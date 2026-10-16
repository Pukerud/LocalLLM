"""Offline, mocked tests for the uninstalled SC117 evidence-runner candidate."""
import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sc117_qualification_runner as runner


class SC117QualificationRunnerTests(unittest.TestCase):
    def test_root_guard_and_missing_proofs_block_before_publication(self):
        with patch.object(runner.os, 'geteuid', return_value=1234, create=True):
            with self.assertRaisesRegex(runner.QualificationError, 'requires effective UID 0'):
                runner.run()
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner, 'capture_live_snapshot') as snapshot, \
                patch.object(runner, 'validate_live_api_contract') as api, \
                patch.object(runner, 'run_active_probes') as probes, \
                patch.object(runner, '_publish_report') as publish:
            with self.assertRaisesRegex(runner.QualificationError, 'issuance is blocked before inference') as error:
                runner.run()
            snapshot.assert_called_once()
            api.assert_called_once()
            probes.assert_not_called()
            publish.assert_not_called()
        for missing in runner.MISSING_EVIDENCE:
            self.assertIn(missing, str(error.exception))

    def test_no_cli_arguments(self):
        with patch.object(runner.sys, 'argv', ['sc117-qualifier', '--report', '/tmp/report']), \
                patch.object(runner, 'run') as run:
            self.assertEqual(runner.main(), 2)
            run.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX owner/mode/no-follow semantics')
    def test_controlled_json_file_no_follow_owner_mode_and_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"schema_version":1}', encoding='utf-8')
            path.chmod(0o600)
            owner_uid = path.stat().st_uid
            with patch.object(runner, 'ROOT_UID', owner_uid):
                value, digest, metadata = runner.read_json_file(
                    path, label='test config', owner_uid=owner_uid, mode=0o600, max_bytes=1024)
            self.assertEqual(value, {'schema_version': 1})
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(metadata.st_ino, path.lstat().st_ino)
            link = Path(directory) / 'linked.json'
            link.symlink_to(path)
            with patch.object(runner, 'ROOT_UID', owner_uid), self.assertRaises(runner.QualificationError):
                runner.read_json_file(link, label='linked config', owner_uid=owner_uid,
                                      mode=0o600, max_bytes=1024)
            path.chmod(0o644)
            with patch.object(runner, 'ROOT_UID', owner_uid), self.assertRaises(runner.QualificationError):
                runner.read_json_file(path, label='wide config', owner_uid=owner_uid,
                                      mode=0o600, max_bytes=1024)

    def test_trial_state_paths_are_derived_not_caller_selected(self):
        home = Path.home() / 'operator-fixture'
        config = {'schema_version': 1, 'operator_uid': 1001,
                  'operator_home': str(home), 'project_root': str(Path.cwd()),
                  'data_root': str(Path.cwd() / 'strata-data')}
        paths = runner.trial_paths(config)
        self.assertEqual(paths['state_root'], home / runner.TRIAL_STATE_SUFFIX)
        self.assertEqual(paths['state_file'], paths['state_root'] / 'server.json')
        self.assertEqual(paths['run_config'], paths['state_root'] / 'run-config.json')
        with self.assertRaises(runner.QualificationError):
            runner.validate_root_config(dict(config, unexpected_path='/tmp/evil'))

    def test_identity_and_process_change_fail_closed(self):
        before = {'server': {'pid': 12, 'start_ticks': 400}, 'config_sha256': 'a' * 64}
        runner.assert_identity_unchanged(before, copy.deepcopy(before))
        changed = copy.deepcopy(before)
        changed['server']['start_ticks'] += 1
        with self.assertRaisesRegex(runner.QualificationError, 'changed during qualification'):
            runner.assert_identity_unchanged(before, changed)
        recorded = {'pid': 12, 'start_ticks': 400, 'exe': '/engine/strata', 'uid': 1001,
                    'cmd': ['strata'], 'pgid': 12, 'sid': 12}
        runner.verify_process_identity(recorded, dict(recorded))
        with patch.object(runner, 'process_identity', return_value=dict(recorded, start_ticks=401)) as process:
            current = process(12)
            with self.assertRaisesRegex(runner.QualificationError, 'no longer matches'):
                runner.verify_process_identity(recorded, current)

    def test_proc_tcp_listener_is_loopback_unique_and_fd_owned(self):
        header = 'sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode'
        row = '0: 0100007F:46A0 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 777'
        listener = runner.parse_proc_tcp_tables(header + '\n' + row, header)
        self.assertEqual(listener, {'address': '127.0.0.1', 'port': 18080,
                                    'inode': 777, 'family': 4})
        self.assertEqual(runner.assert_listener_fd_owner(listener, ['socket:[777]'], expected_pid=12)['pid'], 12)
        with self.assertRaisesRegex(runner.QualificationError, 'uniquely bound'):
            runner.parse_proc_tcp_tables(header + '\n' + row.replace('0100007F', '00000000'), header)
        with self.assertRaisesRegex(runner.QualificationError, 'not owned'):
            runner.assert_listener_fd_owner(listener, ['socket:[778]'], expected_pid=12)

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX file-owner/mode semantics')
    def test_engine_log_cursor_reads_only_complete_appended_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine.log'
            path.write_bytes(b'prior line\n')
            path.chmod(0o600)
            cursor = runner.log_cursor(path, owner_uid=path.stat().st_uid)
            with path.open('ab') as stream:
                stream.write(b'new event\npartial event')
            self.assertEqual(runner.read_log_since(cursor), ['new event'])
            path.write_bytes(b'')
            with self.assertRaisesRegex(runner.QualificationError, 'truncated'):
                runner.read_log_since(cursor)

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX file-owner/mode semantics')
    def test_engine_log_capture_rejects_early_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine.log'
            path.write_bytes(b'prior line\n')
            path.chmod(0o600)
            cursor = runner.log_cursor(path, owner_uid=path.stat().st_uid)
            with path.open('ab') as stream:
                stream.write(b'new event\n')
            with patch.object(runner.os, 'read', return_value=b''):
                with self.assertRaisesRegex(runner.QualificationError, 'incomplete'):
                    runner.read_log_since(cursor)

    def test_gpu_process_inventory_rejects_unknown_workloads(self):
        rows = '12, GPU-00000001, 100\n13, GPU-00000002, 200\n'
        parsed = runner.parse_gpu_compute_processes(
            rows, allowed_pids={12, 13}, expected_uuids={'GPU-00000001', 'GPU-00000002'})
        self.assertEqual({item['pid'] for item in parsed}, {12, 13})
        with self.assertRaisesRegex(runner.QualificationError, 'unrecognized GPU workload'):
            runner.parse_gpu_compute_processes(
                rows.replace('13,', '14,'), allowed_pids={12, 13},
                expected_uuids={'GPU-00000001', 'GPU-00000002'})

    def test_loopback_listener_must_be_unique_and_owned_by_recorded_server(self):
        good = 'LISTEN 0 128 127.0.0.1:18080 0.0.0.0:* users:(("python",pid=444,fd=7))'
        self.assertEqual(runner.validate_listener_output(good, expected_pid=444),
                         {'address': '127.0.0.1', 'port': 18080, 'pid': 444})
        for line in (
                good.replace('127.0.0.1:18080', '0.0.0.0:18080'),
                good.replace('pid=444', 'pid=445'),
                good.replace('LISTEN', 'UNCONN'),
                good.replace('fd=7))', 'fd=7),("other",pid=445,fd=8))'),
                good + '\n' + good):
            with self.subTest(line=line), self.assertRaises(runner.QualificationError):
                runner.validate_listener_output(line, expected_pid=444)

    def test_measured_mtp_requires_depth_four_and_positive_accepted_tokens(self):
        response = {'timings': {'draft_n': 9, 'draft_n_accepted': 3}}
        self.assertEqual(runner.parse_mtp_evidence(response, configured_depth=4)['accepted_tokens'], 3)
        for bad in (
                {'timings': {'draft_n': 9, 'draft_n_accepted': 0}},
                {'timings': {'draft_n': 2, 'draft_n_accepted': 3}},
                {'timings': {'draft_n': True, 'draft_n_accepted': 1}},
        ):
            with self.subTest(bad=bad), self.assertRaises(runner.QualificationError):
                runner.parse_mtp_evidence(bad, configured_depth=4)

    def test_context_requires_count_endpoint_measurement_completion_and_tail(self):
        count = {'input_tokens': 255001}
        completion = {'content': [{'type': 'text', 'text': 'tail: KAPPA-TAIL'}],
                      'stop_reason': 'end_turn',
                      'usage': {'input_tokens': 255001, 'cache_read_input_tokens': 0, 'output_tokens': 2}}
        proof = runner.parse_context_evidence(count, completion, tail_marker='KAPPA-TAIL', elapsed_ms=50)
        self.assertEqual(proof['prompt_tokens'], 255001)
        self.assertTrue(proof['tail_marker_found'])
        for bad_count, bad_response, marker in (
                ({'input_tokens': 254999}, completion, 'KAPPA-TAIL'),
                (count, dict(completion, usage={'input_tokens': 255001, 'cache_read_input_tokens': 0,
                                                'output_tokens': 0}), 'KAPPA-TAIL'),
                (count, completion, 'MISSING'),
        ):
            with self.subTest(bad_count=bad_count, marker=marker), self.assertRaises(runner.QualificationError):
                runner.parse_context_evidence(bad_count, bad_response, tail_marker=marker, elapsed_ms=50)

    def test_cache_requires_correlated_abab_sequence_and_native_park_restore_events(self):
        logs = ['native: strata serve: conversation cache: parked 512 tokens; parked=1 bytes=2048 evictions=0',
                'native: strata serve: conversation cache: parked 496 tokens; parked=2 bytes=4096 evictions=0',
                'native: strata serve: conversation cache: restored 512 tokens; parked=1 bytes=2048',
                'native: strata serve: conversation cache: parked 520 tokens; parked=2 bytes=4096 evictions=0',
                'native: strata serve: conversation cache: restored 496 tokens; parked=1 bytes=2048']
        responses = [
            {'conversation': 'A', 'turn': 1, 'response': {'timings': {'cache_n': 0}}},
            {'conversation': 'B', 'turn': 1, 'response': {'timings': {'cache_n': 0}}},
            {'conversation': 'A', 'turn': 2, 'response': {'timings': {'cache_n': 128}}},
            {'conversation': 'B', 'turn': 2, 'response': {'timings': {'cache_n': 96}}},
        ]
        result = runner.parse_cache_evidence(logs, responses, probe_start_index=0)
        self.assertEqual(result['reused_tokens'], 224)
        for missing_logs, bad_responses in (
                (['conversation cache: parked'], responses),
                (logs, responses[:3]),
                (logs, [dict(responses[0]), dict(responses[1]),
                        {'conversation': 'A', 'turn': 2, 'response': {'timings': {'cache_n': 0}}}, responses[3]]),
                (logs, [dict(responses[0], conversation='B'), *responses[1:]]),
                (logs, [{'conversation_cache': {'reused_tokens': 100}}] * 4),
        ):
            with self.subTest(logs=missing_logs, responses=bad_responses), \
                    self.assertRaises(runner.QualificationError):
                runner.parse_cache_evidence(missing_logs, bad_responses, probe_start_index=0)
        with self.assertRaisesRegex(runner.QualificationError, 'preceding in-window park'):
            runner.parse_cache_evidence(logs, responses, probe_start_index=2)

    def test_vision_requires_exact_projector_process_actual_backend_and_image_success(self):
        digest = 'a' * 64
        process = {'pid': 77, 'exe': '/pinned/engine/strata-vision', 'config_sha256': 'b' * 64}
        response = {'id': 'chatcmpl-' + '1' * 24,
                    'choices': [{'message': {'content': runner.VISION_MARKER}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 20}}
        gpu_uuids = [f'GPU-{i:08x}' for i in range(4)]
        observed = {'source': 'pinned_strata_0.1.39_gpu_vision_path', 'device': 'CUDA',
                    'precision': 'bf16', 'projector_sha256': digest, 'vision_pid': 77,
                    'expected_gpu_uuids': gpu_uuids,
                    'gpu_uuids_before': gpu_uuids, 'gpu_uuids_after': gpu_uuids}
        proof = runner.parse_vision_evidence(
            response, projector_sha256=digest, expected_projector_sha256=digest,
            vision_process=process, expected_vision_process=copy.deepcopy(process),
            runtime_observation=observed)
        self.assertTrue(proof['gpu'])
        for changes in (
                {'projector_sha256': 'c' * 64},
                {'vision_process': dict(process, pid=78)},
                {'runtime_observation': dict(observed, device='CPU')},
                {'runtime_observation': dict(observed, precision='FP16')},
                {'image_response': {'choices': [], 'usage': {}}},
        ):
            args = {'projector_sha256': digest, 'expected_projector_sha256': digest,
                    'vision_process': process, 'expected_vision_process': process,
                    'runtime_observation': observed, 'image_response': response}
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(runner.QualificationError):
                runner.parse_vision_evidence(**args)

    def test_prefill_requires_multiple_explicit_measured_auto_samples(self):
        samples = [{'selected_prefill': 'auto', 'prompt_total': 70000,
                    'response_id': 'chatcmpl-' + '1' * 24,
                    'progress_points': [{'prompt_read': n, 'prompt_total': 70000,
                                         'prefill_tok_s_mean': 100.0} for n in (32768, 65536, 70000)]}
                   for _ in range(3)]
        proof = runner.parse_prefill_evidence(samples, selected_prefill='auto')
        self.assertEqual(len(proof['samples']), 3)
        self.assertEqual(proof['samples'][0]['observed_chunk_tokens'], [32768, 32768, 4464])
        with self.assertRaisesRegex(runner.QualificationError, 'live auto-prefill progress'):
            runner.parse_prefill_evidence([{'timings': {'prompt_n': 5000, 'prompt_ms': 100}}] * 3,
                                          selected_prefill='auto')
        with self.assertRaises(runner.QualificationError):
            runner.parse_prefill_evidence(samples[:2], selected_prefill='auto')
        with self.assertRaises(runner.QualificationError):
            runner.parse_prefill_evidence(samples, selected_prefill='1024')

    def test_runtime_metrics_telemetry_requires_four_fresh_indexed_gpu_rows(self):
        metrics = {'time': 1000.0, 'hardware_static': {'gpu_count': 4},
                   'hardware': {'ram_used': 1024**3, 'ram_total': 8 * 1024**3,
                                'gpus': [{'index': i, 'util': 10 + i, 'mem_used': 2 * 1024**3,
                                          'mem_total': 24 * 1024**3, 'temp': 55, 'power': 100}
                                         for i in range(4)]}}
        machine = {'gpu_uuids': [f'GPU-{i:08x}' for i in range(4)]}
        telemetry = runner.parse_runtime_telemetry(
            metrics, machine=machine, captured_monotonic=1, now_monotonic=1.1,
            now_wallclock=1001)
        self.assertTrue(telemetry['passed'])
        self.assertEqual(len(telemetry['gpus']), 4)
        self.assertEqual(telemetry['sample_age_milliseconds'], 1000)
        with self.assertRaisesRegex(runner.QualificationError, 'incomplete, malformed, or stale'):
            runner.parse_runtime_telemetry(
                metrics, machine=machine, captured_monotonic=1, now_monotonic=1.1,
                now_wallclock=1006)

    def test_live_api_contract_matches_pinned_039_health_schema(self):
        # From pinned 0.1.39 commit 6f32ec0, serve/server.py do_GET: /health returns
        # status="ok" (not an undocumented "s" field).
        health = {'status': 'ok', 'service': 'strata', 'model': runner.MODEL_ID,
                  'max_context': 262144, 'images': True, 'api_key': False, 'loaded': True}
        status = {'service': 'strata', 'model': runner.MODEL_ID,
                  'engine': runner.RUNTIME_VERSION, 'loaded': True,
                  'context': {'native': 262144}, 'vision': {'enabled': True},
                  'concurrency': {'serving': 1, 'requested': 1}, 'activity': {'in_flight': 0}}
        metrics = {'engine': {'model': runner.MODEL_ID, 'max_context': 262144, 'images': True},
                   'live': {'state': 'idle', 'queued': 0}, 'totals': {'requests': 0}}
        payloads = {'/health': health, '/v1/status': status, '/metrics': metrics}
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner, 'api_request', side_effect=lambda method, path: payloads[path]):
            contract = runner.validate_live_api_contract()
        self.assertEqual(contract['health'], health)
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner, 'api_request', side_effect=lambda method, path:
                          dict(payloads[path], status='missing', s=True) if path == '/health' else payloads[path]):
            with self.assertRaisesRegex(runner.QualificationError, 'exact loaded .39'):
                runner.validate_live_api_contract()

    def test_api_client_is_loopback_only_whitelisted_and_bounded(self):
        response = type('Response', (), {'read': lambda self, n: b'{"status":"ok"}',
                                         '__enter__': lambda self: self,
                                         '__exit__': lambda self, *args: None})()
        opener = type('Opener', (), {'open': lambda self, request, timeout: response})()
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner.urllib.request, 'build_opener', return_value=opener) as build:
            result = runner.api_request('GET', '/health')
            self.assertEqual(result, {'status': 'ok'})
            handlers = build.call_args.args
            self.assertTrue(any(isinstance(handler, runner.urllib.request.ProxyHandler) and
                                handler.proxies == {} for handler in handlers))
            self.assertTrue(any(isinstance(handler, runner._NoRedirectHandler) for handler in handlers))
            with self.assertRaisesRegex(runner.QualificationError, 'Only fixed read-only'):
                runner.api_request('GET', 'http://example.invalid/health')
            for path in ('/unload', '/v1/chat/completions', '/v1/messages/count_tokens'):
                with self.subTest(path=path), self.assertRaisesRegex(runner.QualificationError,
                                                                     'Only fixed read-only'):
                    runner.api_request('POST', path, {})
            build.assert_called_once()

    def test_pinned_prefill_parser_uses_metrics_progress_not_invented_timing_fields(self):
        sample = {'selected_prefill': 'auto', 'prompt_total': 70000,
                  'response_id': 'chatcmpl-' + '1' * 24,
                  'progress_points': [{'prompt_read': n, 'prompt_total': 70000,
                                       'prefill_tok_s_mean': 100.0} for n in (32768, 65536, 70000)]}
        proof = runner.parse_prefill_evidence([sample, sample, sample], selected_prefill='auto')
        self.assertEqual(proof['samples'][0]['observed_chunk_tokens'], [32768, 32768, 4464])
        fabricated = {'selected_prefill': 'auto', 'prompt_total': 70000,
                      'timings': {'effective_prefill_tokens': 32768, 'prompt_n': 70000, 'prompt_ms': 1}}
        with self.assertRaises(runner.QualificationError):
            runner.parse_prefill_evidence([fabricated] * 3, selected_prefill='auto')

    def test_readonly_capture_only_reads_metrics_and_never_promotes_probe_checks(self):
        idle = {'live': {'state': 'idle', 'queued': 0}, 'totals': {'requests': 3}}
        before = {'machine_identity': {'gpu_uuids': [f'GPU-{i:08x}' for i in range(4)]},
                  'engine_log_cursor': {'path': '/fixed/log', 'offset': 0},
                  'trial': {'hashes': {}, 'state': {}, 'run_config': {}},
                  'process_family': [], 'listener': {}, 'gpu_compute_processes': []}
        calls = []
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner, 'api_request', side_effect=lambda method, path, **kwargs:
                          calls.append((method, path)) or idle), \
                patch.object(runner, 'parse_runtime_telemetry', return_value={'passed': True}), \
                patch.object(runner, 'read_log_since', return_value=[]), \
                patch.object(runner, '_report_live_identity', return_value={'pid': 1}):
            result = runner.capture_readonly_observations(before, {'metrics': idle})
        self.assertTrue(result['inference_probes_issued'] is False)
        self.assertEqual(calls, [('GET', '/metrics'), ('GET', '/metrics')])
        self.assertFalse(any(result.get('checks', {}).values()))

    def test_api_client_is_strictly_get_only(self):
        with patch.object(runner.os, 'geteuid', return_value=0, create=True):
            with self.assertRaisesRegex(runner.QualificationError, 'Only fixed read-only'):
                runner.api_request('POST', '/v1/chat/completions', {'messages': []})
            with self.assertRaisesRegex(runner.QualificationError, 'Only fixed read-only'):
                runner.api_request('POST', '/v1/messages/count_tokens', {'messages': []})

    def test_fresh_four_gpu_and_ram_telemetry_is_parsed_strictly(self):
        gpu = '\n'.join(f'GPU-{i:08x}, 555.1, {100 + i}, {20 + i}' for i in range(4))
        meminfo = 'MemTotal: 100000 kB\nMemAvailable: 16777216 kB\n'
        telemetry = runner.parse_telemetry(gpu, meminfo, captured_monotonic=10.0, now_monotonic=11.0)
        self.assertTrue(telemetry['passed'])
        self.assertEqual(telemetry['ram_available_mib'], 16384)
        with self.assertRaises(runner.QualificationError):
            runner.parse_telemetry(gpu, meminfo, captured_monotonic=10.0, now_monotonic=16.0)
        with self.assertRaises(runner.QualificationError):
            runner.parse_telemetry(gpu.splitlines()[0], meminfo,
                                   captured_monotonic=10.0, now_monotonic=10.1)

    def _valid_report(self, runner_hash):
        machine = {'machine_id_sha256': 'd' * 64,
                   'gpu_uuids': [f'GPU-{i:08x}' for i in range(4)], 'driver_version': '555.1'}
        runtime_assets = [['/opt/strata/engine/strata', 4096, '4' * 64],
                          ['/opt/strata/engine/strata-vision', 4096, '5' * 64]]
        python_pin = ['/opt/venv/bin/python', '6' * 64]
        fingerprint_inputs = {'runtime_assets': runtime_assets,
                              'runtime_python': python_pin, 'test_artifact_identity': 'pinned'}
        settings = dict(runner.DEFAULT_SETTINGS)
        fingerprint = runner.profile_fingerprint(fingerprint_inputs, machine, settings)
        process_template = {'start_ticks': 100, 'exe_device': 1, 'exe_inode': 1000,
                            'uid': 1001, 'pgid': 100, 'sid': 100}
        processes = [
            dict(process_template, pid=100, exe=python_pin[0], exe_sha256=python_pin[1],
                 exe_size=8192, cmd=['python', 'server.py']),
            dict(process_template, pid=101, exe=runtime_assets[0][0], exe_sha256=runtime_assets[0][2],
                 exe_size=runtime_assets[0][1], cmd=['strata', '--serve']),
            dict(process_template, pid=102, exe=runtime_assets[1][0], exe_sha256=runtime_assets[1][2],
                 exe_size=runtime_assets[1][1], cmd=['strata-vision', '--gpu']),
        ]
        live_material = {'root_config_sha256': '1' * 64, 'pin_policy_sha256': 'f' * 64,
                         'state_sha256': '2' * 64, 'run_config_sha256': '3' * 64,
                         'process_family': processes,
                         'listener': {'address': '127.0.0.1', 'port': 18080, 'pid': 100},
                         'gpu_processes': [[101, machine['gpu_uuids'][0]],
                                           [102, machine['gpu_uuids'][1]]]}
        binding = {'live_sha256': runner.canonical_sha256(live_material),
                   'live_material': live_material,
                   'artifact_sha256': runner.canonical_sha256(fingerprint_inputs),
                   'machine_identity': machine, 'profile_fingerprint': fingerprint}
        binding_hash = runner.canonical_sha256(binding)
        ids = {name: 'chatcmpl-' + f'{index + 1:024x}'
               for index, name in enumerate(probe for probe in runner.PROBE_ORDER
                                             if not probe.startswith('count_'))}
        receipts, counter = [], 0
        for index, probe in enumerate(runner.PROBE_ORDER):
            is_count = probe.startswith('count_')
            start = index * 2
            cache_events = []
            if probe == 'cache_A1':
                cache_events = [{'event': 'parked', 'tokens': 512, 'parked': 1,
                                 'bytes': 2048, 'evictions': 0}]
            elif probe == 'cache_B1':
                cache_events = [{'event': 'parked', 'tokens': 496, 'parked': 2,
                                 'bytes': 4096, 'evictions': 0}]
            elif probe == 'cache_A2':
                cache_events = [{'event': 'restored', 'tokens': 512, 'parked': 1,
                                 'bytes': 2048, 'evictions': 0}]
            elif probe == 'cache_B2':
                cache_events = [{'event': 'parked', 'tokens': 520, 'parked': 2,
                                 'bytes': 4096, 'evictions': 0},
                                {'event': 'restored', 'tokens': 496, 'parked': 1,
                                 'bytes': 2048, 'evictions': 0}]
            cursor_before = {'path': str(Path(tempfile.gettempdir()) / 'sc117-engine.log'), 'identity': [1, 2, 3, 4, 5, 6],
                             'offset': start}
            cursor_after = dict(cursor_before, offset=start + 1)
            receipts.append({
                'probe': probe, 'probe_id': f'{index + 1:024x}',
                'response_id': None if is_count else ids[probe],
                'started_at': start + 1, 'finished_at': start + 1.5,
                'request_counter_before': counter,
                'request_counter_after': counter if is_count else counter + 1,
                'prompt_tokens': 255001 if probe in ('count_context', 'native_context') else 60000
                if probe.startswith(('count_prefill_', 'prefill_')) else 20,
                'completion_tokens': 0 if is_count else 2,
                'finish': 'counted' if is_count else 'stop',
                'log_cursor_before': cursor_before, 'log_cursor_after': cursor_after,
                'cache_events': cache_events, 'identity_before_sha256': binding_hash,
                'identity_after_sha256': binding_hash,
            })
            if not is_count:
                counter += 1
        smoke = {'passed': True, 'response_id': ids['smoke']}
        context = {'prompt_tokens': 255001, 'completion_tokens': 2, 'tail_marker_found': True,
                   'elapsed_milliseconds': 50, 'response_id': ids['native_context']}
        vision = {'passed': True, 'gpu': True, 'precision': 'bf16', 'prompt_tokens': 20,
                  'response_id': ids['vision'], 'projector_sha256': runner.SC117_ASSET_PINS[2][2],
                  'vision_process': dict(processes[2], ppid=100),
                  'runtime_observation': {
                      'source': 'pinned_strata_0.1.39_gpu_vision_path', 'device': 'CUDA',
                      'precision': 'bf16', 'projector_sha256': runner.SC117_ASSET_PINS[2][2],
                      'vision_pid': 102, 'gpu_uuids_before': machine['gpu_uuids'][:1],
                      'gpu_uuids_after': machine['gpu_uuids'][:1],
                      'expected_gpu_uuids': machine['gpu_uuids']}}
        mtp = {'passed': True, 'depth': 4, 'head_variant': runner.MTP_HEAD_VARIANT,
               'drafted_tokens': 8, 'accepted_tokens': 2, 'response_id': ids['mtp']}
        cache_sequence = []
        for conversation, turn in (('A', 1), ('B', 1), ('A', 2), ('B', 2)):
            probe = f'cache_{conversation}{turn}'
            receipt = next(item for item in receipts if item['probe'] == probe)
            cache_sequence.append({'conversation': conversation, 'turn': turn,
                                   'response_id': ids[probe], 'reused_tokens': 8 if turn == 2 else 0,
                                   'cache_events': receipt['cache_events']})
        cache = {'parked': True, 'restored': True, 'reused': True,
                 'sequence': cache_sequence}
        prefill_samples = []
        for index in (1, 2, 3):
            prefill_samples.append({'selected_prefill': 'auto', 'prompt_total': 60000,
                                    'response_id': ids[f'prefill_{index}'],
                                    'prompt_read_samples': [20000, 40000, 60000],
                                    'observed_chunk_tokens': [20000, 20000, 20000],
                                    'observed_prefill_tok_s_mean': [100.0, 100.0, 100.0]})
        prefill = {'passed': True, 'selected_prefill': 'auto', 'samples': prefill_samples}
        telemetry = {'passed': True, 'source': 'strata_0.1.39_GET_/metrics',
                     'ram_total_mib': 16384, 'ram_used_mib': 4096,
                     'sample_age_milliseconds': 1,
                     'gpus': [{'index': i, 'utilization_percent': 0, 'memory_used_mib': 1,
                               'memory_total_mib': 100, 'temperature_c': 50, 'power_w': 10}
                              for i in range(4)]}
        return {
            'schema_version': 1, 'status': 'passed', 'evidence_runner_sha256': runner_hash,
            'profile': runner.PROFILE, 'model_id': runner.MODEL_ID,
            'runtime_version': runner.RUNTIME_VERSION, 'runtime_commit': runner.RUNTIME_COMMIT,
            'profile_fingerprint': fingerprint, 'machine_identity': machine,
            'settings': settings, 'profile_fingerprint_inputs': fingerprint_inputs,
            'checks': {name: True for name in runner.REQUIRED_CHECKS},
            'smoke': smoke, 'native_context': context, 'vision': vision, 'mtp_acceptance': mtp,
            'conversation_cache': cache, 'prefill_screen': prefill, 'resource_telemetry': telemetry,
            'live_identity': {'before_sha256': binding['live_sha256'],
                              'after_sha256': binding['live_sha256']},
            'artifact_identity': {'before_sha256': binding['artifact_sha256'],
                                  'after_sha256': binding['artifact_sha256'],
                                  'fingerprint_inputs_sha256': binding['artifact_sha256'],
                                  'root_pin_policy_sha256': 'f' * 64},
            'measured_observations': {
                'identity_binding': {'before': binding, 'after': copy.deepcopy(binding)},
                'probe_receipts': receipts, 'baseline_request_count': 0,
            },
            'recorded_at': '2026-10-16T00:00:00Z',
        }

    def test_report_staging_is_root_checked_atomic_no_follow_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_file = root / 'installed-runner'
            runner_file.write_text('#!/usr/bin/python3\n', encoding='utf-8')
            runner_file.chmod(0o555)
            report_path = root / 'qualification.json'
            report = self._valid_report(hashlib.sha256(runner_file.read_bytes()).hexdigest())
            session = runner._new_probe_session({}, {}, {})
            owner_uid = runner_file.stat().st_uid
            with patch.object(runner, 'ROOT_UID', owner_uid), \
                    patch.object(runner, 'RUNNER_PATH', runner_file), \
                    patch.object(runner, 'REPORT_PATH', report_path), \
                    patch.object(runner.os, 'geteuid', return_value=owner_uid, create=True), \
                    patch.object(runner, '_assert_report_parent'), \
                    patch.object(runner, '_fsync_directory'), \
                    patch.object(runner, 'REPORT_ISSUANCE_ENABLED', True), \
                    patch.object(runner, 'sha256_file', return_value=report['evidence_runner_sha256']), \
                    patch.object(runner, '_build_report', return_value=report):
                result = runner._publish_report(session)
                self.assertEqual(result, report_path)
                self.assertEqual(stat.S_IMODE(report_path.lstat().st_mode), 0o444)
                serialized_report = json.loads(report_path.read_text())
                self.assertEqual(serialized_report, json.loads(json.dumps(report)))
                with self.assertRaisesRegex(runner.QualificationError, 'already exists'):
                    runner._publish_report(session)
                self.assertEqual(json.loads(report_path.read_text()), serialized_report)

    def test_symlink_report_target_is_not_followed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_file = root / 'installed-runner'
            runner_file.write_text('runner', encoding='utf-8')
            runner_file.chmod(0o555)
            victim = root / 'victim'
            victim.write_text('preserve')
            target = root / 'qualification.json'
            target.symlink_to(victim)
            report = self._valid_report(hashlib.sha256(runner_file.read_bytes()).hexdigest())
            session = runner._new_probe_session({}, {}, {})
            owner_uid = runner_file.stat().st_uid
            with patch.object(runner, 'ROOT_UID', owner_uid), \
                    patch.object(runner, 'RUNNER_PATH', runner_file), \
                    patch.object(runner, 'REPORT_PATH', target), \
                    patch.object(runner.os, 'geteuid', return_value=owner_uid, create=True), \
                    patch.object(runner, '_assert_report_parent'), \
                    patch.object(runner, '_fsync_directory'), \
                    patch.object(runner, 'REPORT_ISSUANCE_ENABLED', True), \
                    patch.object(runner, 'sha256_file', return_value=report['evidence_runner_sha256']), \
                    patch.object(runner, '_build_report', return_value=report):
                with self.assertRaises(runner.QualificationError):
                    runner._publish_report(session)
            self.assertEqual(victim.read_text(), 'preserve')
            self.assertTrue(target.is_symlink())

    def test_unreviewed_runner_cannot_publish_even_a_well_formed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'qualification.json'
            session = runner._new_probe_session({}, {}, {})
            fabricated_report = self._valid_report('e' * 64)
            with self.assertRaisesRegex(runner.QualificationError, 'caller-supplied report payloads'):
                runner._publish_report(fabricated_report)
            with patch.object(runner, 'REPORT_PATH', target), \
                    patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                    patch.object(runner, 'REPORT_ISSUANCE_ENABLED', False):
                with self.assertRaisesRegex(runner.QualificationError, 'publication is disabled'):
                    runner._publish_report(session)
            self.assertFalse(target.exists())

    def test_missing_or_mismatched_proof_never_calls_report_publication(self):
        report = self._valid_report('e' * 64)
        evidence = {'checks': {name: True for name in runner.REQUIRED_CHECKS}}
        with patch.object(runner, '_publish_report') as publish:
            with self.assertRaisesRegex(runner.QualificationError, 'missing measured proof'):
                runner.assert_complete_evidence(evidence)
            with self.assertRaisesRegex(runner.QualificationError, 'runner-captured probe sessions'):
                runner._publish_if_complete(evidence)
            publish.assert_not_called()

        session = runner._new_probe_session({}, {}, {})
        mismatched = copy.deepcopy(report)
        mismatched['native_context']['prompt_tokens'] += 1
        with patch.object(runner.os, 'geteuid', return_value=0, create=True), \
                patch.object(runner, '_build_report', return_value=report):
            with self.assertRaisesRegex(runner.QualificationError, 'differs from runner-captured evidence'):
                runner._publish_report_payload(mismatched, runner_sha256='e' * 64, session=session)

    def test_report_rejects_fabricated_probes_and_mismatched_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runner_file = Path(directory) / 'runner'
            runner_file.write_text('runner')
            runner_file.chmod(0o555)
            digest = hashlib.sha256(runner_file.read_bytes()).hexdigest()
            report = self._valid_report(digest)
            report['measured_observations'] = {
                'identity_binding': report['measured_observations']['identity_binding'],
                'probe_receipts': [], 'baseline_request_count': 0}
            with self.assertRaisesRegex(runner.QualificationError, 'runner-captured probe receipts'):
                runner._validate_report_payload(report, runner_sha256=digest)

            after = report['measured_observations']['identity_binding']['after']
            after['live_material']['state_sha256'] = '9' * 64
            after['live_sha256'] = runner.canonical_sha256(after['live_material'])
            with self.assertRaisesRegex(runner.QualificationError, 'identity changed'):
                runner._validate_report_payload(report, runner_sha256=digest)

    def test_report_schema_rejects_forged_boolean_mtp_and_missing_raw_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            runner_file = Path(directory) / 'runner'
            runner_file.write_text('runner')
            runner_file.chmod(0o555)
            digest = hashlib.sha256(runner_file.read_bytes()).hexdigest()
            report = self._valid_report(digest)
            runner._validate_report_payload(report, runner_sha256=digest)
            bad = copy.deepcopy(report)
            bad['mtp_acceptance']['accepted_tokens'] = 0
            with self.assertRaises(runner.QualificationError):
                runner._validate_report_payload(bad, runner_sha256=digest)
            bad = copy.deepcopy(report)
            bad['native_context']['prompt_tokens'] = True
            with self.assertRaises(runner.QualificationError):
                runner._validate_report_payload(bad, runner_sha256=digest)


if __name__ == '__main__':
    unittest.main()
