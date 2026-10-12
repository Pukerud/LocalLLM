"""CPU-only gates for the isolated, opt-in Q4_K_S / Strata 0.1.43 candidate."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

if sys.platform != 'linux':
    raise unittest.SkipTest('Linux controller imports')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_api_client
import orca_q4ks_v0143_assets as assets
import orca_q4ks_v0143_profile as candidate
import orca_q4ks_v0143_native_validation as native_validation
import prepare_orca_q4ks_v0143 as prepare
import orca_q4ks_v0143_validation as validation_runner
from engine_safety import SafetyError


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CandidateIdentity(unittest.TestCase):
    def test_candidate_has_a_distinct_profile_runtime_tree_and_required_build_flags(self):
        root = Path('/tmp/strata-data')
        self.assertEqual(assets.SOURCE_COMMIT, '7695859aa13afb248bdb4d36f9b2e95707db079d')
        self.assertEqual(assets.RUNTIME_VERSION, '0.1.43')
        self.assertNotEqual(assets.profile_root(root), assets.baseline_profile_root(root))
        self.assertNotEqual(assets.source_root(root), assets.pinned_runtime_source(root))
        self.assertNotIn(assets.PROFILE, __import__('strata_runtime').PROFILES)
        self.assertEqual(assets.BUILD_OPTIONS['STRATA_ORCA_Q4KS_MMQ'], 'ON')
        self.assertEqual(assets.BUILD_OPTIONS['STRATA_NATIVE_EXPERTS'], 'ON')
        self.assertEqual(assets.BUILD_OPTIONS['STRATA_BUILD_TESTS'], 'ON')
        contract = assets.cmake_cache_contract(Path('/tools/cmake'), Path('/runtime/llama'))
        self.assertIn('STRATA_ORCA_Q4KS_MMQ:BOOL=ON', contract)
        self.assertIn('STRATA_BUILD_TESTS:BOOL=ON', contract)
        self.assertIn('STRATA_NATIVE_EXPERTS:BOOL=ON', contract)

    def test_initial_candidate_config_is_only_32k_text_and_uses_distinct_engine(self):
        root = Path('/tmp/strata-data')
        cfg = candidate.make_config(root)
        args = cfg['args']
        self.assertEqual(args[args.index('--max-context') + 1], '32768')
        self.assertEqual(cfg['exe'], str(assets.source_root(root) / 'engine/strata'))
        self.assertEqual(cfg['cwd'], str(assets.source_root(root)))
        self.assertNotIn('--vision', args)
        self.assertNotIn('vision', cfg)
        self.assertEqual(cfg['gpu_order'], 'as_given')
        self.assertFalse(cfg.get('parallel'))
        cache_args = ['--conversation-cache-mib', '8192', '--conversation-cache-slots', '4']
        self.assertEqual(args[args.index('--conversation-cache-mib') - 0:
                                args.index('--conversation-cache-mib') + len(cache_args)], cache_args)

    def test_all_candidate_config_variants_keep_the_eight_gib_cache_contract(self):
        root = Path('/tmp/strata-data')
        expected = ['--conversation-cache-mib', '8192', '--conversation-cache-slots', '4']
        for context in (32768, 262144):
            with self.subTest(context=context):
                config = candidate.make_config(root, context=context)
                args = config['args']
                start = args.index('--conversation-cache-mib')
                self.assertEqual(args[start:start + len(expected)], expected)
        with mock.patch.object(candidate, '_runtime_material', return_value=(None,) * 7):
            vision = candidate.make_config(root, context=262144, vision=True)
        args = vision['args']
        start = args.index('--conversation-cache-mib')
        self.assertEqual(args[start:start + len(expected)], expected)
        self.assertIs(vision['vision']['gpu'], False)

    def test_explicit_native_vision_cannot_be_created_without_native_context(self):
        with self.assertRaisesRegex(SafetyError, 'native-context migration'):
            candidate.make_config(Path('/tmp/strata-data'), context=32768, vision=True)

    def test_shared_asset_contract_survives_json_round_trip(self):
        manifest = {
            'assets': [{'path': '/b', 'bytes': 2, 'sha256': 'b'},
                       {'path': '/a', 'bytes': 1, 'sha256': 'a'}],
            'pack_assets': [{'path': '/p', 'bytes': 3, 'sha256': 'p'}],
        }
        contract = candidate._shared_asset_contract(manifest)
        self.assertEqual(json.loads(json.dumps(contract)), contract)
        self.assertEqual(contract['assets'][0], ['/a', 1, 'a'])


class RuntimeRouting(unittest.TestCase):
    def test_auto_remains_q4_v039_and_explicit_041_uses_only_candidate_gate(self):
        import strata_launcher as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = (Path('/old-source'), Path('/baseline/config.json'), {'args': ['--max-context', '32768']})
            optin = (Path('/candidate-source'), Path('/candidate/config.json'), {'args': ['--max-context', '32768']})
            selection = root / 'runtime-selection.json'
            selection.write_text('{"version":"0.1.43"}\n')
            before = selection.read_bytes()
            with mock.patch.object(launcher, 'data_root', return_value=root), \
                 mock.patch.object(launcher, 'ready_q4ks', return_value=baseline) as base_ready, \
                 mock.patch.object(launcher.q4ks_v0143, 'ready_q4ks_v0143', return_value=optin) as q4_candidate, \
                 mock.patch.object(launcher.runtimes, 'configure', side_effect=AssertionError('generic manager used')):
                result_default = launcher.ready_runtime(assets.HOST_PROFILE, runtime='auto', parallel=1, batch_groups=1)
                result_candidate = launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.43', parallel=1, batch_groups=1)
            self.assertEqual(result_default[0], baseline[0])
            self.assertEqual(result_default[3], '0.1.39')
            self.assertEqual(result_candidate[0], optin[0])
            self.assertEqual(result_candidate[3], '0.1.43')
            base_ready.assert_called_once_with(root)
            q4_candidate.assert_called_once_with(root)
            self.assertEqual(selection.read_bytes(), before)

    def test_q4_candidate_defaults_to_loopback_and_allows_only_explicit_trusted_lan_hosts(self):
        import strata_launcher as launcher
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher._effective_host(assets.HOST_PROFILE, '0.1.43', '0.0.0.0'), '127.0.0.1')
        for requested in ('127.0.0.1', '0.0.0.0'):
            with self.subTest(requested=requested), mock.patch.dict(os.environ, {'STRATA_HOST': requested}):
                self.assertEqual(launcher._effective_host(assets.HOST_PROFILE, '0.1.43', '127.0.0.1',
                                                          context=262144), requested)
        for version in ('0.1.41', '0.1.43'):
            with self.subTest(version=version), mock.patch.dict(os.environ, {'STRATA_HOST': '0.0.0.0'}):
                with self.assertRaisesRegex(SafetyError, '32K validation must remain loopback-only'):
                    launcher._effective_host(assets.HOST_PROFILE, version, '127.0.0.1', context=32768)
        for requested in ('192.168.1.69', 'localhost', '::1', ''):
            with self.subTest(requested=requested), mock.patch.dict(os.environ, {'STRATA_HOST': requested}):
                with self.assertRaisesRegex(SafetyError, 'may only be 127.0.0.1 or 0.0.0.0'):
                    launcher._effective_host(assets.HOST_PROFILE, '0.1.43', '127.0.0.1')
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher._effective_host(assets.HOST_PROFILE, '0.1.39', '0.0.0.0'), '0.0.0.0')

    def test_q4_candidate_start_refuses_root_before_profile_inspection(self):
        import strata_launcher as launcher
        with mock.patch.object(launcher.os, 'geteuid', return_value=0), \
             mock.patch.object(launcher, 'ready_runtime', side_effect=AssertionError('profile inspected')):
            with self.assertRaisesRegex(SafetyError, 'normal operator'):
                launcher.start(None, profile=assets.HOST_PROFILE, runtime='0.1.43')

    def test_candidate_start_accepts_only_the_identity_bound_hostllm_or_hive_authorization(self):
        import strata_launcher as launcher
        lease_id = 'a' * 64
        token = 'b' * 48
        held = {'phase': 'held', 'port': 8080, 'lease_id': lease_id}
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=held):
            auth = launcher.require_q4ks_candidate_authorization(
                8080, lease_id, host_launch_reserved=True, runtime_version='0.1.43')
            self.assertEqual(auth, {'authorization': 'hostllm', 'lease_id': lease_id})
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'MINER_RUN'
            marker.touch()
            stop_marker = Path(temp) / 'MINER_STOP'
            current = Path(temp) / 'cur_miner'
            current.write_text('custom\n')
            with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': token}, clear=True), \
                 mock.patch.object(launcher.host_lifecycle, 'RUN', marker), \
                 mock.patch.object(launcher.host_lifecycle, 'STOP', stop_marker), \
                 mock.patch.object(launcher, 'HIVE_CUR_MINER', current), \
                 mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=None):
                auth = launcher.require_q4ks_candidate_authorization(
                    8080, host_launch_reserved=False, runtime_version='0.1.43')
            self.assertEqual(auth, {'authorization': 'hive', 'launch_id': token})
            for selected, lease in (('other', None), ('custom', held)):
                current.write_text(selected + '\n')
                with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': token}, clear=True), \
                     mock.patch.object(launcher.host_lifecycle, 'RUN', marker), \
                     mock.patch.object(launcher.host_lifecycle, 'STOP', stop_marker), \
                     mock.patch.object(launcher, 'HIVE_CUR_MINER', current), \
                     mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=lease):
                    with self.assertRaises(SafetyError):
                        launcher.require_q4ks_candidate_authorization(
                            8080, host_launch_reserved=False, runtime_version='0.1.43')
            current.write_text('custom\n')
            marker.unlink()
            with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': token}, clear=True), \
                 mock.patch.object(launcher.host_lifecycle, 'RUN', marker), \
                 mock.patch.object(launcher.host_lifecycle, 'STOP', stop_marker), \
                 mock.patch.object(launcher, 'HIVE_CUR_MINER', current), \
                 mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=None):
                with self.assertRaisesRegex(SafetyError, 'MINER_RUN'):
                    launcher.require_q4ks_candidate_authorization(
                        8080, host_launch_reserved=False, runtime_version='0.1.43')
            marker.touch()
            stop_marker.touch()
            with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': token}, clear=True), \
                 mock.patch.object(launcher.host_lifecycle, 'RUN', marker), \
                 mock.patch.object(launcher.host_lifecycle, 'STOP', stop_marker), \
                 mock.patch.object(launcher, 'HIVE_CUR_MINER', current), \
                 mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=None):
                with self.assertRaisesRegex(SafetyError, 'stop marker'):
                    launcher.require_q4ks_candidate_authorization(
                        8080, host_launch_reserved=False, runtime_version='0.1.43')
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(launcher.host_lifecycle, 'read_lease', return_value=held):
            with self.assertRaisesRegex(SafetyError, 'reserved HostLLM lease'):
                launcher.require_q4ks_candidate_authorization(
                    8080, lease_id, host_launch_reserved=False, runtime_version='0.1.43')

    def test_q4_slot_profiles_keep_039_single_and_allow_only_043_two_by_two(self):
        import strata_launcher as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate_config = {'args': ['--max-context', '262144', '--conversation-cache-mib', '8192',
                                         '--conversation-cache-slots', '4'],
                                'vision': {'gpu': False}}
            with mock.patch.object(launcher, 'data_root', return_value=root), \
                 mock.patch.object(launcher.q4ks_v0143, 'ready_q4ks_v0143',
                                   return_value=(Path('/candidate'), Path('/candidate/config.json'),
                                                 json.loads(json.dumps(candidate_config)))):
                with self.assertRaisesRegex(SafetyError, 'isolated to the explicit Orca Q4 candidate'):
                    launcher.ready_runtime('iq3_s', runtime='0.1.43', parallel=1, batch_groups=1)
                with self.assertRaisesRegex(SafetyError, 'baseline or an explicit isolated .41/.43 candidate'):
                    launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.38', parallel=1, batch_groups=1)
                with self.assertRaisesRegex(SafetyError, 'preserved Q4 .39.*one slot'):
                    launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.39', parallel=2, batch_groups=2)
                with self.assertRaisesRegex(SafetyError, '1/1 or 2/2'):
                    launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.43', parallel=2, batch_groups=1)
                source, path, single, version = launcher.ready_runtime(
                    assets.HOST_PROFILE, runtime='0.1.43', parallel=1, batch_groups=1)
                self.assertEqual(version, '0.1.43')
                self.assertNotIn('parallel', single)
                self.assertNotIn('--batch-groups', single['args'])
                source, path, dual, version = launcher.ready_runtime(
                    assets.HOST_PROFILE, runtime='0.1.43', parallel=2, batch_groups=2)
                self.assertEqual(version, '0.1.43')
                self.assertEqual(dual['parallel'], 2)
                self.assertEqual(dual['args'][-2:], ['--batch-groups', '2'])
                initial_config = json.loads(json.dumps(candidate_config))
                initial_config['args'][1] = '32768'
                with mock.patch.object(launcher.q4ks_v0143, 'ready_q4ks_v0143',
                                       return_value=(Path('/candidate'), Path('/candidate/config.json'),
                                                     initial_config)):
                    with self.assertRaisesRegex(SafetyError, '32K validation is restricted to one slot'):
                        launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.43',
                                               parallel=2, batch_groups=2)


class BoundGates(unittest.TestCase):
    def test_candidate_atomic_json_is_private(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            candidate.atomic_json(path, {'lease_id': 'not-a-real-secret'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            candidate._require_private_report(path, 'test evidence')
            self.assertEqual(json.loads(path.read_text()), {'lease_id': 'not-a-real-secret'})
            path.chmod(0o644)
            with self.assertRaisesRegex(SafetyError, 'private operator-owned'):
                candidate._require_private_report(path, 'test evidence')

    def test_prepared_manifest_digest_is_exact_and_rechecked_after_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = candidate.profile_root(root) / 'prepared.json'
            path.parent.mkdir(parents=True)
            raw = b'{"profile":"candidate"}\n'
            path.write_bytes(raw)
            manifest, digest = candidate._json_with_sha256(path, 'test manifest')
            self.assertEqual(manifest, {'profile': 'candidate'})
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            candidate.require_prepared_manifest_unchanged(root, digest)
            path.write_bytes(raw + b' ')
            with self.assertRaisesRegex(SafetyError, 'changed during native validation'):
                candidate.require_prepared_manifest_unchanged(root, digest)

    def test_local_api_client_forces_proxy_free_loopback_transport(self):
        opener = mock.Mock()
        response = object()
        opener.open.return_value = response
        with (mock.patch.dict(os.environ, {'HTTP_PROXY': 'http://proxy.invalid:9999',
                                           'HTTPS_PROXY': 'http://proxy.invalid:9999',
                                           'ALL_PROXY': 'http://proxy.invalid:9999', 'NO_PROXY': ''}),
              mock.patch.object(local_api_client.urllib.request, 'build_opener', return_value=opener) as build):
            request = local_api_client.urllib.request.Request('http://127.0.0.1:8080/health')
            self.assertIs(local_api_client.open_local_api(request, timeout=4), response)
        handlers = build.call_args.args
        proxy_handler = next(handler for handler in handlers
                             if isinstance(handler, local_api_client.urllib.request.ProxyHandler))
        redirect_handler = next(handler for handler in handlers
                                if isinstance(handler, local_api_client.urllib.request.HTTPRedirectHandler))
        self.assertEqual(proxy_handler.proxies, {})
        self.assertIsNone(redirect_handler.redirect_request(
            request, None, 307, 'Temporary Redirect', {}, 'http://attacker.invalid/collect'))
        opener.open.assert_called_once_with(request, timeout=4)
        with self.assertRaisesRegex(ValueError, '127.0.0.1'):
            local_api_client.open_local_api('http://192.168.1.69:8080/health', timeout=4)

    def test_idle_gates_reject_queued_requests(self):
        with (mock.patch.object(validation_runner, 'established_clients', return_value=[]),
              mock.patch.object(validation_runner, 'request_json',
                                return_value={'activity': {'in_flight': 0, 'queued': 1}})):
            with self.assertRaisesRegex(SafetyError, 'active/queued'):
                validation_runner.require_idle('http://127.0.0.1:8080', 8080, None)
        with (mock.patch.object(native_validation, 'established_connections', return_value=[]),
              mock.patch.object(native_validation, 'api_json',
                                return_value={'activity': {'in_flight': 0, 'queued': 1}})):
            with self.assertRaisesRegex(SafetyError, 'active/queued'):
                native_validation.require_idle('http://127.0.0.1:8080', 8080, None)
        status = {'activity': {'in_flight': 0}}
        with (mock.patch.object(validation_runner, 'established_clients', return_value=[]),
              mock.patch.object(validation_runner, 'request_json', return_value=status)):
            self.assertIs(validation_runner.require_idle('http://127.0.0.1:8080', 8080, None), status)
        with (mock.patch.object(native_validation, 'established_connections', return_value=[]),
              mock.patch.object(native_validation, 'api_json', return_value=status)):
            self.assertIs(native_validation.require_idle('http://127.0.0.1:8080', 8080, None), status)

    def test_message_input_count_includes_cache_reads_and_rejects_malformed_usage(self):
        self.assertEqual(native_validation.message_input_tokens(
            {'input_tokens': 7, 'cache_read_input_tokens': 20}), 27)
        self.assertEqual(native_validation.message_input_tokens({'input_tokens': 7}), 7)
        self.assertIsNone(native_validation.message_input_tokens({'input_tokens': True}))
        self.assertIsNone(native_validation.message_input_tokens(
            {'input_tokens': 7, 'cache_read_input_tokens': -1}))
        self.assertIsNone(native_validation.message_input_tokens(None))

    def test_native_token_count_checks_idle_before_and_after_each_call(self):
        with mock.patch.object(native_validation, 'require_idle') as idle, \
             mock.patch.object(native_validation, 'api_json', return_value={'input_tokens': 12}) as request:
            count = native_validation.count_tokens('http://127.0.0.1:8080', 8080, 'model', 'prompt', 'key')
        self.assertEqual(count, 12)
        self.assertEqual(idle.call_args_list, [mock.call('http://127.0.0.1:8080', 8080, 'key'),
                                               mock.call('http://127.0.0.1:8080', 8080, 'key')])
        self.assertEqual(request.call_count, 1)

    def test_native_probe_answers_and_resource_sampler_fail_closed(self):
        own_endpoint = ('127.0.0.1', 12345)
        self.assertEqual(native_validation.parse_endpoint('[::ffff:127.0.0.1]:12345'), own_endpoint)
        self.assertNotEqual(native_validation.parse_endpoint('192.168.1.9:12345'), own_endpoint)
        self.assertTrue(native_validation.exact_arithmetic_answer('437', 'stop'))
        self.assertFalse(native_validation.exact_arithmetic_answer('1437', 'stop'))
        self.assertFalse(native_validation.exact_arithmetic_answer('437', 'length'))
        self.assertTrue(native_validation.exact_color_answer('Magenta.', 'stop'))
        self.assertFalse(native_validation.exact_color_answer('not magenta', 'stop'))
        self.assertFalse(native_validation.exact_color_answer('magenta', 'length'))

        sampler = native_validation.Sampler()
        with self.assertRaisesRegex(SafetyError, 'missing or stale'):
            native_validation.assert_sampler_safe(sampler)
        sampler.samples = 1
        sampler.last_sample_at = time.monotonic() - 10
        sampler.ram_min = candidate.NATIVE_VALIDATION_RAM_FLOOR_MIB + 1
        sampler.gpu_peak = [20000, 21000, 22000, 23000]
        with self.assertRaisesRegex(SafetyError, 'stale'):
            native_validation.assert_sampler_safe(sampler)
        sampler.last_sample_at = time.monotonic()
        sampler.gpu_peak = [20000, 21000, 22000, 23000]
        native_validation.assert_sampler_safe(sampler)
        sampler.errors = ['nvidia-smi failed']
        with self.assertRaisesRegex(SafetyError, 'telemetry failed'):
            native_validation.assert_sampler_safe(sampler)
        sampler.errors = []
        sampler.gpu_peak[0] = candidate.NATIVE_VALIDATION_GPU_CEILING_MIB + 1
        with self.assertRaisesRegex(SafetyError, 'exceeds'):
            native_validation.assert_sampler_safe(sampler)
        sampler.gpu_peak[0] = 20000
        sampler.ram_min = candidate.NATIVE_VALIDATION_RAM_FLOOR_MIB - 1
        with self.assertRaisesRegex(SafetyError, 'RAM'):
            native_validation.assert_sampler_safe(sampler)

    def test_aborted_native_probe_waits_for_server_side_idle(self):
        with (mock.patch.object(native_validation, 'api_json', side_effect=[
                    {'activity': {'in_flight': 1}},
                    {'activity': {'in_flight': 0}}]),
              mock.patch.object(native_validation, 'established_connections', side_effect=[['peer'], []]),
              mock.patch.object(native_validation.time, 'monotonic', side_effect=[0.0, 0.0, 1.0]),
              mock.patch.object(native_validation.time, 'sleep')):
            native_validation.wait_server_idle('http://127.0.0.1:8080', 8080, None, timeout=10)
        with (mock.patch.object(native_validation, 'api_json',
                                return_value={'activity': {'in_flight': 1, 'queued': 0}}),
              mock.patch.object(native_validation, 'established_connections', return_value=['peer']),
              mock.patch.object(native_validation.time, 'monotonic', side_effect=[0.0, 0.0, 5.0]),
              mock.patch.object(native_validation.time, 'sleep')):
            with self.assertRaisesRegex(SafetyError, 'Could not verify server-side idle'):
                native_validation.wait_server_idle('http://127.0.0.1:8080', 8080, None, timeout=5)

    def test_native_report_is_bound_to_config_runner_server_lease_and_probes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pr = root / 'candidate'
            pr.mkdir()
            config = pr / 'config.json'
            config.write_text('{}\n')
            report = {
                'profile': assets.HOST_PROFILE, 'model_id': assets.MODEL_ID,
                'source_commit': assets.SOURCE_COMMIT, 'runtime_version': assets.RUNTIME_VERSION,
                'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
                'validation_runner_sha256': candidate.sha256(Path(native_validation.__file__).resolve()),
                'local_api_client_sha256': candidate.sha256(Path(local_api_client.__file__).resolve()),
                'prepared_manifest_sha256': 'a' * 64,
                'config_sha256': candidate.sha256(config), 'context_configured': 262144,
                'vision_configured': True, 'vision_cpu_only': True, 'passed': True,
                'live_server_identity': {'pid': 12, 'start_ticks': 34, 'uid': 56, 'pgid': 12,
                                         'sid': 12, 'exe': '/tmp/strata-python', 'cmd': ['python']},
                'live_hosting_lease_id': 'a' * 64,
                'health': {'status': 'ok', 'loaded': True, 'max_context': 262144, 'images': True},
                'text_smoke': {'passed': True, 'answer': '437', 'finish_reason': 'stop'},
                'token_count_control': {'passed': True, 'counted_tokens': 10,
                                        'response_input_tokens': 10,
                                        'uncached_response_input_tokens': 7,
                                        'cache_read_input_tokens': 3, 'answer': '437',
                                        'stop_reason': 'end_turn'},
                'safety': {'ram_floor_mib': candidate.NATIVE_VALIDATION_RAM_FLOOR_MIB,
                           'gpu_memory_ceiling_mib': candidate.NATIVE_VALIDATION_GPU_CEILING_MIB,
                           'max_request_seconds': candidate.NATIVE_VALIDATION_MAX_REQUEST_S},
                'vision': {'passed': True, 'expected': 'magenta', 'answer': 'Magenta',
                           'finish_reason': 'stop', 'cpu_only_configured': True},
                'full_context': {'passed': True, 'count_match': True, 'counted_prompt_tokens': 260000,
                                 'response_input_tokens': 260000,
                                 'uncached_response_input_tokens': 260000,
                                 'cache_read_input_tokens': 0, 'marker_retrieved': True,
                                 'expected_tail_marker': 'marker', 'answer': 'marker',
                                 'stop_reason': 'end_turn', 'elapsed_milliseconds': 100000},
                'resource_measurement': {
                    'ram_available_min_mib': candidate.NATIVE_VALIDATION_RAM_FLOOR_MIB + 1000,
                    'gpu_memory_peak_mib_by_gpu': [20000, 21000, 22000, 23000],
                    'gpu_memory_peak_aggregate_mib': 86000, 'sampler_samples': 10,
                    'last_sample_age_ms': 100, 'sampler_thread_stopped': True,
                    'sampler_error_count': 0, 'sampler_errors': []},
            }
            native_path = pr / 'validation-native.json'
            native_path.write_text(json.dumps(report))
            native_path.chmod(0o600)
            prepared_manifest = {'local_api_client_sha256': candidate.sha256(Path(local_api_client.__file__).resolve())}
            prepared_manifest_sha256 = 'a' * 64

            def evidence_summary():
                with (mock.patch.object(candidate, 'profile_root', return_value=pr),
                      mock.patch.object(candidate, '_prepared', return_value=(
                          prepared_manifest, None, None, None, None, None, None, prepared_manifest_sha256))):
                    return candidate.candidate_evidence_summary(root)

            summary = evidence_summary()
            self.assertTrue(any('native/CPU-vision probes match this config' in line for line in summary))
            report['token_count_control']['cache_read_input_tokens'] = 4
            native_path.write_text(json.dumps(report))
            inconsistent_usage = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in inconsistent_usage))
            report['token_count_control']['cache_read_input_tokens'] = 3
            native_path.write_text(json.dumps(report))
            native_path.chmod(0o644)
            nonprivate = evidence_summary()
            self.assertTrue(any('unreadable' in line for line in nonprivate))
            native_path.chmod(0o600)
            config.write_text('{"changed":true}\n')
            stale = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in stale))
            report['config_sha256'] = candidate.sha256(config)
            report['resource_measurement']['gpu_memory_peak_mib_by_gpu'][0] = \
                candidate.NATIVE_VALIDATION_GPU_CEILING_MIB + 1
            native_path.write_text(json.dumps(report))
            over_budget = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in over_budget))
            report['resource_measurement']['gpu_memory_peak_mib_by_gpu'][0] = 20000
            report['resource_measurement']['gpu_memory_peak_aggregate_mib'] = 1000
            native_path.write_text(json.dumps(report))
            inconsistent_aggregate = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in inconsistent_aggregate))
            report['resource_measurement']['gpu_memory_peak_aggregate_mib'] = 86000
            report['full_context']['elapsed_milliseconds'] = \
                candidate.NATIVE_VALIDATION_MAX_REQUEST_S * 1000 + 1
            native_path.write_text(json.dumps(report))
            timeout_report = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in timeout_report))
            report['full_context']['elapsed_milliseconds'] = 100000
            report['token_count_control']['answer'] = '1437'
            native_path.write_text(json.dumps(report))
            false_arithmetic = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in false_arithmetic))
            report['token_count_control']['answer'] = '437'
            report['local_api_client_sha256'] = '0' * 64
            native_path.write_text(json.dumps(report))
            helper_mismatch = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in helper_mismatch))
            report['local_api_client_sha256'] = candidate.sha256(Path(local_api_client.__file__).resolve())
            prepared_manifest['local_api_client_sha256'] = '0' * 64
            native_path.write_text(json.dumps(report))
            manifest_helper_mismatch = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in manifest_helper_mismatch))
            prepared_manifest['local_api_client_sha256'] = candidate.sha256(
                Path(local_api_client.__file__).resolve())
            prepared_manifest_sha256 = '0' * 64
            native_path.write_text(json.dumps(report))
            manifest_mismatch = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in manifest_mismatch))
            prepared_manifest_sha256 = 'a' * 64
            report['live_hosting_lease_id'] = 'not-a-lease'
            native_path.write_text(json.dumps(report))
            unbound = evidence_summary()
            self.assertTrue(any('stale/incomplete' in line for line in unbound))

    def test_32k_validation_requires_exact_candidate_pin_config_checks_memory_and_mtp(self):
        runner_sha = candidate.sha256(Path(candidate.__file__).with_name('orca_q4ks_v0143_validation.py'))
        helper_sha = candidate.sha256(Path(candidate.__file__).with_name('local_api_client.py'))
        manifest = {'runtime_assets': [{'sha256': 'engine-sha'}], 'validation_runner_sha256': runner_sha,
                    'local_api_client_sha256': helper_sha}
        report = {
            'profile': assets.PROFILE, 'validation_runner_sha256': runner_sha,
            'local_api_client_sha256': helper_sha, 'model_id': assets.MODEL_ID,
            'source_commit': assets.SOURCE_COMMIT, 'runtime_version': assets.RUNTIME_VERSION,
            'runtime_sha256': 'engine-sha',
            'asset_sha256': {name: value[1] for name, value in assets.ASSETS.items()},
            'config_sha256': 'config-sha', 'context': 32768, 'parallel': 1, 'text_only': True,
            'passed': True, 'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
            'checks': {'health': True, 'arithmetic': True, 'strict_json': True,
                       'tool_call': True, 'memory_recorded': True,
                       'conversation_cache_configuration': True, 'conversation_reuse': True},
            'memory_peak_mib': 10000, 'gpu_memory_peak_mib_by_gpu': [9000, 9500, 10000, 9800],
            'gpu_memory_peak_aggregate_mib': 38300, 'ram_available_min_mib': 40000,
            'live_status': {'context': 32768, 'vision_enabled': False, 'serving_slots': 1,
                            'conversation_cache': {'enabled': True, 'budget_mib': 8192, 'slots': 4,
                                                  'park_events': 1, 'restored_conversations': 1,
                                                  'reused_requests': 1, 'reused_tokens': 128}},
            'validation_server_identity': {'pid': 12, 'start_ticks': 34, 'uid': 56, 'pgid': 12, 'sid': 12,
                                           'exe': '/tmp/strata-python', 'cmd': ['python']},
            'validation_hosting_lease_id': 'a' * 64,
            'mtp': {'drafted_tokens': 10, 'accepted_tokens': 8},
        }
        candidate._validation_report(Path('/tmp'), report, 'config-sha', manifest)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pr = candidate.profile_root(root)
            pr.mkdir(parents=True)
            config = pr / 'config.json'
            config.write_text('{}\n')
            evidence = dict(report, config_sha256=candidate.sha256(config),
                            recorded_server_identity=report['validation_server_identity'],
                            recorded_hosting_lease_id=report['validation_hosting_lease_id'])
            validation_path = pr / 'validation-32k.json'
            validation_path.write_text(json.dumps(evidence))
            validation_path.chmod(0o600)
            prepared = (manifest, None, None, None, None, None, None, 'manifest-sha')
            with mock.patch.object(candidate, 'profile_root', return_value=pr), \
                 mock.patch.object(candidate, '_prepared', return_value=prepared):
                summary = candidate.candidate_evidence_summary(root)
            self.assertTrue(any('32K result is an operator self-attestation' in line for line in summary), repr(summary))
            validation_path.chmod(0o644)
            with mock.patch.object(candidate, 'profile_root', return_value=pr), \
                 mock.patch.object(candidate, '_prepared', return_value=prepared):
                private_summary = candidate.candidate_evidence_summary(root)
            self.assertTrue(any('unreadable/stale' in line for line in private_summary))
        bad = dict(report, context=262144)
        with self.assertRaisesRegex(SafetyError, '32K text-only'):
            candidate._validation_report(Path('/tmp'), bad, 'config-sha', manifest)
        forged = dict(report, validation_runner_sha256='0' * 64)
        with self.assertRaisesRegex(SafetyError, '32K text-only'):
            candidate._validation_report(Path('/tmp'), forged, 'config-sha', manifest)
        bad = dict(report, mtp={'drafted_tokens': 0, 'accepted_tokens': 0})
        with self.assertRaisesRegex(SafetyError, 'MTP drafted/accepted'):
            candidate._validation_report(Path('/tmp'), bad, 'config-sha', manifest)

    def test_validation_runner_uses_private_api_key_header_and_per_response_mtp(self):
        import io
        with mock.patch.object(validation_runner, 'open_local_api',
                                return_value=io.BytesIO(b'{"ok":true}')) as direct_open:
            self.assertEqual(validation_runner.request_json('http://127.0.0.1:8080', '/health',
                                                            api_key='unit-test-key'), {'ok': True})
        request = direct_open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer unit-test-key')
        self.assertEqual(validation_runner._record_mtp({'timings': {'draft_n': 10, 'draft_n_accepted': 8}}), (10, 8))
        self.assertTrue(validation_runner._arithmetic_pass(' 437\n', 'stop'))
        self.assertFalse(validation_runner._arithmetic_pass('4370', 'stop'))
        self.assertFalse(validation_runner._arithmetic_pass('The answer is 437', 'stop'))
        with self.assertRaisesRegex(SafetyError, 'MTP counters'):
            validation_runner._record_mtp({'timings': {'draft_n': 10, 'draft_n_accepted': 11}})

    def test_cache_validation_alternates_full_histories_and_requires_metrics_restore_and_reuse(self):
        before = {'conversation_cache': {'enabled': True, 'budget_mib': 8192, 'slots': 4,
                                         'parks': 0, 'restores': 0, 'requests_reused': 0,
                                         'reused_tokens': 0}}
        after = {'conversation_cache': {'enabled': True, 'budget_mib': 8192, 'slots': 4,
                                        'parks': 2, 'restores': 2, 'requests_reused': 2,
                                        'reused_tokens': 256}}
        calls = []
        def completion(base, body, api_key):
            calls.append({**body, 'messages': [dict(message) for message in body['messages']]})
            text = ('READY', 'READY', 'SILVER', 'ORANGE')[len(calls) - 1]
            return {}, {'message': {'content': text}, 'finish_reason': 'stop'}
        with mock.patch.object(validation_runner, '_completion', side_effect=completion), \
             mock.patch.object(validation_runner, 'request_json', return_value=after) as request, \
             mock.patch.object(validation_runner, 'require_idle') as idle:
            result = validation_runner._probe_conversation_cache(
                'http://127.0.0.1:8080', 8080, {'model': assets.MODEL_ID, 'max_tokens': 32}, 'key', before)
        self.assertEqual(idle.call_count, 8)
        self.assertTrue(all('conversation_id' not in call for call in calls))
        self.assertEqual([len(call['messages']) for call in calls], [1, 1, 3, 3])
        self.assertEqual(calls[2]['messages'][1], {'role': 'assistant', 'content': 'READY'})
        self.assertEqual(calls[3]['messages'][1], {'role': 'assistant', 'content': 'READY'})
        self.assertEqual(result['park_events'], 2)
        self.assertEqual(result['restored_conversations'], 2)
        self.assertEqual(result['reused_requests'], 2)
        self.assertEqual(result['reused_tokens'], 256)
        request.assert_called_once_with('http://127.0.0.1:8080', '/metrics', timeout=8, api_key='key')
        disabled = {'conversation_cache': {'enabled': False, 'budget_mib': 8192, 'slots': 4,
                                            'parks': 1, 'restores': 1, 'requests_reused': 1,
                                            'reused_tokens': 128}}
        with self.assertRaisesRegex(SafetyError, 'enabled'):
            validation_runner._conversation_cache_metrics(disabled)
        calls.clear()
        with mock.patch.object(validation_runner, '_completion', side_effect=completion), \
             mock.patch.object(validation_runner, 'request_json', return_value=before), \
             mock.patch.object(validation_runner, 'require_idle'):
            with self.assertRaisesRegex(SafetyError, 'did not park and restore a conversation'):
                validation_runner._probe_conversation_cache(
                    'http://127.0.0.1:8080', 8080,
                    {'model': assets.MODEL_ID, 'max_tokens': 32}, 'key', before)

    def test_validation_runner_refuses_a_concurrent_validation_lock(self):
        import fcntl
        root = Path(tempfile.mkdtemp())
        try:
            lock_path = root / 'q4ks-v0143-validation.lock'
            with lock_path.open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(validation_runner, 'strata_state', return_value=root):
                    with self.assertRaisesRegex(SafetyError, 'already active'):
                        with validation_runner._validation_lock():
                            self.fail('second validation unexpectedly acquired the lock')
        finally:
            import shutil
            shutil.rmtree(root)

    def test_validation_has_no_caller_supplied_report_recorder_and_labels_operator_attestation(self):
        import inspect
        self.assertFalse(hasattr(candidate, 'record_32k_validation'))
        self.assertFalse(hasattr(candidate, '_store_32k_validation'))
        self.assertEqual(tuple(inspect.signature(validation_runner._run_locked).parameters), ('root',))
        runner_sha = candidate.sha256(Path(candidate.__file__).with_name('orca_q4ks_v0143_validation.py'))
        helper_sha = candidate.sha256(Path(candidate.__file__).with_name('local_api_client.py'))
        manifest = {'runtime_assets': [{'sha256': 'engine-sha'}], 'validation_runner_sha256': runner_sha,
                    'local_api_client_sha256': helper_sha}
        report = {
            'profile': assets.PROFILE, 'validation_runner_sha256': runner_sha,
            'local_api_client_sha256': helper_sha, 'model_id': assets.MODEL_ID,
            'source_commit': assets.SOURCE_COMMIT, 'runtime_version': assets.RUNTIME_VERSION,
            'runtime_sha256': 'engine-sha',
            'asset_sha256': {name: value[1] for name, value in assets.ASSETS.items()},
            'config_sha256': 'config-sha', 'context': 32768, 'parallel': 1, 'text_only': True,
            'passed': True, 'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
            'checks': {'health': True, 'arithmetic': True, 'strict_json': True,
                       'tool_call': True, 'memory_recorded': True,
                       'conversation_cache_configuration': True, 'conversation_reuse': True},
            'memory_peak_mib': 10000, 'gpu_memory_peak_mib_by_gpu': [9000, 9500, 10000, 9800],
            'gpu_memory_peak_aggregate_mib': 38300, 'ram_available_min_mib': 40000,
            'live_status': {'context': 32768, 'vision_enabled': False, 'serving_slots': 1,
                            'conversation_cache': {'enabled': True, 'budget_mib': 8192, 'slots': 4,
                                                  'park_events': 1, 'restored_conversations': 1,
                                                  'reused_requests': 1, 'reused_tokens': 128}},
            'validation_server_identity': {'pid': 12, 'start_ticks': 34, 'uid': 56, 'pgid': 12, 'sid': 12,
                                           'exe': '/tmp/strata-python', 'cmd': ['python']},
            'validation_hosting_lease_id': 'a' * 64,
            'mtp': {'drafted_tokens': 10, 'accepted_tokens': 8},
        }
        candidate._validation_report(Path('/tmp'), report, 'config-sha', manifest)
        unmarked = dict(report)
        unmarked.pop('evidence_class')
        with self.assertRaisesRegex(SafetyError, 'pinned 32K'):
            candidate._validation_report(Path('/tmp'), unmarked, 'config-sha', manifest)

    def test_native_migration_requires_candidate_32k_evidence_and_leaves_config_untouched(self):
        root = Path(tempfile.mkdtemp())
        try:
            lock = root / 'prepare.lock'
            lock.touch()
            config = candidate.make_config(root)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            before = config_path.read_bytes()
            with mock.patch.object(candidate, '_profile_lock', return_value=lock), \
                 mock.patch.object(candidate, 'docker_empty'), \
                 mock.patch.object(candidate, 'ready_q4ks_v0143', return_value=(root, config_path, config)):
                with self.assertRaisesRegex(SafetyError, '32K validation'):
                    candidate.configure_native_context(root)
            self.assertEqual(config_path.read_bytes(), before)
            self.assertFalse((root / 'native-context-migration.json').exists())
        finally:
            import shutil
            shutil.rmtree(root)

    def test_candidate_vision_provenance_survives_projector_hardlink_count_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pr = candidate.profile_root(root)
            asset_dir = pr / 'assets'
            asset_dir.mkdir(parents=True)
            source = root / 'data/projector.gguf'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'projector fixture')
            source.chmod(0o444)
            target = asset_dir / candidate.baseline_profile.VISION_PROJECTOR_FILENAME
            os.link(source, target)
            helper = root / 'runtime/engine/strata-vision'
            helper.parent.mkdir(parents=True)
            helper.write_bytes(b'vision helper fixture')
            helper_sha = sha(helper.read_bytes())
            source_info = source.stat()
            source_record = {'path': str(source), 'bytes': source_info.st_size,
                             'mtime_ns': source_info.st_mtime_ns, 'sha256': sha(source.read_bytes())}
            runtime_record = {'path': str(helper), 'sha256': helper_sha}
            sidecar = {
                'profile': candidate.PROFILE, 'model_id': candidate.MODEL_ID,
                'source_commit': candidate.SOURCE_COMMIT, 'runtime_version': candidate.RUNTIME_VERSION,
                'helper': candidate.record(helper, helper_sha), 'projector_source': str(source),
                'projector_source_record': source_record,
                'projector': candidate.record(target, candidate.baseline_profile.VISION_PROJECTOR_SHA256),
            }
            (pr / 'vision-prepared.json').write_text(json.dumps(sidecar))
            extra = root / 'extra-projector-link.gguf'
            with mock.patch.object(candidate.baseline_profile, '_verified_iq3_projector',
                                   return_value=(source, source_record)), \
                 mock.patch.object(candidate.runtimes, 'read_manifest',
                                   return_value={'runtime_assets': [runtime_record]}):
                os.link(target, extra)
                candidate._verified_vision_sidecar(root, pr, helper)
                extra.unlink()
                candidate._verified_vision_sidecar(root, pr, helper)

    def test_vision_rollback_does_not_require_candidate_vision_artifacts(self):
        root = Path(tempfile.mkdtemp())
        try:
            pr = candidate.profile_root(root)
            pr.mkdir(parents=True)
            lock = pr / 'prepare.lock'
            lock.touch()
            config_path = pr / 'config.json'
            config_path.write_text('{}\n')
            migration_path = pr / 'native-context-migration.json'
            migration_path.write_text('{}\n')
            text = {'args': ['--max-context', '262144']}
            vision = {'args': ['--max-context', '262144', '--vision']}
            prepared = ({}, root, pr, vision, {}, {}, root / 'broken-helper', 'manifest-sha')
            with mock.patch.object(candidate, '_profile_lock', return_value=lock), \
                 mock.patch.object(candidate, 'docker_empty'), \
                 mock.patch.object(candidate, '_prepared', return_value=prepared) as prepared_gate, \
                 mock.patch.object(candidate, '_check_parity'), \
                 mock.patch.object(candidate, '_config_with_vision_paths', return_value=vision), \
                 mock.patch.object(candidate, 'make_config', return_value=text), \
                 mock.patch.object(candidate, '_update_native') as update, \
                 mock.patch.object(candidate, '_validate_vision_sidecar') as validate:
                candidate.rollback_cpu_vision(root)
            self.assertTrue(prepared_gate.call_args.kwargs['allow_broken_vision'])
            validate.assert_not_called()
            self.assertEqual(update.call_args.args[-1], text)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_native_text_readiness_ignores_broken_vision_provenance_until_vision_is_enabled(self):
        root = Path(tempfile.mkdtemp())
        try:
            pr = root / 'profiles' / assets.PROFILE
            pr.mkdir(parents=True)
            sidecar = pr / 'vision-prepared.json'
            sidecar.write_text('{}\n')
            text = {'args': ['--max-context', '262144']}
            vision = {'args': ['--max-context', '262144', '--vision']}
            helper = root / 'helper'
            with mock.patch.object(candidate, 'make_config', return_value={'args': ['--max-context', '262144']}), \
                 mock.patch.object(candidate, '_config_with_vision_paths', return_value={
                     'args': ['--max-context', '262144', '--vision']}), \
                 mock.patch.object(candidate, '_validate_vision_sidecar') as validate:
                self.assertEqual(candidate._native_context_expected_config(root, pr, text, helper),
                                 {'args': ['--max-context', '262144']})
                validate.assert_not_called()
                expected_vision = candidate._native_context_expected_config(root, pr, vision, helper)
                self.assertEqual(expected_vision['args'], ['--max-context', '262144', '--vision'])
                validate.assert_called_once_with(root, pr, helper)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_numeric_pid_only_is_not_a_candidate_identity(self):
        self.assertFalse(candidate._valid_identity({'pid': 123, 'start_ticks': 456}))
        self.assertFalse(candidate._valid_identity({'pid': 123, 'start_ticks': 456,
                                                    'exe': '/usr/bin/python3'}))

    def test_candidate_runtime_helper_must_match_its_manifest_hash(self):
        import strata_runtime as runtimes
        root = Path(tempfile.mkdtemp())
        try:
            runtime = assets.pinned_runtime_source(root)
            (runtime / 'engine').mkdir(parents=True)
            helper = runtime / 'engine/strata-vision'
            helper.write_bytes(b'exact helper')
            engine = runtime / 'engine/strata'
            engine.write_bytes(b'exact engine')
            (runtime / 'engine/BUILD.json').write_text(json.dumps(
                {'version': '0.1.43', 'vision_src': assets.VISION_SOURCE_SHA256}))
            llama = runtime / 'third_party/llama.cpp'
            llama.mkdir(parents=True)
            (llama / 'CMakeLists.txt').write_text('pinned')
            (runtime / 'setup.py').write_text('# exact pinned source')
            env = root / 'runtimes/0.1.43/python'
            python = env / 'bin/python'
            python.parent.mkdir(parents=True)
            python.write_bytes(b'python')
            (runtime / '.venv').symlink_to(env, target_is_directory=True)
            (root / 'runtimes/0.1.43/prepared.json').write_text('{}\n')

            def record(path):
                info = path.stat()
                return {'path': str(path), 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                        'sha256': candidate.sha256(path)}

            runtime_manifest = {'version': '0.1.43', 'source_commit': runtimes.PINS[assets.RUNTIME_VERSION],
                                'llama_commit': runtimes.LLAMA_PIN, 'source': str(runtime),
                                'cuda_arch': 86, 'cuda_toolkit': '/usr/local/cuda-12.9',
                                'runtime_assets': [record(engine), record(helper)],
                                'python_environment': str(env), 'python_assets': []}
            with mock.patch.object(candidate.runtimes, 'read_manifest', return_value=runtime_manifest), \
                 mock.patch.object(candidate.runtimes, 'command_output',
                                   side_effect=[runtimes.PINS[assets.RUNTIME_VERSION], '']), \
                 mock.patch.object(candidate.subprocess, 'check_output',
                                   return_value=assets.VISION_SOURCE_SHA256), \
                 mock.patch.object(candidate, 'hash_tree', return_value='llama-tree-hash'):
                resolved = candidate._runtime_material(root)
            self.assertEqual(resolved[0], runtime)
            self.assertEqual(resolved[3]['sha256'], sha(b'exact helper'))
            runtime_manifest['runtime_assets'][1]['sha256'] = '0' * 64
            runtime_manifest['runtime_assets'][1] = record(helper)
            with mock.patch.object(candidate.runtimes, 'read_manifest', return_value=runtime_manifest), \
                 mock.patch.object(candidate.runtimes, 'command_output',
                                   side_effect=[runtimes.PINS[assets.RUNTIME_VERSION], '']), \
                 mock.patch.object(candidate.subprocess, 'check_output', return_value='6f47618c713ce0a2'), \
                 mock.patch.object(candidate, 'hash_tree', return_value='llama-tree-hash'):
                with self.assertRaisesRegex(SafetyError, 'exact .43 pin'):
                    candidate._runtime_material(root)
            runtime_manifest['runtime_assets'][1]['sha256'] = '0' * 64
            with mock.patch.object(candidate.runtimes, 'read_manifest', return_value=runtime_manifest), \
                 mock.patch.object(candidate.runtimes, 'command_output',
                                   side_effect=[runtimes.PINS[assets.RUNTIME_VERSION], '']), \
                 mock.patch.object(candidate.subprocess, 'check_output',
                                   return_value=assets.VISION_SOURCE_SHA256), \
                 mock.patch.object(candidate, 'hash_tree', return_value='llama-tree-hash'):
                with self.assertRaisesRegex(SafetyError, 'hash changed'):
                    candidate._runtime_material(root)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_candidate_source_state_allows_only_pinned_links_and_build_outputs(self):
        source = Path('/tmp/orca-q4ks-v0143/source')
        status = '?? .venv\0?? third_party/llama.cpp\0?? engine/strata\0' \
                 '?? build-orca-q4ks-v0143/CMakeCache.txt\0'
        with mock.patch.object(candidate.subprocess, 'check_output', return_value=status):
            candidate._assert_candidate_source_status(source, built=True)
        with mock.patch.object(candidate.subprocess, 'check_output', return_value=status + '?? stray.py\0'):
            with self.assertRaisesRegex(SafetyError, 'unexpected untracked input'):
                candidate._assert_candidate_source_status(source, built=True)
        with mock.patch.object(candidate.subprocess, 'check_output', return_value=' M serve/server.py\0'):
            with self.assertRaisesRegex(SafetyError, 'tracked modification'):
                candidate._assert_candidate_source_status(source, built=True)

    def test_candidate_hash_tree_is_content_bound_and_rejects_symlinks(self):
        root = Path(tempfile.mkdtemp())
        try:
            tree = root / 'llama.cpp'
            (tree / 'ggml').mkdir(parents=True)
            source = tree / 'ggml/CMakeLists.txt'
            source.write_text('pinned source')
            first = candidate.hash_tree(tree)
            source.write_text('changed source')
            self.assertNotEqual(first, candidate.hash_tree(tree))
            link = tree / 'escape'
            link.symlink_to(source)
            with self.assertRaisesRegex(SafetyError, 'symlink'):
                candidate.hash_tree(tree)
        finally:
            import shutil
            shutil.rmtree(root)

    def test_prepare_refuses_root_before_creating_candidate_files(self):
        root = Path(tempfile.mkdtemp()) / 'candidate'
        try:
            with mock.patch.object(prepare, 'BASE', root), \
                 mock.patch.object(prepare, 'PROFILE_ROOT', root / 'profiles' / assets.PROFILE), \
                 mock.patch.object(prepare.os, 'geteuid', return_value=0):
                with self.assertRaisesRegex(SafetyError, 'normal operator; do not use sudo'):
                    prepare.prepare()
            self.assertFalse(root.exists())
        finally:
            import shutil
            shutil.rmtree(root.parent)

    def test_prepare_requires_hostllm_pause_lease_before_candidate_files(self):
        root = Path(tempfile.mkdtemp()) / 'candidate'
        try:
            with mock.patch.object(prepare, 'BASE', root), \
                 mock.patch.object(prepare, 'PROFILE_ROOT', root / 'profiles' / assets.PROFILE), \
                 mock.patch.object(prepare.os, 'geteuid', return_value=1000), \
                 mock.patch.object(prepare.hosting, 'read_lease', return_value=None):
                with self.assertRaisesRegex(SafetyError, 'held HostLLM pause lease'):
                    prepare.prepare()
            self.assertFalse(root.exists())
        finally:
            import shutil
            shutil.rmtree(root.parent)

    def test_parity_refuses_root_before_hostllm_or_gpu_inspection(self):
        with mock.patch.object(prepare.os, 'geteuid', return_value=0), \
             mock.patch.object(prepare, 'docker_empty') as docker, \
             mock.patch.object(prepare.hosting, 'read_lease') as lease:
            with self.assertRaisesRegex(SafetyError, 'normal operator; do not use sudo'):
                prepare.verify_parity('a' * 64, 8080)
        docker.assert_not_called()
        lease.assert_not_called()
