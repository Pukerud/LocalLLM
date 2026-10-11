"""CPU-only Q4_K_S profile gates; no real model downloads/builds/GPU tests."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux controller imports')
import orca_q4ks_assets as assets
import orca_q4ks_profile as profile
import prepare_orca_q4ks as prepare
from engine_safety import SafetyError


def sha(data):
    return hashlib.sha256(data).hexdigest()


LEASE_ID = 'a' * 64


class PinnedQ4Assets(unittest.TestCase):
    def test_exact_publisher_shards_and_separate_runtime_pin(self):
        self.assertEqual(assets.TOTAL_BYTES, 111769915520)
        self.assertEqual(list(assets.ASSETS.values()), [
            (44888775360, '8f940bf64c9a62a4dce1935999ac8ed287bb90ff5a4e5cc22ddac767cd56557d'),
            (44934400672, 'd19d80982e5fa2416a21342be71e5d8041c0c088720ac5cc54bca79a9eafb10c'),
            (21946739488, '6be547f9575cfdf23314426d62fb65ef4052841dd156f7b77dd79a93da3e4087'),
        ])
        self.assertEqual(assets.SOURCE_COMMIT, '6f32ec070f23ced9f50e704d854d775da52591ab')
        self.assertEqual(assets.VISION_PROJECTOR_FILENAME, 'mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf')
        self.assertEqual(assets.VISION_PROJECTOR_BYTES, 907543296)
        self.assertEqual(assets.VISION_PROJECTOR_SHA256,
                         'f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9')
        self.assertEqual(assets.CMAKE_VERSION, '3.31.6')
        self.assertEqual(assets.CMAKE_WHEEL_BYTES, 27800904)
        self.assertEqual(assets.CMAKE_WHEEL_SHA256,
                         '1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec')
        self.assertNotEqual(assets.runtime_root(Path('/tmp/data')),
                            Path('/tmp/data/runtimes/0.1.39'))
        self.assertNotIn(assets.PROFILE, __import__('strata_runtime').PROFILES)


class Q4Profile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = assets.runtime_root(self.root) / 'source'
        self.pinned_source = assets.pinned_runtime_source(self.root)
        self.pr = self.root / 'profiles' / assets.PROFILE
        self.pr.mkdir(parents=True)
        self.pack = self.root / 'data/packs' / assets.PROFILE
        self.paths = [self.root / 'data/models' / assets.PROFILE / name for name in assets.ASSETS]
        self.fixture_assets = {name: (12, sha(b'pinned bytes')) for name in assets.ASSETS}
        self.fixture_projector = b'verified fixture projector'
        self.fixture_iq3_assets = dict(profile.IQ3XXS_ASSETS)
        self.fixture_iq3_assets[profile.VISION_PROJECTOR_FILENAME] = (
            len(self.fixture_projector), sha(self.fixture_projector))
        self.pinned_manifest = {'runtime_assets': []}
        def command_output(cmd):
            if 'diff' in cmd:
                return ''
            if cmd[0].endswith('/bin/cmake'):
                return f'cmake version {assets.CMAKE_VERSION}\n'
            if cmd[0].endswith('/bin/nvcc'):
                return 'nvcc: NVIDIA CUDA compiler\nBuild cuda_12.9\n'
            return assets.BASE_SOURCE_COMMIT if str(self.root / 'source-99f3dbd0b21d') in cmd else assets.SOURCE_COMMIT

        self.patchers = [
            mock.patch.object(profile, 'ASSETS', self.fixture_assets),
            mock.patch.object(profile, 'IQ3XXS_ASSETS', self.fixture_iq3_assets),
            mock.patch.object(profile, 'VISION_PROJECTOR_BYTES', len(self.fixture_projector)),
            mock.patch.object(profile, 'VISION_PROJECTOR_SHA256', sha(self.fixture_projector)),
            mock.patch.object(prepare, 'ASSETS', self.fixture_assets),
            mock.patch.object(prepare, 'BASE', self.root),
            mock.patch.object(profile.runtimes, 'prepared_runtime',
                              return_value=(self.pinned_source, self.pinned_manifest)),
            mock.patch.object(profile, 'command_output', side_effect=command_output),
        ]
        for patcher in self.patchers:
            patcher.start()

        def put(path, data=b'pinned bytes'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o444)
            st = path.stat()
            return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                    'ctime_ns': st.st_ctime_ns, 'device': st.st_dev, 'inode': st.st_ino,
                    'mode': st.st_mode & 0o7777, 'uid': st.st_uid, 'gid': st.st_gid,
                    'sha256': sha(data)}

        asset_records = [put(path) for path in self.paths]
        self.engine = self.source / 'engine/strata'
        runtime_records = [put(self.engine, b'q4 engine')]
        self.tests = {
            'native_expert_parity': self.source / 'build-orca-q4ks/native_expert_parity',
            'ple_q5_parity': self.source / 'build-orca-q4ks/ple_q5_parity',
        }
        test_records = [put(path, path.name.encode()) for path in self.tests.values()]
        put(self.pinned_source / '.venv/bin/python')
        self.vision_engine = self.pinned_source / 'engine/strata-vision'
        self.pinned_manifest['runtime_assets'] = [put(self.vision_engine, b'pinned .39 vision engine')]
        self.cmake = assets.cmake_binary(self.root)
        self.cmake_wheel = assets.cmake_wheel(self.root)
        put(self.cmake, b'pinned q4 cmake binary')
        self.cmake.chmod(0o555)
        put(self.cmake_wheel, b'pinned q4 cmake wheel')
        real_sha256 = profile.sha256
        def pinned_tool_hash(path):
            path = Path(path)
            if path == self.cmake:
                return 'pinned-cmake-binary-sha256'
            if path == self.cmake_wheel:
                return assets.CMAKE_WHEEL_SHA256
            return real_sha256(path)
        self.patchers.append(mock.patch.object(profile, 'sha256', side_effect=pinned_tool_hash))
        self.patchers[-1].start()
        pinned_llama = self.pinned_source / 'third_party/llama.cpp'
        (pinned_llama / 'ggml').mkdir(parents=True)
        (pinned_llama / 'ggml/CMakeLists.txt').write_text('')
        (pinned_llama / 'gguf-py').mkdir()
        q4_llama = self.source / 'third_party/llama.cpp'
        q4_llama.parent.mkdir(parents=True)
        q4_llama.symlink_to(pinned_llama, target_is_directory=True)
        (self.source / '.venv').symlink_to((self.pinned_source / '.venv').resolve(), target_is_directory=True)
        mtp_record = put(self.root / 'data/mtp/rt/experts.bin')
        mtp_tensor_record = put(self.root / 'data/mtp/tensors/draft.bin')
        base_source = self.root / 'source-99f3dbd0b21d'
        (base_source / 'tools').mkdir(parents=True)
        expert_record = put(base_source / 'data/expert-profile.bin')
        (base_source / 'tools/mtp_fetch.py').write_text("SHA256 = {'draft': '" + sha(b'pinned bytes') + "'}\n")
        required = ['dense.bin', 'index.txt', 'native_experts.txt', 'compat-bf16.json',
                    'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja',
                    'tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/token_type.json']
        pack_records = [put(self.pack / relative) for relative in required]
        self.cache = self.source / 'build-orca-q4ks/CMakeCache.txt'
        put(self.cache, (f'CMAKE_COMMAND:INTERNAL={self.cmake}\n' + profile.CMAKE_CACHE_CONTRACT).encode())
        (self.pr / 'prepare.lock').touch()
        self.cfg_path = self.pr / 'config.json'
        self.cfg = profile.make_config(self.root, context=32768)
        self.cfg_path.write_text(json.dumps(self.cfg))
        self.manifest = {
            'profile': assets.PROFILE, 'model_id': assets.MODEL_ID,
            'model_repository': assets.REPOSITORY, 'model_revision': assets.REVISION,
            'source_commit': assets.SOURCE_COMMIT, 'source': str(self.source),
            'runtime_version': '0.1.39', 'config': str(self.cfg_path),
            'runtime_assets': runtime_records, 'test_binaries': test_records,
            'assets': asset_records, 'pack': str(self.pack), 'pack_assets': pack_records,
            'mtp_runtime': mtp_record, 'mtp_tensors': [mtp_tensor_record],
            'mtp_source_commit': assets.BASE_SOURCE_COMMIT, 'expert_profile': expert_record,
            'build_options': assets.BUILD_OPTIONS, 'llama_cpp_commit': __import__('strata_runtime').LLAMA_PIN,
            'llama_cpp_source': str(pinned_llama.resolve()),
            'build_toolchain': {'cmake': str(self.cmake),
                                'cmake_version': f'cmake version {assets.CMAKE_VERSION}',
                                'cmake_sha256': 'pinned-cmake-binary-sha256',
                                'cmake_wheel': str(self.cmake_wheel),
                                'cmake_wheel_sha256': assets.CMAKE_WHEEL_SHA256,
                                'nvcc': '/usr/local/cuda-12.9/bin/nvcc',
                                'nvcc_version': 'Build cuda_12.9'},
            'python_environment': str((self.pinned_source / '.venv').resolve()),
            'runtime_selection_independent': True, 'inference_tested': False,
            'vision_enabled': False, 'configured_context': 32768,
        }
        (self.pr / 'prepared.json').write_text(json.dumps(self.manifest))
        self.write_parity()

    def write_parity(self):
        (self.pr / 'parity.json').write_text(json.dumps({
            'profile': assets.PROFILE, 'source_commit': assets.SOURCE_COMMIT,
            'asset_sha256': {name: value[1] for name, value in self.fixture_assets.items()},
            'runtime_sha256': self.manifest['runtime_assets'][0]['sha256'],
            'tests': {'native_expert_parity': {'passed': True}, 'ple_q5_parity': {'passed': True}},
        }))

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tmp.cleanup()

    def test_only_separate_pinned_text_32k_profile_is_ready(self):
        source, path, cfg = profile.ready_q4ks(self.root)
        self.assertEqual(source, self.source)
        self.assertEqual(path, self.cfg_path)
        args = cfg['args']
        self.assertEqual(args[args.index('--max-context') + 1], '32768')
        self.assertEqual(args[args.index('--kv') + 1], 'int8')
        self.assertEqual(args[args.index('--ple-io') + 1], 'mmap')
        self.assertEqual(args[args.index('--native') + 1], args[args.index('--ple-gguf') + 1])
        self.assertNotIn('--vision', args)
        self.assertNotIn('vision', cfg)
        self.assertFalse(cfg.get('parallel'))

    def test_unexpected_experts_bin_override_refuses_readiness(self):
        override = self.pack / 'experts.bin'
        override.write_bytes(b'unexpected override')
        override.chmod(0o444)
        with self.assertRaisesRegex(SafetyError, 'forbidden expert override'):
            profile.ready_q4ks(self.root)

    def test_symlinked_pack_parent_outside_data_root_refuses_readiness(self):
        outside = self.root.parent / (self.root.name + '-outside-pack')
        outside.mkdir()
        moved = self.root / 'pack-preserved'
        self.pack.rename(moved)
        self.pack.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(SafetyError, 'outside STRATA_DATA_ROOT'):
                profile.ready_q4ks(self.root)
        finally:
            self.pack.unlink()
            moved.rename(self.pack)
            outside.rmdir()

    def test_symlinked_expert_profile_parent_outside_data_root_refuses_readiness(self):
        base_source = self.root / 'source-99f3dbd0b21d'
        data = base_source / 'data'
        moved = base_source / 'data-preserved'
        outside = self.root.parent / (self.root.name + '-outside-expert')
        outside.mkdir()
        data.rename(moved)
        data.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(SafetyError, 'outside STRATA_DATA_ROOT'):
                profile.ready_q4ks(self.root)
        finally:
            data.unlink()
            moved.rename(data)
            outside.rmdir()

    def test_symlinked_mtp_parent_outside_data_root_refuses_readiness(self):
        mtp = self.root / 'data/mtp'
        moved = self.root / 'mtp-preserved'
        outside = self.root.parent / (self.root.name + '-outside-mtp')
        outside.mkdir()
        mtp.rename(moved)
        mtp.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(SafetyError, 'outside STRATA_DATA_ROOT'):
                profile.ready_q4ks(self.root)
        finally:
            mtp.unlink()
            moved.rename(mtp)
            outside.rmdir()

    def test_missing_parity_refuses_readiness(self):
        (self.pr / 'parity.json').unlink()
        with self.assertRaisesRegex(SafetyError, 'parity'):
            profile.ready_q4ks(self.root)

    def test_readiness_detects_same_size_shard_change_with_restored_mtime(self):
        path = self.paths[0]
        saved = self.manifest['assets'][0]
        old_mtime = path.stat().st_mtime_ns
        path.chmod(0o644)
        path.write_bytes(b'changed data')
        os.utime(path, ns=(old_mtime, old_mtime))
        path.chmod(0o444)
        with self.assertRaisesRegex(SafetyError, 'changed'):
            profile.check_record(path, saved, self.fixture_assets[path.name][1], sealed=True)

    def test_each_parity_program_is_required(self):
        parity_path = self.pr / 'parity.json'
        parity = json.loads(parity_path.read_text())
        parity['tests']['ple_q5_parity']['passed'] = False
        parity_path.write_text(json.dumps(parity))
        with self.assertRaisesRegex(SafetyError, 'ple_q5_parity'):
            profile.ready_q4ks(self.root)

    def test_generic_runtime_selection_is_not_consulted_or_changed(self):
        import strata_launcher as launcher
        selection = self.root / 'runtime-selection.json'
        selection.write_text('{"version":"0.1.41"}\n')
        before = selection.read_bytes()
        with mock.patch.object(launcher, 'data_root', return_value=self.root), \
             mock.patch.object(launcher, 'ready_q4ks', return_value=(self.source, self.cfg_path, self.cfg)) as q4ready, \
             mock.patch.object(launcher.runtimes, 'configure', side_effect=AssertionError('generic runtime manager used')):
            result = launcher.ready_runtime(assets.PROFILE, runtime='auto', parallel=1, batch_groups=1)
        self.assertEqual(result[0], self.source)
        self.assertEqual(result[3], '0.1.39')
        q4ready.assert_called_once_with(self.root)
        self.assertEqual(selection.read_bytes(), before)

    def test_explicit_041_routes_to_candidate_and_other_runtime_or_slots_are_refused(self):
        import strata_launcher as launcher
        candidate_result = (self.source / 'candidate', self.pr / 'candidate.json', self.cfg)
        with mock.patch.object(launcher, 'data_root', return_value=self.root), \
             mock.patch.object(launcher.q4ks_v0141, 'ready_q4ks_v0141', return_value=candidate_result) as ready_candidate:
            result = launcher.ready_runtime(assets.PROFILE, runtime='0.1.41', parallel=1, batch_groups=1)
            with self.assertRaisesRegex(SafetyError, 'baseline or the explicit .41'):
                launcher.ready_runtime(assets.PROFILE, runtime='0.1.38', parallel=1, batch_groups=1)
            with self.assertRaisesRegex(SafetyError, 'single-slot'):
                launcher.ready_runtime(assets.PROFILE, runtime='auto', parallel=2, batch_groups=1)
        self.assertEqual(result[0], candidate_result[0])
        self.assertEqual(result[3], '0.1.41')
        ready_candidate.assert_called_once_with(self.root)

    def test_launcher_exposes_future_vision_enable_and_rollback_operations(self):
        import strata_launcher as launcher
        for option, operation in (('--enable-q4ks-vision', launcher.configure_q4ks_vision),
                                  ('--rollback-q4ks-vision', launcher.rollback_q4ks_vision)):
            with (mock.patch.object(launcher, 'data_root', return_value=self.root),
                  mock.patch.object(launcher, operation.__name__) as called,
                  mock.patch.object(sys, 'argv', ['strata_launcher.py', option, '--profile', assets.PROFILE])):
                self.assertEqual(launcher.main(), 0)
                called.assert_called_once_with(self.root)

    def test_launcher_evidence_summary_binds_native_report_to_current_config(self):
        import strata_launcher as launcher
        original = json.dumps(profile.make_config(self.root, context=32768)).encode()
        self.cfg_path.write_bytes(original)
        checks = {'health': True, 'arithmetic': True, 'strict_json': True,
                  'tool_call': True, 'memory_recorded': True}
        report32 = {'profile': assets.PROFILE, 'model_id': assets.MODEL_ID,
                    'source_commit': assets.SOURCE_COMMIT, 'runtime_version': assets.RUNTIME_VERSION,
                    'config_sha256': sha(original), 'context': 32768, 'parallel': 1,
                    'text_only': True, 'passed': True, 'checks': checks,
                    'memory_peak_mib': 12000, 'mtp': {'drafted_tokens': 10, 'accepted_tokens': 9},
                    'recorded_server_identity': {'pid': 123, 'start_ticks': 456, 'exe': '/usr/bin/python3'}}
        (self.pr / 'validation-32k.json').write_text(json.dumps(report32))
        self.assertIn('Recorded 32K text-only gate passed',
                      '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path)))

        native_cfg = profile.make_config(self.root, context=262144, vision=True)
        native_bytes = json.dumps(native_cfg).encode()
        (self.pr / 'context-config-before.json').write_bytes(original)
        self.cfg_path.write_bytes(native_bytes)
        native = {'profile': assets.PROFILE, 'model_id': assets.MODEL_ID,
                  'source_commit': assets.SOURCE_COMMIT, 'runtime_version': assets.RUNTIME_VERSION,
                  'config_sha256': sha(native_bytes), 'context_configured': 262144,
                  'vision_configured': True, 'vision_cpu_only': True, 'passed': True,
                  'live_server_identity': {'pid': 456, 'start_ticks': 789, 'exe': '/usr/bin/python3'},
                  'health': {'status': 'ok', 'max_context': 262144, 'images': True, 'loaded': True},
                  'vision': {'passed': True, 'expected': 'magenta', 'answer': 'Magenta',
                             'cpu_only_configured': True},
                  'full_context': {'passed': True, 'count_match': True, 'counted_prompt_tokens': 260996,
                                   'marker_retrieved': True, 'expected_tail_marker': 'Q4-MARKER',
                                   'answer': 'Q4-MARKER', 'response_input_tokens': 260996,
                                   'context_fill_ratio': 0.5},
                  'resource_measurement': {'sampler_errors': []}}
        report_path = self.pr / 'validation-native.json'
        report_path.write_text(json.dumps(native))
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertIn('260,996/262,144 tokens', summary)
        self.assertIn('99.56%', summary)  # Derived from response_input_tokens, not the contradictory stored ratio.
        self.assertIn('synthetic image passed', summary)

        native['vision']['answer'] = 'not Magenta'
        report_path.write_text(json.dumps(native))
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertNotIn('synthetic image passed', summary)
        native['vision']['answer'] = 'Magenta'
        native['vision'] = None
        report_path.write_text(json.dumps(native))
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertIn('Stored native/vision report is unreadable', summary)
        report_path.write_text('[' * 1500 + ']' * 1500)
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertIn('Stored native/vision report is unreadable', summary)
        native['vision'] = {'passed': True, 'expected': 'magenta', 'answer': 'Magenta',
                            'cpu_only_configured': True}
        report32['checks'] = None
        (self.pr / 'validation-32k.json').write_text(json.dumps(report32))
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertIn('Stored 32K report is unreadable', summary)
        report32['checks'] = checks
        (self.pr / 'validation-32k.json').write_text(json.dumps(report32))
        report_path.write_text(json.dumps(native))

        native_cfg['sampling']['top_k'] = 10
        self.cfg_path.write_text(json.dumps(native_cfg))
        summary = '\n'.join(launcher.q4ks_evidence_summary(self.root, self.cfg_path))
        self.assertIn('stale/incomplete for this config', summary)

    def test_live_validation_requires_identity_bound_q4_text_only_server(self):
        state_root = self.root / 'state'
        state_root.mkdir()
        config = profile.make_config(self.root, context=32768)
        config['log'] = str(self.root / 'server.log')
        run_cfg = state_root / 'run-config.json'
        run_cfg.write_text(json.dumps(config))
        identity = {'pid': 42, 'start_ticks': 7}
        state = {'profile': assets.PROFILE, 'runtime_version': '0.1.39',
                 'source_commit': assets.SOURCE_COMMIT, 'identity': identity,
                 'config': str(run_cfg), 'parallel': 1, 'batch_groups': 1,
                 'allowed_executables': [str((self.source / '.venv/bin/python').resolve()),
                                         str((self.source / 'engine/strata').resolve())]}
        (state_root / 'server.json').write_text(json.dumps(state))
        with mock.patch.object(profile, 'strata_state', return_value=state_root), \
             mock.patch.object(profile, 'proc', return_value=identity), \
             mock.patch.object(profile, 'same_process', return_value=True):
            self.assertEqual(profile._assert_live_q4ks(self.root)['profile'], assets.PROFILE)
            config['args'].append('--vision')
            run_cfg.write_text(json.dumps(config))
            with self.assertRaisesRegex(SafetyError, 'text-only'):
                profile._assert_live_q4ks(self.root)

    def test_native_context_refuses_missing_live_validation(self):
        args = self.cfg['args']
        args[args.index('--max-context') + 1] = '262144'
        args.extend(['--kv-resident', '32768'])
        self.cfg_path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(SafetyError, '32K validation'):
            profile.ready_q4ks(self.root)

    def test_migration_refuses_missing_32k_validation(self):
        with mock.patch.object(profile, 'docker_empty'):
            with self.assertRaisesRegex(SafetyError, '32K validation'):
                profile.configure_native_context(self.root)
        self.assertEqual(json.loads(self.cfg_path.read_text())['args'], self.cfg['args'])

    def test_measured_32k_gate_allows_only_explicit_native_migration(self):
        report = {
            'profile': assets.PROFILE, 'model_id': assets.MODEL_ID,
            'source_commit': assets.SOURCE_COMMIT, 'runtime_version': '0.1.39',
            'runtime_sha256': self.manifest['runtime_assets'][0]['sha256'],
            'asset_sha256': {name: value[1] for name, value in self.fixture_assets.items()},
            'config_sha256': sha(self.cfg_path.read_bytes()), 'context': 32768,
            'parallel': 1, 'text_only': True,
            'checks': {'health': True, 'arithmetic': True, 'strict_json': True,
                       'tool_call': True, 'memory_recorded': True},
            'memory_peak_mib': 19000, 'mtp': {'drafted_tokens': 100, 'accepted_tokens': 41},
        }
        report_path = self.root / 'live-report.json'
        report_path.write_text(json.dumps(report))
        selection = self.root / 'runtime-selection.json'
        selection.write_text('{"version":"0.1.39"}\n')
        run_config = self.root / 'state/run-config.json'
        run_config.parent.mkdir(parents=True)
        run_config.write_text('{"active":"unchanged"}\n')
        selection_before, run_config_before = selection.read_bytes(), run_config.read_bytes()
        with mock.patch.object(builtins, 'print'), \
             mock.patch.object(profile, '_assert_live_q4ks', return_value={
                 'profile': assets.PROFILE, 'identity': {'pid': 42, 'start_ticks': 7},
                 'config': str(self.root / 'state/run-config.json')}):
            profile.record_32k_validation(self.root, report_path)
        with mock.patch.object(profile, 'docker_empty'), mock.patch.object(builtins, 'print'):
            profile.configure_native_context(self.root)
        self.assertEqual(selection.read_bytes(), selection_before)
        self.assertEqual(run_config.read_bytes(), run_config_before)
        native = profile.ready_q4ks(self.root)[2]['args']
        self.assertEqual(native[native.index('--max-context') + 1], '262144')
        self.assertEqual(native[native.index('--kv-resident') + 1], '32768')
        self.assertEqual(json.loads((self.pr / 'context-config-before.json').read_text()), self.cfg)

    def test_mtp_acceptance_must_be_measured_before_native_migration(self):
        report = {
            'profile': assets.PROFILE, 'model_id': assets.MODEL_ID, 'source_commit': assets.SOURCE_COMMIT,
            'runtime_version': '0.1.39', 'runtime_sha256': self.manifest['runtime_assets'][0]['sha256'],
            'asset_sha256': {name: value[1] for name, value in self.fixture_assets.items()},
            'config_sha256': sha(self.cfg_path.read_bytes()), 'context': 32768, 'parallel': 1,
            'text_only': True, 'checks': {'health': True, 'arithmetic': True, 'strict_json': True,
                'tool_call': True, 'memory_recorded': True}, 'memory_peak_mib': 19000,
            'mtp': {'drafted_tokens': 0, 'accepted_tokens': 0},
        }
        report_path = self.root / 'bad-live-report.json'; report_path.write_text(json.dumps(report))
        with mock.patch.object(profile, '_assert_live_q4ks', return_value={
                'profile': assets.PROFILE, 'identity': {'pid': 42, 'start_ticks': 7},
                'config': str(self.root / 'state/run-config.json')}):
            with self.assertRaisesRegex(SafetyError, 'MTP drafted/accepted'):
                profile.record_32k_validation(self.root, report_path)
        self.assertFalse((self.pr / 'validation-32k.json').exists())

    def write_passing_validation(self):
        report = {
            'profile': assets.PROFILE, 'model_id': assets.MODEL_ID, 'source_commit': assets.SOURCE_COMMIT,
            'runtime_version': assets.RUNTIME_VERSION, 'runtime_sha256': self.manifest['runtime_assets'][0]['sha256'],
            'asset_sha256': {name: value[1] for name, value in self.fixture_assets.items()},
            'config_sha256': sha(self.cfg_path.read_bytes()), 'context': 32768, 'parallel': 1,
            'text_only': True, 'checks': {'health': True, 'arithmetic': True, 'strict_json': True,
                'tool_call': True, 'memory_recorded': True}, 'memory_peak_mib': 19000,
            'mtp': {'drafted_tokens': 100, 'accepted_tokens': 41}, 'passed': True,
            'recorded_server_identity': {'pid': 42, 'start_ticks': 7},
            'recorded_server_config': str(self.root / 'state/run-config.json'),
            'recorded_at': '2026-01-01T00:00:00Z',
        }
        (self.pr / 'validation-32k.json').write_text(json.dumps(report))

    def migrate_to_native_context(self):
        self.write_passing_validation()
        with mock.patch.object(profile, 'docker_empty'), mock.patch.object(builtins, 'print'):
            profile.configure_native_context(self.root)

    def prepare_iq3_projector(self, data=b'verified fixture projector'):
        from orca_assets import ASSETS as iq3_assets, PROFILE as iq3_profile, REPOSITORY as iq3_repo, REVISION as iq3_revision
        projector = self.root / 'data/models' / iq3_profile / profile.VISION_PROJECTOR_FILENAME
        projector.parent.mkdir(parents=True, exist_ok=True)
        projector.write_bytes(data)
        projector.chmod(0o444)
        st = projector.stat()
        record = {'path': str(projector), 'bytes': len(data), 'mtime_ns': st.st_mtime_ns,
                  'sha256': sha(data)}
        manifest = {'profile': iq3_profile, 'source_commit': assets.IQ3XXS_SOURCE_COMMIT,
                    'model_repository': iq3_repo, 'model_revision': iq3_revision, 'assets': [record]}
        (self.root / 'profiles' / iq3_profile).mkdir(parents=True)
        (self.root / 'profiles' / iq3_profile / 'prepared.json').write_text(json.dumps(manifest))
        return projector, data

    def configure_test_vision(self):
        projector_data = self.fixture_projector
        self.prepare_iq3_projector(projector_data)
        self.migrate_to_native_context()
        with mock.patch.object(profile, 'docker_empty'), mock.patch.object(builtins, 'print'):
            profile.configure_q4ks_vision(self.root)
        return projector_data

    def test_cpu_vision_requires_migrated_native_context(self):
        projector_data = b'verified fixture projector'
        with mock.patch.object(profile, 'VISION_PROJECTOR_BYTES', len(projector_data)), \
             mock.patch.object(profile, 'VISION_PROJECTOR_SHA256', sha(projector_data)):
            self.prepare_iq3_projector(projector_data)
            with mock.patch.object(profile, 'docker_empty'):
                with self.assertRaisesRegex(SafetyError, 'native-context migration'):
                    profile.configure_q4ks_vision(self.root)
        self.assertFalse((self.pr / 'vision-prepared.json').exists())
        self.assertNotIn('vision', json.loads(self.cfg_path.read_text()))

    def test_cpu_vision_uses_verified_read_only_hardlink_and_exact_config(self):
        projector_data = self.configure_test_vision()
        projector = self.root / 'data/models' / assets.PROFILE / profile.VISION_PROJECTOR_FILENAME
        source_projector = self.root / 'data/models/orca-iq3_xxs' / profile.VISION_PROJECTOR_FILENAME
        self.assertTrue(os.path.samefile(projector, source_projector))
        self.assertEqual(projector.stat().st_mode & 0o222, 0)
        self.assertEqual(projector.read_bytes(), projector_data)
        cfg = profile.ready_q4ks(self.root)[2]
        self.assertEqual(cfg, profile.make_config(self.root, context=262144, vision=True))
        self.assertEqual(cfg['vision']['exe'], str(self.vision_engine))
        self.assertIs(cfg['vision']['gpu'], False)
        self.assertNotIn('--vram-reserve-mib', cfg['args'])
        self.assertIn('--vision', cfg['args'])
        migration = json.loads((self.pr / 'native-context-migration.json').read_text())
        self.assertEqual(migration['current_config_sha256'], sha(self.cfg_path.read_bytes()))
        self.assertTrue((self.pr / 'vision-prepared.json').is_file())
        manifest = json.loads((self.pr / 'prepared.json').read_text())
        self.assertEqual({Path(item['path']).name for item in manifest['assets']}, set(assets.ASSETS))
        parity = json.loads((self.pr / 'parity.json').read_text())
        self.assertEqual(parity['asset_sha256'], {name: value[1] for name, value in self.fixture_assets.items()})
        report = json.loads((self.pr / 'validation-32k.json').read_text())
        self.assertEqual((report['context'], report['text_only']), (32768, True))

    def test_cpu_vision_readiness_requires_separate_sidecar(self):
        self.configure_test_vision()
        (self.pr / 'vision-prepared.json').unlink()
        with self.assertRaisesRegex(SafetyError, 'vision-prepared provenance'):
            profile.ready_q4ks(self.root)

    def test_cpu_vision_readiness_survives_candidate_hardlink_count_changes(self):
        self.configure_test_vision()
        baseline_projector = self.root / 'data/models' / assets.PROFILE / profile.VISION_PROJECTOR_FILENAME
        candidate_projector = self.root / 'profiles/orca-q4_k_s-v0141/assets' / profile.VISION_PROJECTOR_FILENAME
        candidate_projector.parent.mkdir(parents=True)
        os.link(baseline_projector, candidate_projector)
        try:
            with mock.patch.object(profile, 'docker_empty'):
                self.assertEqual(profile.ready_q4ks(self.root)[2],
                                 profile.make_config(self.root, context=262144, vision=True))
            candidate_projector.unlink()
            with mock.patch.object(profile, 'docker_empty'):
                self.assertEqual(profile.ready_q4ks(self.root)[2],
                                 profile.make_config(self.root, context=262144, vision=True))
        finally:
            candidate_projector.unlink(missing_ok=True)

    def test_cpu_vision_refuses_unverified_or_writable_projector(self):
        data = b'verified fixture projector'
        with mock.patch.object(profile, 'VISION_PROJECTOR_BYTES', len(data)), \
             mock.patch.object(profile, 'VISION_PROJECTOR_SHA256', sha(data)):
            source, _ = self.prepare_iq3_projector(data)
            self.migrate_to_native_context()
            manifest_path = self.root / 'profiles/orca-iq3_xxs/prepared.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['assets'][0]['sha256'] = '0' * 64
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(profile, 'docker_empty'):
                with self.assertRaisesRegex(SafetyError, 'IQ3_XXS projector pin'):
                    profile.configure_q4ks_vision(self.root)
            manifest['assets'][0]['sha256'] = sha(data)
            manifest_path.write_text(json.dumps(manifest))
            source.chmod(0o644)
            with mock.patch.object(profile, 'docker_empty'):
                with self.assertRaisesRegex(SafetyError, 'read-only'):
                    profile.configure_q4ks_vision(self.root)
        self.assertFalse((self.pr / 'vision-prepared.json').exists())

    def test_cpu_vision_refuses_projector_path_escape(self):
        outside = self.root.parent / (self.root.name + '-projector')
        outside.write_bytes(b'verified fixture projector')
        data = b'verified fixture projector'
        with mock.patch.object(profile, 'VISION_PROJECTOR_BYTES', len(data)), \
             mock.patch.object(profile, 'VISION_PROJECTOR_SHA256', sha(data)):
            source, _ = self.prepare_iq3_projector(data)
            source.chmod(0o644)
            source.unlink()
            source.symlink_to(outside)
            self.migrate_to_native_context()
            with mock.patch.object(profile, 'docker_empty'):
                with self.assertRaisesRegex(SafetyError, 'outside STRATA_DATA_ROOT'):
                    profile.configure_q4ks_vision(self.root)
        self.assertFalse((self.pr / 'vision-prepared.json').exists())
        outside.unlink()

    def test_cpu_vision_config_and_sidecar_mismatches_fail_readiness(self):
        self.configure_test_vision()
        cfg = json.loads(self.cfg_path.read_text())
        cfg['vision']['gpu'] = True
        self.cfg_path.write_text(json.dumps(cfg))
        migration_path = self.pr / 'native-context-migration.json'
        migration = json.loads(migration_path.read_text())
        migration['current_config_sha256'] = sha(self.cfg_path.read_bytes())
        migration_path.write_text(json.dumps(migration))
        with self.assertRaisesRegex(SafetyError, 'exact pinned'):
            profile.ready_q4ks(self.root)
        cfg = profile.make_config(self.root, context=262144, vision=True)
        self.cfg_path.write_text(json.dumps(cfg))
        migration['current_config_sha256'] = sha(self.cfg_path.read_bytes())
        migration_path.write_text(json.dumps(migration))
        sidecar_path = self.pr / 'vision-prepared.json'
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['projector']['sha256'] = '0' * 64
        sidecar_path.write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(SafetyError, 'vision provenance'):
            profile.ready_q4ks(self.root)

    def test_explicit_vision_rollback_restores_262k_text_only_and_coherent_hash(self):
        self.configure_test_vision()
        active = self.root / 'state/run-config.json'
        active.parent.mkdir(parents=True)
        active.write_text('{"active":"unchanged"}\n')
        server_state = self.root / 'state/server.json'
        server_state.write_text('{"server":"unchanged"}\n')
        selection = self.root / 'runtime-selection.json'
        selection.write_text('{"version":"0.1.41"}\n')
        before = active.read_bytes(), server_state.read_bytes(), selection.read_bytes()
        original_evidence = ((self.pr / 'context-config-before.json').read_bytes(),
                             (self.pr / 'validation-32k.json').read_bytes())
        sidecar_path = self.pr / 'vision-prepared.json'
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['projector']['sha256'] = '0' * 64
        sidecar_path.write_text(json.dumps(sidecar))
        with mock.patch.object(profile, 'docker_empty'), mock.patch.object(builtins, 'print'):
            profile.rollback_q4ks_vision(self.root)
        cfg = profile.ready_q4ks(self.root)[2]
        self.assertEqual(cfg, profile.make_config(self.root, context=262144))
        migration = json.loads((self.pr / 'native-context-migration.json').read_text())
        self.assertEqual(migration['current_config_sha256'], sha(self.cfg_path.read_bytes()))
        self.assertEqual((active.read_bytes(), server_state.read_bytes(), selection.read_bytes()), before)
        self.assertEqual(((self.pr / 'context-config-before.json').read_bytes(),
                          (self.pr / 'validation-32k.json').read_bytes()), original_evidence)
        self.assertTrue((self.pr / 'vision-prepared.json').is_file())

    def test_vision_configured_state_requires_valid_migration_hash(self):
        self.configure_test_vision()
        migration_path = self.pr / 'native-context-migration.json'
        migration = json.loads(migration_path.read_text())
        migration['current_config_sha256'] = '0' * 64
        migration_path.write_text(json.dumps(migration))
        with self.assertRaisesRegex(SafetyError, 'migration evidence'):
            profile.ready_q4ks(self.root)

    def test_parity_refuses_root_even_with_hostllm_lease(self):
        with mock.patch.object(prepare.os, 'geteuid', return_value=0), \
             mock.patch.object(prepare, 'docker_empty') as docker, \
             mock.patch.object(prepare, 'launch_gate') as gate:
            with self.assertRaisesRegex(SafetyError, 'normal operator; do not use sudo'):
                prepare.verify_parity(LEASE_ID)
        docker.assert_not_called()
        gate.assert_not_called()

    def test_prepare_refuses_root_before_writing_profile_state(self):
        unused_profile = self.root / 'must-not-be-created'
        with mock.patch.object(prepare, 'PROFILE_ROOT', unused_profile), \
             mock.patch.object(prepare.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(SafetyError, 'normal operator; do not use sudo'):
                prepare.prepare()
        self.assertFalse(unused_profile.exists())

    @unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(__import__('signal'), 'pidfd_send_signal'),
                         'Linux pidfd support is required')
    def test_parity_identity_failure_stops_and_reaps_owned_pidfd_child(self):
        import subprocess
        real_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        command = [sys.executable, '-c', 'import time; time.sleep(30)']
        with mock.patch.object(prepare.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(prepare, 'proc', side_effect=SafetyError('injected identity read failure')):
            with self.assertRaisesRegex(SafetyError, 'injected identity read failure'):
                prepare._run_parity_program(command, self.root, dict(os.environ))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    @unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(__import__('signal'), 'pidfd_send_signal'),
                         'Linux pidfd support is required')
    def test_sigterm_ignoring_parity_child_is_killed_only_via_pidfd_and_reaped(self):
        import subprocess
        import time
        ready = self.root / 'ignore-term-ready'
        script = ('import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  f'pathlib.Path({str(ready)!r}).touch(); time.sleep(30)')
        real_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        def fail_after_ready(pid):
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not ready.exists():
                raise AssertionError('child did not install its SIGTERM handler')
            raise SafetyError('injected identity read failure')

        command = [sys.executable, '-c', script]
        with mock.patch.object(prepare, 'PARITY_STOP_GRACE_SECONDS', 0.1), \
             mock.patch.object(prepare.subprocess, 'Popen', side_effect=spawn), \
             mock.patch.object(prepare, 'proc', side_effect=fail_after_ready):
            with self.assertRaisesRegex(SafetyError, 'injected identity read failure'):
                prepare._run_parity_program(command, self.root, dict(os.environ))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_parity_requires_hostllm_pause_lease(self):
        with mock.patch.object(prepare, 'docker_empty'), \
             mock.patch.object(prepare.hosting, 'read_lease', return_value=None), \
             mock.patch.object(prepare, 'active_engine', return_value='none'), \
             mock.patch.object(prepare, 'launch_gate') as gate:
            with self.assertRaisesRegex(SafetyError, 'matching one-time HostLLM'):
                prepare.verify_parity(LEASE_ID)
        gate.assert_not_called()

    def test_parity_refuses_running_server_without_subprocess_or_stop(self):
        (self.pr / 'parity.json').unlink()
        with mock.patch.object(prepare, 'docker_empty'), \
             mock.patch.object(prepare.hosting, 'read_lease', return_value={'phase': 'held', 'port': 8080, 'lease_id': LEASE_ID}), \
             mock.patch.object(prepare, 'active_engine', return_value='strata'), \
             mock.patch.object(prepare, 'launch_gate') as gate, \
             mock.patch.object(prepare, '_run_parity_program') as run:
            with self.assertRaisesRegex(SafetyError, 'will not disturb or stop'):
                prepare.verify_parity(LEASE_ID)
        gate.assert_not_called()
        run.assert_not_called()
        self.assertFalse((self.pr / 'parity.json').exists())

    def test_pack_rejects_unexpected_experts_bin_before_promotion(self):
        pending_pack = self.root / 'pack-experts-test'
        required = ['dense.bin', 'index.txt', 'native_experts.txt', 'compat-bf16.json',
                    'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja', 'tokenizer/vocab.json',
                    'tokenizer/merges.txt', 'tokenizer/token_type.json']

        tool = self.source / 'tools/iq_pack.py'
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_text('# pinned fixture pack tool\n')

        def create_pack(command, **kwargs):
            output = Path(command[command.index('--out') + 1])
            for relative in required + ['experts.bin']:
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')

        with mock.patch.object(prepare, 'BASE', self.root), \
             mock.patch.object(prepare, 'PACK', pending_pack), \
             mock.patch.object(prepare.subprocess, 'run', side_effect=create_pack):
            with self.assertRaisesRegex(SafetyError, 'must not contain an experts.bin'):
                prepare._pack(self.paths, self.pinned_source / '.venv/bin/python', self.source)
        pending_dir = pending_pack.with_name(pending_pack.name + '.preparing')
        self.assertFalse(pending_pack.exists())
        self.assertTrue((pending_dir / 'experts.bin').is_file())

    def test_failed_parity_program_does_not_create_pass_record(self):
        import subprocess
        (self.pr / 'parity.json').unlink()
        with mock.patch.object(prepare, 'docker_empty'), \
             mock.patch.object(prepare.hosting, 'read_lease', return_value={'phase': 'held', 'port': 8080, 'lease_id': LEASE_ID}), \
             mock.patch.object(prepare, 'active_engine', return_value='none'), \
             mock.patch.object(prepare, 'launch_gate'), \
             mock.patch.object(prepare, 'protected_snapshot', return_value={'hashes': {}, 'iq3_xxs_file_identities': {}}), \
             mock.patch.object(prepare, 'live_strata_identity', return_value=None), \
             mock.patch.object(prepare, 'assert_protected'), \
             mock.patch.object(prepare, 'assert_live_strata_unchanged'), \
             mock.patch.object(prepare, '_run_parity_program', side_effect=[None, subprocess.CalledProcessError(1, ['ple_q5_parity'])]):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.verify_parity(LEASE_ID)
        self.assertFalse((self.pr / 'parity.json').exists())

    def test_real_shard_parity_records_both_programs_only_after_success(self):
        port = 18199
        with mock.patch.object(builtins, 'print'), \
             mock.patch.object(prepare, 'docker_empty'), \
             mock.patch.object(prepare.hosting, 'read_lease', return_value={'phase': 'held', 'port': port, 'lease_id': LEASE_ID}), \
             mock.patch.object(prepare, 'active_engine', return_value='none'), \
             mock.patch.object(prepare, 'launch_gate') as gate, \
             mock.patch.object(prepare, 'protected_snapshot', return_value={'hashes': {}, 'iq3_xxs_file_identities': {}}), \
             mock.patch.object(prepare, 'live_strata_identity', return_value=None), \
             mock.patch.object(prepare, 'assert_protected'), \
             mock.patch.object(prepare, 'assert_live_strata_unchanged'), \
             mock.patch.object(prepare, '_run_parity_program') as run:
            prepare.verify_parity(LEASE_ID, port)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(gate.call_count >= 3)
        self.assertTrue(all(call.args == (port,) for call in gate.call_args_list))
        native_args = run.call_args_list[0].args[0]
        self.assertEqual(native_args[-5:], ['0', '5', '6', '20', '47'])
        ple_args = run.call_args_list[1].args[0]
        self.assertEqual(Path(ple_args[0]).name, 'ple_q5_parity')
        parity = json.loads((self.pr / 'parity.json').read_text())
        self.assertTrue(all(parity['tests'][name]['passed'] for name in assets.PARITY_TESTS))


class CMakeProvisioning(unittest.TestCase):
    def setUp(self):
        import io
        import zipfile
        import stat
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        binary = b"#!/bin/sh\nprintf 'cmake version 3.31.6\n'\n"
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as wheel:
            info = zipfile.ZipInfo('cmake/data/bin/cmake')
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            wheel.writestr(info, binary)
        self.wheel_bytes = archive.getvalue()
        self.patches = [
            mock.patch.object(prepare, 'BASE', self.root),
            mock.patch.object(prepare, 'CMAKE_WHEEL_BYTES', len(self.wheel_bytes)),
            mock.patch.object(prepare, 'CMAKE_WHEEL_SHA256', sha(self.wheel_bytes)),
            mock.patch.object(prepare, 'CMAKE_WHEEL_URL', 'https://files.pythonhosted.org/pinned-cmake.whl'),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tmp.cleanup()

    def response(self):
        import io
        class Response(io.BytesIO):
            status = 200
            def __init__(self, data):
                super().__init__(data)
                self.headers = {'Content-Length': str(len(data))}
        return Response(self.wheel_bytes)

    def test_installs_exact_wheel_in_q4_runtime_and_reuses_it(self):
        with mock.patch.object(prepare.urllib.request, 'urlopen', return_value=self.response()) as fetch:
            binary = prepare._ensure_cmake()
        self.assertEqual(binary, assets.cmake_binary(self.root))
        self.assertTrue(os.access(binary, os.X_OK))
        self.assertEqual(prepare._cmake_version(binary), f'cmake version {assets.CMAKE_VERSION}')
        self.assertEqual(assets.cmake_wheel(self.root).read_bytes(), self.wheel_bytes)
        self.assertFalse((assets.cmake_wheel(self.root).with_name(assets.CMAKE_WHEEL_FILENAME + '.part')).exists())
        self.assertEqual(fetch.call_count, 1)
        with mock.patch.object(prepare.urllib.request, 'urlopen', side_effect=AssertionError('must reuse pinned wheel')):
            self.assertEqual(prepare._ensure_cmake(), binary)

    def test_wrong_wheel_hash_fails_closed_and_removes_only_its_partial(self):
        self.patches.append(mock.patch.object(prepare, 'CMAKE_WHEEL_SHA256', '0' * 64))
        self.patches[-1].start()
        with mock.patch.object(prepare.urllib.request, 'urlopen', return_value=self.response()):
            with self.assertRaisesRegex(SafetyError, 'SHA-256'):
                prepare._ensure_cmake()
        self.assertFalse(assets.cmake_wheel(self.root).exists())
        self.assertFalse(assets.cmake_wheel(self.root).with_name(assets.CMAKE_WHEEL_FILENAME + '.part').exists())

    def test_unsafe_wheel_member_is_rejected_without_escape(self):
        import io
        import zipfile
        import stat
        outside = self.root.parent / (self.root.name + '-escaped')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as wheel:
            info = zipfile.ZipInfo('../escaped')
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            wheel.writestr(info, b'bad')
        data = archive.getvalue()
        self.patches.append(mock.patch.object(prepare, 'CMAKE_WHEEL_BYTES', len(data)))
        self.patches[-1].start()
        self.patches.append(mock.patch.object(prepare, 'CMAKE_WHEEL_SHA256', sha(data)))
        self.patches[-1].start()
        with mock.patch.object(prepare.urllib.request, 'urlopen', return_value=self.response_for(data)):
            with self.assertRaisesRegex(SafetyError, 'unsafe member path'):
                prepare._ensure_cmake()
        self.assertFalse(outside.exists())
        self.assertFalse(assets.cmake_root(self.root).exists())

    def response_for(self, data):
        import io
        class Response(io.BytesIO):
            status = 200
            def __init__(self, payload):
                super().__init__(payload)
                self.headers = {'Content-Length': str(len(payload))}
        return Response(data)


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.patches = [mock.patch.object(prepare, 'DEST', self.dest),
                        mock.patch.object(prepare, 'BASE', self.dest.parent),
                        mock.patch.object(prepare, 'ASSETS', {'fixture.gguf': (12, sha(b'pinned bytes'))}),
                        mock.patch.object(prepare, 'hf_token', return_value=None)]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tmp.cleanup()

    def test_resumable_download_requires_exact_content_range_and_hash(self):
        import io
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 4-11/12'}
        partial = self.dest / 'fixture.gguf.part'
        partial.write_bytes(b'pinn')
        with mock.patch.object(prepare.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(b'ed bytes')
            path = prepare.download('fixture.gguf', 12, sha(b'pinned bytes'))
            request = opener.return_value.open.call_args.args[0]
        self.assertEqual(path.read_bytes(), b'pinned bytes')
        self.assertEqual(request.get_header('Range'), 'bytes=4-')
        self.assertFalse(partial.exists())

    def test_wrong_existing_asset_is_never_overwritten(self):
        path = self.dest / 'fixture.gguf'
        path.write_bytes(b'wrong')
        with mock.patch.object(prepare.urllib.request, 'build_opener') as opener:
            with self.assertRaises(SafetyError):
                prepare.download('fixture.gguf', 12, sha(b'pinned bytes'))
            opener.assert_not_called()
        self.assertEqual(path.read_bytes(), b'wrong')

    def test_checksum_mismatch_is_not_promoted(self):
        import io
        class Response(io.BytesIO):
            status = 200
            headers = {}
        with mock.patch.object(prepare.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(b'pinned bytes')
            with self.assertRaisesRegex(SafetyError, 'checksum mismatch'):
                prepare.download('fixture.gguf', 12, '0' * 64)
        self.assertFalse((self.dest / 'fixture.gguf').exists())
        self.assertEqual((self.dest / 'fixture.gguf.part').read_bytes(), b'pinned bytes')

    def test_ignored_resume_range_keeps_original_partial(self):
        import io
        class Response(io.BytesIO):
            status = 200
            headers = {}
        partial = self.dest / 'fixture.gguf.part'
        partial.write_bytes(b'pinn')
        with mock.patch.object(prepare.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(b'ed bytes')
            with self.assertRaisesRegex(SafetyError, 'resume range'):
                prepare.download('fixture.gguf', 12, sha(b'pinned bytes'))
        self.assertEqual(partial.read_bytes(), b'pinn')


if __name__ == '__main__':
    unittest.main()
