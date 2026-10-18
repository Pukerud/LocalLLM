"""CPU-only checks for the pinned, opt-in Orca Q4_K_S / Strata 0.1.44.1 route."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

if sys.platform != 'linux':
    raise unittest.SkipTest('Linux controller imports')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orca_q4ks_v0141_assets as prior_assets
import orca_q4ks_v01441_assets as assets
import orca_q4ks_v01441_profile as candidate
from engine_safety import SafetyError


class CandidateIdentity(unittest.TestCase):
    def test_candidate_is_separate_from_the_default_runtime_manager(self):
        import strata_runtime
        root = Path('/tmp/strata-data')
        self.assertEqual(assets.RUNTIME_VERSION, '0.1.44.1')
        self.assertEqual(assets.SOURCE_COMMIT, '296b59d2f73c8f04a805df29c2fa97655adace78')
        self.assertEqual(assets.LLAMA_COMMIT, '3cf03257f219afbe7334045ff7c6a06ac68c627d')
        self.assertNotIn(assets.RUNTIME_VERSION, strata_runtime.PINS)
        self.assertNotEqual(assets.profile_root(root), prior_assets.profile_root(root))
        self.assertNotEqual(assets.source_root(root), prior_assets.source_root(root))
        self.assertEqual(assets.CONFIG_SHA256, '9cd479e3e889efb43b45745193e746d75f33065f06bb9d6db8862c9e870e186d')
        self.assertEqual(len(assets.SOURCE_MANIFEST_SHA256), 64)
        self.assertRegex(assets.SOURCE_MANIFEST_SHA256, r'^[0-9a-f]{64}$')
        self.assertEqual(assets.SOURCE_MANIFEST_SHA256,
                         'a8de552ff8096c190ece7388b61ea594104e3bbd062c852e20d03a8a586260ea')

    def test_candidate_config_requires_all_tuned_and_protected_settings(self):
        root = Path('/tmp/strata-data')
        model = root / 'data/models/orca-q4_k_s/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf'
        args = [
            '--pack', str(root / 'data/packs/orca-q4_k_s'),
            '--native', str(model), '--ple-gguf', str(model), '--ple-io', 'mmap',
            '--expert-profile', str(root / 'source-99f3dbd0b21d/data/expert-profile.bin'),
            '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '4',
            '--mtp', str(root / 'data/mtp/rt'), '--max-context', '262144',
            '--kv', 'int8', '--kv-resident', '32768', '--vision',
            '--conversation-cache-mib', '8192', '--conversation-cache-slots', '4',
            '--trim-stage-weights', '--pipeline-windows', '2',
        ]
        cfg = {
            'exe': str(assets.engine_path(root)), 'cwd': str(assets.source_root(root)),
            'args': args, 'tokenizer': str(root / 'data/packs/orca-q4_k_s/tokenizer'),
            'gpu': [0, 1, 2, 3], 'layer_split': '14,24,36',
            'model_name': assets.MODEL_ID, 'host': '127.0.0.1', 'port': 18844,
            'gpu_order': 'as_given', 'lazy_load': True,
            'env': {'STRATA_PARALLEL': '1', 'STRATA_BATCH_GROUPS': '1'},
            'vision': {
                'exe': str(assets.vision_path(root)), 'model': str(model),
                'mmproj': str(prior_assets.profile_root(root) / 'assets' /
                              'mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf'),
                'gpu': False, 'max_tokens': 1024,
            },
        }
        candidate._validate_config(root, cfg)
        self.assertIn('--pipeline-windows', cfg['args'])
        self.assertEqual(cfg['args'][cfg['args'].index('--pipeline-windows') + 1], '2')
        selected = candidate._selected_runtime_config(cfg)
        self.assertNotIn('--pipeline-windows', selected['args'])
        self.assertIn('--pipeline-windows', cfg['args'])  # calibrated evidence source stays immutable

        changed = json.loads(json.dumps(cfg))
        changed['layer_split'] = 'auto'
        with self.assertRaisesRegex(SafetyError, 'stage trimming'):
            candidate._validate_config(root, changed)
        changed = json.loads(json.dumps(cfg))
        changed['args'][changed['args'].index('--pipeline-windows') + 1] = '1'
        with self.assertRaisesRegex(SafetyError, 'pipeline-windows'):
            candidate._validate_config(root, changed)
        changed = json.loads(json.dumps(cfg))
        changed['args'].remove('--trim-stage-weights')
        with self.assertRaisesRegex(SafetyError, 'stage trimming'):
            candidate._validate_config(root, changed)

    def test_readiness_text_discloses_exploratory_pipeline_setting(self):
        text = candidate.readiness_line()
        self.assertIn('prefill auto', text)
        self.assertIn('split 14,24,36', text)
        self.assertIn('pipeline-windows 2 omitted (decode gain unconfirmed)', text)


class SourceTreeIntegrity(unittest.TestCase):
    def test_full_extracted_tree_must_match_pinned_archive_not_just_manifested_files(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'strata-v0.1.44.1'
            (source / 'serve').mkdir(parents=True)
            (source / 'serve/server.py').write_text('pinned server bytes\\n')
            (source / 'README.md').write_text('pinned docs\\n')
            archive_path = root / 'source.tar.gz'
            with tarfile.open(archive_path, 'w:gz') as archive:
                archive.add(source, arcname='strata-v0.1.44.1')

            candidate._verify_extracted_source(archive_path, source)
            (source / 'serve/server.py').write_text('tampered server bytes\\n')
            with self.assertRaisesRegex(SafetyError, 'differ from the pinned archive'):
                candidate._verify_extracted_source(archive_path, source)
            (source / 'serve/server.py').write_text('pinned server bytes\\n')
            (source / 'serve/injected.py').write_text('unlisted source\\n')
            with self.assertRaisesRegex(SafetyError, 'inventory differs'):
                candidate._verify_extracted_source(archive_path, source)


class RuntimeRouting(unittest.TestCase):
    def test_auto_selects_045_and_039_is_explicit_rollback_one_slot_only(self):
        import strata_launcher as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = (Path('/q4-039'), Path('/q4-039/config.json'), {'args': ['--max-context', '262144']})
            old_candidate = (Path('/q4-041'), Path('/q4-041/config.json'), {'args': ['--max-context', '262144']})
            candidate_result = (Path('/q4-0441'), Path('/q4-0441/config.json'),
                                {'args': ['--max-context', '262144']})
            standard_result = (Path('/q4-045'), Path('/q4-045/config.json'),
                               {'args': ['--max-context', '262144']})
            selection = root / 'runtime-selection.json'
            selection.write_text('{"version":"0.1.41"}\n')
            before = selection.read_bytes()
            with mock.patch.object(launcher, 'data_root', return_value=root), \
                 mock.patch.object(launcher, 'ready_q4ks', return_value=baseline) as base_ready, \
                 mock.patch.object(launcher.q4ks_v0141, 'ready_q4ks_v0141',
                                   return_value=old_candidate) as old_ready, \
                 mock.patch.object(launcher.q4ks_v01441, 'ready_q4ks_v01441',
                                   return_value=candidate_result) as new_ready, \
                 mock.patch.object(launcher, 'ready_q4ks_v0145',
                                   return_value=standard_result) as standard_ready, \
                 mock.patch.object(launcher.runtimes, 'configure',
                                   side_effect=AssertionError('generic runtime manager used')):
                default = launcher.ready_runtime(assets.HOST_PROFILE, runtime='auto', parallel=1, batch_groups=1)
                rollback = launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.39', parallel=1, batch_groups=1)
                old = launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.41', parallel=1, batch_groups=1)
                new = launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.44.1', parallel=1, batch_groups=1)
                with self.assertRaisesRegex(SafetyError, 'single-slot'):
                    launcher.ready_runtime(assets.HOST_PROFILE, runtime='auto', parallel=2, batch_groups=2)
                with self.assertRaisesRegex(SafetyError, 'single-slot'):
                    launcher.ready_runtime(assets.HOST_PROFILE, runtime='0.1.44.1', parallel=2, batch_groups=2)
            self.assertEqual(default[0], standard_result[0])
            self.assertEqual(default[3], '0.1.45')
            self.assertEqual(rollback[0], baseline[0])
            self.assertEqual(rollback[3], '0.1.39')
            self.assertEqual(old[0], old_candidate[0])
            self.assertEqual(old[3], '0.1.41')
            self.assertEqual(new[0], candidate_result[0])
            self.assertEqual(new[3], '0.1.44.1')
            base_ready.assert_called_once_with(root)
            old_ready.assert_called_once_with(root)
            self.assertEqual(new_ready.call_count, 1)
            new_ready.assert_called_once_with(root)
            standard_ready.assert_called_once_with(root)
            self.assertEqual(selection.read_bytes(), before)

    def test_0441_launch_environment_strips_arbitrary_engine_and_python_overrides(self):
        import strata_launcher as launcher
        source = {
            'PATH': '/usr/bin', 'HOME': '/home/user', 'STRATA_API_KEY': 'secret',
            'STRATA_STAGE_PIN': '1', 'STRATA_PREFILL_CPU_SHARE': '0',
            'STRATA_ROUTE_RESIDENT': '0.5', 'PYTHONPATH': '/attacker',
            'LD_PRELOAD': '/attacker.so', 'LD_LIBRARY_PATH': '/attacker/lib',
            'HOSTLLM_LAUNCH_LEASE_ID': 'a' * 64, 'HOSTLLM_LAUNCH_LOCK_FD': '9',
            'HIVE_STRATA_LAUNCH_ID': 'b' * 48, 'PYTHONDONTWRITEBYTECODE': '0',
        }
        clean = launcher._q4_v01441_launch_environment(source)
        expected = {key: source[key] for key in (
            'PATH', 'HOME', 'HOSTLLM_LAUNCH_LEASE_ID',
            'HOSTLLM_LAUNCH_LOCK_FD', 'HIVE_STRATA_LAUNCH_ID')}
        expected['PYTHONDONTWRITEBYTECODE'] = '1'
        self.assertEqual(clean, expected)

    def test_0441_host_binding_is_loopback_by_default_or_explicit_trusted_lan(self):
        import strata_launcher as launcher
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launcher._effective_host(assets.HOST_PROFILE, '0.1.44.1', '0.0.0.0'), '127.0.0.1')
        with mock.patch.dict(os.environ, {'STRATA_HOST': '0.0.0.0'}):
            self.assertEqual(launcher._effective_host(assets.HOST_PROFILE, '0.1.44.1', '127.0.0.1'), '0.0.0.0')
        with mock.patch.dict(os.environ, {'STRATA_HOST': '192.168.1.69'}):
            with self.assertRaisesRegex(SafetyError, '0.0.0.0'):
                launcher._effective_host(assets.HOST_PROFILE, '0.1.44.1', '127.0.0.1')

    def test_candidate_start_refuses_root_before_profile_readiness(self):
        import strata_launcher as launcher
        with mock.patch.object(launcher.os, 'geteuid', return_value=0), \
             mock.patch.object(launcher, 'ready_runtime', side_effect=AssertionError('candidate inspected')):
            with self.assertRaisesRegex(SafetyError, 'normal operator'):
                launcher.start(None, profile=assets.HOST_PROFILE, runtime='0.1.44.1')


class EvidenceGate(unittest.TestCase):
    def test_live_report_gate_requires_cache_vision_context_and_memory_evidence(self):
        report_config = str(assets.config_path(Path('/tmp')))
        inputs = {'candidate_config': report_config,
                  'candidate_config_sha256': assets.CONFIG_SHA256,
                  'engine_artifact_sha256': assets.ENGINE_SHA256,
                  'vision_artifact_sha256': assets.VISION_SHA256}
        results = {
            'test': 'strata_machine_specific_tuning', 'arm': 'integration-auto-trim-windows2',
            'mode': 'full', 'status': 'passed', 'candidate_runtime': assets.RUNTIME_VERSION,
            'candidate_commit': assets.SOURCE_COMMIT, 'llama_cpp_commit': assets.LLAMA_COMMIT,
            'candidate_config': report_config,
            'candidate_config_sha256': assets.CONFIG_SHA256,
            'protected_features': {'native_context': 262144, 'cpu_vision': True,
                                   'conversation_cache_mib': 8192, 'conversation_cache_slots': 4},
            'cache_test': {'passed': True, 'api_cached_token_hits': 3, 'restore_log_events': 16},
            'vision_test': {'cpu_vision_log_confirmed': True, 'expected_color': 'red',
                            'expected_color_found_in_answer': True, 'answer_text': 'Red'},
            'long_prefill_test': {'usage': {'prompt_tokens': 31509, 'completion_tokens': 16,
                                             'prompt_tokens_details': {'cached_tokens': 0}}},
            'full_context_test': {'native_context_tokens': 262144,
                                  'usage': {'prompt_tokens': 256860, 'completion_tokens': 16,
                                            'prompt_tokens_details': {'cached_tokens': 0}}},
            'shutdown': {'clean': True},
            'telemetry': {'sha256': 'c6b91bf5f2c1a9cb594eaa309245627e5a8d3492d5d357cf71beadd2fe53a450',
                          'min_ram_available_bytes': 40 * 1024**3,
                          'per_gpu': {str(i): {'peak_memory_used_mib': 22000} for i in range(4)}},
        }
        candidate._validate_results(Path('/tmp'), results, inputs)
        results['full_context_test']['usage']['prompt_tokens_details']['cached_tokens'] = 1
        with self.assertRaisesRegex(SafetyError, 'native-context'):
            candidate._validate_results(Path('/tmp'), results, inputs)


if __name__ == '__main__':
    unittest.main()
