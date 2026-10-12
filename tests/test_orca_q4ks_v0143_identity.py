"""CPU-only gates for the isolated, opt-in Q4_K_S / Strata 0.1.43 candidate."""
import hashlib
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

import local_api_client
import orca_q4ks_v0143_assets as assets
import orca_q4ks_v0143_profile as candidate
import orca_q4ks_v0143_native_validation as native_validation
import orca_q4ks_v0143_validation as validation_runner
import strata_runtime as runtimes
from engine_safety import SafetyError


class V0143Isolation(unittest.TestCase):
    def test_exact_pin_and_candidate_paths_do_not_alias_v0141_or_default(self):
        root = Path('/tmp/strata-data')
        self.assertEqual(runtimes.VERSION, '0.1.41')
        self.assertEqual(assets.RUNTIME_VERSION, '0.1.43')
        self.assertEqual(assets.SOURCE_COMMIT, '7695859aa13afb248bdb4d36f9b2e95707db079d')
        self.assertEqual(runtimes.PINS['0.1.43'], assets.SOURCE_COMMIT)
        self.assertNotEqual(assets.profile_root(root), Path('/tmp/strata-data/profiles/orca-q4_k_s-v0141'))
        self.assertNotEqual(assets.source_root(root), Path('/tmp/strata-data/runtimes/orca-q4_k_s-v0141/source'))
        self.assertNotIn(assets.PROFILE, runtimes.PROFILES)
        self.assertEqual(assets.VISION_SOURCE_SHA256, '1fb2b3fb797ec402')
        self.assertNotEqual(assets.VISION_SOURCE_SHA256, '6f47618c713ce0a2')

    def test_route_tail_skip_default_is_not_suppressed(self):
        env = runtimes.runtime_env('0.1.43', 1)
        self.assertEqual(env['STRATA_VERIFY_ALL_RESIDENT'], '1')
        self.assertEqual(env['STRATA_BATCH_MTP'], '0')
        self.assertNotIn('STRATA_ROUTE_TAIL_SKIP', env)
        self.assertNotIn('STRATA_ROUTE_TAIL_SKIP', runtimes.SAFE_ENV)

    def test_candidate_config_uses_only_the_v0143_engine_and_source_contract(self):
        root = Path('/tmp/strata-data')
        config = candidate.make_config(root)
        self.assertEqual(config['exe'], str(assets.source_root(root) / 'engine/strata'))
        self.assertEqual(config['cwd'], str(assets.source_root(root)))
        self.assertNotEqual(config['exe'], str(Path('/tmp/strata-data/runtimes/orca-q4_k_s-v0141/source/engine/strata')))
        self.assertEqual(config['args'][config['args'].index('--max-context') + 1], '32768')
        self.assertNotIn('--vision', config['args'])

    def test_v0141_validation_evidence_cannot_pass_v0143_gate(self):
        runner_sha = candidate.sha256(Path(validation_runner.__file__).resolve())
        api_sha = candidate.sha256(Path(local_api_client.__file__).resolve())
        manifest = {'runtime_assets': [{'sha256': 'engine-sha'}], 'validation_runner_sha256': runner_sha,
                    'local_api_client_sha256': api_sha}
        report = {'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
                  'passed': True, 'profile': assets.PROFILE, 'validation_runner_sha256': runner_sha,
                  'local_api_client_sha256': api_sha, 'model_id': assets.MODEL_ID,
                  'source_commit': 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf',
                  'runtime_version': '0.1.41', 'runtime_sha256': 'engine-sha',
                  'asset_sha256': {name: item[1] for name, item in assets.ASSETS.items()},
                  'config_sha256': 'config-sha', 'context': 32768, 'parallel': 1, 'text_only': True,
                  'checks': {}, 'memory_peak_mib': 1, 'gpu_memory_peak_mib_by_gpu': [1, 1, 1, 1],
                  'gpu_memory_peak_aggregate_mib': 4, 'ram_available_min_mib': 1,
                  'live_status': {}, 'validation_server_identity': {},
                  'validation_hosting_lease_id': 'a' * 64, 'mtp': {'drafted_tokens': 1, 'accepted_tokens': 0}}
        with self.assertRaisesRegex(SafetyError, '32K text-only'):
            candidate._validation_report(Path('/tmp'), report, 'config-sha', manifest)

    def test_slot_policy_is_strictly_one_one_or_two_two(self):
        import strata_launcher as launcher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native_cfg = {'args': ['--max-context', '262144'], 'vision': {'gpu': False}}
            ready = (Path('/candidate'), Path('/candidate/config.json'), native_cfg)
            with mock.patch.object(launcher, 'data_root', return_value=root), \
                 mock.patch.object(launcher.q4ks_v0143, 'ready_q4ks_v0143', return_value=ready):
                for slots, groups in ((1, 1), (2, 2)):
                    result = launcher.ready_runtime(assets.HOST_PROFILE, '0.1.43', slots, groups)
                    self.assertEqual(result[3], '0.1.43')
                for slots, groups in ((1, 2), (2, 1)):
                    with self.assertRaises(SafetyError):
                        launcher.ready_runtime(assets.HOST_PROFILE, '0.1.43', slots, groups)
            initial_cfg = {'args': ['--max-context', '32768'], 'vision': None}
            with mock.patch.object(launcher, 'data_root', return_value=root), \
                 mock.patch.object(launcher.q4ks_v0143, 'ready_q4ks_v0143',
                                   return_value=(Path('/candidate'), Path('/candidate/config.json'), initial_cfg)):
                self.assertEqual(launcher.ready_runtime(assets.HOST_PROFILE, '0.1.43', 1, 1)[3], '0.1.43')
                with self.assertRaisesRegex(SafetyError, '32K validation is restricted to one slot'):
                    launcher.ready_runtime(assets.HOST_PROFILE, '0.1.43', 2, 2)


if __name__ == '__main__':
    unittest.main()
