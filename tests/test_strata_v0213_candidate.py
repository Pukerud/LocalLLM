"""Offline candidate/ownership contracts; never start the native engine."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as S
import strata_launcher as L
import strata_v0213_runtime as R
import unsloth_v31_profile as OLD
import unsloth_v31_v0159_profile as C
from engine_safety import SafetyError

ROOT = Path(__file__).resolve().parents[1]


class Candidate(unittest.TestCase):
    def setUp(self):
        # A deployment may now be approved. Locked fixtures must NEVER borrow its
        # real permission or reach real lifecycle work, regardless of test order.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        approval = patch.object(C, 'APPROVAL', Path(folder.name) / 'absent-approval.json')
        approval.start()
        self.addCleanup(approval.stop)

    def test_same_immutable_model_features_and_limits(self):
        a = OLD.canonical_config('/data', '/home/test')
        b = C.canonical_config('/data', '/home/test')
        self.assertEqual(a['sampling'], b['sampling'])
        self.assertEqual(a['gpu'], b['gpu'])
        self.assertEqual(b['parallel'], 2)
        self.assertEqual(a['layer_split'], b['layer_split'])
        self.assertEqual(a['gpu_order'], b['gpu_order'])
        expected = copy.deepcopy(a['args'])
        expected[expected.index('--expert-profile') + 1] = str(C.state_root('/home/test') / 'expert-profile.bin')
        self.assertEqual(b['args'], expected)
        for key in ('model', 'mmproj', 'gpu', 'max_tokens'):
            self.assertEqual(a['vision'][key], b['vision'][key])
        self.assertFalse(b['vision']['gpu'])
        self.assertEqual(b['drafting'], 'mtp')
        self.assertEqual(b['split_engine'], 'off')
        self.assertEqual(b['lib_dirs'], ['/data/runtimes/0.1.59/source/engine'])
        self.assertNotEqual(C.state_root(), OLD.state_root())
        self.assertEqual(b['env']['STRATA_HC_Q8'], '0')  # Not the separate model's Q8 PLE!

    def test_only_reviewed_exact_speed_candidates(self):
        for k in ('STRATA_IQ4_W4A8', 'STRATA_SELECT_TWO_PASS', 'STRATA_CKPT_POOL', 'STRATA_SLOT_COPY_DIRECT'):
            self.assertEqual(C.ENV[k], '1')
        for k in ('STRATA_GDN_TC', 'STRATA_GDN_CONVL2', 'STRATA_ROUTE_PRIOR', 'STRATA_ROUTE_RESIDENT',
                  'STRATA_ROUTE_RESIDENT_MTP', 'STRATA_BATCH_MTP', 'STRATA_PW_WITH_BATCH'):
            self.assertEqual(C.ENV[k], '0')
        self.assertEqual(C.ENV['STRATA_KEEP_LEFTOVERS'], '1')
        self.assertEqual(C.ENV['STRATA_EXTERNAL_LIFECYCLE'], '1')
        self.assertEqual(C.ENV['STRATA_START_RETRIES'], '0')

    def test_mutated_config_refused(self):
        cfg = C.canonical_config('/data')
        for key, value in (('parallel', 1), ('drafting', 'auto'), ('layer_split', '14,24,36'), ('port', 18080)):
            altered = copy.deepcopy(cfg)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(SafetyError):
                C.validate_config('/data', altered)
        C.validate_config('/data', cfg)

    def test_no_environment_variable_is_activation_authority(self):
        with patch.dict(os.environ, {'ACTIVATE': '1', 'STRATA_UNSLOTH_V31_V0159_TRIAL': '1', 'ALLOW_GPU': '1'}):
            with self.assertRaisesRegex(SafetyError, 'LOCKED'):
                C.authorize_start('/data', C.canonical_config('/data'), host='0.0.0.0', port=8080, state=C.state_root())

    def test_locked_hive_choice_refuses_before_inspection_or_actions(self):
        with patch.object(S, 'inspect_owned_workload') as inspect, patch.object(S, 'run') as run, patch.object(S, 'write_journal') as journal:
            with self.assertRaisesRegex(SafetyError, 'LOCKED'):
                S.switch_profile(C.HIVE_PROFILE)
            inspect.assert_not_called()
            run.assert_not_called()
            journal.assert_not_called()

    def test_locked_direct_start_refuses_before_readiness_credentials_or_state(self):
        with patch.object(L, 'ready_runtime') as ready, patch.object(L, 'consume_api_key') as key, patch.object(L, 'write_state') as write:
            with self.assertRaisesRegex(SafetyError, 'LOCKED'):
                L.start(Mock(), C.PROFILE, C.VERSION, 2, 2)
            ready.assert_not_called()
            key.assert_not_called()
            write.assert_not_called()

    def test_cli_locked_start_creates_no_state_or_lease(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ['launcher', '--quickstart', '--profile', C.PROFILE, '--runtime', C.VERSION, '--parallel', '2', '--batch-groups', '2']
            with patch.object(sys, 'argv', argv), patch.object(L.os, 'geteuid', return_value=1000), patch.object(L, 'strata_state', return_value=Path(d)), patch.object(L.host_lifecycle, 'launch_reservation') as reserve, patch.object(L, 'start') as start:
                self.assertEqual(L.main(), 1)
                reserve.assert_not_called()
                start.assert_not_called()
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_offline_registry_is_not_production_approval(self):
        self.assertIn(C.HIVE_PROFILE, S.PREPARED_PROFILES)
        self.assertNotIn(C.HIVE_PROFILE, S.PROFILES)
        with patch.object(S.subprocess, 'run') as run, self.assertRaisesRegex(SafetyError, 'LOCKED'):
            S.preflight_profile(C.HIVE_PROFILE)
        run.assert_not_called()

    def test_readiness_requires_explicit_runtime_and_two_by_two(self):
        for runtime, p, g, ple in (('auto', 2, 2, False), ('0.1.46', 2, 2, False), ('0.1.59', 1, 1, False), ('0.1.59', 2, 1, False), ('0.1.59', 2, 2, True)):
            with self.subTest(runtime=runtime, p=p, g=g, ple=ple), patch.object(C, 'ready') as ready, self.assertRaises(SafetyError):
                L.ready_runtime(C.PROFILE, runtime, p, g, ple)
            ready.assert_not_called()
        with patch.object(C, 'ready', return_value='checked'):
            self.assertEqual(L.ready_runtime(C.PROFILE, C.VERSION, 2, 2), 'checked')

    def test_foreign_state_refused_without_reading_it(self):
        with self.assertRaisesRegex(SafetyError, 'isolated'):
            C.require_state(OLD.state_root())

    def test_wrapper_is_guarded_before_first_mutation(self):
        source = (ROOT / 'hive-llm-miner/profiles' / C.HIVE_PROFILE / 'h-run.sh').read_text()
        self.assertLess(source.index('C.authorize_activation()'), source.index('request_hive_stop()'))
        for key in ('owned_state_identity', 'carries_token', 'START_STATE_IDENTITY', 'STOPPER_ARMED',
                    'docker_workload_info', 'check_port_free', 'unsloth_v31_v0159', C.STATE_NAME, C.READY):
            self.assertIn(key, source)
        self.assertNotIn('systemctl stop', source)
        conf = (ROOT / 'hive-llm-miner/profiles' / C.HIVE_PROFILE / 'llm-hosting.conf').read_text()
        self.assertIn('INFERENCE_TESTED=0', conf)
        self.assertIn('ACTIVATION_ALLOWED=0', conf)

    def test_low_selector_status_reads_policy_but_real_gate_still_precedes_mutation(self):
        source = (ROOT / 'hive-llm-miner/profile-menu.sh').read_text()
        labels = source[source.index('profile_label()'):source.index('read_current_profile()')]
        self.assertIn('C.authorize_activation()', labels)
        self.assertIn('TRIAL AUTHORIZED', labels)
        self.assertIn('LOCKED', labels)
        gate = source.index('if [[ "$target" == unsloth-v31-iq4xs-q8-cvec-2slot-v0159 ]]')
        self.assertLess(gate, source.index('umask 077; exec 9>'))
        self.assertIn('C.authorize_activation()', source[gate:source.index('umask 077; exec 9>')])

    def test_official_package_pins_and_narrow_lifecycle_patch(self):
        self.assertEqual(R.VERSION, '0.1.59')
        self.assertEqual(R.ASSET_BYTES, 723498702)
        self.assertEqual(len(R.ASSET_SHA256), 64)
        self.assertEqual(set(R.OFFICIAL_FILES), {'strata', 'BUILD.json', 'libstrata_kq.so', 'libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13'})
        self.assertNotIn(b'api_key', R.SERVER_NEW)
        self.assertIn(b'STRATA_EXTERNAL_LIFECYCLE', R.SERVER_NEW)
        self.assertNotEqual(R.SERVER_ORIGINAL_SHA256, R.SERVER_PATCHED_SHA256)

    def test_patch_refuses_unpinned_preimage_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'serve/server.py'
            path.parent.mkdir()
            original = b'not the pinned upstream source\n'
            path.write_bytes(original)
            with self.assertRaises(SafetyError):
                R.patch_server(d)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
