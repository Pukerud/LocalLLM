"""Offline PLE-only variants, immutable pins, isolated lifecycle and add-only package tests."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest import mock

import engine_safety as E
import heretic_profile as H
import hive_profile_switch as S
import install_ple_q8_profiles as I
import iq3_s_sc117_abliterated_profile as SC
import ple_q8_assets as A
import ple_q8_profile as Q
import prepare_ple_q8 as P
import strata_launcher as L

REPO = Path(__file__).resolve().parents[1]
READER_SOURCE = Path('/home/user/.local/share/localllm-strata/runtimes/0.1.46/source')


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def gguf_fixture(kind=8, rows=1):
    header = (b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + string(A.TENSOR) +
              struct.pack('<IQQIQ', 2, 160, rows, kind, 0))
    header += b'\0' * (-len(header) % 32)
    return header + (struct.pack('<e', .125) + bytes(range(32))) * 5 * rows


class Contracts(unittest.TestCase):
    def test_pinned_original_q8_table_not_fp8_or_requantized_iq4(self):
        self.assertEqual(A.SHAPE, [160, 320001536])
        self.assertEqual(A.SIZE, 54400261312)
        self.assertEqual(A.TENSOR_BYTES, 320001536 * 170)
        self.assertEqual(A.SHA256, '34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20')
        self.assertIn(A.REVISION, A.url())
        self.assertEqual(len(A.HIVE_PROFILES), 2)

    def test_state_isolation_and_rejection_of_original_or_wrong_profile_state(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Q, 'owner_home', return_value=Path(tmp)):
            for profile, name in A.STATES.items():
                expected = Path(tmp) / '.local/state/locallm-strata' / name
                Q.require_state(profile, expected)
                with self.assertRaises(E.SafetyError):
                    Q.require_state(profile, expected.parent / 'sc117-iq3s-trial')
            with self.assertRaises(E.SafetyError):
                Q.state_root('iq3_s')
            parent = Path(tmp) / '.local/state/locallm-strata'
            parent.mkdir(parents=True)
            (parent / A.STATES[H.PROFILE]).symlink_to(Path(tmp), target_is_directory=True)
            with self.assertRaises(E.SafetyError):
                Q.require_state(H.PROFILE, parent / A.STATES[H.PROFILE])

    def test_overlay_changes_only_ple_adaptive_location_and_api_identity(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Q, 'owner_home', return_value=Path(tmp)), \
             mock.patch.object(Q, 'ready_asset', return_value=Path('/verified/Q8_0.gguf')):
            original = H.canonical_config(Path('/data'), Path('/runtime'))
            for profile in A.STATES:
                baseline = copy.deepcopy(original)
                result = Q.overlay('/data', profile, baseline)
                self.assertEqual(baseline, original)
                for key in original:
                    if key not in ('args', 'model_name', 'aliases'):
                        self.assertEqual(result[key], original[key], key)
                before, after = list(original['args']), list(result['args'])
                for flag in ('--ple-gguf', '--expert-profile'):
                    i = before.index(flag) + 1
                    self.assertNotEqual(before[i], after[i])
                    before[i] = after[i]
                self.assertEqual(before, after)
                self.assertIn('-ple-q8_0', result['model_name'])
                self.assertIn(profile + '-ple-q8', result['aliases'])
                self.assertFalse(result['vision']['gpu'])
                self.assertEqual(result['reasoning_effort'], 'high')

    def test_bad_baseline_flags_and_whole_table_ram_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Q, 'owner_home', return_value=Path(tmp)), \
             mock.patch.object(Q, 'ready_asset', return_value=Path('/table')):
            for mutation in ('duplicate', 'ram'):
                cfg = H.canonical_config('/data', '/runtime')
                if mutation == 'duplicate':
                    cfg['args'] += ['--ple-gguf', '/other']
                else:
                    cfg['args'][cfg['args'].index('--ple-io') + 1] = 'ram'
                with self.subTest(mutation=mutation), self.assertRaises(E.SafetyError):
                    Q.overlay('/data', H.PROFILE, cfg)

    def test_sc117_q8_trial_requires_explicit_variant_and_its_exact_root(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Path, 'home', return_value=Path(tmp)), \
             mock.patch.dict(os.environ, {SC.TRIAL_ENV: '1'}, clear=True):
            base = Path(tmp) / '.local/state/locallm-strata'
            ordinary = base / SC.TRIAL_STATE_ROOT_NAME
            candidate = base / A.STATES[SC.PROFILE]
            kwargs = dict(root='/data', host='0.0.0.0', port=8080)
            self.assertTrue(SC.allow_trial_start(**kwargs, state_root=ordinary))
            self.assertFalse(SC.allow_trial_start(**kwargs, state_root=candidate))
            self.assertTrue(SC.allow_trial_start(**kwargs, state_root=candidate, ple_q8=True))
            self.assertFalse(SC.allow_trial_start(**kwargs, state_root=ordinary, ple_q8=True))
            self.assertFalse(SC.allow_trial_start(root='/data', host='127.0.0.1', port=8080,
                                                  state_root=candidate, ple_q8=True))

    def test_variant_dispatch_keeps_original_heretic_state_check_out_of_q8_path(self):
        cfg = H.canonical_config('/data', '/runtime')
        result = (Path('/runtime'), Path('/config'), cfg)
        with mock.patch.object(Q, 'require_state') as require, \
             mock.patch.object(Q, 'overlay', return_value={'q8': True}) as overlay, \
             mock.patch.object(H, 'ready', return_value=result), mock.patch.object(H, 'require_state') as base_state:
            self.assertEqual(L.ready_runtime(H.PROFILE, '0.1.46', 1, 1, ple_q8=True),
                             (result[0], result[1], {'q8': True}, '0.1.46'))
            base_state.assert_not_called()
            require.assert_called_once()
            overlay.assert_called_once()
        with mock.patch.object(Q, 'require_state'), mock.patch.object(H, 'ready') as ready:
            with self.assertRaises(E.SafetyError):
                L.ready_runtime(H.PROFILE, '0.1.45', 1, 1, ple_q8=True)
            ready.assert_not_called()

    def test_picker_recognizes_both_variant_states_without_inherited_environment(self):
        for name in A.STATES.values():
            with self.subTest(state=name), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                base = home / '.local/state/locallm-strata'
                isolated = base / name
                isolated.mkdir(parents=True)
                identity = {'pid': 7654321}
                (isolated / 'server.json').write_text(json.dumps({'identity': identity}))
                with mock.patch.object(E, 'owner_home', return_value=home), mock.patch.object(E, 'strata_state', return_value=base), \
                     mock.patch.object(E, 'proc', return_value=identity), mock.patch.object(E, 'same_process', return_value=True):
                    self.assertEqual(E.active_engine(), 'strata')

    def test_exact_preflight_for_both_hive_variants_and_rejects_old_readiness(self):
        for profile, base in A.HIVE_PROFILES.items():
            baseline = H.READY if base == H.PROFILE else S.SC117_V0146_READINESS
            for good in (True, False):
                stdout = baseline + (A.READY_SUFFIX if good else '') + '\n'
                completed = type('Completed', (), {'returncode': 0, 'stdout': stdout})()
                with self.subTest(profile=profile, valid=good), mock.patch.object(S.shutil, 'which', return_value='/usr/sbin/runuser'), \
                     mock.patch.object(S.subprocess, 'run', return_value=completed) as run:
                    if good:
                        S.preflight_profile(profile)
                        command = run.call_args.args[0]
                        self.assertIn('--ple-q8', command)
                        self.assertIn('STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/' + A.STATES[base], command)
                        self.assertIn(base, command)
                    else:
                        with self.assertRaises(E.SafetyError):
                            S.preflight_profile(profile)

    def test_readiness_suffix_in_all_real_consumers_and_owned_variant_marker(self):
        for name in ['HostLLM.sh', 'hive-llm-miner/profile-menu.sh',
                     *['hive-llm-miner/profiles/' + p + '/h-run.sh' for p in A.HIVE_PROFILES]]:
            text = (REPO / name).read_text()
            self.assertIn(A.READY_SUFFIX, text, name)
        for profile in A.HIVE_PROFILES:
            text = (REPO / 'hive-llm-miner/profiles' / profile / 'h-run.sh').read_text()
            self.assertIn("state.get('ple_q8') is not True", text)
            self.assertIn(A.SHA256, text)
            for action in ('--stop --ple-q8', '--check-ready --ple-q8', '--quickstart --ple-q8'):
                self.assertIn(action, text)

    def test_menu_mapping_and_runtime_versions_are_explicit(self):
        text = (REPO / 'HostLLM.sh').read_text()
        for row in ['[3] Swift 1.5 Heretic RE — IQ4_NL', '[4] SC117 GSQ-RCO IQ3_S (abliterated) — IQ4_NL',
                    '[5] Swift 1.5 Heretic RE — Q8_0', '[6] SC117 GSQ-RCO IQ3_S (abliterated) — Q8_0',
                    '[15] Set next Hive miner - SC117 IQ3_S | IQ4_NL',
                    '[16] Set next Hive miner - Swift 1.5 Heretic RE | Q8_0',
                    '[17] Set next Hive miner - SC117 IQ3_S | Q8_0']:
            self.assertIn(row, text)
        self.assertIn('sc117_trial_command --stop --ple-q8', text)
        self.assertIn('heretic_profile_command --stop --ple-q8', text)
        self.assertIn('--prepare-ple-q8', (REPO / 'v1strata.sh').read_text())

    def test_start_rehashes_q8_data_before_any_frontend_or_gpu_spawn(self):
        with mock.patch.object(L.os, 'geteuid', return_value=1000), \
             mock.patch.object(L, 'consume_api_key', return_value=None), \
             mock.patch.object(L.host_lifecycle, 'inherited_launch_reservation_fd', return_value=None), \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(L, 'ready_runtime', return_value=(Path('/runtime'), Path('/cfg'), {}, '0.1.46')), \
             mock.patch.object(Q, 'ready_asset', side_effect=E.SafetyError('data changed')) as verify, \
             mock.patch.object(L.subprocess, 'Popen') as spawn:
            with self.assertRaises(E.SafetyError):
                L.start(mock.Mock(), H.PROFILE, '0.1.46', 1, 1, ple_q8=True)
            verify.assert_called_once_with(L.data_root(), full=True)
            spawn.assert_not_called()

    def test_variant_mismatch_cannot_stop_another_ple_session(self):
        for requested_q8 in (False, True):
            with self.subTest(q8=requested_q8), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(L, 'strata_state', return_value=Path(tmp)), \
                 mock.patch.object(Q, 'require_state'), \
                 mock.patch.object(L, 'acquire_action_lock', return_value={'ple_q8': not requested_q8}), \
                 mock.patch.object(L, 'family', return_value=[object()]), mock.patch.object(L, 'stop') as stop, \
                 mock.patch('sys.argv', ['launcher', '--stop', '--profile', H.PROFILE] + (['--ple-q8'] if requested_q8 else [])):
                self.assertEqual(L.main(), 1)
                stop.assert_not_called()

    def test_preparation_refuses_root_before_network_or_file_work(self):
        with mock.patch.object(os, 'geteuid', return_value=0), mock.patch('sys.argv', ['prepare', '--download-only']), \
             mock.patch.object(P, 'download') as download:
            with self.assertRaises(SystemExit):
                P.main()
            download.assert_not_called()


@unittest.skipUnless((READER_SOURCE / 'tools/gguf_reader.py').is_file(), 'pinned local header reader not installed')
class Manifest(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            content = gguf_fixture()
            stack.enter_context(mock.patch.multiple(A, SIZE=len(content), SHA256=hashlib.sha256(content).hexdigest(),
                                                     SHAPE=[160, 1], TENSOR_BYTES=170))
            import strata_runtime
            stack.enter_context(mock.patch.object(strata_runtime, 'prepared_runtime', return_value=(READER_SOURCE, {})))
            path = A.table_path(root)
            path.parent.mkdir(parents=True, mode=0o700)
            path.write_bytes(content)
            path.chmod(0o600)
            P.prepare(root)
            yield root, path

    def test_real_header_fixture_roundtrip_and_full_data_hash(self):
        with self.fixture() as (root, path):
            self.assertEqual(Q.ready_asset(root, full=True), path)
            info = path.stat()
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            with self.assertRaises(E.SafetyError):
                Q.ready_asset(root, full=True)

    def test_manifest_and_data_drift_never_silently_adopted(self):
        for mutation in ('hash', 'revision', 'path', 'inode', 'tensor', 'mtime', 'symlink'):
            with self.subTest(mutation=mutation), self.fixture() as (root, path):
                manifest = path.parent / 'prepared.json'
                d = json.loads(manifest.read_text())
                if mutation == 'hash': d['sha256'] = '0' * 64
                elif mutation == 'revision': d['revision'] = '0' * 40
                elif mutation == 'path': d['path'] = '/elsewhere/table.gguf'
                elif mutation == 'inode': d['inode'] += 1
                elif mutation == 'tensor': d['tensor']['type'] = 'IQ4_NL'
                elif mutation == 'mtime': d['mtime_ns'] -= 1
                elif mutation == 'symlink':
                    other = path.with_suffix('.other')
                    path.rename(other)
                    path.symlink_to(other)
                manifest.write_text(json.dumps(d))
                with self.assertRaises(E.SafetyError):
                    Q.ready_asset(root)

    def test_wrong_tensor_encoding_and_shape_rejected(self):
        with self.fixture() as (root, path):
            for kind, rows in [(20, 1), (8, 2)]:
                path.write_bytes(gguf_fixture(kind, rows))
                with self.assertRaises(E.SafetyError):
                    P.validate_header(root, path)


class AddOnlyPackage(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        for name in I.FILES:
            (source / name).write_text('new candidate ' + name)
        profiles = root / 'profiles'
        profiles.mkdir()
        active = profiles / 'active-sc117'
        active.mkdir()
        for name in I.FILES:
            (active / name).write_text('unchanged running ' + name)
        return source, profiles, active

    def test_add_is_atomic_idempotent_and_never_changes_active_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, profiles, active = self.fixture(root)
            before = {p.name: p.read_bytes() for p in active.iterdir()}
            destination = profiles / 'new-q8'
            self.assertTrue(I.install_inactive_profile(source, destination))
            self.assertFalse(I.install_inactive_profile(source, destination))
            self.assertEqual(before, {p.name: p.read_bytes() for p in active.iterdir()})
            self.assertEqual(set(p.name for p in destination.iterdir()), set(I.FILES))

    def test_differing_existing_or_symlinked_profile_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, profiles, active = self.fixture(Path(tmp))
            with self.assertRaises(E.SafetyError):
                I.install_inactive_profile(source, active)
            self.assertTrue((active / 'h-run.sh').read_text().startswith('unchanged running'))
            link = profiles / 'linked'
            link.symlink_to(active, target_is_directory=True)
            with self.assertRaises(E.SafetyError):
                I.install_inactive_profile(source, link)

    def test_selector_publication_does_not_touch_current_or_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, profiles, active = self.fixture(root)
            current = root / 'current'
            current.symlink_to('profiles/active-sc117')
            src = root / 'new-menu.sh'
            dest = root / 'profile-menu.sh'
            src.write_text('new labels')
            dest.write_text('old labels')
            before = {p.name: p.read_bytes() for p in active.iterdir()}
            self.assertTrue(I.publish_menu(src, dest))
            self.assertFalse(I.publish_menu(src, dest))
            self.assertEqual(os.readlink(current), 'profiles/active-sc117')
            self.assertEqual(before, {p.name: p.read_bytes() for p in active.iterdir()})


if __name__ == '__main__':
    unittest.main()
