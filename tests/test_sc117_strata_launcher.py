"""Focused offline tests for SC117 dispatch and secret handling in the launcher."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name, **attributes):
    value = types.ModuleType(name)
    for key, item in attributes.items():
        setattr(value, key, item)
    return value


class LauncherModule:
    """Load strata_launcher with inert dependency stubs; no host or GPU probes run."""

    @staticmethod
    def load():
        safe_error = type('SafetyError', (RuntimeError,), {})
        stubs = {
            'fcntl': module('fcntl', flock=lambda *_: None, LOCK_EX=1, LOCK_NB=2),
            'engine_safety': module('engine_safety', SafetyError=safe_error,
                                    command_output=lambda *_: '', docker_empty=lambda: None,
                                    launch_gate=lambda *_: None, owner_home=lambda: Path('/tmp'),
                                    proc=lambda *_: None, same_process=lambda *_: False,
                                    signal_identity=lambda *_: None, strata_state=lambda: Path('/tmp/state')),
            'local_api_client': module('local_api_client', open_local_api=lambda *_a, **_k: None),
            'hosting_lifecycle': module('hosting_lifecycle'),
            'prepare_strata': module('prepare_strata', DIGESTS={}),
            'orca_assets': module('orca_assets', MODEL_ID='orca', PROFILE='orca-profile'),
            'orca_profile': module('orca_profile', configure_native_context=lambda *_: None,
                                   ready_orca=lambda *_: None),
            'orca_q4ks_assets': module('orca_q4ks_assets', PROFILE='q4-profile', MODEL_ID='q4',
                                       SOURCE_COMMIT='q4-commit', RUNTIME_VERSION='0.1.39'),
            'orca_q4ks_profile': module('orca_q4ks_profile', configure_native_context=lambda *_: None,
                                        configure_q4ks_vision=lambda *_: None,
                                        profile_root=lambda *_: Path('/tmp'), ready_q4ks=lambda *_: None,
                                        record_32k_validation=lambda *_: None,
                                        rollback_q4ks_vision=lambda *_: None),
            'orca_q4ks_v0141_profile': module('orca_q4ks_v0141_profile'),
            'orca_q4ks_v01441_profile': module('orca_q4ks_v01441_profile',
                                               RUNTIME_VERSION='0.1.44.1',
                                               SOURCE_COMMIT='v01441-pin',
                                               ready_q4ks_v01441=lambda *_: None),
            'iq3_s_sc117_abliterated_assets': module('iq3_s_sc117_abliterated_assets'),
            'iq3_s_sc117_abliterated_profile': module(
                'iq3_s_sc117_abliterated_profile', PROFILE='iq3_s-sc117-abliterated',
                RUNTIME_VERSION='0.1.39', TRIAL_RUNTIME_VERSION='0.1.44.1',
                TRIAL_ENV='STRATA_SC117_IQ3S_TRIAL', settings_for_start=lambda **_: {},
                build_effective_config=lambda *_: {}, build_v01441_trial_config=lambda *_: {}),
            'prepare_iq3_s_sc117_abliterated': module(
                'prepare_iq3_s_sc117_abliterated', verify_readiness=lambda *_a, **_k: None),
            'strata_runtime': module('strata_runtime', LEGACY_VERSION='0.1.39', VERSION='0.1.41',
                                     BASE_VERSION='0.1.38', PINS={'0.1.39': 'pin'},
                                     runtime_env=lambda *_: {}, configure=lambda *_a: None),
        }
        name = '_test_strata_launcher_sc117'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'strata_launcher.py')
        loaded = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(loaded)
        return loaded, safe_error


LAUNCHER, SAFETY_ERROR = LauncherModule.load()


class SC117LauncherSecretTests(unittest.TestCase):
    def test_launcher_consumes_and_closes_only_the_pipe_descriptor(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'private-token')
            os.close(write_fd)
            write_fd = None
            with mock.patch.dict(os.environ, {'STRATA_API_KEY_FD': str(read_fd)}, clear=True):
                self.assertEqual(LAUNCHER.consume_api_key(), 'private-token')
                self.assertNotIn('STRATA_API_KEY_FD', os.environ)
            with self.assertRaises(OSError):
                os.fstat(read_fd)
        finally:
            if write_fd is not None:
                os.close(write_fd)
            try:
                os.close(read_fd)
            except OSError:
                pass

    def test_launcher_rejects_plaintext_environment_credentials(self):
        with mock.patch.dict(os.environ, {'STRATA_API_KEY': 'private-token'}, clear=True):
            with self.assertRaisesRegex(SAFETY_ERROR, 'environment input is unsupported'):
                LAUNCHER.consume_api_key()
            self.assertNotIn('STRATA_API_KEY', os.environ)

    def test_persistent_run_config_always_redacts_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run-config.json'
            config = {'host': '127.0.0.1', 'api_key': 'private-token', 'args': ['--max-context', '262144']}
            LAUNCHER.write_private_run_config(path, config)
            saved = json.loads(path.read_text())
            self.assertNotIn('api_key', saved)
            self.assertEqual(saved['host'], '127.0.0.1')
            self.assertEqual(config['api_key'], 'private-token')

    def test_sc117_dispatch_requires_explicit_039_and_builds_separate_profile(self):
        profile = LAUNCHER.sc117_profile
        source, config_path, baseline, effective = Path('/source'), Path('/base.json'), {'args': []}, {'sc117': True}
        with mock.patch.object(LAUNCHER, 'data_root', return_value=Path('/data')), \
                mock.patch.object(LAUNCHER, 'ready', return_value=(source, config_path, baseline)) as ready, \
                mock.patch.object(LAUNCHER.runtimes, 'configure',
                                  return_value=(source, config_path, baseline, '0.1.39')) as configure, \
                mock.patch.object(profile, 'settings_for_start', return_value={'pinned': True}), \
                mock.patch.object(profile, 'build_effective_config', return_value=effective):
            result = LAUNCHER.ready_runtime(profile.PROFILE, '0.1.39', 1, 1)
            ready.assert_called_once_with(profile.PROFILE)
            configure.assert_called_once_with(Path('/data'), source, config_path, baseline,
                                               'iq3_s', '0.1.39', 1, 1)
            self.assertEqual(result, (source, config_path, effective, '0.1.39'))
        with self.assertRaisesRegex(SAFETY_ERROR, 'pinned .39 runtime or explicit loopback-only .44.1 trial'):
            LAUNCHER.ready_runtime(profile.PROFILE, 'auto', 1, 1)
        with self.assertRaisesRegex(SAFETY_ERROR, 'one serving slot and one batch group'):
            LAUNCHER.ready_runtime(profile.PROFILE, '0.1.39', 2, 1)

    def test_sc117_0441_route_is_explicit_trial_and_uses_q4_candidate_preset(self):
        profile = LAUNCHER.sc117_profile
        candidate = (Path('/q4-0441'), Path('/q4-0441/config.json'), {'args': ['--trim-stage-weights']})
        effective = {'model_name': 'sc117'}
        with mock.patch.dict(os.environ, {'STRATA_SC117_IQ3S_TRIAL': '1'}, clear=True), \
                mock.patch.object(LAUNCHER, 'data_root', return_value=Path('/data')), \
                mock.patch.object(LAUNCHER.prepare_sc117, 'verify_readiness') as assets_ready, \
                mock.patch.object(LAUNCHER.q4ks_v01441, 'ready_q4ks_v01441', return_value=candidate) as q4_ready, \
                mock.patch.object(profile, 'build_v01441_trial_config', return_value=effective) as build:
            result = LAUNCHER.ready_runtime(profile.PROFILE, '0.1.44.1', 1, 1)
            assets_ready.assert_called_once_with(Path('/data'), full_hash=False)
            q4_ready.assert_called_once_with(Path('/data'))
            build.assert_called_once_with(Path('/data'), candidate[2])
            self.assertEqual(result, (candidate[0], candidate[1], effective, '0.1.44.1'))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SAFETY_ERROR, 'trial-only until its own qualification'):
                LAUNCHER.ready_runtime(profile.PROFILE, '0.1.44.1', 1, 1)

    def test_sc117_baseline_readiness_does_not_require_generic_model_or_pack_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source-99f3dbd0b21d'
            engine = source / 'engine'
            python = source / '.venv/bin/python'
            engine.mkdir(parents=True)
            python.parent.mkdir(parents=True)
            python.write_bytes(b'python-placeholder')
            executables = [engine / 'strata', engine / 'strata-vision']
            for path in executables:
                path.write_bytes(path.name.encode())
            model_relatives = (
                'data/models/IQ3_S/shard-1.gguf',
                'data/models/IQ3_S/shard-2.gguf',
                'data/models/projector.gguf',
            )
            model_paths = [root / relative for relative in model_relatives]
            pack = root / 'data/packs/iq3_s'
            cfg = {
                'exe': str(executables[0]), 'cwd': str(source),
                'args': ['--max-context', '262144', '--kv', 'int8', '--spec', '4',
                         '--vram-reserve-mib', '2048', '--kv-resident', '32768',
                         '--spec-min-p', '0.5', '--pack', str(pack), '--native', str(model_paths[0]),
                         '--ple-gguf', str(model_paths[1]), '--mtp', str(root / 'data/mtp/rt'),
                         '--prefill', 'auto', '--vision'],
                'gpu': [0, 1, 2, 3], 'layer_split': 'auto', 'tokenizer': str(pack / 'tokenizer'),
                'vision': {'exe': str(executables[1]), 'model': str(model_paths[0]),
                           'mmproj': str(model_paths[2]), 'gpu': True},
            }
            config_path = source / 'strata-iq3_s.json'
            config_path.write_text(json.dumps(cfg))
            runtime_assets = [{'path': str(path), 'bytes': path.stat().st_size,
                               'mtime_ns': path.stat().st_mtime_ns,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in executables]
            (root / 'prepared.json').write_text(json.dumps({
                'source_commit': LAUNCHER.SOURCE_COMMIT, 'runtime_assets': runtime_assets,
            }))
            with mock.patch.object(LAUNCHER, 'DIGESTS', dict.fromkeys(model_relatives, 'unused')), \
                    mock.patch.object(LAUNCHER, 'data_root', return_value=root), \
                    mock.patch.object(LAUNCHER, 'command_output', return_value=LAUNCHER.SOURCE_COMMIT), \
                    mock.patch.object(LAUNCHER.prepare_sc117, 'verify_readiness') as verify:
                ready_source, ready_config, ready_cfg = LAUNCHER.ready(LAUNCHER.sc117_profile.PROFILE)
                verify.assert_called_once_with(root, full_hash=False)
            self.assertEqual(ready_source, source)
            self.assertEqual(ready_config, config_path)
            self.assertEqual(ready_cfg, cfg)
            self.assertFalse(pack.exists())
            self.assertFalse(any(path.exists() for path in model_paths))


if __name__ == '__main__':
    unittest.main()
