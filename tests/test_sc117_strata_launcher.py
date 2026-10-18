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
                RUNTIME_VERSION='0.1.39', TRIAL_RUNTIME_VERSION='0.1.45',
                TRIAL_STATE_ROOT_NAME='sc117-iq3s-trial', TRIAL_BIND_HOST='0.0.0.0', TRIAL_PORT=8080,
                TRIAL_ENV='STRATA_SC117_IQ3S_TRIAL', TRIAL_DEFAULT_SPEC='3',
                TRIAL_DEFAULT_PIPELINE_WINDOWS='2',
                TRIAL_OVERRIDE_ENV={'STRATA_SC117_IQ3S_TRIAL_PREFILL': 'prefill'},
                TRIAL_EXPERIMENT_ENV={'STRATA_SC117_IQ3S_TRIAL_GPU_ORDER'},
                settings_for_start=lambda **_: {}, authorize_start=lambda *_a, **_k: 'trial',
                require_trial_binding=lambda host: None,
                build_effective_config=lambda *_: {}, build_v01441_trial_config=lambda *_: {}),
            'prepare_iq3_s_sc117_abliterated': module(
                'prepare_iq3_s_sc117_abliterated', verify_readiness=lambda *_a, **_k: None),
            'strata_runtime': module('strata_runtime', LEGACY_VERSION='0.1.39', VERSION='0.1.41',
                                     NEXT_VERSION='0.1.45', BASE_VERSION='0.1.38', PINS={'0.1.39': 'pin'},
                                     runtime_env=lambda *_: {}, prepared_runtime=lambda *_: (Path('/runtimes/0.1.45/source'), {}),
                                     configure=lambda *_a: None),
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

    def test_sc117_dispatch_retires_039_and_requires_explicit_045_trial(self):
        profile = LAUNCHER.sc117_profile
        with mock.patch.object(LAUNCHER, 'data_root', return_value=Path('/data')), \
                mock.patch.object(LAUNCHER, 'ready') as ready, \
                mock.patch.object(LAUNCHER.runtimes, 'configure') as configure:
            for runtime in ('auto', '0.1.39', '0.1.41'):
                with self.subTest(runtime=runtime), self.assertRaisesRegex(
                        SAFETY_ERROR, 'only its explicit trusted-LAN .45 trial is allowed'):
                    LAUNCHER.ready_runtime(profile.PROFILE, runtime, 1, 1)
            ready.assert_not_called()
            configure.assert_not_called()
        with self.assertRaisesRegex(SAFETY_ERROR, 'one serving slot and one batch group'):
            LAUNCHER.ready_runtime(profile.PROFILE, '0.1.45', 2, 1)

    def test_sc117_trial_launcher_requires_the_exact_trusted_lan_host(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(LAUNCHER._effective_host(LAUNCHER.sc117_profile.PROFILE,
                                                      '0.1.45', '127.0.0.1'), '0.0.0.0')
        with mock.patch.dict(os.environ, {'STRATA_HOST': '127.0.0.1'}, clear=True):
            with self.assertRaisesRegex(SAFETY_ERROR, 'fixed to trusted-LAN'):
                LAUNCHER._effective_host(LAUNCHER.sc117_profile.PROFILE, '0.1.45', '0.0.0.0')

    def test_sc117_trial_preflight_requires_exact_isolation_and_cpu_vision(self):
        profile = LAUNCHER.sc117_profile
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            state = home / '.local/state/locallm-strata/sc117-iq3s-trial'
            env = {
                'HOME': str(home), 'STRATA_SC117_IQ3S_TRIAL': '1',
                'STRATA_RUNTIME': '0.1.45', 'STRATA_HOST': '0.0.0.0',
                'STRATA_PORT': '8080', 'STRATA_STATE_ROOT': str(state),
                'PYTHONPYCACHEPREFIX': str(state / 'pycache'),
                'PYTHONDONTWRITEBYTECODE': '1', 'SETUPTOOLS_USE_DISTUTILS': 'stdlib',
            }
            cfg = {'vision': {'gpu': False}, 'args': ['--spec', '3', '--pipeline-windows', '2']}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(LAUNCHER, 'data_root', return_value=Path('/data')), \
                    mock.patch.object(LAUNCHER, 'strata_state', return_value=state), \
                    mock.patch.object(profile, 'settings_for_start', return_value={'trial': True}), \
                    mock.patch.object(profile, 'authorize_start', return_value='trial') as authorize:
                result = LAUNCHER.sc117_trial_readiness_line(profile.PROFILE, '0.1.45', 1, 1, cfg)
            self.assertEqual(result, LAUNCHER.SC117_V0145_TRIAL_READY_CONFIG)
            self.assertTrue(result.startswith('TRIAL CONFIG:'))
            self.assertNotIn('READY CONFIG:', result)
            authorize.assert_called_once_with(Path('/data'), {'trial': True}, host='0.0.0.0', port=8080,
                                              state_root=state)
            bad_cfg = {'vision': {'gpu': False}, 'args': ['--spec', '4', '--pipeline-windows', '2']}
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(SAFETY_ERROR, '--spec 3 and --pipeline-windows 2'):
                    LAUNCHER.sc117_trial_readiness_line(profile.PROFILE, '0.1.45', 1, 1, bad_cfg)

            for key, value in (('STRATA_HOST', '127.0.0.1'), ('STRATA_PORT', '18080'),
                               ('STRATA_STATE_ROOT', str(home / '.local/state/locallm-strata')),
                               ('PYTHONPYCACHEPREFIX', str(home / 'wrong-cache'))):
                bad_env = dict(env)
                bad_env[key] = value
                with mock.patch.dict(os.environ, bad_env, clear=True):
                    with self.assertRaisesRegex(SAFETY_ERROR, 'exact trusted-LAN bind, port, state'):
                        LAUNCHER.sc117_trial_readiness_line(profile.PROFILE, '0.1.45', 1, 1, cfg)
            with mock.patch.dict(os.environ, env | {'STRATA_SC117_IQ3S_TRIAL_GPU_ORDER': '1,0,2,3'}, clear=True):
                with self.assertRaisesRegex(SAFETY_ERROR, 'does not accept experimental tuning overrides'):
                    LAUNCHER.sc117_trial_readiness_line(profile.PROFILE, '0.1.45', 1, 1, cfg)
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(SAFETY_ERROR, 'requires CPU vision'):
                    LAUNCHER.sc117_trial_readiness_line(profile.PROFILE, '0.1.45', 1, 1,
                                                        {'vision': {'gpu': True}})

    def test_sc117_045_route_is_explicit_trial_and_uses_q4_preset(self):
        profile = LAUNCHER.sc117_profile
        candidate = (Path('/q4-preset'), Path('/q4-preset/config.json'),
                     {'args': ['--trim-stage-weights'], 'vision': {'gpu': False}})
        effective = {'model_name': 'sc117', 'vision': {'gpu': False}, 'args': []}
        runtime_source = Path('/runtimes/0.1.45/source')
        with mock.patch.dict(os.environ, {'STRATA_SC117_IQ3S_TRIAL': '1'}, clear=True), \
                mock.patch.object(LAUNCHER, 'data_root', return_value=Path('/data')), \
                mock.patch.object(LAUNCHER.prepare_sc117, 'verify_readiness') as assets_ready, \
                mock.patch.object(LAUNCHER.q4ks_v01441, 'ready_q4ks_v01441', return_value=candidate) as q4_ready, \
                mock.patch.object(LAUNCHER.runtimes, 'prepared_runtime', return_value=(runtime_source, {})) as prepared, \
                mock.patch.object(profile, 'build_v01441_trial_config', return_value=effective) as build:
            result = LAUNCHER.ready_runtime(profile.PROFILE, '0.1.45', 1, 1)
            assets_ready.assert_called_once_with(Path('/data'), full_hash=False)
            q4_ready.assert_called_once_with(Path('/data'))
            prepared.assert_called_once_with(Path('/data'), '0.1.45')
            build.assert_called_once_with(Path('/data'), candidate[2])
            self.assertEqual(result[0:2], (runtime_source, candidate[1]))
            self.assertEqual(result[3], '0.1.45')
            self.assertEqual(result[2]['exe'], str(runtime_source / 'engine/strata'))
            self.assertEqual(result[2]['vision']['exe'], str(runtime_source / 'engine/strata-vision-cpu'))
            self.assertEqual(result[2]['_launcher_python'], str(runtime_source / '.venv/bin/python'))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SAFETY_ERROR, 'only through its explicit unqualified trusted-LAN trial'):
                LAUNCHER.ready_runtime(profile.PROFILE, '0.1.45', 1, 1)

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
