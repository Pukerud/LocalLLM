"""CPU-only configuration and qualification tests for the SC117 IQ3_S candidate."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import iq3_s_sc117_abliterated_assets as assets
import iq3_s_sc117_abliterated_profile as profile
import sc117_qualification_runner as runner
SafetyError = profile.SafetyError
TEST_MACHINE = {
    'machine_id_sha256': 'm' * 64,
    'gpu_uuids': [f'GPU-{index:08x}' for index in range(4)],
    'driver_version': 'test-driver',
}


class SC117ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_shard1 = self.root / 'data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
        self.old_shard2 = self.root / assets.SHARD2_REUSE_SOURCE_RELATIVE
        self.old_projector = self.root / 'data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf'
        self.old_pack = self.root / 'data/packs/iq3_s'
        self.old_mtp = self.root / 'data/mtp/rt'
        self.source = self.root / 'runtimes/0.1.39/source'
        self.baseline = {
            'exe': str(self.source / 'engine/strata'),
            'cwd': str(self.source),
            'args': [
                '--max-context', '262144', '--kv', 'int8', '--kv-resident', '32768',
                '--spec', '4', '--spec-min-p', '0.5', '--vram-reserve-mib', '2048',
                '--vision', '--pack', str(self.old_pack), '--native', str(self.old_shard1),
                '--ple-gguf', str(self.old_shard2), '--mtp', str(self.old_mtp),
                '--prefill', 'auto',
            ],
            'tokenizer': str(self.old_pack / 'tokenizer'),
            'gpu': [0, 1, 2, 3], 'layer_split': 'auto',
            'model_name': 'baseline-model', 'aliases': ['baseline'],
            'host': '0.0.0.0', 'port': 8080,
            'vision': {'exe': str(self.source / 'engine/strata-vision'),
                       'model': str(self.old_shard1), 'mmproj': str(self.old_projector),
                       'gpu': True, 'max_tokens': 1024,
                       'nested': {'unchanged': ['deep', 'data']}},
            'other': {'preserve': True},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_effective_config_rewires_sc117_paths_on_deep_copy(self):
        baseline_before = copy.deepcopy(self.baseline)
        legacy_config = self.root / profile.BASELINE_CONFIG_RELATIVE
        legacy_config.parent.mkdir(parents=True)
        legacy_config.write_text(json.dumps(self.baseline))
        legacy_pack_file = self.old_pack / 'dense.bin'
        legacy_mtp_file = self.old_mtp / 'experts.bin'
        legacy_pack_file.parent.mkdir(parents=True)
        legacy_mtp_file.parent.mkdir(parents=True)
        legacy_pack_file.write_bytes(b'old pack stays read-only')
        legacy_mtp_file.write_bytes(b'old MTP stays read-only')
        protected_before = {path: path.read_bytes() for path in
                            (legacy_config, legacy_pack_file, legacy_mtp_file)}
        effective = profile.build_effective_config(self.root, self.baseline)
        args = effective['args']

        def flag_value(flag):
            self.assertEqual(args.count(flag), 1)
            return args[args.index(flag) + 1]

        self.assertEqual(flag_value('--native'), str(assets.shard1_path(self.root)))
        self.assertEqual(flag_value('--ple-gguf'), str(assets.shard2_path(self.root)))
        self.assertEqual(flag_value('--pack'), str(assets.pack_root(self.root)))
        self.assertEqual(flag_value('--mtp'), str(assets.mtp_runtime_root(self.root)))
        self.assertEqual(effective['tokenizer'], str(assets.pack_root(self.root) / 'tokenizer'))
        self.assertEqual(effective['vision']['model'], str(assets.shard1_path(self.root)))
        self.assertEqual(effective['vision']['mmproj'], str(assets.projector_path(self.root)))
        self.assertEqual(effective['model_name'], assets.MODEL_ID)
        self.assertEqual(self.baseline, baseline_before)
        self.assertEqual(protected_before, {path: path.read_bytes() for path in protected_before})
        self.assertIsNot(effective, self.baseline)
        self.assertIsNot(effective['vision'], self.baseline['vision'])
        self.assertEqual(effective['vision']['nested'], self.baseline['vision']['nested'])
        self.assertEqual(flag_value('--max-context'), '262144')
        self.assertEqual(flag_value('--kv'), 'int8')
        self.assertEqual(flag_value('--kv-resident'), '32768')
        self.assertEqual(flag_value('--spec'), '4')
        self.assertEqual(effective['gpu'], [0, 1, 2, 3])
        for key in ('exe', 'cwd', 'host', 'port', 'other'):
            self.assertEqual(effective[key], self.baseline[key])

    def test_config_rejects_a_non_baseline_identity_or_duplicate_rewire_flags(self):
        bad = copy.deepcopy(self.baseline)
        bad['args'][bad['args'].index('--mtp') + 1] = str(assets.mtp_runtime_root(self.root))
        with self.assertRaises(SafetyError):
            profile.build_effective_config(self.root, bad)
        bad = copy.deepcopy(self.baseline)
        bad['args'].extend(['--pack', str(self.old_pack)])
        with self.assertRaises(SafetyError):
            profile.build_effective_config(self.root, bad)

    def test_v01441_trial_config_reuses_q4_settings_and_rewires_sc117_assets(self):
        q4_assets = types.ModuleType('orca_q4ks_v01441_assets')
        q4_assets.engine_path = lambda root: Path(root) / 'candidate/build/engine/strata'
        q4_assets.source_root = lambda root: Path(root) / 'candidate/source'
        q4_assets.python_path = lambda root: Path(root) / 'candidate/python'
        q4_assets.vision_path = lambda root: Path(root) / 'candidate/build/vision/bin/strata-vision'
        q4_model = self.root / 'data/models/orca-q4_k_s/model.gguf'
        candidate = {
            'exe': str(q4_assets.engine_path(self.root)),
            'cwd': str(q4_assets.source_root(self.root)),
            'args': ['--pack', str(self.root / 'data/packs/orca-q4_k_s'),
                     '--native', str(q4_model), '--ple-gguf', str(q4_model), '--ple-io', 'mmap',
                     '--expert-profile', str(self.root / 'source-99f3dbd0b21d/data/expert-profile.bin'),
                     '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '4',
                     '--mtp', str(self.root / 'data/mtp/rt'), '--max-context', '262144',
                     '--kv', 'int8', '--kv-resident', '32768', '--vision',
                     '--conversation-cache-mib', '8192', '--conversation-cache-slots', '4',
                     '--trim-stage-weights'],
            'tokenizer': str(self.root / 'data/packs/orca-q4_k_s/tokenizer'),
            'gpu': [0, 1, 2, 3], 'gpu_order': 'as_given', 'layer_split': '14,24,36',
            'parallel': None, 'host': '127.0.0.1', 'port': 18844, 'lazy_load': True,
            'env': {'STRATA_PARALLEL': '1', 'STRATA_BATCH_GROUPS': '1'},
            '_launcher_python': str(q4_assets.python_path(self.root)),
            'vision': {'exe': str(q4_assets.vision_path(self.root)), 'model': str(q4_model),
                       'mmproj': str(self.root / 'q4-projector.gguf'), 'gpu': False, 'max_tokens': 1024},
        }
        with patch.dict(sys.modules, {'orca_q4ks_v01441_assets': q4_assets}):
            effective = profile.build_v01441_trial_config(self.root, candidate)
        args = effective['args']
        self.assertEqual(args[args.index('--native') + 1], str(assets.shard1_path(self.root)))
        self.assertEqual(args[args.index('--ple-gguf') + 1], str(assets.shard2_path(self.root)))
        self.assertEqual(args[args.index('--pack') + 1], str(assets.pack_root(self.root)))
        self.assertEqual(args[args.index('--mtp') + 1], str(assets.mtp_runtime_root(self.root)))
        self.assertIn('--trim-stage-weights', args)
        self.assertNotIn('--pipeline-windows', args)
        self.assertEqual(effective['layer_split'], '14,24,36')
        self.assertEqual(effective['vision']['gpu'], False)
        self.assertEqual(effective['vision']['mmproj'], str(assets.projector_path(self.root)))
        self.assertEqual(effective['model_name'], assets.MODEL_ID)
        candidate['args'].extend(['--pipeline-windows', '2'])
        with patch.dict(sys.modules, {'orca_q4ks_v01441_assets': q4_assets}):
            with self.assertRaisesRegex(SafetyError, 'pipeline-windows'):
                profile.build_v01441_trial_config(self.root, candidate)

    def test_default_settings_are_strict_and_trial_overrides_are_explicit(self):
        self.assertEqual(profile.PROFILE, 'iq3_s-sc117-abliterated')
        self.assertEqual(profile.MODEL_ID, assets.MODEL_ID)
        self.assertEqual(profile.DEFAULT_SETTINGS['context'], 262144)
        self.assertEqual(profile.DEFAULT_SETTINGS['kv'], 'int8')
        self.assertEqual(profile.DEFAULT_SETTINGS['kv_resident'], 32768)
        self.assertEqual(profile.DEFAULT_SETTINGS['vision'], 'gpu-bf16')
        self.assertEqual(profile.DEFAULT_SETTINGS['mtp_depth'], 4)
        self.assertEqual(profile.DEFAULT_SETTINGS['parallel'], 1)
        self.assertEqual(profile.DEFAULT_SETTINGS['batch_groups'], 1)
        self.assertEqual(profile.DEFAULT_SETTINGS['prefill'], 'auto')
        self.assertEqual(profile.DEFAULT_SETTINGS['layer_split'], 'auto')
        self.assertEqual(profile.DEFAULT_SETTINGS['conversation_cache_mib'], 8192)
        self.assertEqual(profile.DEFAULT_SETTINGS['conversation_cache_slots'], 4)
        with patch.dict(os.environ, {'STRATA_SC117_IQ3S_TRIAL_PREFILL': '4096'}, clear=True):
            with self.assertRaises(SafetyError):
                profile.settings_for_start(allow_trial_env=True)
        env = {
            'STRATA_SC117_IQ3S_TRIAL': '1',
            'STRATA_SC117_IQ3S_TRIAL_PREFILL': '4096',
            'STRATA_SC117_IQ3S_TRIAL_SPLIT': '14,24,36',
            'STRATA_SC117_IQ3S_TRIAL_CACHE_MIB': '4096',
            'STRATA_SC117_IQ3S_TRIAL_CACHE_SLOTS': '2',
        }
        with patch.dict(os.environ, env, clear=True):
            settings = profile.settings_for_start(allow_trial_env=True)
            self.assertEqual(settings['prefill'], '4096')
            self.assertEqual(settings['layer_split'], '14,24,36')
            self.assertEqual(settings['conversation_cache_mib'], 4096)
            self.assertEqual(settings['conversation_cache_slots'], 2)
            with self.assertRaises(SafetyError):
                profile.settings_for_start(allow_trial_env=False)
        for change in (
                {'context': 131072}, {'kv': 'fp16'}, {'kv_resident': 65536},
                {'mtp_depth': 2}, {'vision': 'cpu'}, {'parallel': 2},
                {'batch_groups': 2}, {'prefill': 'automatic'},
                {'layer_split': '14,24'}, {'layer_split': '24,14,36'}, {'layer_split': '14,24,37'},
                {'conversation_cache_mib': True},
                {'unexpected': 'setting'}):
            with self.subTest(change=change), self.assertRaises(SafetyError):
                profile.validate_settings(dict(profile.DEFAULT_SETTINGS, **change))

    def test_effective_settings_are_provisional_and_preserve_fixed_contract(self):
        configured = profile.build_effective_config(self.root, self.baseline)
        self.assertEqual(configured['layer_split'], 'auto')
        self.assertEqual(configured['args'][configured['args'].index('--prefill') + 1], 'auto')
        self.assertEqual(configured['args'][configured['args'].index('--conversation-cache-mib') + 1], '8192')
        self.assertEqual(configured['args'][configured['args'].index('--conversation-cache-slots') + 1], '4')
        line = profile.default_readiness_line()
        self.assertIn('PROVISIONAL', line)
        self.assertIn('not qualified', line.lower())
        self.assertIn('prefill auto', line)
        self.assertIn('8192 MiB / 4', line)

    def _fingerprint_runtime_fixture(self, *, shared_environment=False):
        prepared_path = assets.manifest_path(self.root)
        pack_manifest_path = assets.pack_manifest_path(self.root)
        prepared_path.parent.mkdir(parents=True)
        prepared_path.write_text('{"candidate":"assets-a"}')
        pack_manifest_path.write_text(json.dumps({"candidate": "pack-a", "pack_files": [
            {"path": "dense.bin", "bytes": 1, "sha256": "a" * 64}]}))
        base_config_path = self.root / 'source-99f3dbd0b21d/strata-iq3_s.json'
        base_config_path.parent.mkdir(parents=True)
        base_config_path.write_text(json.dumps(self.baseline))
        runtime_root = assets.strata_039_source(self.root).parent
        source = runtime_root / 'source'
        source.mkdir(parents=True)
        module = source / 'sc117_import_probe.py'
        module.write_text('VALUE = 1\n')
        subprocess.run(['git', 'init', '-q', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'add', str(module)], check=True)
        overlay = runtime_root / 'python'
        shared_environment_path = self.root / 'source-99f3dbd0b21d/.venv'
        environment = shared_environment_path if shared_environment else overlay
        shared_python = shared_environment_path / 'bin/python'
        shared_python.parent.mkdir(parents=True)
        shared_python.write_bytes(b'fixture shared Python')
        shared_site = shared_environment_path / 'lib/python3.12/site-packages'
        shared_site.mkdir(parents=True)
        overlay_site = overlay / 'lib/python3.12/site-packages'
        if not shared_environment:
            runtime_python = overlay / 'bin/python'
            runtime_python.parent.mkdir(parents=True)
            runtime_python.write_bytes(b'fixture runtime python')
            overlay_site.mkdir(parents=True)
            (overlay_site / 'strata_shared_readonly.pth').write_text(str(shared_site) + '\n')
            environment = overlay
        platform_root = self.root / 'trusted-platform'
        python_home = platform_root / 'bin'
        python_home.mkdir(parents=True)
        (platform_root / 'lib/python3.12').mkdir(parents=True)
        (environment / 'pyvenv.cfg').write_text(
            f'home = {python_home}\ninclude-system-site-packages = false\nversion = 3.12.3\n')
        (source / '.venv').symlink_to(environment, target_is_directory=True)
        runtime_manifest = {'version': '0.1.39', 'source_commit': profile.RUNTIME_COMMIT,
                            'source': str(source), 'python_environment': str(environment),
                            'runtime_assets': [{'path': '/runtime/strata', 'bytes': 10,
                                                'sha256': 'a' * 64}]}
        (runtime_root / 'prepared.json').write_text(json.dumps(runtime_manifest))
        return (prepared_path, pack_manifest_path, base_config_path, runtime_root,
                runtime_manifest, module, overlay_site)

    @contextmanager
    def _patched_runtime_fingerprint(self, runtime_root, runtime_manifest):
        with patch.object(profile.prepare_sc117, 'verify_readiness', return_value={}) as verify:
            with patch.object(profile.strata_runtime, 'prepared_runtime',
                              return_value=(runtime_root / 'source', runtime_manifest)):
                with patch.object(profile, 'machine_identity', return_value=TEST_MACHINE):
                    with patch.object(profile, '_assert_root_controlled_platform_directory'):
                        yield verify

    def test_fingerprint_binds_prepared_pack_runtime_baseline_and_settings(self):
        (prepared_path, pack_manifest_path, base_config_path, runtime_root,
         runtime_manifest, _, _) = self._fingerprint_runtime_fixture()
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest) as verify:
            original = profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
            verify.assert_called_with(self.root, full_hash=False)
            profile.fingerprint(self.root, profile.DEFAULT_SETTINGS, verify_content=True)
            verify.assert_called_with(self.root, full_hash=True)
            changed_settings = dict(profile.DEFAULT_SETTINGS, prefill='4096')
            self.assertNotEqual(original, profile.fingerprint(self.root, changed_settings))
            changed_runtime = dict(runtime_manifest, source_commit='b' * 40)
            with patch.object(profile, 'RUNTIME_COMMIT', 'b' * 40):
                with patch.object(profile.strata_runtime, 'prepared_runtime',
                                  return_value=(runtime_root / 'source', changed_runtime)):
                    self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))
            prepared_path.write_text('{"candidate":"assets-b"}')
            self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))
            prepared_path.write_text('{"candidate":"assets-a"}')
            pack_manifest_path.write_text(json.dumps({"candidate": "pack-b", "pack_files": [
                {"path": "dense.bin", "bytes": 1, "sha256": "a" * 64}]}))
            self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))
            pack_manifest_path.write_text(json.dumps({"candidate": "pack-a", "pack_files": [
                {"path": "dense.bin", "bytes": 1, "sha256": "a" * 64}]}))
            base_config_path.write_text(json.dumps(dict(self.baseline, marker='changed')))
            self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))
            with patch.object(profile, 'machine_identity', return_value=dict(TEST_MACHINE, driver_version='new-driver')):
                self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))
            pyvenv_cfg = Path(runtime_manifest['python_environment']) / 'pyvenv.cfg'
            pyvenv_cfg.write_text(pyvenv_cfg.read_text() + '# changed platform contract\n')
            self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))

    def test_fingerprint_binds_python_runtime_configuration_and_stdlib_identity(self):
        (_, _, _, runtime_root, runtime_manifest, _, _) = self._fingerprint_runtime_fixture()
        environment = Path(runtime_manifest['python_environment'])
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            original = profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
            platform = profile._python_runtime_platform_identity(environment)
            self.assertEqual(platform['python_version'], '3.12.3')
            self.assertEqual(platform['stdlib_path'], str(self.root / 'trusted-platform/lib/python3.12'))
            self.assertEqual(platform['pyvenv_cfg_path'], str(environment / 'pyvenv.cfg'))
            pyvenv_cfg = environment / 'pyvenv.cfg'
            pyvenv_cfg.write_text(pyvenv_cfg.read_text().replace('version = 3.12.3', 'version = 3.12.4'))
            self.assertNotEqual(original, profile.fingerprint(self.root, profile.DEFAULT_SETTINGS))

    def test_runtime_rejects_enabled_system_site_packages(self):
        (_, _, _, runtime_root, runtime_manifest, _, _) = self._fingerprint_runtime_fixture()
        environment = Path(runtime_manifest['python_environment'])
        pyvenv_cfg = environment / 'pyvenv.cfg'
        pyvenv_cfg.write_text(pyvenv_cfg.read_text().replace(
            'include-system-site-packages = false', 'include-system-site-packages = true'))
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, 'disable system site-packages'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_executable_bytecode_caches(self):
        (_, _, _, runtime_root, runtime_manifest, module, overlay_site) = self._fingerprint_runtime_fixture()
        source = runtime_root / 'source'
        source_cache = module.parent / '__pycache__' / 'sc117_import_probe.cpython-312.pyc'
        package_cache = overlay_site / '__pycache__' / 'third_party.cpython-312.pyc'
        source_cache.parent.mkdir()
        package_cache.parent.mkdir()
        source_cache.write_bytes(b'bytecode that could replace tracked source')
        package_cache.write_bytes(b'bytecode that could replace installed package source')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            for cache in (source_cache, package_cache):
                with self.subTest(cache=cache), self.assertRaisesRegex(SafetyError, 'cached Python bytecode'):
                    profile._runtime_python_import_surface(
                        self.root, runtime_root, source, runtime_manifest)
                cache.unlink()
            profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_importable_build_artifacts_but_ignores_invalid_components(self):
        (_, _, _, runtime_root, runtime_manifest, module, _) = self._fingerprint_runtime_fixture()
        llama_artifact = module.parent / 'third_party/llama.cpp/build/generated_helper.py'
        llama_artifact.parent.mkdir(parents=True)
        llama_artifact.write_text('VALUE = "llama.cpp artifact"\n')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            for component in ('build', 'build-extra', 'odd-name', 'testing', '_deps', 'cmakefiles', '__pycache__'):
                with self.subTest(component=component):
                    artifact = module.parent / component / 'generated_helper.py'
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text('VALUE = "untracked artifact"\n')
                    with self.assertRaisesRegex(SafetyError, 'untracked.*Python import'):
                        profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
                    artifact.unlink()
            profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_pth_paths_into_excluded_trees(self):
        (_, _, _, runtime_root, runtime_manifest, _, overlay_site) = self._fingerprint_runtime_fixture()
        source = runtime_root / 'source'
        targets = (source / '.git', source / '.venv', source / 'build',
                   source / 'third_party/llama.cpp')
        (source / 'build').mkdir()
        (source / 'third_party/llama.cpp').mkdir(parents=True)
        pth = overlay_site / 'strata_shared_readonly.pth'
        for target in targets:
            with self.subTest(target=target):
                pth.write_text(str(target) + '\n')
                with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
                    with self.assertRaisesRegex(SafetyError, r'\.pth.*excluded|\.pth.*expected'):
                        profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
        pth.write_text(str(self.root / 'source-99f3dbd0b21d/.venv/lib/python3.12/site-packages') + '\n')

    def test_runtime_python_surface_rejects_unexpected_executable_pth(self):
        (_, _, _, runtime_root, runtime_manifest, _, overlay_site) = self._fingerprint_runtime_fixture()
        pth = overlay_site / 'strata_shared_readonly.pth'
        for line in ('import os; os.environ["SC117_BAD"] = "1"',
                     '\timport\tos; os.environ["SC117_BAD"] = "1"'):
            with self.subTest(line=line):
                pth.write_text(line + '\n')
                with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
                    with self.assertRaisesRegex(SafetyError, 'executable .pth'):
                        profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
        self.assertNotIn('SC117_BAD', os.environ)
        pth.write_text('importer\n')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaises(SafetyError) as error:
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
        self.assertNotIn('executable .pth', str(error.exception))

    def test_runtime_python_surface_rejects_extra_pth_files_and_non_overlay_pth(self):
        (_, _, _, runtime_root, runtime_manifest, _, overlay_site) = self._fingerprint_runtime_fixture()
        extra_overlay_pth = overlay_site / 'unexpected.pth'
        extra_overlay_pth.write_text('')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, r'\.pth.*unexpected'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)
        extra_overlay_pth.unlink()
        shared_site = self.root / 'source-99f3dbd0b21d/.venv/lib/python3.12/site-packages'
        extra_shared_pth = shared_site / 'unexpected.pth'
        extra_shared_pth.write_text('')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, r'\.pth.*unexpected'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_accepts_shared_python_environment_without_overlay(self):
        (_, _, _, runtime_root, runtime_manifest, _, _) = self._fingerprint_runtime_fixture(
            shared_environment=True)
        self.assertFalse((runtime_root / 'python').exists())
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_pth_parse_and_hash_metadata_must_match_final_record(self):
        (_, _, _, runtime_root, runtime_manifest, _, overlay_site) = self._fingerprint_runtime_fixture()
        pth = overlay_site / 'strata_shared_readonly.pth'
        original_read = profile._stable_file_read
        mutated = False

        def read_then_mutate(path, *args, **kwargs):
            nonlocal mutated
            result = original_read(path, *args, **kwargs)
            if Path(path) == pth and not mutated:
                pth.write_text(pth.read_text() + '# changed after parse\n')
                mutated = True
            return result

        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with patch.object(profile, '_stable_file_read', side_effect=read_then_mutate):
                with self.assertRaisesRegex(SafetyError, 'changed during inventory'):
                    profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_customization_packages(self):
        (_, _, _, runtime_root, runtime_manifest, _, overlay_site) = self._fingerprint_runtime_fixture()
        customizer = overlay_site / 'sitecustomize'
        customizer.mkdir()
        (customizer / '__init__.py').write_text('VALUE = 1')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, 'customization package'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_untracked_importable_module(self):
        (_, _, _, runtime_root, runtime_manifest, module, _) = self._fingerprint_runtime_fixture()
        rogue = module.parent / 'untracked_module.py'
        rogue.write_text('VALUE = "untracked"\n')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, 'untracked.*Python import'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_import_files_outside_allowed_roots(self):
        (_, _, _, runtime_root, runtime_manifest, _, _) = self._fingerprint_runtime_fixture()
        (runtime_root / 'unapproved_module.py').write_text('VALUE = 1')
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, 'outside allowed roots'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_runtime_python_surface_rejects_symlinked_import_file(self):
        (_, _, _, runtime_root, runtime_manifest, module, _) = self._fingerprint_runtime_fixture()
        (module.parent / 'linked_module.py').symlink_to(module)
        with self._patched_runtime_fingerprint(runtime_root, runtime_manifest):
            with self.assertRaisesRegex(SafetyError, 'symlink'):
                profile.fingerprint(self.root, profile.DEFAULT_SETTINGS)

    def test_normal_qualification_requires_a_pinned_runner_and_all_live_evidence(self):
        self.assertFalse(hasattr(profile, 'save_qualification'))
        valid_report = self._passing_report()
        valid_fingerprint = valid_report['profile_fingerprint']
        with patch.object(profile, 'fingerprint', return_value=valid_fingerprint), \
                patch.object(profile, 'qualification_path',
                             side_effect=lambda root: profile.profile_root(root) / 'qualification.json'), \
                patch.object(profile, '_validate_qualification_location'), \
                patch.object(profile, 'machine_identity', return_value=TEST_MACHINE):
            passed, message = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
            self.assertFalse(passed)
            self.assertIn('missing', message.lower())
            path = profile.profile_root(self.root) / 'qualification.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(valid_report))
            path.chmod(0o444)
            with patch.object(profile, 'trusted_qualification_runner_sha256',
                              side_effect=SafetyError('trusted runner unavailable')):
                passed, message = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
            self.assertFalse(passed)
            self.assertIn('trusted runner unavailable', message)

            with patch.object(profile, 'trusted_qualification_runner_sha256', return_value='r' * 64), \
                    patch.object(profile, 'trusted_qualification_pin_policy_sha256', return_value='p' * 64):
                passed, message = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
                self.assertTrue(passed, message)
                report = json.loads(path.read_text())
                report['profile_fingerprint'] = '0' * 64
                path.chmod(0o600)
                path.write_text(json.dumps(report))
                passed, _ = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
                self.assertFalse(passed)

                report = self._passing_report()
                report['settings'] = dict(profile.DEFAULT_SETTINGS, prefill='4096')
                path.chmod(0o600)
                path.write_text(json.dumps(report))
                passed, _ = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
                self.assertFalse(passed)

                report = self._passing_report()
                report['checks'].pop('prefill_screen')
                path.chmod(0o600)
                path.write_text(json.dumps(report))
                passed, _ = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
                self.assertFalse(passed)

                report = self._passing_report()
                report['native_context']['prompt_tokens'] = 254999
                path.chmod(0o600)
                path.write_text(json.dumps(report))
                passed, _ = profile.qualification_status(self.root, profile.DEFAULT_SETTINGS)
                self.assertFalse(passed)

                report = self._passing_report()
                expected = report['profile_fingerprint']
                report['mtp_acceptance']['accepted_tokens'] = 0
                self.assertIsNotNone(profile._validate_report(
                    report, profile.DEFAULT_SETTINGS, expected, 'r' * 64,
                    machine=TEST_MACHINE, pin_policy_sha256='p' * 64))
                report = self._passing_report()
                expected = report['profile_fingerprint']
                report['schema_version'] = True
                self.assertIsNotNone(profile._validate_report(
                    report, profile.DEFAULT_SETTINGS, expected, 'r' * 64,
                    machine=TEST_MACHINE, pin_policy_sha256='p' * 64))
                report = self._passing_report()
                expected = report['profile_fingerprint']
                report['machine_identity'] = dict(TEST_MACHINE, driver_version='other-driver')
                self.assertIsNotNone(profile._validate_report(
                    report, profile.DEFAULT_SETTINGS, expected, 'r' * 64,
                    machine=TEST_MACHINE, pin_policy_sha256='p' * 64))
                with self.assertRaises(SafetyError):
                    profile.require_qualification(self.root, profile.DEFAULT_SETTINGS)

    def test_non_loopback_sc117_binding_requires_api_key(self):
        profile.require_api_key_for_binding('127.0.0.1', None)
        profile.require_api_key_for_binding('0.0.0.0', 'test-key')
        with self.assertRaisesRegex(SafetyError, 'requires an API key'):
            profile.require_api_key_for_binding('0.0.0.0', None)

    def test_authorize_start_gates_normal_launch_and_allows_only_explicit_trial(self):
        home = self.root / 'home'
        state = home / '.local/state/locallm-strata' / profile.TRIAL_STATE_ROOT_NAME
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(profile, 'require_qualification') as qualified:
            self.assertEqual(profile.authorize_start(self.root, dict(profile.DEFAULT_SETTINGS),
                                                     host='0.0.0.0', port=8080,
                                                     state_root=home / '.local/state/locallm-strata'),
                             'qualified')
            qualified.assert_called_once()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(profile, 'require_qualification', side_effect=SafetyError('report missing')):
            with self.assertRaisesRegex(SafetyError, 'report missing'):
                profile.authorize_start(self.root, dict(profile.DEFAULT_SETTINGS), host='0.0.0.0',
                                        port=8080, state_root=home / '.local/state/locallm-strata')
        settings = dict(profile.DEFAULT_SETTINGS)
        cache_prefix = state / 'pycache'
        with patch('pathlib.Path.home', return_value=home), \
                patch.dict(os.environ, {'STRATA_SC117_IQ3S_TRIAL': '1',
                                        'PYTHONPYCACHEPREFIX': str(cache_prefix),
                                        'SETUPTOOLS_USE_DISTUTILS': 'stdlib'}, clear=True), \
                patch.object(profile, '_verify_trial_artifacts') as trial_artifacts:
            self.assertEqual(profile.authorize_start(self.root, settings,
                                                     host='127.0.0.1', port=18080, state_root=state), 'trial')
            trial_artifacts.assert_called_once_with(self.root, settings, state)
            with self.assertRaises(SafetyError):
                profile.authorize_start(self.root, settings, host='0.0.0.0',
                                        port=18080, state_root=state)
            trial_artifacts.assert_called_once_with(self.root, settings, state)

    def test_readiness_recomputes_profile_fingerprint_and_rejects_fabricated_probe_receipts(self):
        report = self._passing_report()
        fingerprint = report['profile_fingerprint']
        self.assertEqual(profile._report_profile_fingerprint(
            report['profile_fingerprint_inputs'], TEST_MACHINE, profile.DEFAULT_SETTINGS), fingerprint)
        self.assertEqual(runner.profile_fingerprint(
            report['profile_fingerprint_inputs'], TEST_MACHINE, profile.DEFAULT_SETTINGS), fingerprint)
        report['measured_observations']['probe_receipts'][0]['probe'] = 'caller_fabricated'
        message = profile._validate_report(
            report, profile.DEFAULT_SETTINGS, fingerprint, 'r' * 64,
            machine=TEST_MACHINE, pin_policy_sha256='p' * 64)
        self.assertIsNotNone(message)
        report = self._passing_report()
        report['profile_fingerprint_inputs']['baseline_config_sha256'] = '0' * 64
        message = profile._validate_report(
            report, profile.DEFAULT_SETTINGS, fingerprint, 'r' * 64,
            machine=TEST_MACHINE, pin_policy_sha256='p' * 64)
        self.assertIn('fingerprint does not recompute', message)

    def test_unpinned_evidence_runner_keeps_all_normal_qualification_blocked(self):
        self.assertIsNone(profile.TRUSTED_QUALIFICATION_RUNNER_SHA256)
        with self.assertRaisesRegex(SafetyError, 'runner is not installed'):
            profile.trusted_qualification_runner_sha256()

    def test_production_qualification_path_is_outside_model_users_writable_data_tree(self):
        if os.name == 'posix':
            self.assertEqual(profile.qualification_path(self.root), profile.TRUSTED_QUALIFICATION_PATH)
            self.assertNotEqual(profile.qualification_path(self.root).parent, profile.profile_root(self.root))

    def test_trial_start_is_loopback_port_and_state_root_bound(self):
        home = self.root / 'home'
        state = home / '.local/state/locallm-strata' / profile.TRIAL_STATE_ROOT_NAME
        with patch('pathlib.Path.home', return_value=home), \
                patch.dict(os.environ, {'STRATA_SC117_IQ3S_TRIAL': '1'}, clear=True):
            self.assertTrue(profile.allow_trial_start(self.root, host='127.0.0.1', port=18080,
                                                      state_root=state))
            self.assertFalse(profile.allow_trial_start(self.root, host='0.0.0.0', port=18080,
                                                       state_root=state))
            self.assertFalse(profile.allow_trial_start(self.root, host='127.0.0.1', port=8080,
                                                       state_root=state))
            self.assertFalse(profile.allow_trial_start(self.root, host='127.0.0.1', port=18080,
                                                       state_root=home / '.local/state/locallm-strata'))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(profile.allow_trial_start(self.root, host='127.0.0.1', port=18080,
                                                       state_root=state))

    def _passing_report(self):
        settings = dict(profile.DEFAULT_SETTINGS)
        fingerprint_inputs = {
            'profile': profile.PROFILE, 'model_id': profile.MODEL_ID,
            'model_repository': assets.REPOSITORY, 'model_revision': assets.REVISION,
            'prepared_asset_manifest_sha256': 'a' * 64,
            'source_asset_pins': [('model', 'model_shard', 1, 'b' * 64)],
            'native_pack_manifest_sha256': 'c' * 64,
            'native_pack_files': [('dense.bin', 1, 'd' * 64)],
            'runtime_version': profile.RUNTIME_VERSION, 'runtime_commit': profile.RUNTIME_COMMIT,
            'runtime_manifest_sha256': 'e' * 64,
            'runtime_assets': [('/runtime/strata', 1, '1' * 64),
                               ('/runtime/strata-vision', 1, '2' * 64)],
            'runtime_python': ('/runtime/python', '3' * 64),
            'runtime_python_platform': {'python_version': '3.12.1'},
            'runtime_python_import_surface': [('file', 'module.py', '4' * 64)],
            'baseline_config_path': '/runtime/baseline.json',
            'baseline_config_sha256': '5' * 64,
        }
        fingerprint = profile._report_profile_fingerprint(fingerprint_inputs, TEST_MACHINE, settings)
        policy_sha256 = 'p' * 64
        live_material = {'pin_policy_sha256': policy_sha256, 'state_sha256': '6' * 64}
        live_sha256 = profile._canonical_report_sha256(live_material)
        artifact_sha256 = profile._canonical_report_sha256(fingerprint_inputs)
        binding = {'live_sha256': live_sha256, 'live_material': live_material,
                   'artifact_sha256': artifact_sha256, 'machine_identity': TEST_MACHINE,
                   'profile_fingerprint': fingerprint}
        binding_hash = profile._canonical_report_sha256(binding)
        response_ids = {name: 'chatcmpl-' + f'{index + 1:024x}' for index, name in enumerate(
            name for name in profile.REPORT_PROBE_ORDER if not name.startswith('count_'))}
        receipts, counter = [], 0
        for index, probe in enumerate(profile.REPORT_PROBE_ORDER):
            counted = probe.startswith('count_')
            start = index * 2
            cursor = {'path': str(Path(tempfile.gettempdir()) / 'sc117-engine.log'), 'identity': [1, 2, 3, 4, 5, 6],
                      'offset': start}
            cache_events = []
            if probe in ('cache_A1', 'cache_B1'):
                cache_events = [{'event': 'parked', 'tokens': 512, 'parked': 1,
                                 'bytes': 2048, 'evictions': 0}]
            elif probe in ('cache_A2', 'cache_B2'):
                cache_events = [{'event': 'restored', 'tokens': 512, 'parked': 1,
                                 'bytes': 2048, 'evictions': 0}]
            receipts.append({
                'probe': probe, 'probe_id': f'{index + 1:024x}',
                'response_id': None if counted else response_ids[probe],
                'started_at': start + 1, 'finished_at': start + 1.5,
                'request_counter_before': counter,
                'request_counter_after': counter if counted else counter + 1,
                'prompt_tokens': 255000 if probe in ('count_context', 'native_context') else
                60000 if probe.startswith(('count_prefill_', 'prefill_')) else 20,
                'completion_tokens': 0 if counted else 8 if probe == 'native_context' else 2,
                'finish': 'counted' if counted else 'stop',
                'log_cursor_before': cursor, 'log_cursor_after': dict(cursor, offset=start + 1),
                'cache_events': cache_events, 'identity_before_sha256': binding_hash,
                'identity_after_sha256': binding_hash,
            })
            if not counted:
                counter += 1
        cache = {'parked': True, 'restored': True, 'reused': True,
                 'sequence': [{'conversation': letter, 'turn': turn,
                               'response_id': response_ids[f'cache_{letter}{turn}'],
                               'reused_tokens': 10 if turn == 2 else 0,
                               'cache_events': next(item['cache_events'] for item in receipts
                                                    if item['probe'] == f'cache_{letter}{turn}')}
                              for letter, turn in (('A', 1), ('B', 1), ('A', 2), ('B', 2))]}
        return {
            'schema_version': 1, 'status': 'passed', 'profile': profile.PROFILE,
            'model_id': profile.MODEL_ID, 'machine_identity': TEST_MACHINE,
            'runtime_version': profile.RUNTIME_VERSION, 'runtime_commit': profile.RUNTIME_COMMIT,
            'profile_fingerprint': fingerprint, 'profile_fingerprint_inputs': fingerprint_inputs,
            'evidence_runner_sha256': 'r' * 64, 'settings': settings,
            'checks': {name: True for name in profile.REQUIRED_CHECKS},
            'smoke': {'passed': True, 'response_id': response_ids['smoke']},
            'native_context': {'prompt_tokens': 255000, 'completion_tokens': 8,
                               'tail_marker_found': True, 'response_id': response_ids['native_context']},
            'vision': {'passed': True, 'gpu': True, 'precision': 'bf16', 'prompt_tokens': 20,
                       'projector_sha256': assets.ASSETS_BY_ROLE['vision_projector'].sha256,
                       'response_id': response_ids['vision']},
            'conversation_cache': cache,
            'prefill_screen': {'passed': True, 'selected_prefill': 'auto',
                               'samples': [{'selected_prefill': 'auto', 'prompt_total': 60000,
                                            'response_id': response_ids[f'prefill_{index}'],
                                            'prompt_read_samples': [20000, 40000, 60000]}
                                           for index in (1, 2, 3)]},
            'mtp_acceptance': {'passed': True, 'depth': 4,
                               'head_variant': assets.MTP_SOURCE_VARIANT,
                               'drafted_tokens': 20, 'accepted_tokens': 10,
                               'response_id': response_ids['mtp']},
            'resource_telemetry': {'passed': True, 'source': 'strata_0.1.39_GET_/metrics',
                                   'sample_age_milliseconds': 1,
                                   'gpus': [{'index': i} for i in range(4)]},
            'live_identity': {'before_sha256': live_sha256, 'after_sha256': live_sha256},
            'artifact_identity': {'before_sha256': artifact_sha256, 'after_sha256': artifact_sha256,
                                  'fingerprint_inputs_sha256': artifact_sha256,
                                  'root_pin_policy_sha256': policy_sha256},
            'measured_observations': {
                'identity_binding': {'before': binding, 'after': copy.deepcopy(binding)},
                'probe_receipts': receipts, 'baseline_request_count': 0,
            },
            'recorded_at': '2026-10-16T00:00:00Z',
        }


if __name__ == '__main__':
    unittest.main()
