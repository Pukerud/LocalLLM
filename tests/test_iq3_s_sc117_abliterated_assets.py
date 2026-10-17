"""CPU-only identity and path-safety tests for the separate SC117 IQ3_S assets."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import iq3_s_sc117_abliterated_assets as assets
import prepare_iq3_s_sc117_abliterated as prepare
from prepare_iq3_s_sc117_abliterated import SafetyError


class SC117AssetPins(unittest.TestCase):
    def test_exact_repository_revision_and_asset_pins(self):
        self.assertEqual(assets.PROFILE, 'iq3_s-sc117-abliterated')
        self.assertEqual(assets.REPOSITORY,
                         'SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF')
        self.assertEqual(assets.REVISION,
                         'eec4e10a2c29b440a2ae85591fa11569ce752963')
        expected = {
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf':
                ('model_shard_1', 55_063_448_064,
                 '40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c'),
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf':
                ('model_shard_2', 28_800_138_432,
                 '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113'),
            'mmproj-Qwen3.8-Flash-Next-BF16.gguf':
                ('vision_projector', 907_542_944,
                 '2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355'),
            'mtp-q2_0.gguf':
                ('mtp_source_stock_non_abliterated', 889_017_216,
                 'a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d'),
            'strata/rt/dense.bin':
                ('mtp_runtime_dense', 116_099_072,
                 'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a'),
            'strata/rt/experts.bin':
                ('mtp_runtime_experts', 707_788_800,
                 '09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f'),
            'strata/rt/draft_vocab.bin':
                ('mtp_runtime_draft_vocab', 425_196,
                 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'),
        }
        for repo_path, (role, size, sha256) in expected.items():
            with self.subTest(repo_path=repo_path):
                item = assets.ASSETS_BY_REPO_PATH[repo_path]
                self.assertEqual((item.role, item.size, item.sha256), (role, size, sha256))
        dense_text = assets.ASSETS_BY_REPO_PATH['strata/rt/dense.txt']
        self.assertEqual((dense_text.role, dense_text.size, dense_text.git_blob_oid),
                         ('mtp_runtime_dense_config', 1_880,
                          '0a6b354ada9f26d07c4d51684089bdc2fda06a1b'))
        self.assertEqual(assets.MODEL_TOTAL_BYTES, 84_771_129_440)
        self.assertEqual(assets.MTP_SOURCE_VARIANT, 'stock/non-abliterated')

    def test_roles_and_storage_are_isolated_from_existing_iq3_s(self):
        root = Path('/data/localllm-strata')
        model = assets.model_root(root)
        pack = assets.pack_root(root)
        mtp = assets.mtp_runtime_root(root)
        self.assertEqual(pack, root / 'data/packs/iq3_s-sc117-abliterated')
        self.assertEqual(mtp, root / 'data/mtp/iq3_s-sc117-abliterated/rt')
        self.assertNotEqual(model, root / 'data/models/IQ3_S')
        self.assertNotEqual(pack, root / 'data/packs/iq3_s')
        self.assertNotEqual(mtp, root / 'data/mtp/rt')
        self.assertEqual(assets.shard2_reuse_source(root),
                         root / 'data/models/IQ3_S/'
                         'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf')
        self.assertNotIn('mtp-q8_0.gguf', assets.ASSETS_BY_REPO_PATH)


class SC117PreparationSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        # Pack recording intentionally seals outputs read-only; restore temporary fixtures for Windows cleanup.
        for current, dirs, files in os.walk(self.root, topdown=False):
            for name in files:
                try:
                    (Path(current) / name).chmod(0o600)
                except OSError:
                    pass
            for name in dirs:
                try:
                    (Path(current) / name).chmod(0o700)
                except OSError:
                    pass
        self.temp.cleanup()

    def test_new_pack_path_is_required_and_existing_pack_is_never_reused(self):
        prepare.require_new_pack_path(self.root)
        pack = assets.pack_root(self.root)
        pack.mkdir(parents=True)
        with self.assertRaisesRegex(SafetyError, 'new|exists|overwrit'):
            prepare.require_new_pack_path(self.root)

    def test_symlinked_pack_path_is_rejected(self):
        outside = self.root.parent / (self.root.name + '-outside')
        outside.mkdir()
        try:
            assets.pack_root(self.root).parent.mkdir(parents=True)
            assets.pack_root(self.root).symlink_to(outside, target_is_directory=True)
            with self.assertRaises(SafetyError):
                prepare.require_new_pack_path(self.root)
        finally:
            assets.pack_root(self.root).unlink(missing_ok=True)
            outside.rmdir()

    def test_allowlist_rejects_unpinned_or_traversal_paths(self):
        for bad in ('../../outside.gguf', 'mtp-q8_0.gguf', 'strata/rt/other.bin'):
            with self.subTest(path=bad), self.assertRaises(SafetyError):
                prepare.select_assets([bad])

    def test_shard2_reuse_is_explicit_and_bound_to_the_exact_old_file_pin(self):
        shard2 = assets.ASSETS_BY_REPO_PATH[
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf']
        shard1 = assets.ASSETS_BY_REPO_PATH[
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf']
        self.assertEqual(prepare.reuse_source_for(self.root, shard2, enabled=True),
                         assets.shard2_reuse_source(self.root))
        self.assertIsNone(prepare.reuse_source_for(self.root, shard2, enabled=False))
        with self.assertRaises(SafetyError):
            prepare.reuse_source_for(self.root, shard1, enabled=True)

    def test_changed_target_is_refused_without_network_or_overwrite(self):
        item = assets.ASSETS_BY_REPO_PATH[
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf']
        target = assets.asset_path(self.root, item)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'pre-existing changed file')
        before = target.read_bytes()
        with mock.patch.object(prepare.urllib.request, 'urlopen',
                               side_effect=AssertionError('network must not be used')) as network:
            with self.assertRaisesRegex(SafetyError, 'wrong size|checksum'):
                prepare.prepare_assets(self.root, [item.repo_path])
        network.assert_not_called()
        self.assertEqual(target.read_bytes(), before)

    def test_interrupted_partial_is_refused_without_network(self):
        item = assets.ASSETS_BY_REPO_PATH[
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf']
        target = assets.asset_path(self.root, item)
        target.parent.mkdir(parents=True)
        partial = target.with_name(f'.{target.name}.{assets.PROFILE}.part')
        partial.write_bytes(b'interrupted')
        with mock.patch.object(prepare.urllib.request, 'urlopen',
                               side_effect=AssertionError('network must not be used')) as network:
            with self.assertRaisesRegex(SafetyError, 'partial'):
                prepare.prepare_assets(self.root, [item.repo_path])
        network.assert_not_called()
        self.assertEqual(partial.read_bytes(), b'interrupted')

    def test_asset_verifier_rejects_symlink_even_if_target_is_external(self):
        item = assets.ASSETS_BY_REPO_PATH[
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf']
        target = assets.asset_path(self.root, item)
        target.parent.mkdir(parents=True)
        outside = self.root.parent / (self.root.name + '-asset')
        outside.write_bytes(b'fixture')
        try:
            target.symlink_to(outside)
            with self.assertRaisesRegex(SafetyError, 'outside|symlink|regular'):
                prepare.verify_asset_file(self.root, item, target)
        finally:
            target.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)

    def test_native_pack_helper_only_returns_pinned_command_and_never_runs_packer(self):
        source = assets.strata_039_source(self.root)
        tool = source / 'tools/iq_pack.py'
        tool.parent.mkdir(parents=True)
        tool.write_text('# deliberately not executed\\n', encoding='utf-8')
        python = source / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture python executable')
        pin = prepare.strata_runtime.PINS['0.1.39']

        def git_output(_source, *args):
            if args == ('rev-parse', 'HEAD'):
                return (pin + '\n').encode()
            if args == ('show', f'{pin}:tools/iq_pack.py'):
                return tool.read_bytes()
            if args == ('diff', '--name-only', 'HEAD'):
                return b''
            raise AssertionError(args)

        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={'profile': assets.PROFILE}), \
                mock.patch.object(prepare.strata_runtime, 'prepared_runtime', return_value=(source, {})), \
                mock.patch.object(prepare, '_git_output', side_effect=git_output):
            command = prepare.native_pack_command(self.root, strata_source=source)
        self.assertEqual(command, [str(python), str(tool), '--gguf', str(assets.shard1_path(self.root)),
                                   '--out', str(assets.pack_root(self.root))])
        self.assertFalse(assets.pack_root(self.root).exists())
        self.assertFalse(assets.pack_request_path(self.root).exists())

    def test_native_pack_command_keeps_the_virtualenv_entrypoint_when_python_is_a_symlink(self):
        source = assets.strata_039_source(self.root)
        tool = source / 'tools/iq_pack.py'
        tool.parent.mkdir(parents=True)
        tool.write_text('# pinned fixture\\n', encoding='utf-8')
        python = source / '.venv/bin/python'
        target = python.parent / 'python3'
        python.parent.mkdir(parents=True)
        target.write_bytes(b'fixture interpreter binary')
        try:
            python.symlink_to(target.name)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f'platform cannot create the venv symlink fixture: {exc}')
        pin = prepare.strata_runtime.PINS['0.1.39']

        def git_output(_source, *args):
            if args == ('rev-parse', 'HEAD'):
                return pin.encode()
            if args == ('show', f'{pin}:tools/iq_pack.py'):
                return tool.read_bytes()
            if args == ('diff', '--name-only', 'HEAD'):
                return b''
            raise AssertionError(args)

        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}):
            with mock.patch.object(prepare.strata_runtime, 'prepared_runtime', return_value=(source, {})):
                with mock.patch.object(prepare, '_git_output', side_effect=git_output):
                    command = prepare.native_pack_command(self.root, strata_source=source)
        self.assertEqual(command[0], str(python))
        self.assertEqual(python.resolve(), target.resolve())

    def test_pack_request_rejects_boolean_schema_version(self):
        path = assets.pack_request_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema_version': True}), encoding='utf-8')
        path.chmod(0o444)
        with self.assertRaisesRegex(SafetyError, 'unsupported schema version'):
            prepare._read_pack_request(self.root, {})

    def test_build_pack_runs_only_the_pinned_command_in_a_clean_environment(self):
        source = assets.strata_039_source(self.root)
        python = source / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture python executable')
        command = [str(python), str(source / 'tools/iq_pack.py'), '--gguf',
                   str(assets.shard1_path(self.root)), '--out', str(assets.pack_root(self.root))]
        expected = {'profile': assets.PROFILE, 'status': 'recorded'}
        packer = {'version': prepare.PACKER_VERSION,
                  'source_commit': prepare.strata_runtime.PINS['0.1.39'],
                  'source': str(source), 'tool_path': str(source / 'tools/iq_pack.py'),
                  'tool_sha256': 'a' * 64, 'python_path': str(python),
                  'python_resolved_path': str(python.resolve()), 'python_sha256': 'b' * 64}
        with mock.patch.object(prepare, 'native_pack_command', return_value=command), \
                mock.patch.object(prepare, '_pinned_packer_identity', return_value=packer), \
                mock.patch.object(prepare.subprocess, 'run',
                                  return_value=mock.Mock(returncode=0, stdout='packer completed')) as runner, \
                mock.patch.object(prepare, 'record_pack', return_value=expected) as recorder:
            result = prepare.build_native_pack(self.root, strata_source=source)
        self.assertEqual(result, expected)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[0], command)
        self.assertEqual(runner.call_args.kwargs['cwd'], source)
        self.assertEqual(set(runner.call_args.kwargs['env']),
                         {'HOME', 'PATH', 'LANG', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE'})
        recorder.assert_called_once_with(self.root, strata_source=source)

    def test_native_pack_helper_rejects_arbitrary_tool_source(self):
        source = self.root / 'arbitrary-strata'
        tool = source / 'tools/iq_pack.py'
        tool.parent.mkdir(parents=True)
        tool.write_text('pass\\n', encoding='utf-8')
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}):
            with self.assertRaisesRegex(SafetyError, 'only data-root'):
                prepare.native_pack_command(self.root, strata_source=source)

    def test_manifest_validation_fails_closed_on_wrong_pin_duplicate_or_extra_asset(self):
        with mock.patch.object(prepare, '_asset_file_state', return_value={'mtime_ns': 123, 'ctime_ns': 456, 'mode': 0o444}):
            manifest = prepare._manifest_with_file_states(self.root)
            prepare.validate_manifest(manifest, self.root)
        variants = []
        wrong_revision = dict(manifest, revision='0' * 40)
        variants.append(wrong_revision)
        wrong_role = json.loads(json.dumps(manifest))
        wrong_role['assets'][0]['role'] = 'mtp'
        variants.append(wrong_role)
        duplicate = json.loads(json.dumps(manifest))
        duplicate['assets'][1] = duplicate['assets'][0]
        variants.append(duplicate)
        extra = json.loads(json.dumps(manifest))
        extra['unexpected'] = True
        variants.append(extra)
        unsupported_schema = json.loads(json.dumps(manifest))
        unsupported_schema['schema_version'] = prepare.MANIFEST_SCHEMA + 1
        variants.append(unsupported_schema)
        for malformed in variants:
            with self.subTest(manifest=malformed), \
                    mock.patch.object(prepare, '_asset_file_state', return_value={'mtime_ns': 123, 'ctime_ns': 456, 'mode': 0o444}), \
                    self.assertRaises(SafetyError):
                prepare.validate_manifest(malformed, self.root)

    def test_readiness_rejects_malformed_manifest_without_network_or_repair(self):
        pr = assets.profile_root(self.root)
        pr.mkdir(parents=True)
        path = assets.manifest_path(self.root)
        path.write_text('{not json', encoding='utf-8')
        before = path.read_bytes()
        with self.assertRaises(SafetyError):
            prepare.verify_asset_readiness(self.root)
        self.assertEqual(path.read_bytes(), before)

    def _create_pinned_packer(self):
        source = assets.strata_039_source(self.root)
        tool = source / 'tools/iq_pack.py'
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b'fixed pinned iq_pack fixture\n')
        python = source / '.venv/bin/python'
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b'fixture python executable')
        return source, tool

    def _git_output(self, _source, *args):
        pin = prepare.strata_runtime.PINS['0.1.39']
        if args == ('rev-parse', 'HEAD'):
            return (pin + '\n').encode()
        if args == ('show', f'{pin}:tools/iq_pack.py'):
            return (assets.strata_039_source(self.root) / 'tools/iq_pack.py').read_bytes()
        if args == ('diff', '--name-only', 'HEAD'):
            return b''
        raise AssertionError(args)

    def _authorize_pack_and_write_outputs(self):
        source, tool = self._create_pinned_packer()
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={'profile': assets.PROFILE}), \
                mock.patch.object(prepare.strata_runtime, 'prepared_runtime', return_value=(source, {})), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output):
            command = prepare.native_pack_command(self.root, strata_source=source)
            packer = prepare._pinned_packer_identity(self.root, source)
            prepare._atomic_json_at(self.root, assets.pack_request_path(self.root),
                                    prepare._request_payload(self.root, packer), label='native pack request')
        self.assertEqual(command[-2:], ['--out', str(assets.pack_root(self.root))])
        pack = assets.pack_root(self.root)
        (pack / 'tokenizer').mkdir(parents=True)
        shard_records = prepare._pack_source_shards(self.root)
        conversions = {
            'schema': 1,
            'tool': 'tools/iq_pack.py',
            'source_shards': [{'name': Path(record['path']).name, 'size': record['bytes']}
                              for record in shard_records],
            'tensors': [],
        }
        (pack / 'conversions.json').write_text(json.dumps(conversions), encoding='utf-8')
        (pack / 'dense.bin').write_bytes(b'd' * 64)
        index_head = (f'# strata pack index v3 -- generated by tools/iq_pack.py (native experts) '
                      f'from {Path(shard_records[0]["path"]).name}\n')
        index_row = ' '.join(['blk.0.fixture.weight'] + ['0'] * 18)
        (pack / 'index.txt').write_text(index_head + '# align 64 pool 64 tensors 1\n' +
                                        index_row + '\n', encoding='utf-8')
        (pack / 'native_experts.txt').write_text(
            '# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off '
            '[shard] (n_expert 1, total 4; absolute offsets in source)\n'
            '0 1 1 0 4 0 4 8\n', encoding='utf-8')
        tokenizer = pack / 'tokenizer'
        (tokenizer / 'chat_template.jinja').write_text('template\n', encoding='utf-8')
        (tokenizer / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
        (tokenizer / 'token_type.json').write_text('[0,1]', encoding='utf-8')
        (tokenizer / 'tokenizer.json').write_text('{"model": {}}', encoding='utf-8')
        (tokenizer / 'vocab.json').write_text('{"hello": 0}', encoding='utf-8')
        return source, tool

    def _record_pack(self, source):
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={'profile': assets.PROFILE}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output):
            return prepare.record_pack(self.root, strata_source=source)

    def _protected_sentinels(self):
        paths = [self.root / 'data/models/IQ3_S/base.gguf',
                 self.root / 'data/packs/iq3_s/dense.bin',
                 self.root / 'data/mtp/base.gguf']
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'protected base file')
        return {path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_mode, path.stat().st_ino) for path in paths}

    def test_pack_recording_binds_all_outputs_and_keeps_base_model_pack_mtp_and_tool_unchanged(self):
        protected = self._protected_sentinels()
        source, tool = self._authorize_pack_and_write_outputs()
        tool_before = (tool.read_bytes(), tool.stat().st_mtime_ns, tool.stat().st_mode)
        result = self._record_pack(source)
        self.assertEqual(result['profile'], assets.PROFILE)
        self.assertEqual(result['packer']['source_commit'], prepare.strata_runtime.PINS['0.1.39'])
        self.assertEqual(result['packer']['tool_sha256'], prepare.sha256_file(tool))
        self.assertEqual([item['path'] for item in result['pack_files']], list(prepare.PACK_REQUIRED_FILES))
        self.assertEqual(result['source_shards'], prepare._pack_source_shards(self.root))
        self.assertFalse(assets.pack_request_path(self.root).exists())
        self.assertEqual(tool_before, (tool.read_bytes(), tool.stat().st_mtime_ns, tool.stat().st_mode))
        for path, before in protected.items():
            self.assertEqual((path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns,
                              path.stat().st_mode, path.stat().st_ino), before)
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output):
            ready, message = prepare.readiness(self.root)
        self.assertTrue(ready, message)

    def test_same_size_pack_mutation_with_restored_mtime_is_rejected_by_ctime(self):
        source, _ = self._authorize_pack_and_write_outputs()
        self._record_pack(source)
        template = assets.pack_root(self.root) / 'tokenizer/chat_template.jinja'
        before = template.stat()
        template.chmod(0o644)
        template.write_text('tamplate\n', encoding='utf-8')
        template.chmod(0o444)
        os.utime(template, ns=(before.st_atime_ns, before.st_mtime_ns))
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={} ), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output):
            if os.name == 'posix':
                with self.assertRaisesRegex(SafetyError, 'ctime'):
                    prepare.verify_pack_readiness(self.root)
                with self.assertRaisesRegex(SafetyError, 'ctime'):
                    prepare.verify_pack_readiness(self.root, full_hash=True)
            else:
                # Windows st_ctime is creation time, so full hashes remain necessary there.
                prepare.verify_pack_readiness(self.root)
                with self.assertRaisesRegex(SafetyError, 'checksum mismatch'):
                    prepare.verify_pack_readiness(self.root, full_hash=True)

    def test_pack_manifest_rejects_malformed_and_mismatched_report_bindings(self):
        source, _ = self._authorize_pack_and_write_outputs()
        self._record_pack(source)
        path = assets.pack_manifest_path(self.root)
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output):
            value = json.loads(path.read_text(encoding='utf-8'))
            value['source_shards'][0]['sha256'] = '0' * 64
            path.chmod(0o644)
            path.write_text(json.dumps(value), encoding='utf-8')
            path.chmod(0o444)
            with self.assertRaisesRegex(SafetyError, 'source_shards'):
                prepare.verify_pack_readiness(self.root)
            value['source_shards'][0]['sha256'] = prepare._pack_source_shards(self.root)[0]['sha256']
            value['schema_version'] = prepare.PACK_MANIFEST_SCHEMA + 1
            path.chmod(0o644)
            path.write_text(json.dumps(value), encoding='utf-8')
            path.chmod(0o444)
            with self.assertRaisesRegex(SafetyError, 'schema version'):
                prepare.verify_pack_readiness(self.root)
            path.chmod(0o644)
            path.write_text('{bad json', encoding='utf-8')
            path.chmod(0o444)
            with self.assertRaises(SafetyError):
                prepare.verify_pack_readiness(self.root)

    def test_malformed_or_mismatched_packer_report_refuses_pack_record(self):
        source, _ = self._authorize_pack_and_write_outputs()
        conversions = assets.pack_root(self.root) / 'conversions.json'
        report = json.loads(conversions.read_text(encoding='utf-8'))
        report['source_shards'][0]['name'] = 'other-model.gguf'
        conversions.write_text(json.dumps(report), encoding='utf-8')
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output), \
                self.assertRaisesRegex(SafetyError, 'conversions.json'):
            prepare.record_pack(self.root, strata_source=source)
        self.assertFalse(assets.pack_manifest_path(self.root).exists())

    def test_pack_record_rejects_pack_older_than_explicit_command(self):
        source, _ = self._authorize_pack_and_write_outputs()
        pack = assets.pack_root(self.root)
        request_info = assets.pack_request_path(self.root).stat()
        old_ns = request_info.st_mtime_ns - 5_000_000_000
        os.utime(pack, ns=(old_ns, old_ns))
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output), \
                self.assertRaisesRegex(SafetyError, 'not fresh'):
            prepare.record_pack(self.root, strata_source=source)

    def test_pack_record_rejects_unknown_file(self):
        source, _ = self._authorize_pack_and_write_outputs()
        (assets.pack_root(self.root) / 'unexpected.txt').write_text('unknown', encoding='utf-8')
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output), \
                self.assertRaisesRegex(SafetyError, 'unknown'):
            prepare.record_pack(self.root, strata_source=source)

    def test_pack_record_rejects_temporary_file(self):
        source, _ = self._authorize_pack_and_write_outputs()
        (assets.pack_root(self.root) / 'dense.bin.tmp').write_bytes(b'incomplete')
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output), \
                self.assertRaisesRegex(SafetyError, 'temporary'):
            prepare.record_pack(self.root, strata_source=source)

    def test_pack_record_rejects_symlinks(self):
        source, _ = self._authorize_pack_and_write_outputs()
        pack = assets.pack_root(self.root)
        target = pack / 'tokenizer/vocab.json'
        target.unlink()
        outside = self.root / 'outside-vocab.json'
        outside.write_text('{"outside": true}', encoding='utf-8')
        try:
            target.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest('symlink creation is unavailable')
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare, '_git_output', side_effect=self._git_output), \
                self.assertRaisesRegex(SafetyError, 'symlink'):
            prepare.record_pack(self.root, strata_source=source)

    def test_stale_039_source_commit_is_rejected(self):
        source, _ = self._create_pinned_packer()
        with mock.patch.object(prepare, 'verify_asset_readiness', return_value={}), \
                mock.patch.object(prepare.strata_runtime, 'prepared_runtime', return_value=(source, {})), \
                mock.patch.object(prepare, '_git_output', return_value=b'0' * 40):
            with self.assertRaisesRegex(SafetyError, 'stale or mismatched'):
                prepare.native_pack_command(self.root, strata_source=source)
        self.assertFalse(assets.pack_request_path(self.root).exists())

    def test_ordinary_source_readiness_does_not_hash_large_assets(self):
        path = assets.manifest_path(self.root)
        path.parent.mkdir(parents=True)
        state = {'mtime_ns': 123, 'ctime_ns': 456, 'mode': 0o444}
        with mock.patch.object(prepare, '_asset_file_state', return_value=state):
            manifest = prepare._manifest_with_file_states(self.root)
        path.write_text(json.dumps(manifest), encoding='utf-8')
        path.chmod(0o444)
        with mock.patch.object(prepare, '_asset_file_state', return_value=state), \
                mock.patch.object(prepare, 'verify_asset_file') as verify_file:
            prepare.verify_asset_readiness(self.root)
            self.assertEqual([call.kwargs['full_hash'] for call in verify_file.call_args_list],
                             [False] * len(assets.ASSETS))
            verify_file.reset_mock()
            prepare.verify_asset_readiness(self.root, full_hash=True)
            self.assertEqual([call.kwargs['full_hash'] for call in verify_file.call_args_list],
                             [True] * len(assets.ASSETS))

    def test_git_blob_hash_is_computed_using_git_object_format(self):
        data = b'fixture\n'
        expected = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        self.assertEqual(prepare.git_blob_oid(data), expected)


if __name__ == '__main__':
    unittest.main()
