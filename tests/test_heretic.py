import contextlib
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import heretic_assets as A
import heretic_profile as H
import prepare_heretic as P
import strata_launcher as L
from engine_safety import SafetyError

REPO=Path(__file__).resolve().parents[1]


class Contract(unittest.TestCase):
    def test_publisher_pins_and_legacy_shard_names(self):
        self.assertEqual(len(A.SHARDS),8);self.assertEqual(A.ZIP[0],'strata-pack-heretic-v2-70g-fix.zip')
        self.assertEqual(len(A.REVISION),40)
        for n,size,sha in A.ASSETS:
            self.assertGreater(size,0);self.assertEqual(len(sha),64)
            if n.endswith('.gguf'):self.assertIn('FIX-00001-of-00008-',A.local_name(n))
        self.assertEqual(set(A.PACK_FILES),set(A.PACK_PINS))

    def test_exact_native_context_vision_tuning_cache_and_thinking(self):
        cfg=H.canonical_config(Path('/root-data'),Path('/runtime'))
        args=cfg['args']
        for flag,value in [('--max-context','262144'),('--spec','3'),('--pipeline-windows','2'),
                           ('--conversation-cache-mib','8192'),('--conversation-cache-slots','4'),
                           ('--kv','int8'),('--kv-resident','32768'),('--prefill','auto')]:
            self.assertEqual(args.count(flag),1);self.assertEqual(args[args.index(flag)+1],value)
        self.assertIn('--trim-stage-weights',args);self.assertEqual(cfg['layer_split'],'14,24,36')
        self.assertFalse(cfg['vision']['gpu']);self.assertIn('strata-vision-cpu',cfg['vision']['exe'])
        self.assertEqual(cfg['reasoning_effort'],'high');self.assertEqual(cfg['parallel'],1)
        self.assertNotIn('api_key',cfg);self.assertFalse(cfg['lazy_load']);self.assertEqual(cfg['idle_unload_s'],0)
        self.assertIn('heretic-0.1.46-bf16',cfg['exe']);self.assertEqual(cfg['cwd'],'/runtime')

    def test_state_is_separate(self):
        with mock.patch.object(H,'owner_home',return_value=Path('/operator')):
            H.require_state(Path('/operator/.local/state/locallm-strata')/A.STATE_NAME)
            with self.assertRaises(SafetyError):H.require_state(Path('/operator/.local/state/locallm-strata/sc117-iq3s-trial'))

    def test_all_three_real_readiness_consumers_match(self):
        for name in ['HostLLM.sh','hive-llm-miner/profile-menu.sh','hive-llm-miner/profiles/swift15-heretic-re/h-run.sh']:
            with self.subTest(file=name):self.assertIn(H.READY,(REPO/name).read_text())

    def test_menu_retired_rows_absent_but_heretic_and_sc117_retained(self):
        text=(REPO/'HostLLM.sh').read_text();rows=[l for l in text.splitlines() if 'echo ' in l and '[' in l]
        for n in ['[5] TEST Orca','[6] DeepSeek','[14] Set next Hive miner - Orca','[15] Set next Hive miner - Orca','[16] Set next Hive miner - DeepSeek','[3] Orca']:
            self.assertFalse(any(n in l for l in rows),n)
        self.assertIn('[3] Swift 1.5 Heretic RE',text);self.assertIn('[14] Set next Hive miner - Swift 1.5 Heretic RE',text)
        self.assertIn('[17] Set next Hive miner - SC117',text);self.assertIn('[4] SC117',text)

    def test_runtime_and_slots_refused_before_preparation(self):
        for version,parallel in [('0.1.45',1),('0.1.51',1),('0.1.46',2)]:
            with self.subTest(version=version,slots=parallel),mock.patch.object(H,'ready') as ready:
                with self.assertRaises(SafetyError):L.ready_runtime(H.PROFILE,version,parallel,1)
                ready.assert_not_called()

    def test_dispatch_is_isolated_and_pinned(self):
        result=(Path('/runtime'),Path('/cfg'),{})
        with mock.patch.object(H,'require_state') as state,mock.patch.object(H,'ready',return_value=result),\
             mock.patch.object(L,'strata_state',return_value=Path('/heretic-state')):
            self.assertEqual(L.ready_runtime(H.PROFILE,'0.1.46',1,1),(*result,'0.1.46'))
            state.assert_called_once_with(Path('/heretic-state'))

    def test_picker_recognizes_isolated_owned_state_without_inherited_env(self):
        import engine_safety as E
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);base=home/'.local/state/locallm-strata';isolated=base/A.STATE_NAME
            isolated.mkdir(parents=True);identity={'pid':12345,'uid':os.geteuid()}
            (isolated/'server.json').write_text(json.dumps({'identity':identity}))
            with mock.patch.object(E,'owner_home',return_value=home),mock.patch.object(E,'strata_state',return_value=base),\
                 mock.patch.object(E,'proc',return_value=identity),mock.patch.object(E,'same_process',return_value=True):
                self.assertEqual(E.active_engine(),'strata')
            (isolated/'server.json').unlink();(isolated/'server.json').symlink_to(home/'outside.json')
            with mock.patch.object(E,'owner_home',return_value=home),mock.patch.object(E,'strata_state',return_value=base):
                with self.assertRaises(SafetyError):E.active_engine()

    def test_switch_requires_pinned_preflight_and_retired_models_are_not_targets(self):
        import hive_profile_switch as S
        self.assertIn(A.PROFILE,S.PROFILES)
        self.assertNotIn('deepseek41-abiray-iq2_xxs',S.PROFILES)
        self.assertFalse(any(p.startswith('orca-') for p in S.PROFILES))
        with mock.patch.object(S.shutil,'which',return_value=None),mock.patch.object(S,'run') as mutate:
            with self.assertRaises(SafetyError):S.preflight_profile(A.PROFILE)
            mutate.assert_not_called()
        result=type('Result',(),{'returncode':0,'stdout':H.READY+'\n'})()
        with mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=result) as invoke:
            S.preflight_profile(A.PROFILE)
            args=invoke.call_args.args[0]
            self.assertIn('STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/swift15-heretic-re',args)
            self.assertIn('0.1.46',args)

    def test_asset_paths_refuse_symlinks_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'linked').symlink_to(root,target_is_directory=True)
            for p in [root/'linked/file',root.parent/'escape']:
                with self.assertRaises(SafetyError):P.safe(root,p)

    def test_zip_is_data_only_and_refuses_links_duplicates_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with mock.patch.object(A,'PACK_FILES',('index.txt',)),mock.patch.object(A,'PACK_PINS',{'index.txt':(2,'unused')}):
                good=root/'good.zip'
                with zipfile.ZipFile(good,'w') as z:z.writestr(A.PACK_ROOT+'/index.txt',b'ok')
                P.extract_pack(good,root/'pack');self.assertEqual((root/'pack/index.txt').read_bytes(),b'ok')
                for mode in ['traversal','link','duplicate','size']:
                    bad=root/(mode+'.zip')
                    with zipfile.ZipFile(bad,'w') as z:
                        info=zipfile.ZipInfo(A.PACK_ROOT+'/index.txt' if mode!='traversal' else '../index.txt')
                        if mode=='link':info.external_attr=(stat.S_IFLNK|0o777)<<16
                        z.writestr(info,b'bad' if mode=='size' else b'ok')
                        if mode=='duplicate':z.writestr(info,b'ok')
                    with self.subTest(mode=mode),self.assertRaises(SafetyError):P.extract_pack(bad,root/mode)


class Readiness(unittest.TestCase):
    def fixture(self,root,stack):
        pr=A.profile_root(root);source=root/'runtime';pr.mkdir(parents=True)
        def asset(name,data):
            p=pr/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return P.record(p)
        shards=(('s1.gguf',3,hashlib.sha256(b'one').hexdigest()),('s2.gguf',3,hashlib.sha256(b'two').hexdigest()))
        projector=('vision.gguf',3,hashlib.sha256(b'vis').hexdigest())
        mtp=(('dense.bin',3,hashlib.sha256(b'mtp').hexdigest()),)
        stack.enter_context(mock.patch.multiple(A,SHARDS=shards,PROJECTOR=projector,MTP=mtp,PACK_FILES=('index.txt',),
            PACK_PINS={'index.txt':(3,hashlib.sha256(b'idx').hexdigest())}))
        assets=[asset('models/s1.gguf',b'one'),asset('models/s2.gguf',b'two'),asset('support/vision.gguf',b'vis'),asset('support/mtp/dense.bin',b'mtp')]
        pack=[asset('pack/index.txt',b'idx')]
        rt=H.runtimes.runtime_root(root,H.VERSION)/'prepared.json';rt.parent.mkdir(parents=True);rt.write_text('{}')
        cfg=pr/'config.json';cfg.write_text(json.dumps(H.canonical_config(root,source)))
        compat=H.C.folder(root)/'prepared.json';compat.parent.mkdir(parents=True);compat.write_text('{}')
        stack.enter_context(mock.patch.object(H.C,'ready',return_value=H.C.engine_path(root)))
        manifest={'repository':A.REPOSITORY,'revision':A.REVISION,'runtime_version':H.VERSION,
                  'compatibility_manifest_sha256':H.runtimes.digest(compat),
                  'runtime_manifest_sha256':H.runtimes.digest(rt),'publisher_pack_sha256':A.ZIP[2],
                  'assets':assets,'pack':pack,'config':P.record(cfg)}
        (pr/'prepared.json').write_text(json.dumps(manifest))
        stack.enter_context(mock.patch.object(H.runtimes,'prepared_runtime',return_value=(source,{})))
        return pr,cfg,manifest

    def test_sealed_tiny_fixture_roundtrip_and_tamper_rejection(self):
        for mutation in ['none','config','pack','shard','support','manifest']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
                root=Path(tmp);pr,cfg,m=self.fixture(root,stack)
                if mutation=='config':cfg.write_text('{}')
                if mutation=='pack':(pr/'pack/index.txt').write_bytes(b'bad')
                if mutation=='shard':(pr/'models/s1.gguf').unlink();(pr/'models/s1.gguf').symlink_to(pr/'models/s2.gguf')
                if mutation=='support':(pr/'support/vision.gguf').unlink()
                if mutation=='manifest':m['revision']='drift';(pr/'prepared.json').write_text(json.dumps(m))
                if mutation=='none':self.assertEqual(H.ready(root)[1],cfg)
                else:
                    with self.assertRaises(SafetyError):H.ready(root)

    def test_no_silent_adoption_of_self_reported_pack_hash(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            root=Path(tmp);pr,cfg,m=self.fixture(root,stack)
            (pr/'pack/index.txt').write_bytes(b'new');m['pack']=[P.record(pr/'pack/index.txt')]
            (pr/'prepared.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(SafetyError,'publisher pack checksum'):H.ready(root)


class Compatibility(unittest.TestCase):
    def test_refuses_original_source_drift_for_every_patch(self):
        for relative in H.C.ORIGINALS:
            with self.subTest(file=relative),self.assertRaises(SafetyError):H.C.patch(relative,b'unreviewed source')

    def test_exact_patch_anchors_add_only_bf16_path_and_refuse_reapplication(self):
        import ast
        tree=ast.parse(Path(H.C.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='patch')
        branch=next(n for n in function.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call))
        bodies=[branch.body,branch.orelse[0].body,branch.orelse[0].orelse]
        for relative,body in zip(H.C.ORIGINALS,bodies):
            calls=[n for stmt in body for n in ast.walk(stmt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='replace']
            raw=('\n'.join(n.args[1].value for n in calls)).encode()
            with self.subTest(file=relative),mock.patch.dict(H.C.ORIGINALS,{relative:hashlib.sha256(raw).hexdigest()}):
                patched=H.C.patch(relative,raw)
                self.assertNotEqual(patched,raw);self.assertIn(b'Bf16InF32',patched)
                with self.assertRaises(SafetyError):H.C.patch(relative,patched)

    def test_artifact_drift_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(b'known');sha=hashlib.sha256(b'known').hexdigest();H.C.check(p,sha)
            with self.assertRaises(SafetyError):H.C.check(p,'0'*64)
            link=Path(tmp)/'link';link.symlink_to(p)
            with self.assertRaises(SafetyError):H.C.check(link,sha)


if __name__=='__main__':unittest.main()
