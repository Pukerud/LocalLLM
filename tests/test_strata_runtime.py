"""CPU-only provenance, slot-policy and side-by-side rollback checks."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux runtime controller imports')
import strata_runtime as r
from engine_safety import SafetyError


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.base=self.root/'source-99f3dbd0b21d';self.source=r.runtime_root(self.root)/'source'
        py=self.base/'.venv/bin/python';py.parent.mkdir(parents=True);py.write_text('fake private Python')
        self.source.mkdir(parents=True);(self.source/'.venv').symlink_to(self.base/'.venv',target_is_directory=True)
        self.config=self.base/'strata-iq3_s.json'
        self.base_cfg={'exe':str(self.base/'engine/strata'),'args':['--max-context','262144','--kv','int8',
            '--kv-resident','32768','--spec','4'],'vision':{'exe':str(self.base/'engine/strata-vision'),'gpu':True},
            'gpu':[0,1,2,3],'layer_split':'auto','tokenizer':'original/tokenizer'}
        self.config.write_text(json.dumps(self.base_cfg));assets=[]
        for name in ['strata','strata-vision']:
            p=self.source/'engine'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
            assets.append({'path':str(p),'bytes':p.stat().st_size,'sha256':r.digest(p)})
        self.manifest={'version':r.VERSION,'source_commit':r.PIN,'llama_commit':r.LLAMA_PIN,'source':str(self.source),
            'cuda_arch':86,'cuda_toolkit':'/usr/local/cuda-12.9','runtime_assets':assets,
            'python_environment':str(self.base/'.venv'),
            'baseline_configs':{'iq3_s':{'path':str(self.config),'sha256':r.digest(self.config)}}}
        self.mp=r.runtime_root(self.root)/'prepared.json';self.save()
        self.patch=mock.patch.object(r,'command_output',side_effect=lambda cmd:'' if 'diff' in cmd else
                                     r.PINS[Path(cmd[cmd.index('-C')+1]).parent.name]);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def save(self):self.mp.write_text(json.dumps(self.manifest))

    def cfg(self,version=r.VERSION,slots=1,groups=1):
        return r.configure(self.root,self.base,self.config,self.base_cfg,'iq3_s',version,slots,groups)

    def test_absent_selection_keeps_original_runtime(self):
        self.assertEqual(r.selected_version(self.root),r.BASE_VERSION)
        self.assertEqual(self.cfg('auto')[0],self.base)

    def test_candidate_solo_preserves_model_vision_context_and_mtp(self):
        src,path,cfg,version=self.cfg()
        self.assertEqual(src,self.source);self.assertEqual(path,self.config);self.assertEqual(version,r.VERSION)
        self.assertEqual(cfg['args'],self.base_cfg['args']);self.assertEqual(cfg['tokenizer'],self.base_cfg['tokenizer'])
        self.assertEqual(cfg['vision']['gpu'],True);self.assertNotIn('parallel',cfg)
        self.assertEqual(cfg['runtime_env'],r.runtime_env(r.VERSION,1))
        self.assertEqual(cfg['gpu_order'],'as_given')
        self.assertEqual(self.base_cfg['exe'],str(self.base/'engine/strata'))

    def test_two_slots_in_one_engine_and_optional_pipeline(self):
        cfg=self.cfg(slots=2,groups=2)[2]
        self.assertEqual(cfg['parallel'],2);self.assertEqual(cfg['args'][-2:],['--batch-groups','2'])
        self.assertEqual(cfg['runtime_env']['STRATA_VERIFY_ALL_RESIDENT'],'1')
        self.assertEqual(cfg['runtime_env']['STRATA_BATCH_MTP'],'0')
        self.assertEqual(cfg['gpu'],[0,1,2,3]);self.assertEqual(cfg['layer_split'],'auto')
        self.assertEqual(self.base_cfg['args'][-2:],['--spec','4']) # input not mutated

    def install_legacy(self):
        folder=r.runtime_root(self.root,r.LEGACY_VERSION);source=folder/'source';source.mkdir(parents=True)
        (source/'.venv').symlink_to(self.base/'.venv',target_is_directory=True)
        assets=[]
        for name in ['strata','strata-vision']:
            p=source/'engine'/name;p.parent.mkdir(exist_ok=True);p.write_text(name+' legacy')
            assets.append({'path':str(p),'bytes':p.stat().st_size,'sha256':r.digest(p)})
        manifest={**self.manifest,'source':str(source),'version':r.LEGACY_VERSION,
                  'source_commit':r.PINS[r.LEGACY_VERSION],'runtime_assets':assets,
                  'validation':{p:{'single_slot_passed':True,'source_commit':r.PINS[r.LEGACY_VERSION]} for p in r.PROFILES}}
        (folder/'prepared.json').write_text(json.dumps(manifest))
        return source

    def test_tested_legacy_selection_and_workaround_retained(self):
        source=self.install_legacy();r.select(self.root,r.LEGACY_VERSION)
        self.assertEqual(self.cfg('auto')[0],source)
        cfg=self.cfg(r.LEGACY_VERSION,slots=2,groups=1)[2]
        self.assertEqual(cfg['runtime_env'],r.BATCH_ENV)
        self.assertEqual(cfg['args'][-2:],['--batch-groups','1'])

    def test_new_one_group_is_explicit_not_auto(self):
        self.assertEqual(self.cfg(slots=2,groups=1)[2]['args'][-2:],['--batch-groups','1'])

    def test_preparation_does_not_rewrite_other_runtime(self):
        import prepare_strata_runtime as prep
        import strata_launcher as s
        import hosting_lifecycle as h
        self.install_legacy()
        before=(r.runtime_root(self.root,r.LEGACY_VERSION)/'prepared.json').read_bytes()
        with mock.patch.object(sys,'argv',['prepare_strata_runtime.py','--runtime',r.LEGACY_VERSION]),\
             mock.patch.object(s,'data_root',return_value=self.root),\
             mock.patch.object(s,'read_state',return_value={'runtime_version':r.LEGACY_VERSION}),\
             mock.patch.object(s,'runtime_gate'),mock.patch.object(s,'family',return_value=[{'pid':1}]),\
             mock.patch.object(h,'gpu_jobs',return_value=[]),mock.patch.object(prep,'docker_empty'):
            with self.assertRaisesRegex(SafetyError,'runtime is active'):prep.main()
        self.assertEqual((r.runtime_root(self.root,r.LEGACY_VERSION)/'prepared.json').read_bytes(),before)

    def test_old_runtime_cannot_claim_multiple_slots(self):
        with self.assertRaisesRegex(SafetyError,'single-slot'):self.cfg(r.BASE_VERSION,2)

    def test_preparation_cannot_rewrite_active_candidate(self):
        import prepare_strata_runtime as prep
        import strata_launcher as s
        import hosting_lifecycle as h
        with mock.patch.object(sys,'argv',['prepare_strata_runtime.py']),mock.patch.object(s,'data_root',return_value=self.root),\
             mock.patch.object(s,'read_state',return_value={'runtime_version':r.VERSION}),\
             mock.patch.object(s,'runtime_gate'),mock.patch.object(s,'family',return_value=[{'pid':1}]),\
             mock.patch.object(h,'gpu_jobs',return_value=[]),mock.patch.object(prep,'docker_empty'),\
             mock.patch.object(prep.subprocess,'run') as run:
            with self.assertRaisesRegex(SafetyError,'runtime is active'):prep.main()
        run.assert_not_called()

    def test_untested_larger_slot_allocations_refused(self):
        with self.assertRaisesRegex(SafetyError,'separate GPU'):self.cfg(slots=4)

    def test_orca_pipelined_vision_failure_not_offered_as_validated(self):
        with self.assertRaisesRegex(SafetyError,'concurrent vision'):
            r.configure(self.root,self.base,self.config,self.base_cfg,'orca-iq3_xxs',r.VERSION,2,2)

    def test_new_orca_two_groups_require_actual_exact_pin_stamp(self):
        self.manifest['validation']={'orca-iq3_xxs':{'two_group_vision_passed':True,'source_commit':r.PINS[r.LEGACY_VERSION]}};self.save()
        with self.assertRaisesRegex(SafetyError,'concurrent vision'):
            r.configure(self.root,self.base,self.config,self.base_cfg,'orca-iq3_xxs',r.VERSION,2,2)
        self.manifest['validation']['orca-iq3_xxs']['source_commit']=r.PIN
        self.manifest['baseline_configs']['orca-iq3_xxs']=self.manifest['baseline_configs']['iq3_s'];self.save()
        cfg=r.configure(self.root,self.base,self.config,self.base_cfg,'orca-iq3_xxs',r.VERSION,2,2)[2]
        self.assertEqual(cfg['parallel'],2);self.assertEqual(cfg['args'][-2:],['--batch-groups','2'])

    def test_old_orca_two_groups_stay_refused_even_with_a_new_pass_flag(self):
        self.install_legacy();path=r.runtime_root(self.root,r.LEGACY_VERSION)/'prepared.json'
        manifest=json.loads(path.read_text());manifest['validation']['orca-iq3_xxs']['two_group_vision_passed']=True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SafetyError,'concurrent vision'):
            r.configure(self.root,self.base,self.config,self.base_cfg,'orca-iq3_xxs',r.LEGACY_VERSION,2,2)

    def test_rebuild_invalidates_selected_unvalidated_candidate(self):
        (self.root/'runtime-selection.json').write_text(json.dumps({'version':r.VERSION}))
        with self.assertRaisesRegex(SafetyError,'validation expired'):self.cfg('auto')

    def test_invalid_group_count_is_refused(self):
        with self.assertRaisesRegex(SafetyError,'divide'):self.cfg(slots=2,groups=4)

    def test_modified_native_config_not_silently_rewritten(self):
        self.config.write_text('changed')
        with self.assertRaisesRegex(SafetyError,'Baseline profile changed'):self.cfg()
        self.assertEqual(self.config.read_text(),'changed')

    def test_changed_engine_hash_refused(self):
        (self.source/'engine/strata').write_text('different')
        with self.assertRaisesRegex(SafetyError,'executable'):self.cfg()

    def test_source_pin_mismatch_refused(self):
        with mock.patch.object(r,'command_output',return_value='wrong'):
            with self.assertRaisesRegex(SafetyError,'checkout'):self.cfg()

    def test_tracked_source_modifications_refused(self):
        with mock.patch.object(r,'command_output',side_effect=lambda cmd:'serve/server.py' if 'diff' in cmd else r.PIN):
            with self.assertRaisesRegex(SafetyError,'tracked'):self.cfg()

    def test_foreign_private_environment_refused(self):
        self.manifest['python_environment']=str(self.root/'foreign');self.save()
        with self.assertRaisesRegex(SafetyError,'private Python'):self.cfg()

    def test_changed_private_dependency_metadata_refused(self):
        p=self.base/'.venv/pyvenv.cfg';p.write_text('original')
        self.manifest['python_assets']=[{'path':str(p),'sha256':r.digest(p)}];self.save();p.write_text('changed')
        with self.assertRaisesRegex(SafetyError,'dependency contract'):self.cfg()

    def test_effective_slot_fallback_refused(self):
        import strata_launcher as s
        response=io.BytesIO(json.dumps({'engine':r.VERSION,'concurrency':{'serving':1}}).encode())
        with mock.patch.object(s.urllib.request,'urlopen',return_value=response):
            with self.assertRaisesRegex(SafetyError,'silent single-slot'):s.verify_serving({'parallel':2},r.VERSION,18080)

    def test_effective_slots_checked_with_private_api_auth(self):
        import strata_launcher as s
        response=io.BytesIO(json.dumps({'engine':r.VERSION,'concurrency':{'serving':2}}).encode())
        with mock.patch.object(s.urllib.request,'urlopen',return_value=response) as opened:
            self.assertEqual(s.verify_serving({'parallel':2,'api_key':'cpu-fixture'},r.VERSION,18080),2)
        self.assertEqual(opened.call_args.args[0].get_header('Authorization'),'Bearer cpu-fixture')

    def test_native_explicit_groups_identity_checked(self):
        import strata_launcher as s
        state={'allowed_executables':['python','native']}
        for requested,actual in [(1,1),(2,2)]:
            with mock.patch.object(s,'family',return_value=[{'exe':'native','cmd':['native','--batch-groups',str(actual)]}]):
                s.verify_batch_groups(state,r.VERSION,2,requested)
        for args in [['native'],['native','--batch-groups','auto'],['native','--batch-groups','2']]:
            with mock.patch.object(s,'family',return_value=[{'exe':'native','cmd':args}]):
                with self.assertRaisesRegex(SafetyError,'automatic grouping'):s.verify_batch_groups(state,r.VERSION,2,1)

    def test_wrong_effective_runtime_refused(self):
        import strata_launcher as s
        response=io.BytesIO(json.dumps({'engine':'0.1.38','concurrency':{'serving':1}}).encode())
        with mock.patch.object(s.urllib.request,'urlopen',return_value=response):
            with self.assertRaises(SafetyError):s.verify_serving({'parallel':1},r.VERSION,18080)

    def test_promotion_requires_real_checks_on_both_models(self):
        with self.assertRaisesRegex(SafetyError,'Both profiles'):r.select(self.root,r.VERSION)
        self.assertFalse((self.root/'runtime-selection.json').exists())
        self.manifest['validation']={p:{'single_slot_passed':True,'source_commit':r.PIN} for p in r.PROFILES};self.save()
        r.select(self.root,r.VERSION);self.assertEqual(r.selected_version(self.root),r.VERSION)
        r.select(self.root,r.BASE_VERSION);self.assertEqual(r.selected_version(self.root),r.BASE_VERSION)

    def test_old_pin_validation_cannot_promote_new_runtime(self):
        self.manifest['validation']={p:{'single_slot_passed':True,'source_commit':r.PINS[r.LEGACY_VERSION]} for p in r.PROFILES};self.save()
        with self.assertRaisesRegex(SafetyError,'exact pin'):r.select(self.root,r.VERSION)
        self.assertFalse((self.root/'runtime-selection.json').exists())

    def test_unknown_default_selection_fails_closed(self):
        (self.root/'runtime-selection.json').write_text(json.dumps({'version':'untrusted'}))
        with self.assertRaises(SafetyError):r.selected_version(self.root)


if __name__=='__main__':unittest.main()
