import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
sys.path[:0]=['/home/user/LocalLLM','/home/user/LocalLLM/tests']
import strata_night_confirm as F
import strata_night_holdout as H
import unsloth_v31_v0159_tuned_profile as U
import v0159_tuned_strata_launcher as W
from engine_safety import SafetyError

class ConfirmationTests(unittest.TestCase):
    def test_holdout_frozen(self):
        self.assertEqual(len(H.cases()),24)
        self.assertEqual(H.Q.sha(H.cases()),'c82d2d539b5efbacdee2cd0e7ff7af5739d5f49ba5f43ef6303783e6f7dd06c5')
    def test_holdout_has_reasoning_multilingual_memory(self):
        names={x['id'].split('-')[0] for x in H.cases()}
        self.assertTrue({'high','multilingual','multiturn','data'}<=names)
        self.assertEqual(sum(x.get('high',False) for x in H.cases()),12)
    def test_final_gates_not_screen_tolerance(self):
        r={'half_prefill_speedup':1.2,'geomean_latency_speedup':1.2,'latency_speedups':{'a':1},'pair_speedup':1,'minimum_decode_ratio':.96}
        self.assertTrue(F.qualifies(r))
        for key,value in [('half_prefill_speedup',1.09),('geomean_latency_speedup',1.09),('pair_speedup',.94),('minimum_decode_ratio',.94)]:
            d=copy.deepcopy(r);d[key]=value;self.assertFalse(F.qualifies(d))
        r['latency_speedups']['a']=.94;self.assertFalse(F.qualifies(r))
    def test_permanent_grant_absent_refuses(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(U,'APPROVAL',Path(d)/'absent'):
            with self.assertRaises(SafetyError):U.authorize_activation()
    def test_permanent_recipe_absent_refuses(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(U,'RECIPE',Path(d)/'absent'):
            with self.assertRaises(SafetyError):U.recipe()
    def test_shutdown_state_boundary_does_not_require_recipe(self):
        with mock.patch.object(U,'recipe',side_effect=AssertionError('No recipe read during stop boundary')):
            U.require_state(U.state_root())
            with self.assertRaises(SafetyError):U.require_state(U.C.state_root())
    def test_permanent_distinct_identity(self):
        self.assertNotEqual(U.PROFILE,U.C.PROFILE);self.assertNotEqual(U.state_root(),U.C.state_root())
        self.assertNotEqual(U.APPROVAL,U.C.APPROVAL)
        self.assertIn('strata_launcher.py',Path(W.__file__).name)
    def test_tuned_shim_no_default_fallback(self):
        for args in [[],['--quickstart'],['--quickstart','--profile','other','--runtime','0.1.59'],['--status','--profile',U.PROFILE,'--runtime','0.1.46']]:
            with mock.patch.object(sys,'argv',['shim',*args]),mock.patch.object(W.L,'main') as main:
                with self.assertRaises(SafetyError):W.main()
                main.assert_not_called()
    def test_tuned_shim_duplicate_flags_refused(self):
        args=['--status','--profile',U.PROFILE,'--profile',U.PROFILE,'--runtime','0.1.59']
        with mock.patch.object(sys,'argv',['shim',*args]):
            with self.assertRaises(SafetyError):W.main()
    def test_tuned_shim_binds_only_this_process(self):
        original=W.L.v3159
        args=['--status','--profile',U.PROFILE,'--runtime','0.1.59']
        with mock.patch.object(sys,'argv',['shim',*args]),mock.patch.object(W.L,'main',return_value=0):
            try:self.assertEqual(W.main(),0);self.assertIs(W.L.v3159,U)
            finally:W.L.v3159=original
    def test_provider_frozen_features(self):
        recipe={'case':'balance13','spec':U.T.registry()['balance13']}
        with mock.patch.object(U,'recipe',return_value=recipe):
            p=U.provider();cfg=p.canonical_config(U.T.ROOT);base=U.C.canonical_config(U.T.ROOT,Path('/home/user'))
            self.assertEqual(cfg['vision'],base['vision']);self.assertEqual(cfg['sampling'],base['sampling'])
            a=cfg['args'][:];a[a.index('--expert-profile')+1]=base['args'][base['args'].index('--expert-profile')+1]
            self.assertEqual(a,base['args']);self.assertEqual(cfg['parallel'],2);self.assertEqual(cfg['drafting'],'mtp')
            self.assertEqual(cfg['layer_split'],'13,25,37');self.assertEqual(p.state_root(),U.state_root());self.assertEqual(p.PROFILE,U.PROFILE)
            self.assertEqual(p.MODEL_ID,U.MODEL_ID);self.assertEqual(p.TRIAL_ENV,U.TRIAL_ENV)
    def test_provider_no_temporary_permission(self):
        recipe={'case':'balanced','spec':U.T.registry()['balanced']}
        with mock.patch.object(U,'recipe',return_value=recipe):
            p=U.provider();self.assertEqual(p.APPROVAL,U.APPROVAL)
            self.assertEqual(p.authorize_activation,U.authorize_activation)
    def test_wrong_home_refused(self):
        with self.assertRaises(SafetyError):U.canonical_config(U.T.ROOT,Path('/root'))
    def test_provider_overrides_refused(self):
        recipe={'case':'balanced','spec':U.T.registry()['balanced']}
        with mock.patch.object(U,'recipe',return_value=recipe),mock.patch.object(U.T.TrialProfile,'require_state'):
            env={U.TRIAL_ENV:'1','STRATA_RUNTIME':'0.1.59','STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2','STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080',
                 'PYTHONDONTWRITEBYTECODE':'1','SETUPTOOLS_USE_DISTUTILS':'stdlib','PYTHONPYCACHEPREFIX':str(U.state_root()/'pycache'),'STRATA_HC_Q8':'1'}
            with mock.patch.dict(os.environ,env,clear=True):
                with self.assertRaises(SafetyError):U.require_environment(U.state_root())
    def test_inactive_tuned_bundle_preserves_token_owned_contract(self):
        import re
        bundle=U.REPO/'hive-llm-miner/profiles'/U.HIVE_PROFILE
        text=(bundle/'h-run.sh').read_text()
        self.assertEqual(set(x.name for x in bundle.iterdir()),{'h-run.sh','h-stats.sh','h-config.sh','h-manifest.conf','llm-hosting.conf'})
        self.assertEqual(re.search(r"^expected_readiness='(.*)'$",text,re.M).group(1),U.READY)
        self.assertIn('v0159_tuned_strata_launcher.py',text)
        self.assertIn(U.TRIAL_ENV+'=1',text)
        self.assertIn('HIVE_STRATA_LAUNCH_ID',text)
        self.assertIn('start_ticks',text);self.assertIn('owned_state_identity',text)
        self.assertIn('H.read_lease() is None',text);self.assertIn('A.hardware()',text)
        self.assertNotIn('pkill',text);self.assertNotIn('killall',text);self.assertNotIn('--gpu-reset',text)
    def test_aliases_keep_existing_point59_clients(self):
        recipe={'case':'balanced','spec':U.T.registry()['balanced']}
        with mock.patch.object(U,'recipe',return_value=recipe):
            cfg=U.canonical_config(U.T.ROOT)
            self.assertIn(U.C.MODEL_ID,cfg['aliases']);self.assertIn(U.MODEL_ID,cfg['aliases'])
    def test_unqualified_recipe_refused(self):
        d={'schema':1,'case':'balanced','spec':U.T.registry()['balanced'],'source_commit':U.R.PIN,'build':{},'dependencies':[],
           'qualification':{'path':'results/strata-v0159-overnight-qualification-20261114.json','sha256':'x'}}
        with mock.patch.object(U,'safe_data',side_effect=[d,{'phase':'NOT-QUALIFIED'}]),mock.patch.object(U.R,'digest',return_value='x'):
            with self.assertRaises(SafetyError):U.recipe()
if __name__=='__main__':unittest.main()
