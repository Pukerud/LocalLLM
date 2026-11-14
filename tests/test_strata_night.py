import ast,copy,hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
sys.path[:0]=['/home/user/LocalLLM','/home/user/LocalLLM/tests']
import strata_night_trial as T
import strata_night_client as Q
import strata_night_controller as K
from engine_safety import SafetyError

class NightTests(unittest.TestCase):
    def test_unknown_case_run_refused(self):
        for case,run in [('bad','x'),('official','../live'),('official','')]:
            with self.assertRaises(SafetyError):T.TrialProfile(case,run)
    def test_frozen_features_every_case(self):
        base=T.C.canonical_config(T.ROOT,Path('/home/user'))
        for name in T.registry():
            p=T.TrialProfile(name,'fixture-'+name);cfg=p.canonical_config()
            self.assertEqual(cfg['vision'],base['vision']);self.assertEqual(cfg['gpu'],base['gpu'])
            args=cfg['args'][:];args[args.index('--expert-profile')+1]=base['args'][base['args'].index('--expert-profile')+1]
            self.assertEqual(args,base['args']);self.assertEqual(cfg['parallel'],2)
            self.assertEqual(cfg['drafting'],'mtp');self.assertEqual(cfg['split_engine'],'off')
            self.assertEqual(cfg['sampling'],base['sampling']);self.assertNotEqual(p.state_root(),T.C.state_root())
    def test_config_drift_refused(self):
        p=T.TrialProfile('q5','fixture');c=p.canonical_config();c['parallel']=1
        with self.assertRaises(SafetyError):p.validate_config(T.ROOT,c)
    def test_cannot_read_other_state(self):
        p=T.TrialProfile('official','fixture')
        with self.assertRaises(SafetyError):p.require_state(T.C.state_root())
    def test_environment_override_refused(self):
        p=T.TrialProfile('official','fixture');env=K.environment(p);env['STRATA_GDN_TC']='1'
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(p,'require_state'):
            with self.assertRaises(SafetyError):p.require_environment(p.state_root())
    def test_missing_approval_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            p=T.TrialProfile('official','fixture',Path(d)/'missing')
            with self.assertRaises((SafetyError,OSError)):p.authorize_activation()
    def test_bad_stop_digest_refuses_before_L_main(self):
        p=T.TrialProfile('official','fixture')
        with mock.patch.object(T,'from_environment',return_value=p),mock.patch.object(T.L,'read_state',return_value={}),mock.patch.object(T.L.host_lifecycle,'read_lease',return_value=None),mock.patch.object(T.L,'main') as main,mock.patch.object(sys,'argv',['trial','--stop']):
            with self.assertRaises(SafetyError):T.main()
            main.assert_not_called()
    def test_production_module_not_mutated(self):
        original=T.L.v3159
        with mock.patch.object(T,'from_environment',return_value=T.TrialProfile('official','fixture')),mock.patch.object(T.L,'main',return_value=0),mock.patch.object(sys,'argv',['trial','--status']):
            try:self.assertEqual(T.main(),0);self.assertIsNot(T.L.v3159,original)
            finally:T.L.v3159=original
        self.assertEqual(T.C.PROFILE,original.PROFILE)
    def test_objective_answers(self):
        for c in Q.quality_cases():
            if c['type'] in ('number','exact','contains-number'):self.assertTrue(Q.grade(c,c['expected']))
    def test_json_strict(self):
        c={'type':'json','expected':[1,2]}
        self.assertTrue(Q.grade(c,'[1,2]'));self.assertFalse(Q.grade(c,'Here: [1,2]'))
    def test_code_sandbox_and_correctness(self):
        c=next(x for x in Q.quality_cases() if x['id']=='coding-0')
        self.assertTrue(Q.grade(c,'def solve(xs):\n return sum(x*x for x in xs if x%2==0)'))
        for text in ['import os\ndef solve(xs): return []','def solve(xs): return __import__("os")','def solve(xs): return ().__class__','def solve(xs): return 0']:
            self.assertFalse(Q.grade(c,text))
    def test_new_loss_detection(self):
        r={'cases':[{'id':'a','passed':True},{'id':'b','passed':False}]}
        self.assertEqual(K.no_new_losses({'cases':[{'id':'a','passed':False}]},r),['a'])
    def test_rank_requires_all_workloads(self):
        x={'medians':{k:{'wall_s':10,'prefill':100,'decode':50} for k in ('short','interactive','medium','half')},'pair':{'wall_s':10}}
        self.assertEqual(K.ranking(x,x)['geomean_latency_speedup'],1)
    def test_policy_binds_existing_gates(self):
        for name in ('strata_launcher.py','engine_safety.py','hive_profile_switch.py','hosting_lifecycle.py'):
            self.assertIn(name,T.policies())
    def test_no_global_cap_change(self):
        for case in T.registry():
            p=T.TrialProfile(case,'fixture')
            self.assertNotIn('max_tokens',p.canonical_config()['sampling'])
            self.assertNotIn('--n-predict',p.canonical_config()['args'])
if __name__=='__main__':unittest.main()
