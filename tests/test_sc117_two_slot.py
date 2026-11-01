"""Bounded two-slot profile, unchanged legacy gates and no-stop menu publication contracts."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import engine_safety as E
import hive_profile_switch as S
import install_ple_q8_profiles as I
import ple_q8_assets as A
import ple_q8_profile as Q
import sc117_q8_2slot_profile as T
import strata_launcher as L

REPO=Path(__file__).resolve().parents[1]

def base():
    return {'args':['--native','/original/model','--ple-gguf','/shared/q8','--ple-io','mmap',
      '--expert-profile','/original/state/expert-profile.bin','--max-context','262144','--kv','int8','--kv-resident','32768',
      '--spec','3','--pipeline-windows','2','--conversation-cache-mib','8192','--conversation-cache-slots','4',
      '--native-dense-gguf','/original/shard1','--native-dense-gguf','/original/shard2'],
      'env':{'STRATA_PARALLEL':'1','STRATA_BATCH_GROUPS':'1'},'vision':{'gpu':False,'exe':'/runtime/cpu-vision'},
      'gpu':[0,1,2,3],'gpu_order':'as_given','layer_split':'14,24,36','exe':'/runtime/strata','cwd':'/runtime',
      'tokenizer':'/original/tokenizer','sampling':{'temperature':1,'top_p':.95,'top_k':20},'reasoning_effort':'high',
      'model_name':'original-ple-q8_0','aliases':['original']}

class Profile(unittest.TestCase):
    def test_overlay_only_changes_measured_parameters_and_isolates_state(self):
        old=base();before=copy.deepcopy(old);cfg=T.build_config(old)
        self.assertEqual(old,before)
        args=list(before['args'])
        for flag,value in [('--conversation-cache-mib','16384'),('--conversation-cache-slots','8'),('--expert-profile',str(T.state_root()/'expert-profile.bin'))]:
            args[args.index(flag)+1]=value
        self.assertEqual(cfg['args'],args+['--batch-groups','2'])
        self.assertEqual(cfg['parallel'],2)
        self.assertEqual(cfg['env'],{'STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2'})
        for key in ('vision','gpu','gpu_order','layer_split','exe','cwd','tokenizer','sampling','reasoning_effort'):
            self.assertEqual(cfg[key],before[key])
        self.assertTrue(cfg['model_name'].endswith('-ple-q8_0-2slot'))
        self.assertNotIn('--batch-mtp',cfg['args'])
        self.assertNotEqual(T.state_root(),Q.state_root(T.SC.PROFILE))

    def test_overlay_rejects_unreviewed_batching_or_cache_baseline(self):
        for flag in ['--batch','--batch=2','--slots','--slots=2','--batch-groups','--batch-groups=2','--batch-mtp','--batch-mtp=1']:
            cfg=base();cfg['args'].append(flag)
            with self.subTest(flag=flag),self.assertRaises(E.SafetyError):T.build_config(cfg)
        cfg=base();cfg['args'][cfg['args'].index('--conversation-cache-slots')+1]='8'
        with self.assertRaises(E.SafetyError):T.build_config(cfg)

    def test_new_profile_never_widens_legacy_one_slot_or_045_gates(self):
        for runtime,slots,groups,q8 in [('0.1.45',2,2,False),('auto',2,2,False),('0.1.46',1,1,False),('0.1.46',2,1,False),('0.1.46',2,2,True)]:
            with self.subTest(runtime=runtime,slots=slots,groups=groups,q8=q8),mock.patch.object(T,'ready') as ready,self.assertRaises(E.SafetyError):
                L.ready_runtime(T.PROFILE,runtime,slots,groups,ple_q8=q8)
                ready.assert_not_called()
        with self.assertRaises(E.SafetyError):L._ready_runtime(T.SC.PROFILE,'0.1.46',2,2)

    def test_environment_and_state_are_exact_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(T,'owner_home',return_value=Path(tmp)):
            state=T.state_root();env={T.SC.TRIAL_ENV:'1','STRATA_RUNTIME':'0.1.46','STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2',
                'STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080','PYTHONDONTWRITEBYTECODE':'1','SETUPTOOLS_USE_DISTUTILS':'stdlib',
                'PYTHONPYCACHEPREFIX':str(state/'pycache')}
            with mock.patch.dict(os.environ,env,clear=True):T.require_environment(state)
            for key,value in [('STRATA_RUNTIME','0.1.45'),('STRATA_PARALLEL','1'),('STRATA_BATCH_GROUPS','1'),('STRATA_HOST','127.0.0.1'),('STRATA_PORT','18080'),(T.SC.TRIAL_ENV,'0'),('PYTHONPYCACHEPREFIX','/other'),('STRATA_SC117_IQ3S_TRIAL_SPEC','4')]:
                with self.subTest(key=key),mock.patch.dict(os.environ,dict(env,**{key:value}),clear=True),self.assertRaises(E.SafetyError):T.require_environment(state)
            with self.assertRaises(E.SafetyError):T.require_state(state.parent/'sc117-iq3s-ple-q8-trial')
            state.parent.mkdir(parents=True);state.symlink_to(Path(tmp)/'elsewhere',target_is_directory=True)
            with self.assertRaises(E.SafetyError):T.require_state(state)

    def test_authorization_checks_full_assets_without_qualification_or_root(self):
        cfg=T.build_config(base())
        with mock.patch.object(T.os,'geteuid',return_value=1000),mock.patch.object(T,'validate_config') as validate,\
             mock.patch.object(T.SC,'_verify_v01441_trial_artifacts') as verify:
            self.assertEqual(T.authorize_start('/data',cfg,host='0.0.0.0',port=8080,state=T.state_root()),'trial')
            validate.assert_called_once();verify.assert_called_once_with('/data',T.state_root())
        for host,port,uid in [('127.0.0.1',8080,1000),('0.0.0.0',18080,1000),('0.0.0.0',8080,0)]:
            with mock.patch.object(T.os,'geteuid',return_value=uid),self.assertRaises(E.SafetyError):
                T.authorize_start('/data',cfg,host=host,port=port,state=T.state_root())

    def test_configuration_drift_fails_before_authorization(self):
        expected=T.build_config(base())
        with mock.patch.object(T,'ready',return_value=(Path('/runtime'),Path('/config'),expected,T.VERSION)):
            T.validate_config('/data',T.state_root(),copy.deepcopy(expected))
            for flag,value in [('--max-context','131072'),('--kv','f16'),('--spec','2'),('--conversation-cache-slots','4'),('--batch-groups','1')]:
                changed=copy.deepcopy(expected);changed['args'][changed['args'].index(flag)+1]=value
                with self.subTest(flag=flag),self.assertRaises(E.SafetyError):T.validate_config('/data',T.state_root(),changed)

    def test_full_q8_hash_still_precedes_any_gpu_spawn(self):
        with mock.patch.object(L.os,'geteuid',return_value=1000),mock.patch.object(L,'consume_api_key',return_value=None),\
             mock.patch.object(L,'ready_runtime',return_value=(Path('/runtime'),Path('/config'),base(),'0.1.46')),\
             mock.patch.object(Q,'ready_asset',side_effect=E.SafetyError('tamper')) as verify,mock.patch.object(L.subprocess,'Popen') as spawn:
            with self.assertRaises(E.SafetyError):L.start(mock.Mock(),T.PROFILE,'0.1.46',2,2)
            verify.assert_called_once_with(L.data_root(),full=True);spawn.assert_not_called()

    def test_wrong_profile_cannot_stop_the_two_slot_state(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(L,'strata_state',return_value=Path(tmp)),mock.patch.object(T,'require_state'),\
             mock.patch.object(L,'acquire_action_lock',return_value={'profile':T.SC.PROFILE,'ple_q8':True}),mock.patch.object(L,'family',return_value=[object()]),\
             mock.patch.object(L,'stop') as stop,mock.patch('sys.argv',['launcher','--stop','--profile',T.PROFILE]):
            self.assertEqual(L.main(),1);stop.assert_not_called()

    def test_switch_preflight_matches_exact_new_contract_before_mutations(self):
        self.assertIn(T.HIVE_PROFILE,S.PROFILES);self.assertIn(T.HIVE_PROFILE,I.PROFILES)
        for text in [T.READY+'\n',T.READY+' drift\n']:
            result=type('Result',(),{'returncode':0,'stdout':text})()
            with mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=result) as call,mock.patch.object(S,'run') as mutate:
                if text==T.READY+'\n':S.preflight_profile(T.HIVE_PROFILE)
                else:
                    with self.assertRaises(E.SafetyError):S.preflight_profile(T.HIVE_PROFILE)
                args=call.call_args.args[0];self.assertIn(T.PROFILE,args);self.assertIn('STRATA_PARALLEL=2',args)
                self.assertIn('STRATA_RUNTIME=0.1.46',args);self.assertNotIn('--ple-q8',args);mutate.assert_not_called()

    def test_all_readiness_consumers_and_wrapper_ownership_match(self):
        consumers=['HostLLM.sh','hive-llm-miner/profile-menu.sh','hive-llm-miner/profiles/'+T.HIVE_PROFILE+'/h-run.sh']
        for path in consumers:self.assertIn(T.READY,(REPO/path).read_text())
        wrapper=(REPO/consumers[-1]).read_text()
        for needle in ["state.get('sc117_q8_two_slot') is not True","state.get('parallel')!=2","state.get('batch_groups')!=2",'hive_launch_id_sha256','carries_token','START_STATE_IDENTITY','--parallel 2 --batch-groups 2']:
            self.assertIn(needle,wrapper)
        self.assertNotIn('--quickstart --ple-q8',wrapper)

    def test_actual_menu_dispatch_7_and_18_preserves_other_entries(self):
        for choice,expected in [('7','--profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 2'),('18',T.HIVE_PROFILE)]:
            command='''source "$1/HostLLM.sh"
detect_engine() { echo none; }; show_workload_status() { :; }; host_gpu_summary() { :; }
run_selected() { printf 'DISPATCH %s\\n' "$*"; }; set_next_hive_profile() { printf 'DISPATCH %s\\n' "$*"; }
main < <(printf '%s\\n11\\n' "$2")
'''
            p=subprocess.run(['/bin/bash','-c',command,'test',str(REPO),choice],text=True,capture_output=True,timeout=5)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn(expected,p.stdout)
            self.assertIn('[17] Set next Hive miner - SC117 IQ3_S | Q8_0',p.stdout)

if __name__=='__main__':unittest.main()
