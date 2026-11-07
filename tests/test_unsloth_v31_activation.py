import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unsloth_v31_profile as F
import unsloth_v31_assets as A
import stage_unsloth_v31 as B
import strata_launcher as L
import hive_profile_switch as S
from engine_safety import SafetyError
ROOT=Path(__file__).resolve().parents[1]

class Activation(unittest.TestCase):
    def env(self):
        state=F.state_root()
        return {F.TRIAL_ENV:'1','STRATA_RUNTIME':F.VERSION,'STRATA_PARALLEL':'2',
                'STRATA_BATCH_GROUPS':'2','STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080',
                'PYTHONDONTWRITEBYTECODE':'1','SETUPTOOLS_USE_DISTUTILS':'stdlib',
                'PYTHONPYCACHEPREFIX':str(state/'pycache')}

    def test_exact_environment_and_no_other_profile_tuning(self):
        with patch.object(F,'require_state'),patch.dict(os.environ,self.env(),clear=True):
            F.require_environment(F.state_root())
            for k,v in (('STRATA_RUNTIME','0.1.45'),('STRATA_PARALLEL','1'),('STRATA_BATCH_GROUPS','1'),('STRATA_HOST','127.0.0.1'),(F.TRIAL_ENV,'0')):
                with self.subTest(k=k),patch.dict(os.environ,{k:v}),self.assertRaises(SafetyError):
                    F.require_environment(F.state_root())
            with patch.dict(os.environ,{'STRATA_ROPE_SCALE':'2'}),self.assertRaises(SafetyError):
                F.require_environment(F.state_root())

    def test_explicit_46_two_by_two_inherent_q8_only_before_readiness(self):
        for runtime,p,g,ple in (('auto',2,2,False),('0.1.45',2,2,False),('0.1.46',1,1,False),('0.1.46',2,1,False),('0.1.46',2,2,True)):
            with patch.object(F,'ready') as ready,self.subTest(runtime=runtime,p=p,g=g,ple=ple),self.assertRaises(SafetyError):
                L.ready_runtime(F.PROFILE,runtime,p,g,ple)
            ready.assert_not_called()
        with patch.object(F,'ready',return_value='exact') as ready:
            self.assertEqual(L.ready_runtime(F.PROFILE,'0.1.46',2,2),'exact')
            ready.assert_called_once()

    def test_ready_verifies_immutable_inputs_not_overwriting_live_adaptation(self):
        with patch.object(F,'require_environment'),patch.object(B,'check') as check:
            source,path,cfg,version=F.ready(Path('/data'),F.state_root())
            check.assert_called_once_with(Path('/data'),pristine_state=False)
            self.assertEqual(cfg,F.canonical_config(Path('/data')))
            self.assertEqual(version,'0.1.46')

    def test_start_root_and_wrong_binding_refused(self):
        cfg=F.canonical_config('/data')
        for uid,host,port in ((0,'0.0.0.0',8080),(1000,'127.0.0.1',8080),(1000,'0.0.0.0',18080)):
            with patch.object(F.os,'geteuid',return_value=uid),patch.object(F,'require_environment') as env,self.assertRaises(SafetyError):
                F.authorize_start('/data',cfg,host=host,port=port,state=F.state_root())
            env.assert_not_called()
        with patch.object(F.os,'geteuid',return_value=1000),patch.object(F,'require_environment'):
            self.assertEqual(F.authorize_start('/data',cfg,host='0.0.0.0',port=8080,state=F.state_root()),'trial')

    def test_source_start_has_full_hash_stable_identity_and_projection_enabled(self):
        text=(ROOT/'strata_launcher.py').read_text()
        self.assertIn('stage_unsloth_v31.check(data_root(), full=True, pristine_state=False)',text)
        self.assertIn("cfg['sampling']['experimental_speed_projection'] = True",text)
        self.assertIn('profile in (sc117_two.PROFILE, v31.PROFILE) else proc(child.pid)',text)
        self.assertIn("state.update(unsloth_v31=True, ple_q8=True, ple_sha256=v31.A.ASSETS[-1][4])",text)
        self.assertIn(F.HIVE_PROFILE,S.PROFILES)

    def test_candidate_preflight_has_exact_identity_environment_no_lifecycle(self):
        response=Mock(returncode=0,stdout='Runtime ready\n'+F.READY+'\n')
        with patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),patch.object(S.subprocess,'run',return_value=response) as run:
            S.preflight_profile(F.HIVE_PROFILE)
        args=run.call_args.args[0]
        self.assertEqual(args[0],'/usr/sbin/runuser')
        for flag in (F.TRIAL_ENV+'=1','STRATA_PARALLEL=2','STRATA_BATCH_GROUPS=2','STRATA_PORT=8080',F.PROFILE):self.assertIn(flag,args)
        self.assertNotIn('stop',args)
        response.stdout=F.READY.replace('262144','524288')+'\n'
        with patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),patch.object(S.subprocess,'run',return_value=response),self.assertRaises(SafetyError):
            S.preflight_profile(F.HIVE_PROFILE)

    def test_hive_wrapper_token_markers_hash_state_and_renter_gates_retained(self):
        base=ROOT/'hive-llm-miner/profiles'/F.HIVE_PROFILE
        self.assertEqual(set(x.name for x in base.iterdir()),{'h-run.sh','h-config.sh','h-stats.sh','h-manifest.conf','llm-hosting.conf'})
        text=(base/'h-run.sh').read_text()
        for s in ('owned_state_identity','carries_token','START_STATE_IDENTITY','STOPPER_ARMED','docker_workload_info','runuser','unsloth_v31',A.ASSETS[-1][4],F.STATE_NAME,F.READY):self.assertIn(s,text)
        self.assertNotIn('sc117_q8_two_slot',text)
        self.assertNotIn('STRATA_SC117_IQ3S_TRIAL',text)
        self.assertNotIn('systemctl stop',text)
        self.assertEqual(subprocess.run(['bash','-n',str(base/'h-run.sh')]).returncode,0)

    def test_wrong_state_variant_hash_or_count_never_signalled(self):
        for changes in ({'unsloth_v31':False},{'ple_sha256':'wrong'},{'parallel':1},{'batch_groups':1}):
            with tempfile.TemporaryDirectory() as d:
                state={'profile':F.PROFILE,'runtime_version':'0.1.46','parallel':2,'batch_groups':2,'ple_q8':True,'unsloth_v31':True,'ple_sha256':A.ASSETS[-1][4]}
                state.update(changes)
                argv=['launcher','--stop','--profile',F.PROFILE]
                with patch.object(sys,'argv',argv),patch.object(F,'require_state'),patch.object(L,'strata_state',return_value=Path(d)),patch.object(L,'acquire_action_lock',return_value=state),patch.object(L,'family',return_value=[{}]),patch.object(L,'stop') as stop:
                    self.assertEqual(L.main(),1)
                stop.assert_not_called()

    def test_owned_menu_stop_refuses_manual_work_before_any_action(self):
        with patch.object(S,'JOURNAL',Path('/nonexistent-v31-test-journal')):
            with patch.object(S.os,'readlink',return_value='profiles/'+F.HIVE_PROFILE),patch.object(S,'inspect_owned_workload',side_effect=SafetyError('manual/unowned')),patch.object(S,'run') as run:
                with self.assertRaisesRegex(SafetyError,'manual/unowned'):
                    S.stop_owned_v31()
                run.assert_not_called()

    def test_owned_menu_stop_uses_normal_miner_stop_and_waits_without_service_stop(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(S,'JOURNAL',Path(d)/'journal'),patch.object(S.os,'readlink',return_value='profiles/'+F.HIVE_PROFILE),patch.object(S,'inspect_owned_workload',return_value={1:{}}),patch.object(S,'capture_owned_teardown',return_value=['exact']) as capture,patch.object(S,'docker_empty'),patch.object(S,'run') as run,patch.object(S.hosting,'STOP',Path(d)/'stop'),patch.object(S,'wait_owned_teardown') as wait,patch.object(S.hosting,'wait_idle'),patch.object(S,'owned_wrappers',return_value={}),patch.object(S,'llm_processes',return_value=[]):
                S.stop_owned_v31()
                run.assert_called_once_with([S.hosting.MINER,'stop','60'],timeout=90)
                wait.assert_called_once_with(['exact'],timeout=120)
                self.assertEqual((Path(d)/'stop').read_text(),'1\n')

    def test_hive_menu_new_number_and_no_retired_number_reuse(self):
        text=(ROOT/'HostLLM.sh').read_text()
        self.assertIn('[21] Set next Hive miner - Unsloth V3.1 IQ4_XS',text)
        self.assertIn('21) set_next_hive_profile '+F.HIVE_PROFILE,text)
        for n in (2,3,4,5,6,8,13,14,15,16,17,19):self.assertNotRegex(text,rf'echo .*\[{n}\]')
        low=(ROOT/'hive-llm-miner/profile-menu.sh').read_text()
        self.assertIn("printf '  [9] %s",low)
        self.assertIn(F.HIVE_PROFILE,low)
        self.assertIn('assert_idle',low)

if __name__=='__main__':unittest.main()
