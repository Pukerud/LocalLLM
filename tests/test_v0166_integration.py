"""CPU-only contracts for the separate .66 option; never publishes permission or calls live lifecycle."""
import copy,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as S,strata_launcher as L,engine_safety as E
import unsloth_v31_v0166_profile as P
import v0166_strata_launcher as V
from engine_safety import SafetyError

class Native66Integration(unittest.TestCase):
 def test_independent_prepared_profile_never_membership_only_authority(self):
  self.assertIn(P.PROFILE,S.PREPARED_PROFILES);self.assertNotIn(P.PROFILE,S.PROFILES)
  self.assertIs(S.prepared_profile_module(P.PROFILE),P)
  self.assertNotEqual(P.APPROVAL,S.v3159.APPROVAL);self.assertNotEqual(P.APPROVAL,S.v3159_tuned.APPROVAL)
 def test_locked_refuses_before_inspection_journal_or_mutation(self):
  with mock.patch.object(P,'authorize_activation',side_effect=SafetyError('66 LOCKED')),mock.patch.object(S,'inspect_owned_workload') as inspect,mock.patch.object(S,'run') as run,mock.patch.object(S,'write_journal') as journal:
   with self.assertRaisesRegex(SafetyError,'66 LOCKED'):S.switch_profile(P.PROFILE)
   inspect.assert_not_called();run.assert_not_called();journal.assert_not_called()
 def test_same_weights_features_except_explicit_context_rope_and_separate_routing_state(self):
  old=P.C.canonical_config(P.ROOT,Path('/home/user'));new=P.canonical_config();args=list(new['args'])
  args[args.index('--expert-profile')+1]=old['args'][old['args'].index('--expert-profile')+1]
  self.assertEqual(args[args.index('--max-context')+1],'524288');args[args.index('--max-context')+1]='262144'
  self.assertEqual(args[-6:],['--rope-scaling','yarn','--rope-scale','2','--yarn-orig-ctx','262144']);del args[-6:]
  self.assertEqual(args,old['args'])
  self.assertNotIn('max_tokens',new.get('sampling',{}));self.assertEqual(P.UI_MAX_TOKENS,65536)
  self.assertIn('yarn2-524k',P.STATE_NAME);self.assertNotEqual(P.STATE_NAME,P.OLD_NATIVE_STATE_NAME)
  allowed={'args','exe','cwd','lib_dirs','layer_split','model_name','aliases','env','undeclared_tool_calls','vision'}
  self.assertEqual({k:v for k,v in old.items() if k not in allowed},{k:v for k,v in new.items() if k not in allowed})
  vision=dict(new['vision']);vision['exe']=old['vision']['exe'];self.assertEqual(vision,old['vision'])
  self.assertEqual(new['layer_split'],'12,24,36');self.assertEqual(new['undeclared_tool_calls'],'call')
  self.assertNotIn('api_key',new)
 def test_new_native_tuning_is_compatible_not_fake_mmq_reuse_or_expert_skipping(self):
  env=P.canonical_config()['env']
  for k,v in {'STRATA_PF_GROUP_SCRATCH':'0','STRATA_PF_HC_REGION':'1','STRATA_PW_WITH_BATCH':'1','STRATA_EXP_MODE':'8','STRATA_ROUTE_TAIL_SKIP':'0','STRATA_HC_Q8':'0','STRATA_GDN_TC':'0','STRATA_GDN_CONVL2':'0','STRATA_SPLIT_CPU_SHARE':'0','STRATA_QFUSE':'0','STRATA_DEQUANT_SIDE':'0'}.items():self.assertEqual(env[k],v)
 def test_config_mutation_refuses(self):
  for key,value in (('parallel',1),('layer_split','auto'),('undeclared_tool_calls','text'),('drafting','dflash2')):
   cfg=P.canonical_config();cfg[key]=value
   with self.subTest(key=key),self.assertRaises(SafetyError):P.validate_config(P.ROOT,cfg)
 def test_runtime_and_source_are_separate_exact_release(self):
  self.assertEqual((P.VERSION,P.R.TAG,P.R.PIN),('0.1.66','v0.2.20','a34979fc94f3d126b9bb8210f4ad311345688b52'))
  self.assertNotIn('/0.1.59/',P.canonical_config()['exe']);self.assertEqual(P.R.OFFICIAL_FILES['strata'][0],395465704)
 def test_explicit_entrypoint_rejects_wrong_or_duplicate_profile_before_core(self):
  for args in ([],['--profile',P.C.PROFILE,'--runtime',P.VERSION],['--profile',P.PROFILE,'--runtime','0.1.59'],['--profile',P.PROFILE,'--profile',P.PROFILE,'--runtime',P.VERSION]):
   with self.subTest(args=args),mock.patch.object(sys,'argv',['entry',*args]),mock.patch.object(V.L,'main') as core:
    with self.assertRaises(SafetyError):V.main()
    core.assert_not_called()
 def test_foreign_state_refuses(self):
  for p in (P.C.state_root(),S.v3159_tuned.state_root(),P.state_root().parent/P.OLD_NATIVE_STATE_NAME,Path('/tmp/unowned')):
   with self.subTest(state=p),self.assertRaises(SafetyError):P.require_state(p)
 def test_offline_ready_does_not_authorize_start(self):
  result=types.SimpleNamespace(returncode=0,stdout=P.READY+'\n',stderr='')
  with mock.patch.object(P,'authorize_activation') as auth,mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=result) as run:
   S.preflight_profile(P.PROFILE,offline=True);auth.assert_not_called();cmd=run.call_args.args[0]
   self.assertIn(str(S.SCRIPT_ROOT/'v0166_strata_launcher.py'),cmd);self.assertIn('--check-ready',cmd);self.assertNotIn('--quickstart',cmd)
 def test_live_ready_checks_own_permission_first(self):
  with mock.patch.object(P,'authorize_activation',side_effect=SafetyError('LOCKED')),mock.patch.object(S.subprocess,'run') as run:
   with self.assertRaises(SafetyError):S.preflight_profile(P.PROFILE)
   run.assert_not_called()
 def test_no_other_permission_borrowed(self):
  with mock.patch.object(S.v3159,'authorize_activation',side_effect=SafetyError('OLD LOCKED')),mock.patch.object(P,'authorize_activation') as other,mock.patch.object(S,'inspect_owned_workload') as inspect:
   with self.assertRaises(SafetyError):S.switch_profile(S.v3159.PROFILE)
   other.assert_not_called();inspect.assert_not_called()
 def test_state_detection_without_inherited_environment(self):
  with tempfile.TemporaryDirectory() as t:
   home=Path(t);p=home/'.local/state/locallm-strata'/P.STATE_NAME;p.mkdir(parents=True)
   i={'pid':345678,'start_ticks':1,'exe':'/usr/bin/python3.10','uid':1000,'cmd':['python','-m','serve.server'],'pgid':345678,'sid':345678}
   (p/'server.json').write_text(json.dumps({'identity':i,'profile':P.PROFILE}))
   with mock.patch.object(E,'owner_home',return_value=home),mock.patch.object(E,'strata_state',return_value=home/'unused'),mock.patch.object(E,'proc',return_value=i):self.assertEqual(E.active_engine(),'strata')
 def test_menu_ids_and_lock_before_selector_mutation(self):
  root=S.SCRIPT_ROOT;low=(root/'hive-llm-miner/profile-menu.sh').read_text();high=(root/'HostLLM.sh').read_text()
  self.assertIn('12|'+P.PROFILE,low);self.assertIn('24) set_next_hive_profile '+P.PROFILE,high)
  self.assertLess(low.index('if [[ "$target" == '+P.PROFILE+' ]]'),low.index('umask 077; exec 9>'))
  self.assertIn('--stop-owned-v3166',high)
 def test_family_extension_keeps_exact_identity_gates(self):
  text=(S.SCRIPT_ROOT/'strata_launcher.py').read_text()
  self.assertIn("state.get('profile') == '"+P.PROFILE+"'",text)
  for x in ("p['uid'] != leader['uid']","p.get('ppid') != leader['pid']","p['exe'] not in allowed","not same_process(known, p)"):self.assertIn(x,text)
 def test_supported_no_key_opt_out_is_version_scoped(self):
  text=(S.SCRIPT_ROOT/'strata_launcher.py').read_text()
  self.assertIn("profile == v3159.PROFILE and version == '0.1.66'",text);self.assertIn("server_args.append('--no-api-key')",text)
 def test_actual_wrapper_readiness_literal_matches_provider(self):
  text=(S.SCRIPT_ROOT/'hive-llm-miner/profiles'/P.PROFILE/'h-run.sh').read_text()
  import re
  expected=re.search(r"^expected_readiness='([^']+)'$",text,re.M).group(1)
  self.assertEqual(expected,P.READY)
 def test_matching_permission_still_cannot_restart_on_faulted_boot(self):
  value=P.approval_contract()
  with mock.patch.object(P,'safe_data',return_value=value),mock.patch.object(P,'boot_digest',return_value=P.FAULT_BOOT_SHA256),mock.patch.object(P.R,'ready') as ready:
   with self.assertRaisesRegex(SafetyError,'fresh healthy reboot'):P.authorize_activation()
   ready.assert_not_called()
 def test_new_boot_requires_the_same_exact_permission_and_runtime(self):
  value=P.approval_contract()
  with mock.patch.object(P,'safe_data',return_value=value),mock.patch.object(P,'boot_digest',return_value='another-boot'),mock.patch.object(P.R,'ready') as ready:
   self.assertEqual(P.authorize_activation(),'trial');ready.assert_called_once_with(P.ROOT)
  bad=copy.deepcopy(value);bad['configuration_sha256']='tampered'
  with mock.patch.object(P,'safe_data',return_value=bad),mock.patch.object(P,'boot_digest',return_value='another-boot'),mock.patch.object(P.R,'ready') as ready:
   with self.assertRaises(SafetyError):P.authorize_activation()
   ready.assert_not_called()
 def test_new_wrapper_keeps_power_independent_hardware_and_no_renter_override(self):
  text=(S.SCRIPT_ROOT/'hive-llm-miner/profiles'/P.PROFILE/'h-run.sh').read_text()
  for x in ('A.hardware()','C.authorize_activation()','H.read_lease()','owned_state_identity','HIVE_STRATA_LAUNCH_ID','runuser -u'):self.assertIn(x,text)
  self.assertNotIn('systemctl stop osn',text);self.assertNotIn('power.limit',text)

if __name__=='__main__':unittest.main()
