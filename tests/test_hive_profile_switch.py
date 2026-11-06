"""CPU-only retained-profile switch transactions; rental admission never gets stopped."""
from contextlib import ExitStack
import os,subprocess,sys,tempfile,unittest,shlex
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as S
from engine_safety import SafetyError
TARGET=S.sc117_two.HIVE_PROFILE

class SwitchTransaction(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.current=self.root/'current';self.current.symlink_to('profiles/swift15u-q8');self.stop=self.root/'STOP';self.journal=self.root/'journal'
  self.service='active';self.miner=True;self.events=[]
  self.wrappers={10:{'pid':10,'start_ticks':11,'exe':'/bin/bash','uid':0,'cmd':['bash','/test/h-run.sh'],'pgid':10,'sid':10}}
  self.stack=ExitStack();self.addCleanup(self.stack.close)
  def p(obj,name,**kw):return self.stack.enter_context(mock.patch.object(obj,name,**kw))
  p(S,'CURRENT',new=self.current);p(S,'JOURNAL',new=self.journal);p(S,'SCRIPT_ROOT',new=self.root)
  p(S.hosting,'ROOT',new=self.root);p(S.hosting,'STOP',new=self.stop);p(S.hosting,'service_state',side_effect=lambda:self.service)
  p(S.hosting,'wait_idle');p(S,'docker_empty');p(S,'llm_processes',return_value=[]);p(S.hosting,'gpu_jobs',return_value=[])
  self.inspected=p(S,'inspect_owned_workload',side_effect=lambda:self.wrappers if self.miner else {})
  p(S,'owned_wrappers',side_effect=lambda:self.wrappers if self.miner else {});p(S,'wait_owned_teardown');p(S,'package_current',return_value=True)
  self.ready=p(S,'preflight_profile');self.execute=p(S,'run',side_effect=self.command)
 def command(self,args,timeout=180):
  self.events.append(tuple(args))
  self.assertNotEqual(args,['systemctl','stop',S.hosting.SERVICE],'rental admission must never be stopped')
  if args==[S.hosting.MINER,'stop','60']:self.miner=False
  elif '--apply-hive-profile' in args:
   self.assertFalse(self.miner);self.assertTrue(self.stop.exists());self.current.unlink();self.current.symlink_to('profiles/'+args[-1])
  elif args==[S.hosting.MINER,'start']:self.miner=True
  elif args==['systemctl','start',S.hosting.SERVICE]:self.service='active'
  else:self.fail('Unexpected lifecycle command')
 def test_running_swift_switches_to_retained_two_slot_without_stopping_admission(self):
  S.switch_profile(TARGET)
  self.assertEqual(os.readlink(self.current),'profiles/'+TARGET);self.assertTrue(self.miner);self.assertEqual(self.service,'active')
  self.assertEqual([x[1] for x in self.events],['stop','--apply-hive-profile','start']);self.assertFalse(self.journal.exists());self.assertFalse(self.stop.exists())
 def test_exact_readiness_is_before_any_mutation(self):
  def ready(profile):
   self.assertEqual(profile,TARGET);self.assertEqual(self.service,'active');self.assertTrue(self.miner);self.assertFalse(self.stop.exists());self.assertEqual(self.events,[])
  self.ready.side_effect=ready;S.switch_profile(TARGET);self.ready.assert_called_once_with(TARGET)
 def test_missing_runuser_refuses_before_mutation(self):
  self.ready.side_effect=SafetyError('runuser is unavailable')
  with self.assertRaisesRegex(SafetyError,'runuser'):S.switch_profile(TARGET)
  self.execute.assert_not_called();self.assertTrue(self.miner);self.assertFalse(self.journal.exists())
 def test_failed_readiness_refuses_before_mutation(self):
  self.ready.side_effect=SafetyError('readiness failed before any pause')
  with self.assertRaisesRegex(SafetyError,'before any pause'):S.switch_profile(TARGET)
  self.execute.assert_not_called();self.assertFalse(self.stop.exists());self.assertEqual(self.service,'active')
 def test_rental_arriving_before_owned_stop_blocks_and_keeps_admission_active(self):
  self.inspected.side_effect=[self.wrappers,SafetyError('new rental'),SafetyError('new rental')]
  with self.assertRaisesRegex(SafetyError,'new rental'):S.switch_profile(TARGET)
  self.assertEqual(self.service,'active');self.assertTrue(self.miner);self.assertEqual(self.events,[]);self.assertTrue(self.journal.exists())
 def test_stop_race_accepted_only_after_verified_idle(self):
  original=self.command
  def race(args,timeout=180):
   if args==[S.hosting.MINER,'stop','60']:self.miner=False;raise SafetyError('No miner screens found')
   original(args,timeout)
  self.execute.side_effect=race;S.switch_profile(TARGET);self.assertTrue(self.miner);self.assertFalse(self.journal.exists());self.assertEqual(self.service,'active')
 def test_same_current_profile_needs_no_stop(self):
  S.switch_profile('swift15u-q8');self.execute.assert_not_called();self.assertFalse(self.journal.exists())
 def test_already_idle_miner_not_started(self):
  self.miner=False;S.switch_profile(TARGET);self.assertFalse(self.miner);self.assertNotIn((S.hosting.MINER,'start'),self.events);self.assertEqual(self.service,'active')
 def test_inactive_service_and_operator_stop_preserved(self):
  self.service='inactive';self.miner=False;self.stop.write_text('operator-stop\n');S.switch_profile(TARGET)
  self.assertEqual(self.service,'inactive');self.assertFalse(self.miner);self.assertEqual(self.stop.read_text(),'operator-stop\n')
 def test_second_readiness_failure_restores_old_owned_miner_not_cloud_settings(self):
  original=self.command
  def fail(args,timeout=180):
   if '--apply-hive-profile' in args:self.assertTrue(self.stop.exists());raise SafetyError('late readiness failed')
   original(args,timeout)
  self.execute.side_effect=fail
  with self.assertRaisesRegex(SafetyError,'late readiness'):S.switch_profile(TARGET)
  self.assertEqual(os.readlink(self.current),'profiles/swift15u-q8');self.assertTrue(self.miner);self.assertEqual(self.service,'active');self.assertFalse(self.journal.exists())
 def test_unowned_workload_retains_journal_without_restart(self):
  self.inspected.side_effect=[self.wrappers,self.wrappers,SafetyError('new rental')]
  with self.assertRaisesRegex(SafetyError,'new rental'):S.switch_profile(TARGET)
  self.assertEqual(self.service,'active');self.assertFalse(self.miner);self.assertTrue(self.journal.exists());self.assertEqual(self.journal.stat().st_mode&0o777,0o600)
  self.assertNotIn((S.hosting.MINER,'start'),self.events)
 def test_recovery_keeps_exact_state_root(self):
  command=S.recovery_command();self.assertIn('HOSTLLM_STATE_ROOT='+str(self.root),command);self.assertTrue(command.endswith('--recover'))
 def test_existing_journal_not_overwritten(self):
  self.journal.write_text('existing')
  with self.assertRaisesRegex(SafetyError,'needs recovery'):S.switch_profile(TARGET)
  self.execute.assert_not_called();self.assertEqual(self.journal.read_text(),'existing')
 def test_retired_target_cannot_reach_lifecycle(self):
  for target in ('hauhau-q8-fastmtp-q4kv-xhigh','swift15-heretic-re','iq3_s-sc117-abliterated-v0146'):
   with self.assertRaisesRegex(SafetyError,'retired'):S.switch_profile(target)
  self.execute.assert_not_called();self.inspected.assert_not_called()

class PreflightChecks(unittest.TestCase):
 def ready(self):return subprocess.CompletedProcess([],0,stdout=S.sc117_two.READY+'\n',stderr='')
 def test_runuser_resolved_from_trusted_paths_and_clean_two_slot_environment(self):
  with mock.patch.dict(os.environ,{'PATH':'/tmp/untrusted'}),mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser') as lookup,mock.patch.object(S.subprocess,'run',return_value=self.ready()) as launch:
   S.preflight_profile(TARGET)
  lookup.assert_called_once_with('runuser',path='/usr/sbin:/usr/bin:/sbin:/bin');a=launch.call_args.args[0]
  self.assertEqual(a[:6],['/usr/sbin/runuser','-u','user','--','/usr/bin/env','-i'])
  for value in ('STRATA_RUNTIME=0.1.46','STRATA_HOST=0.0.0.0','STRATA_PORT=8080','STRATA_PARALLEL=2','STRATA_BATCH_GROUPS=2','STRATA_SC117_IQ3S_TRIAL=1'):self.assertIn(value,a)
  self.assertFalse(any(x.startswith('STRATA_API_KEY=') for x in a));self.assertEqual(launch.call_args.kwargs,{'capture_output':True,'text':True,'timeout':600})
 def test_missing_runuser_is_a_readiness_failure(self):
  with mock.patch.object(S.shutil,'which',return_value=None),mock.patch.object(S.subprocess,'run') as invoke:
   with self.assertRaisesRegex(SafetyError,'runuser'):S.preflight_profile(TARGET)
  invoke.assert_not_called()
 def test_legacy_runuser_location_supported(self):
  with mock.patch.object(S.shutil,'which',return_value='/bin/runuser'),mock.patch.object(S.subprocess,'run',return_value=self.ready()) as launch:S.preflight_profile(TARGET)
  self.assertEqual(launch.call_args.args[0][0],'/bin/runuser')
 def test_exact_unambiguous_readiness_only(self):
  for text in ('prefix '+S.sc117_two.READY,S.sc117_two.READY+'\n'+S.sc117_two.READY,S.sc117_two.READY+'\nREADY CONFIG: contradictory'):
   with self.subTest(text=text),mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout=text,stderr='')):
    with self.assertRaisesRegex(SafetyError,'before any pause'):S.preflight_profile(TARGET)
 def test_other_historical_metadata_is_not_a_switch_target(self):
  self.assertNotIn('orca-q4_k_s-v01441',S.PROFILES)
  with mock.patch.object(S.shutil,'which') as lookup,mock.patch.object(S.subprocess,'run') as invoke:S.preflight_profile('orca-q4_k_s-v01441')
  lookup.assert_not_called();invoke.assert_not_called()
 def test_real_resolved_runuser_stub_is_executed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);stub=p/'runuser';trace=p/'argv';stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > '+str(trace)+'\nprintf "%s\\n" '+shlex.quote(S.sc117_two.READY)+'\n');stub.chmod(0o755)
   with mock.patch.object(S.shutil,'which',return_value=str(stub)):S.preflight_profile(TARGET)
   a=trace.read_text().splitlines();self.assertEqual(a[:5],['-u','user','--','/usr/bin/env','-i']);self.assertEqual(a[-6:],['--runtime','0.1.46','--parallel','2','--batch-groups','2'])

class OwnershipChecks(unittest.TestCase):
 @staticmethod
 def identity(pid,parent,uid=1000):return {'pid':pid,'ppid':parent,'uid':uid,'start_ticks':pid*10,'exe':'/usr/bin/python3','cmd':['python3'],'pgid':pid,'sid':pid}
 def test_gpu_ancestry_must_reach_exact_live_root_hive_wrapper(self):
  wrapper=self.identity(10,1,0);launcher=self.identity(20,10);native=self.identity(30,20);identities={10:wrapper,20:launcher,30:native}
  with mock.patch.object(S,'proc',side_effect=identities.get):self.assertTrue(S.belongs_to_wrapper(native,{10:wrapper}));self.assertFalse(S.belongs_to_wrapper(native,{}))
  with mock.patch.object(S,'proc',return_value=dict(native,ppid=1)):self.assertFalse(S.belongs_to_wrapper(native,{10:wrapper}))
 def test_rental_and_manual_lease_block_before_stop(self):
  with mock.patch.object(S,'docker_empty',side_effect=SafetyError('rental')),mock.patch.object(S,'owned_wrappers') as wrappers:
   with self.assertRaisesRegex(SafetyError,'rental'):S.inspect_owned_workload()
   wrappers.assert_not_called()
  with mock.patch.object(S,'docker_empty'),mock.patch.object(S.hosting,'read_lease',return_value={'phase':'held'}),mock.patch.object(S,'owned_wrappers') as wrappers:
   with self.assertRaisesRegex(SafetyError,'manual HostLLM'):S.inspect_owned_workload()
   wrappers.assert_not_called()
if __name__=='__main__':unittest.main()
