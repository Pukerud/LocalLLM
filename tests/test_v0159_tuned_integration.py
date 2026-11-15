"""Pure mocked integration contracts; no real permission, GPU, service or lifecycle mutation."""
import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,'/home/user/LocalLLM')
import hive_profile_switch as S
import strata_launcher as L
import engine_safety as E
import unsloth_v31_v0159_tuned_profile as U
from engine_safety import SafetyError

class TunedIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(U,'APPROVAL',Path(self.tmp.name)/'absent-tuned').start()
        mock.patch.object(S.v3159,'APPROVAL',Path(self.tmp.name)/'absent-original').start()
    def test_separate_prepared_not_membership_authorized(self):
        self.assertIn(U.HIVE_PROFILE,S.PREPARED_PROFILES);self.assertNotIn(U.HIVE_PROFILE,S.PROFILES)
        self.assertIs(S.prepared_profile_module(U.HIVE_PROFILE),U)
        self.assertIs(S.prepared_profile_module(S.v3159.HIVE_PROFILE),S.v3159)
        with self.assertRaises(SafetyError):S.prepared_profile_module('unknown')
    def test_tuned_lock_refuses_before_inspection_or_action(self):
        with mock.patch.object(U,'authorize_activation',side_effect=SafetyError('TUNED LOCKED')) as auth,mock.patch.object(S,'inspect_owned_workload') as inspect,mock.patch.object(S,'run') as run,mock.patch.object(S,'write_journal') as journal:
            with self.assertRaisesRegex(SafetyError,'TUNED LOCKED'):S.switch_profile(U.HIVE_PROFILE)
            auth.assert_called_once();inspect.assert_not_called();run.assert_not_called();journal.assert_not_called()
    def test_original_does_not_borrow_tuned_grant(self):
        with mock.patch.object(S.v3159,'authorize_activation',side_effect=SafetyError('ORIGINAL LOCKED')),mock.patch.object(U,'authorize_activation') as other,mock.patch.object(S,'inspect_owned_workload') as inspect:
            with self.assertRaisesRegex(SafetyError,'ORIGINAL LOCKED'):S.switch_profile(S.v3159.HIVE_PROFILE)
            other.assert_not_called();inspect.assert_not_called()
    def test_tool_compat_is_only_additional_serving_setting(self):
        recipe={'case':'batch-pipeline','spec':U.T.registry()['batch-pipeline']}
        with mock.patch.object(U,'recipe',return_value=recipe):
            provider=U.provider()
            previous=U.T.TrialProfile.canonical_config(provider,U.T.ROOT)
            previous['aliases']=[U.MODEL_ID,U.PROFILE,U.C.MODEL_ID,U.C.PROFILE]
            cfg=provider.canonical_config(U.T.ROOT)
            self.assertEqual(cfg,{**previous,'undeclared_tool_calls':'call'})
            self.assertEqual(U.canonical_config(U.T.ROOT),cfg)
            self.assertEqual(U.validate_config(U.T.ROOT,cfg),cfg)
            self.assertEqual(provider.validate_config(U.T.ROOT,cfg),cfg)
    def test_tool_compat_cannot_silently_revert_or_change(self):
        recipe={'case':'batch-pipeline','spec':U.T.registry()['batch-pipeline']}
        with mock.patch.object(U,'recipe',return_value=recipe):
            cfg=U.canonical_config(U.T.ROOT)
            for value in ('text',True,'unknown',None):
                bad=dict(cfg)
                if value is None:bad.pop('undeclared_tool_calls')
                else:bad['undeclared_tool_calls']=value
                with self.subTest(value=value),self.assertRaises(SafetyError):U.validate_config(U.T.ROOT,bad)
    def test_tool_compat_not_added_to_official_or_native_profiles(self):
        self.assertNotIn('undeclared_tool_calls',U.C.canonical_config(U.T.ROOT,Path('/home/user')))
        self.assertNotIn('undeclared_tool_calls',U.C.BASE.canonical_config(U.T.ROOT,Path('/home/user')))
    def test_offline_readiness_not_start_permission(self):
        result=types.SimpleNamespace(returncode=0,stdout='Runtime ready\n'+U.READY+'\n',stderr='')
        with mock.patch.object(U,'authorize_activation') as auth,mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=result) as run:
            S.preflight_profile(U.HIVE_PROFILE,offline=True);auth.assert_not_called()
            args=run.call_args.args[0]
            self.assertIn(str(S.SCRIPT_ROOT/'v0159_tuned_strata_launcher.py'),args)
            self.assertIn(U.TRIAL_ENV+'=1',args);self.assertIn('--check-ready',args)
            self.assertNotIn('--quickstart',args)
    def test_live_readiness_correct_permission_first(self):
        with mock.patch.object(U,'authorize_activation',side_effect=SafetyError('LOCKED')),mock.patch.object(S.subprocess,'run') as run:
            with self.assertRaises(SafetyError):S.preflight_profile(U.HIVE_PROFILE)
            run.assert_not_called()
    def test_wrong_readiness_line_refuses(self):
        result=types.SimpleNamespace(returncode=0,stdout='READY CONFIG: another model\n',stderr='')
        with mock.patch.object(S.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(S.subprocess,'run',return_value=result):
            with self.assertRaises(SafetyError):S.preflight_profile(U.HIVE_PROFILE,offline=True)
    def test_prepared_active_publisher_uses_real_preflight(self):
        import install_retained_profiles as P
        with mock.patch.object(S,'preflight_profile',side_effect=SafetyError('NO GRANT')) as gate:
            with self.assertRaises(SafetyError):P.validate_active_selection(U.HIVE_PROFILE)
            gate.assert_called_once_with(U.HIVE_PROFILE)
    def test_new_state_is_detected_without_inherited_environment(self):
        base=Path(self.tmp.name);root=base/'.local/state/locallm-strata'/U.STATE_NAME;root.mkdir(parents=True)
        identity={'pid':345678,'start_ticks':1,'exe':'/usr/bin/python3.10','uid':1000,'cmd':['python','-m','serve.server'],'pgid':345678,'sid':345678}
        (root/'server.json').write_text(json.dumps({'identity':identity,'profile':U.PROFILE}))
        with mock.patch.object(E,'owner_home',return_value=base),mock.patch.object(E,'strata_state',return_value=base/'unused'),mock.patch.object(E,'proc',return_value=identity):
            self.assertEqual(E.active_engine(),'strata')
    def test_selector_high_low_ids_and_early_tuned_gate(self):
        root=S.SCRIPT_ROOT
        text=(root/'hive-llm-miner/profile-menu.sh').read_text()
        gate=text.index('if [[ "$target" == '+U.HIVE_PROFILE+' ]]')
        lock=text.index('umask 077; exec 9>')
        self.assertLess(gate,lock)
        self.assertIn('11|'+U.HIVE_PROFILE,text)
        host=(root/'HostLLM.sh').read_text()
        self.assertIn('23) set_next_hive_profile '+U.HIVE_PROFILE,host)
        self.assertIn('--stop-owned-v3159-tuned',host)
    def test_family_extension_not_broad_session_adoption(self):
        text=(S.SCRIPT_ROOT/'strata_launcher.py').read_text()
        self.assertIn("state.get('profile') == '"+U.PROFILE+"'",text)
        self.assertIn("state.get('runtime_version') == '0.1.59'",text)
        for guard in ("p['uid'] != leader['uid']","p.get('ppid') != leader['pid']","p['exe'] not in allowed","not same_process(known, p)"):
            self.assertIn(guard,text)
if __name__=='__main__':unittest.main()
