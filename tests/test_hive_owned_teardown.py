"""Drain only exact recorded identities; never signal, adopt or stop unknown work."""
from contextlib import ExitStack
from pathlib import Path
import sys,unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as S
from engine_safety import SafetyError


class OwnedTeardown(unittest.TestCase):
    def process(self,pid=42,tick=123):
        return {'pid':pid,'ppid':1,'start_ticks':tick,'exe':'/test/python','uid':1000,
                'cmd':['python','-m','serve.server'],'pgid':42,'sid':42}

    def setup_guard(self,stack):
        stack.enter_context(mock.patch.object(S,'docker_empty'))
        stack.enter_context(mock.patch.object(S.hosting,'read_lease',return_value=None))
        stack.enter_context(mock.patch.object(S.hosting,'gpu_jobs',return_value=[]))
        stack.enter_context(mock.patch.object(S,'owned_wrappers',return_value={}))
        stack.enter_context(mock.patch.object(S,'belongs_to_wrapper',return_value=False))
        stack.enter_context(mock.patch.object(S.time,'sleep'))

    def test_recorded_orphan_drains_without_a_wrapper_or_any_signal(self):
        p=self.process()
        with ExitStack() as stack:
            self.setup_guard(stack)
            stack.enter_context(mock.patch.object(S,'llm_processes',side_effect=[[p],[]]))
            with mock.patch.object(S,'run') as action:
                S.wait_owned_teardown([S.teardown_identity(p)])
            action.assert_not_called()

    def test_reused_pid_and_new_manual_process_are_refused_immediately(self):
        for p in [self.process(tick=999),self.process(pid=99)]:
            with self.subTest(p=p),ExitStack() as stack:
                self.setup_guard(stack)
                stack.enter_context(mock.patch.object(S,'llm_processes',return_value=[p]))
                with self.assertRaisesRegex(SafetyError,'new manual/unowned process'):
                    S.wait_owned_teardown([S.teardown_identity(self.process())])

    def test_changed_command_session_or_executable_is_not_the_old_teardown(self):
        original=self.process()
        for key,value in [('cmd',['python','manual.py']),('sid',999),('exe','/tmp/replacement')]:
            p=dict(original,**{key:value})
            with self.subTest(key=key),ExitStack() as stack:
                self.setup_guard(stack)
                stack.enter_context(mock.patch.object(S,'llm_processes',return_value=[p]))
                with self.assertRaises(SafetyError):S.wait_owned_teardown([S.teardown_identity(original)])

    def test_timeout_and_new_maintenance_lease_do_not_restore_or_signal(self):
        p=self.process()
        with ExitStack() as stack:
            self.setup_guard(stack)
            stack.enter_context(mock.patch.object(S,'llm_processes',return_value=[p]))
            stack.enter_context(mock.patch.object(S.time,'monotonic',side_effect=[0,91]))
            with self.assertRaisesRegex(SafetyError,'no unknown process was stopped'):
                S.wait_owned_teardown([S.teardown_identity(p)])
        with mock.patch.object(S,'docker_empty'),mock.patch.object(S.hosting,'read_lease',return_value={'foreign':True}),self.assertRaisesRegex(SafetyError,'Another hosting lease'):
            S.wait_owned_teardown([])

    def test_journal_identity_omits_sensitive_command_arguments(self):
        p=self.process();p['cmd']=['miner','--password','private-fixture']
        record=S.teardown_identity(p)
        self.assertNotIn('cmd',record);self.assertNotIn('private-fixture',str(record))
        self.assertEqual(len(record['cmd_sha256']),64)


if __name__=='__main__':unittest.main()
