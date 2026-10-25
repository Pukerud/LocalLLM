"""CPU-only cleanup-lock race tests; no server, signal, service or GPU action."""
import sys
from pathlib import Path
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import strata_launcher as L
from engine_safety import SafetyError


class StopLock(unittest.TestCase):
    def state(self,tick=123):
        return {'identity':{'pid':42,'start_ticks':tick,'exe':'/test/python','uid':1000,
                            'cmd':['python','-m','serve.server'],'pgid':42,'sid':42}}

    def test_hive_foreground_cleanup_wins_stop_waits_and_accepts_absent_state(self):
        with mock.patch.object(L,'read_state',side_effect=[self.state(),None]),\
             mock.patch.object(L.fcntl,'flock',side_effect=[BlockingIOError(11,'busy'),None]) as flock,\
             mock.patch.object(L.time,'sleep') as sleep:
            self.assertIsNone(L.acquire_action_lock(object(),stopping=True))
            self.assertEqual(flock.call_count,2);sleep.assert_called_once_with(.1)

    def test_same_recorded_frontend_is_returned_for_identity_bound_stop(self):
        before=self.state();after=self.state()
        with mock.patch.object(L,'read_state',side_effect=[before,after]),mock.patch.object(L.fcntl,'flock'):
            self.assertIs(L.acquire_action_lock(object(),stopping=True),after)

    def test_reused_pid_or_new_session_is_not_adopted(self):
        for before in [self.state(),None]:
            with self.subTest(before=before),mock.patch.object(L,'read_state',side_effect=[before,self.state(999)]),\
                 mock.patch.object(L.fcntl,'flock'),self.assertRaisesRegex(SafetyError,'replacement session was not stopped'):
                L.acquire_action_lock(object(),stopping=True)

    def test_model_start_stays_nonblocking_and_never_reads_or_adopts_a_stop_target(self):
        with mock.patch.object(L,'read_state') as state,mock.patch.object(L.fcntl,'flock',side_effect=BlockingIOError(11,'busy')),\
             mock.patch.object(L.time,'sleep') as sleep,self.assertRaises(BlockingIOError):
            L.acquire_action_lock(object())
        state.assert_not_called();sleep.assert_not_called()

    def test_timeout_remains_fail_closed(self):
        with mock.patch.object(L,'read_state',return_value=self.state()),mock.patch.object(L.fcntl,'flock',side_effect=BlockingIOError(11,'busy')),\
             mock.patch.object(L.time,'monotonic',side_effect=[0,46]),self.assertRaisesRegex(SafetyError,'no unknown process was stopped'):
            L.acquire_action_lock(object(),stopping=True)


if __name__=='__main__':unittest.main()
