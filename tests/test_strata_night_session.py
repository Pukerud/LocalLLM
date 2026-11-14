import tempfile,types,unittest
from pathlib import Path
from unittest import mock
import sys
sys.path.insert(0,'/home/user/LocalLLM')
import strata_night_session as M
from engine_safety import SafetyError

class QuiescentSessionTests(unittest.TestCase):
    def test_waits_only_for_natural_clear(self):
        s=M.Session()
        with mock.patch.object(M.K.A,'hardware') as hw,mock.patch.object(M.K.S,'docker_empty'),mock.patch.object(M.K.H,'read_lease',return_value={'phase':'held','lease_id':s.lease}),mock.patch.object(M.K.S,'llm_processes',side_effect=[['unknown'],[],[]]),mock.patch.object(M.K.H,'gpu_jobs',return_value=[]),mock.patch.object(M.K.H,'active_engine',return_value='none'),mock.patch.object(M.time,'monotonic',side_effect=[0,1,2,3,4,8]),mock.patch.object(M.time,'sleep'),mock.patch.object(M.K.S,'stop_owned_v31') as stop:
            s.quiet();self.assertEqual(hw.call_count,3);stop.assert_not_called()
    def test_renter_never_overridden(self):
        s=M.Session()
        with mock.patch.object(M.K.A,'hardware'),mock.patch.object(M.K.S,'docker_empty',side_effect=SafetyError('rental')),mock.patch.object(M.K.S,'stop_owned_v31') as stop:
            with self.assertRaises(SafetyError):s.quiet()
            stop.assert_not_called()
    def test_changed_lease_refuses(self):
        s=M.Session()
        with mock.patch.object(M.K.A,'hardware'),mock.patch.object(M.K.S,'docker_empty'),mock.patch.object(M.K.H,'read_lease',return_value={'phase':'held','lease_id':'other'}):
            with self.assertRaises(SafetyError):s.quiet()
    def test_pre_spawn_refusal_new_empty_label_only(self):
        s=M.Session();s.quiet=mock.Mock()
        with tempfile.TemporaryDirectory() as d:
            p=types.SimpleNamespace(state_root=lambda:Path(d)/'not-created')
            with mock.patch.object(M.K.Session,'start',side_effect=[SafetyError('LLM/GPU/manual work present; no trial start'),'ready']) as start,mock.patch.object(M.K.T,'TrialProfile',return_value=p),mock.patch.object(M.K,'save') as save:
                self.assertEqual(s.start('official','new'), 'ready')
                self.assertEqual([x.args[1] for x in start.call_args_list],['new-p1','new-p2'])
                self.assertEqual(save.call_count,1);self.assertEqual(s.quiet.call_count,2)
    def test_spawned_runtime_never_retried(self):
        s=M.Session();s.quiet=mock.Mock()
        def fail(*args):s.worker=object();raise SafetyError('LLM/GPU/manual work present; no trial start')
        with mock.patch.object(M.K.Session,'start',side_effect=fail) as start,mock.patch.object(M.K.T,'TrialProfile'),mock.patch.object(M.K,'save') as save:
            with self.assertRaises(SafetyError):s.start('official','new')
            self.assertEqual(start.call_count,1);save.assert_not_called()
    def test_other_failure_never_retried(self):
        s=M.Session();s.quiet=mock.Mock()
        with mock.patch.object(M.K.Session,'start',side_effect=SafetyError('Exact trial child exited')) as start,mock.patch.object(M.K.T,'TrialProfile'),mock.patch.object(M.K,'save') as save:
            with self.assertRaises(SafetyError):s.start('official','new')
            self.assertEqual(start.call_count,1);save.assert_not_called()
if __name__=='__main__':unittest.main()
