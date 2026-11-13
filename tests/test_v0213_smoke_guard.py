"""Future live client guards, tested with data/mocks only: no model generation or HTTP."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import v0213_live_smoke as M
from engine_safety import SafetyError


class SmokeGuard(unittest.TestCase):
    def test_current_46_or_wrong_features_never_pass_target_guard(self):
        valid={'loaded':True,'engine':'0.1.59','model':M.F.MODEL_ID,
               'context':{'native':262144},'concurrency':{'serving':2}}
        M.require_candidate_status(valid)
        for k,v in (('engine','0.1.46'),('model','current-21'),('loaded',False),
                    ('context',{'native':131072}),('concurrency',{'serving':1})):
            bad=copy.deepcopy(valid);bad[k]=v
            with self.subTest(k=k),self.assertRaisesRegex(RuntimeError,'no generation sent'):
                M.require_candidate_status(bad)

    def test_waiter_never_generates_on_old_target_and_times_out_cleanly(self):
        with patch.object(M,'get',return_value={'engine':'0.1.46'}), \
             patch.object(M.time,'monotonic',side_effect=[0,1]),patch.object(M.C,'stream') as stream:
            with self.assertRaisesRegex(RuntimeError,'no generation sent'):M.wait_for_candidate(0)
            stream.assert_not_called()

    def test_waiter_accepts_only_exact_new_loaded_target(self):
        good={'loaded':True,'engine':'0.1.59','model':M.F.MODEL_ID,'context':{'native':262144},'concurrency':{'serving':2}}
        with patch.object(M,'get',side_effect=[{'engine':'0.1.46'},good]), \
             patch.object(M.time,'monotonic',return_value=0),patch.object(M.time,'sleep') as sleep:
            self.assertEqual(M.wait_for_candidate(),good);sleep.assert_called_once_with(2)

    def test_missing_permission_refuses_even_before_any_http(self):
        with patch.object(sys,'argv',['smoke','--phase','not-permitted']), \
             patch.object(M.os,'geteuid',return_value=1000), \
             patch.object(M.F,'authorize_activation',side_effect=SafetyError('LOCKED')), \
             patch.object(M,'request') as request,patch.object(M,'get') as get,patch.object(M.C,'stream') as stream:
            with self.assertRaises(SafetyError):M.main()
            request.assert_not_called();get.assert_not_called();stream.assert_not_called()


if __name__=='__main__':unittest.main()
