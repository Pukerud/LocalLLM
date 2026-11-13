"""Offline gate/handoff fixtures only. Never publish a real approval or call real lifecycle operations."""
import copy
import os
import stat
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_safety import SafetyError
import unsloth_v31_v0159_profile as C
import strata_v0213_activation as A


class Gate(unittest.TestCase):
    def test_permission_template_is_bound_to_all_actual_critical_policies(self):
        value = C.approval_contract()
        self.assertEqual(value['profile'], C.HIVE_PROFILE)
        self.assertEqual(value['scope'], C.CONFIRMATION)
        self.assertEqual(value['owner_uid'], 1000)
        self.assertNotIn('pid', value)
        self.assertNotIn('launch_token', value)
        self.assertIn('strata_launcher.py', value['policy_sha256'])
        self.assertEqual(C.validate_approval(value), 'trial')
        for field in ('runtime', 'model_revision', 'ple_sha256', 'configuration_sha256', 'scope'):
            bad = copy.deepcopy(value);bad[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(SafetyError):
                C.validate_approval(bad)
        bad = copy.deepcopy(value);bad['policy_sha256']['strata_launcher.py'] = 'wrong'
        with self.assertRaises(SafetyError):
            C.validate_approval(bad)

    def test_new_public_approval_directory_has_explicit_mode_after_mkdir(self):
        parent=MagicMock()
        parent.exists.return_value=False
        parent.lstat.return_value=SimpleNamespace(st_mode=stat.S_IFDIR|0o700,st_uid=0)
        A.approval_directory(parent)
        parent.mkdir.assert_called_once_with(mode=0o755)
        parent.chmod.assert_called_once_with(0o755)

    def test_existing_private_directory_is_not_silently_broadened(self):
        parent=MagicMock()
        parent.exists.return_value=True
        parent.lstat.return_value=SimpleNamespace(st_mode=stat.S_IFDIR|0o700,st_uid=0)
        with self.assertRaisesRegex(SafetyError,'explicit administrator correction'):
            A.approval_directory(parent)
        parent.chmod.assert_not_called()

    def test_owner_approval_check_is_actual_unprivileged_key_free_policy_read(self):
        with patch.object(A.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            A.check_owner_approval()
        command=run.call_args.args[0]
        self.assertEqual(command[:4],['/usr/sbin/runuser','-u','user','--'])
        self.assertIn('C.authorize_activation()',command[-1])
        self.assertIn('PYTHONDONTWRITEBYTECODE=1',command)
        self.assertNotIn('STRATA_API_KEY',' '.join(command))
        self.assertNotIn('activate(',command[-1])

    def test_unreadable_owner_permission_refuses_before_switch(self):
        with patch.object(A.subprocess,'run',return_value=SimpleNamespace(returncode=1)), \
             patch.object(A.S,'switch_profile') as switch:
            with self.assertRaisesRegex(SafetyError,'no stop/switch'):
                A.check_owner_approval()
        switch.assert_not_called()

    def test_no_confirmation_means_no_reservation_or_permission_or_switch(self):
        for confirmation in (None, '', 'OK', 'yes', 'OPERATOR-APPROVED'):
            with self.subTest(confirmation=confirmation), patch.object(A.S.hosting, 'launch_reservation') as reservation, \
                 patch.object(A, 'publish_approval') as publish, patch.object(A.S, 'switch_profile') as switch:
                with self.assertRaises(SafetyError):A.activate(confirmation)
                reservation.assert_not_called();publish.assert_not_called();switch.assert_not_called()

    def test_unprivileged_confirmation_cannot_grant_permission(self):
        with patch.object(A.os, 'geteuid', return_value=1000), patch.object(A.S.hosting, 'launch_reservation') as reservation:
            with self.assertRaises(SafetyError):A.activate(C.CONFIRMATION)
            reservation.assert_not_called()

    def test_check_cannot_publish_approval_or_switch(self):
        with patch.object(A, 'check', return_value={'offline_ready':True}) as check, \
             patch.object(A.os, 'geteuid', return_value=0), \
             patch.object(A.S.hosting, 'launch_reservation', return_value=MagicMock()), \
             patch.object(A, 'publish_approval') as grant, patch.object(A.S, 'switch_profile') as switch:
            self.assertEqual(A.main(['--check']), 0)
            check.assert_called_once();grant.assert_not_called();switch.assert_not_called()

    def test_authorized_future_handoff_waits_naturally_then_uses_existing_transaction(self):
        identity={'pid':1234,'start_ticks':5678,'exe':'/owned/python','uid':1000,'pgid':1234,'sid':1234,'cmd':['owned']}
        state={'identity':identity}
        events=[]
        with patch.object(A.os,'geteuid',return_value=0), \
             patch.object(A.S.hosting,'launch_reservation',return_value=MagicMock()), \
             patch.object(A,'check',side_effect=lambda:events.append('check')), \
             patch.object(A,'current_state',return_value=state), \
             patch.object(A,'wait_for_natural_idle',side_effect=lambda s:events.append('wait')), \
             patch.object(A,'hardware',side_effect=lambda:events.append('hardware')), \
             patch.object(A.S,'docker_empty'), patch.object(A.S,'inspect_owned_workload'), \
             patch.object(A,'request_counts',return_value=0), \
             patch.object(A,'publish_approval',side_effect=lambda:events.append('permission')), \
             patch.object(A.S,'switch_profile',side_effect=lambda p:events.append(('switch',p))):
            A.activate(C.CONFIRMATION)
        self.assertEqual(events,['check','wait','hardware','permission',('switch',C.HIVE_PROFILE)])

    def test_new_request_appearing_prevents_permission_and_stop(self):
        identity={'pid':1234,'start_ticks':5678,'exe':'/owned/python','uid':1000,'pgid':1234,'sid':1234,'cmd':['owned']}
        with patch.object(A.os,'geteuid',return_value=0), patch.object(A.S.hosting,'launch_reservation',return_value=MagicMock()), \
             patch.object(A,'check'), patch.object(A,'current_state',return_value={'identity':identity}), \
             patch.object(A,'wait_for_natural_idle'), patch.object(A,'hardware'), patch.object(A.S,'docker_empty'), \
             patch.object(A.S,'inspect_owned_workload'), patch.object(A,'request_counts',return_value=1), \
             patch.object(A,'publish_approval') as grant, patch.object(A.S,'switch_profile') as switch:
            with self.assertRaises(SafetyError):A.activate(C.CONFIRMATION)
            grant.assert_not_called();switch.assert_not_called()

    def test_active_requests_timeout_without_cancellation_or_stop(self):
        identity={'pid':1234,'start_ticks':5678,'exe':'/owned/python','uid':1000,'pgid':1234,'sid':1234,'cmd':['owned']}
        state={'identity':identity}
        with patch.object(A.S,'docker_empty'), patch.object(A.S,'inspect_owned_workload'), \
             patch.object(A,'current_state',return_value=state), patch.object(A,'request_counts',return_value=2), \
             patch.object(A.time,'monotonic',side_effect=[0,301]), patch.object(A.S,'run') as run:
            with self.assertRaisesRegex(SafetyError,'no cancellation or stop'):
                A.wait_for_natural_idle(state)
            run.assert_not_called()


if __name__=='__main__':unittest.main()
