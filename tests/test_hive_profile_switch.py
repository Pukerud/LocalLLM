"""CPU-only switch transactions; all service/miner/process boundaries are mocked."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as switch
from engine_safety import SafetyError


class SwitchTransaction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.current = self.root / 'current'
        self.current.symlink_to('profiles/orca-q4_k_s-v01441')
        self.stop = self.root / 'MINER_STOP'
        self.journal = self.root / 'switch.json'
        self.service = 'active'
        self.miner = True
        self.events = []
        self.wrappers = {10: {'pid': 10}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        def patch(obj, name, **kwargs):
            return self.stack.enter_context(mock.patch.object(obj, name, **kwargs))
        patch(switch, 'CURRENT', new=self.current)
        patch(switch, 'JOURNAL', new=self.journal)
        patch(switch, 'SCRIPT_ROOT', new=self.root)
        patch(switch.hosting, 'ROOT', new=self.root)
        patch(switch.hosting, 'STOP', new=self.stop)
        patch(switch.hosting, 'service_state', side_effect=lambda: self.service)
        patch(switch.hosting, 'wait_idle')
        patch(switch, 'docker_empty')
        self.inspected = patch(switch, 'inspect_owned_workload', side_effect=lambda: self.wrappers if self.miner else {})
        patch(switch, 'owned_wrappers', side_effect=lambda: self.wrappers if self.miner else {})
        patch(switch, 'llm_processes', return_value=[])
        patch(switch, 'package_current', return_value=True)
        self.execute = patch(switch, 'run', side_effect=self.command)

    def command(self, args, timeout=180):
        self.events.append(tuple(args))
        if args == ['systemctl', 'stop', switch.hosting.SERVICE]:
            self.service = 'inactive'
        elif args == [switch.hosting.MINER, 'stop', '60']:
            self.assertEqual(self.service, 'inactive', 'pause the auto-restarter BEFORE miner stop')
            self.miner = False
        elif '--apply-hive-profile' in args:
            self.assertEqual(self.service, 'inactive')
            self.assertFalse(self.miner)
            self.assertTrue(self.stop.exists(), 'hold MINER_STOP during slow readiness and pointer swap')
            self.current.unlink()
            self.current.symlink_to('profiles/' + args[-1])
        elif args == [switch.hosting.MINER, 'start']:
            self.assertEqual(self.service, 'inactive', 'start selected miner before resuming OctaSpace')
            self.miner = True
        elif args == ['systemctl', 'start', switch.hosting.SERVICE]:
            self.service = 'active'
        else:
            self.fail(f'unexpected command: {args}')

    def test_running_orca_switches_to_sc117_without_restart_window(self):
        switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertEqual(os.readlink(self.current), 'profiles/iq3_s-sc117-abliterated')
        self.assertEqual([event[1] for event in self.events], ['stop', 'stop', '--apply-hive-profile', 'start', 'start'])
        self.assertEqual(self.service, 'active')
        self.assertTrue(self.miner)
        self.assertFalse(self.stop.exists())
        self.assertFalse(self.journal.exists())

    def test_octa_stop_may_already_stop_the_miner(self):
        original = self.command
        def stop_together(args, timeout=180):
            original(args, timeout)
            if args == ['systemctl', 'stop', switch.hosting.SERVICE]:
                self.miner = False
        self.execute.side_effect = stop_together
        switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertNotIn((switch.hosting.MINER, 'stop', '60'), self.events)
        self.assertTrue(self.miner)
        self.assertEqual(os.readlink(self.current), 'profiles/iq3_s-sc117-abliterated')

    def test_stop_race_is_accepted_only_after_verified_idle(self):
        original = self.command
        def stop_race(args, timeout=180):
            if args == [switch.hosting.MINER, 'stop', '60']:
                self.miner = False
                raise SafetyError('No miner screens found')
            original(args, timeout)
        self.execute.side_effect = stop_race
        switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertTrue(self.miner)
        self.assertEqual(self.service, 'active')
        self.assertFalse(self.journal.exists())

    def test_same_current_profile_is_instant_without_stop_or_asset_reload(self):
        switch.switch_profile('orca-q4_k_s-v01441')
        self.execute.assert_not_called()
        self.assertFalse(self.journal.exists())

    def test_already_idle_miner_is_not_started_by_the_switch(self):
        self.miner = False
        switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertFalse(self.miner)
        self.assertNotIn((switch.hosting.MINER, 'start'), self.events)
        self.assertEqual(self.service, 'active')

    def test_inactive_service_and_existing_stop_marker_are_preserved(self):
        self.service = 'inactive'
        self.miner = False
        self.stop.write_text('operator-stop\n')
        switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertEqual(self.service, 'inactive')
        self.assertEqual(self.stop.read_text(), 'operator-stop\n')
        self.assertFalse(self.miner)

    def test_readiness_failure_restores_old_miner_and_service(self):
        original = self.command
        def fail_readiness(args, timeout=180):
            if '--apply-hive-profile' in args:
                self.assertEqual(self.service, 'inactive')
                self.assertTrue(self.stop.exists())
                raise SafetyError('readiness failed')
            original(args, timeout)
        self.execute.side_effect = fail_readiness
        with self.assertRaisesRegex(SafetyError, 'readiness failed'):
            switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertEqual(os.readlink(self.current), 'profiles/orca-q4_k_s-v01441')
        self.assertTrue(self.miner)
        self.assertEqual(self.service, 'active')
        self.assertFalse(self.journal.exists())

    def test_new_unowned_workload_retains_recovery_journal_without_restart(self):
        self.inspected.side_effect = [self.wrappers, self.wrappers, SafetyError('new rental')]
        with self.assertRaisesRegex(SafetyError, 'new rental'):
            switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertEqual(self.service, 'inactive')
        self.assertFalse(self.miner)
        self.assertTrue(self.journal.exists())
        self.assertEqual(self.journal.stat().st_mode & 0o777, 0o600)
        self.assertNotIn((switch.hosting.MINER, 'start'), self.events)

    def test_recovery_command_preserves_the_hostllm_state_root(self):
        command = switch.recovery_command()
        self.assertIn(f'HOSTLLM_STATE_ROOT={self.root}', command)
        self.assertTrue(command.endswith('--recover'))

    def test_preexisting_switch_journal_is_not_overwritten(self):
        self.journal.write_text('old transaction')
        with self.assertRaisesRegex(SafetyError, 'needs recovery') as caught:
            switch.switch_profile('iq3_s-sc117-abliterated')
        self.assertIn(f'HOSTLLM_STATE_ROOT={self.root}', str(caught.exception))
        self.assertIn('--recover', str(caught.exception))
        self.execute.assert_not_called()
        self.assertEqual(self.journal.read_text(), 'old transaction')


class OwnershipChecks(unittest.TestCase):
    @staticmethod
    def identity(pid, parent, uid=1000):
        return {'pid': pid, 'ppid': parent, 'uid': uid, 'start_ticks': pid * 10,
                'exe': '/usr/bin/python3', 'cmd': ['python3'], 'pgid': pid, 'sid': pid}

    def test_gpu_ancestry_must_reach_the_verified_root_hive_wrapper(self):
        wrapper = self.identity(10, 1, 0)
        launcher = self.identity(20, 10)
        native = self.identity(30, 20)
        identities = {10: wrapper, 20: launcher, 30: native}
        with mock.patch.object(switch, 'proc', side_effect=identities.get):
            self.assertTrue(switch.belongs_to_wrapper(native, {10: wrapper}))
            self.assertFalse(switch.belongs_to_wrapper(native, {}))
        changed = dict(native, ppid=1)
        with mock.patch.object(switch, 'proc', return_value=changed):
            self.assertFalse(switch.belongs_to_wrapper(native, {10: wrapper}))

    def test_rental_and_hostllm_lease_block_before_any_stop(self):
        with mock.patch.object(switch, 'docker_empty', side_effect=SafetyError('rental')), \
                mock.patch.object(switch, 'owned_wrappers') as wrappers:
            with self.assertRaisesRegex(SafetyError, 'rental'):
                switch.inspect_owned_workload()
            wrappers.assert_not_called()
        with mock.patch.object(switch, 'docker_empty'), \
                mock.patch.object(switch.hosting, 'read_lease', return_value={'phase': 'held'}), \
                mock.patch.object(switch, 'owned_wrappers') as wrappers:
            with self.assertRaisesRegex(SafetyError, 'manual HostLLM'):
                switch.inspect_owned_workload()
            wrappers.assert_not_called()


if __name__ == '__main__':
    unittest.main()
