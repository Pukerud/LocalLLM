"""CPU-only switch transactions; all service/miner/process boundaries are mocked."""
from contextlib import ExitStack
import os
from pathlib import Path
import shlex
import subprocess
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
        self.wrappers = {10: {'pid':10,'start_ticks':11,'exe':'/bin/bash','uid':0,
                              'cmd':['bash','/test/h-run.sh'],'pgid':10,'sid':10}}
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
        patch(switch.hosting, 'gpu_jobs', return_value=[])
        patch(switch, 'wait_owned_teardown')
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

    def test_v0146_sc117_preflight_runs_before_any_service_or_miner_pause(self):
        calls=[]
        def preflight(profile):
            self.assertEqual(profile,switch.SC117_V0146_PROFILE)
            self.assertEqual(self.service,'active')
            self.assertTrue(self.miner)
            self.assertFalse(self.stop.exists())
            calls.append(profile)
        with mock.patch.object(switch,'preflight_profile',side_effect=preflight):
            switch.switch_profile(switch.SC117_V0146_PROFILE)
        self.assertEqual(calls,[switch.SC117_V0146_PROFILE])
        self.assertEqual(os.readlink(self.current),f'profiles/{switch.SC117_V0146_PROFILE}')
        self.assertTrue(self.miner)
        self.assertEqual(self.service,'active')

    def test_v0146_missing_runuser_refuses_before_any_mutation(self):
        with mock.patch.object(switch.shutil,'which',return_value=None), \
                mock.patch.object(switch.subprocess,'run') as launch:
            with self.assertRaisesRegex(SafetyError,'runuser is unavailable'):
                switch.switch_profile(switch.SC117_V0146_PROFILE)
        launch.assert_not_called()
        self.execute.assert_not_called()
        self.assertEqual(self.service,'active')
        self.assertTrue(self.miner)
        self.assertFalse(self.stop.exists())
        self.assertFalse(self.journal.exists())
        self.assertEqual(os.readlink(self.current),'profiles/orca-q4_k_s-v01441')

    def test_v0146_failed_preflight_refuses_before_any_mutation(self):
        failed=subprocess.CompletedProcess([],1,stdout='',stderr='readiness failed')
        with mock.patch.object(switch.shutil,'which',return_value='/usr/sbin/runuser'), \
                mock.patch.object(switch.subprocess,'run',return_value=failed):
            with self.assertRaisesRegex(SafetyError,'before any service/miner pause'):
                switch.switch_profile(switch.SC117_V0146_PROFILE)
        self.execute.assert_not_called()
        self.assertEqual(self.service,'active')
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


class PreflightChecks(unittest.TestCase):
    def test_sbin_runuser_is_resolved_from_trusted_paths_not_inherited_path(self):
        ready=subprocess.CompletedProcess([],0,stdout=switch.SC117_V0146_READINESS+'\n',stderr='')
        with mock.patch.dict(os.environ,{'PATH':'/tmp/untrusted-bin'}), \
                mock.patch.object(switch.shutil,'which',return_value='/usr/sbin/runuser') as lookup, \
                mock.patch.object(switch.subprocess,'run',return_value=ready) as launch:
            switch.preflight_profile(switch.SC117_V0146_PROFILE)
        lookup.assert_called_once_with('runuser',path='/usr/sbin:/usr/bin:/sbin:/bin')
        command=launch.call_args.args[0]
        self.assertEqual(command[:6],['/usr/sbin/runuser','-u','user','--','/usr/bin/env','-i'])
        self.assertIn('STRATA_RUNTIME=0.1.46',command)
        self.assertIn('STRATA_HOST=0.0.0.0',command)
        self.assertIn('STRATA_PORT=8080',command)
        self.assertIn('STRATA_SC117_IQ3S_TRIAL=1',command)
        self.assertFalse(any(arg.startswith('STRATA_API_KEY=') for arg in command))
        self.assertEqual(launch.call_args.kwargs,{'capture_output':True,'text':True,'timeout':600})

    def test_legacy_system_runuser_location_is_also_supported(self):
        ready=subprocess.CompletedProcess([],0,stdout=switch.SC117_V0146_READINESS,stderr='')
        with mock.patch.object(switch.shutil,'which',return_value='/bin/runuser'), \
                mock.patch.object(switch.subprocess,'run',return_value=ready) as launch:
            switch.preflight_profile(switch.SC117_V0146_PROFILE)
        self.assertEqual(launch.call_args.args[0][0],'/bin/runuser')

    def test_exact_preflight_line_remains_required(self):
        for output in ('prefix '+switch.SC117_V0146_READINESS,
                       switch.SC117_V0146_READINESS+'\n'+switch.SC117_V0146_READINESS,
                       switch.SC117_V0146_READINESS+'\nREADY CONFIG: contradictory'):
            ready=subprocess.CompletedProcess([],0,stdout=output,stderr='')
            with self.subTest(output=output), \
                    mock.patch.object(switch.shutil,'which',return_value='/usr/sbin/runuser'), \
                    mock.patch.object(switch.subprocess,'run',return_value=ready):
                with self.assertRaisesRegex(SafetyError,'before any service/miner pause'):
                    switch.preflight_profile(switch.SC117_V0146_PROFILE)

    def test_other_profiles_do_not_invoke_sc117_046_preflight(self):
        with mock.patch.object(switch.shutil,'which') as lookup, \
                mock.patch.object(switch.subprocess,'run') as launch:
            switch.preflight_profile('orca-q4_k_s-v01441')
        lookup.assert_not_called()
        launch.assert_not_called()

    def test_resolved_runuser_subprocess_is_really_executed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);stub=root/'runuser';trace=root/'argv'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > '+str(trace)+'\nprintf "%s\\n" '+
                            shlex.quote(switch.SC117_V0146_READINESS)+'\n')
            stub.chmod(0o755)
            with mock.patch.object(switch.shutil,'which',return_value=str(stub)):
                switch.preflight_profile(switch.SC117_V0146_PROFILE)
            args=trace.read_text().splitlines()
            self.assertEqual(args[:5],['-u','user','--','/usr/bin/env','-i'])
            self.assertEqual(args[-6:],['--runtime','0.1.46','--parallel','1','--batch-groups','1'])


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
