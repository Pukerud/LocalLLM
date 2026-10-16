"""Mocked host boundaries: no real miner/systemctl/Docker changes by these tests."""
import fcntl
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux hosting controller')
import hosting_lifecycle as h
from engine_safety import SafetyError


class HostingLease(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []
        self.service = 'active'
        self.jobs = [{'pid': 123, 'exe': '/hive/miners/custom/wildrig-multi-hiveos/wildrig-multi'}]
        self.patchers = [mock.patch.object(h, 'ROOT', self.root), mock.patch.object(h, 'STOP', self.root / 'STOP'),
                         mock.patch.object(h, 'RUN', self.root / 'RUN'), mock.patch.object(h, 'docker_empty'),
                         mock.patch.object(h, 'active_engine', return_value='none'),
                         mock.patch.object(h, 'gpu_jobs', side_effect=lambda: list(self.jobs)),
                         mock.patch.object(h, 'only_hive_miners', side_effect=lambda jobs: all('wildrig' in p['exe'] for p in jobs)),
                         mock.patch.object(h, 'service_state', side_effect=lambda: self.service),
                         mock.patch.object(h, 'action', side_effect=self.action),
                         mock.patch.object(h, 'wait_idle', side_effect=self.idle),
                         mock.patch.object(h, 'active_api_connections', return_value=''),
                         mock.patch.object(h.socket, 'socket')]
        self.mocks = [p.start() for p in self.patchers]
        (self.root / 'RUN').write_text('1')

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        self.temp.cleanup()

    def idle(self, *args):
        if self.jobs:
            raise SafetyError('GPUs busy')

    def action(self, args):
        self.calls.append(args)
        if args[:2] == ['systemctl', 'stop']:
            self.service = 'inactive'
        elif args[:2] == ['systemctl', 'start']:
            self.service = 'active'
        elif args == [h.MINER, 'stop']:
            self.jobs = []
            h.RUN.unlink(missing_ok=True)
            h.STOP.unlink(missing_ok=True)  # real Hive consumes this marker
        elif args == [h.MINER, 'start']:
            self.jobs = [{'pid': 124, 'exe': '/hive/miners/custom/wildrig-multi-hiveos/wildrig-multi'}]
            h.RUN.write_text('1')
            h.STOP.unlink(missing_ok=True)
        return 0

    def test_lease_path_reports_the_resolved_state_without_creating_files(self):
        with mock.patch.object(h, 'ROOT', self.root), \
             mock.patch.object(sys, 'argv', ['hosting_lifecycle.py', 'lease-path']), \
             mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(h.main(), 0)
        self.assertEqual(output.getvalue().strip(), str(self.root / 'pause.json'))
        self.assertFalse((self.root / 'lifecycle.lock').exists())

    def test_verify_lease_checks_exact_held_identity_under_shared_reservation(self):
        token = 'a' * 64
        h.write_lease({'phase': 'held', 'port': 8080, 'lease_id': token})
        import sys
        with h.launch_reservation() as lock, \
             mock.patch.dict(os.environ, {'HOSTLLM_LAUNCH_LOCK_FD': str(lock.fileno()),
                                          'HOSTLLM_LAUNCH_LEASE_ID': token}), \
             mock.patch.object(sys, 'argv', ['hosting_lifecycle.py', 'verify-lease', '--lease-id', token]):
            self.assertEqual(h.main(), 0)
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(sys, 'argv', ['hosting_lifecycle.py', 'verify-lease', '--lease-id', 'b' * 64]):
            self.assertEqual(h.main(), 1)

    def test_inherited_reservation_release_requires_hostllm_lock_inode(self):
        lock_path = self.root / 'lifecycle.lock'
        lock_path.touch()
        fd = os.open(lock_path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with mock.patch.dict(os.environ, {'HOSTLLM_LAUNCH_LOCK_FD': str(fd)}), \
                 mock.patch.object(h, 'ROOT', self.root):
                self.assertTrue(h.release_inherited_launch_reservation())
            probe = os.open(lock_path, os.O_RDONLY)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
        finally:
            os.close(fd)

    def test_competing_launch_reservations_serialize_until_identity_handoff(self):
        acquired = threading.Event()
        def competing_start():
            with h.launch_reservation():
                acquired.set()
        with h.launch_reservation():
            thread = threading.Thread(target=competing_start)
            thread.start()
            self.assertFalse(acquired.wait(.15))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set())

    def test_launch_reservation_blocks_exclusive_lifecycle_mutation(self):
        lock_path = self.root / 'lifecycle.lock'
        with h.launch_reservation():
            fd = os.open(lock_path, os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
        fd = os.open(lock_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    def test_parity_style_new_lease_refuses_existing_lease(self):
        h.write_lease({'phase': 'held', 'port': 8080, 'lease_id': 'b' * 64})
        with self.assertRaisesRegex(SafetyError, 'refusing to borrow'):
            h.begin(8080, require_new_lease=True, lease_id='a' * 64)
        self.assertFalse(self.calls)
        self.assertEqual(h.read_lease()['phase'], 'held')

    def test_begin_cli_does_not_release_preexisting_lease_after_race(self):
        import sys
        existing = {'phase': 'held', 'port': 8080, 'lease_id': 'b' * 64}
        h.write_lease(existing)
        with mock.patch.object(h.os, 'geteuid', return_value=0), \
             mock.patch.object(h, 'release') as release, \
             mock.patch.object(sys, 'argv', ['hosting_lifecycle.py', 'begin', '--require-new-lease',
                                             '--lease-id', 'a' * 64]):
            self.assertEqual(h.main(), 1)
        release.assert_not_called()
        self.assertEqual(h.read_lease(), existing)

    def test_lease_release_requires_matching_identity(self):
        token = 'a' * 64
        h.begin(8080, require_new_lease=True, lease_id=token)
        with self.assertRaisesRegex(SafetyError, 'identity changed'):
            h.release('b' * 64)
        self.assertEqual(h.read_lease()['lease_id'], token)
        h.release(token)
        self.assertIsNone(h.read_lease())

    def test_pause_then_restore_previous_state(self):
        h.begin(8080)
        self.assertEqual(self.calls[:2], [['systemctl', 'stop', 'osn.service'], [h.MINER, 'stop']])
        self.assertTrue(h.STOP.exists())
        self.assertEqual(h.read_lease()['phase'], 'held')
        h.release()
        self.assertFalse((self.root / 'pause.json').exists())
        self.assertEqual(self.service, 'active')
        self.assertTrue(h.RUN.exists())
        self.assertFalse(h.STOP.exists())

    def test_hostllm_does_not_stop_live_managed_qwen_without_an_exclusive_drain(self):
        with mock.patch.object(h, 'active_engine', return_value='qwen38'), \
             mock.patch.object(h, 'hive_qwen', return_value={'pid': 123, 'cmd': ['llama-server']}), \
             mock.patch.object(h, 'managed_qwen_port', return_value=8080):
            with self.assertRaisesRegex(SafetyError, 'cannot drain new clients'):
                h.begin(8080)
        self.assertEqual(self.calls, [])
        self.assertIsNone(h.read_lease())

    def test_active_hive_qwen_api_session_prevents_pause_before_mutation(self):
        with mock.patch.object(h, 'active_engine', return_value='qwen38'), \
             mock.patch.object(h, 'hive_qwen', return_value={'pid': 123, 'cmd': ['llama-server']}), \
             mock.patch.object(h, 'managed_qwen_port', return_value=8080), \
             mock.patch.object(h, 'active_api_connections', return_value='ESTAB 0 0 192.168.1.69:8080 192.168.1.22:50123'):
            with self.assertRaisesRegex(SafetyError, 'Active API session'):
                h.begin(8080)
        self.assertEqual(self.calls, [])
        self.assertIsNone(h.read_lease())

    def test_managed_qwen_clients_are_checked_on_recorded_port_not_menu_port(self):
        with mock.patch.object(h, 'active_engine', return_value='qwen38'), \
             mock.patch.object(h, 'hive_qwen', return_value={'pid': 123, 'cmd': ['llama-server']}), \
             mock.patch.object(h, 'managed_qwen_port', return_value=8091), \
             mock.patch.object(h, 'active_api_connections', return_value='ESTAB port 8091') as clients:
            with self.assertRaisesRegex(SafetyError, 'managed Qwen port 8091'):
                h.begin(8080)
        clients.assert_called_once_with(8091)
        self.assertEqual(self.calls, [])
        self.assertIsNone(h.read_lease())

    def test_managed_qwen_port_matches_command_and_actual_listener_owner(self):
        state_root = self.root / 'qwen-state'
        state_root.mkdir()
        (state_root / 'server.info').write_text('port=8091\n')
        identity = {'pid': 123, 'cmd': ['llama-server', '--port', '8091']}
        output = 'LISTEN 0 128 0.0.0.0:8091 0.0.0.0:* users:(("llama-server",pid=123,fd=3))'
        with mock.patch.object(h, 'qwen_state', return_value=state_root), \
             mock.patch.object(h, 'command_output', return_value=output):
            self.assertEqual(h.managed_qwen_port(identity), 8091)
        with mock.patch.object(h, 'qwen_state', return_value=state_root), \
             mock.patch.object(h, 'command_output', return_value=output.replace('pid=123', 'pid=124')):
            with self.assertRaisesRegex(SafetyError, 'listener ownership'):
                h.managed_qwen_port(identity)
        with mock.patch.object(h, 'qwen_state', return_value=state_root):
            with self.assertRaisesRegex(SafetyError, 'disagrees'):
                h.managed_qwen_port({'pid': 123, 'cmd': ['llama-server', '--port', '8080']})

    def test_manual_unmanaged_llm_does_not_get_stopped_as_a_miner(self):
        with mock.patch.object(h, 'active_engine', return_value='qwen38'), \
             mock.patch.object(h, 'hive_qwen', return_value=None):
            with self.assertRaises(SafetyError):
                h.begin(8080)
        self.assertFalse(self.calls)

    def test_rental_blocks_before_any_mutation(self):
        with mock.patch.object(h, 'docker_empty', side_effect=SafetyError('rental')):
            with self.assertRaises(SafetyError):
                h.begin(8080)
        self.assertFalse(self.calls)
        self.assertIsNone(h.read_lease())

    def test_unknown_gpu_blocks_before_any_mutation(self):
        self.jobs = [{'pid': 321, 'exe': '/opt/renter/workload'}]
        with self.assertRaises(SafetyError):
            h.begin(8080)
        self.assertFalse(self.calls)
        self.assertIsNone(h.read_lease())

    def test_busy_port_blocks_before_any_mutation(self):
        sock = self.mocks[-1].return_value.__enter__.return_value
        sock.bind.side_effect = OSError('busy')
        with self.assertRaises(SafetyError):
            h.begin(8080)
        self.assertFalse(self.calls)

    def test_pausing_lease_saved_before_service_stop(self):
        original = self.action
        def action(args):
            self.assertIsNotNone(h.read_lease())
            return original(args)
        with mock.patch.object(h, 'action', side_effect=action):
            h.begin(8080)

    def test_already_stopped_miner_rc_one_is_accepted_only_when_idle(self):
        original = self.action
        def action(args):
            rc = original(args)
            return 1 if args == [h.MINER, 'stop'] else rc
        with mock.patch.object(h, 'action', side_effect=action):
            h.begin(8080)
        self.assertEqual(h.read_lease()['phase'], 'held')
        self.assertTrue(h.STOP.exists())

    def test_live_llm_keeps_pause(self):
        h.begin(8080)
        count = len(self.calls)
        with mock.patch.object(h, 'active_engine', return_value='strata'):
            h.release()
        self.assertEqual(len(self.calls), count)
        self.assertIsNotNone(h.read_lease())
        self.assertTrue(h.STOP.exists())

    def test_unknown_job_prevents_restoration(self):
        h.begin(8080)
        count = len(self.calls)
        self.jobs = [{'pid': 321, 'exe': '/opt/renter/workload'}]
        with self.assertRaises(SafetyError):
            h.release()
        self.assertEqual(len(self.calls), count)
        self.assertTrue(h.STOP.exists())
        self.assertIsNotNone(h.read_lease())

    def test_workload_appearing_during_service_restore_keeps_miner_suppressed(self):
        h.begin(8080)
        original_action = self.action

        def action(args):
            result = original_action(args)
            if args == ['systemctl', 'start', 'osn.service']:
                self.jobs = [{'pid': 321, 'exe': '/opt/renter/workload'}]
            return result

        with mock.patch.object(h, 'action', side_effect=action):
            with self.assertRaisesRegex(SafetyError, 'GPU workload'):
                h.release()
        self.assertTrue(h.STOP.exists())
        self.assertEqual(h.read_lease()['phase'], 'restoring')
        self.assertEqual(self.service, 'active')
        self.assertNotIn([h.MINER, 'start'], self.calls)

    def test_workload_appearing_during_miner_start_retains_pause_lease(self):
        h.begin(8080)
        original_action = self.action

        def action(args):
            result = original_action(args)
            if args == [h.MINER, 'start']:
                self.jobs = [{'pid': 321, 'exe': '/opt/renter/workload'}]
            return result

        with mock.patch.object(h, 'action', side_effect=action):
            with self.assertRaisesRegex(SafetyError, 'during miner restoration'):
                h.release()
        self.assertTrue(h.STOP.exists())
        self.assertEqual(h.read_lease()['phase'], 'restoring')
        self.assertEqual(self.service, 'active')

    def test_status_distinguishes_held_from_interrupted_restoration(self):
        import contextlib
        import io
        import sys
        h.write_lease({'phase': 'restoring', 'lease_id': 'a' * 64})
        output = io.StringIO()
        with mock.patch.object(h.os, 'geteuid', return_value=0), \
             mock.patch.object(sys, 'argv', ['hosting_lifecycle.py', 'status']), \
             contextlib.redirect_stdout(output):
            self.assertEqual(h.main(), 0)
        self.assertIn('recovery (phase=restoring)', output.getvalue())

    def test_docker_uncertainty_prevents_restoration(self):
        h.begin(8080)
        count = len(self.calls)
        with mock.patch.object(h, 'docker_empty', side_effect=SafetyError('unknown')):
            with self.assertRaises(SafetyError):
                h.release()
        self.assertEqual(len(self.calls), count)
        self.assertTrue(h.STOP.exists())

    def test_inactive_services_are_not_started(self):
        self.service = 'inactive'
        self.jobs = []
        h.RUN.unlink()
        h.begin(8080)
        h.release()
        self.assertNotIn(['systemctl', 'start', 'osn.service'], self.calls)
        self.assertNotIn([h.MINER, 'start'], self.calls)

    def test_reopened_menu_can_keep_and_release_lease(self):
        h.begin(8080)
        count = len(self.calls)
        h.begin(8080)
        self.assertEqual(len(self.calls), count)
        h.release()
        self.assertIsNone(h.read_lease())

    def test_original_stop_marker_is_preserved(self):
        self.jobs = []
        h.RUN.unlink()
        h.STOP.write_text('original')
        h.begin(8080)
        h.release()
        self.assertEqual(h.STOP.read_text(), 'original')
        self.assertNotIn([h.MINER, 'start'], self.calls)


if __name__ == '__main__':
    unittest.main()
