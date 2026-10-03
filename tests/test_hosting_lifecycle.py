"""Mocked host boundaries: no real miner/systemctl/Docker changes by these tests."""
import os
from pathlib import Path
import sys
import tempfile
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

    def test_known_hive_llm_can_be_paused_even_when_it_owns_api_port(self):
        (self.root / 'server.info').write_text('port=8080\n')
        self.mocks[-1].return_value.__enter__.return_value.bind.side_effect = OSError('Hive LLM owns port')
        with mock.patch.object(h, 'active_engine', return_value='qwen38'), \
             mock.patch.object(h, 'hive_qwen', return_value={'pid':123}), \
             mock.patch.object(h, 'qwen_state', return_value=self.root):
            h.begin(8080)
        self.assertEqual(h.read_lease()['phase'], 'held')
        self.assertIn([h.MINER, 'stop'], self.calls)

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
