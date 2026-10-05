"""Linux CPU-only safety/lifecycle checks. No CUDA inference, Hive, or service actions."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform != 'linux':
    raise unittest.SkipTest('Real procfs/pidfd tests require Linux')
import engine_safety as guard
import strata_launcher as launcher


class Gates(unittest.TestCase):
    def test_docker_unknown_blocks_before_gpu(self):
        with mock.patch.object(guard, 'docker_empty', side_effect=guard.SafetyError('unknown')), \
             mock.patch.object(guard, 'command_output') as command:
            with self.assertRaises(guard.SafetyError):
                guard.launch_gate(18081)
            command.assert_not_called()

    def test_miner_blocks_without_stopping_it(self):
        with mock.patch.object(guard, 'docker_empty'), \
             mock.patch.object(guard, 'command_output', return_value='123, ./wildrig-multi\n'), \
             mock.patch.object(subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(guard.SafetyError, 'Manually stop'):
                guard.launch_gate(18081)
            spawn.assert_not_called()

    def test_unidentified_gpu_process_blocks(self):
        with mock.patch.object(guard, 'docker_empty'), mock.patch.object(guard, 'command_output', return_value='N/A\n'):
            with self.assertRaises(guard.SafetyError):
                guard.launch_gate(18081)

    def test_port_time_wait_is_not_an_active_listener(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0)); port = server.getsockname()[1]
            server.listen()
            client.connect(('127.0.0.1', port))
            conn, _ = server.accept()
            conn.shutdown(socket.SHUT_WR); conn.close()
            self.assertEqual(client.recv(1), b'')
        with mock.patch.object(guard, 'docker_empty'), mock.patch.object(guard, 'command_output', return_value=''):
            guard.launch_gate(port)

    def test_gpu_query_failure_blocks(self):
        with mock.patch.object(guard, 'docker_empty'), \
             mock.patch.object(guard, 'command_output', side_effect=guard.SafetyError('query failed')):
            with self.assertRaises(guard.SafetyError):
                guard.launch_gate(18081)

    def test_busy_port_blocks(self):
        with socket.socket() as sock:
            sock.bind(('0.0.0.0', 0))
            port = sock.getsockname()[1]
            with mock.patch.object(guard, 'docker_empty'), mock.patch.object(guard, 'command_output', return_value=''):
                with self.assertRaisesRegex(guard.SafetyError, 'port'):
                    guard.launch_gate(port)

    def test_rental_race_is_rechecked(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with mock.patch.object(guard, 'docker_empty', side_effect=[None, guard.SafetyError('rental')]) as docker, \
             mock.patch.object(guard, 'command_output', return_value=''):
            with self.assertRaisesRegex(guard.SafetyError, 'rental'):
                guard.launch_gate(port)
            self.assertEqual(docker.call_count, 2)

    def test_docker_running_blocks(self):
        with mock.patch.object(guard.shutil, 'which', return_value='/usr/bin/docker'), \
             mock.patch.object(guard, 'command_output', return_value='container-id\n'):
            with self.assertRaises(guard.SafetyError):
                guard.docker_empty()


class ProcessIdentity(unittest.TestCase):
    def test_own_proc_is_coherent(self):
        p = guard.proc(os.getpid())
        self.assertEqual(p['pid'], os.getpid())
        self.assertGreater(p['start_ticks'], 0)
        self.assertTrue(Path(p['exe']).is_absolute())

    def test_cuda_exit_mm_before_zombie_is_terminal(self):
        fields = ['S', '1', '1', '1', '0', '0', str(0x4)] + ['0'] * 20
        self.assertTrue(guard.terminal_stat(fields))
        fields[6] = str(0x400000)
        self.assertFalse(guard.terminal_stat(fields))

    def test_zombie_is_terminal(self):
        child = subprocess.Popen(['/bin/true'])
        try:
            time.sleep(0.1)
            self.assertIsNone(guard.proc(child.pid))
        finally:
            child.wait()

    def test_pidfd_stop_of_owned_child(self):
        for _ in range(10):
            child = subprocess.Popen(['/bin/sleep', '30'])
            try:
                p = guard.proc(child.pid)
                guard.signal_identity(p, signal.SIGTERM)
                self.assertEqual(child.wait(timeout=3), -signal.SIGTERM)
                self.assertIsNone(guard.proc(child.pid))
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()

    def test_wrong_start_time_is_not_signalled(self):
        child = subprocess.Popen(['/bin/sleep', '30'])
        try:
            p = guard.proc(child.pid)
            p['start_ticks'] += 1
            with self.assertRaisesRegex(guard.SafetyError, 'identity'):
                guard.signal_identity(p, signal.SIGTERM)
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait()

    def test_missing_pidfd_has_no_numeric_fallback(self):
        with mock.patch.object(guard.os, 'pidfd_open', create=True) as fd, \
             mock.patch.object(guard, 'same_process', return_value=False), \
             mock.patch.object(guard.os, 'kill') as numeric:
            fd.return_value = os.open('/dev/null', os.O_RDONLY)
            with self.assertRaises(guard.SafetyError):
                guard.signal_identity(guard.proc(os.getpid()), signal.SIGTERM)
            numeric.assert_not_called()

    def test_qwen_state_without_provenance_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(guard, 'qwen_state', return_value=Path(tmp)):
            (Path(tmp) / 'server.info').write_text('pid=1\nmodel=test.gguf\n')
            with self.assertRaisesRegex(guard.SafetyError, 'provenance'):
                guard.qwen_record()


class QwenOwnedStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.exe = cls.root / 'build/bin/llama-server'
        cls.exe.parent.mkdir(parents=True)
        src = cls.root / 'fake.c'
        src.write_text('#include <unistd.h>\nint main(void) { sleep(30); return 0; }\n')
        subprocess.run(['gcc', str(src), '-o', str(cls.exe)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, change_ticks=False, change_model=False):
        child = subprocess.Popen([str(self.exe), '--model', str(self.root / 'test.gguf')])
        state = self.root / 'state'
        state.mkdir(exist_ok=True)
        p = guard.proc(child.pid)
        (state / 'server.info').write_text(f'pid={child.pid}\nstart_ticks={p["start_ticks"] + int(change_ticks)}\n'
                                           f'model={self.root / ("wrong.gguf" if change_model else "test.gguf")}\n')
        (state / 'server.pid').write_text(str(child.pid))
        try:
            with mock.patch.object(guard, 'qwen_state', return_value=state):
                if change_ticks or change_model:
                    with self.assertRaises(guard.SafetyError):
                        guard.stop_qwen(from_owner=True)
                    self.assertIsNone(child.poll())
                else:
                    guard.stop_qwen(from_owner=True)
                    self.assertEqual(child.wait(timeout=3), -signal.SIGTERM)
                    self.assertFalse((state / 'server.info').exists())
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait()

    def test_real_owned_qwen_pidfd_teardown(self):
        self.run_case()

    def test_wrong_qwen_start_time_does_not_stop(self):
        self.run_case(change_ticks=True)

    def test_wrong_qwen_model_does_not_stop(self):
        self.run_case(change_model=True)


class StrataState(unittest.TestCase):
    def test_private_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, 'strata_state', return_value=Path(tmp)):
            state = {'identity': guard.proc(os.getpid()), 'port': 18081}
            launcher.write_state(state)
            self.assertEqual(launcher.read_state(), state)
            self.assertFalse(list(Path(tmp).glob('*.tmp')))

    def test_runtime_gate_allows_only_owned_gpu_pids(self):
        with mock.patch.object(launcher, 'docker_empty'), \
             mock.patch.object(launcher, 'family', return_value=[{'pid': 123}]), \
             mock.patch.object(launcher, 'command_output', return_value='123\n123\n'):
            launcher.runtime_gate({})

    def test_runtime_gate_yields_to_new_gpu_workload(self):
        with mock.patch.object(launcher, 'docker_empty'), \
             mock.patch.object(launcher, 'family', return_value=[{'pid': 123}]), \
             mock.patch.object(launcher, 'command_output', return_value='123\n456\n'), \
             mock.patch.object(launcher, 'signal_identity') as send:
            with self.assertRaisesRegex(guard.SafetyError, 'yielding only Strata'):
                launcher.runtime_gate({})
            send.assert_not_called()

    def test_runtime_docker_uncertainty_fails_closed(self):
        with mock.patch.object(launcher, 'docker_empty', side_effect=guard.SafetyError('unknown')), \
             mock.patch.object(launcher, 'command_output') as command:
            with self.assertRaises(guard.SafetyError):
                launcher.runtime_gate({})
            command.assert_not_called()

    def test_guard_does_not_query_gpu_after_external_stop(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, 'strata_state', return_value=Path(tmp)), \
             mock.patch.object(launcher, 'read_state', return_value=None), \
             mock.patch.object(launcher, 'runtime_gate') as gate:
            self.assertFalse(launcher.guard_during_run({'identity': guard.proc(os.getpid())}))
            gate.assert_not_called()

    def test_guard_does_not_query_gpu_during_requested_stop(self):
        state = {'identity': guard.proc(os.getpid()), 'stop_requested': True}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, 'strata_state', return_value=Path(tmp)), \
             mock.patch.object(launcher, 'read_state', return_value=state), \
             mock.patch.object(launcher, 'runtime_gate') as gate:
            self.assertFalse(launcher.guard_during_run(state))
            gate.assert_not_called()

    def test_stale_foreground_does_not_recreate_state(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(launcher, 'strata_state', return_value=Path(tmp)):
            state = {'identity': guard.proc(os.getpid())}
            self.assertFalse(launcher.remember_children(state))
            self.assertIsNone(launcher.read_state())

    def test_unidentified_child_blocks_before_any_signal(self):
        leader = guard.proc(os.getpid())
        state = {'identity': leader, 'allowed_executables': []}
        with mock.patch.object(launcher, 'read_state', return_value=state), \
             mock.patch.object(launcher, 'signal_identity') as send:
            with self.assertRaises(guard.SafetyError):
                launcher.stop()
            send.assert_not_called()

    def test_changed_leader_blocks(self):
        leader = guard.proc(os.getpid())
        leader['start_ticks'] += 1
        with self.assertRaises(guard.SafetyError):
            launcher.family({'identity': leader, 'allowed_executables': [leader['exe']]})

    def test_orphan_family_requires_recorded_child_identity(self):
        child = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True)
        try:
            identity = guard.proc(child.pid)
            old = dict(identity, pid=999999999)
            with self.assertRaisesRegex(guard.SafetyError, 'recorded'):
                launcher.family({'identity': old, 'allowed_executables': [identity['exe']]})
        finally:
            child.terminate()
            child.wait()


class ForegroundIntegration(unittest.TestCase):
    def test_cpu_http_start_request_external_stop_and_reap(self):
        self.run_case('iq3_s')

    def test_cpu_orca_profile_request_external_stop_and_reap(self):
        self.run_case('orca-iq3_xxs')

    def run_case(self, profile):
        """Fake CPU frontend exercises the real launcher lifecycle, not model inference."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            (source / '.venv/bin').mkdir(parents=True)
            (source / '.venv/bin/python').symlink_to(sys.executable)
            (source / 'serve').mkdir()
            (source / 'serve/__init__.py').write_text('')
            (source / 'serve/server.py').write_text('''import argparse,json\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\np=argparse.ArgumentParser();p.add_argument('--port',type=int);a,_=p.parse_known_args()\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')\n def do_POST(self):\n  self.rfile.read(int(self.headers.get('Content-Length','0')));self.send_response(200);self.end_headers();self.wfile.write(b'{"choices":[{"message":{"content":"ANSWER=42"}}]}')\nHTTPServer(('127.0.0.1',a.port),H).serve_forever()\n''')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            state_root = root / 'state'
            cfg = {'args': [], 'exe': '/bin/sleep', 'vision': {'exe': '/bin/sleep'}, 'host': '127.0.0.1', 'port': port}
            driver = root / 'driver.py'
            driver.write_text(f'''import sys,pathlib,fcntl\nsys.path.insert(0,{str(ROOT)!r})\nimport strata_launcher as s\ns.ready=lambda *args: (pathlib.Path({str(source)!r}),pathlib.Path('unused'),{cfg!r})\ns.launch_gate=lambda port: None\ns.runtime_gate=lambda state: None\ns.main()\n''')
            env = dict(os.environ, STRATA_DATA_ROOT=str(root), STRATA_STATE_ROOT=str(state_root), STRATA_HEALTH_TIMEOUT='10')
            with (root / 'driver.log').open('w') as log:
                driver_proc = subprocess.Popen([sys.executable, str(driver), '--quickstart', '--profile', profile],
                                               env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    if launcher.health(port):
                        break
                    if driver_proc.poll() is not None:
                        self.fail((root / 'driver.log').read_text())
                    time.sleep(0.1)
                else:
                    self.fail('CPU fixture health timeout')
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             data=b'{"messages":[],"max_tokens":32}',
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=3) as response:
                    data = json.load(response)
                self.assertEqual(data['choices'][0]['message']['content'], 'ANSWER=42')
                state = json.loads((state_root / 'server.json').read_text())
                effective = json.loads((state_root / 'run-config.json').read_text())
                self.assertEqual(state['profile'], profile)
                self.assertEqual(effective['model_name'], 'qwen3.8-flash-next-orca-iq3_xxs-strata' if profile ==
                                 'orca-iq3_xxs' else 'qwen3.8-flash-next-iq3_s-strata')
                wrong = 'iq3_s' if profile == 'orca-iq3_xxs' else 'orca-iq3_xxs'
                refused = subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop', '--profile', wrong],
                                          env=env, capture_output=True, timeout=5)
                self.assertNotEqual(refused.returncode, 0)
                self.assertTrue(launcher.health(port))
                refused_runtime = subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop',
                                                   '--runtime', '0.1.39'],env=env,capture_output=True,timeout=5)
                self.assertNotEqual(refused_runtime.returncode,0)
                self.assertTrue(launcher.health(port))
                subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop'], env=env,
                               capture_output=True, text=True, check=True, timeout=25)
                self.assertEqual(driver_proc.wait(timeout=12), 0, (root / 'driver.log').read_text())
                self.assertFalse((state_root / 'server.json').exists())
                self.assertFalse(launcher.health(port))
                self.assertEqual((state_root / 'run-config.json').stat().st_mode & 0o777, 0o600)
            finally:
                if driver_proc.poll() is None:
                    driver_proc.terminate()
                    try:
                        driver_proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        driver_proc.kill()
                        driver_proc.wait()
                if (state_root / 'server.json').exists():
                    subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop'], env=env, timeout=25)


if __name__ == '__main__':
    unittest.main()
