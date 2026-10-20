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


class HiveLaunchToken(unittest.TestCase):
    def setUp(self):
        self.token = 'a' * 48
        self.identity = {'pid': 101, 'start_ticks': 10, 'exe': '/usr/bin/python3', 'uid': os.getuid(),
                         'cmd': ['/venv/bin/python', '-m', 'serve.server', '--engine', 'strata',
                                 '--config', '/state/run-config.json', '--host', '0.0.0.0',
                                 '--port', '8080'], 'ppid': 202, 'pgid': 101, 'sid': 101}
        self.server = dict(self.identity)
        self.parent = {'pid': 202, 'start_ticks': 9, 'exe': '/usr/bin/python3', 'uid': os.getuid(),
                       'pgid': 101, 'sid': 101, 'cmd': ['python3', '/home/user/LocalLLM/strata_launcher.py', '--quickstart',
                               '--profile', 'orca-q4_k_s']}

    def test_generic_stop_without_token_is_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HIVE_STRATA_LAUNCH_ID', None)
            launcher.require_hive_launch_identity({})

    def test_invalid_token_fails_closed(self):
        with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': 'wrong'}, clear=False):
            with self.assertRaisesRegex(guard.SafetyError, 'Invalid Hive'):
                launcher.require_hive_launch_identity({})

    def test_live_server_and_parent_must_carry_matching_token(self):
        with (
            mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': self.token}, clear=False),
            mock.patch.object(launcher, 'proc', side_effect=[self.server, self.parent]),
            mock.patch.object(Path, 'read_bytes', return_value=b'OTHER=1' + bytes([0])),
        ):
            with self.assertRaisesRegex(guard.SafetyError, 'token does not match'):
                launcher.require_hive_launch_identity({'identity': self.identity})

    def test_matching_live_server_and_parent_token_is_accepted(self):
        token_env = f'HIVE_STRATA_LAUNCH_ID={self.token}'.encode() + bytes([0])
        with (
            mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': self.token}, clear=False),
            mock.patch.object(launcher, 'proc', side_effect=[self.server, self.parent]),
            mock.patch.object(Path, 'read_bytes', return_value=token_env),
        ):
            launcher.require_hive_launch_identity({'identity': self.identity})

    def test_recorded_hive_stop_requires_the_same_token_and_parent_identity(self):
        import hashlib
        token_env = f'HIVE_STRATA_LAUNCH_ID={self.token}'.encode() + bytes([0])
        state = {'identity': self.identity,
                 'hive_launch_id_sha256': hashlib.sha256(self.token.encode()).hexdigest(),
                 'hive_launcher_identity': self.parent}
        with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': self.token}, clear=False), \
             mock.patch.object(launcher, 'proc', side_effect=[self.server, self.parent]), \
             mock.patch.object(Path, 'read_bytes', return_value=token_env):
            launcher.require_hive_launch_identity(state)
        wrong = dict(state, hive_launch_id_sha256='0' * 64)
        with mock.patch.dict(os.environ, {'HIVE_STRATA_LAUNCH_ID': self.token}, clear=False):
            with self.assertRaisesRegex(guard.SafetyError, 'does not match recorded'):
                launcher.require_hive_launch_identity(wrong)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HIVE_STRATA_LAUNCH_ID', None)
            with self.assertRaisesRegex(guard.SafetyError, 'token is missing'):
                launcher.require_hive_launch_identity(state)


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

    def test_qwen_command_line_api_port_is_parsed_and_ambiguous_values_refused(self):
        self.assertEqual(guard.qwen_command_port(['llama-server']), 8080)
        self.assertEqual(guard.qwen_command_port(['llama-server', '--port', '8091']), 8091)
        self.assertEqual(guard.qwen_command_port(['llama-server', '--port=8092']), 8092)
        with self.assertRaisesRegex(guard.SafetyError, 'ambiguous'):
            guard.qwen_command_port(['llama-server', '--port', '8091', '--port', '8092'])


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
                                           f'model={self.root / ("wrong.gguf" if change_model else "test.gguf")}\n'
                                           'port=8080\n')
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

    def test_family_skips_process_disappearing_during_proc_stat_read(self):
        exe = str(Path(sys.executable).resolve())
        leader = {'pid': 41001, 'start_ticks': 123, 'exe': exe, 'ppid': 1,
                  'pgid': 41001, 'sid': 41001, 'uid': os.getuid(), 'cmd': [exe]}
        state = {'identity': leader, 'allowed_executables': [exe], 'children': []}
        paths = [Path('/proc/41001'), Path('/proc/41002')]

        def read_stat(path, *args, **kwargs):
            if path == Path('/proc/41001/stat'):
                return '41001 (python) S 1 41001 41001 0'
            if path == Path('/proc/41002/stat'):
                raise ProcessLookupError(3, 'No such process')
            raise AssertionError(f'unexpected proc read: {path}')

        with mock.patch.object(launcher, 'proc', side_effect=lambda pid: leader if pid == 41001 else None), \
             mock.patch.object(Path, 'iterdir', return_value=iter(paths)), \
             mock.patch.object(Path, 'read_text', new=read_stat):
            self.assertEqual(launcher.family(state), [leader])

    def test_runtime_gate_does_not_claim_vanished_child_gpu_pid(self):
        leader = {'pid': 41001}
        child = {'pid': 41002, 'start_ticks': 456, 'pgid': 41001, 'sid': 41001}
        state = {'identity': leader, 'children': [child]}
        with mock.patch.object(launcher, 'docker_empty'), \
             mock.patch.object(launcher, 'family', return_value=[leader]), \
             mock.patch.object(Path, 'read_text', side_effect=ProcessLookupError(3, 'No such process')), \
             mock.patch.object(launcher, 'command_output', return_value='41002'):
            with self.assertRaisesRegex(guard.SafetyError, 'yielding only Strata'):
                launcher.runtime_gate(state)

    def test_stop_removes_stale_state_without_signalling_a_pid(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(launcher, 'strata_state', return_value=Path(tmp)), \
             mock.patch.object(launcher, 'family', return_value=[]), \
             mock.patch.object(launcher, 'signal_identity') as send:
            state = {'identity': {'pid': 41001, 'start_ticks': 123, 'exe': '/usr/bin/python3',
                                  'uid': os.getuid(), 'cmd': ['python3'], 'pgid': 41001, 'sid': 41001}}
            launcher.write_state(state)
            launcher.stop()
            self.assertFalse((Path(tmp) / 'server.json').exists())
            send.assert_not_called()

    def test_foreground_cleanup_is_idempotent_after_external_stop_removed_state(self):
        identity = {'pid': 41001, 'start_ticks': 123, 'exe': '/usr/bin/python3',
                    'uid': os.getuid(), 'cmd': ['python3'], 'pgid': 41001, 'sid': 41001}
        with mock.patch.object(launcher, 'read_state', return_value=None), \
             mock.patch.object(launcher, 'proc', return_value=None), \
             mock.patch.object(launcher, 'require_hive_launch_identity') as token_check, \
             mock.patch.object(launcher, 'signal_identity') as send:
            launcher.stop(expected={'identity': identity, 'hive_launch_id_sha256': 'a' * 64})
        token_check.assert_not_called()
        send.assert_not_called()


class IsolatedGroupFamily(unittest.TestCase):
    def setUp(self):
        self.leader={'pid':41001,'start_ticks':123,'exe':str(Path(sys.executable).resolve()),
                     'uid':os.getuid(),'ppid':1,'pgid':41001,'sid':41001,'cmd':['python','-m','serve.server']}
        self.child={'pid':41002,'start_ticks':456,'exe':'/candidate/native','uid':os.getuid(),
                    'ppid':41001,'pgid':41002,'sid':41001,'cmd':['/candidate/native','--serve']}
        self.state={'identity':self.leader,'runtime_version':'0.1.46','children':[],
                    'allowed_executables':[self.leader['exe'],'/candidate/native','/candidate/vision']}

    def family(self, *, live=True):
        identities={41002:self.child}
        if live:identities[41001]=self.leader
        def read_stat(path,*args,**kwargs):
            p=identities[int(path.parent.name)]
            return f'{p["pid"]} (fixture) S {p["ppid"]} {p["pgid"]} {p["sid"]} 0'
        with mock.patch.object(launcher,'proc',side_effect=lambda pid:identities.get(pid)), \
             mock.patch.object(Path,'iterdir',return_value=iter(Path('/proc')/str(pid) for pid in identities)), \
             mock.patch.object(Path,'read_text',new=read_stat):
            return launcher.family(self.state)

    def test_v0146_direct_native_child_in_own_group_is_verified(self):
        self.assertIn(self.child,self.family())
        self.child['exe']='/candidate/vision'
        self.assertIn(self.child,self.family())

    def test_0146_group_contract_remains_when_candidate_version_advances(self):
        with mock.patch.object(launcher.runtimes,'TEST_VERSION','0.1.47'):
            self.assertIn(self.child,self.family())

    def test_other_releases_still_refuse_separate_groups(self):
        for version in ('0.1.39','0.1.41','0.1.44.1','0.1.45'):
            self.state['runtime_version']=version
            with self.assertRaisesRegex(guard.SafetyError,'unverified isolated child group'):self.family()

    def test_foreign_parent_executable_uid_group_and_python_are_refused(self):
        original=dict(self.child)
        for field,value in (('ppid',999),('exe','/unknown/job'),('uid',os.getuid()+1),
                            ('pgid',41003),('exe',self.leader['exe'])):
            self.child=dict(original,**{field:value})
            with self.assertRaisesRegex(guard.SafetyError,'process-family identity mismatch'):self.family()

    def test_unrecorded_orphan_refused_but_exact_recorded_child_can_teardown(self):
        with self.assertRaisesRegex(guard.SafetyError,'unverified isolated child group'):self.family(live=False)
        self.state['children']=[dict(self.child)]
        self.child['ppid']=1
        self.assertEqual(self.family(live=False),[self.child])
        self.child['start_ticks']+=1
        with self.assertRaisesRegex(guard.SafetyError,'unverified isolated child group'):self.family(live=False)

    def test_changed_recorded_child_refused_even_if_parent_still_matches(self):
        self.state['children']=[dict(self.child)]
        self.child['cmd']=['/candidate/native','--other-session']
        with self.assertRaisesRegex(guard.SafetyError,'unverified isolated child group'):self.family()


class ForegroundIntegration(unittest.TestCase):
    def test_cpu_http_start_request_external_stop_and_reap(self):
        self.run_case('iq3_s')

    def test_cpu_orca_profile_request_external_stop_and_reap(self):
        self.run_case('orca-iq3_xxs')

    def test_cpu_q4ks_text_only_profile_request_external_stop_and_reap(self):
        self.run_case('orca-q4_k_s')

    def test_cpu_q4ks_start_replaces_dead_hive_state_before_reply(self):
        self.run_case('orca-q4_k_s', stale_hive_state=True)

    def test_cpu_v0146_native_groups_request_external_stop_and_reap(self):
        self.run_case('orca-q4_k_s', isolated_groups=True)

    def run_case(self, profile, *, stale_hive_state=False, isolated_groups=False):
        """Fake CPU frontend exercises the real launcher lifecycle, not model inference."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            (source / '.venv/bin').mkdir(parents=True)
            (source / '.venv/bin/python').symlink_to(sys.executable)
            (source / 'serve').mkdir()
            (source / 'serve/__init__.py').write_text('')
            status_version = ('0.1.46' if isolated_groups else '0.1.45') if profile == 'orca-q4_k_s' else '0.1.39'
            server_script = '''import argparse,json\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\np=argparse.ArgumentParser();p.add_argument('--port',type=int);a,_=p.parse_known_args()\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200);self.end_headers();self.wfile.write(b'{"engine":"__ENGINE_VERSION__","loaded":true,"concurrency":{"serving":1},"status":"ok"}' if self.path == '/v1/status' else b'{"status":"ok"}')\n def do_POST(self):\n  self.rfile.read(int(self.headers.get('Content-Length','0')));self.send_response(200);self.end_headers();self.wfile.write(b'{"choices":[{"message":{"content":"ANSWER=42"}}]}')\nHTTPServer(('127.0.0.1',a.port),H).serve_forever()\n'''
            if isolated_groups:
                server_script='import os,subprocess\nchildren=[subprocess.Popen(["/bin/sleep","30"],preexec_fn=os.setpgrp) for _ in range(2)]\n'+server_script
            (source / 'serve/server.py').write_text(server_script.replace('__ENGINE_VERSION__', status_version))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            state_root = root / 'state'
            if stale_hive_state:
                state_root.mkdir()
                dead_pid = int(Path('/proc/sys/kernel/pid_max').read_text()) + 1
                (state_root / 'server.json').write_text(json.dumps({
                    'identity': {'pid': dead_pid, 'start_ticks': 1, 'exe': sys.executable,
                                 'uid': os.getuid(), 'cmd': ['python3'], 'pgid': dead_pid,
                                 'sid': dead_pid, 'ppid': dead_pid},
                    'allowed_executables': [], 'children': [], 'profile': profile,
                    'hive_launch_id_sha256': '0' * 64,
                }))
            cfg = {'args': ['--max-context', '32768'] if profile == 'orca-q4_k_s' else [],
                   'exe': '/bin/sleep', 'host': '127.0.0.1', 'port': port}
            if profile != 'orca-q4_k_s':
                cfg['vision'] = {'exe': '/bin/sleep'}
            driver = root / 'driver.py'
            host_state_root = root / 'hostllm-state'
            host_state_root.mkdir()
            release_probe = root / 'release_probe.py'
            release_probe.write_text(f'''import fcntl,json,os,sys
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from engine_safety import proc,same_process
lock_fd=os.open(sys.argv[1],os.O_RDONLY)
try:
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    Path(sys.argv[3]).write_text('contended')
else:
    raise SystemExit('release probe acquired the lock before startup handoff')
fcntl.flock(lock_fd,fcntl.LOCK_EX)
state=json.loads(Path(sys.argv[2]).read_text())
if not same_process(state['identity'],proc(state['identity']['pid'])):
    raise SystemExit('launcher released before durable identity verification')
Path(sys.argv[4]).write_text('identity-recorded')
os.close(lock_fd)
''')
            driver.write_text(f'''import contextlib,pathlib,subprocess,sys,time
sys.path.insert(0,{str(ROOT)!r})
import strata_launcher as s
s.ready=lambda *args: (pathlib.Path({str(source)!r}),pathlib.Path('unused'),{cfg!r})
s.ready_q4ks=lambda root: (pathlib.Path({str(source)!r}),pathlib.Path('unused'),{cfg!r})
original_ready_runtime=s.ready_runtime
def fixture_ready_runtime(profile='iq3_s',runtime='auto',parallel=1,batch_groups=1):
    if profile=='orca-q4_k_s':
        return pathlib.Path({str(source)!r}),pathlib.Path('unused'),{cfg!r},{status_version!r}
    return original_ready_runtime(profile,runtime,parallel,batch_groups)
s.ready_runtime=fixture_ready_runtime
# The fake-runner test isolates lifecycle handoff; candidate authorization has dedicated tests.
s.require_q4ks_candidate_authorization=lambda *args,**kwargs:{{'authorization':'hostllm','lease_id':'a'*64}}
s.require_q4ks_candidate_lease=lambda *args,**kwargs:'a'*64
s.q4ks_v01441.candidate_evidence_summary=lambda root: []
s.launch_gate=lambda port: None
s.runtime_gate=lambda state: None
s.host_lifecycle.active_engine=lambda: 'none'
original=s.host_lifecycle.launch_reservation
@contextlib.contextmanager
def observed_reservation():
    with original() as lock:
        probe=subprocess.Popen([sys.executable,{str(release_probe)!r},{str(host_state_root / 'lifecycle.lock')!r},
                                {str(state_root / 'server.json')!r},{str(root / 'release-attempted')!r},
                                {str(root / 'release-observed')!r}])
        deadline=time.monotonic()+5
        while not pathlib.Path({str(root / 'release-attempted')!r}).exists():
            if probe.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('release probe did not confirm lock contention')
            time.sleep(0.01)
        try:
            yield lock
        finally:
            probe.wait(timeout=5)
            if probe.returncode:
                raise RuntimeError('concurrent release probe failed')
s.host_lifecycle.launch_reservation=observed_reservation
s.main()
''')
            env = dict(os.environ, STRATA_DATA_ROOT=str(root), STRATA_STATE_ROOT=str(state_root),
                       HOSTLLM_STATE_ROOT=str(host_state_root), STRATA_HEALTH_TIMEOUT='10')
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
                handoff_deadline = time.monotonic() + 3
                while time.monotonic() < handoff_deadline and not (root / 'release-observed').exists():
                    time.sleep(0.01)
                self.assertTrue((root / 'release-attempted').exists())
                self.assertTrue((root / 'release-observed').exists(), 'release crossed startup before identity handoff')
                req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                             data=b'{"messages":[],"max_tokens":32}',
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=3) as response:
                    data = json.load(response)
                self.assertEqual(data['choices'][0]['message']['content'], 'ANSWER=42')
                state = json.loads((state_root / 'server.json').read_text())
                effective = json.loads((state_root / 'run-config.json').read_text())
                self.assertEqual(state['profile'], profile)
                expected_model = {'orca-iq3_xxs': 'qwen3.8-flash-next-orca-iq3_xxs-strata',
                                  'orca-q4_k_s': 'qwen3.8-flash-next-orca-q4_k_s-strata'}
                self.assertEqual(effective['model_name'], expected_model.get(profile, 'qwen3.8-flash-next-iq3_s-strata'))
                if profile == 'orca-q4_k_s':
                    self.assertIs(effective['lazy_load'], False)
                    self.assertEqual(effective['idle_unload_s'], 0)
                    self.assertNotIn('--vision', effective['args'])
                    self.assertNotIn('vision', effective)
                wrong = {'orca-iq3_xxs': 'iq3_s', 'orca-q4_k_s': 'orca-iq3_xxs'}.get(profile, 'orca-iq3_xxs')
                refused = subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop', '--profile', wrong],
                                          env=env, capture_output=True, timeout=5)
                self.assertNotEqual(refused.returncode, 0)
                self.assertTrue(launcher.health(port))
                refused_version = '0.1.41' if profile == 'orca-q4_k_s' else '0.1.39'
                refused_runtime = subprocess.run([sys.executable, str(ROOT / 'strata_launcher.py'), '--stop',
                                                   '--runtime', refused_version],env=env,capture_output=True,timeout=5)
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
