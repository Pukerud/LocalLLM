#!/usr/bin/env python3
"""Read-only launch gates and identity-bound stopping; never manages Hive/OctaSpace."""
import argparse
import csv
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import socket
import subprocess
import time


class SafetyError(RuntimeError):
    pass


def owner_home():
    if os.geteuid() == 0:
        name = os.environ.get('QWEN38_OWNER_USER') or os.environ.get('SUDO_USER') or 'user'
        try:
            home = pwd.getpwnam(name).pw_dir
            if home != '/root':
                return Path(home)
        except KeyError:
            pass
    return Path.home()


def qwen_state():
    return Path(os.environ.get('QWEN38_STATE_ROOT', owner_home() / '.local/state/locallm-qwen38'))


def strata_state():
    return Path(os.environ.get('STRATA_STATE_ROOT', owner_home() / '.local/state/locallm-strata'))


def terminal_stat(fields):
    # CUDA context teardown may keep /proc present after exit_mm removed exe/cmdline.
    # PF_EXITING (0x4) is irrevocable exit, even before the task becomes a zombie.
    return fields[0] in ('Z', 'X', 'x') or bool(int(fields[6]) & 0x4)


def proc(pid, retries=4):
    """Bracket matching exe/cmdline snapshots with stat reads; terminal states are gone."""
    p = Path('/proc') / str(pid)
    for attempt in range(retries):
        try:
            a = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if terminal_stat(a):
                return None
            exe = os.readlink(p / 'exe')
            cmd = (p / 'cmdline').read_bytes().decode('utf-8', 'strict').rstrip('\0').split('\0')
            uid = (p / 'status').stat().st_uid
            b = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if terminal_stat(b):
                return None
            exe2 = os.readlink(p / 'exe')
            cmd2 = (p / 'cmdline').read_bytes().decode('utf-8', 'strict').rstrip('\0').split('\0')
            uid2 = (p / 'status').stat().st_uid
            c = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if terminal_stat(c):
                return None
            if a[19] == b[19] == c[19] and a[1:4] == b[1:4] == c[1:4] and \
                    exe == exe2 and cmd == cmd2 and uid == uid2 and cmd and exe:
                return {'pid': int(pid), 'start_ticks': int(c[19]), 'exe': exe,
                        'ppid': int(c[1]), 'pgid': int(c[2]), 'sid': int(c[3]), 'uid': uid, 'cmd': cmd}
        except FileNotFoundError:
            if not p.exists():
                return None
        except (PermissionError, UnicodeError) as exc:
            raise SafetyError(f'Cannot identify PID {pid}: {exc}') from exc
        except (IndexError, ValueError, OSError):
            pass
        time.sleep(0.02 * (attempt + 1))
    raise SafetyError(f'PID {pid} did not have a coherent readable identity')


def same_process(a, b):
    return bool(a and b and all(a[k] == b[k] for k in ('pid', 'start_ticks', 'exe', 'uid', 'cmd', 'pgid', 'sid')))


def signal_identity(identity, sig):
    """A pidfd, not a reusable numeric PID, is the signalling target."""
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise SafetyError('Linux pidfd support is required; numeric-kill fallback is forbidden')
    try:
        fd = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        current = proc(identity['pid'])
        if current is None:
            return
        if not same_process(identity, current):
            raise SafetyError(f"PID {identity['pid']} changed identity; refusing to signal")
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'Cannot safely inspect {args[0]}') from exc


def docker_empty():
    if not shutil.which('docker'):
        raise SafetyError('Docker inspection unavailable; starts blocked')
    try:
        text = command_output(['docker', 'ps', '--format', '{{.ID}}'])
    except SafetyError:
        if os.geteuid() == 0:
            raise
        text = command_output(['sudo', '-n', 'docker', 'ps', '--format', '{{.ID}}'])
    if text.strip():
        raise SafetyError('Docker workload/rental present; starts blocked')


def launch_gate(port=8080):
    docker_empty()
    text = command_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
    rows = [row for row in csv.reader(text.splitlines()) if row and any(x.strip() for x in row)]
    if rows:
        names = ', '.join(row[1].strip() if len(row) > 1 else 'unidentified workload' for row in rows)
        raise SafetyError(f'GPUs are busy ({names}). Manually stop the miner/workload first; nothing was stopped.')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            raise SafetyError(f'TCP port {port} is busy; starts blocked') from exc
    # Catch a rental that arrived during GPU/port inspection.
    docker_empty()


def qwen_record():
    info = qwen_state() / 'server.info'
    if not info.exists():
        return None
    fields = dict(line.split('=', 1) for line in info.read_text().splitlines() if '=' in line)
    try:
        pid = int(fields['pid'])
        ticks = int(fields['start_ticks'])
    except (KeyError, ValueError) as exc:
        raise SafetyError('Qwen state lacks start-time provenance; stop it through its owning launcher/Hive') from exc
    p = proc(pid)
    if p is None:
        return None
    if p['start_ticks'] != ticks or not p['exe'].endswith('/build/bin/llama-server'):
        raise SafetyError('Qwen PID identity does not match its state; stop refused')
    args = p['cmd']
    if '--model' not in args or args[args.index('--model') + 1] != fields.get('model'):
        raise SafetyError('Qwen model/command identity does not match its state')
    return p


def stop_qwen(from_owner=False):
    p = qwen_record()
    if not p:
        print('No identity-verified Qwen server is running.')
        return
    # Hive supervises its custom LLM; killing only its child would restart it.
    cur = Path('/run/hive/cur_miner')
    if not from_owner and cur.exists() and cur.read_text().strip() == 'custom':
        raise SafetyError('Hive custom miner may supervise Qwen. Run miner stop manually first.')
    signal_identity(p, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        now = proc(p['pid'])
        if now is None:
            break
        if not same_process(p, now):
            raise SafetyError('Qwen identity changed while stopping')
        time.sleep(0.1)
    else:
        signal_identity(p, signal.SIGKILL)
        for _ in range(50):
            if proc(p['pid']) is None:
                break
            time.sleep(0.1)
        else:
            raise SafetyError('Qwen stop could not be confirmed')
    # Do not remove a concurrent launch's state.
    info = qwen_state() / 'server.info'
    if info.exists() and f'start_ticks={p["start_ticks"]}\n' in info.read_text():
        info.unlink()
        (qwen_state() / 'server.pid').unlink(missing_ok=True)
    print('Identity-verified Qwen server stopped.')


def active_engine():
    s = strata_state() / 'server.json'
    if s.exists():
        state = json.loads(s.read_text())
        p = proc(state['identity']['pid'])
        if same_process(state['identity'], p):
            return 'strata'
    try:
        if qwen_record():
            return 'qwen38'
    except SafetyError:
        return 'unmanaged'
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if cmd and (os.path.basename(cmd[0]) in (b'llama-server', b'strata') or
                    b'serve.server' in cmd or any(x.endswith(b'/tabbyAPI/main.py') for x in cmd[:3])):
            return 'unmanaged'
    return 'none'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['active', 'preflight', 'stop-qwen'])
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--owner-stop', action='store_true', help='owning Qwen launcher/Hive teardown, still identity-bound')
    a = ap.parse_args()
    try:
        if a.action == 'active':
            print(active_engine())
        elif a.action == 'preflight':
            launch_gate(a.port)
            print('Launch allowed: Docker empty, GPUs idle, port free. Hive/OctaSpace unchanged.')
        else:
            stop_qwen(from_owner=a.owner_stop)
    except (SafetyError, PermissionError, ValueError) as exc:
        print(f'BLOCKED: {exc}', file=__import__('sys').stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
