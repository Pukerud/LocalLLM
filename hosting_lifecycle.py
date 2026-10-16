#!/usr/bin/env python3
"""HostLLM's persistent miner/OctaSpace pause lease. Never stops renters or unknown GPU jobs."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import sys
import time

from engine_safety import (SafetyError, active_engine, command_output, docker_empty, owner_home,
                           proc, qwen_command_port, qwen_record, qwen_state, same_process)

ROOT = Path(os.environ.get('HOSTLLM_STATE_ROOT', owner_home() / '.local/state/hostllm'))
STOP = Path('/run/hive/MINER_STOP')
RUN = Path('/run/hive/MINER_RUN')
MINER = '/hive/bin/miner'
SERVICE = 'osn.service'


def service_state():
    state = command_output(['systemctl', 'show', SERVICE, '--property=ActiveState', '--value']).strip()
    if state not in ('active', 'inactive', 'failed'):
        raise SafetyError(f'OctaSpace state {state!r} is not safe to change')
    return state


def action(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return result.returncode


def gpu_jobs():
    text = command_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    result = []
    for line in set(text.splitlines()):
        if not line.strip():
            continue
        if not line.strip().isdigit():
            raise SafetyError('Unidentified GPU workload; no miner/service changes permitted')
        p = proc(int(line.strip()))
        if p is None:
            raise SafetyError('GPU workload is still exiting; wait for it to clear')
        result.append(p)
    return result


def hive_qwen():
    """The configured Hive miner may itself be Qwen; prove its live installed supervisor and state root."""
    record = qwen_record()
    if not record:
        return None
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(a.endswith(b'h-run.sh') for a in args):
                continue
            wrapper = proc(int(p.name))
            if not wrapper or wrapper['uid'] != record['uid']:
                continue
            cwd = Path(os.readlink(p / 'cwd'))
            scripts = [Path(a) if a.startswith('/') else cwd / a
                       for a in wrapper['cmd'] if a.endswith('h-run.sh')]
            if not any(str(a.resolve()) == '/hive/miners/custom/llm-hosting/h-run.sh' for a in scripts):
                continue
            env = dict(x.split(b'=', 1) for x in (p / 'environ').read_bytes().split(b'\0') if b'=' in x)
            tracked = Path(env.get(b'QWEN38_STATE_ROOT', b'/home/user/.local/state/locallm-qwen38').decode())
            if (tracked == qwen_state() and same_process(wrapper, proc(wrapper['pid']))
                    and same_process(record, qwen_record())):
                return record
        except FileNotFoundError:
            continue
    return None


def only_hive_miners(jobs):
    cur = Path('/run/hive/cur_miner')
    selected = cur.read_text().strip() if cur.exists() else ''
    managed = hive_qwen()
    return bool(selected) and all(p['exe'].startswith(f'/hive/miners/{selected}/') or
                                  (managed and p['pid'] == managed['pid']) for p in jobs)


def write_lease(lease):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT / f'.pause.{os.getpid()}.{secrets.token_hex(16)}.tmp'
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(temp, flags, 0o600)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, 'w') as stream:
            json.dump(lease, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, ROOT / 'pause.json')
        directory_fd = os.open(ROOT, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def read_lease():
    path = ROOT / 'pause.json'
    return json.loads(path.read_text()) if path.exists() else None


@contextmanager
def launch_reservation():
    """Exclusively reserve model startup through durable live-identity recording."""
    inherited = inherited_launch_reservation_fd()
    if inherited is not None:
        with os.fdopen(os.dup(inherited), 'r') as lock:
            yield lock
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'lifecycle.lock'
    flags = os.O_CREAT | os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock


def inherited_launch_reservation_fd():
    raw_fd = os.environ.get('HOSTLLM_LAUNCH_LOCK_FD')
    if raw_fd is None:
        return None
    try:
        fd = int(raw_fd)
        info = os.fstat(fd)
        lock_info = (ROOT / 'lifecycle.lock').stat()
    except (ValueError, OSError) as exc:
        raise SafetyError('Inherited HostLLM launch reservation is unavailable') from exc
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (lock_info.st_dev, lock_info.st_ino):
        raise SafetyError('Inherited HostLLM launch reservation does not match its lifecycle lock')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SafetyError('Inherited HostLLM launch reservation is not exclusive') from exc
    return fd


def release_inherited_launch_reservation():
    fd = inherited_launch_reservation_fd()
    if fd is None:
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    return True


def active_api_connections(port):
    if not 1 <= port <= 65535:
        raise SafetyError('Invalid API port for active-session check')
    return command_output(['ss', '-Htn', 'state', 'established', f'( sport = :{port} )']).strip()


def managed_qwen_port(identity):
    info = qwen_state() / 'server.info'
    try:
        fields = dict(line.split('=', 1) for line in info.read_text().splitlines() if '=' in line)
        port = int(fields['port'])
    except (OSError, KeyError, ValueError) as exc:
        raise SafetyError('Managed Qwen server port cannot be verified; pause refused') from exc
    if (not 1 <= port <= 65535 or not isinstance(identity, dict) or
            qwen_command_port(identity.get('cmd', [])) != port):
        raise SafetyError('Managed Qwen server port disagrees with its live command line; pause refused')
    output = command_output(['ss', '-Hltpn', f'( sport = :{port} )'])
    listeners = set(re.findall(r'pid=(\d+)', output))
    if listeners != {str(identity.get('pid'))}:
        raise SafetyError('Managed Qwen listener ownership is unverified; pause refused')
    return port


def wait_idle(timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        docker_empty()
        if not command_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip():
            return
        time.sleep(.5)
    raise SafetyError('GPUs did not clear; no unknown workload was killed and mining was not restarted')


def begin(port, *, require_new_lease=False, lease_id=None):
    existing = read_lease()
    if existing and (require_new_lease or lease_id):
        raise SafetyError('A HostLLM pause lease already exists; refusing to borrow or restore another session\'s lease')
    if require_new_lease and not lease_id:
        raise SafetyError('A new lease ID is required for an isolated Q4 parity session')
    if lease_id and not re.fullmatch(r'[a-f0-9]{64}', lease_id):
        raise SafetyError('Invalid HostLLM lease ID')
    docker_empty()  # BEFORE touching miner/osn: every container may be a rental.
    active = active_engine()
    managed = hive_qwen() if active == 'qwen38' else None
    if active != 'none' and not managed:
        raise SafetyError('An LLM is already running; stop its owned hosting session first')
    if managed:
        actual_port = managed_qwen_port(managed)
        if active_api_connections(actual_port):
            raise SafetyError(f'Active API session on managed Qwen port {actual_port}; hosting pause refused')
        raise SafetyError('HostLLM cannot drain new clients from a live Qwen listener; stop it through its owner and retry')
    jobs = gpu_jobs()
    if jobs and not only_hive_miners(jobs):
        raise SafetyError('Unidentified/non-Hive GPU workload; nothing was stopped')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            raise SafetyError(f'Port {port} is occupied; nothing was stopped') from exc
    if read_lease():
        if service_state() == 'active':
            raise SafetyError('Existing pause lease conflicts with active OctaSpace; use [9] to recover first')
        wait_idle()
        STOP.write_text('1\n')
        print('Existing HostLLM pause retained; miner and OctaSpace remain stopped.')
        return
    lease = {'version': 1, 'phase': 'pausing', 'osn_was_active': service_state() == 'active',
             'miner_was_running': bool(jobs) or RUN.exists(), 'stop_was_present': STOP.exists(),
             'stop_value': STOP.read_text() if STOP.exists() else None, 'port': port,
             'lease_id': lease_id or secrets.token_hex(32)}
    write_lease(lease)  # Durable before the first mutation; [9] can recover after an interrupted launcher.
    docker_empty()
    if lease['osn_was_active']:
        print('Pausing OctaSpace (osn.service)...', flush=True)
        if action(['systemctl', 'stop', SERVICE]) != 0 or service_state() == 'active':
            raise SafetyError('Could not pause OctaSpace; start cancelled')
    docker_empty()
    print('Stopping Hive miner for hosting...', flush=True)
    rc = action([MINER, 'stop'])
    # Hive consumes MINER_STOP on stop/start. Reassert after stop, even when it says already stopped.
    STOP.write_text('1\n')
    wait_idle()
    if rc and RUN.exists():
        raise SafetyError('Hive miner stop failed; hosting cancelled')
    lease['phase'] = 'held'
    write_lease(lease)
    print('Hosting pause active: miner stopped, OctaSpace paused, GPUs idle.', flush=True)


def release(expected_lease_id=None):
    lease = read_lease()
    if not lease:
        if expected_lease_id:
            raise SafetyError('The expected HostLLM lease no longer exists')
        return
    if expected_lease_id and lease.get('lease_id') != expected_lease_id:
        raise SafetyError('HostLLM lease identity changed; refusing to release another session\'s pause')
    if active_engine() != 'none':
        print('Hosting remains active; miner/OctaSpace stay paused. Use [9] after stopping the LLM.')
        return
    docker_empty()
    try:
        jobs = gpu_jobs()
    except SafetyError:
        wait_idle()  # e.g. a recorded CUDA process is still in kernel teardown
        jobs = []
    if jobs:
        wait_idle()  # do not restore services while an external GPU workload remains
    docker_empty()
    lease['phase'] = 'restoring'
    write_lease(lease)
    # Keep the miner's stop marker asserted until OctaSpace is restored and GPUs are rechecked.
    # If another workload appears during restoration, leave the lease/marker for explicit recovery.
    STOP.write_text('1\n')
    if lease['osn_was_active']:
        if (service_state() != 'active' and action(['systemctl', 'start', SERVICE])) or service_state() != 'active':
            raise SafetyError('OctaSpace restoration failed; pause lease retained for [9] recovery')
    docker_empty()
    try:
        jobs = gpu_jobs()
    except SafetyError:
        try:
            wait_idle()  # e.g. a recorded CUDA process is still in kernel teardown
        except SafetyError as exc:
            raise SafetyError('GPU workload appeared during restoration; miner remains paused for [9] recovery') from exc
        jobs = []
    if jobs:
        try:
            wait_idle()  # never restore the miner while any other GPU workload is present
        except SafetyError as exc:
            raise SafetyError('GPU workload appeared during restoration; miner remains paused for [9] recovery') from exc
    docker_empty()
    if gpu_jobs():
        raise SafetyError('GPU workload appeared during restoration; miner remains paused for [9] recovery')

    # Restore only the prior state, never start a miner that was originally inactive.
    if lease['miner_was_running'] and not lease['stop_was_present']:
        docker_empty()
        if gpu_jobs():
            raise SafetyError('GPU workload appeared before miner restoration; pause lease retained')
        STOP.unlink(missing_ok=True)
        try:
            docker_empty()
            if gpu_jobs():
                STOP.write_text('1\n')
                raise SafetyError('GPU workload appeared immediately before miner restoration; pause retained')
            rc = action([MINER, 'start'])
        except BaseException:
            STOP.write_text('1\n')
            raise
        if rc and not RUN.exists():
            STOP.write_text('1\n')
            raise SafetyError('Miner restoration failed; pause lease retained for [9] recovery')
        try:
            docker_empty()
            jobs = gpu_jobs()
        except SafetyError as exc:
            STOP.write_text('1\n')
            raise SafetyError('Workload state changed during miner restoration; pause retained for [9] recovery') from exc
        if jobs and not only_hive_miners(jobs):
            STOP.write_text('1\n')
            raise SafetyError('Unidentified GPU workload appeared during miner restoration; pause retained')
    elif lease['stop_was_present']:
        STOP.write_text(lease['stop_value'])
    else:
        STOP.unlink(missing_ok=True)
    (ROOT / 'pause.json').unlink()
    print('Previous miner/OctaSpace state restored.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['begin', 'release', 'status', 'managed-qwen', 'verify-lease', 'lease-path'])
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--require-new-lease', action='store_true',
                    help='for begin only: refuse an existing pause lease instead of sharing it')
    ap.add_argument('--lease-id', help='bind begin/release to a 64-hex HostLLM lease identity')
    args = ap.parse_args()
    if args.require_new_lease and args.mode != 'begin':
        ap.error('--require-new-lease is valid only with begin')
    if args.lease_id and args.mode not in ('begin', 'release', 'verify-lease'):
        ap.error('--lease-id is valid only with begin, release, or verify-lease')
    if args.lease_id and not re.fullmatch(r'[a-f0-9]{64}', args.lease_id):
        ap.error('--lease-id must be 64 lowercase hexadecimal characters')
    if args.require_new_lease and not args.lease_id:
        ap.error('--require-new-lease requires --lease-id')
    if args.mode == 'verify-lease' and not args.lease_id:
        ap.error('verify-lease requires --lease-id')
    if args.mode == 'lease-path':
        print(ROOT / 'pause.json')
        return 0
    if os.geteuid() != 0 and args.mode != 'verify-lease':
        # Narrow, explicit sudo; no configuration/ownership changes to grant extra access.
        command = ['sudo', '-n', 'env', f'HOSTLLM_STATE_ROOT={ROOT}',
                   f'QWEN38_STATE_ROOT={os.environ.get("QWEN38_STATE_ROOT", owner_home() / ".local/state/locallm-qwen38")}',
                   f'STRATA_STATE_ROOT={os.environ.get("STRATA_STATE_ROOT", owner_home() / ".local/state/locallm-strata")}',
                   sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        return subprocess.call(command)
    try:
        if args.mode == 'managed-qwen':
            return 0 if hive_qwen() else 1
        if args.mode == 'verify-lease':
            try:
                with launch_reservation():
                    lease = read_lease()
                    if (not isinstance(lease, dict) or lease.get('phase') != 'held' or
                            lease.get('lease_id') != args.lease_id):
                        raise SafetyError('HostLLM launch lease is missing, changed, or not held')
                    if active_engine() != 'none':
                        raise SafetyError('An LLM became active before HostLLM startup acquired its reservation')
            except (SafetyError, OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f'BLOCKED: {exc}', file=sys.stderr)
                return 1
            return 0
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / 'lifecycle.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.mode == 'begin':
                lease_before = read_lease()
                try:
                    begin(args.port, require_new_lease=args.require_new_lease, lease_id=args.lease_id)
                except (SafetyError, OSError, subprocess.SubprocessError):
                    lease_after = read_lease()
                    if lease_before is None and lease_after:
                        try:
                            release(lease_after.get('lease_id'))
                        except (SafetyError, OSError, subprocess.SubprocessError):
                            print('New pause lease retained; restoration waits for safe idle GPUs/Docker.', file=sys.stderr)
                    raise
            elif args.mode == 'release':
                release(args.lease_id)
            else:
                lease = read_lease()
                if not lease:
                    print('not paused by HostLLM')
                elif lease.get('phase') == 'held':
                    print('paused for hosting')
                else:
                    print(f'HostLLM lifecycle requires recovery (phase={lease.get("phase", "unknown")})')
    except (SafetyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
