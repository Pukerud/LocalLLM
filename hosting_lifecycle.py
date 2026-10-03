#!/usr/bin/env python3
"""HostLLM's persistent miner/OctaSpace pause lease. Never stops renters or unknown GPU jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from engine_safety import (SafetyError, active_engine, command_output, docker_empty, owner_home,
                           proc, qwen_record, qwen_state, same_process)

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
    temp = ROOT / 'pause.tmp'
    temp.write_text(json.dumps(lease, indent=2) + '\n')
    os.replace(temp, ROOT / 'pause.json')


def read_lease():
    path = ROOT / 'pause.json'
    return json.loads(path.read_text()) if path.exists() else None


def wait_idle(timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        docker_empty()
        if not command_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip():
            return
        time.sleep(.5)
    raise SafetyError('GPUs did not clear; no unknown workload was killed and mining was not restarted')


def begin(port):
    docker_empty()  # BEFORE touching miner/osn: every container may be a rental.
    active = active_engine()
    managed = hive_qwen() if active == 'qwen38' else None
    if active != 'none' and not managed:
        raise SafetyError('An LLM is already running; stop its owned hosting session first')
    jobs = gpu_jobs()
    if jobs and not only_hive_miners(jobs):
        raise SafetyError('Unidentified/non-Hive GPU workload; nothing was stopped')
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as exc:
            info = qwen_state() / 'server.info'
            fields = dict(line.split('=', 1) for line in info.read_text().splitlines() if '=' in line) if managed else {}
            if not managed or fields.get('port') != str(port):
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
             'stop_value': STOP.read_text() if STOP.exists() else None, 'port': port}
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


def release():
    lease = read_lease()
    if not lease:
        return
    if active_engine() != 'none':
        print('Hosting remains active; miner/OctaSpace stay paused. Use [9] after stopping the LLM.')
        return
    docker_empty()
    try:
        jobs = gpu_jobs()
    except SafetyError:
        wait_idle()  # e.g. an owned CUDA process is still in kernel teardown
        jobs = []
    if jobs and not only_hive_miners(jobs):
        wait_idle()  # never kill an unknown job just to restore mining
    docker_empty()
    lease['phase'] = 'restoring'
    write_lease(lease)
    if lease['stop_was_present']:
        STOP.write_text(lease['stop_value'])
    else:
        STOP.unlink(missing_ok=True)
    # Restore only what was running before hosting, never start an originally inactive service/miner.
    if lease['osn_was_active']:
        if (service_state() != 'active' and action(['systemctl', 'start', SERVICE])) or service_state() != 'active':
            raise SafetyError('OctaSpace restoration failed; pause lease retained for [9] recovery')
    docker_empty()
    if lease['miner_was_running'] and not lease['stop_was_present']:
        jobs = gpu_jobs()
        if jobs and not only_hive_miners(jobs):
            raise SafetyError('New GPU workload; automatic miner restart refused')
        if not jobs:
            rc = action([MINER, 'start'])
            if rc and not RUN.exists():
                raise SafetyError('Miner restoration failed; pause lease retained for [9] recovery')
    elif lease['stop_was_present']:
        STOP.write_text(lease['stop_value'])
    (ROOT / 'pause.json').unlink()
    print('Previous miner/OctaSpace state restored.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['begin', 'release', 'status', 'managed-qwen'])
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()
    if os.geteuid() != 0:
        # Narrow, explicit sudo; no configuration/ownership changes to grant extra access.
        command = ['sudo', '-n', 'env', f'HOSTLLM_STATE_ROOT={ROOT}',
                   f'QWEN38_STATE_ROOT={os.environ.get("QWEN38_STATE_ROOT", owner_home() / ".local/state/locallm-qwen38")}',
                   f'STRATA_STATE_ROOT={os.environ.get("STRATA_STATE_ROOT", owner_home() / ".local/state/locallm-strata")}',
                   sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        return subprocess.call(command)
    try:
        if args.mode == 'managed-qwen':
            return 0 if hive_qwen() else 1
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT / 'lifecycle.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.mode == 'begin':
                try:
                    begin(args.port)
                except (SafetyError, OSError, subprocess.SubprocessError):
                    if read_lease():
                        try:
                            release()
                        except (SafetyError, OSError, subprocess.SubprocessError):
                            print('Pause lease retained; restoration waits for safe idle GPUs/Docker.', file=sys.stderr)
                    raise
            elif args.mode == 'release':
                release()
            else:
                lease = read_lease()
                print('paused for hosting' if lease else 'not paused by HostLLM')
    except (SafetyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
