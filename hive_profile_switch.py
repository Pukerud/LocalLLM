#!/usr/bin/env python3
"""Serialized Hive LLM profile switching. Never stops rentals or manual/unowned servers."""
import argparse
import filecmp
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import stat
import subprocess
import sys

import hosting_lifecycle as hosting
from engine_safety import SafetyError, command_output, docker_empty, proc, same_process

SCRIPT_ROOT = Path(__file__).resolve().parent
WRAPPER = Path('/hive/miners/custom/llm-hosting/h-run.sh')
CURRENT = WRAPPER.parent / 'current'
CUR_MINER = Path('/run/hive/cur_miner')
JOURNAL = hosting.ROOT / 'hive-profile-switch.json'
PROFILES = ('swift15u-q8', 'hauhau-q8-fastmtp-q4kv-xhigh', 'orca-q4_k_s-v01441',
            'iq3_s-sc117-abliterated', 'orca-q4_k_s', 'orca-q4_k_s-v0141',
            'orca-q4_k_s-v0141-2slot')


def say(message):
    print(f'[Hive switch] {message}', flush=True)


def recovery_command():
    return (f'sudo env HOSTLLM_STATE_ROOT={shlex.quote(str(hosting.ROOT))} '
            f'python3 {shlex.quote(__file__)} --recover')


def llm_processes():
    """Inspect only recognized launchers/servers, not unrelated root/renter processes."""
    result = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        names = {os.path.basename(arg) for arg in args}
        if not (names & {b'llama-server', b'strata', b'strata-vision', b'strata_launcher.py', b'v1qwen38.sh'} or
                b'serve.server' in args or b'llama_cpp.server' in args):
            continue
        identity = proc(int(directory.name))
        if identity:
            result.append(identity)
    return result


def owned_wrappers():
    info = WRAPPER.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise SafetyError('Installed Hive LLM wrapper is not a trusted root-owned file')
    wrappers = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if os.fsencode(WRAPPER) not in args:
            continue
        identity = proc(int(directory.name))
        if identity and identity['uid'] == 0 and identity['exe'] in ('/usr/bin/bash', '/bin/bash'):
            wrappers[identity['pid']] = identity
    return wrappers


def belongs_to_wrapper(identity, wrappers):
    """Require a coherent live ancestry chain; never authorize a reusable bare PID."""
    seen = set()
    current = identity
    while current and current['pid'] not in seen:
        seen.add(current['pid'])
        live = proc(current['pid'])
        if not same_process(current, live) or current['ppid'] != live['ppid']:
            return False
        if current['pid'] in wrappers:
            return same_process(wrappers[current['pid']], live)
        if current['ppid'] <= 1:
            return False
        current = proc(current['ppid'])
    return False


def inspect_owned_workload():
    docker_empty()  # BEFORE any service/miner stop: every container may be a rental.
    if hosting.read_lease() is not None:
        raise SafetyError('A manual HostLLM hosting session is active; stop it with Ctrl+C/[9] before changing Hive')
    wrappers = owned_wrappers()
    processes = llm_processes()
    jobs = hosting.gpu_jobs()
    if wrappers and (CUR_MINER.is_symlink() or CUR_MINER.read_text().strip() != 'custom'):
        raise SafetyError('Hive miner selection is not exclusively custom; nothing was stopped')
    if any(not belongs_to_wrapper(identity, wrappers) for identity in processes + jobs):
        raise SafetyError('A manual/unowned LLM or GPU workload is present; nothing was stopped')
    # Repeat the rental check after walking process identities.
    docker_empty()
    return wrappers


def run(command, timeout=180):
    result = subprocess.run(command, timeout=timeout)
    if result.returncode:
        raise SafetyError(f'{Path(command[0]).name} failed (exit {result.returncode}); see its output above')


def write_journal(value):
    hosting.ROOT.mkdir(parents=True, exist_ok=True)
    if JOURNAL.is_symlink():
        raise SafetyError('Hive profile-switch journal is symlinked')
    temporary = JOURNAL.with_name(f'.hive-switch.{os.getpid()}.{secrets.token_hex(8)}.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, JOURNAL)
    finally:
        temporary.unlink(missing_ok=True)


def read_journal():
    info = JOURNAL.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600 or
            info.st_size > 16384):
        raise SafetyError('Hive switch recovery journal is unsafe')
    value = json.loads(JOURNAL.read_text())
    if (value.get('version') != 1 or type(value.get('osn_was_active')) is not bool or
            type(value.get('miner_was_running')) is not bool or
            type(value.get('stop_was_present')) is not bool or
            not isinstance(value.get('stop_value'), str)):
        raise SafetyError('Hive switch recovery journal is malformed')
    return value


def restore(previous):
    wrappers = inspect_owned_workload()
    if hosting.STOP.is_symlink():
        raise SafetyError('Hive stop marker became symlinked; recovery is required')
    if previous['stop_was_present']:
        hosting.STOP.write_text(previous['stop_value'])
    else:
        hosting.STOP.unlink(missing_ok=True)
    # Start Hive first, while OctaSpace is still paused. New launchers wait for
    # our HostLLM lifecycle reservation until the transaction returns.
    if previous['miner_was_running'] and not wrappers:
        say('Starting the selected Hive LLM miner...')
        run([hosting.MINER, 'start'], timeout=90)
    if previous['osn_was_active'] and hosting.service_state() != 'active':
        say('Restoring OctaSpace...')
        run(['systemctl', 'start', hosting.SERVICE], timeout=60)
    JOURNAL.unlink()


def package_current(profile):
    source = SCRIPT_ROOT / 'hive-llm-miner'
    files = [('profile-menu.sh', 'profile-menu.sh')] + [
        (f'profiles/{profile}/{name}', f'profiles/{profile}/{name}')
        for name in ('h-manifest.conf', 'h-config.sh', 'h-run.sh', 'h-stats.sh', 'llm-hosting.conf')]
    return all((source / src).is_file() and (WRAPPER.parent / dst).is_file() and
               not (WRAPPER.parent / dst).is_symlink() and
               filecmp.cmp(source / src, WRAPPER.parent / dst, shallow=False) for src, dst in files)


def switch_profile(profile):
    if JOURNAL.exists() or JOURNAL.is_symlink():
        raise SafetyError(f'An earlier switch needs recovery: {recovery_command()}')
    wrappers = inspect_owned_workload()
    if os.readlink(CURRENT) == f'profiles/{profile}' and package_current(profile):
        say(f'{profile} is already selected; no stop/reload is needed.')
        return
    if hosting.STOP.is_symlink():
        raise SafetyError('Hive stop marker is symlinked; nothing was stopped')
    previous = {'version': 1, 'osn_was_active': hosting.service_state() == 'active',
                'miner_was_running': bool(wrappers), 'stop_was_present': hosting.STOP.exists(),
                'stop_value': hosting.STOP.read_text() if hosting.STOP.exists() else '',
                'target': profile, 'previous_selection': os.readlink(CURRENT), 'stage': 'prepared'}
    write_journal(previous)
    try:
        if previous['osn_was_active']:
            say('Pausing OctaSpace so it cannot restart the old miner during selection...')
            run(['systemctl', 'stop', hosting.SERVICE], timeout=60)
        remaining = inspect_owned_workload()
        if remaining:
            say('Stopping only the verified Hive LLM through miner stop...')
            try:
                run([hosting.MINER, 'stop', '60'], timeout=90)
            except SafetyError:
                # OctaSpace may finish stopping its screen between our identity
                # snapshot and miner stop. Accept only a fully verified idle host.
                hosting.wait_idle(timeout=75)
                if inspect_owned_workload() or llm_processes():
                    raise
                say('Hive already stopped while OctaSpace was pausing; verified idle.')
        # miner stop consumes MINER_STOP. Keep it asserted throughout slow asset
        # readiness/pointer changes; never leave a race for the auto-restarter.
        hosting.STOP.write_text('1\n')
        hosting.wait_idle(timeout=75)
        if owned_wrappers() or llm_processes():
            raise SafetyError('A launcher remains/appeared after Hive stop; profile unchanged')
        docker_empty()
        previous['stage'] = 'selecting'
        write_journal(previous)
        say(f'Selecting {profile}; asset checks may take a minute. The old miner stays stopped.')
        run([str(SCRIPT_ROOT / 'HostLLM.sh'), '--apply-hive-profile', profile], timeout=600)
        if os.readlink(CURRENT) != f'profiles/{profile}':
            raise SafetyError('Hive profile pointer did not match the requested model')
        previous['stage'] = 'selected'
        write_journal(previous)
    finally:
        try:
            restore(previous)
        except BaseException as error:
            say(f'Restoration waits for safe recovery: {error}. Run {recovery_command()}')
            raise
    say(f'Selected {profile}. Previous miner/OctaSpace running state restored; no manual stop race.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--profile', choices=PROFILES)
    group.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('run through HostLLM or sudo')
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('profile switch interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with hosting.launch_reservation():
            if args.recover:
                restore(read_journal())
                say('Previous miner/OctaSpace state restored.')
            else:
                switch_profile(args.profile)
    except (SafetyError, OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        say(f'ERROR: {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
