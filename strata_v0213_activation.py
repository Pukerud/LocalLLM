#!/usr/bin/env python3
"""Two-phase .59 handoff: --check is read-only; --activate REQUIRES later explicit operator confirmation.
Nothing is scheduled. No saved process identity, offline receipt or environment variable grants activation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import urllib.request

from engine_safety import SafetyError, proc, same_process
import hive_profile_switch as S
import strata_launcher as L
import unsloth_v31_profile as OLD
import unsloth_v31_v0159_profile as C

STATE = Path('/home/user/.local/state/locallm-strata') / OLD.STATE_NAME / 'server.json'
GPUS = ('GPU-b4b74f60-cb72-137e-2edd-ff3a98c23141', 'GPU-aa05a399-603d-8cb7-1f8b-a452692e9c73',
        'GPU-8bcaf2d8-276f-6058-b88b-15d4a6e1e109', 'GPU-335cd5c4-a226-bd46-42d4-ef96dea816c9')
POWERS = (300.0, 300.0, 300.0, 280.0)
# Exact operator-approved profiles (2026-11-22); never set/restore power limits here.
APPROVED_POWER_PROFILES = (POWERS, (300.0, 300.0, 300.0, 300.0))


def hardware():
    """Read only. Unapproved caps or identity/fault changes block; no GPU setting is written."""
    if os.uname().release != '6.10.0-hiveos':
        raise SafetyError('Kernel changed; no switch authorized by this prepared policy')
    r = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version,power.limit,memory.total',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    rows = list(csv.reader(r.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 4:
        raise SafetyError('Original four-GPU inventory unavailable; no switch')
    powers = []
    for i, row in enumerate(rows):
        if len(row) != 4:
            raise SafetyError(f'GPU{i} inventory row is incomplete; nothing stopped')
        if row[0].strip() != GPUS[i]:
            raise SafetyError(f'GPU{i} identity/order changed; nothing stopped')
        if row[1].strip() != '595.91.07':
            raise SafetyError(f'GPU{i} driver changed: expected 595.91.07, observed {row[1].strip()}; nothing stopped')
        try:
            power, memory = float(row[2]), int(row[3])
        except ValueError as exc:
            raise SafetyError(f'GPU{i} power/VRAM telemetry is unreadable; nothing stopped') from exc
        if memory != 24576:
            raise SafetyError(f'GPU{i} VRAM changed: expected 24576 MiB, observed {memory}; nothing stopped')
        powers.append(power)
    kernel = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=15, check=True).stdout
    if re.search(r'NVRM:.*\bXid\b|AMD-Vi:.*IO_PAGE_FAULT|Node Reboot Required', kernel, re.I):
        raise SafetyError('Current-boot GPU/kernel fault: no switch, retry, reset or recovery is attempted')
    if tuple(powers) not in APPROVED_POWER_PROFILES:
        observed = '/'.join(f'{p:g}' for p in powers)
        approved = ' or '.join('/'.join(f'{p:g}' for p in profile) for profile in APPROVED_POWER_PROFILES)
        raise SafetyError(f'GPU power caps {observed} W are not approved (expected {approved} W); no power settings changed')


def current_state():
    if STATE.is_symlink():
        raise SafetyError('Current server state is symlinked')
    value = json.loads(STATE.read_text())
    if ((value.get('profile'), value.get('runtime_version'), value.get('parallel'), value.get('batch_groups')) !=
            (OLD.PROFILE, OLD.VERSION, 2, 2) or value.get('unsloth_v31') is not True or
            not same_process(value.get('identity'), proc(value.get('identity', {}).get('pid')))):
        raise SafetyError('Expected exact current .46 V3.1 server changed; no adoption/stop')
    if not L.family(value):
        raise SafetyError('Exact current .46 process family is unavailable')
    return value


def request_counts():
    # No Authorization/API key and no generation. Unknown telemetry refuses instead of guessing idle.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8080/metrics', timeout=5) as response:
        data = json.load(response)
    live = data.get('live') or {}
    count = live.get('running')
    if type(count) is not int or count < 0:
        raise SafetyError('Cannot prove active-request count; no switch')
    return count


def wait_for_natural_idle(original, timeout=300):
    deadline = time.monotonic() + timeout
    while True:
        S.docker_empty()
        S.inspect_owned_workload()
        fresh = current_state()
        if not same_process(original['identity'], fresh['identity']):
            raise SafetyError('Running server changed while waiting; replacement never adopted')
        if request_counts() == 0:
            return
        if time.monotonic() >= deadline:
            raise SafetyError('Requests remain active; no cancellation or stop performed')
        time.sleep(1)


def approval_directory(parent):
    """Public permission DATA, never secrets. Explicit mode survives a restrictive root umask."""
    created = not parent.exists()
    if created:
        parent.mkdir(mode=0o755)
    st = parent.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
        raise SafetyError('Unsafe root approval directory; nothing published')
    if created:
        parent.chmod(0o755)  # mkdir(mode) alone becomes0700 under umask077.
    elif not st.st_mode & 0o001:
        raise SafetyError('Existing approval directory is not owner-traversable; explicit administrator correction required')


def check_owner_approval():
    """Real UID1000 policy read BEFORE any stop; root-readable is not sufficient."""
    command = ['/usr/sbin/runuser', '-u', 'user', '--', '/usr/bin/env', '-i',
               'HOME=/home/user', 'USER=user', 'LOGNAME=user', 'PATH=/usr/bin:/bin',
               'PYTHONDONTWRITEBYTECODE=1', '/usr/bin/python3', '-c',
               'import sys;sys.path.insert(0,"/home/user/LocalLLM");'
               'import unsloth_v31_v0159_profile as C;C.authorize_activation()']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise SafetyError('UID1000 cannot validate exact activation permission; no stop/switch performed')


def publish_approval():
    """Called ONLY by explicitly confirmed activate(), after fresh readiness/ownership/idle checks."""
    path = C.APPROVAL
    expected = C.approval_contract()
    if path.exists() or path.is_symlink():
        C.authorize_activation()  # Existing DIFFERENT/stale policy is never silently replaced.
        check_owner_approval()
        return False
    parent = path.parent
    approval_directory(parent)
    temp = parent / ('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, 'w') as f:
            os.fchmod(f.fileno(), 0o644)
            json.dump(expected, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        # Atomic no-overwrite publication (no lost race with another operator).
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    C.authorize_activation()
    check_owner_approval()
    return True


def check():
    """No approval, state write, request, stop, selection or start."""
    S.docker_empty()
    S.inspect_owned_workload()
    if S.hosting.read_lease() is not None or S.JOURNAL.exists() or S.JOURNAL.is_symlink():
        raise SafetyError('Lease/journal present; no readiness claim')
    if os.readlink(S.CURRENT) != 'profiles/' + OLD.HIVE_PROFILE:
        raise SafetyError('Current Hive selection differs; nothing changed')
    current_state()
    hardware()
    S.preflight_profile(C.HIVE_PROFILE, offline=True)
    if not S.package_current(C.HIVE_PROFILE):
        raise SafetyError('Inactive installed bundle/selector differs; publish it without stopping the server first')
    return {'offline_ready': True, 'approval_present': C.APPROVAL.exists(),
            'gpu_inference_tested': False, 'current_server_changed': False,
            'downloads_or_builds_remaining': False, 'activation_requires_confirmation': True}


def activate(confirmation):
    # An absent/wrong confirmation refuses BEFORE reservation, readiness or any lifecycle action.
    if confirmation != C.CONFIRMATION:
        raise SafetyError('Explicit later operator approval required; no action taken')
    if os.geteuid() != 0:
        raise SafetyError('Only the root lifecycle administrator may authorize this handoff')
    with S.hosting.launch_reservation():
        check()
        original = current_state()
        wait_for_natural_idle(original)
        hardware(); S.docker_empty(); S.inspect_owned_workload()
        if not same_process(original['identity'], current_state()['identity']) or request_counts() != 0:
            raise SafetyError('Session/request changed immediately before handoff; no stop')
        publish_approval()
        S.switch_profile(C.HIVE_PROFILE)  # Existing exact token/pidfd/parent/rental/miner transaction, not a saved PID.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--activate', action='store_true')
    ap.add_argument('--confirm')
    a = ap.parse_args(argv)
    try:
        if a.check:
            if a.confirm is not None:
                raise SafetyError('--check never consumes activation confirmation')
            if os.geteuid() != 0:
                raise SafetyError('Run read-only lifecycle readiness as root; builds/check workers remain UID1000')
            with S.hosting.launch_reservation():
                print(json.dumps(check(), indent=2))
        else:
            activate(a.confirm)
        return 0
    except (SafetyError, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        print('BLOCKED:', e)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
