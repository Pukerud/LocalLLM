#!/usr/bin/env python3
"""Serialized Hive LLM profile switching. Never stops rentals or manual/unowned servers."""
import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time

import hosting_lifecycle as hosting
from engine_safety import SafetyError, command_output, docker_empty, proc, same_process

SCRIPT_ROOT = Path(__file__).resolve().parent
WRAPPER = Path('/hive/miners/custom/llm-hosting/h-run.sh')
CURRENT = WRAPPER.parent / 'current'
CUR_MINER = Path('/run/hive/cur_miner')
JOURNAL = hosting.ROOT / 'hive-profile-switch.json'
PROFILES = ('swift15u-q8', 'hauhau-q8-fastmtp-q4kv-xhigh', 'orca-q4_k_s-v01441',
            'iq3_s-sc117-abliterated', 'iq3_s-sc117-abliterated-v0146',
            'orca-q4_k_s', 'orca-q4_k_s-v0141',
            'orca-q4_k_s-v0141-2slot')
DEEPSEEK_PROFILE = 'deepseek41-abiray-iq2_xxs'
PROFILES = (*PROFILES, DEEPSEEK_PROFILE)
SC117_V0146_PROFILE = 'iq3_s-sc117-abliterated-v0146'
SC117_V0146_READINESS = (
    'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.46; trusted-LAN=0.0.0.0:8080 (unauthenticated); '
    'native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; '
    'one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; '
    'conversation cache=8192 MiB / 4 slots; pipeline-windows=2; new SM86 release defaults enabled. This is not qualification.'
)


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
        if not (names & {b'llama-server', b'strata', b'strata-dsv41', b'strata-vision', b'strata_launcher.py', b'v1qwen38.sh'} or
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


def verified_hive_fallback_parent(identity):
    """Hive CLI starts may inherit a user-session cgroup. Prove the exact root
    parent running the installed Wildrig h-run.sh rather than trusting that group.
    Hive ships these scripts owned by its trusted operator, as it does /hive.
    """
    import pwd
    script = Path('/hive/miners/custom/wildrig-multi-hiveos/h-run.sh')
    parent = proc(identity.get('ppid', 0))
    if (not parent or parent.get('uid') != 0 or parent.get('exe') not in ('/bin/bash', '/usr/bin/bash') or
            parent.get('cmd', [None, None])[1:2] != [str(script)] or parent.get('sid') != identity.get('sid')):
        return False
    info = script.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, pwd.getpwnam('user').pw_uid) or
            info.st_mode & 0o022 or not same_process(parent, proc(parent['pid']))):
        return False
    return True


def verified_fallback_miner_jobs(jobs):
    """Only observed root Wildrig with Hive service or verified script ownership."""
    if not jobs or CUR_MINER.is_symlink() or not CUR_MINER.is_file() or CUR_MINER.read_text().strip() != 'custom':
        return []
    for identity in jobs:
        if identity.get('uid') != 0 or identity.get('exe') != '/hive/miners/custom/wildrig-multi-hiveos/wildrig-multi':
            return []
        group = (Path('/proc') / str(identity['pid']) / 'cgroup').read_text()
        if not any('/system.slice/' + unit in group for unit in ('osn.service', 'hive.service')) and not verified_hive_fallback_parent(identity):
            return []
        if not same_process(identity, proc(identity['pid'])):
            raise SafetyError('Fallback miner identity changed; no service was stopped')
    return jobs


def inspect_owned_workload(*, allow_fallback_miner=False):
    docker_empty()  # BEFORE any service/miner stop: every container may be a rental.
    if hosting.read_lease() is not None:
        raise SafetyError('A manual HostLLM hosting session is active; stop it with Ctrl+C/[9] before changing Hive')
    wrappers = owned_wrappers()
    processes = llm_processes()
    jobs = hosting.gpu_jobs()
    if wrappers and (CUR_MINER.is_symlink() or CUR_MINER.read_text().strip() != 'custom'):
        raise SafetyError('Hive miner selection is not exclusively custom; nothing was stopped')
    fallback = verified_fallback_miner_jobs(jobs) if allow_fallback_miner and not wrappers else []
    fallback_pids = {identity['pid'] for identity in fallback}
    if any(not belongs_to_wrapper(identity, wrappers) and identity['pid'] not in fallback_pids for identity in processes + jobs):
        raise SafetyError('A manual/unowned LLM or GPU workload is present; nothing was stopped')
    # Repeat the rental check after walking process identities.
    docker_empty()
    return wrappers


def teardown_identity(identity):
    # Persist no mining command arguments, API keys, or other secrets.
    return {**{key: identity[key] for key in ('pid','start_ticks','exe','uid','pgid','sid')},
            'cmd_sha256': hashlib.sha256(json.dumps(identity['cmd'],separators=(',',':')).encode()).hexdigest()}


def capture_owned_teardown(wrappers, fallback):
    known={}
    for identity in [*wrappers.values(),*llm_processes(),*hosting.gpu_jobs()]:
        if (not any(same_process(identity,p) for p in [*wrappers.values(),*fallback]) and
                not belongs_to_wrapper(identity,wrappers)):
            raise SafetyError('A new unowned process appeared before Hive stop; nothing was stopped')
        known[identity['pid']]=teardown_identity(identity)
    return list(known.values())


def wait_owned_teardown(records, timeout=90):
    """Wait, never signal: a recorded native/launcher can outlive its Hive shell.
    Only unchanged recorded identities or new children of its STILL LIVE exact
    wrapper may drain. Reused PIDs, manual launches and rentals stay refused.
    """
    known={record['pid']:record for record in records}
    deadline=time.monotonic()+timeout
    while True:
        docker_empty()
        if hosting.read_lease() is not None:raise SafetyError('Another hosting lease appeared during Hive teardown')
        wrappers=owned_wrappers()
        original={pid:p for pid,p in wrappers.items() if teardown_identity(p)==known.get(pid)}
        remaining=[*wrappers.values(),*llm_processes(),*hosting.gpu_jobs()]
        for identity in remaining:
            record=teardown_identity(identity)
            if record!=known.get(identity['pid']) and not belongs_to_wrapper(identity,original):
                raise SafetyError('A new manual/unowned process appeared during Hive teardown; it was not stopped')
            known[identity['pid']]=record
        if not remaining:return
        if time.monotonic()>=deadline:raise SafetyError('Recorded Hive teardown did not finish; no unknown process was stopped')
        time.sleep(.1)


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
    if previous.get('owned_teardown') and previous.get('stage') in ('prepared','selecting'):
        wait_owned_teardown(previous['owned_teardown'])
    wrappers = inspect_owned_workload()
    if hosting.STOP.is_symlink():
        raise SafetyError('Hive stop marker became symlinked; recovery is required')
    previous['stage']='restoring'
    write_journal(previous)
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


def flight_sheet_current(wallet=Path('/hive-config/wallet.conf'), rig=Path('/hive-config/rig.conf')):
    expected = ((rig, {'MINER':'custom'}),
                (wallet, {'CUSTOM_MINER':'llm-hosting',
                          'CUSTOM_CONFIG_FILENAME':'/hive/miners/custom/llm-hosting/llm-hosting.conf',
                          'CUSTOM_INSTALL_URL':''}))
    try:
        for path, fields in expected:
            if path.is_symlink():return False
            lines=path.read_text().splitlines()
            for key,value in fields.items():
                if [line for line in lines if line.startswith(key+'=')] != [key+'='+value]:return False
    except OSError:return False
    return True


def package_current(profile):
    if not flight_sheet_current():return False
    source = SCRIPT_ROOT / 'hive-llm-miner'
    files = [('profile-menu.sh', 'profile-menu.sh')] + [
        (f'profiles/{profile}/{name}', f'profiles/{profile}/{name}')
        for name in ('h-manifest.conf', 'h-config.sh', 'h-run.sh', 'h-stats.sh', 'llm-hosting.conf')]
    return all((source / src).is_file() and (WRAPPER.parent / dst).is_file() and
               not (WRAPPER.parent / dst).is_symlink() and
               filecmp.cmp(source / src, WRAPPER.parent / dst, shallow=False) for src, dst in files)


def preflight_profile(profile):
    """Run the new SC117 route's exact readiness check before pausing Hive/OctaSpace."""
    if profile == DEEPSEEK_PROFILE:
        from deepseek41_profile import READY
        runuser = shutil.which('runuser', path='/usr/sbin:/usr/bin:/sbin:/bin')
        if runuser is None:
            raise SafetyError('runuser is unavailable; DeepSeek readiness failed before service pause')
        command = [runuser, '-u', 'user', '--', '/usr/bin/env', '-i',
                   'HOME=/home/user', 'USER=user', 'LOGNAME=user', 'PATH=/usr/local/bin:/usr/bin:/bin',
                   'PYTHONDONTWRITEBYTECODE=1', 'STRATA_HOST=0.0.0.0', 'STRATA_PORT=8080',
                   'STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/deepseek41-abiray',
                   '/home/user/LocalLLM/v1strata.sh', '--check-ready', '--profile', profile,
                   '--runtime', '0.1.51', '--parallel', '1', '--batch-groups', '1']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SafetyError('DeepSeek readiness failed before any service/miner pause') from exc
        if result.returncode or [line for line in result.stdout.splitlines() if line.startswith('READY CONFIG:')] != [READY]:
            raise SafetyError('DeepSeek exact native-context/vision readiness failed before any service/miner pause')
        say('DeepSeek exact readiness passed before service/miner pause.')
        return
    if profile != SC117_V0146_PROFILE:
        return
    # util-linux installs runuser under sbin on HiveOS/Ubuntu. Search only
    # trusted system directories, never an inherited/user-writable PATH.
    runuser = shutil.which('runuser', path='/usr/sbin:/usr/bin:/sbin:/bin')
    if runuser is None:
        raise SafetyError('runuser is unavailable in trusted system paths; no service/miner pause was started')
    trial_state = '/home/user/.local/state/locallm-strata/sc117-iq3s-trial'
    command = [
        runuser, '-u', 'user', '--', '/usr/bin/env', '-i',
        'HOME=/home/user', 'USER=user', 'LOGNAME=user',
        'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin',
        'PYTHONDONTWRITEBYTECODE=1', f'PYTHONPYCACHEPREFIX={trial_state}/pycache',
        'SETUPTOOLS_USE_DISTUTILS=stdlib', 'STRATA_SC117_IQ3S_TRIAL=1',
        'STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata', f'STRATA_STATE_ROOT={trial_state}',
        'STRATA_RUNTIME=0.1.46', 'STRATA_PARALLEL=1', 'STRATA_BATCH_GROUPS=1',
        'STRATA_PORT=8080', 'STRATA_HOST=0.0.0.0',
        '/home/user/LocalLLM/v1strata.sh', '--check-ready', '--profile', 'iq3_s-sc117-abliterated',
        '--runtime', '0.1.46', '--parallel', '1', '--batch-groups', '1',
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'SC117 .46 readiness failed before any service/miner pause: {exc}') from exc
    lines = [line for line in result.stdout.splitlines() if line.startswith('TRIAL CONFIG:')]
    if (result.returncode or lines != [SC117_V0146_READINESS] or 'READY CONFIG:' in result.stdout):
        raise SafetyError('SC117 .46 exact unqualified readiness failed before any service/miner pause')
    say('SC117 .46 exact unqualified readiness passed; no service/miner state has changed.')


def switch_profile(profile):
    if JOURNAL.exists() or JOURNAL.is_symlink():
        raise SafetyError(f'An earlier switch needs recovery: {recovery_command()}')
    inspection = {'allow_fallback_miner': True} if profile == DEEPSEEK_PROFILE else {}
    wrappers = inspect_owned_workload(**inspection)
    fallback = verified_fallback_miner_jobs(hosting.gpu_jobs()) if inspection and not wrappers else []
    preflight_profile(profile)
    if not fallback and os.readlink(CURRENT) == f'profiles/{profile}' and package_current(profile):
        say(f'{profile} is already selected; no stop/reload is needed.')
        return
    if hosting.STOP.is_symlink():
        raise SafetyError('Hive stop marker is symlinked; nothing was stopped')
    previous = {'version': 1, 'osn_was_active': hosting.service_state() == 'active',
                'miner_was_running': bool(wrappers or fallback), 'stop_was_present': hosting.STOP.exists(),
                'stop_value': hosting.STOP.read_text() if hosting.STOP.exists() else '',
                'target': profile, 'previous_selection': os.readlink(CURRENT), 'stage': 'prepared',
                'owned_teardown': capture_owned_teardown(wrappers, fallback)}
    write_journal(previous)
    try:
        if previous['osn_was_active']:
            say('Pausing OctaSpace so it cannot restart the old miner during selection...')
            run(['systemctl', 'stop', hosting.SERVICE], timeout=60)
        remaining = inspect_owned_workload(**inspection)
        remaining_fallback = verified_fallback_miner_jobs(hosting.gpu_jobs()) if fallback else []
        if remaining_fallback and any(not any(same_process(job, original) for original in fallback) for job in remaining_fallback):
            raise SafetyError('A different fallback miner appeared; nothing else will be stopped')
        if remaining or remaining_fallback:
            say('Stopping only the verified Hive-owned workload through miner stop...')
            try:
                run([hosting.MINER, 'stop', '60'], timeout=90)
            except SafetyError:
                # OctaSpace may finish stopping its screen between our identity
                # snapshot and miner stop. Accept only a fully verified idle host.
                hosting.wait_idle(timeout=75)
                wait_owned_teardown(previous['owned_teardown'])
                if inspect_owned_workload() or llm_processes():
                    raise
                say('Hive already stopped while OctaSpace was pausing; verified idle.')
        # miner stop consumes MINER_STOP. Keep it asserted throughout slow asset
        # readiness/pointer changes; never leave a race for the auto-restarter.
        hosting.STOP.write_text('1\n')
        hosting.wait_idle(timeout=75)
        wait_owned_teardown(previous['owned_teardown'])
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
