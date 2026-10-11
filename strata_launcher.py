#!/usr/bin/env python3
"""Prepared Strata model launcher. No downloads, updates, driver or service changes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from engine_safety import (SafetyError, command_output, docker_empty, launch_gate, owner_home,
                           proc, same_process, signal_identity, strata_state)
from local_api_client import open_local_api
import hosting_lifecycle as host_lifecycle
from prepare_strata import DIGESTS
from orca_assets import MODEL_ID as ORCA_MODEL_ID, PROFILE as ORCA_PROFILE
from orca_profile import configure_native_context, ready_orca
from orca_q4ks_assets import (PROFILE as ORCA_Q4KS_PROFILE, MODEL_ID as ORCA_Q4KS_MODEL_ID,
                              SOURCE_COMMIT as ORCA_Q4KS_SOURCE_COMMIT,
                              RUNTIME_VERSION as ORCA_Q4KS_RUNTIME)
from orca_q4ks_profile import (configure_native_context as configure_q4ks_native_context,
                               configure_q4ks_vision, profile_root as q4ks_profile_root,
                               ready_q4ks, record_32k_validation as record_q4ks_32k_validation,
                               rollback_q4ks_vision)
import orca_q4ks_v0141_profile as q4ks_v0141
import strata_runtime as runtimes

SOURCE_COMMIT = '99f3dbd0b21d1401b3769e0c0d963913607f380b'
MODEL_REVISION = 'ed59f92082b1e93c0e96d60a8b11aab089b52f09'
MODEL = 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S'
HIVE_CUR_MINER = Path('/run/hive/cur_miner')


def data_root():
    return Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))


def ready(profile='iq3_s'):
    root = data_root()
    if profile == ORCA_PROFILE:
        return ready_orca(root)
    if profile == ORCA_Q4KS_PROFILE:
        return ready_q4ks(root)
    if profile != 'iq3_s':
        raise SafetyError('Unknown prepared Strata profile')
    manifest = json.loads((root / 'prepared.json').read_text())
    if manifest['source_commit'] != SOURCE_COMMIT:
        raise SafetyError('Prepared Strata source pin mismatch')
    source = root / 'source-99f3dbd0b21d'
    actual = command_output(['git', '-c', f'safe.directory={source}', '-C', str(source), 'rev-parse', 'HEAD']).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Strata source checkout changed since preparation')
    runtimes = {r['path']: r for r in manifest.get('runtime_assets', [])}
    for executable in [source / 'engine/strata', source / 'engine/strata-vision']:
        record = runtimes.get(str(executable))
        if not record or not executable.is_file() or executable.stat().st_size != record['bytes'] or \
                executable.stat().st_mtime_ns != record['mtime_ns'] or \
                hashlib.sha256(executable.read_bytes()).hexdigest() != record['sha256']:
            raise SafetyError(f'Compiled runtime missing or changed: {executable}')
    config_path = source / 'strata-iq3_s.json'
    cfg = json.loads(config_path.read_text())
    args = cfg['args']
    for key, expected in [('exe', source / 'engine/strata')]:
        if Path(cfg[key]).resolve() != expected.resolve():
            raise SafetyError('Unexpected Strata engine path')
    if Path(cfg['vision']['exe']).resolve() != (source / 'engine/strata-vision').resolve():
        raise SafetyError('Unexpected vision helper path')
    model_paths = [root / relative for relative in DIGESTS]
    pack = root / 'data/packs/iq3_s'
    for key, value in [('--max-context', '262144'), ('--kv', 'int8'), ('--spec', '4'),
                       ('--vram-reserve-mib', '2048'), ('--kv-resident', '32768'), ('--spec-min-p', '0.5'),
                       ('--pack', str(pack)), ('--native', str(model_paths[0])),
                       ('--ple-gguf', str(model_paths[1])), ('--mtp', str(root / 'data/mtp/rt'))]:
        if key not in args or args[args.index(key) + 1] != value:
            raise SafetyError(f'Unexpected Strata setting: {key}')
    if cfg['vision']['model'] != str(model_paths[0]) or cfg['vision']['mmproj'] != str(model_paths[2]):
        raise SafetyError('Vision must use the verified Flash-Next model/projector pair')
    if cfg['tokenizer'] != str(pack / 'tokenizer') or not (pack / 'dense.bin').is_file() or \
            not (pack / 'tokenizer/tokenizer.json').is_file():
        raise SafetyError('Prepared model pack/tokenizer missing or miswired')
    if cfg.get('gpu') != [0, 1, 2, 3] or cfg.get('layer_split') != 'auto':
        raise SafetyError('Expected the prepared four-GPU automatic layer split')
    if '--vision' not in args or not cfg.get('vision') or not cfg['vision'].get('gpu'):
        raise SafetyError('Prepared GPU vision configuration missing')
    if any(a.startswith('--cvec') for a in args):
        raise SafetyError('Experimental speed projection must remain off')
    verified = {asset['path']: asset for asset in manifest['assets']}
    for relative, digest in DIGESTS.items():
        p = root / relative
        asset = verified.get(str(p))
        if not asset or asset['sha256'] != digest or not p.is_file() or p.stat().st_size != asset['bytes']:
            raise SafetyError(f'Missing or changed model asset: {p}')
        if 'mtime_ns' in asset and p.stat().st_mtime_ns != asset['mtime_ns']:
            raise SafetyError(f'Model asset changed since checksum verification: {p}')
    for p in [source / '.venv/bin/python', Path(cfg['exe']), Path(cfg['vision']['exe']),
              Path(args[args.index('--mtp') + 1]) / 'experts.bin']:
        if not p.is_file():
            raise SafetyError(f'Prepared runtime asset missing: {p}')
    return source, config_path, cfg


def _effective_host(profile, version, configured_host):
    requested = os.environ.get('STRATA_HOST')
    if profile == ORCA_Q4KS_PROFILE and version == '0.1.41':
        if requested not in (None, '127.0.0.1', '0.0.0.0'):
            raise SafetyError('Q4 .41 STRATA_HOST may only be 127.0.0.1 or 0.0.0.0')
        return requested or '127.0.0.1'
    return requested or configured_host


def ready_runtime(profile='iq3_s', runtime='auto', parallel=1, batch_groups=1):
    if profile == ORCA_Q4KS_PROFILE:
        if runtime not in ('auto', ORCA_Q4KS_RUNTIME, '0.1.41'):
            raise SafetyError(f'Q4_K_S supports its preserved .39 baseline or the explicit .41 candidate only')
        if runtime == '0.1.41':
            if (parallel, batch_groups) not in ((1, 1), (2, 2)):
                raise SafetyError('Q4 .41 supports only the single-slot profile or the tested 2-slot/2-group profile')
            source, path, cfg = q4ks_v0141.ready_q4ks_v0141(data_root())
            if parallel == 2:
                cfg['parallel'] = parallel
                cfg['args'] += ['--batch-groups', str(batch_groups)]
            return source, path, cfg, '0.1.41'
        if parallel != 1 or batch_groups != 1:
            raise SafetyError('The preserved Q4 .39 profile is restricted to one slot and one batch group')
        source, path, cfg = ready_q4ks(data_root())
        return source, path, cfg, ORCA_Q4KS_RUNTIME
    source, path, cfg = ready() if profile == 'iq3_s' else ready(profile)
    return runtimes.configure(data_root(), source, path, cfg, profile, runtime, parallel, batch_groups)


def _q4ks_recorded_identity_valid(value):
    return (isinstance(value, dict) and type(value.get('pid')) is int and value['pid'] > 0 and
            type(value.get('start_ticks')) is int and value['start_ticks'] > 0 and
            isinstance(value.get('exe'), str) and value['exe'].startswith('/'))


def q4ks_evidence_summary(root, config_path):
    """Summarize stored, config-hash-bound Q4 live reports without turning readiness into a quality claim."""
    pr = q4ks_profile_root(root)
    lines = []
    validation32 = pr / 'validation-32k.json'
    if not validation32.is_symlink() and validation32.is_file():
        try:
            report = json.loads(validation32.read_text())
            if not isinstance(report, dict):
                raise TypeError('report root must be an object')
            initial = pr / 'context-config-before.json'
            evidence_config = initial if initial.is_file() and not initial.is_symlink() else config_path
            checks, mtp = report.get('checks'), report.get('mtp')
            if not isinstance(checks, dict) or not isinstance(mtp, dict):
                raise TypeError('checks and MTP must be objects')
            drafted, accepted = mtp.get('drafted_tokens'), mtp.get('accepted_tokens')
            valid = (report.get('profile') == ORCA_Q4KS_PROFILE and
                     report.get('model_id') == ORCA_Q4KS_MODEL_ID and
                     report.get('source_commit') == ORCA_Q4KS_SOURCE_COMMIT and
                     report.get('runtime_version') == ORCA_Q4KS_RUNTIME and
                     report.get('config_sha256') == hashlib.sha256(evidence_config.read_bytes()).hexdigest() and
                     report.get('context') == 32768 and report.get('parallel') == 1 and
                     report.get('text_only') is True and report.get('passed') is True and
                     all(checks.get(name) is True for name in
                         ('health', 'arithmetic', 'strict_json', 'tool_call', 'memory_recorded')) and
                     type(report.get('memory_peak_mib')) is int and report['memory_peak_mib'] > 0 and
                     type(drafted) is int and drafted > 0 and type(accepted) is int and
                     0 <= accepted <= drafted and
                     _q4ks_recorded_identity_valid(report.get('recorded_server_identity')))
            if valid:
                lines.append(f'Recorded 32K text-only gate passed (MTP {accepted}/{drafted}; '
                             f'per-GPU peak {report["memory_peak_mib"]} MiB). Historical 32K evidence only.')
            else:
                lines.append('Stored 32K report is stale/incomplete and is not counted as live evidence.')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
            lines.append('Stored 32K report is unreadable and is not counted as live evidence.')
    else:
        lines.append('No recorded live 32K text-only validation; native-context migration remains gated.')

    native = pr / 'validation-native.json'
    if not native.is_symlink() and native.is_file():
        try:
            report = json.loads(native.read_text())
            if not isinstance(report, dict):
                raise TypeError('report root must be an object')
            vision, full = report.get('vision'), report.get('full_context')
            health, resources = report.get('health'), report.get('resource_measurement')
            if not all(isinstance(item, dict) for item in (vision, full, health, resources)):
                raise TypeError('vision/context/health/resources must be objects')
            marker = full.get('expected_tail_marker')
            answer = vision.get('answer')
            magenta_answer = (isinstance(answer, str) and
                              answer.strip().strip(' .,!?:;').casefold() == 'magenta')
            response_tokens = full.get('response_input_tokens')
            full_answer = full.get('answer')
            valid = (report.get('profile') == ORCA_Q4KS_PROFILE and
                     report.get('model_id') == ORCA_Q4KS_MODEL_ID and
                     report.get('source_commit') == ORCA_Q4KS_SOURCE_COMMIT and
                     report.get('runtime_version') == ORCA_Q4KS_RUNTIME and
                     report.get('config_sha256') == hashlib.sha256(config_path.read_bytes()).hexdigest() and
                     report.get('context_configured') == 262144 and
                     report.get('vision_configured') is True and report.get('vision_cpu_only') is True and
                     report.get('passed') is True and
                     _q4ks_recorded_identity_valid(report.get('live_server_identity')) and
                     health.get('status') == 'ok' and health.get('loaded') is True and
                     health.get('max_context') == 262144 and health.get('images') is True and
                     vision.get('passed') is True and vision.get('expected') == 'magenta' and
                     vision.get('cpu_only_configured') is True and magenta_answer and
                     full.get('passed') is True and full.get('count_match') is True and
                     full.get('counted_prompt_tokens') == response_tokens and
                     full.get('marker_retrieved') is True and isinstance(marker, str) and marker and
                     isinstance(full_answer, str) and marker in full_answer and
                     type(response_tokens) is int and 255000 <= response_tokens <= 262080 and
                     resources.get('sampler_errors') == [])
            if valid:
                fill_ratio = response_tokens / 262144
                lines.append(f'Recorded native/CPU-vision probes match this config: synthetic image passed; '
                             f'{response_tokens:,}/262,144 tokens ({fill_ratio:.2%}) '
                             'with tail-marker retrieval. One probe, not general quality/sustained-load validation.')
            else:
                lines.append('Stored native/vision report is stale/incomplete for this config; no matching live-test claim.')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
            lines.append('Stored native/vision report is unreadable and is not counted as live evidence.')
    else:
        lines.append('No matching native-context/CPU-vision live report is recorded.')
    return lines


def require_q4ks_candidate_lease(port, expected_lease_id=None):
    lease = host_lifecycle.read_lease()
    lease_id = lease.get('lease_id') if isinstance(lease, dict) else None
    if (not isinstance(lease, dict) or lease.get('phase') != 'held' or lease.get('port') != port or
            not isinstance(lease_id, str) or not re.fullmatch(r'[a-f0-9]{64}', lease_id) or
            (expected_lease_id is not None and lease_id != expected_lease_id)):
        raise SafetyError('Q4 .41 candidate requires its matching active HostLLM pause lease')
    return lease_id


def require_q4ks_candidate_authorization(port, expected_lease_id=None, *, host_launch_reserved=False):
    """Check HostLLM's inherited reservation or the Hive profile's token/marker coordination gate."""
    token = os.environ.get('HIVE_STRATA_LAUNCH_ID')
    if token is None:
        if not host_launch_reserved or not expected_lease_id:
            raise SafetyError('Q4 .41 candidate requires a reserved HostLLM lease or per-run Hive launch token')
        return {'authorization': 'hostllm',
                'lease_id': require_q4ks_candidate_lease(port, expected_lease_id)}
    if (not re.fullmatch(r'[a-f0-9]{48}', token) or host_launch_reserved or expected_lease_id is not None or
            'HOSTLLM_LAUNCH_LOCK_FD' in os.environ or 'HOSTLLM_LAUNCH_LEASE_ID' in os.environ):
        raise SafetyError('Q4 .41 Hive launch token is invalid or conflicts with HostLLM authorization')
    run_marker = host_lifecycle.RUN
    if run_marker.is_symlink() or not run_marker.is_file():
        raise SafetyError('Q4 .41 Hive authorization requires /run/hive/MINER_RUN')
    stop_marker = host_lifecycle.STOP
    if stop_marker.is_symlink() or stop_marker.exists():
        raise SafetyError('Q4 .41 Hive authorization refuses a pending Hive stop marker')
    if HIVE_CUR_MINER.is_symlink() or not HIVE_CUR_MINER.is_file():
        raise SafetyError('Q4 .41 Hive authorization requires a readable /run/hive/cur_miner')
    try:
        selected = HIVE_CUR_MINER.read_text().strip()
    except OSError as exc:
        raise SafetyError('Q4 .41 Hive miner selection cannot be verified') from exc
    if selected != 'custom':
        raise SafetyError('Q4 .41 Hive authorization requires cur_miner == custom')
    if host_lifecycle.read_lease() is not None:
        raise SafetyError('Q4 .41 Hive authorization refuses a conflicting HostLLM lease')
    return {'authorization': 'hive', 'launch_id': token}


def read_state():
    p = strata_state() / 'server.json'
    return json.loads(p.read_text()) if p.exists() else None


def write_state(state):
    root = strata_state()
    root.mkdir(parents=True, exist_ok=True)
    p = root / 'server.json'
    tmp = root / f'.server.{os.getpid()}.tmp'
    with tmp.open('w') as f:
        json.dump(state, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def remember_children(state):
    """Do not let a foreground watcher recreate state removed by an external stop."""
    root = strata_state()
    with (root / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read_state()
        if current is None or not same_process(current['identity'], state['identity']):
            return False
        # Keep recorded identities through CUDA exit_mm/context teardown, not just live snapshots.
        recorded = {p['pid']: p for p in state.get('children', [])}
        recorded.update({p['pid']: p for p in family(state) if p['pid'] != state['identity']['pid']})
        children = list(recorded.values())
        state['children'] = children
        if current.get('children') != children:
            write_state(state)
    return True


def family(state):
    leader = state['identity']
    live = proc(leader['pid'])
    if live and not same_process(leader, live):
        raise SafetyError('Strata frontend PID changed identity; stop refused')
    allowed = {str(Path(p).resolve()) for p in state['allowed_executables']}
    recorded = {p['pid']: p for p in state.get('children', [])}
    members = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        # Read stat first to avoid inspecting unrelated root/renter executables.
        try:
            stat = (d / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(stat[3]) != leader['sid'] or stat[0] in ('Z', 'X', 'x'):
                continue
        except (FileNotFoundError, ProcessLookupError):
            # A process can exit between /proc enumeration and opening stat.
            continue
        p = proc(int(d.name))
        if p is None:
            continue
        if p['sid'] != leader['sid'] or p['pgid'] != leader['pgid'] or p['uid'] != leader['uid']:
            raise SafetyError('Strata process-family identity mismatch')
        if live is None and not same_process(recorded.get(p['pid']), p):
            raise SafetyError('Frontend exited before this child identity was recorded; stop refused')
        if p['exe'] not in allowed:
            raise SafetyError(f"Unidentified process in Strata session: {p['pid']}; no process was stopped")
        if p['exe'] == leader['exe'] and not same_process(leader, p):
            raise SafetyError('Unidentified Python child in Strata session; stop refused')
        members.append(p)
    return members


def runtime_gate(state):
    """Yield our own runtime if a rental/new miner appears; never signal that workload."""
    docker_empty()
    owned = {p['pid'] for p in family(state)}
    for p in state.get('children', []):
        try:
            fields = (Path('/proc') / str(p['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) == p['start_ticks'] and int(fields[2]) == p['pgid'] and int(fields[3]) == p['sid']:
                owned.add(p['pid'])  # recorded CUDA task still tearing down; never a reusable bare PID
        except (FileNotFoundError, ProcessLookupError):
            # Do not treat a vanished recorded PID as an owned live GPU process.
            pass
    text = command_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.strip().isdigit() or int(line.strip()) not in owned:
            raise SafetyError('New/unidentified GPU workload detected; yielding only Strata')


def guard_during_run(state):
    # Serialize against external --stop; don't mistake its draining CUDA PIDs for a new workload.
    with (strata_state() / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read_state()
        if not current or not same_process(current['identity'], state['identity']) or current.get('stop_requested'):
            return False
        if state.get('profile') == ORCA_Q4KS_PROFILE and state.get('runtime_version') == '0.1.41':
            if state.get('hive_launch_id_sha256'):
                if host_lifecycle.read_lease() is not None:
                    raise SafetyError('Hive Q4 .41 session conflicts with a HostLLM pause lease')
                require_hive_launch_identity(state)
            else:
                require_q4ks_candidate_lease(state.get('port'), state.get('hosting_lease_id'))
        runtime_gate(state)
        return True


def _verified_hive_process_parent(identity, token, expected_parent=None):
    server = proc(identity.get('pid'))
    if not same_process(identity, server):
        raise SafetyError('Hive Strata server identity changed; stop refused')
    parent_pid = identity.get('ppid')
    if type(parent_pid) is not int or server.get('ppid') != parent_pid:
        raise SafetyError('Hive Strata launcher parent changed; stop refused')
    parent = proc(parent_pid)
    if (not parent or parent.get('uid') != identity.get('uid') or
            not any('strata_launcher.py' in arg for arg in parent['cmd'])):
        raise SafetyError('Hive Strata launcher identity is unavailable; stop refused')
    if expected_parent is not None and not same_process(expected_parent, parent):
        raise SafetyError('Hive Strata launcher process identity changed; stop refused')
    expected = f'HIVE_STRATA_LAUNCH_ID={token}'.encode()
    for pid in (identity['pid'], parent_pid):
        try:
            environment = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
        except OSError as exc:
            raise SafetyError('Cannot verify Hive Strata launch token; stop refused') from exc
        if expected not in environment:
            raise SafetyError('Hive Strata launch token does not match the live server; stop refused')
    return parent


def require_hive_launch_identity(state):
    """When Hive supplies a launch token, stop only that token- and process-bound server family."""
    token = os.environ.get('HIVE_STRATA_LAUNCH_ID')
    recorded_hash = state.get('hive_launch_id_sha256')
    if token is None:
        if recorded_hash is not None:
            raise SafetyError('Hive Strata launch token is missing; stop refused')
        return
    if not re.fullmatch(r'[a-f0-9]{48}', token):
        raise SafetyError('Invalid Hive Strata launch identity; stop refused')
    if recorded_hash is not None and recorded_hash != hashlib.sha256(token.encode()).hexdigest():
        raise SafetyError('Hive Strata launch token does not match recorded process identity; stop refused')
    identity = state.get('identity', {})
    _verified_hive_process_parent(identity, token, state.get('hive_launcher_identity'))


def stop(expected=None):
    state = expected if expected is not None else read_state()
    if not state:
        print('Strata is not running.')
        return
    current = read_state()
    if expected is not None and (not current or
                                 not same_process(current.get('identity'), state.get('identity'))):
        identity = state.get('identity', {})
        live = proc(identity.get('pid')) if identity else None
        if live and same_process(identity, live):
            raise SafetyError('Strata ownership state changed while its recorded process is live; stop refused')
        print('Recorded Strata identity is no longer active/owned; no process was signalled.')
        return
    if current and same_process(current.get('identity'), state.get('identity')):
        state = current
    require_hive_launch_identity(state)
    members = family(state)  # Validate the entire family before the first signal.
    current = read_state()
    if current and same_process(current['identity'], state['identity']):
        state = dict(current, stop_requested=True)
        write_state(state)
    for p in sorted(members, key=lambda p: p['pid'] == state['identity']['pid']):
        signal_identity(p, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        remaining = family(state)
        if not remaining:
            break
        time.sleep(0.1)
    else:
        for p in family(state):
            signal_identity(p, signal.SIGKILL)
        for _ in range(50):
            if not family(state):
                break
            time.sleep(0.1)
        else:
            raise SafetyError('Strata teardown could not be confirmed')
    if read_state() == state:
        (strata_state() / 'server.json').unlink(missing_ok=True)
    print('Only the identity-verified Strata frontend/engine/vision helper were stopped.')


def health(port):
    try:
        with open_local_api(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def verify_serving(cfg, version, port):
    """Do not silently advertise two slots if upstream fell back to one after allocation failure."""
    if version == '0.1.38':
        return 1
    headers = {'Authorization': 'Bearer ' + cfg['api_key']} if cfg.get('api_key') else {}
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/status', headers=headers)
    with open_local_api(request, timeout=5) as response:
        status = json.load(response)
    count = (status.get('concurrency') or {}).get('serving')
    if status.get('engine') != version or type(count) is not int or count != cfg.get('parallel', 1):
        raise SafetyError('Effective runtime/slot count differs from the request; no silent single-slot fallback')
    return count


def verify_batch_groups(state, version, parallel, groups):
    if parallel == 1 or version == runtimes.BASE_VERSION:
        return
    native = [p for p in family(state) if p['exe'] == state['allowed_executables'][1]]
    if len(native) != 1:
        raise SafetyError('Cannot verify the identity-bound native batch configuration')
    args = native[0]['cmd']
    if ('--batch-groups' not in args or args[args.index('--batch-groups') + 1] != str(groups)):
        raise SafetyError('Native batching must receive the explicit requested groups; automatic grouping refused')


def start(lock, profile='iq3_s', runtime='auto', parallel=1, batch_groups=1, hosting_lock=None):
    if os.geteuid() == 0 and profile == ORCA_Q4KS_PROFILE:
        raise SafetyError('Q4 Strata starts must run as the normal operator; use sudo only for identity-bound administration')
    inherited_fd = host_lifecycle.inherited_launch_reservation_fd()
    requested_lease_id = os.environ.get('HOSTLLM_LAUNCH_LEASE_ID')
    if (inherited_fd is None) != (requested_lease_id is None):
        raise SafetyError('HostLLM launch lease/reservation must be supplied together')
    if requested_lease_id is not None:
        lease = host_lifecycle.read_lease()
        if (not re.fullmatch(r'[a-f0-9]{64}', requested_lease_id) or not isinstance(lease, dict) or
                lease.get('phase') != 'held' or lease.get('lease_id') != requested_lease_id):
            raise SafetyError('HostLLM launch lease changed before model startup')
    source, config_path, cfg, version = ready_runtime(profile, runtime, parallel, batch_groups)
    state = read_state()
    if state and family(state):
        raise SafetyError('Strata is already running; use its existing web UI')
    active = host_lifecycle.active_engine()
    if active != 'none':
        raise SafetyError(f'Another LLM session is active ({active}); start refused')
    port = int(os.environ.get('STRATA_PORT', cfg.get('port', 8080)))
    hosting_lease_id = None
    hive_launch_id = None
    if profile == ORCA_Q4KS_PROFILE and version == '0.1.41':
        authorization = require_q4ks_candidate_authorization(
            port, requested_lease_id, host_launch_reserved=inherited_fd is not None)
        if authorization['authorization'] == 'hostllm':
            hosting_lease_id = authorization['lease_id']
        else:
            hive_launch_id = authorization['launch_id']
    launch_gate(port)
    state_root = strata_state()
    state_root.mkdir(parents=True, exist_ok=True)
    # A private effective config permits optional API credentials without editing the pinned source config.
    cfg['host'] = _effective_host(profile, version, cfg.get('host', '0.0.0.0'))
    cfg['port'] = port
    if profile == ORCA_PROFILE:
        cfg['model_name'] = ORCA_MODEL_ID
        cfg['aliases'] = ['orca', 'orca-strata']
    elif profile == ORCA_Q4KS_PROFILE:
        cfg['model_name'] = ORCA_Q4KS_MODEL_ID
        cfg['aliases'] = ['orca-q4ks', 'orca-q4_k_s']
    else:
        cfg['model_name'] = 'qwen3.8-flash-next-iq3_s-strata'
        cfg['aliases'] = ['qwen38', 'strata']
    cfg['sampling'] = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
                       'experimental_speed_projection': False}
    if os.environ.get('STRATA_API_KEY'):
        cfg['api_key'] = os.environ['STRATA_API_KEY']
    run_config = state_root / 'run-config.json'
    log_root = data_root() / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / f'server-{time.strftime("%Y%m%d-%H%M%S")}.log'
    cfg['log'] = str(log.with_name(log.stem + '-engine.log'))
    fd = os.open(run_config, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(cfg, f, indent=2)
    env = dict(os.environ)
    env.update(runtimes.runtime_env(version, parallel))
    if version == runtimes.LEGACY_VERSION and parallel > 1:
        print('Pinned 0.1.39 batching: all-resident zero-doorbell optimization disabled to avoid the verified layer-0 timeout.', flush=True)
    elif version == runtimes.VERSION:
        print('Pinned 0.1.41: upstream per-window all-resident fix enabled; precision-changing/unsupported opt-ins disabled.', flush=True)
    lib_dirs = cfg.get('lib_dirs', [])
    if lib_dirs:
        env['LD_LIBRARY_PATH'] = ':'.join(lib_dirs) + ':' + env.get('LD_LIBRARY_PATH', '')
    if profile == ORCA_PROFILE:
        print('EXPERIMENTAL Orca Uncensored IQ3_XXS | native 262144 | INT8 streaming KV (32K resident) | MTP | high | F16 GPU vision configured')
        print('EXPERIMENTAL: bounded text/vision checks are documented in ORCA_IQ3XXS.md; not populated-context quality or peak-memory proof.')
    elif profile == ORCA_Q4KS_PROFILE:
        context = cfg['args'][cfg['args'].index('--max-context') + 1]
        vision_cfg = cfg.get('vision')
        vision_state = 'CPU vision configured' if isinstance(vision_cfg, dict) and vision_cfg.get('gpu') is False else 'text-only'
        slot_state = 'one slot' if parallel == 1 else f'{parallel} serving slots / {batch_groups} batch groups (experimental)'
        print(f'Orca Q4_K_S | profile config: {context} context | {vision_state} | {slot_state} | original Flash-Next MTP')
        if parallel == 2 and version == '0.1.41':
            print('Experimental 2-slot profile: one bounded 4.5K-prompt probe left only 457 MiB reported VRAM free; no throughput gain is established.')
        print('Readiness verifies assets/configuration; stored live reports describe bounded probes, not general quality/load.')
        evidence = (q4ks_v0141.candidate_evidence_summary(data_root()) if version == '0.1.41'
                    else q4ks_evidence_summary(data_root(), config_path))
        for evidence_line in evidence:
            print(evidence_line)
    else:
        print('EXPERIMENTAL Strata IQ3_S | BF16 GPU vision | native 262144 | INT8 KV | MTP | high reasoning')
    print(f'Strata {version}; four-GPU layer split auto; requested slots={parallel}, batch groups={batch_groups}. Log: {log}', flush=True)
    if parallel > 1:
        print('Experimental batching: concurrent slots decode without MTP drafts; solo requests retain MTP. Verify effective serving count.', flush=True)
    if cfg['host'] == '0.0.0.0' and not cfg.get('api_key'):
        print('WARNING: unauthenticated LAN API. Use STRATA_API_KEY before exposing beyond a trusted LAN.')
    python = source / '.venv/bin/python'
    with log.open('a') as out:
        child = subprocess.Popen([str(python), '-m', 'serve.server', '--engine', 'strata', '--config', str(run_config),
                                  '--host', cfg['host'], '--port', str(port)], cwd=source, env=env,
                                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(child.pid)
    if not identity or identity['exe'] != str(python.resolve()) or identity['sid'] != child.pid:
        # Popen's exec handshake plus retained child identity, never a guessed global process name.
        child.terminate()
        child.wait(timeout=10)
        raise SafetyError('Frontend launch identity could not be recorded')
    hive_parent_identity = None
    if hive_launch_id:
        try:
            hive_parent_identity = _verified_hive_process_parent(identity, hive_launch_id)
        except Exception:
            signal_identity(identity, signal.SIGTERM)
            child.wait(timeout=10)
            raise
    state = {'identity': identity, 'port': port, 'log': str(log),
             'source_commit': runtimes.PINS[version],
             'runtime_version': version, 'parallel': parallel, 'batch_groups': batch_groups, 'profile': profile,
             'config': str(run_config), 'allowed_executables': [str(python.resolve()), cfg['exe']] +
             ([cfg['vision']['exe']] if cfg.get('vision') else [])}
    if hosting_lease_id or requested_lease_id:
        state['hosting_lease_id'] = hosting_lease_id or requested_lease_id
    if hive_launch_id:
        state['hive_launch_id_sha256'] = hashlib.sha256(hive_launch_id.encode()).hexdigest()
        state['hive_launcher_identity'] = hive_parent_identity
    write_state(state)
    # Serialize the launch handoff, not the whole foreground server lifetime.
    host_lifecycle.release_inherited_launch_reservation()
    if hosting_lock is not None:
        fcntl.flock(hosting_lock.fileno(), fcntl.LOCK_UN)
    fcntl.flock(lock, fcntl.LOCK_UN)
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    old_handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        deadline = time.monotonic() + int(os.environ.get('STRATA_HEALTH_TIMEOUT', '600'))
        next_report = 0
        next_guard = 0
        while not health(port):
            if not remember_children(state) or interrupted:
                return  # external stop/Ctrl+C, not a failed model-quality check
            if child.poll() is not None:
                raise SafetyError(f'Strata did not become ready; see {log}')
            if time.monotonic() >= next_guard:
                if not guard_during_run(state):
                    return
                next_guard = time.monotonic() + 3
            if time.monotonic() >= deadline:
                raise SafetyError(f'Strata health timeout; see {log}')
            if time.monotonic() >= next_report:
                print('Loading Strata; no long-context request is being sent...', flush=True)
                next_report = time.monotonic() + 15
            time.sleep(1)
        serving = verify_serving(cfg, version, port)
        verify_batch_groups(state, version, parallel, batch_groups)
        print(f'Strata ready: this host, port {port}, serving slots={serving}, API /v1. Ctrl+C stops only Strata.', flush=True)
        while not interrupted and child.poll() is None:
            if time.monotonic() >= next_guard:
                if not guard_during_run(state):
                    break
                next_guard = time.monotonic() + 3
            if not remember_children(state):
                break
            time.sleep(1)
        if not interrupted and child.poll() is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)  # wait for any external stop/state removal
            if read_state() == state:
                raise SafetyError(f'Strata frontend exited unexpectedly ({child.returncode}); see {log}')
    finally:
        original_error = sys.exc_info()[1]
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stop(expected=state)
        except (SafetyError, OSError) as cleanup_error:
            if original_error is None:
                raise
            print(f'Cleanup also blocked: {cleanup_error}. Original failure: {original_error}', file=sys.stderr)
            # Keep the state/lease for recovery and propagate the original failure, not a masked teardown error.
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print('WARNING: frontend did not reap; inspect its recorded identity.', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--quickstart', '--start', action='store_true')
    modes.add_argument('--stop', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--check-ready', action='store_true')
    modes.add_argument('--configure-native-context', action='store_true')
    modes.add_argument('--enable-q4ks-vision', action='store_true')
    modes.add_argument('--rollback-q4ks-vision', action='store_true')
    modes.add_argument('--record-q4ks-32k-validation', metavar='REPORT')
    modes.add_argument('--validate-q4ks141-32k', action='store_true')
    modes.add_argument('--configure-q4ks141-native-context', action='store_true')
    modes.add_argument('--enable-q4ks141-vision', action='store_true')
    modes.add_argument('--rollback-q4ks141-vision', action='store_true')
    modes.add_argument('--select-runtime', choices=runtimes.VERSIONS)
    ap.add_argument('--profile', choices=['iq3_s', ORCA_PROFILE, ORCA_Q4KS_PROFILE], default=None)
    ap.add_argument('--runtime', choices=['auto', *runtimes.VERSIONS], default=os.environ.get('STRATA_RUNTIME', 'auto'))
    ap.add_argument('--parallel', type=int, choices=[1, 2], default=os.environ.get('STRATA_PARALLEL', '1'))
    ap.add_argument('--batch-groups', type=int, choices=[1, 2], default=os.environ.get('STRATA_BATCH_GROUPS', '1'))
    a = ap.parse_args()
    profile = a.profile or 'iq3_s'
    q4_mutation = not (a.stop or a.status or a.check_ready or a.select_runtime)
    if os.geteuid() == 0 and profile == ORCA_Q4KS_PROFILE and q4_mutation:
        print('BLOCKED: Q4 starts/configuration/evidence updates must run as the normal operator; use sudo only for identity-bound administration.', file=sys.stderr)
        return 1
    try:
        if a.select_runtime:
            # No running model is stopped/reconfigured. An active session reads its existing private run-config.
            for p in runtimes.PROFILES:
                ready(p)
            runtimes.select(data_root(), a.select_runtime)
        elif a.configure_native_context:
            if profile == ORCA_Q4KS_PROFILE:
                configure_q4ks_native_context(data_root())
            elif profile == ORCA_PROFILE:
                configure_native_context(data_root())
            else:
                raise SafetyError('Only prepared Orca profiles support explicit native-context migration; IQ3_S already requires 262144')
        elif a.enable_q4ks_vision:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('CPU vision configuration is only available for --profile orca-q4_k_s')
            configure_q4ks_vision(data_root())
        elif a.rollback_q4ks_vision:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('Q4 CPU-vision rollback is only available for --profile orca-q4_k_s')
            rollback_q4ks_vision(data_root())
        elif a.record_q4ks_32k_validation:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('A live 32K validation report can only be recorded for --profile orca-q4_k_s')
            record_q4ks_32k_validation(data_root(), a.record_q4ks_32k_validation)
        elif a.validate_q4ks141_32k:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('The live .41 32K gate is only available for --profile orca-q4_k_s')
            import orca_q4ks_v0141_validation
            orca_q4ks_v0141_validation.run(data_root())
        elif a.configure_q4ks141_native_context:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('.41 native-context migration is only available for --profile orca-q4_k_s')
            q4ks_v0141.configure_native_context(data_root())
        elif a.enable_q4ks141_vision:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('.41 CPU vision is only available for --profile orca-q4_k_s')
            q4ks_v0141.configure_cpu_vision(data_root())
        elif a.rollback_q4ks141_vision:
            if profile != ORCA_Q4KS_PROFILE:
                raise SafetyError('.41 CPU-vision rollback is only available for --profile orca-q4_k_s')
            q4ks_v0141.rollback_cpu_vision(data_root())
        elif a.check_ready:
            source, config, cfg, version = ready_runtime(profile, a.runtime, a.parallel, a.batch_groups)
            print(f'Runtime ready: {version}, requested slots={a.parallel}, batch groups={a.batch_groups}')
            if profile == ORCA_PROFILE:
                print('PREPARED: pinned Orca IQ3_XXS, own compatibility pack/tokenizer, F16 vision configured, native 262144, INT8 streaming KV.')
                print('Readiness verifies assets/config only; bounded live results are separate in ORCA_IQ3XXS.md. No full-context quality claim.')
            elif profile == ORCA_Q4KS_PROFILE:
                context = cfg['args'][cfg['args'].index('--max-context') + 1]
                vision_cfg = cfg.get('vision')
                vision_state = 'CPU vision configured' if isinstance(vision_cfg, dict) and vision_cfg.get('gpu') is False else 'text-only'
                if version == '0.1.41':
                    cache_args = cfg.get('args', [])
                    cache_index = cache_args.index('--conversation-cache-mib')
                    cache_budget = cache_args[cache_index + 1]
                    cache_slots = cache_args[cache_args.index('--conversation-cache-slots') + 1]
                    if cfg.get('parallel', 1) == 2:
                        slot_state = '2 serving slots; batch groups=2'
                    else:
                        slot_state = 'single-slot'
                    print(f'READY CONFIG: pinned Q4_K_S; future profile context={context}; {vision_state}; '
                          f'{slot_state}; conversation cache={cache_budget} MiB / {cache_slots} cache slots.')
                else:
                    print(f'READY CONFIG: pinned Q4_K_S; future profile context={context}; {vision_state}; single-slot.')
                print('Readiness verifies pinned files/configuration only; it is not a general quality or sustained-load test.')
                evidence = (q4ks_v0141.candidate_evidence_summary(data_root()) if version == '0.1.41'
                            else q4ks_evidence_summary(data_root(), config))
                for evidence_line in evidence:
                    print(evidence_line)
            else:
                print('PREPARED: pinned IQ3_S, BF16 GPU vision, native 262144, INT8 KV, MTP, four-GPU auto split.')
                print('Readiness verifies assets/config; bounded live validation is documented in STRATA_IQ3S.md (not full-context quality).')
        elif a.status:
            state = read_state()
            live_state = None
            if state and family(state):
                live_state = state
                print(f'Strata running ({state.get("profile", "iq3_s")}, runtime {state.get("runtime_version", "0.1.38")}, '
                      f'requested slots={state.get("parallel", 1)}); port {state["port"]}; log {state["log"]}')
            else:
                print('Strata stopped.')
            try:
                q4_config_path = q4ks_profile_root(data_root()) / 'config.json'
                q4_config = json.loads(q4_config_path.read_text())
                q4_args = q4_config.get('args', [])
                q4_context = q4_args[q4_args.index('--max-context') + 1]
                q4_vision_cfg = q4_config.get('vision')
                if isinstance(q4_vision_cfg, dict) and q4_vision_cfg.get('gpu') is False:
                    q4_vision_state = 'CPU vision configured'
                elif '--vision' not in q4_args and q4_vision_cfg is None:
                    q4_vision_state = 'text-only'
                else:
                    q4_vision_state = 'vision config unrecognized'
                print(f'Q4 .39 baseline profile file: context={q4_context}, {q4_vision_state}.')
                for evidence_line in q4ks_evidence_summary(data_root(), q4_config_path):
                    print(evidence_line)
                candidate_path = q4ks_v0141.profile_root(data_root()) / 'config.json'
                if candidate_path.is_file() and not candidate_path.is_symlink():
                    candidate_config = json.loads(candidate_path.read_text())
                    candidate_args = candidate_config.get('args', [])
                    candidate_context = candidate_args[candidate_args.index('--max-context') + 1]
                    candidate_vision = candidate_config.get('vision')
                    if isinstance(candidate_vision, dict) and candidate_vision.get('gpu') is False:
                        candidate_vision_state = 'CPU vision configured'
                    elif candidate_vision is None and '--vision' not in candidate_args:
                        candidate_vision_state = 'text-only'
                    else:
                        candidate_vision_state = 'unrecognized vision config'
                    active_candidate = bool(live_state and live_state.get('profile') == ORCA_Q4KS_PROFILE and
                                            live_state.get('runtime_version') == '0.1.41')
                    print(f'Separate Q4 .41 candidate profile file: context={candidate_context}, '
                          f'{candidate_vision_state}; never selected by auto.')
                    if active_candidate:
                        live_config_path = Path(live_state.get('config', ''))
                        if (live_config_path.is_file() and not live_config_path.is_symlink() and
                                live_config_path.resolve().is_relative_to(strata_state().resolve())):
                            active_config = json.loads(live_config_path.read_text())
                            active_args = active_config.get('args', [])
                            active_context = active_args[active_args.index('--max-context') + 1]
                            active_vision = active_config.get('vision')
                            active_vision_state = ('CPU vision configured' if isinstance(active_vision, dict) and
                                                    active_vision.get('gpu') is False else
                                                    'text-only' if active_vision is None and '--vision' not in active_args
                                                    else 'unrecognized vision config')
                            print(f'Active Q4 .41 private run-config: context={active_context}, '
                                  f'{active_vision_state}.')
                        else:
                            print('Active Q4 .41 private run-config is unavailable/unverified.')
                    for evidence_line in q4ks_v0141.candidate_evidence_summary(data_root()):
                        print(evidence_line)
            except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, SafetyError):
                print('Q4 profile configuration is unavailable or unreadable; no readiness/live-validation claim.')
        else:
            root = strata_state()
            root.mkdir(parents=True, exist_ok=True)
            with (root / 'lifecycle.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if a.stop:
                    state = read_state()
                    if a.profile and state and state.get('profile', 'iq3_s') != profile and family(state):
                        raise SafetyError('A different Strata profile is running; profile-specific stop refused')
                    explicit_runtime = any(x == '--runtime' or x.startswith('--runtime=') for x in sys.argv[1:])
                    if (explicit_runtime and a.runtime != 'auto' and state and
                            state.get('runtime_version', '0.1.38') != a.runtime and family(state)):
                        raise SafetyError('A different Strata runtime is running; runtime-specific stop refused')
                    stop()
                else:
                    with host_lifecycle.launch_reservation() as hosting_lock:
                        start(lock, profile, a.runtime, a.parallel, a.batch_groups, hosting_lock)
    except (SafetyError, FileNotFoundError, PermissionError, ValueError, TypeError, KeyError, IndexError, BlockingIOError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
