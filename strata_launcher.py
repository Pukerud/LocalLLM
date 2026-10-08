#!/usr/bin/env python3
"""Prepared Strata model launcher. No downloads, updates, driver or service changes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from engine_safety import (SafetyError, command_output, docker_empty, launch_gate, owner_home,
                           proc, same_process, signal_identity, strata_state)
from prepare_strata import DIGESTS
from orca_assets import MODEL_ID as ORCA_MODEL_ID, PROFILE as ORCA_PROFILE
from orca_profile import configure_native_context, ready_orca
import strata_runtime as runtimes

SOURCE_COMMIT = '99f3dbd0b21d1401b3769e0c0d963913607f380b'
MODEL_REVISION = 'ed59f92082b1e93c0e96d60a8b11aab089b52f09'
MODEL = 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S'


def data_root():
    return Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))


def ready(profile='iq3_s'):
    root = data_root()
    if profile == ORCA_PROFILE:
        return ready_orca(root)
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


def ready_runtime(profile='iq3_s', runtime='auto', parallel=1, batch_groups=1):
    source, path, cfg = ready() if profile == 'iq3_s' else ready(profile)
    return runtimes.configure(data_root(), source, path, cfg, profile, runtime, parallel, batch_groups)


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
        except FileNotFoundError:
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
        except FileNotFoundError:
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
        runtime_gate(state)
        return True


def stop(expected=None):
    state = expected if expected is not None else read_state()
    if not state:
        print('Strata is not running.')
        return
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
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def verify_serving(cfg, version, port):
    """Do not silently advertise two slots if upstream fell back to one after allocation failure."""
    if version == '0.1.38':
        return 1
    headers = {'Authorization': 'Bearer ' + cfg['api_key']} if cfg.get('api_key') else {}
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/status', headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
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


def start(lock, profile='iq3_s', runtime='auto', parallel=1, batch_groups=1):
    source, config_path, cfg, version = ready_runtime(profile, runtime, parallel, batch_groups)
    state = read_state()
    if state and family(state):
        raise SafetyError('Strata is already running; use its existing web UI')
    port = int(os.environ.get('STRATA_PORT', cfg.get('port', 8080)))
    launch_gate(port)
    state_root = strata_state()
    state_root.mkdir(parents=True, exist_ok=True)
    # A private effective config permits optional API credentials without editing the pinned source config.
    cfg['host'] = os.environ.get('STRATA_HOST', cfg.get('host', '0.0.0.0'))
    cfg['port'] = port
    cfg['model_name'] = ORCA_MODEL_ID if profile == ORCA_PROFILE else 'qwen3.8-flash-next-iq3_s-strata'
    cfg['aliases'] = ['orca', 'orca-strata'] if profile == ORCA_PROFILE else ['qwen38', 'strata']
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
    state = {'identity': identity, 'port': port, 'log': str(log),
             'source_commit': runtimes.PINS[version],
             'runtime_version': version, 'parallel': parallel, 'batch_groups': batch_groups, 'profile': profile,
             'config': str(run_config), 'allowed_executables': [str(python.resolve()), cfg['exe'], cfg['vision']['exe']]}
    write_state(state)
    # Serialize the launch handoff, not the whole foreground server lifetime.
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
    modes.add_argument('--select-runtime', choices=runtimes.VERSIONS)
    ap.add_argument('--profile', choices=['iq3_s', ORCA_PROFILE], default=None)
    ap.add_argument('--runtime', choices=['auto', *runtimes.VERSIONS], default=os.environ.get('STRATA_RUNTIME', 'auto'))
    ap.add_argument('--parallel', type=int, choices=[1, 2], default=os.environ.get('STRATA_PARALLEL', '1'))
    ap.add_argument('--batch-groups', type=int, choices=[1, 2], default=os.environ.get('STRATA_BATCH_GROUPS', '1'))
    a = ap.parse_args()
    profile = a.profile or 'iq3_s'
    try:
        if a.select_runtime:
            # No running model is stopped/reconfigured. An active session reads its existing private run-config.
            for p in runtimes.PROFILES:
                ready(p)
            runtimes.select(data_root(), a.select_runtime)
        elif a.configure_native_context:
            if profile != ORCA_PROFILE:
                raise SafetyError('Only the prepared Orca 32K-to-native migration is supported; IQ3_S already requires 262144')
            configure_native_context(data_root())
        elif a.check_ready:
            source, config, cfg, version = ready_runtime(profile, a.runtime, a.parallel, a.batch_groups)
            print(f'Runtime ready: {version}, requested slots={a.parallel}, batch groups={a.batch_groups}')
            if profile == ORCA_PROFILE:
                print('PREPARED: pinned Orca IQ3_XXS, own compatibility pack/tokenizer, F16 vision configured, native 262144, INT8 streaming KV.')
                print('Readiness verifies assets/config only; bounded live results are separate in ORCA_IQ3XXS.md. No full-context quality claim.')
            else:
                print('PREPARED: pinned IQ3_S, BF16 GPU vision, native 262144, INT8 KV, MTP, four-GPU auto split.')
                print('Readiness verifies assets/config; bounded live validation is documented in STRATA_IQ3S.md (not full-context quality).')
        elif a.status:
            state = read_state()
            if state and family(state):
                print(f'Strata running ({state.get("profile", "iq3_s")}, runtime {state.get("runtime_version", "0.1.38")}, '
                      f'requested slots={state.get("parallel", 1)}); port {state["port"]}; log {state["log"]}')
            else:
                print('Strata stopped.')
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
                    start(lock, profile, a.runtime, a.parallel, a.batch_groups)
    except (SafetyError, FileNotFoundError, PermissionError, ValueError, TypeError, KeyError, IndexError, BlockingIOError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
