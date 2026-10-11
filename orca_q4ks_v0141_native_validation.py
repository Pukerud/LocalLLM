#!/usr/bin/env python3
"""Bounded native-context/CPU-vision probes for the isolated Q4_K_S / Strata 0.1.41 profile.

Never starts or stops a service. Requires an already-running, identity-bound HostLLM launch.
"""
import base64
import binascii
import fcntl
import http.client
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import zlib

from engine_safety import SafetyError, owner_home, proc, same_process, strata_state
from local_api_client import open_local_api
import hosting_lifecycle as host_lifecycle
import orca_q4ks_v0141_assets as pins
import orca_q4ks_v0141_profile as candidate

ROOT = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
API_HOST = '127.0.0.1'
TARGET_PROMPT_TOKENS = 261_000
MIN_RAM_AVAILABLE_MIB = candidate.NATIVE_VALIDATION_RAM_FLOOR_MIB
MAX_GPU_USED_MIB = candidate.NATIVE_VALIDATION_GPU_CEILING_MIB
MAX_LONG_REQUEST_S = candidate.NATIVE_VALIDATION_MAX_REQUEST_S


def api_json(base, path, body=None, timeout=180, api_key=None):
    data = None if body is None else json.dumps(body, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with open_local_api(request, timeout=timeout) as response:
        return json.load(response)


def normalize_address(value):
    try:
        parsed = ipaddress.ip_address(value.split('%', 1)[0])
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
            parsed = parsed.ipv4_mapped
        return str(parsed)
    except ValueError:
        return value.casefold()


def parse_endpoint(value):
    if value.startswith('['):
        address, separator, port = value[1:].rpartition(']:')
        if not separator:
            raise ValueError(f'Invalid bracketed socket endpoint: {value}')
    else:
        address, separator, port = value.rpartition(':')
        if not separator:
            raise ValueError(f'Invalid socket endpoint: {value}')
    return normalize_address(address), int(port)


def established_connections(port):
    output = subprocess.check_output(
        ['ss', '-Htn', 'state', 'established', f'( sport = :{port} )'], text=True)
    return [line for line in output.splitlines() if line.strip()]


def require_idle(base, port, api_key):
    if established_connections(port):
        raise SafetyError('An API client is already connected; no validation request was sent')
    status = api_json(base, '/v1/status', timeout=8, api_key=api_key)
    activity = status.get('activity')
    # Strata 0.1.41 folds busy and queued work into in_flight; check a separate
    # queued counter only on compatible APIs that actually expose it.
    if (not isinstance(activity, dict) or type(activity.get('in_flight')) is not int or
            activity['in_flight'] != 0 or
            ('queued' in activity and (type(activity['queued']) is not int or activity['queued'] != 0))):
        raise SafetyError('The API reports an active/queued request; no validation request was sent')
    if established_connections(port):
        raise SafetyError('A client arrived during the idle check; no validation request was sent')
    return status


def require_live_candidate(root):
    manifest, _, profile_root, config, engine_record, _, _, manifest_sha256 = candidate._prepared(root)
    candidate._check_parity(root, manifest, engine_record)
    config_path = profile_root / 'config.json'
    expected = candidate.make_config(root, context=262144, vision=True)
    if config != expected:
        raise SafetyError('Q4 .41 profile is not the exact migrated native-context/CPU-vision preset')

    state_path = strata_state() / 'server.json'
    state = json.loads(state_path.read_text())
    identity = state.get('identity')
    current = proc(identity.get('pid')) if isinstance(identity, dict) else None
    if (state.get('profile') != pins.HOST_PROFILE or state.get('runtime_version') != pins.RUNTIME_VERSION or
            state.get('source_commit') != pins.SOURCE_COMMIT or not same_process(identity, current) or
            state.get('parallel') != 1 or state.get('batch_groups') != 1):
        raise SafetyError('Live server is not the identity-verified single-slot Q4 .41 candidate')

    lease = host_lifecycle.read_lease()
    lease_id = lease.get('lease_id') if isinstance(lease, dict) else None
    if (not isinstance(lease, dict) or lease.get('phase') != 'held' or
            lease.get('port') != state.get('port') or
            not isinstance(lease_id, str) or len(lease_id) != 64 or
            state.get('hosting_lease_id') != lease_id):
        raise SafetyError('Live Q4 .41 server is not bound to its active HostLLM lease')

    run_config_path = Path(state.get('config', ''))
    expected_run_config = strata_state() / 'run-config.json'
    if (run_config_path != expected_run_config or run_config_path.is_symlink() or
            not run_config_path.resolve().is_relative_to(strata_state().resolve())):
        raise SafetyError('Live private run-config is not the HostLLM-owned config')
    run_config = json.loads(run_config_path.read_text())
    keys = ('exe', 'cwd', 'args', 'tokenizer', 'gpu', 'layer_split', 'gpu_order',
            'model_name', 'aliases', 'sampling', 'vision')
    if (run_config.get('host') not in ('127.0.0.1', '0.0.0.0') or
            any(run_config.get(key) != expected.get(key) for key in keys)):
        raise SafetyError('Live server config differs from the exact Q4 .41 native/CPU-vision preset')
    if run_config.get('vision', {}).get('gpu') is not False:
        raise SafetyError('Q4 .41 vision is not configured CPU-only')
    expected_executables = {
        str((candidate.source_root(root) / '.venv/bin/python').resolve()),
        str((candidate.source_root(root) / 'engine/strata').resolve()),
        str((candidate.pinned_runtime_source(root) / 'engine/strata-vision').resolve()),
    }
    if set(state.get('allowed_executables', [])) != expected_executables:
        raise SafetyError('Live Q4 .41 process allowlist does not exactly pin the CPU vision helper')
    if not 1 <= int(state.get('port', 0)) <= 65535:
        raise SafetyError('Live candidate API port is invalid')
    return state, config_path, run_config, lease_id, manifest_sha256


def png_chunk(kind, data):
    payload = kind + data
    return (len(data).to_bytes(4, 'big') + payload +
            (zlib.crc32(payload) & 0xffffffff).to_bytes(4, 'big'))


def marker_png():
    width = height = 256
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend((255, 0, 255) if 48 <= x < 208 and 48 <= y < 208 else (255, 255, 255))
    return (b'\x89PNG\r\n\x1a\n' +
            png_chunk(b'IHDR', width.to_bytes(4, 'big') + height.to_bytes(4, 'big') + bytes((8, 2, 0, 0, 0))) +
            png_chunk(b'IDAT', zlib.compress(bytes(raw), 6)) + png_chunk(b'IEND', b''))


def count_tokens(base, port, model, prompt, api_key):
    require_idle(base, port, api_key)
    body = {'model': model, 'max_tokens': 64, 'thinking': {'type': 'disabled'},
            'messages': [{'role': 'user', 'content': prompt}]}
    result = api_json(base, '/v1/messages/count_tokens', body, timeout=180, api_key=api_key)
    require_idle(base, port, api_key)
    value = result.get('input_tokens')
    if type(value) is not int or value <= 0:
        raise SafetyError(f'Unexpected count_tokens response: {result!r}')
    return value


def build_near_full_prompt(base, port, model, api_key):
    prefix = ('Context-window validation filler follows. It is deliberately repetitive; '
              'use the final instruction and marker only.\n')
    filler = ' cedar quartz orbit maple violet turbine delta 17 birch lumen cobalt\n'
    marker = 'Q4-141-NATIVE-CTX-39B7'
    suffix = f'\nFinal instruction: What is the exact marker immediately above? Reply with only that marker.\nMarker: {marker}'

    def prompt_for(repetitions):
        return prefix + filler * repetitions + suffix

    low_count = count_tokens(base, port, model, prompt_for(0), api_key)
    probe_count = count_tokens(base, port, model, prompt_for(512), api_key)
    per_repeat = max(1.0, (probe_count - low_count) / 512)
    high = max(512, int((TARGET_PROMPT_TOKENS - low_count) / per_repeat * 1.12) + 8)
    low, best = 0, None
    while low <= high:
        mid = (low + high) // 2
        prompt = prompt_for(mid)
        count = count_tokens(base, port, model, prompt, api_key)
        if count <= TARGET_PROMPT_TOKENS:
            best = (mid, prompt, count)
            low = mid + 1
        else:
            high = mid - 1
    if best is None or best[2] < 255_000:
        raise SafetyError(f'Could not fit a near-full prompt; best count={best[2] if best else None}')
    return best[1], best[2], best[0], marker


def memory_snapshot():
    mem = Path('/proc/meminfo').read_text()
    available = next(int(line.split()[1]) for line in mem.splitlines()
                     if line.startswith('MemAvailable:')) // 1024
    text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.STDOUT, timeout=6)
    gpu = [int(line.strip()) for line in text.splitlines() if line.strip()]
    if len(gpu) != 4:
        raise SafetyError(f'Expected four GPU memory readings, got {len(gpu)}')
    return available, gpu


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.ram_min = None
        self.gpu_peak = []
        self.aggregate_peak = 0
        self.samples = 0
        self.last_sample_at = None
        self.errors = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                ram, gpu = memory_snapshot()
                self.ram_min = ram if self.ram_min is None else min(self.ram_min, ram)
                self.gpu_peak = gpu[:] if not self.gpu_peak else [max(a, b) for a, b in zip(self.gpu_peak, gpu)]
                self.aggregate_peak = max(self.aggregate_peak, sum(gpu))
                self.samples += 1
                self.last_sample_at = time.monotonic()
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop_event.wait(1)


def assert_sampler_safe(sampler):
    if sampler.errors:
        raise SafetyError('Resource telemetry failed: ' + sampler.errors[0])
    if sampler.samples == 0 or sampler.last_sample_at is None or time.monotonic() - sampler.last_sample_at > 5:
        raise SafetyError('RAM/GPU telemetry is missing or stale')
    if sampler.ram_min is None or sampler.ram_min < MIN_RAM_AVAILABLE_MIB:
        raise SafetyError(f'Available RAM is below the {MIN_RAM_AVAILABLE_MIB} MiB safety floor')
    if len(sampler.gpu_peak) != 4 or max(sampler.gpu_peak) > MAX_GPU_USED_MIB:
        raise SafetyError(f'Four-GPU telemetry is missing or exceeds {MAX_GPU_USED_MIB} MiB per GPU')


def exact_arithmetic_answer(answer, finish_reason):
    return isinstance(answer, str) and answer.strip() == '437' and finish_reason == 'stop'


def exact_color_answer(answer, finish_reason):
    return (isinstance(answer, str) and answer.strip().strip(' .,!?:;').casefold() == 'magenta' and
            finish_reason == 'stop')


def text_blocks(response):
    return ''.join(block.get('text', '') for block in response.get('content', [])
                   if isinstance(block, dict) and block.get('type') == 'text')


def message_input_tokens(usage):
    """Return total prompt tokens, including Anthropic cache reads, or None if malformed."""
    if not isinstance(usage, dict):
        return None
    uncached = usage.get('input_tokens')
    cached = usage.get('cache_read_input_tokens', 0)
    if (type(uncached) is not int or uncached < 0 or
            type(cached) is not int or cached < 0):
        return None
    return uncached + cached


def wait_server_idle(base, port, api_key, timeout=120):
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        try:
            last_state = api_json(base, '/v1/status', timeout=8, api_key=api_key)
            peers = established_connections(port)
            activity = last_state.get('activity')
            if (isinstance(activity, dict) and type(activity.get('in_flight')) is int and
                    activity['in_flight'] == 0 and
                    ('queued' not in activity or
                     (type(activity.get('queued')) is int and activity['queued'] == 0)) and not peers):
                return last_state
        except Exception as exc:
            last_state = {'status_check_error': str(exc)}
        time.sleep(2)
    raise SafetyError(f'Could not verify server-side idle after client disconnect within {timeout}s: {last_state!r}')


def _run_long_request(base, port, api_key, model, body, sampler, interrupt_event=None):
    assert_sampler_safe(sampler)
    encoded = json.dumps(body, separators=(',', ':')).encode()
    result_box, conn_box = {}, {}
    lock = threading.Lock()
    started = time.monotonic()

    def request_thread():
        conn = http.client.HTTPConnection(API_HOST, port, timeout=MAX_LONG_REQUEST_S + 120)
        with lock:
            conn_box['conn'] = conn
        try:
            conn.connect()
            local_address = conn.sock.getsockname()
            with lock:
                conn_box['client_endpoint'] = (normalize_address(local_address[0]), int(local_address[1]))
            headers = {'Content-Type': 'application/json', 'Content-Length': str(len(encoded))}
            if api_key:
                headers['Authorization'] = 'Bearer ' + api_key
            conn.request('POST', '/v1/messages', body=encoded, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            result_box['http_status'] = response.status
            result_box['body'] = json.loads(raw)
        except Exception as exc:
            result_box['error'] = repr(exc)
        finally:
            conn.close()

    thread = threading.Thread(target=request_thread, daemon=True)
    thread.start()
    abort_reason = None
    last_log = 0.0
    while thread.is_alive():
        time.sleep(2)
        if interrupt_event is not None and interrupt_event.is_set():
            abort_reason = 'operator interrupted the near-full-context request'
        elapsed = time.monotonic() - started
        if elapsed > MAX_LONG_REQUEST_S:
            abort_reason = 'near-full-context request exceeded the 30-minute safety limit'
        if sampler.errors:
            abort_reason = 'resource telemetry failed during the near-full-context request'
        if sampler.last_sample_at is None or time.monotonic() - sampler.last_sample_at > 5:
            abort_reason = 'resource telemetry became stale during the near-full-context request'
        if sampler.ram_min is not None and sampler.ram_min < MIN_RAM_AVAILABLE_MIB:
            abort_reason = f'available RAM fell below {MIN_RAM_AVAILABLE_MIB} MiB safety floor'
        if sampler.gpu_peak and max(sampler.gpu_peak) > MAX_GPU_USED_MIB:
            abort_reason = f'GPU allocation crossed the {MAX_GPU_USED_MIB} MiB safety threshold'
        if not abort_reason:
            try:
                with lock:
                    own_endpoint = conn_box.get('client_endpoint')
                peers = []
                for line in established_connections(port):
                    peer_endpoint = parse_endpoint(line.split()[-1])
                    if peer_endpoint != own_endpoint:
                        peers.append(line)
                if peers:
                    abort_reason = 'another API client connected during the single-slot probe; cancelling this request'
            except Exception as exc:
                abort_reason = f'could not verify API isolation during the probe: {exc}'
        if abort_reason:
            with lock:
                conn = conn_box.get('conn')
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            break
        if elapsed - last_log >= 30:
            last_log = elapsed
            print(json.dumps({'near_full_context_elapsed_s': round(elapsed),
                              'ram_available_min_mib': sampler.ram_min,
                              'gpu_memory_peak_mib': sampler.gpu_peak}), flush=True)
    thread.join(timeout=120)
    failure = abort_reason
    if thread.is_alive():
        failure = failure or 'client thread did not settle after the disconnect'
    elif 'error' in result_box:
        failure = failure or ('near-full-context HTTP request failed: ' + result_box['error'])
    elif result_box.get('http_status') != 200 or not isinstance(result_box.get('body'), dict):
        failure = failure or f'near-full-context API returned HTTP {result_box.get("http_status")}'
    elapsed = time.monotonic() - started
    if elapsed > MAX_LONG_REQUEST_S:
        failure = failure or 'near-full-context request exceeded the 30-minute safety limit'
    if failure:
        wait_server_idle(base, port, api_key, timeout=120)
        raise SafetyError(f'{failure}; server-side idle was confirmed after disconnect')
    return result_box['body'], elapsed


def run_long_request(base, port, api_key, model, body, sampler):
    if threading.current_thread() is not threading.main_thread():
        return _run_long_request(base, port, api_key, model, body, sampler)
    interrupt_event = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda _signum, _frame: interrupt_event.set())
    try:
        return _run_long_request(base, port, api_key, model, body, sampler, interrupt_event)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _run_locked(root):
    state, config_path, run_config, lease_id, prepared_manifest_sha256 = require_live_candidate(root)
    report_path = candidate.profile_root(root) / 'validation-native.json'
    if report_path.exists() or report_path.is_symlink():
        raise SafetyError('Q4 .41 native report already exists; inspect it rather than overwrite')
    port = state['port']
    base = f'http://{API_HOST}:{port}'
    api_key = run_config.get('api_key')
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise SafetyError('Q4 .41 private API-key value is malformed')

    require_idle(base, port, api_key)
    health = api_json(base, '/health', timeout=10, api_key=api_key)
    status = require_idle(base, port, api_key)
    if (health.get('status') != 'ok' or health.get('model') != pins.MODEL_ID or
            health.get('max_context') != 262144 or health.get('images') is not True or
            health.get('loaded') is not True or status.get('model') != pins.MODEL_ID or
            status.get('context', {}).get('native') != 262144 or
            status.get('vision', {}).get('enabled') is not True or
            status.get('concurrency', {}).get('serving') != 1):
        raise SafetyError(f'Live Q4 .41 health/status mismatch: {health!r}, {status!r}')

    report = {'profile': pins.HOST_PROFILE, 'model_id': pins.MODEL_ID,
              'source_commit': pins.SOURCE_COMMIT, 'runtime_version': pins.RUNTIME_VERSION,
              'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
              'validation_runner_sha256': candidate.sha256(Path(__file__).resolve()),
              'local_api_client_sha256': candidate.sha256(Path(__file__).with_name('local_api_client.py')),
              'prepared_manifest_sha256': prepared_manifest_sha256,
              'config_sha256': candidate.sha256(config_path), 'context_configured': 262144,
              'vision_configured': True, 'vision_cpu_only': run_config['vision'].get('gpu') is False,
              'live_server_identity': state['identity'], 'live_hosting_lease_id': lease_id,
              'health': health, 'vision': {'passed': False}, 'full_context': {'passed': False},
              'safety': {'ram_floor_mib': MIN_RAM_AVAILABLE_MIB,
                         'gpu_memory_ceiling_mib': MAX_GPU_USED_MIB,
                         'max_request_seconds': MAX_LONG_REQUEST_S}}
    sampler = Sampler()
    sampler.start()
    try:
        sample_deadline = time.monotonic() + 10
        while sampler.samples == 0 and time.monotonic() < sample_deadline:
            if sampler.errors:
                raise SafetyError('Resource telemetry failed before probes: ' + sampler.errors[0])
            time.sleep(0.1)
        assert_sampler_safe(sampler)
        require_idle(base, port, api_key)
        arithmetic = api_json(base, '/v1/chat/completions', {
            'model': pins.MODEL_ID, 'temperature': 0, 'max_tokens': 64,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': 'Compute 19 * 23. Reply with only the integer.'}]},
            timeout=180, api_key=api_key)
        arithmetic_choice = arithmetic.get('choices', [{}])[0]
        arithmetic_text = arithmetic_choice.get('message', {}).get('content') or ''
        report['text_smoke'] = {'passed': exact_arithmetic_answer(
                                    arithmetic_text, arithmetic_choice.get('finish_reason')),
                                'answer': arithmetic_text[:120],
                                'finish_reason': arithmetic_choice.get('finish_reason')}
        require_idle(base, port, api_key)

        image_body = {'model': pins.MODEL_ID, 'temperature': 0, 'max_tokens': 64,
                      'chat_template_kwargs': {'enable_thinking': False},
                      'messages': [{'role': 'user', 'content': [
                          {'type': 'text', 'text': 'The image has a plain white background and one large square. What color is the square? Reply with one color word.'},
                          {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                              base64.b64encode(marker_png()).decode('ascii')}}]}]}
        image_result = api_json(base, '/v1/chat/completions', image_body, timeout=600, api_key=api_key)
        image_choice = image_result.get('choices', [{}])[0]
        image_text = image_choice.get('message', {}).get('content') or ''
        vision_passed = exact_color_answer(image_text, image_choice.get('finish_reason'))
        report['vision'] = {'passed': vision_passed, 'expected': 'magenta',
                            'answer': image_text[:500], 'finish_reason': image_choice.get('finish_reason'),
                            'cpu_only_configured': True,
                            'prompt_tokens': image_result.get('usage', {}).get('prompt_tokens')}
        require_idle(base, port, api_key)

        model = pins.MODEL_ID
        tiny_prompt = 'Compute 19 * 23. Reply with only the integer.'
        tiny_count = count_tokens(base, port, model, tiny_prompt, api_key)
        require_idle(base, port, api_key)
        tiny_body = {'model': model, 'max_tokens': 64, 'thinking': {'type': 'disabled'},
                     'messages': [{'role': 'user', 'content': tiny_prompt}]}
        tiny_response = api_json(base, '/v1/messages', tiny_body, timeout=180, api_key=api_key)
        tiny_text = text_blocks(tiny_response)
        tiny_usage = tiny_response.get('usage', {})
        tiny_uncached = tiny_usage.get('input_tokens') if isinstance(tiny_usage, dict) else None
        tiny_cached = tiny_usage.get('cache_read_input_tokens', 0) if isinstance(tiny_usage, dict) else None
        tiny_total = message_input_tokens(tiny_usage)
        report['token_count_control'] = {
            'passed': tiny_total == tiny_count and tiny_text.strip() == '437' and
                      tiny_response.get('stop_reason') == 'end_turn',
            'counted_tokens': tiny_count, 'response_input_tokens': tiny_total,
            'uncached_response_input_tokens': tiny_uncached,
            'cache_read_input_tokens': tiny_cached,
            'answer': tiny_text[:120], 'stop_reason': tiny_response.get('stop_reason')}
        require_idle(base, port, api_key)
        prompt, counted, repetitions, marker = build_near_full_prompt(base, port, model, api_key)
        require_idle(base, port, api_key)
        if counted + 64 >= 262144:
            raise SafetyError(f'Near-full prompt leaves no safe output room: {counted} + 64')
        long_body = {'model': model, 'max_tokens': 64, 'thinking': {'type': 'disabled'},
                     'messages': [{'role': 'user', 'content': prompt}]}
        long_response, elapsed = run_long_request(base, port, api_key, model, long_body, sampler)
        answer = text_blocks(long_response)
        usage = long_response.get('usage', {})
        response_input = message_input_tokens(usage)
        uncached_response_input = usage.get('input_tokens') if isinstance(usage, dict) else None
        cache_read_input = usage.get('cache_read_input_tokens', 0) if isinstance(usage, dict) else None
        retrieved = marker in answer
        report['full_context'] = {
            'passed': (type(response_input) is int and response_input >= 255000 and retrieved and
                       long_response.get('stop_reason') == 'end_turn' and elapsed <= MAX_LONG_REQUEST_S),
            'target_prompt_tokens': TARGET_PROMPT_TOKENS, 'counted_prompt_tokens': counted,
            'response_input_tokens': response_input,
            'uncached_response_input_tokens': uncached_response_input,
            'cache_read_input_tokens': cache_read_input,
            'context_fill_ratio': round((response_input or 0) / 262144, 6),
            'filler_repetitions': repetitions, 'expected_tail_marker': marker,
            'marker_retrieved': retrieved, 'answer': answer[:500],
            'stop_reason': long_response.get('stop_reason'), 'elapsed_seconds': round(elapsed, 1),
            'elapsed_milliseconds': int(elapsed * 1000),
            'memory_available_min_mib': sampler.ram_min,
            'gpu_memory_peak_mib_by_gpu': sampler.gpu_peak,
            'gpu_memory_peak_aggregate_mib': sampler.aggregate_peak}
        report['full_context']['count_match'] = response_input == counted
    finally:
        sampler.stop_event.set()
        sampler.join(timeout=10)

    final_status = require_idle(base, port, api_key)
    candidate.require_prepared_manifest_unchanged(root, prepared_manifest_sha256)
    final_state = json.loads((strata_state() / 'server.json').read_text())
    final_lease = host_lifecycle.read_lease()
    if (not same_process(state['identity'], proc(state['identity']['pid'])) or
            final_state.get('identity') != state.get('identity') or
            final_state.get('profile') != pins.HOST_PROFILE or
            final_state.get('runtime_version') != pins.RUNTIME_VERSION or
            final_state.get('source_commit') != pins.SOURCE_COMMIT or
            final_state.get('hosting_lease_id') != lease_id or
            not isinstance(final_lease, dict) or final_lease.get('phase') != 'held' or
            final_lease.get('lease_id') != lease_id or
            final_status.get('context', {}).get('native') != 262144 or
            final_status.get('vision', {}).get('enabled') is not True or
            candidate.sha256(config_path) != report['config_sha256']):
        raise SafetyError('Q4 .41 server/config/HostLLM lease identity changed during native validation')

    sample_age_ms = (int((time.monotonic() - sampler.last_sample_at) * 1000)
                     if sampler.last_sample_at is not None else None)
    resource = {'ram_available_min_mib': sampler.ram_min,
                'gpu_memory_peak_mib_by_gpu': sampler.gpu_peak,
                'gpu_memory_peak_aggregate_mib': sampler.aggregate_peak,
                'sampler_samples': sampler.samples, 'last_sample_age_ms': sample_age_ms,
                'sampler_thread_stopped': not sampler.is_alive(),
                'sampler_error_count': len(sampler.errors),
                'sampler_errors': sampler.errors[:5]}
    report['resource_measurement'] = resource
    report['passed'] = bool(report.get('text_smoke', {}).get('passed') and
                            report.get('vision', {}).get('passed') and
                            report.get('token_count_control', {}).get('passed') and
                            report.get('full_context', {}).get('passed') and
                            report.get('full_context', {}).get('count_match') and
                            resource['sampler_error_count'] == 0 and resource['sampler_errors'] == [] and
                            resource['sampler_samples'] > 0 and resource['sampler_thread_stopped'] is True and
                            type(sample_age_ms) is int and
                            0 <= sample_age_ms <= 5000 and len(sampler.gpu_peak) == 4 and
                            all(type(value) is int and 0 < value <= MAX_GPU_USED_MIB for value in sampler.gpu_peak) and
                            max(sampler.gpu_peak) <= sampler.aggregate_peak <= sum(sampler.gpu_peak) and
                            sampler.ram_min is not None and
                            sampler.ram_min >= MIN_RAM_AVAILABLE_MIB)
    report['claim_scope'] = ('One controlled near-full-context tail-marker retrieval and one synthetic CPU-vision image probe; '
                             'not general quality, throughput, worst-case memory, or sustained-load qualification.')
    report['recorded_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({'passed': report['passed'], 'text_smoke': report.get('text_smoke'),
                      'vision': report['vision'], 'full_context': report['full_context'],
                      'resource_measurement': resource, 'report': str(report_path)}, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(2)


def run(root=ROOT):
    if os.geteuid() == 0:
        raise SafetyError('Q4 .41 native validation must run as the normal operator; do not use sudo')
    lock_path = candidate._profile_lock(root)
    fd = os.open(lock_path, os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SafetyError('Another Q4 .41 profile operation/validation is already active') from exc
        return _run_locked(root)


if __name__ == '__main__':
    try:
        run()
    except (SafetyError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, RuntimeError,
            urllib.error.URLError, subprocess.SubprocessError,
            http.client.HTTPException, binascii.Error) as exc:
        print(f'Q4 .41 native-context/CPU-vision validation BLOCKED: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
