#!/usr/bin/env python3
"""Run the bounded Q4 .43 32K live gate under its identity-bound HostLLM lease."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

from engine_safety import SafetyError, owner_home, strata_state
from local_api_client import open_local_api
import orca_q4ks_v0143_assets as pins
import orca_q4ks_v0143_profile as candidate


ROOT = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))


def request_json(base, path, body=None, timeout=180, api_key=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(base + path, data=data, headers=headers)
    with open_local_api(request, timeout=timeout) as response:
        return json.load(response)


def established_clients(port):
    output = subprocess.check_output(
        ['ss', '-Htnp', 'state', 'established', f'( sport = :{port} )'], text=True)
    return [line for line in output.splitlines() if line.strip()]


def require_idle(base, port, api_key):
    if established_clients(port):
        raise SafetyError('An API client is already connected; leaving it untouched')
    status = request_json(base, '/v1/status', timeout=8, api_key=api_key)
    activity = status.get('activity')
    # Strata 0.1.43 folds busy and queued work into in_flight; check a separate
    # queued counter only on compatible APIs that actually expose it.
    if (not isinstance(activity, dict) or type(activity.get('in_flight')) is not int or
            activity['in_flight'] != 0 or
            ('queued' in activity and (type(activity['queued']) is not int or activity['queued'] != 0))):
        raise SafetyError('API reports an active/queued request; leaving it untouched')
    deadline = time.monotonic() + 2
    while established_clients(port) and time.monotonic() < deadline:
        time.sleep(0.05)
    if established_clients(port):
        raise SafetyError('Status connection remained open; no validation request was sent')
    return status


class MemorySampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.per_gpu_peak = []
        self.aggregate_peak_mib = 0
        self.ram_available_min_mib = None
        self.errors = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    text=True, stderr=subprocess.STDOUT, timeout=5)
                current = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if len(current) != 4:
                    raise RuntimeError(f'Expected four GPU memory readings, received {len(current)}')
                if not self.per_gpu_peak:
                    self.per_gpu_peak = current[:]
                else:
                    self.per_gpu_peak = [max(old, new) for old, new in zip(self.per_gpu_peak, current)]
                self.aggregate_peak_mib = max(self.aggregate_peak_mib, sum(current))
                meminfo = Path('/proc/meminfo').read_text()
                available_kib = next(int(line.split()[1]) for line in meminfo.splitlines()
                                     if line.startswith('MemAvailable:'))
                available_mib = available_kib // 1024
                self.ram_available_min_mib = (available_mib if self.ram_available_min_mib is None
                                               else min(self.ram_available_min_mib, available_mib))
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop_event.wait(0.5)


def _completion(base, body, api_key):
    response = request_json(base, '/v1/chat/completions', body, timeout=240, api_key=api_key)
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        raise SafetyError('Completion returned no choices')
    return response, choices[0]


def _conversation_cache_metrics(metrics):
    if not isinstance(metrics, dict):
        raise SafetyError('/metrics did not return a JSON object')
    cache = metrics.get('conversation_cache')
    if (not isinstance(cache, dict) or cache.get('enabled') is not True or
            type(cache.get('budget_mib')) is not int or cache['budget_mib'] != 8192 or
            type(cache.get('slots')) is not int or cache['slots'] != 4):
        raise SafetyError('/metrics does not report the enabled 8192 MiB/four-slot conversation cache')
    counters = ('parks', 'restores', 'requests_reused', 'reused_tokens')
    if any(type(cache.get(key)) is not int or cache[key] < 0 for key in counters):
        raise SafetyError('/metrics conversation-cache park/restore/reuse counters are missing or invalid')
    return {'enabled': True, 'budget_mib': 8192, 'slots': 4,
            'parks': cache['parks'], 'restores': cache['restores'],
            'requests_reused': cache['requests_reused'], 'reused_tokens': cache['reused_tokens']}


def _probe_conversation_cache(base, port, base_body, api_key, initial_metrics):
    before = _conversation_cache_metrics(initial_metrics)
    histories = {
        'A': [{'role': 'user', 'content':
               'Remember the private word SILVER for conversation A. Reply with exactly READY.'}],
        'B': [{'role': 'user', 'content':
               'Remember the private word ORANGE for conversation B. Reply with exactly READY.'}],
    }
    for key in ('A', 'B'):
        require_idle(base, port, api_key)
        _, choice = _completion(base, dict(base_body, messages=histories[key]), api_key)
        text = choice.get('message', {}).get('content') or ''
        if (not isinstance(text, str) or not text.strip() or
                text.strip().strip(' .,!?:;').upper() != 'READY' or choice.get('finish_reason') != 'stop'):
            raise SafetyError(f'Conversation-cache probe did not finish the expected short reply for {key}')
        histories[key].append({'role': 'assistant', 'content': text})
        require_idle(base, port, api_key)

    for key, expected_word in (('A', 'SILVER'), ('B', 'ORANGE')):
        require_idle(base, port, api_key)
        messages = histories[key] + [{'role': 'user', 'content':
                                      'What private word did I ask you to remember? Reply with only the word.'}]
        _, choice = _completion(base, dict(base_body, messages=messages), api_key)
        text = choice.get('message', {}).get('content') or ''
        if (not isinstance(text, str) or choice.get('finish_reason') != 'stop' or
                text.strip().strip(' .,!?:;').upper() != expected_word):
            raise SafetyError(f'Conversation {key} did not restore its private word')
        require_idle(base, port, api_key)

    after = _conversation_cache_metrics(request_json(base, '/metrics', timeout=8, api_key=api_key))
    if after['budget_mib'] != before['budget_mib'] or after['slots'] != before['slots']:
        raise SafetyError('Conversation-cache configuration changed during the alternating-conversation probe')
    parks = after['parks'] - before['parks']
    restored = after['restores'] - before['restores']
    reused_requests = after['requests_reused'] - before['requests_reused']
    reused_tokens = after['reused_tokens'] - before['reused_tokens']
    if parks < 1 or restored < 1 or reused_requests < 1 or reused_tokens <= 0:
        raise SafetyError('A/B/A/B did not park and restore a conversation and reuse prompt tokens according to /metrics')
    return {'enabled': True, 'budget_mib': 8192, 'slots': 4,
            'park_events': parks, 'restored_conversations': restored,
            'reused_requests': reused_requests, 'reused_tokens': reused_tokens}


def _arithmetic_pass(text, finish_reason):
    return isinstance(text, str) and text.strip() == '437' and finish_reason == 'stop'


def _record_mtp(response):
    timings = response.get('timings') or {}
    drafted, accepted = timings.get('draft_n'), timings.get('draft_n_accepted')
    if (type(drafted) is not int or type(accepted) is not int or drafted < 0 or
            not 0 <= accepted <= drafted):
        raise SafetyError(f'Original Flash-Next MTP counters are missing/invalid: {timings!r}')
    return drafted, accepted


@contextmanager
def _validation_lock():
    state_root = strata_state()
    state_root.mkdir(parents=True, exist_ok=True)
    path = state_root / 'q4ks-v0143-validation.lock'
    if path.is_symlink():
        raise SafetyError('Q4 .43 validation lock must not be a symlink')
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SafetyError('Another Q4 .43 validation is already active') from exc
        yield


def _run_locked(root):
    source, config_path, config = candidate.ready_q4ks_v0143(root)
    context = config['args'][config['args'].index('--max-context') + 1]
    if context != '32768' or config.get('vision') is not None or '--vision' in config['args']:
        raise SafetyError('The candidate gate is only for the initial 32K text-only profile')
    live = candidate._assert_live_candidate(root)
    port = live['port']
    run_config = json.loads(Path(live['config']).read_text())
    api_key = run_config.get('api_key')
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise SafetyError('Candidate run-config API key is malformed')
    base = f'http://127.0.0.1:{port}'
    require_idle(base, port, api_key)
    health = request_json(base, '/health', timeout=8, api_key=api_key)
    if (health.get('status') != 'ok' or health.get('model') != pins.MODEL_ID or
            health.get('max_context') != 32768 or health.get('images') is not False or
            health.get('loaded') is not True):
        raise SafetyError(f'Unexpected candidate health/profile response: {health!r}')
    status = require_idle(base, port, api_key)
    if (status.get('model') != pins.MODEL_ID or status.get('context', {}).get('native') != 32768 or
            status.get('vision', {}).get('enabled') is not False or
            status.get('concurrency', {}).get('serving') != 1):
        raise SafetyError('Live status is not the exact Q4 .43 single-slot 32K text-only candidate')
    cache_initial_metrics = request_json(base, '/metrics', timeout=8, api_key=api_key)
    _conversation_cache_metrics(cache_initial_metrics)

    base_body = {'model': pins.MODEL_ID, 'temperature': 0, 'max_tokens': 128,
                 'chat_template_kwargs': {'enable_thinking': False}}
    checks = {'health': True, 'conversation_cache_configuration': True}
    drafted = accepted = 0
    sampler = MemorySampler()
    sampler.start()
    try:
        require_idle(base, port, api_key)
        response, choice = _completion(base, dict(base_body, messages=[{'role': 'user',
            'content': 'Compute 19 * 23. Reply with only the integer.'}]), api_key)
        text = choice.get('message', {}).get('content') or ''
        checks['arithmetic'] = _arithmetic_pass(text, choice.get('finish_reason'))
        if not checks['arithmetic']:
            raise SafetyError(f'Arithmetic validation failed: {text[:160]!r}')
        require_idle(base, port, api_key)
        d, a = _record_mtp(response)
        drafted += d
        accepted += a

        require_idle(base, port, api_key)
        response, choice = _completion(base, dict(base_body, response_format={'type': 'json_object'},
            messages=[{'role': 'user', 'content':
                       'Return a JSON object with exactly one key "answer" and integer value 437.'}]), api_key)
        text = choice.get('message', {}).get('content') or ''
        parsed = json.loads(text)
        checks['strict_json'] = (isinstance(parsed, dict) and parsed == {'answer': 437} and
                                 choice.get('finish_reason') == 'stop')
        if not checks['strict_json']:
            raise SafetyError(f'Strict JSON validation failed: {text[:200]!r}')
        require_idle(base, port, api_key)
        d, a = _record_mtp(response)
        drafted += d
        accepted += a

        require_idle(base, port, api_key)
        tool = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'},
            'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}
        response, choice = _completion(base, dict(base_body, tools=[tool], tool_choice={
            'type': 'function', 'function': {'name': 'add'}},
            messages=[{'role': 'user', 'content': 'Call add with a=19 and b=23. Do not answer directly.'}]), api_key)
        calls = choice.get('message', {}).get('tool_calls') or []
        if len(calls) != 1 or calls[0].get('function', {}).get('name') != 'add':
            raise SafetyError('Tool-call validation did not produce one add() call')
        arguments = json.loads(calls[0]['function']['arguments'])
        checks['tool_call'] = arguments == {'a': 19, 'b': 23}
        if not checks['tool_call']:
            raise SafetyError(f'Tool-call arguments were incorrect: {arguments!r}')
        require_idle(base, port, api_key)
        d, a = _record_mtp(response)
        drafted += d
        accepted += a

        # Take the counter baseline immediately before A/B/A/B so earlier smoke probes cannot satisfy this gate.
        cache_probe_metrics = request_json(base, '/metrics', timeout=8, api_key=api_key)
        _conversation_cache_metrics(cache_probe_metrics)
        cache_report = _probe_conversation_cache(base, port, base_body, api_key, cache_probe_metrics)
        checks['conversation_reuse'] = (cache_report['park_events'] >= 1 and
                                        cache_report['restored_conversations'] >= 1 and
                                        cache_report['reused_requests'] >= 1 and
                                        cache_report['reused_tokens'] > 0)
        if not checks['conversation_reuse']:
            raise SafetyError('A/B/A/B did not park and restore a history while reusing prompt tokens')
    finally:
        sampler.stop_event.set()
        sampler.join(timeout=8)

    if sampler.is_alive() or sampler.errors or not sampler.per_gpu_peak or any(x <= 0 for x in sampler.per_gpu_peak):
        raise SafetyError('GPU/RAM sampling did not complete cleanly: ' + '; '.join(sampler.errors[-3:]))
    if drafted <= 0 or not 0 <= accepted <= drafted:
        raise SafetyError('Original Flash-Next MTP did not record valid drafted/accepted counters')
    final_status = require_idle(base, port, api_key)
    live_end = candidate._assert_live_candidate(root)
    if (live_end.get('identity') != live.get('identity') or
            live_end.get('hosting_lease_id') != live.get('hosting_lease_id')):
        raise SafetyError('Candidate process/HostLLM lease changed during the validation window')
    manifest, *_ = candidate._prepared(root)
    engine_sha = manifest['runtime_assets'][0]['sha256']
    report = {
        'profile': pins.PROFILE, 'model_id': pins.MODEL_ID, 'source_commit': pins.SOURCE_COMMIT,
        'runtime_version': pins.RUNTIME_VERSION, 'runtime_sha256': engine_sha,
        'asset_sha256': {name: digest for name, (_, digest) in pins.ASSETS.items()},
        'config_sha256': candidate.sha256(config_path), 'context': 32768, 'parallel': 1,
        'text_only': True, 'passed': True,
        'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
        'checks': {'health': True, **checks, 'memory_recorded': True},
        'memory_peak_mib': max(sampler.per_gpu_peak),
        'gpu_memory_peak_mib_by_gpu': sampler.per_gpu_peak,
        'gpu_memory_peak_aggregate_mib': sampler.aggregate_peak_mib,
        'ram_available_min_mib': sampler.ram_available_min_mib,
        'mtp': {'drafted_tokens': drafted, 'accepted_tokens': accepted},
        'live_status': {'context': final_status['context']['native'],
                        'vision_enabled': final_status['vision']['enabled'],
                        'serving_slots': final_status['concurrency']['serving'],
                        'conversation_cache': cache_report},
        'validation_runner_sha256': candidate.sha256(Path(__file__).resolve()),
        'local_api_client_sha256': candidate.sha256(Path(__file__).with_name('local_api_client.py')),
        'validation_server_identity': live['identity'],
        'validation_hosting_lease_id': live['hosting_lease_id'],
        'api_key_enabled': bool(api_key),
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    candidate._validation_report(root, report, candidate.sha256(config_path), manifest)
    target = candidate.profile_root(root) / 'validation-32k.json'
    if target.exists() or target.is_symlink():
        raise SafetyError('Q4 .43 32K evidence already exists; inspect rather than overwrite')
    candidate.atomic_json(target, dict(report, recorded_server_identity=live_end['identity'],
                                       recorded_server_config=live_end['config'],
                                       recorded_hosting_lease_id=live_end['hosting_lease_id'],
                                       recorded_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(json.dumps({'checks': report['checks'], 'mtp': report['mtp'],
                      'memory_peak_mib_by_gpu': report['gpu_memory_peak_mib_by_gpu'],
                      'aggregate_peak_mib': report['gpu_memory_peak_aggregate_mib'],
                      'evidence_class': report['evidence_class'],
                      'cryptographic_attestation': report['cryptographic_attestation'],
                      'report': str(target)}, indent=2))


def run(root=ROOT):
    if os.geteuid() == 0:
        raise SafetyError('Q4 .43 validation must run as the normal operator; do not use sudo')
    with _validation_lock():
        return _run_locked(root)


if __name__ == '__main__':
    try:
        run()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Q4 .43 32K validation BLOCKED: {exc}', file=__import__('sys').stderr)
        raise SystemExit(1)
