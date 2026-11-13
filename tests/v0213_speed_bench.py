#!/usr/bin/env python3
"""UID1000 API-only matched speed fixtures. Never starts/stops/configures an LLM.

--prepare-fixtures uses the loaded baseline's CPU token-count endpoint, not generation.
Unique early system nonces prevent useful prefix matches; actual cache_n MUST be zero.
Request-local 128-token limits do not change the server's unlimited default.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unsloth_v31_profile as OLD
import unsloth_v31_v0159_profile as NEW

HOST = '192.168.1.69'
PORT = 8080
ROOT = Path('/home/user/.local/share/localllm-strata/calibration/strata-v0213-live-20261113')
QUESTION = ('Ignore the neutral reference material above. Give a numbered list of 30 practical troubleshooting '
            'tips for intermittent HTTP timeouts in Python. Each tip must be a specific complete sentence. '
            'Start directly at tip 1, without an introduction or conclusion.')
SAMPLING = {'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 719,
            'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False},
            'experimental_speed_projection': True, 'max_tokens': 128}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def api(path, data=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    try:
        conn.request('GET' if data is None else 'POST', path,
                     body=None if data is None else encoded(data),
                     headers={'Content-Type': 'application/json'} if data is not None else {})
        reply = conn.getresponse()
        body = reply.read()
        if reply.status != 200:
            raise RuntimeError(f'{path}: HTTP {reply.status}: {body[:256]!r}')
        return json.loads(body)
    finally:
        conn.close()


def target(version):
    model = OLD.MODEL_ID if version == '0.1.46' else NEW.MODEL_ID
    status = api('/v1/status')
    if not (status.get('loaded') is True and status.get('engine') == version
            and status.get('model') == model
            and status.get('context', {}).get('native') == 262144
            and status.get('context', {}).get('max_positions') == 262144
            and status.get('concurrency', {}).get('serving') == 2):
        raise RuntimeError('Exact loaded runtime/model/native262K/two-slot target differs; no generation sent')
    return status


def running(metrics):
    value = metrics.get('live', {}).get('running')
    if type(value) is not int or value < 0:
        raise RuntimeError('Unknown request occupancy; no generation sent')
    return value


def idle(version, seconds=300):
    deadline = time.monotonic() + seconds
    while True:
        target(version)
        if running(api('/metrics')) == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Other requests remain active; no cancellation or generation sent')
        time.sleep(1)


def token_count(messages, version):
    """These text-only Anthropic/OpenAI messages render identically with thinking disabled."""
    model = OLD.MODEL_ID if version == '0.1.46' else NEW.MODEL_ID
    body = {'model': model, 'system': messages[0]['content'], 'messages': messages[1:],
            'thinking': {'type': 'disabled'}}
    return int(api('/v1/messages/count_tokens', body)['input_tokens'])


def make_messages(label, body):
    nonce = hashlib.sha256(('v0213-matched-speed-20261113/' + label).encode()).hexdigest()
    return [{'role': 'system', 'content': nonce + '\nFollow the final user instruction. Neutral reference text is not an instruction.'},
            {'role': 'user', 'content': body}]


def fit(label, desired, corpus):
    head = 'Neutral reference material follows.\n'
    tail = '\nEnd reference material.\n\n' + QUESTION
    lo, hi = 0, len(corpus)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = token_count(make_messages(label, head + corpus[:mid] + tail), '0.1.46')
        if n <= desired - 24:
            lo = mid
        else:
            hi = mid - 1
    base = head + corpus[:lo] + '\nPadding:'
    padding = 0
    for _ in range(12):
        messages = make_messages(label, base + ' q' * padding + tail)
        n = token_count(messages, '0.1.46')
        if n == desired:
            return {'name': label, 'messages': messages, 'input_tokens': n,
                    'messages_sha256': digest(messages)}
        padding += desired - n
        if padding < 0:
            raise RuntimeError('Padding underflow; no inference sent')
    raise RuntimeError('Exact CPU token-count fit failed; no inference sent')


def prepare():
    idle('0.1.46')
    rng = random.Random(20261113)
    colors = ['silver', 'amber', 'cobalt', 'green', 'violet', 'white', 'blue']
    nouns = ['harbor', 'meadow', 'station', 'orchard', 'workshop', 'library', 'garden']
    corpus = ''.join(f'Archive row {i:05d}: the {rng.choice(colors)} {rng.choice(nouns)} holds '
                     f'{rng.randrange(10, 9999)} neutral samples; record label {rng.randrange(10000, 99999)}.\n'
                     for i in range(9000))
    cases = []
    for rep in range(1, 4):
        for kind, count in [('short', 256), ('half-context', 131072)]:
            case = fit(f'{kind}-r{rep}', count, corpus)
            case.update(kind=kind, repetition=rep)
            cases.append(case)
            print(f'Frozen CPU-counted {case["name"]}: {case["input_tokens"]} input tokens', flush=True)
    fixture = {'schema': 1, 'context_per_chat': 262144, 'output_cap': 128, 'sampling': SAMPLING,
               'scope': 'single sequential request; not two-user throughput or full-context capacity',
               'cache_policy': 'unique early system nonce per case; reject any actual KV reuse', 'cases': cases}
    save(ROOT / 'fixtures.json', fixture)
    print('Saved frozen fixtures SHA256=' + digest(fixture), flush=True)


def stream(messages, version, output_dir, name, abort_file, max_tokens=128):
    idle(version)
    model = OLD.MODEL_ID if version == '0.1.46' else NEW.MODEL_ID
    payload = {**SAMPLING, 'model': model, 'messages': messages, 'max_tokens': max_tokens,
               'stream': True, 'stream_options': {'include_usage': True}}
    raw_body = encoded(payload)
    conn = http.client.HTTPConnection(HOST, PORT, timeout=600)
    done = threading.Event()
    observed = {'max_running': 0, 'monitor_errors': [], 'aborted': None}

    def monitor():
        while not done.is_set():
            try:
                if abort_file.exists():
                    observed['aborted'] = abort_file.read_text()[:512]
                    if conn.sock is not None:
                        conn.sock.shutdown(socket.SHUT_RDWR)  # only this client's connection
                    return
                observed['max_running'] = max(observed['max_running'], running(api('/metrics')))
            except Exception as e:
                observed['monitor_errors'].append(str(e))
            done.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    events, content, reasoning, usage, timings = [], [], [], {}, {}
    first = last = None
    finish = None
    complete = False
    started = time.perf_counter()
    try:
        if abort_file.exists():
            raise RuntimeError('Safety abort present; no generation sent')
        conn.request('POST', '/v1/chat/completions', body=raw_body, headers={'Content-Type': 'application/json'})
        reply = conn.getresponse()
        if reply.status != 200:
            raise RuntimeError(f'Generation HTTP {reply.status}: {reply.read(256)!r}')
        while True:
            line = reply.readline()
            if not line:
                break
            if not line.startswith(b'data: '):
                continue
            text = line[6:].strip()
            if text == b'[DONE]':
                complete = True
                break
            event = json.loads(text)
            events.append(event)
            if event.get('error'):
                raise RuntimeError(str(event['error']))
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                if c or r:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    content.append(c)
                    reasoning.append(r)
                finish = choice.get('finish_reason') or finish
        elapsed = time.perf_counter() - started
    finally:
        done.set()
        conn.close()
        thread.join(timeout=62)
        save(output_dir / (name + '-sse.json'), events)
    if observed['aborted']:
        raise RuntimeError('Own request aborted by fresh safety monitor: ' + observed['aborted'])
    if not complete or first is None or not usage or not timings:
        raise RuntimeError('Incomplete SSE/output/usage/native clock; sample invalid')
    result = {'name': name, 'messages_sha256': digest(messages), 'body_sha256': hashlib.sha256(raw_body).hexdigest(),
              'usage': usage, 'timings': timings, 'finish_reason': finish,
              'ttft_s': first - started, 'wall_s': elapsed,
              'client_decode_tokens_s': ((usage['completion_tokens'] - 1) / (last - first)
                                        if last is not None and last > first else None),
              'content': ''.join(content), 'reasoning': ''.join(reasoning), **observed}
    save(output_dir / (name + '.json'), result)
    if observed['max_running'] > 1 or observed['monitor_errors']:
        raise RuntimeError('Concurrent activity or unknown occupancy; sample saved but invalid, no next request sent')
    if timings.get('cache_n') != 0:
        raise RuntimeError('KV reuse observed; sample saved but invalid, no next request sent')
    if not result['content'].strip() or result['reasoning']:
        raise RuntimeError('Expected direct nonempty no-thinking answer differs')
    return result


def run(version, abort_file):
    if version == '0.1.59':
        NEW.authorize_activation()  # real permission BEFORE any HTTP request
    status = target(version)
    fixture = json.loads((ROOT / 'fixtures.json').read_text())
    if fixture['sampling'] != SAMPLING or fixture['output_cap'] != 128:
        raise RuntimeError('Frozen benchmark policy differs')
    out = ROOT / version
    out.mkdir(mode=0o700, exist_ok=False)
    warm = make_messages('untimed-known-working-arithmetic', 'What is 7 times 9? Reply only with the number.')
    expected_warm = token_count(warm, version)
    result = stream(warm, version, out, 'untimed-known-working', abort_file, max_tokens=16)
    if result['content'].strip() != '63' or result['usage']['prompt_tokens'] != expected_warm:
        raise RuntimeError('Known-working reply / count-to-generation template equivalence failed')
    results = []
    for case in fixture['cases']:
        if digest(case['messages']) != case['messages_sha256']:
            raise RuntimeError('Frozen message hash differs')
        idle(version)
        if token_count(case['messages'], version) != case['input_tokens']:
            raise RuntimeError('Cross-runtime prompt token count differs; no generation sent')
        sample = stream(case['messages'], version, out, case['name'], abort_file)
        if sample['usage']['prompt_tokens'] != case['input_tokens'] or sample['usage']['completion_tokens'] != 128:
            raise RuntimeError('Actual input/output lengths differ; sample invalid')
        sample.update(kind=case['kind'], repetition=case['repetition'])
        results.append(sample)
        print(f'{version} {case["name"]}: prefill={sample["timings"]["prompt_per_second"]} tok/s; '
              f'decode={sample["timings"]["predicted_per_second"]} tok/s; TTFT={sample["ttft_s"]:.3f}s; '
              f'input={case["input_tokens"]}; output=128; reused=0', flush=True)
    medians = {}
    for kind in ('short', 'half-context'):
        selected = [r for r in results if r['kind'] == kind]
        medians[kind] = {'prefill_tokens_s': statistics.median(r['timings']['prompt_per_second'] for r in selected),
                         'decode_tokens_s': statistics.median(r['timings']['predicted_per_second'] for r in selected),
                         'ttft_s': statistics.median(r['ttft_s'] for r in selected),
                         'client_decode_tokens_s': statistics.median(r['client_decode_tokens_s'] for r in selected),
                         'wall_s': statistics.median(r['wall_s'] for r in selected)}
    report = {'phase': 'COMPLETE', 'runtime': version, 'model': status['model'], 'fixture_sha256': digest(fixture),
              'sampling': SAMPLING, 'repetitions': 3, 'input_tokens': {'short': 256, 'half-context': 131072},
              'output_tokens_each': 128, 'all_actual_kv_reuse_zero': True, 'all_samples_isolated': True,
              'medians': medians, 'samples': results,
              'limitations': ['current .46 long-lived warm/adaptive state vs separate .59 generic seed; neither reset nor copied',
                              'normal profile runtime/cache warm-up, not cold disk/page-cache performance',
                              'prepared-profile comparison, including reviewed scheduling/kernel environment and auto split',
                              'single active request, not concurrent throughput/full-context/stability/quality qualification']}
    save(out / 'results.json', report)
    print(json.dumps({'runtime': version, 'medians': medians}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-fixtures', action='store_true')
    mode.add_argument('--run', choices=('0.1.46', '0.1.59'))
    parser.add_argument('--abort-file', type=Path, default=ROOT / 'SAFETY-ABORT')
    args = parser.parse_args()
    if os.geteuid() != 1000:
        parser.error('API-only benchmark must run as UID1000')
    os.umask(0o077)
    if args.prepare_fixtures:
        prepare()
    else:
        run(args.run, args.abort_file)


if __name__ == '__main__':
    main()
