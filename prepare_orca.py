#!/usr/bin/env python3
"""Explicit CPU/download-only Orca preparation. Reuses (never rebuilds) the live Strata runtime."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from engine_safety import SafetyError, docker_empty, owner_home, proc, same_process, strata_state
from orca_assets import ASSETS, MODEL_ID, PROFILE, REPOSITORY, REVISION
from prepare_strata import PIN

BASE = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
SOURCE = BASE / 'source-99f3dbd0b21d'
DEST = BASE / 'data/models' / PROFILE
PACK = BASE / 'data/packs' / PROFILE
PROFILE_ROOT = BASE / 'profiles' / PROFILE


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


class PrivateRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward a Hugging Face bearer token to a CDN or another host."""
    def redirect_request(self, request, fp, code, msg, headers, url):
        if urllib.parse.urlsplit(url).scheme != 'https':
            raise SafetyError('Refusing a non-HTTPS model redirect')
        redirected = super().redirect_request(request, fp, code, msg, headers, url)
        if urllib.parse.urlsplit(url).hostname != 'huggingface.co':
            redirected.remove_header('Authorization')
        return redirected


def hf_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token.strip()
    paths = [owner_home() / '.cache/huggingface/token', owner_home() / '.huggingface/token']
    if os.environ.get('HF_TOKEN_PATH'):
        paths.insert(0, Path(os.environ['HF_TOKEN_PATH']))
    for p in paths:
        if p.is_file():
            return p.read_text().strip()
    return None


def download(name, size, sha):
    """Single-stream, resumable, full-file verified; never overwrite a completed mismatching asset."""
    path = DEST / name
    if path.exists():
        if path.stat().st_size != size or digest(path) != sha:
            raise SafetyError(f'Existing Orca asset differs from its pin; not overwritten: {path}')
        print(f'Already verified: {name}', flush=True)
        return path
    part = path.with_name(path.name + '.part')
    if part.exists() and part.stat().st_size > size:
        raise SafetyError(f'Oversized partial download; not overwritten: {part}')
    url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
    for attempt in range(10):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            break
        headers = {'User-Agent': 'LocalLLM-install-only/1', 'Accept-Encoding': 'identity'}
        token = hf_token()
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url + f'?download=true&nonce={time.time_ns()}', headers=headers)
        try:
            with urllib.request.build_opener(PrivateRedirect()).open(request, timeout=60) as response:
                if offset:
                    content_range = response.headers.get('Content-Range', '')
                    if response.status != 206 or not content_range.startswith(f'bytes {offset}-') or \
                            not content_range.endswith(f'/{size}'):
                        raise SafetyError('Server did not honor the pinned-file resume range; partial retained')
                elif response.status not in (200, 206):
                    raise SafetyError(f'Unexpected download response: {response.status}')
                with part.open('ab') as out:
                    count, report = offset, time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        if count + len(block) > size:
                            raise SafetyError('Download exceeds pinned size; partial retained')
                        out.write(block); count += len(block)
                        if time.monotonic() - report > 15:
                            print(f'{name}: {count / size:.1%} ({count / 1e9:.2f}/{size / 1e9:.2f} GB)', flush=True)
                            report = time.monotonic()
                    out.flush(); os.fsync(out.fileno())
                if count != size:
                    raise OSError(f'Truncated download: {count}/{size}')
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SafetyError('Hugging Face gated access is required. Accept access on the Orca model page and '
                                  'configure HF_TOKEN or the owner\'s private ~/.cache/huggingface/token; '
                                  'no credentials are logged or saved in the repository.') from None
            if attempt == 9:
                raise SafetyError(f'Pinned model download failed (HTTP {exc.code}); partial retained') from None
            print(f'Download HTTP {exc.code}; resuming in 5s.', flush=True)
            time.sleep(5)
        except (OSError, TimeoutError) as exc:
            if attempt == 9:
                raise
            print(f'Download interrupted ({type(exc).__name__}); resuming in 5s.', flush=True)
            time.sleep(5)
    if not part.is_file() or part.stat().st_size != size:
        raise SafetyError(f'Incomplete download: {name}')
    print(f'Full-file SHA-256: {name}', flush=True)
    if digest(part) != sha:
        raise SafetyError(f'Checksum mismatch; partial retained, never promoted: {part}')
    os.replace(part, path)
    return path


def capture_live():
    p = strata_state() / 'server.json'
    state = json.loads(p.read_text()) if p.exists() else None
    if state and not same_process(state['identity'], proc(state['identity']['pid'])):
        raise SafetyError('Existing Strata state is not coherent; preparation will not try to repair it')
    protected = [BASE / 'prepared.json', SOURCE / 'strata-iq3_s.json', strata_state() / 'run-config.json',
                 owner_home() / '.local/state/hostllm/pause.json', Path('/hive-config/rig.conf'),
                 Path('/hive-config/wallet.conf'), Path('/hive-config/watchdog.conf')]
    return {'identity': state['identity'] if state else None,
            'protected': {str(p): digest(p) for p in protected if p.is_file()},
            'markers': {n: (Path('/run/hive') / n).exists() for n in ['MINER_RUN', 'MINER_STOP']},
            'osn': subprocess.check_output(['systemctl', 'show', 'osn.service', '--property=ActiveState', '--value'],
                                          text=True).strip()}


def assert_preserved(before):
    if before['identity'] and not same_process(before['identity'], proc(before['identity']['pid'])):
        raise SafetyError('Original Strata instance changed during preparation; nothing will be restarted')
    for name, sha in before['protected'].items():
        if not Path(name).is_file() or digest(Path(name)) != sha:
            raise SafetyError(f'Protected configuration changed during preparation: {name}')
    now = {n: (Path('/run/hive') / n).exists() for n in ['MINER_RUN', 'MINER_STOP']}
    osn = subprocess.check_output(['systemctl', 'show', 'osn.service', '--property=ActiveState', '--value'], text=True).strip()
    if now != before['markers'] or osn != before['osn']:
        raise SafetyError('Service/marker state changed externally; preparation did not restore or change it')


def main():
    if sys.argv[1:] not in ([], ['--download-only']):
        raise SafetyError('Only --download-only is accepted by this install-only helper')
    docker_empty()
    actual = subprocess.check_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN:
        raise SafetyError('Existing Strata source pin differs; no update/rebuild is allowed')
    python = SOURCE / '.venv/bin/python'
    if not python.is_file():
        raise SafetyError('Prepare the original Strata runtime first; no package installation is permitted here')
    existing = json.loads((BASE / 'prepared.json').read_text())
    if existing['source_commit'] != PIN:
        raise SafetyError('Original runtime manifest pin mismatch')
    for r in existing['runtime_assets']:
        p = Path(r['path'])
        if p.stat().st_size != r['bytes'] or p.stat().st_mtime_ns != r['mtime_ns'] or digest(p) != r['sha256']:
            raise SafetyError('Existing runtime changed; it will not be replaced')
    before = capture_live()
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(PROFILE_ROOT / 'preparation-before.json', before)
    DEST.mkdir(parents=True, exist_ok=True)
    missing = sum(size for name, (size, _) in ASSETS.items() if not (DEST / name).exists())
    if shutil.disk_usage(BASE).free < missing + 15 * 1024**3:
        raise SafetyError('Insufficient free space for pinned assets, packing and safety margin')
    paths = [download(name, size, sha) for name, (size, sha) in ASSETS.items()]
    assert_preserved(before)
    if '--download-only' in sys.argv:
        print('DOWNLOADS VERIFIED. No inference, runtime update or service operation was performed.', flush=True)
        return
    docker_empty()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    # Work in a NEW pack only: no --base or experts.bin; no original dense/tokenizer reuse.
    if not PACK.exists():
        pending = PACK.with_name(PACK.name + '.preparing')
        if pending.exists():
            raise SafetyError(f'An interrupted pack exists; inspect it rather than overwrite: {pending}')
        subprocess.run([str(python), str(SOURCE / 'tools/iq_pack.py'), '--gguf', str(paths[0]), '--out', str(pending),
                        '--compat-bf16'], cwd=SOURCE, env=env, check=True)
        for rel in ['dense.bin', 'index.txt', 'native_experts.txt', 'tokenizer/tokenizer.json',
                    'tokenizer/chat_template.jinja', 'compat-bf16.json']:
            if not (pending / rel).is_file():
                raise SafetyError(f'Orca pack incomplete: {rel}')
        os.rename(pending, PACK)
    else:
        # Only reuse a pack bound by an earlier completed preparation, never a same-named foreign pack.
        previous = PROFILE_ROOT / 'prepared.json'
        if not previous.is_file():
            raise SafetyError('Existing Orca pack has no preparation provenance; inspect it instead of reusing it')
        prior = json.loads(previous.read_text())
        if prior.get('model_repository') != REPOSITORY or prior.get('model_revision') != REVISION:
            raise SafetyError('Existing Orca pack belongs to another model/revision')
        records = prior.get('pack_assets', [])
        if not records:
            raise SafetyError('Existing Orca pack has no recorded hashes')
        for record in records:
            p = Path(record['path'])
            if not p.resolve().is_relative_to(PACK.resolve()) or not p.is_file() or \
                    p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
                raise SafetyError('Existing Orca pack changed; it is never overwritten/rebound')
    # Fresh offline checks without the upstream verifier's shared verified.json stamp writes.
    pins = {'__name__':'orca_offline_mtp_pins', '__file__':str(SOURCE / 'tools/mtp_fetch.py')}
    exec(compile((SOURCE / 'tools/mtp_fetch.py').read_text(), pins['__file__'], 'exec'), pins)
    for name, expected_sha in pins['SHA256'].items():
        p = BASE / 'data/mtp/tensors' / (name + '.bin')
        if not p.is_file() or digest(p) != expected_sha:
            raise SafetyError(f'Original pinned Flash-Next MTP tensor missing or changed: {name}')
    print(f'Original MTP: {len(pins["SHA256"])} full tensor hashes verified offline; no shared stamp writes.', flush=True)
    mtp = BASE / 'data/mtp/rt'
    if not (mtp / 'experts.bin').is_file():
        raise SafetyError('Prepared original Flash-Next MTP runtime missing')
    config = {'exe': str(SOURCE / 'engine/strata'), 'args': [
        '--pack', str(PACK), '--native', str(paths[0]), '--ple-gguf', str(paths[0]),
        '--expert-profile', str(SOURCE / 'data/expert-profile.bin'), '--expert-cache', 'auto', '--prefill', '512',
        '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(mtp), '--max-context', '32768', '--kv', 'int8',
        '--vision', '--vram-reserve-mib', '2048'],
        'cwd': str(SOURCE), 'tokenizer': str(PACK / 'tokenizer'), 'gpu': [0,1,2,3], 'layer_split': 'auto',
        'model_name': MODEL_ID, 'aliases': ['orca', 'orca-strata'], 'host': '0.0.0.0', 'port': 8080,
        'vision': {'exe': str(SOURCE / 'engine/strata-vision'), 'model': str(paths[0]), 'mmproj': str(paths[2]),
                   'gpu': True, 'max_tokens': 1024},
        'lib_dirs': ['/usr/local/cuda-12.9/bin', '/usr/local/cuda-12.9/lib64'],
        'sampling': {'temperature':1.0,'top_p':0.95,'top_k':20,'experimental_speed_projection':False}}
    atomic_json(PROFILE_ROOT / 'config.json', config)
    # These are integrity checks, NOT a claim that the actual model/vision has run on the GPUs.
    records = [{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':ASSETS[p.name][1]}
               for p in paths]
    pack_records = []
    for p in sorted(PACK.rglob('*')):
        if p.is_file():
            pack_records.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest(p)})
    assert_preserved(before)
    atomic_json(PROFILE_ROOT / 'prepared.json', {'profile':PROFILE,'source_commit':PIN,'model_repository':REPOSITORY,
        'model_revision':REVISION,'assets':records,'pack_assets':pack_records,
        'runtime_assets':existing['runtime_assets'],'config':str(PROFILE_ROOT / 'config.json'),
        'inference_tested':False,'vision_configured':True,'vision_inference_tested':False,
        'initial_context':32768,'compat_bf16':True})
    print('ORCA PREPARED: assets/pack verified, original runtime and running Strata preserved. GPU inference/vision UNTESTED.',
          flush=True)


if __name__ == '__main__':
    try:
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        with (PROFILE_ROOT / 'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except (SafetyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'PREPARATION BLOCKED: {exc}', file=sys.stderr)
        raise SystemExit(1)
