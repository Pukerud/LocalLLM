#!/usr/bin/env python3
"""CPU-only IQ4_XS V3.1 packing/checks. Produces an INACTIVE config; cannot start/switch any model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys

from engine_safety import SafetyError, owner_home
from prepare_ple_q8 import safe
import prepare_unsloth_v31 as D
import unsloth_v31_assets as A
import unsloth_v31_profile as F
import strata_runtime as R
import iq3_s_sc117_abliterated_assets as S


def record(path, sha=None):
    path = safe(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha or D.digest(path), 'identity': list(D.identity(path))}


def checked_record(row, full=False):
    path = safe(Path(row['path']))
    if list(D.identity(path)) != row['identity'] or path.stat().st_size != row['bytes']:
        raise SafetyError('Prepared file identity changed: ' + path.name)
    if full and D.digest(path) != row['sha256']:
        raise SafetyError('Prepared file checksum changed: ' + path.name)
    return path


def downloads(root, full=False):
    folder = A.folder(root)
    d = json.loads(safe(folder / 'downloaded.json').read_text())
    if (d.get('repository'), d.get('revision'), d.get('quant_commit')) != (A.REPOSITORY, A.REVISION, A.QUANT_COMMIT):
        raise SafetyError('V3.1 download provenance differs')
    rows = d.get('assets', [])
    if len(rows) != len(A.ASSETS):
        raise SafetyError('Incomplete V3.1 downloads')
    for pin, row in zip(A.ASSETS, rows):
        role, remote, local, size, sha = pin
        if (row['role'], row['repo_path'], row['path'], row['bytes'], row['sha256']) != (role, remote, str(folder / local), size, sha):
            raise SafetyError('V3.1 asset pin differs')
        checked_record(row, full)
    return rows


def runtime(root):
    source, manifest = R.prepared_runtime(Path(root), A.RUNTIME)
    if manifest['llama_commit'] != A.GGUF_PY_COMMIT:
        raise SafetyError('CPU packer llama.cpp revision pin changed')
    verify_gguf_py(source / 'third_party/llama.cpp')
    return source, manifest


def verify_gguf_py(base, expected_sha=A.GGUF_PY_TREE_SHA256, expected_count=A.GGUF_PY_FILES):
    # setup.get_llama_cpp extracts a pinned ZIP; rev-parse here would incorrectly climb to Strata's parent repo.
    # Exact file hashes were independently checked against ggml-org's pinned Git tree before recording this pin.
    base = Path(base)
    rows = []
    for path in sorted((base / 'gguf-py').rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        safe(path)
        data = path.read_bytes()
        rows.append({'path': str(path.relative_to(base)), 'bytes': len(data),
                     'git_blob_oid': hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest(),
                     'sha256': hashlib.sha256(data).hexdigest()})
    sha = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if len(rows) != expected_count or sha != expected_sha:
        raise SafetyError('CPU packer vendored gguf-py Python tree changed')
    return sha


def reader(source):
    sys.path.insert(0, str(source / 'tools'))
    from gguf_reader import GGUFFile
    return GGUFFile


def validate_headers(root, source):
    G = reader(source)
    files, names, total = [], set(), 0
    for pin in A.ASSETS[:4]:
        path = A.folder(root) / pin[2]
        g = G(path)
        if pin[0] == 'model_1':
            for k, v in {'general.architecture': 'qwen4exp', 'qwen4exp.block_count': 48,
                         'qwen4exp.embedding_length': 2560, 'qwen4exp.expert_count': 512,
                         'qwen4exp.context_length': A.CONTEXT, 'split.count': 4,
                         'split.tensors.count': 1224}.items():
                if g.metadata.get(k) != v:
                    raise SafetyError('V3.1 model geometry changed: ' + k)
        types = {}
        for t in g.tensors:
            if t.name in names or g.data_start + t.offset + t.expected_bytes() > pin[3]:
                raise SafetyError('Duplicate/out-of-bounds model tensor')
            names.add(t.name)
            types[t.type_name] = types.get(t.type_name, 0) + 1
        total += len(g.tensors)
        files.append({'path': str(path), 'tensors': len(g.tensors), 'types': types})
    if total != 1224:
        raise SafetyError('Incomplete V3.1 tensor set')
    g = G(A.folder(root) / A.ASSETS[-1][2])
    if len(g.tensors) != 1:
        raise SafetyError('New Q8 PLE must be table-only')
    t = g.tensors[0]
    if (t.name != 'per_layer_token_embd.weight' or t.type_name != 'Q8_0' or
            list(t.shape) != [160, 320001536] or t.offset != 0 or
            g.data_start + t.expected_bytes() != A.ASSETS[-1][3]):
        raise SafetyError('New Q8 PLE geometry/bounds differ')
    return {'shards': files, 'tensor_count': total, 'ple': {'type': t.type_name, 'shape': list(t.shape), 'data_offset': g.data_start}}


def dependencies(root, source):
    rows = []
    vision = safe(Path(root) / A.VISION_REL)
    if vision.stat().st_size != A.VISION_SIZE or D.digest(vision) != A.VISION_SHA:
        raise SafetyError('Matching BF16 vision projector missing/changed')
    rows.append(record(vision, A.VISION_SHA))
    for a in S.ASSETS:
        if a.role.startswith('mtp_runtime_'):
            path = safe(S.asset_path(Path(root), a))
            if path.stat().st_size != a.size:
                raise SafetyError('Stock Strata MTP file size changed')
            sha = D.digest(path)
            if a.sha256 and sha != a.sha256:
                raise SafetyError('Stock Strata MTP checksum changed')
            if a.git_blob_oid:
                data = path.read_bytes()
                if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != a.git_blob_oid:
                    raise SafetyError('Stock Strata MTP text pin changed')
            rows.append(record(path, sha))
    original = source / 'data/experimental-speed-projection' / A.VECTOR_NAME
    if original.stat().st_size != A.VECTOR_SIZE or D.digest(original) != A.VECTOR_SHA:
        raise SafetyError('Original-Qwen projection vector checksum changed')
    target = safe(A.folder(root) / A.VECTOR_NAME)
    if target.exists():
        raise SafetyError('Unrecorded vector already present; no overwrite')
    with target.open('xb') as out:
        out.write(original.read_bytes())
    g = reader(source)(target)
    if (g.metadata.get('general.architecture') != 'controlvector' or
            {t.name for t in g.tensors} != {f'direction.{n}' for n in range(1, 48)}):
        raise SafetyError('Projection vector layer set differs')
    with target.open('rb') as f:
        for t in g.tensors:
            if list(t.shape) != [2560] or t.type_name != 'F32':
                raise SafetyError('Projection vector type/width changed')
            f.seek(g.data_start + t.offset)
            values = struct.unpack('<2560f', f.read(2560 * 4))
            norm = math.fsum(x * x for x in values)
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
                raise SafetyError('Projection direction is not a finite unit vector')
    target.chmod(0o444)
    rows.append(record(target, A.VECTOR_SHA))
    return rows


def prepare(root):
    root = Path(root)
    folder = A.folder(root)
    if (folder / 'prepared.json').exists():
        return check(root)
    assets = downloads(root)
    source, manifest = runtime(root)
    headers = validate_headers(root, source)
    pack = safe(folder / 'native-pack', directory=True)
    if pack.exists():
        raise SafetyError('Unrecorded native-pack already exists; no overwrite/rebuild')
    deps = dependencies(root, source)
    command = [str(source / '.venv/bin/python'), str(source / 'tools/iq_pack.py'), '--gguf',
               str(folder / A.ASSETS[0][2]), '--out', str(pack), '--compat-bf16']
    env = {'HOME': str(owner_home()), 'USER': os.environ.get('USER', 'user'),
           'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    request = {'command': command, 'tool_sha256': D.digest(source / 'tools/iq_pack.py'),
               'source_commit': manifest['source_commit'], 'gguf_py_commit': A.GGUF_PY_COMMIT,
               'gguf_py_tree_sha256': A.GGUF_PY_TREE_SHA256, 'no_gpu_or_inference': True}
    D.atomic_json(folder / 'pack-request.json', request)
    with (folder / 'cpu-pack.log').open('x') as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise SafetyError('Pinned CPU packer failed; log/partial retained, no runtime start attempted')
    expert_lines = [x for x in (pack / 'native_experts.txt').read_text().splitlines() if x and not x.startswith('#')]
    if len(expert_lines) != 48 or [int(x.split()[0]) for x in expert_lines] != list(range(48)) or (pack / 'experts.bin').exists():
        raise SafetyError('Native expert table must cover 48 layers without duplicate expert copies')
    state = safe(A.state(owner_home()), directory=True)
    if state.exists() and any(state.iterdir()):
        raise SafetyError('Candidate state must be fresh/empty; no conversation/cache adoption')
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    seed = safe(state / 'expert-profile.bin')
    if seed.exists():
        raise SafetyError('Fresh candidate expert profile already exists; no state/history adoption')
    original = safe(source / 'data/expert-profile.bin')
    if original.stat().st_size != A.ROUTING_SEED_SIZE or D.digest(original) != A.ROUTING_SEED_SHA:
        raise SafetyError('Pinned runtime-shipped original-model routing seed changed')
    with seed.open('xb') as out:
        out.write(original.read_bytes())
    seed.chmod(0o600)
    config = F.canonical_config(root)
    F.validate_config(root, config)
    config_path = folder / 'config.inactive.json'
    D.atomic_json(config_path, config)
    config_path.chmod(0o444)
    files = []
    for path in sorted(pack.rglob('*')):
        if path.is_dir():
            continue
        safe(path)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise SafetyError('Special/symlinked native pack file')
        path.chmod(0o444)
        files.append(record(path))
    for row in assets:
        path = Path(row['path'])
        checked_record(row)
        path.chmod(0o444)
        row['identity'] = list(D.identity(path))
    d = json.loads((folder / 'downloaded.json').read_text())
    d['assets'] = assets
    D.atomic_json(folder / 'downloaded.json', d)
    value = {'phase': 'OFFLINE-PREPARED-AWAITING-EXPLICIT-CONTINUE', 'activation_allowed': False,
             'gpu_inference_tested': False, 'repository': A.REPOSITORY, 'revision': A.REVISION,
             'quant_commit': A.QUANT_COMMIT, 'runtime_version': A.RUNTIME,
             'runtime_source_commit': manifest['source_commit'], 'assets': assets, 'headers': headers,
             'dependencies': deps, 'pack_files': files, 'pack_request': request,
             'config': record(config_path), 'fresh_adaptive_profile': record(seed),
             'limits': {'context_per_slot': A.CONTEXT, 'parallel': A.PARALLEL, 'groups': A.GROUPS,
                        'conversation_cache_mib': A.CACHE_MIB, 'cached_histories': A.HISTORIES},
             'mtp': 'Existing matching stock/non-abliterated Qwen Strata Q2_0 expert drafter, Q8 dense. Not the incompatible llama.cpp Q8 MTP file; concurrent pipelined drafts remain off.',
             'conversion_note': 'Pinned iq_pack --compat-bf16 converts small unsupported projections to BF16 with recorded rounding; original GGUFs/experts/native dense/Q8 PLE bytes are not rewritten.',
             'pending': ['Explicit operator continue', 'Fresh rentals/manual ownership/active-request checks',
                         'Separate bounded GPU startup and LAN text/thinking/vision/tools/SSE/Host-Origin tests',
                         'Measured fit/performance/quality and new-profile menu activation'],
             'scope': 'Offline checks only. No serving readiness, two-full-context capacity, eight-full-history capacity, quality equivalence, speed gain or stability qualification.'}
    D.atomic_json(folder / 'prepared.json', value)
    return check(root)


def check(root, full=False, *, pristine_state=True):
    root = Path(root)
    folder = A.folder(root)
    d = json.loads(safe(folder / 'prepared.json').read_text())
    source, manifest = runtime(root)
    if (d.get('phase') != 'OFFLINE-PREPARED-AWAITING-EXPLICIT-CONTINUE' or d.get('activation_allowed') is not False or
            d.get('gpu_inference_tested') is not False or d.get('runtime_source_commit') != manifest['source_commit'] or
            (d.get('repository'), d.get('revision'), d.get('quant_commit')) != (A.REPOSITORY, A.REVISION, A.QUANT_COMMIT)):
        raise SafetyError('Inactive candidate contract changed')
    expected_limits = {'context_per_slot': A.CONTEXT, 'parallel': A.PARALLEL, 'groups': A.GROUPS,
                       'conversation_cache_mib': A.CACHE_MIB, 'cached_histories': A.HISTORIES}
    if d.get('runtime_version') != A.RUNTIME or d.get('limits') != expected_limits:
        raise SafetyError('Inactive native context/parallel/cache/runtime policy changed')
    assets = downloads(root, full)
    if d['assets'] != assets:
        raise SafetyError('Prepared/download manifest identities differ')
    pins = {str(root / A.VISION_REL): (A.VISION_SIZE, A.VISION_SHA, None),
            str(folder / A.VECTOR_NAME): (A.VECTOR_SIZE, A.VECTOR_SHA, None)}
    for a in S.ASSETS:
        if a.role.startswith('mtp_runtime_'):
            pins[str(S.asset_path(root, a))] = (a.size, a.sha256, a.git_blob_oid)
    if len(d['dependencies']) != len(pins) or {x['path'] for x in d['dependencies']} != set(pins):
        raise SafetyError('Inactive dependency set changed')
    for row in d['dependencies']:
        size, sha, git_oid = pins[row['path']]
        if row['bytes'] != size or (sha and row['sha256'] != sha):
            raise SafetyError('Inactive dependency pin changed')
        if git_oid:
            data = Path(row['path']).read_bytes()
            if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != git_oid:
                raise SafetyError('Inactive text dependency pin changed')
    actual = {str(p) for p in (folder / 'native-pack').rglob('*') if not p.is_dir()}
    if actual != {x['path'] for x in d['pack_files']} or len(actual) != len(d['pack_files']):
        raise SafetyError('Unrecorded/missing native pack file; no adoption')
    if d['config']['path'] != str(folder / 'config.inactive.json') or d['fresh_adaptive_profile']['path'] != str(A.state(owner_home()) / 'expert-profile.bin'):
        raise SafetyError('Inactive config/state path changed')
    seed = d['fresh_adaptive_profile']
    if seed['bytes'] != A.ROUTING_SEED_SIZE or seed['sha256'] != A.ROUTING_SEED_SHA:
        raise SafetyError('Inactive fresh routing seed pin changed')
    if pristine_state and {p.name for p in A.state(owner_home()).iterdir()} != {'expert-profile.bin'}:
        raise SafetyError('Inactive state has unrecorded server/history/cache files; no adoption')
    want_command = [str(source / '.venv/bin/python'), str(source / 'tools/iq_pack.py'), '--gguf',
                    str(folder / A.ASSETS[0][2]), '--out', str(folder / 'native-pack'), '--compat-bf16']
    want_request = {'command': want_command, 'tool_sha256': D.digest(source / 'tools/iq_pack.py'),
                    'source_commit': manifest['source_commit'], 'gguf_py_commit': A.GGUF_PY_COMMIT,
                    'gguf_py_tree_sha256': A.GGUF_PY_TREE_SHA256, 'no_gpu_or_inference': True}
    if d['pack_request'] != want_request or json.loads((folder / 'pack-request.json').read_text()) != want_request:
        raise SafetyError('Inactive CPU pack provenance/command changed')
    records = [*d['dependencies'], *d['pack_files'], d['config']]
    if pristine_state:
        records.append(d['fresh_adaptive_profile'])
    for row in records:
        checked_record(row, full=True)
    F.validate_config(root, json.loads(Path(d['config']['path']).read_text()))
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--check', action='store_true')
    p.add_argument('--full-hash', action='store_true')
    a = p.parse_args(argv)
    if os.geteuid() == 0:
        p.error('Run CPU preparation as the unprivileged model owner')
    if a.full_hash and not a.check:
        p.error('--full-hash requires --check')
    root = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
    folder = safe(A.folder(root), directory=True)
    fd = os.open(safe(folder / 'prepare.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        import fcntl
        with os.fdopen(fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check(root, a.full_hash) if a.check else prepare(root)
    except (SafetyError, OSError, ValueError, KeyError) as e:
        print('OFFLINE PREPARATION BLOCKED:', e, file=sys.stderr)
        return 1
    print('OFFLINE PREPARED: V3.1 IQ4_XS / Q8 PLE / original-Qwen projection / CPU BF16 vision / stock Strata MTP / native262K / 2 slots / 8 histories. Await explicit continue; NO model was loaded or switched.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
