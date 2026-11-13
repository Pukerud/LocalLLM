"""Pinned Strata app 0.2.13 / engine 0.1.59, side-by-side and GPU-OFF during preparation.
No updater, model download, activation, driver install or lifecycle operation.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from engine_safety import SafetyError

VERSION = '0.1.59'
TAG = 'v0.2.13'
PIN = 'd2f096475817d4666c364c820b0ca830f751cfc5'
ASSET = 'strata-engine-0.1.59-linux-x86_64-cuda-cuda13.tar.gz'
ASSET_BYTES = 723498702
ASSET_SHA256 = '42ab04fa3201e41bef5fac28c05f5c35c909419ec937d879b505bf1516742610'
URL = 'https://github.com/Niko1221/Strata/releases/download/' + TAG + '/' + ASSET
REPOSITORY = 'https://github.com/Niko1221/Strata.git'
SERVER_ORIGINAL_SHA256 = '398dea98fd319ad3b570a006dd1a4c1907480cb1ab07737368382620d85f2d14'
SERVER_PATCHED_SHA256 = '1229beb5123d20442e040df1b8a38f2c649615414aea41d1cb8101bbdd795700'
OFFICIAL_FILES = {
    'strata': (316017888, '0ba914b28df1c8e09c0b5209c2ca6040c75fb650a7408ff3d9ef3ad159a13eb9'),
    'BUILD.json': (187, '15c8c2e698a8a60deb0cf1a7a6968ce76a53a5e75cd477b8176a00aff56d202c'),
    'libstrata_kq.so': (35881360, 'f64075cd5774915867af0012a68abd0af05f57ab0af635d674a1724f558d64d0'),
    'libcudart.so.13': (704288, '96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0'),
    'libcublas.so.13': (54177976, '25cde432108543626df85d887c730bf2cab96803ca16f9654e4abc8d2b45833e'),
    'libcublasLt.so.13': (541607888, '1489aeb75d55ee55782798343ac66f57771d561cc1fe1e0491568deefa0e6515'),
}
SERVER_OLD = b'        (PR #637)."""\n        info = dict(self.info)\n'
SERVER_NEW = (b'        (PR #637)."""\n'
              b'        if os.environ.get("STRATA_EXTERNAL_LIFECYCLE") == "1":\n'
              b'            raise EngineDied("Engine restart requires a fresh HostLLM/Hive ownership, rental and hardware check")\n'
              b'        info = dict(self.info)\n')


def patch_server(source):
    path = Path(source) / 'serve/server.py'
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha == SERVER_PATCHED_SHA256:
        return
    if sha != SERVER_ORIGINAL_SHA256 or data.count(SERVER_OLD) != 1:
        raise SafetyError('Exact external-lifecycle patch preimage changed; no patch applied')
    data = data.replace(SERVER_OLD, SERVER_NEW)
    if hashlib.sha256(data).hexdigest() != SERVER_PATCHED_SHA256:
        raise SafetyError('External-lifecycle patch postimage changed')
    path.write_bytes(data)


def folder(root):
    return Path(root) / 'runtimes' / VERSION


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def checked_file(path):
    path = Path(path)
    for p in (path, *path.parents):
        if p.is_symlink():
            raise SafetyError('Candidate runtime path is symlinked: ' + str(p))
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 1000 or st.st_mode & 0o022 or st.st_nlink != 1:
        raise SafetyError('Candidate runtime file owner/type/mode/link changed: ' + str(path))
    return path


def record(path):
    path = checked_file(path)
    st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'sha256': digest(path)}


def check_record(row):
    path = checked_file(row['path'])
    if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise SafetyError('Candidate runtime artifact checksum changed: ' + path.name)
    return path


def source_check(source):
    source = Path(source)
    git = ['git', '-c', 'safe.directory=' + str(source), '-C', str(source)]
    if subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise SafetyError('Strata 0.2.13 source commit changed')
    changed = subprocess.check_output(git + ['diff', 'HEAD', '--name-only'], text=True).splitlines()
    sha = digest(source / 'serve/server.py')
    if changed not in ([], ['serve/server.py']) or sha not in (SERVER_ORIGINAL_SHA256, SERVER_PATCHED_SHA256):
        raise SafetyError('Strata 0.2.13 source differs beyond the exact external-lifecycle patch')


def ready(root):
    base = folder(root)
    manifest = json.loads(checked_file(base / 'prepared.json').read_text())
    if (manifest.get('version'), manifest.get('source_commit'), manifest.get('release_archive_sha256')) != (VERSION, PIN, ASSET_SHA256):
        raise SafetyError('Strata 0.2.13 runtime provenance changed')
    if manifest.get('gpu_inference_tested') is not False or manifest.get('activation_allowed') is not False:
        raise SafetyError('Offline runtime receipt is immutable history, not activation authority')
    source = base / 'source'
    source_check(source)
    if digest(source / 'serve/server.py') != SERVER_PATCHED_SHA256:
        raise SafetyError('Candidate external-lifecycle guard is missing')
    expected = {str(p) for p in (source / 'engine').rglob('*') if p.is_file()}
    rows = manifest['runtime_assets']
    names = {Path(r['path']).name for r in rows}
    if ({r['path'] for r in rows} != expected or len(rows) != len(expected) or
            names != set(OFFICIAL_FILES) | {'strata-vision-cpu'}):
        raise SafetyError('Unrecorded/missing candidate engine/library file')
    for row in rows:
        pin = OFFICIAL_FILES.get(Path(row['path']).name)
        if pin and (row['bytes'], row['sha256']) != pin:
            raise SafetyError('Official .59 binary/library pin differs even if a local receipt was rewritten')
    for row in [*rows, *manifest['python_assets']]:
        check_record(row)
    if manifest.get('cpu_vision') != 'CPU-only build; STRATA_VISION_CUDA=OFF':
        raise SafetyError('Candidate must retain CPU-only BF16 vision')
    return source, manifest
