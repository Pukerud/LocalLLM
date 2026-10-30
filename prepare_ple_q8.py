#!/usr/bin/env python3
"""Download/verify one pinned Q8_0 PLE shard as the operator; never run inference or services."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import urllib.request

from engine_safety import SafetyError, owner_home
import ple_q8_assets as A


def safe(path, *, directory=False):
    path = Path(path)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise SafetyError('Q8_0 PLE path is symlinked: ' + str(part))
    if path.exists():
        info = path.lstat()
        if (not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) or
                info.st_uid != os.geteuid() or info.st_mode & 0o022):
            raise SafetyError('Q8_0 PLE path has unsafe type/owner/mode: ' + str(path))
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def validate_header(root, path):
    # The already-pinned local runtime reader parses headers only, never model data/code.
    import strata_runtime as runtimes
    source, _ = runtimes.prepared_runtime(root, '0.1.46')
    sys.path.insert(0, str(source / 'tools'))
    from gguf_reader import GGUFFile
    g = GGUFFile(path)
    if len(g.tensors) != 1:
        raise SafetyError('Q8_0 PLE shard must contain exactly its one original table')
    t = g.tensors[0]
    if (t.name != A.TENSOR or t.type_name != 'Q8_0' or t.shape != A.SHAPE or
            t.offset != 0 or t.expected_bytes() != A.TENSOR_BYTES or
            g.data_start + A.TENSOR_BYTES != A.SIZE):
        raise SafetyError('Q8_0 PLE tensor format/shape/bounds differ from the pinned table')
    return {'name': t.name, 'type': t.type_name, 'shape': t.shape,
            'data_offset': g.data_start, 'data_bytes': A.TENSOR_BYTES}


def download(root):
    dest = A.table_path(root)
    parent = safe(dest.parent, directory=True)
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    safe(parent, directory=True)
    safe(dest)
    if dest.exists():
        if dest.stat().st_size != A.SIZE or digest(dest) != A.SHA256:
            raise SafetyError('Existing Q8_0 PLE table differs; it was not overwritten')
        print('Pinned Q8_0 PLE file already present and checksum-verified.', flush=True)
        return
    tmp = safe(dest.with_suffix(dest.suffix + '.part'))
    start = tmp.stat().st_size if tmp.exists() else 0
    if start > A.SIZE:
        raise SafetyError('Q8_0 PLE partial download is oversized')
    if os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize < A.SIZE - start + 2 * 1024**3:
        raise SafetyError('Not enough disk space for Q8_0 PLE and 2 GiB headroom')
    for attempt in range(6):
        start = tmp.stat().st_size if tmp.exists() else 0
        if start == A.SIZE:
            break
        request = urllib.request.Request(A.url(), headers={'Range': f'bytes={start}-'} if start else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                    raise SafetyError('Q8_0 PLE server did not honor the exact resume offset')
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'ab') as out:
                    total = start
                    report_at = total + 1024**3
                    while block := response.read(8 * 1024**2):
                        total += len(block)
                        if total > A.SIZE:
                            raise SafetyError('Q8_0 PLE response exceeds the pinned length')
                        out.write(block)
                        if total >= report_at:
                            print(f'Q8_0 PLE: {total}/{A.SIZE} bytes', flush=True)
                            report_at += 1024**3
                    out.flush()
                    os.fsync(out.fileno())
            if tmp.stat().st_size != A.SIZE:
                raise OSError('Q8_0 PLE short download')
            break
        except SafetyError:
            raise
        except (OSError, TimeoutError) as exc:
            if attempt == 5:
                raise
            print('Q8_0 PLE transfer interrupted; retrying:', type(exc).__name__, flush=True)
            time.sleep(3 * (attempt + 1))
    if tmp.stat().st_size != A.SIZE or digest(tmp) != A.SHA256:
        raise SafetyError('Q8_0 PLE full size/SHA-256 verification failed; partial file retained')
    if dest.exists():
        raise SafetyError('Q8_0 PLE destination appeared during download; refusing replacement')
    os.rename(tmp, dest)
    print('Downloaded and SHA-256 verified original Q8_0 PLE shard; no other Q6 model shards downloaded.', flush=True)


def prepare(root):
    path = safe(A.table_path(root))
    if not path.is_file() or path.stat().st_size != A.SIZE or digest(path) != A.SHA256:
        raise SafetyError('Q8_0 PLE is missing or checksum-mismatched; download it first')
    tensor = validate_header(root, path)
    info = path.stat()
    manifest = {'repository': A.REPOSITORY, 'revision': A.REVISION, 'repo_path': A.REPO_PATH,
                'sha256': A.SHA256, 'bytes': A.SIZE, 'path': str(path), 'mtime_ns': info.st_mtime_ns,
                'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid, 'tensor': tensor,
                'scope': 'Data/header checks only; no GPU load, inference, quality or speed qualification.'}
    dest = safe(A.folder(root) / 'prepared.json')
    temp = safe(dest.with_suffix('.json.part'))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, dest)
    print('Q8_0 PLE manifest prepared; inference and quality/speed testing remain pending.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--download-only', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    a = ap.parse_args()
    if os.geteuid() == 0:
        raise SystemExit('Run Q8_0 PLE preparation as the unprivileged model owner, never root')
    root = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
    try:
        folder = safe(A.folder(root), directory=True)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = safe(folder / 'prepare.lock')
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            download(root) if a.download_only else prepare(root)
    except (SafetyError, OSError, ValueError) as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
