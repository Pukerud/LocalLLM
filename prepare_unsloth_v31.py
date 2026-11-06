#!/usr/bin/env python3
"""Prepare pinned V3.1 IQ4_XS DATA ONLY. No start, stop, switch, calibration, or GPU inference commands."""
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
from prepare_ple_q8 import safe
import unsloth_v31_assets as A

BLOCK = 4 * 1024**2


def identity(path):
    s = Path(path).lstat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_uid, s.st_mode)


def digest(path):
    before = identity(path)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        if identity(path) != before:
            raise SafetyError('File changed before checksum')
        while block := f.read(BLOCK):
            h.update(block)
    if identity(path) != before:
        raise SafetyError('File changed during checksum')
    return h.hexdigest()


def atomic_json(path, data):
    path = safe(path)
    tmp = safe(path.with_suffix(path.suffix + '.new'))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint(out, start, end):
    out.flush()
    os.fsync(out.fileno())
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(out.fileno(), start, end - start, os.POSIX_FADV_DONTNEED)


def request(url, start=0, end=None):
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'LocalLLM-data-only-preparation'}
    if start or end is not None:
        headers['Range'] = f'bytes={start}-' + (str(end) if end is not None else '')
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    if 'Range' in headers and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
        response.close()
        raise SafetyError('Server did not honor exact byte range')
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        response.close()
        raise SafetyError('Compressed model response refused')
    return response


def reuse_payload(root, row, dest, *, get=request, rate=256 * 1024**2):
    """Copy, never edit/link the source. Complete hashes prove byte-exact publisher identity."""
    if row[0] not in A.REUSE:
        return False
    relative, old_sha = A.REUSE[row[0]]
    source = safe(Path(root) / relative)
    if not source.exists() or source.stat().st_size != row[3]:
        return False
    with get(A.url(row), 0, 191) as response:
        header = response.read(193)
    if len(header) != 192 or header[:4] != b'GGUF':
        return False
    print(row[0] + ': checking exact header/payload reuse without touching the original...', flush=True)
    tmp = safe(dest.with_suffix('.reuse.part'))
    before = identity(source)
    started, total, synced = time.monotonic(), 192, 0
    old_h, new_h = hashlib.sha256(), hashlib.sha256()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out, source.open('rb') as src:
        old_h.update(src.read(192))
        new_h.update(header)
        out.write(header)
        while block := src.read(BLOCK):
            total += len(block)
            old_h.update(block)
            new_h.update(block)
            out.write(block)
            if total - synced >= 256 * 1024**2:
                checkpoint(out, synced, total)
                synced = total
            delay = total / rate - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
        checkpoint(out, synced, total)
    if identity(source) != before or old_h.hexdigest() != old_sha:
        raise SafetyError('Original table changed or its pinned checksum differs; new copy retained for inspection')
    if total != row[3] or new_h.hexdigest() != row[4]:
        tmp.unlink()  # Only our own just-created copy, never the shared source.
        print(row[0] + ': payload differs; downloading the exact new file instead.', flush=True)
        return False
    if digest(tmp) != row[4] or dest.exists():
        raise SafetyError('Reconstructed file failed on-disk checksum / destination appeared')
    os.rename(tmp, dest)
    print(row[0] + ': exact NEW full-file SHA verified using an unchanged existing payload.', flush=True)
    return True


def download_one(row, dest, *, rate=80 * 1024**2, get=request):
    safe(dest.parent, directory=True).mkdir(parents=True, exist_ok=True, mode=0o700)
    safe(dest)
    if dest.exists():
        if dest.stat().st_size != row[3] or digest(dest) != row[4]:
            raise SafetyError('Existing candidate differs; refusing overwrite: ' + dest.name)
        return
    tmp = safe(dest.with_suffix(dest.suffix + '.part'))
    for attempt in range(8):
        start = tmp.stat().st_size if tmp.exists() else 0
        if start > row[3]:
            raise SafetyError('Oversized partial file')
        if start == row[3]:
            break
        space = os.statvfs(dest.parent)
        if space.f_bavail * space.f_frsize < row[3] - start + 8 * 1024**3:
            raise SafetyError('Not enough disk space plus 8 GiB headroom')
        try:
            with get(A.url(row), start) as response:
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != row[3] - start:
                    raise SafetyError('Response length differs from the pinned file/resume offset')
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'ab') as out:
                    total, synced = start, start
                    began = time.monotonic()
                    report = began
                    while block := response.read(BLOCK):
                        total += len(block)
                        if total > row[3]:
                            raise SafetyError('Response exceeds pinned length')
                        out.write(block)
                        if total - synced >= 256 * 1024**2:
                            checkpoint(out, synced, total)
                            synced = total
                        delay = (total - start) / rate - (time.monotonic() - began)
                        if delay > 0:
                            time.sleep(delay)
                        if time.monotonic() - report >= 30:
                            print(f'{row[0]}: {total}/{row[3]} bytes ({total / row[3]:.1%})', flush=True)
                            report = time.monotonic()
                    checkpoint(out, synced, total)
            if total != row[3]:
                raise OSError('Short model response')
            break
        except SafetyError:
            raise
        except (OSError, TimeoutError) as exc:
            # Never log redirect URLs or signed CDN query strings.
            print(f'{row[0]} transfer interrupted ({type(exc).__name__}); retry {attempt + 1}/8', flush=True)
            if attempt == 7:
                raise SafetyError('Transfer retries exhausted; partial retained') from None
            time.sleep(3 * (attempt + 1))
    print(row[0] + ': checking complete on-disk SHA-256...', flush=True)
    if tmp.stat().st_size != row[3] or digest(tmp) != row[4]:
        raise SafetyError('Pinned size/SHA mismatch; partial retained, not adopted')
    if dest.exists():
        raise SafetyError('Destination appeared; refusing replacement')
    os.rename(tmp, dest)
    print(row[0] + ': complete SHA-256 verified.', flush=True)


def download_all(root, rate):
    folder = A.folder(root)
    rows = []
    for row in A.ASSETS:
        path = folder / row[2]
        safe(path.parent, directory=True).mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            if path.stat().st_size != row[3] or digest(path) != row[4]:
                raise SafetyError('Existing prepared asset differs')
        elif not reuse_payload(root, row, path):
            download_one(row, path, rate=rate)
        info = identity(path)
        rows.append({'role': row[0], 'repo_path': row[1], 'path': str(path), 'bytes': row[3],
                     'sha256': row[4], 'identity': list(info)})
    atomic_json(folder / 'downloaded.json', {'repository': A.REPOSITORY, 'revision': A.REVISION,
                'quant_commit': A.QUANT_COMMIT, 'assets': rows, 'inference_tested': False,
                'scope': 'Exact data checksums only; no serving readiness/fit/quality/speed claim.'})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-only', action='store_true', required=True)
    p.add_argument('--rate-mib', type=float, default=80)
    args = p.parse_args(argv)
    if os.geteuid() == 0:
        p.error('Preparation must run as the unprivileged model owner, never root')
    if not 8 <= args.rate_mib <= 256:
        p.error('Rate must be 8..256 MiB/s')
    root = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
    folder = safe(A.folder(root), directory=True)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(safe(folder / 'prepare.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            download_all(root, args.rate_mib * 1024**2)
    except (SafetyError, OSError, ValueError) as exc:
        print('PREPARATION BLOCKED:', type(exc).__name__, str(exc) if isinstance(exc, SafetyError) else '', file=sys.stderr)
        return 1
    print('V3.1 IQ4_XS and new Q8 PLE data verified. No running model/config/service was changed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
