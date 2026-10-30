"""Opt-in PLE-only Q8_0 variants. Original profiles, runtime and live state remain unchanged."""
import copy
import json
import os
from pathlib import Path
import stat

from engine_safety import SafetyError, owner_home
import ple_q8_assets as A


def state_root(profile):
    if profile not in A.STATES:
        raise SafetyError('Q8_0 PLE is supported only for Heretic RE and SC117')
    return owner_home() / '.local/state/locallm-strata' / A.STATES[profile]


def require_state(profile, state):
    expected = state_root(profile)
    if Path(state) != expected:
        raise SafetyError('Q8_0 PLE requires its separate state/cache: ' + str(expected))
    for path in (expected, *expected.parents):
        if path.is_symlink():
            raise SafetyError('Q8_0 PLE state path is symlinked')
    if expected.exists() and (not expected.is_dir() or expected.stat().st_uid != os.geteuid()):
        raise SafetyError('Q8_0 PLE state type/owner changed')


def ready_asset(root, *, full=False):
    from prepare_ple_q8 import digest, safe, validate_header
    root = Path(root)
    folder = safe(A.folder(root), directory=True)
    manifest = safe(folder / 'prepared.json')
    try:
        d = json.loads(manifest.read_text())
    except FileNotFoundError:
        raise SafetyError('Q8_0 PLE is not prepared; run --prepare-ple-q8 --download-only, then --prepare') from None
    path = safe(A.table_path(root))
    if (d.get('repository') != A.REPOSITORY or d.get('revision') != A.REVISION or
            d.get('repo_path') != A.REPO_PATH or d.get('sha256') != A.SHA256 or
            d.get('bytes') != A.SIZE or d.get('path') != str(path)):
        raise SafetyError('Q8_0 PLE provenance/checksum pin differs')
    info = path.stat()
    if ((info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino, info.st_uid) !=
            (A.SIZE, d.get('mtime_ns'), d.get('device'), d.get('inode'), d.get('uid')) or
            d.get('tensor') != validate_header(root, path) or (full and digest(path) != A.SHA256)):
        raise SafetyError('Q8_0 PLE verified data/header/identity changed')
    return path


def identity(cfg, profile):
    original = cfg['model_name'].removesuffix('-ple-q8_0')
    cfg['model_name'] = original + '-ple-q8_0'
    cfg['aliases'] = list(dict.fromkeys([profile + '-ple-q8', original, *cfg.get('aliases', [])]))


def overlay(root, profile, cfg):
    """Change only the PLE input, adaptive-profile location and API identity."""
    require_state(profile, state_root(profile))
    table = ready_asset(root)
    cfg = copy.deepcopy(cfg)
    args = cfg['args']
    for flag in ('--ple-gguf', '--ple-io', '--expert-profile'):
        if args.count(flag) != 1:
            raise SafetyError('Q8_0 PLE requires exactly one ' + flag)
    if args[args.index('--ple-io') + 1] != 'mmap':
        raise SafetyError('Q8_0 PLE must remain memory-mapped; whole-table RAM locking is not allowed')
    args[args.index('--ple-gguf') + 1] = str(table)
    args[args.index('--expert-profile') + 1] = str(state_root(profile) / 'expert-profile.bin')
    identity(cfg, profile)
    return cfg


def seed_profile(root, profile, state):
    require_state(profile, state)
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.chmod(0o700)
    dest = state / 'expert-profile.bin'
    if dest.is_symlink():
        raise SafetyError('Q8_0 PLE expert profile is symlinked')
    if not dest.exists():
        seed = Path(root) / 'source-99f3dbd0b21d/data/expert-profile.bin'
        if seed.is_symlink():
            raise SafetyError('Q8_0 PLE seed is symlinked')
        with seed.open('rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024**2:
                raise SafetyError('Q8_0 PLE seed type/size changed')
            data = f.read(4 * 1024**2 + 1)
            after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
            raise SafetyError('Q8_0 PLE seed changed while being copied')
        with dest.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
    info = dest.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or
            info.st_mode & 0o077 or info.st_size > 4 * 1024**2):
        raise SafetyError('Q8_0 PLE expert profile type/owner/mode/size changed')
