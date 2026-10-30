#!/usr/bin/env python3
"""Add only two inactive Q8_0 PLE Hive profiles and the selector; never stop/restart or change current."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import tempfile

from engine_safety import SafetyError, docker_empty
import ple_q8_assets as A

ROOT = Path('/hive/miners/custom/llm-hosting')
SOURCE = Path(__file__).resolve().parent / 'hive-llm-miner'
LOCK = Path('/run/llm-hosting-profile-menu.lock')
FILES = ('h-manifest.conf', 'h-config.sh', 'h-run.sh', 'h-stats.sh', 'llm-hosting.conf')


def checked_file(path, *, installed=False):
    path = Path(path)
    info = path.lstat()
    owner = os.geteuid()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
            (installed and (info.st_uid != owner or info.st_mode & 0o022))):
        raise SafetyError('Unsafe Q8_0 PLE package file: ' + str(path))
    return path.read_bytes()


def digest(path):
    return hashlib.sha256(checked_file(path, installed=True)).hexdigest()


def trusted_layout():
    operator = pwd.getpwnam('user').pw_uid
    for path in (Path('/run'), Path('/hive'), Path('/hive/miners'), Path('/hive/miners/custom'), ROOT, ROOT / 'profiles'):
        info = path.lstat()
        owner_ok = info.st_uid == 0 or (path == Path('/hive') and info.st_uid == operator and stat.S_IMODE(info.st_mode) == 0o755)
        if not stat.S_ISDIR(info.st_mode) or not owner_ok or info.st_mode & 0o022:
            raise SafetyError('Unsafe root-owned Hive package parent: ' + str(path))


def active_snapshot():
    target = os.readlink(ROOT / 'current')
    parts = Path(target).parts
    if len(parts) != 2 or parts[0] != 'profiles' or parts[1] in ('.', '..'):
        raise SafetyError('Current Hive pointer is not an exact installed profile')
    active = ROOT / target
    info = active.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise SafetyError('Active Hive profile directory is unsafe')
    links = {}
    for name in FILES:
        stable = ROOT / name
        if not stable.is_symlink() or os.readlink(stable) != 'current/' + name:
            raise SafetyError('Stable Hive entrypoint is not the unchanged current link')
        links[name] = os.readlink(stable)
    return {'current': target, 'stable_links': links,
            'active_files': {name: digest(active / name) for name in FILES},
            'miner_markers': {name: Path('/run/hive', name).read_bytes().hex() if Path('/run/hive', name).exists() else None
                              for name in ('MINER_RUN', 'MINER_STOP')}}


def install_inactive_profile(source, destination):
    """Create a complete new directory, or accept an exactly identical prior add-only install."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_dir():
        raise SafetyError('Q8_0 PLE source directory is missing or symlinked')
    contents = {name: checked_file(source / name) for name in FILES}
    if destination.exists() or destination.is_symlink():
        info = destination.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022 or
                set(p.name for p in destination.iterdir()) != set(FILES) or
                any(checked_file(destination / name, installed=True) != contents[name] for name in FILES)):
            raise SafetyError('Existing Q8_0 PLE profile differs; add-only mode will not overwrite it')
        return False
    stage = Path(tempfile.mkdtemp(prefix='.ple-q8-stage-', dir=destination.parent))
    stage.chmod(0o755)
    # A failed/incomplete stage is deliberately retained for inspection, never removed recursively.
    for name, data in contents.items():
        path = stage / name
        with path.open('xb') as f:
            os.fchmod(f.fileno(), 0o755 if name.endswith('.sh') else 0o644)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if checked_file(path, installed=True) != data:
            raise SafetyError('Q8_0 PLE staged file verification failed')
    if destination.exists() or destination.is_symlink():
        raise SafetyError('Q8_0 PLE destination appeared during staging; it was not overwritten')
    os.rename(stage, destination)
    return True


def publish_menu(source, destination):
    data = checked_file(source)
    old = checked_file(destination, installed=True)
    if old == data:
        return False
    fd, name = tempfile.mkstemp(prefix='.ple-q8-menu-', dir=destination.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            os.fchmod(f.fileno(), 0o755)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if checked_file(destination, installed=True) != old:
            raise SafetyError('Installed selector changed during add-only publication')
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--add-only', action='store_true', required=True)
    ap.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('Add-only Hive profile installation requires root; model preparation must run as user')
    try:
        trusted_layout()
        fd = os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        import hive_profile_switch as switch
        with switch.hosting.launch_reservation(), os.fdopen(fd, 'r+') as lock:
            if switch.hosting.read_lease() is not None or switch.JOURNAL.exists() or switch.JOURNAL.is_symlink():
                raise SafetyError('A hosting lease/switch journal exists; add-only installation refused')
            info = os.fstat(lock.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
                    stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise SafetyError('Hive selector lock type/owner/mode is unsafe')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            docker_empty()
            before = active_snapshot()
            for profile in A.HIVE_PROFILES:
                switch.preflight_profile(profile)  # metadata/file checks as user; no inference or lifecycle actions
            trusted_layout()
            docker_empty()
            if active_snapshot() != before:
                raise SafetyError('Hive selection/active files/markers changed; no selector publication')
            added = {profile: install_inactive_profile(SOURCE / 'profiles' / profile, ROOT / 'profiles' / profile)
                     for profile in A.HIVE_PROFILES}
            if active_snapshot() != before:
                raise SafetyError('Active Hive state changed while adding inactive profiles; selector unchanged')
            changed = publish_menu(SOURCE / 'profile-menu.sh', ROOT / 'profile-menu.sh')
            if active_snapshot() != before:
                raise SafetyError('Active Hive state changed externally during selector publication; inspect before switching')
            print(json.dumps({'added_inactive_profiles': added, 'selector_updated': changed,
                              'current_unchanged': True, 'active_profile_files_unchanged': True,
                              'stable_entrypoint_links_unchanged': True, 'miner_markers_unchanged': True,
                              'lifecycle_actions': [], 'inference_tested': False}, indent=2))
    except (SafetyError, OSError, ValueError) as exc:
        print('BLOCKED:', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
