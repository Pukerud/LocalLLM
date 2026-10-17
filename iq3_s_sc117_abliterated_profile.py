"""Isolated SC117 IQ3_S serving config and live-qualification gate.

The 8192 MiB/four-slot cache, automatic prefill, and automatic split are provisional
starting points only. They are not qualified until the identity-bound live checks pass.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

import iq3_s_sc117_abliterated_assets as assets
import prepare_iq3_s_sc117_abliterated as prepare_sc117
from prepare_iq3_s_sc117_abliterated import SafetyError

strata_runtime = prepare_sc117.strata_runtime
PROFILE = assets.PROFILE
MODEL_ID = assets.MODEL_ID
RUNTIME_VERSION = '0.1.39'
RUNTIME_COMMIT = strata_runtime.PINS[RUNTIME_VERSION]
TRIAL_RUNTIME_VERSION = '0.1.44.1'
TRIAL_DISTUTILS_PTH_RELATIVE = Path('distutils-precedence.pth')
TRIAL_DISTUTILS_PTH_SHA256 = '7ea7ffef3fe2a117ee12c68ed6553617f0d7fd2f0590257c25c484959a3b7373'
TRIAL_DISTUTILS_PTH_CONTENTS = (b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
                               b"os.environ.get(var, 'stdlib') == 'local'; enabled and "
                               b"__import__('_distutils_hack').add_shim(); \n")
BASELINE_CONFIG_RELATIVE = Path('source-99f3dbd0b21d') / 'strata-iq3_s.json'
TRIAL_STATE_ROOT_NAME = 'sc117-iq3s-trial'
TRIAL_BIND_HOST = '0.0.0.0'
TRIAL_PORT = 8080
TRIAL_ENV = 'STRATA_SC117_IQ3S_TRIAL'
TRIAL_OVERRIDE_ENV = {
    'STRATA_SC117_IQ3S_TRIAL_PREFILL': 'prefill',
    'STRATA_SC117_IQ3S_TRIAL_SPLIT': 'layer_split',
    'STRATA_SC117_IQ3S_TRIAL_CACHE_MIB': 'conversation_cache_mib',
    'STRATA_SC117_IQ3S_TRIAL_CACHE_SLOTS': 'conversation_cache_slots',
}
TRIAL_DEFAULT_SPEC = '3'
TRIAL_DEFAULT_PIPELINE_WINDOWS = '2'
TRIAL_EXPERIMENT_ENV = {
    'STRATA_SC117_IQ3S_TRIAL_PCIE_FRAC',
    'STRATA_SC117_IQ3S_TRIAL_POOL_WORKERS',
    'STRATA_SC117_IQ3S_TRIAL_TRIM_STAGE_WEIGHTS',
    'STRATA_SC117_IQ3S_TRIAL_GPU_ORDER',
    'STRATA_SC117_IQ3S_TRIAL_PIPELINE_WINDOWS',
    'STRATA_SC117_IQ3S_TRIAL_SPEC',
}
TRIAL_ENV_PREFIX = 'STRATA_SC117_IQ3S_TRIAL_'
DEFAULT_SETTINGS = {
    'context': 262144,
    'kv': 'int8',
    'kv_resident': 32768,
    'mtp_depth': 4,
    'vision': 'gpu-bf16',
    'parallel': 1,
    'batch_groups': 1,
    'prefill': 'auto',
    'layer_split': 'auto',
    'conversation_cache_mib': 8192,
    'conversation_cache_slots': 4,
}
REQUIRED_CHECKS = (
    'smoke',
    'native_context_tail',
    'gpu_vision',
    'mtp_acceptance',
    'conversation_cache_park_restore_reuse',
    'prefill_screen',
    'resource_telemetry',
)
TRUSTED_QUALIFICATION_PATH = Path('/var/lib/localllm-strata/qualifications/sc117-iq3s.json')
QUALIFICATION_REPORT_MAX_BYTES = 8 * 1024 * 1024
TRUSTED_QUALIFICATION_RUNNER = Path('/usr/local/libexec/localllm-strata/sc117-qualifier')
TRUSTED_QUALIFICATION_PIN_POLICY = Path('/etc/localllm-strata/sc117-artifact-pins.json')
# No root-owned evidence runner is installed. Keep this unset so no caller-supplied report is
# treated as qualification; SC117 normal starts remain disabled regardless of any report.
TRUSTED_QUALIFICATION_RUNNER_SHA256 = None
REPORT_FINGERPRINT_INPUT_FIELDS = frozenset({
    'profile', 'model_id', 'model_repository', 'model_revision', 'prepared_asset_manifest_sha256',
    'source_asset_pins', 'native_pack_manifest_sha256', 'native_pack_files', 'runtime_version',
    'runtime_commit', 'runtime_manifest_sha256', 'runtime_assets', 'runtime_python',
    'runtime_python_platform', 'runtime_python_import_surface', 'baseline_config_path',
    'baseline_config_sha256',
})
REPORT_PROBE_ORDER = (
    'smoke', 'count_context', 'native_context', 'vision', 'mtp',
    'cache_A1', 'cache_B1', 'cache_A2', 'cache_B2',
    'count_prefill_1', 'prefill_1', 'count_prefill_2', 'prefill_2',
    'count_prefill_3', 'prefill_3',
)


def profile_root(root):
    return Path(root) / 'profiles' / PROFILE


def qualification_path(root):
    # On Hive/Linux, qualification is an operator-owned attestation outside the
    # model user's writable data tree. Windows keeps a local test fixture path.
    if os.name == 'posix':
        return TRUSTED_QUALIFICATION_PATH
    return profile_root(root) / 'qualification.json'


def _validate_qualification_location(path, *, require_file):
    if os.name != 'posix':
        return
    path = Path(path)
    if path != TRUSTED_QUALIFICATION_PATH:
        raise SafetyError('SC117 qualification must use the fixed root-controlled attestation path')
    directories = (Path('/'), Path('/var'), Path('/var/lib'),
                   Path('/var/lib/localllm-strata'), path.parent)
    for directory in directories:
        try:
            info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 trusted qualification directory is unavailable: {directory}: {exc}') from exc
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or
                stat.S_IMODE(info.st_mode) & 0o022):
            raise SafetyError(f'SC117 trusted qualification directory is not root-controlled: {directory}')
    try:
        info = path.lstat()
    except FileNotFoundError:
        if require_file:
            raise SafetyError('SC117 IQ3_S identity-bound qualification report is missing') from None
        return
    except OSError as exc:
        raise SafetyError(f'SC117 trusted qualification report is unavailable: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
            stat.S_IMODE(info.st_mode) != 0o444):
        raise SafetyError('SC117 trusted qualification report must be root-owned, immutable-mode, regular, and non-symlink')


def _stable_file_read(path, label, *, max_bytes=None, include_contents=False):
    """Read and hash a regular file through one no-follow descriptor, checking stable identity."""
    path = Path(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 {label} is missing or unreadable: {path}: {exc}') from exc
    if not stat.S_ISREG(before.st_mode) or (max_bytes is not None and before.st_size > max_bytes):
        raise SafetyError(f'SC117 {label} must be a bounded regular non-symlink file: {path}')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SafetyError(f'SC117 {label} changed or became unreadable: {path}: {exc}') from exc
    digest = hashlib.sha256()
    contents = [] if include_contents else None
    total = 0
    try:
        try:
            opened = os.fstat(descriptor)
            def identity(info):
                return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                        info.st_uid, info.st_gid, info.st_size)

            def descriptor_state(info):
                return identity(info) + (info.st_mtime_ns, info.st_ctime_ns)

            def path_state(info):
                values = identity(info) + (info.st_mtime_ns,)
                return values + ((info.st_ctime_ns,) if os.name != 'nt' else ())
            if not stat.S_ISREG(opened.st_mode) or identity(before) != identity(opened):
                raise SafetyError(f'SC117 {label} changed while opening: {path}')
            with os.fdopen(descriptor, 'rb', closefd=False) as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if max_bytes is not None and total > max_bytes:
                        raise SafetyError(f'SC117 {label} exceeds its maximum size: {path}')
                    digest.update(block)
                    if contents is not None:
                        contents.append(block)
            after_fd = os.fstat(descriptor)
            after_path = path.lstat()
        except SafetyError:
            raise
        except OSError as exc:
            raise SafetyError(f'SC117 {label} changed while hashing: {path}: {exc}') from exc
        if (descriptor_state(opened) != descriptor_state(after_fd) or
                path_state(opened) != path_state(after_path)):
            raise SafetyError(f'SC117 {label} changed while hashing: {path}')
    finally:
        os.close(descriptor)
    return (digest.hexdigest(), b''.join(contents) if contents is not None else None,
            _metadata_record(path, after_path))


def _stable_file_hash(path, label, *, max_bytes=None):
    return _stable_file_read(path, label, max_bytes=max_bytes)[0]


def _assert_root_controlled_platform_directory(path, label):
    """Require platform Python paths to stay in root-owned, non-writable directories."""
    path = Path(path)
    if not path.is_absolute():
        raise SafetyError(f'SC117 {label} must be an absolute platform path: {path}')
    for directory in reversed((path, *path.parents)):
        try:
            info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 {label} directory is unavailable: {directory}: {exc}') from exc
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or
                stat.S_IMODE(info.st_mode) & 0o022):
            raise SafetyError(f'SC117 {label} must be root-controlled and non-writable: {directory}')


def _python_runtime_platform_identity(environment):
    """Bind venv settings and trust only root-controlled CPython platform directories."""
    pyvenv_cfg = Path(environment) / 'pyvenv.cfg'
    digest, contents, metadata = _stable_file_read(
        pyvenv_cfg, 'Python pyvenv.cfg', max_bytes=64 * 1024, include_contents=True)
    try:
        lines = contents.decode('utf-8').splitlines()
    except UnicodeError as exc:
        raise SafetyError(f'SC117 Python pyvenv.cfg is unreadable: {pyvenv_cfg}: {exc}') from exc
    values = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        key, separator, value = stripped.partition('=')
        key = key.strip().lower()
        if not separator or not key or key in values:
            raise SafetyError(f'SC117 Python pyvenv.cfg has malformed or duplicate settings: {pyvenv_cfg}')
        values[key] = value.strip()
    home_value = values.get('home')
    version = values.get('version')
    version_match = re.fullmatch(r'(\d+)\.(\d+)(?:\.(\d+))?', version or '')
    if (not home_value or not version_match or
            values.get('include-system-site-packages', '').lower() != 'false'):
        raise SafetyError('SC117 Python pyvenv.cfg must identify an absolute home/version and disable system site-packages: '
                          f'{pyvenv_cfg}')
    home = Path(home_value)
    if not home.is_absolute():
        raise SafetyError(f'SC117 Python pyvenv.cfg home must be absolute: {pyvenv_cfg}')
    try:
        resolved_home = home.resolve(strict=True)
        stdlib_candidates = [resolved_home.parent / libdir /
                             f'python{version_match.group(1)}.{version_match.group(2)}'
                             for libdir in ('lib', 'lib64')]
        stdlib_matches = [candidate.resolve(strict=True) for candidate in stdlib_candidates
                          if candidate.is_dir()]
        if not resolved_home.is_dir() or len(set(stdlib_matches)) != 1:
            raise OSError('Python home or unique standard-library path is unavailable')
        stdlib_path = stdlib_matches[0]
        if os.name == 'posix':
            _assert_root_controlled_platform_directory(resolved_home, 'Python home')
            _assert_root_controlled_platform_directory(stdlib_path, 'Python standard library')
        final_metadata = _metadata_record(pyvenv_cfg, pyvenv_cfg.lstat())
    except OSError as exc:
        raise SafetyError(f'SC117 Python home/stdlib path is unavailable: {exc}') from exc
    if final_metadata != metadata:
        raise SafetyError(f'SC117 Python pyvenv.cfg changed during inventory: {pyvenv_cfg}')
    return {
        'python_version': version,
        'python_home': str(resolved_home),
        'stdlib_path': str(stdlib_path),
        'pyvenv_cfg_path': str(pyvenv_cfg),
        'pyvenv_cfg_sha256': digest,
        'pyvenv_cfg_metadata': metadata,
    }


_PYTHON_IMPORT_SUFFIXES = ('.py', '.pyw', '.pyc', '.pyo', '.pth', '.so', '.pyd', '.dll', '.zip', '.egg', '.pyz')
_EXCLUDED_IMPORT_COMPONENTS = frozenset({'.git', '.venv'})


def _is_python_import_file(path):
    return Path(path).name.endswith(_PYTHON_IMPORT_SUFFIXES)


def _import_component_is_valid(component):
    # Dynamic imports can load filesystem names that are not Python identifiers (for example,
    # importlib.import_module('package-name.plugin')). Only VCS/venv metadata is unreachable here.
    lowered = component.lower()
    return (component not in ('', '.', '..') and '.' not in component and
            lowered not in _EXCLUDED_IMPORT_COMPONENTS)


def _is_importable_directory(path, import_root):
    try:
        relative = Path(path).relative_to(Path(import_root))
    except ValueError:
        return False
    return all(_import_component_is_valid(part) for part in relative.parts)


def _is_importable_python_path(path, import_root):
    """Recognize module-shaped files below an actual import root, not build artifacts."""
    path = Path(path)
    if not _is_python_import_file(path) or not _is_importable_directory(path.parent, import_root):
        return False
    name = path.name
    suffix = next(suffix for suffix in _PYTHON_IMPORT_SUFFIXES if name.endswith(suffix))
    if suffix == '.pth':
        # site processes .pth files by suffix; their basename need not be an identifier.
        return True
    if suffix in ('.zip', '.egg', '.pyz', '.pyc', '.pyo'):
        # Bytecode caches are read while importing their corresponding .py module even though
        # CPython's cache basename includes an ABI tag (module.cpython-312.pyc).
        return bool(name[:-len(suffix)])
    module_name = name[:-len(suffix)]
    if suffix in ('.so', '.pyd', '.dll'):
        # Native extensions may carry an ABI tag between the module name and suffix.
        module_name = module_name.split('.', 1)[0]
    return bool(module_name) and '.' not in module_name


def _path_within(path, roots):
    path = Path(path)
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def _checked_runtime_directory(path, label):
    path = Path(path)
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SafetyError(f'SC117 {label} is missing or unavailable: {path}: {exc}') from exc
    if not stat.S_ISDIR(info.st_mode) or resolved != path.absolute():
        raise SafetyError(f'SC117 {label} must be a real non-symlink directory: {path}')
    return path


def _python_site_packages(environment, label):
    matches = sorted((Path(environment) / 'lib').glob('python*/site-packages'))
    if len(matches) != 1:
        raise SafetyError(f'SC117 {label} must contain exactly one Python site-packages root')
    return _checked_runtime_directory(matches[0], label + ' site-packages')


def _tracked_import_files(source):
    try:
        result = subprocess.run(['git', '-c', f'safe.directory={source}', '-C', str(source),
                                 'ls-files', '-z', '--cached'], check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'SC117 Strata source import inventory failed: {exc}') from exc
    if result.returncode != 0:
        raise SafetyError('SC117 Strata source is not a readable Git checkout for import verification')
    try:
        return {Path(os.fsdecode(record)).as_posix() for record in result.stdout.split(b'\0') if record}
    except (UnicodeError, ValueError) as exc:
        raise SafetyError(f'SC117 Strata source import inventory is malformed: {exc}') from exc


def _metadata_record(path, info):
    return (str(path), stat.S_IMODE(info.st_mode), info.st_dev, info.st_ino, info.st_nlink,
            info.st_uid, info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _runtime_python_import_surface(root, runtime_root, runtime_source, manifest, *,
                                   allow_inactive_bytecode=False):
    """Inventory importable Python paths without starting the prepared interpreter."""
    root = Path(root)
    runtime_root = Path(runtime_root)
    source = _checked_runtime_directory(runtime_source, 'Strata source import root')
    shared_environment = root / 'source-99f3dbd0b21d/.venv'
    shared_environment = _checked_runtime_directory(shared_environment, 'shared Strata Python environment')
    overlay_path = runtime_root / 'python'
    environment_value = manifest.get('python_environment')
    if not isinstance(environment_value, str) or Path(environment_value) not in (overlay_path, shared_environment):
        raise SafetyError('SC117 Strata Python environment is outside the prepared overlay/shared roots')
    environment = Path(environment_value)
    overlay = None
    overlay_site = None
    if environment == overlay_path:
        overlay = _checked_runtime_directory(overlay_path, 'Strata Python overlay')
        overlay_site = _python_site_packages(overlay, 'Strata Python overlay')
    try:
        environment_resolved = environment.resolve(strict=True)
        venv_info = (source / '.venv').lstat()
        venv_resolved = (source / '.venv').resolve(strict=True)
    except OSError as exc:
        raise SafetyError(f'SC117 prepared source/.venv link is unavailable: {exc}') from exc
    if (not stat.S_ISLNK(venv_info.st_mode) or venv_resolved != environment_resolved):
        raise SafetyError('SC117 source/.venv must link to its recorded overlay or shared environment')
    shared_site = _python_site_packages(shared_environment, 'shared Strata Python environment')
    site_roots = (overlay_site, shared_site) if overlay_site is not None else (shared_site,)
    import_roots = (source,) + site_roots
    environment_roots = (overlay, shared_environment) if overlay is not None else (shared_environment,)
    tracked = _tracked_import_files(source)
    records = []
    import_files = []
    pth_files = []
    pth_hashes = {}
    directory_snapshots = []

    def visit(directory, *, import_root, source_tree=False):
        if not _is_importable_directory(directory, import_root):
            return
        if directory.name.lower() in ('sitecustomize', 'usercustomize'):
            raise SafetyError(f'SC117 Python import surface rejects customization package: {directory}')
        try:
            dir_info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 Python import directory is unavailable: {directory}: {exc}') from exc
        if not stat.S_ISDIR(dir_info.st_mode):
            raise SafetyError(f'SC117 Python import directory must be a non-symlink directory: {directory}')
        records.append(('directory',) + _metadata_record(directory, dir_info))
        directory_snapshots.append((directory, _metadata_record(directory, dir_info)))
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SafetyError(f'SC117 Python import directory is unreadable: {directory}: {exc}') from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SafetyError(f'SC117 Python import entry is unavailable: {path}: {exc}') from exc
            if source_tree and path == source / '.venv':
                # This one alias is part of prepare_strata_runtime.py's documented layout.
                if not stat.S_ISLNK(info.st_mode):
                    raise SafetyError('SC117 source/.venv must be the prepared environment symlink')
                continue
            if source_tree and path == source / '.git':
                if not stat.S_ISDIR(info.st_mode):
                    raise SafetyError('SC117 Strata source .git metadata must be a directory')
                continue
            if stat.S_ISLNK(info.st_mode):
                # Symlinks beneath excluded/non-module components cannot participate in imports.
                relative_parts = path.relative_to(import_root).parts
                if any(not _import_component_is_valid(part) for part in relative_parts):
                    continue
                raise SafetyError(f'SC117 Python import surface rejects symlink: {path}')
            if stat.S_ISDIR(info.st_mode):
                visit(path, import_root=import_root, source_tree=source_tree)
                continue
            if not _is_importable_python_path(path, import_root):
                continue
            if path.suffix.lower() in ('.pyc', '.pyo'):
                # Only an explicit, isolated trial may ignore canonical __pycache__ entries, and
                # only because its server subprocess uses PYTHONPYCACHEPREFIX to bypass them.
                # Qualification and normal readiness continue to reject every executable cache.
                if allow_inactive_bytecode and path.suffix.lower() == '.pyc' and path.parent.name == '__pycache__' and stat.S_ISREG(info.st_mode):
                    try:
                        source_path = Path(importlib.util.source_from_cache(str(path)))
                        source_info = source_path.lstat()
                    except (OSError, ValueError, NotImplementedError) as exc:
                        raise SafetyError(f'SC117 trial Python cache has no canonical source: {path}') from exc
                    if source_path.parent != path.parent.parent or not stat.S_ISREG(source_info.st_mode):
                        raise SafetyError(f'SC117 trial Python cache has no regular adjacent source: {path}')
                    continue
                raise SafetyError(f'SC117 runtime rejects executable cached Python bytecode: {path}')
            if not stat.S_ISREG(info.st_mode):
                raise SafetyError(f'SC117 Python import artifact must be a regular file: {path}')
            lowered_name = path.name.lower()
            if lowered_name.startswith(('sitecustomize.', 'usercustomize.')):
                raise SafetyError(f'SC117 Python import surface rejects customization module: {path}')
            if source_tree:
                relative = path.relative_to(source).as_posix()
                if relative not in tracked:
                    raise SafetyError(f'SC117 untracked Python import artifact: {path}')
            import_files.append(path)
            if path.name.endswith('.pth'):
                pth_files.append((path, directory))

    for source_root in import_roots:
        visit(source_root, import_root=source_root, source_tree=(source_root == source))

    expected_pth = overlay_site / 'strata_shared_readonly.pth' if overlay_site is not None else None
    trial_distutils_pths = ({site / TRIAL_DISTUTILS_PTH_RELATIVE for site in site_roots}
                             if allow_inactive_bytecode else set())
    allowed_pth = {expected_pth} if expected_pth is not None else set()
    allowed_pth.update(trial_distutils_pths)
    if any(path not in allowed_pth for path, _ in pth_files):
        unexpected = next(path for path, _ in pth_files if path not in allowed_pth)
        raise SafetyError(f'SC117 Python .pth file is unexpected or in an unapproved location: {unexpected}')
    if expected_pth is not None and all(path != expected_pth for path, _ in pth_files):
        raise SafetyError(f'SC117 required overlay Python .pth file is missing: {expected_pth}')

    for pth_path, _site_root in pth_files:
        try:
            digest, contents, stable_metadata = _stable_file_read(
                pth_path, 'Python .pth file', max_bytes=1024 * 1024, include_contents=True)
            lines = contents.decode('utf-8').splitlines()
        except UnicodeError as exc:
            raise SafetyError(f'SC117 Python .pth file is unreadable: {pth_path}: {exc}') from exc
        pth_hashes[pth_path] = (digest, stable_metadata)
        if pth_path in trial_distutils_pths:
            if (digest != TRIAL_DISTUTILS_PTH_SHA256 or contents != TRIAL_DISTUTILS_PTH_CONTENTS or
                    os.environ.get('SETUPTOOLS_USE_DISTUTILS') != 'stdlib'):
                raise SafetyError('SC117 trial distutils .pth differs from its pinned no-op behavior')
            continue
        added_paths = []
        for line in lines:
            candidate = line.strip()
            if candidate.startswith(chr(0xfeff)):
                candidate = candidate[1:].lstrip()
            if not candidate or candidate.startswith('#'):
                continue
            if candidate.startswith(('import ', 'import' + chr(9))):
                raise SafetyError(f'SC117 unexpected executable .pth line: {pth_path}')
            if candidate != str(shared_site):
                raise SafetyError(f'SC117 Python .pth path reaches an excluded tree or differs from the expected shared site-packages: {pth_path}')
            added_paths.append(candidate)
        if added_paths != [str(shared_site)]:
            raise SafetyError(f'SC117 Python .pth must contain exactly the shared site-packages path: {pth_path}')

    # Ignore artifacts in excluded/non-importable trees, while retaining a fail-closed check for
    # Python-shaped files placed at the runtime root rather than one of its declared import roots.
    def walk_error(exc):
        raise SafetyError(f'SC117 runtime import inventory is unreadable: {exc}') from exc

    def in_environment_without_site_packages(path):
        return (_path_within(path, environment_roots) and
                not _path_within(path, site_roots))

    for directory, dirnames, filenames in os.walk(runtime_root, topdown=True, onerror=walk_error,
                                                  followlinks=False):
        current = Path(directory)
        kept = []
        for dirname in dirnames:
            child = current / dirname
            if child == source / '.git' or child == source / '.venv':
                continue
            if _path_within(child, (source,)) and not _is_importable_directory(child, source):
                continue
            if (_path_within(child, site_roots) and
                    not any(_is_importable_directory(child, site) for site in site_roots)):
                continue
            if not in_environment_without_site_packages(child):
                relative_to_runtime = child.relative_to(runtime_root).parts
                if any(not _import_component_is_valid(part) for part in relative_to_runtime):
                    continue
            try:
                info = child.lstat()
            except OSError as exc:
                raise SafetyError(f'SC117 runtime import inventory changed during scan: {child}: {exc}') from exc
            if stat.S_ISLNK(info.st_mode):
                if _is_python_import_file(child):
                    import_root = next((candidate_root for candidate_root in import_roots
                                        if _path_within(child, (candidate_root,))), None)
                    if (import_root is None or
                            _is_importable_python_path(child, import_root) or
                            _is_importable_python_path(child, runtime_root)):
                        raise SafetyError(f'SC117 Python import surface rejects symlink: {child}')
                continue
            kept.append(dirname)
        dirnames[:] = kept
        for filename in filenames:
            path = current / filename
            if not _is_python_import_file(path):
                continue
            if in_environment_without_site_packages(path):
                continue
            import_root = next((candidate_root for candidate_root in import_roots
                                if _path_within(path, (candidate_root,))), None)
            if import_root is not None:
                if not _is_importable_python_path(path, import_root):
                    continue
            elif not _is_importable_python_path(path, runtime_root):
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                raise SafetyError(f'SC117 runtime import inventory changed during scan: {path}: {exc}') from exc
            if import_root is None:
                raise SafetyError(f'SC117 runtime Python import file is outside allowed roots: {path}')
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SafetyError(f'SC117 Python import artifact must be a regular non-symlink file: {path}')

    for path in import_files:
        if path in pth_hashes:
            digest, stable_metadata = pth_hashes[path]
        else:
            digest, _, stable_metadata = _stable_file_read(path, 'runtime Python import artifact')
        try:
            after_info = path.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 Python import artifact changed during inventory: {path}: {exc}') from exc
        final_metadata = _metadata_record(path, after_info)
        if not stat.S_ISREG(after_info.st_mode) or final_metadata != stable_metadata:
            raise SafetyError(f'SC117 Python import artifact changed during inventory: {path}')
        records.append(('file',) + final_metadata + (digest,))
    for directory, before_record in directory_snapshots:
        try:
            after_info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 Python import directory changed during inventory: {directory}: {exc}') from exc
        if (not stat.S_ISDIR(after_info.st_mode) or
                _metadata_record(directory, after_info) != before_record):
            raise SafetyError(f'SC117 Python import directory changed during inventory: {directory}')
    return sorted(records)


def _baseline_paths(root):
    root = Path(root)
    old_shard2 = root / assets.SHARD2_REUSE_SOURCE_RELATIVE
    old_shard1_name = old_shard2.name.replace('00002-of-00002', '00001-of-00002')
    return {
        'native': old_shard2.with_name(old_shard1_name),
        'ple': old_shard2,
        'pack': root / 'data/packs/iq3_s',
        'mtp': root / 'data/mtp/rt',
        'projector': root / 'data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf',
    }


def _flag_value(args, flag, *, expected=None):
    if any(item.startswith(flag + '=') for item in args):
        raise SafetyError(f'Validated .39 IQ3_S config must use separate {flag} and value arguments')
    if args.count(flag) != 1:
        raise SafetyError(f'Validated .39 IQ3_S config must contain exactly one {flag}')
    index = args.index(flag)
    if index + 1 >= len(args) or args[index + 1].startswith('--'):
        raise SafetyError(f'Validated .39 IQ3_S {flag} has no value')
    value = args[index + 1]
    if expected is not None and value != str(expected):
        raise SafetyError(f'Validated .39 IQ3_S config has an unexpected {flag} path/value')
    return value


def _validate_baseline_config(root, cfg):
    """Fail closed unless the input is the already-validated .39 IQ3_S baseline config."""
    if not isinstance(cfg, dict):
        raise SafetyError('Validated .39 IQ3_S config must be an object')
    source = assets.strata_039_source(Path(root))
    if (cfg.get('exe') != str(source / 'engine/strata') or cfg.get('cwd') != str(source) or
            cfg.get('gpu') != [0, 1, 2, 3] or cfg.get('layer_split') != 'auto'):
        raise SafetyError('SC117 requires the validated four-GPU Strata 0.1.39 baseline config')
    args = cfg.get('args')
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise SafetyError('Validated .39 IQ3_S config has malformed engine arguments')
    paths = _baseline_paths(root)
    for flag, key in (('--native', 'native'), ('--ple-gguf', 'ple'), ('--pack', 'pack'), ('--mtp', 'mtp')):
        _flag_value(args, flag, expected=paths[key])
    for flag, expected in (('--max-context', '262144'), ('--kv', 'int8'),
                           ('--kv-resident', '32768'), ('--spec', '4')):
        _flag_value(args, flag, expected=expected)
    if ('--vision' not in args or args.count('--vision') != 1 or
            any(item.startswith('--vision=') for item in args)):
        raise SafetyError('Validated .39 IQ3_S config must retain GPU vision')
    parallel = cfg.get('parallel')
    for flag in ('--batch-groups', '--parallel'):
        if ((flag in args or any(item.startswith(flag + '=') for item in args)) and
                _flag_value(args, flag) != '1'):
            raise SafetyError('SC117 IQ3_S is restricted to one serving slot and one batch group')
    if parallel is not None and (type(parallel) is not int or parallel != 1):
        raise SafetyError('SC117 IQ3_S is restricted to one serving slot and one batch group')
    if cfg.get('tokenizer') != str(paths['pack'] / 'tokenizer'):
        raise SafetyError('Validated .39 IQ3_S config has an unexpected tokenizer path')
    vision = cfg.get('vision')
    if (not isinstance(vision, dict) or vision.get('exe') != str(source / 'engine/strata-vision') or
            vision.get('model') != str(paths['native']) or vision.get('mmproj') != str(paths['projector']) or
            vision.get('gpu') is not True):
        raise SafetyError('Validated .39 IQ3_S config does not use the baseline BF16 GPU-vision model')
    # Candidate tuning flags are applied after validation. The baseline must not already contain
    # cache options, and exactly one prefill value is required for controlled overrides.
    _flag_value(args, '--prefill')
    for flag in ('--conversation-cache-mib', '--conversation-cache-slots'):
        if flag in args or any(item.startswith(flag + '=') for item in args):
            raise SafetyError(f'Validated .39 IQ3_S baseline unexpectedly contains {flag}')
    return args, paths


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise SafetyError('SC117 IQ3_S settings must contain exactly the pinned setting keys')
    for key in ('context', 'kv_resident', 'mtp_depth', 'parallel', 'batch_groups'):
        if type(settings[key]) is not int or settings[key] != DEFAULT_SETTINGS[key]:
            raise SafetyError('SC117 IQ3_S must retain native 262144 context, INT8 KV, 32768 resident, MTP=4, one slot/group')
    if (type(settings['kv']) is not str or settings['kv'] != 'int8' or
            type(settings['vision']) is not str or settings['vision'] != 'gpu-bf16'):
        raise SafetyError('SC117 IQ3_S must retain INT8 KV and BF16 GPU vision')
    prefill = settings['prefill']
    if not isinstance(prefill, str) or prefill not in ('auto', '1024', '2048', '4096', '8192'):
        raise SafetyError('SC117 IQ3_S prefill must be auto or a supported 1K–8K preset')
    split = settings['layer_split']
    if not isinstance(split, str):
        raise SafetyError('SC117 IQ3_S split must be auto or three comma-separated stage boundaries')
    if split != 'auto':
        if not re.fullmatch(r'[0-9]+,[0-9]+,[0-9]+', split):
            raise SafetyError('SC117 IQ3_S split must be auto or three comma-separated stage boundaries')
        boundaries = tuple(int(value) for value in split.split(','))
        if not (0 < boundaries[0] < boundaries[1] < boundaries[2] <= 36):
            raise SafetyError('SC117 IQ3_S trial split boundaries must increase within the 36-layer model')
    mib, slots = settings['conversation_cache_mib'], settings['conversation_cache_slots']
    if type(mib) is not int or mib not in (0, 4096, 8192, 12288, 16384):
        raise SafetyError('SC117 IQ3_S conversation cache must be 0, 4096, 8192, 12288, or 16384 MiB')
    if (type(slots) is not int or (mib == 0 and slots != 0) or
            (mib > 0 and slots not in (1, 2, 4))):
        raise SafetyError('SC117 IQ3_S cache slots must be 0 when disabled, otherwise 1, 2, or 4')


def settings_for_start(*, allow_trial_env=False):
    """Resolve only the bounded trial tuning variables; no fixed contract is overridable."""
    settings = dict(DEFAULT_SETTINGS)
    marker = os.environ.get(TRIAL_ENV)
    supplied_trial_names = {name for name in os.environ if name.startswith(TRIAL_ENV_PREFIX)}
    unknown = supplied_trial_names - set(TRIAL_OVERRIDE_ENV) - TRIAL_EXPERIMENT_ENV
    if unknown:
        raise SafetyError(f'Unallowlisted SC117 IQ3_S trial override: {sorted(unknown)[0]}')
    if marker is not None and marker != '1':
        raise SafetyError(f'SC117 IQ3_S trial mode requires {TRIAL_ENV}=1')
    if supplied_trial_names and marker != '1':
        raise SafetyError(f'SC117 IQ3_S trial overrides require explicit {TRIAL_ENV}=1')
    if marker == '1' and not allow_trial_env:
        raise SafetyError('SC117 IQ3_S trial settings are not allowed for normal readiness')
    if marker == '1':
        for name, setting in TRIAL_OVERRIDE_ENV.items():
            value = os.environ.get(name)
            if value is None:
                continue
            if setting in ('conversation_cache_mib', 'conversation_cache_slots'):
                if not re.fullmatch(r'[0-9]+', value):
                    raise SafetyError(f'Invalid SC117 IQ3_S trial override: {name}')
                value = int(value)
            settings[setting] = value
    validate_settings(settings)
    return settings


def apply_settings(cfg, settings):
    """Apply tunables only to a deep copy of the validated baseline-derived config."""
    validate_settings(settings)
    cloned = copy.deepcopy(cfg)
    args = cloned.get('args')
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise SafetyError('SC117 effective config has malformed engine arguments')
    _flag_value(args, '--prefill')
    args[args.index('--prefill') + 1] = settings['prefill']
    if any(flag in args or any(item.startswith(flag + '=') for item in args)
           for flag in ('--conversation-cache-mib', '--conversation-cache-slots')):
        raise SafetyError('SC117 baseline config already contains conversation-cache options')
    if settings['conversation_cache_mib']:
        args.extend(['--conversation-cache-mib', str(settings['conversation_cache_mib']),
                     '--conversation-cache-slots', str(settings['conversation_cache_slots'])])
    cloned['layer_split'] = settings['layer_split']
    cloned['model_name'] = MODEL_ID
    cloned['aliases'] = ['qwen38-iq3s-sc117-abliterated', PROFILE]
    return cloned


def build_effective_config(root, validated_baseline_cfg, settings=None):
    """Clone a validated .39 config and point only model/pack/MTP/tokenizer/vision at SC117."""
    args, _ = _validate_baseline_config(root, validated_baseline_cfg)
    settings = dict(DEFAULT_SETTINGS) if settings is None else settings
    cloned = copy.deepcopy(validated_baseline_cfg)
    rewires = {
        '--native': assets.shard1_path(root),
        '--ple-gguf': assets.shard2_path(root),
        '--pack': assets.pack_root(root),
        '--mtp': assets.mtp_runtime_root(root),
    }
    for flag, destination in rewires.items():
        cloned['args'][cloned['args'].index(flag) + 1] = str(destination)
    cloned['tokenizer'] = str(assets.pack_root(root) / 'tokenizer')
    cloned['vision']['model'] = str(assets.shard1_path(root))
    cloned['vision']['mmproj'] = str(assets.projector_path(root))
    return apply_settings(cloned, settings)


def default_readiness_line():
    return ('PROVISIONAL CONFIG: SC117 abliterated IQ3_S; Strata=0.1.39; context=262144; INT8 KV '
            '(32768 resident); MTP=4 from the stock/non-abliterated head; BF16 GPU vision; one serving slot/group; '
            'prefill auto; split auto; conversation cache=8192 MiB / 4 slots. These starting defaults are not qualified; '
            'live smoke, near-native context, GPU vision, measurable MTP acceptance, cache park/restore/reuse, '
            'prefill screen, and telemetry are required.')


def readiness_line(settings=None):
    """Describe effective settings without implying that configuration equals qualification."""
    settings = dict(DEFAULT_SETTINGS) if settings is None else settings
    validate_settings(settings)
    cache = (f"{settings['conversation_cache_mib']} MiB / {settings['conversation_cache_slots']} slots"
             if settings['conversation_cache_mib'] else 'disabled')
    mtp_note = 'SC117 strata/rt; stock/non-abliterated head'
    return (f'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime={RUNTIME_VERSION}; context=262144; '
            f'INT8 KV (32768 resident); MTP=4 ({mtp_note}); BF16 GPU vision; one serving slot/group; '
            f'prefill={settings["prefill"]}; split={settings["layer_split"]}; '
            f'conversation cache={cache}.')


def machine_identity():
    """Identify the host and all four GPUs so qualification cannot be copied to another node."""
    if os.name != 'posix':
        raise SafetyError('SC117 qualification requires a Linux host identity')
    machine_id_path = Path('/etc/machine-id')
    try:
        machine_info = machine_id_path.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 host machine-id is unavailable: {exc}') from exc
    if (not stat.S_ISREG(machine_info.st_mode) or machine_info.st_uid != 0 or
            stat.S_IMODE(machine_info.st_mode) & 0o022):
        raise SafetyError('SC117 host machine-id must be a root-controlled regular file')
    machine_id_sha256 = _stable_file_hash(machine_id_path, 'host machine-id', max_bytes=256)
    try:
        machine_id = machine_id_path.read_text(encoding='ascii').strip()
    except (OSError, UnicodeError) as exc:
        raise SafetyError(f'SC117 host machine-id is unavailable: {exc}') from exc
    if not re.fullmatch(r'[0-9a-fA-F]{32}', machine_id):
        raise SafetyError('SC117 host machine-id is malformed')
    smi = Path('/usr/bin/nvidia-smi')
    try:
        resolved_smi = smi.resolve(strict=True)
        smi_info = resolved_smi.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 trusted nvidia-smi binary is unavailable: {exc}') from exc
    directories = {Path('/'), smi.parent, *resolved_smi.parents}
    for directory in directories:
        try:
            directory_info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 GPU inventory path is unavailable: {directory}: {exc}') from exc
        if (not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != 0 or
                stat.S_IMODE(directory_info.st_mode) & 0o022):
            raise SafetyError(f'SC117 GPU inventory path is not root-controlled: {directory}')
    if (not stat.S_ISREG(smi_info.st_mode) or smi_info.st_uid != 0 or
            stat.S_IMODE(smi_info.st_mode) & 0o022):
        raise SafetyError('SC117 nvidia-smi binary must be root-controlled and non-writable')
    try:
        output = subprocess.check_output(
            [str(smi), '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.PIPE, timeout=15,
            env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'SC117 trusted GPU inventory failed: {exc}') from exc
    rows = [row for row in csv.reader(output.splitlines()) if row and any(field.strip() for field in row)]
    if len(rows) != 4 or any(len(row) != 2 for row in rows):
        raise SafetyError('SC117 qualification requires exactly four inventoried GPU UUID/driver records')
    uuids = [row[0].strip() for row in rows]
    drivers = {row[1].strip() for row in rows}
    if (len(set(uuids)) != 4 or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', item) for item in uuids) or
            len(drivers) != 1 or not next(iter(drivers))):
        raise SafetyError('SC117 GPU identity or driver inventory is malformed/inconsistent')
    return {'machine_id_sha256': machine_id_sha256, 'gpu_uuids': sorted(uuids),
            'driver_version': next(iter(drivers))}


def fingerprint(root, settings, *, verify_content=False, _machine_identity=None,
                allow_inactive_bytecode=False):
    """Bind live evidence to pinned asset/pack contents, .39 runtime, baseline config, and settings.

    Qualification issuance requests full verification of the pinned model and native-pack assets.
    The small Python import surface is content-hashed on every call, using stable no-follow reads,
    so normal readiness binds imported bytes as well as paths and filesystem metadata.
    """
    root = Path(root)
    validate_settings(settings)
    try:
        # Full hashing is mandatory before an attestation is issued. Normal status checks also
        # rehash the bounded Python import surface below, without rehashing large model/pack assets.
        prepare_sc117.verify_readiness(root, full_hash=verify_content)
        prepared_manifest_sha = _stable_file_hash(assets.manifest_path(root), 'prepared asset manifest',
                                                  max_bytes=1024 * 1024)
        pack_manifest_path = assets.pack_manifest_path(root)
        pack_manifest_sha = _stable_file_hash(pack_manifest_path, 'native pack manifest',
                                              max_bytes=1024 * 1024)
        pack_manifest_value = json.loads(pack_manifest_path.read_text(encoding='utf-8'))
        pack_files = pack_manifest_value.get('pack_files')
        if not isinstance(pack_files, list):
            raise SafetyError('SC117 native pack manifest has no file identities')
        normalized_pack_files = []
        for record in pack_files:
            if (not isinstance(record, dict) or not isinstance(record.get('path'), str) or
                    type(record.get('bytes')) is not int or
                    not isinstance(record.get('sha256'), str) or
                    not re.fullmatch(r'[a-f0-9]{64}', record['sha256'])):
                raise SafetyError('Malformed SC117 native pack file identity')
            normalized_pack_files.append((record['path'], record['bytes'], record['sha256']))
        if len({row[0] for row in normalized_pack_files}) != len(normalized_pack_files):
            raise SafetyError('Duplicate SC117 native pack file identity')
        source_asset_pins = sorted((item.repo_path, item.role, item.size,
                                    item.sha256 or item.git_blob_oid) for item in assets.ASSETS)
        baseline_config = root / BASELINE_CONFIG_RELATIVE
        baseline_config_sha = _stable_file_hash(baseline_config, '.39 baseline config')
        runtime_root = strata_runtime.runtime_root(root, RUNTIME_VERSION)
        runtime_manifest_path = runtime_root / 'prepared.json'
        runtime_manifest_sha = _stable_file_hash(runtime_manifest_path, '.39 runtime manifest',
                                                 max_bytes=8 * 1024 * 1024)
        runtime_source, runtime_manifest = strata_runtime.prepared_runtime(root, RUNTIME_VERSION)
        if (runtime_source != runtime_root / 'source' or
                runtime_manifest.get('version') != RUNTIME_VERSION or
                runtime_manifest.get('source_commit') != RUNTIME_COMMIT or
                runtime_manifest.get('source') != str(runtime_root / 'source')):
            raise SafetyError('SC117 qualification requires the exact prepared Strata 0.1.39 runtime pin')
        runtime_python_import_surface = _runtime_python_import_surface(
            root, runtime_root, runtime_source, runtime_manifest,
            allow_inactive_bytecode=allow_inactive_bytecode)
        runtime_python = (runtime_source / '.venv/bin/python').resolve(strict=True)
        runtime_python_sha = _stable_file_hash(runtime_python, 'pinned .39 Python interpreter',
                                               max_bytes=512 * 1024 * 1024)
        runtime_python_platform = _python_runtime_platform_identity(
            Path(runtime_manifest['python_environment']))
        runtime_assets = runtime_manifest.get('runtime_assets')
        if not isinstance(runtime_assets, list) or not runtime_assets:
            raise SafetyError('Pinned Strata 0.1.39 runtime manifest has no runtime asset records')
        normalized_runtime_assets = []
        for record in runtime_assets:
            if (not isinstance(record, dict) or not isinstance(record.get('path'), str) or
                    type(record.get('bytes')) is not int or
                    not isinstance(record.get('sha256'), str) or
                    not re.fullmatch(r'[a-f0-9]{64}', record['sha256'])):
                raise SafetyError('Malformed Strata 0.1.39 runtime asset record')
            normalized_runtime_assets.append((record['path'], record['bytes'], record['sha256']))
        if len({row[0] for row in normalized_runtime_assets}) != len(normalized_runtime_assets):
            raise SafetyError('Duplicate Strata 0.1.39 runtime asset record')
    except SafetyError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SafetyError(f'SC117 qualification provenance is unavailable: {exc}') from exc
    machine = _machine_identity if _machine_identity is not None else machine_identity()
    identity = {
        'profile': PROFILE,
        'machine_identity': machine,
        'model_id': MODEL_ID,
        'model_repository': assets.REPOSITORY,
        'model_revision': assets.REVISION,
        'prepared_asset_manifest_sha256': prepared_manifest_sha,
        'source_asset_pins': source_asset_pins,
        'native_pack_manifest_sha256': pack_manifest_sha,
        'native_pack_files': sorted(normalized_pack_files),
        'runtime_version': RUNTIME_VERSION,
        'runtime_commit': RUNTIME_COMMIT,
        'runtime_manifest_sha256': runtime_manifest_sha,
        'runtime_assets': sorted(normalized_runtime_assets),
        'runtime_python': (str(runtime_python), runtime_python_sha),
        'runtime_python_platform': runtime_python_platform,
        'runtime_python_import_surface': runtime_python_import_surface,
        'baseline_config_path': str(baseline_config),
        'baseline_config_sha256': baseline_config_sha,
        'settings': dict(settings),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def trusted_qualification_pin_policy_sha256():
    """Hash the root-only reviewed runtime/interpreter/import/baseline/pack pin policy."""
    path = TRUSTED_QUALIFICATION_PIN_POLICY
    _assert_root_controlled_platform_directory(path.parent, 'SC117 artifact-pin policy')
    try:
        info = path.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 trusted artifact-pin policy is unavailable: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
            stat.S_IMODE(info.st_mode) != 0o444 or info.st_size > 1024 * 1024):
        raise SafetyError('SC117 artifact-pin policy must be root-owned, mode 0444, regular, and bounded')
    return _stable_file_hash(path, 'trusted SC117 artifact-pin policy', max_bytes=1024 * 1024)


def _report_profile_fingerprint(artifact_inputs, machine, settings):
    if (not isinstance(artifact_inputs, dict) or
            set(artifact_inputs) != REPORT_FINGERPRINT_INPUT_FIELDS):
        raise SafetyError('SC117 qualification report fingerprint inputs have an unexpected field set')
    identity = dict(artifact_inputs)
    identity['machine_identity'] = machine
    identity['settings'] = dict(settings)
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _canonical_report_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def trusted_qualification_runner_sha256():
    """Return the hash of the installed root-owned evidence runner, or fail closed."""
    expected = TRUSTED_QUALIFICATION_RUNNER_SHA256
    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise SafetyError('SC117 trusted evidence runner is not installed; qualification reports cannot unlock normal hosting')
    path = TRUSTED_QUALIFICATION_RUNNER
    directories = (Path('/'), Path('/usr'), Path('/usr/local'), Path('/usr/local/libexec'), path.parent)
    for directory in directories:
        try:
            directory_info = directory.lstat()
        except OSError as exc:
            raise SafetyError(f'SC117 trusted evidence-runner directory is unavailable: {directory}: {exc}') from exc
        if (not stat.S_ISDIR(directory_info.st_mode) or directory_info.st_uid != 0 or
                stat.S_IMODE(directory_info.st_mode) & 0o022):
            raise SafetyError(f'SC117 trusted evidence-runner directory is not root-controlled: {directory}')
    try:
        info = path.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 trusted evidence runner is unavailable: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
            stat.S_IMODE(info.st_mode) != 0o555):
        raise SafetyError('SC117 evidence runner must be root-owned, mode 0555, regular, and non-symlink')
    actual = _stable_file_hash(path, 'trusted evidence runner', max_bytes=16 * 1024 * 1024)
    if actual != expected:
        raise SafetyError('SC117 trusted evidence runner hash differs from the reviewed pin')
    return actual


def _validate_report(report, settings, expected_fingerprint, runner_sha256, *, machine=None,
                     pin_policy_sha256=None):
    required = {
        'schema_version', 'status', 'evidence_runner_sha256', 'profile', 'model_id',
        'runtime_version', 'runtime_commit', 'profile_fingerprint', 'profile_fingerprint_inputs',
        'machine_identity', 'settings', 'checks', 'smoke', 'native_context', 'vision',
        'mtp_acceptance', 'conversation_cache', 'prefill_screen', 'resource_telemetry',
        'live_identity', 'artifact_identity', 'measured_observations', 'recorded_at',
    }
    if not isinstance(report, dict) or set(report) != required:
        return 'SC117 IQ3_S qualification report fields are missing or unreviewed'
    if (type(report.get('schema_version')) is not int or report['schema_version'] != 1 or
            report.get('status') != 'passed' or report.get('evidence_runner_sha256') != runner_sha256 or
            report.get('profile') != PROFILE or report.get('model_id') != MODEL_ID or
            report.get('runtime_version') != RUNTIME_VERSION or report.get('runtime_commit') != RUNTIME_COMMIT):
        return 'SC117 IQ3_S qualification report identity does not match the pinned runner/profile'
    current_machine = machine if machine is not None else machine_identity()
    if report.get('machine_identity') != current_machine:
        return 'SC117 IQ3_S qualification report belongs to a different host/GPU identity'
    report_settings = report.get('settings')
    try:
        validate_settings(report_settings)
    except SafetyError:
        return 'SC117 IQ3_S qualification report settings are malformed'
    if report_settings != settings:
        return 'SC117 IQ3_S qualification report settings do not match the requested config'
    try:
        computed_fingerprint = _report_profile_fingerprint(
            report['profile_fingerprint_inputs'], current_machine, report_settings)
    except (SafetyError, TypeError, ValueError):
        return 'SC117 IQ3_S report profile fingerprint inputs are malformed'
    if (computed_fingerprint != report.get('profile_fingerprint') or
            computed_fingerprint != expected_fingerprint):
        return 'SC117 IQ3_S report profile fingerprint does not recompute from current identity/settings'
    checks = report.get('checks')
    if (not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS) or
            any(checks.get(name) is not True for name in REQUIRED_CHECKS)):
        return 'SC117 IQ3_S qualification report lacks a required true check'
    context = report.get('native_context')
    if (not isinstance(context, dict) or type(context.get('prompt_tokens')) is not int or
            not 255000 <= context['prompt_tokens'] <= 262144 or
            type(context.get('completion_tokens')) is not int or context['completion_tokens'] < 1 or
            context.get('tail_marker_found') is not True):
        return 'SC117 IQ3_S report needs a verified native-context tail at >=255K prompt tokens'
    vision = report.get('vision')
    if (not isinstance(vision, dict) or vision.get('passed') is not True or
            vision.get('gpu') is not True or vision.get('precision') != 'bf16' or
            vision.get('projector_sha256') != assets.ASSETS_BY_ROLE['vision_projector'].sha256):
        return 'SC117 IQ3_S report lacks the pinned BF16 GPU-vision result'
    cache = report.get('conversation_cache')
    if (not isinstance(cache, dict) or any(cache.get(name) is not True
                                           for name in ('parked', 'restored', 'reused'))):
        return 'SC117 IQ3_S report lacks conversation-cache park/restore/reuse evidence'
    prefill = report.get('prefill_screen')
    if (not isinstance(prefill, dict) or prefill.get('passed') is not True or
            prefill.get('selected_prefill') != settings['prefill'] or
            not isinstance(prefill.get('samples'), list) or len(prefill['samples']) != 3):
        return 'SC117 IQ3_S report lacks three measured prefill samples for the selected setting'
    mtp = report.get('mtp_acceptance')
    if (not isinstance(mtp, dict) or mtp.get('passed') is not True or
            mtp.get('depth') != settings['mtp_depth'] or
            mtp.get('head_variant') != assets.MTP_SOURCE_VARIANT or
            type(mtp.get('drafted_tokens')) is not int or mtp['drafted_tokens'] < 1 or
            type(mtp.get('accepted_tokens')) is not int or
            not 1 <= mtp['accepted_tokens'] <= mtp['drafted_tokens']):
        return 'SC117 IQ3_S report needs measurable MTP=4 acceptance with the pinned stock/non-abliterated head'
    telemetry = report.get('resource_telemetry')
    if (not isinstance(telemetry, dict) or telemetry.get('passed') is not True or
            telemetry.get('source') != 'strata_0.1.39_GET_/metrics' or
            not isinstance(telemetry.get('gpus'), list) or len(telemetry['gpus']) != 4 or
            type(telemetry.get('sample_age_milliseconds')) is not int or
            not 0 <= telemetry['sample_age_milliseconds'] <= 5000):
        return 'SC117 IQ3_S report lacks fresh four-GPU .39 resource telemetry'
    live = report.get('live_identity')
    artifact = report.get('artifact_identity')
    observations = report.get('measured_observations')
    if (not isinstance(live, dict) or set(live) != {'before_sha256', 'after_sha256'} or
            not isinstance(artifact, dict) or set(artifact) != {
                'before_sha256', 'after_sha256', 'fingerprint_inputs_sha256', 'root_pin_policy_sha256'} or
            not isinstance(observations, dict) or set(observations) != {
                'identity_binding', 'probe_receipts', 'baseline_request_count'} or
            artifact.get('root_pin_policy_sha256') != pin_policy_sha256 or
            type(observations.get('baseline_request_count')) is not int or
            observations['baseline_request_count'] < 0):
        return 'SC117 IQ3_S report lacks identity-bound measured probe observations'
    identity_binding = observations['identity_binding']
    binding_fields = {'live_sha256', 'live_material', 'artifact_sha256',
                      'machine_identity', 'profile_fingerprint'}
    if not isinstance(identity_binding, dict) or set(identity_binding) != {'before', 'after'}:
        return 'SC117 IQ3_S report identity snapshots are malformed'
    before, after = identity_binding['before'], identity_binding['after']
    if (not isinstance(before, dict) or not isinstance(after, dict) or
            set(before) != binding_fields or set(after) != binding_fields or before != after):
        return 'SC117 IQ3_S live/artifact identity changed across captured probes'
    for snapshot in (before, after):
        if (snapshot.get('machine_identity') != current_machine or
                snapshot.get('profile_fingerprint') != computed_fingerprint or
                snapshot.get('artifact_sha256') != _canonical_report_sha256(report['profile_fingerprint_inputs']) or
                snapshot.get('live_sha256') != _canonical_report_sha256(snapshot.get('live_material')) or
                not isinstance(snapshot.get('live_material'), dict) or
                snapshot['live_material'].get('pin_policy_sha256') != pin_policy_sha256):
            return 'SC117 IQ3_S report identity snapshot does not recompute against current pins'
    if (live.get('before_sha256') != before['live_sha256'] or
            live.get('after_sha256') != after['live_sha256'] or
            artifact.get('before_sha256') != before['artifact_sha256'] or
            artifact.get('after_sha256') != after['artifact_sha256'] or
            artifact.get('fingerprint_inputs_sha256') != before['artifact_sha256']):
        return 'SC117 IQ3_S report identity summary does not match its snapshots'
    receipts = observations['probe_receipts']
    if (not isinstance(receipts, list) or len(receipts) != len(REPORT_PROBE_ORDER) or
            tuple(item.get('probe') for item in receipts if isinstance(item, dict)) != REPORT_PROBE_ORDER):
        return 'SC117 IQ3_S report lacks the exact runner-captured synthetic probe sequence'
    before_hash, after_hash = (_canonical_report_sha256(before), _canonical_report_sha256(after))
    previous_counter = observations['baseline_request_count']
    previous_finish = 0.0
    by_probe = {}
    seen_probe_ids, seen_response_ids = set(), set()
    expected_log_identity = None
    receipt_fields = {'probe', 'probe_id', 'response_id', 'started_at', 'finished_at',
                      'request_counter_before', 'request_counter_after', 'prompt_tokens',
                      'completion_tokens', 'finish', 'log_cursor_before', 'log_cursor_after',
                      'cache_events', 'identity_before_sha256', 'identity_after_sha256'}
    for item in receipts:
        if not isinstance(item, dict) or set(item) != receipt_fields:
            return 'SC117 IQ3_S probe receipt contains missing/fabricated fields'
        if (not isinstance(item.get('probe_id'), str) or
                not re.fullmatch(r'[a-f0-9]{24}', item['probe_id']) or item['probe_id'] in seen_probe_ids or
                type(item.get('request_counter_before')) is not int or
                type(item.get('request_counter_after')) is not int or
                item['request_counter_before'] != previous_counter or
                type(item.get('prompt_tokens')) is not int or item['prompt_tokens'] < 1 or
                type(item.get('completion_tokens')) is not int or item['completion_tokens'] < 0 or
                type(item.get('started_at')) not in (int, float) or
                type(item.get('finished_at')) not in (int, float) or
                not math.isfinite(item['started_at']) or not math.isfinite(item['finished_at']) or
                item['started_at'] < previous_finish or item['finished_at'] < item['started_at'] or
                item.get('identity_before_sha256') != before_hash or
                item.get('identity_after_sha256') != after_hash):
            return 'SC117 IQ3_S probe receipt is not bound to ordered counters/time/identity'
        cursor_before, cursor_after = item.get('log_cursor_before'), item.get('log_cursor_after')
        for cursor in (cursor_before, cursor_after):
            if (not isinstance(cursor, dict) or set(cursor) != {'path', 'identity', 'offset'} or
                    not isinstance(cursor.get('path'), str) or not Path(cursor['path']).is_absolute() or
                    not isinstance(cursor.get('identity'), (list, tuple)) or len(cursor['identity']) != 6 or
                    any(type(value) is not int for value in cursor['identity']) or
                    type(cursor.get('offset')) is not int or cursor['offset'] < 0):
                return 'SC117 IQ3_S probe receipt log cursor is malformed'
        cursor_identity = (cursor_before['path'], tuple(cursor_before['identity']))
        if expected_log_identity is None:
            expected_log_identity = cursor_identity
        if (cursor_before['path'] != cursor_after['path'] or
                tuple(cursor_before['identity']) != tuple(cursor_after['identity']) or
                cursor_identity != expected_log_identity or
                cursor_after['offset'] < cursor_before['offset'] or
                not isinstance(item.get('cache_events'), list)):
            return 'SC117 IQ3_S probe receipt log window is not stable/ordered'
        for event in item['cache_events']:
            if (not isinstance(event, dict) or set(event) != {
                    'event', 'tokens', 'parked', 'bytes', 'evictions'} or
                    event.get('event') not in ('parked', 'restored') or
                    any(type(event.get(key)) is not int or event[key] < 0
                        for key in ('tokens', 'parked', 'bytes', 'evictions'))):
                return 'SC117 IQ3_S probe receipt has malformed native cache events'
        count_probe = item['probe'].startswith('count_')
        if (count_probe and (item.get('response_id') is not None or item.get('finish') != 'counted' or
                             item['completion_tokens'] != 0 or
                             item['request_counter_after'] != item['request_counter_before']) or
                not count_probe and (not isinstance(item.get('response_id'), str) or
                                     not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', item['response_id']) or
                                     item.get('finish') != 'stop' or item['completion_tokens'] < 1 or
                                     item['request_counter_after'] != item['request_counter_before'] + 1)):
            return 'SC117 IQ3_S probe receipt is not correlated to a successful API response'
        seen_probe_ids.add(item['probe_id'])
        if not count_probe:
            if item['response_id'] in seen_response_ids:
                return 'SC117 IQ3_S report repeats a server response ID across probes'
            seen_response_ids.add(item['response_id'])
        by_probe[item['probe']] = item
        previous_counter = item['request_counter_after']
        previous_finish = item['finished_at']
    if not isinstance(report['smoke'], dict) or report['smoke'].get('passed') is not True:
        return 'SC117 IQ3_S report lacks a passing runner-captured smoke proof'
    for probe, proof in (('smoke', report['smoke']), ('native_context', context),
                         ('vision', vision), ('mtp', mtp)):
        if not isinstance(proof, dict) or proof.get('response_id') != by_probe[probe]['response_id']:
            return 'SC117 IQ3_S measured proof is not bound to its captured response ID'
    if (by_probe['count_context']['prompt_tokens'] != context['prompt_tokens'] or
            by_probe['native_context']['prompt_tokens'] != context['prompt_tokens'] or
            by_probe['native_context']['completion_tokens'] != context['completion_tokens'] or
            by_probe['vision']['prompt_tokens'] != vision.get('prompt_tokens')):
        return 'SC117 IQ3_S context/vision totals do not match captured receipts'
    for index, sample in enumerate(prefill['samples'], 1):
        if (not isinstance(sample, dict) or sample.get('response_id') != by_probe[f'prefill_{index}']['response_id'] or
                sample.get('selected_prefill') != settings['prefill'] or
                type(sample.get('prompt_total')) is not int or
                not isinstance(sample.get('prompt_read_samples'), list) or
                len(sample['prompt_read_samples']) < 2):
            return 'SC117 IQ3_S prefill samples are not bound to their captured requests'
    cache_sequence = cache.get('sequence')
    if not isinstance(cache_sequence, list) or len(cache_sequence) != 4:
        return 'SC117 IQ3_S report lacks the four-turn cache sequence'
    for letter, turn in (('A', 1), ('B', 1), ('A', 2), ('B', 2)):
        sample = next((item for item in cache_sequence
                       if isinstance(item, dict) and item.get('conversation') == letter and
                       item.get('turn') == turn), None)
        receipt = by_probe[f'cache_{letter}{turn}']
        if (sample is None or sample.get('response_id') != receipt['response_id'] or
                sample.get('cache_events') != receipt.get('cache_events') or
                type(sample.get('reused_tokens')) is not int or
                (turn == 2 and sample['reused_tokens'] <= 0)):
            return 'SC117 IQ3_S cache evidence is not bound to its captured native log windows'
    for index in (2, 3):
        current_events = by_probe[f'cache_{("A", "B")[index - 2]}2']['cache_events']
        preceding_events = [event for name in ('cache_A1', 'cache_B1', 'cache_A2', 'cache_B2')[:index]
                            for event in by_probe[name]['cache_events']]
        if (not any(event.get('event') == 'restored' for event in current_events) or
                not any(event.get('event') == 'parked' for event in preceding_events)):
            return 'SC117 IQ3_S cache restore lacks its serialized preceding park event'
    if not isinstance(report.get('recorded_at'), str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', report['recorded_at']):
        return 'SC117 IQ3_S qualification report timestamp is malformed'
    return None


def qualification_status(root, settings):
    try:
        validate_settings(settings)
    except SafetyError as exc:
        return False, str(exc)
    path = qualification_path(root)
    if path.is_symlink() or not path.is_file():
        return False, 'SC117 IQ3_S identity-bound qualification report is missing'
    try:
        _validate_qualification_location(path, require_file=True)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > QUALIFICATION_REPORT_MAX_BYTES:
            return False, 'SC117 IQ3_S qualification report is oversized, linked, or invalid'
        runner_sha256 = trusted_qualification_runner_sha256()
        pin_policy_sha256 = trusted_qualification_pin_policy_sha256()
        _report_digest, report_bytes, report_metadata = _stable_file_read(
            path, 'qualification report', max_bytes=QUALIFICATION_REPORT_MAX_BYTES, include_contents=True)
        if (os.name == 'posix' and
                (report_metadata[1] != 0o444 or report_metadata[5] != 0 or
                 report_metadata[4] != 1)):
            return False, 'SC117 IQ3_S qualification report changed owner/mode/link identity while reading'
        report = json.loads(report_bytes.decode('utf-8'))
        machine = machine_identity()
        expected_fingerprint = fingerprint(root, settings, _machine_identity=machine)
    except (OSError, UnicodeError, json.JSONDecodeError, SafetyError, ValueError, TypeError) as exc:
        return False, f'SC117 IQ3_S qualification provenance/report is unavailable: {exc}'
    message = _validate_report(report, settings, expected_fingerprint, runner_sha256,
                               machine=machine, pin_policy_sha256=pin_policy_sha256)
    return (False, message) if message else (True, 'qualified')


def require_qualification(root, settings):
    passed, message = qualification_status(root, settings)
    if not passed:
        raise SafetyError(message)


def allow_trial_start(root, *, host, port, state_root):
    """Permit the explicit unauthenticated trusted-LAN trial with dedicated isolated state."""
    if (os.environ.get(TRIAL_ENV) != '1' or host != TRIAL_BIND_HOST or
            type(port) is not int or port != TRIAL_PORT):
        return False
    expected = Path.home() / '.local/state/locallm-strata' / TRIAL_STATE_ROOT_NAME
    candidate = Path(state_root)
    if candidate != expected:
        return False
    # Do not let a symlinked home-state component redirect trial state into production state.
    current = Path.home()
    for part in ('.local', 'state', 'locallm-strata', TRIAL_STATE_ROOT_NAME):
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            return False
    return True


def require_trial_binding(host):
    """SC117's unqualified trial is deliberately restricted to its trusted-LAN bind."""
    if host != TRIAL_BIND_HOST:
        raise SafetyError(f'SC117 trial binding is fixed to trusted-LAN {TRIAL_BIND_HOST}; other hosts are not permitted')


def _verify_trial_environment(state_root):
    cache_prefix = Path(state_root) / 'pycache'
    if os.environ.get('PYTHONPYCACHEPREFIX') != str(cache_prefix):
        raise SafetyError('SC117 trial requires its isolated Python bytecode-cache prefix')
    if os.environ.get('SETUPTOOLS_USE_DISTUTILS') != 'stdlib':
        raise SafetyError('SC117 trial requires the pinned distutils .pth to remain inert')


def _verify_trial_artifacts(root, settings, state_root):
    """Verify pinned .39 inputs while ensuring the trial interpreter ignores shared bytecode caches."""
    _verify_trial_environment(state_root)
    fingerprint(root, settings, verify_content=True, allow_inactive_bytecode=True)


def _verify_v01441_trial_artifacts(root, state_root):
    """Verify SC117 data plus the independently pinned .44.1 candidate for this trial only."""
    _verify_trial_environment(state_root)
    prepare_sc117.verify_readiness(Path(root), full_hash=True)
    from orca_q4ks_v01441_profile import ready_q4ks_v01441
    ready_q4ks_v01441(Path(root))


def build_v01441_trial_config(root, candidate_config):
    """Rewire the exact Q4 .44.1 serving preset to SC117 assets without changing its tuning."""
    import orca_q4ks_v01441_assets as q4_assets
    root = Path(root)
    cfg = copy.deepcopy(candidate_config)
    args = cfg.get('args')
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise SafetyError('SC117 .44.1 candidate config has malformed arguments')
    if ('--pipeline-windows' in args or
            any(item.startswith('--pipeline-windows=') for item in args)):
        raise SafetyError('SC117 .44.1 baseline must omit the unconfirmed pipeline-windows override')
    for flag, expected in (('--prefill', 'auto'), ('--spec', '4'), ('--max-context', '262144'),
                           ('--kv', 'int8'), ('--kv-resident', '32768'),
                           ('--conversation-cache-mib', '8192'),
                           ('--conversation-cache-slots', '4')):
        _flag_value(args, flag, expected=expected)
    if args.count('--trim-stage-weights') != 1 or cfg.get('layer_split') != '14,24,36':
        raise SafetyError('SC117 .44.1 baseline must retain split 14,24,36 and --trim-stage-weights')
    if cfg.get('gpu') != [0, 1, 2, 3] or cfg.get('gpu_order') != 'as_given':
        raise SafetyError('SC117 .44.1 baseline must retain the Q4 GPU order')
    if cfg.get('parallel') is not None or cfg.get('env') != {
            'STRATA_PARALLEL': '1', 'STRATA_BATCH_GROUPS': '1'}:
        raise SafetyError('SC117 .44.1 baseline must remain single-slot/single-group')
    vision = cfg.get('vision')
    if (not isinstance(vision, dict) or vision.get('exe') != str(q4_assets.vision_path(root)) or
            vision.get('gpu') is not False or vision.get('max_tokens') != 1024):
        raise SafetyError('SC117 .44.1 baseline requires the pinned CPU vision helper')
    rewires = {
        '--native': assets.shard1_path(root),
        '--ple-gguf': assets.shard2_path(root),
        '--pack': assets.pack_root(root),
        '--mtp': assets.mtp_runtime_root(root),
    }
    for flag, destination in rewires.items():
        if args.count(flag) != 1:
            raise SafetyError(f'SC117 .44.1 candidate must contain exactly one {flag}')
        args[args.index(flag) + 1] = str(destination)
    settings = settings_for_start(allow_trial_env=True)
    if 'STRATA_SC117_IQ3S_TRIAL_PREFILL' in os.environ:
        _flag_value(args, '--prefill')
        args[args.index('--prefill') + 1] = settings['prefill']
    if 'STRATA_SC117_IQ3S_TRIAL_SPLIT' in os.environ:
        cfg['layer_split'] = settings['layer_split']
    if ('STRATA_SC117_IQ3S_TRIAL_CACHE_MIB' in os.environ or
            'STRATA_SC117_IQ3S_TRIAL_CACHE_SLOTS' in os.environ):
        cache_mib = settings['conversation_cache_mib']
        cache_slots = settings['conversation_cache_slots']
        for flag in ('--conversation-cache-mib', '--conversation-cache-slots'):
            if args.count(flag) != 1:
                raise SafetyError(f'SC117 candidate must contain exactly one {flag}')
        cache_index = args.index('--conversation-cache-mib')
        slots_index = args.index('--conversation-cache-slots')
        if cache_mib == 0:
            for index in sorted((cache_index, slots_index), reverse=True):
                del args[index:index + 2]
        else:
            args[cache_index + 1] = str(cache_mib)
            args[slots_index + 1] = str(cache_slots)
    cfg['tokenizer'] = str(assets.pack_root(root) / 'tokenizer')
    cfg['model_name'] = MODEL_ID
    cfg['aliases'] = ['qwen38-iq3s-sc117-abliterated', PROFILE]
    vision['model'] = str(assets.shard1_path(root))
    vision['mmproj'] = str(assets.projector_path(root))
    _apply_v01441_trial_tunables(cfg)
    return cfg


def _apply_v01441_trial_tunables(cfg):
    supplied = TRIAL_EXPERIMENT_ENV.intersection(os.environ)
    if supplied and os.environ.get(TRIAL_ENV) != '1':
        raise SafetyError('SC117 .44.1 tuning overrides require an explicit isolated trial')
    args = cfg['args']
    spec = os.environ.get('STRATA_SC117_IQ3S_TRIAL_SPEC', TRIAL_DEFAULT_SPEC)
    if not re.fullmatch(r'[2-6]', spec) or args.count('--spec') != 1:
        raise SafetyError('SC117 trial --spec must be one integer from 2 through 6')
    args[args.index('--spec') + 1] = spec
    pipeline = os.environ.get('STRATA_SC117_IQ3S_TRIAL_PIPELINE_WINDOWS', TRIAL_DEFAULT_PIPELINE_WINDOWS)
    if pipeline not in ('0', '2') or '--pipeline-windows' in args or any(
            item.startswith('--pipeline-windows=') for item in args):
        raise SafetyError('SC117 trial --pipeline-windows must be 0 or 2; this engine supports at most 2')
    args.extend(['--pipeline-windows', pipeline])
    pcie = os.environ.get('STRATA_SC117_IQ3S_TRIAL_PCIE_FRAC')
    if pcie is not None:
        if (not re.fullmatch(r'(?:0(?:[.][0-9]+)?|1(?:[.]0+)?)', pcie) or
                not math.isfinite(float(pcie)) or not 0 <= float(pcie) <= 1 or
                '--pcie-frac' in args or any(item.startswith('--pcie-frac=') for item in args)):
            raise SafetyError('SC117 trial --pcie-frac must be a unique value in [0, 1]')
        args.extend(['--pcie-frac', pcie])
    workers = os.environ.get('STRATA_SC117_IQ3S_TRIAL_POOL_WORKERS')
    if workers is not None:
        if (not re.fullmatch(r'[0-9]+', workers) or int(workers) > 128 or
                '--pool-workers' in args or any(item.startswith('--pool-workers=') for item in args)):
            raise SafetyError('SC117 trial --pool-workers must be a unique integer from 0 through 128')
        args.extend(['--pool-workers', workers])
    trim = os.environ.get('STRATA_SC117_IQ3S_TRIAL_TRIM_STAGE_WEIGHTS')
    if trim is not None:
        if trim not in ('0', '1'):
            raise SafetyError('SC117 trial trim-stage-weights override must be 0 or 1')
        if trim == '0':
            args.remove('--trim-stage-weights')
    gpu_order = os.environ.get('STRATA_SC117_IQ3S_TRIAL_GPU_ORDER')
    if gpu_order is not None:
        if not re.fullmatch(r'[0-3](?:,[0-3]){3}', gpu_order):
            raise SafetyError('SC117 trial GPU order must be a permutation of 0,1,2,3')
        order = [int(item) for item in gpu_order.split(',')]
        if sorted(order) != [0, 1, 2, 3]:
            raise SafetyError('SC117 trial GPU order must be a permutation of 0,1,2,3')
        cfg['gpu'] = order
        cfg['gpu_order'] = 'as_given'


def authorize_start(root, settings, *, host, port, state_root):
    """Permit only the isolated unqualified trial; normal/qualified starts are retired."""
    if os.environ.get(TRIAL_ENV) != '1':
        raise SafetyError('SC117 remains unqualified; normal/qualified hosting is disabled, and only the isolated .44.1 trial is allowed')
    trial_settings = settings_for_start(allow_trial_env=True)
    validate_settings(settings)
    if (settings != trial_settings or os.environ.get('STRATA_RUNTIME') != TRIAL_RUNTIME_VERSION or
            not allow_trial_start(root, host=host, port=port, state_root=state_root)):
        raise SafetyError('Unqualified SC117 trials require Strata .44.1, trusted-LAN 0.0.0.0:8080, and isolated sc117-iq3s-trial state')
    # Trials are explicitly not qualification evidence. The isolated cache prefix makes these
    # existing __pycache__ entries inactive; no normal/qualified start path is enabled.
    _verify_v01441_trial_artifacts(root, state_root)
    return 'trial'
