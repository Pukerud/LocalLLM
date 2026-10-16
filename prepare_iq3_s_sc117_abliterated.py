#!/usr/bin/env python3
"""Safely stage pinned assets; build only the explicitly requested native pack, never inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import types
import urllib.parse
import urllib.request

import iq3_s_sc117_abliterated_assets as assets
try:
    import strata_runtime
except ModuleNotFoundError as exc:  # engine_safety imports pwd, which is unavailable on native Windows.
    if exc.name != 'pwd':
        raise
    shim = types.ModuleType('engine_safety')

    class _RuntimeSafetyError(RuntimeError):
        pass

    def _runtime_command_output(*_args, **_kwargs):
        raise RuntimeError('Strata runtime commands are unavailable in the SC117 Windows pin loader')

    shim.SafetyError = _RuntimeSafetyError
    shim.command_output = _runtime_command_output
    runtime_spec = importlib.util.spec_from_file_location(
        '_sc117_strata_runtime', Path(__file__).with_name('strata_runtime.py'))
    strata_runtime = importlib.util.module_from_spec(runtime_spec)
    previous_safety = sys.modules.get('engine_safety')
    sys.modules['engine_safety'] = shim
    try:
        runtime_spec.loader.exec_module(strata_runtime)
    finally:
        if previous_safety is None:
            sys.modules.pop('engine_safety', None)
        else:
            sys.modules['engine_safety'] = previous_safety

try:
    from engine_safety import SafetyError
except ImportError:  # Keep this standalone identity/preparation module importable on Windows too.
    class SafetyError(RuntimeError):
        pass

USER_AGENT = 'LocalLLM-SC117-asset-preparation/1'
CHUNK_BYTES = 8 * 1024 * 1024
MANIFEST_SCHEMA = 2
PACK_MANIFEST_SCHEMA = 2
PACKER_VERSION = '0.1.39'
PACK_REQUIRED_FILES = (
    'conversions.json',
    'dense.bin',
    'index.txt',
    'native_experts.txt',
    'tokenizer/chat_template.jinja',
    'tokenizer/merges.txt',
    'tokenizer/token_type.json',
    'tokenizer/tokenizer.json',
    'tokenizer/vocab.json',
)
MAX_PACK_REPORT_BYTES = 128 * 1024 * 1024


def _root_path(root: Path) -> Path:
    path = Path(root)
    if not path.is_absolute():
        raise SafetyError('SC117 data root must be an absolute path')
    try:
        if path.is_symlink():
            raise SafetyError('SC117 data root must not be a symlink')
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise SafetyError(f'Cannot resolve SC117 data root: {exc}') from exc
    if resolved == Path(resolved.anchor):
        raise SafetyError('SC117 data root cannot be a filesystem root')
    return resolved


def _check_beneath(root: Path, path: Path) -> Path:
    """Reject lexical escapes, symlink components, and non-directory parents."""
    base = _root_path(root)
    candidate = Path(os.path.abspath(Path(path)))
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        raise SafetyError(f'SC117 path resolves outside its data root: {candidate}') from None
    try:
        if not candidate.resolve(strict=False).is_relative_to(base):
            raise SafetyError(f'SC117 path resolves outside its data root: {candidate}')
    except (OSError, RuntimeError) as exc:
        raise SafetyError(f'Cannot resolve SC117 path {candidate}: {exc}') from exc

    current = base
    components = tuple(relative.parts)
    for index, component in enumerate(components):
        current = current / component
        try:
            item = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SafetyError(f'Cannot inspect SC117 path {current}: {exc}') from exc
        if stat.S_ISLNK(item.st_mode):
            raise SafetyError(f'SC117 paths may not contain symlinks: {current}')
        if index < len(components) - 1 and not stat.S_ISDIR(item.st_mode):
            raise SafetyError(f'SC117 path parent is not a directory: {current}')
    return candidate


def _ensure_directory(root: Path, directory: Path) -> Path:
    base = _root_path(root)
    target = _check_beneath(base, directory)
    relative = target.relative_to(base)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SafetyError(f'Cannot create SC117 data root: {exc}') from exc
    if base.is_symlink() or not base.is_dir():
        raise SafetyError('SC117 data root changed or is not a directory')
    current = base
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise SafetyError(f'Cannot create SC117 directory {current}: {exc}') from exc
        try:
            item = current.lstat()
        except OSError as exc:
            raise SafetyError(f'Cannot inspect SC117 directory {current}: {exc}') from exc
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise SafetyError(f'SC117 path is not a real directory: {current}')
    _check_beneath(base, target)
    return target


def require_new_pack_path(root: Path) -> Path:
    """The candidate must use a distinct, not-yet-created native pack directory."""
    base = _root_path(root)
    path = _check_beneath(base, assets.pack_root(base))
    if os.path.lexists(path):
        raise SafetyError(f'SC117 native pack path must be new; existing path will not be reused/overwritten: {path}')
    return path


def select_assets(repo_paths=None):
    """Resolve caller selections only through the fixed publisher-file allowlist."""
    if repo_paths is None:
        return assets.ASSETS
    if isinstance(repo_paths, (str, bytes)):
        raise SafetyError('SC117 file selection must be a sequence of exact allowlisted repository paths')
    try:
        requested = list(repo_paths)
    except TypeError:
        raise SafetyError('SC117 file selection must be a sequence of exact allowlisted repository paths') from None
    if not requested:
        raise SafetyError('SC117 preparation requires at least one explicitly allowlisted file')
    if any(not isinstance(name, str) for name in requested):
        raise SafetyError('SC117 file selection entries must be exact repository-path strings')
    if len(set(requested)) != len(requested):
        raise SafetyError('SC117 file selection contains duplicates')
    unknown = [name for name in requested if name not in assets.ASSETS_BY_REPO_PATH]
    if unknown:
        raise SafetyError(f'SC117 file is not in the pinned download allowlist: {unknown[0]!r}')
    return tuple(assets.ASSETS_BY_REPO_PATH[name] for name in requested)


def _asset_file_state(root: Path, item) -> dict:
    base = _root_path(root)
    target = _check_beneath(base, assets.asset_path(base, item))
    try:
        info = target.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 pinned asset is missing: {target}') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 asset {target}: {exc}') from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size != item.size:
        raise SafetyError(f'SC117 pinned asset is not a regular file of the expected size: {target}')
    return {'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns,
            'mode': stat.S_IMODE(info.st_mode)}


def expected_manifest(root: Path) -> dict:
    base = _root_path(root)
    asset_records = []
    for item in assets.ASSETS:
        record = {
            'repo_path': item.repo_path,
            'role': item.role,
            'path': str(assets.asset_path(base, item)),
            'bytes': item.size,
        }
        if item.sha256 is not None:
            record['sha256'] = item.sha256
        if item.git_blob_oid is not None:
            record['git_blob_oid'] = item.git_blob_oid
        asset_records.append(record)
    return {
        'schema_version': MANIFEST_SCHEMA,
        'profile': assets.PROFILE,
        'model_id': assets.MODEL_ID,
        'repository': assets.REPOSITORY,
        'revision': assets.REVISION,
        'mtp_source_variant': assets.MTP_SOURCE_VARIANT,
        'mtp_source_note': assets.MTP_SOURCE_NOTE,
        'pack_path': str(assets.pack_root(base)),
        'pack_manifest_path': str(assets.pack_manifest_path(base)),
        'pack_state': 'tracked_separately',
        'assets': asset_records,
    }


def _manifest_with_file_states(root: Path) -> dict:
    base = _root_path(root)
    value = expected_manifest(base)
    for record, item in zip(value['assets'], assets.ASSETS):
        record['file_state'] = _asset_file_state(base, item)
    return value


def validate_manifest(value, root: Path) -> dict:
    """Require pinned provenance and unchanged size/mtime/mode fingerprints."""
    if not isinstance(value, dict):
        raise SafetyError('Malformed SC117 preparation manifest: expected a JSON object')
    expected = expected_manifest(root)
    if set(value) != set(expected):
        raise SafetyError('Malformed SC117 preparation manifest: missing or unexpected top-level fields')
    if type(value.get('schema_version')) is not int or value['schema_version'] != MANIFEST_SCHEMA:
        raise SafetyError('Malformed SC117 preparation manifest: unsupported schema version')
    records = value.get('assets')
    if not isinstance(records, list) or len(records) != len(assets.ASSETS):
        raise SafetyError('Malformed SC117 preparation manifest: pinned asset list is incomplete')
    for record, pinned, item in zip(records, expected['assets'], assets.ASSETS):
        if not isinstance(record, dict) or set(record) != set(pinned) | {'file_state'}:
            raise SafetyError('Malformed SC117 preparation manifest: asset record fields are invalid')
        if {key: record[key] for key in pinned} != pinned:
            raise SafetyError('SC117 preparation manifest does not match pinned repository/assets/roles/paths')
        state = record.get('file_state')
        if (not isinstance(state, dict) or set(state) != {'mtime_ns', 'ctime_ns', 'mode'} or
                type(state.get('mtime_ns')) is not int or type(state.get('ctime_ns')) is not int or
                type(state.get('mode')) is not int):
            raise SafetyError('Malformed SC117 preparation manifest: asset file state is invalid')
        if state != _asset_file_state(root, item):
            raise SafetyError(f'SC117 asset size/mtime/mode changed since preparation: {item.repo_path}')
    for key in expected.keys() - {'assets'}:
        if value[key] != expected[key]:
            raise SafetyError('SC117 preparation manifest does not match pinned repository/assets/roles/paths')
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def git_blob_oid(data: bytes) -> str:
    """Return the SHA-1 object id Git assigns to a blob containing ``data``."""
    digest = hashlib.sha1()
    digest.update(b'blob ' + str(len(data)).encode('ascii') + b'\0')
    digest.update(data)
    return digest.hexdigest()


def _expected_digest(path: Path, item) -> str:
    if item.sha256 is not None:
        return sha256_file(path)
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(b'blob ' + str(size).encode('ascii') + b'\0')
    with Path(path).open('rb') as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def verify_asset_file(root: Path, item, path: Path = None, *, full_hash=True) -> Path:
    """Verify a pinned regular file; ordinary readiness uses metadata, explicit reverify hashes bytes."""
    if item.repo_path not in assets.ASSETS_BY_REPO_PATH or assets.ASSETS_BY_REPO_PATH[item.repo_path] != item:
        raise SafetyError('Cannot verify an asset outside the immutable SC117 allowlist')
    base = _root_path(root)
    target = _check_beneath(base, Path(path) if path is not None else assets.asset_path(base, item))
    try:
        before = target.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 pinned asset is missing: {target}') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 asset {target}: {exc}') from exc
    if not stat.S_ISREG(before.st_mode):
        raise SafetyError(f'SC117 asset must be a regular non-symlink file: {target}')
    if before.st_size != item.size:
        raise SafetyError(f'SC117 asset has wrong size ({before.st_size}/{item.size}): {target}')
    actual = _expected_digest(target, item) if full_hash else None
    try:
        after = target.lstat()
    except OSError as exc:
        raise SafetyError(f'SC117 asset changed while being verified: {target}') from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                       before.st_ctime_ns, stat.S_IMODE(before.st_mode))
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                      after.st_ctime_ns, stat.S_IMODE(after.st_mode))
    if identity_before != identity_after:
        raise SafetyError(f'SC117 asset changed while being verified: {target}')
    expected = item.sha256 if item.sha256 is not None else item.git_blob_oid
    if full_hash and actual != expected:
        raise SafetyError(f'SC117 asset checksum/blob-id mismatch; not accepted: {target}')
    return target


def verify_asset_readiness(root: Path, *, full_hash=False) -> dict:
    """Validate provenance and metadata cheaply; hash large sources only for explicit reverify."""
    base = _root_path(root)
    path = _check_beneath(base, assets.manifest_path(base))
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 assets are not prepared; manifest is missing: {path}') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 manifest: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024 or
            stat.S_IMODE(info.st_mode) != 0o444):
        raise SafetyError('SC117 preparation manifest must be a small immutable 0444 regular file')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Malformed SC117 preparation manifest: {exc}') from exc
    validate_manifest(value, base)
    _reject_partial_files(base)
    for item in assets.ASSETS:
        verify_asset_file(base, item, full_hash=full_hash)
    return value


def _reject_partial_files(root: Path):
    base = _root_path(root)
    for item in assets.ASSETS:
        destination = assets.asset_path(base, item)
        temporary = _check_beneath(base, _partial_path(destination))
        if os.path.lexists(temporary):
            raise SafetyError(f'Interrupted SC117 partial exists; inspect/remove explicitly: {temporary}')


def readiness(root: Path, *, full_hash=False):
    try:
        verify_readiness(root, full_hash=full_hash)
    except SafetyError as exc:
        return False, str(exc)
    return True, 'all pinned SC117 source assets and native pack are ready'


def reuse_source_for(root: Path, item, *, enabled=False):
    """Opt-in reuse is permitted for the publisher-identical old IQ3_S shard 2 only."""
    if not enabled:
        return None
    if (item.role != 'model_shard_2' or item.repo_path !=
            'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf'):
        raise SafetyError('Only the exact SC117 shard 2 may use the verified old-shard reuse path')
    return assets.shard2_reuse_source(_root_path(root))


def _partial_path(path: Path) -> Path:
    return path.with_name(f'.{path.name}.{assets.PROFILE}.part')


def _check_regular_or_missing(root: Path, path: Path, *, label: str):
    target = _check_beneath(root, path)
    try:
        info = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect {label} {target}: {exc}') from exc
    if not stat.S_ISREG(info.st_mode):
        raise SafetyError(f'{label} must be a regular non-symlink file: {target}')
    return info


def _copy_reused_shard(root: Path, item, source: Path, temporary: Path):
    source = _check_beneath(root, source)
    info = _check_regular_or_missing(root, source, label='shard-2 reuse source')
    if info is None:
        raise SafetyError(f'Opt-in SC117 shard-2 reuse source is missing: {source}')
    if info.st_size != item.size:
        raise SafetyError(f'Old shard 2 does not match the exact SC117 size pin: {source}')
    before_id = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise SafetyError(f'Cannot open old shard-2 reuse source: {exc}') from exc
    try:
        opened = os.fstat(source_fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != before_id:
            raise SafetyError('Old shard 2 changed before verified reuse; source is untouched')
        out_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, 'O_NOFOLLOW', 0), 0o600)
        count = 0
        digest = hashlib.sha256()
        with os.fdopen(source_fd, 'rb', closefd=False) as source_stream, os.fdopen(out_fd, 'wb') as output:
            while block := source_stream.read(CHUNK_BYTES):
                count += len(block)
                if count > item.size:
                    raise SafetyError('Old shard 2 exceeds the pinned size; reuse refused')
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        source_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if count != item.size or digest.hexdigest() != item.sha256 or source_after != before_id:
            raise SafetyError('Old shard 2 failed exact size/hash verification; reuse refused')
    finally:
        os.close(source_fd)
    os.chmod(temporary, 0o444)
    _fsync_directory(temporary.parent)


def _fsync_directory(directory: Path):
    if os.name == 'nt':
        return
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _download_to_temporary(item, temporary: Path):
    url = (f'https://huggingface.co/{assets.REPOSITORY}/resolve/{assets.REVISION}/' +
           urllib.parse.quote(item.repo_path, safe='/'))
    request = urllib.request.Request(url, headers={
        'User-Agent': USER_AGENT,
        'Accept-Encoding': 'identity',
    })
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, 'O_NOFOLLOW', 0), 0o600)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(request, timeout=60) as response:
            if getattr(response, 'status', 200) != 200:
                raise SafetyError(f'Unexpected HTTP response for pinned SC117 file: {response.status}')
            while block := response.read(CHUNK_BYTES):
                count += len(block)
                if count > item.size:
                    raise SafetyError('SC117 download exceeded its pinned size; partial retained')
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # Keep any partial as evidence. A subsequent invocation refuses rather than resumes/overwrites it.
        raise
    if count != item.size:
        raise SafetyError(f'Truncated SC117 download; partial retained ({count}/{item.size}): {temporary}')
    if item.sha256 is not None and digest.hexdigest() != item.sha256:
        raise SafetyError(f'SC117 download SHA-256 mismatch; partial retained: {temporary}')
    if item.git_blob_oid is not None:
        # Dense.txt is tiny; hash the Git object prefix while streaming was deliberately avoided so that
        # the actual file length participates in Git's object header.
        if _expected_digest(temporary, item) != item.git_blob_oid:
            raise SafetyError(f'SC117 dense.txt Git blob mismatch; partial retained: {temporary}')
    os.chmod(temporary, 0o444)
    _fsync_directory(temporary.parent)


def _atomic_finalize(root: Path, item, temporary: Path, destination: Path):
    verify_asset_file(root, item, temporary)
    try:
        if os.name == 'nt':
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination, follow_symlinks=False)
            temporary.unlink()
    except FileExistsError as exc:
        raise SafetyError(f'SC117 target appeared during preparation; not overwritten: {destination}') from exc
    except OSError as exc:
        raise SafetyError(f'Cannot atomically finalize SC117 asset {destination}: {exc}') from exc
    _fsync_directory(destination.parent)
    verify_asset_file(root, item, destination)


def _atomic_json_at(root: Path, target: Path, value: dict, *, label: str):
    base = _root_path(root)
    target = _check_beneath(base, target)
    _ensure_directory(base, target.parent)
    if os.path.lexists(target):
        raise SafetyError(f'SC117 {label} already exists; it will not be overwritten: {target}')
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.{time.time_ns()}.tmp')
    content = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            if os.name == 'nt':
                os.rename(temporary, target)
            else:
                os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise SafetyError(f'SC117 {label} appeared during creation; not overwritten: {target}') from exc
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_manifest(root: Path, value: dict):
    _atomic_json_at(root, assets.manifest_path(_root_path(root)), value, label='preparation manifest')


def prepare_assets(root: Path, repo_paths=None, *, reuse_shard2=False) -> dict | None:
    """Fetch selected pinned files only; publish a manifest only after all pins verify.

    Existing verified files are retained. Any changed target, symlink, interrupted partial,
    old manifest, or pre-existing pack path is refused rather than repaired or overwritten.
    """
    base = _root_path(root)
    selected = select_assets(repo_paths)
    shard2 = assets.ASSETS_BY_ROLE['model_shard_2']
    if reuse_shard2 and shard2 not in selected:
        raise SafetyError('--reuse-shard2 requires the exact shard-2 file in the allowlist selection')
    manifest_path = _check_beneath(base, assets.manifest_path(base))
    if os.path.lexists(manifest_path):
        return verify_asset_readiness(base)
    require_new_pack_path(base)

    # Preflight all selected outputs before creating anything: never repair a mismatched file
    # or overwrite an interrupted temporary download.
    missing = []
    for item in selected:
        destination = _check_beneath(base, assets.asset_path(base, item))
        temporary = _check_beneath(base, _partial_path(destination))
        if os.path.lexists(temporary):
            raise SafetyError(f'Interrupted SC117 partial exists; inspect/remove explicitly: {temporary}')
        if os.path.lexists(destination):
            verify_asset_file(base, item, destination)
        else:
            missing.append(item)

    reuse_source = reuse_source_for(base, shard2, enabled=True) if reuse_shard2 else None
    if reuse_source is not None and shard2 in missing:
        _check_beneath(base, reuse_source)
        source_info = _check_regular_or_missing(base, reuse_source, label='shard-2 reuse source')
        if source_info is None or source_info.st_size != shard2.size:
            raise SafetyError('Old shard 2 is missing or has a different size; exact reuse refused')

    for item in missing:
        destination = assets.asset_path(base, item)
        _ensure_directory(base, destination.parent)
        temporary = _partial_path(destination)
        if item is shard2 and reuse_source is not None:
            _copy_reused_shard(base, item, reuse_source, temporary)
        else:
            _download_to_temporary(item, temporary)
        _atomic_finalize(base, item, temporary, destination)

    if all(_asset_present_and_verified(base, item) for item in assets.ASSETS):
        _reject_partial_files(base)
        value = _manifest_with_file_states(base)
        validate_manifest(value, base)
        _atomic_manifest(base, value)
        return verify_asset_readiness(base, full_hash=True)
    return None


def _asset_present_and_verified(root: Path, item) -> bool:
    path = assets.asset_path(root, item)
    if not os.path.lexists(path):
        return False
    verify_asset_file(root, item, path)
    return True


def _git_output(source: Path, *args) -> bytes:
    try:
        result = subprocess.run(['git', '-c', f'safe.directory={source}', '-C', str(source), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise SafetyError(f'Cannot inspect pinned Strata packer source: {exc}') from exc
    if result.returncode != 0:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise SafetyError(f'Cannot inspect pinned Strata packer source: {detail or "git failed"}')
    return result.stdout


def _pinned_packer_identity(root: Path, strata_source: Path = None) -> dict:
    base = _root_path(root)
    expected_source = _check_beneath(base, assets.strata_039_source(base))
    source = expected_source if strata_source is None else Path(strata_source)
    if not source.is_absolute():
        raise SafetyError('Pinned Strata 0.1.39 source path must be absolute')
    source = _check_beneath(base, source)
    if source != expected_source:
        raise SafetyError('SC117 packs accept only data-root/runtimes/0.1.39/source')
    try:
        source_info = source.lstat()
    except OSError as exc:
        raise SafetyError(f'Pinned Strata 0.1.39 source is missing: {source}') from exc
    if not stat.S_ISDIR(source_info.st_mode):
        raise SafetyError('Pinned Strata 0.1.39 source must be a real directory')
    tool = _check_beneath(base, source / 'tools/iq_pack.py')
    try:
        info = tool.lstat()
    except OSError as exc:
        raise SafetyError(f'Pinned Strata tools/iq_pack.py is missing: {tool}') from exc
    if not stat.S_ISREG(info.st_mode):
        raise SafetyError('Pinned Strata tools/iq_pack.py must be a regular non-symlink file')
    try:
        pin = strata_runtime.PINS['0.1.39']
    except (AttributeError, KeyError, TypeError):
        raise SafetyError('Strata runtime does not define the pinned 0.1.39 source commit') from None
    if not isinstance(pin, str) or not re.fullmatch(r'[0-9a-f]{40}', pin):
        raise SafetyError('Strata runtime does not define the pinned 0.1.39 source commit')
    head = _git_output(source, 'rev-parse', 'HEAD').decode('ascii', errors='replace').strip()
    if head != pin:
        raise SafetyError(f'Strata packer source commit is stale or mismatched ({head!r}; expected {pin})')
    dirty = _git_output(source, 'diff', '--name-only', 'HEAD').decode('utf-8', errors='replace').strip()
    if dirty:
        raise SafetyError(f'Strata 0.1.39 packer source checkout has tracked modifications: {dirty}')
    committed_tool = _git_output(source, 'show', f'{pin}:tools/iq_pack.py')
    tool_hash = sha256_file(tool)
    try:
        tool_after = tool.lstat()
    except OSError as exc:
        raise SafetyError(f'Strata tools/iq_pack.py changed while being inspected: {exc}') from exc
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            tool_after.st_dev, tool_after.st_ino, tool_after.st_size,
            tool_after.st_mtime_ns, tool_after.st_ctime_ns):
        raise SafetyError('Strata tools/iq_pack.py changed while being inspected')
    if hashlib.sha256(committed_tool).hexdigest() != tool_hash:
        raise SafetyError('Strata tools/iq_pack.py differs from the pinned 0.1.39 source commit')
    interpreter = source / '.venv/bin/python'
    try:
        interpreter_resolved = interpreter.resolve(strict=True)
        interpreter_before = interpreter_resolved.stat()
        if not stat.S_ISREG(interpreter_before.st_mode):
            raise SafetyError('Pinned Strata packer interpreter is not a regular file')
        interpreter_hash = sha256_file(interpreter_resolved)
        interpreter_after = interpreter_resolved.stat()
    except OSError as exc:
        raise SafetyError(f'Pinned Strata 0.1.39 Python interpreter is unavailable: {exc}') from exc
    if ((interpreter_before.st_dev, interpreter_before.st_ino, interpreter_before.st_size,
         interpreter_before.st_mtime_ns, interpreter_before.st_ctime_ns) !=
            (interpreter_after.st_dev, interpreter_after.st_ino, interpreter_after.st_size,
             interpreter_after.st_mtime_ns, interpreter_after.st_ctime_ns)):
        raise SafetyError('Pinned Strata packer interpreter changed while being inspected')
    return {
        'version': PACKER_VERSION,
        'source_commit': pin,
        'source': str(source),
        'tool_path': str(tool),
        'tool_sha256': tool_hash,
        # Execute through the venv entrypoint so CPython discovers pyvenv.cfg; record and hash the
        # resolved executable separately to bind the actual interpreter binary.
        'python_path': str(interpreter),
        'python_resolved_path': str(interpreter_resolved),
        'python_sha256': interpreter_hash,
    }


def _pack_source_shards(root: Path) -> list:
    base = _root_path(root)
    records = []
    for role in ('model_shard_1', 'model_shard_2'):
        item = assets.ASSETS_BY_ROLE[role]
        target = _check_beneath(base, assets.asset_path(base, item))
        records.append({'repo_path': item.repo_path, 'path': str(target), 'bytes': item.size,
                        'sha256': item.sha256})
    return records


def _json_file(path: Path, *, label: str):
    try:
        info = path.lstat()
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 {label}: {exc}') from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PACK_REPORT_BYTES:
        raise SafetyError(f'SC117 {label} must be a small regular non-symlink file')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Malformed SC117 {label}: {exc}') from exc


def _read_pack_text(pack: Path, relative: str) -> str:
    path = pack / relative
    try:
        info = path.lstat()
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 pack output {relative}: {exc}') from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PACK_REPORT_BYTES:
        raise SafetyError(f'SC117 pack report is not a regular bounded file: {relative}')
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise SafetyError(f'Malformed SC117 pack output {relative}: {exc}') from exc


def _validate_native_experts(pack: Path, source_shards: list):
    text = _read_pack_text(pack, 'native_experts.txt')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not (lines[0].startswith('# strata native experts v3:') or
                         lines[0].startswith('# strata native experts v4:')):
        raise SafetyError('Malformed SC117 pack completion report: native_experts.txt header is invalid')
    match = re.search(r'\(n_expert (\d+), total (\d+);', lines[0])
    if not match:
        raise SafetyError('Malformed SC117 pack completion report: native_experts.txt totals are missing')
    experts, total = map(int, match.groups())
    if experts < 1 or total < 1 or len(lines) < 2:
        raise SafetyError('Malformed SC117 pack completion report: native_experts.txt has no layers')
    allowed_shards = {Path(record['path']).name for record in source_shards}
    expected_offset = 0
    for layer, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) not in (8, 9):
            raise SafetyError('Malformed SC117 pack completion report: native expert row is invalid')
        try:
            values = [int(field, 10) for field in fields[:8]]
        except ValueError:
            raise SafetyError('Malformed SC117 pack completion report: native expert row is not numeric') from None
        row_layer, _, _, offset, blob_bytes, *_ = values
        if row_layer != layer or offset != expected_offset or blob_bytes < 1:
            raise SafetyError('Mismatched SC117 pack completion report: native expert layers/offsets are inconsistent')
        expected_offset += blob_bytes * experts
        if len(fields) == 9:
            names = fields[8].split(',')
            if any(name and name not in allowed_shards for name in names):
                raise SafetyError('Mismatched SC117 pack completion report: native experts name an unknown '
                                  'source shard')
    if expected_offset != total:
        raise SafetyError('Mismatched SC117 pack completion report: native expert total does not match its rows')


def _validate_pack_outputs(pack: Path, source_shards: list):
    native = pack / 'native_experts.txt'
    if not native.is_file():
        raise SafetyError('SC117 native pack is incomplete: native_experts.txt completion marker is missing')
    _validate_native_experts(pack, source_shards)

    index = _read_pack_text(pack, 'index.txt')
    lines = [line.strip() for line in index.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[0].startswith('# strata pack index v3 -- generated by tools/iq_pack.py'):
        raise SafetyError('Malformed SC117 pack report: index.txt header is invalid')
    if f'from {Path(source_shards[0]["path"]).name}' not in lines[0]:
        raise SafetyError('Mismatched SC117 pack report: index.txt names another source shard')
    header = re.fullmatch(r'# align 64 pool (\d+) tensors (\d+)', lines[1])
    if not header:
        raise SafetyError('Malformed SC117 pack report: index.txt pool/tensor header is invalid')
    pool_bytes, tensor_count = map(int, header.groups())
    names = set()
    rows = lines[2:]
    if not rows or len(rows) != tensor_count:
        raise SafetyError('Malformed SC117 pack report: index tensor count does not match its rows')
    for row in rows:
        fields = row.split()
        if len(fields) != 19 or fields[0] in names:
            raise SafetyError('Malformed SC117 pack report: index tensor row is invalid or duplicated')
        try:
            [int(field, 10) for field in fields[1:]]
        except ValueError:
            raise SafetyError('Malformed SC117 pack report: index tensor row contains nonnumeric fields') from None
        names.add(fields[0])
    if (pack / 'dense.bin').stat().st_size != pool_bytes:
        raise SafetyError('Mismatched SC117 pack report: dense.bin size differs from index pool size')

    conversions = _json_file(pack / 'conversions.json', label='conversions.json pack report')
    expected_input = [{'name': Path(record['path']).name, 'size': record['bytes']}
                      for record in source_shards]
    if (not isinstance(conversions, dict) or type(conversions.get('schema')) is not int or
            conversions.get('schema') != 1 or conversions.get('tool') != 'tools/iq_pack.py' or
            conversions.get('source_shards') != expected_input or
            not isinstance(conversions.get('tensors'), list)):
        raise SafetyError('Malformed or mismatched SC117 conversions.json pack report')

    for relative in ('tokenizer/tokenizer.json', 'tokenizer/vocab.json', 'tokenizer/token_type.json'):
        parsed = _json_file(pack / relative, label=f'{relative} tokenizer output')
        if relative != 'tokenizer/token_type.json' and not isinstance(parsed, dict):
            raise SafetyError(f'Malformed SC117 tokenizer JSON output: {relative}')
    for relative in ('tokenizer/chat_template.jinja', 'tokenizer/merges.txt'):
        if not _read_pack_text(pack, relative).strip():
            raise SafetyError(f'SC117 tokenizer output is empty: {relative}')


def _scan_pack_files(root: Path, *, full_hash=False, require_sealed=False) -> list:
    base = _root_path(root)
    pack = _check_beneath(base, assets.pack_root(base))
    try:
        pack_info = pack.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 native pack is missing: {pack}') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 native pack: {exc}') from exc
    if not stat.S_ISDIR(pack_info.st_mode):
        raise SafetyError('SC117 native pack must be a real non-symlink directory')
    if require_sealed and stat.S_IMODE(pack_info.st_mode) != 0o555:
        raise SafetyError('SC117 native pack directory mode changed; expected immutable 0555')

    found_files, found_dirs = {}, set()

    def visit(directory: Path):
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SafetyError(f'Cannot inspect SC117 pack directory {directory}: {exc}') from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(pack).as_posix()
            try:
                info = path.lstat()
            except OSError as exc:
                raise SafetyError(f'Cannot inspect SC117 pack path {path}: {exc}') from exc
            if stat.S_ISLNK(info.st_mode):
                raise SafetyError(f'SC117 native pack may not contain symlinks: {relative}')
            if stat.S_ISDIR(info.st_mode):
                if relative != 'tokenizer':
                    raise SafetyError(f'Unknown SC117 native pack directory: {relative}')
                if require_sealed and stat.S_IMODE(info.st_mode) != 0o555:
                    raise SafetyError('SC117 tokenizer directory mode changed; expected immutable 0555')
                found_dirs.add(relative)
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                if relative not in PACK_REQUIRED_FILES:
                    is_temporary = (relative.startswith('.') or
                                    relative.endswith(('.tmp', '.part', '.preparing', '~')))
                    kind = 'temporary' if is_temporary else 'unknown'
                    raise SafetyError(f'SC117 native pack contains {kind} file: {relative}')
                if info.st_nlink != 1:
                    raise SafetyError(f'SC117 native pack file has an unsafe hard link: {relative}')
                if require_sealed and stat.S_IMODE(info.st_mode) != 0o444:
                    raise SafetyError(f'SC117 native pack file mode changed; expected immutable 0444: {relative}')
                if info.st_size <= 0:
                    raise SafetyError(f'SC117 native pack file is empty: {relative}')
                found_files[relative] = info
            else:
                raise SafetyError(f'SC117 native pack contains a non-regular file: {relative}')

    visit(pack)
    expected_files = set(PACK_REQUIRED_FILES)
    if set(found_files) != expected_files:
        missing = sorted(expected_files - set(found_files))
        if 'native_experts.txt' in missing:
            raise SafetyError('SC117 native pack is incomplete: native_experts.txt completion marker is missing')
        raise SafetyError(f'SC117 native pack is incomplete; required output missing: {missing[0]}')
    if found_dirs != {'tokenizer'}:
        raise SafetyError('SC117 native pack has an unexpected/missing tokenizer directory')
    source_shards = _pack_source_shards(base)
    _validate_pack_outputs(pack, source_shards)

    records = []
    for relative in PACK_REQUIRED_FILES:
        path = pack / relative
        before = found_files[relative]
        record = {'path': relative, 'bytes': before.st_size,
                  'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns,
                  'mode': stat.S_IMODE(before.st_mode)}
        if full_hash:
            record['sha256'] = sha256_file(path)
            after = path.lstat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns, stat.S_IMODE(before.st_mode)) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                        after.st_ctime_ns, stat.S_IMODE(after.st_mode)):
                raise SafetyError(f'SC117 native pack changed while hashing: {relative}')
        records.append(record)
    return records


def _seal_pack(root: Path):
    base = _root_path(root)
    pack = _check_beneath(base, assets.pack_root(base))
    # Verify single-link regular files before chmod so a malicious hard link cannot change a base asset's mode.
    _scan_pack_files(base)
    for relative in PACK_REQUIRED_FILES:
        path = _check_beneath(base, pack / relative)
        before = path.lstat()
        flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            if os.name == 'posix':
                before_identity += (before.st_ctime_ns,)
                opened_identity += (opened.st_ctime_ns,)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                    opened_identity != before_identity):
                raise SafetyError(f'SC117 native pack file changed before sealing: {relative}')
            os.fchmod(fd, 0o444)
        finally:
            os.close(fd)
    for directory in (pack / 'tokenizer', pack):
        _check_beneath(base, directory)
        os.chmod(directory, 0o555)
    _fsync_directory(pack)


def _pack_manifest_identity(root: Path, packer: dict) -> dict:
    base = _root_path(root)
    return {
        'schema_version': PACK_MANIFEST_SCHEMA,
        'profile': assets.PROFILE,
        'model_id': assets.MODEL_ID,
        'repository': assets.REPOSITORY,
        'revision': assets.REVISION,
        'pack_path': str(assets.pack_root(base)),
        'source_shards': _pack_source_shards(base),
        'packer': packer,
    }


def _request_payload(root: Path, packer: dict) -> dict:
    identity = _pack_manifest_identity(root, packer)
    return {'schema_version': 1, 'profile': assets.PROFILE, 'pack_path': identity['pack_path'],
            'source_shards': identity['source_shards'], 'packer': packer}


def _read_pack_request(root: Path, packer: dict) -> tuple[dict, os.stat_result]:
    base = _root_path(root)
    path = _check_beneath(base, assets.pack_request_path(base))
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise SafetyError('SC117 native pack request is missing; obtain the explicit pinned pack command '
                          'first') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 native pack request: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024 or
            stat.S_IMODE(info.st_mode) != 0o444):
        raise SafetyError('SC117 native pack request must be a small immutable 0444 regular file')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Malformed SC117 native pack request: {exc}') from exc
    expected = _request_payload(base, packer)
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or
            value.get('schema_version') != 1):
        raise SafetyError('Malformed SC117 native pack request: unsupported schema version')
    if value != expected:
        raise SafetyError('SC117 native pack request does not match the pinned source and packer')
    return value, info


def _pack_manifest_file_records(root: Path, *, full_hash=False) -> list:
    return _scan_pack_files(root, full_hash=full_hash, require_sealed=True)


def validate_pack_manifest(value, root: Path, *, full_hash=False, _sources_verified=False) -> dict:
    """Validate a pack report against exact source/tool pins and current file fingerprints."""
    if not isinstance(value, dict):
        raise SafetyError('Malformed SC117 native pack manifest: expected a JSON object')
    base = _root_path(root)
    if not _sources_verified:
        verify_asset_readiness(base, full_hash=full_hash)
    identity = _pinned_packer_identity(base)
    expected = _pack_manifest_identity(base, identity)
    expected_keys = set(expected) | {'pack_files'}
    if set(value) != expected_keys:
        raise SafetyError('Malformed SC117 native pack manifest: missing or unexpected fields')
    if type(value.get('schema_version')) is not int or value['schema_version'] != PACK_MANIFEST_SCHEMA:
        raise SafetyError('Malformed SC117 native pack manifest: unsupported schema version')
    for key, pinned in expected.items():
        if value.get(key) != pinned:
            raise SafetyError(f'Mismatched SC117 native pack manifest: pinned {key} does not match')
    recorded = value.get('pack_files')
    if not isinstance(recorded, list) or len(recorded) != len(PACK_REQUIRED_FILES):
        raise SafetyError('Malformed SC117 native pack manifest: complete pack file report is required')
    actual = _pack_manifest_file_records(base, full_hash=full_hash)
    for record, observed, relative in zip(recorded, actual, PACK_REQUIRED_FILES):
        required = {'path', 'bytes', 'mtime_ns', 'ctime_ns', 'mode', 'sha256'}
        if not isinstance(record, dict) or set(record) != required:
            raise SafetyError('Malformed SC117 native pack manifest: pack file record fields are invalid')
        if (record.get('path') != relative or type(record.get('bytes')) is not int or
                type(record.get('mtime_ns')) is not int or type(record.get('ctime_ns')) is not int or
                type(record.get('mode')) is not int or
                record.get('mode') != 0o444 or
                not isinstance(record.get('sha256'), str) or
                not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise SafetyError('Malformed SC117 native pack manifest: pack file record is invalid')
        for key in ('path', 'bytes', 'mtime_ns', 'ctime_ns', 'mode'):
            if record[key] != observed[key]:
                raise SafetyError(f'SC117 native pack file size/mtime/ctime/mode changed: {relative}')
        if full_hash and record['sha256'] != observed['sha256']:
            raise SafetyError(f'SC117 native pack file checksum mismatch: {relative}')
    return value


def verify_pack_readiness(root: Path, *, full_hash=False, _sources_verified=False) -> dict:
    base = _root_path(root)
    if not _sources_verified:
        verify_asset_readiness(base, full_hash=full_hash)
    request = _check_beneath(base, assets.pack_request_path(base))
    if os.path.lexists(request):
        raise SafetyError(f'SC117 native pack request remains unrecorded: {request}')
    path = _check_beneath(base, assets.pack_manifest_path(base))
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 native pack is not recorded; manifest is missing: {path}') from None
    except OSError as exc:
        raise SafetyError(f'Cannot inspect SC117 native pack manifest: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024 or
            stat.S_IMODE(info.st_mode) != 0o444):
        raise SafetyError('SC117 native pack manifest must be a small immutable 0444 regular file')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Malformed SC117 native pack manifest: {exc}') from exc
    return validate_pack_manifest(value, base, full_hash=full_hash, _sources_verified=True)


def verify_readiness(root: Path, *, full_hash=False) -> dict:
    base = _root_path(root)
    verify_asset_readiness(base, full_hash=full_hash)
    return verify_pack_readiness(base, full_hash=full_hash, _sources_verified=True)


def record_pack(root: Path, *, strata_source: Path = None) -> dict:
    """Record a completed fresh pack; never runs the packer and never overwrites provenance."""
    base = _root_path(root)
    verify_asset_readiness(base, full_hash=True)
    manifest_path = _check_beneath(base, assets.pack_manifest_path(base))
    if os.path.lexists(manifest_path):
        raise SafetyError(f'SC117 native pack manifest already exists; it will not be overwritten: {manifest_path}')
    packer = _pinned_packer_identity(base, strata_source)
    _, request_info = _read_pack_request(base, packer)
    pack = _check_beneath(base, assets.pack_root(base))
    try:
        pack_info = pack.lstat()
    except FileNotFoundError:
        raise SafetyError(f'SC117 native pack has not completed: {pack}') from None
    if not stat.S_ISDIR(pack_info.st_mode) or pack_info.st_mtime_ns < request_info.st_mtime_ns:
        raise SafetyError('SC117 native pack is not fresh relative to the explicit pack command')
    _scan_pack_files(base)
    _seal_pack(base)
    file_records = _pack_manifest_file_records(base, full_hash=True)
    value = _pack_manifest_identity(base, packer)
    value['pack_files'] = file_records
    _atomic_json_at(base, manifest_path, value, label='native pack manifest')
    request = assets.pack_request_path(base)
    try:
        request.chmod(0o600)
        request.unlink()
        _fsync_directory(request.parent)
    except OSError as exc:
        raise SafetyError(f'SC117 pack manifest recorded but request cleanup failed; inspect {request}: '
                          f'{exc}') from exc
    return verify_pack_readiness(base, _sources_verified=True)


def native_pack_command(root: Path, *, strata_source: Path = None) -> list[str]:
    """Print (but never execute) the pinned 0.1.39 native pack command."""
    base = _root_path(root)
    require_new_pack_path(base)
    verify_asset_readiness(base)
    if os.path.lexists(assets.pack_manifest_path(base)):
        raise SafetyError('SC117 native pack is already recorded; it will not be rebuilt')
    request = _check_beneath(base, assets.pack_request_path(base))
    if os.path.lexists(request):
        raise SafetyError('SC117 native pack request is pending; inspect it before requesting another command')
    identity = _pinned_packer_identity(base, strata_source)
    runtime_source, _ = strata_runtime.prepared_runtime(base, PACKER_VERSION)
    if runtime_source != assets.strata_039_source(base):
        raise SafetyError('SC117 native pack requires the exact prepared Strata 0.1.39 runtime')
    return [identity['python_path'], identity['tool_path'], '--gguf', str(assets.shard1_path(base)),
            '--out', str(assets.pack_root(base))]


def build_native_pack(root: Path, *, strata_source: Path = None) -> dict:
    """Run the pinned packer in a clean environment, then seal and record its output."""
    base = _root_path(root)
    source = _check_beneath(base, strata_source or assets.strata_039_source(base))
    command = native_pack_command(base, strata_source=source)
    packer = _pinned_packer_identity(base, source)
    _atomic_json_at(base, assets.pack_request_path(base), _request_payload(base, packer),
                    label='native pack request')
    clean_env = {
        'HOME': str(Path.home()),
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'LANG': 'C.UTF-8',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONNOUSERSITE': '1',
    }
    try:
        result = subprocess.run(command, cwd=source, env=clean_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
    except OSError as exc:
        raise SafetyError(f'Cannot execute the pinned SC117 native packer: {exc}') from exc
    if result.returncode != 0:
        output = (result.stdout or '')[-4000:]
        raise SafetyError('Pinned SC117 native packer failed; partial output/request retained for inspection: ' +
                          output)
    return record_pack(base, strata_source=source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(os.environ.get('STRATA_DATA_ROOT', Path.home() / '.local/share/localllm-strata'))
    parser.add_argument('--data-root', type=Path, default=default_root)
    parser.add_argument('--file', dest='files', action='append', choices=assets.ASSET_PATHS,
                        help='Exact pinned repository path to prepare; repeatable (default: all assets)')
    parser.add_argument('--reuse-shard2', action='store_true',
                        help='Copy old IQ3_S shard 2 only after exact SC117 size/SHA verification')
    parser.add_argument('--check', action='store_true',
                        help='Verify complete assets and native-pack readiness without downloading')
    parser.add_argument('--full-hash', '--reverify', dest='full_hash', action='store_true',
                        help='Recompute every pinned source and pack SHA-256 (slow; use for explicit re-verification)')
    parser.add_argument('--build-pack', action='store_true',
                        help='Run the pinned Strata 0.1.39 packer in a clean environment, then seal/hash its output')
    parser.add_argument('--strata-source', type=Path,
                        help='Use only data-root/runtimes/0.1.39/source when printing/recording the pinned packer')
    args = parser.parse_args(argv)
    try:
        if args.build_pack and (args.check or args.files or args.reuse_shard2):
            raise SafetyError('--build-pack cannot be combined with --check, --file, or --reuse-shard2')
        if args.full_hash and not args.check:
            raise SafetyError('--full-hash/--reverify is supported with --check')
        if args.check and args.strata_source:
            raise SafetyError('--check cannot also select a native packer source')
        if args.build_pack:
            result = build_native_pack(args.data_root, strata_source=args.strata_source)
            print('SC117 native pack built by the pinned Strata 0.1.39 packer, sealed, hashed, and recorded.')
            return 0
        if args.check:
            result = verify_readiness(args.data_root, full_hash=args.full_hash)
            print('SC117 pinned sources and completed native pack are ready' +
                  (' (full hashes verified).' if args.full_hash else ' (metadata readiness verified).'))
            return 0
        result = prepare_assets(args.data_root, args.files, reuse_shard2=args.reuse_shard2)
        if result is None:
            print('Selected SC117 files verified; other pinned files remain unstaged. No manifest published.')
        else:
            print('SC117 pinned source assets are ready. Native pack and inference were not run.')
        if args.strata_source:
            command = native_pack_command(args.data_root, strata_source=args.strata_source)
            print('Explicit native pack command (not run):')
            print(' '.join(__import__('shlex').quote(part) for part in command))
        return 0
    except (SafetyError, OSError, ValueError) as exc:
        print(f'SC117 PREPARATION BLOCKED: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
