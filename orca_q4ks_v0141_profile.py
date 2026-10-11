"""Fail-closed readiness and 32K-gated migration for the opt-in Q4_K_S / Strata 0.1.41 path."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from engine_safety import SafetyError, docker_empty, proc, same_process, strata_state
import hosting_lifecycle as host_lifecycle
import orca_q4ks_profile as baseline_profile
import strata_runtime as runtimes
from strata_runtime import LLAMA_PIN
from orca_q4ks_v0141_assets import (ASSETS, BUILD_OPTIONS, HOST_PROFILE, MODEL_ID, PARITY_TESTS,
                                   PROFILE, RUNTIME_VERSION, SOURCE_COMMIT, baseline_profile_root,
                                   baseline_model_root, baseline_pack_root, build_root, candidate_root,
                                   cmake_cache_contract, pinned_runtime_source, profile_root, source_root)

CUDA_LIB_DIRS = ['/usr/local/cuda-12.9/bin', '/usr/local/cuda-12.9/lib64']
CONVERSATION_CACHE_ARGS = ['--conversation-cache-mib', '8192', '--conversation-cache-slots', '4']
NATIVE_VALIDATION_RAM_FLOOR_MIB = 12_288
NATIVE_VALIDATION_GPU_CEILING_MIB = 24_300
NATIVE_VALIDATION_MAX_REQUEST_S = 1_800


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def hash_tree(path):
    """Content-address a pinned dependency tree, rejecting links/special files."""
    root = Path(path)
    h = hashlib.sha256()
    for item in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        info = item.lstat()
        relative = item.relative_to(root).as_posix().encode()
        if stat.S_ISLNK(info.st_mode):
            raise SafetyError(f'Q4 .41 dependency tree contains a symlink: {item}')
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise SafetyError(f'Q4 .41 dependency tree contains a special file: {item}')
        h.update(len(relative).to_bytes(8, 'big'))
        h.update(relative)
        with item.open('rb') as stream:
            while block := stream.read(8 * 1024 * 1024):
                h.update(block)
    return h.hexdigest()


def record(path, expected_sha=None):
    path = Path(path)
    info = path.stat()
    return {'path': str(path), 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'ctime_ns': info.st_ctime_ns, 'device': info.st_dev, 'inode': info.st_ino,
            'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid,
            'sha256': expected_sha or sha256(path)}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _beneath(root, path, label):
    root = Path(root).resolve()
    path = Path(path)
    if not path.resolve().is_relative_to(root):
        raise SafetyError(f'Q4 .41 {label} resolves outside STRATA_DATA_ROOT')
    return path


def _json(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetyError(f'Q4 .41 {label} is missing or invalid: {exc}') from exc
    if not isinstance(value, dict):
        raise SafetyError(f'Q4 .41 {label} must be a JSON object')
    return value


def _json_with_sha256(path, label):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SafetyError(f'Q4 .41 {label} is missing or invalid: {exc}') from exc
    if not isinstance(value, dict):
        raise SafetyError(f'Q4 .41 {label} must be a JSON object')
    return value, hashlib.sha256(raw).hexdigest()


def require_prepared_manifest_unchanged(root, expected_sha256):
    path = profile_root(root) / 'prepared.json'
    if path.is_symlink() or not path.is_file():
        raise SafetyError('Q4 .41 preparation manifest disappeared or became a symlink during validation')
    try:
        current = sha256(path)
    except OSError as exc:
        raise SafetyError(f'Could not recheck the Q4 .41 preparation manifest: {exc}') from exc
    if current != expected_sha256:
        raise SafetyError('Q4 .41 preparation manifest changed during native validation')
    return current


def _require_private_report(path, label):
    try:
        info = Path(path).lstat()
    except OSError as exc:
        raise SafetyError(f'Q4 .41 {label} is missing or unreadable: {exc}') from exc
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or
            info.st_uid != os.getuid()):
        raise SafetyError(f'Q4 .41 {label} must be a private operator-owned regular file')
    return info


def _assert_candidate_source_status(source, *, built):
    source = Path(source)
    try:
        status = subprocess.check_output(
            ['git', '-c', f'safe.directory={source}', '-C', str(source), 'status', '--porcelain=v1',
             '-z', '--untracked-files=all'], text=True, stderr=subprocess.PIPE)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'Could not verify isolated Q4 .41 source state: {exc}') from exc
    allowed = {'.venv', 'third_party/llama.cpp'}
    if built:
        allowed.add('engine/strata')
    build_name = 'build-orca-q4ks-v0141'
    for entry in status.split('\0'):
        if not entry:
            continue
        if entry[:2] != '??':
            raise SafetyError(f'Q4 .41 pinned source has a tracked modification: {entry[3:]}')
        path = entry[3:]
        if path in allowed or (built and (path == build_name or path.startswith(build_name + '/'))):
            continue
        raise SafetyError(f'Q4 .41 source contains an unexpected untracked input: {path}')


def _file_record(path, saved, *, sealed=False, hash_file=True):
    path = Path(path)
    if path.is_symlink() or not saved or saved.get('path') != str(path) or not path.is_file():
        raise SafetyError(f'Q4 .41 pinned file is missing or linked: {path}')
    info = path.stat()
    identity = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns,
                'device': info.st_dev, 'inode': info.st_ino, 'mode': stat.S_IMODE(info.st_mode),
                'uid': info.st_uid, 'gid': info.st_gid}
    required_identity = ('bytes', 'mtime_ns')
    if any(saved.get(key) != identity[key] for key in required_identity) or any(
            key in saved and saved.get(key) != value for key, value in identity.items()):
        raise SafetyError(f'Q4 .41 pinned file identity changed: {path}')
    if sealed and info.st_mode & 0o222:
        raise SafetyError(f'Q4 .41 pinned file is not read-only: {path}')
    if hash_file and sha256(path) != saved.get('sha256'):
        raise SafetyError(f'Q4 .41 pinned file hash changed: {path}')


def _baseline(root):
    """The .39 profile remains the default and is read-only shared input for this candidate."""
    _, config_path, config = baseline_profile.ready_q4ks(root)
    manifest_path = baseline_profile_root(root) / 'prepared.json'
    manifest = _json(manifest_path, 'baseline .39 manifest')
    if (manifest.get('profile') != HOST_PROFILE or manifest.get('runtime_version') != '0.1.39' or
            manifest.get('source_commit') != baseline_profile.SOURCE_COMMIT):
        raise SafetyError('Q4 .41 requires the exact separately prepared Q4 .39 baseline assets')
    return manifest_path, manifest, config_path, config


def _runtime_material(root, *, validate_helper=True):
    root = Path(root)
    runtime_manifest_path = root / 'runtimes/0.1.41/prepared.json'
    runtime_source = pinned_runtime_source(root)
    runtime_parent = root / 'runtimes/0.1.41'
    for path, label in ((root / 'runtimes', '.41 runtime directory'),
                        (runtime_parent, '.41 runtime version directory'),
                        (runtime_manifest_path, '.41 runtime manifest'),
                        (runtime_source, '.41 runtime source')):
        _beneath(root, path, label)
        if path.is_symlink():
            raise SafetyError(f'Q4 .41 {label} must not be a symlink')
    manifest = runtimes.read_manifest(root, RUNTIME_VERSION)
    if (manifest.get('version') != RUNTIME_VERSION or manifest.get('source_commit') != runtimes.PIN or
            manifest.get('llama_commit') != LLAMA_PIN or manifest.get('source') != str(runtime_source) or
            manifest.get('cuda_arch') != 86 or manifest.get('cuda_toolkit') != '/usr/local/cuda-12.9'):
        raise SafetyError('Q4 .41 runtime manifest/source/toolchain does not match its exact prepared pin')
    if runtimes.command_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                                'rev-parse', 'HEAD']).strip() != runtimes.PIN or \
            runtimes.command_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                                     'diff', '--name-only', 'HEAD']).strip():
        raise SafetyError('Prepared .41 tracked source differs from its exact pin')
    if (runtime_source / 'engine').is_symlink() or (runtime_source / 'third_party').is_symlink():
        raise SafetyError('Prepared .41 engine/llama parent directories must not be symlinks')
    runtime_assets = {item.get('path'): item for item in manifest.get('runtime_assets', [])}
    helper = runtime_source / 'engine/strata-vision'
    helper_record = runtime_assets.get(str(helper))
    engine = runtime_source / 'engine/strata'
    engine_record = runtime_assets.get(str(engine))
    if not helper_record or not engine_record:
        raise SafetyError('Prepared .41 manifest does not bind both runtime executables')
    _file_record(engine, engine_record, sealed=False, hash_file=True)
    if validate_helper:
        _file_record(helper, helper_record, sealed=False, hash_file=True)
    llama = runtime_source / 'third_party/llama.cpp'
    if llama.is_symlink() or not llama.is_dir() or not llama.resolve().is_relative_to(root.resolve()):
        raise SafetyError('Prepared .41 llama.cpp tree is missing, linked, or outside STRATA_DATA_ROOT')
    if ((runtime_source / 'setup.py').is_symlink() or not (runtime_source / 'setup.py').is_file()):
        raise SafetyError('Prepared .41 setup source is missing or linked')
    python = runtime_source / '.venv/bin/python'
    old_python = root / 'source-99f3dbd0b21d/.venv/bin/python'
    environment = Path(manifest.get('python_environment', ''))
    allowed_envs = [root / 'source-99f3dbd0b21d/.venv', runtime_parent / 'python']
    if (not python.is_file() or python.resolve() != old_python.resolve() or environment not in allowed_envs or
            (runtime_source / '.venv').resolve() != environment.resolve()):
        raise SafetyError('Prepared .41 private Python environment differs from its manifest')
    for item in manifest.get('python_assets', []):
        path = Path(item.get('path', ''))
        if (not path.resolve().is_relative_to(environment.resolve()) or not path.is_file() or
                sha256(path) != item.get('sha256')):
            raise SafetyError('Prepared .41 Python dependency contract changed')
    script = ('import setup; assert setup.LLAMA_CPP_COMMIT == ' + repr(LLAMA_PIN) +
              '; print(setup.source_hash(setup.VISION_SOURCES))')
    try:
        vision_source_hash = subprocess.check_output([str(python), '-c', script], cwd=runtime_source,
                                                     text=True, stderr=subprocess.PIPE, timeout=60).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'Could not verify .41 llama/vision source contracts: {exc}') from exc
    build_record_path = runtime_source / 'engine/BUILD.json'
    if build_record_path.is_symlink():
        raise SafetyError('Prepared .41 helper build record must not be a symlink')
    build_json = _json(build_record_path, '.41 helper build record')
    if build_json.get('version') != RUNTIME_VERSION or build_json.get('vision_src') != vision_source_hash:
        raise SafetyError('Prepared .41 vision source record does not match the pinned runtime source')
    # The generic .41 manifest predates this candidate and is intentionally unchanged.
    # Bind the exact dependency contents in the isolated candidate manifest instead.
    llama_sha = hash_tree(llama)
    return runtime_source, manifest, helper, helper_record, llama, llama_sha, vision_source_hash


def _config_with_vision_paths(root, *, context=262144):
    root = Path(root)
    cfg = baseline_profile.make_config(root, context=context, vision=True)
    source = source_root(root)
    cfg['exe'] = str(source / 'engine/strata')
    cfg['cwd'] = str(source)
    cfg['gpu_order'] = 'as_given'
    cfg['vision']['exe'] = str(pinned_runtime_source(root) / 'engine/strata-vision')
    cfg['vision']['mmproj'] = str(profile_root(root) / 'assets' / baseline_profile.VISION_PROJECTOR_FILENAME)
    cfg['args'].extend(CONVERSATION_CACHE_ARGS)
    return cfg


def make_config(root, *, context=32768, vision=False):
    root = Path(root)
    if context not in (32768, 262144) or (vision and context != 262144):
        raise SafetyError('Q4 .41 CPU vision requires its validated native-context migration')
    if vision:
        cfg = _config_with_vision_paths(root, context=context)
        _runtime_material(root, validate_helper=True)
    else:
        cfg = baseline_profile.make_config(root, context=context, vision=False)
        source = source_root(root)
        cfg['exe'] = str(source / 'engine/strata')
        cfg['cwd'] = str(source)
        cfg['gpu_order'] = 'as_given'
        cfg['args'].extend(CONVERSATION_CACHE_ARGS)
    return cfg


def _shared_asset_contract(base_manifest):
    return {
        'assets': sorted([item['path'], item['bytes'], item['sha256']]
                          for item in base_manifest.get('assets', [])),
        'pack_assets': sorted([item['path'], item['bytes'], item['sha256']]
                              for item in base_manifest.get('pack_assets', [])),
    }


def _prepared(root, *, allow_broken_vision=False):
    root = Path(root)
    pr = profile_root(root)
    _beneath(root, pr, 'candidate profile directory')
    if (root / 'profiles').is_symlink() or pr.is_symlink():
        raise SafetyError('Q4 .41 candidate profile directory must not be a symlink')
    manifest_path = pr / 'prepared.json'
    if manifest_path.is_symlink():
        raise SafetyError('Q4 .41 candidate preparation manifest must not be a symlink')
    manifest, manifest_sha256 = _json_with_sha256(manifest_path, 'candidate preparation manifest')
    config_path = pr / 'config.json'
    if config_path.is_symlink():
        raise SafetyError('Q4 .41 candidate config must not be a symlink')
    prepared_config = _json(config_path, 'candidate config')
    base_path, base_manifest, _, _ = _baseline(root)
    validate_helper = prepared_config.get('vision') is not None and not allow_broken_vision
    runtime_source, runtime_manifest, helper, helper_record, llama, llama_sha, vision_hash = \
        _runtime_material(root, validate_helper=validate_helper)
    source = source_root(root)
    candidate_dir = candidate_root(root)
    _beneath(root, candidate_dir, 'candidate runtime directory')
    _beneath(root, source, 'candidate source tree')
    if (root / 'runtimes').is_symlink() or candidate_dir.is_symlink() or source.is_symlink() or not source.is_dir():
        raise SafetyError('Q4 .41 candidate source tree is missing or linked')
    actual = runtimes.command_output(['git', '-c', f'safe.directory={source}', '-C', str(source),
                                      'rev-parse', 'HEAD']).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Q4 .41 candidate source differs from its exact immutable pin')
    _assert_candidate_source_status(source, built=True)
    llama_link = source / 'third_party/llama.cpp'
    if not llama_link.is_symlink() or llama_link.resolve() != llama.resolve():
        raise SafetyError('Q4 .41 build must link only the manifest-verified .41 llama.cpp source')
    python_link = source / '.venv'
    if not python_link.is_symlink() or python_link.resolve() != (runtime_source / '.venv').resolve():
        raise SafetyError('Q4 .41 source must use its exact prepared private Python environment')

    expected_fields = {
        'profile': PROFILE, 'host_profile': HOST_PROFILE, 'model_id': MODEL_ID,
        'model_repository': baseline_profile.REPOSITORY,
        'model_revision': baseline_profile.REVISION,
        'baseline_profile': HOST_PROFILE,
        'runtime_version': RUNTIME_VERSION, 'source_commit': SOURCE_COMMIT,
        'runtime_selection_independent': True,
        'source': str(source), 'runtime_source': str(runtime_source),
        'runtime_manifest_sha256': sha256(root / 'runtimes/0.1.41/prepared.json'),
        'validation_runner_sha256': sha256(Path(__file__).with_name('orca_q4ks_v0141_validation.py')),
        'local_api_client_sha256': sha256(Path(__file__).with_name('local_api_client.py')),
        'runtime_helper': helper_record, 'llama_commit': LLAMA_PIN,
        'llama_source': str(llama), 'llama_tree_sha256': llama_sha,
        'vision_source_sha256': vision_hash,
        'baseline_manifest_sha256': sha256(base_path),
        'shared_assets': _shared_asset_contract(base_manifest),
        'build_options': BUILD_OPTIONS,
        'configured_context': 32768,
        'vision_enabled': False,
        'inference_tested': False,
        'config': str(pr / 'config.json'),
    }
    for key, value in expected_fields.items():
        if manifest.get(key) != value:
            raise SafetyError(f'Q4 .41 preparation provenance mismatch: {key}')

    cmake = Path(manifest.get('build_toolchain', {}).get('cmake', ''))
    cmake_sha = manifest.get('build_toolchain', {}).get('cmake_sha256')
    if (not cmake.is_file() or cmake.is_symlink() or sha256(cmake) != cmake_sha or
            manifest.get('build_toolchain') != base_manifest.get('build_toolchain')):
        raise SafetyError('Q4 .41 build tool must match the separately hash-verified Q4 .39 CMake toolchain')
    candidate_build = build_root(root)
    cache = candidate_build / 'CMakeCache.txt'
    if candidate_build.is_symlink():
        raise SafetyError('Q4 .41 distinct CMake build directory must not be a symlink')
    if manifest.get('build_cache') != str(cache):
        raise SafetyError('Q4 .41 cache path is not its separate profile build tree')
    _beneath(root, cache, 'candidate CMake cache')
    if cache.is_symlink() or not cache.is_file():
        raise SafetyError('Q4 .41 distinct CMake cache is missing')
    cache_text = cache.read_text(errors='replace')
    for contract in cmake_cache_contract(cmake, llama):
        if contract not in cache_text:
            raise SafetyError(f'Q4 .41 distinct build is missing required CMake option: {contract}')

    binaries = {item.get('path'): item for item in manifest.get('runtime_assets', [])}
    engine = source / 'engine/strata'
    engine_record = binaries.get(str(engine))
    _file_record(engine, engine_record, sealed=True)
    test_binaries = {item.get('path'): item for item in manifest.get('test_binaries', [])}
    for name in PARITY_TESTS:
        binary = build_root(root) / name
        _file_record(binary, test_binaries.get(str(binary)), sealed=True)

    initial_config = pr / 'context-config-before.json'
    if initial_config.is_symlink():
        raise SafetyError('Q4 .41 initial config evidence must not be a symlink')
    if initial_config.is_file():
        initial_sha = sha256(initial_config)
    else:
        initial_sha = sha256(config_path)
    if manifest.get('initial_config_sha256') != initial_sha:
        raise SafetyError('Q4 .41 initial config hash differs from its preparation record')
    if manifest.get('inference_tested') is not False:
        raise SafetyError('Q4 .41 preparation must not claim inference validation')
    return manifest, source, pr, prepared_config, engine_record, base_manifest, helper, manifest_sha256


def _check_parity(root, manifest, engine_record):
    path = profile_root(root) / 'parity.json'
    _beneath(root, path, 'candidate parity evidence')
    if path.is_symlink():
        raise SafetyError('Q4 .41 parity evidence must not be a symlink')
    parity = _json(path, 'candidate parity evidence')
    pinned = {name: value[1] for name, value in ASSETS.items()}
    if (parity.get('evidence_class') != 'operator_self_attestation' or
            parity.get('cryptographic_attestation') is not False or
            parity.get('profile') != PROFILE or parity.get('runtime_version') != RUNTIME_VERSION or
            parity.get('source_commit') != SOURCE_COMMIT or
            parity.get('runtime_sha256') != engine_record.get('sha256') or
            parity.get('asset_sha256') != pinned):
        raise SafetyError('Q4 .41 parity evidence is not bound to this exact source/runtime/model')
    tests = parity.get('tests')
    if not isinstance(tests, dict):
        raise SafetyError('Q4 .41 parity evidence has no test-result object')
    for name in PARITY_TESTS:
        result = tests.get(name)
        if not isinstance(result, dict) or result.get('passed') is not True:
            raise SafetyError(f'Q4 .41 required real-shard parity test missing/failed: {name}')


def _validation_report(root, report, config_sha, manifest):
    required = ('health', 'arithmetic', 'strict_json', 'tool_call', 'memory_recorded',
                'conversation_cache_configuration', 'conversation_reuse')
    pinned = {name: value[1] for name, value in ASSETS.items()}
    if (report.get('evidence_class') != 'operator_self_attestation' or
            report.get('cryptographic_attestation') is not False or
            report.get('passed') is not True or report.get('profile') != PROFILE or
            report.get('validation_runner_sha256') != manifest.get('validation_runner_sha256') or
            report.get('local_api_client_sha256') != manifest.get('local_api_client_sha256') or
            report.get('model_id') != MODEL_ID or report.get('source_commit') != SOURCE_COMMIT or
            report.get('runtime_version') != RUNTIME_VERSION or
            report.get('runtime_sha256') != manifest['runtime_assets'][0]['sha256'] or
            report.get('asset_sha256') != pinned or report.get('config_sha256') != config_sha or
            report.get('context') != 32768 or report.get('parallel') != 1 or report.get('text_only') is not True):
        raise SafetyError('Q4 .41 validation report does not match the pinned 32K text-only profile')
    checks = report.get('checks')
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in required):
        raise SafetyError('Q4 .41 validation report is missing a passing functional/memory gate')
    memory_by_gpu = report.get('gpu_memory_peak_mib_by_gpu')
    if (type(report.get('memory_peak_mib')) is not int or report['memory_peak_mib'] <= 0 or
            not isinstance(memory_by_gpu, list) or len(memory_by_gpu) != 4 or
            any(type(value) is not int or value <= 0 for value in memory_by_gpu) or
            report['memory_peak_mib'] != max(memory_by_gpu) or
            type(report.get('gpu_memory_peak_aggregate_mib')) is not int or
            report['gpu_memory_peak_aggregate_mib'] <= 0 or
            type(report.get('ram_available_min_mib')) is not int or report['ram_available_min_mib'] <= 0):
        raise SafetyError('Q4 .41 validation must record positive per-GPU, aggregate, and RAM measurements')
    live_status = report.get('live_status')
    if (not isinstance(live_status, dict) or live_status.get('context') != 32768 or
            live_status.get('vision_enabled') is not False or live_status.get('serving_slots') != 1):
        raise SafetyError('Q4 .41 validation must record the exact live 32K/text-only/single-slot status')
    cache = live_status.get('conversation_cache')
    if (not isinstance(cache, dict) or cache.get('enabled') is not True or
            cache.get('budget_mib') != 8192 or cache.get('slots') != 4 or
            type(cache.get('park_events')) is not int or cache['park_events'] < 1 or
            type(cache.get('restored_conversations')) is not int or cache['restored_conversations'] < 1 or
            type(cache.get('reused_requests')) is not int or cache['reused_requests'] < 1 or
            type(cache.get('reused_tokens')) is not int or cache['reused_tokens'] <= 0):
        raise SafetyError('Q4 .41 validation must record A/B/A/B parking, restoration, and token reuse under an 8192 MiB/four-slot cache')
    if (not _valid_identity(report.get('validation_server_identity')) or
            not _valid_lease_id(report.get('validation_hosting_lease_id'))):
        raise SafetyError('Q4 .41 validation must bind checks to the starting server and HostLLM lease identities')
    mtp = report.get('mtp')
    drafted = mtp.get('drafted_tokens') if isinstance(mtp, dict) else None
    accepted = mtp.get('accepted_tokens') if isinstance(mtp, dict) else None
    if type(drafted) is not int or drafted <= 0 or type(accepted) is not int or not 0 <= accepted <= drafted:
        raise SafetyError('Q4 .41 validation must measure original Flash-Next MTP drafted/accepted tokens')


def _valid_lease_id(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def _valid_identity(identity):
    return (isinstance(identity, dict) and type(identity.get('pid')) is int and identity['pid'] > 0 and
            type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0 and
            isinstance(identity.get('exe'), str) and identity['exe'].startswith('/') and
            type(identity.get('uid')) is int and type(identity.get('pgid')) is int and
            type(identity.get('sid')) is int and isinstance(identity.get('cmd'), list))


def _native_context_expected_config(root, pr, config, helper):
    """Text-only readiness and rollback do not depend on optional vision artifacts."""
    text = make_config(root, context=262144)
    vision = _config_with_vision_paths(root, context=262144)
    if config == vision:
        if (Path(pr) / 'vision-prepared.json').is_symlink():
            raise SafetyError('Q4 .41 CPU-vision provenance must not be a symlink')
        _validate_vision_sidecar(root, pr, helper)
        return vision
    return text


def ready_q4ks_v0141(root):
    """Read-only readiness for only the explicitly requested .41 Q4 candidate."""
    manifest, source, pr, cfg, engine_record, _, helper, _ = _prepared(root)
    _check_parity(root, manifest, engine_record)
    config_path = pr / 'config.json'
    config = cfg
    context = None
    args = config.get('args')
    if isinstance(args, list) and args.count('--max-context') == 1:
        try:
            context = int(args[args.index('--max-context') + 1])
        except (ValueError, IndexError):
            pass
    migration_path = pr / 'native-context-migration.json'
    validation_path = pr / 'validation-32k.json'
    vision_path = pr / 'vision-prepared.json'
    backup_path = pr / 'context-config-before.json'
    for path, label in ((migration_path, 'migration evidence'), (validation_path, '32K evidence'),
                        (backup_path, 'initial config evidence')):
        _beneath(root, path, label)
        if path.is_symlink():
            raise SafetyError(f'Q4 .41 {label} must not be a symlink')
    if context == 32768:
        if migration_path.exists() or vision_path.exists() or vision_path.is_symlink():
            raise SafetyError('Q4 .41 initial 32K profile conflicts with a migration/vision record')
        expected = make_config(root, context=32768)
        if validation_path.exists():
            _require_private_report(validation_path, '32K validation evidence')
            report = _json(validation_path, '32K validation evidence')
            _validation_report(root, report, sha256(config_path), manifest)
            if (report.get('passed') is not True or
                    not _valid_identity(report.get('recorded_server_identity')) or
                    not _valid_lease_id(report.get('recorded_hosting_lease_id'))):
                raise SafetyError('Q4 .41 32K validation evidence is incomplete or not identity/lease-bound')
    elif context == 262144:
        if not backup_path.is_file() or not validation_path.is_file() or not migration_path.is_file():
            raise SafetyError('Q4 .41 native context requires its own recorded passing 32K gate first')
        backup = _json(backup_path, 'initial config evidence')
        _require_private_report(validation_path, '32K validation evidence')
        validation = _json(validation_path, '32K validation evidence')
        _validation_report(root, validation, sha256(backup_path), manifest)
        migration = _json(migration_path, 'native-context migration evidence')
        migration_fields = {'profile', 'source_commit', 'context', 'initial_config_sha256',
                            'validation_sha256', 'current_config_sha256', 'recorded_at'}
        if (set(migration) != migration_fields or
                not isinstance(migration.get('recorded_at'), str) or
                backup != make_config(root, context=32768) or validation.get('passed') is not True or
                not _valid_identity(validation.get('recorded_server_identity')) or
                not _valid_lease_id(validation.get('recorded_hosting_lease_id')) or
                migration.get('profile') != PROFILE or migration.get('source_commit') != SOURCE_COMMIT or
                migration.get('context') != 262144 or
                migration.get('initial_config_sha256') != sha256(backup_path) or
                migration.get('validation_sha256') != sha256(validation_path) or
                migration.get('current_config_sha256') != sha256(config_path)):
            raise SafetyError('Q4 .41 native-context migration evidence is stale or inconsistent')
        expected = _native_context_expected_config(root, pr, config, helper)
    else:
        raise SafetyError('Q4 .41 profile must remain 32K until its own bound validation is recorded')
    if config != expected:
        raise SafetyError('Q4 .41 config differs from its exact pinned text-only/CPU-vision preset')
    return source, config_path, config


def _assert_live_candidate(root):
    state_path = strata_state() / 'server.json'
    state = _json(state_path, 'live ownership state')
    identity = state.get('identity')
    current = proc(identity.get('pid')) if _valid_identity(identity) else None
    if (state.get('profile') != HOST_PROFILE or state.get('runtime_version') != RUNTIME_VERSION or
            state.get('source_commit') != SOURCE_COMMIT or not same_process(identity, current)):
        raise SafetyError('Live process is not the identity-verified Q4 .41 candidate')
    lease = host_lifecycle.read_lease()
    lease_id = lease.get('lease_id') if isinstance(lease, dict) else None
    if (not isinstance(lease, dict) or lease.get('phase') != 'held' or lease.get('port') != state.get('port') or
            not isinstance(lease_id, str) or not re.fullmatch(r'[a-f0-9]{64}', lease_id) or
            state.get('hosting_lease_id') != lease_id):
        raise SafetyError('Live Q4 .41 candidate is not held under its recorded HostLLM lease identity')
    config_path = Path(state.get('config', ''))
    expected_config_path = strata_state() / 'run-config.json'
    if (config_path != expected_config_path or config_path.is_symlink() or
            not config_path.resolve().is_relative_to(strata_state().resolve())):
        raise SafetyError('Live Q4 .41 run config path is not the HostLLM-owned private config')
    expected_python = source_root(root) / '.venv/bin/python'
    if identity.get('exe') != str(expected_python.resolve()):
        raise SafetyError('Live Q4 .41 frontend executable is not the prepared candidate Python')
    live = _json(config_path, 'live run config')
    expected = make_config(root, context=32768)
    keys = ('exe', 'cwd', 'args', 'tokenizer', 'gpu', 'layer_split', 'gpu_order',
            'model_name', 'aliases', 'sampling')
    if (live.get('host') != '127.0.0.1' or
            any(live.get(key) != expected.get(key) for key in keys) or live.get('vision') is not None or
            state.get('parallel', 1) != 1 or state.get('batch_groups', 1) != 1):
        raise SafetyError('Live server is not the exact 32K single-slot text-only Q4 .41 profile')
    python = source_root(root) / '.venv/bin/python'
    engine = source_root(root) / 'engine/strata'
    if set(state.get('allowed_executables', [])) != {str(python.resolve()), str(engine.resolve())}:
        raise SafetyError('Live Q4 .41 process allowlist differs or contains an unverified helper')
    return state


def _profile_lock(root):
    path = profile_root(root) / 'prepare.lock'
    _beneath(root, path, 'candidate lock')
    if path.is_symlink() or not path.is_file():
        raise SafetyError('Q4 .41 candidate lock is missing or linked')
    return path


def _update_native(root, config_path, migration_path, previous_config, previous_migration, updated):
    try:
        atomic_json(config_path, updated)
        migration = json.loads(previous_migration)
        migration['current_config_sha256'] = sha256(config_path)
        atomic_json(migration_path, migration)
        ready_q4ks_v0141(root)
    except Exception:
        config_path.write_bytes(previous_config)
        migration_path.write_bytes(previous_migration)
        raise


def configure_native_context(root):
    pr = profile_root(root)
    lock_fd = os.open(_profile_lock(root), os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        _, config_path, config = ready_q4ks_v0141(root)
        if config != make_config(root, context=32768):
            raise SafetyError('Q4 .41 native-context migration starts only from the exact 32K text-only preset')
        validation_path = pr / 'validation-32k.json'
        _require_private_report(validation_path, 'candidate 32K validation')
        validation = _json(validation_path, 'candidate 32K validation')
        manifest, *_ = _prepared(root)
        _validation_report(root, validation, sha256(config_path), manifest)
        if (validation.get('evidence_class') != 'operator_self_attestation' or
                validation.get('cryptographic_attestation') is not False or
                validation.get('passed') is not True or
                not _valid_identity(validation.get('recorded_server_identity')) or
                not _valid_lease_id(validation.get('recorded_hosting_lease_id')) or
                not validation.get('recorded_server_config')):
            raise SafetyError('Q4 .41 native-context migration requires its recorded identity-bound 32K gate')
        backup_path = pr / 'context-config-before.json'
        migration_path = pr / 'native-context-migration.json'
        if backup_path.exists() or migration_path.exists():
            raise SafetyError('Unexpected existing Q4 .41 migration evidence; inspect rather than overwrite')
        previous = config_path.read_bytes()
        atomic_json(backup_path, config)
        try:
            atomic_json(config_path, make_config(root, context=262144))
            atomic_json(migration_path, {
                'profile': PROFILE, 'source_commit': SOURCE_COMMIT, 'context': 262144,
                'initial_config_sha256': sha256(backup_path),
                'validation_sha256': sha256(validation_path),
                'current_config_sha256': sha256(config_path),
                'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            ready_q4ks_v0141(root)
        except Exception:
            config_path.write_bytes(previous)
            migration_path.unlink(missing_ok=True)
            backup_path.unlink(missing_ok=True)
            raise
    print('Q4 .41 future profile migrated to native context only after its own passing 32K evidence; CPU vision remains off.')


def _verified_vision_sidecar(root, pr, helper):
    source_projector, source_record = baseline_profile._verified_iq3_projector(root, hash_file=True)
    target = pr / 'assets' / baseline_profile.VISION_PROJECTOR_FILENAME
    sidecar_path = pr / 'vision-prepared.json'
    _beneath(root, target, 'candidate CPU projector')
    _beneath(root, sidecar_path, 'candidate vision evidence')
    sidecar = _json(sidecar_path, 'candidate CPU-vision evidence')
    runtime_record = {item.get('path'): item for item in runtimes.read_manifest(root, RUNTIME_VERSION).get('runtime_assets', [])}.get(str(helper))
    if (not target.is_file() or target.is_symlink() or not runtime_record or sha256(helper) != runtime_record.get('sha256') or
            not os.path.samefile(source_projector, target) or target.stat().st_mode & 0o222):
        raise SafetyError('Q4 .41 CPU-vision helper/projector is not the exact verified read-only input')
    expected = {'profile': PROFILE, 'model_id': MODEL_ID, 'source_commit': SOURCE_COMMIT,
                'runtime_version': RUNTIME_VERSION, 'helper': record(helper, runtime_record['sha256']),
                'projector_source': str(source_projector), 'projector_source_record': source_record,
                'projector': record(target, baseline_profile.VISION_PROJECTOR_SHA256)}
    actual_projector = sidecar.get('projector')
    if not isinstance(actual_projector, dict):
        raise SafetyError('Q4 .41 CPU-vision projector record is malformed')
    recorded_ctime = actual_projector.get('ctime_ns')
    if type(recorded_ctime) is not int or recorded_ctime < 0:
        raise SafetyError('Q4 .41 CPU-vision projector ctime record is malformed')
    # Adding/removing another hardlink changes inode ctime, not the verified projector contents.
    comparable_sidecar = dict(sidecar)
    comparable_projector = dict(actual_projector)
    comparable_projector.pop('ctime_ns')
    comparable_sidecar['projector'] = comparable_projector
    expected_projector = dict(expected['projector'])
    expected_projector.pop('ctime_ns')
    expected['projector'] = expected_projector
    if comparable_sidecar != expected:
        raise SafetyError('Q4 .41 CPU-vision provenance does not match its .41 helper/projector pins')


def _validate_vision_sidecar(root, pr, helper):
    _verified_vision_sidecar(root, pr, helper)


def configure_cpu_vision(root):
    pr = profile_root(root)
    lock_fd = os.open(_profile_lock(root), os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        _, config_path, config = ready_q4ks_v0141(root)
        native_text = make_config(root, context=262144)
        native_vision = make_config(root, context=262144, vision=True)
        if config == native_vision:
            return
        if config != native_text:
            raise SafetyError('Q4 .41 CPU vision requires native-context migration after its own 32K validation')
        _, _, helper, _, _, _, _ = _runtime_material(root)
        source_projector, source_record = baseline_profile._verified_iq3_projector(root, hash_file=True)
        asset_dir = pr / 'assets'
        _beneath(root, asset_dir, 'candidate asset directory')
        if asset_dir.is_symlink():
            raise SafetyError('Q4 .41 candidate asset directory must not be a symlink')
        asset_dir.mkdir(exist_ok=True)
        target = asset_dir / baseline_profile.VISION_PROJECTOR_FILENAME
        sidecar_path = pr / 'vision-prepared.json'
        if (target.exists() or target.is_symlink() or sidecar_path.exists() or sidecar_path.is_symlink()):
            raise SafetyError('Unexpected Q4 .41 projector/vision evidence already exists; refusing overwrite')
        if source_projector.stat().st_dev != target.parent.stat().st_dev:
            raise SafetyError('Q4 .41 candidate and pinned projector are on different filesystems; hardlink refused')
        os.link(source_projector, target, follow_symlinks=False)
        previous_config = config_path.read_bytes()
        migration_path = pr / 'native-context-migration.json'
        previous_migration = migration_path.read_bytes()
        try:
            helper_record = {item.get('path'): item for item in runtimes.read_manifest(root, RUNTIME_VERSION).get('runtime_assets', [])}.get(str(helper))
            if not helper_record:
                raise SafetyError('Q4 .41 prepared manifest does not bind its vision helper')
            atomic_json(sidecar_path, {
                'profile': PROFILE, 'model_id': MODEL_ID, 'source_commit': SOURCE_COMMIT,
                'runtime_version': RUNTIME_VERSION, 'helper': record(helper, helper_record['sha256']),
                'projector_source': str(source_projector), 'projector_source_record': source_record,
                'projector': record(target, baseline_profile.VISION_PROJECTOR_SHA256),
            })
            updated = make_config(root, context=262144, vision=True)
            _update_native(root, config_path, migration_path, previous_config, previous_migration, updated)
        except Exception:
            target.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
            raise
    print('Q4 .41 future profile configured for CPU vision after its own 32K gate and native migration; active server untouched.')


def rollback_cpu_vision(root):
    pr = profile_root(root)
    lock_fd = os.open(_profile_lock(root), os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        manifest, source, checked_pr, config, engine_record, _, helper, _ = _prepared(root, allow_broken_vision=True)
        _check_parity(root, manifest, engine_record)
        config_path = checked_pr / 'config.json'
        if config != _config_with_vision_paths(root, context=262144):
            raise SafetyError('Q4 .41 vision rollback requires its exact migrated CPU-vision preset')
        migration_path = checked_pr / 'native-context-migration.json'
        migration = migration_path.read_bytes()
        previous = config_path.read_bytes()
        _update_native(root, config_path, migration_path, previous, migration, make_config(root, context=262144))
    print('Q4 .41 future config rolled back to native-context text-only; projector provenance retained.')


def candidate_evidence_summary(root):
    pr = profile_root(root)
    config_path = pr / 'config.json'
    if not config_path.is_file():
        return ['No separate Q4 .41 candidate evidence is recorded.']
    output = []
    prepared_manifest = None
    prepared_manifest_sha256 = None
    validation_path = pr / 'validation-32k.json'
    if not validation_path.is_file() or validation_path.is_symlink():
        output.append('No Q4 .41 identity-bound 32K text-only validation; native-context/CPU vision remain gated.')
    else:
        try:
            _require_private_report(validation_path, '32K evidence')
            prepared = _prepared(root)
            prepared_manifest, prepared_manifest_sha256 = prepared[0], prepared[-1]
            report = _json(validation_path, '32K evidence')
            _validation_report(root, report, sha256(pr / 'context-config-before.json')
                               if (pr / 'context-config-before.json').is_file() else sha256(config_path),
                               prepared_manifest)
            if (report.get('evidence_class') == 'operator_self_attestation' and
                    report.get('cryptographic_attestation') is False and report.get('passed') is True and
                    _valid_identity(report.get('recorded_server_identity')) and
                    _valid_lease_id(report.get('recorded_hosting_lease_id'))):
                output.append('Q4 .41 32K result is an operator self-attestation (not cryptographically verified); migration remains explicitly opt-in.')
            else:
                output.append('Stored Q4 .41 32K report is stale/incomplete and is not counted.')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, SafetyError):
            output.append('Stored Q4 .41 32K report is unreadable/stale and is not counted.')

    native_path = pr / 'validation-native.json'
    if not native_path.is_file() or native_path.is_symlink():
        output.append('No matching Q4 .41 native-context/CPU-vision live report is recorded.')
    else:
        try:
            native_info = _require_private_report(native_path, 'native-context/vision evidence')
            report = _json(native_path, 'native-context/vision evidence')
            if prepared_manifest is None:
                prepared = _prepared(root)
                prepared_manifest, prepared_manifest_sha256 = prepared[0], prepared[-1]
            health = report.get('health')
            vision = report.get('vision')
            full = report.get('full_context')
            resources = report.get('resource_measurement')
            text_smoke = report.get('text_smoke')
            token_control = report.get('token_count_control')
            safety = report.get('safety')
            runner = Path(__file__).with_name('orca_q4ks_v0141_native_validation.py')
            answer = vision.get('answer') if isinstance(vision, dict) else None
            marker = full.get('expected_tail_marker') if isinstance(full, dict) else None
            tokens = full.get('response_input_tokens') if isinstance(full, dict) else None
            elapsed_ms = full.get('elapsed_milliseconds') if isinstance(full, dict) else None
            gpu_peak = resources.get('gpu_memory_peak_mib_by_gpu') if isinstance(resources, dict) else None
            aggregate_peak = resources.get('gpu_memory_peak_aggregate_mib') if isinstance(resources, dict) else None
            valid = (
                report.get('evidence_class') == 'operator_self_attestation' and
                report.get('cryptographic_attestation') is False and
                report.get('validation_runner_sha256') == sha256(runner) and
                report.get('local_api_client_sha256') == sha256(Path(__file__).with_name('local_api_client.py')) ==
                prepared_manifest.get('local_api_client_sha256') and
                report.get('prepared_manifest_sha256') == prepared_manifest_sha256 and
                stat.S_ISREG(native_info.st_mode) and stat.S_IMODE(native_info.st_mode) == 0o600 and
                native_info.st_uid == os.getuid() and
                report.get('profile') == HOST_PROFILE and report.get('model_id') == MODEL_ID and
                report.get('source_commit') == SOURCE_COMMIT and report.get('runtime_version') == RUNTIME_VERSION and
                report.get('config_sha256') == sha256(config_path) and
                report.get('context_configured') == 262144 and report.get('vision_configured') is True and
                report.get('vision_cpu_only') is True and report.get('passed') is True and
                isinstance(text_smoke, dict) and text_smoke.get('passed') is True and
                text_smoke.get('answer', '').strip() == '437' and text_smoke.get('finish_reason') == 'stop' and
                isinstance(token_control, dict) and token_control.get('passed') is True and
                token_control.get('answer', '').strip() == '437' and
                type(token_control.get('uncached_response_input_tokens')) is int and
                token_control['uncached_response_input_tokens'] >= 0 and
                type(token_control.get('cache_read_input_tokens')) is int and
                token_control['cache_read_input_tokens'] >= 0 and
                token_control.get('response_input_tokens') ==
                token_control['uncached_response_input_tokens'] + token_control['cache_read_input_tokens'] and
                token_control.get('counted_tokens') == token_control.get('response_input_tokens') and
                token_control.get('stop_reason') == 'end_turn' and
                isinstance(safety, dict) and safety.get('ram_floor_mib') == NATIVE_VALIDATION_RAM_FLOOR_MIB and
                safety.get('gpu_memory_ceiling_mib') == NATIVE_VALIDATION_GPU_CEILING_MIB and
                safety.get('max_request_seconds') == NATIVE_VALIDATION_MAX_REQUEST_S and
                _valid_identity(report.get('live_server_identity')) and
                _valid_lease_id(report.get('live_hosting_lease_id')) and
                isinstance(health, dict) and health.get('status') == 'ok' and health.get('loaded') is True and
                health.get('max_context') == 262144 and health.get('images') is True and
                isinstance(vision, dict) and vision.get('passed') is True and vision.get('expected') == 'magenta' and
                vision.get('cpu_only_configured') is True and vision.get('finish_reason') == 'stop' and
                isinstance(answer, str) and answer.strip().strip(' .,!?:;').casefold() == 'magenta' and
                isinstance(full, dict) and full.get('passed') is True and full.get('count_match') is True and
                type(full.get('uncached_response_input_tokens')) is int and
                full['uncached_response_input_tokens'] >= 0 and
                type(full.get('cache_read_input_tokens')) is int and full['cache_read_input_tokens'] >= 0 and
                full.get('response_input_tokens') ==
                full['uncached_response_input_tokens'] + full['cache_read_input_tokens'] and
                full.get('counted_prompt_tokens') == tokens and full.get('marker_retrieved') is True and
                type(elapsed_ms) is int and 0 <= elapsed_ms <= NATIVE_VALIDATION_MAX_REQUEST_S * 1000 and
                isinstance(marker, str) and marker and isinstance(full.get('answer'), str) and
                marker in full['answer'] and full.get('stop_reason') == 'end_turn' and
                type(tokens) is int and 255000 <= tokens <= 262080 and
                isinstance(resources, dict) and resources.get('sampler_errors') == [] and
                resources.get('sampler_error_count') == 0 and type(resources.get('sampler_samples')) is int and
                resources['sampler_samples'] > 0 and resources.get('sampler_thread_stopped') is True and
                type(resources.get('last_sample_age_ms')) is int and 0 <= resources['last_sample_age_ms'] <= 5000 and
                type(resources.get('ram_available_min_mib')) is int and
                resources['ram_available_min_mib'] >= NATIVE_VALIDATION_RAM_FLOOR_MIB and
                isinstance(gpu_peak, list) and len(gpu_peak) == 4 and
                all(type(value) is int and 0 < value <= NATIVE_VALIDATION_GPU_CEILING_MIB
                    for value in gpu_peak) and
                type(aggregate_peak) is int and max(gpu_peak) <= aggregate_peak <= sum(gpu_peak)
            )
            if valid:
                output.append(f'Recorded Q4 .41 native/CPU-vision probes match this config: synthetic image passed; '
                              f'{tokens:,}/262,144 tokens with tail-marker retrieval. One bounded probe, not general quality/load qualification.')
            else:
                output.append('Stored Q4 .41 native/vision report is stale/incomplete for this config and is not counted.')
        except (OSError, ValueError, KeyError, TypeError, AttributeError, SafetyError):
            output.append('Stored Q4 .41 native/vision report is unreadable and is not counted.')
    return output
