"""Strict readiness and explicitly gated context migration for Orca Q4_K_S."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from engine_safety import SafetyError, command_output, docker_empty, proc, same_process, strata_state
from orca_q4ks_assets import (ASSETS, BUILD_OPTIONS, CMAKE_VERSION, CMAKE_WHEEL_SHA256, MODEL_ID,
                              PARITY_TESTS, PROFILE, BASE_SOURCE_COMMIT, REPOSITORY, REVISION,
                              RUNTIME_VERSION, SOURCE_COMMIT, IQ3XXS_PROFILE, IQ3XXS_SOURCE_COMMIT,
                              IQ3XXS_REPOSITORY, IQ3XXS_REVISION, IQ3XXS_ASSETS,
                              VISION_PROJECTOR_BYTES, VISION_PROJECTOR_FILENAME,
                              VISION_PROJECTOR_SHA256, cmake_binary,
                              cmake_wheel, model_root, pack_root, pinned_runtime_source,
                              profile_root, source_root)
import strata_runtime as runtimes
from strata_runtime import LLAMA_PIN

CUDA_LIB_DIRS = ['/usr/local/cuda-12.9/bin', '/usr/local/cuda-12.9/lib64']
CMAKE_CACHE_CONTRACT = '\n'.join(
    [f'{key}:BOOL={value}' for key, value in BUILD_OPTIONS.items() if key != 'CMAKE_CUDA_ARCHITECTURES'] +
    [f"CMAKE_CUDA_ARCHITECTURES:STRING={BUILD_OPTIONS['CMAKE_CUDA_ARCHITECTURES']}",
     'CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda-12.9/bin/nvcc'])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def record(path, expected_sha=None):
    path = Path(path)
    st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'ctime_ns': st.st_ctime_ns, 'device': st.st_dev, 'inode': st.st_ino,
            'mode': stat.S_IMODE(st.st_mode), 'uid': st.st_uid, 'gid': st.st_gid,
            'sha256': expected_sha or sha256(path)}


def atomic_json(path, value):
    return atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode('utf-8'))


def atomic_bytes(path, content):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    with temporary.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_beneath(root, path, label):
    base = Path(root).resolve()
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(base):
        raise SafetyError(f'Q4_K_S {label} resolves outside STRATA_DATA_ROOT: {candidate}')
    return Path(path)


def _assert_tool_path(root, path, label):
    path = Path(path)
    resolved = path.resolve()
    base = Path(root).resolve()
    if resolved.is_relative_to(base):
        return path
    try:
        info = resolved.stat()
    except OSError as exc:
        raise SafetyError(f'Q4_K_S {label} is missing: {resolved}') from exc
    if (not path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
            info.st_mode & 0o022 or not info.st_mode & 0o111):
        raise SafetyError(f'Q4_K_S {label} resolves outside STRATA_DATA_ROOT to an untrusted tool: {resolved}')
    return path


def check_record(path, saved, expected_sha=None, *, hash_file=False, sealed=False):
    path = Path(path)
    if path.is_symlink() or not saved or saved.get('path') != str(path) or not path.is_file():
        raise SafetyError(f'Q4_K_S asset missing or changed: {path}')
    st = path.stat()
    identity = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns,
                'device': st.st_dev, 'inode': st.st_ino, 'mode': stat.S_IMODE(st.st_mode),
                'uid': st.st_uid, 'gid': st.st_gid}
    if any(saved.get(key) != value for key, value in identity.items()) or \
            (expected_sha and saved.get('sha256') != expected_sha):
        raise SafetyError(f'Q4_K_S asset missing or changed: {path}')
    # Pinned downloaded/build outputs are sealed read-only after initial SHA-256 verification.
    if sealed and identity['mode'] & 0o222:
        raise SafetyError(f'Q4_K_S asset is not sealed read-only: {path}')
    if hash_file and sha256(path) != saved.get('sha256'):
        raise SafetyError(f'Q4_K_S asset hash changed: {path}')


def make_config(root, *, context=32768, vision=False):
    root = Path(root)
    source = source_root(root)
    shards = [model_root(root) / name for name in ASSETS]
    pack = pack_root(root)
    if context not in (32768, 262144) or (vision and context != 262144):
        raise SafetyError('Q4_K_S vision requires the migrated native 262144 context')
    args = ['--pack', str(pack), '--native', str(shards[0]), '--ple-gguf', str(shards[0]),
            '--ple-io', 'mmap', '--expert-profile', str(root / 'source-99f3dbd0b21d/data/expert-profile.bin'),
            '--expert-cache', 'auto', '--prefill', '512', '--spec', '4', '--mtp', str(root / 'data/mtp/rt'),
            '--max-context', str(context), '--kv', 'int8']
    if context == 262144:
        args += ['--kv-resident', '32768']
    cfg = {
        'exe': str(source / 'engine/strata'), 'cwd': str(source), 'args': args,
        'tokenizer': str(pack / 'tokenizer'), 'gpu': [0, 1, 2, 3], 'layer_split': 'auto',
        'model_name': MODEL_ID, 'aliases': ['orca-q4ks', 'orca-q4_k_s'],
        'host': '0.0.0.0', 'port': 8080, 'lib_dirs': list(CUDA_LIB_DIRS),
        'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
                     'experimental_speed_projection': False},
    }
    if vision:
        args.append('--vision')
        cfg['vision'] = {
            'exe': str(pinned_runtime_source(root) / 'engine/strata-vision'),
            'model': str(shards[0]),
            'mmproj': str(model_root(root) / VISION_PROJECTOR_FILENAME),
            'gpu': False,
            'max_tokens': 1024,
        }
    return cfg


def _load_json(path, message):
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise SafetyError(message) from None
    except (json.JSONDecodeError, OSError) as exc:
        raise SafetyError(f'Invalid Q4_K_S metadata: {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise SafetyError(f'Invalid Q4_K_S metadata; expected a JSON object: {path}')
    return value


def _prepared(root):
    root = Path(root)
    pr = profile_root(root)
    _assert_beneath(root, pr, 'profile directory')
    if pr.is_symlink():
        raise SafetyError('Q4_K_S profile directory is a symlink')
    prepared_path = pr / 'prepared.json'
    _assert_beneath(root, prepared_path, 'preparation manifest')
    manifest = _load_json(prepared_path,
                          'Q4_K_S is not prepared. Run ./v1strata.sh --prepare-orca-q4ks explicitly.')
    if (manifest.get('profile') != PROFILE or manifest.get('model_id') != MODEL_ID or
            manifest.get('model_repository') != REPOSITORY or manifest.get('model_revision') != REVISION or
            manifest.get('source_commit') != SOURCE_COMMIT or manifest.get('source') != str(source_root(root)) or
            manifest.get('runtime_version') != RUNTIME_VERSION or manifest.get('config') != str(pr / 'config.json') or
            manifest.get('pack') != str(pack_root(root)) or manifest.get('build_options') != BUILD_OPTIONS or
            manifest.get('llama_cpp_commit') != LLAMA_PIN or
            manifest.get('llama_cpp_source') != str((pinned_runtime_source(root) / 'third_party/llama.cpp').resolve()) or
            manifest.get('runtime_selection_independent') is not True or
            manifest.get('mtp_source_commit') != BASE_SOURCE_COMMIT or
            manifest.get('inference_tested') is not False or manifest.get('vision_enabled') is not False):
        raise SafetyError('Q4_K_S preparation/source/build provenance mismatch')

    source = source_root(root)
    runtime_source = pinned_runtime_source(root)
    base_source = root / 'source-99f3dbd0b21d'
    pack = pack_root(root)
    for path, label in ((source, 'Q4 source tree'), (runtime_source, 'pinned .39 tool source'),
                        (base_source, 'Flash-Next MTP source'), (pack, 'compatibility pack')):
        _assert_beneath(root, path, label)
    if source.is_symlink() or base_source.is_symlink() or pack.is_symlink():
        raise SafetyError('Q4 source/pack inputs must not be symlinked directories')
    actual = command_output(['git', '-c', f'safe.directory={source}', '-C', str(source), 'rev-parse', 'HEAD']).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Q4_K_S side-by-side source pin changed')
    llama_link = source / 'third_party/llama.cpp'
    pinned_llama = runtime_source / 'third_party/llama.cpp'
    _assert_beneath(root, llama_link, 'Q4 llama.cpp link')
    _assert_beneath(root, pinned_llama, 'pinned llama.cpp source')
    if not llama_link.is_symlink() or llama_link.resolve() != pinned_llama.resolve():
        raise SafetyError('Q4_K_S must reuse the llama.cpp source from the prepared .39 runtime')
    if not (pinned_llama / 'ggml/CMakeLists.txt').is_file() or not (pinned_llama / 'gguf-py').is_dir():
        raise SafetyError('Pinned .39 llama.cpp build dependency is incomplete')
    shared_source, _ = runtimes.prepared_runtime(root, RUNTIME_VERSION)
    _assert_beneath(root, shared_source, 'prepared .39 source')
    if shared_source.resolve() != runtime_source.resolve():
        raise SafetyError('Q4_K_S must use the exact side-by-side Strata 0.1.39 source tree')
    runtime_actual = command_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                                    'rev-parse', 'HEAD']).strip()
    if runtime_actual != SOURCE_COMMIT or manifest.get('python_environment') != str((runtime_source / '.venv').resolve()):
        raise SafetyError('Q4_K_S tool environment is not bound to the separate pinned .39 source')
    if command_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                       'diff', '--name-only', 'HEAD']).strip():
        raise SafetyError('Q4_K_S pinned .39 tool-environment source has tracked modifications')
    python_path = runtime_source / '.venv/bin/python'
    _assert_tool_path(root, python_path, 'Q4 private Python')
    _assert_beneath(root, source / '.venv', 'Q4 private environment link')
    if not python_path.is_file() or (source / '.venv').resolve() != (runtime_source / '.venv').resolve():
        raise SafetyError('Q4_K_S private Python environment link differs from pinned .39')
    if command_output(['git', '-c', f'safe.directory={source}', '-C', str(source), 'diff', '--name-only', 'HEAD']).strip():
        raise SafetyError('Q4_K_S source has tracked modifications')
    toolchain = manifest.get('build_toolchain', {})
    expected_cmake = cmake_binary(root)
    expected_wheel = cmake_wheel(root)
    _assert_beneath(root, expected_cmake, 'isolated Q4 CMake')
    _assert_beneath(root, expected_wheel, 'pinned Q4 CMake wheel')
    if (expected_cmake.is_symlink() or not expected_cmake.is_file() or not os.access(expected_cmake, os.X_OK) or
            expected_wheel.is_symlink() or not expected_wheel.is_file()):
        raise SafetyError('Q4-isolated CMake binary or wheel is missing or linked')
    wheel_sha = sha256(expected_wheel)
    cmake_sha = sha256(expected_cmake)
    if (wheel_sha != CMAKE_WHEEL_SHA256 or toolchain.get('cmake_wheel') != str(expected_wheel) or
            toolchain.get('cmake_wheel_sha256') != wheel_sha or
            toolchain.get('cmake_sha256') != cmake_sha):
        raise SafetyError('Q4-isolated CMake wheel/binary provenance changed since preparation')
    expected_nvcc = Path('/usr/local/cuda-12.9/bin/nvcc')
    if toolchain.get('cmake') != str(expected_cmake) or toolchain.get('nvcc') != str(expected_nvcc):
        raise SafetyError('Q4_K_S toolchain is not bound to the pinned isolated CMake and existing CUDA 12.9')
    cmake_version = command_output([str(expected_cmake), '--version']).splitlines()[0].strip()
    nvcc_version = command_output([str(expected_nvcc), '--version']).splitlines()[-1].strip()
    if (cmake_version != f'cmake version {CMAKE_VERSION}' or toolchain.get('cmake_version') != cmake_version or
            toolchain.get('nvcc_version') != nvcc_version):
        raise SafetyError('Q4_K_S pinned build-tool versions changed since preparation')
    cache = source / 'build-orca-q4ks/CMakeCache.txt'
    _assert_beneath(root, cache, 'Q4 CMake cache')
    if not cache.is_file() or cache.is_symlink():
        raise SafetyError('Q4_K_S CUDA/native-expert CMake build cache is missing')
    cache_text = cache.read_text(errors='replace')
    if f'CMAKE_COMMAND:INTERNAL={expected_cmake}' not in cache_text:
        raise SafetyError('Q4_K_S CMake cache was not configured with its isolated pinned CMake')
    for contract in CMAKE_CACHE_CONTRACT.splitlines():
        if contract not in cache_text:
            raise SafetyError(f'Q4_K_S build is missing required CMake option: {contract}')

    engine = source / 'engine/strata'
    _assert_beneath(root, engine, 'Q4 engine')
    runtime = {item.get('path'): item for item in manifest.get('runtime_assets', [])}
    engine_record = runtime.get(str(engine))
    check_record(engine, engine_record, sealed=True)
    if sha256(engine) != engine_record['sha256']:
        raise SafetyError('Q4_K_S native runtime hash changed')
    binaries = {item.get('path'): item for item in manifest.get('test_binaries', [])}
    for name in PARITY_TESTS:
        binary = source / 'build-orca-q4ks' / name
        _assert_beneath(root, binary, 'Q4 parity binary')
        item = binaries.get(str(binary))
        check_record(binary, item, sealed=True)
        if sha256(binary) != item['sha256']:
            raise SafetyError(f'Q4_K_S parity binary changed: {name}')

    model_dir = model_root(root)
    _assert_beneath(root, model_dir, 'Q4 model directory')
    if model_dir.is_symlink():
        raise SafetyError('Q4_K_S model directory is a symlink')
    asset_records = {item.get('path'): item for item in manifest.get('assets', [])}
    paths = [model_dir / name for name in ASSETS]
    if len(asset_records) != len(paths):
        raise SafetyError('Q4_K_S shard manifest is incomplete')
    for path in paths:
        size, expected = ASSETS[path.name]
        check_record(path, asset_records.get(str(path)), expected, sealed=True)
        if asset_records[str(path)].get('bytes') != size:
            raise SafetyError(f'Q4_K_S shard size differs from the publisher pin: {path.name}')

    pack_records = {item.get('path'): item for item in manifest.get('pack_assets', [])}
    required = ['dense.bin', 'index.txt', 'native_experts.txt', 'compat-bf16.json',
                'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja', 'tokenizer/vocab.json',
                'tokenizer/merges.txt', 'tokenizer/token_type.json']
    for relative in required:
        if str(pack / relative) not in pack_records:
            raise SafetyError(f'Q4_K_S compatibility pack record missing: {relative}')
    for path in pack.rglob('experts.bin'):
        raise SafetyError(f'Q4_K_S pack contains a forbidden expert override: {path}')
    for name, item in pack_records.items():
        path = Path(name)
        _assert_beneath(root, path, 'pack manifest entry')
        if not path.resolve().is_relative_to(pack.resolve()):
            raise SafetyError('Q4_K_S pack manifest points outside its separate profile')
        if path.name == 'experts.bin':
            raise SafetyError('Q4_K_S pack manifest contains a forbidden experts.bin override')
        check_record(path, item, sealed=True)

    base_actual = command_output(['git', '-c', f'safe.directory={base_source}', '-C', str(base_source),
                                  'rev-parse', 'HEAD']).strip()
    if base_actual != BASE_SOURCE_COMMIT:
        raise SafetyError('Original Flash-Next MTP source pin changed')
    if command_output(['git', '-c', f'safe.directory={base_source}', '-C', str(base_source),
                       'diff', '--name-only', 'HEAD']).strip():
        raise SafetyError('Original Flash-Next MTP source has tracked changes')
    mtp_script = base_source / 'tools/mtp_fetch.py'
    _assert_beneath(root, mtp_script, 'Flash-Next MTP pin script')
    if mtp_script.is_symlink() or not mtp_script.is_file():
        raise SafetyError('Flash-Next MTP pin script is missing or linked')
    mtp_pins = {'__name__': 'q4ks_readiness_mtp_pins', '__file__': str(mtp_script)}
    exec(compile(mtp_script.read_text(), str(mtp_script), 'exec'), mtp_pins)
    expected_mtp = mtp_pins['SHA256']

    python = source / '.venv/bin/python'
    _assert_tool_path(root, python, 'Q4 private Python link')
    expert_profile = base_source / 'data/expert-profile.bin'
    _assert_beneath(root, expert_profile, 'shared expert profile')
    mtp = root / 'data/mtp/rt/experts.bin'
    _assert_beneath(root, mtp, 'original MTP runtime')
    if not python.is_file() or expert_profile.is_symlink() or not expert_profile.is_file():
        raise SafetyError('Q4_K_S private Python/shared expert profile is missing or linked')
    check_record(expert_profile, manifest.get('expert_profile'), hash_file=True)
    check_record(mtp, manifest.get('mtp_runtime'), hash_file=False)
    tensor_records = {Path(item.get('path', '')).name.removesuffix('.bin'): item
                      for item in manifest.get('mtp_tensors', [])}
    if set(tensor_records) != set(expected_mtp):
        raise SafetyError('Original Flash-Next MTP tensor verification records are incomplete or foreign')
    mtp_root = root / 'data/mtp'
    _assert_beneath(root, mtp_root, 'original MTP data directory')
    for name, expected_sha in expected_mtp.items():
        item = tensor_records[name]
        tensor = root / 'data/mtp/tensors' / (name + '.bin')
        _assert_beneath(root, tensor, 'original MTP tensor')
        if item.get('sha256') != expected_sha:
            raise SafetyError(f'Q4_K_S MTP tensor record differs from the original draft pin: {name}')
        check_record(tensor, item, expected_sha)
    if '--vision' in manifest.get('config_args', []):
        raise SafetyError('Q4_K_S prepared profile must remain text-only')
    return manifest, source, pr, paths, engine_record


def _check_parity(root, manifest, paths, engine_record):
    parity_path = profile_root(root) / 'parity.json'
    _assert_beneath(root, parity_path, 'parity evidence')
    parity = _load_json(parity_path,
                        'Q4_K_S parity is missing; run --verify-orca-q4ks-parity against all real shards first.')
    pinned_assets = {name: sha for name, (_, sha) in ASSETS.items()}
    if (parity.get('profile') != PROFILE or parity.get('source_commit') != SOURCE_COMMIT or
            parity.get('asset_sha256') != pinned_assets or parity.get('runtime_sha256') != engine_record['sha256']):
        raise SafetyError('Q4_K_S parity evidence does not match the prepared source/runtime/shards')
    tests = parity.get('tests', {})
    for name in PARITY_TESTS:
        if tests.get(name, {}).get('passed') is not True:
            raise SafetyError(f'Q4_K_S required real-shard parity test missing/failed: {name}')
    return parity


def _validate_report(root, report, config_sha):
    manifest, source, pr, paths, engine_record = _prepared(root)
    pinned_assets = {name: sha for name, (_, sha) in ASSETS.items()}
    required_checks = ('health', 'arithmetic', 'strict_json', 'tool_call', 'memory_recorded')
    if (report.get('profile') != PROFILE or report.get('model_id') != MODEL_ID or
            report.get('source_commit') != SOURCE_COMMIT or report.get('runtime_version') != RUNTIME_VERSION or
            report.get('runtime_sha256') != engine_record.get('sha256') or
            report.get('asset_sha256') != pinned_assets or report.get('config_sha256') != config_sha or
            report.get('context') != 32768 or report.get('parallel') != 1 or report.get('text_only') is not True):
        raise SafetyError('Q4_K_S 32K validation report does not match the pinned single-slot text-only profile')
    checks = report.get('checks', {})
    if any(checks.get(name) is not True for name in required_checks):
        raise SafetyError('Q4_K_S 32K validation report is missing a passing functional/memory gate')
    memory = report.get('memory_peak_mib')
    if type(memory) is not int or memory <= 0:
        raise SafetyError('Q4_K_S validation must record a positive measured memory peak')
    mtp = report.get('mtp')
    drafted = mtp.get('drafted_tokens') if isinstance(mtp, dict) else None
    accepted = mtp.get('accepted_tokens') if isinstance(mtp, dict) else None
    if (type(drafted) is not int or drafted <= 0 or type(accepted) is not int or
            accepted < 0 or accepted > drafted):
        raise SafetyError('Q4_K_S validation must measure original Flash-Next MTP drafted/accepted tokens')
    _check_parity(root, manifest, paths, engine_record)


def _verified_iq3_projector(root, *, hash_file):
    root = Path(root)
    profile_dir = root / 'profiles' / IQ3XXS_PROFILE
    source_dir = root / 'data/models' / IQ3XXS_PROFILE
    manifest_path = profile_dir / 'prepared.json'
    projector = source_dir / VISION_PROJECTOR_FILENAME
    for path, label in ((profile_dir, 'IQ3_XXS profile directory'),
                        (source_dir, 'IQ3_XXS projector directory'),
                        (manifest_path, 'IQ3_XXS preparation manifest'),
                        (projector, 'IQ3_XXS projector')):
        _assert_beneath(root, path, label)
    if profile_dir.is_symlink() or source_dir.is_symlink() or manifest_path.is_symlink():
        raise SafetyError('IQ3_XXS projector provenance paths must not be symlinked')
    source_manifest = _load_json(manifest_path, 'A verified same-revision IQ3_XXS projector is required')
    if (source_manifest.get('profile') != IQ3XXS_PROFILE or
            source_manifest.get('model_repository') != IQ3XXS_REPOSITORY or
            source_manifest.get('model_revision') != IQ3XXS_REVISION or
            source_manifest.get('source_commit') != IQ3XXS_SOURCE_COMMIT or
            IQ3XXS_REVISION != REVISION or IQ3XXS_REPOSITORY != REPOSITORY):
        raise SafetyError('IQ3_XXS projector provenance is not pinned to the Q4_K_S model revision')
    source_assets = source_manifest.get('assets')
    if not isinstance(source_assets, list) or any(not isinstance(item, dict) for item in source_assets):
        raise SafetyError('IQ3_XXS projector preparation asset list is malformed')
    matches = [item for item in source_assets if item.get('path') == str(projector)]
    expected_bytes, expected_sha = IQ3XXS_ASSETS[VISION_PROJECTOR_FILENAME]
    if (len(matches) != 1 or expected_bytes != VISION_PROJECTOR_BYTES or
            expected_sha != VISION_PROJECTOR_SHA256):
        raise SafetyError('IQ3_XXS projector pin is missing or inconsistent')
    saved = matches[0]
    try:
        info = projector.lstat()
    except OSError as exc:
        raise SafetyError(f'IQ3_XXS projector is missing: {projector}') from exc
    if (not stat.S_ISREG(info.st_mode) or saved.get('bytes') != expected_bytes or
            saved.get('sha256') != expected_sha or saved.get('bytes') != info.st_size or
            saved.get('mtime_ns') != info.st_mtime_ns or info.st_mode & 0o222):
        raise SafetyError(f'IQ3_XXS projector pin/read-only state changed: {projector}')
    if hash_file and sha256(projector) != expected_sha:
        raise SafetyError(f'IQ3_XXS projector pin mismatch: {projector}')
    return projector, saved


def _verified_vision_engine(root):
    source, runtime_manifest = runtimes.prepared_runtime(root, RUNTIME_VERSION)
    expected_source = pinned_runtime_source(root)
    if source.resolve() != expected_source.resolve():
        raise SafetyError('CPU vision must reuse the exact prepared Strata 0.1.39 runtime')
    engine = expected_source / 'engine/strata-vision'
    _assert_beneath(root, engine, 'prepared CPU vision engine')
    records = [item for item in runtime_manifest.get('runtime_assets', []) if item.get('path') == str(engine)]
    if len(records) != 1 or engine.is_symlink() or not engine.is_file():
        raise SafetyError('Prepared Strata 0.1.39 vision helper provenance is missing or linked')
    item = records[0]
    if item.get('bytes') != engine.stat().st_size or item.get('sha256') != sha256(engine):
        raise SafetyError('Prepared Strata 0.1.39 vision helper pin changed')
    return engine, {'path': str(engine), 'bytes': item.get('bytes'), 'sha256': item.get('sha256')}


def _validate_vision_sidecar(root, pr):
    sidecar_path = pr / 'vision-prepared.json'
    _assert_beneath(root, sidecar_path, 'CPU-vision provenance')
    if sidecar_path.is_symlink():
        raise SafetyError('Q4_K_S CPU-vision provenance must not be a symlink')
    sidecar = _load_json(sidecar_path, 'Q4_K_S CPU vision requires its vision-prepared provenance sidecar')
    engine, engine_pin = _verified_vision_engine(root)
    source_projector, source_record = _verified_iq3_projector(root, hash_file=False)
    target_projector = model_root(root) / VISION_PROJECTOR_FILENAME
    _assert_beneath(root, target_projector, 'Q4 CPU-vision projector')
    try:
        source_info = source_projector.lstat()
        target_info = target_projector.lstat()
    except OSError as exc:
        raise SafetyError('Q4_K_S CPU-vision projector hardlink is missing') from exc
    expected_fields = {'profile', 'model_id', 'model_repository', 'model_revision', 'source_commit',
                       'runtime_version', 'vision_engine', 'projector'}
    projector_fields = {'path', 'source_profile', 'source_path', 'source_repository',
                        'source_revision', 'bytes', 'sha256', 'source_asset_record', 'record'}
    projector = sidecar.get('projector')
    if (set(sidecar) != expected_fields or sidecar.get('profile') != PROFILE or
            sidecar.get('model_id') != MODEL_ID or sidecar.get('model_repository') != REPOSITORY or
            sidecar.get('model_revision') != REVISION or sidecar.get('source_commit') != SOURCE_COMMIT or
            sidecar.get('runtime_version') != RUNTIME_VERSION or sidecar.get('vision_engine') != engine_pin or
            not isinstance(projector, dict) or set(projector) != projector_fields or
            projector.get('path') != str(target_projector) or
            projector.get('source_profile') != IQ3XXS_PROFILE or
            projector.get('source_path') != str(source_projector) or
            projector.get('source_repository') != IQ3XXS_REPOSITORY or
            projector.get('source_revision') != IQ3XXS_REVISION or
            projector.get('bytes') != VISION_PROJECTOR_BYTES or
            projector.get('sha256') != VISION_PROJECTOR_SHA256 or
            projector.get('source_asset_record') != source_record):
        raise SafetyError('Q4_K_S CPU vision provenance does not match the pinned model/projector/runtime')
    if (not stat.S_ISREG(source_info.st_mode) or not stat.S_ISREG(target_info.st_mode) or
            source_info.st_dev != target_info.st_dev or source_info.st_ino != target_info.st_ino or
            target_info.st_mode & 0o222):
        raise SafetyError('Q4_K_S projector is not the verified read-only IQ3_XXS hardlink')
    check_record(target_projector, projector.get('record'), VISION_PROJECTOR_SHA256,
                 hash_file=True, sealed=True)
    return engine


def _native_context_evidence(root, pr, config_path):
    validation_path = pr / 'validation-32k.json'
    original_config_path = pr / 'context-config-before.json'
    migration_path = pr / 'native-context-migration.json'
    for path, label in ((validation_path, '32K validation evidence'),
                        (original_config_path, 'initial config evidence'),
                        (migration_path, 'native-context migration evidence')):
        _assert_beneath(root, path, label)
        if path.is_symlink():
            raise SafetyError(f'Q4_K_S {label} must not be a symlink')
    if not original_config_path.is_file():
        raise SafetyError('Q4_K_S native context requires recorded 32K validation and preserved initial-config evidence')
    original_config = _load_json(original_config_path, 'Q4_K_S initial config evidence is missing')
    original_sha = hashlib.sha256(original_config_path.read_bytes()).hexdigest()
    validation = _load_json(validation_path, 'Q4_K_S native context requires recorded 32K validation first')
    _validate_report(root, validation, original_sha)
    if (validation.get('passed') is not True or not isinstance(validation.get('recorded_server_identity'), dict) or
            not validation.get('recorded_server_config')):
        raise SafetyError('Q4_K_S native context requires a passing identity-bound 32K validation')
    migration = _load_json(migration_path, 'Q4_K_S native-context migration record is missing')
    migration_fields = {'profile', 'source_commit', 'context', 'initial_config_sha256',
                        'validation_sha256', 'current_config_sha256', 'recorded_at'}
    if (set(migration) != migration_fields or migration.get('profile') != PROFILE or
            migration.get('validation_sha256') != sha256(validation_path) or
            migration.get('initial_config_sha256') != original_sha or
            migration.get('current_config_sha256') != sha256(config_path) or
            migration.get('source_commit') != SOURCE_COMMIT or migration.get('context') != 262144 or
            not isinstance(migration.get('recorded_at'), str) or
            original_config != make_config(root, context=32768)):
        raise SafetyError('Q4_K_S native-context migration evidence is stale or inconsistent')
    return migration


def ready_q4ks(root):
    """Read-only readiness; never consults or updates the generic runtime-selection manager."""
    manifest, source, pr, paths, engine_record = _prepared(root)
    _check_parity(root, manifest, paths, engine_record)
    config_path = pr / 'config.json'
    _assert_beneath(root, config_path, 'profile config')
    if config_path.is_symlink():
        raise SafetyError('Q4_K_S profile config must not be a symlink')
    cfg = _load_json(config_path, 'Q4_K_S profile config is missing')
    args = cfg.get('args')
    context = None
    if isinstance(args, list) and args.count('--max-context') == 1:
        try:
            context = int(args[args.index('--max-context') + 1])
        except (ValueError, IndexError):
            pass
    migration_path = pr / 'native-context-migration.json'
    vision_sidecar = pr / 'vision-prepared.json'
    _assert_beneath(root, migration_path, 'native-context migration evidence')
    if migration_path.is_symlink():
        raise SafetyError('Q4_K_S native-context migration evidence must not be a symlink')
    if context == 32768:
        _assert_beneath(root, vision_sidecar, 'CPU-vision provenance')
        if vision_sidecar.is_symlink():
            raise SafetyError('Q4_K_S CPU-vision provenance must not be a symlink')
        expected = make_config(root, context=32768)
        if migration_path.exists():
            raise SafetyError('Q4_K_S 32K profile conflicts with an existing native-context migration record')
        if vision_sidecar.exists():
            raise SafetyError('Q4_K_S CPU vision is only permitted after native-context migration')
    elif context == 262144:
        _native_context_evidence(root, pr, config_path)
        expected_text = make_config(root, context=262144)
        expected_vision = make_config(root, context=262144, vision=True)
        if cfg == expected_vision:
            _validate_vision_sidecar(root, pr)
            expected = expected_vision
        else:
            expected = expected_text
    else:
        raise SafetyError('Q4_K_S context must remain at 32K until the gated native migration')

    if cfg != expected:
        raise SafetyError('Q4_K_S profile config differs from its exact pinned text-only or CPU-vision preset')
    return source, config_path, cfg


def _assert_live_q4ks(root):
    state_path = strata_state() / 'server.json'
    state = _load_json(state_path, 'Record Q4_K_S gates only while its owned server is running')
    identity = state.get('identity')
    if state.get('profile') != PROFILE or state.get('runtime_version') != RUNTIME_VERSION or \
            not same_process(identity, proc(identity.get('pid')) if identity else None):
        raise SafetyError('Live Strata identity/profile is not the pinned Q4_K_S server')
    if state.get('source_commit') != SOURCE_COMMIT:
        raise SafetyError('Live Q4_K_S server source pin differs')
    run_config_path = Path(state.get('config', ''))
    try:
        live_cfg = json.loads(run_config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Cannot verify live Q4_K_S config: {exc}') from exc
    expected = make_config(root, context=32768)
    keys = ('exe', 'cwd', 'args', 'tokenizer', 'gpu', 'layer_split', 'model_name', 'aliases', 'sampling')
    if (any(live_cfg.get(key) != expected.get(key) for key in keys) or
            live_cfg.get('vision') is not None or live_cfg.get('parallel') not in (None, 1) or
            state.get('parallel', 1) != 1 or state.get('batch_groups', 1) != 1):
        raise SafetyError('Live server is not the exact 32K single-slot text-only Q4_K_S profile')
    expected_executables = {str((source_root(root) / '.venv/bin/python').resolve()),
                            str((source_root(root) / 'engine/strata').resolve())}
    if set(state.get('allowed_executables', [])) != expected_executables:
        raise SafetyError('Live Q4_K_S ownership allowlist differs or includes a vision helper')
    return state


def record_32k_validation(root, report_path):
    """Bind an operator's measured first-run report to the live identity-verified Q4 server."""
    _, config_path, _ = ready_q4ks(root)
    live_state = _assert_live_q4ks(root)
    report = _load_json(report_path, 'Q4_K_S validation report file is missing')
    config_sha = sha256(config_path)
    _validate_report(root, report, config_sha)
    target = profile_root(root) / 'validation-32k.json'
    if target.exists():
        raise SafetyError('Q4_K_S 32K validation is already recorded; inspect it rather than overwrite')
    report = dict(report, passed=True, recorded_server_identity=live_state['identity'],
                  recorded_server_config=live_state['config'],
                  recorded_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic_json(target, report)
    print(f'Q4_K_S 32K validation recorded; MTP acceptance measured. No context/runtime selection changed.')


def _q4ks_lock(root):
    pr = profile_root(root)
    lock_path = pr / 'prepare.lock'
    _assert_beneath(root, lock_path, 'preparation lock')
    if lock_path.is_symlink() or not lock_path.is_file():
        raise SafetyError('Q4_K_S preparation lock/provenance is missing or linked')
    return lock_path


def _update_native_config(root, config_path, migration_path, config_bytes, migration_bytes, updated):
    try:
        atomic_json(config_path, updated)
        migration = json.loads(migration_bytes)
        migration['current_config_sha256'] = sha256(config_path)
        atomic_json(migration_path, migration)
        ready_q4ks(root)
    except Exception:
        atomic_bytes(config_path, config_bytes)
        atomic_bytes(migration_path, migration_bytes)
        raise


def _prepare_vision_assets(root, pr):
    source_projector, source_record = _verified_iq3_projector(root, hash_file=True)
    engine, engine_pin = _verified_vision_engine(root)
    _assert_beneath(root, engine, 'prepared CPU vision engine')
    try:
        engine_info = engine.lstat()
    except OSError as exc:
        raise SafetyError('Prepared Strata 0.1.39 CPU vision helper is missing') from exc
    if (not stat.S_ISREG(engine_info.st_mode) or engine_info.st_size != engine_pin['bytes'] or
            sha256(engine) != engine_pin['sha256']):
        raise SafetyError('Prepared Strata 0.1.39 CPU vision helper changed')

    target = model_root(root) / VISION_PROJECTOR_FILENAME
    sidecar_path = pr / 'vision-prepared.json'
    _assert_beneath(root, target, 'Q4 CPU-vision projector')
    _assert_beneath(root, sidecar_path, 'CPU-vision provenance')
    if sidecar_path.is_symlink():
        raise SafetyError('Q4_K_S CPU-vision provenance must not be a symlink')
    if sidecar_path.exists():
        _validate_vision_sidecar(root, pr)
        return
    if model_root(root).is_symlink():
        raise SafetyError('Q4_K_S model directory must not be a symlink')
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise SafetyError(f'Unproven Q4 projector destination exists; refusing overwrite: {target}')
    if source_projector.stat().st_dev != model_root(root).stat().st_dev:
        raise SafetyError('IQ3_XXS projector and Q4_K_S model directory are on different filesystems; hardlink refused')
    try:
        os.link(source_projector, target, follow_symlinks=False)
    except OSError as exc:
        raise SafetyError(f'Could not create safe read-only IQ3_XXS projector hardlink: {exc}') from exc
    try:
        source_info, target_info = source_projector.stat(), target.lstat()
        if (not stat.S_ISREG(target_info.st_mode) or source_info.st_dev != target_info.st_dev or
                source_info.st_ino != target_info.st_ino or target_info.st_mode & 0o222):
            raise SafetyError('Created Q4 projector is not the verified read-only IQ3_XXS hardlink')
        engine_record = engine_pin
        atomic_json(sidecar_path, {
            'profile': PROFILE, 'model_id': MODEL_ID, 'model_repository': REPOSITORY,
            'model_revision': REVISION, 'source_commit': SOURCE_COMMIT,
            'runtime_version': RUNTIME_VERSION, 'vision_engine': engine_record,
            'projector': {
                'path': str(target), 'source_profile': IQ3XXS_PROFILE,
                'source_path': str(source_projector), 'source_repository': IQ3XXS_REPOSITORY,
                'source_revision': IQ3XXS_REVISION, 'bytes': VISION_PROJECTOR_BYTES,
                'sha256': VISION_PROJECTOR_SHA256, 'source_asset_record': source_record,
                'record': record(target, expected_sha=VISION_PROJECTOR_SHA256),
            },
        })
        _validate_vision_sidecar(root, pr)
    except Exception:
        # Remove only the hardlink created here, never the retained IQ3_XXS source.
        try:
            current = target.lstat()
            original = source_projector.stat()
            if (stat.S_ISREG(current.st_mode) and current.st_dev == original.st_dev and
                    current.st_ino == original.st_ino and not sidecar_path.exists()):
                target.unlink()
        except OSError:
            pass
        raise


def configure_q4ks_vision(root):
    """Configure future Q4 launches for CPU vision after native-context readiness."""
    pr = profile_root(root)
    lock_path = _q4ks_lock(root)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        _, config_path, cfg = ready_q4ks(root)
        native_text = make_config(root, context=262144)
        native_vision = make_config(root, context=262144, vision=True)
        if cfg == native_vision:
            print('Q4_K_S future profile already has CPU vision configured; no active server changed.')
            return
        if cfg != native_text:
            raise SafetyError('Q4_K_S CPU vision requires an existing passing native-context migration')
        migration_path = pr / 'native-context-migration.json'
        migration_bytes = migration_path.read_bytes()
        config_bytes = config_path.read_bytes()
        _prepare_vision_assets(root, pr)
        _update_native_config(root, config_path, migration_path, config_bytes, migration_bytes, native_vision)
    print('Q4_K_S future profile configured for CPU vision at native 262144. Active private run-config/server and generic runtime selection untouched; vision is not live-validated.')


def rollback_q4ks_vision(root):
    """Explicitly restore future launches to 262K text-only without removing vision provenance."""
    pr = profile_root(root)
    lock_path = _q4ks_lock(root)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        manifest, source, checked_pr, paths, engine_record = _prepared(root)
        _check_parity(root, manifest, paths, engine_record)
        config_path = checked_pr / 'config.json'
        _assert_beneath(root, config_path, 'profile config')
        if config_path.is_symlink():
            raise SafetyError('Q4_K_S profile config must not be a symlink')
        cfg = _load_json(config_path, 'Q4_K_S profile config is missing')
        _native_context_evidence(root, checked_pr, config_path)
        native_text = make_config(root, context=262144)
        native_vision = make_config(root, context=262144, vision=True)
        if cfg == native_text:
            print('Q4_K_S future profile is already 262K text-only; no active server changed.')
            return
        if cfg != native_vision:
            raise SafetyError('Vision rollback requires the exact migrated 262K CPU-vision config')
        migration_path = checked_pr / 'native-context-migration.json'
        migration_bytes = migration_path.read_bytes()
        config_bytes = config_path.read_bytes()
        _update_native_config(root, config_path, migration_path, config_bytes, migration_bytes, native_text)
    print('Q4_K_S future profile rolled back to 262K text-only. Projector provenance retained; active server and generic runtime selection untouched.')


def configure_native_context(root):
    """Explicitly migrate future Q4 launches only after recorded 32K functional/memory/MTP gates."""
    pr = profile_root(root)
    lock_path = _q4ks_lock(root)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(lock_fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        _, config_path, cfg = ready_q4ks(root)
        if '--max-context' not in cfg['args'] or cfg['args'][cfg['args'].index('--max-context') + 1] != '32768':
            raise SafetyError('Q4_K_S is not in its initial 32K state')
        validation_path = pr / 'validation-32k.json'
        validation = _load_json(validation_path, 'Q4_K_S native migration requires 32K validation first')
        config_sha = sha256(config_path)
        _validate_report(root, validation, config_sha)
        if validation.get('passed') is not True:
            raise SafetyError('Q4_K_S 32K validation did not pass')
        backup = pr / 'context-config-before.json'
        migration_path = pr / 'native-context-migration.json'
        if migration_path.exists():
            raise SafetyError('Existing Q4_K_S native migration record is unexpected; inspect rather than overwrite')
        config_bytes = config_path.read_bytes()
        if backup.exists():
            if backup.read_bytes() != config_bytes:
                raise SafetyError('Existing Q4_K_S 32K config backup differs; refusing migration')
        else:
            atomic_bytes(backup, config_bytes)
        migrated = make_config(root, context=262144)
        try:
            atomic_json(config_path, migrated)
            atomic_json(migration_path, {
                'profile': PROFILE, 'source_commit': SOURCE_COMMIT, 'context': 262144,
                'initial_config_sha256': config_sha, 'validation_sha256': sha256(validation_path),
                'current_config_sha256': sha256(config_path),
                'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            ready_q4ks(root)
        except Exception:
            atomic_bytes(config_path, config_bytes)
            migration_path.unlink(missing_ok=True)
            raise
    print('Q4_K_S future profile migrated to native 262144 / INT8 KV / 32768 resident. Active server and generic runtime selection untouched.')
