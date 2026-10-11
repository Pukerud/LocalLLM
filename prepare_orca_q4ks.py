#!/usr/bin/env python3
"""Explicit, side-by-side Orca Q4_K_S download/build. Never starts inference or changes services."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from engine_safety import (SafetyError, active_engine, docker_empty, launch_gate, owner_home,
                           proc, same_process, signal_identity, strata_state)
import hosting_lifecycle as hosting
from orca_q4ks_assets import (ASSETS, BUILD_OPTIONS, CMAKE_VERSION, CMAKE_WHEEL_BYTES, CMAKE_WHEEL_FILENAME,
                              CMAKE_WHEEL_SHA256, CMAKE_WHEEL_URL, MODEL_ID, PARITY_TESTS, PROFILE,
                              REPOSITORY, REVISION, BASE_SOURCE_COMMIT, RUNTIME_VERSION, SOURCE_COMMIT,
                              TOTAL_BYTES, cmake_binary, cmake_root, cmake_wheel, model_root, pack_root,
                              pinned_runtime_source, profile_root, source_root, toolchain_root)
from orca_q4ks_profile import (CUDA_LIB_DIRS, CMAKE_CACHE_CONTRACT, _assert_tool_path, _prepared,
                               atomic_json, make_config, record)
from prepare_orca import PrivateRedirect, hf_token
from strata_runtime import LLAMA_PIN, prepared_runtime

BASE = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
DEST = model_root(BASE)
PACK = pack_root(BASE)
PROFILE_ROOT = profile_root(BASE)
SOURCE = source_root(BASE)
BUILD = SOURCE / 'build-orca-q4ks'
PREPARED = PROFILE_ROOT / 'prepared.json'
USER_AGENT = 'LocalLLM-install-only/1'
DISK_MARGIN = 15 * 1024**3
PARITY_STOP_GRACE_SECONDS = 10


def _require_unprivileged(operation):
    if os.geteuid() == 0:
        raise SafetyError(f'{operation} must run as the normal operator; do not use sudo')


def _assert_data_path(path, label):
    base = BASE.resolve()
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(base):
        raise SafetyError(f'{label} resolves outside STRATA_DATA_ROOT: {candidate}')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _regular_file(path, *, missing_ok=True):
    try:
        st = Path(path).lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    if not stat.S_ISREG(st.st_mode):
        raise SafetyError(f'Expected a regular Q4_K_S file, refusing symlink/special path: {path}')
    return st


def download(name, size, sha):
    """Single-stream resumable download; completed mismatches are never overwritten."""
    _require_unprivileged('Q4_K_S downloads')
    if DEST.is_symlink():
        raise SafetyError('Q4_K_S model directory is a symlink')
    _assert_data_path(DEST, 'Q4_K_S model directory')
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / name
    existing = _regular_file(path)
    if existing:
        if existing.st_size != size or digest(path) != sha:
            raise SafetyError(f'Existing Q4_K_S asset differs from its pin; not overwritten: {path}')
        print(f'Already verified: {name}', flush=True)
        return path
    part = path.with_name(path.name + '.part')
    part_stat = _regular_file(part)
    if part_stat and part_stat.st_size > size:
        raise SafetyError(f'Oversized Q4_K_S partial; not overwritten: {part}')
    url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
    for attempt in range(10):
        part_stat = _regular_file(part)
        offset = part_stat.st_size if part_stat else 0
        if offset == size:
            break
        headers = {'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'}
        token = hf_token()
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url + f'?download=true&nonce={time.time_ns()}', headers=headers)
        try:
            with urllib.request.build_opener(PrivateRedirect()).open(request, timeout=60) as response:
                if offset:
                    content_range = response.headers.get('Content-Range', '')
                    if (response.status != 206 or not content_range.startswith(f'bytes {offset}-') or
                            not content_range.endswith(f'/{size}')):
                        raise SafetyError('Server did not honor the pinned Q4_K_S resume range; partial retained')
                elif response.status not in (200, 206):
                    raise SafetyError(f'Unexpected Q4_K_S download response: {response.status}')
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                with os.fdopen(fd, 'ab') as out:
                    count, report = offset, time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        if count + len(block) > size:
                            raise SafetyError('Q4_K_S download exceeds publisher-pinned size; partial retained')
                        out.write(block)
                        count += len(block)
                        if time.monotonic() - report > 15:
                            print(f'{name}: {count / size:.1%} ({count / 1e9:.2f}/{size / 1e9:.2f} GB)', flush=True)
                            report = time.monotonic()
                    out.flush()
                    os.fsync(out.fileno())
                if count != size:
                    raise OSError(f'Truncated Q4_K_S download: {count}/{size}')
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise SafetyError('Hugging Face access is gated. Accept access and configure HF_TOKEN or the owner\'s '
                                  'private token file; credentials are not logged or saved.') from None
            if attempt == 9:
                raise SafetyError(f'Pinned Q4_K_S download failed (HTTP {exc.code}); partial retained') from None
            print(f'Download HTTP {exc.code}; resuming in 5s.', flush=True)
            time.sleep(5)
        except (OSError, TimeoutError) as exc:
            if attempt == 9:
                raise
            print(f'Download interrupted ({type(exc).__name__}); resuming in 5s.', flush=True)
            time.sleep(5)
    part_stat = _regular_file(part, missing_ok=False)
    if part_stat.st_size != size:
        raise SafetyError(f'Incomplete Q4_K_S download: {name}')
    print(f'Full-file SHA-256: {name}', flush=True)
    if digest(part) != sha:
        raise SafetyError(f'Q4_K_S checksum mismatch; partial retained, never promoted: {part}')
    try:
        os.link(part, path, follow_symlinks=False)
    except FileExistsError as exc:
        raise SafetyError(f'Q4_K_S destination appeared during download; not overwritten: {path}') from exc
    part.unlink()
    return path


def _file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    return {'device': st.st_dev, 'inode': st.st_ino, 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode}


def protected_snapshot():
    """Hash small shared manifests/configs and capture retained IQ3_XXS file identities."""
    paths = [BASE / 'prepared.json', BASE / 'runtime-selection.json',
             BASE / 'source-99f3dbd0b21d/strata-iq3_s.json',
             BASE / 'source-99f3dbd0b21d/data/expert-profile.bin',
             BASE / 'source-99f3dbd0b21d/engine/strata',
             BASE / 'source-99f3dbd0b21d/engine/strata-vision',
             BASE / 'runtimes/0.1.38/prepared.json', BASE / 'runtimes/0.1.39/prepared.json',
             BASE / 'runtimes/0.1.41/prepared.json',
             BASE / 'profiles/orca-iq3_xxs/prepared.json',
             BASE / 'profiles/orca-iq3_xxs/config.json']
    for version in ('0.1.38', '0.1.39', '0.1.41'):
        paths.extend([BASE / 'runtimes' / version / 'source/engine/strata',
                      BASE / 'runtimes' / version / 'source/engine/strata-vision'])
    hashed = {str(path): digest(path) if path.is_file() else None for path in paths}
    retained_roots = [BASE / 'data/models/orca-iq3_xxs', BASE / 'data/packs/orca-iq3_xxs']
    identities = {}
    for retained in retained_roots:
        if retained.is_dir():
            identities.update({str(path): _file_fingerprint(path) for path in sorted(retained.rglob('*'))
                               if path.is_file()})
    return {'hashes': hashed, 'iq3_xxs_file_identities': identities}


def assert_protected(before):
    for name, expected in before['hashes'].items():
        path = Path(name)
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            raise SafetyError(f'Q4_K_S preparation changed protected generic/IQ3_XXS file: {name}')
    now = {}
    for retained in [BASE / 'data/models/orca-iq3_xxs', BASE / 'data/packs/orca-iq3_xxs']:
        if retained.is_dir():
            now.update({str(path): _file_fingerprint(path) for path in sorted(retained.rglob('*')) if path.is_file()})
    if now != before['iq3_xxs_file_identities']:
        raise SafetyError('Q4_K_S preparation changed retained IQ3_XXS asset identities')


def live_strata_identity():
    state_file = strata_state() / 'server.json'
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text())
        identity = state['identity']
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SafetyError(f'Existing Strata ownership state is incoherent; preparation will not repair it: {exc}') from exc
    if not same_process(identity, proc(identity.get('pid'))):
        raise SafetyError('Existing Strata identity is not coherent; preparation leaves it untouched')
    return identity


def assert_live_strata_unchanged(before):
    after = live_strata_identity()
    if before is None:
        if after is not None:
            raise SafetyError('A Strata instance appeared during Q4_K_S preparation; it was not stopped')
    elif not same_process(before, after):
        raise SafetyError('The original Strata instance changed during Q4_K_S preparation; it was not restarted')


def _verify_original_mtp(base_source):
    _assert_data_path(base_source, 'Flash-Next MTP source')
    if base_source.is_symlink():
        raise SafetyError('Flash-Next MTP source must not be a symlink')
    actual = subprocess.check_output(['git', '-c', f'safe.directory={base_source}', '-C', str(base_source),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != BASE_SOURCE_COMMIT:
        raise SafetyError('Original Flash-Next MTP source pin changed')
    if subprocess.check_output(['git', '-c', f'safe.directory={base_source}', '-C', str(base_source),
                                'diff', '--name-only', 'HEAD'], text=True).strip():
        raise SafetyError('Original Flash-Next MTP source has tracked changes')
    mtp_script = base_source / 'tools/mtp_fetch.py'
    _assert_data_path(mtp_script, 'Flash-Next MTP pin script')
    if mtp_script.is_symlink() or not mtp_script.is_file():
        raise SafetyError('Original Flash-Next MTP pin source is missing; Q4_K_S will not use another draft')
    pins = {'__name__': 'q4ks_mtp_offline_pins', '__file__': str(mtp_script)}
    exec(compile(mtp_script.read_text(), str(mtp_script), 'exec'), pins)
    expert_profile = base_source / 'data/expert-profile.bin'
    _assert_data_path(expert_profile, 'shared expert profile')
    if expert_profile.is_symlink() or not expert_profile.is_file():
        raise SafetyError('Shared Strata expert profile is missing or linked')
    records = []
    for name, expected in pins['SHA256'].items():
        path = BASE / 'data/mtp/tensors' / (name + '.bin')
        _assert_data_path(path, 'original MTP tensor')
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise SafetyError(f'Original pinned Flash-Next MTP tensor missing/changed: {name}')
        records.append(record(path))
    runtime = BASE / 'data/mtp/rt/experts.bin'
    _assert_data_path(runtime, 'original MTP runtime')
    if not runtime.is_file() or runtime.is_symlink():
        raise SafetyError('Original Flash-Next draft runtime is missing')
    return {'runtime': record(runtime), 'tensors': records, 'source_commit': BASE_SOURCE_COMMIT,
            'expert_profile': record(expert_profile)}


def _validate_runtime_environment(runtime_source):
    _assert_data_path(runtime_source, 'pinned .39 tool source')
    prepared_source, _ = prepared_runtime(BASE, RUNTIME_VERSION)
    if prepared_source.resolve() != Path(runtime_source).resolve():
        raise SafetyError('Q4_K_S must use the generic manager\'s exact pinned .39 source as a read-only tool base')
    actual = subprocess.check_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Exact side-by-side Strata 0.1.39 source/environment is required')
    if subprocess.check_output(['git', '-c', f'safe.directory={runtime_source}', '-C', str(runtime_source),
                                'diff', '--name-only', 'HEAD'], text=True).strip():
        raise SafetyError('Pinned Strata 0.1.39 source has tracked modifications')
    venv = runtime_source / '.venv'
    python = venv / 'bin/python'
    _assert_tool_path(BASE, python, 'pinned .39 Python')
    if not python.is_file():
        raise SafetyError('Prepare the pinned Strata 0.1.39 private Python environment first')
    return python, venv


def _prepare_source(runtime_source):
    _assert_data_path(SOURCE, 'Q4_K_S source directory')
    if SOURCE.is_symlink():
        raise SafetyError('Q4_K_S source directory is a symlink; refusing to clone through it')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                        'https://github.com/Niko1221/Strata.git', str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Q4_K_S side-by-side source differs from exact Strata 0.1.39 pin; not overwritten')
    if subprocess.check_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                'diff', '--name-only', 'HEAD'], text=True).strip():
        raise SafetyError('Q4_K_S source has tracked changes; inspect rather than overwrite')
    python, base_venv = _validate_runtime_environment(runtime_source)
    candidate_venv = SOURCE / '.venv'
    if candidate_venv.is_symlink():
        if candidate_venv.resolve() != base_venv.resolve():
            raise SafetyError('Q4_K_S source has an unexpected private-environment link')
    elif candidate_venv.exists():
        raise SafetyError('Q4_K_S source has a foreign private environment; not overwritten')
    else:
        candidate_venv.symlink_to(base_venv.resolve(), target_is_directory=True)
    pinned_llama = runtime_source / 'third_party/llama.cpp'
    _assert_data_path(pinned_llama, 'pinned .39 llama.cpp source')
    if not (pinned_llama / 'ggml/CMakeLists.txt').is_file() or not (pinned_llama / 'gguf-py').is_dir():
        raise SafetyError('Prepared .39 llama.cpp dependency is incomplete; Q4 will not fetch a replacement')
    llama_link = SOURCE / 'third_party/llama.cpp'
    if llama_link.is_symlink():
        if llama_link.resolve() != pinned_llama.resolve():
            raise SafetyError('Q4_K_S llama.cpp link differs from the exact .39 runtime dependency')
    elif llama_link.exists():
        raise SafetyError('Q4_K_S has a non-pinned llama.cpp path; not overwritten')
    else:
        llama_link.parent.mkdir(parents=True, exist_ok=True)
        llama_link.symlink_to(pinned_llama.resolve(), target_is_directory=True)
    if BUILD.exists() or BUILD.is_symlink():
        raise SafetyError(f'Interrupted Q4_K_S build directory exists; inspect it instead of overwriting: {BUILD}')
    return python, base_venv


def _cmake_version(binary):
    try:
        output = subprocess.check_output([str(binary), '--version'], text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'Q4-isolated CMake cannot run: {exc}') from exc
    version = output.splitlines()[0].strip() if output.splitlines() else ''
    if version != f'cmake version {CMAKE_VERSION}':
        raise SafetyError(f'Q4-isolated CMake must be exactly {CMAKE_VERSION}; got {version!r}')
    return version


def _ensure_cmake():
    """Install the pinned binary wheel only inside this Q4 runtime; never touch the shared .39 venv/system."""
    _require_unprivileged('Q4 CMake provisioning')
    tools = toolchain_root(BASE)
    root = cmake_root(BASE)
    wheel = cmake_wheel(BASE)
    part = wheel.with_name(wheel.name + '.part')
    _assert_data_path(tools, 'Q4 toolchain directory')
    _assert_data_path(root, 'Q4 CMake installation')
    _assert_data_path(wheel, 'Q4 CMake wheel')
    if tools.is_symlink() or root.is_symlink() or wheel.is_symlink() or part.is_symlink():
        raise SafetyError('Q4 CMake toolchain paths must not be symlinks')
    tools.mkdir(parents=True, exist_ok=True)

    if wheel.exists():
        info = _regular_file(wheel, missing_ok=False)
        if info.st_size != CMAKE_WHEEL_BYTES or digest(wheel) != CMAKE_WHEEL_SHA256:
            raise SafetyError('Existing Q4 CMake wheel differs from its PyPI pin; never overwritten')
    else:
        if part.exists():
            raise SafetyError('Interrupted Q4 CMake wheel download exists; inspect it rather than overwrite')
        request = urllib.request.Request(CMAKE_WHEEL_URL, headers={
            'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise SafetyError(f'Unexpected Q4 CMake wheel response: {response.status}')
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) != CMAKE_WHEEL_BYTES:
                    raise SafetyError('Q4 CMake wheel Content-Length differs from its publisher pin')
                fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                with os.fdopen(fd, 'wb') as output:
                    count = 0
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > CMAKE_WHEEL_BYTES:
                            raise SafetyError('Q4 CMake wheel exceeds its pinned size')
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            if count != CMAKE_WHEEL_BYTES or digest(part) != CMAKE_WHEEL_SHA256:
                raise SafetyError('Q4 CMake wheel size/SHA-256 differs from the PyPI pin')
            os.link(part, wheel, follow_symlinks=False)
            part.unlink()
            wheel.chmod(0o444)
        except BaseException:
            if part.exists() and not part.is_symlink():
                part.unlink()
            raise

    if not root.exists():
        staging = root.with_name(f'.{root.name}.extracting')
        if staging.exists() or staging.is_symlink():
            raise SafetyError('Interrupted Q4 CMake extraction exists; inspect rather than overwrite')
        staging.mkdir(mode=0o700)
        try:
            with zipfile.ZipFile(wheel) as archive:
                members = archive.infolist()
                if len(members) > 5000 or sum(item.file_size for item in members) > 256 * 1024**2:
                    raise SafetyError('Pinned Q4 CMake wheel has unexpected expansion limits')
                directories = []
                base = staging.resolve()
                for item in members:
                    name = item.filename
                    relative = PurePosixPath(name)
                    if (not name or name.startswith('/') or chr(92) in name or
                            any(part in ('', '.', '..') for part in relative.parts)):
                        raise SafetyError('Pinned Q4 CMake wheel contains an unsafe member path')
                    destination = staging.joinpath(*relative.parts)
                    if not destination.resolve().is_relative_to(base):
                        raise SafetyError('Pinned Q4 CMake wheel member escapes its isolated tool directory')
                    mode = (item.external_attr >> 16) & 0xffff
                    kind = stat.S_IFMT(mode)
                    if kind == stat.S_IFLNK or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise SafetyError('Pinned Q4 CMake wheel contains a symlink or special file')
                    permissions = stat.S_IMODE(mode) & 0o755
                    if item.is_dir() or kind == stat.S_IFDIR:
                        destination.mkdir(parents=True, exist_ok=True)
                        directories.append((destination, permissions or 0o755))
                        continue
                    if destination.exists() or destination.is_symlink():
                        raise SafetyError('Duplicate Q4 CMake wheel member; extraction refused')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, destination.open('xb') as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    destination.chmod(permissions or 0o644)
                for directory, permissions in reversed(directories):
                    directory.chmod(permissions)
            binary = staging / 'cmake/data/bin/cmake'
            if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
                raise SafetyError('Pinned CMake wheel did not provide an executable Linux CMake binary')
            _cmake_version(binary)
            os.rename(staging, root)
        except BaseException:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise

    binary = cmake_binary(BASE)
    _assert_data_path(binary, 'Q4 CMake executable')
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise SafetyError('Q4-isolated CMake binary is missing or linked')
    _cmake_version(binary)
    return binary


def _build(source_python, base_venv):
    _assert_data_path(BUILD, 'Q4_K_S build directory')
    _assert_tool_path(BASE, source_python, 'Q4 build Python')
    _assert_data_path(base_venv, 'Q4 build environment')
    nvcc = Path('/usr/local/cuda-12.9/bin/nvcc')
    if not nvcc.is_file():
        raise SafetyError('Existing CUDA 12.9 nvcc is required; no system installation is allowed')
    cmake = _ensure_cmake()
    _assert_data_path(cmake, 'isolated Q4 CMake')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               CUDA_HOME='/usr/local/cuda-12.9', STRATA_NVCC=str(nvcc),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PATH'] = str(cmake.parent) + ':' + str(base_venv / 'bin') + ':/usr/local/cuda-12.9/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIB_DIRS) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['STRATA_GGUF_PY'] = str(SOURCE / 'third_party/llama.cpp/gguf-py')
    setup_script = f"import setup; assert setup.LLAMA_CPP_COMMIT == {LLAMA_PIN!r}"
    subprocess.run([str(source_python), '-c', setup_script], cwd=SOURCE, env=env, check=True)
    llama = SOURCE / 'third_party/llama.cpp'
    if (not llama.is_symlink() or llama.resolve() !=
            (pinned_runtime_source(BASE) / 'third_party/llama.cpp').resolve() or
            not (llama / 'ggml/CMakeLists.txt').is_file()):
        raise SafetyError('Q4_K_S must use the already-prepared pinned .39 llama.cpp source without refetching')
    cmake_version = subprocess.check_output([str(cmake), '--version'], text=True).splitlines()[0].strip()
    nvcc_version = subprocess.check_output([str(nvcc), '--version'], text=True).splitlines()[-1].strip()
    definitions = [
        '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_NATIVE_EXPERTS=ON',
        '-DSTRATA_ORCA_Q4KS_MMQ=ON', '-DSTRATA_BUILD_TESTS=ON',
        '-DCMAKE_CUDA_ARCHITECTURES:STRING=86', f'-DCMAKE_CUDA_COMPILER={nvcc}',
        f'-DSTRATA_GGML_DIR={llama}',
    ]
    docker_empty()
    subprocess.run([str(cmake), '-S', str(SOURCE), '-B', str(BUILD), *definitions], env=env, check=True)
    subprocess.run([str(cmake), '--build', str(BUILD), '--target', 'strata', *PARITY_TESTS, '-j3'],
                   env=env, check=True)
    cache = BUILD / 'CMakeCache.txt'
    cache_text = cache.read_text(errors='replace')
    for contract in CMAKE_CACHE_CONTRACT.splitlines():
        if contract not in cache_text:
            raise SafetyError(f'Q4_K_S CMake cache does not contain required option: {contract}')
    engine_bin = BUILD / 'strata'
    if engine_bin.is_symlink() or not engine_bin.is_file():
        raise SafetyError('Q4_K_S build did not produce a regular engine binary')
    engine_dir = SOURCE / 'engine'
    if engine_dir.is_symlink():
        raise SafetyError('Q4_K_S engine output directory is a symlink')
    engine_dir.mkdir(exist_ok=True)
    if not engine_dir.resolve().is_relative_to(SOURCE.resolve()):
        raise SafetyError('Q4_K_S engine output directory resolves outside its pinned source tree')
    engine = engine_dir / 'strata'
    if engine.is_symlink() or engine.exists():
        raise SafetyError('Q4_K_S destination engine already exists or is linked; not overwritten')
    with engine.open('xb') as output, engine_bin.open('rb') as source_binary:
        shutil.copyfileobj(source_binary, output, length=8 * 1024 * 1024)
    for name in PARITY_TESTS:
        test_binary = BUILD / name
        if test_binary.is_symlink() or not test_binary.is_file():
            raise SafetyError(f'Q4_K_S parity test target is missing or linked: {name}')
    engine.chmod(0o555)
    for name in PARITY_TESTS:
        (BUILD / name).chmod(0o555)
    wheel = cmake_wheel(BASE)
    toolchain = {'cmake': str(cmake), 'cmake_version': cmake_version,
                 'cmake_sha256': digest(cmake), 'cmake_wheel': str(wheel),
                 'cmake_wheel_sha256': digest(wheel),
                 'nvcc': str(nvcc), 'nvcc_version': nvcc_version}
    return engine, BUILD, llama, toolchain


def _pack(paths, python, source):
    _assert_tool_path(BASE, python, 'Q4 pack Python')
    pack_script = source / 'tools/iq_pack.py'
    _assert_data_path(pack_script, 'pinned Q4 pack tool')
    if pack_script.is_symlink() or not pack_script.is_file():
        raise SafetyError('Pinned Q4 pack tool is missing or linked')
    _assert_data_path(PACK, 'Q4_K_S pack directory')
    pending = PACK.with_name(PACK.name + '.preparing')
    _assert_data_path(pending, 'Q4_K_S pack staging directory')
    if PACK.exists() or PACK.is_symlink() or pending.exists() or pending.is_symlink():
        raise SafetyError('Q4_K_S pack or interrupted pack exists without trusted provenance; inspect, never overwrite')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', STRATA_GGUF_PY=str(source / 'third_party/llama.cpp/gguf-py') )
    subprocess.run([str(python), str(pack_script), '--gguf', str(paths[0]),
                    '--out', str(pending), '--compat-bf16'], cwd=source, env=env, check=True)
    for relative in ['dense.bin', 'index.txt', 'native_experts.txt', 'compat-bf16.json',
                     'tokenizer/tokenizer.json', 'tokenizer/chat_template.jinja', 'tokenizer/vocab.json',
                     'tokenizer/merges.txt', 'tokenizer/token_type.json']:
        if not (pending / relative).is_file() or (pending / relative).is_symlink():
            raise SafetyError(f'Q4_K_S compatibility pack incomplete or linked: {relative}')
    for path in pending.rglob('*'):
        if path.name == 'experts.bin':
            raise SafetyError('Q4_K_S compatibility pack must not contain an experts.bin override')
        if path.is_symlink():
            raise SafetyError(f'Q4_K_S pack contains a symlink; refusing promotion: {path}')
        if path.is_file():
            path.chmod(0o444)
    os.rename(pending, PACK)


def prepare(download_only=False):
    """Prepare a private Q4 tree; caller must explicitly opt in via v1strata.sh."""
    _require_unprivileged('Q4_K_S preparation')
    if not BASE.is_absolute():
        raise SafetyError('STRATA_DATA_ROOT must be an absolute path')
    resolved_base = BASE.resolve()
    if resolved_base == Path('/') or resolved_base == owner_home().resolve():
        raise SafetyError('STRATA_DATA_ROOT must be a dedicated subdirectory, not a filesystem/home root')
    for path, label in ((PROFILE_ROOT, 'Q4_K_S profile directory'), (DEST, 'Q4_K_S model directory'),
                        (PACK, 'Q4_K_S pack directory'), (SOURCE, 'Q4_K_S source directory')):
        _assert_data_path(path, label)
    if PROFILE_ROOT.is_symlink():
        raise SafetyError('Q4_K_S profile directory is a symlink')
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = PROFILE_ROOT / 'prepare.lock'
    _assert_data_path(lock_path, 'Q4_K_S preparation lock')
    if lock_path.is_symlink():
        raise SafetyError('Q4_K_S preparation lock is a symlink; refusing append')
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if PREPARED.exists():
            raise SafetyError('Q4_K_S is already prepared; inspect it and use readiness/parity commands, not a rebuild')
        docker_empty()
        if PACK.exists() or PACK.with_name(PACK.name + '.preparing').exists():
            raise SafetyError('Unproven Q4_K_S pack exists; inspect it rather than overwrite')
        before = protected_snapshot()
        live_before = live_strata_identity()
        base_source = BASE / 'source-99f3dbd0b21d'
        runtime_source = pinned_runtime_source(BASE)
        if not download_only:
            _validate_runtime_environment(runtime_source)
        missing = sum(size for name, (size, _) in ASSETS.items()
                      if not (DEST / name).is_file() or (DEST / name).stat().st_size != size)
        required_free = missing + 2 * 1024**3 + DISK_MARGIN
        if shutil.disk_usage(BASE).free < required_free:
            raise SafetyError(f'Insufficient free space: Q4_K_S needs about {TOTAL_BYTES / 1e9:.2f} GB of shards '
                              'plus a pack/build safety margin')
        paths = [download(name, size, sha) for name, (size, sha) in ASSETS.items()]
        if download_only:
            assert_protected(before)
            assert_live_strata_unchanged(live_before)
            print('Q4_K_S shards checksum-verified only. No source/build/pack/inference/service operation performed.')
            return

        for path in paths:
            path.chmod(0o444)
        source_python, base_venv = _prepare_source(runtime_source)
        engine, build, llama, toolchain = _build(source_python, base_venv)
        _pack(paths, source_python, SOURCE)
        mtp = _verify_original_mtp(base_source)
        cfg = make_config(BASE, context=32768)
        atomic_json(PROFILE_ROOT / 'config.json', cfg)
        asset_records = [record(path) for path in paths]
        pack_records = [record(path) for path in sorted(PACK.rglob('*')) if path.is_file()]
        test_records = [record(build / name) for name in PARITY_TESTS]
        manifest = {
            'profile': PROFILE, 'model_id': MODEL_ID,
            'model_repository': REPOSITORY, 'model_revision': REVISION,
            'source_commit': SOURCE_COMMIT, 'source': str(SOURCE),
            'runtime_version': RUNTIME_VERSION, 'runtime_selection_independent': True,
            'runtime_assets': [record(engine)], 'test_binaries': test_records,
            'assets': asset_records, 'pack': str(PACK), 'pack_assets': pack_records,
            'config': str(PROFILE_ROOT / 'config.json'), 'config_args': cfg['args'],
            'build_options': BUILD_OPTIONS, 'llama_cpp_commit': LLAMA_PIN,
            'llama_cpp_source': str(llama.resolve()), 'build_toolchain': toolchain,
            'python_environment': str((SOURCE / '.venv').resolve()),
            'mtp_runtime': mtp['runtime'], 'mtp_tensors': mtp['tensors'],
            'mtp_source_commit': mtp['source_commit'], 'expert_profile': mtp['expert_profile'],
            'initial_context': 32768, 'kv': 'int8', 'kv_resident': None,
            'vision_enabled': False, 'inference_tested': False,
            'parity_required': list(PARITY_TESTS),
            'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        assert_protected(before)
        assert_live_strata_unchanged(live_before)
        atomic_json(PREPARED, manifest)
        print('Q4_K_S PREPARED ONLY: separate pinned runtime/assets/config; parity and live inference still untested.')


def _run_parity_program(command, source, env):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise SafetyError('Q4 parity requires Linux pidfd support for identity-bound child cleanup')
    child = subprocess.Popen(command, cwd=source, env=env, start_new_session=True)
    identity = None
    pidfd = None
    try:
        try:
            pidfd = os.pidfd_open(child.pid, 0)
        except OSError:
            # Continue only if the ordinary identity-bound stop path can still open a pidfd later.
            pidfd = None
        identity = proc(child.pid)
        expected_exe = str(Path(command[0]).resolve())
        if not identity or identity['exe'] != expected_exe or identity['sid'] != child.pid:
            raise SafetyError('Could not establish the parity process identity; refusing an unverified GPU run')
        while child.poll() is None:
            docker_empty()
            jobs = hosting.gpu_jobs()
            if any(not same_process(identity, job) for job in jobs):
                raise SafetyError('A new or unidentified GPU workload appeared; yielding by stopping only the owned parity process')
            time.sleep(0.5)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
    except BaseException as original:
        if child.poll() is None:
            signalled = False
            signal_error = None
            if identity:
                try:
                    if same_process(identity, proc(child.pid)):
                        signal_identity(identity, signal.SIGTERM)
                        signalled = True
                except (SafetyError, OSError):
                    pass
            if not signalled and pidfd is not None:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                    signalled = True
                except ProcessLookupError:
                    signalled = True  # The pidfd names our exited child; wait below to reap it.
                except OSError as exc:
                    signal_error = exc
            if signalled:
                try:
                    child.wait(timeout=PARITY_STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    # Escalate only through the pidfd captured from our direct child; never use a numeric PID.
                    if pidfd is None:
                        try:
                            current_identity = proc(child.pid)
                        except (OSError, SafetyError, KeyError, ValueError):
                            current_identity = None
                        if not identity or not same_process(identity, current_identity):
                            raise SafetyError('Parity child ignored SIGTERM and its identity could not be revalidated; lease must remain held')
                        try:
                            pidfd = os.pidfd_open(child.pid, 0)
                        except OSError as exc:
                            raise SafetyError('Parity child ignored SIGTERM and no safe pidfd could be opened; lease must remain held') from exc
                    try:
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        raise SafetyError('Could not send pidfd-bound SIGKILL to the owned parity child; lease must remain held') from exc
                    try:
                        child.wait(timeout=PARITY_STOP_GRACE_SECONDS)
                    except subprocess.TimeoutExpired as exc:
                        raise SafetyError('Owned parity process did not exit after pidfd-bound SIGKILL; lease must remain held') from exc
            else:
                # Without an identity or pidfd there is no safe signal target. Do not leave an orphan;
                # wait for the direct child to finish rather than using its numeric PID.
                child.wait()
                detail = ('pidfd signal failed; waited for the owned child to exit' if signal_error else
                          'could not identity-signal parity child; waited for the owned child to exit')
                raise SafetyError(detail) from (signal_error or original)
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)


def verify_parity(lease_id, port=8080):
    """Run both real-shard tests under the one-time HostLLM pause lease created for this run."""
    _require_unprivileged('Q4_K_S parity')
    docker_empty()
    lease = hosting.read_lease()
    if (not lease or lease.get('phase') != 'held' or lease.get('port') != port or
            lease.get('lease_id') != lease_id):
        raise SafetyError('Q4 parity requires the matching one-time HostLLM lease identity and port')
    if active_engine() != 'none':
        raise SafetyError('An LLM is running; parity checks will not disturb or stop it')
    launch_gate(port)
    manifest, source, pr, paths, engine_record = _prepared(BASE)
    for name, path in zip(ASSETS, paths):
        size, expected = ASSETS[name]
        if path.stat().st_size != size or digest(path) != expected:
            raise SafetyError(f'Real-shard parity refused: full publisher checksum differs for {name}')
    launch_gate(port)
    binaries = {item['path']: item for item in manifest['test_binaries']}
    native = source / 'build-orca-q4ks/native_expert_parity'
    ple = source / 'build-orca-q4ks/ple_q5_parity'
    for binary in (native, ple):
        saved = binaries.get(str(binary))
        if not saved or digest(binary) != saved['sha256']:
            raise SafetyError(f'Q4_K_S parity executable missing/changed: {binary.name}')
    before = protected_snapshot()
    live_before = live_strata_identity()
    if live_before:
        raise SafetyError('A Strata instance is running; real-shard parity tests require an idle server')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIB_DIRS) + ':' + env.get('LD_LIBRARY_PATH', '')
    _run_parity_program([str(native), *(str(path) for path in paths), '0', '5', '6', '20', '47'], source, env)
    launch_gate(port)
    _run_parity_program([str(ple), str(paths[0])], source, env)
    launch_gate(port)
    for name, path in zip(ASSETS, paths):
        if digest(path) != ASSETS[name][1]:
            raise SafetyError(f'Q4_K_S parity unexpectedly changed original shard: {name}')
    assert_protected(before)
    assert_live_strata_unchanged(live_before)
    parity = {
        'profile': PROFILE, 'source_commit': SOURCE_COMMIT,
        'runtime_sha256': engine_record['sha256'],
        'asset_sha256': {name: sha for name, (_, sha) in ASSETS.items()},
        'tests': {name: {'passed': True} for name in PARITY_TESTS},
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    atomic_json(pr / 'parity.json', parity)
    print('Q4_K_S real-shard native_expert_parity and ple_q5_parity passed. Inference remains untested.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--download-only', action='store_true',
                       help='verify/resume only the pinned model shards; do not clone/build/pack')
    group.add_argument('--verify-parity', action='store_true',
                       help='run both CUDA parity programs on all pinned Q4 shards under HostLLM lease')
    parser.add_argument('--lease-id', help='one-time identity for the HostLLM parity pause lease')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if args.verify_parity:
        import re
        if not args.lease_id or not re.fullmatch(r'[a-f0-9]{64}', args.lease_id):
            parser.error('--verify-parity requires the 64-hex HostLLM lease ID')
        if not 1 <= args.port <= 65535:
            parser.error('--port must be between 1 and 65535')
        verify_parity(args.lease_id, args.port)
    elif args.lease_id or args.port != 8080:
        parser.error('--lease-id/--port are valid only with --verify-parity')
    else:
        prepare(download_only=args.download_only)


if __name__ == '__main__':
    try:
        main()
    except (SafetyError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'Q4_K_S PREPARATION BLOCKED: {exc}', file=sys.stderr)
        raise SystemExit(1)
