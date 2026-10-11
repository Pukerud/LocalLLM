#!/usr/bin/env python3
"""Explicitly build/test the isolated Q4_K_S / Strata 0.1.41 candidate; never changes .39 defaults."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

from engine_safety import SafetyError, active_engine, docker_empty, launch_gate, owner_home
import hosting_lifecycle as hosting
import orca_q4ks_profile as baseline_profile
import orca_q4ks_v0141_profile as candidate_profile
import strata_runtime as runtimes
from strata_runtime import LLAMA_PIN
from orca_q4ks_v0141_assets import (ASSETS, BUILD_OPTIONS, HOST_PROFILE, MODEL_ID, PARITY_TESTS,
                                    PROFILE, RUNTIME_VERSION, SOURCE_COMMIT, baseline_profile_root,
                                    baseline_model_root, build_root, candidate_root,
                                    cmake_cache_contract, profile_root, source_root)
from prepare_orca_q4ks import _run_parity_program

BASE = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
PROFILE_ROOT = profile_root(BASE)
SOURCE = source_root(BASE)
BUILD = build_root(BASE)
PREPARED = PROFILE_ROOT / 'prepared.json'
PARITY = PROFILE_ROOT / 'parity.json'
CUDA_LIB_DIRS = ['/usr/local/cuda-12.9/bin', '/usr/local/cuda-12.9/lib64']


def _require_unprivileged(operation):
    if os.geteuid() == 0:
        raise SafetyError(f'{operation} must run as the normal operator; do not use sudo')


def _tree_identity(root, records):
    result = {}
    for item in records:
        path = Path(item['path'])
        info = path.stat()
        result[str(path)] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                             info.st_ctime_ns, info.st_mode)
    return result


def _protected_snapshot(root):
    base_profile = baseline_profile_root(root)
    paths = {root / 'prepared.json', root / 'runtime-selection.json',
             root / 'runtimes/0.1.38/prepared.json', root / 'runtimes/0.1.39/prepared.json',
             root / 'runtimes/0.1.41/prepared.json',
             root / 'runtimes/0.1.38/source/engine/strata',
             root / 'runtimes/0.1.38/source/engine/strata-vision',
             root / 'runtimes/0.1.39/source/engine/strata',
             root / 'runtimes/0.1.39/source/engine/strata-vision',
             root / 'runtimes/0.1.41/source/engine/strata',
             root / 'runtimes/0.1.41/source/engine/strata-vision',
             base_profile / 'prepared.json', base_profile / 'config.json',
             base_profile / 'parity.json',
             root / 'runtimes/orca-q4_k_s-0.1.39/source/engine/strata',
             root / 'runtimes/orca-q4_k_s-0.1.39/source/build-orca-q4ks/CMakeCache.txt',
             root / 'runtimes/orca-q4_k_s-0.1.39/source/build-orca-q4ks/native_expert_parity',
             root / 'runtimes/orca-q4_k_s-0.1.39/source/build-orca-q4ks/ple_q5_parity'}
    if base_profile.is_dir():
        for path in base_profile.rglob('*'):
            if path.is_symlink():
                raise SafetyError(f'Q4 .39 profile contains an unexpected symlink: {path}')
            if path.is_file():
                paths.add(path)
    result = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise SafetyError(f'Protected Q4/runtime path is a symlink: {path}')
        result[str(path)] = candidate_profile.sha256(path) if path.is_file() else None
    return result


def _full_verify_shared(root, baseline_manifest):
    for item in baseline_manifest.get('assets', []) + baseline_manifest.get('pack_assets', []):
        path = Path(item['path'])
        if path.is_symlink() or not path.is_file() or candidate_profile.sha256(path) != item.get('sha256'):
            raise SafetyError(f'Q4 .41 refuses an unverified shared Q4 .39 asset: {path}')
    for name, (size, expected) in ASSETS.items():
        path = baseline_model_root(root) / name
        if path.stat().st_size != size or candidate_profile.sha256(path) != expected:
            raise SafetyError(f'Q4 .41 refuses a shard that differs from its publisher pin: {name}')


def _prepare_source(runtime_source, llama):
    if SOURCE.is_symlink():
        raise SafetyError('Q4 .41 candidate source directory is a symlink')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                        'https://github.com/Niko1221/Strata.git', str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
    actual = runtimes.command_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                      'rev-parse', 'HEAD']).strip()
    if actual != SOURCE_COMMIT:
        raise SafetyError('Q4 .41 candidate source differs from the exact pin; not overwritten')
    if runtimes.command_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                'diff', '--name-only', 'HEAD']).strip():
        raise SafetyError('Q4 .41 candidate source has tracked changes; inspect rather than overwrite')
    venv = SOURCE / '.venv'
    pinned_venv = runtime_source / '.venv'
    if venv.is_symlink():
        if venv.resolve() != pinned_venv.resolve():
            raise SafetyError('Q4 .41 candidate Python link differs from the exact prepared .41 environment')
    elif venv.exists():
        raise SafetyError('Q4 .41 candidate has a foreign private environment; not overwritten')
    else:
        venv.symlink_to(pinned_venv.resolve(), target_is_directory=True)
    llama_link = SOURCE / 'third_party/llama.cpp'
    if llama_link.is_symlink():
        if llama_link.resolve() != llama.resolve():
            raise SafetyError('Q4 .41 candidate llama.cpp link differs from the exact .41 runtime tree')
    elif llama_link.exists():
        raise SafetyError('Q4 .41 candidate llama.cpp path is not the exact verified runtime tree')
    else:
        llama_link.parent.mkdir(parents=True, exist_ok=True)
        llama_link.symlink_to(llama.resolve(), target_is_directory=True)
    candidate_profile._assert_candidate_source_status(SOURCE, built=False)
    return venv / 'bin/python'


def _verified_cmake(base_manifest):
    toolchain = base_manifest.get('build_toolchain', {})
    cmake = Path(toolchain.get('cmake', ''))
    wheel = Path(toolchain.get('cmake_wheel', ''))
    if (not cmake.is_file() or cmake.is_symlink() or not os.access(cmake, os.X_OK) or
            candidate_profile.sha256(cmake) != toolchain.get('cmake_sha256') or
            not wheel.is_file() or wheel.is_symlink() or
            candidate_profile.sha256(wheel) != toolchain.get('cmake_wheel_sha256')):
        raise SafetyError('Q4 .41 requires the exact previously hash-verified isolated Q4 CMake toolchain')
    version = subprocess.check_output([str(cmake), '--version'], text=True).splitlines()[0].strip()
    if version != 'cmake version ' + baseline_profile.CMAKE_VERSION:
        raise SafetyError('Q4 .41 isolated CMake version differs from the verified Q4 build toolchain')
    return cmake, toolchain


def _build(python, runtime_source, llama, cmake):
    if BUILD.exists() or BUILD.is_symlink():
        raise SafetyError(f'Interrupted/foreign Q4 .41 build directory exists; inspect, do not overwrite: {BUILD}')
    engine_dir = SOURCE / 'engine'
    engine_path = engine_dir / 'strata'
    if engine_dir.is_symlink() or engine_path.exists() or engine_path.is_symlink():
        raise SafetyError('Q4 .41 candidate engine destination already exists or is linked; not overwritten')
    nvcc = Path('/usr/local/cuda-12.9/bin/nvcc')
    if not nvcc.is_file():
        raise SafetyError('Existing CUDA 12.9 nvcc is required; no system installation is allowed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               CUDA_HOME='/usr/local/cuda-12.9', STRATA_NVCC=str(nvcc), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PATH'] = str(cmake.parent) + ':' + str(runtime_source / '.venv/bin') + \
                  ':/usr/local/cuda-12.9/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIB_DIRS) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['STRATA_GGUF_PY'] = str(llama / 'gguf-py')
    script = ('import setup; assert setup.LLAMA_CPP_COMMIT == ' + repr(LLAMA_PIN) +
              '; print(setup.source_hash(setup.VISION_SOURCES))')
    vision_hash = subprocess.check_output([str(python), '-c', script], cwd=SOURCE, env=env,
                                          text=True, stderr=subprocess.PIPE).strip()
    if vision_hash != candidate_profile._runtime_material(BASE)[-1]:
        raise SafetyError('Q4 .41 build source does not share the exact .41 vision source contract')
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
    for expected in cmake_cache_contract(cmake, llama):
        if expected not in cache_text:
            raise SafetyError(f'Q4 .41 CMake cache does not contain required option: {expected}')
    output_engine = BUILD / 'strata'
    if output_engine.is_symlink() or not output_engine.is_file():
        raise SafetyError('Q4 .41 build did not produce its separate engine binary')
    engine_dir = SOURCE / 'engine'
    if engine_dir.is_symlink():
        raise SafetyError('Q4 .41 candidate engine output directory is a symlink')
    engine_dir.mkdir(exist_ok=True)
    engine = engine_dir / 'strata'
    if engine.exists() or engine.is_symlink():
        raise SafetyError('Q4 .41 candidate engine destination exists; not overwritten')
    with engine.open('xb') as output, output_engine.open('rb') as source_binary:
        shutil.copyfileobj(source_binary, output, length=8 * 1024 * 1024)
    for name in PARITY_TESTS:
        binary = BUILD / name
        if binary.is_symlink() or not binary.is_file():
            raise SafetyError(f'Q4 .41 parity target missing or linked: {name}')
        binary.chmod(0o555)
    engine.chmod(0o555)
    return engine, cache, vision_hash


def prepare():
    _require_unprivileged('Q4 .41 candidate preparation')
    port = int(os.environ.get('STRATA_PORT', '8080'))
    lease = hosting.read_lease()
    if not isinstance(lease, dict) or lease.get('phase') != 'held' or lease.get('port') != port:
        raise SafetyError('Q4 .41 preparation requires a held HostLLM pause lease for its port')
    if active_engine() != 'none':
        raise SafetyError('Q4 .41 preparation requires no active LLM; leave other sessions untouched and use HostLLM lifecycle first')
    launch_gate(port)
    if not BASE.is_absolute() or BASE.resolve() in (Path('/'), owner_home().resolve()):
        raise SafetyError('STRATA_DATA_ROOT must be a dedicated absolute subdirectory')
    if PROFILE_ROOT.is_symlink() or (BASE / 'profiles').is_symlink():
        raise SafetyError('Q4 .41 candidate profile path is a symlink')
    candidate_root_path = candidate_root(BASE)
    if candidate_root_path.is_symlink() or (BASE / 'runtimes').is_symlink():
        raise SafetyError('Q4 .41 candidate runtime path is a symlink')
    candidate_profile._beneath(BASE, PROFILE_ROOT, 'candidate profile directory')
    candidate_profile._beneath(BASE, candidate_root_path, 'candidate runtime directory')
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = PROFILE_ROOT / 'prepare.lock'
    if lock_path.is_symlink():
        raise SafetyError('Q4 .41 candidate lock is a symlink')
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing_outputs = [PROFILE_ROOT / name for name in (
            'config.json', 'prepared.json', 'parity.json', 'evidence.json', 'validation-32k.json',
            'context-config-before.json', 'native-context-migration.json', 'vision-prepared.json', 'assets')]
        if any(path.exists() or path.is_symlink() for path in existing_outputs):
            raise SafetyError('Q4 .41 candidate output already exists; inspect rather than overwrite/rebuild')
        docker_empty()
        base_path, base_manifest, _, _ = candidate_profile._baseline(BASE)
        _full_verify_shared(BASE, base_manifest)
        runtime_source, runtime_manifest, helper, helper_record, llama, llama_sha, vision_hash = \
            candidate_profile._runtime_material(BASE)
        if shutil.disk_usage(BASE).free < 10 * 1024**3:
            raise SafetyError('Q4 .41 candidate build needs at least 10 GiB free; no files are pruned')
        # Capture retained inputs; the candidate build is not allowed to rewrite baseline/runtime evidence.
        protected = _protected_snapshot(BASE)
        shared_records = base_manifest.get('assets', []) + base_manifest.get('pack_assets', [])
        input_identities = _tree_identity(BASE, shared_records)
        python = _prepare_source(runtime_source, llama)
        cmake, toolchain = _verified_cmake(base_manifest)
        engine, cache, candidate_vision_hash = _build(python, runtime_source, llama, cmake)
        candidate_profile._assert_candidate_source_status(SOURCE, built=True)
        if candidate_vision_hash != vision_hash or candidate_profile.hash_tree(llama) != llama_sha:
            raise SafetyError('Q4 .41 build changed its pinned llama/vision source inputs')
        # Freshly revalidate all shared manifests, executable hashes, and file identities after compilation.
        baseline_profile.ready_q4ks(BASE)
        runtime_source_after, runtime_manifest_after = runtimes.prepared_runtime(BASE, RUNTIME_VERSION)
        if (runtime_source_after.resolve() != runtime_source.resolve() or
                runtime_manifest_after != runtime_manifest or _tree_identity(BASE, shared_records) != input_identities):
            raise SafetyError('Q4 .41 build changed a shared baseline/runtime input; candidate is not promoted')
        for name, expected in protected.items():
            path = Path(name)
            current = candidate_profile.sha256(path) if path.exists() else None
            if current != expected:
                raise SafetyError(f'Q4 .41 build changed protected .39/.41/default-selection file: {path}')
        cfg = candidate_profile.make_config(BASE, context=32768)
        atomic = candidate_profile.atomic_json
        atomic(PROFILE_ROOT / 'config.json', cfg)
        runtime_record = {item.get('path'): item for item in runtime_manifest.get('runtime_assets', [])}
        if runtime_record.get(str(helper)) != helper_record:
            raise SafetyError('Q4 .41 helper record changed during candidate preparation')
        manifest = {
            'profile': PROFILE, 'host_profile': HOST_PROFILE, 'model_id': MODEL_ID,
            'model_repository': baseline_profile.REPOSITORY,
            'model_revision': baseline_profile.REVISION,
            'runtime_version': RUNTIME_VERSION, 'runtime_selection_independent': True,
            'source_commit': SOURCE_COMMIT, 'source': str(SOURCE),
            'runtime_source': str(runtime_source),
            'runtime_manifest_sha256': candidate_profile.sha256(BASE / 'runtimes/0.1.41/prepared.json'),
            'validation_runner_sha256': candidate_profile.sha256(
                Path(__file__).with_name('orca_q4ks_v0141_validation.py')),
            'local_api_client_sha256': candidate_profile.sha256(
                Path(__file__).with_name('local_api_client.py')),
            'runtime_helper': helper_record, 'llama_commit': LLAMA_PIN,
            'llama_source': str(llama), 'llama_tree_sha256': llama_sha,
            'vision_source_sha256': candidate_vision_hash,
            'baseline_profile': HOST_PROFILE,
            'baseline_manifest_sha256': candidate_profile.sha256(base_path),
            'shared_assets': candidate_profile._shared_asset_contract(base_manifest),
            'build_options': BUILD_OPTIONS,
            'build_toolchain': toolchain,
            'build_cache': str(cache),
            'runtime_assets': [candidate_profile.record(engine)],
            'test_binaries': [candidate_profile.record(BUILD / name) for name in PARITY_TESTS],
            'config': str(PROFILE_ROOT / 'config.json'),
            'initial_config_sha256': candidate_profile.sha256(PROFILE_ROOT / 'config.json'),
            'config_args': cfg['args'], 'inference_tested': False,
            'vision_enabled': False, 'configured_context': 32768,
            'prepared_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        atomic(PREPARED, manifest)
        atomic(PROFILE_ROOT / 'evidence.json', {
            'profile': PROFILE, 'runtime_version': RUNTIME_VERSION, 'source_commit': SOURCE_COMMIT,
            'parity_passed': False, 'inference_tested': False,
            'scope': 'Separate Q4 .41 32K text-only candidate; no inference claim until a separately recorded live validation.',
        })
        print('Q4 .41 candidate staged in a distinct source/build/profile tree at 32K text-only. Parity and inference remain untested.')


def verify_parity(lease_id, port):
    _require_unprivileged('Q4 .41 parity')
    docker_empty()
    lease = hosting.read_lease()
    if (not lease or lease.get('phase') != 'held' or lease.get('port') != port or
            lease.get('lease_id') != lease_id):
        raise SafetyError('Q4 .41 parity requires its matching one-time HostLLM lease and port')
    if active_engine() != 'none':
        raise SafetyError('An LLM is running; Q4 .41 parity will not stop or disturb it')
    launch_gate(port)
    manifest, source, pr, cfg, engine_record, base_manifest, _, _ = candidate_profile._prepared(BASE)
    if cfg != candidate_profile.make_config(BASE, context=32768):
        raise SafetyError('Q4 .41 parity is permitted only for the staged 32K text-only candidate')
    if PARITY.exists() or PARITY.is_symlink():
        raise SafetyError('Q4 .41 parity is already recorded; inspect it rather than overwrite')
    paths = [baseline_model_root(BASE) / name for name in ASSETS]
    for name, path in zip(ASSETS, paths):
        size, expected = ASSETS[name]
        if path.stat().st_size != size or candidate_profile.sha256(path) != expected:
            raise SafetyError(f'Q4 .41 parity refused: full publisher hash differs for {name}')
    tests = {item['path']: item for item in manifest.get('test_binaries', [])}
    for name in PARITY_TESTS:
        binary = build_root(BASE) / name
        if candidate_profile.sha256(binary) != tests[str(binary)]['sha256']:
            raise SafetyError(f'Q4 .41 parity executable changed: {name}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = ':'.join(CUDA_LIB_DIRS) + ':' + env.get('LD_LIBRARY_PATH', '')
    _run_parity_program([str(build_root(BASE) / PARITY_TESTS[0]), *(str(path) for path in paths),
                         '0', '5', '6', '20', '47'], source, env)
    launch_gate(port)
    _run_parity_program([str(build_root(BASE) / PARITY_TESTS[1]), str(paths[0])], source, env)
    launch_gate(port)
    for name, path in zip(ASSETS, paths):
        if candidate_profile.sha256(path) != ASSETS[name][1]:
            raise SafetyError(f'Q4 .41 parity unexpectedly changed shared shard: {name}')
    value = {
        'evidence_class': 'operator_self_attestation', 'cryptographic_attestation': False,
        'profile': PROFILE, 'runtime_version': RUNTIME_VERSION, 'source_commit': SOURCE_COMMIT,
        'runtime_sha256': engine_record['sha256'],
        'asset_sha256': {name: sha for name, (_, sha) in ASSETS.items()},
        'tests': {name: {'passed': True} for name in PARITY_TESTS},
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    candidate_profile.atomic_json(PARITY, value)
    evidence_path = pr / 'evidence.json'
    evidence = candidate_profile._json(evidence_path, 'candidate evidence')
    evidence.update(parity_passed=True, parity_sha256=candidate_profile.sha256(PARITY),
                    parity_recorded_at=value['recorded_at'])
    candidate_profile.atomic_json(evidence_path, evidence)
    print('Q4 .41 distinct native-expert and Q5 PLE parity tests passed on the exact pinned Q4 shards.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify-parity', action='store_true')
    group.add_argument('--prepare', action='store_true')
    parser.add_argument('--lease-id')
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
        prepare()


if __name__ == '__main__':
    try:
        main()
    except (SafetyError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'Q4 .41 CANDIDATE BLOCKED: {exc}', file=sys.stderr)
        raise SystemExit(1)
