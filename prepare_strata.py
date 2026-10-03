#!/usr/bin/env python3
"""Explicit install-only Strata preparation. Never starts inference or modifies system packages."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from engine_safety import docker_empty, owner_home

PIN = '99f3dbd0b21d1401b3769e0c0d963913607f380b'
BASE = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
SOURCE = BASE / 'source-99f3dbd0b21d'
DIGESTS = {
    'data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf':
        '4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3',
    'data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf':
        '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113',
    'data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf':
        'b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0',
}


def main():
    docker_empty()  # The bare Hive miner may continue; rentals/unknown Docker state block preparation.
    BASE.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none',
                        'https://github.com/Niko1221/Strata.git', str(SOURCE)], check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', PIN], check=True)
    actual = subprocess.check_output(['git', '-c', f'safe.directory={SOURCE}', '-C', str(SOURCE),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN:
        raise RuntimeError('Strata source pin mismatch; existing source will not be overwritten')
    venv = SOURCE / '.venv'
    if not (venv / 'bin/python').exists():
        subprocess.run(['python3', '-m', 'venv', str(venv)], check=True)
    if Path(sys.prefix) != venv:
        os.execv(str(venv / 'bin/python'), [str(venv / 'bin/python'), __file__])
    os.chdir(SOURCE)
    os.environ['PATH'] = str(venv / 'bin') + ':/usr/local/cuda-12.9/bin:' + os.environ['PATH']
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('strata_setup', SOURCE / 'setup.py')
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    def existing_tools(gpu, yes):
        nvcc = Path('/usr/local/cuda-12.9/bin/nvcc')
        if not nvcc.is_file():
            raise RuntimeError('Existing CUDA 12.9 missing; no system installation is permitted')
        if any(int(a) != 86 for a in gpu.get('archs', [gpu['arch']])):
            raise RuntimeError('This preparation profile requires four sm_86 GPUs')
        return str(nvcc), None

    def bounded_build(src, bdir, target, defs, vcvars, bat_name):
        if vcvars:
            raise RuntimeError('Unexpected Windows build')
        subprocess.run(['cmake', '-S', str(src), '-B', str(bdir), '-DCMAKE_BUILD_TYPE=Release', *defs], check=True)
        subprocess.run(['cmake', '--build', str(bdir), '--target', target, '-j4'], check=True)

    def forbidden(*args, **kwargs):
        raise RuntimeError('Inference/calibration forbidden during install-only preparation')

    setup.install_build_tools = existing_tools
    setup.cmake_build = bounded_build
    setup.start = forbidden
    setup.calibrate_config = forbidden
    sys.argv = ['setup.py', '--setup', '--family', 'qwen', '--model', 'IQ3_S', '--context', '262144',
                '--kv', 'int8', '--vision', 'gpu', '--gpus', '0,1,2,3', '--layer-split', 'auto',
                '--low-ram', 'off', '--vram-reserve-mib', '2048', '--host', '0.0.0.0', '--port', '8080',
                '--data-dir', str(BASE / 'data'), '--build', '--no-start', '--yes']
    if setup.main() != 0:
        raise RuntimeError('Install-only preparation failed')
    verified = []
    for relative, expected in DIGESTS.items():
        path = BASE / relative
        digest, count, last = hashlib.sha256(), 0, time.monotonic()
        with path.open('rb') as f:
            while block := f.read(64 * 1024 * 1024):
                digest.update(block)
                count += len(block)
                if time.monotonic() - last > 10:
                    print(f'Checksum {path.name}: {count / path.stat().st_size:.1%}', flush=True)
                    last = time.monotonic()
        if digest.hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch: {path}')
        verified.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': expected,
                         'mtime_ns': path.stat().st_mtime_ns})
    cfg_path = SOURCE / 'strata-iq3_s.json'
    config = json.loads(cfg_path.read_text())
    assert config['gpu'] == [0, 1, 2, 3] and config['layer_split'] == 'auto'
    assert config['args'][config['args'].index('--max-context') + 1] == '262144'
    assert config.get('vision') and config['vision']['gpu']
    assert not any(a.startswith('--cvec') for a in config['args'])
    runtimes = []
    for executable in [Path(config['exe']), Path(config['vision']['exe'])]:
        runtimes.append({'path': str(executable), 'bytes': executable.stat().st_size,
                         'mtime_ns': executable.stat().st_mtime_ns,
                         'sha256': hashlib.sha256(executable.read_bytes()).hexdigest()})
    prepared = BASE / 'prepared.json'
    tmp = prepared.with_suffix('.tmp')
    tmp.write_text(json.dumps({'source_commit': PIN, 'assets': verified, 'runtime_assets': runtimes,
                               'config': str(cfg_path), 'inference_tested': False}, indent=2) + '\n')
    os.replace(tmp, prepared)
    print('PREPARED AND CHECKSUM VERIFIED. No Strata inference/calibration was run.', flush=True)


if __name__ == '__main__':
    main()
