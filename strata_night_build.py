#!/usr/bin/env python3
"""Separate CPU-only .59 OFF/ON Q5-MMQ builds; never executes an inference engine."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/user/.local/share/localllm-strata')
DEST = ROOT / 'runtimes/0.1.59-night-20261113'
SOURCE = ROOT / 'runtimes/0.1.59/source'
CMAKE = str(ROOT / 'source-99f3dbd0b21d/.venv/bin/cmake')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    if os.geteuid() != 1000: raise RuntimeError('Build only as UID1000')
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PATH'] = '/usr/local/cuda-12.9/bin:/usr/bin:/bin'
    os.environ['CUDACXX'] = '/usr/local/cuda-12.9/bin/nvcc'
    import strata_v0213_runtime as R
    R.ready(ROOT)
    if (DEST.is_symlink() or not DEST.is_dir() or DEST.stat().st_uid != 1000 or
            DEST.stat().st_mode & 0o077 or any(DEST.iterdir())):
        raise RuntimeError('Administrator must provision only this new empty UID1000/private build directory')
    build = DEST / 'build'
    started = time.time()
    for flag, name in [('OFF', 'control'), ('ON', 'q5mmq')]:
        out = DEST / name
        out.mkdir(mode=0o700)
        with (out / 'build.log').open('x') as log:
            args = [CMAKE, '-S', str(SOURCE), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_PORTABLE=ON',
                    '-DSTRATA_ENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=86-real',
                    '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc',
                    '-DCUDAToolkit_ROOT=/usr/local/cuda-12.9',
                    '-DSTRATA_ORCA_Q4KS_MMQ=' + flag, '-DSTRATA_BUILD_TESTS=OFF',
                    '-DSTRATA_BUILD_SPLIT=OFF', '-DSTRATA_MMQ_KQUANTS=ON',
                    '-DSTRATA_Q6K_EXPERTS=ON', '-DSTRATA_MMQ_TQ_OVERLAY=ON']
            print('Configuring ' + name, flush=True)
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            subprocess.run([CMAKE, '--build', str(build), '--target', 'strata', 'strata_kq', '-j2'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=5400)
        candidates = [build / 'strata', build / 'bin/strata']
        exe = next(p for p in candidates if p.is_file())
        shutil.copy2(exe, out / 'strata')
        kq = next(build.rglob('libstrata_kq.so'))
        shutil.copy2(kq, out / kq.name)
        shutil.copy2(build / 'CMakeCache.txt', out / 'CMakeCache.txt')
        symbols = subprocess.check_output(['nm', '-C', str(out / 'strata')], text=True)
        q5 = 'void mul_mat_q_case<(ggml_type)6>' in symbols
        if q5 != (flag == 'ON'): raise RuntimeError('Q5 symbol capability differs from build flag')
        deps = subprocess.check_output(['ldd', str(out / 'strata')], text=True)
        if 'not found' in deps: raise RuntimeError('Build dependency missing: ' + deps)
        (out / 'ldd.txt').write_text(deps)
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in (out / 'strata', out / kq.name, out / 'CMakeCache.txt')}
        receipt = {'phase': 'BUILT-NOT-INFERENCE-QUALIFIED', 'source_commit': R.PIN,
                   'source_build_origin': 'final frontend tag source, NOT official rc-native binary',
                   'q5_0_mmq': q5, 'cuda': '12.9.86', 'architecture': '86-real',
                   'uid': os.geteuid(), 'jobs': 2, 'gpus_hidden': True,
                   'configure_argv': args, 'files': files, 'finished': time.time(),
                   'elapsed_total_s': time.time() - started}
        (out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
        for p in (out / 'strata', out / kq.name): p.chmod(0o555)
        print(json.dumps({'built': name, 'q5_mmq': q5, 'files': files}), flush=True)
    R.ready(ROOT)  # Original sealed runtime remains unchanged.
    (DEST / 'complete.json').write_text(json.dumps({'phase': 'BUILDS-COMPLETE', 'elapsed_s': time.time()-started}))

if __name__ == '__main__': main()
