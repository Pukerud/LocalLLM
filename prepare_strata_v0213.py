#!/usr/bin/env python3
"""Prepare only the pinned 0.2.13 runtime beside .46. No GPU inference or activation API."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request

from engine_safety import SafetyError, docker_empty, owner_home
import strata_v0213_runtime as R


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('x') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if path.exists() or path.is_symlink():
        raise SafetyError('Refusing to overwrite preparation evidence: ' + str(path))
    os.rename(temp, path)


def download(base):
    path = base / R.ASSET
    if path.exists():
        if path.stat().st_size != R.ASSET_BYTES or R.digest(path) != R.ASSET_SHA256:
            raise SafetyError('Existing release archive differs; not overwritten')
        return path
    partial = path.with_suffix(path.suffix + '.part')
    # A complete interrupted transfer may be adopted ONLY after its exact whole-file pin matches.
    if not partial.exists():
        start = time.monotonic()
        with urllib.request.urlopen(R.URL, timeout=90) as response, partial.open('xb') as f:
            if response.status != 200 or int(response.headers.get('Content-Length', '-1')) != R.ASSET_BYTES:
                raise SafetyError('Release archive HTTP length/status differs')
            written = 0
            while block := response.read(2 * 1024**2):
                f.write(block)
                written += len(block)
                if written > R.ASSET_BYTES:
                    raise SafetyError('Oversized release archive')
                time.sleep(max(0, written / (32 * 1024**2) - (time.monotonic() - start)))
            f.flush()
            os.fsync(f.fileno())
    if partial.stat().st_size != R.ASSET_BYTES or R.digest(partial) != R.ASSET_SHA256:
        raise SafetyError('Release archive incomplete/hash mismatch; partial retained, no extraction')
    partial.chmod(0o444)
    os.rename(partial, path)
    return path


def extract_archive(archive, destination):
    """No extractall, traversal, devices, executable hooks or escaping links. Links become new regular files."""
    if destination.exists():
        raise SafetyError('Candidate package directory exists without a completed receipt; no overwrite')
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        if len(members) > 500 or sum(m.size for m in members) > 5 * 1024**3:
            raise SafetyError('Release archive exceeds bounded package limits')
        seen = set()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or str(p) in seen or not (m.isfile() or m.isdir() or m.issym() or m.islnk()):
                raise SafetyError('Unsafe/duplicate release archive member')
            seen.add(str(p))
            if m.issym() or m.islnk():
                target = PurePosixPath(m.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise SafetyError('Unsafe archive link')
        destination.mkdir(mode=0o700)
        for m in members:
            target = destination / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            f = tf.extractfile(m)
            if f is None:
                raise SafetyError('Missing archive member content')
            with f, target.open('xb') as out:
                shutil.copyfileobj(f, out, 1024**2)
            target.chmod(0o555 if m.mode & 0o111 else 0o444)
    return destination


def finish(root, source, package, env):
    base = R.folder(root)
    R.patch_server(source)
    venv = source / '.venv'
    if not venv.exists():
        subprocess.run(['/usr/bin/python3', '-m', 'venv', str(venv)], env=env, check=True)
    python = venv / 'bin/python'
    pipenv = {k: v for k, v in env.items() if not k.startswith('PIP_')}
    pipenv['PIP_CONFIG_FILE'] = '/dev/null'
    subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--index-url',
                    'https://pypi.org/simple', '-r', str(source / 'requirements.txt')], env=pipenv, check=True)
    # Verify the official component's Ed25519 signature AND every file hash before invoking any executable.
    script = ('from pathlib import Path; from strata_control import engine_component as E; '
              'c=E.read_manifest(Path(' + repr(str(package)) + ')); '
              'assert c.version=="0.1.59" and "sm_86" in c.archs; '
              'assert E.verify_signature(c)=="verified"; E.verify(c,full=True); '
              'print("Official component signature and all payload hashes verified")')
    subprocess.run([str(python), '-c', script], cwd=source, env=env, check=True)
    engine = source / 'engine'
    engine.mkdir(exist_ok=True)
    # No all-cards split/GPU-vision binary is installed on the serving path.
    for name in ('strata', 'libstrata_kq.so', 'libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13', 'BUILD.json'):
        dest = engine / name
        if dest.exists():
            if R.digest(dest) != R.digest(package / name):
                raise SafetyError('Existing candidate engine copy differs; not overwritten')
        else:
            shutil.copyfile(package / name, dest)
            dest.chmod(0o555 if name in ('strata', 'libstrata_kq.so') else 0o444)
    # The CPU reader uses the exact existing llama.cpp dependency: no system/CUDA installation or GPU build.
    import stage_unsloth_v31 as B
    old_source, old_manifest = B.runtime(root)
    llama = old_source / 'third_party/llama.cpp'
    if old_manifest['llama_commit'] != '3cf03257f219afbe7334045ff7c6a06ac68c627d':
        raise SafetyError('CPU vision dependency revision changed')
    build = source / 'build-vision-cpu'
    subprocess.run(['/usr/bin/cmake', '-S', str(source / 'tools/vision'), '-B', str(build),
                    '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_DIR=' + str(llama),
                    '-DSTRATA_VISION_CUDA=OFF', '-DSTRATA_VISION_HIP=OFF', '-DSTRATA_VISION_SYCL=OFF',
                    '-DSTRATA_PORTABLE=ON'], env=env, check=True)
    subprocess.run(['/usr/bin/cmake', '--build', str(build), '--target', 'strata-vision', '-j2'], env=env, check=True)
    dest = engine / 'strata-vision-cpu'
    built = build / 'bin/strata-vision'
    if dest.exists() and R.digest(dest) != R.digest(built):
        raise SafetyError('Existing CPU reader differs; no overwrite')
    if not dest.exists():
        shutil.copyfile(built, dest)
        dest.chmod(0o555)
    # --version / --geometry / --help exit before model loading or CUDA-device setup; visibility stays empty.
    probes = {}
    for flag in ('--version', '--geometry', '--help'):
        r = subprocess.run([str(engine / 'strata'), flag], cwd=source, env=env, capture_output=True, text=True, timeout=30, check=True)
        probes[flag] = (r.stdout + r.stderr).strip()
    if '0.1.59' not in probes['--version'] or 'flash_next' not in probes['--geometry']:
        raise SafetyError('Actual release executable version/geometry differs')
    r = subprocess.run([str(dest)], env=env, capture_output=True, text=True, timeout=30)
    if r.returncode != 2 or 'usage: strata-vision' not in r.stdout + r.stderr:
        raise SafetyError('CPU vision helper did not start correctly')
    needed = subprocess.check_output(['readelf', '-d', str(dest)], text=True)
    if any(x in needed.lower() for x in ('libcuda', 'libcublas', 'libhip', 'libsycl')):
        raise SafetyError('CPU vision reader unexpectedly links a GPU runtime')
    R.source_check(source)
    site = next((venv / 'lib').glob('python*/site-packages'))
    pyfiles = [venv / 'pyvenv.cfg', *sorted(p for p in site.rglob('*') if p.is_file() and '__pycache__' not in p.parts)]
    value = {'version': R.VERSION, 'app_version': '0.2.13', 'source_commit': R.PIN,
             'release_archive_sha256': R.ASSET_SHA256, 'component_signature': 'verified',
             'binary_build_id': json.loads((package / 'engine-component.json').read_text())['build_id'],
             'frontend_patch': {'upstream_sha256': R.SERVER_ORIGINAL_SHA256, 'patched_sha256': R.SERVER_PATCHED_SHA256,
                                'scope': 'Opt-in external lifecycle: no frontend native-engine restart; no numerical/auth patch'},
             'runtime_assets': [R.record(p) for p in sorted(engine.iterdir()) if p.is_file()],
             'python_assets': [R.record(p) for p in pyfiles],
             'cpu_vision': 'CPU-only build; STRATA_VISION_CUDA=OFF',
             'cpu_vision_llama_commit': old_manifest['llama_commit'], 'no_gpu_probes': probes,
             'activation_allowed': False, 'gpu_inference_tested': False,
             'prepared_at': time.strftime('%FT%TZ', time.gmtime())}
    atomic_json(base / 'prepared.json', value)
    R.ready(root)
    print('0.2.13 / engine 0.1.59 OFFLINE PREPARED; no GPU model loaded, no activation, original .46 untouched.', flush=True)
    return value


def prepare(root):
    base = R.folder(root)
    if os.geteuid() != 1000:
        raise SafetyError('Preparation runs only as user/UID1000')
    if base.is_symlink() or base.stat().st_uid != 1000 or stat.S_IMODE(base.stat().st_mode) != 0o700:
        raise SafetyError('Candidate runtime requires its separately provisioned private owner directory')
    if (base / 'prepared.json').exists():
        return R.ready(root)[1]
    docker_empty()
    if shutil.disk_usage(base).free < 8 * 1024**3:
        raise SafetyError('Need 8 GiB disk space; no original files are removed')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
    if available < 8 * 1024**2:
        raise SafetyError('Need 8 GiB available RAM; no workload is stopped')
    archive = download(base)
    source = base / 'source'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    if not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', R.TAG, '--single-branch', R.REPOSITORY, str(source)], env=env, check=True)
    R.source_check(source)
    receipt = base / 'downloaded.json'
    if receipt.exists():
        d = json.loads(receipt.read_text())
        if d['source_commit'] != R.PIN:
            raise SafetyError('Download source receipt differs')
        for row in [d['archive'], *d['package_files']]:
            R.check_record(row)
        package = base / 'package'
    else:
        package = extract_archive(archive, base / 'package')
        atomic_json(receipt, {'source_commit': R.PIN, 'archive': R.record(archive),
                     'package_files': [R.record(p) for p in sorted(package.rglob('*')) if p.is_file()]})
    return finish(root, source, package, env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--prepare-profile', action='store_true', help='Create only the fresh locked profile/config; never infer or activate')
    a = ap.parse_args(argv)
    root = Path(os.environ.get('STRATA_DATA_ROOT', owner_home() / '.local/share/localllm-strata'))
    try:
        base = R.folder(root)
        if os.geteuid() != 1000:
            raise SafetyError('Run offline preparation as user/UID1000')
        fd = os.open(base / 'prepare.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.check:
                R.ready(root)
            elif a.prepare_profile:
                import unsloth_v31_v0159_profile as C
                C.prepare(root)
                print(C.READY, flush=True)
            else:
                prepare(root)
        return 0
    except (OSError, ValueError, KeyError, SafetyError, subprocess.SubprocessError) as e:
        print('OFFLINE PREPARATION BLOCKED:', e)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
