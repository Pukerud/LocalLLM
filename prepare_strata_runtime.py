#!/usr/bin/env python3
"""Build a pinned runtime beside tested rollbacks; existing CUDA/private dependencies, no setup/update/inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from engine_safety import SafetyError, docker_empty, command_output
from strata_runtime import VERSION, LEGACY_VERSION, BASE_VERSION, PINS, LLAMA_PIN, PROFILES, digest, runtime_root
from prepare_orca import atomic_json


def record(path):
    st = path.stat()
    return {'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=[LEGACY_VERSION, VERSION], default=VERSION)
    version = parser.parse_args().runtime
    pin = PINS[version]
    import strata_launcher as launcher
    import hosting_lifecycle as hosting
    root = launcher.data_root()
    folder = runtime_root(root, version)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        docker_empty()
        state = launcher.read_state()
        jobs = hosting.gpu_jobs()
        if state:
            launcher.runtime_gate(state)
            if state.get('runtime_version', BASE_VERSION) == version and launcher.family(state):
                raise SafetyError('Candidate runtime is active; preparation refuses to rewrite its executables/dependencies')
        elif jobs and not hosting.only_hive_miners(jobs):
            raise SafetyError('Unknown GPU workload; candidate preparation blocked')
        if shutil.disk_usage(root).free < 10 * 1024**3:
            raise SafetyError('Need at least 10 GiB free for the private source/build')
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        if available < 16 * 1024**2:
            raise SafetyError('Need at least 16 GiB available RAM; no live model/miner is stopped to make room')
        configs = {}
        protected = {}
        for profile in PROFILES:
            source0, cfg_path, cfg = launcher.ready(profile)
            configs[profile] = {'path':str(cfg_path),'sha256':digest(cfg_path)}
            for p in [cfg_path, root / 'prepared.json', root / 'profiles/orca-iq3_xxs/prepared.json',
                      source0 / 'engine/strata', source0 / 'engine/strata-vision']:
                protected[str(p)] = digest(p)
        # Preserve previously tested runtime/selection and Hive files, including recorded absence.
        for p in [root / 'runtime-selection.json', Path('/hive-config/rig.conf'), Path('/hive-config/wallet.conf'),
                  Path('/hive-config/watchdog.conf'), Path('/run/hive/cur_miner')]:
            protected[str(p)] = digest(p) if p.exists() else None
        for retained in [LEGACY_VERSION, VERSION]:
            if retained == version:
                continue
            other = runtime_root(root, retained)
            for rel in ['prepared.json', 'source/engine/strata', 'source/engine/strata-vision']:
                p = other / rel
                if p.exists():
                    protected[str(p)] = digest(p)
        source = folder / 'source'
        if not source.exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/Niko1221/Strata.git',str(source)],check=True)
            subprocess.run(['git','-C',str(source),'checkout','--detach',pin],check=True)
        if command_output(['git','-c',f'safe.directory={source}','-C',str(source),'rev-parse','HEAD']).strip() != pin:
            raise SafetyError('Existing candidate source differs from release pin; not overwritten')
        if command_output(['git','-c',f'safe.directory={source}','-C',str(source),'diff','--name-only','HEAD']).strip():
            raise SafetyError('Candidate has tracked local changes; not overwritten')
        venv = source / '.venv'
        old_venv = source0 / '.venv'
        overlay = folder / 'python'
        if not (overlay / 'bin/python').is_file():
            subprocess.run(['python3','-m','venv',str(overlay)],check=True)
        shared = next((old_venv / 'lib').glob('python*/site-packages'))
        site = next((overlay / 'lib').glob('python*/site-packages'))
        pth = site / 'strata_shared_readonly.pth'
        if pth.exists() and pth.read_text().strip() != str(shared):
            raise SafetyError('Unexpected candidate dependency overlay; not overwritten')
        pth.write_text(str(shared) + '\n')
        if venv.is_symlink() and venv.resolve() == old_venv.resolve():
            venv.unlink()  # remove only our candidate symlink, never the original dependency environment
        if not venv.exists():
            venv.symlink_to(overlay, target_is_directory=True)
        if venv.resolve() != overlay.resolve():
            raise SafetyError('Unexpected candidate private Python path')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                   CUDA_HOME='/usr/local/cuda-12.9', STRATA_NVCC='/usr/local/cuda-12.9/bin/nvcc',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        env['PATH'] = str(overlay / 'bin') + ':' + str(old_venv / 'bin') + ':/usr/local/cuda-12.9/bin:' + env['PATH']
        # Optional schema validation lives only in the candidate overlay. No original/system package is modified.
        packages = {'jsonschema': '4.25.1'}
        if version == VERSION:
            packages['psutil'] = '7.2.2'  # native-process activity checks, only in the candidate overlay
        for name, package_version in packages.items():
            check = f"from importlib.metadata import version; print(version({name!r}))"
            found = subprocess.run([str(overlay / 'bin/python'),'-c',check],env=env,capture_output=True,text=True)
            if found.returncode or found.stdout.strip() != package_version or not (site / f'{name}-{package_version}.dist-info/METADATA').is_file():
                pipenv = {k:v for k,v in env.items() if not k.startswith('PIP_')}
                pipenv['PIP_CONFIG_FILE'] = '/dev/null'
                subprocess.run([str(overlay / 'bin/python'),'-m','pip','install','--ignore-installed','--only-binary=:all:',
                                '--index-url','https://pypi.org/simple',f'{name}=={package_version}'],env=pipenv,check=True)
        # Source preparation only: get_llama_cpp extracts the explicitly pinned build dependency.
        # No setup.main(), downloads of model weights, pack conversion or system/tool installation.
        script = "import setup; assert setup.LLAMA_CPP_COMMIT == %r; setup.get_llama_cpp()" % LLAMA_PIN
        subprocess.run([str(venv / 'bin/python'),'-c',script],cwd=source,env=env,check=True)
        llama = source / 'third_party/llama.cpp'
        nvcc = '/usr/local/cuda-12.9/bin/nvcc'
        if not Path(nvcc).is_file():
            raise SafetyError('Existing CUDA12.9 absent; no system installation allowed')
        cmake = str(old_venv / 'bin/cmake')
        build = source / 'build'
        defs = ['-DCMAKE_BUILD_TYPE=Release','-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_BUILD_TESTS=OFF',
                '-DCMAKE_CUDA_ARCHITECTURES=86',f'-DCMAKE_CUDA_COMPILER={nvcc}',f'-DSTRATA_GGML_DIR={llama}']
        docker_empty()
        subprocess.run([cmake,'-S',str(source),'-B',str(build),*defs],env=env,check=True)
        subprocess.run([cmake,'--build',str(build),'--target','strata','-j3'],env=env,check=True)
        engine = source / 'engine'; engine.mkdir(exist_ok=True)
        shutil.copy2(build / 'strata', engine / 'strata')
        # Reuse the proven GPU helper only when both pinned source/build contracts match exactly.
        script = "import setup,json; print(json.dumps({'llama':setup.LLAMA_CPP_COMMIT,'vision':setup.source_hash(setup.VISION_SOURCES)}))"
        vision_contract = json.loads(subprocess.check_output([str(venv / 'bin/python'),'-c',script],cwd=source,env=env,text=True))
        old_meta = json.loads((source0 / 'engine/BUILD.json').read_text())
        reused = vision_contract['vision'] == old_meta.get('vision_src')
        if reused:
            old_script = "import setup; print(setup.LLAMA_CPP_COMMIT)"
            reused = subprocess.check_output([str(old_venv / 'bin/python'),'-c',old_script],cwd=source0,env=env,text=True).strip() == LLAMA_PIN
        if reused:
            shutil.copy2(source0 / 'engine/strata-vision',engine / 'strata-vision')
        else:
            vbuild = source / 'build-vision'
            subprocess.run([cmake,'-S',str(source / 'tools/vision'),'-B',str(vbuild),'-DCMAKE_BUILD_TYPE=Release',
                            f'-DLLAMA_DIR={llama}','-DSTRATA_VISION_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=86',
                            f'-DCMAKE_CUDA_COMPILER={nvcc}'],env=env,check=True)
            subprocess.run([cmake,'--build',str(vbuild),'--target','strata-vision','-j3'],env=env,check=True)
            shutil.copy2(vbuild / 'bin/strata-vision',engine / 'strata-vision')
        subprocess.run([str(engine / 'strata'),'--help'],env=env,stdout=subprocess.DEVNULL,check=True)
        for name, sha in protected.items():
            if (digest(Path(name)) if Path(name).exists() else None) != sha:
                raise SafetyError('Original profile/runtime changed during build; no restart or rollback attempted')
        atomic_json(engine / 'BUILD.json', {'source':'local','version':version,'archs':[86],'vision':'gpu',
            'cuda_dirs':['/usr/local/cuda-12.9/bin','/usr/local/cuda-12.9/lib64'],'toolkit':12,
            'vision_src':vision_contract['vision'],'vision_reused_identical_source':reused})
        atomic_json(folder / 'prepared.json', {'version':version,'source_commit':pin,'llama_commit':LLAMA_PIN,
            'source':str(source),'cuda_arch':86,'cuda_toolkit':'/usr/local/cuda-12.9',
            'runtime_assets':[record(engine / 'strata'),record(engine / 'strata-vision')],
            'baseline_configs':configs,'vision_reused_identical_source':reused,'inference_tested':False,
            'python_environment':str(overlay),'python_packages':packages,'jsonschema_version':'4.25.1',
            'python_assets':[record(overlay / 'pyvenv.cfg'),record(pth)] + [record(p) for p in sorted(site.glob('*.dist-info/METADATA'))],
            'retained_hashes_or_absences':protected,
            'prepared_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print(f'{version} PREPARED beside unchanged tested rollbacks. No model download/repack/inference/service/system change.',flush=True)


if __name__=='__main__':
    try:
        main()
    except (SafetyError, OSError, subprocess.SubprocessError, ValueError) as e:
        print(f'PREPARATION BLOCKED: {e}',file=sys.stderr)
        raise SystemExit(1)
