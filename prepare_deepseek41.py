#!/usr/bin/env python3
"""Explicit pinned DeepSeek preparation; downloads never launch/stop workloads.
Large GGUFs remain in place; the pack references them, without requantizing weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from engine_safety import SafetyError, owner_home
import deepseek41_assets as A
from prepare_orca import atomic_json
from strata_runtime import digest


def data_root():
    return Path(os.environ.get('STRATA_DATA_ROOT', owner_home()/'.local/share/localllm-strata'))


def profile_root():
    return data_root()/'profiles'/A.PROFILE


def runtime_root():
    return data_root()/'runtimes'/'dsv41-0.1.51'


def file_record(path, checksum=None):
    path=Path(path);info=path.stat()
    return {'path':str(path),'bytes':info.st_size,'mtime_ns':info.st_mtime_ns,'sha256':checksum or digest(path)}


def fetch(record):
    relative,size,expected,url=record
    target=profile_root()/'models'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():raise SafetyError('Refusing symlinked model target')
    if target.exists():
        if target.stat().st_size!=size or digest(target)!=expected:raise SafetyError('Existing asset changed; not overwritten: '+str(target))
        return file_record(target,expected)
    part=target.with_name(target.name+'.part')
    if part.is_symlink() or (part.exists() and part.stat().st_size>size):raise SafetyError('Unexpected partial download')
    print('Downloading '+relative+' ('+str(size)+' bytes)',flush=True)
    subprocess.run(['curl','--silent','--show-error','--fail','--location','--retry','5','--retry-delay','3','--connect-timeout','25',
                    '--continue-at','-','--output',str(part),url],check=True)
    if part.stat().st_size!=size or digest(part)!=expected:raise SafetyError('Downloaded size/SHA mismatch: '+relative)
    os.rename(part,target)
    print('Verified '+relative,flush=True)
    return file_record(target,expected)


def download():
    if os.geteuid()==0:raise SafetyError('Download as the unprivileged runtime owner')
    root=profile_root();root.mkdir(parents=True,exist_ok=True)
    sources=A.asset_sources()
    missing=sum(max(0,size-((root/'models'/name).stat().st_size if (root/'models'/name).is_file() else
                            (root/'models'/name).with_name(Path(name).name+'.part').stat().st_size
                            if (root/'models'/name).with_name(Path(name).name+'.part').is_file() else 0)) for name,size,_,_ in sources)
    if shutil.disk_usage(root).free<missing+15*1024**3:raise SafetyError('Need all pinned model files plus 15 GiB headroom; nothing deleted')
    with (root/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Small support files first so runtime/pack integration can be inspected
        # while the four large shards and separate Engram are downloading.
        small=[r for r in sources if r[0] in (A.METADATA,A.MMPROJ)]
        records=[fetch(r) for r in small]
        with ThreadPoolExecutor(max_workers=3) as pool:records+=list(pool.map(fetch,sorted([r for r in sources if r not in small],key=lambda r:-r[1])))
        atomic_json(root/'downloaded.json',{'profile':A.PROFILE,'model_repository':A.REPOSITORY,'model_revision':A.REVISION,
                    'support_repository':A.SUPPORT_REPOSITORY,'support_revision':A.SUPPORT_REVISION,'assets':records,'at':time.time()})
        print('All pinned DeepSeek model/Engram/drafter/BF16 vision assets verified.',flush=True)


SERVER_UPSTREAM_SHA='c9e6d3e382fa521f8469d60c28a701db4417da5dcadea9a131b781fc6d8bf695'
SERVER_PATCHED_SHA='e052a44346ed56972f4a23fafc2bf739411788edbeb4cb95fa5f9544eeb82019'
PATCH_OLD=b'    svc.launch_attestation = launch_attestation\n'
PATCH_NEW=(b'    if cfg.get("arch") == "deepseek41" and cfg.get("deepseek_native_vision") is True:\n'
           b'        from serve.deepseek41_native_vision import enable\n'
           b'        enable(svc)\n'+PATCH_OLD)


def prepare_runtime():
    import hashlib
    from engine_safety import docker_empty
    if os.geteuid()==0:raise SafetyError('Prepare as the unprivileged runtime owner')
    docker_empty()
    rt=runtime_root();source=rt/'source';venv=source/'.venv'
    state=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME/'server.json'
    if state.exists():raise SafetyError('DeepSeek state exists; inspect/stop its owned session before preparation')
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=A.SOURCE_COMMIT:raise SafetyError('DeepSeek source pin mismatch')
    changed=subprocess.check_output(['git','-C',str(source),'diff','--name-only','HEAD'],text=True).splitlines()
    if set(changed)-{'serve/server.py'}:raise SafetyError('Unreviewed DeepSeek tracked edits')
    server=source/'serve/server.py';data=server.read_bytes()
    if hashlib.sha256(data).hexdigest()==SERVER_UPSTREAM_SHA:
        if data.count(PATCH_OLD)!=1:raise SafetyError('Native image hook differs from reviewed source')
        data=data.replace(PATCH_OLD,PATCH_NEW)
        if hashlib.sha256(data).hexdigest()!=SERVER_PATCHED_SHA:raise SafetyError('Image hook patch result mismatch')
        server.write_bytes(data)
    if digest(server)!=SERVER_PATCHED_SHA:raise SafetyError('Unexpected DeepSeek server patch')
    helper=Path(__file__).with_name('deepseek41_native_vision.py')
    target=source/'serve/deepseek41_native_vision.py'
    if target.exists() and digest(target)!=digest(helper):raise SafetyError('Existing native image helper differs; not overwritten')
    if not target.exists():shutil.copyfile(helper,target)
    if not (venv/'bin/python').exists():subprocess.run(['python3','-m','venv',str(venv)],check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('PIP_','PYTHON'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',PIP_CONFIG_FILE='/dev/null')
    subprocess.run([str(venv/'bin/python'),'-m','pip','install','--only-binary=:all:','--index-url','https://pypi.org/simple','-r',str(source/'requirements.txt')],env=env,check=True)
    archive=rt/A.ENGINE_ARCHIVE
    if archive.stat().st_size!=A.ENGINE_SIZE or digest(archive)!=A.ENGINE_SHA256:raise SafetyError('Official DeepSeek archive size/SHA mismatch')
    dest=rt/'engine-packages'
    if dest.exists():raise SafetyError('Engine package destination exists; inspect rather than rewriting it')
    script=('from strata_control.dsv41_engine import install_package; from pathlib import Path; '
            'p=install_package(Path('+repr(str(dest))+'), '+repr(str(archive))+'); assert p is not None; print(p)')
    subprocess.run([str(venv/'bin/python'),'-c',script],cwd=source,env=env,check=True)
    engines=list(dest.glob('*/strata-dsv41'))
    if len(engines)!=1:raise SafetyError('One isolated signed DeepSeek engine package required')
    exe=engines[0]
    subprocess.run([str(exe),'--version'],env=dict(env,LD_LIBRARY_PATH=str(exe.parent)),check=True)
    assets=[file_record(p) for p in sorted(exe.parent.iterdir()) if p.is_file()]
    assets+=[file_record(server),file_record(target),file_record(venv/'pyvenv.cfg'),file_record(source/'requirements.txt')]
    for base in [source/'serve',source/'strata_control',source/'tools']:
        assets += [file_record(p) for p in sorted(base.rglob('*.py')) if '__pycache__' not in p.parts and p not in (server,target)]
    atomic_json(rt/'prepared.json',{'profile':A.PROFILE,'version':A.RUNTIME_VERSION,'app_version':A.APP_VERSION,
                'source_commit':A.SOURCE_COMMIT,'source':str(source),'engine':str(exe),'python':str(venv/'bin/python'),
                'package_sha256':A.ENGINE_SHA256,'package_signed_and_full_hash_verified':True,
                'server_upstream_sha256':SERVER_UPSTREAM_SHA,'server_sha256':SERVER_PATCHED_SHA,
                'native_vision_helper_sha256':digest(helper),'assets':assets,
                'python_metadata':[file_record(p) for p in sorted(venv.glob('lib/python*/site-packages/*.dist-info/METADATA'))],
                'key_free_lan':'upstream optional key; explicit --no-api-key when none supplied; Host/Origin unchanged',
                'system_toolkit_changed':False,'inference_tested':False})
    print('DeepSeek engine 0.1.51 signed package/private Python prepared; old runtimes untouched.',flush=True)


def prepare_pack():
    if os.geteuid()==0:raise SafetyError('Pack as the unprivileged runtime owner')
    root=profile_root();rt=runtime_root();source=rt/'source'
    manifest=json.loads((root/'downloaded.json').read_text());runtime=json.loads((rt/'prepared.json').read_text())
    expected={str(root/'models'/name):(size,checksum) for name,size,checksum,_ in A.asset_sources()}
    if manifest.get('model_revision')!=A.REVISION or manifest.get('support_revision')!=A.SUPPORT_REVISION:raise SafetyError('Download revision mismatch')
    if {row['path'] for row in manifest['assets']}!=set(expected):raise SafetyError('Incomplete model asset set')
    for row in manifest['assets']:
        p=Path(row['path']);info=p.stat();size,checksum=expected[str(p)]
        if p.is_symlink() or info.st_size!=size or info.st_mtime_ns!=row['mtime_ns'] or row['sha256']!=checksum:raise SafetyError('Verified source asset changed')
    pack=root/'pack'
    if pack.exists():raise SafetyError('DeepSeek pack exists; inspect rather than overwriting it')
    python=runtime['python'];models=root/'models'
    subprocess.run([python,str(source/'tools/dsv41/pack_gguf.py'),'--model',str(models),'--shards',*A.SHARDS,
                    '--engram',str(models/A.ENGRAM),'--engram-meta',str(models/A.METADATA),
                    '--dspark-gguf',str(models/A.DRAFTER),'--dspark-noise-token','128799',
                    '--dspark-markov-rank','256','--mmproj',str(models/A.MMPROJ),'--out',str(pack)],check=True)
    shutil.copyfile(source/'data/dsv41/profile-2.00.bin',pack/'expert-profile.bin')
    exe=runtime['engine'];env=dict(os.environ,LD_LIBRARY_PATH=str(Path(exe).parent))
    subprocess.run([exe,'--pack',str(pack),'--pack-info'],env=env,check=True)
    from deepseek41_profile import canonical_config, validate_geometry
    validate_geometry(pack/'ds4.txt')
    cfg=canonical_config(data_root(),runtime)
    atomic_json(root/'config.json',cfg)
    atomic_json(root/'prepared.json',{'profile':A.PROFILE,'model_id':A.MODEL_ID,'model_revision':A.REVISION,
                'support_revision':A.SUPPORT_REVISION,'source_commit':A.SOURCE_COMMIT,'version':A.RUNTIME_VERSION,
                'context':A.NATIVE_CONTEXT,'vision':'native DeepSeek BF16 GPU tower, DSI1 HTTP bridge',
                'assets':manifest['assets'],'pack_assets':[file_record(p) for p in sorted(pack.rglob('*')) if p.is_file()],
                'config':file_record(root/'config.json'),'runtime_manifest_sha256':digest(rt/'prepared.json'),
                'recipe':'plain official GSQ pack, no skip-tensors/dense overrides/requantization/LoRA',
                'inference_tested':False,'full_context_populated_tested':False})
    print('Pinned Abiray native-1M/BF16-vision profile prepared; real serving checks still required.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--download-only',action='store_true')
    group.add_argument('--prepare-runtime',action='store_true')
    group.add_argument('--prepare-pack',action='store_true')
    args=ap.parse_args()
    if args.download_only:download()
    elif args.prepare_runtime:prepare_runtime()
    else:prepare_pack()


if __name__=='__main__':
    try:main()
    except (SafetyError,OSError,ValueError,subprocess.SubprocessError) as exc:
        print('PREPARATION BLOCKED: '+str(exc),flush=True);raise SystemExit(1)
