#!/usr/bin/env python3
"""Prepare pinned publisher DATA only. No inference, service changes, or remote code."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile

from engine_safety import SafetyError, owner_home
from prepare_orca import atomic_json
import strata_runtime as runtimes
import heretic_assets as A
import heretic_profile as H


def data_root():return Path(os.environ.get('STRATA_DATA_ROOT',owner_home()/'.local/share/localllm-strata'))


def safe(root,path):
    root=Path(root);path=Path(path)
    if not root.is_absolute():raise SafetyError('Absolute Heretic data root required')
    try:parts=path.relative_to(root).parts
    except ValueError:raise SafetyError('Heretic destination escaped data root') from None
    for p in (root,*[root.joinpath(*parts[:n]) for n in range(1,len(parts)+1)]):
        if p.is_symlink():raise SafetyError('Heretic destination is symlinked: '+str(p))
    return path


def record(path):
    p=Path(path);s=p.stat()
    if p.is_symlink() or not p.is_file():raise SafetyError('Expected regular Heretic file')
    return {'path':str(p),'bytes':s.st_size,'sha256':runtimes.digest(p),'mtime_ns':s.st_mtime_ns}


def pinned(path,size,sha):
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or runtimes.digest(path)!=sha:
        raise SafetyError('Heretic pinned size/SHA mismatch: '+str(path))


def download_one(row):
    name,size,sha=row;root=data_root();pr=A.profile_root(root)
    p=safe(root,pr/('models' if name.endswith('.gguf') else '')/(A.local_name(name) if name.endswith('.gguf') else name))
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():pinned(p,size,sha);return record(p)
    temp=safe(root,p.with_name(p.name+'.partial'))
    for attempt in range(4):
        try:
            req=urllib.request.Request(f'https://huggingface.co/{A.REPOSITORY}/resolve/{A.REVISION}/{name}?download=true&attempt={attempt}')
            h=hashlib.sha256();total=0
            with urllib.request.urlopen(req,timeout=90) as response,temp.open('wb') as out:
                while block:=response.read(8*1024**2):out.write(block);h.update(block);total+=len(block)
            if total!=size or h.hexdigest()!=sha:raise SafetyError('Downloaded Heretic size/hash mismatch')
            temp.rename(p);p.chmod(0o444);print('Verified '+name,flush=True);return record(p)
        except (OSError,ValueError,SafetyError):
            if attempt==3:raise
            time.sleep(2)


def download():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(download_one,A.ASSETS))
    atomic_json(A.profile_root(data_root())/'downloaded.json',{'repository':A.REPOSITORY,'revision':A.REVISION,'assets':rows})


def extract_pack(archive,dest):
    """Exact allowlisted regular files only; no extractall, links, paths or executable hooks."""
    if dest.exists() or dest.is_symlink():raise SafetyError('Heretic pack destination must be new')
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        if (len(files)!=len(A.PACK_FILES) or {i.filename for i in files}!={A.PACK_ROOT+'/'+p for p in A.PACK_FILES}):
            raise SafetyError('Heretic archive member set differs')
        for i in files:
            relative=i.filename[len(A.PACK_ROOT)+1:]
            if i.file_size!=A.PACK_PINS[relative][0]:raise SafetyError('ZIP member size differs from pinned pack')
        for i in z.infolist():
            pure=PurePosixPath(i.filename)
            if pure.is_absolute() or '..' in pure.parts or '\\' in i.filename:raise SafetyError('Unsafe ZIP path')
            kind=stat.S_IFMT(i.external_attr>>16)
            if kind not in (0,stat.S_IFDIR,stat.S_IFREG):raise SafetyError('ZIP links/special files refused')
        dest.mkdir(mode=0o700)
        for i in files:
            relative=i.filename[len(A.PACK_ROOT)+1:];p=dest/relative;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as source,p.open('xb') as out:shutil.copyfileobj(source,out,8*1024**2)
            p.chmod(0o444)


def copy_pinned(source,dest,size,sha):
    pinned(source,size,sha);safe(data_root(),dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with source.open('rb') as src,dest.open('xb') as out:shutil.copyfileobj(src,out,8*1024**2)
        dest.chmod(0o444)
    pinned(dest,size,sha);return record(dest)


def validate_layout(source,pr):
    """Recompute all 48 expert offsets/types from the eight real GGUF headers, CPU only."""
    script=r'''
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/tools');import iq_pack as P
pr=Path(sys.argv[2]);src=pr/'models'/sys.argv[3];m=P.Model(src)
r=P.expert_layout(m,src)
assert not isinstance(r,str),r
layout,text,n,total=r
actual=(pr/'pack/native_experts.txt').read_text()
clean=lambda t:[line.split() for line in t.splitlines() if line.strip() and not line.startswith('#')]
assert clean(actual)==clean(text),'Publisher expert offsets/types differ from actual GGUFs'
assert n==512 and len(layout)==48
assert m.files[0].metadata['qwen4exp.context_length']==262144
assert m.where['per_layer_token_embd.weight'][3]==m.paths[1],'PLE must be in shard 2'
print(json.dumps({'layers':len(layout),'experts_per_layer':n,'native_expert_bytes':total,'all_tensor_bounds_checked':True,'native_context':262144}))
'''
    result=subprocess.run([str(source/'.venv/bin/python'),'-c',script,str(source),str(pr),A.local_name(A.SHARDS[0][0])],
                          capture_output=True,text=True,check=True,timeout=120)
    print(result.stdout,flush=True)
    return json.loads(result.stdout.splitlines()[-1])


def prepare_pack():
    root=data_root();pr=A.profile_root(root);safe(root,pr)
    if (pr/'prepared.json').exists():H.ready(root);print('Heretic already prepared; no files changed.');return
    source,_=runtimes.prepared_runtime(root,A.RUNTIME_VERSION)
    model_records=[]
    for name,size,sha in A.SHARDS:
        p=safe(root,pr/'models'/A.local_name(name));pinned(p,size,sha);model_records.append(record(p))
    archive=safe(root,pr/A.ZIP[0]);pinned(archive,*A.ZIP[1:])
    pack=safe(root,pr/'pack')
    if not pack.exists():extract_pack(archive,pack)
    from iq3_s_sc117_abliterated_assets import projector_path,mtp_runtime_root
    name,size,sha=A.PROJECTOR
    model_records.append(copy_pinned(projector_path(root),pr/'support'/name,size,sha))
    for name,size,sha in A.MTP:model_records.append(copy_pinned(mtp_runtime_root(root)/name,pr/'support/mtp'/name,size,sha))
    layout=validate_layout(source,pr)
    for name,(size,sha) in A.PACK_PINS.items():pinned(pack/name,size,sha)
    pack_records=[record(pack/name) for name in A.PACK_FILES]
    cfg=pr/'config.json';atomic_json(cfg,H.canonical_config(root,source))
    manifest={'repository':A.REPOSITORY,'revision':A.REVISION,'runtime_version':A.RUNTIME_VERSION,
              'runtime_manifest_sha256':runtimes.digest(runtimes.runtime_root(root,A.RUNTIME_VERSION)/'prepared.json'),
              'compatibility_manifest_sha256':runtimes.digest(H.C.folder(root)/'prepared.json'),
              'publisher_pack_sha256':A.ZIP[2],'assets':model_records,'pack':pack_records,
              'config':record(cfg),'layout_validation':layout,
              'mtp_note':'Same pinned stock/non-abliterated Flash-Next drafter as SC117; target verifies drafts, not claimed Heretic-trained.'}
    atomic_json(pr/'prepared.json',manifest);H.ready(root);print(H.READY,flush=True)


def prepare_compat():
    """Explicit stopped-session overlay build/config upgrade; never changes shared .46."""
    import hosting_lifecycle as lifecycle
    import strata_launcher as launcher
    from engine_safety import docker_empty
    with lifecycle.launch_reservation():
        docker_empty()
        if lifecycle.read_lease() is not None:raise SafetyError('Existing maintenance lease; Heretic unchanged')
        state=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME/'server.json'
        if state.exists() and launcher.family(json.loads(state.read_text())):raise SafetyError('Stop the owned Heretic session first')
        root=data_root();pr=A.profile_root(root);out=safe(root,H.C.folder(root))
        if out.exists() and any(p.is_symlink() for p in out.rglob('*')):raise SafetyError('Compatibility tree contains a symlink')
        H.C.build(root)
        manifest=pr/'prepared.json'
        if not manifest.exists():return
        source,_=runtimes.prepared_runtime(root,A.RUNTIME_VERSION)
        cfg=pr/'config.json';old=json.loads(cfg.read_text());new=H.canonical_config(root,source)
        legacy=dict(new,exe=str(source/'engine/strata'))
        if old not in (new,legacy):raise SafetyError('Heretic compatibility migration refuses config drift')
        m=json.loads(manifest.read_text())
        backup=pr/('before-bf16-compat-'+str(time.time_ns()));backup.mkdir(mode=0o700)
        shutil.copyfile(cfg,backup/'config.json');shutil.copyfile(manifest,backup/'prepared.json')
        atomic_json(cfg,new);m['config']=record(cfg)
        m['compatibility_manifest_sha256']=runtimes.digest(H.C.folder(root)/'prepared.json')
        atomic_json(manifest,m);H.ready(root)
        print('Isolated Heretic BF16 L0 gate compatibility selected; target weights and shared .46 unchanged.',flush=True)


def main():
    if os.geteuid()==0:raise SafetyError('Prepare Heretic only as the unprivileged runtime owner')
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--download-only',action='store_true');g.add_argument('--prepare-pack',action='store_true')
    g.add_argument('--prepare-compat',action='store_true');a=parser.parse_args()
    if a.download_only:download()
    elif a.prepare_compat:prepare_compat()
    else:prepare_pack()


if __name__=='__main__':
    try:main()
    except (SafetyError,OSError,ValueError,subprocess.SubprocessError) as e:
        print('PREPARATION BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(1)
