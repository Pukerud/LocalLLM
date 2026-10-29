"""Isolated pinned Heretic Qwen profile; readiness is not quality qualification."""
import json
import os
from pathlib import Path
import shutil

from engine_safety import SafetyError, owner_home
import strata_runtime as runtimes
import heretic_assets as A
import heretic_bf16_compat as C

PROFILE=A.PROFILE
VERSION=A.RUNTIME_VERSION
READY=('READY CONFIG: Swift 1.5 Heretic RE; Strata=0.1.46; native context=262144; '
       'BF16 CPU vision; one serving slot/group; --spec 3; prefill auto; split 14,24,36; '
       'trim-stage-weights; pipeline-windows 2; conversation cache=8192 MiB / 4 slots; '
       'thinking high; trusted-LAN 0.0.0.0:8080 without mandatory API key. Bounded test, not qualification.')


def require_state(state):
    wanted=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME
    if Path(state)!=wanted:raise SafetyError('Heretic requires its isolated state directory: '+str(wanted))


def canonical_config(root, source):
    root=Path(root);pr=A.profile_root(root);source=Path(source)
    native=pr/'models'/A.local_name(A.SHARDS[0][0]);ple=pr/'models'/A.local_name(A.SHARDS[1][0])
    state=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME
    return {'exe':str(C.engine_path(root)),'cwd':str(source),'_launcher_python':str(source/'.venv/bin/python'),
            'args':['--pack',str(pr/'pack'),'--native',str(native),'--ple-gguf',str(ple),'--ple-io','mmap',
                    '--expert-profile',str(state/'expert-profile.bin'),'--expert-cache','auto','--prefill','auto',
                    '--spec','3','--mtp',str(pr/'support/mtp'),'--max-context',str(A.NATIVE_CONTEXT),
                    '--kv','int8','--kv-resident','32768','--vision','--conversation-cache-mib','8192',
                    '--conversation-cache-slots','4','--trim-stage-weights','--pipeline-windows','2'],
            'tokenizer':str(pr/'pack/tokenizer'),'gpu':[0,1,2,3],'gpu_order':'as_given','layer_split':'14,24,36',
            'model_name':A.MODEL_ID,'aliases':[PROFILE,'heretic-re'],'host':'0.0.0.0','port':8080,
            'lib_dirs':['/usr/local/cuda-12.9/lib64'],'parallel':1,
            'vision':{'exe':str(source/'engine/strata-vision-cpu'),'model':str(native),
                      'mmproj':str(pr/'support'/A.PROJECTOR[0]),'gpu':False,'max_tokens':1024},
            'sampling':{'temperature':1.0,'top_p':0.95,'top_k':20,'experimental_speed_projection':False},
            'reasoning_effort':'high','lazy_load':False,'idle_unload_s':0,'fit_max_tokens':False,
            'env':{'STRATA_PARALLEL':'1','STRATA_BATCH_GROUPS':'1'}}


def check_file(root, record, *, full=False):
    root=Path(root);p=Path(record['path'])
    try:relative=p.relative_to(root)
    except ValueError:raise SafetyError('Heretic file escaped profile root') from None
    for item in (root,*[root.joinpath(*relative.parts[:n]) for n in range(1,len(relative.parts)+1)]):
        if item.is_symlink():raise SafetyError('Heretic asset path is symlinked: '+str(item))
    if not p.is_file():raise SafetyError('Heretic verified file missing: '+str(p))
    st=p.stat()
    if st.st_size!=record['bytes'] or st.st_mtime_ns!=record['mtime_ns'] or (full and runtimes.digest(p)!=record['sha256']):
        raise SafetyError('Heretic verified file changed: '+str(p))


def ready(root):
    root=Path(root);pr=A.profile_root(root);manifest=pr/'prepared.json'
    if manifest.is_symlink():raise SafetyError('Heretic manifest is symlinked')
    try:d=json.loads(manifest.read_text())
    except FileNotFoundError:raise SafetyError('Heretic is not prepared; run --prepare-heretic --prepare-pack as user') from None
    source,_=runtimes.prepared_runtime(root,VERSION)
    compat=C.folder(root)
    for path in (compat,compat/'bin',compat/'src',compat/'src/core',compat/'src/prefill'):
        if path.is_symlink():raise SafetyError('Heretic compatibility path is symlinked')
    C.ready(root)
    if d.get('compatibility_manifest_sha256')!=runtimes.digest(C.folder(root)/'prepared.json'):
        raise SafetyError('Heretic BF16 compatibility manifest binding differs')
    if (d.get('repository')!=A.REPOSITORY or d.get('revision')!=A.REVISION or d.get('runtime_version')!=VERSION or
            d.get('runtime_manifest_sha256')!=runtimes.digest(runtimes.runtime_root(root,VERSION)/'prepared.json')):
        raise SafetyError('Heretic model/runtime provenance mismatch')
    expected={str(pr/'models'/A.local_name(n)):(size,h) for n,size,h in A.SHARDS}
    expected.update({str(pr/'support'/A.PROJECTOR[0]):A.PROJECTOR[1:]})
    expected.update({str(pr/'support/mtp'/n):(size,h) for n,size,h in A.MTP})
    if {r['path'] for r in d['assets']}!=set(expected):raise SafetyError('Heretic asset set incomplete')
    for r in d['assets']:
        if (r['bytes'],r['sha256'])!=expected[r['path']]:raise SafetyError('Heretic source checksum pin differs')
        check_file(pr,r)
    if d.get('publisher_pack_sha256')!=A.ZIP[2] or {r['path'] for r in d['pack']}!={str(pr/'pack'/p) for p in A.PACK_FILES}:
        raise SafetyError('Heretic publisher pack provenance mismatch')
    for r in d['pack']:
        relative=str(Path(r['path']).relative_to(pr/'pack'))
        if (r['bytes'],r['sha256'])!=A.PACK_PINS[relative]:raise SafetyError('Heretic publisher pack checksum differs')
        check_file(pr,r,full=True)
    cfg_path=pr/'config.json';check_file(pr,d['config'],full=True);cfg=json.loads(cfg_path.read_text())
    if cfg!=canonical_config(root,source):raise SafetyError('Heretic context/vision/cache/tuning configuration differs')
    return source,cfg_path,cfg


def seed_profile(root,state):
    require_state(state);state=Path(state)
    if state.is_symlink():raise SafetyError('Heretic private state is symlinked')
    state.mkdir(parents=True,exist_ok=True)
    if state.stat().st_uid!=os.geteuid():raise SafetyError('Heretic state owner changed')
    state.chmod(0o700);dest=state/'expert-profile.bin'
    if dest.is_symlink():raise SafetyError('Heretic expert profile is symlinked')
    if not dest.exists():
        seed=Path(root)/'source-99f3dbd0b21d/data/expert-profile.bin'
        with dest.open('xb') as out,seed.open('rb') as source:
            os.fchmod(out.fileno(),0o600);shutil.copyfileobj(source,out)
    st=dest.stat()
    if not dest.is_file() or st.st_uid!=os.geteuid() or st.st_mode&0o077 or st.st_size>4*1024**2:
        raise SafetyError('Heretic expert profile type/owner/mode/size changed')
