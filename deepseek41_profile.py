"""Isolated native-1M, BF16-vision DeepSeek profile; older runtimes never selected.
Readiness is provenance/configuration, not populated-context or quality qualification.
"""
import json
import os
from pathlib import Path
import subprocess

from engine_safety import SafetyError, owner_home
from strata_runtime import digest
import deepseek41_assets as A

PROFILE=A.PROFILE
VERSION=A.RUNTIME_VERSION
ENGINE_TELEMETRY='ds4-0.1'  # upstream INFO engine identifier; package --version is 0.1.51
READY=('READY CONFIG: DeepSeek V4.1 Flash 1017 Abiray IQ2_XXS; Strata DeepSeek=0.1.51 (app 0.2.6); '
       'native context=1048576; native BF16 GPU vision (DSI1 bridge); one serving slot; six experts/token; '
       'DSpark pinned preset; four GPUs; trusted-LAN 0.0.0.0:8080 without mandatory API key. '
       'Configured native window, not a populated-1M or sustained-load qualification.')


def profile_root(root):return Path(root)/'profiles'/PROFILE

def runtime_root(root):return Path(root)/'runtimes'/'dsv41-0.1.51'


def canonical_config(root, runtime, *, draft_mode='auto'):
    if draft_mode not in ('none','auto'):raise SafetyError('Unsupported DeepSeek draft preset')
    pack=profile_root(root)/'pack'
    state=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME
    return {'exe':runtime['engine'],'cwd':runtime['source'],'_launcher_python':runtime['python'],
            'arch':'deepseek41','deepseek_native_vision':True,'tokenizer':str(pack/'tokenizer'),
            'model_name':A.MODEL_ID,'aliases':[PROFILE,'deepseek41','deepseek-v4.1-flash'],
            'host':'0.0.0.0','port':8080,'gpu':[0,1,2,3],'gpu_order':'as_given',
            'args':['--pack',str(pack),'--max-context',str(A.NATIVE_CONTEXT),'--eos','1',
                    '--reserve-mib','1536','--experts','6','--dspark',draft_mode,'--dspark-conf','0.65','--vision','on',
                    '--expert-profile',str(state/'expert-profile.bin')],
            'parallel':1,'lib_dirs':[str(Path(runtime['engine']).parent)],'cuda':13,'engine_ready_s':3600,
            'lazy_load':False,'idle_unload_s':0,'fit_max_tokens':False,'reasoning_effort':'high',
            'sampling':{'temperature':1.0,'top_p':1.0},
            # DeepSeek .51 rejects Qwen's --conversation-cache-mib. Its RAM cache
            # is count-bounded; SLOT_GB caps the disk-backed store, NOT RAM bytes.
            'env':{'STRATA_DS4_SLOT_DIR':str(state/'ds4-slots'),
                   'STRATA_DS4_SLOT_KEEP':'1','STRATA_DS4_SLOT_CAP':'4',
                   'STRATA_DS4_SLOT_GB':'8','STRATA_DS4_REUSE':'1'}}


def verify_records(records, *, full=False):
    for record in records:
        path=Path(record['path'])
        if path.is_symlink() or not path.is_file():raise SafetyError('DeepSeek verified file missing/symlinked: '+str(path))
        info=path.stat()
        if info.st_size!=record['bytes'] or info.st_mtime_ns!=record['mtime_ns'] or (full and digest(path)!=record['sha256']):
            raise SafetyError('DeepSeek verified file changed: '+str(path))


def validate_geometry(path):
    values=dict(line.split(' ',1) for line in Path(path).read_text().splitlines() if ' ' in line)
    required={'max_position_embeddings':str(A.NATIVE_CONTEXT),'num_experts_per_tok':'6',
              'dspark_noise_token_id':'128799','dspark_markov_rank':'256',
              'vision.num_hidden_layers':'32','image_token_id':'129264'}
    if any(values.get(key)!=value for key,value in required.items()):
        raise SafetyError('DeepSeek native context/stock drafter/BF16 vision geometry mismatch')


def ready(root):
    from prepare_deepseek41 import SERVER_PATCHED_SHA,SERVER_UPSTREAM_SHA
    root=Path(root);pr=profile_root(root);rt=runtime_root(root)
    try:
        runtime=json.loads((rt/'prepared.json').read_text());model=json.loads((pr/'prepared.json').read_text())
    except FileNotFoundError:raise SafetyError('DeepSeek is not prepared: run prepare_deepseek41.py --download-only, --prepare-runtime, then --prepare-pack as user') from None
    source=rt/'source'
    if (runtime.get('source_commit')!=A.SOURCE_COMMIT or runtime.get('version')!=VERSION or
            runtime.get('package_sha256')!=A.ENGINE_SHA256 or runtime.get('package_signed_and_full_hash_verified') is not True or
            runtime.get('source')!=str(source) or runtime.get('python')!=str(source/'.venv/bin/python') or
            model.get('source_commit')!=A.SOURCE_COMMIT or model.get('model_revision')!=A.REVISION or
            model.get('support_revision')!=A.SUPPORT_REVISION or model.get('version')!=VERSION or
            model.get('runtime_manifest_sha256')!=digest(rt/'prepared.json')):
        raise SafetyError('DeepSeek model/runtime provenance mismatch')
    commit=subprocess.check_output(['git','-c',f'safe.directory={source}','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    changed=subprocess.check_output(['git','-c',f'safe.directory={source}','-C',str(source),'diff','--name-only','HEAD'],text=True).splitlines()
    if commit!=A.SOURCE_COMMIT or set(changed)!={'serve/server.py','serve/web/app.js'}:raise SafetyError('DeepSeek source checkout changed outside its exact image/Monitor patches')
    if runtime.get('server_upstream_sha256')!=SERVER_UPSTREAM_SHA or runtime.get('server_sha256')!=SERVER_PATCHED_SHA or digest(source/'serve/server.py')!=SERVER_PATCHED_SHA:
        raise SafetyError('DeepSeek exact native-image hook changed')
    expected_exe=rt/'engine-packages'/'0.1.51-linux-x86_64-cuda-cuda13'/'strata-dsv41'
    if runtime.get('engine')!=str(expected_exe):raise SafetyError('Unexpected DeepSeek engine package path')
    verify_records(runtime['assets'],full=True);verify_records(runtime['python_metadata'],full=True)
    helper=Path(__file__).with_name('deepseek41_native_vision.py')
    if runtime.get('native_vision_helper_sha256')!=digest(helper):raise SafetyError('Installed native-image helper differs from this reviewed launcher')
    from deepseek41_cache_ui import WEB_UPSTREAM_SHA,WEB_PATCHED_SHA
    monitor=Path(__file__).with_name('deepseek41_cache_monitor.py')
    if (runtime.get('native_cache_monitor_version')!=1 or
            runtime.get('native_cache_monitor_helper_sha256')!=digest(monitor) or
            digest(source/'serve/deepseek41_cache_monitor.py')!=digest(monitor) or
            runtime.get('monitor_web_upstream_sha256')!=WEB_UPSTREAM_SHA or
            runtime.get('monitor_web_sha256')!=WEB_PATCHED_SHA or
            digest(source/'serve/web/app.js')!=WEB_PATCHED_SHA):
        raise SafetyError('DeepSeek native cache Monitor provenance mismatch; explicit stopped-session update required')
    wanted={str(pr/'models'/name):(size,checksum) for name,size,checksum,_ in A.asset_sources()}
    if {row['path'] for row in model['assets']}!=set(wanted):raise SafetyError('DeepSeek model asset set incomplete')
    for row in model['assets']:
        if (row['bytes'],row['sha256'])!=wanted[row['path']]:raise SafetyError('DeepSeek model checksum record differs from pinned source')
    verify_records(model['assets']);verify_records(model['pack_assets'],full=True)
    validate_geometry(pr/'pack/ds4.txt')
    config=pr/'config.json';verify_records([model['config']],full=True)
    cfg=json.loads(config.read_text())
    # Stock auto remains the unchanged default; the measured no-drafter option
    # requires an explicit stopped-session selection (never an arbitrary override).
    if cfg not in [canonical_config(root,runtime),canonical_config(root,runtime,draft_mode='none')]:
        raise SafetyError('DeepSeek native context/vision/quality/ownership configuration differs')
    return source,config,cfg


def require_state(root):
    expected=owner_home()/'.local/state/locallm-strata'/A.STATE_NAME
    if Path(root)!=expected:raise SafetyError('DeepSeek requires its separate state directory: '+str(expected))


def seed_hot_profile(root, state):
    import shutil
    state=Path(state)
    if state.is_symlink():raise SafetyError('DeepSeek state directory cannot be symlinked')
    state.mkdir(parents=True,exist_ok=True)
    if not state.is_dir() or state.stat().st_uid!=os.geteuid():raise SafetyError('DeepSeek state directory owner/type changed')
    state.chmod(0o700)
    dest=state/'expert-profile.bin'
    if dest.is_symlink():raise SafetyError('DeepSeek adaptive expert profile cannot be symlinked')
    if not dest.exists():
        source=profile_root(root)/'pack/expert-profile.bin'
        with dest.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600)
            with source.open('rb') as seed:shutil.copyfileobj(seed,stream)
    info=dest.stat()
    if info.st_uid!=os.geteuid() or info.st_mode&0o022 or not dest.is_file() or info.st_size>4*1024*1024:
        raise SafetyError('DeepSeek adaptive expert profile has unsafe ownership/type/size')


def capacity_gate():
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available<64*1024**3:raise SafetyError('DeepSeek native-1M startup needs at least 64 GiB available RAM on this four-GPU preset')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    totals=[int(v.strip()) for v in text.splitlines() if v.strip()]
    if len(totals)!=4 or min(totals)<24000:raise SafetyError('DeepSeek preset requires the four 24GB GPUs; no smaller silent fallback')
