"""Separate operator-approved 0.2.20/0.1.66 option; frozen V3.1 model/features, own state, exact root grant."""
import copy,hashlib,json,os,stat
from pathlib import Path
from engine_safety import SafetyError
import unsloth_v31_v0159_profile as C
import strata_v0220_runtime as R
from unsloth_v31_v0159_tuned_profile import safe_data
ROOT=Path('/home/user/.local/share/localllm-strata')
REPO=Path(__file__).resolve().parent
A=C.A
VERSION=R.VERSION
PROFILE=HIVE_PROFILE=A.PROFILE+'-v0166'
OLD_NATIVE_STATE_NAME=PROFILE+'-trial'
STATE_NAME=PROFILE+'-yarn2-524k-trial'
CONTEXT=524288
UI_MAX_TOKENS=65536
TRIAL_ENV='STRATA_UNSLOTH_V31_V0166_TRIAL'
MODEL_ID=C.BASE.MODEL_ID+'-v0166'
APPROVAL=Path('/var/lib/localllm-strata/approvals/unsloth-v31-iq4xs-v0166.json')
READY=('READY CONFIG: Unsloth V3.1 IQ4_XS app0.2.20/engine0.1.66; Q8 PLE/projection ON4-44/BF16 CPU vision/stock MTP3; '
       'YaRN2/524288 per chat (trained262144)/UI64K/2slots/2groups/16384MiB/8histories/INT8-32768; compatible tuning/native kernels; bounded trial, not qualification.')
LOCKED='Strata0166 option LOCKED: exact source/runtime/config and independent root approval required.'
# Operator approved manual testing AFTER reboot, never another load on the faulted boot.
FAULT_BOOT_SHA256='b161f45d2cb47b93362fc6d5ca62231a9b455f702f756d4a070d61e5eebc1441'
def boot_digest():return hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest()
ENV={**C.ENV,'STRATA_PF_HC_REGION':'1','STRATA_PF_GROUP_SCRATCH':'0','STRATA_PW_WITH_BATCH':'1',
     'STRATA_EXP_MODE':'8','STRATA_LFUSE':'1','STRATA_HC_UPMIX':'1','STRATA_Q80_ROWS1':'1','STRATA_PLE_COPY_STREAM':'1',
     'STRATA_PLE_COMPACT':'0','STRATA_QFUSE':'0','STRATA_DEQUANT_SIDE':'0','STRATA_SPLIT_CPU_SHARE':'0',
     'STRATA_ROUTE_TAIL_SKIP':'0','STRATA_ADAPT_FETCH':'0','STRATA_VHEAD_BITPLANE':'0',
     'STRATA_DENSE_I8_TC':'1','STRATA_DENSE_I8_TC_MIN_T':'9'}
POLICIES=('unsloth_v31_v0166_profile.py','v0166_strata_launcher.py','strata_v0220_runtime.py','prepare_strata_v0220.py',
 'strata_v0166_ui_defaults.py','publish_v0166_defaults.py',
 'unsloth_v31_v0159_profile.py','unsloth_v31_v0159_tuned_profile.py','unsloth_v31_profile.py','unsloth_v31_assets.py','stage_unsloth_v31.py',
 'strata_v0213_runtime.py','strata_v0213_activation.py','strata_launcher.py','hive_profile_switch.py','engine_safety.py','hosting_lifecycle.py',
 'hive-llm-miner/profiles/'+PROFILE+'/h-run.sh')
def state_root(home=None):
 if home is not None and Path(home)!=Path('/home/user'):raise SafetyError('0166 requires exact operator home')
 return Path('/home/user/.local/state/locallm-strata')/STATE_NAME
def canonical_config(root=ROOT,home=None):
 if Path(root)!=ROOT:raise SafetyError('0166 data root differs')
 state=state_root(home);cfg=copy.deepcopy(C.canonical_config(ROOT,Path('/home/user')));base=R.folder(ROOT)
 cfg['args'][cfg['args'].index('--expert-profile')+1]=str(state/'expert-profile.bin')
 cfg['args'][cfg['args'].index('--max-context')+1]=str(CONTEXT)
 cfg['args']+=['--rope-scaling','yarn','--rope-scale','2','--yarn-orig-ctx','262144']
 cfg.update(exe=str(base/'payload/strata'),cwd=str(base/'source/terminal'),lib_dirs=[str(base/'payload')],layer_split='12,24,36',
            model_name=MODEL_ID,aliases=[PROFILE,C.MODEL_ID,C.PROFILE,C.MODEL_ID+'-tuned',C.PROFILE+'-tuned'],
            env=dict(ENV,STRATA_PARALLEL='2',STRATA_BATCH_GROUPS='2'),undeclared_tool_calls='call',engine_silence_s=0)
 cfg['vision']['exe']=str(base/'payload/strata-vision-cpu')
 return cfg
def validate_config(root,cfg,home=None):
 if cfg!=canonical_config(root,home):raise SafetyError('0166 frozen model/features/tuning/configuration differs')
 return cfg
def require_state(state):
 expected=state_root()
 if Path(state)!=expected or any(p.is_symlink() for p in (expected,*expected.parents)):raise SafetyError('0166 exact nonsymlink isolated state required')
 if expected.exists():
  s=expected.lstat()
  if not stat.S_ISDIR(s.st_mode) or s.st_uid!=1000 or s.st_mode&0o077:raise SafetyError('0166 state owner/mode changed')
def require_environment(state):
 require_state(state)
 required={TRIAL_ENV:'1','STRATA_RUNTIME':VERSION,'STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2','STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080',
 'PYTHONDONTWRITEBYTECODE':'1','PYTHONPYCACHEPREFIX':str(state_root()/'pycache'),'SETUPTOOLS_USE_DISTUTILS':'stdlib'}
 if any(os.environ.get(k)!=v for k,v in required.items()):raise SafetyError('0166 clean owner/runtime/state environment differs')
 allowed=set(required)|{'STRATA_STATE_ROOT','STRATA_DATA_ROOT'}
 if any(k.startswith('STRATA_') and k not in allowed for k in os.environ):raise SafetyError('0166 inherited STRATA override refused')
def approval_contract():
 cfg=canonical_config();base=R.folder(ROOT)
 return {'schema':1,'scope':'OPERATOR-APPROVED-STRATA0166-524K-UI64K-DEFAULT-20261124','blocked_boot_sha256':FAULT_BOOT_SHA256,'profile':PROFILE,'owner_uid':1000,'runtime':VERSION,'source_commit':R.PIN,
 'ui_app_original_sha256':R.UI.WEB_ORIGINAL_SHA256,'ui_app_patched_sha256':R.UI.WEB_PATCHED_SHA256,'ui_only_default_max_tokens':UI_MAX_TOKENS,
 'model_revision':A.REVISION,'ple_sha256':A.ASSETS[-1][4],'engine_sha256':R.OFFICIAL_FILES['strata'][1],
 'configuration_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'runtime_manifest_sha256':R.digest(base/'prepared.json'),'policy_sha256':{p:R.digest(REPO/p) for p in POLICIES}}
def authorize_activation():
 try:
  for p in (APPROVAL.parent,*APPROVAL.parent.parents):
   s=p.lstat()
   if not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or s.st_mode&0o022:raise SafetyError(LOCKED)
  if safe_data(APPROVAL,root_only=True)!=approval_contract():raise SafetyError(LOCKED)
  if boot_digest()==FAULT_BOOT_SHA256:raise SafetyError('Strata .66 manual test awaits a fresh healthy reboot; this faulted boot is blocked, no GPU retry')
  R.ready(ROOT);return 'trial'
 except (OSError,ValueError) as e:raise SafetyError(LOCKED) from e
def authorize_start(root,cfg,*,host,port,state):
 authorize_activation()
 if os.geteuid()!=1000 or host!='0.0.0.0' or type(port) is not int or port!=8080:raise SafetyError('0166 inference only UID1000/trusted-LAN8080')
 require_environment(state);validate_config(root,cfg);return 'trial'
def seed_profile(root,state):
 require_state(state);p=Path(state)/'expert-profile.bin';s=p.lstat()
 if p.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_uid!=1000 or s.st_mode&0o077:raise SafetyError('0166 private routing state missing/unsafe')
def check(root,full=False):
 import stage_unsloth_v31 as B
 require_state(state_root());B.check(root,full=full,pristine_state=False);source,_=R.ready(root)
 path=state_root()/'config.expected.json';validate_config(root,json.loads(R.checked_file(path).read_text()));seed_profile(root,state_root())
 return source,path,canonical_config(root),VERSION
def ready(root,state):require_environment(state);return check(root)
def prepare(root=ROOT):
 if os.geteuid()!=1000 or Path(root)!=ROOT:raise SafetyError('0166 preparation only exact UID1000/root')
 R.ready(ROOT);state=state_root();require_state(state)
 if state.exists():return check(root)
 state.mkdir(mode=0o700)
 seed=ROOT/'runtimes/0.1.46/source/data/expert-profile.bin'
 if seed.stat().st_size!=A.ROUTING_SEED_SIZE or R.digest(seed)!=A.ROUTING_SEED_SHA:raise SafetyError('Generic routing seed differs')
 for name,data in [('expert-profile.bin',seed.read_bytes()),('config.expected.json',json.dumps(canonical_config(root),indent=2).encode())]:
  with (state/name).open('xb') as f:os.fchmod(f.fileno(),0o600);f.write(data)
 return check(root)
