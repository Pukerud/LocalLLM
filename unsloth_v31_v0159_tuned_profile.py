"""Distinct, fail-closed overnight winner profile. UNREGISTERED/LOCKED until qualification.
Original Hive22/runtime/state/approval stay separate. No lifecycle or automatic activation here.
A frozen reviewed recipe is configuration DATA; only a separate exact root approval permits start.
"""
import copy,hashlib,json,os,stat
from pathlib import Path
import unsloth_v31_v0159_profile as C
import strata_night_trial as T
import strata_v0213_runtime as R
from engine_safety import SafetyError

PROFILE=HIVE_PROFILE=C.PROFILE+'-tuned'
STATE_NAME=PROFILE+'-trial'
MODEL_ID=C.MODEL_ID+'-tuned'
VERSION=C.VERSION
A=C.A
TRIAL_ENV='STRATA_UNSLOTH_V31_V0159_TUNED_TRIAL'
APPROVAL=Path('/var/lib/localllm-strata/approvals/unsloth-v31-iq4xs-v0159-tuned.json')
RECIPE=Path(__file__).resolve().with_name('strata_v0159_tuned_recipe.json')
READY=('READY CONFIG: Unsloth V3.1 IQ4_XS .59 tuned; Q8 PLE/projection ON4-44/BF16 CPU vision/stock MTP3; '
       'native262144 per chat/2slots/2groups/16384MiB/8histories/INT8-32768; '
       'bounded overnight qualification only, not universal quality/full-window concurrency assurance.')
LOCKED='Tuned .59 profile is LOCKED: exact qualified recipe/runtime/source/root permission required.'
REPO=Path(__file__).resolve().parent
POLICIES=('unsloth_v31_v0159_tuned_profile.py','v0159_tuned_strata_launcher.py','strata_night_trial.py',
          'unsloth_v31_v0159_profile.py','unsloth_v31_profile.py','unsloth_v31_assets.py','stage_unsloth_v31.py',
          'strata_v0213_runtime.py','strata_v0213_activation.py','strata_launcher.py','engine_safety.py','hosting_lifecycle.py','hive_profile_switch.py',
          'hive-llm-miner/profiles/'+PROFILE+'/h-run.sh')

def encoded(d):return json.dumps(d,sort_keys=True,separators=(',',':')).encode()
def safe_data(path,maxbytes=32768,root_only=False):
    for p in (path,*path.parents):
        if p.is_symlink():raise SafetyError(LOCKED+' Symlinked data path.')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as f:
        s=os.fstat(f.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_uid not in ((0,) if root_only else (0,1000)) or s.st_mode&0o022 or s.st_nlink!=1 or s.st_size>maxbytes:
            raise SafetyError(LOCKED+' Unsafe data owner/type/mode/size/link.')
        if root_only and stat.S_IMODE(s.st_mode)!=0o644:raise SafetyError(LOCKED+' Approval must be root0644.')
        return json.load(f)

def recipe():
    try:d=safe_data(RECIPE)
    except (OSError,ValueError) as e:raise SafetyError(LOCKED) from e
    if (set(d)!=set(('schema','case','spec','source_commit','build','dependencies','qualification')) or d.get('schema')!=1 or
            d.get('case') not in T.registry() or d['case'] in ('official','control') or
            d.get('spec')!=T.registry()[d['case']] or d.get('source_commit')!=R.PIN):
        raise SafetyError(LOCKED+' Recipe differs from bounded registry/source.')
    qualification=d.get('qualification') or {}
    if set(qualification)!=set(('path','sha256')) or qualification['path']!='results/strata-v0159-overnight-qualification-20261114.json':
        raise SafetyError(LOCKED+' Qualification reference differs.')
    qpath=REPO/qualification['path']
    if R.digest(qpath)!=qualification['sha256']:raise SafetyError(LOCKED+' Qualification receipt changed.')
    q=safe_data(qpath,maxbytes=262144)
    if (q.get('phase')!='QUALIFICATION-PASSED' or q.get('case')!=d['case'] or
            q.get('speed_gate') is not True or q.get('new_quality_losses')!=[] or
            q.get('native_long_and_features') is not True):raise SafetyError(LOCKED+' Qualification not passed.')
    return d

def environment():
    d=recipe();env=dict(C.ENV,STRATA_PF_HC_REGION='0',STRATA_PF_GROUP_SCRATCH='0');env.update(d['spec']['env']);return env

# Imported before qualification is safe: no selector integration or start authority exists.
# A serving launcher imports this afresh AFTER the frozen recipe has been installed.
try:ENV=environment()
except (SafetyError,OSError):ENV=dict(C.ENV,STRATA_PF_HC_REGION='0',STRATA_PF_GROUP_SCRATCH='0')

def state_root(home=None):
    home=Path('/home/user') if home is None else Path(home)
    return home/'.local/state/locallm-strata'/STATE_NAME

def provider():
    d=recipe();p=T.TrialProfile(d['case'],'permanent-winner')
    p.PROFILE=PROFILE;p.HIVE_PROFILE=HIVE_PROFILE;p.MODEL_ID=MODEL_ID;p.TRIAL_ENV=TRIAL_ENV;p.ENV=environment();p.APPROVAL=APPROVAL
    p.state_root=state_root;p.authorize_activation=authorize_activation;p.require_environment=require_environment
    original_config=p.canonical_config
    def compatible_config(root=T.ROOT,home=None):
        cfg=original_config(root,home)
        cfg['aliases']=[MODEL_ID,PROFILE,C.MODEL_ID,C.PROFILE]
        cfg['undeclared_tool_calls']='call'
        return cfg
    p.canonical_config=compatible_config
    return p

def canonical_config(root,home=None):
    if home is not None and Path(home)!=Path('/home/user'):raise SafetyError('Tuned profile requires exact operator home')
    return provider().canonical_config(root,home)
def validate_config(root,cfg,home=None):
    if cfg!=canonical_config(root,home):raise SafetyError('Tuned frozen model/features/recipe configuration changed')
    return cfg
def require_state(state):
    # Shutdown must remain possible after a recipe/permission fault. This fixed
    # state boundary never depends on mutable selection DATA or start authority.
    expected=state_root()
    if Path(state)!=expected or any(p.is_symlink() for p in (expected,*expected.parents)):
        raise SafetyError('Tuned requires its exact nonsymlink isolated state')
    if expected.exists():
        s=expected.lstat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid!=1000 or s.st_mode&0o077:
            raise SafetyError('Tuned private state owner/type/mode changed')
def require_environment(state):
    require_state(state)
    required={TRIAL_ENV:'1','STRATA_RUNTIME':VERSION,'STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2',
              'STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080','PYTHONDONTWRITEBYTECODE':'1',
              'SETUPTOOLS_USE_DISTUTILS':'stdlib','PYTHONPYCACHEPREFIX':str(state_root()/'pycache')}
    if any(os.environ.get(k)!=v for k,v in required.items()):raise SafetyError('Tuned exact clean owner/runtime/state environment differs')
    allowed=set(required)|{'STRATA_STATE_ROOT','STRATA_DATA_ROOT'}
    if any(k.startswith('STRATA_') and k not in allowed for k in os.environ):raise SafetyError('Tuned inherited STRATA override refused')

def check_runtime():
    d=recipe();folder=T.BUILD/d['spec']['engine'];build=d['build']
    if set(build)!=set(('strata','libstrata_kq.so','build.json')):raise SafetyError('Tuned build seal entries differ')
    for name,row in build.items():
        if set(row)!=set(('bytes','sha256')):raise SafetyError('Tuned build seal schema differs')
        p=R.checked_file(folder/name)
        if p.stat().st_size!=row['bytes'] or R.digest(p)!=row['sha256']:raise SafetyError('Tuned build payload changed: '+name)
    b=json.loads((folder/'build.json').read_text())
    if b.get('source_commit')!=R.PIN or b.get('q5_0_mmq')!=(d['spec']['engine']=='q5mmq'):
        raise SafetyError('Tuned native build source/capability differs')
    deps=d['dependencies']
    if not isinstance(deps,list) or not deps or len({x.get('path') for x in deps})!=len(deps):
        raise SafetyError('Tuned dependency list is empty/ambiguous')
    names={Path(x.get('path','')).name for x in deps}
    if not all(any(n.startswith(prefix) for n in names) for prefix in ('libcudart.so.12','libcublas.so.12','libcublasLt.so.12','libcuda.so.')):
        raise SafetyError('Tuned CUDA/runtime/driver dependency coverage is incomplete')
    for row in deps:
        if set(row)!=set(('path','bytes','sha256')):raise SafetyError('Tuned dependency schema differs')
        p=Path(row['path'])
        if not p.is_absolute() or p.is_symlink():raise SafetyError('Tuned dependency must use exact resolved path')
        s=p.stat()
        if not stat.S_ISREG(s.st_mode) or s.st_uid!=0 or s.st_mode&0o022 or s.st_size!=row['bytes'] or R.digest(p)!=row['sha256']:
            raise SafetyError('Tuned system dependency changed; no automatic rebuild/upgrade/retry')
    return d

def approval_contract():
    d=recipe();cfg=canonical_config(T.ROOT,Path('/home/user'))
    return {'schema':1,'scope':'OPERATOR-APPROVED-OVERNIGHT-QUALIFIED-TRIAL','profile':PROFILE,'owner_uid':1000,
            'runtime':VERSION,'source_commit':R.PIN,'model_revision':A.REVISION,'ple_sha256':A.ASSETS[-1][4],
            'configuration_sha256':hashlib.sha256(encoded(cfg)).hexdigest(),'recipe_sha256':R.digest(RECIPE),
            'policy_sha256':{p:R.digest(REPO/p) for p in POLICIES}}

def authorize_activation():
    try:
        for p in (APPROVAL.parent,*APPROVAL.parent.parents):
            s=p.lstat()
            if not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or s.st_mode&0o022:raise SafetyError(LOCKED+' Unsafe approval directory.')
        value=safe_data(APPROVAL,root_only=True)
        if value!=approval_contract():raise SafetyError(LOCKED+' Permission differs from exact policy/config/assets.')
        check_runtime();return 'trial'
    except (OSError,ValueError) as e:raise SafetyError(LOCKED) from e

def authorize_start(root,cfg,*,host,port,state):
    authorize_activation()
    if os.geteuid()!=1000 or host!='0.0.0.0' or type(port) is not int or port!=8080:
        raise SafetyError('Tuned inference only UID1000/trusted-LAN8080')
    require_environment(state);validate_config(root,cfg);return 'trial'

def check(root,full=False):
    # Offline immutable readiness is NOT activation permission. Core start calls
    # authorize_activation before this and authorize_start again before any spawn.
    import stage_unsloth_v31 as B
    p=provider();p.require_state(state_root());check_runtime()
    B.check(root,full=full,pristine_state=False);source,_=R.ready(root)
    path=state_root()/'config.expected.json'
    validate_config(root,json.loads(path.read_text()));p.seed_profile(root,state_root())
    return source,path,canonical_config(root),VERSION
def ready(root,state):require_environment(state);return check(root)
def seed_profile(root,state):return provider().seed_profile(root,state)
def prepare(root):
    if Path(root)!=T.ROOT or os.geteuid()!=1000:raise SafetyError('Tuned preparation requires exact root/UID1000')
    check_runtime();R.ready(root)
    p=provider()
    # Only this disposable PREPARATION object's no-GPU method is used. The module's
    # live authorizer is never replaced and no permission file is written here.
    p.authorize_activation=lambda:'offline-data-readiness-only'
    return p.prepare()
