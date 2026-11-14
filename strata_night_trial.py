#!/usr/bin/env python3
"""Exact-approved temporary .59 adapter; delegates ALL live lifecycle work to strata_launcher.

Only this process binds its profile provider. Production modules/files/configs are not edited.
Fresh private state + a root-pinned case registry are mandatory; no env-only authorization.
"""
import copy, hashlib, json, os
from pathlib import Path
import re, stat, sys, time
import unsloth_v31_v0159_profile as C
import strata_launcher as L
import strata_v0213_runtime as R
from engine_safety import SafetyError

ROOT=Path('/home/user/.local/share/localllm-strata')
REPO=Path('/home/user/LocalLLM')
BUILD=ROOT/'runtimes/0.1.59-night-20261113'
APPROVAL=Path('/var/lib/localllm-strata/approvals/strata-v0159-night-20261113.json')
PREFIX='strata-v0159-night-20261113-'
POLICIES=('strata_night_trial.py','strata_launcher.py','engine_safety.py','hosting_lifecycle.py',
          'hive_profile_switch.py','unsloth_v31_v0159_profile.py','strata_v0213_runtime.py',
          'unsloth_v31_profile.py','stage_unsloth_v31.py','strata_night_controller.py',
          'strata_night_safety.py','tests/strata_night_client.py','strata_night_build.py')

def registry():
    def c(engine='q5mmq',env=None,split='auto'):return {'engine':engine,'env':env or {},'layer_split':split}
    scratch={'STRATA_PF_HC_REGION':'1','STRATA_PF_GROUP_SCRATCH':'1'}
    # Every delta is independently bounded here, then pinned in the root permission DATA.
    return {
      'official':c('official'),'control':c('control'),'q5':c(),
      'hc-region':c(env={'STRATA_PF_HC_REGION':'1'}),'scratch':c(env=scratch),
      'balanced':c(env=scratch,split='12,24,36'),
      'balance13':c(env=scratch,split='13,25,37'),
      'balance14':c(env=scratch,split='14,26,38'),
      # Runtime candidates are initially screened against the same fixed layout/memory baseline.
      **{name:c(env={**scratch,**delta},split='12,24,36') for name,delta in {
        'iq4-off':{'STRATA_IQ4_W4A8':'0'},'iq4-force':{'STRATA_IQ4_W4A8':'2'},
        'select-off':{'STRATA_SELECT_TWO_PASS':'0'},
        'gdn-tc':{'STRATA_GDN_TC':'1'},'gdn-conv':{'STRATA_GDN_CONVL2':'2'},
        'gdn-both':{'STRATA_GDN_TC':'1','STRATA_GDN_CONVL2':'2'},
        'hc-q8':{'STRATA_HC_Q8':'1'},'batch-pipeline':{'STRATA_PW_WITH_BATCH':'1'},
        'combined-gdn-hc':{'STRATA_GDN_TC':'1','STRATA_GDN_CONVL2':'2','STRATA_HC_Q8':'1'},
        'combined-pipeline':{'STRATA_GDN_TC':'1','STRATA_GDN_CONVL2':'2','STRATA_HC_Q8':'1','STRATA_PW_WITH_BATCH':'1'},
      }.items()}}

def policies():return {p:R.digest(REPO/p) for p in POLICIES}
def safe_approval(path=APPROVAL):
    for p in (path.parent,*path.parent.parents):
        st=p.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid!=0 or st.st_mode&0o022:raise SafetyError('Unsafe night approval parent')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd) as f:
        st=os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o644 or st.st_nlink!=1 or st.st_size>32768:
            raise SafetyError('Unsafe night approval file')
        d=json.load(f)
    if (d.get('scope')!='OPERATOR-AUTONOMOUS-OVERNIGHT-20261113' or d.get('cases')!=registry() or
            d.get('policy_sha256')!=policies() or d.get('owner_uid')!=1000 or
            d.get('expires',0)<time.time() or d.get('expires',0)>time.time()+8*3600):
        raise SafetyError('Night scope/source/cases/expiry changed; nothing started')
    return d

class TrialProfile:
    PROFILE=C.PROFILE;HIVE_PROFILE=C.HIVE_PROFILE;VERSION=C.VERSION;TRIAL_ENV=C.TRIAL_ENV;A=C.A;R=R
    READY='TEMPORARY EXACT-APPROVED OVERNIGHT TRIAL; frozen model/vision/native262K/2/2/8; not promotion'
    def __init__(self,case,run,approval=None):
        if case not in registry() or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',run):raise SafetyError('Unknown night case/run')
        self.case=case;self.run=run;self.spec=copy.deepcopy(registry()[case]);self.APPROVAL=approval or APPROVAL
        self.MODEL_ID=C.MODEL_ID+'-night-'+case
        self.ENV=dict(C.ENV,STRATA_PF_HC_REGION='0',STRATA_PF_GROUP_SCRATCH='0')
        self.ENV.update(self.spec['env'])
    def state_root(self):return Path('/home/user/.local/state/locallm-strata')/(PREFIX+self.run)
    def canonical_config(self,root=ROOT,home=None):
        if Path(root)!=ROOT:raise SafetyError('Night data root differs')
        cfg=copy.deepcopy(C.canonical_config(ROOT,Path('/home/user')))
        cfg['args'][cfg['args'].index('--expert-profile')+1]=str(self.state_root()/'expert-profile.bin')
        if self.spec['engine']!='official':
            folder=BUILD/self.spec['engine'];cfg['exe']=str(folder/'strata');cfg['lib_dirs']=[str(folder),'/usr/local/cuda-12.9/lib64']
        cfg.update(model_name=self.MODEL_ID,aliases=[self.MODEL_ID],env=dict(self.ENV,STRATA_PARALLEL='2',STRATA_BATCH_GROUPS='2'),layer_split=self.spec['layer_split'])
        return cfg
    def validate_config(self,root,cfg,home=None):
        if cfg!=self.canonical_config(root):raise SafetyError('Night config differs from exact frozen-feature case')
        return cfg
    def require_state(self,state):
        expected=self.state_root()
        if Path(state)!=expected or any(p.is_symlink() for p in (expected,*expected.parents)):raise SafetyError('Night requires its exact nonsymlink state')
        if expected.exists():
            s=expected.lstat()
            if not stat.S_ISDIR(s.st_mode) or s.st_uid!=1000 or s.st_mode&0o077:raise SafetyError('Unsafe night state owner/mode')
    def require_environment(self,state):
        self.require_state(state)
        wanted={'STRATA_NIGHT_CASE':self.case,'STRATA_NIGHT_RUN':self.run,self.TRIAL_ENV:'1',
                'STRATA_RUNTIME':'0.1.59','STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2',
                'STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080','PYTHONDONTWRITEBYTECODE':'1',
                'PYTHONPYCACHEPREFIX':str(self.state_root()/'pycache'),'SETUPTOOLS_USE_DISTUTILS':'stdlib'}
        if any(os.environ.get(k)!=v for k,v in wanted.items()):raise SafetyError('Night clean owner/state/runtime environment differs')
        allowed=set(wanted)|{'STRATA_STATE_ROOT','STRATA_DATA_ROOT'}
        if any(k.startswith('STRATA_') and k not in allowed for k in os.environ):raise SafetyError('Inherited STRATA tuning override refused')
    def authorize_activation(self):
        d=safe_approval(self.APPROVAL)
        kind=self.spec['engine'];cfg=self.canonical_config()
        if kind=='official':
            if d['engines']['official']!=R.OFFICIAL_FILES['strata'][1] or R.digest(Path(cfg['exe']))!=d['engines']['official']:
                raise SafetyError('Official night engine changed')
        else:
            info=json.loads((BUILD/kind/'build.json').read_text())
            if info.get('source_commit')!=R.PIN or info.get('q5_0_mmq')!=(kind=='q5mmq'):raise SafetyError('Night build provenance/capability differs')
            for name in ('strata','libstrata_kq.so'):
                p=BUILD/kind/name;s=p.lstat()
                if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=1000 or s.st_mode&0o022 or R.digest(p)!=d['engines'][kind][name]:
                    raise SafetyError('Night engine/dependency bytes or owner changed')
        return 'trial'
    def authorize_start(self,root,cfg,*,host,port,state):
        self.authorize_activation()
        if os.geteuid()!=1000 or host!='0.0.0.0' or port!=8080:raise SafetyError('Night inference only UID1000/trusted-LAN8080')
        self.require_environment(state);self.validate_config(root,cfg);return 'trial'
    def check(self,root,full=False):
        self.authorize_activation();self.require_state(self.state_root())
        import stage_unsloth_v31 as B
        B.check(root,full=full,pristine_state=False);source,_=R.ready(root)
        path=self.state_root()/'config.expected.json'
        self.validate_config(root,json.loads(path.read_text()));self.seed_profile(root,self.state_root())
        return source,path,self.canonical_config(root),self.VERSION
    def ready(self,root,state):self.require_environment(state);return self.check(root)
    def seed_profile(self,root,state):
        self.require_state(state);p=Path(state)/'expert-profile.bin';s=p.lstat()
        if not stat.S_ISREG(s.st_mode) or p.is_symlink() or s.st_uid!=1000 or s.st_mode&0o077:raise SafetyError('Unsafe private night seed')
    def prepare(self):
        self.authorize_activation()
        if os.geteuid()!=1000:raise SafetyError('Night state prepared only UID1000')
        state=self.state_root();self.require_state(state)
        state.mkdir(mode=0o700,exist_ok=False)
        seed=ROOT/'runtimes/0.1.46/source/data/expert-profile.bin'
        if R.digest(seed)!=C.A.ROUTING_SEED_SHA:raise SafetyError('Generic seed changed')
        for name,data in [('expert-profile.bin',seed.read_bytes()),('config.expected.json',json.dumps(self.canonical_config(),indent=2).encode())]:
            with (state/name).open('xb') as f:os.fchmod(f.fileno(),0o600);f.write(data)
        self.check(ROOT)

def from_environment():return TrialProfile(os.environ.get('STRATA_NIGHT_CASE',''),os.environ.get('STRATA_NIGHT_RUN',''))
def main():
    p=from_environment()
    if sys.argv[1:]==['--prepare']:p.prepare();return 0
    if '--stop' in sys.argv[1:]:
        # Fresh controller-provided digest of this live identity + exact task lease.
        # Never stop a replacement merely because it occupies this case's state path.
        state=L.read_state();lease=L.host_lifecycle.read_lease()
        expected=os.environ.get('STRATA_NIGHT_STOP_IDENTITY_SHA256','')
        binding={k:(state or {}).get(k) for k in ('identity','hosting_lease_id','allowed_executables')}
        actual=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (not re.fullmatch(r'[a-f0-9]{64}',expected) or expected!=actual or not state or
                not isinstance(lease,dict) or lease.get('phase')!='held' or
                lease.get('lease_id')!=state.get('hosting_lease_id') or not L.family(state)):
            raise SafetyError('Night stop lacks fresh exact process/lease authority; no replacement stopped')
    # Per-process dependency injection only, with a stricter, root-approved, exact config provider.
    # All original launch/stop/rental/lease/child-identity/pidfd checks remain implemented by L.
    L.v3159=p
    return L.main()
if __name__=='__main__':raise SystemExit(main())
