#!/usr/bin/env python3
"""Explicit final publication/activation steps for a QUALIFIED overnight winner.
No code editing, updater, GPU reset, rental stop, historical controller replay or numeric kill.
Default is read-only status; each mutable phase has its own fresh proof/exclusive output fence.
"""
import argparse,hashlib,json,os,secrets,stat,subprocess,sys,time,urllib.request
from pathlib import Path
import hive_profile_switch as S
import hosting_lifecycle as H
import strata_launcher as L
import strata_v0213_activation as A
import unsloth_v31_v0159_profile as C
import unsloth_v31_v0159_tuned_profile as U
import strata_night_controller as K
import strata_night_safety as N
import strata_night_trial as T
from engine_safety import SafetyError,proc,same_process

PRIVATE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113/integration')
REPO=Path('/home/user/LocalLLM')

def save(name,value):K.save(PRIVATE/name,value)
def load(name):return json.loads((PRIVATE/name).read_text())
def data_publish(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'w') as f:os.fchmod(f.fileno(),0o644);json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8080'+path,timeout=5) as f:return json.load(f)
def idle(state,timeout=600):
    deadline=time.monotonic()+timeout
    while True:
        A.hardware();S.docker_empty()
        if not same_process(state['identity'],proc(state['identity']['pid'])):raise SafetyError('Frontend changed while waiting; no adoption/stop')
        live=get('/metrics')['live'];keys=('running','queued','waiting','admitting','outside_slots')
        if any(type(live.get(k)) is not int for k in keys):raise SafetyError('Unknown request/admission occupancy; no stop')
        if all(live[k]==0 for k in keys):return
        if time.monotonic()>deadline:raise SafetyError('Requests remain active; none cancelled/stopped')
        time.sleep(.5)
def state(profile):return json.loads((profile.state_root()/'server.json').read_text())
def permission_as_owner(module):
    code='import sys;sys.path.insert(0,"/home/user/LocalLLM");import '+module+' as P;P.authorize_activation()'
    subprocess.run(K.user_command(['-c',code]),check=True,timeout=120,capture_output=True)
def confirm_holder():
    d=load('before-source.private.json');p=proc(d['holder']['pid'])
    if (not same_process(d['holder'],p) or str(Path(__file__)) not in p['cmd'] or '--hold' not in p['cmd'] or
            (PRIVATE/'release').exists()):raise SafetyError('Fresh integration reservation holder unavailable; no publication')
    A.hardware();S.docker_empty();return d

def hold():
    U.recipe();U.check_runtime();PRIVATE.mkdir(mode=0o700,exist_ok=False)
    with H.launch_reservation():
        A.hardware();S.inspect_owned_workload();C.authorize_activation()
        if H.read_lease() or S.JOURNAL.exists() or os.readlink(S.CURRENT)!='profiles/'+C.HIVE_PROFILE:raise SafetyError('Original22 selection/lease/journal differs')
        before=state(C);idle(before)
        d={'at':time.time(),'holder':proc(os.getpid()),'original_approval':json.loads(C.APPROVAL.read_text()),
           'original_approval_sha256':T.R.digest(C.APPROVAL),'original_state':before,
           'temporary_grant_sha256':T.R.digest(T.APPROVAL) if T.APPROVAL.exists() else None,
           'selection':os.readlink(S.CURRENT),'admission':H.service_state(),'boot':N.BOOT,
           'cloud_hashes':{n:T.R.digest(Path('/hive-config')/n) for n in ('rig.conf','wallet.conf','nvidia-oc.conf')}}
        save('before-source.private.json',d);save('lock-ready.private.json',{'at':time.time(),'reservation_held':True})
        deadline=time.monotonic()+2400
        while not (PRIVATE/'release').exists():
            A.hardware();S.docker_empty()
            if time.monotonic()>deadline:raise SafetyError('Integration reservation time limit; no automatic deployment')
            time.sleep(2)
    save('lock-released.private.json',{'at':time.time()})

def grants():
    before=confirm_holder();old=before['original_approval'];new=C.approval_contract()
    allowed={'strata_launcher.py','engine_safety.py','hive_profile_switch.py'}
    changes={k for k in old['policy_sha256'] if old['policy_sha256'][k]!=new['policy_sha256'].get(k)}
    if (not changes or not changes<=allowed or {k:v for k,v in old.items() if k!='policy_sha256'}!={k:v for k,v in new.items() if k!='policy_sha256'} or
            set(old['policy_sha256'])!=set(new['policy_sha256'])):
        raise SafetyError('Original permission changed beyond reviewed core integration; nothing refreshed')
    if T.R.digest(C.APPROVAL)!=before['original_approval_sha256']:raise SafetyError('Another operator changed original permission; no overwrite')
    C.APPROVAL.rename(PRIVATE/'original-v0159-approval.audit.json');data_publish(C.APPROVAL,new)
    C.authorize_activation();permission_as_owner('unsloth_v31_v0159_profile')
    # Offline permanent state is NEW/generic-only; no trial/live adaptation/history is copied.
    with (PRIVATE/'prepare-permanent.log').open('x') as log:
        subprocess.run(K.user_command([REPO/'v0159_tuned_strata_launcher.py','--prepare-profile']),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    if U.APPROVAL.exists() or U.APPROVAL.is_symlink():raise SafetyError('Tuned permission already exists; no overwrite/replay')
    data_publish(U.APPROVAL,U.approval_contract());U.authorize_activation();permission_as_owner('unsloth_v31_v0159_tuned_profile')
    if T.APPROVAL.exists():
        if T.R.digest(T.APPROVAL)!=before['temporary_grant_sha256']:raise SafetyError('Temporary grant changed; not archived')
        T.APPROVAL.rename(PRIVATE/'completed-night-scope.audit.json')
    save('grants-published.private.json',{'at':time.time(),'original_configuration_unchanged':True,
        'reviewed_original_source_changes':sorted(changes),'tuned_case':U.recipe()['case'],'real_uid1000_permission_checks':True,
        'temporary_scope_archived':True,'no_lifecycle_action':True})

def release():
    confirm_holder();save('release',{'at':time.time()})
    until=time.monotonic()+30
    while not (PRIVATE/'lock-released.private.json').exists():
        if time.monotonic()>until:raise SafetyError('Reservation holder did not release')
        time.sleep(.25)

def publish():
    import install_retained_profiles as P
    value=P.publish();save('inactive-publication.private.json',value);print(json.dumps(value,indent=2))

def ready(label,previous=None):
    deadline=time.monotonic()+1200
    while True:
        A.hardware();S.docker_empty()
        try:
            d=get('/v1/status')
            if (d.get('loaded') is True and d.get('engine')==U.VERSION and d.get('model')==U.MODEL_ID and
                    d.get('context',{}).get('max_positions')==262144 and d.get('concurrency',{}).get('serving')==2):
                current=state(U);S.inspect_owned_workload();N.check(current)
                if (current.get('profile')!=U.PROFILE or current.get('parallel')!=2 or current.get('batch_groups')!=2 or
                        current.get('allowed_executables',[None,None])[1]!=U.canonical_config(T.ROOT)['exe']):
                    raise SafetyError('Exact tuned profile/count/executable identity differs')
                if previous and (same_process(previous['identity'],current['identity']) or previous.get('hive_launch_id_sha256')==current.get('hive_launch_id_sha256')):
                    raise SafetyError('Owned restart did not produce new frontend/new token')
                env=(Path('/proc')/str(current['identity']['pid'])/'environ').read_bytes().split(b'\0')
                tokens=[x.split(b'=',1)[1].decode() for x in env if x.startswith(b'HIVE_STRATA_LAUNCH_ID=')]
                if len(tokens)!=1:raise SafetyError('Live Hive token unavailable; never guessed/adopted')
                L._verified_hive_process_parent(current['identity'],tokens[0],current.get('hive_launcher_identity'))
                save(label,{'at':time.time(),'state':current,'status':d,'live_parent_token_proof':True,'all_uid1000':True})
                return current
        except (OSError,ValueError):pass
        if time.monotonic()>deadline:raise SafetyError('Normal tuned Hive startup not ready; NO duplicate/retry')
        time.sleep(2)

def activate():
    if (PRIVATE/'before-activation.private.json').exists():raise SafetyError('Activation already attempted; never replay')
    with H.launch_reservation():
        A.hardware();S.inspect_owned_workload();C.authorize_activation();U.authorize_activation()
        if os.readlink(S.CURRENT)!='profiles/'+C.HIVE_PROFILE:raise SafetyError('Original22 no longer selected; no adoption/switch')
        current=state(C);idle(current)
        save('before-activation.private.json',{'at':time.time(),'state':current,'selection':os.readlink(S.CURRENT)})
        S.switch_profile(U.HIVE_PROFILE)
    ready('initial-tuned-ready.private.json')
    print('Tuned normal Hive profile loaded; no duplicated startup.',flush=True)

def restart():
    if (PRIVATE/'before-owned-restart.private.json').exists():raise SafetyError('Owned restart already attempted; never replay')
    with H.launch_reservation():
        A.hardware();S.inspect_owned_workload();U.authorize_activation()
        if H.read_lease() or S.JOURNAL.exists() or os.readlink(S.CURRENT)!='profiles/'+U.HIVE_PROFILE or H.STOP.exists():
            raise SafetyError('Fresh tuned owned-stop/marker/lease selection differs')
        previous=state(U);idle(previous)
        save('before-owned-restart.private.json',{'at':time.time(),'state':previous,'stop_before':False,'run_before':H.RUN.exists()})
        S.stop_owned_v31(U.HIVE_PROFILE)
        save('owned-stop-complete.private.json',{'at':time.time(),'old_session_drained':True,
             'admission_active':H.service_state()=='active','no_lease_created':H.read_lease() is None,
             'normal_supervision_to_be_observed_without_duplicate_start':True})
    # OctaSpace's legitimate regular miner-start supervision has continued throughout
    # the task (observed every minute). Let IT restore the selected qualified miner.
    # No miner-start command, new pause lease, replacement stop or duplicate launch.
    ready('owned-restarted-tuned-ready.private.json',previous)
    print('Exact tuned owned stop/drain and normal new-token/new-frontend restart passed.',flush=True)

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group();g.add_argument('--hold',action='store_true');g.add_argument('--grants',action='store_true');g.add_argument('--release',action='store_true');g.add_argument('--publish',action='store_true');g.add_argument('--activate',action='store_true');g.add_argument('--restart',action='store_true');a=ap.parse_args()
    if not any(vars(a).values()):print(json.dumps({'profile':U.HIVE_PROFILE,'recipe_present':U.RECIPE.exists(),'approval_present':U.APPROVAL.exists(),'mutation':False}));return
    if os.geteuid()!=0:ap.error('Narrow lifecycle/publication administrator only')
    os.umask(0o077)
    if a.hold:hold()
    elif a.grants:grants()
    elif a.release:release()
    elif a.publish:publish()
    elif a.activate:activate()
    elif a.restart:restart()
if __name__=='__main__':main()
