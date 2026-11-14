#!/usr/bin/env python3
"""Bounded overnight lifecycle administrator. Fresh authority only; never replay this task.
Default is READ-ONLY planning. --execute is the operator-approved overnight run.
Promotion is deliberately separate from screening; original Hive is restored on normal exit.
"""
import argparse, csv, hashlib, json, math, os
from pathlib import Path
import secrets, stat, subprocess, sys, time
import urllib.request
import hive_profile_switch as S
import hosting_lifecycle as H
import strata_launcher as L
import strata_v0213_activation as A
import unsloth_v31_v0159_profile as C
import strata_night_trial as T
import strata_night_safety as N
from engine_safety import SafetyError, proc, same_process

PRIVATE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113')
CAL=N.ROOT
CLIENT=Path('/home/user/LocalLLM/tests/strata_night_client.py')
DEADLINE=1794636000  # 2026-11-14 06:00 UTC; reserve confirmation/deployment before morning.

def save(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())

def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8080'+path,timeout=5) as f:return json.load(f)

def natural_idle(state,seconds=600):
    end=time.monotonic()+seconds
    while True:
        if not same_process(state['identity'],proc(state['identity']['pid'])):raise SafetyError('Frontend changed while waiting; no adoption/stop')
        S.docker_empty();A.hardware()
        d=get('/metrics')['live']
        if type(d.get('running')) is not int:raise SafetyError('Unknown request occupancy')
        if d['running']==0:return
        if time.monotonic()>end:raise SafetyError('Requests remain active; no cancellation/stop')
        time.sleep(.5)

def environment(p):
    return {'HOME':'/home/user','USER':'user','LOGNAME':'user','PATH':'/usr/local/cuda-12.9/bin:/usr/bin:/bin',
            'PYTHONDONTWRITEBYTECODE':'1','SETUPTOOLS_USE_DISTUTILS':'stdlib',
            'PYTHONPYCACHEPREFIX':str(p.state_root()/'pycache'),p.TRIAL_ENV:'1','STRATA_RUNTIME':'0.1.59',
            'STRATA_DATA_ROOT':str(T.ROOT),'STRATA_STATE_ROOT':str(p.state_root()),
            'STRATA_PARALLEL':'2','STRATA_BATCH_GROUPS':'2','STRATA_HOST':'0.0.0.0','STRATA_PORT':'8080',
            'STRATA_NIGHT_CASE':p.case,'STRATA_NIGHT_RUN':p.run}

def user_command(argv,env=None):
    env=env or {'HOME':'/home/user','USER':'user','LOGNAME':'user','PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
    return ['/usr/sbin/runuser','-u','user','--','/usr/bin/env','-i',*[k+'='+str(v) for k,v in env.items()],'/usr/bin/python3',*map(str,argv)]

def approve():
    if os.geteuid()!=0:raise SafetyError('Root must publish bounded permission')
    if T.APPROVAL.exists():T.safe_approval();return
    builds={}
    for name in ('control','q5mmq'):
        folder=T.BUILD/name;d=json.loads((folder/'build.json').read_text())
        if d['source_commit']!=T.R.PIN or d['q5_0_mmq']!=(name=='q5mmq'):raise SafetyError('Build provenance mismatch')
        builds[name]={n:T.R.digest(folder/n) for n in ('strata','libstrata_kq.so')}
        if any(builds[name][n]!=d['files'][n]['sha256'] for n in builds[name]):raise SafetyError('Build seal drift')
    builds['official']=T.R.OFFICIAL_FILES['strata'][1]
    d={'schema':1,'scope':'OPERATOR-AUTONOMOUS-OVERNIGHT-20261113','owner_uid':1000,
       'expires':min(DEADLINE+3600,time.time()+7*3600),'cases':T.registry(),'engines':builds,'policy_sha256':T.policies()}
    fd=os.open(T.APPROVAL,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'w') as f:os.fchmod(f.fileno(),0o644);json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
    T.safe_approval();digest=T.R.digest(T.APPROVAL)
    save(PRIVATE/('permission-'+digest[:12]+'-published.private.json'),{'at':time.time(),'permission_sha256':digest})

def check_offline():
    revision=hashlib.sha256(json.dumps(T.policies(),sort_keys=True).encode()).hexdigest()[:10]
    p=T.TrialProfile('official','offline-e2e-'+revision)
    env=environment(p)
    for args in (['--prepare'],['--check-ready','--profile',p.PROFILE,'--runtime','0.1.59','--parallel','2','--batch-groups','2']):
        with (CAL/('offline-'+revision+'-'+('prepare' if args[0]=='--prepare' else 'ready')+'.log')).open('x') as f:
            subprocess.run(user_command([Path(T.__file__),*args],env),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    print('Real UID1000 exact-grant private-state/check-ready E2E passed; zero GPU startup.',flush=True)

class Session:
    def __init__(self):self.lease=secrets.token_hex(32);self.worker=None;self.case=None;self.state=None;self.before=None
    def acquire(self):
        with H.launch_reservation():
            C.authorize_activation();A.check_owner_approval();A.hardware();S.inspect_owned_workload()
            if S.JOURNAL.exists() or H.read_lease() or os.readlink(S.CURRENT)!='profiles/'+C.HIVE_PROFILE:raise SafetyError('Original Hive selection/lease/journal changed')
            original=json.loads((C.state_root()/'server.json').read_text())
            if original.get('profile')!=C.PROFILE or not original.get('unsloth_v31_v0159'):raise SafetyError('Original .59 is not exact owned model')
            natural_idle(original)
            self.before={'at':time.time(),'lease_id':self.lease,'state':original,'selection':os.readlink(S.CURRENT),
                         'run':H.RUN.exists(),'stop':H.STOP.exists(),'admission':H.service_state(),
                         'approval_sha256':T.R.digest(C.APPROVAL),'boot':N.BOOT}
            if not self.before['run'] or self.before['stop'] or self.before['admission']!='active':raise SafetyError('Original supervision state differs')
            save(PRIVATE/'before.private.json',self.before)
            S.stop_owned_v31(C.HIVE_PROFILE)
            H.begin(8080,require_new_lease=True,lease_id=self.lease)
            lease=H.read_lease()
            if lease.get('lease_id')!=self.lease or lease.get('phase')!='held':raise SafetyError('Exact fresh task lease not established')
            # begin() necessarily sees the stop marker CREATED BY THE OWNED STOP ABOVE.
            # Restore the proven pre-task miner/marker state, not that intermediate state.
            lease.update(miner_was_running=True,stop_was_present=False,stop_value=None,
                         task='strata-v0159-overnight-20261113',owned_pre_stop_proof=str(PRIVATE/'before.private.json'))
            H.write_lease(lease)
            save(PRIVATE/'lease-held.private.json',{'at':time.time(),'lease_id':self.lease})
        print('Exact .59 stopped/fully drained; fresh task lease held; admission active.',flush=True)
    def start(self,case,run):
        if self.worker is not None:raise SafetyError('Refuse overlapping trial workers')
        if (CAL/'SAFETY-ABORT').exists() or time.time()>DEADLINE:raise SafetyError('Safety/time bound reached; no next start')
        p=T.TrialProfile(case,run);env=environment(p)
        with (CAL/(run+'-prepare.log')).open('x') as log:
            subprocess.run(user_command([Path(T.__file__),'--prepare'],env),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        with H.launch_reservation() as lock:
            A.hardware();S.docker_empty()
            lease=H.read_lease()
            if not lease or lease.get('lease_id')!=self.lease or lease.get('phase')!='held':raise SafetyError('Task lease changed')
            if S.llm_processes() or H.gpu_jobs() or H.active_engine()!='none':raise SafetyError('LLM/GPU/manual work present; no trial start')
            env.update(HOSTLLM_LAUNCH_LEASE_ID=self.lease,HOSTLLM_LAUNCH_LOCK_FD=str(lock.fileno()))
            log=(CAL/(run+'-launcher.log')).open('x')
            args=[Path(T.__file__),'--quickstart','--profile',p.PROFILE,'--runtime','0.1.59','--parallel','2','--batch-groups','2']
            self.worker=subprocess.Popen(user_command(args,env),stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),));log.close()
            self.case=p
            # Retain the own unreaped administrative child; trust ONLY its live UID1000 launcher child.
            deadline=time.monotonic()+240
            while not (p.state_root()/'server.json').exists():
                A.hardware();S.docker_empty()
                if self.worker.poll() is not None:raise SafetyError('Own trial launcher exited before identity capture; see log')
                if time.monotonic()>deadline:raise SafetyError('No exact frontend handoff; no duplicate/retry')
                time.sleep(.5)
            state=json.loads((p.state_root()/'server.json').read_text())
            parent=proc(state['identity']['ppid']);admin=proc(self.worker.pid)
            if (not admin or admin['ppid']!=os.getpid() or admin['exe']!='/usr/sbin/runuser' or not parent or parent['uid']!=1000 or
                    parent['ppid']!=admin['pid'] or str(Path(T.__file__)) not in parent['cmd'] or
                    state.get('hosting_lease_id')!=self.lease or state.get('allowed_executables',[None,None])[1]!=p.canonical_config()['exe'] or
                    state.get('identity',{}).get('uid')!=1000 or not same_process(state['identity'],proc(state['identity']['pid']))):
                raise SafetyError('Fresh own-child/frontend/lease/executable/UID ancestry proof failed; no adoption')
            self.state=state;save(PRIVATE/(run+'-spawn.private.json'),{'state':state,'admin_identity':admin,'launcher_identity':parent})
        ready=time.monotonic()+900
        with N.Watch(run+'-startup',self.state):
            while True:
                if self.worker.poll() is not None:raise SafetyError('Exact trial child exited during startup; no retry')
                try:
                    d=get('/v1/status')
                    if (d.get('loaded') is True and d.get('model')==p.MODEL_ID and d.get('engine')=='0.1.59' and
                            d.get('concurrency',{}).get('serving')==2 and d.get('context',{}).get('max_positions')==262144):break
                except (OSError,ValueError):pass
                if time.monotonic()>ready:raise SafetyError('Exact trial target did not load; no generation/duplicate')
                time.sleep(1)
        natural_idle(self.state);print('READY',run,case,flush=True)
        self.client('known',run+'-known')
        return p
    def client(self,mode,label,reps=1):
        out=CAL/label
        with N.Watch(label,self.state):
            with (CAL/(label+'.log')).open('x') as log:
                child=subprocess.Popen(user_command([CLIENT,'--mode',mode,'--model',self.case.MODEL_ID,'--out',out,'--reps',str(reps)]),stdout=log,stderr=subprocess.STDOUT)
                until=time.monotonic()+2400
                while child.poll() is None:
                    if (CAL/'SAFETY-ABORT').exists():
                        # The API-only client sees this and closes only its own HTTP connection.
                        child.wait(timeout=30);raise SafetyError('Safety observer stopped own client; no further inference')
                    if time.monotonic()>until:
                        N.abort('Own bounded client exceeded time budget');child.wait(timeout=30);raise SafetyError('Client budget exhausted')
                    time.sleep(.5)
                if child.returncode:raise RuntimeError('Client failure: '+label+'; see private output')
        return json.loads((out/'complete.json').read_text())
    def stop(self):
        if self.worker is None:return
        p=self.case
        with H.launch_reservation():
            A.hardware();S.docker_empty()
            if self.state and same_process(self.state['identity'],proc(self.state['identity']['pid'])):
                natural_idle(self.state)
                current=json.loads((p.state_root()/'server.json').read_text())
                if not same_process(self.state['identity'],current['identity']) or current.get('hosting_lease_id')!=self.lease:
                    raise SafetyError('Trial changed before stop; no replacement stopped')
                binding={k:current.get(k) for k in ('identity','hosting_lease_id','allowed_executables')}
                env=environment(p);env['STRATA_NIGHT_STOP_IDENTITY_SHA256']=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                with (CAL/(p.run+'-stop.log')).open('x') as log:
                    subprocess.run(user_command([Path(T.__file__),'--stop','--profile',p.PROFILE,'--runtime','0.1.59'],env),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            self.worker.wait(timeout=180)
            H.wait_idle(timeout=90)
            if S.llm_processes() or H.gpu_jobs() or H.active_engine()!='none':raise SafetyError('Trial did not fully drain; no replacement killed')
        print('DRAINED',p.run,flush=True);self.worker=None;self.case=None;self.state=None
    def restore(self):
        if self.worker is not None:self.stop()
        with H.launch_reservation():
            A.hardware();S.docker_empty()
            if os.readlink(S.CURRENT)!=self.before['selection'] or T.R.digest(C.APPROVAL)!=self.before['approval_sha256']:
                raise SafetyError('Original selection/permission changed; no stale restoration')
            lease=H.read_lease()
            if lease is None:return
            if lease.get('lease_id')!=self.lease:raise SafetyError('Do not release another task lease')
            H.release(expected_lease_id=self.lease)
        # Observe normal Hive startup; NEVER duplicate or replay the activation transaction.
        deadline=time.monotonic()+1200
        while True:
            A.hardware();S.docker_empty()
            try:
                status=get('/v1/status')
                if status.get('loaded') and status.get('model')==C.MODEL_ID:
                    state=json.loads((C.state_root()/'server.json').read_text())
                    if same_process(self.before['state']['identity'],state['identity']):raise SafetyError('Restoration did not produce a fresh frontend')
                    S.inspect_owned_workload();N.check(state);save(PRIVATE/'restored.private.json',{'at':time.time(),'state':state,'status':status});break
            except (OSError,ValueError):pass
            if time.monotonic()>deadline:raise SafetyError('Normal Hive restoration not ready; no duplicate startup')
            time.sleep(2)
        print('Original Hive[22]/.59 restored, new owned session, admission active.',flush=True)

def no_new_losses(report,reference):
    got={c['id']:c['passed'] for c in report['cases']}
    return [c['id'] for c in reference['cases'] if c['passed'] and not got.get(c['id'],False)]

def ranking(bench,reference):
    ratios={k:reference['medians'][k]['wall_s']/bench['medians'][k]['wall_s'] for k in reference['medians']}
    pair=reference['pair']['wall_s']/bench['pair']['wall_s']
    geo=math.exp(sum(math.log(x) for x in [*ratios.values(),pair])/5)
    return {'geomean_latency_speedup':geo,'latency_speedups':ratios,'pair_speedup':pair,
            'half_prefill_speedup':bench['medians']['half']['prefill']/reference['medians']['half']['prefill'],
            'minimum_decode_ratio':min(bench['medians'][k]['decode']/reference['medians'][k]['decode'] for k in reference['medians'])}

def screen(cases):
    s=Session();s.acquire();results=[]
    reference=json.loads((CAL/'original-live-speed/benchmark.json').read_text())
    qualityref=json.loads((CAL/'original-live-quality/quality.json').read_text())
    try:
        for index,case in enumerate(cases):
            if time.time()>DEADLINE-3600:print('Time reserve reached: stop new screens.',flush=True);break
            # Do not spend extra pipeline VRAM unless the balanced screen actually has margin.
            if case in ('batch-pipeline','combined-pipeline'):
                logs=list(CAL.glob('balanced-screen-speed-resources.jsonl'))
                if not logs:
                    results.append({'case':case,'excluded':'No measured balanced-layout memory margin'});continue
                d=[json.loads(x) for x in logs[0].read_text().splitlines()]
                if min(x['gpu_free_mib'][0] for x in d)<1024 or min(min(x['gpu_free_mib'][1:]) for x in d)<512:
                    results.append({'case':case,'excluded':'Conservative extra-pipeline headroom gate (<1024MiB first GPU or <512MiB others)'});continue
            run=case+'-screen'
            try:
                s.start(case,run)
                q=s.client('quick-quality',run+'-quality')
                b=s.client('speed',run+'-speed')
                row={'case':case,'run':run,'benchmark':b,'quick_quality':q,'new_quick_losses':no_new_losses(q,{'cases':[x for x in qualityref['cases'] if x['id'] in {y['id'] for y in q['cases']}]}),
                     'rank':ranking(b,reference)}
                results.append(row);save(CAL/(run+'-summary.json'),row)
                print('SCREEN',case,json.dumps(row['rank']),'new quick losses',row['new_quick_losses'],flush=True)
                s.stop()
            except Exception as e:
                results.append({'case':case,'error':str(e)})
                save(PRIVATE/(run+'-error.private.json'),{'error':str(e),'at':time.time()})
                # Fail closed: unexpected startup/functional failure ends this batch, not an automatic GPU retry.
                raise
        save(CAL/'screen-results.json',{'phase':'SCREENS-COMPLETE','results':results,'promotion_performed':False})
    finally:
        # No original-model restart is attempted after a real hardware/kernel fault.
        try:A.hardware()
        except Exception as e:
            N.abort(str(e));save(PRIVATE/'hardware-hold.private.json',{'error':str(e),'lease_id':s.lease,'no_reset_or_retry':True});raise
        s.restore()
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--approve-offline',action='store_true');ap.add_argument('--cases',nargs='+',choices=list(T.registry()),default=list(T.registry()));a=ap.parse_args()
    if not (a.execute or a.approve_offline):print(json.dumps({'cases':T.registry(),'deadline_utc':'2026-11-14T06:00:00Z','mutation':False},indent=2));return
    if os.geteuid()!=0:ap.error('Narrow lifecycle administrator requires root')
    os.umask(0o077)
    if a.approve_offline:approve();check_offline();return
    T.safe_approval();screen(a.cases)
if __name__=='__main__':main()
