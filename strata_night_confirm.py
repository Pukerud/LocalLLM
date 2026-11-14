#!/usr/bin/env python3
"""Fresh confirmation phase, never a replay of completed screen identities/leases.
Creates a new private phase namespace; restores original Hive22 on normal completion.
No automatic permanent deployment. Requires completed screens and exact live original .59.
"""
import argparse,hashlib,json,math,os
from pathlib import Path
import statistics,subprocess,time
import strata_night_controller as K
import strata_night_trial as T
import strata_night_safety as N
from engine_safety import SafetyError
from strata_night_session import Session

CAL=N.ROOT
REPO=Path('/home/user/LocalLLM')
PHASE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113/confirmation')

def auxiliary(s,script,label,args):
    out=CAL/label
    with N.Watch(label,s.state):
        with (CAL/(label+'.log')).open('x') as log:
            argv=[REPO/'tests'/script,*args]
            subprocess.run(K.user_command(argv),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    if script=='strata_night_holdout.py':return json.loads((out/'holdout.json').read_text())
    return json.loads((CAL/'features'/label/'result.json').read_text())

def holdout(s,label):return auxiliary(s,'strata_night_holdout.py',label,['--model',s.case.MODEL_ID,'--out',str(CAL/label)])
def features(s,label):return auxiliary(s,'strata_night_features.py',label,['--case',s.case.case,'--run',s.case.run,'--phase',label])

def pooled(labels):
    values={k:[] for k in ('short','interactive','medium','half')};pairs=[]
    for label in labels:
        folder=CAL/label
        for p in folder.glob('*-r*.json'):
            d=json.loads(p.read_text())
            if d.get('kind') in values:values[d['kind']].append(d)
        pairs.append(json.loads((folder/'pair.json').read_text()))
    med={k:{'prefill':statistics.median(x['timings']['prompt_per_second'] for x in rows),
            'decode':statistics.median(x['timings']['predicted_per_second'] for x in rows),
            'wall_s':statistics.median(x['wall_s'] for x in rows),'ttft_s':statistics.median(x['ttft_s'] for x in rows),
            'samples':len(rows)} for k,rows in values.items()}
    samples=[x for rows in values.values() for x in rows]
    mtp={'samples':len(samples),'active_samples':sum(x['timings'].get('draft_n',0)>0 for x in samples),
         'drafted':sum(x['timings'].get('draft_n',0) for x in samples),
         'accepted':sum(x['timings'].get('draft_n_accepted',0) for x in samples)}
    return {'medians':med,'pair':{'wall_s':statistics.median(x['wall_s'] for x in pairs)},'labels':labels,'solo_mtp':mtp}

def qualifies(rank):
    return (rank['half_prefill_speedup']>=1.10 and rank['geomean_latency_speedup']>=1.10 and
            min(rank['latency_speedups'].values())>=1/1.05 and rank['pair_speedup']>=1/1.05 and
            rank['minimum_decode_ratio']>=.95)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--max-candidates',type=int,choices=[1,2,3,4],default=3);a=ap.parse_args()
    screen=json.loads((CAL/'screen-results.json').read_text())
    if screen.get('phase')!='SCREENS-COMPLETE':raise SafetyError('Screens incomplete; no confirmation scheduled')
    # Single-repetition screen is noisy: retain decode down to0.88 for investigation,
    # but FINAL repeated no-regression gate remains>=0.95. No criterion is relaxed for deployment.
    eligible=[x for x in screen['results'] if 'rank' in x and x['case'] not in ('official','control') and
              not x.get('new_quick_losses') and x['rank']['half_prefill_speedup']>=1.08 and
              x['rank']['geomean_latency_speedup']>=1.07 and x['rank']['minimum_decode_ratio']>=.88 and
              min(x['rank']['latency_speedups'].values())>=.90 and x['rank']['pair_speedup']>=.90]
    eligible.sort(key=lambda x:x['rank']['geomean_latency_speedup'],reverse=True)
    chosen=eligible[:a.max_candidates]
    # Keep one exact-kernel/memory/layout finalist even if several arithmetic-changing
    # variants lead the noisy screen. Otherwise a shared numerical regression could
    # discard a nearly-as-fast conservative winner without ever testing it broadly.
    conservative=next((x for x in eligible if x['case'] in ('balanced','balance13','balance14')),None)
    if conservative and conservative not in chosen:
        chosen=(chosen[:-1] if chosen else [])+[conservative]
    if not a.execute:
        print(json.dumps({'shortlist':[{'case':x['case'],'rank':x['rank']} for x in chosen],'mutation':False},indent=2));return
    if os.geteuid()!=0:raise SafetyError('Only the lifecycle administrator may confirm')
    os.umask(0o077);PHASE.mkdir(mode=0o700,exist_ok=False);K.PRIVATE=PHASE
    # These newly added API-only checks are frozen before running any holdout.
    K.save(PHASE/'plan.private.json',{'at':time.time(),'shortlist':[x['case'] for x in chosen],
        'screen_sha256':hashlib.sha256((CAL/'screen-results.json').read_bytes()).hexdigest(),
        'auxiliary_sha256':{p:T.R.digest(REPO/'tests'/p) for p in ('strata_night_holdout.py','strata_night_features.py')},
        'confirmation_source_sha256':T.R.digest(Path(__file__)),
        'wait_session_source_sha256':T.R.digest(REPO/'strata_night_session.py'),
        'final_gates':{'half_prefill':1.10,'geometric_latency':1.10,'decode_ratio':.95,'max_latency_regression':.05},
        'default_unchanged':True,'state_copied_or_reset':False})
    s=Session();results=[];references=[];qualified=[]
    try:
        s.acquire()
        s.start('official','confirm-reference-a')
        label='confirm-reference-a-speed';s.client('speed',label,reps=3);references.append(label)
        qref=s.client('quality','confirm-reference-a-quality');href=holdout(s,'confirm-reference-a-holdout')
        prior=json.loads((CAL/'original-live-quality/quality.json').read_text())
        # Preserve every task passed by either the original warm reference or fresh official reference.
        old={x['id']:x['passed'] for x in prior['cases']}
        qref['cases']=[{'id':x['id'],'passed':x['passed'] or old.get(x['id'],False)} for x in qref['cases']]
        s.stop()
        for selected in chosen:
            case=selected['case']
            if time.time()>K.DEADLINE-5400:break
            run='confirm-'+case+'-a';s.start(case,run)
            b=s.client('speed',run+'-speed',reps=3)
            q=s.client('quality',run+'-quality');h=holdout(s,run+'-holdout')
            losses=K.no_new_losses(q,qref)+K.no_new_losses(h,href)
            row={'case':case,'first_speed_label':run+'-speed','quality':q,'holdout':h,'new_losses':losses}
            results.append(row);K.save(CAL/(run+'-summary.json'),row)
            print('CONFIRM-QUALITY',case,q['passed'],h['passed'],'new losses',losses,flush=True)
            s.stop()
        # Reversed-order second reference; actual native-window/functional checks also qualify reference.
        s.start('official','confirm-reference-b')
        label='confirm-reference-b-speed';s.client('speed',label,reps=3);references.append(label)
        s.client('long','confirm-reference-b-long');features(s,'confirm-reference-b-features');s.stop()
        ref=pooled(references)
        K.save(CAL/'confirmed-reference.json',ref)
        for row in results:
            if row['new_losses']:continue
            case=row['case'];run='confirm-'+case+'-b'
            s.start(case,run);s.client('speed',run+'-speed',reps=3)
            b=pooled([row['first_speed_label'],run+'-speed']);rank=K.ranking(b,ref)
            mtp_ok=b['solo_mtp']['active_samples']==b['solo_mtp']['samples']
            row.update(pooled_benchmark=b,rank=rank,solo_mtp_gate=mtp_ok,speed_gate=qualifies(rank) and mtp_ok)
            if row['speed_gate']:
                s.client('long',run+'-long');features(s,run+'-features')
                row['native_long_and_features']=True;qualified.append(row)
            K.save(CAL/(run+'-summary.json'),row)
            print('CONFIRMED',case,json.dumps(rank),'speed gate',row['speed_gate'],flush=True)
            s.stop()
        qualified.sort(key=lambda x:x['rank']['geomean_latency_speedup'],reverse=True)
        winner=qualified[0]['case'] if qualified else None
        if qualified:
            close=[x for x in qualified if qualified[0]['rank']['geomean_latency_speedup']/x['rank']['geomean_latency_speedup']<=1.03]
            def conservative(x):
                env=T.registry()[x['case']]['env']
                return all(env.get(k,'0')=='0' for k in ('STRATA_GDN_TC','STRATA_GDN_CONVL2','STRATA_HC_Q8'))
            simpler=next((x for x in close if conservative(x)),None)
            if simpler:winner=simpler['case']
        report={'phase':'CONFIRMATION-COMPLETE','qualified':[x['case'] for x in qualified],
                'winner_candidate':winner,'selection_rule':'Prefer exact-kernel/memory/layout/scheduling recipe without GDN/HC precision changes when within3% of fastest qualifying result','references':ref,
                'results':results,'permanent_deployment_performed':False,
                'limitations':['72 objective tasks, not universal intelligence equivalence',
                               'one populated native window at a time, not two simultaneous full windows',
                               'six sequential samples per workload per confirmed arm; two pair observations']}
        K.save(CAL/'confirmation-results.json',report)
    finally:
        try:K.A.hardware()
        except Exception as e:
            N.abort(str(e));K.save(PHASE/'hardware-hold.private.json',{'error':str(e),'lease_id':s.lease,'no_reset_or_retry':True});raise
        if s.before:s.restore()
    print('CONFIRMATION COMPLETE. Candidate:',report['winner_candidate'],'Original Hive22 restored.',flush=True)
if __name__=='__main__':main()
