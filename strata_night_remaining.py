#!/usr/bin/env python3
"""Fresh phase for six never-started cases after a PRE-SPAWN guard refusal.
No completed controller/lease/identity replay; all previous results retained.
"""
import argparse,json,os,time
from pathlib import Path
import strata_night_controller as K
import strata_night_trial as T
from strata_night_session import Session

CAL=K.CAL
PRIVATE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113/screen-part2')
REMAINING=('gdn-conv','gdn-both','hc-q8','batch-pipeline','combined-gdn-hc','combined-pipeline')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    if not args.execute:print(json.dumps({'remaining':REMAINING,'mutation':False}));return
    if os.geteuid()!=0:raise RuntimeError('Lifecycle administrator only')
    os.umask(0o077);T.safe_approval()
    if not (K.PRIVATE/'restored.private.json').is_file() or (CAL/'screen-results.json').exists():
        raise RuntimeError('Previous phase not cleanly restored or screens already complete; no replay')
    previous=[json.loads(p.read_text()) for p in CAL.glob('*-screen-summary.json')]
    if {x['case'] for x in previous}!=set(T.registry())-set(REMAINING):raise RuntimeError('Partial inventory differs; no inferred tests')
    samples=[json.loads(x) for x in (CAL/'balanced-screen-speed-resources.jsonl').read_text().splitlines()]
    minimum=[min(x['gpu_free_mib'][i] for x in samples) for i in range(4)]
    if minimum[0]<1024 or min(minimum[1:])<512:raise RuntimeError('Measured pipeline memory gate not satisfied; no extra allocation')
    PRIVATE.mkdir(mode=0o700,exist_ok=False);K.PRIVATE=PRIVATE
    K.save(PRIVATE/'plan.private.json',{'at':time.time(),'cases':REMAINING,'measured_pipeline_minimum_free_mib':minimum,
        'reason':'Prior gate refused BEFORE any gdn-conv runtime was spawned; previous original .59 restored.',
        'wait_only_no_gate_filtering':True,'remaining_cases_have_never_loaded':True})
    reference=json.loads((CAL/'original-live-speed/benchmark.json').read_text())
    qualityref=json.loads((CAL/'original-live-quality/quality.json').read_text())
    s=Session();results=[]
    try:
        s.acquire()
        for case in REMAINING:
            p=s.start(case,case+'-remaining-screen');run=p.run
            q=s.client('quick-quality',run+'-quality');b=s.client('speed',run+'-speed')
            ids={y['id'] for y in q['cases']};ref={'cases':[x for x in qualityref['cases'] if x['id'] in ids]}
            row={'case':case,'run':run,'benchmark':b,'quick_quality':q,
                 'new_quick_losses':K.no_new_losses(q,ref),'rank':K.ranking(b,reference)}
            results.append(row);K.save(CAL/(run+'-summary.json'),row)
            print('SCREEN',case,json.dumps(row['rank']),'new quick losses',row['new_quick_losses'],flush=True)
            s.stop()
        K.save(CAL/'screen-results.json',{'phase':'SCREENS-COMPLETE','results':previous+results,'promotion_performed':False,
            'pre_spawn_refusal_preserved':True,'all18_cases_measured':True,'no_unknown_process_stopped':True})
    finally:
        try:K.A.hardware()
        except Exception as e:
            K.N.abort(str(e));K.save(PRIVATE/'hardware-hold.private.json',{'error':str(e),'lease_id':s.lease,'no_reset_or_retry':True});raise
        if s.before:s.restore()
if __name__=='__main__':main()
