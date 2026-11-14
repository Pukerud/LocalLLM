#!/usr/bin/env python3
"""Fresh post-candidate reference bracket, not a replay of a completed controller.
The earlier controller's two blocks use repeated order, not literal ABBA. This final
fresh generic reference brackets both candidate blocks against beginning/end controls.
No setting change, new variant, approval grant or permanent deployment.
"""
import argparse,json,os,time
from pathlib import Path
import strata_night_controller as K
import strata_night_trial as T
from strata_night_session import Session
from strata_night_confirm import pooled,qualifies
from engine_safety import SafetyError

CAL=K.CAL
PRIVATE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113/final-reference-bracket')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    original=json.loads((CAL/'confirmation-results.json').read_text())
    if not a.execute:print(json.dumps({'phase':'fresh-final-reference-bracket','mutation':False}));return
    if os.geteuid()!=0 or original.get('phase')!='CONFIRMATION-COMPLETE' or not (PRIVATE.parent/'confirmation/restored.private.json').exists():
        raise SafetyError('Previous confirmation not completed/restored; no bracket start')
    if time.time()>K.DEADLINE-5400:raise SafetyError('Final deployment/restoration reserve too short; no new bracket')
    os.umask(0o077);T.safe_approval();PRIVATE.mkdir(mode=0o700,exist_ok=False);K.PRIVATE=PRIVATE
    K.save(PRIVATE/'plan.private.json',{'at':time.time(),'scope':'Fresh final official .59 reference, same generic seed/fixtures/all sampling penalties; no state reset/copy',
        'prior_order':'referenceA,candidatesA,referenceB,candidatesB','final_order':'...,freshReferenceC',
        'main_comparison':'six beginning/end reference samples versus six candidate samples; middle reference retained as drift check'})
    s=Session()
    try:
        s.acquire();s.start('official','confirm-final-reference-c');label='confirm-final-reference-c-speed'
        s.client('speed',label,reps=3);s.stop()
        reference=pooled(['confirm-reference-a-speed',label])
        rows=[];qualified=[]
        for old in original['results']:
            row=dict(old)
            if not row.get('pooled_benchmark'):rows.append(row);continue
            rank=K.ranking(row['pooled_benchmark'],reference)
            ok=qualifies(rank) and row.get('solo_mtp_gate') and not row.get('new_losses') and row.get('native_long_and_features')
            row.update(rank=rank,speed_gate=bool(ok),final_reference_bracket=True)
            rows.append(row)
            if ok:qualified.append(row)
        qualified.sort(key=lambda x:x['rank']['geomean_latency_speedup'],reverse=True)
        winner=qualified[0]['case'] if qualified else None
        if qualified:
            near=[x for x in qualified if qualified[0]['rank']['geomean_latency_speedup']/x['rank']['geomean_latency_speedup']<=1.03]
            conservative=next((x for x in near if all(T.registry()[x['case']]['env'].get(k,'0')=='0'
                for k in ('STRATA_GDN_TC','STRATA_GDN_CONVL2','STRATA_HC_Q8'))),None)
            if conservative:winner=conservative['case']
        report={'phase':'CONFIRMATION-COMPLETE','qualified':[x['case'] for x in qualified],
                'winner_candidate':winner,'selection_rule':original['selection_rule'],'references':reference,
                'middle_reference_drift_check':original['references'],'results':rows,'permanent_deployment_performed':False,
                'control_order':'Fresh referenceA -> candidateA blocks -> fresh referenceB -> candidateB blocks -> fresh referenceC; main rates useA/C bracket',
                'limitations':original['limitations']+['Bracketing reduces temporal drift; this is not a randomized/exhaustive factorial trial']}
        K.save(CAL/'confirmation-bracketed-results.json',report)
        print('Final bracket gates:',[(x['case'],x.get('speed_gate')) for x in rows],'candidate:',winner,flush=True)
    finally:
        try:K.A.hardware()
        except Exception as e:
            K.N.abort(str(e));K.save(PRIVATE/'hardware-hold.private.json',{'error':str(e),'lease_id':s.lease,'no_reset_or_retry':True});raise
        if s.before:s.restore()
if __name__=='__main__':main()
