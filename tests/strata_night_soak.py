#!/usr/bin/env python3
"""API-only bounded mixed-load soak of an exact already-loaded tuned model.
No lifecycle/settings/state reset, external tools, downloads or service actions.
"""
import argparse,concurrent.futures,json,os,threading,time,uuid
from pathlib import Path
import strata_night_client as Q

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--minutes',type=int,choices=[15,20,30,45],default=30)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if os.geteuid()!=1000 or a.out.exists() or Q.ROOT not in a.out.parents:ap.error('UID1000/fresh calibration output required')
    os.umask(0o077);Q.MODEL=a.model;a.out.mkdir(mode=0o700,parents=True)
    fixture=json.loads((Q.ROOT/'fixtures.json').read_text());messages={c['kind']:c['messages'][1]['content'] for c in fixture['cases'] if c['repetition']==1}
    Q.idle();before=Q.target()['activity']['requests'];start=time.monotonic();until=start+a.minutes*60
    rows=[];requests=0;cycle=0;peak=0
    def record(label,r):
        Q.save(a.out/(label+'.json'),r);rows.append({'label':label,'wall_s':r['wall_s'],'input':r['prompt_tokens'],'output':r['output_tokens'],
                                                  'complete':r['complete'],'drafted':r['timings'].get('draft_n',0)})
    while time.monotonic()<until:
        cycle+=1;tick=time.monotonic();Q.idle();nonce='soak-'+str(cycle)+'-'+uuid.uuid4().hex
        kind='half' if cycle%8==0 else 'medium' if cycle%4==0 else 'interactive'
        r=Q.stream(Q.msg(nonce,messages[kind]),cap=128);requests+=1;record('cycle-'+str(cycle)+'-'+kind,r)
        if cycle%3==0:
            barrier=threading.Barrier(2)
            def pair(i):
                barrier.wait(10)
                return Q.stream(Q.msg(nonce+'-pair-'+str(i),f'Begin with the exact tag SOAK{cycle}X{i}. Then list all integers 1 through 500.'),cap=256)
            Q.idle()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(pair,i) for i in (1,2)]
                while not all(f.done() for f in futures):peak=max(peak,Q.occupancy());time.sleep(.08)
                got=[f.result() for f in futures]
            for i,row in enumerate(got,1):
                if f'SOAK{cycle}X{i}' not in row['content'] or f'SOAK{cycle}X{3-i}' in row['content']:raise RuntimeError('Soak pair isolation failed')
                record('cycle-'+str(cycle)+'-pair-'+str(i),row)
            requests+=2
        if cycle%5==0:
            content=[{'type':'text','text':'Name each image color in order, separated by a comma. No explanation.'},
                     {'type':'image_url','image_url':{'url':Q.png((0,0,255))}},
                     {'type':'image_url','image_url':{'url':Q.png((255,0,0))}}]
            r=Q.stream([{'role':'user','content':content}],cap=64);requests+=1;ans=r['content'].lower()
            if not ('blue' in ans and 'red' in ans and ans.index('blue')<ans.index('red')):raise RuntimeError('Soak vision order failed')
            record('cycle-'+str(cycle)+'-vision',r)
        if cycle%10==0:
            r=Q.stream(Q.msg(nonce+'-high','Calculate 37 times 57 minus 10. State the final result.'),cap=1536,
                       extra={'reasoning_effort':'high','chat_template_kwargs':{'enable_thinking':True}});requests+=1
            if '2099' not in r['content'] or not r['reasoning']:raise RuntimeError('Soak high reasoning failed')
            record('cycle-'+str(cycle)+'-high',r)
        if cycle%7==0:
            r=Q.stream(Q.msg(nonce+'-known','What is 7 times 9? Reply only with the number.'),cap=16);requests+=1
            if r['content'].strip()!='63':raise RuntimeError('Soak known-working63 failed')
            record('cycle-'+str(cycle)+'-known',r)
        props=Q.get('/props')
        if props['default_generation_settings']['params'].get('n_predict')!=-1 or props.get('total_slots')!=2:
            raise RuntimeError('Soak output-default/slot contract changed')
        print('Soak cycle',cycle,'requests',requests,'elapsed_s',round(time.monotonic()-start,1),flush=True)
        # Warm/reuse behavior is intentional here, not a cache-free speed comparison.
        # A small cadence interval exercises cache parking and keeps resource sampling meaningful.
        delay=min(max(0,30-(time.monotonic()-tick)),max(0,until-time.monotonic()))
        time.sleep(delay)
    Q.idle();after=Q.target()['activity']['requests']
    if after-before!=requests:raise RuntimeError('Other activity arrived during soak; no user request was cancelled')
    if peak!=2:raise RuntimeError('Two occupied slots not observed during mixed soak')
    result={'phase':'SOAK-COMPLETE','model':a.model,'seconds':time.monotonic()-start,'cycles':cycle,'requests':requests,
            'observed_request_delta':after-before,'two_occupied_slots':peak,'complete_streams':len(rows),
            'vision':True,'high_reasoning':True,'no_global_output_cap':True,'rows':rows,
            'scope':'Bounded mixed warm-cache soak, not long-term stability or two simultaneous full-window qualification'}
    Q.save(a.out/'soak.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
