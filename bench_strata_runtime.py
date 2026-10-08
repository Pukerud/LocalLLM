#!/usr/bin/env python3
"""Bounded live Strata comparison. No lifecycle/service operations; preserved native context and sampling.
Records task/TTFT/engine decode/prefill separately, queued pairs vs concurrent pairs, and observed GPU/RAM use.
"""
import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib


def image_uri(left, right):
    colors={'red':b'\xff\x00\x00','blue':b'\x00\x00\xff'}
    def chunk(k,d):return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d))
    raw=(b'\x00'+colors[left]*128+colors[right]*128)*128
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',256,128,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(png).decode()


class Client:
    def __init__(self, base, model):self.base=base.rstrip('/');self.model=model
    def get(self,path):
        with urllib.request.urlopen(self.base+path,timeout=6) as r:return json.load(r)
    def post(self,body,stream=False):
        payload={'model':self.model,'temperature':1.0,'top_p':0.95,'top_k':20,'reasoning_effort':'high',**body}
        payload['stream']=stream
        req=urllib.request.Request(self.base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        try:
            response=urllib.request.urlopen(req,timeout=180)
        except urllib.error.HTTPError as error:
            detail=error.read().decode('utf8','replace')[:2048]
            raise RuntimeError(f'HTTP {error.code}: {detail}') from error
        with response:
            if not stream:
                data=json.load(response);return data,time.perf_counter()-start
            ttft=None;content=[];reasoning=[];usage=None;timings=None;finish=None;chunks=0
            for raw in response:
                line=raw.decode('utf8').strip()
                if not line.startswith('data: '):continue
                if line=='data: [DONE]':break
                event=json.loads(line[6:]);chunks+=1
                if 'error' in event:raise RuntimeError(event['error'])
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    text=delta.get('content') or '';think=delta.get('reasoning_content') or ''
                    if (text or think) and ttft is None:ttft=time.perf_counter()-start
                    content.append(text);reasoning.append(think);finish=choice.get('finish_reason') or finish
                usage=event.get('usage') or usage;timings=event.get('timings') or timings
        elapsed=time.perf_counter()-start
        if not usage or not usage.get('completion_tokens') or ttft is None:
            raise RuntimeError('Missing real streamed token/usage evidence')
        return {'seconds':elapsed,'ttft_seconds':ttft,'usage':usage,'timings':timings,'finish_reason':finish,
                'answer_chars':sum(map(len,content)),'reasoning_chars':sum(map(len,reasoning)),
                'answer_tail':''.join(content)[-180:],'sse_chunks':chunks}


def functional(c, slots, results=None):
    if results is None:results=[]
    def check(name,body,validate):
        d,seconds=c.post({**body,'seed':1234});m=d['choices'][0]['message'];passed=validate(m)
        row={'name':name,'passed':passed,'seconds':seconds,'usage':d.get('usage'),'timings':d.get('timings')}
        results.append(row)
        if not passed:raise RuntimeError('Functional failure '+name+': '+json.dumps(m))
    check('known_working_math',{'messages':[{'role':'user','content':'What is 6 times 7? Finish with ANSWER=42.'}],
          'max_tokens':256,'reasoning_budget_tokens':128},lambda m:'ANSWER=42' in m.get('content',''))
    check('json',{'messages':[{'role':'user','content':'Return JSON with exactly total 42 and city Oslo.'}],
          'max_tokens':128,'reasoning_effort':'none','response_format':{'type':'json_object'}},
          lambda m:json.loads(m['content'])=={'total':42,'city':'Oslo'})
    tool={'type':'function','function':{'name':'get_weather','description':'Get weather for a city',
        'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}
    check('one_tool',{'messages':[{'role':'user','content':'Call get_weather exactly once for Oslo, with no text answer.'}],
          'tools':[tool],'tool_choice':{'type':'function','function':{'name':'get_weather'}},'parallel_tool_calls':False,
          'reasoning_effort':'none','max_tokens':128},lambda m:len(m.get('tool_calls',[]))==1 and
          m['tool_calls'][0]['function']['name']=='get_weather' and
          json.loads(m['tool_calls'][0]['function']['arguments'])=={'city':'Oslo'})
    def code_valid(m):
        text=m.get('content','').strip()
        if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
        tree=ast.parse(text)
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
        fn=tree.body[0];body=fn.body
        if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant):body=body[1:]
        if fn.name!='add' or [v.arg for v in fn.args.args]!=['a','b'] or len(body)!=1 or not isinstance(body[0],ast.Return):return False
        expr=body[0].value
        return isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.Add) and isinstance(expr.left,ast.Name) and isinstance(expr.right,ast.Name) and {expr.left.id,expr.right.id}=={'a','b'}
    check('code',{'messages':[{'role':'user','content':'Write only Python code defining add(a, b) that returns a+b. No example call or markdown.'}],
          'max_tokens':384,'reasoning_budget_tokens':128},code_valid)
    def vision(left,right):
        body={'messages':[{'role':'user','content':[{'type':'text','text':'Return JSON naming the color on each half: keys left and right, lowercase names.'},
              {'type':'image_url','image_url':{'url':image_uri(left,right)}}]}],
              'max_tokens':128,'reasoning_effort':'none','seed':1234,'response_format':{'type':'json_object'}}
        d,seconds=c.post(body);m=d['choices'][0]['message'];passed=json.loads(m['content'])=={'left':left,'right':right}
        return {'name':'vision_'+left,'passed':passed,'seconds':seconds,'usage':d.get('usage'),'timings':d.get('timings')}
    if slots>1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(lambda pair:vision(*pair),[('red','blue'),('blue','red')]))
        results+=rows
    else:results += [vision('red','blue'),vision('blue','red')]
    if not all(x['passed'] for x in results):raise RuntimeError('Swapped-image correctness failed')
    return results


def prompt(case, repetition, member):
    # Distinct early prefix prevents live conversation/prefix cache from becoming an uncached-prefill claim.
    nonce=f'fixture_{case}_{repetition}_{member}'
    if case=='code':
        return nonce+'\nExplain and implement a robust Python LRU cache with capacity limits, get/put, and tests. '+\
               'Include the reasoning behind eviction and several edge cases. Write at least 500 words or equivalent code.'
    ledger='\n'.join(f'entry {i:03}: batch {i%13} stores {i*7+19} units with a checksum of {i*11+31}.' for i in range(200))
    return nonce+'\n'+ledger+'\nWrite a detailed 500-word technical explanation of how to process this ledger safely. '+\
           'Cover validation, reproducibility, indexing, concurrency and error handling; do not merely repeat the entries.'


def overlap_check(c):
    """Bounded long/long/short admission regression, not a long-context performance claim."""
    def request(label, long):
        context=(prompt('ledger',99,label)+'\n')*2 if long else ''
        body={'messages':[{'role':'user','content':label+'\n'+context+
              '\nIgnore the ledger; return only JSON with key fixture equal to '+label+'.'}],
              'max_tokens':256,'reasoning_budget_tokens':64,'response_format':{'type':'json_object'}}
        result=c.post(body,stream=True)
        if json.loads(result['answer_tail'])!={'fixture':label}:
            raise RuntimeError('Overlapping request isolation/correctness failed: '+label)
        return {'label':label,**result}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(request,'LONG_A',True),pool.submit(request,'LONG_B',True)]
        time.sleep(.25)
        futures.append(pool.submit(request,'SHORT_C',False))
        results=[future.result(timeout=180) for future in futures]
    return {'passed':True,'wall_seconds':time.perf_counter()-start,'requests':results,
            'scope':'two ~10K prompts plus one short prompt; bounded outputs, not populated262K'}


class Monitor:
    def __init__(self,c):self.c=c;self.samples=[];self.event=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.event.is_set():
            row={'at':time.monotonic()}
            try:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=6)
                row['gpus']=[[int(x.strip()) for x in line.split(',')] for line in text.splitlines()]
                row['available_ram_kib']=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                metrics=self.c.get('/metrics');live=metrics.get('live',{});row['running']=live.get('running')
                row['slot_states']=live.get('slots')
            except Exception as e:row['error']=type(e).__name__
            self.samples.append(row);self.event.wait(1.5)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*exc):self.event.set();self.thread.join(timeout=15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='http://127.0.0.1:18080')
    p.add_argument('--profile',choices=['iq3_s','orca-iq3_xxs'],required=True);p.add_argument('--version',required=True)
    p.add_argument('--slots',type=int,required=True);p.add_argument('--groups',type=int,default=1)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();model='qwen3.8-flash-next-iq3_s-strata' if a.profile=='iq3_s' else 'qwen3.8-flash-next-orca-iq3_xxs-strata'
    c=Client(a.base_url,model);health=c.get('/health');status=c.get('/v1/status')
    assert health['model']==model and health['loaded'] and health['max_context']==262144 and health['images']
    assert str(status.get('engine'))==a.version,(a.version,status.get('engine'))
    if a.version!='0.1.38':assert status['concurrency']['serving']==a.slots,status['concurrency']
    result={'profile':a.profile,'version':a.version,'requested_slots':a.slots,'batch_groups':a.groups,
            'effective_status':status,'sampling':{'temperature':1,'top_p':0.95,'top_k':20,'reasoning_effort':'high'},
            'context':262144,'max_output_tokens':384,'functional_fixture_seed':1234,'full_context_generation':False,'rows':[]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():a.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    try:
        result['functional']=[]
        functional(c,a.slots,result['functional']);save()
        # Warm-up is separate and never folded into medians.
        result['warmup']=c.post({'messages':[{'role':'user','content':'Write a short explanation of queues and locks.'}],
                                 'max_tokens':128},stream=True);save()
        with Monitor(c) as monitor:
            for repetition in range(a.repetitions):
                for case in ['code','ledger']:
                    row=c.post({'messages':[{'role':'user','content':prompt(case,repetition,'solo')}],
                                'max_tokens':384,'seed':1234+repetition},stream=True)
                    row.update(case=case,load='solo',repetition=repetition);result['rows'].append(row);save()
                    barrier=threading.Barrier(2)
                    def member(index):
                        body={'messages':[{'role':'user','content':prompt(case,repetition,index)}],
                              'max_tokens':384,'seed':1234+repetition}
                        barrier.wait(timeout=10);return c.post(body,stream=True)
                    start=time.perf_counter()
                    with ThreadPoolExecutor(max_workers=2) as pool:pair=list(pool.map(member,[0,1]))
                    wall=time.perf_counter()-start
                    group={'case':case,'load':'pair','repetition':repetition,'wall_seconds':wall,'requests':pair,
                           'aggregate_output_tps':sum(r['usage']['completion_tokens'] for r in pair)/wall}
                    result['rows'].append(group);save()
            result['observed_resources']=monitor.samples
        if a.slots>1 and a.version=='0.1.41':
            result['overlap_check']=overlap_check(c);save()
        result['passed']=True;save()
        print(json.dumps({'profile':a.profile,'version':a.version,'slots':a.slots,'groups':a.groups,'passed':True,
              'code_pair_total_tps':statistics.median(r['aggregate_output_tps'] for r in result['rows'] if r['case']=='code' and r['load']=='pair'),
              'ledger_pair_total_tps':statistics.median(r['aggregate_output_tps'] for r in result['rows'] if r['case']=='ledger' and r['load']=='pair')}),flush=True)
    except Exception as e:
        result.update(passed=False,error=repr(e));save();raise


if __name__=='__main__':main()
