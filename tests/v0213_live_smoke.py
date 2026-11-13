"""Future approved LIVE .59 smoke. UID1000 API client only. Hard-refuses the current .46 model before any generation; never starts/stops a server."""
import argparse,concurrent.futures,json,os,time,urllib.request,urllib.error,threading
from pathlib import Path
import sys
sys.path.insert(0,'/home/user/LocalLLM')
import unsloth_v31_v0159_profile as F
import _v0213_e2e_client as C
C.BASE='http://192.168.1.69:8080'

def request(path,body=None,headers=None):
 h=dict(headers or {})
 if body is not None:h['Content-Type']='application/json'
 req=urllib.request.Request(C.BASE+path,json.dumps(body).encode() if body is not None else None,h)
 try:
  with urllib.request.urlopen(req,timeout=180) as r:return r.status,r.read()
 except urllib.error.HTTPError as e:return e.code,e.read()
def get(path):
 code,data=request(path);assert code==200,(path,code,data[:200]);return json.loads(data)
def chat(messages,cap=32,**extra):
 body={'model':F.MODEL_ID,'messages':messages,'max_tokens':cap,'temperature':0,'reasoning_effort':'none',**extra}
 code,data=request('/v1/chat/completions',body);assert code==200,(code,data[:500]);return json.loads(data)
def save(out,name,d):
 p=out/name;p.write_text(json.dumps(d,indent=2)+'\n');p.chmod(0o600)

def require_candidate_status(status):
 if (status.get('loaded') is not True or status.get('engine')!='0.1.59' or status.get('model')!=F.MODEL_ID or
     (status.get('context') or {}).get('native')!=262144 or (status.get('concurrency') or {}).get('serving')!=2):
  raise RuntimeError('Live smoke refuses: exact loaded .59 model/native262K/two-slot target not established; no generation sent.')

def wait_for_candidate(seconds=900):
 deadline=time.monotonic()+seconds
 while True:
  try:
   status=get('/v1/status');require_candidate_status(status);return status
  except (RuntimeError,urllib.error.URLError,TimeoutError):
   if time.monotonic()>=deadline:raise RuntimeError('Exact .59 target did not become ready; no generation sent.')
   time.sleep(2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--quick',action='store_true');ap.add_argument('--wait-seconds',type=int,default=900);a=ap.parse_args();assert os.geteuid()==1000
 F.authorize_activation()  # No client/model test without later explicit root-published operator permission.
 assert __import__('re').fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',a.phase),'Invalid output phase'
 base=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0213-menu-prep-20261113/live');base.mkdir(mode=0o700,parents=True,exist_ok=True)
 out=base/a.phase;out.mkdir(mode=0o700,exist_ok=False)
 status=wait_for_candidate(a.wait_seconds)
 props=get('/props');assert props['total_slots']==2 and props['default_generation_settings']['n_ctx']==262144
 code,html=request('/');assert code==200 and b'<html' in html.lower()
 result={'model':F.MODEL_ID,'uid':1000,'key_free_lan':True,'loaded_runtime':'0.1.59','native_context_per_slot':262144,'serving_slots':2,'web_ui':True}
 save(out,'status.private.json',{'status':status,'props':props,'health':get('/health')})
 text=C.stream(C.message('What is 9 times 7? Reply with only the number.'),cap=16,model=F.MODEL_ID);assert text['content'].strip()=='63';save(out,'text.private.json',text);result['text63_complete_sse']=True
 if a.quick:
  save(out,'result.json',result);print(json.dumps(result),flush=True);return
 thought=C.stream(C.message('Calculate 37 times 57 minus 10. State the final result.'),cap=1024,thinking='high',model=F.MODEL_ID);assert '2099' in thought['content'],thought;assert thought['reasoning'];save(out,'thinking.private.json',thought);result['high2099_reasoning']=True
 off=C.stream(C.message('What is 9 times 7? Reply with only the number.'),cap=16,model=F.MODEL_ID,extra={'experimental_speed_projection':False});assert off['content'].strip()=='63'
 projections=[x.get('projection') for x in get('/metrics').get('requests',[])];assert True in projections and False in projections,projections;result['loaded_projection_default_on_request_toggle_off']=True
 red,blue=C.png((255,0,0)),C.png((0,0,255))
 for name,images,want in [('red-blue',[red,blue],['red','blue']),('blue-red',[blue,red],['blue','red'])]:
  content=[{'type':'text','text':'Name each image color in order, separated by a comma. No explanation.'}]+[{'type':'image_url','image_url':{'url':x}} for x in images]
  row=C.stream([{'role':'user','content':content}],cap=48,model=F.MODEL_ID);answer=row['content'].lower();assert all(w in answer for w in want) and answer.index(want[0])<answer.index(want[1]),row;save(out,'vision-'+name+'.private.json',row)
 result['swapped_ordered_vision']=True
 barrier=threading.Barrier(2)
 def pair(tag):
  barrier.wait(timeout=10)
  return C.stream(C.message('Start with the exact tag '+tag+'. Then list the integers 1 through 120 in order.'),cap=192,model=F.MODEL_ID)
 peak=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(pair,t) for t in ('CEDAR37','MAPLE82')]
  while not all(f.done() for f in futures):
   slots=get('/slots');peak=max(peak,sum(bool(s.get('is_processing')) for s in slots));time.sleep(.08)
  rows=[f.result() for f in futures]
 assert peak==2,('Two simultaneously occupied slots not observed',peak)
 assert 'CEDAR37' in rows[0]['content'] and 'MAPLE82' not in rows[0]['content'] and 'MAPLE82' in rows[1]['content'] and 'CEDAR37' not in rows[1]['content'],rows
 save(out,'concurrent.private.json',rows);result['concurrent_tags_complete_sse']=True;result['observed_occupied_slots']=peak
 body={'model':F.MODEL_ID,'messages':[{'role':'user','content':'Call emit_number with value 63.'}],'temperature':0,'max_tokens':128,'reasoning_effort':'none','tools':[{'type':'function','function':{'name':'emit_number','description':'Return the requested number.','parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}}],'tool_choice':{'type':'function','function':{'name':'emit_number'}}}
 code,data=request('/v1/chat/completions',body);assert code==200,(code,data[:500]);tools=json.loads(data);calls=tools['choices'][0]['message'].get('tool_calls') or [];assert calls and calls[0]['function']['name']=='emit_number' and json.loads(calls[0]['function']['arguments'])['value']==63,tools
 save(out,'tools.private.json',tools);result['tools_format_only_no_execution']=True
 for name,headers,want in [('bad_host',{'Host':'evil.invalid:8080'},403),('bad_origin',{'Origin':'http://evil.invalid'},403),('same_origin',{'Origin':C.BASE},200)]:
  code,_=request('/v1/chat/completions',{'model':F.MODEL_ID,'messages':[{'role':'user','content':'Reply with only 63.'}],'max_tokens':8,'temperature':0,'reasoning_effort':'none'},headers);assert code==want,(name,code);result[name]=code
 histories=[];follow=[]
 for i in range(8):
  code=str(740101+i*101)
  messages=[{'role':'system','content':f'v31_history_{i}_isolated_fixture\nRemember CODE={code}.'},{'role':'user','content':('block '*4096)+'\nWhat is CODE? Reply only with its numeric value.'}]
  reply=chat(messages);answer=reply['choices'][0]['message']['content'].strip();assert answer==code,(i,answer)
  histories.append((messages,answer,code))
 for i in reversed(range(8)):
  messages,answer,code=histories[i];row=chat(messages+[{'role':'assistant','content':answer},{'role':'user','content':'Repeat CODE. Only its numeric value.'}]);assert row['choices'][0]['message']['content'].strip()==code
  reused=(row.get('timings') or {}).get('cache_n',(row.get('usage') or {}).get('prompt_tokens_details',{}).get('cached_tokens',0));follow.append({'history':i,'reused':reused})
 assert sum(x['reused']>1000 for x in follow)==8,follow
 result['cached_histories_reused']=8;save(out,'cache.private.json',{'followups':follow,'metrics':get('/metrics').get('conversation_cache')})
 result['n_predict_default']=props['default_generation_settings'].get('params',{}).get('n_predict');result['scope']='Bounded short functional and ~4K-history cache fixtures; not full-context concurrency, broad quality, speed comparison or sustained stability qualification.'
 save(out,'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
