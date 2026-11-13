"""Bounded V3.1 .59 client primitives. No lifecycle commands, model download or external tool execution."""
import urllib.request,urllib.error,json,time,base64,zlib,struct,uuid,concurrent.futures,statistics
BASE='http://192.168.1.69:8080'
def get(path):
 with urllib.request.urlopen(BASE+path,timeout=20) as r:return json.load(r)
def png(color):
 def c(tag,data):return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
 w=h=256;raw=(b'\0'+bytes(color)*w)*h
 b=b'\x89PNG\r\n\x1a\n'+c(b'IHDR',struct.pack('>2I5B',w,h,8,2,0,0,0))+c(b'IDAT',zlib.compress(raw))+c(b'IEND',b'')
 return 'data:image/png;base64,'+base64.b64encode(b).decode()
def stream(messages,*,cap=256,thinking='none',model='',extra=None):
 body={'model':model,'messages':messages,'max_tokens':cap,'temperature':0,'top_p':1,'top_k':1,'min_p':0,'seed':719,
       'presence_penalty':0,'frequency_penalty':0,'repeat_penalty':1.0,'stream':True,'stream_options':{'include_usage':True},
       'reasoning_effort':thinking,'chat_template_kwargs':{'enable_thinking':thinking!='none'},'cache_prompt':False,'multi_agent':False}
 if extra:body.update(extra)
 start=time.monotonic();first=None;last=None;content='';reasoning='';usage={};timings={};done=False;finish=None;events=0
 req=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=240) as response:
  for line in response:
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':done=True;break
   d=json.loads(data);events+=1
   if d.get('error'):raise RuntimeError(str(d['error']))
   if d.get('usage'):usage=d['usage']
   if d.get('timings'):timings=d['timings']
   for choice in d.get('choices',[]):
    delta=choice.get('delta',{});a=delta.get('content') or '';b=delta.get('reasoning_content') or delta.get('reasoning') or ''
    if a or b:
     now=time.monotonic();first=first or now;last=now
    content+=a;reasoning+=b
    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
 end=time.monotonic()
 if not done or finish is None or not first:raise RuntimeError('Incomplete/no-output SSE')
 n=usage.get('completion_tokens');p=usage.get('prompt_tokens')
 if not isinstance(n,int) or n<1:raise RuntimeError('Backend omitted actual completion-token count')
 return {'content':content,'reasoning':reasoning,'prompt_tokens':p,'output_tokens':n,'usage':usage,'timings':timings,'events':events,'done':done,'finish':finish,
         'seconds':end-start,'ttft_s':first-start,'decode_client_tps':(n-1)/(last-first) if last>first and n>1 else None,'first':first,'last':last,'start':start,'end':end}
def message(prompt,nonce=None):
 return [{'role':'system','content':'Follow the user instructions carefully. Independent fixture '+(nonce or uuid.uuid4().hex)+'.'},{'role':'user','content':prompt}]
