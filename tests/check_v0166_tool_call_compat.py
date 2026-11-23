#!/usr/bin/env python3
"""CPU-only HTTP/SSE check of Hive24's actual canonical tool-call setting.
Run with the separate .66 Python environment. Uses a scripted MockEngine on a
loopback ephemeral port: no LLM/GPU, production endpoint, lifecycle or tool I/O.
"""
import json,os,sys,urllib.request
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import unsloth_v31_v0166_profile as U

TOOLS=[{'type':'function','function':{'name':'emit_number','parameters':{
    'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}}]

def call(name):
    return '<tool_call>\n<function='+name+'>\n<parameter=value>\n63\n</parameter>\n</function>\n</tool_call>'

def main():
    if os.geteuid()!=1000 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise SystemExit('Run as UID1000 with GPUs hidden; no changes to a running model')
    cfg=U.canonical_config(U.ROOT,Path('/home/user'))
    assert cfg['undeclared_tool_calls']=='call'
    source,_=U.R.ready(U.ROOT)
    sys.path.insert(0,str(source))
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve,undeclared_tool_calls_setting
    tok=ByteTokenizer()
    unknown=call('not_advertised')
    scripts=[unknown,call('emit_number'),'63','63',unknown]*2+[unknown,'Write <tool_call> and <function=x> to call a tool.',unknown,call('emit_number')]
    engine=MockEngine(tok,scripts,max_context=262144)
    svc=Service(engine,tok,ChatTemplate(Path(cfg['tokenizer'])/'chat_template.jinja'),model_name='cpu-tool-compat-fixture')
    svc.undeclared_tool_calls=undeclared_tool_calls_setting(cfg)
    httpd=serve(svc,host='127.0.0.1',port=0)
    base='http://127.0.0.1:'+str(httpd.server_address[1])
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    requests=0
    def post(messages,stream,tools=TOOLS):
        nonlocal requests
        body={'model':svc.model,'messages':messages,'tools':tools,'max_tokens':4096,
              'reasoning_effort':'none','stream':stream,'stream_options':{'include_usage':True}}
        req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        with opener.open(req,timeout=15) as r:
            assert r.status==200
            raw=r.read().decode()
        requests+=1
        if not stream:
            d=json.loads(raw);c=d['choices'][0]
            return c['message'],c['finish_reason']
        assert 'data: [DONE]' in raw
        message={'role':'assistant','content':'','tool_calls':[]};calls={};finish=None
        for line in raw.splitlines():
            if not line.startswith('data: {'):continue
            d=json.loads(line[6:]);assert not d.get('error')
            for choice in d.get('choices',[]):
                delta=choice.get('delta') or {};message['content']+=delta.get('content') or ''
                finish=choice.get('finish_reason') or finish
                for tc in delta.get('tool_calls') or []:
                    entry=calls.setdefault(tc['index'],{'id':tc.get('id'),'type':'function','function':{'name':'','arguments':''}})
                    if tc.get('id'):entry['id']=tc['id']
                    entry['function']['name']+=tc.get('function',{}).get('name') or ''
                    entry['function']['arguments']+=tc.get('function',{}).get('arguments') or ''
        message['tool_calls']=[calls[i] for i in sorted(calls)]
        return message,finish
    try:
        for stream in (False,True):
            messages=[{'role':'user','content':'Calculate 7 times 9 using emit_number, then answer.'}]
            first,finish=post(messages,stream)
            tc=first['tool_calls'][0]
            assert finish=='tool_calls' and tc['function']['name']=='not_advertised'
            # Client allowlist rejects the name; the server never executes it.
            assert tc['function']['name'] not in {'emit_number'}
            messages += [first,{'role':'tool','tool_call_id':tc['id'],'content':'Unknown tool. Available tool: emit_number(value: integer).'}]
            second,finish=post(messages,stream)
            tc=second['tool_calls'][0]
            assert finish=='tool_calls' and tc['function']['name']=='emit_number'
            assert json.loads(tc['function']['arguments'])=={'value':63}
            messages += [second,{'role':'tool','tool_call_id':tc['id'],'content':'63'}]
            final,finish=post(messages,stream)
            assert finish=='stop' and final['content'].strip()=='63' and not final.get('tool_calls')
            known,finish=post([{'role':'user','content':'What is 7 times 9? Reply only with the number.'}],stream,tools=[])
            assert finish=='stop' and known['content'].strip()=='63' and not known.get('tool_calls')
            no_tools,finish=post([{'role':'user','content':'No tools are available.'}],stream,tools=[])
            assert finish=='stop' and not no_tools.get('tool_calls') and 'not_advertised' in no_tools['content']
        svc.undeclared_tool_calls='text'  # Isolate the policy's negative control, not .66's separate retry feature.
        svc.tool_markup_retry=False       # MOCK control only; canonical candidate configuration stays unchanged.
        default,finish=post([{'role':'user','content':'Call a tool.'}],False)
        assert finish=='stop' and not default.get('tool_calls') and 'not_advertised' in default['content']
        svc.undeclared_tool_calls='call'
        svc.tool_markup_retry=True
        prose,finish=post([{'role':'user','content':'Explain tool syntax.'}],False)
        assert finish=='stop' and not prose.get('tool_calls') and 'Write' in prose['content']
        svc.undeclared_tool_calls='text'
        repaired,finish=post([{'role':'user','content':'Call emit_number with value 63.'}],False)
        assert finish=='tool_calls' and repaired['tool_calls'][0]['function']['name']=='emit_number'
        assert json.loads(repaired['tool_calls'][0]['function']['arguments'])=={'value':63}
        print(json.dumps({'phase':'CPU-HTTP-SSE-TOOL-COMPAT-PASSED','setting':cfg['undeclared_tool_calls'],
            'requests':requests,'streaming_and_nonstreaming':True,'unknown_name_forwarded_unchanged':True,
            'client_allowlist_error_then_corrected_call_and_final63':True,'known_working_key_free63':True,
            'no_tools_still_returns_text':True,'default_text_control_with_markup_retry_disabled_on_mock_only':True,
            'new_markup_retry_corrects_rejected_name':True,'prose_not_executed':True,
            'gpu_or_model_inference':False,'external_tools_executed':False,'production_endpoint_contacted':False}))
    finally:
        httpd.shutdown();httpd.server_close()
if __name__=='__main__':main()
