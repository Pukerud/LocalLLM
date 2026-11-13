#!/usr/bin/env python3
"""CPU-only .59 command assembly and real loopback MOCK frontend. Never loads a GPU/model or changes Hive.
The HTTP answer is scripted, NOT evidence of model inference, vision quality or effective native slots.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from engine_safety import proc, same_process, signal_identity
import unsloth_v31_v0159_profile as C
import strata_v0213_runtime as R


def run(out):
    if os.geteuid() != 1000:
        raise RuntimeError('Run only as user/UID1000')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    root = Path('/home/user/.local/share/localllm-strata')
    source, _ = R.ready(root)
    cfg = C.canonical_config(root)
    sys.path.insert(0, str(source))
    from serve import server as S
    # Prevent even accidental process launch during the command-assembly check; only hardware identification is stubbed.
    with patch.object(S.drafting, 'detect_hardware', return_value=('cuda', ['sm_86'] * 4)), \
         patch.object(S, 'drafting_card_class', return_value='other'), \
         patch.object(S.drafting, 'engine_form', return_value=S.drafting.FORM_FILE), \
         patch.object(S.subprocess, 'Popen', side_effect=AssertionError('No native launch permitted')):
        assembled = S.engine_args(cfg, cfg['exe'], announce=False)
        child = S.child_env(cfg)
        assert S.mtp_retry_args(cfg, cfg['exe']) is None
        assert S.nodraft_retry_args(cfg, cfg['exe']) is None
    for flag, value in (('--max-context','262144'),('--batch','2'),('--batch-groups','2'),('--kv','int8'),
                        ('--kv-resident','32768'),('--conversation-cache-mib','16384'),
                        ('--conversation-cache-slots','8'),('--spec','3'),('--layer-split','auto')):
        assert assembled.count(flag) == 1 and assembled[assembled.index(flag) + 1] == value, flag
    for item in cfg['args']:
        assert item in assembled, item
    assert '--vision' in assembled and '--mtp' in assembled and '--draft' not in assembled
    assert child['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    for k, v in C.ENV.items():
        assert child[k] == v, k
    (out / 'assembled-arguments.json').write_text(json.dumps({'args':assembled,'engine_env':C.ENV,
                    'scope':'Command assembly only; no native process/GPU started'},indent=2))
    # Confirm the exact external-lifecycle patch refuses without closing/signalling/spawning anything.
    with patch.dict(os.environ, {'STRATA_EXTERNAL_LIFECYCLE':'1'}):
        with patch.object(S.StrataEngine, 'close') as close, patch.object(S.subprocess, 'Popen') as spawn:
            try:
                S.StrataEngine.restart(object())
            except S.EngineDied as e:
                assert 'ownership, rental and hardware' in str(e)
            else:
                raise AssertionError('External-lifecycle restart guard did not refuse')
            close.assert_not_called(); spawn.assert_not_called()
    ready = out / 'mock-ready.json'
    home = out / 'home'; home.mkdir(mode=0o700)
    env = {'HOME':str(home),'USER':'user','LOGNAME':'user','PATH':'/usr/local/bin:/usr/bin:/bin',
           'LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'',
           'HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','STRATA_KEEP_LEFTOVERS':'1',
           'STRATA_EXTERNAL_LIFECYCLE':'1','STRATA_START_RETRIES':'0','OMP_NUM_THREADS':'1',
           'OPENBLAS_NUM_THREADS':'1','STRATA_RUNTIME_DIR':str(out/'runtime')}
    args = [str(source/'.venv/bin/python'),'-m','serve.server','--engine','mock','--host','127.0.0.1',
            '--port','0','--script','63','--offline','--ready-file',str(ready),
            '--tokenizer',str(out/'no-model-tokenizer')]
    results = {'uid':os.geteuid(),'candidate_model_loaded':False,'gpu_inference_tested':False,
               'native_command_assembly':True,'external_restart_refused':True}
    with (out/'mock-server.log').open('x') as log:
        p = subprocess.Popen(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = None
        try:
            deadline = time.monotonic()+40
            while not ready.exists():
                if p.poll() is not None:
                    raise RuntimeError('Mock frontend exited; see mock-server.log')
                if time.monotonic()>deadline:
                    raise TimeoutError('Mock frontend readiness')
                time.sleep(.05)
            identity = proc(p.pid)
            assert identity and identity['uid']==1000 and identity['ppid']==os.getpid() and identity['cmd']==args
            info=json.loads(ready.read_text()); port=info['bound_port']
            assert port!=8080 and info['bound_host']=='127.0.0.1'
            def request(path, payload=None, headers=None):
                conn=http.client.HTTPConnection('127.0.0.1',port,timeout=15)
                h={'Content-Type':'application/json',**(headers or {})}
                assert not any(k.lower() in ('authorization','x-api-key') for k in h)
                conn.request('GET' if payload is None else 'POST',path,None if payload is None else json.dumps(payload),h)
                r=conn.getresponse();code=r.status;data=r.read();conn.close();return code,data
            code,body=request('/health'); assert code==200
            health=json.loads(body); assert health.get('api_key') is False
            results['key_free_health']=True
            code,body=request('/'); assert code==200 and b'<html' in body.lower()
            results['web_ui']=True
            code,body=request('/v1/models'); assert code==200
            results['models']=True
            payload={'messages':[{'role':'user','content':'Return the scripted answer.'}], 'max_tokens':16,'stream':True,'reasoning_effort':'none'}
            code,body=request('/v1/chat/completions',payload)
            (out/'scripted.sse').write_bytes(body)
            assert code==200 and b'data: [DONE]' in body
            chunks=[json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith('data: {')]
            text=''.join(str(x.get('choices',[{}])[0].get('delta',{}).get('content','')) for x in chunks)
            assert text.strip()=='63', repr(text)
            results['scripted_63_complete_sse']=True
            code,_=request('/v1/chat/completions',payload,{'Host':'evil.invalid'}); assert code==403
            results['bad_host_403']=True
            code,_=request('/v1/chat/completions',payload,{'Origin':'https://evil.invalid'}); assert code==403
            results['foreign_origin_403']=True
            code,_=request('/v1/chat/completions',payload,{'Origin':f'http://127.0.0.1:{port}'}); assert code==200
            results['same_origin_200']=True
        finally:
            if p.poll() is None:
                current=proc(p.pid)
                if identity is None:
                    if not current or current['ppid']!=os.getpid() or current['cmd']!=args or current['uid']!=1000:
                        raise RuntimeError('Own mock child identity unavailable; no guessed stop')
                    identity=current
                if not same_process(identity,current):
                    raise RuntimeError('Own mock child changed; replacement was not signalled')
                signal_identity(identity,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if same_process(identity,proc(p.pid)):
                        signal_identity(identity,signal.SIGKILL)
                    p.wait(timeout=10)
            results['owned_mock_shutdown']=p.poll() is not None
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.out)
