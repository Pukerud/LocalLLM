#!/usr/bin/env python3
"""Real .59 CPU-only image encoder probe. No LLM/main-model inference, GPU access or Hive changes.
Tests embeddings/protocol/cache, NOT whether the language model names image colors correctly.
"""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import unsloth_v31_v0159_profile as C
import strata_v0213_runtime as R
from engine_safety import proc


def main(out):
    if os.geteuid()!=1000:raise RuntimeError('UID1000 only')
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    root=Path('/home/user/.local/share/localllm-strata')
    source,_=R.ready(root)
    sys.path.insert(0,str(source))
    from serve import server as S
    from PIL import Image
    import numpy as np
    cfg=copy.deepcopy(C.canonical_config(root)['vision'])
    assert cfg['gpu'] is False
    cfg.update(threads=1,max_tokens=16)  # Probe-local resource bound, NEVER rewrite the canonical profile.
    env={'HOME':'/home/user','USER':'user','LOGNAME':'user','PATH':'/usr/bin:/bin',
         'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    rows=[];vision=None
    with (out/'encoder.log').open('x') as log:
        try:
            with patch.object(S.Vision,'work_dir',return_value=out/'encoder-work'):
                vision=S.Vision(cfg,log=log,env=env)
            identity=proc(vision.proc.pid)
            assert identity and identity['uid']==1000 and identity['exe']==cfg['exe']
            assert '--gpu' not in identity['cmd'] and '--threads' in identity['cmd']
            for color in ((255,0,0),(0,0,255),(255,0,0)):
                f=io.BytesIO();Image.new('RGB',(56,56),color).save(f,format='PNG')
                path,n=vision.encode(f.getvalue())
                blob=path.read_bytes();magic,tokens,nx,ny,dim=struct.unpack('<5i',blob[:20])
                assert magic==0x31455653 and n==tokens and 0<n<=16 and dim==2560
                values=np.frombuffer(blob[20:],dtype='<f4')
                assert values.size==tokens*dim and np.isfinite(values).all()
                rows.append({'color':list(color),'tokens':tokens,'dimension':dim,
                             'sha256':__import__('hashlib').sha256(blob).hexdigest(),'mtime_ns':path.stat().st_mtime_ns})
            assert rows[0]['sha256']!=rows[1]['sha256']
            assert rows[0]['sha256']==rows[2]['sha256'] and rows[0]['mtime_ns']==rows[2]['mtime_ns']
        finally:
            if vision is not None:vision.shutdown()
    result={'uid':1000,'cpu_vision_protocol':True,'dimension':2560,'finite_red_blue_embeddings':True,
            'encoder_cache_reuse':True,'cpu_threads':1,'probe_max_tokens':16,
            'canonical_max_tokens_unchanged':1024,'main_llm_loaded':False,'gpu_inference_tested':False,
            'owned_encoder_shutdown':vision is not None and vision.proc is None,'samples':rows}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
