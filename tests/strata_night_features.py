#!/usr/bin/env python3
"""Reuse the proven API-only feature suite in this NEW task's output namespace.
No lifecycle, no historical controller invocation, and no mutation of production modules on disk.
"""
import argparse,os,sys
from pathlib import Path
sys.path[:0]=['/home/user/LocalLLM','/home/user/LocalLLM/tests']
import strata_night_trial as T
import v0213_live_smoke as M

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=list(T.registry()),required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--phase',required=True);a=ap.parse_args()
    if os.geteuid()!=1000:ap.error('UID1000 only')
    p=T.TrialProfile(a.case,a.run);p.authorize_activation()
    M.F=p  # Separate provider object, never mutate the real original profile module.
    original=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0213-menu-prep-20261113/live')
    destination=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0159-overnight-20261113/features')
    M.Path=lambda value:destination if str(value)==str(original) else Path(value)
    oldsave=M.save
    def save(out,name,value):
        if name=='result.json' and value.get('n_predict_default',-1)!=-1:raise RuntimeError('Unlimited output default changed')
        return oldsave(out,name,value)
    M.save=save;sys.argv=[str(__file__),'--phase',a.phase,'--wait-seconds','60'];M.main()
if __name__=='__main__':main()
