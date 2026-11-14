#!/usr/bin/env python3
"""Approved permanent tuned-profile API checks. No lifecycle commands or model changes."""
import argparse,os,sys
from pathlib import Path
sys.path[:0]=['/home/user/LocalLLM','/home/user/LocalLLM/tests']
import unsloth_v31_v0159_tuned_profile as U
import v0213_live_smoke as M

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--quick',action='store_true');a=ap.parse_args()
    if os.geteuid()!=1000:ap.error('UID1000 only')
    U.authorize_activation()
    M.F=U
    original=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0213-menu-prep-20261113/live')
    destination=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0159-overnight-20261113/permanent-features')
    M.Path=lambda value:destination if str(value)==str(original) else Path(value)
    save=M.save
    def checked_save(out,name,value):
        if name=='result.json' and value.get('n_predict_default',-1)!=-1:raise RuntimeError('Unlimited default changed')
        return save(out,name,value)
    M.save=checked_save;sys.argv=[str(__file__),'--phase',a.phase,'--wait-seconds','120']+(['--quick'] if a.quick else []);M.main()
if __name__=='__main__':main()
