#!/usr/bin/env python3
"""Explicit isolated tuned-profile entrypoint; all lifecycle gates remain in strata_launcher.
No default/retired-profile fallback, no updater, no environment-only approval.
"""
import sys
from pathlib import Path
import strata_launcher as L
import unsloth_v31_v0159_tuned_profile as P
from engine_safety import SafetyError

def main():
    if sys.argv[1:]==['--prepare-profile']:
        P.prepare(Path('/home/user/.local/share/localllm-strata'));return 0
    args=sys.argv[1:]
    for flag,want in (('--profile',P.PROFILE),('--runtime',P.VERSION)):
        if args.count(flag)!=1 or any(x.startswith(flag+'=') for x in args):
            raise SafetyError('Tuned entrypoint requires one explicit '+flag+' '+want)
        i=args.index(flag)
        if i+1>=len(args) or args[i+1]!=want:raise SafetyError('Tuned entrypoint refuses another profile/runtime')
    if '--quickstart' in args or '--check-ready' in args:
        P.ENV=P.environment()  # Fresh reviewed recipe, never a stale import-time fallback.
    # A separate strict profile MODULE supplies immutable recipe/root permission.
    # No on-disk production module, running process or stop identity is replaced.
    L.v3159=P
    return L.main()
if __name__=='__main__':
    try:raise SystemExit(main())
    except SafetyError as e:print('BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(1)
