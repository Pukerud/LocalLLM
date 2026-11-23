#!/usr/bin/env python3
"""Explicit isolated .66 entrypoint; exact root permission and all common lifecycle gates retained."""
import sys
from pathlib import Path
import strata_launcher as L
import unsloth_v31_v0166_profile as P
from engine_safety import SafetyError

def main():
 if sys.argv[1:]==['--prepare-profile']:P.prepare();return 0
 args=sys.argv[1:]
 for flag,want in (('--profile',P.PROFILE),('--runtime',P.VERSION)):
  if args.count(flag)!=1 or any(x.startswith(flag+'=') for x in args):raise SafetyError('0166 requires one explicit '+flag)
  i=args.index(flag)
  if i+1>=len(args) or args[i+1]!=want:raise SafetyError('0166 entrypoint refuses another profile/runtime')
 L.v3159=P
 return L.main()
if __name__=='__main__':
 try:raise SystemExit(main())
 except SafetyError as e:print('BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(1)
