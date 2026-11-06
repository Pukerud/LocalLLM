#!/usr/bin/env python3
"""Prepare pinned TensorSharp for the existing Swift file, entirely as UID1000.
No model download, system toolkit/driver install, GPU inference or lifecycle action.
"""
import argparse,os,subprocess,sys
from pathlib import Path
import tensorsharp_runtime as T
import swift15_assets as A
from engine_safety import SafetyError
REPO=Path(__file__).resolve().parent
READY='READY TENSORSHARP SOFTWARE: 2.8.6; pinned 9d18a64dbf07; GGML CUDA SM86; original Swift Q8_K_XL + BF16 projector; native 262144; no inference.'

def main():
 ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--check-ready',action='store_true');args=ap.parse_args()
 if os.geteuid()!=1000:raise SafetyError('TensorSharp downloads/builds/preparation must run as user UID1000, never root')
 A.ready()
 if args.check_ready:T.ready();print(READY);return 0
 if T.manifest().exists():
  T.ready(software_only=True);print('BUILT SOFTWARE ONLY; GPU DEPLOYMENT DISABLED: '+T.GPU_DEPLOYMENT_BLOCKED);return 0
 root=T.root();root.mkdir(mode=0o700,parents=True,exist_ok=True)
 if root.is_symlink() or root.stat().st_uid!=1000 or root.stat().st_mode&0o077:raise SafetyError('Unsafe private TensorSharp preparation root')
 clean={'HOME':'/home/user','USER':'user','LOGNAME':'user','PATH':'/usr/sbin:/usr/bin:/sbin:/bin','PYTHONDONTWRITEBYTECODE':'1','GIT_TERMINAL_PROMPT':'0'}
 for path,script in ((root/'dotnet-10.0.403','tensorsharp_bootstrap_sdk.py'),(root/'cuda-13.0.2','tensorsharp_bootstrap_cuda.py')):
  if path.exists() or path.is_symlink():raise SafetyError('Unsealed existing toolkit requires inspection; no overwrite or adoption')
  subprocess.run(['/usr/bin/python3',str(REPO/script)],env=clean,check=True)
 if T.source().exists() or T.source().is_symlink():raise SafetyError('Unsealed existing TensorSharp source requires inspection; no overwrite')
 subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false','clone','--no-checkout','https://github.com/zhongkaifu/TensorSharp.git',str(T.source())],env=clean,check=True)
 subprocess.run(['git','-c','core.hooksPath=/dev/null','checkout','--detach',T.COMMIT],cwd=T.source(),env=clean,check=True)
 T.patch()
 subprocess.run(['nice','-n','15','ionice','-c','3','bash',str(REPO/'build_tensorsharp.sh')],env=clean,check=True)
 T.seal();T.ready(software_only=True);print('BUILT SOFTWARE ONLY; GPU DEPLOYMENT DISABLED: '+T.GPU_DEPLOYMENT_BLOCKED);return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (SafetyError,OSError,subprocess.CalledProcessError) as e:print('BLOCKED:',e,file=sys.stderr);raise SystemExit(1)
