#!/usr/bin/env python3
"""CPU-only side-by-side preparation. Never invokes an updater, model inference or lifecycle action."""
import argparse,fcntl,json,os,shutil,subprocess
from pathlib import Path
import strata_v0220_runtime as R
import strata_v0213_runtime as OLD
from engine_safety import SafetyError,docker_empty
from prepare_strata_v0213 import atomic_json,extract_archive
ROOT=Path('/home/user/.local/share/localllm-strata')
def prepare():
 if os.geteuid()!=1000 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SafetyError('Prepare as UID1000 with GPUs hidden')
 base=R.folder(ROOT)
 if (base/'prepared.json').exists():
  R.patch_server(base/'source');R.UI.patch(base/'source');return R.ready(ROOT)
 docker_empty();archive=R.checked_file(base/R.ASSET)
 if archive.stat().st_size!=R.ASSET_BYTES or R.digest(archive)!=R.ASSET_SHA256:raise SafetyError('Pinned archive incomplete/different')
 checkout=base/'source';R.source_check(checkout,patched=False);source=checkout/'terminal';package=base/'package'
 if not package.exists():extract_archive(archive,package)
 python=source/'.venv/bin/python'
 # A fresh private venv uses only the pinned release requirements, never system packages.
 if not python.exists():
  subprocess.run(['/usr/bin/python3','-m','venv',str(source/'.venv')],check=True)
  subprocess.run([str(python),'-m','pip','install','--only-binary=:all:','--index-url','https://pypi.org/simple','-r',str(source/'requirements.txt')],check=True)
 script='from pathlib import Path;from strata_control import engine_component as E;c=E.read_manifest(Path('+repr(str(package))+'));assert c.version=="0.1.66" and "sm_86" in c.archs;assert E.verify_signature(c)=="verified";E.verify(c,full=True)'
 subprocess.run([str(python),'-c',script],cwd=source,check=True)
 R.patch_server(checkout);R.UI.patch(checkout);payload=base/'payload';payload.mkdir(mode=0o700,exist_ok=True)
 for name,pin in R.OFFICIAL_FILES.items():
  p=R.checked_file(package/name)
  if (p.stat().st_size,R.digest(p))!=pin:raise SafetyError('Official payload differs: '+name)
  dest=payload/name
  if dest.exists():
   if R.digest(dest)!=pin[1]:raise SafetyError('Existing new-runtime file differs; not overwritten')
  else:shutil.copyfile(p,dest);dest.chmod(0o555 if name=='strata' else 0o444)
 old_source,old=OLD.ready(ROOT);vision=next(r for r in old['runtime_assets'] if Path(r['path']).name=='strata-vision-cpu');OLD.check_record(vision)
 dest=payload/'strata-vision-cpu'
 if not dest.exists():shutil.copyfile(vision['path'],dest);dest.chmod(0o555)
 if R.digest(dest)!=vision['sha256']:raise SafetyError('Preserved CPU encoder differs')
 env=dict(os.environ,LD_LIBRARY_PATH=str(payload),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
 probes={}
 for flag in ('--version','--geometry','--help'):
  p=subprocess.run([str(payload/'strata'),flag],env=env,cwd=source,capture_output=True,text=True,timeout=30,check=True);probes[flag]=(p.stdout+p.stderr).strip()
 if VERSION_NOT_IN(probes):raise SafetyError('Actual native version/geometry differs')
 needed=subprocess.check_output(['readelf','-d',str(dest)],text=True).lower()
 if any(x in needed for x in ('libcuda','libcublas','libhip','libsycl')):raise SafetyError('CPU encoder has a GPU dependency')
 site=next((source/'.venv/lib').glob('python*/site-packages'))
 pyfiles=[source/'.venv/pyvenv.cfg',*sorted(p for p in site.rglob('*') if p.is_file() and '__pycache__' not in p.parts)]
 value={'version':R.VERSION,'app_version':'0.2.20','source_commit':R.PIN,'release_archive_sha256':R.ASSET_SHA256,'component_signature':'verified',
        'binary_build':json.loads((package/'BUILD.json').read_text()),'runtime_assets':[R.record(p) for p in sorted(payload.iterdir())],
        'python_assets':[R.record(p) for p in pyfiles],'cpu_vision':'Byte-identical pinned .59 BF16 CPU encoder; protocol checked separately',
        'cpu_vision_source':vision,'frontend_original_sha256':R.SERVER_ORIGINAL_SHA256,'frontend_patched_sha256':R.SERVER_PATCHED_SHA256,
        'auth':'Supported upstream --no-api-key; preserve Host/Origin and optional auth; no auth source patch',
        'activation_allowed':False,'gpu_inference_tested':False,'no_gpu_probes':probes}
 atomic_json(base/'prepared.json',value);R.ready(ROOT);print('OFFLINE PREPARED0220/0166; no model loaded, no lifecycle/system/power action')
def VERSION_NOT_IN(probes):return R.VERSION not in probes['--version'] or 'flash_next' not in probes['--geometry']
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true',required=True);ap.parse_args()
 base=R.folder(ROOT);fd=os.open(base/'prepare.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'r+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
if __name__=='__main__':main()
