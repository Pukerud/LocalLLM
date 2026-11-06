#!/usr/bin/env python3
"""Exact NVIDIA redistributables -> a private CUDA13 compiler/toolkit; no installer, root or driver change."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import os,json,hashlib,urllib.request,tarfile,shutil
assert os.geteuid()==1000
ROOT=Path('/home/user/.local/share/localllm-tensorsharp');ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not ROOT.is_symlink()
DEST=ROOT/'cuda-13.0.2';ARCH=ROOT/'cuda13-archives';PARTS=ROOT/'cuda13-components'
PINS=[
 ('cuda_nvcc','cuda_nvcc-linux-x86_64-13.0.88-archive.tar.xz','48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a',26657716),
 ('cuda_crt','cuda_crt-linux-x86_64-13.0.88-archive.tar.xz','5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0',79800),
 ('libnvvm','libnvvm-linux-x86_64-13.0.88-archive.tar.xz','17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353',43645408),
 ('cuda_cudart','cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz','25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1',1487960),
 ('cuda_cccl','cuda_cccl-linux-x86_64-13.0.85-archive.tar.xz','ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60',991284),
 ('cuda_nvrtc','cuda_nvrtc-linux-x86_64-13.0.88-archive.tar.xz','00038aac08e1dba6f1933237dbfb217ac6452ae24fab970edcac808f103ca64b',116779532),
 ('libcublas','libcublas-linux-x86_64-13.1.0.3-archive.tar.xz','88bc951efd906032a371153ca61975e0d9c4761e4012169169a6b3a47931606e',838952932),
 ('cuda_nvtx','cuda_nvtx-linux-x86_64-13.0.85-archive.tar.xz','ed150e6fb1b50663ff068cccee3c5e2ca581c3b939b321656afbc9193671137d',94140)]
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for p in (ARCH,PARTS):p.mkdir(mode=0o700,exist_ok=True);assert not p.is_symlink()
assert not DEST.exists(),'Private toolkit already exists; refuse overwrite'
assert shutil.disk_usage(ROOT).free>12*1024**3

def prepare(pin):
 family,name,sha,size=pin;p=ARCH/name;assert not p.is_symlink()
 if not p.exists():
  url=f'https://developer.download.nvidia.com/compute/cuda/redist/{family}/linux-x86_64/{name}'
  with urllib.request.urlopen(url,timeout=180) as src,p.open('xb') as out:
   while b:=src.read(4*1024**2):out.write(b)
 assert p.stat().st_size==size and digest(p)==sha,'CUDA component hash/size mismatch: '+family
 dest=PARTS/family;assert not dest.exists();dest.mkdir(mode=0o700)
 with tarfile.open(p,'r:xz') as tf:
  members=tf.getmembers();assert sum(m.size for m in members)<5*1024**3
  for m in members:
   rel=PurePosixPath(m.name);assert not rel.is_absolute() and '..' not in rel.parts
   assert m.isdir() or m.isfile() or m.issym() or m.islnk()
   if m.issym() or m.islnk():
    target=(dest/str(rel)).parent/m.linkname if m.issym() else dest/m.linkname
    assert target.resolve().is_relative_to(dest.resolve()),'CUDA component link escapes directory'
   m.mode &= 0o755
  tf.extractall(dest,members=members)
 dirs=list(dest.iterdir());assert len(dirs)==1 and dirs[0].is_dir()
 print('Verified/extracted '+family,flush=True);return family,dirs[0]
with ThreadPoolExecutor(max_workers=3) as pool:parts=list(pool.map(prepare,PINS))
DEST.mkdir(mode=0o700)
for family,part in parts:
 for p in sorted(part.rglob('*')):
  rel=p.relative_to(part)
  if rel.parts[0] in ('LICENSE','LICENSE.txt','EULA.txt'):rel=Path('licenses')/family/rel
  dest=DEST/rel
  if p.is_dir():dest.mkdir(mode=0o755,parents=True,exist_ok=True);continue
  dest.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
  if dest.exists() or dest.is_symlink():
   assert p.is_file() and dest.is_file() and digest(p)==digest(dest),'Conflicting CUDA artifact: '+str(rel);continue
  if p.is_symlink():dest.symlink_to(os.readlink(p))
  else:os.link(p,dest)
if not (DEST/'lib64').exists():(DEST/'lib64').symlink_to('lib')
(DEST/'version.txt').write_text('CUDA Version 13.0.2\n')
(DEST/'provenance.json').write_text(json.dumps({'source':'NVIDIA CUDA13.0.2 official redist manifest','components':PINS,'system_modified':False},indent=2)+'\n')
assert (DEST/'bin/nvcc').is_file()
print('Private CUDA13.0.2 assembled; system CUDA12.9/driver untouched.',flush=True)
