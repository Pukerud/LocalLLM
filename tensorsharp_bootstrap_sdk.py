#!/usr/bin/env python3
"""User-local SHA512-bound .NET10 SDK, no system install or external setup script."""
from pathlib import Path,PurePosixPath
import os,hashlib,urllib.request,tarfile,stat
assert os.geteuid()==1000
ROOT=Path('/home/user/.local/share/localllm-tensorsharp');ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not ROOT.is_symlink()
URL='https://builds.dotnet.microsoft.com/dotnet/Sdk/10.0.403/dotnet-sdk-10.0.403-linux-x64.tar.gz'
HASH='b7b3d8461cf98b07d2552ce9199702fa3ad010ac2671a5f8cd5136f7534e50f6855ba7bbb02b11494563f1d04156276a31cda1103b900e5eecd0d9473dc7b9c5'
p=ROOT/'dotnet-sdk-10.0.403-linux-x64.tar.gz';dest=ROOT/'dotnet-10.0.403'
assert not p.is_symlink() and not dest.is_symlink()
if not p.exists():
 with urllib.request.urlopen(URL,timeout=120) as src,p.open('xb') as out:
  while b:=src.read(4*1024**2):out.write(b)
h=hashlib.sha512()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()==HASH,'SDK SHA512 mismatch'
assert not dest.exists(),'SDK directory already exists; refusing overwrite'
dest.mkdir(mode=0o700)
with tarfile.open(p,'r:gz') as tf:
 members=tf.getmembers();assert sum(m.size for m in members)<4*1024**3
 for m in members:
  name=PurePosixPath(m.name);assert not name.is_absolute() and '..' not in name.parts
  assert m.isfile() or m.isdir() or m.issym() or m.islnk()
  if m.issym() or m.islnk():
   target=(dest/str(name)).parent/m.linkname if m.issym() else dest/m.linkname
   assert target.resolve().is_relative_to(dest.resolve()),'SDK link escapes directory'
 for m in members:
  m.mode &= 0o755
 tf.extractall(dest,members=members)
print('SHA512-verified user-local .NET10.0.403 extracted; no system software modified.',flush=True)
