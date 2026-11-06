"""Immutable shared Swift1.5 Uncensored Q8_K_XL data pins; both engines use these unchanged files."""
from pathlib import Path
import hashlib,os,stat
from engine_safety import SafetyError,owner_home
REPOSITORY='ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF'
REVISION='490e3d954a5d7616cccb69df3da437d66c20991a'
MODEL='Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q8_K_XL.gguf'
PROJECTOR='mmproj-BF16.gguf'
PINS={MODEL:(31_457_990_272,'9076f99a2b1c0d232d5c169bb117e919ca21bafbe092be8fc8d9c4d645ed09c9'),PROJECTOR:(931_145_888,'19acd7fcf4ee504328566eddf0f8e603ef853a205b4830209d2a4908213abdea')}
CONTEXT=262144

def directory():return owner_home()/'.local/share/localllm-qwen38/models/swift15-uncensored'
def model():return directory()/MODEL
def projector():return directory()/PROJECTOR

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def ready(*,full=False):
 result={}
 for name,(size,sha) in PINS.items():
  p=directory()/name
  for parent in (p,*p.parents):
   if parent.is_symlink():raise SafetyError('Shared Swift asset path is symlinked')
  info=p.lstat()
  if not stat.S_ISREG(info.st_mode) or info.st_size!=size or info.st_uid not in (0,os.geteuid()) or info.st_mode&0o022:
   raise SafetyError('Shared Swift immutable file owner/type/size/mode changed: '+name)
  with p.open('rb') as f:
   if f.read(4)!=b'GGUF':raise SafetyError('Shared Swift input is not GGUF')
  if full and digest(p)!=sha:raise SafetyError('Shared Swift SHA256 mismatch: '+name)
  after=p.stat()
  if (info.st_ino,info.st_dev,info.st_size,info.st_mtime_ns)!=(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns):
   raise SafetyError('Shared Swift asset changed during verification')
  result[name]={'size':size,'sha256':sha,'inode':info.st_ino,'device':info.st_dev,'mtime_ns':info.st_mtime_ns}
 return result
