"""Separate pinned app0.2.20 / engine0.1.66 runtime. No update, activation or GPU operation."""
import hashlib,json,os,stat,subprocess
from pathlib import Path
from engine_safety import SafetyError
from strata_v0213_runtime import digest,checked_file,record,check_record,SERVER_OLD,SERVER_NEW
VERSION='0.1.66'
TAG='v0.2.20'
PIN='a34979fc94f3d126b9bb8210f4ad311345688b52'
ASSET='strata-engine-0.1.66-linux-x86_64-cuda-cuda12.tar.gz'
ASSET_BYTES=782739940
ASSET_SHA256='74980aebedc70fb2cfcd96b3c15874cb3e7e517700ecdf8ed811247a9a577f30'
URL='https://github.com/Niko1221/Strata/releases/download/'+TAG+'/'+ASSET
REPOSITORY='https://github.com/Niko1221/Strata.git'
SERVER_ORIGINAL_SHA256='5e20466881840a7c4a76091f28b3538b6f77d80dd68dcf4e803b381384c83417'
SERVER_PATCHED_SHA256='7308281e145e839d854d739150b8423c6e9817d5ad1a21f5a0066b91a85eae14'
OFFICIAL_FILES={
 'strata':(395465704,'57996ee170a43e33c4b8f92f7a3cd4c92a8786fac821b36a62f127202bbaa13a'),
 'BUILD.json':(186,'da7909cb41746ec76f0000b6a0dd29097ad2513a3ff86bba5b5617a6407028d6'),
 'libcudart.so.12':(741088,'256e6409e4f06f618e1fb53d4844a6b81cdded1013afa8ade40c22f99eb133b7'),
 'libcublas.so.12':(105140976,'5757ab5839fb4f203ca47ecb336110d10f4a5606b1e097f195fbca89774569e2'),
 'libcublasLt.so.12':(749210000,'2c9006a75c74b3bea2dc7ae2ec38ab038b0e45ea02cb4b717a915e8a5796acb1'),
}
def folder(root):return Path(root)/'runtimes'/VERSION
def source_check(source,*,patched=True):
 checkout=Path(source);git=['git','-c','safe.directory='+str(checkout),'-C',str(checkout)]
 if subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()!=PIN:raise SafetyError('Strata0220 source pin changed')
 changed=subprocess.check_output(git+['diff','HEAD','--name-only'],text=True).splitlines()
 sha=digest(checkout/'terminal/serve/server.py')
 if changed not in ([],['terminal/serve/server.py']) or sha not in ((SERVER_PATCHED_SHA256,) if patched else (SERVER_ORIGINAL_SHA256,SERVER_PATCHED_SHA256)):
  raise SafetyError('Strata0220 source differs beyond exact external-lifecycle patch')
def patch_server(checkout):
 p=Path(checkout)/'terminal/serve/server.py';data=p.read_bytes();sha=digest(p)
 if sha==SERVER_PATCHED_SHA256:return
 if sha!=SERVER_ORIGINAL_SHA256 or data.count(SERVER_OLD)!=1:raise SafetyError('Strata0220 patch preimage changed')
 data=data.replace(SERVER_OLD,SERVER_NEW)
 if hashlib.sha256(data).hexdigest()!=SERVER_PATCHED_SHA256:raise SafetyError('Strata0220 patch postimage changed')
 p.write_bytes(data)
def ready(root):
 base=folder(root);m=json.loads(checked_file(base/'prepared.json').read_text());source_check(base/'source')
 if (m.get('version'),m.get('source_commit'),m.get('release_archive_sha256'),m.get('component_signature'))!=(VERSION,PIN,ASSET_SHA256,'verified'):
  raise SafetyError('Strata0220 runtime provenance differs')
 if m.get('activation_allowed') is not False or m.get('gpu_inference_tested') is not False:raise SafetyError('Offline runtime receipt is history, never live authority')
 engine=base/'payload';rows=m['runtime_assets']
 if len(rows)!=len(OFFICIAL_FILES)+1 or {Path(r['path']).name for r in rows}!=set(OFFICIAL_FILES)|{'strata-vision-cpu'}:
  raise SafetyError('Strata0220 payload coverage differs')
 if {r['path'] for r in rows}!={str(p) for p in engine.iterdir() if p.is_file()}:raise SafetyError('Unrecorded Strata0220 payload')
 for row in rows:
  pin=OFFICIAL_FILES.get(Path(row['path']).name)
  if pin and (row['bytes'],row['sha256'])!=pin:raise SafetyError('Strata0220 official payload pin changed')
  check_record(row)
 for row in m['python_assets']:check_record(row)
 if m.get('cpu_vision')!='Byte-identical pinned .59 BF16 CPU encoder; protocol checked separately':raise SafetyError('CPU vision provenance differs')
 return base/'source/terminal',m
