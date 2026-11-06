#!/usr/bin/env python3
"""Publish retained inactive bundles/selector without touching the active model or services.
Only explicitly retired inactive package files are removed; model data retirement is separate.
"""
import argparse,fcntl,hashlib,json,os,shutil,stat,tempfile,time
from pathlib import Path
import install_ple_q8_profiles as U
import hive_profile_switch as S
from engine_safety import SafetyError,docker_empty
RETIRED=('hauhau-q8-fastmtp-q4kv-xhigh','swift15-heretic-re','iq3_s-sc117-abliterated',
         'iq3_s-sc117-abliterated-v0146','swift15-heretic-re-ple-q8','iq3_s-sc117-abliterated-ple-q8-v0146')

def stable():
 out=U.active_snapshot()
 out['service']=S.hosting.service_state()
 out['cloud_hashes']={n:hashlib.sha256(Path('/hive-config',n).read_bytes()).hexdigest() for n in ('rig.conf','wallet.conf','nvidia-oc.conf')}
 return out

def unused(paths):
 prefixes=tuple(str(p)+'/' for p in paths)
 if not prefixes:return
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   if any(v in (d/'cmdline').read_bytes().decode(errors='replace') or v in (d/'maps').read_text() for v in prefixes):
    raise SafetyError('An inactive package is referenced by a live process; nothing removed')
   for f in (d/'fd').iterdir():
    try:target=os.readlink(f)
    except FileNotFoundError:continue
    if target.startswith(prefixes):raise SafetyError('Inactive package has a live FD; nothing removed')
  except (FileNotFoundError,ProcessLookupError):continue
  except PermissionError:raise SafetyError('Cannot prove retired package unused')

def publish():
 if os.geteuid()!=0:raise SafetyError('Root package publication only; runtime preparation runs as user')
 U.trusted_layout()
 fd=os.open(U.LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with S.hosting.launch_reservation(),os.fdopen(fd,'r+') as lock:
  st=os.fstat(lock.fileno())
  if not stat.S_ISREG(st.st_mode) or st.st_uid or stat.S_IMODE(st.st_mode)!=0o600 or st.st_nlink!=1:raise SafetyError('Unsafe selector lock')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if S.hosting.read_lease() is not None or S.JOURNAL.exists() or S.JOURNAL.is_symlink():raise SafetyError('Hosting lease/switch journal present; no publication')
  docker_empty();S.inspect_owned_workload();before=stable()
  active=Path(before['current']).name
  if active not in S.PROFILES:raise SafetyError('An explicitly retired/unknown profile is still selected; not removed or stopped')
  for profile in S.PROFILES:S.preflight_profile(profile)
  candidates={}
  for profile in RETIRED:
   path=U.ROOT/'profiles'/profile
   if not path.exists() and not path.is_symlink():continue
   info=path.lstat()
   if not stat.S_ISDIR(info.st_mode) or info.st_uid or info.st_mode&0o022 or set(x.name for x in path.iterdir())!=set(U.FILES):raise SafetyError('Unexpected retired package contents; not deleted')
   candidates[profile]={n:U.checked_file(path/n,installed=True) for n in U.FILES}
  unused([U.ROOT/'profiles'/p for p in candidates]);docker_empty()
  if stable()!=before:raise SafetyError('Active/selection/cloud state changed before publication')
  added={p:U.install_inactive_profile(U.SOURCE/'profiles'/p,U.ROOT/'profiles'/p) for p in S.PROFILES if p!=active}
  if any(U.checked_file(U.SOURCE/'profiles'/active/n)!=U.checked_file(U.ROOT/'profiles'/active/n,installed=True) for n in U.FILES):
   raise SafetyError('Active bundle differs; no hot overwrite allowed')
  if stable()!=before:raise SafetyError('Active state changed; selector not published')
  changed=U.publish_menu(U.SOURCE/'profile-menu.sh',U.ROOT/'profile-menu.sh')
  # Keep a small root-only audit/rollback copy of inactive package metadata.
  backup=Path(tempfile.mkdtemp(prefix='retired-profile-bundles-',dir='/var/lib/localllm-strata'))
  for p,files in candidates.items():
   d=backup/p;d.mkdir(mode=0o700)
   for n,b in files.items():
    f=d/n;f.write_bytes(b);f.chmod(0o600)
  docker_empty();unused([U.ROOT/'profiles'/p for p in candidates])
  if stable()!=before:raise SafetyError('State changed before inactive package removal')
  for p,files in candidates.items():
   d=U.ROOT/'profiles'/p
   if {n:U.checked_file(d/n,installed=True) for n in U.FILES}!=files:raise SafetyError('Inactive package changed; removal refused')
   for n in U.FILES:(d/n).unlink()
   d.rmdir()
  if stable()!=before:raise SafetyError('External active/cloud state change during publication; inspect before switching')
  return {'added_inactive_profiles':added,'selector_updated':changed,'retired_inactive_bundles_removed':list(candidates),
          'current_unchanged':True,'active_bundle_unchanged':True,'stable_links_unchanged':True,'miner_markers_unchanged':True,
          'cloud_configuration_unchanged':True,'admission_service_unchanged':True,'lifecycle_actions':[]}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--publish',action='store_true',required=True);ap.parse_args()
 try:print(json.dumps(publish(),indent=2));return 0
 except (OSError,ValueError,SafetyError) as e:print('BLOCKED:',e);return 1
if __name__=='__main__':raise SystemExit(main())
