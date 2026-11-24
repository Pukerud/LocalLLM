#!/usr/bin/env python3
"""Publish ONLY menu[24]'s explicitly approved defaults while its selected bundle is STOPPED.
No miner/service action, selector switch, cloud write, GPU setting or deleted bundle.
Existing generic add-only publishers deliberately refuse an active-selection overwrite;
this separately authorized narrow path requires zero wrappers/LLMs/GPU work under both locks.
"""
import argparse,ctypes,fcntl,hashlib,json,os,stat,tempfile
from pathlib import Path
from engine_safety import SafetyError,docker_empty,active_engine
import hosting_lifecycle as H
import hive_profile_switch as S
import install_ple_q8_profiles as U
import unsloth_v31_v0166_profile as P
import strata_v0213_activation as A

def exchange(left,right):
    lib=ctypes.CDLL(None,use_errno=True)
    fn=getattr(lib,'renameat2',None)
    if fn is None:raise SafetyError('Atomic directory exchange is required; no non-atomic fallback')
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(left),-100,os.fsencode(right),2):
        err=ctypes.get_errno();raise SafetyError('Atomic menu24 exchange failed: '+os.strerror(err))

def stable():
    return {'current':os.readlink(S.CURRENT),
            'links':{n:os.readlink(U.ROOT/n) for n in U.FILES},
            'markers':{n:(Path('/run/hive')/n).read_bytes().hex() if (Path('/run/hive')/n).exists() else None for n in ('MINER_RUN','MINER_STOP')},
            'osn':H.service_state(),
            'cloud':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,('/hive-config/rig.conf','/hive-config/wallet.conf','/hive-config/nvidia-oc.conf')) if p.exists()}}

def stopped_gate():
    docker_empty();A.hardware()
    if H.read_lease() is not None or S.JOURNAL.exists() or S.JOURNAL.is_symlink():raise SafetyError('Existing hosting transaction; no publication')
    wrappers=S.inspect_owned_workload()
    if wrappers or H.RUN.exists() or active_engine()!='none' or H.gpu_jobs():
        raise SafetyError('Menu24 bundle must be genuinely stopped/unused; no hot overwrite or stop authority')
    if os.readlink(S.CURRENT)!='profiles/'+P.PROFILE:raise SafetyError('This narrow defaults publication requires selected menu24')
    P.authorize_activation()

def publish():
    if os.geteuid()!=0:raise SafetyError('Root-only package publication; preparation/inference remains UID1000')
    U.trusted_layout()
    with H.launch_reservation():
        fd=os.open(U.LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'r+') as lock:
            info=os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or stat.S_IMODE(info.st_mode)!=0o600 or info.st_nlink!=1:
                raise SafetyError('Unsafe selector publication lock')
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            stopped_gate();before=stable();S.preflight_profile(P.PROFILE)
            source=U.SOURCE/'profiles'/P.PROFILE;dest=U.ROOT/'profiles'/P.PROFILE
            info=dest.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode&0o022 or {p.name for p in dest.iterdir()}!=set(U.FILES):
                raise SafetyError('Existing menu24 directory is unsafe/incomplete')
            old={n:U.checked_file(dest/n,installed=True) for n in U.FILES}
            wanted={n:U.checked_file(source/n) for n in U.FILES}
            archive=None
            if old!=wanted:
                stage=Path(tempfile.mkdtemp(prefix='.v0166-defaults-stage-',dir=dest.parent));stage.chmod(0o755)
                for n,data in wanted.items():
                    with (stage/n).open('xb') as out:
                        os.fchmod(out.fileno(),0o755 if n.endswith('.sh') else 0o644);out.write(data);out.flush();os.fsync(out.fileno())
                    if U.checked_file(stage/n,installed=True)!=data:raise SafetyError('Staged menu24 verification failed')
                stopped_gate()
                if stable()!=before or any(U.checked_file(dest/n,installed=True)!=old[n] for n in U.FILES):
                    raise SafetyError('Operator/target state changed; staged directory retained, nothing exchanged')
                exchange(stage,dest);archive=str(stage) # Original entire bundle preserved, never deleted/replayed.
            stopped_gate()
            if stable()!=before:raise SafetyError('External state changed; no stale restoration, inspect fresh')
            changed=U.publish_menu(U.SOURCE/'profile-menu.sh',U.ROOT/'profile-menu.sh')
            if stable()!=before:raise SafetyError('External state changed during selector refresh; no stale restore')
            return {'profile':P.PROFILE,'selected_unchanged':True,'stopped_bundle_updated':old!=wanted,
                    'old_bundle_preserved_at':archive,'selector_updated':changed,'lifecycle_actions':[],
                    'cloud_service_markers_links_unchanged':True,
                    'new_bundle_sha256':{n:hashlib.sha256(wanted[n]).hexdigest() for n in U.FILES}}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--publish',action='store_true',required=True);ap.parse_args()
    try:print(json.dumps(publish(),indent=2));return 0
    except (OSError,ValueError,SafetyError) as e:print('BLOCKED:',e);return 1
if __name__=='__main__':raise SystemExit(main())
