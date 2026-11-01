"""Explicit .46-only SC117 Q8 two-user preset. Legacy profiles/gates remain unchanged."""
import copy
import os
from pathlib import Path
import stat

from engine_safety import SafetyError, owner_home
import iq3_s_sc117_abliterated_profile as SC
import ple_q8_profile as Q

PROFILE = 'sc117-q8-2slot'
VERSION = '0.1.46'
STATE_NAME = 'sc117-iq3s-ple-q8-2slot-trial'
HIVE_PROFILE = 'iq3_s-sc117-abliterated-ple-q8-2slot-v0146'
MODEL_ID = SC.MODEL_ID + '-ple-q8_0-2slot'
READY = ('TRIAL CONFIG: unqualified SC117 Q8_0 2-slot; Strata=0.1.46; '
         'trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); '
         'BF16 CPU vision; --spec=3; pipeline-windows=2; split=14,24,36; '
         'two serving slots / two pipelined groups; conversation cache=16384 MiB / 8 histories; '
         'solo MTP; no concurrent drafts; mmap PLE; isolated state/cache. '
         'Bounded throughput evidence, not qualification; long-prefill stream stalls possible.')


def state_root():
    return owner_home() / '.local/state/locallm-strata' / STATE_NAME


def require_state(state):
    expected = state_root()
    if Path(state) != expected:
        raise SafetyError('SC117 two-slot preset requires its exact separate state/cache: ' + str(expected))
    for path in (expected, *expected.parents):
        if path.is_symlink():
            raise SafetyError('SC117 two-slot state path is symlinked')
    if expected.exists() and (not expected.is_dir() or expected.stat().st_uid != os.geteuid()):
        raise SafetyError('SC117 two-slot state owner/type changed')


def require_environment(state):
    require_state(state)
    required = {SC.TRIAL_ENV: '1', 'STRATA_RUNTIME': VERSION, 'STRATA_PARALLEL': '2',
                'STRATA_BATCH_GROUPS': '2', 'STRATA_HOST': '0.0.0.0', 'STRATA_PORT': '8080',
                'PYTHONDONTWRITEBYTECODE': '1', 'SETUPTOOLS_USE_DISTUTILS': 'stdlib',
                'PYTHONPYCACHEPREFIX': str(Path(state) / 'pycache')}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise SafetyError('SC117 two-slot preset requires its exact .46/2-slot/2-group trusted-LAN trial environment')
    if any(key in os.environ for key in (*SC.TRIAL_OVERRIDE_ENV, *SC.TRIAL_EXPERIMENT_ENV)):
        raise SafetyError('SC117 two-slot menu preset does not accept tuning overrides')


def build_config(cfg):
    """Only the benchmark winner's declared changes, after canonical SC117/Q8 verification."""
    cfg = copy.deepcopy(cfg)
    args = cfg['args']
    if cfg.get('parallel') is not None or cfg.get('env') != {'STRATA_PARALLEL': '1', 'STRATA_BATCH_GROUPS': '1'}:
        raise SafetyError('SC117 two-slot overlay requires the exact one-slot canonical input')
    for flag, old, new in (('--conversation-cache-mib', '8192', '16384'),
                           ('--conversation-cache-slots', '4', '8')):
        if args.count(flag) != 1 or args[args.index(flag) + 1] != old:
            raise SafetyError('SC117 two-slot baseline cache policy changed')
        args[args.index(flag) + 1] = new
    if any(a in ('--batch', '--slots', '--batch-groups', '--batch-mtp') or
           a.startswith(('--batch=', '--slots=', '--batch-groups=', '--batch-mtp=')) for a in args):
        raise SafetyError('SC117 two-slot overlay refuses pre-existing batching overrides')
    if args.count('--expert-profile') != 1:
        raise SafetyError('SC117 two-slot adaptive-profile input changed')
    args[args.index('--expert-profile') + 1] = str(state_root() / 'expert-profile.bin')
    args.extend(['--batch-groups', '2'])
    cfg['parallel'] = 2
    cfg['env'] = {'STRATA_PARALLEL': '2', 'STRATA_BATCH_GROUPS': '2'}
    cfg['lazy_load'] = False
    cfg['idle_unload_s'] = 0
    cfg['model_name'] = MODEL_ID
    cfg['aliases'] = list(dict.fromkeys([PROFILE, SC.MODEL_ID, SC.PROFILE, *cfg.get('aliases', [])]))
    return cfg


def ready(root, state):
    require_environment(state)
    # Reuse the exact retained SC117 baseline and prepared runtime, never loosen its one-slot gate.
    import strata_launcher as L
    source, path, cfg, version = L._ready_runtime(SC.PROFILE, VERSION, 1, 1)
    if version != VERSION:
        raise SafetyError('SC117 two-slot runtime pin changed')
    cfg = Q.overlay(root, SC.PROFILE, cfg)
    return source, path, build_config(cfg), VERSION


def validate_config(root, state, cfg):
    _source, _path, expected, _version = ready(root, state)
    keys = ('args', 'exe', 'cwd', 'gpu', 'gpu_order', 'layer_split', 'parallel', 'env',
            'vision', 'tokenizer', 'lib_dirs', 'model_name', 'aliases', 'reasoning_effort', 'lazy_load', 'idle_unload_s')
    if any(cfg.get(key) != expected.get(key) for key in keys):
        raise SafetyError('SC117 two-slot configuration differs from the exact measured preset')


def authorize_start(root, cfg, *, host, port, state):
    if os.geteuid() == 0 or host != '0.0.0.0' or type(port) is not int or port != 8080:
        raise SafetyError('SC117 two-slot trial requires unprivileged trusted-LAN 0.0.0.0:8080')
    validate_config(root, state, cfg)
    SC._verify_v01441_trial_artifacts(root, state)
    return 'trial'


def seed_profile(root, state):
    require_state(state)
    state = Path(state)
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    state.chmod(0o700)
    dest = state / 'expert-profile.bin'
    if dest.is_symlink():
        raise SafetyError('SC117 two-slot adaptive profile is symlinked')
    if not dest.exists():
        seed = Path(root) / 'source-99f3dbd0b21d/data/expert-profile.bin'
        if seed.is_symlink():
            raise SafetyError('SC117 two-slot seed is symlinked')
        with seed.open('rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024**2:
                raise SafetyError('SC117 two-slot seed type/size changed')
            data = f.read(4 * 1024**2 + 1)
            after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
            raise SafetyError('SC117 two-slot seed changed during copy')
        with dest.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)
    info = dest.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_size > 4 * 1024**2:
        raise SafetyError('SC117 two-slot adaptive profile owner/type/mode/size changed')
