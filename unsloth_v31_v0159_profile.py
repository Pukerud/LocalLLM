"""Inactive, fully isolated .59 alternative to Hive [21]. Activation remains explicitly LOCKED.
Same frozen model/PLE/projection/vision/MTP/slots/context/cache; scheduling and exact-kernel candidates only.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

from engine_safety import SafetyError, owner_home
import unsloth_v31_profile as BASE
import unsloth_v31_assets as A
import strata_v0213_runtime as R

PROFILE = A.PROFILE + '-v0159'
HIVE_PROFILE = PROFILE
VERSION = R.VERSION
STATE_NAME = PROFILE + '-trial'
TRIAL_ENV = 'STRATA_UNSLOTH_V31_V0159_TRIAL'
MODEL_ID = BASE.MODEL_ID + '-v0159'
LOCKED = 'Strata .59 candidate is OFFLINE ONLY / LOCKED. Explicit operator go and separate GPU/LAN validation required; running server untouched.'
APPROVAL = Path('/var/lib/localllm-strata/approvals/unsloth-v31-iq4xs-v0159.json')
CONFIRMATION = 'OPERATOR-APPROVED-LIVE-TRIAL'
READY = ('OFFLINE CONFIG: Unsloth V3.1 IQ4_XS; Strata=0.1.59 (app 0.2.13); '
         'native262144/2slots/2groups/16384MiB/8histories; Q8 PLE; projection ON layers4-44; '
         'BF16 CPU vision; stock MTP3; INT8/32768 KV; isolated state; '
         'scheduling + exact IQ4/scorer candidates; immutable offline readiness only; live approval / GPU validation separate.')
# The .59 defaults include some arithmetic-changing paths. Do not silently enable those in this comparison.
ENV = {
    'STRATA_CKPT_POOL': '1', 'STRATA_CKPT_POOL_READY': '4', 'STRATA_SLOT_CKPT_MOVE': '1',
    'STRATA_SLOT_COPY_DIRECT': '1', 'STRATA_SLOT_COPY_BATCHSYNC': '1',
    'STRATA_BATCH_GDN_GROUPS': '1', 'STRATA_BATCH_ATTN_ROWS': '1',
    'STRATA_BATCH_SELF_COMMIT': '1', 'STRATA_QSA_SLOTS': '1',
    'STRATA_IQ4_W4A8': '1', 'STRATA_SELECT_TWO_PASS': '1',
    'STRATA_GDN_TC': '0', 'STRATA_GDN_CONVL2': '0', 'STRATA_HC_Q8': '0',
    'STRATA_ROUTE_PRIOR': '0', 'STRATA_ROUTE_RESIDENT': '0', 'STRATA_ROUTE_RESIDENT_MTP': '0',
    'STRATA_EMB_REUSE_ACCOUNT': '0', 'STRATA_PREFILL_CPU_SHARE': '0', 'STRATA_BATCH_MTP': '0',
    'STRATA_STAGE_PIN': '0', 'STRATA_NV_CHUNK': '0', 'STRATA_PW_WITH_BATCH': '0',
    # Upstream orphan cleanup/retry cannot adopt a different process or bypass fresh renter/hardware gates.
    'STRATA_KEEP_LEFTOVERS': '1', 'STRATA_START_RETRIES': '0', 'STRATA_EXTERNAL_LIFECYCLE': '1',
    'STRATA_ENGINE_NICE': '0',
}


def folder(root):
    return Path(root) / 'profiles' / PROFILE


def state_root(home=None):
    return (Path(home) if home is not None else owner_home()) / '.local/state/locallm-strata' / STATE_NAME


def canonical_config(root, home=None):
    root = Path(root)
    home = Path(home) if home is not None else owner_home()
    cfg = copy.deepcopy(BASE.canonical_config(root, home))
    source = R.folder(root) / 'source'
    args = cfg['args']
    args[args.index('--expert-profile') + 1] = str(state_root(home) / 'expert-profile.bin')
    cfg.update(exe=str(source / 'engine/strata'), cwd=str(source),
               model_name=MODEL_ID, aliases=[PROFILE], drafting='mtp', split_engine='off',
               lib_dirs=[str(source / 'engine')], env=dict(ENV, STRATA_PARALLEL='2', STRATA_BATCH_GROUPS='2'),
               engine_silence_s=0)
    cfg['vision']['exe'] = str(source / 'engine/strata-vision-cpu')
    return cfg


def validate_config(root, cfg, home=None):
    if cfg != canonical_config(root, home):
        raise SafetyError('Strata .59 candidate changed: exact same GGUF/Q8/projection/CPU vision/MTP/native262K/2/2/8 required')
    return cfg


def require_state(state):
    expected = state_root()
    if Path(state) != expected or any(p.is_symlink() for p in (expected, *expected.parents)):
        raise SafetyError('Strata .59 requires its own exact isolated nonsymlinked state; .46 state never adopted')
    if expected.exists():
        st = expected.stat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            raise SafetyError('Strata .59 private state owner/type/mode changed')


def require_environment(state):
    require_state(state)
    required = {TRIAL_ENV: '1', 'STRATA_RUNTIME': VERSION, 'STRATA_PARALLEL': '2', 'STRATA_BATCH_GROUPS': '2',
                'STRATA_HOST': '0.0.0.0', 'STRATA_PORT': '8080', 'PYTHONDONTWRITEBYTECODE': '1',
                'SETUPTOOLS_USE_DISTUTILS': 'stdlib', 'PYTHONPYCACHEPREFIX': str(Path(state) / 'pycache')}
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise SafetyError('Strata .59 candidate requires its exact explicit owner/.59/2/2/private-state environment')
    for k in os.environ:
        if k.startswith(('STRATA_SC117_', 'STRATA_SPEC_', 'STRATA_CVEC_', 'STRATA_ROPE_', 'STRATA_YARN_')) or k == BASE.TRIAL_ENV:
            raise SafetyError('Strata .59 inherited tuning/old-profile override refused')


def approval_contract():
    """Data-only permission template. Constructing it NEVER grants approval or starts/stops anything."""
    root = Path('/home/user/.local/share/localllm-strata')
    repo = Path(__file__).resolve().parent
    policies = ['unsloth_v31_v0159_profile.py', 'strata_v0213_runtime.py', 'strata_launcher.py',
                'hive_profile_switch.py', 'engine_safety.py', 'strata_v0213_activation.py',
                'hive-llm-miner/profiles/' + HIVE_PROFILE + '/h-run.sh']
    cfg = json.dumps(canonical_config(root, Path('/home/user')), sort_keys=True, separators=(',', ':')).encode()
    return {'schema': 1, 'scope': CONFIRMATION, 'activation_allowed': True, 'profile': HIVE_PROFILE,
            'runtime': VERSION, 'source_commit': R.PIN, 'model_revision': A.REVISION,
            'ple_sha256': A.ASSETS[-1][4], 'engine_sha256': R.OFFICIAL_FILES['strata'][1],
            'configuration_sha256': hashlib.sha256(cfg).hexdigest(), 'owner_uid': 1000,
            'policy_sha256': {p: R.digest(repo / p) for p in policies}}


def validate_approval(value):
    if value != approval_contract():
        raise SafetyError(LOCKED + ' Approval does not match these exact prepared policies/assets.')
    return 'trial'


def authorize_activation():
    # No env/CLI flag, offline validation receipt, saved token or PID is approval.
    # Only a later explicitly authorized root action can publish this separate data-only permission.
    try:
        parent = APPROVAL.parent
        for p in (parent, *parent.parents):
            st = p.lstat()
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
                raise SafetyError(LOCKED + ' Unsafe approval directory.')
        fd = os.open(APPROVAL, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as f:
            st = os.fstat(f.fileno())
            if (not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or stat.S_IMODE(st.st_mode) != 0o644 or
                    st.st_nlink != 1 or st.st_size > 16384):
                raise SafetyError(LOCKED + ' Unsafe approval owner/type/mode/size.')
            value = json.load(f)
        return validate_approval(value)
    except (OSError, ValueError) as e:
        raise SafetyError(LOCKED) from e


def authorize_start(root=None, cfg=None, *, host=None, port=None, state=None):
    authorize_activation()  # Preparation is never start authority.
    if os.geteuid() == 0 or host != '0.0.0.0' or type(port) is not int or port != 8080:
        raise SafetyError('Strata .59 starts only as its unprivileged owner on trusted-LAN 0.0.0.0:8080')
    require_environment(state)
    validate_config(root, cfg)
    return 'trial'


def check(root, full=False):
    import stage_unsloth_v31 as B
    B.check(root, full=full, pristine_state=False)  # shared immutable data only; never read/copy live histories
    source, runtime = R.ready(root)
    path = folder(root) / 'config.inactive.json'
    d = json.loads(R.checked_file(folder(root) / 'prepared.json').read_text())
    if (d.get('phase'), d.get('activation_allowed'), d.get('gpu_inference_tested')) != ('OFFLINE-PREPARED-LOCKED', False, False):
        raise SafetyError('Strata .59 offline preparation receipt changed')
    if (d.get('source_commit') != R.PIN or d.get('model_revision') != A.REVISION or
            d.get('model_assets') != [list(row) for row in A.ASSETS]):
        raise SafetyError('Strata .59 immutable model/runtime pins differ')
    R.check_record(d['config'])
    validate_config(root, json.loads(path.read_text()))
    require_state(state_root())
    seed = state_root() / 'expert-profile.bin'
    if not seed.is_file() or seed.is_symlink():
        raise SafetyError('Strata .59 own routing seed is missing/unsafe')
    return source, path, canonical_config(root), VERSION


def ready(root, state):
    require_environment(state)
    return check(root)


def seed_profile(root, state):
    require_state(state)
    path = Path(state) / 'expert-profile.bin'
    if not path.exists():
        raise SafetyError('Strata .59 must be offline prepared before activation; no live-profile adoption')
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise SafetyError('Strata .59 routing profile owner/type/mode changed')


def prepare(root):
    if os.geteuid() != 1000:
        raise SafetyError('Prepare the .59 profile only as user/UID1000')
    from prepare_strata_v0213 import atomic_json
    import stage_unsloth_v31 as B
    root = Path(root)
    R.ready(root)
    B.check(root, pristine_state=False)
    dest = folder(root)
    if (dest / 'prepared.json').exists():
        return check(root)
    if dest.is_symlink() or dest.stat().st_uid != os.geteuid() or stat.S_IMODE(dest.stat().st_mode) != 0o700:
        raise SafetyError('Provision only the new private .59 profile directory; originals never rewritten')
    state = state_root()
    require_state(state)
    if state.exists() and any(state.iterdir()):
        raise SafetyError('Fresh .59 state must be empty; no cached conversations/adaptation copied')
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    original_seed = root / 'runtimes/0.1.46/source/data/expert-profile.bin'
    if original_seed.stat().st_size != A.ROUTING_SEED_SIZE or R.digest(original_seed) != A.ROUTING_SEED_SHA:
        raise SafetyError('Pinned generic routing seed changed')
    with (state / 'expert-profile.bin').open('xb') as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(original_seed.read_bytes())
    cfg_path = dest / 'config.inactive.json'
    atomic_json(cfg_path, canonical_config(root))
    cfg_path.chmod(0o444)
    atomic_json(dest / 'prepared.json', {'phase': 'OFFLINE-PREPARED-LOCKED', 'activation_allowed': False,
                'gpu_inference_tested': False, 'source_commit': R.PIN, 'model_revision': A.REVISION,
                'model_assets': A.ASSETS, 'config': R.record(cfg_path),
                'generic_routing_seed_sha256': A.ROUTING_SEED_SHA,
                'state_copied_from_live': False, 'original_data_rewritten': False,
                'scope': 'Shared checksum-pinned immutable model/derived-pack data; no GPU serving/quality/speed claim'})
    return check(root)
