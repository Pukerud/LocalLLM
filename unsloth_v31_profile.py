"""Explicit operator-approved IQ4_XS V3.1 .46-only two-chat trial; isolated state and exact pins."""
import os
import stat
from pathlib import Path
import unsloth_v31_assets as A
from engine_safety import SafetyError, owner_home

PROFILE = A.PROFILE
VERSION = A.RUNTIME
STATE_NAME = PROFILE + '-trial'
HIVE_PROFILE = PROFILE + '-v0146'
MODEL_ID = 'qwen3.8-flash-next-unsloth-v31-iq4_xs-ple-q8_0-projection-2slot'
TRIAL_ENV = 'STRATA_UNSLOTH_V31_TRIAL'
READY = ('TRIAL CONFIG: Unsloth V3.1 IQ4_XS; Strata=0.1.46; trusted-LAN=0.0.0.0:8080 (unauthenticated); '
         'native 262144 per slot; two slots / two pipelined groups; conversation cache=16384 MiB / 8 histories; '
         'Q8_0 mmap PLE; original-Qwen projection on (layers 4-44); BF16 CPU vision; INT8 KV (32768 resident); '
         '--spec=3; pipeline-windows=2; solo MTP; no concurrent drafts; automatic split; isolated state/cache. '
         'Bounded functional trial only; no full-context capacity/quality/speed qualification.')


def canonical_config(root, home=None):
    root = Path(root)
    home = Path(home) if home is not None else owner_home()
    source = root / 'runtimes' / A.RUNTIME / 'source'
    folder = A.folder(root)
    models = [folder / row[2] for row in A.ASSETS if row[0].startswith('model_')]
    args = ['--pack', str(folder / 'native-pack'), '--native', str(models[0]),
            '--ple-gguf', str(folder / A.ASSETS[-1][2]), '--ple-io', 'mmap',
            '--expert-profile', str(A.state(home) / 'expert-profile.bin'), '--expert-cache', 'auto',
            '--prefill', 'auto', '--spec', '3', '--mtp', str(root / A.MTP_REL),
            '--max-context', str(A.CONTEXT), '--kv', 'int8', '--kv-resident', '32768', '--vision',
            '--conversation-cache-mib', str(A.CACHE_MIB), '--conversation-cache-slots', str(A.HISTORIES),
            '--trim-stage-weights', '--pipeline-windows', '2', '--batch-groups', str(A.GROUPS)]
    for model in models:
        args += ['--native-dense-gguf', str(model)]
    args += ['--control-vector-scaled', str(folder / A.VECTOR_NAME) + ':1.0',
             '--control-vector-layer-range', '4', '44', '--cvec-mode', 'project', '--cvec-dir', 'per-layer']
    return {
        'exe': str(source / 'engine/strata'), 'cwd': str(source), 'args': args,
        'tokenizer': str(folder / 'native-pack/tokenizer'), 'gpu': [0, 1, 2, 3],
        # Larger/different quant: do NOT inherit SC117's measured fixed split or claim its performance.
        'layer_split': 'auto', 'gpu_order': 'as_given',
        'model_name': 'qwen3.8-flash-next-unsloth-v31-iq4_xs-ple-q8_0-projection-2slot',
        'aliases': [A.PROFILE], 'host': '0.0.0.0', 'port': 8080,
        'lib_dirs': ['/usr/local/cuda-12.9/bin', '/usr/local/cuda-12.9/lib64'],
        'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'experimental_speed_projection': True},
        'vision': {'exe': str(source / 'engine/strata-vision-cpu'), 'model': str(models[0]),
                   'mmproj': str(root / A.VISION_REL), 'gpu': False, 'max_tokens': 1024},
        'lazy_load': False, 'idle_unload_s': 0, 'parallel': A.PARALLEL,
        'env': {'STRATA_PARALLEL': str(A.PARALLEL), 'STRATA_BATCH_GROUPS': str(A.GROUPS)},
    }


def validate_config(root, config, home=None):
    if config != canonical_config(root, home):
        raise SafetyError('V3.1 config changed: exact native262K/2slots/2groups/8histories/Q8/projection required')
    return config


def state_root():
    return A.state(owner_home())


def require_state(state):
    expected = state_root()
    if Path(state) != expected or any(p.is_symlink() for p in (expected, *expected.parents)):
        raise SafetyError('V3.1 requires its exact isolated nonsymlinked state/cache')
    if expected.exists():
        info = expected.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise SafetyError('V3.1 isolated state owner/type/mode changed')


def require_environment(state):
    require_state(state)
    required = {TRIAL_ENV:'1', 'STRATA_RUNTIME':VERSION, 'STRATA_PARALLEL':'2',
                'STRATA_BATCH_GROUPS':'2', 'STRATA_HOST':'0.0.0.0', 'STRATA_PORT':'8080',
                'PYTHONDONTWRITEBYTECODE':'1', 'SETUPTOOLS_USE_DISTUTILS':'stdlib',
                'PYTHONPYCACHEPREFIX':str(Path(state)/'pycache')}
    if any(os.environ.get(k) != v for k,v in required.items()):
        raise SafetyError('V3.1 requires its exact explicit .46/2/2/native262K trusted-LAN trial environment')
    for key in os.environ:
        if key.startswith(('STRATA_SC117_', 'STRATA_SPEC_', 'STRATA_CVEC_', 'STRATA_ROPE_', 'STRATA_YARN_')):
            raise SafetyError('V3.1 inherited tuning/other-profile override refused')


def ready(root, state):
    require_environment(state)
    import stage_unsloth_v31 as B
    B.check(root, pristine_state=False)
    source = Path(root)/'runtimes'/VERSION/'source'
    cfg = canonical_config(root)
    return source, A.folder(root)/'config.inactive.json', cfg, VERSION


def authorize_start(root=None, cfg=None, *, host=None, port=None, state=None):
    if os.geteuid() == 0 or host != '0.0.0.0' or type(port) is not int or port != 8080:
        raise SafetyError('V3.1 starts only as the unprivileged owner on trusted-LAN 0.0.0.0:8080')
    require_environment(state)
    validate_config(root, cfg)
    return 'trial'


def seed_profile(root, state):
    require_state(state)
    path = Path(state)/'expert-profile.bin'
    if not path.exists():
        import prepare_unsloth_v31 as D
        source = Path(root)/'runtimes'/VERSION/'source/data/expert-profile.bin'
        if D.digest(source) != A.ROUTING_SEED_SHA:
            raise SafetyError('V3.1 generic original-model routing seed pin changed')
        with path.open('xb') as out:
            os.fchmod(out.fileno(),0o600)
            out.write(source.read_bytes())
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or
            info.st_mode & 0o077 or not 0 < info.st_size <= 4*1024**2):
        raise SafetyError('V3.1 routing profile owner/type/mode/size changed')
