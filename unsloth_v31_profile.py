"""Inactive IQ4_XS V3.1 configuration. Intentionally NOT registered with any launcher/selector."""
from pathlib import Path
import unsloth_v31_assets as A
from engine_safety import SafetyError, owner_home


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
        raise SafetyError('Inactive V3.1 config changed: exact native262K/2slots/2groups/8histories/Q8/projection required')
    return config


def authorize_start(*args, **kwargs):
    raise SafetyError('V3.1 candidate is OFFLINE ONLY. Await explicit operator continue, fresh rental/ownership checks, and separate activation/testing; no automatic start.')
