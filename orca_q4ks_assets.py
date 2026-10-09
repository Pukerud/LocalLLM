"""Immutable publisher/source pins for the separate Orca Q4_K_S experiment."""
from pathlib import Path

from orca_assets import (ASSETS as IQ3XXS_ASSETS, PROFILE as IQ3XXS_PROFILE,
                         REPOSITORY as IQ3XXS_REPOSITORY, REVISION as IQ3XXS_REVISION)

REPOSITORY = 'orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF'
REVISION = 'e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc'
PROFILE = 'orca-q4_k_s'
MODEL_ID = 'qwen3.8-flash-next-orca-q4_k_s-strata'
SOURCE_COMMIT = '6f32ec070f23ced9f50e704d854d775da52591ab'
RUNTIME_VERSION = '0.1.39'
IQ3XXS_SOURCE_COMMIT = '99f3dbd0b21d1401b3769e0c0d963913607f380b'
BASE_SOURCE_COMMIT = '99f3dbd0b21d1401b3769e0c0d963913607f380b'

ASSETS = {
    'Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf':
        (44888775360, '8f940bf64c9a62a4dce1935999ac8ed287bb90ff5a4e5cc22ddac767cd56557d'),
    'Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00002-of-00003.gguf':
        (44934400672, 'd19d80982e5fa2416a21342be71e5d8041c0c088720ac5cc54bca79a9eafb10c'),
    'Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00003-of-00003.gguf':
        (21946739488, '6be547f9575cfdf23314426d62fb65ef4052841dd156f7b77dd79a93da3e4087'),
}
TOTAL_BYTES = sum(size for size, _ in ASSETS.values())

# Reuse the already-pinned projector from the same publisher revision only. Q4 keeps it
# outside the three-shard parity asset list and records its hardlink in a separate sidecar.
VISION_PROJECTOR_FILENAME = 'mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf'
VISION_PROJECTOR_BYTES, VISION_PROJECTOR_SHA256 = IQ3XXS_ASSETS[VISION_PROJECTOR_FILENAME]
if (REVISION != IQ3XXS_REVISION or REPOSITORY != IQ3XXS_REPOSITORY or
        (VISION_PROJECTOR_BYTES, VISION_PROJECTOR_SHA256) !=
        (907543296, 'f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9')):
    raise ValueError('Q4_K_S CPU-vision projector must remain bound to the same-revision IQ3_XXS pin')

BUILD_OPTIONS = {
    'STRATA_ENABLE_CUDA': 'ON',
    'STRATA_NATIVE_EXPERTS': 'ON',
    'STRATA_ORCA_Q4KS_MMQ': 'ON',
    'STRATA_BUILD_TESTS': 'ON',
    'CMAKE_CUDA_ARCHITECTURES': '86',
}
PARITY_TESTS = ('native_expert_parity', 'ple_q5_parity')

# Isolated Linux x86_64 build tool. Never install into the shared .39 environment or system paths.
CMAKE_VERSION = '3.31.6'
CMAKE_WHEEL_FILENAME = 'cmake-3.31.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
CMAKE_WHEEL_BYTES = 27800904
CMAKE_WHEEL_SHA256 = '1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec'
CMAKE_WHEEL_URL = ('https://files.pythonhosted.org/packages/59/e8/096984b89133681533650b9078c5ed1c5c9b534e869b5487f22d4de1935c/'
                   + CMAKE_WHEEL_FILENAME)


def data_root_from(home: Path) -> Path:
    return home / '.local/share/localllm-strata'


def runtime_root(root: Path) -> Path:
    """Q4's runtime is intentionally outside strata_runtime's version manager."""
    return root / 'runtimes' / f'{PROFILE}-{RUNTIME_VERSION}'


def source_root(root: Path) -> Path:
    return runtime_root(root) / 'source'


def toolchain_root(root: Path) -> Path:
    return runtime_root(root) / 'tools'


def cmake_root(root: Path) -> Path:
    return toolchain_root(root) / f'cmake-{CMAKE_VERSION}'


def cmake_binary(root: Path) -> Path:
    return cmake_root(root) / 'cmake/data/bin/cmake'


def cmake_wheel(root: Path) -> Path:
    return toolchain_root(root) / CMAKE_WHEEL_FILENAME


def pinned_runtime_source(root: Path) -> Path:
    """Read-only source/environment pin reused as the Q4 build's toolchain base."""
    return root / 'runtimes' / RUNTIME_VERSION / 'source'


def model_root(root: Path) -> Path:
    return root / 'data/models' / PROFILE


def pack_root(root: Path) -> Path:
    return root / 'data/packs' / PROFILE


def profile_root(root: Path) -> Path:
    return root / 'profiles' / PROFILE
