"""Pins and isolated paths for the opt-in Orca Q4_K_S / Strata 0.1.41 candidate."""
from pathlib import Path

import orca_q4ks_assets as baseline
import strata_runtime as runtimes

PROFILE = 'orca-q4_k_s-v0141'
HOST_PROFILE = baseline.PROFILE
MODEL_ID = baseline.MODEL_ID
REPOSITORY = baseline.REPOSITORY
REVISION = baseline.REVISION
RUNTIME_VERSION = '0.1.41'
SOURCE_COMMIT = runtimes.PIN
BASELINE_RUNTIME_VERSION = baseline.RUNTIME_VERSION
ASSETS = baseline.ASSETS
PARITY_TESTS = baseline.PARITY_TESTS
BUILD_OPTIONS = {
    'STRATA_ENABLE_CUDA': 'ON',
    'STRATA_NATIVE_EXPERTS': 'ON',
    'STRATA_ORCA_Q4KS_MMQ': 'ON',
    'STRATA_BUILD_TESTS': 'ON',
    'CMAKE_CUDA_ARCHITECTURES': '86',
}


def profile_root(root: Path) -> Path:
    return root / 'profiles' / PROFILE


def candidate_root(root: Path) -> Path:
    return root / 'runtimes' / PROFILE


def source_root(root: Path) -> Path:
    return candidate_root(root) / 'source'


def build_root(root: Path) -> Path:
    return source_root(root) / 'build-orca-q4ks-v0141'


def baseline_profile_root(root: Path) -> Path:
    return baseline.profile_root(root)


def baseline_model_root(root: Path) -> Path:
    return baseline.model_root(root)


def baseline_pack_root(root: Path) -> Path:
    return baseline.pack_root(root)


def pinned_runtime_source(root: Path) -> Path:
    return root / 'runtimes' / RUNTIME_VERSION / 'source'


def cmake_cache_contract(cmake: Path, llama: Path) -> tuple[str, ...]:
    return (
        f'CMAKE_COMMAND:INTERNAL={cmake}',
        'CMAKE_BUILD_TYPE:STRING=Release',
        'STRATA_ENABLE_CUDA:BOOL=ON',
        'STRATA_NATIVE_EXPERTS:BOOL=ON',
        'STRATA_ORCA_Q4KS_MMQ:BOOL=ON',
        'STRATA_BUILD_TESTS:BOOL=ON',
        'CMAKE_CUDA_ARCHITECTURES:STRING=86',
        'CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda-12.9/bin/nvcc',
        f'STRATA_GGML_DIR:PATH={llama}',
    )
