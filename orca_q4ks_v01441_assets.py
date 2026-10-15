"""Pins and machine-specific paths for the opt-in Orca Q4_K_S / Strata 0.1.44.1 profile."""
from pathlib import Path

import orca_q4ks_assets as baseline
import orca_q4ks_v0141_assets as q4ks_v0141

PROFILE = 'orca-q4_k_s-v01441'
HOST_PROFILE = baseline.PROFILE
MODEL_ID = baseline.MODEL_ID
RUNTIME_VERSION = '0.1.44.1'
SOURCE_COMMIT = '296b59d2f73c8f04a805df29c2fa97655adace78'
LLAMA_COMMIT = '3cf03257f219afbe7334045ff7c6a06ac68c627d'

# This candidate intentionally consumes the exact artifacts from the completed .69
# calibration workspace. It is not part of the generic runtime-selection manager.
TUNING_WORKSPACE = 'calibration/orca-q4ks-machine-tuning-v01441-20261015'
SOURCE_WORKSPACE = 'calibration/orca-q4_k_s-v01441-prefill-20261015-retry2'
ARM = 'integration-auto-trim-windows2'
SOURCE_ARCHIVE = 'strata-v0144.1-296b59d-source.tar.gz'
SOURCE_ARCHIVE_SHA256 = '90a002c9d62f3aa723495cb11f82adb3645666f31c169050d3683f60cac24c7f'
SOURCE_MANIFEST_SHA256 = 'a8de552ff8096c190ece738b61ea594104e3bbd062c852e20d03a8a586260ea'
BUILD_MANIFEST_SHA256 = '06eb73365aa692efdc5f4cfec421563ed9f9b538d7d9724b8d3a725edd978d57'
CONFIG_SHA256 = '9cd479e3e889efb43b45745193e746d75f33065f06bb9d6db8862c9e870e186d'
INPUT_MANIFEST_SHA256 = 'c5d3731b4e7c2ae6be811a417c99f02f57f11359c2db55aa5938afbd24d553be'
RESULTS_SHA256 = 'ea6f9da791ae3ce4e214de361abc562ec412bd75cc7561562217eec8b84cbd6d'
ENGINE_SHA256 = '5776265eecde746a7542d682f63b455b076d582437d41d0281385211bd6f617e'
NATIVE_PARITY_SHA256 = 'e11804ab9f352bf97adf011acbfe3705e014edb41df4b4381f8b07322fa8978e'
PLE_PARITY_SHA256 = '759e9945220576b1706a4b59169b3a1eb74e1dc2c37ba9700a6653a3be5927f5'
VISION_SHA256 = '8a1d33ac3af8bc1630eb7a48b238de3f0086d30aa17d7f0c55f991738dbbfb5e'
KERNEL_LIBRARY_SHA256 = '8822fdf0db0e761d0ea39eb1270b84ce93c649d0052e62d1007892aa86405711'

BUILD_OPTIONS = {
    'CMAKE_BUILD_TYPE': 'Release',
    'CMAKE_CUDA_ARCHITECTURES': '86',
    'STRATA_ENABLE_CUDA': 'ON',
    'STRATA_NATIVE_EXPERTS': 'ON',
    'STRATA_ORCA_Q4KS_MMQ': 'ON',
    'STRATA_MMQ_KQUANTS': 'ON',
    'STRATA_BUILD_TESTS': 'ON',
}
VISION_BUILD_OPTIONS = {
    'STRATA_VISION_CUDA': 'OFF',
    'STRATA_PORTABLE': 'OFF',
}


def tuning_root(root: Path) -> Path:
    return root / TUNING_WORKSPACE


def source_workspace(root: Path) -> Path:
    return root / SOURCE_WORKSPACE


def source_archive(root: Path) -> Path:
    return source_workspace(root) / SOURCE_ARCHIVE


def profile_root(root: Path) -> Path:
    return tuning_root(root) / 'arms' / ARM


def source_root(root: Path) -> Path:
    return tuning_root(root) / 'strata-v0.1.44.1'


def build_root(root: Path) -> Path:
    return tuning_root(root) / 'build'


def engine_path(root: Path) -> Path:
    return build_root(root) / 'engine' / 'strata'


def vision_path(root: Path) -> Path:
    return build_root(root) / 'vision' / 'bin' / 'strata-vision'


def build_manifest_path(root: Path) -> Path:
    return tuning_root(root) / 'step2-build-manifest.json'


def source_manifest_path(root: Path) -> Path:
    return source_workspace(root) / 'step2-source-manifest.json'


def input_manifest_path(root: Path) -> Path:
    return profile_root(root) / 'input-manifest.json'


def results_path(root: Path) -> Path:
    return profile_root(root) / 'results.json'


def config_path(root: Path) -> Path:
    return profile_root(root) / 'config.json'


def python_path(root: Path) -> Path:
    """Use the same private Python environment that executed the validated candidate."""
    return q4ks_v0141.source_root(root) / '.venv' / 'bin' / 'python'
