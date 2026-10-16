"""Immutable asset identity and isolated paths for the SC117 abliterated IQ3_S candidate."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROFILE = 'iq3_s-sc117-abliterated'
MODEL_ID = 'qwen3.8-flash-next-sc117-abliterated-iq3_s'
REPOSITORY = 'SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF'
REVISION = 'eec4e10a2c29b440a2ae85591fa11569ce752963'
MTP_SOURCE_VARIANT = 'stock/non-abliterated'
MTP_SOURCE_NOTE = ('This repository MTP is stock/non-abliterated; mtp-q2_0.gguf is the pinned source artifact, '
                   'not the root llama.cpp Q8 MTP.')


@dataclass(frozen=True)
class Asset:
    """One publisher-pinned file and its isolated local destination."""

    repo_path: str
    role: str
    size: int
    destination: str
    sha256: Optional[str] = None
    git_blob_oid: Optional[str] = None


_ASSET_ROWS = (
    ('IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf',
     'model_shard_1', 55_063_448_064,
     '40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c'),
    ('IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf',
     'model_shard_2', 28_800_138_432,
     '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113'),
    ('mmproj-Qwen3.8-Flash-Next-BF16.gguf',
     'vision_projector', 907_542_944,
     '2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355'),
    ('mtp-q2_0.gguf',
     'mtp_source_stock_non_abliterated', 889_017_216,
     'a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d'),
    ('strata/rt/dense.bin',
     'mtp_runtime_dense', 116_099_072,
     'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a'),
    ('strata/rt/experts.bin',
     'mtp_runtime_experts', 707_788_800,
     '09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f'),
    ('strata/rt/draft_vocab.bin',
     'mtp_runtime_draft_vocab', 425_196,
     'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'),
)

ASSETS = tuple(
    Asset(repo_path, role, size, f'data/models/{PROFILE}/{repo_path}', sha256=sha)
    if role in {'model_shard_1', 'model_shard_2', 'vision_projector'} else
    Asset(repo_path, role, size,
          f'data/mtp/{PROFILE}/source/mtp-q2_0.gguf' if role == 'mtp_source_stock_non_abliterated'
          else f'data/mtp/{PROFILE}/rt/{Path(repo_path).name}', sha256=sha)
    for repo_path, role, size, sha in _ASSET_ROWS
) + (
    Asset('strata/rt/dense.txt', 'mtp_runtime_dense_config', 1_880,
          f'data/mtp/{PROFILE}/rt/dense.txt',
          git_blob_oid='0a6b354ada9f26d07c4d51684089bdc2fda06a1b'),
)
ASSETS_BY_REPO_PATH = {asset.repo_path: asset for asset in ASSETS}
ASSETS_BY_ROLE = {asset.role: asset for asset in ASSETS}
ASSET_PATHS = tuple(asset.repo_path for asset in ASSETS)
MODEL_ASSETS = tuple(asset for asset in ASSETS if asset.role in
                     {'model_shard_1', 'model_shard_2', 'vision_projector'})
MODEL_TOTAL_BYTES = sum(asset.size for asset in MODEL_ASSETS)

# This is the existing base IQ3_S shard-2 location. It is a read-only, opt-in reuse
# source; no other base model/projector/MTP path is eligible for reuse.
SHARD2_REUSE_SOURCE_RELATIVE = (
    'data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf')


def _root(root: Path) -> Path:
    return Path(root)


def data_root(root: Path) -> Path:
    return _root(root)


def model_root(root: Path) -> Path:
    return _root(root) / 'data/models' / PROFILE


def asset_path(root: Path, asset: Asset) -> Path:
    if asset.repo_path not in ASSETS_BY_REPO_PATH or ASSETS_BY_REPO_PATH[asset.repo_path] != asset:
        raise ValueError('Asset is not in the immutable SC117 allowlist')
    return _root(root) / asset.destination


def shard1_path(root: Path) -> Path:
    return asset_path(root, ASSETS_BY_ROLE['model_shard_1'])


def shard2_path(root: Path) -> Path:
    return asset_path(root, ASSETS_BY_ROLE['model_shard_2'])


def projector_path(root: Path) -> Path:
    return asset_path(root, ASSETS_BY_ROLE['vision_projector'])


def mtp_source_root(root: Path) -> Path:
    return _root(root) / 'data/mtp' / PROFILE / 'source'


def mtp_runtime_root(root: Path) -> Path:
    return _root(root) / 'data/mtp' / PROFILE / 'rt'


def mtp_source_path(root: Path) -> Path:
    return asset_path(root, ASSETS_BY_ROLE['mtp_source_stock_non_abliterated'])


def pack_root(root: Path) -> Path:
    return _root(root) / 'data/packs' / PROFILE


def profile_root(root: Path) -> Path:
    return _root(root) / 'profiles' / PROFILE


def manifest_path(root: Path) -> Path:
    return profile_root(root) / 'prepared.json'


def pack_manifest_path(root: Path) -> Path:
    return profile_root(root) / 'native-pack.json'


def pack_request_path(root: Path) -> Path:
    return profile_root(root) / 'native-pack.request.json'


def strata_039_source(root: Path) -> Path:
    return _root(root) / 'runtimes' / '0.1.39' / 'source'


def shard2_reuse_source(root: Path) -> Path:
    return _root(root) / SHARD2_REUSE_SOURCE_RELATIVE
