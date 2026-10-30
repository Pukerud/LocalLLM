"""Pinned original-Qwen Q8_0 PLE table, shared read-only by two opt-in variants."""
from pathlib import Path

REPOSITORY = 'unsloth/Qwen3.8-Flash-Next-GGUF'
REVISION = '38bb39ee97821de2c9009abb7e93950eec396e66'
REPO_PATH = 'UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf'
SIZE = 54_400_261_312
SHA256 = '34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20'
SHAPE = [160, 320_001_536]
TENSOR = 'per_layer_token_embd.weight'
TENSOR_BYTES = 54_400_261_120
FOLDER = 'qwen-flash-next-ple-q8'
STATES = {'swift15-heretic-re': 'swift15-heretic-re-ple-q8',
          'iq3_s-sc117-abliterated': 'sc117-iq3s-ple-q8-trial'}
HIVE_PROFILES = {'swift15-heretic-re-ple-q8': 'swift15-heretic-re',
                 'iq3_s-sc117-abliterated-ple-q8-v0146': 'iq3_s-sc117-abliterated'}
READY_SUFFIX = (' PLE OVERRIDE: original-Qwen Q8_0 n-gram; mmap; separate Q8_0 state/cache; '
                'bounded live smoke passed; matched quality/speed A/B pending.')


def folder(root):
    return Path(root) / 'profiles' / FOLDER


def table_path(root):
    return folder(root) / Path(REPO_PATH).name


def url():
    return f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{REPO_PATH}'
