"""Data-only pins for the explicitly V3.1 IQ4_XS release; never a serving profile."""
from pathlib import Path

REPOSITORY = 'unsloth/Qwen3.8-Flash-Next-GGUF'
REVISION = 'd96201194f103c906994f283f504209dc8d2a10c'
QUANT_COMMIT = '814db05f88b3b827ea7cfcab5344fd218de494d2'  # Better UD 3.1 quants
PROFILE = 'unsloth-v31-iq4xs-q8-cvec-2slot'
RUNTIME = '0.1.46'
# Existing runtime extracts this pinned llama.cpp snapshot as ZIP, not a Git submodule.
GGUF_PY_COMMIT = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
GGUF_PY_FILES = 23
GGUF_PY_TREE_SHA256 = '06548579a064de90e0544d6a00ef27fd1271d0d74130b6d5c88abc5639f1000f'
CONTEXT, PARALLEL, GROUPS = 262144, 2, 2
CACHE_MIB, HISTORIES = 16384, 8
# role, repository path, local relative path, exact byte count, SHA-256
ASSETS = (
    ('model_1', 'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf',
     'models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf', 10946618,
     '93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef'),
    ('model_2', 'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf',
     'models/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf', 49604491008,
     '13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb'),
    ('model_3', 'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf',
     'models/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf', 15256783680,
     'f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87'),
    ('model_4', 'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf',
     'models/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf', 28800138432,
     '5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36'),
    ('ple_q8', 'engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file',
     'Qwen3.8-Flash-Next-engram-Q8_0.gguf', 54400261312,
     'e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a'),
)
# Only exact known original-Qwen table payloads are eligible for a copy + new header.
# BOTH old and reconstructed NEW complete-file hashes must match. Otherwise download normally.
REUSE = {
    'model_4': ('data/models/iq3_s-sc117-abliterated/IQ3_S/'
                'Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf',
                '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113'),
    'ple_q8': ('profiles/qwen-flash-next-ple-q8/'
               'Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf',
               '34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20'),
}
VECTOR_NAME = 'Qwen3.8-Flash-Next-experimental-speed-projection.gguf'
VECTOR_SHA = 'ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b'
VECTOR_SIZE = 483520
ROUTING_SEED_SIZE = 196632
ROUTING_SEED_SHA = '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
# Keep the already-supported stock/non-abliterated Qwen Strata MTP, not llama.cpp's incompatible Q8 MTP layout.
VISION_REL = 'data/models/iq3_s-sc117-abliterated/mmproj-Qwen3.8-Flash-Next-BF16.gguf'
VISION_SIZE = 907542944
VISION_SHA = '2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355'
MTP_REL = 'data/mtp/iq3_s-sc117-abliterated/rt'


def folder(root):
    return Path(root) / 'profiles' / PROFILE


def url(row):
    return f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{row[1]}?download=true'


def state(root_home):
    return Path(root_home) / '.local/state/locallm-strata' / (PROFILE + '-trial')
