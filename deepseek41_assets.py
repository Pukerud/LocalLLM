"""Pinned Abiray DeepSeek 4.1 Flash IQ2_XXS assets, with original BF16 vision.
No claim that startup at native context qualifies a populated one-million-token request.
"""
PROFILE = 'deepseek41-abiray-iq2_xxs'
MODEL_ID = 'DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS'
REPOSITORY = 'Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF'
REVISION = '99acb20f7c83a37d7f0ea6082e924f3d61615d72'
SUPPORT_REPOSITORY = 'unsloth/DeepSeek-V4.1-Flash-1017-GGUF'
SUPPORT_REVISION = 'f924b94f19c1fea02c06a65c8475b0a45e2b88b1'
SOURCE_COMMIT = '4b9e05a2a806a3fe24f7b2560d98626c37dade35'
APP_VERSION = '0.2.6'
RUNTIME_VERSION = '0.1.51'
NATIVE_CONTEXT = 1048576
STATE_NAME = 'deepseek41-abiray'
SHARDS = [f'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf' for i in range(1,5)]
ENGRAM = 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
MMPROJ = 'mmproj-BF16.gguf'
METADATA = 'UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
DRAFTER = 'dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'
MODEL_FILES = {
 SHARDS[0]: (39075714624, '89156abe782f5e6416563a91c7b0d0fc382a8fdbecb222d1d50fb54a5eac0868'),
 SHARDS[1]: (39388449504, 'ea63f4a003f90bbf324f07f585a3e51a6391673c3ef203d1b883e236747feb38'),
 SHARDS[2]: (39246891616, 'b347a95133cfa45c017945186a8da5f7ab69bcdeb5439e50c2f95605d543b113'),
 SHARDS[3]: (20870099072, 'f49bfdecba1b9d04289fb5d1a86efa3a62830202f3c7b271a664d2651d7d3473'),
 ENGRAM: (110595290816, 'eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a'),
}
SUPPORT_FILES = {
 METADATA: (5780589, 'cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f'),
 MMPROJ: (972208288, '9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9'),
 DRAFTER: (7976914240, '6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a'),
}
ENGINE_ARCHIVE = 'strata-dsv41-0.1.51-linux-x86_64-cuda-cuda13.tar.gz'
ENGINE_SIZE = 439928756
ENGINE_SHA256 = '88c9c54ecc3ffce6f2ddbef281a7066146694be85e4ace99d31e5cdd691f0603'
ENGINE_URL = 'https://github.com/Niko1221/Strata/releases/download/v0.2.6/' + ENGINE_ARCHIVE


def asset_sources():
    return [(name, size, digest, f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true')
            for repo,revision,files in [(REPOSITORY,REVISION,MODEL_FILES),(SUPPORT_REPOSITORY,SUPPORT_REVISION,SUPPORT_FILES)]
            for name,(size,digest) in files.items()]
