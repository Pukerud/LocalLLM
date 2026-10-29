"""Pinned publisher data; no code from the model repository is executed."""
from pathlib import Path

PROFILE = 'swift15-heretic-re'
MODEL_ID = 'Swift-1.5-Flash-Next-Heretic-ReverseEngineer'
REPOSITORY = 'trubisky/Swift-1.5-Flash-Next-Heretic-ReverseEngineer'
REVISION = 'a7a90ef46b79b746c484d094fd5814c359543d77'
RUNTIME_VERSION = '0.1.46'
STATE_NAME = 'swift15-heretic-re'
NATIVE_CONTEXT = 262144
ASSETS = (('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00001-of-00008.gguf', 3208309664, '9344341eab2c2d9d59c8a4a9533c8cfc515547675d7a139d17957968a3019c5b'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00002-of-00008.gguf', 28800138432, '56a185873792b560485ee971ba5c4f9c5558def5e3e70dbf0f8925e9849eb513'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00003-of-00008.gguf', 13700060352, 'e6c22d9c0a22e39fb25decc81d8862667d8184907c623b182d9e2771ebfb9e03'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00004-of-00008.gguf', 13766580576, 'a8e29ba699751bf05250824b7075aee2383048b39290ab55fb50c677925a1a37'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00005-of-00008.gguf', 13643236928, 'afa6f9afdeccef869b8f9ed1b3800246a626a9ff4e34859926800ed17e0cbdec'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00006-of-00008.gguf', 13670914048, 'da51b3a9d3f485a54e69f16338f7f1dc21f0032b470116c75c508f1422446848'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00007-of-00008.gguf', 13751645536, 'a232c44cfc9fe5555ab5dde1c629120eada854f598a8c26247aac552b68697c2'), ('Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-00008-of-00008.gguf', 8352778560, '2148d4cf9c7d03c953ec3c95fc28f23cbad13547be4ebfd4ecf4d590d1498cd3'), ('strata-pack-heretic-v2-70g-fix.zip', 1527244892, '0dd16206360250679997ed007ba095e5b53083b4658d1d3da0ca1da19028818c'))
SHARDS = ASSETS[:-1]
ZIP = ASSETS[-1]
PACK_ROOT = 'heretic-v2-70g-fix'
PACK_FILES = ('compat-bf16.json', 'dense.bin', 'index.txt', 'native_experts.txt',
              'conversions.json', 'tokenizer/chat_template.jinja', 'tokenizer/vocab.json',
              'tokenizer/tokenizer.json', 'tokenizer/token_type.json', 'tokenizer/merges.txt')
# The publisher pack names its original eight-way split this way. Preserve the
# pack bytes by downloading the identical published shards under those names.
def local_name(name):
    return name.replace('FIX-', 'FIX-00001-of-00008-', 1)

def profile_root(root):return Path(root)/'profiles'/PROFILE

PROJECTOR = ('mmproj-Qwen3.8-Flash-Next-BF16.gguf', 907542944,
             '2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355')
# Same stock/non-abliterated Flash-Next MTP head as retained SC117. Drafts are
# verified by the actual target; this is not claimed to be a Heretic-trained head.
MTP = (('dense.bin',116099072,'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a'),
       ('dense.txt',1880,'8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac'),
       ('experts.bin',707788800,'09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f'),
       ('draft_vocab.bin',425196,'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'))

# Derived from the exact SHA-pinned publisher ZIP above, not caller manifests.
PACK_PINS = {'compat-bf16.json': (41981, '23b044960f9353fbc4390c95714600426fb9a1ab95b6ba2b4b61dd6560141299'), 'tokenizer/chat_template.jinja': (8952, 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'), 'tokenizer/vocab.json': (5737005, '4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4'), 'tokenizer/tokenizer.json': (541, 'c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf'), 'tokenizer/token_type.json': (744960, '5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b'), 'tokenizer/merges.txt': (3353258, '4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332'), 'dense.bin': (1517063680, '92093eb794720774dd6601437df4fee5b603934b4a28000d74f58c84e69cbe70'), 'native_experts.txt': (6747, '6901fbe12716ddc4f359f2dba8e63aea80b085fc76b37e2c5842dffbdc90f5e1'), 'conversions.json': (191682, 'efad860e5a63b5b1759b0fd90448ef8e16bacdacfa209efda87a097b9bba5477'), 'index.txt': (93680, '3b24d58bcbece3d4927dd6c68bc0317186662c0e42becddd48f549e9763df306')}
