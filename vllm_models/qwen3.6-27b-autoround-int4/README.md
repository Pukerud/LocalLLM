---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.6-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen3_5
- autoround
- int4
- w4g128
- w4a16
- quantization
- vllm
- multimodal
- mtp
- speculative-decoding
---

# Qwen3.6-27B INT4 AutoRound

A **W4A16 (INT4 weight, FP16 activation) quantization** of [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), produced with [Intel's AutoRound](https://github.com/intel/auto-round).

## TL;DR

- **Base**: Qwen3.6-27B (27B dense VLM, Apr 21 2026)
- **Quant**: INT4 W4A16, group_size 128, symmetric
- **Tool**: `auto-round` (default recipe, 200 iters, torch.compile)
- **Size**: 18 GB (down from ~54 GB BF16) — **3x reduction**
- **MTP preserved**: The native Multi-Token Prediction head is kept in BF16, enabling **native speculative decoding** in vLLM (≈90% draft acceptance in our tests, ~2x throughput)
- **Accuracy**: Default AutoRound recipe preserves quality well; layer-norm weights, router layers, RMSNorm, `linear_attn.in_proj_a/b`, and MTP's fusion `fc` are kept unquantized (they're small and benefit from full precision)

## Quick inference with vLLM (with MTP speculative decoding)

Requires vLLM that supports Qwen3_5 MTP (most recent nightlies — tested with `eugr/spark-vllm-docker` fork `0.19.1rc1.dev39+g7055d32a7`):

```bash
vllm serve Lorbus/Qwen3.6-27B-int4-AutoRound \
  --dtype half \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype tq-t4nc \
  --max-num-seqs 3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --port 8888 --host 0.0.0.0 \
  --trust-remote-code \
  --compilation-config.cudagraph_mode none \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

Notes:
- `--kv-cache-dtype tq-t4nc` (TurboQuant 4-bit) halves KV memory vs fp8. Use `--kv-cache-dtype fp8` for mainline vLLM without the TurboQuant fork.
- `--compilation-config.cudagraph_mode none` is currently needed on Blackwell consumer (SM120/SM121) GPUs — CUDA graph capture hits a `cudaErrorStreamCaptureInvalidated` on the MTP module in some vLLM nightlies.
- `--speculative-config` enables the model's native MTP head as a built-in drafter.

### OpenAI-compatible request

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8888/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="Lorbus/Qwen3.6-27B-int4-AutoRound",
    messages=[{"role": "user", "content": "Write a quicksort in Python."}],
    max_tokens=512,
)
print(r.choices[0].message.content)
```

### Transformers (no spec decoding)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    "Lorbus/Qwen3.6-27B-int4-AutoRound",
    trust_remote_code=True,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("Lorbus/Qwen3.6-27B-int4-AutoRound")
msg = [{"role": "user", "content": "Explain quantum computing briefly."}]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(m.device)
print(tok.decode(m.generate(ids, max_new_tokens=256)[0]))
```

## Quantization details

| Field | Value |
|------|------|
| Base | `Qwen/Qwen3.6-27B` |
| Method | AutoRound (`intel/auto-round`), default recipe |
| Scheme | W4A16 (4-bit weights, FP16 activations) |
| Bits | 4 |
| Group size | 128 |
| Symmetric | yes |
| Packing format | `auto_round:auto_gptq` |
| Unquantized layers | `linear_attn.in_proj_a/b`, `mtp.fc`, all LayerNorms and RMSNorms, router gates |
| Calibration samples | 128 (default) |
| Iterations | 200 |
| torch.compile | enabled |
| GPU used for quant | 1× RTX 5090 (32 GB, SM120), `low_gpu_mem_usage=True` |
| Quant wall time | ~1h 40min |

### Unquantized layers — why

- **`linear_attn.in_proj_a/b`**: these are low-rank projections in Qwen3.6's Gated DeltaNet. Their shapes are not divisible by 32 (group_size), so AutoRound skips them. They account for a tiny fraction of parameters.
- **`mtp.fc`**: the Multi-Token Prediction fusion layer. AutoRound initially quantized it to GPTQ-packed INT4, but vLLM's `Qwen3_5MTP` loader expects an unquantized `fc.weight`. We dequantized it to BF16 so MTP works natively. If you use this quant **without** MTP, the `fc` weight is still there and harmless.
- **Norms, routers**: precision-sensitive and very small.

### MTP fix — what's different from a vanilla AutoRound run

A plain `auto-round` run on a Qwen3.5/3.6 model packs `mtp.fc` as INT4. In that form, vLLM skips
loading the layer entirely (param name mismatch between `fc.qweight` and the expected `fc.weight`),
which makes MTP speculative decoding produce **0% acceptance**.

This release dequantizes `mtp.fc` back to BF16 after AutoRound finishes. The layer is only
~100 MB (5120 × 10240 × 2 bytes) so the file size impact is negligible. Result: MTP works
out of the box and reaches **~80-90% draft acceptance** on typical prompts.

## Performance

Benchmarked on **1× RTX 5090 (32 GB)** with vLLM + TurboQuant 4-bit KV cache + MTP:

| Prompt type | max_tokens | Throughput |
|-------------|-----------:|-----------:|
| "Write a haiku" | 128 | **58 tok/s** |
| "Explain quantum computing in 3 paragraphs" | 256 | **60 tok/s** |
| "Write 8 paragraphs about deep learning history" | 1024 | **60 tok/s** |
| "What is 127*83? Show reasoning" | 256 | **61 tok/s** |

With MTP off (`--speculative-config` removed): **~32 tok/s**. The 2x speedup comes from MTP speculative decoding with ~85% acceptance.

## Known limitations

- The model is a **vision-language model** — you can feed images in OpenAI-compat messages with an `image_url` content part. Image quantization was not the focus here; MoonViT encoder weights are kept at their original precision (BF16/FP16 as in the base model).
- `bits: 4` at `group_size: 128` prioritizes throughput/memory over maximal accuracy. For accuracy-critical work, try the `auto-round-best` recipe (1000 iters, ~5-10x slower) or a higher bit width.
- Not tested extensively beyond 128K context. Qwen3.6's `partial_rotary_factor` RoPE scaling is preserved, so 262K should work.

## Reproduction

```bash
pip install auto-round-nightly

auto-round \
  --model Qwen/Qwen3.6-27B \
  --scheme W4A16 \
  --format auto_round \
  --output_dir Qwen3.6-27B-int4-AutoRound \
  --enable_torch_compile \
  --low_gpu_mem_usage \
  --device_map 0
```

Then dequantize `mtp.fc` for MTP compatibility — see the `dequant_mtp_fc.py` script below:

<details>
<summary>mtp.fc dequant script (click to expand)</summary>

```python
#!/usr/bin/env python3
"""Dequantize mtp.fc from GPTQ INT4 back to bf16 so vLLM's MTP loader picks it up."""
import json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("Qwen3.6-27B-int4-AutoRound")
EXTRA = BASE / "model_extra_tensors.safetensors"
INDEX = BASE / "model.safetensors.index.json"

tensors = {}
with safe_open(EXTRA, framework="pt") as f:
    meta = f.metadata() or {}
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

qw = tensors["mtp.fc.qweight"]    # [1280, 5120] int32
qz = tensors["mtp.fc.qzeros"]     # [80, 640] int32
sc = tensors["mtp.fc.scales"]     # [80, 5120] fp16

in_features = qw.shape[0] * 8     # 10240
out_features = qw.shape[1]        # 5120
group_size = 128
num_groups = in_features // group_size   # 80

def unpack_int32_4bit(packed, axis, factor=8):
    dev = packed.device
    shifts = torch.arange(0, 32, 4, device=dev, dtype=torch.int32)
    expanded = (packed.unsqueeze(axis + 1) >> shifts.view([8 if i == axis + 1 else 1 for i in range(packed.ndim + 1)])) & 0xF
    new_shape = list(packed.shape); new_shape[axis] *= factor
    return expanded.reshape(new_shape).to(torch.int8)

w_int = unpack_int32_4bit(qw, axis=0)   # [10240, 5120]
z_int = unpack_int32_4bit(qz, axis=1)   # [80, 5120]

w_grouped = w_int.view(num_groups, group_size, out_features).to(torch.float32)
w_fp32 = (w_grouped - z_int.unsqueeze(1).to(torch.float32)) * sc.unsqueeze(1).to(torch.float32)
w_final = w_fp32.view(in_features, out_features).t().contiguous().to(torch.bfloat16)   # [5120, 10240]

# Replace
for k in ("mtp.fc.qweight", "mtp.fc.qzeros", "mtp.fc.scales"):
    del tensors[k]
tensors["mtp.fc.weight"] = w_final

save_file(tensors, str(EXTRA), metadata=meta)

# Update index
idx = json.loads(INDEX.read_text())
for k in ("mtp.fc.qweight", "mtp.fc.qzeros", "mtp.fc.scales"):
    idx["weight_map"].pop(k, None)
idx["weight_map"]["mtp.fc.weight"] = EXTRA.name
# Recompute total_size
from collections import defaultdict
shard_sizes = defaultdict(int)
for sf in set(idx["weight_map"].values()):
    with safe_open(BASE / sf, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            shard_sizes[sf] += t.numel() * t.element_size()
idx["metadata"]["total_size"] = sum(shard_sizes.values())
INDEX.write_text(json.dumps(idx, indent=2))
```

</details>

## Acknowledgements

- [Alibaba / Qwen team](https://huggingface.co/Qwen) for the base [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) model
- [Intel AutoRound](https://github.com/intel/auto-round) team for the quantization framework
- [@eugr](https://github.com/eugr) for the [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) fork and TurboQuant KV cache work
- [vLLM project](https://github.com/vllm-project/vllm) for the inference engine and Qwen3_5 MTP support

## License

Apache 2.0 — same as [Qwen3.6-27B base](https://huggingface.co/Qwen/Qwen3.6-27B).

## Citation

If you use this quant, please cite the original Qwen3.6 release (see base model card) and the AutoRound paper:

```bibtex
@article{cheng2023autoround,
  title   = {Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author  = {Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal = {arXiv preprint arXiv:2309.05516},
  year    = {2023}
}
```
