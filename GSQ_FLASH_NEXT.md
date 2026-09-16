# Experimental Flash-Next GSQ-RCO

`v1qwen38.sh --quickstart` now offers [6] Q2_0 (speed), [7] IQ2_XS, and [8] IQ3_XXS (publisher's quality recommendation). Production default [4] is unchanged. These are experimental, not measured performance improvements or replacements for Hauhau.

Source: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF pinned at `1c04b8102ca5346f1faf4d9914503e378d713021`. Both shards and BF16 projector are SHA256-verified. Identical second shards share one local file. Downloads including projector are approximately 67.3/68.9/76.7 GB. Existing pinned qwen4exp-capable upstream runtime `4cbe8b070bb040f3b95845408f100fbf5fb746f1` is reused, without modifying Hauhau's runtime.

Normal launch: vision, xhigh reasoning, Q8 KV, mmap and lazy PLE; no unverified MTP sidecar. On a host with at least two selected RTX 3090 24GB GPUs, Q2_0 now defaults to **one native 262144-token slot on the first two GPUs, split 23:25**. This balances the projector/input overhead better than an equal split. Explicit `QWEN38_GPU_INDICES` or `QWEN38_TENSOR_SPLIT` bypasses automatic tuning. IQ variants and other hardware retain the conservative 32768-token/all-selected-GPU configuration.

`QWEN38_GSQ_TUNED=0` restores the original all-GPU/32K behavior. `QWEN38_GSQ_CTX=131072` gives additional memory headroom; allowed range is 4096–262144. Native-262K allocation with short requests is not a full-context quality/throughput certification. Very large images or nearly full histories remain untested. Unlike Hauhau this is a qwen4exp MoE with a lazily mapped PLE table.

```bash
# Preparation only; does not stop serving:
bash v1qwen38.sh --download --profile gsq-q2
# Manual launch (same lifecycle cautions as other interactive profiles):
bash v1qwen38.sh --quickstart --profile gsq-q2
# Existing short benchmark and menu speed cache:
bash v1qwen38.sh --speed-test --profile gsq-q2
```

Replace gsq-q2 with gsq-iq2 or gsq-iq3. The existing speed-test uses smoke-mode reasoning-off, so those numbers are NOT normal xhigh throughput. Standard speed-all does not automatically download/test these large experimental models.

On Hive, enter the existing authorized maintenance/manual-hosting flow before switching away from a miner-managed server. Do not run alongside a renter or an already allocated GPU workload; this addition does not change Hive/osn settings or its production flight sheet. Downloads and menu/static command checks are not an inference-quality certification; first actual model load and speed results remain experimental until measured.

## Node .69 tuning measurements (2026-09-16)

Same Q2_0 weights, Q8 KV, BF16 vision, xhigh, layer mode; three short prompts, fixed seed, warmup excluded, capped at512 generated tokens. Capped samples are throughput measurements, not completed-answer quality passes. Repeated original four-GPU baseline brackets the candidate trials.

| Configuration | Short-prompt decode |
|---|---:|
| 4 GPUs, lazy PLE, 32K | about63 tok/s |
| 4 GPUs, CPU-resident PLE, 32K | about61 tok/s (rejected) |
| 2 GPUs, lazy PLE, 32K/128K/262K | about69–70 tok/s |
| 3 GPUs, lazy PLE, 262K | about64 tok/s |

Changing allocation capacity alone did not improve decode; reducing active GPUs did. At about9660 prompt tokens, the two-GPU262K trial measured about704 prefill tok/s and59 decode tok/s. Do not extrapolate short-prompt70tok/s to a populated262K context. Balanced two-GPU configuration is checked separately for arithmetic, strict JSON, exactly-one tool call, and a red/blue image with thinking enabled. No full-context generation is used.

Regression test: `bash tests/gsq-profile-test.sh`.
