# Experimental Flash-Next GSQ-RCO

`v1qwen38.sh --quickstart` now offers [6] Q2_0 (speed), [7] IQ2_XS, and [8] IQ3_XXS (publisher's quality recommendation). Production default [4] is unchanged. These are experimental, not measured performance improvements or replacements for Hauhau.

Source: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF pinned at `1c04b8102ca5346f1faf4d9914503e378d713021`. Both shards and BF16 projector are SHA256-verified. Identical second shards share one local file. Downloads including projector are approximately 67.3/68.9/76.7 GB. Existing pinned qwen4exp-capable upstream runtime `4cbe8b070bb040f3b95845408f100fbf5fb746f1` is reused, without modifying Hauhau's runtime.

Normal launch: vision, xhigh reasoning, one 32768-token slot, Q8 KV, layer splitting across selected GPUs, mmap and lazy PLE. No unverified MTP sidecar. `QWEN38_GSQ_CTX=65536` optionally adjusts context (4096–262144); larger allocations are not validated and can exhaust VRAM. Unlike Hauhau this is a qwen4exp MoE with an SSD-backed PLE table.

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

Regression test: `bash tests/gsq-profile-test.sh`.
