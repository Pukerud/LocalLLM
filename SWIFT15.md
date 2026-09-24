# Swift 1.5 Qwen3.8-27B — experimental menu profiles

Added 2026-09-24 from https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF.

- Menu9 / `swift15-q8`: Q8_0 fidelity reference, 29.05GB weights plus0.93GB F16 projector.
- Menu10 / `swift15-q4`: Q4_K_M speed/memory comparison, 17.44GB weights plus same projector.
- Both configure native262144 tokens, one slot, Q8 K/V, GPU vision, thinking on/xhigh/preserved, temperature1/top-p0.95/top-k20, layer split across selected GPUs.
- Independent from the older Swift BF16 option5. Production default4 and GSQ options6–8 unchanged.
- Uses existing qwen35-capable upstream runtime at4cbe8b070bb040f3b95845408f100fbf5fb746f1, without replacing Hauhau's patched runtime. No MTP enabled: no validated matching draft was established. Non-none speculative overrides are rejected.
- HF revision pinned at a1614465cfa35d04d3e8575d713fa779662b5eab. Weights/projector SHA256-verified; release license downloaded beside weights.

```bash
# Prepare without stopping the current service:
bash v1qwen38.sh --download --profile swift15-q8
# After entering the usual safe manual-hosting maintenance flow:
bash v1qwen38.sh --quickstart --profile swift15-q8
# Optional smaller weight quant:
bash v1qwen38.sh --quickstart --profile swift15-q4
```

Native262K is a configured allocation, not a claim of completed full-context quality or throughput validation. Menu/profile/command and download-routing tests are performed without loading the new model. Inference validation remains pending. These files download when requested; merely adding menu entries does not download them.

`--speed-test --profile swift15-q8` uses the launcher's EXISTING smoke mode:4K allocation and reasoning off. It measures a bounded decode scenario, NOT normal native262K/xhigh task latency. For meaningful comparison use normal launch plus identical bounded chat requests and report prompt time, output length, decode, and total completion time. Do not benchmark by generating an entire262K context. New profiles are excluded from automatic standard speed-all downloads/tests.

## License and expected gains

Swift Open License v1.0 is NOT unrestricted Apache2.0 for the adapted weights. See the publisher LICENSE for commercial-use revenue threshold/affiliate terms (US$1million) and enterprise licensing. The base Qwen license remains separate.

Publisher's measured reasoning-token reductions depend on workload. The headline9.18x is a specific agent demo's total time, not a general llama.cpp tokens/s speedup. Evaluation uses BF16/vLLM for main comparisons, not this pinned Q8/Q4 launcher. Do not claim GGUF quality parity or speed before measuring. Q8 is the initial reference; Q4 trades fidelity for footprint and potentially bandwidth.

Tests: `bash tests/swift15-profile-test.sh` and `bash tests/gsq-profile-test.sh`.
