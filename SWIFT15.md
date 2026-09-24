# Swift 1.5 Qwen3.8-27B — experimental menu profiles

Added 2026-09-24 from https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF.

- Menu9 / `swift15-q8`: Q8_0 fidelity reference, 29.05GB weights plus0.93GB F16 projector.
- Menu10 / `swift15-q4`: Q4_K_M speed/memory comparison, 17.44GB weights plus same projector.
- Both configure native262144 tokens, one slot, Q8 K/V, GPU vision, thinking on/xhigh/preserved, temperature1/top-p0.95/top-k20, layer split across selected GPUs.
- Independent from the older Swift BF16 option5. Production default4 and GSQ options6–8 unchanged.
- Uses existing qwen35-capable upstream runtime at4cbe8b070bb040f3b95845408f100fbf5fb746f1, without replacing Hauhau's patched runtime. Embedded native MTP enabled by default at draft depth3: `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`. No external draft file is needed. `--no-spec` disables it; `--spec native` explicitly enables it. `QWEN38_SWIFT15_MTP_N_MAX=2` can change depth (allowed1–7); depth3 is a starting setting, not a measured optimum. Unrelated FastMTP/DFlash overrides remain rejected.
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

## Official MTP verification (correction)

The [original model card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b#optional-mtp-decoding) explicitly states that the published weights include the base MTP head, and documents vLLM/SGLang use. Source config at bc7a1e10b689648585a3ef41494c8d84cf77271a has `mtp_num_hidden_layers: 1`; the safetensors index contains15 MTP tensors.

More importantly, direct partial-file parsing of BOTH pinned Q8_0 and Q4_K_M GGUF headers confirms `general.architecture=qwen35`, `qwen35.block_count=65`, `qwen35.nextn_predict_layers=1`, and15 `blk.64.*` head tensors including `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, and `nextn.shared_head_norm`. This corrects the initial menu's unnecessary MTP disablement. Do not mistake absence of a separate MTP filename for absence of embedded MTP weights.

The shared pinned runtime already supports qwen35 native `draft-mtp`, used by the existing TURBO/Swift profiles. Launcher tests verify correct flags and no external draft. Actual Swift1.5 native-MTP inference, acceptance rate, VRAM and speed have not yet been benchmarked on .69; official head availability is not a claim of measured acceleration. Native262K and vision remain configured, unchanged.

Tests: `bash tests/swift15-profile-test.sh` and `bash tests/gsq-profile-test.sh`.
