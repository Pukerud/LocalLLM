# Orca Uncensored IQ3_XXS — separate experimental Strata choice

**[4] Orca** does not replace **[3] GSQ-RCO IQ3_S**, and selecting it while Strata is running only displays status.
It does not stop/restart/test the existing instance. Use [9] yourself when you actually intend to switch.
Swift remains the default and Hauhau remains the fallback; their settings are unchanged.

## Pinned assets and runtime

- Model: `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`
- Model revision: `e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc`
- Shared **existing**, not rebuilt/updated Strata source: `99f3dbd0b21d1401b3769e0c0d963913607f380b`
- Exact file sizes/SHA-256: [orca_assets.py](orca_assets.py)
- Both IQ3_XXS shards total **85,202,668,032 bytes** (79.35 GiB), plus the publisher's **907,543,296-byte F16 projector**.
  Download size is not resident RAM or VRAM.

Under `~/.local/share/localllm-strata/`:

```text
data/models/orca-iq3_xxs/       # own verified GGUFs + F16 mmproj
data/packs/orca-iq3_xxs/        # own dense/index/native-expert files + tokenizer + conversion manifest
profiles/orca-iq3_xxs/         # config.json + prepared.json, independent of original prepared.json
```

`--compat-bf16` converts selected small projections with nearest-even BF16 rounding. It does not restore original
BF16 precision; expert/embedding/PLE table bytes are not requantized. No pack/tokenizer is borrowed from GSQ-RCO.
The original model-appropriate Flash-Next MTP is reused, not a dense Swift/Hauhau/FastMTP/DFlash draft.

## Explicit preparation only

Hugging Face currently gates the files. Accept access at the model page and provide a read token via `HF_TOKEN`,
`HF_TOKEN_PATH`, or the owner's private `~/.cache/huggingface/token`. Never paste tokens into Git, command arguments,
or public logs. When using sudo, the owner is `SUDO_USER`, so the normal user cache is consulted.
For an environment token, use `sudo --preserve-env=HF_TOKEN`; no global configuration is changed by the helper.

```bash
cd /home/user/LocalLLM
sudo -n ./v1strata.sh --prepare-orca
./v1strata.sh --check-ready --profile orca-iq3_xxs
```

Preparation is resumable and verifies every complete download by full-file SHA-256. Mismatching completed files
are never overwritten. HTTP authorization failures fail fast; bearer credentials are stripped on CDN redirects.
Preparation uses one download stream, lowest CPU/idle IO priority, a private lock, CPU-only packing and the existing
Python/runtime. No Hugging Face package installation, CUDA build/calibration, model inference, miner/service command,
clock/driver/watchdog/flight-sheet change or original asset replacement occurs. Docker uncertainty/rentals block
preparation. Original runtime identity and protected configuration/service/marker state are checked for preservation.

## Hosting and limits

After preparation, reopen HostLLM to see **[4]**. The menu will not automatically switch away from a running model.
When no other Strata worker is running, the same identity/rental guards and persistent miner/OctaSpace pause lease
apply. Missing preparation fails **before** the pause. [9] stops the owned engine and safely restores prior state.
The two Strata profiles share a single ownership/state root; simultaneous blind launches are intentionally blocked.
An explicit profile-specific CLI stop refuses a different live profile.

Initial Orca settings: **32768 context, INT8 KV, prefill 512, four-GPU contiguous automatic split, 2048 MiB reserve,
MTP `--spec 4 --spec-min-p 0.5`, high maximum thinking**. Its publisher F16 GPU vision is configured with 1024 image
tokens but has not been validated locally. These initial settings are not the original IQ3_S defaults, which retain
native 262144/INT8 streaming KV and BF16 vision.

The existing Strata frontend/UI/OpenAI/Anthropic APIs are reused. Default trusted-LAN binding is `0.0.0.0:8080`;
`STRATA_HOST`, `STRATA_PORT`, `STRATA_API_KEY` apply. Model ID: `qwen3.8-flash-next-orca-iq3_xxs-strata`; aliases:
`orca`, `orca-strata`. No public unauthenticated exposure. One FIFO generation sequence, not extra concurrent slots.

## Validation scope

**Preparation/readiness and CPU mock-frontend tests are not real Orca inference.** No local model-quality, MTP
acceptance, image correctness, speed or peak-memory claim is made. The currently running IQ3_S instance is deliberately
not used for a test request or stopped for a canary. First real Orca tests must wait until the user authorizes switching:
bounded arithmetic/code/JSON/one-tool checks, swapped-image tests, high reasoning and MTP acceptance, then measurements.
No full-context generation and no automatic extension to 262K.

Upstream [ORCA.md](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/ORCA.md)
reports IQ3_XXS text-only validation on different hardware. It is not a `.69` vision/performance validation.
