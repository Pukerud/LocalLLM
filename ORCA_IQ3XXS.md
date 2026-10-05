# Orca Uncensored IQ3_XXS — separate experimental Strata choice

**[4] Orca** does not replace **[3] GSQ-RCO IQ3_S**, and selecting it while Strata is running only displays status.

**2026-10-05 runtime follow-up:** [STRATA_V0139.md](STRATA_V0139.md) records the side-by-side 0.1.39 upgrade and
single/two-slot measurements. One slot remains default; Orca's two-group pipeline failed concurrent-image validation
and is refused. The preparation provenance below refers to the preserved original runtime/assets, not a new conversion.
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

Orca settings: **native 262144 context, INT8 streaming KV with 32768 resident, prefill 512, four-GPU contiguous
automatic split, 2048 MiB reserve, MTP `--spec 4 --spec-min-p 0.5`, high maximum thinking**. Its publisher F16 GPU vision is configured with 1024 image
tokens. Both profiles use the same native context/streaming policy; IQ3_S retains BF16 vision and Orca its own F16
projector. The resident 32K is a GPU working window, not the total context. No RoPE extension is enabled.

The first Orca installation at `48e3df9` deliberately used 32K. After this update, migrate the verified cached profile:

```bash
sudo -n ./v1strata.sh --configure-native-context --profile orca-iq3_xxs
```

This validates the pinned asset records/runtime/pack, preserves historical preparation metadata/backups, and writes
only the Orca profile config/metadata. It neither downloads weights nor touches the active run-config/process.
That engine needs an owned, guarded restart to use the new context. Full explicit preparation also writes native
settings for fresh installations.

The existing Strata frontend/UI/OpenAI/Anthropic APIs are reused. Default trusted-LAN binding is `0.0.0.0:8080`;
`STRATA_HOST`, `STRATA_PORT`, `STRATA_API_KEY` apply. Model ID: `qwen3.8-flash-next-orca-iq3_xxs-strata`; aliases:
`orca`, `orca-strata`. No public unauthenticated exposure. One FIFO generation sequence by default; validated
0.1.39 two-slot/one-group batching is an explicit alternative in the same engine, not an extra worker.

## Native-context validation — 2026-10-04

On `.69`, both live profiles reported **262144** through `/health`, canonical/alias `/v1/models` entries, `/slots`,
and `/props`. Each passed bounded arithmetic, JSON, exactly one forced Oslo tool call and red/blue plus swapped-image
checks. The fixed Pi extension's real HTTP discovery/session registration reported 262144 for every ID/alias on both
servers. Linux CPU/controller tests passed **66 as user and root**; Node context tests passed **8 including the live API**
for each profile. The final session was returned to the user's latest selected IQ3_S profile.

The first unconstrained Orca image fixture failed strict JSON parsing. Repeating the bounded vision tests with explicit
`response_format: json_object` passed both swapped images; no weight, runtime, sampling default or MTP setting was
changed. One image response recorded 6 accepted / 11 drafted tokens, a tiny-request observation, not representative
MTP acceptance. Post-short-request Orca GPU used/free MiB were 19727/4400, 17585/6542, 16517/7610, 19039/5085:
not peaks or full-context guarantees. Raw local evidence is under `logs/native-context-20261004/` and
`profiles/orca-iq3_xxs/validation-native-context-20261004.json`.

## Validation scope

**Preparation/readiness and CPU mock-frontend tests are not real Orca inference.** Historical install-only evidence
preserved the original IQ3_S session; subsequent native-context validation is recorded separately, not retroactively
added to the initial preparation results. Native allocation plus bounded arithmetic/JSON/tool/image requests is not
a full-context quality, speed or peak-memory claim. No full-context generation and no RoPE extension past 262144.

Upstream [ORCA.md](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/ORCA.md)
reports IQ3_XXS text-only validation on different hardware. It is not a `.69` vision/performance validation.
