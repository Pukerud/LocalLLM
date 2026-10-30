# Swift 1.5 Flash-Next Heretic RE — pinned Strata profile

2026-10-29; `.69`, four RTX 3090s. Profile/state **`swift15-heretic-re`**, direct HostLLM **[3]**, Hive **[14]**. Swift's existing dense Q8 profile remains the overall default. Heretic's real native window is **262144**, not DeepSeek's historical 1M window.

**2026-10-30 addition:** original IQ4_NL PLE remains [3]/Hive [14]; opt-in Q8_0 PLE uses direct **[5]** / Hive **[16]**, the same isolated .46 engine/tuning and separate state/cache. See [PLE_Q8.md](PLE_Q8.md). The tests below are the original IQ4_NL profile's historical evidence, **not Q8 inference/quality qualification**. Initial no-stop publication preserved SC117. Later explicitly authorized Q8 [5]/[16] functional smoke and a 259835-token Hive retrieval passed; see [the separate live receipt](results/ple-q8-live-menus-20261030.json). Original SC117 .46/IQ4_NL was restored afterward. These bounded tests are not a matched quality/speed A/B.

## Publisher inputs

- Repository: `trubisky/Swift-1.5-Flash-Next-Heretic-ReverseEngineer`.
- Immutable revision: **`a7a90ef46b79b746c484d094fd5814c359543d77`**.
- All eight `Qwen3.8-Flash-Next-Heretic-V2-70G-FIX-0000X-of-00008.gguf` shards: publisher mixed IQ4_NL/custom recipe. Exact sizes/SHA-256 are in `heretic_assets.py`; no target tensors are omitted or requantized.
- Supplied `strata-pack-heretic-v2-70g-fix.zip`: **1,527,244,892 bytes**, SHA-256 **`0dd16206360250679997ed007ba095e5b53083b4658d1d3da0ca1da19028818c`**. It contains data only: dense/index/expert mapping/tokenizer and publisher BF16-conversion reports. Exact extracted sizes/hashes are pinned separately; no downloaded code is executed.
- The pack names its original split with a doubled prefix (`...FIX-00001-of-00008-0000X-of-00008.gguf`). Downloads map the unchanged published bytes to those local names; the pack is not edited to conceal offset/name differences.
- CPU inspection reconstructs every expert tensor offset/type across **48 layers × 512 experts** (74,606,182,400 native expert bytes), including the cross-shard role mapping at layer 41, and checks every GGUF tensor against its actual file bounds. The native PLE table is in shard 2.
- BF16 Flash-Next projector and the stock/non-abliterated MTP head are independently copied from checksum-verified retained inputs. They live in this model's own `support/`, so future removal of unrelated model directories cannot break it. The drafter is **not claimed to be Heretic-trained**; the actual target verifies every draft. Target outputs/vision weights are not replaced by a draft model.

## Exact serving policy

Existing pinned Strata **0.1.46**, source **`90767252f420e25223c739d604145ad15911d48b`**, CUDA 12.9/sm_86, existing private Python and CPU-vision helper. **The inference executable is a separately built Heretic-only compatibility overlay**, explained below; the original .46 binary/source used by SC117 is unchanged.

- Native/served context **262144**, no context extension or silent fit/truncation.
- Four GPUs in their original order, split **14,24,36**, one serving slot/group.
- **BF16 CPU vision** (up to 1024 tokens/image), preserving GPU memory for the target.
- INT8 streaming KV with **32768 resident**, eager model/vision loading, idle unload disabled.
- `--spec 3 --pipeline-windows 2`, automatic prefill, trimmed stage weights.
- Conversation cache **8192 MiB / four parked histories**. These are not four concurrent serving slots. Large histories compete for the byte budget; four near-full-window histories fitting is not promised.
- High thinking default; temperature 1.0, top-p 0.95, top-k 20. Test calls can explicitly request greedy/no-thinking without changing defaults.
- Private learned expert-profile copy and isolated state; no writes to another model's learned profile.
- Trusted-LAN `0.0.0.0:8080`, **no required API key**. Host/DNS-rebinding, browser Origin and optional deliberately configured authentication remain intact. Never expose this unauthenticated service publicly.

Inherited tuning is a starting policy, not a new comparative performance qualification.

## Why the BF16 compatibility overlay is necessary

The supplied pack is structurally valid and the plain .46 engine loads it. **Its first real inference nevertheless failed**:

```text
verify: blk.0.attn_gate.weight is not served natively (run with --native)
```

`--native` was present. The publisher deliberately keeps that sensitive matrix **BF16**, while .46's GDN verify/prefill/decoder assumes a quantized native matrix. Loading/readiness alone therefore was not adequate testing.

`heretic_bf16_compat.py` adds an isolated, checksum-bound fallback to the existing BF16 projection kernels for **only layer 0's 2560×6144 attention gate**. Other native projection/error paths remain. It builds as the unprivileged operator from exact original source/static-library/compile/link inputs; shared .46 sources/binaries, model files, system CUDA/NVIDIA/kernel and power policy are not modified.

| Original source file | Exact original SHA-256 |
|---|---|
| `src/core/layer.cpp` | `f284c2ad8d48e82a658dbebfd24357943aba35e96237759d1765472dbe9bda2e` |
| `src/core/verify.cpp` | `db798abe2f2e77289d82cdc45d983c5ccf50a3de99fc4b5699ef02c32bb6b47c` |
| `src/prefill/prefill.cpp` | `a0890f6184c20fcb1729e7fa6ce79df8b5d42b710b310baac21974674208b6ad` |

Exact patched hashes, original static archives/flags/link/object hashes, implementation fingerprint and resulting executable SHA are recorded in `runtimes/heretic-0.1.46-bf16/prepared.json`. The profile binds that manifest hash and refuses source/artifact drift. This is not a generic .46 modification or a promotion of a different runtime release.

The actual **31,457,280-byte** gate in the supplied pack is bit-identical to the real GGUF slice: SHA **`2bce627de1b1aca8725741bb4ce7c2986230ab38de5288b90ffbd488ebe4dda5`**. No gate requantization, conversion, omission or lowered-quality workaround.

A four-row numerical test on that actual matrix passed:

- Verify/decode MMVF versus double-accumulation CPU oracle: relative RMS **9.50e-8**.
- BF16 high+low prefill GEMM versus the same oracle: relative RMS **3.62e-6**.
- Multi-row/single-row MMVF outputs: **bitwise equal**.

The test is `tests/heretic_bf16_gate_parity.cpp`; it uses existing CUDA/BF16 kernels and does not modify weights. It is a projection check, not general model-quality qualification.

## Preparation

No normal start downloads or upgrades anything. As the normal operator:

```bash
# Only if the root-owned runtimes parent prevents creation of this NEW directory:
sudo install -d -o "$USER" -g "$(id -gn)" -m 700 \
  "$HOME/.local/share/localllm-strata/runtimes/heretic-0.1.46-bf16"

./v1strata.sh --prepare-heretic --download-only
./v1strata.sh --prepare-heretic --prepare-compat
./v1strata.sh --prepare-heretic --prepare-pack
```

The already prepared pinned .46 build and its static build inputs are required; source/library hashes are fixed, not accepted from caller manifests. Projector/MTP support staging currently requires the pinned SC117 support inputs locally. Missing/drifting files fail closed rather than download another variant silently. `--prepare-compat` also upgrades the exact stopped pre-overlay Heretic config with small backups; another lease/rental or a live Heretic family blocks activation.

## Validation and scope

**Completed:** actual root-terminal [3] direct hosting and [14] Hive selection/start, no-key LAN health/models/web, arithmetic `63`, high-thinking arithmetic `2099` with observed reasoning, swapped red/green and ordered two-image native CPU vision, same-origin requests and hostile Host/Origin **403**, complete SSE `4`/finish/`[DONE]`, and exactly one declared `add_integers(20,22)` call (format only; no tool executed).

Four independent histories each restored the correct private fixture code. Followups reused **4736/4736/5136/4736 tokens** with native restore counters, under the actual **8192 MiB/four-history** policy. Browser Monitor showed the correct RAM budget/slots and live thinking; a subsequent operator generation was left untouched.

One counted **259835-token** prompt (99.12% of the full window) returned tail marker **271828**, with no API cache reuse, in **190.67 s**. Native prefill timing was **1364.4 tok/s**. One-second telemetry (182 samples) recorded per-GPU maxima **20962/19748/18836/23894 MiB** and **70/71/66/72°C**. This is one populated-window retrieval, not broad long-context or stability qualification.

A separate bounded 192-token code diagnostic reported **148.8 tok/s**, 140 draft tokens/125 accepted. It is not a matched performance comparison or optimal-tuning claim.

Direct Ctrl+C stopped only its identity-bound frontend/native/vision family and released its own fresh lease, restoring its actual prior inactive admission state. Final normal Hive has token-bound ownership, OctaSpace restored **active**, no pause lease or pending recovery journal, and was left serving Heretic. **418 Python tests OK (one skipped), seven shell suites and the actual-matrix CUDA parity test passed.**

[Sanitized results](results/heretic-re-menu3-menu14-20261029.json). Raw fixtures, root-menu PTYs, receipts, failures and telemetry: `/var/lib/localllm-strata/heretic-20261029/`. These bounded tests do not establish broad quality, long-running stability, concurrent throughput or optimal tuning.

The first direct attempt was safely blocked when restoring OctaSpace restarted retained SC117; no server was adopted or killed to override that guard. Subsequent maintenance held admission closed during direct testing. The originally orphaned SC117 session was stopped **only after explicit operator approval** and exact recorded identity checks. Retained SC117 then cold-started and answered `63` successfully after Orca/DeepSeek weights were absent.

## Retirement of old model weights

Only the explicitly authorized Orca Q4_K_S model/pack/projector roots and DeepSeek model-profile data root were removed: **373,291,543,291 logical bytes**, with shared hardlinks distinguished from physically reclaimed space. Original inventory and the narrow intra-plan hardlink-count adjustment were retained. No active mapping/FD referenced those roots; no workload was stopped for deletion. Runtime/build/preset metadata and user conversation state were retained.

SC117 now reads its exact SHA-pinned serving-preset template without checking retired Orca weights. Its own model/pack/MTP, selected runtime, isolated state, bind and trial restrictions remain checked. Its retained .45/.46 routes remain **unqualified**. Retired Orca and DeepSeek code/reports are historical, not runnable installed rollbacks.
