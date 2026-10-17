# Orca Uncensored Q4_K_S — bounded live validation recorded

**Status:** all publisher-pinned shards and both real-shard parity tests passed. Under the HostLLM lease, Q4 passed the initial 32K text-only functional/MTP/memory gates, then a native-context/CPU-vision probe: a synthetic magenta-square image was identified correctly, and one 260,996-token prompt (99.56% of the configured 262,144 window) returned its tail marker. The future profile is configured for native 262144, INT8 KV, 32768 resident, CPU vision, and one slot. HostLLM option [4] routes to Q4 .39; IQ3_XXS weights and pack were later removed, so it is not a runnable rollback. The previously verified Swift/Qwen Q8 profile remains the explicit fallback. **These are bounded probes, not a general long-context quality, throughput, worst-case-memory, multi-user, or sustained-load qualification.**

The 32K report records arithmetic, strict JSON, tool-call, and health passes; MTP drafted 38 tokens and accepted 36. Peak per-GPU memory was 23,340 / 21,948 / 20,494 / 23,150 MiB (sum of per-device peaks: 88,932 MiB); minimum available host RAM was 49,938 MiB. The native/vision report records the synthetic-image answer `Magenta`; the long prompt used 260,996 input tokens, returned `Q4-NATIVE-CTX-8F31`, and finished in 312.6 seconds. Minimum available RAM was 44,293 MiB. Per-GPU peaks were 23,500 / 22,416 / 20,962 / 23,310 MiB (sum: 90,188 MiB); the four cards are separate memory pools, not one shared pool. The prompt was deliberately repetitive and tests one tail retrieval only.

Reports are retained at `profiles/orca-q4_k_s/validation-32k.json` and `profiles/orca-q4_k_s/validation-native.json` on `.69`; staged raw test output is under `.q4ks-staging-20261009`. `validation-native.json` is mode 0600 and bound to the live server identity and current config hash.

`native_expert_parity` reported zero failures at selected layers 0, 5, 6, 20, and 47. `ple_q5_parity` reported 320,001,536 rows with max absolute difference `0.000e+00`. These verify the tested kernels/tables against their references; they are not end-to-end model inference or quality tests.

The IQ3_XXS model and compatibility pack were removed from `.69` to reclaim disk, so the retained profile/evidence metadata is not a usable CLI rollback. The Q4 build and `.39` selection do not change that state. Do not attempt an IQ3_XXS start unless its assets are restored and readiness passes; the Swift/Qwen Q8 profile is the previous runnable fallback.

## Exact pins and separate storage

- Model repository: `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`
- Model revision: `e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc`
- Separate engine source: Strata **0.1.39**, commit `6f32ec070f23ced9f50e704d854d775da52591ab`
- Base Flash-Next draft/MTP source: existing original source pin `99f3dbd0b21d1401b3769e0c0d963913607f380b`; the bounded 32K run measured 38 drafted / 36 accepted tokens.

| Original shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf` | 44,888,775,360 | `8f940bf64c9a62a4dce1935999ac8ed287bb90ff5a4e5cc22ddac767cd56557d` |
| `Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00002-of-00003.gguf` | 44,934,400,672 | `d19d80982e5fa2416a21342be71e5d8041c0c088720ac5cc54bca79a9eafb10c` |
| `Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00003-of-00003.gguf` | 21,946,739,488 | `6be547f9575cfdf23314426d62fb65ef4052841dd156f7b77dd79a93da3e4087` |

Total shard download: **111,769,915,520 bytes (~111.77 GB / 104.1 GiB)**. The original shards remain in place. Expected storage is separate:

```text
data/models/orca-q4_k_s/                 # three publisher-pinned original shards
data/packs/orca-q4_k_s/                   # own ~1.43 GiB --compat-bf16 pack/tokenizer
profiles/orca-q4_k_s/                    # config, provenance, parity and validation records
runtimes/orca-q4_k_s-0.1.39/source/      # separate CUDA/native-expert Q4-enabled build
```

This runtime is deliberately **not** part of `strata_runtime.py`'s selected-runtime manager. It does not change
`runtime-selection.json`, or replace/modify normal `.38`, `.39`, or `.41` engines. The current `iq3_s` default
and `.39` selection are not changed; the IQ3_XXS weights/pack were separately removed from `.69`.

## Explicit preparation and parity gates

The helpers do not stop services or start a model. The combined preparation command downloads/verifies the exact
shards, checks out a separate Q4-only source tree, reuses the exact llama.cpp dependency from the pinned `.39`
runtime, builds the native engine and parity binaries, packs shard 1 with `--compat-bf16`, and writes a **32K,
single-slot, text-only** Q4 config. It does not run the parity programs or model inference.

```bash
cd /home/user/LocalLLM
./v1strata.sh --prepare-orca-q4ks --download-only  # optional: downloads/checksums only
./v1strata.sh --prepare-orca-q4ks
./HostLLM.sh --verify-q4ks-parity  # requires no active LLM or pre-existing pause lease
./v1strata.sh --check-ready --profile orca-q4_k_s
```

Preparation requires the already prepared, exact side-by-side Strata 0.1.39 runtime and its read-only private Python
environment, plus the existing CUDA 12.9 toolchain. CMake 3.31.6 is fetched as a pinned Linux x86_64 wheel
(27,800,904 bytes; SHA-256 `1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec`) and extracted only under
`runtimes/orca-q4_k_s-0.1.39/tools/`; it does not modify the shared `.39` environment or system CMake. The exact
`.39` source is built separately with CUDA, native experts, `STRATA_ORCA_Q4KS_MMQ=ON`, and `STRATA_BUILD_TESTS=ON`.
Downloads are promoted only after full-file SHA-256 verification. Existing completed mismatches and unproven partial
packs/builds are never overwritten. HF credentials are read privately and never logged. Preparation and parity must
run as the normal operator; both reject effective UID 0, so CUDA/model binaries are never launched as root. Data
paths are resolved and constrained beneath `STRATA_DATA_ROOT`.

Before readiness permits [4], both parity programs must pass against all publisher-pinned shards. Run them only through HostLLM's guarded parity command above; do not launch the CUDA binaries directly.

Run both real-shard tests only through `./HostLLM.sh --verify-q4ks-parity`: it refuses an active LLM or
pre-existing pause lease, uses HostLLM's identity-bound miner/OctaSpace pause-and-restore lifecycle, verifies
Docker/GPU/port idleness, and monitors for newly appearing workloads. If another workload appears, it terminates
only its own identity-verified parity process; renters and unknown jobs are never stopped. Raw
`v1strata.sh --verify-orca-q4ks-parity` is an internal low-level command and requires the matching one-time
HostLLM lease ID and port; the wrapper supplies them. It independently rejects root. Run HostLLM as the normal operator (never with `sudo`), and
do not launch the CUDA binaries directly. Both parity programs passed on `.69` before the recorded live runs.

## Engine profile and recorded gates

The Q4-specific runtime must keep all three shard filenames together. Strata discovers shards 2 and 3 from shard 1.
`--native` and `--ple-gguf` both point to shard 1. The profile uses the model's own packed tokenizer, Q5_0 PLE via
`--ple-io mmap`, original Flash-Next MTP, one slot, INT8 KV, and no `experts.bin` override or generic IQ3 pack.
The initial configuration was text-only/32K. On `.69`, it passed health, arithmetic, strict JSON, tool call, positive
measured memory, and MTP measurement (38 drafted / 36 accepted). Only after recording that identity-bound report was
the future profile migrated to 262144 context with INT8 KV and 32768 resident, then CPU vision was enabled and tested.
The 32K result does not stand in for native-context or image validation.

HostLLM checks `--check-ready` before calling `hosting begin`; missing preparation/parity therefore cannot pause the
miner or OctaSpace. If another Strata profile is already running, menu option [4] displays its status and leaves it
alone. The initial stage did not enable vision, batching, or a parallel override. After the required 32K gates,
`.69` was explicitly migrated to native context and CPU vision, then tested under HostLLM. The configured profile
remains single-slot; readiness validates pins/configuration while stored live reports describe the limited probes.

## Recording the 32K live gates and later migration

After real-shard parity succeeds, the first Q4 inference run must use **only the initial text-only 32K profile**.
A successful readiness check is not a model test. The `.69` 32K run was recorded from the live, identity-verified
Q4 server; its report was passed to the migration gate:

```bash
./v1strata.sh --record-q4ks-32k-validation /path/to/q4ks-32k-report.json --profile orca-q4_k_s
./v1strata.sh --configure-native-context --profile orca-q4_k_s
```

The report must bind to the exact Q4 source/runtime/config/shard hashes and record `context: 32768`, `parallel: 1`,
`text_only: true`, passing health/arithmetic/strict-JSON/tool-call and memory-recorded gates, a positive measured
memory peak, and original MTP `drafted_tokens`/`accepted_tokens`. For example (fill hashes/measurements from the
pinned preparation and actual run):

```json
{
  "profile": "orca-q4_k_s",
  "model_id": "qwen3.8-flash-next-orca-q4_k_s-strata",
  "source_commit": "6f32ec070f23ced9f50e704d854d775da52591ab",
  "runtime_version": "0.1.39",
  "runtime_sha256": "<prepared Q4 engine SHA-256>",
  "asset_sha256": {
    "Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf": "8f940bf64c9a62a4dce1935999ac8ed287bb90ff5a4e5cc22ddac767cd56557d",
    "Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00002-of-00003.gguf": "d19d80982e5fa2416a21342be71e5d8041c0c088720ac5cc54bca79a9eafb10c",
    "Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00003-of-00003.gguf": "6be547f9575cfdf23314426d62fb65ef4052841dd156f7b77dd79a93da3e4087"
  },
  "config_sha256": "<sha256 of profiles/orca-q4_k_s/config.json>",
  "context": 32768,
  "parallel": 1,
  "text_only": true,
  "checks": {"health": true, "arithmetic": true, "strict_json": true, "tool_call": true, "memory_recorded": true},
  "memory_peak_mib": 0,
  "mtp": {"drafted_tokens": 0, "accepted_tokens": 0}
}
```

Replace the zero/placeholder measurements with actual positive measurements; the example is not a passing report.
MTP acceptance is required as a measurement, not assumed beneficial. The recording command additionally verifies the
live server identity, model/config and exact pinned runtime. It does not change a running config or restart a server.

Only the explicit migration command, after those checks, changes the **future Q4 profile config** to native
262144, INT8 KV and 32768 resident. It preserves the original 32K config and validation evidence, does not alter
the active private `run-config.json`, and does not touch generic runtime selection or any IQ3_XXS asset. The
subsequent `.69` probe populated 260,996 tokens and retrieved one tail marker; this does not qualify general
262K-context quality or sustained behavior. Native migration remains single-slot; CPU vision is configured below.

## CPU-vision configuration and bounded image probe

Only after the native-context migration passes readiness, CPU vision may be configured for **future** Q4 launches; this step is already complete on `.69`:

```bash
./v1strata.sh --enable-q4ks-vision --profile orca-q4_k_s
./v1strata.sh --rollback-q4ks-vision --profile orca-q4_k_s  # explicit 262K text-only rollback
```

The enable operation reuses the same-revision IQ3_XXS projector only when its existing preparation manifest and
full-file pin are valid, the file is read-only, and a same-filesystem hardlink is safe. The exact projector pin is
`mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, 907,543,296 bytes, SHA-256
`f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9`. Otherwise it fails closed; it does not
invoke token/auth/download helpers. It reuses the prepared `.39` `engine/strata-vision` validated runtime binary,
sets vision `gpu: false`, and adds no VRAM reserve or new build/helper. `vision-prepared.json` records this provenance
separately; the Q4 three-shard parity manifest and shard asset list are unchanged.

On `.69`, the pinned IQ3_XXS projector was full-hash verified, made read-only, and linked into Q4 before the IQ3_XXS
model/pack cleanup. The Q4 hardlink remains the CPU-vision input; it is not evidence that the IQ3_XXS model is still
runnable. The CPU helper and projector pins are recorded separately in `vision-prepared.json`; the Q4 three-shard
parity inputs remain unchanged.

The operation updates only the future profile configuration plus its migration hash and vision-preparation files;
it does not alter the active private `run-config.json`/server or generic runtime selection. The rollback restores the
exact 262K text-only preset and coherent migration hash while retaining projector/provenance files. The original
32K report remains identity-bound **text-only** evidence. The later native/vision report is bound to the native
config hash and live identity, but its one synthetic image and one tail-marker retrieval do not establish general
image or full-context quality.

## Operational scope

The pinned 111.77 GB shards, separate 0.1.39 CUDA build, compatibility pack, preparation manifest, parity record,
32K report and native/vision report are present under Q4-only paths on `.69`. The future profile is native 262144,
INT8 KV / 32768 resident, CPU vision, and one slot. Q4 was stopped through its identity-bound launcher after
validation; the previous miner and OctaSpace state were restored. IQ3_XXS model/pack assets were later removed; the
at that time, the generic `.39` runtime and normal default remained intact; the current Orca Q4 standard is `.44.1` and `.39` is rollback-only. No renter, Docker workload, or unknown GPU job was stopped.

Upstream technical basis: [Strata `ORCA_Q4_K_S.md` at the pinned 0.1.39 commit](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/ORCA_Q4_K_S.md).
