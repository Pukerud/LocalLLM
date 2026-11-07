# Unsloth Dynamic V3.1 IQ4_XS — active Hive trial, 2026-11-07

**Operator-authorized switch and bounded functional tests completed. This model is now the ACTIVE Hive miner**, on the normal trusted-LAN URL **http://192.168.1.69:8080/**, **without an API key or Authorization header**. Swift remains the installation default, not the active selection.

## Menu and exact profile

| Entry | Meaning |
|---|---|
| HostLLM **[21]** | Select this model through the checked Hive lifecycle |
| Low-level Hive **[9]** | Idle-only next-start selection |
| HostLLM **[9]** | Stop this verified owned Hive model through its supervisor; never guess its launch token |

Launcher profile: `unsloth-v31-iq4xs-q8-cvec-2slot`.
Hive bundle: `unsloth-v31-iq4xs-q8-cvec-2slot-v0146`.
State: `~/.local/state/locallm-strata/unsloth-v31-iq4xs-q8-cvec-2slot-trial/`.

Retained alternatives remain Swift **[1]/[12]**, SC117 two-user **[7]/[18]**. No retired high-level [2–6]/[8]/[13–17]/[19] number was reused. Low-level [8] is still SC117, not TensorSharp. TensorSharp software/routes remain retired; **no TensorSharp or512K retry**.

## What actually runs

- **Unsloth UD-IQ4_XS Dynamic V3.1**, exact frozen repo `d96201194f103c906994f283f504209dc8d2a10c`, weight update `814db05f88b3b827ea7cfcab5344fd218de494d2` (“Better UD3.1 quants”). This is the original model with runtime projection, **not SC117's permanently abliterated weights**.
- Existing pinned **Strata0.1.46**, native engine and CPU vision as **UID1000**; no runtime/driver/kernel/system CUDA upgrade or GPU/power/unlock change.
- **262144 tokens per chat**, **two serving slots / two batch groups**. No512K/YaRN extension. Both slots were observed occupied concurrently.
- **16384MiB / eight cached histories**, isolated state. All8 synthetic ~4K histories reused their cached prefixes. This is not eight full-context histories or full-context concurrency qualification.
- **Q8_0 mmap engram/PLE**, not Q8 main/expert weights. All four original publisher shards remain the explicit native-dense inputs; the new table is separately pinned. Old/new table payload reuse was proven by independent full-file hashes, never assumed from size.
- **Original-Qwen refusal-direction projection ON**, per-layer, scale1, layers4–44. Actual engine loading and per-request on/off telemetry were verified. Disable for a request with `"experimental_speed_projection": false`. It changes activations, not original GGUF files, can change ordinary predictions, and does not guarantee SC117 equivalence or zero refusals.
- **BF16 CPU vision**, verified by ordered red/blue and blue/red images.
- Matching **stock/non-abliterated Qwen Strata MTP**, Q2_0 expert / Q8 dense drafter, **spec3**; actual short solo fixtures drafted/accepted tokens. No extra Unsloth headline speed multiplier is claimed. Concurrent pipelined requests do not draft.
- `--pipeline-windows 2` remains in the declared config, but the actual .46 engine explicitly disables that particular option with `--batch` slots. This is distinct from the configured **two batch groups**. Do not describe an ignored option as an extra active optimization.
- Automatic layer split, existing GPU order, auto prefill, trimmed stage weights, INT8 KV with32768 resident. Automatic placement is not the old SC117 measured fixed split or a speed qualification. Initial short-load snapshots showed tight first-GPU free VRAM; full-context headroom has **not** been established.
- Temperature1/top-p.95/top-k20, projection default on. **`n_predict=-1`**, no newly fixed output/thinking budget. Input/history/images/reasoning/output still share each native window.

The pinned CPU packer records **460 compatibility conversions,196 nonexact** for small unsupported projections in a separate derived pack. Original downloaded GGUF/expert/native-dense/PLE files were not rewritten. BF16 compatibility rounding and activation projection mean published Unsloth benchmark equivalence is not established.

## Actual tests and scope

UID1000 requests to the real LAN URL, **without Authorization/x-api-key headers**:

- UI/health/models/loaded `.46`, native262144 and2 slots; health `api_key:false`.
- Arithmetic63 with complete SSE; high-thinking2099 with reasoning.
- Projection default-on and explicit per-request off, confirmed in actual request telemetry.
- Swapped/ordered images, concurrent isolated tags with **two simultaneously occupied slots**, complete streams.
- Forced tool-call **format** carrying63; **no tool executed**.
- Model-using POST bad Host403, foreign Origin403, same Origin200.
- **8/8 cached ~4K histories** reused.
- Actual Hive-owned stop/drain/restart, new launch token and new exact process identity, followed by a fresh key-free LAN63/SSE check. **Left running as the miner.**

These are bounded functional/cache checks, **not** a matched speed comparison, broad intelligence/refusal/quant-quality test, long-output test, two-full262K workload, eight-full-context cache proof or sustained stability qualification.

## Lifecycle and safety

The legacy current Swift server was actually **UID0** and could not be proven Hive-owned through its daemonized ancestry. It was **not adopted**. The operator explicitly authorized stopping that exact current session: fresh state/model/UID/birth/argv/GPU checks and naturally finished requests preceded its **identity-bound owner stop with pidfds**, then its known idle Hive supervision drained. New inference is UID1000. Swift/SC117 data/settings remain retained; only active selection changed on request.

New starts require the exact `.46/2/2/native262K` trial environment, source/data/derived-file hashes, dedicated state, full model hash before spawn, serialized lifecycle authorization and a fresh per-run Hive token. Parent/UID/executable/argv/birth/session/token and stable spawned-child capture remain enforced. Wrong profile/PLE/hash/count/runtime/state refuses stop or startup. No numeric-PID fallback or unknown-session adoption.

**Renters always win; OctaSpace admission stayed active throughout.** No renter/render container/service was stopped, cloud rig/wallet/OC bytes were unchanged, and the four original GPUs/current-boot fault scan passed. No lease, switch journal, stop marker or pending controller/test/switch/reset/reboot remains. Preserve Host/Origin and deliberately configured optional authentication. Trusted firewalled LAN only, not public hosting.

Offline `prepared.json` remains the historical data-only receipt with its original `activation_allowed:false`/`gpu_inference_tested:false` fields; it is not the live result or a process-stop authority. Actual successful activation is separately recorded in [sanitized results](results/unsloth-v31-iq4xs-activation-20261107.json). The original [preparation report](UNSLOTH_V31_IQ4XS_PREP.md) is historical, not a request to undo this later activation.

Private completed operational evidence: `/var/lib/localllm-strata/unsloth-v31-iq4xs-activation-20261106/` (namespace follows the earlier preparation date; actual activation logs are November7). Saved controllers/PIDs/tokens are completed history, **never replay or signal authority**.
