# LocalLLM — Swift, SC117 and Unsloth V3.1

HostLLM and Hive custom-miner profiles for `.69`: four RTX 3090s and approximately 128 GB RAM. **Swift remains the installation default. Unsloth V3.1 IQ4_XS is now ACTIVE as Hive [21]** after explicit operator approval. Swift and SC117 native-262K two-user **[18]** remain retained.

## Latest activation — 2026-11-07, IQ4_XS Hive [21] ACTIVE

[Unsloth V3.1 IQ4_XS active trial](UNSLOTH_V31_IQ4XS.md): real key-free LAN text/thinking/ordered vision, projection on/off telemetry, two occupied chat slots/complete SSE, tool-format-only, Host/Origin guards and **8/8 ~4K cache histories** passed. Actual owned Hive stop/restart/new-token verification passed, and **the new model was left running as the miner**, UID1000, native262144/2slots/2groups/16GiB/eight histories, CPU vision, Q8 PLE and projection on. No full-context/quality/speed/stability qualification. Swift's exact current owner session was explicitly stopped after requests drained, not adopted as Hive-owned. Admission/cloud settings retained; no lease/journal/stop marker or pending controller/reboot. See [sanitized results](results/unsloth-v31-iq4xs-activation-20261107.json).

## Historical preparation — 2026-11-06, before later switch approval

[Unsloth V3.1 **UD-IQ4_XS** offline preparation](UNSLOTH_V31_IQ4XS_PREP.md): exact pinned main shards, new Q8 PLE, original-Qwen refusal projection, CPU BF16 vision, supported stock Strata MTP and an **inactive native262K / 2-slot / 8-history** configuration. **Await separate explicit continue before stopping/switching the current LLM, loading/testing the candidate or publishing a serving profile.** Offline checks are not GPU fit/quality/speed qualification. Q4's engram-packaging update was corrected before any Q4 download; IQ4_XS has an explicit “Better UD 3.1 quants” weight-file commit.

Requested TensorSharp **[8]/[19]** menu entries and proven-unused private source/.NET/CUDA13/NuGet/build software have now been removed. Original Swift/SC117 assets, active installed bundle/selection, services and cloud settings are preserved. Only history remains; no TensorSharp/512K retry.

## Historical recovery checkpoint — 2026-11-06, before the later Swift selection

**Cleanup completed; native262K Hive [18] recovered and ACTIVE; TensorSharp GPU deployment rejected.** Hauhau and the single-slot Heretic/SC117 routes were removed. Only unused Hauhau/Heretic model trees were deleted, freeing approximately **137 GiB**; shared SC117 weights, original dense shards, Q8 PLE, CPU vision, runtime and preset metadata remain intact.

During TensorSharp TP4’s **2048-token startup warm-up**, GPU3 fell off the bus (**Xid79**, followed by **Xid154: Node Reboot Required**). No TensorSharp HTTP inference/speed result was obtained, and it is **not deployed**. This was not a 512K test. Root cause remains unknown.

After the operator’s manual restart and “continue,” **all four original GPUs recovered, the current boot’s fault scan was clear, and native262K [18] is ACTIVE**. Fresh rental/ownership checks preceded release of only this task’s exact fault lease; an existing owned startup was observed without duplication. UID1000 key-free LAN text/thinking/vision/concurrent-tags/complete-SSE/tools-format and Host/Origin checks passed. OctaSpace admission stayed active; no renter/container/service was stopped and cloud settings were preserved. No lease/journal, test/controller, reset, reboot or retry is pending.

See **[the full checkpoint](TENSORSHARP_SWIFT_20261106.md)** and [sanitized results](results/tensorsharp-swift-cleanup-20261106.json). Earlier pause instructions and controllers are historical; do not replay them or retry TensorSharp/512K.

## Current menu

| Direct | Hive | Model/runtime |
|---:|---:|---|
| **1** | **12** | Swift 1.5 Uncensored **Q8_K_XL**, existing pinned llama.cpp; BF16 vision, native MTP n3, xhigh, two native-262K slots |
| **7** | **18** | SC117 abliterated IQ3_S with **Q8_0 PLE**, Strata **.46**; two pipelined slots/groups, 16 GiB/eight histories, CPU vision; unqualified trial |
| — | **21** | **ACTIVE** Unsloth V3.1 **IQ4_XS**, Q8 PLE / original-Qwen projection ON, .46 / CPU vision / stock MTP; two native262K chats / 16 GiB / eight histories; bounded functional trial |

**[9]** owned stop, **[10]** update, **[11]** exit. Former **[2–6], [8], [13–17] and [19]** are gone; their numbers were not reused. The low-level Hive selector has **[1] Swift / [8] SC117 two-user / [9] Unsloth V3.1 IQ4_XS**. High-level [9] uses the verified Hive supervisor to stop the active V3.1 profile.

Hauhau, Heretic and the original one-slot SC117 IQ4_NL/Q8 routes are retired. Orca/DeepSeek retirement also remains in force. Historical source, wrappers and reports remain audit evidence where useful; they are not current serving instructions or permission to redownload retired models.

## Retained settings

### Swift

The retained file is the **27B dense Qwen model**, not Flash-Next:
`Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q8_K_XL.gguf`.

- Original pinned model/projector retained unchanged; unused TensorSharp software is retired.
- Existing llama.cpp `4cbe8b07`, four-GPU layer split, Q8 K/V, BF16 vision, embedded target-verified MTP depth 3.
- Two **262144-token** slots. Its `--ctx-size 524288` is the aggregate allocation for two native windows, **not** a 512K-per-slot extension.
- Normal xhigh thinking; temperature 1, top-p .95, top-k 20, min-p/presence 0, repetition 1.
- `v1qwen38.sh --print-command --profile swift15u-q8` prints JSON argv without downloading, building or starting inference.

### SC117 two-user

[SC117_Q8_2SLOT.md](SC117_Q8_2SLOT.md) documents the separately bounded `sc117-q8-2slot` / `iq3_s-sc117-abliterated-ple-q8-2slot-v0146` preset:

- Strata .46; **native262144 per slot**, two slots/two pipelined groups.
- INT8 KV/32768 resident, BF16 CPU vision, split `14,24,36`, spec 3/windows 2.
- **16384 MiB/eight cached histories**, isolated state/adaptive profile; not proof of eight full-context histories.
- Solo MTP remains; concurrent windows have no drafts. Long prefill can pause another stream.
- Q8_0 applies to the **PLE table only**, not main weights or KV. Original SC117 shards remain required native-dense/provenance dependencies even though the serving PLE is Q8.
- Unqualified trial. **No 512K promotion/retry** or output/thinking-budget change.

## Installation and lifecycle

```bash
cd /home/user/LocalLLM
./HostLLM.sh
```

For an existing live package, after operator-owned preparation and with **no lease/journal/rental**, the narrow publication mode preserves current selection, active files, stable links, miner markers, services and cloud configuration:

```bash
sudo ./install-hive-llm-miner.sh --publish-retained-profiles
```

`--add-ple-q8-profiles` remains a compatibility spelling for this **retained-only** publisher; it cannot reintroduce the retired profiles. The ordinary full installer remains **idle-only** and never stops a workload to make itself run. Never use publication/installation to bypass any future fault hold.

High-level Hive selection verifies exact ownership/ancestry and readiness before changing an owned workload. **Renters always have priority. OctaSpace admission stays running.** New rental/unknown work blocks or yields only our LLM. No `pkill`, numeric-PID signal fallback, GPU reset, driver/clock/power change or cloud-agent suppression.

Use only on a trusted, firewalled LAN: **http://192.168.1.69:8080/**, **no mandatory key, dummy token or Authorization header**. Preserve Host/Origin checks and deliberately configured optional auth. Not public hosting.

## TensorSharp retirement

The failed **2.8.6** trial is historical, not a serving route. Its private source/.NET10/CUDA13/NuGet/build archives and repository preparation/build helpers were explicitly retired after proving them unused. Small calibration/results and private metadata remain audit evidence. System CUDA12.9, driver and actual Swift model/projector were not changed. External/manual TensorSharp detection remains a safety blocker, not stop authority.

The earlier CPU-only LAN tests did not qualify GPU serving: TP4 faulted before HTTP readiness. **No winning configuration or speed improvement exists. No TensorSharp/512K retry.** See the historical [trial/recovery report](TENSORSHARP_SWIFT_20261106.md).

Historical evidence: [concurrency](SC117_Q8_CONCURRENCY_20261101.md), [two-user activation](SC117_Q8_2SLOT.md), [rejected 512K test](SC117_512K_TEST_20261101.md), [PLE pins](PLE_Q8.md), [retired Heretic](HERETIC_RE.md). Their old menu/install instructions are superseded above.
