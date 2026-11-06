# LocalLLM — Swift and SC117

HostLLM and Hive custom-miner profiles for `.69`: four RTX 3090s and approximately 128 GB RAM. **Swift remains the installation default.** SC117 native-262K two-user **[18]** is the operator’s selected miner.

## Latest checkpoint — 2026-11-06

**Cleanup completed; native262K Hive [18] recovered and ACTIVE; TensorSharp GPU deployment rejected.** Hauhau and the single-slot Heretic/SC117 routes were removed. Only unused Hauhau/Heretic model trees were deleted, freeing approximately **137 GiB**; shared SC117 weights, original dense shards, Q8 PLE, CPU vision, runtime and preset metadata remain intact.

During TensorSharp TP4’s **2048-token startup warm-up**, GPU3 fell off the bus (**Xid79**, followed by **Xid154: Node Reboot Required**). No TensorSharp HTTP inference/speed result was obtained, and it is **not deployed**. This was not a 512K test. Root cause remains unknown.

After the operator’s manual restart and “continue,” **all four original GPUs recovered, the current boot’s fault scan was clear, and native262K [18] is ACTIVE**. Fresh rental/ownership checks preceded release of only this task’s exact fault lease; an existing owned startup was observed without duplication. UID1000 key-free LAN text/thinking/vision/concurrent-tags/complete-SSE/tools-format and Host/Origin checks passed. OctaSpace admission stayed active; no renter/container/service was stopped and cloud settings were preserved. No lease/journal, test/controller, reset, reboot or retry is pending.

See **[the full checkpoint](TENSORSHARP_SWIFT_20261106.md)** and [sanitized results](results/tensorsharp-swift-cleanup-20261106.json). Earlier pause instructions and controllers are historical; do not replay them or retry TensorSharp/512K.

## Current menu

| Direct | Hive | Model/runtime |
|---:|---:|---|
| **1** | **12** | Swift 1.5 Uncensored **Q8_K_XL**, existing pinned llama.cpp; BF16 vision, native MTP n3, xhigh, two native-262K slots |
| **7** | **18** | SC117 abliterated IQ3_S with **Q8_0 PLE**, Strata **.46**; two pipelined slots/groups, 16 GiB/eight histories, CPU vision; unqualified trial |
| **8** | **19** | TensorSharp / same Swift file — **DISABLED information entries**, not serving or installable profiles |

**[9]** owned stop, **[10]** update, **[11]** exit. Requested former **[2–6] and [13–17]** are gone; their numbers were not reused. The low-level Hive selector retains **[1] Swift / [8] SC117 two-user** only.

Hauhau, Heretic and the original one-slot SC117 IQ4_NL/Q8 routes are retired. Orca/DeepSeek retirement also remains in force. Historical source, wrappers and reports remain audit evidence where useful; they are not current serving instructions or permission to redownload retired models.

## Retained settings

### Swift

The retained file is the **27B dense Qwen model**, not Flash-Next:
`Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q8_K_XL.gguf`.

- Original pinned model/projector unchanged; shared by any explicitly authorized future TensorSharp investigation.
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

## TensorSharp status

Pinned **2.8.6 / `9d18a64dbf07efa7b0fb5b7d34dda97f05e2c368`**, private .NET10.0.403/CUDA13.0.2, GGML CUDA SM86. System CUDA12.9/driver unchanged. Exact Host/Origin/fixed-model patch; no numerical source patch.

Software build and CPU-only unauthenticated LAN UI/Host/Origin/pidfd-shutdown tests passed. GPU layer split was refused by `qwen35`; TP4 disabled MTP, then faulted during warm-up before serving readiness. **No winning configuration or speed improvement exists.** `prepare_tensorsharp.py --prepare` can verify software only; `--check-ready` refuses GPU deployment. Do not retry automatically after reboot.

Historical evidence: [concurrency](SC117_Q8_CONCURRENCY_20261101.md), [two-user activation](SC117_Q8_2SLOT.md), [rejected 512K test](SC117_512K_TEST_20261101.md), [PLE pins](PLE_Q8.md), [retired Heretic](HERETIC_RE.md). Their old menu/install instructions are superseded above.
