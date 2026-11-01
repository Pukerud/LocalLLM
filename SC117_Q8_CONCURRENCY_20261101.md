# SC117 Q8_0 PLE — .46 concurrency and conversation-cache measurements

2026-11-01, `.69`, four RTX 3090s. **Strata 0.1.46 only.** Seven completed fresh-process runs cover six configurations, including a before/after baseline. The operator explicitly authorized stopping the current server and testing settings. An unnecessary extra .45 baseline was begun for direct option [6], then cancelled when the operator clarified .46; none of its partial measurements are included. Direct [6]'s .45 defaults are not changed or qualified by these results.

## Speed summary

Median speeds, tokens/s. **Solo decode** is the native decode rate. **Two-client total** is actual generated tokens divided by the synchronized pair's complete wall time, including queue, prefill and transport. They measure different things: the one-slot baseline runs the pair **one after another**, whereas the two-slot cases really decoded two streams together.

| .46 configuration | Cache GiB / parked limit | Solo decode | Two-client total, short input | Two-client total, 11.8K input | Last client's first-token wait, short input |
|---|---:|---:|---:|---:|---:|
| Current: one serving slot, normal MTP | 8 / 4 | **134.3** | **121.2** | **53.9** | **3.47 s** |
| One slot, eight cached histories | 8 / 8 | 133.3 | 121.2 | 53.8 | 3.45 s |
| One slot, larger cache budget | 16 / 8 | 133.9 | 121.5 | 53.8 | 3.47 s |
| **Two slots, two pipelined groups** | **16 / 8** | **138.4** | **153.0** | **57.9** | **0.76 s** |
| Two slots, one group, no batch-MTP | 16 / 8 | 133.1 | 114.4 | 44.8 | 0.73 s |
| Two slots, one group, batch-MTP | 16 / 8 | 134.5 | 140.6 | 47.4 | 0.75 s |

The baseline is pooled from its before/after runs. Individually, short-input solo decode was **134.2 / 134.4**, combined pair throughput **121.1 / 121.3**: drift about **0.15% / 0.18%**. Candidate deltas remain bounded workload measurements, not universal speed guarantees.

### What wins here

- **Two pipelined slots** lead: approximately **+26.3% combined throughput** on short prompts and **+7.3%** on the 11.8K prompts versus the pooled one-slot queue baseline.
- That is not twice the speed for each user. Concurrent short-prompt streams had a median of about **86 tokens/s per active user**, versus approximately **133** while each baseline user had its sequential turn. Their waiting time is much lower.
- Batch-MTP genuinely ran (two slot drafters, about **49 MiB private state each**) and improved the one-group two-slot arm, but its **140.6** total tokens/s still lost to the pipelined arm's **153.0**.
- **MTP was not disabled globally.** All solo speed fixtures reported draft tokens. Two pipelined groups use no drafts while concurrently batching; the solo path retains normal MTP. Batch-MTP uses one proposal per slot, not the normal solo `--spec 3` path.

## Larger cache

All eight approximately 4K-token histories returned their correct private test marker. With the current four-parked-history limit, **5/8** revisits reused state (four parked entries plus the active history). With an eight-entry limit, **8/8** reused state. The 8-GiB/eight-history case actually reported eight parked entries using about **2.72 GiB**.

Raising the history count caused no meaningful decode-rate change. Doubling the byte budget from 8 to 16 GiB did not improve this workload further; it only gives larger histories more headroom. **Neither budget guarantees eight fully populated 262K conversations fit.** The byte budget is a shared ceiling, not RAM reserved per history.

## Long-prompt trade-off

A cold approximately 29K-token prompt joined a continuing 1536-token decode, two repetitions:

| Mode | Continuing stream's largest token gap | Joining request's first-token wait |
|---|---:|---:|
| One-slot queue | 0.035 s | 16.7 s |
| Two pipelined groups | **6.93 s** | **6.89 s** |
| Two slots, one group, no batch-MTP | 3.71 s | 19.09 s |
| Two slots, one group, batch-MTP | 3.72 s | 19.10 s |

Pipelining starts the joining request sooner and has the best pair throughput, but **a long prefill temporarily pauses the other stream**. Experimental concurrent-prefill/environment switches were not enabled. Do not promise uninterrupted streaming from a slot-count change alone.

## Method and checks

- Identical pinned SC117 IQ3_S main weights and original-Qwen Q8_0 PLE; same .46 binary/source, CPU BF16 vision, native **262144**, INT8/32768 resident KV, split `14,24,36`, `--spec 3 --pipeline-windows 2`, sampling defaults and stock target-verified MTP support.
- Fresh owned process per case, same frozen adaptive-profile seed, no cross-case KV. Normal OS caches were not flushed. Q8 full checksum rechecked before every start; canonical model/runtime provenance verified before the window.
- Three timed repetitions each of solo/pair short **93-token** and approximately **11782-token** inputs, output cap **384 tokens**. All timed prompt-cache counts were zero. Solo and synchronized-pair warmups were excluded.
- Speed fixtures used greedy sampling and **thinking disabled per request**; a separate real high-thinking arithmetic check passed. These are not high-thinking task-completion or quality benchmarks.
- Every completed case passed known answer `63`, swapped real images, tool-call formatting without tool execution, simultaneous distinct-tag correctness, complete SSE, key-free LAN UI/health and Host/Origin **403**.
- Effective two-slot counts and actual concurrent decoding were observed, not inferred from configuration. Requested counts were not silently accepted after fallback.
- GPU power limits stayed **300/300/300/280 W**; maximum sampled temperature across the seven .46 runs was **75 C**. No driver/kernel/toolkit, clocks/power policy, weights, normal presets or installed Hive profile files were changed.
- Models ran as UID **1000** on temporary trusted-LAN benchmark port **18080**, isolated from normal clients on 8080. Lifecycle reservations, exact pause lease, rental checks and exact executable/UID/birth/session/parent identities were retained. Stops used pidfds and only the experiment's own family.
- The initial setup paused OSN before its descendant model was explicitly stopped; that also drained the model. The original server was restored, then the order was corrected to owned Hive teardown before pausing admission. No performance result came from that setup attempt.

**Final state verified:** the original Q8 SC117 **.46 Hive [17]** server is restored with its exact original semantic configuration (one slot, 8-GiB/four-history cache); **OctaSpace active**, no pause lease or switch journal, known LAN answer `63` passed. Rig/wallet/OC hashes, GPU limits and selected profile matched the baseline. Regression: **441 source Python tests OK/one skipped, seven shell suites passed, two private experiment-contract tests passed.** Normal menu/profile defaults are unchanged; no winner is permanently applied. See [sanitized measurements and restoration receipt](results/sc117-q8-concurrency-20261101.json). Raw fixtures, answers, process identities and configuration snapshots remain private.

## Limits

No .45 speed claim; no IQ4_NL/Q8 quality comparison, broad answer-quality equivalence, long-term stability qualification, or two simultaneously populated near-262K histories/eight full-context parked histories. SC117 remains an **unqualified trusted-LAN trial**. The recommendation is workload-specific: pipelined two slots for combined throughput and reduced queueing; one slot if uninterrupted single-user streaming is the overriding priority.
