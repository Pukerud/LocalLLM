# Strata 0.1.41 qualification — 2026-10-08, four RTX 3090s

## Decision: prepared and tested, **not promoted**

The node keeps **0.1.39, one slot** selected for future IQ3_S and Orca launches. 0.1.41 is installed separately and can be requested explicitly; 0.1.38 and 0.1.39 executables, dependencies, model/config assets and the selection file were preserved.

Both models passed bounded 0.1.41 GPU correctness checks. Nevertheless, the new four-way automatic placement regressed uncached prompt processing. Preserving the fast existing setup is more important than adopting a newer version number. No manual split, experimental placement tuning or upstream C++ patch was substituted to conceal the regression.

| Runtime | Immutable source commit |
|---|---|
| Original 0.1.38 | `99f3dbd0b21d1401b3769e0c0d963913607f380b` |
| Selected 0.1.39 | `6f32ec070f23ced9f50e704d854d775da52591ab` |
| Optional 0.1.41 | `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` |

The llama.cpp vision dependency remains `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Actual installed commits, not mutable release tags, are authoritative.

## Controlled preparation and policy

- CUDA 12.9 / `sm_86` Release build, three low-priority CPU compilation jobs; no driver/system-tool change, reboot, clock/watchdog/flight-sheet/miner-selection edit, model download, model repack or repeated cleanup.
- Candidate-only Python overlay: `jsonschema==4.25.1`, `psutil==7.2.2`, with existing private dependencies shared read-only. The 0.1.41 vision-source contract changed, so its GPU helper was rebuilt, not silently reused.
- Native context **262144**, INT8 streaming KV / **32768 resident**, original Flash-Next MTP, high reasoning, temperature **1**, top-p **0.95**, top-k **20** retained. IQ3_S uses BF16 GPU vision and prefill auto; Orca retains its independent pack/tokenizer/F16 projector and prefill 512.
- GPU order explicitly `as_given`; contiguous split remains **auto**. Two-slot launches explicitly pass `--batch-groups 1` or `2`. New upstream AUTO grouping must not silently change the intended preset.
- 0.1.39 multi-slot still uses `STRATA_VERIFY_ALL_RESIDENT=0`; solo remains unchanged. Tested 0.1.41 uses `STRATA_VERIFY_ALL_RESIDENT=1`, exercising the upstream per-window fix. Stage pinning, residency-biased routing, embedding-account reuse, CPU-share prefill and batch-MTP opt-ins remain off. Concurrent batch-MTP is not supported on this layer-split topology; solo requests, including one active user in a two-slot allocation, still recorded original MTP drafts.
- Only **1/2 slots** and groups 1/2 are offered. Orca 0.1.39 groups 2 remains refused. Orca 0.1.41 groups 2 requires the exact-pin, real concurrent-pixel qualification stamp; it passed this run, but is not the default.
- Every GPU arm went through maintained HostLLM pause/restoration, with fresh Docker/workload checks and identity-bound teardown. Unknown renters/jobs, external sessions and busy requests are not adopted or killed.

## Real GPU measurements

Three repetitions per case, distinct early request prefixes and **zero solo cached prompt tokens**. Warm-up excluded. High reasoning, 384-output-token cap including reasoning. Code prompt: 102 tokens; ledger prompt: 5294 tokens. These are capped stream measurements, **not complete coding/agent-task latency**.

Solo rates below are **native engine decode**, excluding prefill/queue. Pair rates are **total outputs / entire two-request wall time**, including prefill/admission/queue. One-slot pairs queue; two-slot pairs share one real engine. TTFT uses the first nonempty reasoning/answer delta, not an SSE role header.

| Profile / runtime / slots / groups | Solo code decode | Solo ledger decode | Code pair aggregate | Ledger pair aggregate |
|---|---:|---:|---:|---:|
| IQ3_S / .39 / 1 / 1 | 105.3 | 110.5 | 94.75 | 60.64 |
| IQ3_S / .41 / 1 / 1 | 105.3 | 108.7 | 92.14 | 53.83 |
| IQ3_S / .39 / 2 / 2, guarded | 94.8 | 98.8 | 120.25 | 56.73 |
| IQ3_S / .41 / 2 / 2 | 102.7 | 124.4 | 128.48 | 54.19 |
| IQ3_S / .41 / 2 / 1 | 109.1 | 109.8 | 95.78 | 36.52 |
| Orca / .39 / 1 / 1 | 108.5 | 115.4 | 97.94 | 64.51 |
| Orca / .41 / 1 / 1 | 110.0 | 116.2 | 99.95 | 51.53 |
| Orca / .39 / 2 / 1, guarded | 96.3 | 104.2 | 96.92 | 46.73 |
| Orca / .41 / 2 / 1 | 114.0 | 119.0 | 104.20 | 35.33 |
| Orca / .41 / 2 / 2, qualified | 110.9 | 118.0 | 142.63 | 50.92 |

All rates are tokens/second. These are fresh same-day baselines; do not interpret differences from the separate October 5 report as an upgrade gain.

### Why the newer runtime was not selected

| Single-slot ledger observation | 0.1.39 | 0.1.41 |
|---|---:|---:|
| IQ3_S prefill, input tok/s | 1851.6 | 1410.2 (**−23.8%**) |
| Orca prefill, input tok/s | 1987.8 | 1252.7 (**−37.0%**) |
| IQ3_S solo TTFT, s | 2.886 | 3.773 |
| Orca solo TTFT, s | 2.689 | 4.244 |
| IQ3_S pair last TTFT, s | 9.554 | 10.721 |
| Orca pair last TTFT, s | 8.756 | 11.675 |

Old auto boundaries were `13,25,36`. New auto chose `18,19,33` for IQ3_S and `17,18,33` for Orca: **CUDA1 got just one layer**. About 17–18 GiB remained unused there. IQ3_S logged the narrow-stage warning, reducing auto prefill from **8192 to 768 tokens**, with cache borrowing/streaming. Orca retained requested prefill 512 but also regressed. Decode was broadly similar; this is a prompt-processing/layout regression, not evidence of a universal decode collapse.

Two-group .41 improved short code pair throughput over .41's single-slot queue, but its ledger pair still lagged the selected .39 single-slot queue. Batching is workload-dependent, not universally faster.

## Correctness, failures and bounded coverage

- **10 passing GPU arms**, plus **one separately retained failed baseline fixture**. Passing arms checked math, JSON, exactly one correctly named tool call, a structurally validated Python `add`, and actual red/blue swapped pixels. Multi-slot arms sent both images concurrently; `/v1/status` proved two allocated slots and sampled `/metrics` observed two running requests.
- All four .41 two-slot modes also completed two **10625-token** prompts plus a **137-token** prompt, with distinct exact JSON labels and the same native process identity afterward. This bounded long/long/short admission check is **not** a filled-262K or sustained-capacity test; short TTFT ranged roughly 14–29 seconds in this fixture.
- Orca's initially unseeded .39 solo tool check returned plain `get_weather city=Oslo`, not an API tool call. That failed arm, native/launcher logs and restored baseline remain retained. Functional fixtures then used request-scoped seed **1234**, with unchanged temperature/top-p/top-k; seeded old/new Orca checks passed. No failure was relabeled or included in passing medians. Earlier IQ3_S single-slot checks passed unseeded; performance requests already used fixed seeds.
- A driver resume initially hit the restored managed Swift server while it still returned HTTP 503 during loading. It aborted before the next hosting mutation, retained the failure, and was corrected to wait for readable idle metadata rather than stop anything to force readiness.
- Maintained Linux suite: **100 tests passed as ordinary user and root**. Python/shell syntax plus retained-menu/Swift/Swift-Uncensored/GSQ shell regressions passed. Printed ERROR lines in rejection fixtures are expected.
- Upstream frontend CPU suite: raw **593 tests**, **one failure**, **8 skipped**. `GpuChoice.test_vision_device` assumed equal GPU speed scores, but real 3090 rankings reordered 0,1. A scoped equal-score mock **only in that fixture** yielded **593 run / 8 skipped / no failures**; ordering tests and pinned production source were unchanged. CPU mocks are not GPU performance evidence.

Sampled performance-run minimum free VRAM ranged from **839 MiB** in the retained .39 IQ3_S solo arm to **1073 MiB** on a .41 Orca two-slot arm. Available RAM minima were approximately **55.9–66.0 GiB**. Sampling was every 1.5 seconds and excluded image/warm-up/overlap phases; peaks can be missed. Configured 2048 MiB reserve is not a guaranteed free-memory floor. No four-slot, worst-case/full-context memory, long-context quality, video, full Codex CLI or sustained multi-user claim is made.

## Operation

Reopen old menus after pulling the updated controller. Existing selection stays .39:

```bash
./HostLLM.sh                                      # [3] IQ3_S or [4] Orca: selected .39, one slot
STRATA_RUNTIME=0.1.41 ./HostLLM.sh                 # explicit optional .41, one slot
STRATA_RUNTIME=0.1.41 STRATA_PARALLEL=2 STRATA_BATCH_GROUPS=2 ./HostLLM.sh
sudo -n ./v1strata.sh --select-runtime 0.1.39       # future launches only, does not replace an active engine
sudo -n ./v1strata.sh --select-runtime 0.1.38       # original rollback, one slot only
```

Use owned stop/recovery **[9]** before switching. Preparation is separate and explicit:

```bash
sudo -n ./v1strata.sh --prepare-runtime --runtime 0.1.41
```

It refuses to rebuild the active candidate and invalidates its old validation after rebuilding. Ordinary menu launches never download/repack/update a runtime. Default binding remains **trusted LAN** `0.0.0.0:8080`; do not expose an unauthenticated endpoint publicly.

Machine-readable sanitized results: [results/strata-v0141-20261008.json](results/strata-v0141-20261008.json). Postdeployment normal-menu API/vision/Pi checks are recorded separately after the maintained fast-forward deployment; the optional runtime is not automatically promoted.
