# Strata 0.1.39 — guarded upgrade and slot comparison on `.69`, 2026-10-05

## Decision

Use **0.1.39, one slot** for both menu choices [3]/[4] on this prepared node. Preserve 0.1.38 as an explicit rollback.
Swift Q8 remains the main model and Hauhau remains the fallback; their runtimes/defaults were not changed.
Two slots are **opt-in**, not a universal speed upgrade. Larger counts have not been validated and are refused by this controller.

The checked source is `6f32ec070f23ced9f50e704d854d775da52591ab` (release v0.1.39), beside the original
`99f3dbd0b21d1401b3769e0c0d963913607f380b` (0.1.38). Both retain native **262144**, INT8 streaming KV with
**32768 resident**, high reasoning, original Flash-Next MTP, GPU vision, four-GPU automatic contiguous layer splitting,
and temperature 1 / top_p 0.95 / top_k 20. IQ3_S keeps `--prefill auto`; Orca keeps `--prefill 512`.
No RoPE extension, weight/pack/tokenizer conversion, driver/clock/watchdog/flight-sheet change or model download was made.

## Measured single-stream decoding

Actual four-RTX-3090 inference, median of three 384-token, high-reasoning streams per case. Counts include reasoning.
Decode figures exclude prompt processing/queue time; capped stream latency is not completion of a whole agent task.

| Model | Case | 0.1.38, 1 slot | 0.1.39, 1 slot | Change |
| --- | --- | ---: | ---: | ---: |
| IQ3_S | Code, 102-token prompt | 55.6 tok/s | 68.7 tok/s | +23.6% |
| IQ3_S | Ledger, 5294-token prompt | 60.0 | 73.3 | +22.2% |
| Orca IQ3_XXS | Code | 56.7 | 71.1 | +25.4% |
| Orca IQ3_XXS | Ledger | 61.9 | 76.8 | +24.1% |

Ledger prompt processing medians: IQ3_S **1156.4 → 1554.9 input tok/s**; Orca **917.1 → 1765.6**.
All solo measured prefixes had zero cached tokens. Distinct early prefixes were used for each repetition/member,
and warm-up was excluded. These are local bounded results, not the release's other-machine headline claims.

## Two simultaneous clients: queue versus actual batch slots

These are **aggregate output tokens / entire pair wall time**, including prefill, admission and queueing;
do not compare them directly with the preceding engine-only decode rates.

| Model / case | 0.1.39, 1 slot (queued) | 0.1.39, 2 slots | Pair wall time, queued → 2 slots | Last first-token time, queued → 2 slots |
| --- | ---: | ---: | ---: | ---: |
| IQ3_S / code, **2 pipeline groups** | 62.1 tok/s | 78.3 tok/s | 12.37 → 9.81 s | 6.65 → 1.56 s |
| IQ3_S / ledger, **2 groups** | 44.8 | 43.7 | 17.13 → 17.57 s | 12.58 → 9.15 s |
| Orca / code, **1 batch group** | 65.2 | 66.5 | 11.78 → 11.55 s | 6.39 → 1.03 s |
| Orca / ledger, **1 group** | 48.4 | 35.1 | 15.88 → 21.89 s | 11.19 → 13.30 s |

First-token timing counts actual reasoning or answer text, not an empty SSE role header. Native `/v1/status`
confirmed two allocated serving slots and `/metrics` observed two running batch slots in these arms.
Solo decoding with two allocated slots was lower: IQ3_S code/ledger **62.5/67.0**, Orca **65.4/71.4 tok/s**.
Concurrent batch windows do not use MTP drafts; a request alone still uses the original MTP path.

IQ3_S's guarded **one-group** comparison also passed: code pair **60.6 tok/s**, ledger **37.1 tok/s**,
with last first-token times **1.57/9.13 s**. It reduces short-prompt waiting but is slower in total than one queued slot;
two pipeline groups are preferable for its overlapping code requests. See [machine-readable results](results/strata-v0139-20261005.json).

**Interpretation:** IQ3_S's two-group mode helps overlapping short requests. Longer prompt admissions and the
loss of batched MTP can erase that benefit. Orca's one-group mode primarily buys short-prompt waiting time;
its ledger pair lost approximately 27% aggregate throughput. One slot remains the faster general default.

## Discovered failures and guarded policy

1. **Unmodified all-resident batching timed out.** The first IQ3_S 2-slot/1-group concurrent-image arm hit
   `verify batch: timed out at layer 0`; the engine released its GPU waits and exited with code 1. It was not
   promoted or counted as a speed result. Source inspection found the all-resident capture's zero-doorbell
   path incompatible with the batch runner's CPU-doorbell waits at this pin.
   The controller therefore sets upstream's existing **`STRATA_VERIFY_ALL_RESIDENT=0` only for multi-slot launches**.
   It leaves the fast zero-doorbell path enabled for single-slot, keeps MTP/weights unchanged, and does not patch
   the tagged C++ source. Successful guarded two-slot checks include concurrent swapped-image requests.
2. **Orca 2 groups failed concurrent-image JSON validation (HTTP 502).** The native engine remained responsive,
   but one answer was not valid JSON despite explicit json_object formatting. No cause beyond that retained
   evidence is claimed. This mode is refused **before pausing mining/services**; use Orca `--batch-groups 1`.
3. The upstream CPU suite initially failed a Responses JSON-schema test because `jsonschema` was absent.
   `jsonschema==4.25.1` was installed only in the candidate private dependency overlay. Original/system Python
   packages were not changed. The rerun passed: **268 tests, 7 skipped** (not GPU inference tests).

The controller rejects silent upstream allocation fallback: a request for 2 slots must actually serve 2, or the
owned engine is stopped and the normal hosting lease restored. Profile- and explicit runtime-qualified stops
refuse a different live model/version. Rebuilding invalidates default-selection validation until rechecked.

## Installation and operation

On a fresh compatible node, ordinary launches remain 0.1.38 until an explicit preparation and validated selection.
The prepared `.69` node selects 0.1.39 for future launches; no active process is replaced by selection alone.

```bash
sudo -n ./v1strata.sh --prepare-runtime        # CPU/build only; existing CUDA 12.9/sm_86/private tooling
./v1strata.sh --check-ready --profile iq3_s --runtime 0.1.39
sudo -n ./v1strata.sh --select-runtime 0.1.39 # requires recorded real single-slot checks for BOTH models
./HostLLM.sh                                 # [3]/[4], one slot by default

# Explicit optional batching (stop the owned running model with [9] first):
STRATA_PARALLEL=2 STRATA_BATCH_GROUPS=2 ./HostLLM.sh  # IQ3_S [3]; Orca [4] intentionally refused
STRATA_PARALLEL=2 STRATA_BATCH_GROUPS=1 ./HostLLM.sh  # either profile

# Future-launch rollback; active engine untouched:
sudo -n ./v1strata.sh --select-runtime 0.1.38
```

`STRATA_RUNTIME=0.1.38|0.1.39` is a per-invocation override. Direct `--runtime`, `--parallel` and `--batch-groups`
are also supported. Use **HostLLM**, not raw `--quickstart`, for automatic mining/OctaSpace pausing and restoration.
Readiness covers model, runtime, slot/group policy and provenance before any pause. Reopen an old menu after updating.

Runtime: `~/.local/share/localllm-strata/runtimes/0.1.39/source`; private dependency overlay: sibling `python/`.
Existing dependencies are reused read-only; pinned optional schema validation is in the overlay. The original
runtime/configs/assets remain intact. Vision helper reuse required identical source and llama.cpp dependency
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; candidate executable hashes and private-environment contracts are checked.

## Scope and resources

- Maintained Linux CPU/controller suite: **90 passed as user and root**; shell/menu/Swift/GSQ regressions passed.
- Real single-slot model checks: arithmetic, strict JSON, exactly one tool call, simple Python, red/blue image and
  swapped image; both profiles passed on both versions. Passing guarded multi arms also checked concurrent images.
- Model, pack/tokenizer, original MTP and baseline config provenance were preserved. Rentals/unknown workloads
  were never stopped. Tests used identity-bound stops, the existing pause lease, and restored the mining/OctaSpace baseline.
- Observed available RAM minima: ~64 GiB single-slot, ~57 GiB guarded two-slot. Not worst-case/filled-context guarantees.
- IQ3_S 0.1.39 single-slot auto-prefill temporarily left only **839 MiB free on GPU3** in sampled observations,
  despite retaining the configured 2048 MiB reserve. Guarded IQ3_S two-slot minimums were **1564/2850/3042/4137 MiB**.
  Orca two-slot/one-group minimums were **3760/5924/6996/4471 MiB**. Sampling can miss true peaks; no 4-slot promise.
- **No populated-262K generation, sustained multi-user capacity, worst-case VRAM or full Codex CLI validation.**
  New Responses endpoint unit tests are not an E2E Codex-client claim.

Private raw evidence is under `logs/v0139-upgrade-20261004/` (the directory name retains the original preparation date;
GPU runs are dated 2026-10-05). Failed arms remain separate from passing results. Public reports contain no tokens,
miner command lines, wallets or credential-bearing URLs.
