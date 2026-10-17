# HOWTOTUNE — repeatable Strata model tuning

A practical protocol for measuring Strata inference settings without confusing a promising result with a qualification or production-ready profile. Reuse the methodology across models, but **never copy settings from one model, runtime, GPU layout, or host to another without retesting**.

## Safety first

On shared Hive/OctaSpace machines, model tuning is a maintenance operation:

- Check that Docker/rental state is inspectable and no rental, manual HostLLM session, unrelated GPU process, or active API client can be interrupted. If identity or ownership is uncertain, **stop before touching anything**.
- Use the existing serialized HostLLM/Hive lifecycle and ownership checks. Pause OctaSpace only after the guard verifies its current state and the workload owner; record the prior Hive profile and service state. Never use broad `pkill`, kill a rental, or stop a miner/service by hand to make a benchmark fit.
- Run inference as the unprivileged model owner. Keep test state, logs, and caches separate from the normal profile. Do not edit model assets, packs, normal launch flags, or a production/default profile while screening.
- Keep unauthenticated APIs on a trusted, firewalled network. This SC117 trial binds `0.0.0.0:8080` without authentication; never expose it publicly.
- If a test fails to stop cleanly, an unknown process appears, or restoration cannot be verified, leave workloads paused and follow the recovery journal. Do not restart Hive/OctaSpace blindly.

For this repository, begin with `README.md`, `STRATA_SC117_IQ3S.md`, `hosting_lifecycle.py`, `engine_safety.py`, and the selected profile's launcher/wrapper. The existing menu/profile safeguards deliberately refuse trial overrides. **Do not bypass them with a manual shell launch.** A tuning controller must separately perform the serialized pause, verify Hive-wrapper ownership and GPU idleness, pass only explicitly allowlisted trial variables to an unprivileged Strata process, record its identity, stop only that identity, wait for GPU/API teardown, and restore the exact prior state. The controller used for the 2026-10-17 SC117 measurements was a temporary, root-orchestrated test helper; it was not installed as a normal start path.

## Measurement protocol

1. **Pin the experiment.** Record host and GPU identities, OS/driver, model/revision/quantization, Strata version/build, profile/config hash, context/KV/vision/MTP/slot settings, effective CLI arguments, and every override. Do not compare across runtime or model changes as though they were the same experiment.
2. **Capture a baseline.** Start from the current validated configuration. Use a cold server start for every arm and a known-working warm-up excluded from timing. Run a baseline before and after a candidate series to reveal drift (thermal, clock, background-load, or order effects).
3. **Change one variable at a time.** Keep all untested settings at baseline. Once individual effects are understood, screen combinations as separate arms; interactions are possible. A setting that is faster in isolation need not help in a combination.
4. **Use fixed, reproducible prompts.** Keep prompt text, seeds, sampling, output limits, reasoning mode, endpoint, and model ID identical between arms. For uncached prefill, restart the model for each arm, add a distinct fixture prefix, and verify the engine reports zero cached prompt tokens. Never reuse private user prompts as benchmark data.
5. **Separate the metrics.** Measure prompt throughput/TTFT separately from output decode throughput. Use the server's engine timing fields (`prompt_n`, `prompt_per_second`, `prompt_ms`, `predicted_n`, `predicted_per_second`, `predicted_ms`, `cache_n`) rather than deriving decode rate from end-to-end wall time. Fix the output cap and confirm every sample reaches the same useful minimum; record output length and finish reason.
6. **Repeat and summarize robustly.** Run at least three independent requests per arm and report each sample plus the median. Repeat the leading candidate in a fresh start; bracket it with baseline repeats where possible. Small differences within run-to-run spread are inconclusive. Retain failures, errors, and incomplete arms rather than silently dropping them.
7. **Record resource and behavior evidence.** Sample per-GPU utilization, clocks, temperature, power and VRAM, plus RAM; capture engine configuration before/after. Record MTP drafted/accepted counts, startup time, TTFT, failures, and whether the model loaded. A throughput run does not by itself assess answer quality, long-context behavior, vision, concurrent load, sustained stability, or safety.
8. **Restore and verify.** Stop only the benchmark's identity-verified process. Confirm no listener, LLM process, or GPU job from the test remains; restore the exact selected Hive profile and prior OctaSpace state; verify the model is loaded and a known-working request succeeds. Preserve the result JSON and recovery evidence.

A useful report is machine-readable JSON: host/model/runtime pins; baseline/effective arguments; per-arm overrides; exact prompt fixture IDs/hashes; sampling settings; per-request token/timing data; medians; MTP counters; telemetry; startup/shutdown/restore status; and scope/limitations. Keep benchmark evidence distinct from any model qualification record.

## SC117 `.44.1` trial: current evidence

The live screen was run on **OctaSpace-WorkerFifteen**, four RTX 3090s (96 GiB aggregate VRAM), Strata `0.1.44.1`, model `qwen3.8-flash-next-sc117-abliterated-iq3_s` (publisher revision `eec4e10a2c29b440a2ae85591fa11569ce752963`), one serving slot/group, context 262,144, INT8 KV with 32,768 resident, CPU vision, and the existing 8,192 MiB/four-slot conversation cache. Each arm was cold-started, warmed separately, then tested with three fresh ~5.3K-token prefill prompts (16-token output cap) and three fresh 384-token decode probes (temperature 0.7, top-p 0.95, top-k 20, high reasoning). Prompts/seeds were held constant, `cache_n=0` was required, per-GPU telemetry was collected, and medians are reported.

The prefill fixtures were generated as follows; `i` is 0, 1, and 2, with seed `8100 + i`:

```python
ledger = "\n".join(
    f"entry {n:03}: batch {n % 13} stores {n * 7 + 19} units with a checksum of {n * 11 + 31}."
    for n in range(200)
)
prefill = (
    f"fixture_sc117_tune_prefill_{i:02d}\n" + ledger +
    "\nWrite a detailed 500-word technical explanation of how to process this ledger safely. "
    "Cover validation, reproducibility, indexing, concurrency and error handling; do not merely repeat the entries."
)
```

The decode fixture for each `i` used prefix `decode_probe_sc117_{i:02d}\n`, followed by: `Explain how binary search locates a value in a sorted list. Give a worked, numbered example using the values 3, 7, 11, 18, 24, 31, 42, 56, 73, and 91, searching for 42. Continue with useful detail.` It used seed `9100 + i` and a 384-token cap. The separate warm-up was `What is 2 + 2? Reply with only the number.` with temperature 0, reasoning effort `none`, and a 16-token cap; it had to return `4` and was excluded from the medians.

| Arm | Trial difference | Prefill median (tok/s) | Decode median (tok/s) | MTP accepted/drafted |
|---|---|---:|---:|---:|
| Baseline A | Existing defaults | 1,826.4 | 149.8 | 70.03% |
| Baseline B | Existing defaults, repeat | 1,822.7 | 148.6 | 70.03% |
| Pipeline 2 | `--pipeline-windows 2` | 1,823.7 | 167.1 | 61.62% |
| Pipeline 2 repeat | Same | 1,814.9 | 168.9 | 61.89% |
| Spec 3 only | `--spec 3` | 1,792.9 | 144.9 | 75.83% |
| **Pipeline 2 + Spec 3** | `--pipeline-windows 2 --spec 3` | 1,810.5 / 1,799.4 | **178.9 / 177.9** | 71.4% / 71.4% |
| Pipeline 2 + split 12,24,36 | `--pipeline-windows 2`, split override | 1,818.1 | 165.6 | 61.62% |
| Full exploratory combination | Pipeline 2, Spec 3, PCIe fraction 0.20, 3 pool workers, split 12,24,36 | 1,806.7; repeats 1,810.8 / 1,823.2 | 178.6; repeats 175.1 / 178.3 | 71.5% / 71.4% / 71.5% |

The exact **Pipeline 2 + Spec 3** candidate repeated at **178.4 tok/s** median-of-run-medians, about **19.5% above** the two-baseline median (149.2 tok/s). Its prefill median across the two runs was about 1,805 tok/s (roughly 1% below baseline), so this is a **decode-throughput** result, not a universal speedup. Pipeline 2 alone repeated around 168 tok/s (+12.6%), while Spec 3 alone was slower than baseline. The combined result demonstrates an interaction; the split override did not help Pipeline 2 in its isolated screen. Changing PCIe fraction or pool workers alone did not produce a convincing improvement. Trimming stage weights off reduced free VRAM and did not materially improve throughput; GPU reordering and the alternate split were also unhelpful in the single-variable screens.

**Interpretation:** on this host/runtime/model and these short, single-request probes, `--pipeline-windows 2 --spec 3` is the best simple *trial candidate measured so far*. Keep the existing split, PCIe fraction, pool-worker count, trimming, context, cache, KV, vision, and serving-slot settings unchanged. Do not persist these overrides in HostLLM/Hive: normal menu startup continues to use baseline defaults and intentionally rejects experimental variables. These measurements are not SC117 qualification, do not change its unqualified status, and do not establish quality, full-context throughput, multi-user capacity, or a result transferable to another host/model/runtime.

After all trial arms, the restored default Hive profile was loaded and passed a separate `2 + 2 → 4` request; the response, timing, and final loaded status are recorded in the main report. This smoke check verifies recovery/availability, not general answer quality.

Raw machine-readable evidence:

- [Main SC117 tuning sweep and baseline A/B](results/sc117-strata-tuning-20261017.json)
- [Pipeline/spec interaction and combined repeats](results/sc117-strata-confirmation-20261017.json)
- [Exact Pipeline 2 + Spec 3 repeat](results/sc117-strata-exact-confirmation-20261017.json)

## Reusing the SC117 trial knobs safely

The variables below are explicit, bounded **trial-only** controls implemented by `iq3_s_sc117_abliterated_profile.py`; they are not normal Hive or HostLLM settings. The normal menu preflight rejects these experimental environment variables. A new comparison must use a reviewed maintenance controller as described above, not a hand-edited Hive profile or a direct unaudited launcher command.

| Variable | Purpose | Tested / accepted example |
|---|---|---|
| `STRATA_SC117_IQ3S_TRIAL_PIPELINE_WINDOWS` | Strata pipeline windows | `2` (tested); only `0` or `2` accepted |
| `STRATA_SC117_IQ3S_TRIAL_SPEC` | Speculative decode `--spec` | `3` (tested); integer 2–6 |
| `STRATA_SC117_IQ3S_TRIAL_PCIE_FRAC` | PCIe fraction | `0.2` (screened); numeric 0–1 |
| `STRATA_SC117_IQ3S_TRIAL_POOL_WORKERS` | Pool worker count | `3` (screened); integer 0–128 |
| `STRATA_SC117_IQ3S_TRIAL_TRIM_STAGE_WEIGHTS` | Trim stage weights toggle | `0` (screened) or `1` |
| `STRATA_SC117_IQ3S_TRIAL_GPU_ORDER` | Permutation of GPUs 0–3 | `1,2,3,0` (screened) |
| `STRATA_SC117_IQ3S_TRIAL_PREFILL` | Prefill setting | `auto`, `1024`, `2048`, `4096`, or `8192` |
| `STRATA_SC117_IQ3S_TRIAL_SPLIT` | Three layer-split boundaries | `12,24,36` (screened); `auto` also allowed |
| `STRATA_SC117_IQ3S_TRIAL_CACHE_MIB` / `_CACHE_SLOTS` | Conversation-cache budget | 0/disabled, or bounded MiB and 1/2/4 slots |

Only pipeline windows and `--spec` were retained in the current leading candidate; all other values stayed at baseline. Apply one knob at a time first, then test the candidate combination, repeat it, and verify all effective arguments from the live engine. Never infer the full table is tested just because the profile accepts a setting.

## Generic API measurement sketch

For a compatible, already-isolated Strata benchmark service, POST fixed prompts to `/v1/chat/completions` and persist the complete request/response timing metadata. For example, use `max_tokens: 384`, stable distinct fixture IDs and seeds, `temperature: 0.7`, `top_p: 0.95`, `top_k: 20`, `reasoning_effort: "high"`, and streaming enabled. Read the final `usage` and `timings` event; require `timings.predicted_n == usage.completion_tokens`, require zero `cache_n` for an uncached comparison, and report median `predicted_per_second`. For prefill, use a separate fixed ~5K-token fixture, small fixed output cap, and report `prompt_n`, `prompt_per_second`, `prompt_ms`, TTFT, and `cache_n` independently. Do not mix prefill requests into decode medians. A timing result is valid only if the model ID/runtime/effective settings and token counts are also recorded.

`bench_strata_runtime.py` provides functional and paired-load probes for its supported model IDs; verify that a target profile is explicitly supported before using it. Its concurrency/functional results are not a substitute for the single-request, uncached per-arm procedure above.

## Qualification remains separate

SC117 is still an unqualified trial. A throughput win cannot satisfy its qualification gate. See [STRATA_SC117_IQ3S.md](STRATA_SC117_IQ3S.md) for model/runtime provenance and the required live smoke, native-context-tail, GPU-vision, measured-MTP, cache-park/restore/reuse, prefill-screen, and resource-telemetry checks. A measured decode winner is not permission to enable normal hosting or to alter the default profile.
