# DeepSeek .51 speed comparison — 2026-10-26

## Result

On this four-3090 host, **fully unloading DSpark (`--dspark none`)** was the useful measured option. Two cold model starts at the original **1,048,576-token window**, six experts/token and native vision enabled produced:

| Native timing, medians of arm medians | Stock auto, 1M | No drafter, 1M | Change |
|---|---:|---:|---:|
| Decode | 33.425 tok/s | 36.125 tok/s | **+8.08%** |
| Prefill, 2075 tokens | 169.20 tok/s | 202.25 tok/s | **+19.53%** |

`none` matters: `off` still loads the drafter's weights in this engine. No drafter frees that memory for the target model's routed experts; this is not reducing six experts to five, changing quantization, disabling vision or reducing the context window. This is a modest gain, not Flash-Next-like throughput. Model quality relative to Flash-Next was not benchmarked.

**Activation is deferred:** after the comparison restored canonical Hive/OctaSpace, an actual Blender rental occupied the GPUs. The explicit preset selector correctly refused with `Docker workload/rental present; starts blocked`. No rental or unknown process was interrupted. The existing normal profile remains stock-auto; the measured option and exact rollback are available for a later owned idle maintenance window.

## Compared settings

| Case | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| Baseline 1M A | 33.95 | 170.3 |
| 262K, auto | 35.60 | 177.7 |
| 262K, 12 CPU threads | 34.00 | 177.5 |
| 262K, drafter on / block 3 | 35.30 | 178.8 |
| 262K, no drafter | 35.60 | 210.8 |
| 262K, mmap experts on | 32.40 | 118.4 |
| Baseline 1M B | 32.90 | 168.1 |
| 1M, no drafter A | 35.95 | 202.4 |
| 1M, no drafter B | 36.30 | 202.1 |

262K alone helped a little, but did not justify discarding the larger window for this use. Twelve CPU threads and mmap were not wins here. No lower-quality five-expert preset was applied. No kernel/driver/system-toolkit, power-limit or clock-policy change was made; observed operator settings were held across arms.

## Method and safety

- Same pinned Abiray IQ2_XXS/Engram, stock DSpark assets, original BF16 vision tower and signed .51 binary. Every experimental launcher first verified normal profile/runtime/source provenance, then used a strictly allowlisted **in-memory** candidate configuration. Normal config/manifests were not rewritten by benchmarks.
- Cold model start per arm (not host power cycling), the same shipped expert-placement seed per arm, and one exact newly created HostLLM pause lease. Existing leases/rentals/manual work were not borrowed or killed. Only identity-recorded families were stopped; canonical Hive and OctaSpace were restored after both experiment phases.
- Nine requests/arm: three repetitions each of bounded code decode, prose decode and a 2075-token known-answer tail-retrieval prefill. Greedy temperature 0/top-p 1, reasoning none, 256 output cap for decode/8 for prefill. All fixture hashes and actual prompt lengths matched; API cache reuse was disabled and verified zero. SSE completion, native timings and TTFT were recorded. One-second GPU/RAM telemetry retained privately.
- Baseline A/B bracket and two independent full-context no-drafter starts. Baseline decode drift was about 3%, so smaller one-off differences should not be treated as precise gains.
- Native vision loaded in every arm and reported available; the final post-preset image smoke is pending the rental-free activation window. Earlier cache-Monitor E2E included actual same-origin red-image inference. This is not full-context population, broad quality, sustained-load, indefinite-stability or throughput qualification.

[Sanitized measurements](results/deepseek41-speed-20261026.json). Private raw sessions, logs, receipts and telemetry are under `/var/lib/localllm-strata/deepseek41-speed{-refine,}-20261026/`; unprivileged fixtures/adapters live in the matching calibration directories.

## Explicit selection and rollback

After safely stopping the **owned** DeepSeek session and only when no rental/other maintenance is active, as the normal operator:

```bash
STRATA_STATE_ROOT="$HOME/.local/state/locallm-strata/deepseek41-abiray" \
  ./v1strata.sh --prepare-deepseek41 --select-performance-preset full-quality-no-drafter

# Exact stock .51 rollback, same context/vision/cache/weights:
STRATA_STATE_ROOT="$HOME/.local/state/locallm-strata/deepseek41-abiray" \
  ./v1strata.sh --prepare-deepseek41 --select-performance-preset stock-auto
```

The selector verifies provenance, backs up small config/manifest files, and accepts only those two exact configurations. Root, live target families, foreign maintenance leases and rentals are refused. It never stops a workload itself. New installs and existing stock configs keep auto unless explicitly selected.

## Why a response can still take minutes

The inspected existing request was decoding around **44 tok/s**, but had already spent approximately **30,000 tokens thinking** and had a near-1M output allowance. Reasoning length and total output are separate from context reservation and raw decode throughput. This release defaults to a 32K DeepSeek thinking budget when none is supplied.

For latency-sensitive calls, a bounded `reasoning_budget_tokens` and `max_tokens`, or `reasoning_effort: "none"`, can reduce time much more than an 8% kernel/runtime gain. These are task/quality tradeoffs, not free throughput improvements; no default reasoning budget or output cap was silently reduced. A large context window does not require generating that many output tokens.
