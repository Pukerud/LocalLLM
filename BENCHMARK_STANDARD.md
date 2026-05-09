# Benchmark Standard

Every engine benchmark in the README follows this exact format.

## Hardware Block

Every benchmark section starts with a hardware spec line:

```
_Benchmarked on: GPU | VRAM | Context | KV cache | Date_
```

## Table Format

Standard columns for all engines:

| Target Model | Config | Speed (tok/s) | Prompt (tok/s) | Accept Rate | Notes |
|---|---|---:|---:|---:|---|

### Column definitions

| Column | What it measures | Source |
|--------|-----------------|--------|
| **Target Model** | Short model name (quant) | — |
| **Config** | Key config flags (draft model, MTP tokens, etc.) | — |
| **Speed (tok/s)** | Generation speed — completion tokens / wall time | `eval time` from server log or API `usage.completion_tokens` / elapsed |
| **Prompt (tok/s)** | Prompt evaluation speed | `prompt eval time` from server log |
| **Accept Rate** | Speculative decoding acceptance: accepted / generated draft tokens | `draft acceptance rate` from server log |
| **Notes** | Context size, special flags, OOM warnings, etc. | — |

## Benchmark conditions

- **Prompt:** `"Write a Python function that finds the longest increasing subsequence in a list of integers. Include proper type hints, docstring, and a few test cases with assertions."`
- **Max tokens:** 1024 completion tokens
- **Temperature:** 0.6
- **Top-K:** 20
- **Min-P:** 0.0
- **Reasoning:** ON (if engine supports it)

These are fixed across ALL engine benchmarks so results are comparable.

## Variations allowed per engine

Engines that have different modes (e.g. draft models, MTP token counts) get a sub-table showing the variation. The table structure stays the same.

Example variations:
- **BeeLlama DFlash:** different draft model quants
- **MTP:** different MTP token counts (1, 3, 5)
- **vLLM:** different context presets

## README placement

Each engine's benchmark goes directly under that engine's section in the README, right before the next engine section or `---` divider.

## Example (filled in)

### Benchmarks (RTX 3090, 24 GB, 100K ctx, turbo3_tcq KV, 2025-05-09)

_Prompt: "Write a Python function..." | 1024 max tokens | temp=0.6 | top-k=20 | reasoning on_

#### Target: Qwen3.6-27B-Q4_K_M

| Draft | Accept Rate | Accepted / Generated | Speed (tok/s) | Prompt (tok/s) |
|-------|-------------|---------------------|---------------|----------------|
| Q6_K (1.4 GB) | 47.2% | 860/1822 | **102.9** | 188.4 |
| Q8_0 (1.8 GB) | 41.8% | 856/2049 | 101.7 | 184.9 |
| IQ4_XS (892 MB) | **48.0%** | 853/1778 | 98.4 | 187.8 |
| Q5_K_M (1.2 GB) | 41.5% | 833/2008 | 90.0 | 184.6 |

**Best:** Q6_K draft = 102.9 tok/s at 47.2% acceptance.
