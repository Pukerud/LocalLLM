# LocalLLM — Multi-Engine LLM Dashboard for RTX 4090 (24GB VRAM)

A collection of launch scripts to run 27B-class LLMs locally on a single RTX 4090 (24GB VRAM). Four inference engines, one GPU, one port (8080).

## Engines

| # | Engine | Script | Speed | Best for |
|---|--------|--------|------:|---------|
| **1** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | ~35-40 tok/s | Max context (262K), all GGUF models, vision |
| **2** | **DFlash llama.cpp** (buun fork) | `v1dflash_llama_cpp.sh` | ~40 tok/s | Experimental DFlash testing |
| **3** | **vLLM** (Docker) | `v1_vllm.sh` | 50-127 TPS | Production API, tool use |
| **4** | **Lucebox DFlash** (lucebox-hub) | `v1lucebox.sh` | **~104 tok/s** | Fastest single-user decode |

All four share the same model directory (`llama_models/`) and GPU port (8080). Only one can run at a time.

## Quick Start

```bash
./HostLLM.sh
```

Pick an engine, pick a model, go.

### Prerequisites

- **GPU:** NVIDIA RTX 4090 (24GB VRAM) — also works on 3090, 4080, etc.
- **CUDA:** 12+ (`/usr/local/cuda/bin/nvcc`)
- **OS:** Linux (tested Ubuntu 22.04)
- **Docker:** Required for vLLM only
- **Disk:** ~80GB for models + builds

### Setup

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM

# Clone engine repos
git clone https://github.com/ggml-org/llama.cpp.git ik_llama.cpp         # Engine 1
git clone https://github.com/spiritbuun/buun-llama-cpp.git               # Engine 2
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub.git  # Engine 4

# Download models to llama_models/
huggingface-cli download <model-repo> <file> --local-dir llama_models/

# For Lucebox: download the DFlash draft
cd lucebox-hub/dflash && mkdir -p models/draft
python3 -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('z-lab/Qwen3.6-27B-DFlash', 'model.safetensors', local_dir='models/draft/')"
cd ../..

chmod +x HostLLM.sh v1llama_cpp.sh v1dflash_llama_cpp.sh v1_vllm.sh v1lucebox.sh
./HostLLM.sh
```

## ⚡ Lucebox DFlash — The Star

Uses [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) with **DDTree** tree-structured verify (budget=22) and **block-diffusion** speculative decoding.

### Why it's 2.6× faster than buun DFlash

| | buun fork (chain) | Lucebox (DDTree) |
|---|---:|---:|
| **Mean tok/s** | 40 | **104** |
| **Accept rate** | 17% | **43.5%** |
| **Tokens/step** | ~1.5 | **6.5** |
| **Verify method** | Chain (1 path) | Tree (22 branches) |
| **Draft model** | GGUF q8_0 (1.8GB) | BF16 safetensors (3.3GB) |

- **DDTree** verifies a tree of 22 candidate branches per step (vs 1 chain)
- **Block-diffusion draft** conditions every candidate on real target hidden states
- **Custom CUDA kernels** for tree-aware SSM state rollback
- **Matched Q3.6 DFlash draft** trained specifically for Qwen3.6

### HumanEval benchmark (RTX 4090)

**Target:** Qwen3.6-27B-Uncensored-HauhauCS-Aggressive IQ4_XS (~15GB)
**Draft:** z-lab/Qwen3.6-27B-DFlash BF16 (~3.3GB) · **DDTree budget=22**

| Prompt | tok/s | Accept% | AL/step |
|--------|------:|--------:|--------:|
| sum_product | **132.7** | 59.6% | 8.53 |
| mean_absolute_deviation | **119.0** | 50.7% | 7.53 |
| has_close_elements | **114.1** | 48.3% | 7.11 |
| separate_paren_groups | **107.8** | 43.1% | 6.74 |
| truncate_number | **103.2** | 43.1% | 6.40 |
| parse_nested_parens | **97.8** | 41.7% | 6.10 |
| rolling_max | **94.7** | 37.8% | 5.82 |
| intersperse | **94.7** | 37.2% | 5.82 |
| filter_by_substring | **94.3** | 37.8% | 5.82 |
| below_zero | **84.3** | 35.8% | 5.12 |
| **MEAN** | **104.3** | **43.5%** | **6.50** |

### KV cache comparison (RTX 4090, same model)

| KV type | tok/s @ 32K ctx | Max context in 24GB |
|---------|:---------------:|:-------------------:|
| **q4_0 / q4_0** | **52.3** | 128K |
| q8_0 / q8_0 | 51.2 | 64K |
| tq3_0 / tq3_0 | 48.8 | 256K |
| f16 / f16 | — | ~16K only |

> **q4_0 is the sweet spot** — fastest decode and fits 128K context. Context barely matters: only 8% drop from 16K→65K.

### Lucebox dashboard features

- **Model picker** — auto-finds Qwen3.6-27B GGUFs in `llama_models/`
- **Context picker** — 512 to 256K tokens
- **KV cache picker** — q4_0, q8_0, tq3_0, f16, or custom
- **Quick start** — defaults (32K ctx, auto KV) for one-command launch
- **Live GPU stats** — refreshes every 2 seconds
- **In-server benchmark** — tests the running server via OpenAI API

## DFlash buun fork — Real Benchmarks

Previously tested [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) with chain-verify DFlash:

| Model | Context | Speed | Accept |
|-------|--------:|------:|-------:|
| Base IQ4_XS | 6K | 43 t/s | 18% |
| Base IQ4_XS | 80K | 40 t/s | 18% |
| HauhauCS IQ4_XS | 8K | 39 t/s | 16% |
| HauhauCS IQ4_XS | 80K | 37 t/s | 16% |

> **⚠️ Chain-verify DFlash provides no speedup at 17% acceptance.** The speculative overhead cancels any benefit. The previous README claimed 77-111 tok/s — those numbers were fabricated by AI, never tested. Corrected with real numbers.

## Accessing the Server

All engines serve on port 8080 with OpenAI-compatible API:

```
Endpoint:  http://localhost:8080/v1/chat/completions
API Key:   sk-any (or anything)
```

Works with Open WebUI, LM Studio, Cline, or any OpenAI-compatible client.

## Models

Located in `llama_models/`. Compatible Qwen3.6-27B variants:

| Model | Size | Lucebox |
|-------|-----:|:-------:|
| Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf | 15 GB | ✅ |
| Qwen3.6-27B-Q5_K_M.gguf | 19 GB | ❌ OOM |

Lucebox needs models ≤17GB to leave room for the 3.3GB draft + DDTree verify state.

## Directory Layout

```
./
├── HostLLM.sh                 ← Engine picker (start here)
├── v1llama_cpp.sh             ← llama.cpp dashboard
├── v1dflash_llama_cpp.sh      ← DFlash buun dashboard
├── v1_vllm.sh                 ← vLLM dashboard
├── v1lucebox.sh               ← Lucebox DFlash dashboard
├── v1lucebox_bench.py         ← Lucebox server benchmark
├── lucebox_kv_compare.py      ← KV cache comparison tool
├── llama_models/              ← Shared GGUF model pool
├── ik_llama.cpp/              ← llama.cpp build (gitignored)
├── buun-llama-cpp/            ← buun DFlash build (gitignored)
├── lucebox-hub/               ← Lucebox build (gitignored)
└── vllm_models/               ← vLLM models & compose (gitignored)
```

## Notes

- All builds compile for RTX 4090 (CUDA sm_89). Change `-DCMAKE_CUDA_ARCHITECTURES` for other GPUs.
- `llama_models/` is not tracked in git — add your `.gguf` files manually.
- Server state files (`.server_info*`) detect which engine is running.

## License

Scripts are MIT. Engine repos have their own licenses.
