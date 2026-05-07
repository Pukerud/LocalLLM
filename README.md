# LocalLLM -- Multi-Engine LLM Dashboard for RTX 4090 (24GB VRAM)

<video src="https://github.com/Pukerud/LocalLLM/releases/download/speed-demo-v1/speed.demo.1x4090.mp4" controls autoplay muted loop></video>

> **Qwen3.6-27B at ~100 tok/s on a single RTX 4090** using llama.cpp MTP ([ggml-org/llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673))

---

A collection of launch scripts to run 27B-class LLMs locally on a single RTX 4090 (24GB VRAM). Five inference engines, one GPU, one port (8080).

## Engines

| # | Engine | Script | Speed | Best for |
|---|--------|--------|------:|---------|
| **1** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | ~35-40 tok/s | Max context (262K), all GGUF models, vision |
| **2** | **DFlash llama.cpp** (buun fork) | `v1dflash_llama_cpp.sh` | ~40 tok/s | Experimental DFlash testing |
| **3** | **vLLM** (Docker) | `v1_vllm.sh` | ~70 tok/s* | Production API, tool use, MTP spec-decode |
| **4** | **Lucebox DFlash** (lucebox-hub) | `v1lucebox.sh` | **~104 tok/s** | Fastest single-user decode |
| **5** | **llama.cpp MTP** (ggml-org/llama.cpp PR #22673) | `v1llama_mtp.sh` | TBD | Native MTP speculative decoding, up to 180K context |

All five share the same model directory (`llama_models/`) and GPU port (8080). Only one can run at a time.

\* *vLLM speed varies by preset: 20K ctx ~110 tok/s, 48K ctx ~70 tok/s, 128K ctx ~55 tok/s (benchmarked on RTX 4090).*

## Quick Start

```bash
./HostLLM.sh
```

Pick an engine, pick a model, go.

### Prerequisites

- **GPU:** NVIDIA RTX 4090 (24GB VRAM) -- also works on 3090, 4080, etc.
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

# Engine 5 (MTP) is set up through its dashboard: run v1llama_mtp.sh -> [0] -> [1]

# Download models to llama_models/
huggingface-cli download <model-repo> <file> --local-dir llama_models/

# For Lucebox: download the DFlash draft
cd lucebox-hub/dflash && mkdir -p models/draft
python3 -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('z-lab/Qwen3.6-27B-DFlash', 'model.safetensors', local_dir='models/draft/')"
cd ../..

chmod +x HostLLM.sh v1llama_cpp.sh v1dflash_llama_cpp.sh v1_vllm.sh v1lucebox.sh v1llama_mtp.sh
./HostLLM.sh
```

## ⚡ vLLM -- The Speed King

Uses [vLLM](https://github.com/vllm-project/vllm) in Docker with [Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) for MTP speculative decoding.

### 5 Presets (pick based on your needs)

| # | Preset | KV | MTP | Context | Best for |
|---|--------|----|:---:|--------:|----------|
| **1** | Fast Chat | fp8 | 5 | 20K | Short conversations, max speed |
| **2** | General Chat | fp8 | 3 | 48K | Best all-rounder, stable |
| **3** | IDE/Tools | fp8 | 3 | 63K | Cline/Cursor (fp8 ceiling) |
| **4** | Long Vision | TQ3 | 3 | 128K | Long docs, tool calls |
| **5** | Long Text | TQ3 | 3 | 150K | Max context on 24GB |

All presets include tool call support (`qwen3_coder` parser).

### Key learnings from benchmarking

- **fp8 KV + MTP spec-decode** is the biggest win -- 2x faster than the old TQ3/MTP=3 config
- **MTP=5 crashes on long output** (CUDA illegal memory access bug in vLLM) -- only safe for short chat
- **MTP=3 is rock-solid** -- use for anything that generates 4K+ tokens
- **fp8 KV ceiling is ~63K context** on 24GB -- above that, TQ3 (3-bit) KV is needed
- **Genesis patches P67/P82** accelerate spec-decode on fp8 KV; **P65/P66** needed for TQ3 compatibility

### vLLM setup

```bash
# Clone with submodules (includes Genesis patches)
git clone --recurse-submodules https://github.com/Pukerud/LocalLLM.git

# Download the AutoRound INT4 model
huggingface-cli download	huahuacs/Qwen3.6-27B-AutoRound-INT4 --local-dir vllm_models/qwen3.6-27b-autoround-int4

# Pull the vLLM Docker image
docker pull vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08

# Launch
./v1_vllm.sh
# Pick [0] to install/update first, then choose a preset
```

---

## ⚡ Lucebox DFlash -- The Star

Uses [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) with **DDTree** tree-structured verify (budget=22) and **block-diffusion** speculative decoding.

### Why it's 2.6x faster than buun DFlash

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
**Draft:** z-lab/Qwen3.6-27B-DFlash BF16 (~3.3GB) * **DDTree budget=22**

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
| f16 / f16 | -- | ~16K only |

> **q4_0 is the sweet spot** -- fastest decode and fits 128K context. Context barely matters: only 8% drop from 16K->65K.

### Lucebox dashboard features

- **Model picker** -- auto-finds Qwen3.6-27B GGUFs in `llama_models/`
- **Context picker** -- 512 to 256K tokens
- **KV cache picker** -- q4_0, q8_0, tq3_0, f16, or custom
- **Quick start** -- defaults (32K ctx, auto KV) for one-command launch
- **Live GPU stats** -- refreshes every 2 seconds
- **In-server benchmark** -- tests the running server via OpenAI API

## DFlash buun fork -- Real Benchmarks

Previously tested [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) with chain-verify DFlash:

| Model | Context | Speed | Accept |
|-------|--------:|------:|-------:|
| Base IQ4_XS | 6K | 43 t/s | 18% |
| Base IQ4_XS | 80K | 40 t/s | 18% |
| HauhauCS IQ4_XS | 8K | 39 t/s | 16% |
| HauhauCS IQ4_XS | 80K | 37 t/s | 16% |

> **[!] Chain-verify DFlash provides no speedup at 17% acceptance.** The speculative overhead cancels any benefit. The previous README claimed 77-111 tok/s -- those numbers were fabricated by AI, never tested. Corrected with real numbers.

## llama.cpp MTP -- Native Speculative Decoding

Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) which adds **native Multi-Token Prediction** for Qwen3.6 models.

### How it works

Qwen3.6-27B includes built-in MTP tensor layers in its weights. This PR:

- Preserves MTP layers during GGUF conversion (existing converters strip them)
- Uses the MTP heads for **self-speculative decoding** -- no separate draft model needed
- Achieves ~2.5x speedup over standard autoregressive decoding

### Key differences from other engines

| | DFlash (Engine 2) | vLLM MTP (Engine 3) | **llama.cpp MTP (Engine 5)** |
|---|---|---|---|
| Draft model | Separate GGUF | Built-in (vLLM native) | **Built-in (GGUF native)** |
| Context | Up to 80K | Up to 48K (fp8) / 150K (TQ3) | **Up to 180K** |
| Vision | No | No | No |
| Build | buun fork | Docker + Genesis patches | **Stock llama.cpp + PR** |
| Chat template | Standard | Standard | **Fixed (froggeric)** |

### Setup

```bash
./v1llama_mtp.sh

# [0] Install/Update -- clones llama.cpp, fetches PR #22673, compiles for RTX 4090
# [1] Convert Model -- downloads Qwen3.6-27B from HuggingFace, converts to GGUF with MTP layers
#     - Auto-quantizes to Q5_K_M (recommended for 180K context on 24GB)
#     - Also offers Q4_K_M, Q6_K, Q8_0, F16
# [2] Download Template -- fetches froggeric's fixed jinja chat templates
#     - Fixes 7 vLLM-specific bugs in the original Qwen template
# [3] Start Server -- select model, context, MTP tokens, template
```

**Recommended defaults for RTX 4090:**
- Quantization: **Q5_K_M** (~18.5 GB)
- Context: **184320 (180K)** with q4_0 KV
- MTP tokens: **5** (speculative lookahead)
- Template: froggeric fixed (auto-downloaded)
- Thinking: OFF (best speed)

### Chat template fix

The original Qwen3.6 jinja template has [7 bugs from vLLM-specific workarounds](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) that break in other tools. The MTP dashboard downloads and applies these fixed templates automatically.

---

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

Lucebox needs models <=17GB to leave room for the 3.3GB draft + DDTree verify state.

## Directory Layout

```
./
|-- HostLLM.sh                 <-- Engine picker (start here)
|-- v1llama_cpp.sh             <-- llama.cpp dashboard
|-- v1dflash_llama_cpp.sh      <-- DFlash buun dashboard
|-- v1_vllm.sh                 <-- vLLM dashboard
|-- v1lucebox.sh               <-- Lucebox DFlash dashboard
|-- v1llama_mtp.sh             <-- llama.cpp MTP dashboard (PR #22673)
|-- v1lucebox_bench.py         <-- Lucebox server benchmark
|-- lucebox_kv_compare.py      <-- KV cache comparison tool
|-- llama_models/              <-- Shared GGUF model pool
|-- ik_llama.cpp/              <-- llama.cpp build (gitignored)
|-- buun-llama-cpp/            <-- buun DFlash build (gitignored)
|-- llama_cpp_mtp/             <-- llama.cpp MTP build (gitignored)
|-- lucebox-hub/               <-- Lucebox build (gitignored)
|-- vllm_models/
|   |-- compose/               <-- vLLM Docker Compose presets
|   |-- genesis/               <-- Genesis vLLM patches (submodule)
|   |-- patch_tolist_cudagraph.py
|   `-- qwen3.6-27b-autoround-int4/  <-- AutoRound INT4 model weights
```

## Notes

- All builds compile for RTX 4090 (CUDA sm_89). Change `-DCMAKE_CUDA_ARCHITECTURES` for other GPUs.
- `llama_models/` is not tracked in git -- add your `.gguf` files manually.
- `vllm_models/` compose files and Genesis patches are tracked. Model weights (`.safetensors`) are gitignored.
- Server state files (`.server_info*`) detect which engine is running. Each engine has its own: `.server_info` (llama.cpp), `.server_info_dflash` (DFlash), `.server_info_mtp` (MTP).
- vLLM presets use `--language-model-only` (no vision). For vision, use llama.cpp or Lucebox.
- MTP GGUF files must be converted with the PR #22673 converter -- existing GGUFs do not have MTP layers.

## License

Scripts are MIT. Engine repos have their own licenses.
