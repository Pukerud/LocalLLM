# HostLLM — Local LLM Engine Manager

A multi-engine dashboard for running local LLMs on NVIDIA GPUs. Pick your engine, configure your model, and go.

## Engines

| # | Engine | Script | What it does |
|---|--------|--------|-------------|
| **1** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | Max context (262K), all GGUF models, adaptive vision, speculative decoding, benchmarks |
| **2** | **DFlash llama.cpp** (buun-llama-cpp) | `v1dflash_llama_cpp.sh` | DFlash block-diffusion speculative decoding for faster inference |
| **3** | **vLLM** (Docker) | `v1_vllm.sh` | Max throughput (50-127 TPS), tool calls, Docker-based |

All three share the same model directory (`llama_models/`) and GPU port (8080). Only one can run at a time.

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1llama_cpp.sh v1dflash_llama_cpp.sh v1_vllm.sh
./HostLLM.sh
```

### Prerequisites

- **NVIDIA GPU** with CUDA 12+
- **CUDA toolkit**: `/usr/local/cuda/bin/nvcc`
- **Docker** (for vLLM only)
- **Build tools**: `gcc`, `g++`, `cmake`, `git`
- **Utilities**: `curl`, `jq`, `wget`

## llama.cpp Dashboard (ik_llama.cpp)

Full-featured dashboard with:

- **Adaptive server launch** — text-only or vision (GPU or CPU mmproj offload)
- **OpenClaw mode** — long context (64K/128K/256K) with optional vision + draft models
- **Cowork server** — Anthropic/Claude-compatible API with `--alias`
- **Benchmarking** — practical q4_0 baseline, KV quality matrix, full model sweep
- **Model management** — download from URL, delete, list with benchmark stats
- **Install/update** — clones and builds [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for RTX 4090 (sm_89)

## DFlash Dashboard (buun-llama-cpp)

Specialized dashboard for [DFlash speculative decoding](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF):

- **Block-diffusion draft model** — `--spec-type dflash` for faster token generation
- **Thinking forced OFF** — required for good DFlash acceptance rates
- **Vision support** — GPU or CPU mmproj offload
- **Install/update** — clones and builds [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) with flash attention flags

### DFlash Draft Models

Download from [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF):

| File | Size | Notes |
|------|------|-------|
| `dflash-draft-3.6-q8_0.gguf` | 1.75 GB | **Recommended** — matches F16 acceptance |
| `dflash-draft-3.6-q4_k_m.gguf` | 1.03 GB | Only if VRAM-constrained |

### DFlash Tested Performance (RTX 4090, 24 GB VRAM)

Qwen3.6-27B models with `dflash-draft-3.6-q8_0.gguf`, `--reasoning off`, no KV cache flags:

| Model | Model Size | Context | Speed | DFlash Acceptance | Result |
|-------|-----------|---------|-------|-------------------|--------|
| IQ4_XS | ~15 GB | 6,048 | 111 t/s | 68% | ✅ |
| IQ4_XS | ~15 GB | 32,768 | 63 t/s | 18% | ✅ |
| IQ4_XS | ~15 GB | 65,536 | 73 t/s | 40% | ✅ |
| IQ4_XS | ~15 GB | **81,920** | **77 t/s** | **46%** | ✅ **Best for IQ4_XS** |
| IQ4_XS | ~15 GB | 98,304 | — | — | ❌ OOM (tree buffers) |
| Q5_K_M | ~18 GB | 6,048 | — | — | ✅ |
| Q5_K_M | ~18 GB | **16,384** | **70 t/s** | **38%** | ✅ **Best for Q5_K_M** |
| Q5_K_M | ~18 GB | 32,768 | — | — | ❌ OOM (tree buffers) |

> **Why the OOM?** DFlash allocates ~1.4 GB tree verify buffers (`48 layers × 20 tokens`) + ~300 MB recurrent state on top of normal model + KV memory. The script auto-detects model file size and warns if the selected context may exceed VRAM.

> **Why no KV cache flags?** The HuggingFace example doesn't use `-ctk`/`-ctv`. Adding them changed the memory layout and left no headroom for DFlash's runtime allocations. Match the example exactly for stability.

> **Vision + DFlash?** The buun-llama-cpp fork does **not** support combining DFlash speculative decoding with multimodal/vision. The server reports "speculative decoding is not supported by multimodal" and then segfaults. Vision works without DFlash using `--no-mmproj-offload` for CPU mmproj offload.

## vLLM Dashboard

Docker-based vLLM with model selection, compose profiles, and container management.

## Directory Layout

```
./
├── HostLLM.sh                 ← Engine picker (start here)
├── v1llama_cpp.sh             ← llama.cpp dashboard
├── v1dflash_llama_cpp.sh      ← DFlash dashboard
├── v1_vllm.sh                 ← vLLM dashboard
├── llama_models/              ← Shared GGUF model pool
│   ├── *.gguf                 ← Target models
│   ├── dflash-draft-*.gguf    ← DFlash draft models
│   └── mmproj-*.gguf          ← Vision projectors
├── ik_llama.cpp/              ← llama.cpp build (gitignored)
├── buun-llama-cpp/            ← DFlash build (gitignored)
└── vllm_models/               ← vLLM models & compose (gitignored)
```

## Notes

- All builds compile natively for RTX 4090 (CUDA arch 89). Change `-DCMAKE_CUDA_ARCHITECTURES` in the install functions if you have a different GPU.
- The `llama_models/` directory is not tracked in git — add your `.gguf` files manually.
- Server state files (`.server_info`, `.server_info_dflash`) are used to detect which engine is running.
