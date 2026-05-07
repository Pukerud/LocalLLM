# LocalLLM -- Multi-Engine LLM Dashboard for RTX 4090 (24GB VRAM)

https://github.com/user-attachments/assets/daa56313-deff-4664-a614-f2472aac92f6

> **Qwen3.6-27B at ~100 tok/s on a single RTX 4090** using llama.cpp MTP ([ggml-org/llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673))

---

A collection of launch scripts to run 27B-class LLMs locally on a single RTX 4090 (24GB VRAM). Five inference engines, one GPU, one port (8080).

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

Pick an engine from the menu — each dashboard handles building, model downloading, and server startup automatically.

### Prerequisites

- **GPU:** NVIDIA RTX 4090 (24GB VRAM) — also works on 3090, 4080, etc.
- **CUDA:** 12+ (`/usr/local/cuda/bin/nvcc`)
- **OS:** Linux (tested Ubuntu 22.04)
- **Docker:** Required for vLLM (engine 3) only
- **Disk:** ~80GB for models + builds

## Engines

| # | Engine | Script | Speed | Status | Best for |
|---|--------|--------|------:|--------|----------|
| **1** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | ~35-40 tok/s | ✅ Stable | Vision, max context (262K), most reliable |
| **2** | **DFlash** (buun fork) | `v1dflash_llama_cpp.sh` | — | ❌ Not working | Under development |
| **3** | **vLLM** (Docker) | `v1_vllm.sh` | ~70 tok/s* | ✅ Working | Production API, tool use, MTP spec-decode |
| **4** | **Lucebox DFlash** (lucebox-hub) | `v1lucebox.sh` | **~104 tok/s** | ⚠️ Unstable | Fastest when stable, needs tuning |
| **5** | **llama.cpp MTP** (PR #22673) | `v1llama_mtp.sh` | **~100 tok/s** | ✅ Working | Fast text-only, native MTP, ~256K context |

All five share `llama_models/` and port 8080. Only one runs at a time.

**Recommended today:** Engine **1** (stable, vision, max context) or Engine **5** (fastest text-only with MTP).

\* *vLLM speed varies by preset: 20K ctx ~110 tok/s, 48K ctx ~70 tok/s, 128K ctx ~55 tok/s.*

---

## Engine 1 — llama.cpp (ik_llama.cpp)

The reliable daily driver. Vision support, up to 262K context, works with all GGUF models. ~35-40 tok/s without speculative decoding.

**Dashboard:** `./v1llama_cpp.sh` → Install → Pick model → Go.

---

## Engine 3 — vLLM (Docker)

Production-grade API with tool call support. Uses [vLLM](https://github.com/vllm-project/vllm) with [Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) for MTP speculative decoding.

**5 Presets:**

| # | Preset | KV | MTP | Context | Best for |
|---|--------|----|:---:|--------:|----------|
| **1** | Fast Chat | fp8 | 5 | 20K | Short conversations, max speed |
| **2** | General Chat | fp8 | 3 | 48K | Best all-rounder, stable |
| **3** | IDE/Tools | fp8 | 3 | 63K | Cline/Cursor (fp8 ceiling) |
| **4** | Long Vision | TQ3 | 3 | 128K | Long docs, tool calls |
| **5** | Long Text | TQ3 | 3 | 150K | Max context on 24GB |

**Key learnings:**
- **MTP=5 crashes on long output** — only safe for short chat
- **MTP=3 is rock-solid** — use for anything generating 4K+ tokens
- **fp8 KV ceiling ~63K** on 24GB — above that, TQ3 needed

**Dashboard:** `./v1_vllm.sh` → **[0] Install/Update** → choose preset → go.

---

## Engine 4 — Lucebox DFlash (⚠️ Unstable)

Uses [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub) with **DDTree** tree-structured verify and block-diffusion speculative decoding. Fastest engine when stable (~104 tok/s), but currently unstable on current settings — needs more tuning.

**Why it's fast:** DDTree verifies 22 candidate branches per step (vs 1 chain in buun's DFlash), achieving 43.5% acceptance rate and ~6.5 tokens per step.

**Dashboard:** `./v1lucebox.sh` → Install → Pick model → Configure → Go.

---

## Engine 5 — llama.cpp MTP (✅ Working)

Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) — **native Multi-Token Prediction** for Qwen3.6 models. No separate draft model needed — the MTP heads are built into the weights.

### How it works

Qwen3.6-27B includes MTP tensor layers. This PR:
- Preserves MTP layers during GGUF conversion (standard converters strip them)
- Uses MTP heads for **self-speculative decoding**
- Achieves ~2.5x speedup over standard autoregressive decoding

### Dashboard menu

```
./v1llama_mtp.sh

[0] Install/Update  -- clones llama.cpp, fetches PR #22673, compiles
[1] Convert Model   -- downloads from HuggingFace, converts to MTP GGUF
[2] Download Template -- froggeric's fixed jinja templates (fixes 7 bugs)
[3] Start Server    -- select model, context, KV cache, MTP tokens
[4] Stop Server
[5] Download MTP GGUF -- download pre-converted models
[6] Status / Logs
[7] Quick Start     -- Reddit PR author's exact command, just pick model
```

### Recommended defaults (RTX 4090)

| Setting | Value | Why |
|---------|-------|-----|
| Quantization | **Q5_K_M** (~19 GB) | Best quality/size ratio for 24GB |
| Context | **262144 (256K)** | Maximum with q4_0 KV |
| KV cache | **q4_0** | Fits 256K in 24GB with MTP |
| MTP tokens | **5** | Speculative lookahead |
| Thinking | **OFF** | Best speed |

### Chat template

The original Qwen3.6 jinja has [7 vLLM-specific bugs](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates). The dashboard auto-downloads froggeric's fixed templates.

---

## Accessing the Server

All engines serve on port 8080 with OpenAI-compatible API:

```
Endpoint:  http://localhost:8080/v1/chat/completions
API Key:   sk-any (or anything)
```

Works with Open WebUI, LM Studio, Cline, or any OpenAI-compatible client.

## Models

Located in `llama_models/`. Any Qwen3.6-27B GGUF variant works with engine 1. Engine 5 requires GGUFs converted with the PR #22673 converter (filenames ending in `-mtp.gguf`).

## Directory Layout

```
./
├── HostLLM.sh              ← Engine picker (start here)
├── v1llama_cpp.sh          ← Engine 1 dashboard
├── v1dflash_llama_cpp.sh   ← Engine 2 dashboard
├── v1_vllm.sh              ← Engine 3 dashboard
├── v1lucebox.sh            ← Engine 4 dashboard
├── v1llama_mtp.sh          ← Engine 5 dashboard
├── llama_models/           ← Shared GGUF model pool
├── llama_cpp_mtp/          ← MTP build (gitignored)
├── ik_llama.cpp/           ← llama.cpp build (gitignored)
├── buun-llama-cpp/         ← DFlash build (gitignored)
├── lucebox-hub/            ← Lucebox build (gitignored)
└── vllm_models/            ← vLLM Docker + model weights (gitignored)
```

## Notes

- All builds compile for RTX 4090 (CUDA sm_89). Change `-DCMAKE_CUDA_ARCHITECTURES` for other GPUs.
- `llama_models/` is gitignored — add your `.gguf` files via dashboard menus or manually.
- MTP GGUF files must be converted with the PR #22673 converter — standard GGUFs don't have MTP layers.
- Each engine tracks its own state via `.server_info*` files — the main menu auto-detects which engine is running.

## License

Scripts are MIT. Engine repos have their own licenses.
