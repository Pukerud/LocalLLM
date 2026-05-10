# LocalLLM — Multi-Engine LLM Dashboard

https://github.com/user-attachments/assets/daa56313-deff-4664-a614-f2472aac92f6

> **Qwen3.6-27B at ~100 tok/s with MTP speculative decoding** using [ggml-org/llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)

---

A collection of launch scripts to run LLMs locally on NVIDIA GPUs. Eight inference engines, one port (8080).

**🟢 NVIDIA GPUs only** — all engines require CUDA.

---

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

Press **[0]** from the main menu for the **one-click Quick Start** — it auto-installs everything, downloads models, and starts the server. No technical knowledge needed.

### What Quick Start does automatically

1. **Installs CUDA Toolkit** if missing (12.4 or 12.8 for Blackwell GPUs)
2. **Builds BeeLlama.cpp** with DFlash, TurboQuant/TCQ, Flash Attention
3. **Downloads 3 models**: target (~15 GB IQ4_XS), draft (~1.2 GB Q5_K_M), mmproj (~0.9 GB)
4. **Detects your GPU(s)** and auto-calculates the optimal context size
5. **Starts the server** with vision, reasoning, and DFlash enabled
6. **Shows connection info** with all API endpoints

### Quick Start final display

```
==================================================================
  BEELLAMA DFLASH SERVER RUNNING
==================================================================

  Model:   Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS
  Draft:   Qwen3.6-27B-DFlash-Q5_K_M
  Context: 262144  |  KV: turbo3_tcq  |  DFlash: cross-ctx 1024
  Vision:  ON (CPU offload)  |  Reasoning: ON
  GPUs:    1x RTX 3090 (24 GB)

  Connect from any device on your network:

  Chat UI:       http://192.168.1.45:8080
  API Base:      http://192.168.1.45:8080/v1
  Anthropic:     http://192.168.1.45:8080/v1/messages

  API Key: any string (e.g. sk-1234) or leave blank

  OpenWebUI:    OpenAI base URL → http://192.168.1.45:8080/v1
  Pi / Codex:   OPENAI_API_BASE=http://192.168.1.45:8080/v1
  Cline / Continue: OpenAI compatible → http://192.168.1.45:8080/v1
  Anthropic SDK:  base_url → http://192.168.1.45:8080/v1
==================================================================

  CPU: 1%
  GPU 0:  45%   |   VRAM: 8.8 GB / 12 GB (72%)   |   Temp: 52 degC
  GPU 1:  38%   |   VRAM: 9.1 GB / 12 GB (75%)   |   Temp: 49 degC
  TOTAL: VRAM: 17.9 GB / 24 GB (74%)   |   GPUs: 2

  [1] Stop server and return to menu
  [2] Return to menu (keep server running)

  Select [1/2]:
```

Stats update live. The server stays running when you press **[2]** — access it from any device on your network.

---

## Main Menu

```
==========================================================
  HostLLM — Engine Picker
==========================================================

  Status:  No engine running

  Quick Start:
  ------------
  [0] Quick Start           BeeLlama DFlash — up to 105 tok/s, vision, reasoning, dual GPU (no draft) (Tested on 3090)
  [1] Quick Start (Legacy)  MTP — up to 100 tok/s, no vision, dual GPU (Tested on 4090)

  Engines:
  -------
  [2] llama.cpp       ik_llama.cpp — max context (262K), all GGUF models
  [3] DFlash llama.cpp buun-llama-cpp — DFlash speculative decoding
  [4] vLLM            Docker — max throughput (50-127 TPS), tool calls
  [5] Lucebox DFlash  lucebox-hub — DDTree (~104 t/s on 4090)
  [6] llama.cpp MTP   ggml-org/llama.cpp — native MTP speculative decoding
  [7] BeeLlama DFlash Anbeeld/beellama.cpp — full dashboard (manual control)
  [8] ZAYA1-8B        Zyphra vLLM — 8B MoE, 760M active, reasoning

  Controls:
  ---------
  [9] Kill All    Stop whatever is running
  [10] Update      Check git repo for newer version
  [11] Exit

  Select:
```

All engine sub-menus have:
- **[99] Back to Main Menu** — returns to this screen
- **[98] Exit** — quits the whole app

---

## Prerequisites

- **GPU:** NVIDIA only (RTX 3060+, 3090, 4070, 4090, 5080, etc.)
  - Minimum 16 GB total VRAM (single or multi-GPU)
  - Multi-GPU supported — model and KV cache are split automatically
- **OS:** Linux (tested Ubuntu 22.04)
- **Disk:** ~80 GB for models + builds
- **Docker:** Required for vLLM (engine 3) only
- **CUDA:** Auto-installed by Quick Start if missing

---

## Supported GPUs

Quick Start auto-detects your GPU(s) and adjusts:

| Setup | Total VRAM | Context | Notes |
|-------|-----------|---------|-------|
| 1× RTX 4090 | 24 GB | ~131K | Recommended |
| 1× RTX 3090 | 24 GB | ~131K | Same VRAM, works great |
| 2× RTX 3090 | 48 GB | ~262K | Max context |
| 2× RTX 3060 | 24 GB | ~50K | Tight but works |
| 2× RTX 4090 | 48 GB | ~262K | Max context |
| RTX 5080/5090 | 16/32 GB | varies | Auto-installs CUDA 12.8 |

---

## Engines

| # | Engine | Script | Speed | Status | Best for |
|---|--------|--------|------:|--------|----------|
| **0** | **Quick Start** | `v1beellama.sh --quickstart` | **up to 105 tok/s** | ✅ | BeeLlama DFlash, vision, reasoning, dual GPU (no draft) |
| **1** | **Quick Start (Legacy)** | `v1llama_mtp.sh --quickstart` | **up to 100 tok/s** | ✅ | MTP, dual GPU, no vision |
| **2** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | ~35-40 tok/s | ✅ Stable | Vision, max context (262K) |
| **3** | **DFlash** (buun fork) | `v1dflash_llama_cpp.sh` | — | ❌ | Under development |
| **4** | **vLLM** (Docker) | `v1_vllm.sh` | ~70 tok/s* | ✅ Working | Production API, tool use |
| **5** | **Lucebox DFlash** | `v1lucebox.sh` | **~104 tok/s** | ⚠️ Unstable | Fastest when stable |
| **6** | **llama.cpp MTP** (PR #22673) | `v1llama_mtp.sh` | **up to 100 tok/s** | ✅ Working | Fast text, native MTP |
| **7** | **BeeLlama DFlash** (Anbeeld) | `v1beellama.sh` | **~100+ tok/s** | ✅ Working | Full dashboard, manual control |
| **8** | **ZAYA1-8B** (Zyphra vLLM) | `v1zaya.sh` | TBD | ✅ Working | 8B MoE / 760M active, reasoning, tools |

All share port 8080. Only one runs at a time.

**Recommended:** Engine **0** (Quick Start) for best speed + quality + vision, or **7** (BeeLlama full dashboard) for manual control.

\* *vLLM speed varies by preset: 20K ctx ~110 tok/s, 48K ctx ~70 tok/s, 128K ctx ~55 tok/s.*

---

## API Endpoints

The server exposes OpenAI and Anthropic-compatible endpoints:

| Endpoint | Purpose |
|----------|---------|
| `http://IP:8080/v1/chat/completions` | OpenAI Chat (main) |
| `http://IP:8080/v1/completions` | Text completions |
| `http://IP:8080/v1/models` | List models |
| `http://IP:8080/v1/responses` | OpenAI Responses API |
| `http://IP:8080/v1/messages` | Anthropic Messages API |
| `http://IP:8080/v1/embeddings` | Embeddings |
| `http://IP:8080/health` | Health check |

### Connecting from clients

| Client | How to connect |
|--------|---------------|
| **OpenWebUI** | OpenAI base URL → `http://IP:8080/v1` |
| **Pi (coding agent)** | `OPENAI_API_BASE=http://IP:8080/v1` |
| **Codex CLI** | `OPENAI_API_BASE=http://IP:8080/v1` |
| **Cline / Continue** | OpenAI compatible → `http://IP:8080/v1` |
| **Cursor** | Set OpenAI base URL → `http://IP:8080/v1` |
| **Anthropic SDK** | `base_url="http://IP:8080/v1"` |

**API Key:** Any string (e.g. `sk-1234`) or leave blank. The server doesn't require authentication.

---

## Engine 5 — llama.cpp MTP (Full Dashboard)

For users who want full control over settings. Uses [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) — native Multi-Token Prediction.

### Dashboard menu

```
 --- SETUP ---
 [0] Install / Update llama.cpp (MTP PR #22673)
 [1] Convert HF Model -> GGUF (preserve MTP layers + quantize)
 [2] Download Fixed Chat Template (froggeric)

 --- MTP SERVER ---
 [3] Start MTP Server (configure context, KV, MTP tokens)
 [7] Quick Start MTP   (Reddit PR params -- just pick model)
 [4] Stop Server

 --- MANAGEMENT ---
 [5] Download Model (.gguf URL)
 [6] Delete Model

 [99] Back to Main Menu
 [98] Exit
```

### Recommended defaults (RTX 4090)

| Setting | Value | Why |
|---------|-------|-----|
| Quantization | **Q5_K_M** (~19 GB) | Best quality/size for 24GB |
| Context | **262144 (256K)** | Max with q4_0 KV |
| KV cache | **q4_0** | Fits 256K in 24GB |
| MTP tokens | **5** | Speculative lookahead |
| Template | **Fixed jinja v9** | Bundled, fixes 7 bugs |

---

## Engine 8 — ZAYA1-8B (Zyphra vLLM)

Runs [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) — a novel MoE architecture with **8B total / 760M active parameters** (16 experts, top-1 routing). Competitive with models 10× its size on math and coding benchmarks.

Uses [Zyphra's vLLM fork](https://github.com/Zyphra/vllm/tree/zaya1-pr) — the architecture (MoE + Mamba cache + CCA + EDA + MOD) is not supported by llama.cpp or any GGUF-based engine.

### Quickstart

```bash
# 1. Install Zyphra's vLLM fork (builds from source, ~10-20 min)
pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr"

# 2. Start the server
vllm serve Zyphra/ZAYA1-8B --port 8080 \
   --mamba-cache-dtype float32 --dtype bfloat16 \
   --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser zaya_xml
```

Or use the dashboard: `./HostLLM.sh` → press **[8]**.

### Dashboard menu

```
 [0] Install / Update Zyphra vLLM fork
 [1] Start Server (local model)
 [2] Start Server (from HuggingFace Hub)
 [5] Download Model (~16.5 GB)
 [D] Delete Model
 [6] Stop Server
 [7] Quick Benchmark
 [8] View Server Logs
 [s] Setup Status
```

### Key specs

| Property | Value |
|----------|-------|
| Total parameters | 8.4B |
| Active per token | 760M |
| Architecture | MoE (16 experts, top-1) + Mamba cache + CCA |
| Disk size (bf16) | 16.5 GB |
| VRAM needed | ~18 GB (fits 3090/4090) |
| Context window | 131K |
| Reasoning | Yes (qwen3 parser) |
| Tool calls | Yes (zaya_xml parser) |

### Recommended sampling

| Use case | Temperature | top-p | top-k |
|----------|------------|-------|-------|
| General / math | 1.0 | 0.95 | -1 |
| Code / agent | 0.6 | 0.95 | -1 |

---

## Engine 6 — BeeLlama DFlash (TurboQuant + Vision + Reasoning)

Uses [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) — a fork with DFlash speculative decoding, TurboQuant/TCQ KV cache compression, working vision, and reasoning support. Follows the [official quickstart guide](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md).

### Advantages over other engines

- **TurboQuant + TCQ KV cache** — `turbo4` (4.125 bpv) and `turbo3_tcq` (3.25 bpv) compress KV cache up to ~5× vs FP16, fitting 120K+ context in 24 GB VRAM with better precision than `q4_0`
- **DFlash speculative decoding** — a small draft model reads hidden states from the target model via cross-attention, predicting multiple tokens ahead for verification in a single forward pass
- **Vision works** — `--mmproj` + `--no-mmproj-offload` runs the multimodal projector on CPU, freeing GPU VRAM
- **Reasoning ON** — thinking tokens give the drafter richer context for better predictions (`--reasoning on` + `preserve_thinking`)
- **Adaptive draft depth** — automatically adjusts draft depth based on real-time acceptance rates

### Dashboard menu

```
 --- SETUP ---
 [0] Install / Update beellama.cpp (Anbeeld/beellama.cpp)

 --- BEELLAMA DFLASH SERVER ---
 [1] Start DFlash Server (Precision preset — Q5_K_S + turbo4 + reasoning ON)
 [2] Start DFlash Server (Speed preset — Q4_K_M + turbo3_tcq)
 [3] Start DFlash Server (Custom — pick everything)
 [4] Stop Server

 --- MANAGEMENT ---
 [5] Download Target Model
 [6] Download DFlash Draft Model
 [7] Download mmproj (vision projector)
 [8] Delete Model

 [99] Back to Main Menu
 [98] Exit
```

### Config presets

| Preset | Target | Draft | K cache | V cache | Context | Reasoning |
|--------|--------|-------|---------|---------|---------|----------|
| **Precision** | Q5_K_S | Q4_K_M | turbo4 | turbo3_tcq | 122800 | ON |
| **Speed/VRAM** | Q4_K_M | Q4_K_M | turbo3_tcq | turbo3_tcq | 131072 | ON |
| **Custom** | manual | manual | manual | manual | manual | manual |

### Required models

**Target model** — from [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF):
- `Qwen3.6-27B-Q5_K_S.gguf` (precision preset)
- `Qwen3.6-27B-Q4_K_M.gguf` (speed/VRAM preset)

**DFlash draft model** — from [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) or [Ardenzard/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF):
- `Qwen3.6-27B-DFlash-Q4_K_M.gguf` (recommended, ~2.7 GB)
- `Qwen3.6-27B-DFlash-Q5_K_M.gguf` (more precision, ~3.1 GB)
- `Qwen3.6-27B-DFlash-IQ4_XS.gguf` (smallest, ~1.5 GB)

**Multimodal projector** (optional, for vision) — from [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF):
- `mmproj-BF16.gguf`

All three can be downloaded from the dashboard menu ([5], [6], [7]).

### Recommended settings (RTX 3090/4090, 24 GB)

| Setting | Precision | Speed/VRAM |
|---------|-----------|------------|
| Target | Q5_K_S (~19 GB) | Q4_K_M (~16 GB) |
| Draft | Q4_K_M (~2.7 GB) | Q4_K_M (~2.7 GB) |
| Context | 122800 | 131072 |
| K cache | turbo4 (4.125 bpv) | turbo3_tcq (3.25 bpv) |
| V cache | turbo3_tcq (3.25 bpv) | turbo3_tcq (3.25 bpv) |
| Cross-ctx | 1024 | 1024 |
| `-ub` | 256 | 256 |
| Reasoning | ON | ON |
| Vision | ON (CPU offload) | ON (CPU offload) |

### Benchmarks (RTX 3090, 24 GB, 100K ctx, turbo3_tcq KV, 2025-05-09)

_Prompt: "Write a Python function that finds the longest increasing subsequence..." | 1024 max tokens | temp=0.6 | top-k=20 | reasoning on_

#### Target: Qwen3.6-27B-Q4_K_M (16 GB)

| Draft | Accept Rate | Accepted / Generated | Speed (tok/s) | Prompt (tok/s) |
|-------|-------------|---------------------|---------------|----------------|
| Q6_K (1.4 GB) | 47.2% | 860/1822 | **102.9** | 188.4 |
| Q8_0 (1.8 GB) | 41.8% | 856/2049 | 101.7 | 184.9 |
| IQ4_XS (892 MB) | **48.0%** | 853/1778 | 98.4 | 187.8 |
| Q5_K_M (1.2 GB) | 41.5% | 833/2008 | 90.0 | 184.6 |

#### Target: Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS (15 GB)

| Draft | Accept Rate | Accepted / Generated | Speed (tok/s) | Prompt (tok/s) |
|-------|-------------|---------------------|---------------|----------------|
| Q5_K_M (1.2 GB) | **49.1%** | 853/1736 | **104.8** | 230.5 |
| Q8_0 (1.8 GB) | 44.0% | 837/1903 | 95.0 | 212.5 |
| Q6_K (1.4 GB) | 43.2% | 822/1901 | 87.9 | 227.5 |
| IQ4_XS (892 MB) | 40.5% | 802/1982 | 83.3 | 242.5 |

#### Target: Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q5_K_M (19 GB)

| Draft | Accept Rate | Accepted / Generated | Speed (tok/s) | Prompt (tok/s) |
|-------|-------------|---------------------|---------------|----------------|
| IQ4_XS (892 MB) | **42.0%** | 798/1898 | 74.1 | 261.8 |
| Q8_0 (1.8 GB) | 39.8% | 817/2054 | **78.5** | 255.0 |
| Q6_K (1.4 GB) | 38.8% | 814/2099 | 76.6 | 255.9 |
| Q5_K_M (1.2 GB) | 38.5% | 807/2097 | 73.8 | 258.6 |

**Best overall:** IQ4_XS target + Q5_K_M draft = **104.8 tok/s** at **49.1%** acceptance.

**Best acceptance:** Q4_K_M target + IQ4_XS draft = **48.0%** at 98.4 tok/s.

Run speed benchmarks with `bash benchmarks/speed_beellama.sh`.

### Quality Benchmark (RTX 3090, 24 GB, 100K ctx, turbo3_tcq KV, 2025-05-09)

_Experimental automated quality test — 4 prompt classes scored automatically. Not a replacement for human evaluation. Uses fixed prompts with objective scoring criteria (code runs, answer matches, constraints counted, keywords checked). Run 3× per model due to temperature variance and report best._

_Prompts: merge intervals (10 assertions), Einstein puzzle (15 clues), 8-constraint essay, CAP theorem (12 concepts) | temp=0.6 | reasoning on_

| Model | Code | Reasoning | Instruction | Knowledge | **Best** | Range |
|-------|------|-----------|-------------|-----------|---------|-------|
| **NEO-CODE Q5_K_M** (19 GB) | 10/10 | **10/10** | **10/10** | 9/10 | **39/40** 🏆 | 35-39 |
| NEO-CODE IQ4_XS (15 GB) | 10/10 | 10/10 | 8/10 | 9/10 | 37/40 | 37 |
| Q4_K_M (16 GB) | 10/10 | 8/10 | 10/10 | 9/10 | 36/40 | 36-37 |

**Q5_K_M is the best model** — highest peak (39/40) with perfect reasoning and instruction scores. IQ4_XS is the value pick (smallest, still 37/40). All models ace code generation.

Run quality benchmarks with `bash benchmarks/quality_test.sh [port] [host]` or test all models with `bash benchmarks/quality_all_targets.sh`.

---

## Directory Layout

```
./
├── HostLLM.sh              ← Engine picker (start here)
├── chat_templates/          ← Bundled fixed jinja templates
├── v1llama_mtp.sh          ← Engine 6 / Legacy Quick Start dashboard
├── v1llama_cpp.sh          ← Engine 1 dashboard
├── v1dflash_llama_cpp.sh   ← Engine 2 dashboard
├── v1_vllm.sh              ← Engine 3 dashboard
├── v1lucebox.sh            ← Engine 4 dashboard
├── v1beellama.sh           ← Engine 0/7 — Quick Start + full dashboard
├── v1zaya.sh               ← Engine 8 — ZAYA1-8B (Zyphra vLLM)
├── benchmarks/             ← Speed & quality benchmark scripts
│   ├── BENCHMARK_STANDARD.md  ← Standard format docs
│   ├── speed_beellama.sh      ← Speed: all target×draft combos
│   ├── quality_test.sh        ← IQ test: code, reasoning, instruction, knowledge
│   └── quality_all_targets.sh ← IQ test all models automatically
├── llama_models/           ← Shared GGUF model pool
├── llama_cpp_mtp/          ← MTP build (gitignored)
├── ik_llama.cpp/           ← llama.cpp build (gitignored)
├── buun-llama-cpp/         ← DFlash build (gitignored)
├── beellama-cpp/           ← BeeLlama build (gitignored)
├── lucebox-hub/            ← Lucebox build (gitignored)
└── vllm_models/            ← vLLM Docker + model weights (gitignored)
├── zaya_models/            ← ZAYA1-8B model weights (gitignored)
```

## Notes

- All builds auto-detect GPU architecture (sm_86, sm_89, sm_120, etc.) — no manual config needed.
- Multi-GPU tensor splitting is automatic — just plug in multiple NVIDIA GPUs.
- `llama_models/` is gitignored — add `.gguf` files via dashboard menus or manually.
- MTP GGUF files must be converted with the PR #22673 converter — standard GGUFs don't have MTP layers.
- Each engine tracks its own state via `.server_info*` files — the main menu auto-detects which is running.
- Use **[9] Update** from the main menu to pull the latest version from GitHub.

## License

Scripts are MIT. Engine repos have their own licenses.
