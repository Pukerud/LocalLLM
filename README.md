# LocalLLM — Multi-Engine LLM Dashboard

https://github.com/user-attachments/assets/daa56313-deff-4664-a614-f2472aac92f6

> **Qwen3.6-27B at ~100 tok/s with MTP speculative decoding** using [ggml-org/llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)

---

A collection of launch scripts to run 27B-class LLMs locally on NVIDIA GPUs. Five inference engines, one port (8080).

**🟢 NVIDIA GPUs only** — all engines require CUDA.

---

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

Press **[0]** from the main menu for the **one-click Quick Start** — it auto-installs everything, downloads a model, and starts the server. No technical knowledge needed.

### What Quick Start does automatically

1. **Installs CUDA Toolkit** if missing (12.4 or 12.8 for Blackwell GPUs)
2. **Builds llama.cpp** with MTP PR #22673
3. **Downloads the default model** (~16 GB, Q4_K_S)
4. **Detects your GPU(s)** and auto-calculates the optimal context size
5. **Applies fixed jinja chat template** (bundled)
6. **Starts the server** and shows connection info

### Quick Start final display

```
==================================================================
  MTP SERVER RUNNING
==================================================================

  Model:   Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf
  Context: 131072  |  KV: q4_0  |  MTP: 5
  GPUs:    2x (24 GB total)

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
  [0] Quick Start   One-click MTP — auto-install, download model, start server

  Engines:
  -------
  [1] llama.cpp       ik_llama.cpp — max context (262K), all GGUF models
  [2] DFlash llama.cpp buun-llama-cpp — DFlash speculative decoding
  [3] vLLM            Docker — max throughput (50-127 TPS), tool calls
  [4] Lucebox DFlash  lucebox-hub — DDTree (~104 t/s on 4090)
  [5] llama.cpp MTP   ggml-org/llama.cpp — native MTP speculative decoding

  Controls:
  ---------
  [6] Kill All    Stop whatever is running
  [7] Update      Check git repo for newer version
  [8] Exit

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
| **0** | **Quick Start** | `v1llama_mtp.sh --quickstart` | auto | ✅ | One-click, noobs |
| **1** | **llama.cpp** (ik_llama.cpp) | `v1llama_cpp.sh` | ~35-40 tok/s | ✅ Stable | Vision, max context (262K) |
| **2** | **DFlash** (buun fork) | `v1dflash_llama_cpp.sh` | — | ❌ | Under development |
| **3** | **vLLM** (Docker) | `v1_vllm.sh` | ~70 tok/s* | ✅ Working | Production API, tool use |
| **4** | **Lucebox DFlash** | `v1lucebox.sh` | **~104 tok/s** | ⚠️ Unstable | Fastest when stable |
| **5** | **llama.cpp MTP** (PR #22673) | `v1llama_mtp.sh` | **~100 tok/s** | ✅ Working | Fast text, native MTP |

All share `llama_models/` and port 8080. Only one runs at a time.

**Recommended:** Engine **0** (Quick Start) or **5** (full MTP dashboard).

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

## Directory Layout

```
./
├── HostLLM.sh              ← Engine picker (start here)
├── chat_templates/          ← Bundled fixed jinja templates
├── v1llama_mtp.sh          ← Engine 5 / Quick Start dashboard
├── v1llama_cpp.sh          ← Engine 1 dashboard
├── v1dflash_llama_cpp.sh   ← Engine 2 dashboard
├── v1_vllm.sh              ← Engine 3 dashboard
├── v1lucebox.sh            ← Engine 4 dashboard
├── llama_models/           ← Shared GGUF model pool
├── llama_cpp_mtp/          ← MTP build (gitignored)
├── ik_llama.cpp/           ← llama.cpp build (gitignored)
├── buun-llama-cpp/         ← DFlash build (gitignored)
├── lucebox-hub/            ← Lucebox build (gitignored)
└── vllm_models/            ← vLLM Docker + model weights (gitignored)
```

## Notes

- All builds auto-detect GPU architecture (sm_86, sm_89, sm_120, etc.) — no manual config needed.
- Multi-GPU tensor splitting is automatic — just plug in multiple NVIDIA GPUs.
- `llama_models/` is gitignored — add `.gguf` files via dashboard menus or manually.
- MTP GGUF files must be converted with the PR #22673 converter — standard GGUFs don't have MTP layers.
- Each engine tracks its own state via `.server_info*` files — the main menu auto-detects which is running.
- Use **[7] Update** from the main menu to pull the latest version from GitHub.

## License

Scripts are MIT. Engine repos have their own licenses.
