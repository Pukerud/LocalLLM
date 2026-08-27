# LocalLLM — Qwen3.8 Inference

Local NVIDIA-GPU launchers for the current Qwen3.8 profiles, with a general llama.cpp fallback. Only one server should use port `8080` at a time.

> **Current primary:** Qwen3.8-27B HauhauCS Q8 + vision + FastMTP, configured for native 262K context on three RTX 3090 GPUs.

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

From the HostLLM menu, press **[Q]**. The Qwen3.8 profile menu offers:

```text
Qwen3.8 Quick Start
  [1] HauhauCS Q8_K_P + BF16 vision + native MTP + 262K       | speed: cached result
  [2] HauhauCS Q8_K_P + BF16 vision + FastMTP + 262K           | speed: cached result
  [3] Flash-Next UD-IQ3_XXS + F16 vision + PR #27742          | speed: cached result
  [4] Flash-Next UD-IQ4_XS + F16 vision + Q8 KV/auto-fit + PR #27742 | speed: cached result
  [s] Run short speed tests for installed profiles
  [q] Cancel
```

The launcher visibly reports:

- model/projector/sidecar checksum progress, rate, and ETA;
- downloads and already-present assets;
- runtime/build status;
- a health-wait heartbeat every ten seconds while the model loads.

No full-context generation is used by the speed tests. The server may still start with its configured native context after the test context has been selected.

## Tested Qwen3.8 profiles

Measured on 2026-08-27 using a 4096-token context, one short coding prompt, and one short story prompt. These are lightweight single-request generation measurements, not full-context benchmarks.

| Profile | Coding | Story | Average | Notes |
|---|---:|---:|---:|---|
| Hauhau Q8 native MTP | 56.87 tok/s | 40.62 tok/s | **48.74 tok/s** | BF16 vision projector |
| Hauhau Q8 FastMTP | 73.47 tok/s | 47.31 tok/s | **60.39 tok/s** | current recommended profile |
| Flash IQ3 | 45.24 tok/s | 45.13 tok/s | **45.19 tok/s** | experimental PR #27742 |
| Flash IQ4 | 43.49 tok/s | 43.22 tok/s | **43.36 tok/s** | experimental PR #27742; native 262K uses Q8 K/V and automatic layer fitting |

Results are cached in:

```text
~/.local/state/locallm-qwen38/speed-results.tsv
```

The menu reads that cache and displays the average beside each profile. Run a profile-specific test with:

```bash
./v1qwen38.sh --speed-test --profile hauhau-q8-fastmtp
./v1qwen38.sh --speed-test-all
```

## Qwen3.8 runtime details

- HauhauCS Q8_K_P GGUF with matching BF16 vision projector.
- Flash-Next UD-IQ3_XXS and UD-IQ4_XS GGUFs use the isolated PR #27742 runtime.
- RTX 3090 builds use CUDA architecture `sm_86`.
- Hauhau and Flash IQ3 use layer split with `--tensor-split 1,1,1` because this host reports PHB topology and no usable peer-to-peer link; Flash IQ4 uses layer split with automatic fitting because its larger weights need a rebalanced placement.
- F16 KV cache is used for the native 262144-token context configuration except Flash IQ4, which uses `q8_0` K/V to fit its larger model at native context.
- FastMTP uses the publisher sidecar and its pinned qwen35-compatible patch.
- All Qwen3.8 data, runtimes, logs, and state live below:

```text
/home/user/.local/share/localllm-qwen38
/home/user/.local/state/locallm-qwen38
```

When HiveOS enters an automatic `sudo -s` shell, the launcher resolves the original `/home/user` owner so root and user shells share the same assets and server state.

## Qwen3.8 server dashboard

After Quick Start finishes, the launcher shows a live dashboard. Press **[2]** to return to the HostLLM menu while keeping the server running, **[1]** to stop it, or **[r]** to refresh.

```text
==================================================================
  QWEN3.8 SERVER RUNNING
==================================================================
  Profile:  Qwen3.8-27B HauhauCS Q8_K_P / vision / FastMTP / 262K
  Model:    Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
  Context:  262144  |  KV: F16  |  Speculation: FastMTP (3-token draft)
  Vision:   ON (BF16 projector)
  GPUs:     3x RTX 3090 (24 GB each)
  Reasoning: ON

  Connect from any device on your network:

  Chat UI:       http://192.168.1.69:8080
  API Base:      http://192.168.1.69:8080/v1
  Anthropic:     http://192.168.1.69:8080/v1/messages

  API Key: any string or blank (not required)

  OpenWebUI:       OpenAI base URL → http://192.168.1.69:8080/v1
  Pi / Codex:      OPENAI_API_BASE=http://192.168.1.69:8080/v1
  Cline / Continue: OpenAI compatible → http://192.168.1.69:8080/v1
  Anthropic SDK:   base_url → http://192.168.1.69:8080/v1
==================================================================

  Health: {"status":"ok"}
  Speed:  avg 60.39 tok/s | coding 73.47 | story 47.31
  CPU: 0%
  GPU 0 :   0% | VRAM: 15.5 GB / 24.0 GB (64%) | Temp: 48 degC
  GPU 1 :   0% | VRAM: 15.5 GB / 24.0 GB (64%) | Temp: 50 degC
  GPU 2 :   0% | VRAM: 18.2 GB / 24.0 GB (75%) | Temp: 46 degC
  TOTAL: VRAM: 49.1 GB / 72.0 GB (68%) | GPUs: 3

  [1] Stop server and return to menu
  [2] Return to menu (keep server running)
  [r] Refresh
```

Direct dashboard/status commands:

```bash
./v1qwen38.sh --status
./v1qwen38.sh --dashboard
./v1qwen38.sh --stop
```

## Current HostLLM menu

The active menu intentionally stays small:

```text
  [Q] Qwen3.8-27B       vision │ native 262K │ FastMTP
  [2] llama.cpp          general GGUF fallback
  [9] Kill All
  [10] Update
  [11] Exit
```

`v1llama_cpp.sh` remains available for manually running other current GGUF models. The removed Qwen3.6-era launchers and tests are recorded below and are no longer offered by HostLLM.

## API connections

The Qwen3.8 server exposes:

| Endpoint | Purpose |
|---|---|
| `http://IP:8080/health` | health check |
| `http://IP:8080/v1/models` | model list |
| `http://IP:8080/v1/chat/completions` | OpenAI-compatible chat |
| `http://IP:8080/v1/completions` | text completions |
| `http://IP:8080/v1/messages` | Anthropic-compatible messages |

No API key is required by the local server. Clients may still send any placeholder key such as `sk-local`.

## Hardware and safety

Validated host:

- 3× NVIDIA RTX 3090, 24 GiB each, compute capability 8.6;
- NVIDIA driver `595.91.07`;
- PHB topology with no usable P2P;
- approximately 125 GiB system RAM and no swap.

The Qwen3.8 launcher does not modify HiveOS, watchdog, miner, or driver configuration. The custom miner remains disabled; `WD_ENABLED=0`, `REBOOT_ON_ERROR=`, `MINER=`, and `MINER2=` remain unchanged in the active rig configuration.

## Future high-throughput candidates

Qwen publishes serving examples for vLLM, SGLang, and TokenSpeed. A future isolated test could use the official `Qwen/Qwen3.8-27B-FP8` safetensors checkpoint with a packaged serving engine. It must be tested on this exact 3×3090 PHB host before replacing the current profile.

BeeLlama Docker images are packaging for BeeLlama's `llama-server`, not a special Qwen3.8 accelerator. Its preview release is rolling and does not contain the custom Hauhau FastMTP or Flash-Next runtime used here.

## Legacy engines and tests

This section preserves the history of the removed Qwen3.6-era entries. Their source files, old model metadata, and menu routes were removed from the active checkout; Git history retains the previous implementation.

### Why Qwen3.6 was removed

The old launchers were built around Qwen3.6 model files, Qwen3.6 draft models, Qwen3.6 chat templates, or Qwen3.6-specific Docker/Genesis configurations. They did not provide a tested Qwen3.8 vision/FastMTP path on this three-3090 host. Keeping them in the main menu made their old 4090 speed claims look current, so Qwen3.8 is now the primary supported model family.

### Removed engines

| Removed entry | Historical purpose | Reason removed |
|---|---|---|
| Legacy MTP Quick Start | Qwen3.6 native MTP, no vision; advertised up to 100 tok/s on a 4090 | Superseded by the tested Qwen3.8 profiles |
| buun-llama-cpp DFlash | Qwen3.6 DFlash speculative decoding | Qwen3.6-only workflow, no vision, and the script used an `sm_89` build assumption |
| Old vLLM Docker profile | Qwen3.6 AutoRound INT4 plus Genesis patches | The stored compose/model setup was Qwen3.6-specific; future vLLM work should use an isolated Qwen3.8 FP8 profile |
| Lucebox DFlash | Qwen3.6 DFlash safetensors draft and DDTree | Unstable, Qwen3.6-only, and compiled with an `sm_89` assumption |
| Upstream llama.cpp MTP dashboard | Qwen3.6 model conversion with PR #22673 MTP layers | Redundant after the Qwen3.8 runtime became the supported path |
| BeeLlama DFlash dashboard | Qwen3.6 DFlash, TurboQuant/TCQ KV cache, vision, and reasoning | Useful historical fork, but not the Hauhau FastMTP or Flash-Next runtime |
| ZAYA1-8B | 8B total / approximately 760M active parameters using Zyphra's experimental vLLM fork | Small-model detour with no value for the current 27B target; installation could hard-lock the host |

### Historical tests and claims

- Legacy MTP was reported at up to approximately 100 tok/s on an RTX 4090 with no vision.
- Lucebox was reported at approximately 104 tok/s on an RTX 4090 under its Qwen3.6/DDTree settings.
- Old vLLM README figures ranged from approximately 50–127 tok/s depending on context preset.
- BeeLlama DFlash benchmarks tested Qwen3.6 target/draft combinations at roughly 100K context with TurboQuant/TCQ KV settings.
- The old benchmark scripts measured different prompts, models, contexts, KV types, and hardware. Their numbers are not directly comparable to the current Qwen3.8 smoke-speed results.

The current comparable lightweight measurements are the Qwen3.8 table above: FastMTP averaged **60.39 tok/s**, while native MTP averaged **48.74 tok/s**.

### BeeLlama preview history

BeeLlama `preview-v0.4.4` is a rolling preview based on a moving branch build. The published CUDA images included:

```text
ghcr.io/anbeeld/beellama.cpp:server-cuda-preview-v0.4.4
ghcr.io/anbeeld/beellama.cpp:server-cuda12-preview-v0.4.4
ghcr.io/anbeeld/beellama.cpp:server-cuda13-preview-v0.4.4
```

The images are convenient server packages. They do not inherently improve inference speed and were not selected for the current Qwen3.8 path.

## Repository layout

```text
HostLLM.sh                 current top-level menu
v1qwen38.sh                Qwen3.8 profiles, tests, and dashboard
v1llama_cpp.sh             general llama.cpp fallback
QWEN38_EXECUTION_PLAN.md   provenance and validation record
```

Generated Qwen3.8 models, runtimes, logs, and state are stored outside the repository under `/home/user/.local/`.

## Validation

Static checks:

```bash
bash -n HostLLM.sh v1qwen38.sh
```

The tested Qwen3.8 path uses short text/vision smoke tests and short speed tests only. No full 262K-context generation or long benchmark is part of the normal launcher workflow.
