# LocalLLM — Qwen3.8 Inference

Local NVIDIA-GPU launchers for the current Qwen3.8 profiles, with ExLlamaV3 and a general llama.cpp fallback. Only one server should use port `8080` at a time.

> **Current primary:** Qwen3.8-27B HauhauCS Q8 + vision + FastMTP, configured for native 262K context on three RTX 3090 GPUs.

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

From the HostLLM menu, press **[1]** for the direct SPEED DEMON profile,
**[Q]** for the Qwen3.8 llama.cpp profile menu, **[2]** for ExLlamaV3, or
**[3]** for the general llama.cpp fallback.

```text
HostLLM — Engine Picker
  [1] SPEED DEMON — Qwen3.8 AWQ INT4 + FP8 DFlash2 | ~123 code* / ~67 tools / ~62 prose tok/s
      native 262K | 2x RTX 3090 | image input + auto tools + thinking ON; FP8 draft text-only; video unvalidated
  [Q] Qwen3.8-27B — vision | native 262K | FastMTP
  [2] ExLlamaV3 — Qwen3.8 EXL3 6bpw + vision | native 262K | speed: cached result
  [3] llama.cpp — general GGUF fallback

Qwen3.8 Quick Start (inside [Q])
  [1] HauhauCS Q8_K_P + BF16 vision + native MTP + 262K       | speed: cached result
  [2] HauhauCS Q8_K_P + BF16 vision + FastMTP + 262K           | speed: cached result
  [3] Flash-Next UD-IQ3_XXS + F16 vision + PR #27742          | speed: cached result
  [4] Flash-Next UD-IQ4_XS + F16 vision + Q8 KV/auto-fit + PR #27742 | speed: cached result
  [5] HauhauCS Q8_K_P + DFlash2 Q4 n=5 (text-only, experimental) | speed: cached result
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

Measured on 2026-08-27–28 using a 4096-token context, one short coding prompt, and one short story prompt. These are lightweight single-request generation measurements, not full-context benchmarks.

| Profile | Coding | Story | Average | Notes |
|---|---:|---:|---:|---|
| Hauhau Q8 native MTP | 56.87 tok/s | 40.62 tok/s | **48.74 tok/s** | BF16 vision projector |
| Hauhau Q8 FastMTP | 73.47 tok/s | 47.31 tok/s | **60.39 tok/s** | current recommended profile |
| Flash IQ3 | 45.24 tok/s | 45.13 tok/s | **45.19 tok/s** | experimental PR #27742 |
| Flash IQ4 | 48.08 tok/s | 48.25 tok/s | **48.16 tok/s** | experimental PR #27742; Q8 K/V and automatic layer fitting, remeasured 2026-08-28 |
| Hauhau Q8 + DFlash2 Q4 n=5 | 86.52 tok/s | 38.67 tok/s | **62.59 tok/s** | upstream master `4e97ac86`; text-only; reversed layer-device order; opt-in candidate |

The DFlash2 row is not a replacement for the vision-capable FastMTP profile. The Q4
DFlash2 drafter currently fails to process multimodal embedding chunks in this
llama.cpp build, so the opt-in profile deliberately does not load a projector.

## SPEED DEMON profile

SPEED DEMON is a separate vLLM Docker profile for fast text and coding work. It
uses the standard (not Hauhau-abliterated) `cyankiwi/Qwen3.8-27B-AWQ-INT4`
target and the tested `TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm` draft by
default. Set `SPEED_DEMON_DRAFT_MODE=bf16` to use the retained BF16 DFlash2
fallback.

| Item | Configuration |
|---|---|
| Runtime | vLLM `0.28.0` + FlashInfer full decode graph overlay from PR #50885 + FP8 DFlash support from PR #53122 |
| GPUs | CUDA0 and CUDA1, 2x RTX 3090; CUDA2 remains unused |
| Context | native `262144` configured context |
| KV/cache | FP8 KV; no LMCache |
| Draft | `TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm`, seven speculative tokens |
| Tools | automatic tool choice with vLLM `qwen3_xml` parser |
| Reasoning | ON by default with vLLM `qwen3` parser; clients may override |
| Measured speed | approximately 123 tok/s mixed coding*, 67 tok/s tools, 62 tok/s prose |
| Model ID | `speed-demon` |

The target model supports image input and the short image smoke test passed. The
FP8 DFlash2 drafter receives text only, not image/video embeddings, so image
answers are still verified by the target but speculative acceptance may be lower.
Video support has not been validated. The profile is therefore labeled **target
vision ON; FP8 DFlash draft text-only; video unvalidated**, with text/code
recommended. The `*` speed figure is a prompt-dependent mixed short-test
reference; easy coding samples reached approximately 121–145 tok/s.

Automatic tool choice is enabled for Qwen Code and Open WebUI using the vLLM
`qwen3_xml` parser, matching this model's `<tool_call><function=...>` template.
Reasoning is ON by default and is parsed with `qwen3`; clients may explicitly
override the thinking setting. Clients should send the normal OpenAI `tools`
payload with `tool_choice: auto`.

SPEED DEMON starts on the normal HostLLM port `8080` and is mutually exclusive
with the llama.cpp engines. HostLLM pauses OctaSpace before starting it and
restores OctaSpace after the engine is stopped. The vLLM image and model assets
are stored outside the repository under:

```text
~/.local/share/localllm-speed-demon
~/.local/state/locallm-speed-demon
```

The direct launcher commands are:

```bash
./v1speeddemon.sh --quickstart   # interactive terminals open the live info dashboard
./v1speeddemon.sh --smoke
./v1speeddemon.sh --status
# Optional retained BF16 drafter fallback:
SPEED_DEMON_DRAFT_MODE=bf16 ./v1speeddemon.sh --quickstart
./v1speeddemon.sh --stop
```

The smoke test uses a short text request and one small image only. It does not
send a long-context generation.

Results are cached in:

```text
~/.local/state/locallm-qwen38/speed-results.tsv
```

The menu reads that cache and displays the average beside each profile. Run a profile-specific test with:

```bash
./v1qwen38.sh --speed-test --profile hauhau-q8-fastmtp
./v1qwen38.sh --speed-test-all
```

## ExLlamaV3 option 2

Option 2 is the highest self-calibrated Qwen3.8 EXL3 vision variant:
`turboderp/Qwen3.8-27B-exl3` revision `SC_6.00bpw_H6_V6`. It contains a 6-bit
text model and a 6-bit vision tower, with the model's native `262144` context.
The model files occupy approximately 21.1 GiB across three safetensors shards.
The 1M context mentioned by the base model card is an optional YaRN extension,
not the native setting used here.

| Item | Configuration |
|---|---|
| Runtime | ExLlamaV3 `1.4.4` + TabbyAPI, isolated Docker image |
| Model | `turboderp/Qwen3.8-27B-exl3` / `SC_6.00bpw_H6_V6` |
| Context | native `262144`; `--quickstart` uses the full configured cache |
| KV/cache | 8-bit K/V by default (`EXLLAMA_CACHE_MODE=Q6` or `Q4` is available) |
| GPUs | autosplit across visible RTX 30-series GPUs; the test used CUDA0/CUDA1 and left CUDA2 free |
| Draft | none; this is a quality/vision/context option, not a DFlash2 speed profile |
| API | TabbyAPI OpenAI-compatible server on port `8080` |

The isolated validation passed model load/health at native 262K configuration,
short text, image input (red on the left and blue on the right), automatic
calculator tool calling, tool-result continuation, and a one-token prefill with
a 259,161-token templated prompt. No long generation was run. The near-native
prefill reached the practical prompt limit imposed by TabbyAPI's cache-chunk
reservation without a CUDA/Xid/runtime error.

Use the launcher directly with:

```bash
./v1exllama.sh --quickstart   # interactive terminals open the live info dashboard
./v1exllama.sh --smoke       # short 4096-token text/image/tool smoke test
./v1exllama.sh --speed-test  # quick 4096-token coding/story measurement
./v1exllama.sh --status
./v1exllama.sh --stop
```

The launcher downloads only the pinned revision, verifies all three large
shards, builds the pinned TabbyAPI/ExLlamaV3 image if needed, and stores all
assets/state below `~/.local/share/locallm-exllama` and
`~/.local/state/locallm-exllama`. Video metadata is present, but long-video
inference remains unvalidated.

## Upstream master and DFlash2 validation

On 2026-08-28, upstream llama.cpp master `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
was built side-by-side with CUDA Toolkit 12.9.86 and `CMAKE_CUDA_ARCHITECTURES=86`.
The existing pinned runtimes were not modified or replaced. No experimental
`top-k.cu` changes were applied.

Short 4096-token A/B checks using the same two prompts measured:

| Profile | Pinned runtime | Upstream master | Result |
|---|---:|---:|---|
| Hauhau FastMTP | 60.36 tok/s | 62.84 tok/s | +4.1%; short vision check passed |
| Flash IQ4 | 46.95 tok/s | 55.56 tok/s | +18.3%; short vision check passed |

These are lightweight single-request measurements, not full-context benchmarks.
The current production FastMTP runtime remains pinned and its historical 60.39
tok/s result remains the production baseline.

The official Q4 DFlash2 draft was downloaded from
`incoai/Qwen3.8-27B-DFlash2-GGUF` and verified with SHA-256:

```text
Qwen3.8-27B-DFlash2-Q4_K_M.gguf
18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594
```

DFlash2 was tested against the existing Hauhau Q8 target at `n_max=3` and `n_max=5`.
The working three-GPU layout uses target devices `CUDA2,CUDA1,CUDA0` and places
the draft on `CUDA0`; the target output projection must be visible to the draft
scheduler. The n=3 run averaged 58.47 tok/s. The n=5 runs averaged about 64.0
tok/s with the projector loaded; the final launcher verification of the text-only profile measured 62.59 tok/s.
Acceptance was workload-dependent: coding was about 0.78–0.91 draft-token
acceptance, while the story prompt was about 0.26–0.44. Three short greedy
parity prompts matched the non-speculative target exactly. Native-262144 health
checks passed for both n=3 and n=5 without sending a long-context request.

The DFlash2 candidate is exposed only as the explicit
`hauhau-q8-dflash2` profile. It defaults to `n=5`, is text-only, and does not
participate in the normal `--speed-test-all` set. Use:

```bash
./v1qwen38.sh --smoke --profile hauhau-q8-dflash2
./v1qwen38.sh --speed-test --profile hauhau-q8-dflash2
./v1qwen38.sh --quickstart --profile hauhau-q8-dflash2
```

The profile defaults to `n=5`; set `QWEN38_DFLASH_N_MAX=3` to run the other
validated draft-length test. The current Hauhau FastMTP and Flash IQ4 profiles remain unchanged and remain the
vision-capable production choices. DFlash2 startup logs and the short A/B results
are retained under the host's `~/.local/share/localllm-qwen38/logs/` and
`~/.local/state/locallm-qwen38-upstream-test/` directories.

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
  [1] SPEED DEMON        Qwen3.8 AWQ INT4 + FP8 DFlash2 │ ~123 code* / ~67 tools / ~62 prose tok/s
      target image input ON │ FP8 DFlash draft text-only │ video unvalidated │ native 262K │ 2x RTX 3090
  [Q] Qwen3.8-27B        vision │ native 262K │ FastMTP
  [2] ExLlamaV3          6bpw EXL3 vision │ native 262K │ TabbyAPI │ speed: cached result
  [3] llama.cpp          general GGUF fallback
  [9] Kill All
  [10] Update
  [11] Exit
```

`v1llama_cpp.sh` remains available for manually running other current GGUF models. The removed Qwen3.6-era launchers and tests are recorded below and are no longer offered by HostLLM.

### OctaSpace coexistence

If the OctaSpace `osn.service` exists and is active, HostLLM pauses it before launching SPEED DEMON, Qwen3.8, ExLlamaV3, or the general llama.cpp engine so both workloads do not compete for the same GPUs. When the engine stops, HostLLM starts OctaSpace again. If a launcher returns while its server is still running, OctaSpace remains paused until **[9] Kill All** stops the engine. A small state marker preserves this behavior if HostLLM is reopened.

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
v1speeddemon.sh            SPEED DEMON vLLM/DFlash2 profile
v1exllama.sh               Qwen3.8 EXL3/ExLlamaV3 + TabbyAPI profile
v1qwen38.sh                Qwen3.8 llama.cpp profiles, tests, and dashboard
v1llama_cpp.sh             general llama.cpp fallback
speed-demon/               pinned SPEED DEMON container overlay
exllama-v3/                pinned TabbyAPI/ExLlamaV3 image overlay
QWEN38_EXECUTION_PLAN.md   provenance and validation record
```

Generated Qwen3.8 models, runtimes, logs, and state are stored outside the repository under `/home/user/.local/`.

## Validation

Static checks:

```bash
bash -n HostLLM.sh v1qwen38.sh v1speeddemon.sh v1exllama.sh
```

The tested Qwen3.8 path uses short text/vision smoke tests and short speed tests only. The ExLlamaV3 validation additionally used a one-token near-native-context prefill; no long generation or long benchmark is part of the normal launcher workflow.
