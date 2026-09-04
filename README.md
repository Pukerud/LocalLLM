# LocalLLM — Qwen3.8 Inference

Local NVIDIA-GPU launchers for the current Qwen3.8 profiles, with a general llama.cpp fallback. Only one server should use port `8080` at a time.

> **Current primary:** Qwen3.8-27B HauhauCS Q8 + vision + FastMTP, auto-scaled across all detected RTX GPUs. The current four-GPU host runs three native-262K slots with Q8 KV.

## Quick Start

```bash
git clone https://github.com/Pukerud/LocalLLM.git
cd LocalLLM
chmod +x HostLLM.sh v1*.sh
./HostLLM.sh
```

From the HostLLM menu, press **[1]** (or **[Q]**) for the Qwen3.8
llama.cpp profile menu.

```text
HostLLM — Engine Picker
  [1] Qwen3.8-27B — vision | auto-scaled native-262K slots | FastMTP + Q8 KV
  [Q] Qwen3.8 profile menu (alias for [1])
  [2] llama.cpp — general GGUF fallback

Qwen3.8 Quick Start (inside [Q]; choose by use case)
  Stable profiles:
  [1] Hauhau Q8 + native MTP | SAME model as [2] | vision | 1 slot / F16 KV / reference fallback | speed: cached result
  [2] Hauhau Q8 + FastMTP | SAME model as [1] | stable production / multi-user | vision | 3 slots / Q8 KV | speed: cached result
  [3] Flash-Next Uncensored IQ4 | DIFFERENT IQ4 model | vision | speed-first | current upstream 4cbe8b070 | speed: cached result
  Experimental:
  [4] Flash-Next IQ4 + n-gram | SAME MODEL AS [3], NOT SMARTER | may be faster or slower | speed: cached result
  [s] Run short speed tests for installed profiles
  DFlash2 is CLI-only: text-only/no vision, experimental
  [q] Cancel
```

The launcher visibly reports:

- model/projector/sidecar checksum progress, rate, and ETA;
- downloads and already-present assets;
- runtime/build status;
- a health-wait heartbeat every ten seconds while the model loads.

No full-context generation is used by the speed tests. The server may still start with its configured native context after the test context has been selected.

## HiveOS LLM miner

The retained Hauhau FastMTP profile can run as HiveOS's official custom miner,
so it appears in the HiveOS dashboard and follows the normal `miner start` /
`miner stop` lifecycle. Install it from the repository root as root:

```bash
./install-hive-llm-miner.sh
miner start
```

The custom miner launches `hauhau-q8-fastmtp` directly and deliberately leaves
`osn.service` running. It auto-detects all available GPUs and uses three native
262K slots on the current four-RTX-3090 host (two slots on the original
three-GPU layout). When OctaSpace rents the node, its normal HiveOS `miner stop`
stops the Qwen server; after the rental, `miner start` brings it back without
restarting OctaSpace. The wrapper fails closed if Docker reports an
unknown/running non-HostLLM workload and cleans up its Hive screen on startup
failure so later `miner start` calls are recoverable. It reports zero hashrate
because the process is an inference server, but its running state and GPU
telemetry remain visible in HiveOS. Remove it with `./uninstall-hive-llm-miner.sh`.

## Tested Qwen3.8 profiles

Measured on 2026-08-27–2026-09-04 using short coding and prose prompts. These are lightweight generation measurements, not full-context benchmarks; the current rows use the configured multi-GPU slot profiles.

| Profile | Coding | Story | Average | Notes |
|---|---:|---:|---:|---|
| Hauhau Q8 native MTP | 56.87 tok/s | 40.62 tok/s | **48.74 tok/s** | BF16 vision projector |
| Hauhau Q8 FastMTP | 83.50 tok/s | 43.70 tok/s | **63.60 tok/s** | current 4-GPU production; FastMTP n=4, 3 slots / 262K each / Q8 KV; 3-run medians, 2026-09-01 |
| Flash IQ4 Uncensored | 67.28 tok/s | 67.12 tok/s | **67.20 tok/s** | cygnal IQ4XS-NGQ4; current upstream 4cbe8b070, 2 slots / 262K each / Q8 K/V; menu speed test, 2026-09-04 |
| Hauhau Q8 + DFlash2 Q4 n=5 | 86.52 tok/s | 38.67 tok/s | **62.59 tok/s** | upstream master `4e97ac86`; text-only; reversed layer-device order; opt-in candidate |

The DFlash2 row is not a replacement for the vision-capable profiles. The Q4
DFlash2 drafter currently fails to process multimodal embedding chunks in this
llama.cpp build, so the opt-in profile deliberately does not load a projector.

### Uncensored Flash-Next replacement validation

The previous official `UD-IQ4_XS` Flash-Next model was tested side-by-side with
`cygnal/Qwen3.8-Flash-Next-Uncensored-IQ4XS-NGQ4-GGUF` on 2026-08-30. Both used
the same qwen4exp PR #27742 runtime, three RTX 3090 GPUs, automatic layer fitting,
Q8 K/V, one slot, and a configured native `262144`-token context. No full-context
generation was sent; the health check verified `n_ctx_slot=262144`.

| Target | Coding | Story | Average | Vision |
|---|---:|---:|---:|---|
| Previous UD-IQ4_XS | 40.96 tok/s | 40.49 tok/s | **40.72 tok/s** | passed |
| Uncensored IQ4XS-NGQ4 | 38.72 tok/s | 39.21 tok/s | **38.97 tok/s** | passed |

The new target was **95.7%** of the previous full-context decode average and its
longer vision response was **95.4%** of the previous model. It also produced a
direct technical answer to the uncensoring probe and emitted a valid `get_weather`
tool call. Its model and projector SHA-256 checks passed. The launcher now uses
this uncensored target; the previous Flash weights were removed after validation.

### 2026-09-01 historical b10731 runtime and speed update

Upstream llama.cpp build `b10731` (`0eadefebd`) was released today. It includes
qwen4exp graph/GDN changes from #27877 and #27880, the indexer-head reduction
from #28023, and the later recurrent-state rollback fix #28123. The rollback
fix is not usable by this uncensored GGUF because it exports no MTP head, but
the qwen4exp runtime changes are usable.

The same uncensored model was A/B tested on all four RTX 3090s with identical
Q8 K/V, automatic layer fitting, and short deterministic requests. The pinned
PR runtime measured approximately `46.45` tok/s code, `46.33` prose, and `44.70`
repeated JSON. `b10731` measured `66.18`, `66.17`, and `63.97` respectively,
without `CUDA_SCALE_LAUNCH_QUEUES`; this is roughly a 43% decode improvement.
`CUDA_SCALE_LAUNCH_QUEUES=4x` was also tested and produced no meaningful decode
gain on this host.

The b10731 target loaded successfully with two native-262K slots
(`--ctx-size 524288`, `n_ctx_slot=262144`) across all four GPUs. Two concurrent
text requests, vision, tool calling, and repeated JSON output all passed. The
configured two-slot profile measured `60.46` tok/s coding and `60.33` prose.

The optional `--spec ngram` / menu entry [4] uses the **same Flash IQ4 model as
[3]**; it is not a smarter or different model. It is workload-dependent
speculation: after warm-up, repeated JSON reached about `129–136` tok/s versus
`58–64` tok/s without speculation in earlier tests; code reached about
`90–145` tok/s, while prose varied from roughly `59–71` tok/s and can be
slower. The fresh current-upstream menu test measured `66.30` coding / `65.69`
story / `66.00` average versus `67.20` for normal Flash, while tool, vision,
and schema-valid JSON checks passed. It remains opt-in rather than the normal
Flash default.

### 2026-09-04 current-upstream Flash menu upgrade

The Flash menu profile is now pinned to current upstream llama.cpp commit
`4cbe8b070bb040f3b95845408f100fbf5fb746f1` instead of the older b10731
runtime. It uses a versioned runtime directory and explicitly selects CUDA
12.9 for fresh builds. The previously installed b10731 runtime remains on the
host as a rollback artifact but is no longer selected by the menu.

Before this menu update, isolated same-flag server A/B testing on all four
RTX 3090s measured approximately 486 versus 14.4 tok/s prompt processing at
512 tokens and 521 versus 13.5 tok/s at 2048 tokens, with current upstream
also decoding substantially faster. Current upstream loaded four native
262144-token slots, and short text, JSON, tool, and BF16 vision checks passed.
The menu-specific build, smoke, native-context health, and active-profile
restoration were revalidated during this upgrade; no full-context generation
was used.

### Two-user maximum-context check

On 2026-08-30, the production Q8 FastMTP model was tested with
`--parallel 2` and `--ctx-size 524288`, giving two slots with `n_ctx_slot=262144`.
The F16-KV version failed allocation on GPU2, while the Q8-KV version loaded
successfully and completed two simultaneous short requests. Peak observed usage
was approximately 22.2 GiB of 24 GiB on the fullest GPU.

On 2026-08-31, the host exposed a fourth RTX 3090. The same model loaded with
all four GPUs, `--tensor-split 1,1,1,1`, `--parallel 3`, and aggregate
`--ctx-size 786432`; llama.cpp reported three `n_ctx_slot=262144` slots and
about 22.4 GiB on the fullest GPU (74.6 GiB total) at allocation. No full-context generation was sent.
The launcher now selects FastMTP `n=4` and three slots automatically on four
or more GPUs, while retaining `n=3` and two slots on a three-GPU host. Set
`QWEN38_FASTMTP_SLOTS=2` to force the conservative two-slot mode or
`QWEN38_FASTMTP_N_MAX=3` to use the previous draft length.

Results are cached in:

```text
~/.local/state/locallm-qwen38/speed-results.tsv
```

The Qwen submenu reads that cache and displays the average beside each profile.
Run a profile-specific test with:

```bash
./v1qwen38.sh --speed-test --profile hauhau-q8-fastmtp
./v1qwen38.sh --speed-test-all
```

## Upstream master and DFlash2 validation

On 2026-08-28, upstream llama.cpp master `4e97ac86ebe2c4cb8212d98d2641ad6768810896`
was built side-by-side with CUDA Toolkit 12.9.86 and `CMAKE_CUDA_ARCHITECTURES=86`.
The existing pinned runtimes were not modified or replaced. No experimental
`top-k.cu` changes were applied.

Short 4096-token A/B checks using the same two prompts measured:

| Profile | Pinned runtime | Upstream master | Result |
|---|---:|---:|---|
| Hauhau FastMTP | 60.36 tok/s | 62.84 tok/s | +4.1%; short vision check passed |
| Previous Flash UD-IQ4_XS | 46.95 tok/s | 55.56 tok/s | +18.3%; short vision check passed |

These are lightweight single-request measurements, not full-context benchmarks.
The 60.39 tok/s FastMTP figure is the historical single-slot F16-KV baseline;
current four-GPU production uses FastMTP n=4, three slots with Q8 KV, and measures
63.60 tok/s on the updated short benchmark (the original three-GPU layout used
n=3 and two slots).

The official Q4 DFlash2 draft was downloaded from
`incoai/Qwen3.8-27B-DFlash2-GGUF` and verified with SHA-256:

```text
Qwen3.8-27B-DFlash2-Q4_K_M.gguf
18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594
```

DFlash2 was tested against the existing Hauhau Q8 target at `n_max=3` and `n_max=5`.
The original three-GPU layout used target devices `CUDA2,CUDA1,CUDA0` and places
the draft on `CUDA0`; the target output projection must be visible to the draft
scheduler. The n=3 run averaged 58.47 tok/s. The n=5 runs averaged about 64.0
tok/s with the projector loaded; the final launcher verification of the text-only profile measured 62.59 tok/s.
Acceptance was workload-dependent: coding was about 0.78–0.91 draft-token
acceptance, while the story prompt was about 0.26–0.44. Three short greedy
parity prompts matched the non-speculative target exactly. Native-262144 health
checks passed for both n=3 and n=5 without sending a long-context request.

The DFlash2 candidate is exposed only as the explicit CLI
`hauhau-q8-dflash2` profile and is intentionally hidden from the normal menu. It defaults to `n=5`, is text-only, and does not
participate in the normal `--speed-test-all` set. Use:

```bash
./v1qwen38.sh --smoke --profile hauhau-q8-dflash2
./v1qwen38.sh --speed-test --profile hauhau-q8-dflash2
./v1qwen38.sh --quickstart --profile hauhau-q8-dflash2
```

The profile defaults to `n=5`; set `QWEN38_DFLASH_N_MAX=3` to run the other
validated draft-length test. The current Hauhau FastMTP and uncensored Flash profiles remain
vision-capable production choices. DFlash2 startup logs and the short A/B results
are retained under the host's `~/.local/share/localllm-qwen38/logs/` and
`~/.local/state/locallm-qwen38-upstream-test/` directories.

## Qwen3.8 runtime details

- HauhauCS Q8_K_P GGUF with matching BF16 vision projector.
- Flash-Next Uncensored IQ4XS-NGQ4 GGUF uses isolated qwen4exp llama.cpp current upstream `4cbe8b070` (`4cbe8b070bb040f3b95845408f100fbf5fb746f1`), with the older b10731 and PR #27742 runtimes retained for rollback.
- RTX 3090 builds use CUDA architecture `sm_86`.
- Hauhau uses layer split across all detected GPUs with an equal dynamic `--tensor-split` (currently `1,1,1,1`); this host reports PHB topology and no usable peer-to-peer link. The uncensored Flash IQ4 target uses layer split with automatic fitting because its larger weights need a rebalanced placement.
- F16 KV cache is used by the single-slot native profiles; FastMTP uses `q8_0` K/V with auto-scaled slots and an aggregate context equal to `262144 × slots`, so every slot retains native 262144 context. Flash uses the same Q8 K/V policy and selects two slots on this four-GPU host.
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
  Profile:  Qwen3.8-27B HauhauCS Q8_K_P / vision / FastMTP / 3 slots / 262K each / Q8 KV
  Model:    Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
  Context:  262144 per slot  |  Slots: 3  |  KV: Q8_0  |  Speculation: FastMTP (3-token draft)
  Vision:   ON (BF16 projector)
  GPUs:     4x RTX 3090 (24 GB each)
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
  GPU 0 :   0% | VRAM: 18.3 GB / 24.0 GB (76%) | Temp: 42 degC
  GPU 1 :   0% | VRAM: 16.7 GB / 24.0 GB (69%) | Temp: 40 degC
  GPU 2 :   0% | VRAM: 17.2 GB / 24.0 GB (71%) | Temp: 42 degC
  GPU 3 :   0% | VRAM: 22.4 GB / 24.0 GB (93%) | Temp: 41 degC
  TOTAL: VRAM: 74.6 GB / 96.0 GB (77%) | GPUs: 4

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
  [1] Qwen3.8-27B        vision │ auto-scaled native 262K slots │ FastMTP + Q8 KV
      Uses all detected GPUs; 3 users + n=4 draft on 4x RTX 3090
      HauhauCS and uncensored Flash-Next profiles with cached speed results
  [Q] Qwen3.8 profile menu (alias for [1])
  [2] llama.cpp          general GGUF fallback
  [9] Kill All
  [10] Update
  [11] Exit
```

`v1llama_cpp.sh` remains available for manually running other current GGUF
models. The removed Qwen3.6-era launchers and tests are recorded below and are
no longer offered by HostLLM.

### OctaSpace coexistence

If the OctaSpace `osn.service` exists and is active, HostLLM pauses it before
launching Qwen3.8 or the general llama.cpp engine so both workloads do not
compete for the same GPUs. When the engine stops, HostLLM starts OctaSpace
again. If a launcher returns while its server is still running, OctaSpace
remains paused until **[9] Kill All** stops the engine. A small state marker
preserves this behavior if HostLLM is reopened.

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

No alternate serving engine is retained in the active configuration. Any future
engine or checkpoint experiment must remain isolated and must pass the same
short text, vision, tool, and safety checks before replacing a current profile.

BeeLlama Docker images are packaging for BeeLlama's `llama-server`, not a special
Qwen3.8 accelerator. Its preview release is rolling and does not contain the
custom Hauhau FastMTP or Flash-Next runtime used here.

## Legacy engines and tests

This section preserves the history of the removed Qwen3.6-era entries. Their source files, old model metadata, and menu routes were removed from the active checkout; Git history retains the previous implementation.

### Why Qwen3.6 was removed

The old launchers were built around Qwen3.6 model files, Qwen3.6 draft models, Qwen3.6 chat templates, or Qwen3.6-specific Docker/Genesis configurations. They did not provide a tested Qwen3.8 vision/FastMTP path on this three-3090 host. Keeping them in the main menu made their old 4090 speed claims look current, so Qwen3.8 is now the primary supported model family.

### Removed engines

| Removed entry | Historical purpose | Reason removed |
|---|---|---|
| Legacy MTP Quick Start | Qwen3.6 native MTP, no vision; advertised up to 100 tok/s on a 4090 | Superseded by the tested Qwen3.8 profiles |
| buun-llama-cpp DFlash | Qwen3.6 DFlash speculative decoding | Qwen3.6-only workflow, no vision, and the script used an `sm_89` build assumption |
| Old vLLM Docker profile | Qwen3.6 AutoRound INT4 plus Genesis patches | Removed with the obsolete Qwen3.6-specific serving path |
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

The current comparable lightweight measurements are the Qwen3.8 table above: four-GPU FastMTP n=4 averages **63.60 tok/s**, while the four-GPU two-slot Flash profile averages **60.40 tok/s** without n-gram speculation.

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
v1qwen38.sh                Qwen3.8 llama.cpp profiles, tests, and dashboard
v1llama_cpp.sh             general llama.cpp fallback
QWEN38_EXECUTION_PLAN.md   provenance and validation record
```

Generated Qwen3.8 models, runtimes, logs, and state are stored outside the repository under `/home/user/.local/`.

## Validation

Static checks:

```bash
bash -n HostLLM.sh v1qwen38.sh v1llama_cpp.sh
```

The tested Qwen3.8 path uses short text/vision smoke tests and short speed tests only. No full 262K-context generation or long benchmark is part of the normal launcher workflow.
