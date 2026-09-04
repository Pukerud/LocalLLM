# Qwen3.8 LocalLLM execution plan

Date: 2026-08-27; updated 2026-09-04
Target: `/home/user/LocalLLM` on `192.168.1.69`

## Guardrails

- Do not start, modify, or enable the custom miner unless explicitly requested for a lifecycle test or deployment.
- Keep `WD_ENABLED=0` and `REBOOT_ON_ERROR=` unchanged.
- Do not change the NVIDIA driver or Hive watchdog configuration.
- Keep the general llama.cpp fallback; archive obsolete Qwen3.6-specific engines and tests after Qwen3.8 validation.
- Do not run a full 262K-context benchmark or long generation. Smoke tests use short text and one small image only.
- Stop every test server before the next test and record logs/VRAM.

## Baseline

- Four RTX 3090 GPUs, 24 GiB each, compute capability 8.6.
- GPUs are connected through PHB and report no usable P2P; Hauhau and TURBO use `--split-mode layer` with a dynamic equal `--tensor-split` across all selected GPUs.
- Host has approximately 125 GiB RAM and no swap.
- Current NVIDIA driver is 595.91.07; use a CUDA 12.9 build/runtime path if a toolkit is required.

## Model/runtime targets

### Stable Qwen3.8-27B

- HauhauCS `Q8_K_P` target GGUF, native context 262144.
- Matching HauhauCS BF16 vision projector.
- First profile: full 262144 context, F16 KV, vision, no speculation for the baseline.
- Follow-up profile: native `draft-mtp`; FastMTP sidecar only if its required qwen35 patch applies and the short smoke test passes.
- Build from the HauhauCS-pinned llama.cpp commit `4df29be4f4c3673f428170fda944a5b19f743bb8`; do not replace existing builds.
- Verify target SHA-256 `4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2`, projector SHA-256 `5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142`, and FastMTP SHA-256 `115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b`.

### Qwen3.8-27B TURBO MTP replacement

- Use `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF` revision `6408ab122`.
- Use the largest listed Q8 MTP target, `...MTP-Q8_0.gguf`, with the repository's `mmproj-BF16.gguf` projector; both are SHA-256 verified by the launcher.
- Use current-upstream llama.cpp `4cbe8b070`, CUDA Toolkit 12.9, `sm_86`, equal layer split, Q8 K/V, embedded native MTP `n=2`, and three native-262K slots on the four-3090 host.
- Keep the model card's thinking defaults: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition penalty 1.0. Use `temperature=0.6` only for precise coding requests when the client overrides sampling.
- Use the supplied BF16 vision projector and validate text, vision, tool behavior, JSON, native-context health, and clean teardown before promotion.

## LocalLLM changes

1. Add isolated runtime/profile scripts rather than mutating the old Qwen3.6 DFlash/Lucebox/BeeLlama paths.
2. Add explicit model metadata: repository, shard/include pattern, projector, checksum/manifest, context, KV type, split mode, and speculation mode.
3. Add safe dependency/build capability checks, including `sm_86`; remove the old `sm_89` assumption for this host.
4. Add short smoke-test helper using `/health`, one short text request, and one small image request.
5. Add Quick Start entries only for profiles that pass the smoke test; do not expose obsolete Qwen3.6-only engines in the active menu.
6. Update README with the model/runtime requirements, short-test policy, and legacy engine history.

## Execution/test order

1. Snapshot git/config/health baseline.
2. Write this plan into the remote LocalLLM checkout.
3. Add scripts/documentation and run static checks.
4. Build/download only the stable Hauhau Q8 target and projector first.
5. Run short text and image smoke tests, with no long-context prompt.
6. If stable, build/download the TURBO Q8 MTP target and run the same short smoke tests. Large model downloads are checksum-verified and lazy.
7. Commit changes with model/runtime provenance and push to the appropriate repository only after validation.

## Success criteria

- The selected server starts with the intended binary and model.
- `/health` returns success.
- A short text request returns a non-empty response.
- A small image request returns a non-empty response when vision is enabled.
- All four GPUs remain visible and there are no CUDA/Xid errors in the test log.
- No miner process starts and Hive safety settings remain unchanged.

## Known deferred work

- No full-context throughput/quality benchmark.
- No long video test; video behavior remains unvalidated.
- No DFlash/Lucebox/BeeLlama port is planned for the TURBO target; its embedded MTP path is the supported speculation mode.

## Execution record

Completed 2026-08-27:

- Added `v1qwen38.sh` and the `HostLLM.sh` `[Q]` entry without changing the old engines.
- Added this plan and a Qwen3.8 section to `README.md`.
- Installed only the build toolchain needed here: build-essential/CMake/Ninja and CUDA Toolkit 12.9.2. The NVIDIA driver stayed at 595.91.07.
- Built the Hauhau runtime at `4df29be4f4c3673f428170fda944a5b19f743bb8` with the publisher FastMTP patch.
- Built the Flash runtime at PR #27742 head `af1ffaf37f1e44edb62e87ab8ddb9bb6840849bc`.
- Downloaded and SHA-256 verified the Hauhau Q8 target, BF16 projector, FastMTP sidecar, and the retained Flash-Next IQ4 shards/F16 projector.
- Short Hauhau baseline, native MTP, and FastMTP smoke tests passed for text plus vision.
- Short Flash-Next IQ4 baseline and `ngram-mod` smoke tests passed for text plus vision.
- Health-only native-context startup checks passed for Hauhau Q8 and Flash IQ4 at 262144. No full-context prompt was sent.
- Added visible checksum/launch progress, cached short speed tests, and a post-launch dashboard with connection URLs, health, GPU state, and tok/s.
- Downloaded and SHA-256 verified the retained Flash-Next UD-IQ4_XS (~88 GiB); its initial F16-KV short 4096-context speed test measured 43.49 tok/s coding, 43.22 tok/s story, 43.36 tok/s average.
- Short 4096-context speed results: Hauhau native 48.74 tok/s, Hauhau FastMTP 60.39 tok/s, and Flash IQ4 48.16 tok/s with Q8 K/V and automatic layer fitting. These are cached in `speed-results.tsv`, not full-context benchmarks.
- Final checks showed the FastMTP server healthy at context 262144, no miner process, all three GPUs visible, `MINER=`/`MINER2=` empty, `WD_ENABLED=0`, and `REBOOT_ON_ERROR=` empty.
- Cleaned the active HostLLM menu and removed obsolete Qwen3.6-era launcher, benchmark, chat-template, and vLLM metadata files; the README now preserves their history and test results.

Completed 2026-08-28:

- Diagnosed Flash IQ4 native-context startup failure as CUDA OOM during KV/compute-buffer allocation, then verified the retained Q8-KV/auto-fit configuration.
- Verified Flash IQ4 at native 262144 with `q8_0` K/V, automatic layer fitting (no explicit tensor split), original batch/ubatch 512/128, all three GPUs, and F16 vision.
- Short text and 64x64 image requests passed; the temporary server was stopped after testing. No full-context generation was run and no Hive/miner settings changed.
- Reran `./v1qwen38.sh --speed-test --profile flash-iq4`: coding 48.08 tok/s, story 48.25 tok/s, average 48.16 tok/s; the launcher stopped the test server cleanly.

Completed 2026-08-28 — upstream master / DFlash2 evaluation:

- Preserved the production runtimes at `/home/user/.local/share/localllm-qwen38/runtimes/llama-qwen38-hauhau` and `llama-qwen4exp-pr27742` unchanged.
- Built upstream llama.cpp master `4e97ac86ebe2c4cb8212d98d2641ad6768810896` side-by-side at `runtimes/llama-upstream-master-4e97ac86`, explicitly using CUDA Toolkit 12.9.86 and `CMAKE_CUDA_ARCHITECTURES=86`. The binary SHA-256 is `b70d9b8cd76e34f0b57b8f9a74621695bd1187b2e3b2a9058423dee520462d3f`.
- Applied only the existing additive Hauhau FastMTP compatibility patch to the side-by-side source; no experimental reverted `top-k.cu` patch was applied.
- Repeated short A/B checks at context 4096: pinned Hauhau FastMTP 60.36 tok/s versus upstream 62.84 tok/s; pinned Flash IQ4 46.95 tok/s versus upstream 55.56 tok/s. Upstream short vision checks passed for both.
- Downloaded only `incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,143,006,752 bytes) and verified SHA-256 `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`.
- Tested DFlash2 Q4 against the existing Hauhau Q8 target at `n_max=3` and `n_max=5`. The working layout reverses target devices to `CUDA2,CUDA1,CUDA0` and places the draft on `CUDA0`, so the shared target output projection is schedulable. n=3 averaged 58.47 tok/s; n=5 averaged approximately 64.0 tok/s with the projector loaded. The final launcher verification of the text-only profile measured 62.59 tok/s.
- DFlash2 acceptance was workload-dependent: coding was approximately 0.78–0.91 and story approximately 0.26–0.44 by accepted/generated draft tokens. Three short greedy parity prompts matched the non-speculative target exactly.
- Native-context health-only startup passed at 262144 for both n=3 and n=5. No full-context generation was sent.
- DFlash2 vision-first tests failed in the draft context with `failed to initialize batch` / `failed to decode mtmd chunk`; the opt-in `hauhau-q8-dflash2` profile therefore deliberately omits the projector and is text-only. It is not a replacement for the vision-capable production profiles and is excluded from normal `--speed-test-all`.
- Added the explicit `hauhau-q8-dflash2` profile with default `n=5`, checksum-verified asset download, reversed device placement, text-only smoke handling, and dashboard/menu labeling. Existing Hauhau FastMTP and Flash IQ4 defaults remain unchanged.

Completed 2026-08-30 — uncensored Flash-Next replacement:

- Downloaded `cygnal/Qwen3.8-Flash-Next-Uncensored-IQ4XS-NGQ4-GGUF` beside the previous `unsloth` UD-IQ4_XS target and verified the GGUF SHA-256 `cedf1e08063f6df77926e1169f67b327dcc6301b5b329589615bdf09d4895f7e` plus BF16 projector SHA-256 `9c56f5aa2d30242325a91aa3e4c03348e9944648f4af6692a7a86db93aae7ffa`.
- Side-by-side full-context allocation and short-generation tests used the same qwen4exp PR #27742 runtime, Q8 K/V, automatic layer fitting, one slot, and `n_ctx_slot=262144`. Both targets passed text and vision.
- At the configured native context, the previous target averaged `40.72` tok/s and the uncensored target averaged `38.97` tok/s (`95.7%`), with longer vision decode at `39.33` versus `37.53` tok/s (`95.4%`).
- The uncensored target returned a direct technical answer to the refusal probe and emitted a valid `get_weather` tool call. The standard 4096-token launcher-style test averaged `47.30` tok/s.
- Promoted the uncensored target as the `flash-iq4` model, removed the previous UD-IQ4_XS weights/projector, and retained the tested qwen4exp runtime. No full-context generation was sent.

Completed 2026-08-31 — four-GPU scaling:

- The worker now exposes four identical RTX 3090 GPUs (`CUDA0`–`CUDA3`, 96 GiB total). All devices report PHB links with no usable peer-to-peer path.
- Hauhau FastMTP allocation passed with all four GPUs, dynamic `--tensor-split 1,1,1,1`, `--parallel 3`, aggregate `--ctx-size 786432`, Q8 K/V, and three `n_ctx_slot=262144` slots. No full-context generation was sent.
- Three-run short-request medians on the 4-GPU/3-slot production server were `70.78` tok/s coding and `46.39` tok/s story, averaging `58.59` tok/s; these were short requests with thinking disabled and are not full-context measurements.
- Updated the launcher to discover/select GPUs, generate device/split arguments, and automatically use three slots on four or more GPUs while retaining two slots on three GPUs. `QWEN38_FASTMTP_SLOTS=2` remains the conservative override.
- Improved HostLLM and Qwen dashboards/menu labels to show the detected GPU inventory.
- Improved the Hive wrapper's exit cleanup so a failed/stopped custom miner does not leave an empty screen that blocks the next `miner start`. The custom miner remains `MINER=custom` with `osn.service` untouched.

Completed 2026-09-01 — upstream/runtime speed research:

- Checked current llama.cpp master and built `b10731` (`0eadefebd`) with CUDA 12.9 / sm_86 in a side-by-side runtime. The qwen4exp model loaded the uncensored GGUF with one and two native-262K slots across all four RTX 3090s; text, vision, tool calling, and JSON validation passed.
- Compared the current pinned qwen4exp runtime with b10731 under identical four-GPU Q8-KV settings. No-spec decode improved from roughly 46 tok/s to 66 tok/s on short code/prose requests; `CUDA_SCALE_LAUNCH_QUEUES=4x` did not materially change decode speed.
- Tested `ngram-mod` on the uncensored Flash GGUF. It is lossless target-verified speculation but workload-dependent: repeated JSON reached roughly 129–136 tok/s, code 90–145 tok/s after warm-up, and prose ranged from 59–71 tok/s. It remains opt-in.
- Tested FastMTP draft lengths n=3, n=4, and n=5 on the four-GPU Hauhau profile. n=4 gave the best balanced short result (83.50 code / 43.70 story / 63.60 average) and was promoted; the production profile remains three native-262K slots.
- Tested Flash with two native-262K slots on b10731. Allocation, two concurrent text requests, vision, JSON, and tool calls passed. The default Flash profile now uses two slots; `QWEN38_FLASH_SLOTS=1` remains available for maximum single-stream speed.

Completed 2026-09-01 — engine retirement and Flash follow-up experiments:

- Removed the alternate vLLM engine, its launcher, Docker overlays, menu routes, model metadata, and documentation. Deleted its remote container images, model snapshots, cache, state, and logs; the general llama.cpp fallback and Qwen profiles remain.
- Built an isolated CUDA/sm_86 llama.cpp runtime from current upstream plus PR #27941 and PR #27977 (`d24c7ae73`, build 10747). Unit architecture checks passed; short no-speculation, n-gram, two-slot concurrency, and n-gram vision checks passed. The short decode A/B was effectively neutral, and a long-context A/B was not completed because an approximately 8.4K-token prefill already took about 503 seconds on this host; no long-context speed claim was promoted.
- Tested the `dzannotti/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` sidecar with the current uncensored Flash target using the experimental PR #28104 MTP stack and the merged recurrent rollback code. Native 262K one-slot allocation required `--fit-target 4096`; at that margin the text server loaded and achieved approximately 96 code / 48 prose / 102 JSON tok/s at n=3, with draft acceptance approximately 1.00 / 0.81 / 1.00. The target vision request completed, but draft acceptance was zero.
- Greedy code and JSON matched the no-draft target, while the prose continuation diverged reproducibly even at n=1. The sidecar was therefore rejected for this uncensored target and deleted; MTP remains disabled and the existing n-gram profile is retained.

Completed 2026-09-04 — current-upstream Flash menu upgrade:

- Replaced the Flash menu profile's b10731 pin with current upstream `4cbe8b070bb040f3b95845408f100fbf5fb746f1` and moved it to the versioned `llama-qwen4exp-upstream-4cbe8b07` runtime directory. Fresh Flash builds explicitly use CUDA Toolkit 12.9 and `sm_86`; the old b10731 runtime remains available for rollback.
- Same-flag isolated testing measured approximately 486 versus 14.4 tok/s prompt processing at 512 tokens and 521 versus 13.5 tok/s at 2048 tokens compared with b10731. Current upstream loaded four native 262144-token slots and passed short text, JSON, tool, and BF16 vision checks.
- Stopped the miner and `osn.service` for the deployment window, then verified the menu's download/build/smoke path, native-context health, and Flash API checks. The menu speed test measured `67.28` coding / `67.12` story / `67.20` average tok/s. Restored the existing Hauhau FastMTP profile afterward; `/health`, thinking, tool behavior, and the normal Hive custom-miner lifecycle passed.

Completed 2026-09-04 — menu clarity and n-gram validation:

- Clarified the choices by use case: [1] is the one-slot F16-KV Hauhau reference/fallback, [2] is the stable multi-user Hauhau FastMTP production profile, [3] is the different uncensored Flash IQ4 model, and [4] is the same Flash model as [3] with experimental speculation rather than a smarter model.
- Removed DFlash2 from the normal numbered menu. The text-only experiment remains available explicitly as `--profile hauhau-q8-dflash2`.
- Retested [4] on current upstream: `66.30` coding / `65.69` story / `66.00` average tok/s versus `67.20` normal Flash. Tool calling, BF16 vision, and schema-valid JSON passed, so n-gram remains an opt-in workload-specific experiment.

Completed 2026-09-04 — TURBO Q8 MTP replacement:

- Downloaded `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF` repository revision `6408ab122688c54ba5b7cea19084307ef153410f`.
- Selected the largest listed Q8 MTP target, `Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf` (30,239,020,576 bytes), and `mmproj-BF16.gguf` (931,145,920 bytes). SHA-256 verification passed: `54f27515edb20675f289f99b9c6d40d114fb634db21bae3fd4c901661aba85b9` and `b0d8d89e9c9c90e0fb8ca74742d9d9bd7cc0f966a29b6f8c14227000ea6bd89e`.
- Built an isolated current-upstream llama.cpp runtime from `4cbe8b070bb040f3b95845408f100fbf5fb746f1` with CUDA Toolkit 12.9.86 and `CMAKE_CUDA_ARCHITECTURES=86`.
- Validated embedded native MTP at `n=2`, equal four-GPU layer split, Q8 K/V, BF16 vision, three slots, and `n_ctx_slot=262144` without sending a long-context request. Text, tool calling, schema-valid JSON, vision, and launcher smoke/teardown passed.
- The short menu speed test measured `58.13` coding / `42.23` story / `50.18` average tok/s at 4096-token smoke context. This is a lightweight generation result, not a full-context benchmark.
- Removed the retired Flash model/projector from the host and replaced its numbered menu entry with `turbo-q8-mtp`; the DFlash2 profile remains explicit CLI-only.

Completed 2026-09-04 — Q4-KV Hauhau production promotion:

- Promoted menu option [4], `hauhau-q8-fastmtp-q4kv-xhigh`, to the default Qwen3.8 profile after the latest operator comparison found better output and speed.
- The production profile keeps FastMTP n=4, three native-262K slots, BF16 vision, and maximum supported `xhigh` reasoning while changing both K and V caches to `q4_0`.
- Updated the Hive custom miner default and launcher default. `hauhau-q8-fastmtp` remains available as the Q8-KV fallback; TURBO and DFlash2 settings remain unchanged.