# Qwen3.8 / Flash-Next LocalLLM execution plan

Date: 2026-08-27
Target: `/home/user/LocalLLM` on `192.168.1.69`

## Guardrails

- Do not start, modify, or enable the custom miner.
- Keep `WD_ENABLED=0` and `REBOOT_ON_ERROR=` unchanged.
- Do not change the NVIDIA driver or Hive watchdog configuration.
- Keep the general llama.cpp fallback; archive obsolete Qwen3.6-specific engines and tests after Qwen3.8 validation.
- Do not run a full 262K-context benchmark or long generation. Smoke tests use short text and one small image only.
- Stop every test server before the next test and record logs/VRAM.

## Baseline

- Three RTX 3090 GPUs, 24 GiB each, compute capability 8.6.
- GPUs are connected through PHB and report no usable P2P; Hauhau and Flash IQ3 use `--split-mode layer` with equal `--tensor-split 1,1,1`, while Flash IQ4 uses layer split with automatic fitting.
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

### Experimental Qwen3.8-Flash-Next

- Build a separate pinned PR #27742 (`qwen4exp`) runtime at head `af1ffaf37f1e44edb62e87ab8ddb9bb6840849bc` (recorded 2026-08-27).
- Use mmap/host handling for the large PLE/n-gram tables; Flash IQ3 uses F16 K/V, while Flash IQ4 uses `q8_0` K/V and automatic layer fitting at native context.
- Initial candidate: Unsloth `UD-IQ3_XXS`; `UD-IQ4_XS` is permitted only after the smaller candidate starts successfully.
- Use `ngram-mod` only after a non-speculative short text/vision smoke test.
- Do not assume an MTP head exists in the Unsloth GGUF; inspect metadata before enabling `draft-mtp`.

### Experimental SPEED DEMON

- Use the packaged vLLM `0.28.0` image with `cyankiwi/Qwen3.8-27B-AWQ-INT4` as the target and `z-lab/Qwen3.8-27B-DFlash2` as the drafter.
- Keep target tensor parallelism at 2 on CUDA0/CUDA1; TP=3 is invalid because the model's 32 attention heads are not divisible by 3. CUDA2 remains unused by this profile.
- Use FP8 KV, no LMCache, DFlash2 with seven speculative tokens, native context `262144`, and the tested FlashInfer full decode graph overlay from vLLM PR #50885.
- The target accepts image input, but the DFlash2 drafter receives text only. Label the profile as target vision enabled, DFlash draft text-only, and video unvalidated.
- Enable automatic tool choice with vLLM's `qwen3_xml` parser, matching the target chat template used by Qwen Code and Open WebUI.
- Treat approximately 144 tok/s coding, 97 tok/s agent, and 58 tok/s prose as short-test reference values, not universal or full-context throughput claims.

## LocalLLM changes

1. Add isolated runtime/profile scripts rather than mutating the old Qwen3.6 DFlash/Lucebox/BeeLlama paths.
2. Add explicit model metadata: repository, shard/include pattern, projector, checksum/manifest, context, KV type, split mode, and speculation mode.
3. Add safe dependency/build capability checks, including `sm_86`; remove the old `sm_89` assumption for this host.
4. Add short smoke-test helper using `/health`, one short text request, and one small image request.
5. Add Quick Start entries only for profiles that pass the smoke test; do not expose obsolete Qwen3.6-only engines in the active menu.
6. Add the tested SPEED DEMON launcher as a top-level HostLLM `[1]` option rather than placing it inside the Qwen submenu.
7. Update README with the model/runtime requirements, short-test policy, and legacy engine history.

## Execution/test order

1. Snapshot git/config/health baseline.
2. Write this plan into the remote LocalLLM checkout.
3. Add scripts/documentation and run static checks.
4. Build/download only the stable Hauhau Q8 target and projector first.
5. Run short text and image smoke tests, with no long-context prompt.
6. If stable, build/download the Flash-Next candidate and run the same short smoke tests. Flash downloads are lazy because the IQ3/IQ4 shard sets are large.
7. Commit changes with model/runtime provenance and push to the appropriate repository only after validation.

## Success criteria

- The selected server starts with the intended binary and model.
- `/health` returns success.
- A short text request returns a non-empty response.
- A small image request returns a non-empty response when vision is enabled.
- All three GPUs remain visible and there are no CUDA/Xid errors in the test log.
- No miner process starts and Hive safety settings remain unchanged.

## Known deferred work

- No full-context throughput/quality benchmark.
- No long video test; SPEED DEMON video behavior remains unvalidated.
- No DFlash/Lucebox/BeeLlama port for Flash-Next.
- SPEED DEMON is experimental and does not replace the production llama.cpp profile; no LMCache or third-GPU arrangement has been promoted.

## Execution record

Completed 2026-08-27:

- Added `v1qwen38.sh` and the `HostLLM.sh` `[Q]` entry without changing the old engines.
- Added this plan and a Qwen3.8 section to `README.md`.
- Installed only the build toolchain needed here: build-essential/CMake/Ninja and CUDA Toolkit 12.9.2. The NVIDIA driver stayed at 595.91.07.
- Built the Hauhau runtime at `4df29be4f4c3673f428170fda944a5b19f743bb8` with the publisher FastMTP patch.
- Built the Flash runtime at PR #27742 head `af1ffaf37f1e44edb62e87ab8ddb9bb6840849bc`.
- Downloaded and SHA-256 verified the Hauhau Q8 target, BF16 projector, FastMTP sidecar, and Flash-Next UD-IQ3_XXS shards/F16 projector.
- Short Hauhau baseline, native MTP, and FastMTP smoke tests passed for text plus vision.
- Short Flash-Next UD-IQ3_XXS baseline and `ngram-mod` smoke tests passed for text plus vision.
- Health-only native-context startup checks passed for Hauhau Q8 at 262144 and Flash UD-IQ3_XXS at 262144. No full-context prompt was sent.
- Added visible checksum/launch progress, cached short speed tests, and a post-launch dashboard with connection URLs, health, GPU state, and tok/s.
- Downloaded and SHA-256 verified Flash-Next UD-IQ4_XS (~88 GiB) after IQ3 passed; its initial F16-KV short 4096-context speed test measured 43.49 tok/s coding, 43.22 tok/s story, 43.36 tok/s average.
- Short 4096-context speed results: Hauhau native 48.74 tok/s, Hauhau FastMTP 60.39 tok/s, Flash IQ3 45.19 tok/s, Flash IQ4 48.16 tok/s with Q8 K/V and automatic layer fitting. These are cached in `speed-results.tsv`, not full-context benchmarks.
- Final checks showed the FastMTP server healthy at context 262144, no miner process, all three GPUs visible, `MINER=`/`MINER2=` empty, `WD_ENABLED=0`, and `REBOOT_ON_ERROR=` empty.
- Cleaned the active HostLLM menu and removed obsolete Qwen3.6-era launcher, benchmark, chat-template, and vLLM metadata files; the README now preserves their history and test results.

Completed 2026-08-28:

- Diagnosed Flash IQ4 native-context startup failure as CUDA OOM during KV/compute-buffer allocation; IQ3 remained healthy.
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

Completed 2026-08-28 — SPEED DEMON integration:

- Added `v1speeddemon.sh`, a side-by-side vLLM launcher using the tested AWQ target, DFlash2 drafter, native 262K context, FP8 KV, and no LMCache.
- Added a derived packaged Docker image definition containing the tested FlashInfer PR #50885 full-decode graph overlay; the stock vLLM image remains the base.
- Added short text/image smoke handling, health/status/dashboard output, persistent container logs, model download support, and separate SPEED DEMON state below `~/.local/share/localllm-speed-demon` and `~/.local/state/locallm-speed-demon`.
- Added SPEED DEMON as HostLLM top-level option `[1]`; `[Q]` continues to select the llama.cpp Qwen3.8 profiles. HostLLM recognizes and stops the vLLM container and preserves OctaSpace pause/resume behavior.
- Kept the vision description precise: the target sees image input, DFlash2 drafts text only, and video remains unvalidated. No production Qwen profile or Hive/miner setting was changed.

Completed 2026-08-29 — SPEED DEMON tool calling:

- Enabled vLLM automatic tool choice with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, fixing the 400 error returned when Qwen Code sends `tool_choice: auto`.
- Added `--reasoning-parser qwen3` and made thinking explicitly ON by default while allowing clients to override it.
- Verified non-streaming and streaming calculator tool calls, tool-result continuation, and thinking-enabled coding/vision requests against isolated vLLM containers; no tool-choice 400s occurred.
