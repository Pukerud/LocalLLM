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
- No long video test.
- No DFlash/Lucebox/BeeLlama port for Flash-Next.
- No vLLM GGUF conversion; vLLM remains a separate future experiment with official safetensors.

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
