# DeepSeek V4.1 Flash 1017 — isolated Abiray IQ2_XXS profile

**New routes:** HostLLM **[6]** (direct hosting) and **[16]** (Hive). The prior SC117 .46 Hive route moves to **[17]**; its low-level Hive selector remains [5]. Older profiles, Qwen runtimes and default selection are retained. This is a separate DeepSeek engine upgrade, not promotion of the existing .46 Qwen candidate.

## Exact assets and runtime

- Profile: `deepseek41-abiray-iq2_xxs`.
- Model: [`Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF`](https://huggingface.co/Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF), revision `99acb20f7c83a37d7f0ea6082e924f3d61615d72`. All four original baked-weight shards and separate IQ4_NL Engram table; **no runtime LoRA or requantization**.
- Original BF16 vision tower, stock/non-abliterated DSpark Q8_0 drafter and Engram metadata shard: `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`, revision `f924b94f19c1fea02c06a65c8475b0a45e2b88b1`. The stock drafter is a separate component, not an abliterated MTP head.
- Total pinned inputs: **258,131,348,749 bytes**. Sizes, revisions and full SHA-256 values live in `deepseek41_assets.py`; downloads resume and verify before promotion. Packs reference these files in place without copying the large weights.
- App/control source **0.2.6**, commit `4b9e05a2a806a3fe24f7b2560d98626c37dade35`; separate **DeepSeek engine 0.1.51**, official signed `strata-dsv41-0.1.51-linux-x86_64-cuda-cuda13.tar.gz`, SHA-256 `88c9c54ecc3ffce6f2ddbef281a7066146694be85e4ace99d31e5cdd691f0603`, native build ID `562a0c2ed`.
- Upstream's native protocol still reports `INFO engine=ds4-0.1`; this legacy API label is not the package version. The signed binary's `--version` is **0.1.51**, build `562a0c2ed`. The launcher accepts that exact telemetry mapping only for this pinned DeepSeek route; older Qwen version checks are unchanged.
- Package signature and every package file checked with the pinned upstream verifier. Bundled private CUDA 13 runtime libraries; **no driver/system CUDA toolkit replacement**. Existing CUDA 12.9 builds remain untouched.
- Separate runtime: `~/.local/share/localllm-strata/runtimes/dsv41-0.1.51/`.
- Model/pack: `~/.local/share/localllm-strata/profiles/deepseek41-abiray-iq2_xxs/`.
- Separate private live state: `~/.local/state/locallm-strata/deepseek41-abiray/`. The immutable shipped hot-expert seed stays in the pack; `--expert-profile` points to its separate private adaptive copy in the state directory, so normal engine shutdown never mutates a checksum-bound pack input. Conversation slots are isolated there too.

## Native context, quality and vision

The base model's native window is **1,048,576 tokens** (reference config pin `1bae0e1088993b9b82de08f59de5be5a34e42170`). This profile explicitly requests that native context, not upstream setup's tested 8192-token default. It retains the model's native positional geometry, **six experts per token** (not the faster reduced-five-expert option), one serving slot, stock DSpark `auto` (or the explicitly selected measured no-drafter preset), all four GPUs, and temperature 1.0 / top-p 1.0 (the engine's supported native top-k default, 64). No output cap is silently substituted for a context-size configuration, and `fit_max_tokens=false` refuses overlong requests instead of truncating them.

A native window configured/allocated at startup does **not** establish populated-1M capacity, retrieval/quality at that length, sustained load, multi-user capacity or a speed gain. These require separate tests. The preset refuses a smaller-than-four-24GB-GPU host or less than 64 GiB available RAM at launch.

### Version-pinned HTTP vision adapter

The release has native DeepSeek BF16 vision kernels and `GENI`/`DSI1` support, but its standard Python HTTP `Service.prepare` still uses the Qwen image-embedding protocol. Upstream setup therefore advertises DeepSeek as text-only. This profile deliberately does not pretend a Qwen projector works for DeepSeek.

`deepseek41_native_vision.py` connects only this isolated .51 profile to the unmodified native engine's image path. It resizes RGB on the CPU using DeepSeek's reference patch/merge geometry and padded image layout, expands token 129264 into the exact image span, and writes a private, per-request **DSI1 pixel file**. The model's own BF16 tower on the first GPU encodes it. Up to 1024 tokens/image and 16 images/request; CPU preprocessing is not a CPU vision encoder. Text-only requests use the original path. Missing native vision refuses startup, never a silent text-only fallback.

Only three hook lines are added to the isolated release's `serve/server.py`:

```python
    if cfg.get("arch") == "deepseek41" and cfg.get("deepseek_native_vision") is True:
        from serve.deepseek41_native_vision import enable
        enable(svc)
```

Original whole-file SHA-256: `c9e6d3e382fa521f8469d60c28a701db4417da5dcadea9a131b781fc6d8bf695`. Patched SHA-256: `e052a44346ed56972f4a23fafc2bf739411788edbeb4cb95fa5f9544eeb82019`. Readiness checks the exact source commit/diff/hash, helper, package artifacts, private dependencies, pinned model records, pack/tokenizer and canonical config. No older source contract is relaxed.

## Several agents / conversation cache

DeepSeek **0.1.51 does not accept** Qwen's `--conversation-cache-mib 8192` or `--conversation-cache-slots` arguments; forwarding them stops native startup. The unpatched 0.2.6 web UI's generic `docs/DETAILS.md` tooltip is not a DeepSeek cache-status test. Its own native cache is already enabled by upstream's frontend; this profile makes the requested policy explicit:

```json
"env": {
  "STRATA_DS4_SLOT_KEEP": "1",
  "STRATA_DS4_SLOT_CAP": "4",
  "STRATA_DS4_SLOT_GB": "8",
  "STRATA_DS4_REUSE": "1",
  "STRATA_DS4_SLOT_DIR": ".../deepseek41-abiray/ds4-slots"
}
```

- **Four RAM-held conversation snapshots**, then LRU spill to the separate private folder. This is a count limit, **not an 8192 MiB RAM-byte limit**. Memory use depends on history length.
- **8192 MiB (8 GiB) persisted checkpoint-payload target**, not an extra VRAM allocation. Native disk LRU trimming excludes the newest single snapshot, so it is not a hard filesystem quota.
- Prefix/checkpoint reuse and reload after restart are native. The upstream default minimum for parking is 1024 tokens and expiry is seven days; smaller side requests do not evict expensive histories.
- Parked multi-history snapshots are **text-only** in this release. Vision remains available, but an image history displaced by another agent is not parked/restored by this cache.
- Four history snapshots are **not four concurrent serving slots**: serving remains one slot/group, six experts, native context 1048576.

### Native Monitor telemetry (2026-10-26 fix)

The original Monitor inferred `enabled=false` from missing **Qwen** `INFO conversation_cache_mib`, even while DeepSeek was restoring histories. `deepseek41_cache_monitor.py` now wraps only this DeepSeek service's metrics: it reads the actual native process spawn environment and incrementally parses bounded `ds4 slots:` stderr events. Real API reuse counters are preserved. Unknown policy is shown as unknown, runtime folder failures are reported, and incomplete/truncated event history is labeled observed rather than invented totals.

The pinned `serve/web/app.js` patch renders a DeepSeek-specific card: **enabled**, four RAM-history slots, **8192 MiB disk target (not a RAM limit)**, native saves/restores, and actual last-request reuse. The generic Qwen renderer is unchanged. No `conversation_cache_mib` INFO value, RAM bytes, hard RAM cap or fake cache hit is manufactured. Native single-slot and text-only parked-history limitations remain visible. The existing three-line server hook is unchanged; it installs the observer before `set_family()`, using native `engine_kind=ds4` at registration and the actual family when handling metrics.

Exact frontend hashes: upstream `09031e89035b33d322f144e4b88c594e6d33ae76917779fd0b2cb77abe3540ab`; patched `df77352df1dc145a089d27713271295b300c00389d4d03d207fbdef85080cb53`. The deployed observer/vision helper hashes and refreshed runtime/model binding are recorded in preparation metadata and checked at readiness. Only `serve/server.py` and `serve/web/app.js` may have these exact tracked changes in this isolated runtime; older runtime contracts remain unchanged.

For an already prepared legacy .51 installation, after stopping its **owned** session and with no other maintenance lease, run as the normal operator:

```bash
STRATA_STATE_ROOT="$HOME/.local/state/locallm-strata/deepseek41-abiray" \
  ./v1strata.sh --prepare-deepseek41 --upgrade-monitor
```

The explicit upgrader validates existing provenance, backs up small files, and changes only the adapter/UI and their bound records—not the signed engine, model tensors, pack, sampling, context or cache settings. It refuses root, live target families, foreign leases, unknown source edits and altered configuration. New preparations include the observer automatically.

## Measured speed option (2026-10-26)

[Matched comparisons and activation/rollback instructions](DEEPSEEK41_SPEED_20261026.md) found that fully unloading the drafter (`--dspark none`) at the unchanged **1M context** improved median decode by **8.08%** and 2075-token prefill by **19.53%** on this host. Six target experts, model tensors and loaded vision remain unchanged; this is not a five-expert preset. `off` is not equivalent because it still loads draft weights. 262K alone was only a modest improvement, and SMT/mmap did not help.

**Stock auto remains the unchanged normal default.** The explicit `full-quality-no-drafter` preset is saved alongside `stock-auto` rollback, but activation and final image smoke are deferred: a real Blender rental began after the tests restored OctaSpace, and the rental gate correctly blocked selection. No rental was interrupted and no forced background activation is scheduled. Do not confuse output/reasoning limits with context size; long thinking can dominate latency independently of the measured throughput gain.

## LAN and ownership

Trusted-LAN `0.0.0.0:8080` is **API-key-free** by default. This upstream version makes keys optional itself; the launcher passes the actual upstream `--no-api-key` opt-out, never a dummy token. Deliberately supplied optional authentication still uses the private stdin/volatile-config path. Host/DNS-rebinding and browser Origin checks remain unchanged. Never expose the port publicly.

Inference/preparation run as `user`. Startup is eager; idle unload is disabled. Hive Ctrl+C can reach both the foreground launcher and its stop CLI: the stop CLI now waits up to 45 seconds for that same per-state cleanup lock, and refuses a replacement frontend identity instead of adopting it. Ordinary starts remain nonblocking. During switching, the helper waits up to 90 seconds for only the exact recorded old process identities (or new children of their still-live verified wrapper) to drain. It never signals an unknown process, treats a reused PID as the old process, or weakens rental/manual-session gates. Direct hosting uses HostLLM's fresh serialized pause lease, and its owned shutdown restores the prior miner/OctaSpace state. Hive uses the existing token-bound trusted wrapper and exact parent/session/executable identity, with only a **literal profile-bound .51** isolated-child-PGID allowance; .46's original contract and older strict grouping remain unchanged. No bare-PID/process-name kill fallback.

Hive [16] additionally recognizes only the inspected **root Wildrig fallback in the Hive/OctaSpace cgroup, or its exact live root Bash parent executing the installed Hive Wildrig `h-run.sh`** (CLI starts can inherit a user-session cgroup). The script must be a regular, non-group/world-writable file owned by root or the trusted Hive operator; the parent session/identity must match. Admission is paused before stopping it. Other miners, unknown GPU work, rentals and manual sessions remain refused. Its preflight runs before any service pause; when returning from a Wildrig flight sheet the package/config refresh ensures Hive actually selects `llm-hosting`, not just a profile symlink. It sets `MINER=custom`, the two LLM custom-miner assignments and an **empty `CUSTOM_INSTALL_URL`** so Hive neither rejects the LLM name against the old Wildrig URL nor redownloads the wrong miner. Unrelated settings and `MINER2` remain unchanged.

## Explicit preparation (as user)

```bash
./v1strata.sh --prepare-deepseek41 --download-only
./v1strata.sh --prepare-deepseek41 --prepare-runtime
./v1strata.sh --prepare-deepseek41 --prepare-pack
```

Runtime preparation needs the already staged pinned source checkout and official package in its isolated runtime directory. It is explicit, not an automatic startup updater. The plain GSQ pack recipe includes the Engram, original metadata, stock drafter and BF16 mmproj, with no `--skip-tensors`/`--dense-override` reconstruction. Older runtimes, model files, configuration and rollback routes are not replaced.

## Validation status

Real root [6] and [16] routes passed bounded native-window/text/vision checks on the four-3090 host. The final [16] retest recovered the operator's exact failed journal and passed the previously failing SC117 .46 stop/selection; a subsequent HiveOS cloud WildRig flight-sheet push interrupted the first DeepSeek load, so the explicit selection was reapplied after that update. Final native 1M metadata, text `63`, red/green swaps, ordered images, same-origin blue image, web UI HTTP 200, complete SSE `4`, >8K tail retrieval and four interleaved agents all passed. Followups restored 2573/2804/3034/2573 cached tokens and read only 13 new prompt tokens each.

**388 Python tests OK (one root-only branch skipped under the normal operator; actual root menu handoff tested live), all seven shell tests passed.** See [sanitized results](results/deepseek41-abiray-menu6-menu16-cache-20261025.json). Preparation tests restored original mining, but after the latest operator request the final state is **Hive DeepSeek serving with OctaSpace active**, not mining. A cloud flight-sheet change can legitimately replace local [16]; keep the Hive cloud custom miner set to `llm-hosting` with no stale download URL when this is the desired workload. No agent/rental safeguard was disabled to force persistence.

These are short functional checks, not populated-1M, sustained-load, broad quality or speed qualification.
