# Unsloth Dynamic V3.1 IQ4_XS — offline preparation, NOT activation

> **Historical preparation stage.** Later separate operator approval authorized the switch/testing and left [IQ4_XS ACTIVE as Hive[21]](UNSLOTH_V31_IQ4XS.md) on November7. The preparation-only hold and registry/start-refusal statements below describe that earlier checkpoint, not the current serving state. Do not undo the later activation or replay saved controllers.

The operator approved preparation only, while continuing to use the existing **Swift** server. **Do not stop/switch it, load the candidate, run replacement inference, or publish/activate a new serving profile until a separate explicit continue.** No TensorSharp/512K retry is authorized.

## Correct release identity

- Repository: `unsloth/Qwen3.8-Flash-Next-GGUF`.
- Frozen repository revision: `d96201194f103c906994f283f504209dc8d2a10c`.
- **UD-IQ4_XS**, actual weight-file commit [`814db05f88b3b827ea7cfcab5344fd218de494d2`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/commit/814db05f88b3b827ea7cfcab5344fd218de494d2), **“Better UD 3.1 quants”**, November 4.
- Q3_K_XL is explicitly V3.1 too, but smaller. Q4_K_XL's November 4 commit says **“Add mix n match engram”**: its directory update was mistakenly treated as proof of a new V3.1 quant. That earlier recommendation was corrected before any Q4 model download.
- Four original IQ4_XS shards: **93,672,359,738 bytes**, including the original **28,800,138,432-byte IQ4_NL PLE shard**.
- Separate newly named original-Qwen **Q8_0 PLE**: **54,400,261,312 bytes**, SHA `e3d76a2747c1fe53e3420d1bb7fd9593aafd03035fee4926317db80ad5c5296a`.
- Main model excluding bundled PLE: approximately **64.87 GB**; main plus Q8 PLE: **119.27 GB**. These are decimal file sizes, **not resident RAM/VRAM or a capacity demonstration**. Original shards are retained for complete split/native-dense provenance.

All five exact file sizes and SHA-256 pins are in `unsloth_v31_assets.py`. A different checksum on the new Q8 filename was not assumed to mean an equivalent file. For table-only files, the preparer may copy an existing original-Qwen payload into a **new inode** with the publisher's new header, but adopts that copy **only if BOTH old and complete new file hashes match their independent pins**. Otherwise it downloads the new file normally. Existing source files are never rewritten, linked, requantized or deleted.

## Inactive candidate policy

- Existing checksum-bound **Strata 0.1.46**, existing private Python, CUDA **12.9** toolchain/runtime and existing trusted-LAN mandatory-key-only patch. **No runtime/driver/kernel/system CUDA upgrade.**
- **262144 context per chat**, **2 serving slots / 2 pipelined groups**; **16384 MiB / 8 cached histories**. Histories are not serving slots, and this is not proof of eight full-context histories or two full windows at once.
- INT8 KV with **32768 resident**, mmap Q8 PLE, auto prefill, trimmed stage weights; same established GPU order.
- **Automatic layer split**, not SC117's fixed measured split: different quant/weights, no inherited performance claim.
- **BF16 CPU vision**, full hash of the retained matching Flash-Next projector verified.
- **Stock/non-abliterated original-Qwen Strata MTP**, existing pinned Q2_0 expert / Q8 dense runtime, spec depth **3**, pipeline windows **2**. This is **not** Unsloth's separately packaged llama.cpp Q8 MTP GGUF: the pinned .46 `mtp_rt.py` expects the native Q2_0 layout. No untested MTP conversion/kernel patch or additional 1.3–1.7x speed claim. Concurrent pipelined windows remain draft-free.
- Pinned **original-Qwen refusal-direction projection**, `--control-vector-scaled <file>:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. Per-request default **on**, can be disabled with `experimental_speed_projection:false`. Model-specific experimental activation projection, **not source-weight abliteration**, not guaranteed SC117 equivalence or unchanged ordinary answers.
- Existing upstream CPU `iq_pack.py --compat-bf16` produces a **separate derived native pack**. It records BF16 compatibility rounding for small unsupported projections in `conversions.json`. It does not rewrite the downloaded GGUFs, routed experts, native dense tensors or Q8 PLE file; no numerical engine-source patch is made.
- Separate fresh candidate state and generic original-model routing seed, **no conversation/KV/adaptive-history copying** from Swift, SC117 or another PLE variant.
- Normal trusted LAN `0.0.0.0:8080`, **no mandatory API key/dummy token/Authorization header**. Existing Host/Origin/configured-key checks preserved. Not public hosting.
- No fixed output/thinking budget added; no live sampling/context/cache configuration changed. **No 512K/YaRN flags.**

## Data-only tools

Run as the unprivileged model owner, with low CPU/I/O priority:

```bash
cd /home/user/LocalLLM
nice -n 19 ionice -c 3 python3 prepare_unsloth_v31.py --download-only --rate-mib 80
nice -n 19 ionice -c 3 python3 stage_unsloth_v31.py --prepare
python3 stage_unsloth_v31.py --check
# Explicit complete model-file rehash when desired / before any later authorized GPU start:
nice -n 19 ionice -c 3 python3 stage_unsloth_v31.py --check --full-hash
```

These tools have **no start/stop/switch/activate option**. The profile module's `authorize_start()` always refuses. The candidate is deliberately **absent** from the production launcher choices, Hive registry and installed selector. This preparation does not weaken existing readiness, rental or ownership checks.

Local staging:

```
/home/user/.local/share/localllm-strata/profiles/unsloth-v31-iq4xs-q8-cvec-2slot/
  models/                         # four pinned complete publisher shards
  Qwen3.8-Flash-Next-engram-Q8_0.gguf
  Qwen3.8-Flash-Next-experimental-speed-projection.gguf
  native-pack/                    # CPU-only derived pack and tokenizer
  config.inactive.json            # NOT the live run-config.json
  downloaded.json
  prepared.json                   # activation_allowed=false; gpu_inference_tested=false
```

Fresh candidate state is `~/.local/state/locallm-strata/unsloth-v31-iq4xs-q8-cvec-2slot-trial/`; no server record is created. Source data and completed pack/config files are read-only. The prepared manifest records exact identities, hashes, dependencies, CPU conversion provenance, limits and pending checks. Offline preparation is **not GPU fit, LAN serving, quality, speed or stability qualification**.

## Current service and cleanup boundary

The actual current pointer is **`profiles/swift15u-q8`**, not the earlier recovery report's SC117 [18]. The existing live Swift session is preserved; its current ancestry is **not adopted as Hive-owned** and the existing inspector's refusal is not relaxed. Final switching must resolve fresh exact manual/owned-session authorization without overriding a renter or an unrelated process.

The separately requested **TensorSharp direct [8] / Hive [19] information entries are removed**, with no number reuse. Five unused repository build/preparation helpers and proven-unused private TensorSharp source/.NET/CUDA13/NuGet/build archives were removed after root process/maps/FD/cwd/executable inspection and inventory under the lifecycle reservation. No runtime process, miner, admission service or container was stopped. Original Swift/SC117 model data, vision, Q8 PLE, .46, CPU vision, installed active bundle/selection/stable links and cloud configuration remain intact. Detection of an externally/manual-started TensorSharp process stays as a **safety blocker**, never stop authority. Small old calibration/results and private retirement metadata remain historical evidence; old controllers are not pending work.

## Separate next phase — only after explicit approval

1. Freshly inspect actual selection, exact process/birth/UID/executable/argv/parent ownership, live requests, Docker/rentals/GPU/manual work, lease/journal and hardware health under the serialized lifecycle reservation. **Renters always win; keep OctaSpace admission running.**
2. Preserve the then-current session/profile/settings; never infer stop authority from this preparation, an old PID or diagnostic record. Drain/stop only a freshly verified and explicitly authorized current session.
3. Separately integrate the isolated candidate into bounded launcher/Hive ownership gates and perform a real startup/LAN functional window without mandatory keys: text, thinking, ordered vision, concurrent slots, complete SSE, tool-format, cache reuse and Host/Origin POST boundaries. Stop on hardware/kernel faults; no resets/retries/promotions.
4. Measure actual fit/performance and report quality limitations. Publish the final runnable menu/profile only with accurate tested scope. Restore the preserved current profile if the candidate fails. No automatic rollback attempt on faulty hardware, renter override, TensorSharp retry or 512K trial.

Sanitized outcome: `results/unsloth-v31-iq4xs-preparation-20261106.json`. Private operational receipts: `/var/lib/localllm-strata/unsloth-v31-iq4xs-prep-20261106/` (not executable continuation authority).
