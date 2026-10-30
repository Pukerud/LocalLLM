# Flash-Next Q8_0 PLE-only candidates — no-interruption preparation

2026-10-30, `.69`. The operator normally runs **SC117 .46**. This update prepares opt-in higher-precision n-gram tables for both retained Flash-Next models **without stopping, switching, reloading or reconfiguring the running miner**. Existing IQ4_NL PLE routes remain available.

## Menu map

| Model / PLE | Direct HostLLM | Hive HostLLM | Runtime |
|---|---:|---:|---|
| Heretic RE / original IQ4_NL | **3** | **14** | .46 + the existing isolated BF16-gate engine |
| SC117 IQ3_S / original IQ4_NL | **4** | **15** | Direct .45; Hive .46 |
| Heretic RE / **Q8_0** | **5** | **16** | Same .46/BF16-gate engine |
| SC117 IQ3_S / **Q8_0** | **6** | **17** | Direct .45; Hive .46 |

Old Hive SC117 [17] becomes [15]; [17] now deliberately names **SC117 with Q8_0 PLE**. This does not change the selected Hive profile. The low-level Hive selector keeps [1]–[5] and adds **[6] Heretic Q8_0** and **[7] SC117 .46 Q8_0**. Swift/Hauhau and the menu's existing Swift installation default are unchanged.

**Q8_0 here describes only the learned PLE/n-gram table.** It does not turn the IQ3_S experts, Heretic's mixed-quantized experts, or INT8 KV into a different model quantization. No runtime upgrade, retraining, new routing shortcut, YaRN or context extension is involved.

## Exact data source

Only the table-only third shard is downloaded; the rest of the Unsloth Q6 model is neither downloaded nor used:

- Repository: `unsloth/Qwen3.8-Flash-Next-GGUF`.
- Revision: **`38bb39ee97821de2c9009abb7e93950eec396e66`**.
- File: `UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf`.
- Bytes: **54,400,261,312** (about **50.66 GiB**).
- SHA-256: **`34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20`**.
- Exactly one tensor: `per_layer_token_embd.weight`, **Q8_0**, shape **[160, 320001536]**, 170 bytes/row.
- Tensor payload: **54,400,261,120 bytes** following the 192-byte header.
- One shared, unchanged published file under `profiles/qwen-flash-next-ple-q8/`; no conversion from the already-lossy IQ4_NL table and no second 54-GB copy for the other model.

Both original model PLE tensors were independently hashed in full. Their **28,800,138,240-byte IQ4_NL payloads are bit-identical**, SHA-256 **`dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3`**. The SC117 publisher also documents that this table was untouched by abliteration. This supports using the same matching original-Qwen replacement with either model; it is not a quality benchmark.

Local preparation verifies the full published file checksum, exact one-tensor header/encoding/shape/bounds and immutable provenance pins. Readiness rejects changed path, size, mtime, inode/device, owner, manifest pins, tensor format or symlinks. Each actual Q8 start rehashes the full sidecar **before spawning any model/GPU process**. Initial startup therefore includes additional checksum time.

## Same serving policy, separate state/cache

The only numerical model input changed is `--ple-gguf`. Keep the original expert/dense weights, pack, tokenizer, MTP, projector, engine and tuning:

- Native **262144** context; no silent fit/truncation or RoPE extension.
- Four GPUs, split **14,24,36**, one serving slot/group.
- BF16 **CPU** vision; INT8 KV with **32768** resident.
- `--spec 3 --pipeline-windows 2`, auto prefill, trimmed stage weights.
- **8192 MiB / four parked histories**, not four concurrent serving slots.
- Existing thinking/sampling policy, target-verified stock/non-abliterated MTP.
- No mandatory LAN key; preserve Host/Origin checks and deliberately configured optional authentication.

The separate API model IDs end in `-ple-q8_0`; existing model aliases remain accepted. Isolated adaptive profiles, runtime configs and state are:

```text
~/.local/state/locallm-strata/swift15-heretic-re-ple-q8/
~/.local/state/locallm-strata/sc117-iq3s-ple-q8-trial/
```

Original IQ4_NL state/configuration and conversation histories are not rewritten or adopted. Native conversation snapshots are process-local; any file-tier compatibility key also includes the PLE input identity. Never reuse KV computed with another PLE table for a quality comparison. Q8-specific stop commands and Hive supervision require the recorded variant marker and exact process identity; a wrong PLE variant is not an authorized stop target. SC117 remains unqualified/trial-only, including the new variants.

## Preparation and no-stop package addition

If the root-owned parent prevents creation of this **new** directory, create only that directory administratively, then perform downloads/preparation as the normal operator:

```bash
sudo install -d -o "$USER" -g "$(id -gn)" -m 700 \
  "$HOME/.local/share/localllm-strata/profiles/qwen-flash-next-ple-q8"

cd /home/user/LocalLLM
./v1strata.sh --prepare-ple-q8 --download-only
./v1strata.sh --prepare-ple-q8 --prepare

# Narrow add-only mode: safe while the current miner remains running.
sudo ./install-hive-llm-miner.sh --add-ple-q8-profiles
```

The downloader is resumable, size/SHA-pinned, serialized and low-priority. No model-repository code is executed. The original model files are kept for rollback, so this adds the full **54.4 GB** of disk data, not just the difference between table formats.

The **add-only** installer serializes against HostLLM/Hive selection, refuses rentals or a pending lease/journal, checks both .46 candidates as unprivileged `user`, publishes only two complete new profile directories and atomically replaces the selector. It refuses to overwrite a differing existing Q8 profile. It verifies that `current`, stable entrypoint links, active profile files and miner markers stay unchanged. It does **not** rewrite rig/wallet files, active wrappers/configs, service state or the selected profile, and contains no stop/start/switch operation.

The ordinary full installer still refuses active workloads and is **not** the no-stop command. Reopen HostLLM after updates; existing shells retain their old menu functions. Merely adding/listing the options does not choose them. A future explicit Hive [16]/[17] selection invokes the existing ownership/rental-checked switching workflow; direct [5]/[6] refuse to replace an already-running Strata instance.

## What was and was not tested

Completed preparation scope: full sidecar checksum/header checks, full original-table identity comparison, actual offline readiness for Heretic .46 and SC117 .45/.46 Q8 variants, unit tests, shell dispatch/safety tests and no-stop package publication. Final counts and preservation receipts are in [the sanitized results](results/ple-q8-no-interruption-20261030.json).

**Not performed:** Q8 model loading/inference, text/image/tool output tests, cache reuse with Q8, long-context Q8 retrieval, throughput/latency measurements or a matched IQ4_NL-versus-Q8 quality A/B. These require a later explicitly authorized model switch; the running SC117 was deliberately left alone. Earlier IQ4_NL smoke/near-native-context evidence does not qualify these Q8 variants.

Higher table precision may improve fidelity, but does not guarantee smarter answers or unchanged speed. SSD access, cold/warm page cache, available RAM, long prompts and the remaining expert quantization still matter. Strata also supports BF16 PLE; Q8_0 is an opt-in precision/storage compromise, not full-BF16 inference or a performance promotion.
