# LocalLLM — Swift, Hauhau and Flash-Next Strata

HostLLM and Hive custom-miner profiles for `.69`: four RTX 3090s, approximately 128 GB RAM. **Swift remains the default; Hauhau remains the retained fallback.** The Flash-Next Heretic model below is separate from the smaller dense Swift/Hauhau models, despite sharing the Swift name.

## Current menu

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
[2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
[3] Swift 1.5 Heretic RE — Strata .46 | native 262K | BF16 CPU vision | spec 3 + windows 2 | one slot
[4] SC117 GSQ-RCO IQ3_S — Strata .45 | unqualified CPU-vision trial | native 262K | one slot

[9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit
[12] Set next Hive miner — Swift 1.5 Uncensored Q8_K_XL
[13] Set next Hive miner — Hauhau Q8_K_P
[14] Set next Hive miner — Swift 1.5 Heretic RE
[17] Set next Hive miner — SC117 .46 unqualified CPU-vision trial
```

The requested old Orca [3]/[5]/[14], DeepSeek [6]/[16], and SC117 .45 Hive [15] rows were removed. [3]/[14] now name **Heretic**, not Orca. SC117 .45 direct [4] and SC117 .46 Hive [17] are retained. The lower-level Hive menu lists only active model choices; retired Orca/DeepSeek names cannot be selected there.

## Heretic RE

Publisher: [trubisky/Swift-1.5-Flash-Next-Heretic-ReverseEngineer](https://huggingface.co/trubisky/Swift-1.5-Flash-Next-Heretic-ReverseEngineer), pinned revision **`a7a90ef46b79b746c484d094fd5814c359543d77`**. See [HERETIC_RE.md](HERETIC_RE.md) for assets, provenance, validation scope and results.

| Setting | Explicit profile |
|---|---|
| Profile | `swift15-heretic-re` |
| Native model/served context | **262144**, from actual GGUF metadata; no extension or silent truncation |
| Model | Publisher's mixed IQ4_NL V2-70G-FIX, all eight SHA-pinned shards |
| Strata | Existing pinned **0.1.46** frontend/vision plus isolated **BF16 L0-gate compatibility engine**, CUDA sm_86; shared SC117 runtime unchanged |
| Vision | **BF16 CPU vision**, original Flash-Next projector; preserves VRAM for the target model |
| Serving | One slot/group, four GPUs, split `14,24,36` |
| Tuning | `--spec 3 --pipeline-windows 2`, automatic prefill, `--trim-stage-weights` |
| KV | INT8, native 262K window with 32768 resident |
| Conversation cache | **8192 MiB budget / four parked histories**, not four serving slots |
| Thinking / sampling | High default; temperature 1.0, top-p 0.95, top-k 20 |
| Load policy | Eager model + vision load; no automatic idle unloading |
| State | `~/.local/state/locallm-strata/swift15-heretic-re/`, separate learned expert profile |

The plain .46 engine loaded this pack but rejected its uncompressed BF16 attention gate on first inference. A narrow isolated compatibility build now uses existing BF16 kernels for that exact gate; actual-matrix CPU/GPU oracle tests and bit-identity checks passed, with no target requantization. See [the compatibility scope](HERETIC_RE.md#why-the-bf16-compatibility-overlay-is-necessary).

The publisher's `strata-pack-heretic-v2-70g-fix.zip` is **data only**. Its exact dense/index/tokenizer/compatibility files are SHA-pinned; no downloaded scripts are executed. Its older, doubled-prefix shard names are preserved as local filenames, mapping to the identical HF-published bytes. All 48 layers' expert offsets/types and every GGUF tensor bound are checked against the real eight shard headers. No requantization or omitted target tensors.

The drafter is the same independently pinned **stock/non-abliterated Flash-Next MTP head** used by SC117, copied into Heretic's own support directory. It is not advertised as a Heretic-trained head; the actual target verifies drafts. Tuning inherited from another model is not proof of optimal performance for this one.

### Prepare explicitly, then start

As the normal operator, not root:

```bash
cd /home/user/LocalLLM
./v1strata.sh --prepare-heretic --download-only
./v1strata.sh --prepare-heretic --prepare-compat
./v1strata.sh --prepare-heretic --prepare-pack
./HostLLM.sh
# [3] direct foreground server; [14] select/switch Hive.
```

Preparation never starts inference or changes services. The root-owned runtime parent may need one administrative creation of the new overlay directory; build and inference still run as `user` (see [preparation details](HERETIC_RE.md#preparation)). Pack preparation reuses the locally pinned SC117 projector/MTP **only after full checksum verification**, copying them into independent support files; those source assets must already be prepared on a new host. It fails closed if anything is missing or differs.

For explicit readiness:

```bash
STRATA_STATE_ROOT="$HOME/.local/state/locallm-strata/swift15-heretic-re" \
  ./v1strata.sh --check-ready --profile swift15-heretic-re \
  --runtime 0.1.46 --parallel 1 --batch-groups 1
```

Readiness is provenance/configuration, **not inference, populated-context, broad-quality or sustained-load qualification**. Ordinary starts never download or upgrade anything.

## Hosting safety and LAN access

```bash
cd /home/user/LocalLLM
./HostLLM.sh
```

- Direct hosting uses a newly created, identity-bound HostLLM pause lease. It pauses OctaSpace/mining only after readiness and rental/workload checks. Ctrl+C stops only that exact frontend/engine/vision family, then restores the recorded previous state after GPU teardown clears.
- Root menus hand Strata inference to unprivileged `user`. No inference/preparation as root, process-name kills, driver resets or clock/power changes.
- Hive selections use a serialized ownership-checked transaction and atomic profile pointer. Hive-owned sessions may be switched; rentals, unidentified jobs and another manual hosting session are never adopted or stopped. A pending transaction can be recovered only after inspecting its exact identity/journal: `sudo python3 hive_profile_switch.py --recover`.
- Hive hosting keeps OctaSpace active and yields its own model if a rental arrives. Do not disable the cloud agent, rental guards or process-ownership checks to force hosting.
- [9] controls the retained isolated states as well as the ordinary Qwen state. Exiting an unrelated menu never releases someone else's lease. Reopen the menu after updates; old shells retain old functions.
- New package installs default to Swift and preserve recognized active selections. A selected retired model is refused rather than silently loaded/migrated from missing weights.

Trusted-LAN endpoints:

- Web UI: <http://192.168.1.69:8080/>
- OpenAI: `http://192.168.1.69:8080/v1`
- Anthropic: `http://192.168.1.69:8080/v1/messages`

**No API key is required**, including after upgrades. Preserve hostile Host/DNS-rebinding and browser Origin rejection and optional explicitly configured authentication. This is for a trusted, firewalled LAN, never public exposure. Heretic defaults to `0.0.0.0:8080`; `STRATA_HOST=127.0.0.1` selects explicit local-only direct hosting. SC117 trials stay fixed to trusted-LAN `0.0.0.0:8080` and remain **unqualified**, not normal/production authorization. Swift/Hauhau retain their llama.cpp UI and existing multi-slot behavior.

## Removed weights and historical evidence

On operator request (2026-10-29), unique **Orca Q4_K_S and DeepSeek Abiray IQ2_XXS weights/packs** were removed from `.69`. They are no longer runnable rollbacks. Retained source/build/configuration and historical reports are not proof their weights are installed. No unrelated model, system NVIDIA/CUDA/kernel, protected Pi skill/extension/settings or user conversation state was deleted.

SC117 no longer needs the retired Orca GGUFs/pack/projector to pass startup. It reuses only the exact SHA-bound audited preset **metadata** and independently checks its own model/pack/MTP and selected .45/.46 runtime. This does not bypass qualification or weaken the trial-only state/bind restrictions.

Historical references (not current menu instructions): [Orca Q4](ORCA_Q4KS.md), [.41](ORCA_Q4KS_V0141.md), [.44.1](ORCA_Q4KS_V01441.md), [DeepSeek](DEEPSEEK41_ABIRAY.md), [DeepSeek speed](DEEPSEEK41_SPEED_20261026.md), [.46 runtime/no-key patch](STRATA_V0146.md), [SC117](STRATA_SC117_IQ3S.md), [measurement protocol](HOWTOTUNE.md), [Swift two-slot](SWIFT15U_2SLOT_2026-09-26.md), [Swift/MTP](SWIFT15.md), [Hauhau baseline](GSQ_TUNING_RESULTS.md), [Hive package](hive-llm-miner/README.md).

The original .38/.39/.41 engine/source pins remain available where compatible assets are retained; .46 is profile-bound, not a generic global upgrade. See [STRATA_IQ3S.md](STRATA_IQ3S.md), [STRATA_V0139.md](STRATA_V0139.md), [STRATA_V0141.md](STRATA_V0141.md). No quality/speed claims are transferred between model families.

Pi clients should use the server's actual per-sequence **`meta.n_ctx`**, not just training-context metadata. No global Pi settings change is needed. Strata is MIT-licensed; weights retain the publisher's Swift Open License/Qwen license terms.
