# SC117 abliterated IQ3_S on Strata — isolated route and qualification plan

**Status: unqualified; preparation and deployment have not been performed.** This page documents the existing opt-in SC117 route. No model files have been downloaded, no native pack built, no inference or qualification run, and no HostLLM/Hive deployment has occurred as part of this documentation task. Do not read the configuration below as a result or readiness claim.

The route replaces the old HostLLM IQ3_S entry at choice **[3]** and uses Hive profile `iq3_s-sc117-abliterated`. It is a distinct model/profile, not the older base IQ3_S route documented in [STRATA_IQ3S.md](STRATA_IQ3S.md). It must remain blocked from normal hosting until an identity-bound qualification is recorded.

## Immutable model and runtime pins

- Model repository: [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/eec4e10a2c29b440a2ae85591fa11569ce752963).
- SC117's [Strata-specific README](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/eec4e10a2c29b440a2ae85591fa11569ce752963/strata/README.md) identifies `mtp-q2_0.gguf` as the source for `tools/mtp_rt.py` and `strata/rt/` as the runtime directory passed to `--mtp`. It explicitly states that the published MTP head is stock/non-abliterated; no abliterated MTP head is published.
- Exact model revision: **`eec4e10a2c29b440a2ae85591fa11569ce752963`**.
- Strata: **0.1.39**, source commit `6f32ec070f23ced9f50e704d854d775da52591ab`.
- Profile: `iq3_s-sc117-abliterated`.
- API model ID: `qwen3.8-flash-next-sc117-abliterated-iq3_s`.

The model revision and Strata revision are separate pins. The packer is `tools/iq_pack.py` from the pinned Strata 0.1.39 checkout, run only through that prepared runtime's `.venv/bin/python`; the resolved interpreter path and SHA-256 are recorded with pack provenance. No arbitrary Python executable or pack copied from the earlier IQ3_S profile is accepted.

### SC117 source files

Destinations below are relative to `STRATA_DATA_ROOT` (default: `$HOME/.local/share/localllm-strata`). SHA-256 values are source-file pins; `dense.txt` is pinned by its Git blob object ID instead.

| Publisher path | Role | Bytes | Pin | Isolated destination |
| --- | --- | ---: | --- | --- |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` | Native shard 1 | 55,063,448,064 | SHA-256 `40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c` | `data/models/iq3_s-sc117-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf` | PLE/shard 2 | 28,800,138,432 | SHA-256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` | `data/models/iq3_s-sc117-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | BF16 vision projector | 907,542,944 | SHA-256 `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` | `data/models/iq3_s-sc117-abliterated/mmproj-Qwen3.8-Flash-Next-BF16.gguf` |
| `mtp-q2_0.gguf` | MTP source artifact | 889,017,216 | SHA-256 `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d` | `data/mtp/iq3_s-sc117-abliterated/source/mtp-q2_0.gguf` |
| `strata/rt/dense.bin` | Strata MTP runtime tensor | 116,099,072 | SHA-256 `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` | `data/mtp/iq3_s-sc117-abliterated/rt/dense.bin` |
| `strata/rt/experts.bin` | Strata MTP runtime tensor | 707,788,800 | SHA-256 `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` | `data/mtp/iq3_s-sc117-abliterated/rt/experts.bin` |
| `strata/rt/draft_vocab.bin` | Strata MTP runtime vocabulary | 425,196 | SHA-256 `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` | `data/mtp/iq3_s-sc117-abliterated/rt/draft_vocab.bin` |
| `strata/rt/dense.txt` | Strata MTP runtime configuration | 1,880 | Git blob `0a6b354ada9f26d07c4d51684089bdc2fda06a1b` | `data/mtp/iq3_s-sc117-abliterated/rt/dense.txt` |

The main shards plus projector total **84,771,129,440 bytes**. The MTP is also model-specific in layout: the launcher passes `--mtp` the SC117 `data/mtp/iq3_s-sc117-abliterated/rt/` directory, which contains the three `strata/rt` binary files and `dense.txt`. The pinned repository's MTP is explicitly **stock/non-abliterated**. It is not an abliterated MTP head and is not the root llama.cpp `mtp-q8_0.gguf`. Its compatibility and useful acceptance with the abliterated main model remain unqualified.

### Isolated files and what must not be reused

With the default data root, the important paths are:

```text
~/.local/share/localllm-strata/
  runtimes/0.1.39/source/                         # pinned Strata checkout and packer
  source-99f3dbd0b21d/strata-iq3_s.json           # validated base config; read-only input
  data/models/iq3_s-sc117-abliterated/           # SC117 shards and its projector
  data/packs/iq3_s-sc117-abliterated/             # new SC117 native pack; never the base pack
  data/mtp/iq3_s-sc117-abliterated/source/        # pinned mtp-q2_0.gguf source artifact
  data/mtp/iq3_s-sc117-abliterated/rt/            # SC117 strata/rt runtime files
  profiles/iq3_s-sc117-abliterated/prepared.json  # pinned source-asset manifest
  profiles/iq3_s-sc117-abliterated/native-pack.request.json
  profiles/iq3_s-sc117-abliterated/native-pack.json
```

The native pack is separate because SC117 tensor offsets differ from the earlier IQ3_S pack. Do not point this profile at `data/packs/iq3_s`, `data/mtp/rt`, or the earlier model/projector paths; do not overwrite the base config or existing pack. SC117 shard 2 is publisher-identical to the existing base IQ3_S shard 2. Reuse is optional, limited to that exact file, and copies it into the isolated SC117 destination only after exact size and SHA-256 verification; it does not hard-link or modify the source.

## Explicit preparation, verification, and pack build

These are **future operator steps**, not actions performed for this document. Set an absolute, non-symlink data root if not using the default. The pinned 0.1.39 source/private Python environment must already be prepared there for pack creation. On a fresh installation, establish the ordinary Strata prerequisites using the procedures in [STRATA_IQ3S.md](STRATA_IQ3S.md) and [STRATA_V0139.md](STRATA_V0139.md); SC117 asset preparation does not install or build the Strata runtime.

```bash
export STRATA_DATA_ROOT="$HOME/.local/share/localllm-strata"

# Fetch and verify every allowlisted SC117 source file at the exact revision.
python3 prepare_iq3_s_sc117_abliterated.py --data-root "$STRATA_DATA_ROOT"

# Optional alternative: use only the existing base IQ3_S shard 2 after exact verification.
# Use this instead of the preceding all-assets command, not in addition to it.
python3 prepare_iq3_s_sc117_abliterated.py --data-root "$STRATA_DATA_ROOT" --reuse-shard2

# Explicitly run the pinned .39 native packer to create, seal, hash and record the new pack.
python3 prepare_iq3_s_sc117_abliterated.py --data-root "$STRATA_DATA_ROOT" --build-pack

# Verify source and pack readiness; --full-hash recomputes all source and pack hashes.
python3 prepare_iq3_s_sc117_abliterated.py --data-root "$STRATA_DATA_ROOT" --check
python3 prepare_iq3_s_sc117_abliterated.py --data-root "$STRATA_DATA_ROOT" --check --full-hash
```

The first preparation command downloads only the fixed asset allowlist and verifies full source hashes before publishing its manifest. It does **not** run the packer or inference. `--check` requires both the completed source manifest and a recorded native pack, so run it after `--build-pack`. Pack creation is a separate explicit action: it requires a new `data/packs/iq3_s-sc117-abliterated/` path, checks the exact 0.1.39 source commit and packer file, and refuses to reuse or overwrite an existing pack/request/manifest. Failed or interrupted outputs are left for inspection rather than silently repaired.

The packer output is validated against both SC117 source shards and must contain the expected conversion/index/native-expert reports, dense weights, and tokenizer files. The resulting pack files are sealed read-only (`0444`, directories `0555`) and hashed into `profiles/iq3_s-sc117-abliterated/native-pack.json`. `--check` performs metadata/provenance readiness; `--check --full-hash` also recomputes every pinned source and pack SHA-256. Neither check runs inference or establishes qualification.

## Configuration target — provisional, not measured

The effective profile deep-copies the existing validated IQ3_S baseline config and rewires only the model shards, projector/vision model, tokenizer, native pack, and MTP directory. The source config and other profiles are preserved. Its fixed serving contract is:

- Strata **0.1.39**, explicitly requested; four configured GPUs `[0,1,2,3]` and automatic layer split inherited from the baseline.
- Native context **262144**; INT8 KV with **32768 resident**; no context extension.
- SC117 Strata `strata/rt` MTP depth **4**, stock/non-abliterated head.
- BF16 GPU vision; one serving slot and one batch group.
- Provisional starting values: `--prefill auto`, layer split `auto`, conversation cache **8192 MiB / 4 cache slots**.

The cache, prefill and split are tunable only in explicit trial mode. Trial prefill is `auto`, `1024`, `2048`, `4096`, or `8192`; split is `auto` or three increasing boundaries within the 36-layer model; cache is `0`, `4096`, `8192`, `12288`, or `16384` MiB, with `0` slots when disabled and otherwise 1, 2, or 4 slots. Context, KV mode/residency, MTP depth, BF16 GPU vision, and one-slot/one-group are not trial overrides. Any setting change changes the qualification fingerprint and needs its own matching attestation. HostLLM and Hive use only the exact defaults above.

These values describe a candidate config, not proven capacity, successful startup, memory headroom, MTP benefit, or quality. In particular, a four-slot conversation-cache configuration is not four concurrent serving slots: serving remains one slot.

## Qualification evidence required

A complete schema-version-1 report must identify the exact profile/model, runtime version and commit, profile fingerprint, and full effective settings. The fingerprint binds the pinned SC117 source hashes, every recorded pack-file hash, the 0.1.39 runtime manifest/executable and interpreter hashes, the unchanged baseline-config hash, every setting, and the host machine-id hash plus all four GPU UUIDs and driver version. For Python it additionally binds the `pyvenv.cfg` bytes/metadata, configured runtime version, and resolved standard-library path; system-site packages must be disabled, and the Python home/stdlib directory chain must be root-owned and non-writable. The standard-library contents are not recursively hashed and remain a trusted platform dependency. The runtime import inventory hashes source/site-package modules, native extensions, and archives, while rejecting executable `.pyc`/`.pyo` caches rather than trusting them. It also rejects untracked source imports, customization modules, symlinked imports, and unexpected/executable `.pth` entries. A qualification issuance must recompute all source and pack hashes; later preflight uses the root-attested manifest digests and Linux ctime-bound file states to avoid rehashing the full model on every start. The still-pending trusted runner must independently verify expected/trusted import and runtime identities and recapture them before and after probing; current point-in-time fingerprinting does not itself eliminate a same-UID check/use race. It must pass **all** required checks:

1. `smoke` — a successful bounded live serving smoke check.
2. `native_context_tail` — a prompt count from **255,000 through 262,144 tokens**, at least one completion token, and retrieval of the deliberately placed tail marker.
3. `gpu_vision` — a passing BF16 GPU-vision result (`passed: true`, `gpu: true`, `precision: "bf16"`).
4. `mtp_acceptance` — **measured** depth-4 MTP counts with `head_variant: "stock/non-abliterated"`, at least one drafted token and **at least one accepted token**; accepted count must be no greater than drafted count. Zero accepted tokens fails. The code does not set a higher acceptance-rate threshold, so do not imply one.
5. `conversation_cache_park_restore_reuse` — evidence that cache state was parked, restored, and reused.
6. `prefill_screen` — a passing screen whose selected prefill exactly matches the attested settings.
7. `resource_telemetry` — passing telemetry evidence.

Each required `checks` entry must be boolean `true`. The report also carries the native-context counts/tail marker, vision details, drafted/accepted MTP counts, cache outcomes, selected prefill, and telemetry outcome. The profile validator checks that telemetry and the prefill screen pass but does not define numeric telemetry limits or a particular prefill performance threshold; retain the measured observations and do not invent pass results. The native-context probe is a bounded tail-retrieval check, not a general long-context quality guarantee.

The attestation location on Linux is fixed and outside the model user's writable data tree:

```text
/var/lib/localllm-strata/qualifications/sc117-iq3s.json
```

It must be a regular non-symlink file owned by root with mode `0444`; its parent directories must be root-owned and not group/other-writable. The candidate runner now defines serialized fixed synthetic probes and response/counter/log-window receipts, but **issuance is still disabled and the CLI exits before inference** while the reviewed root-owned artifact pin policy is unprovisioned, the operator/server share a UID, pinned .39 lacks a per-request BF16 execution attestation, and the installed runner hash is not pinned by readiness. A successful probe session is not enough to waive any of those blockers. The active probe code has not been run against a model; no report can be emitted or imported to unlock normal hosting. Caller-supplied pass assertions are rejected. Stale assets/config/runtime, changed settings, missing checks, or mismatched before/after identity must fail readiness. The asset-preparation CLI does not generate qualification evidence; `--check` cannot stand in for live measurements, and CPU-only unit tests do not qualify the model.

The launcher contains a narrow, explicit unqualified trial exception for controlled testing: it requires `STRATA_SC117_IQ3S_TRIAL=1`, binds to `127.0.0.1:18080`, and uses the separate state directory `~/.local/state/locallm-strata/sc117-iq3s-trial`. It is not normal readiness, does not create an attestation, and must not be used as a route to LAN/Hive deployment. HostLLM and Hive apply the normal qualification gate.

## HostLLM and Hive gates; preserve runtime defaults

Normal readiness is checked with the explicit profile/runtime/slot contract:

```bash
./v1strata.sh --check-ready \
  --profile iq3_s-sc117-abliterated --runtime 0.1.39 \
  --parallel 1 --batch-groups 1
```

Without a matching root-owned report this must fail; asset/pack readiness by itself is not enough. Once qualified, its exact normal readiness line is:

```text
READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.
```

- **HostLLM:** choice [3] checks the explicit .39/one-slot/one-group request, exact readiness line, and qualification before acquiring its hosting pause lease or pausing the miner/OctaSpace service. Its lifecycle safety checks still block on Docker uncertainty/rentals, existing or unidentified GPU work, an occupied API port, or unsafe service state. Use HostLLM for normal hosting and restoration, not a raw launcher as a substitute for its lifecycle controller.
- **Hive:** the root-only profile selector checks readiness before changing the `current` profile pointer and runs only while idle. The SC117 `h-run.sh` wrapper repeats the exact .39/qualification check before launch, and fails closed if the listener/state or port is occupied, Docker is unavailable/nonempty, four GPUs cannot be inventoried, any GPU has less than **23,700 MiB free**, or available RAM is below **12,288 MiB**. The Hive selector does not start/stop the miner or change `osn.service`; use Hive's normal operator workflow only after a successful selection gate.
- **Binding/authentication:** both routes accept only `127.0.0.1` or `0.0.0.0` and default to loopback. Explicit `STRATA_HOST=0.0.0.0` requires an API key in HostLLM or the Hive environment; the launcher enforces this even when started directly. The API key stays out of argv/environment and the persistent run-config, using a private RAM-backed tmpfs config removed on stop. Protect the Hive configuration that stores the key. Traffic is still plain HTTP, so use a trusted LAN or trusted tunnel and never expose the API to an untrusted network.

This profile is pinned to **Strata 0.1.39 regardless of generic runtime selection**; `auto`, 0.1.38, 0.1.41, and 0.1.44.1 are not substitutes. Preserve the existing .39 default/rollback selection. The .44.1 runtime is a separate Orca Q4 candidate and its source, config, evidence, and future-launch selection must remain unchanged by this SC117 route. Preparing or qualifying SC117 does not promote another runtime or rewrite the shared baseline.

## Current disposition

**Unqualified and hard-blocked.** No SC117 download, asset preparation, native pack, inference, live qualification, root attestation, HostLLM launch, or Hive deployment is claimed here. The active-probe runner implementation is uninstalled and unpinned, report issuance remains disabled, and its current known blockers stop it before inference; no qualification report can unlock normal starts. Preserve the existing .39 default and the separate .44.1 candidate.
