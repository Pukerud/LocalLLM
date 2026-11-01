# SC117 Q8 two-user preset — Strata .46

Added on 2026-11-01 at the operator's request, based on [the completed .46 cache/concurrency study](SC117_Q8_CONCURRENCY_20261101.md). This is a separate opt-in profile, **not** a rewrite of the existing one-slot routes or an overall installation-default change.

| Menu | Choice |
|---|---|
| Direct HostLLM | **[7] SC117 Q8 two-user** |
| Hive HostLLM | **[18] SC117 Q8 two-user** |
| Low-level Hive selector | **[8]** |
| Launcher profile | `sc117-q8-2slot` |
| Hive bundle | `iq3_s-sc117-abliterated-ple-q8-2slot-v0146` |

Original SC117 [4]/[6]/Hive [15]/[17], Heretic choices and every existing menu number remain unchanged. Direct [7] and Hive [18] both pin **.46**, never .45. Swift remains the installation's original default; selecting an active Hive miner is a separate operator action.

## Exact policy

- Same SC117 IQ3_S main/expert/dense weights, stock target-verified MTP, pinned .46 shared runtime and original-Qwen **Q8_0 PLE** as [17]. No new weights, quantization, engine patch or data copy.
- **Two serving slots / two pipelined batch groups**; no `--batch-mtp` or experimental concurrent-prefill switches.
- **16384 MiB / eight parked-history slots**. History limit and concurrent serving slots are different settings; the byte budget is shared, not RAM reserved per conversation.
- Genuine **262144 native context**, INT8/32768 resident KV, split `14,24,36`, BF16 **CPU vision**, `--spec 3 --pipeline-windows 2`, auto prefill and trimmed stage weights retained.
- Normal high thinking/sampling defaults retained. **Solo requests use MTP; concurrent pipelined windows use no drafts.**
- Key-free trusted-LAN **0.0.0.0:8080** by default; Host/Origin protections, explicitly configured optional authentication, rental/process-ownership checks and pidfd stops remain intact. SC117 is still **unqualified/trial-only**, not public hosting.

In the matched study this configuration led short-prompt two-client throughput (**153.0 vs 121.2 total tokens/s**) and ~11.8K-prompt throughput (**57.9 vs 53.9**). Individual concurrent streams were about **86 tokens/s** while both ran. A cold ~29K joining prompt paused its companion for approximately **6.93 seconds**. These are bounded fixture measurements, not guarantees or broad quality/context/stability qualification.

## Separate ownership, cache and controls

Private state/adaptive profile:

```text
~/.local/state/locallm-strata/sc117-iq3s-ple-q8-2slot-trial/
```

The API model ID ends in **`-ple-q8_0-2slot`**. Old single-slot state/KV is neither adopted nor reused. This profile has its own exact identity, counts, Q8 checksum marker and Hive launch-token binding. Wrong-profile/variant/runtime controls fail closed. Existing one-slot authorization gates were not widened.

HostLLM [7] supplies the clean unprivileged trial environment, exact readiness check **before** pausing services, and the normal identity-bound HostLLM lease. It refuses to replace an existing Strata session. Hive [18] uses the existing serialized ownership-checked profile-switch workflow and token-owned wrapper shutdown; it never adopts a manual/replacement process. Hive lifecycle control stays with its owning Hive wrapper.

Q8 is inherent to the new launcher profile: do **not** add the old `--ple-q8` modifier. Both requested/effective slots and explicit groups must be **2/2** at runtime **.46**; normal one-slot profiles remain strict. Startup checks the full Q8 sidecar hash before model/GPU spawn and records only the exact stable spawned-child identity. The Hive wrapper requires four initially idle 3090-sized GPUs and at least 64 GiB initial available host memory; this is conservative admission headroom, **not** proof every long-history combination fits.

## Publication and testing

The narrow existing command can now add this **inactive** bundle too:

```bash
sudo ./install-hive-llm-miner.sh --add-ple-q8-profiles
```

It accepts only identical existing Q8 bundles, stages a complete new directory, atomically updates the selector and verifies `current`, active files, stable entrypoints and miner markers unchanged. No stop/start/switch is performed by publication. The ordinary full installer remains idle-only.

The new numerical arguments, runtime, vision and tuning were independently checked against the actual measured winner's run-config; only API identity, normal API port and isolated state/log paths differ. Offline readiness, root handoff, dispatch, exact-readiness-before-pause, wrong-variant stops, immutable asset checks and add-only publication have dedicated regressions. **Actual root Hive [18] activation passed**, including key-free text/vision/tool-format/Host/Origin checks, two observed concurrent slots, complete streams and eight restored histories. Its token-owned Hive stop and normal restart also passed; the new .46 two-user miner is left **active** at the operator's explicit request, with OctaSpace active and no pause lease/switch journal. Direct [7] dispatch/root handoff/readiness boundaries were regression-tested; a separate direct native session was not started while leaving Hive active. **453 Python tests OK/one skipped; eight shell suites passed.** Actual activation/lifecycle evidence is recorded separately in [the menu/activation receipt](results/sc117-two-slot-menu-20261101.json); source/header readiness is not presented as a live menu launch.
