# LocalLLM — Swift, Hauhau, Strata and Orca

Local HostLLM menu definition for `.69` (4× RTX 3090, 128 GB RAM). The standard Strata route is Orca Q4_K_S `.45` with auto prefill and trimmed `14,24,36` split; the windows-2 override is omitted. Orca `.39`, `.41`, and `.44.1` remain explicit rollback routes. SC117 is an unqualified `.45` trial only, exposed on trusted-LAN `0.0.0.0:8080` without API authentication; it uses CPU vision and dedicated state and is not normal or production hosting. SC117 and Orca use the same port and cannot serve concurrently. The `.45` engine/build is pinned and artifact-checked, but no `.45`-specific Q4 speed/quality claim is made.

```text
  [1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
  [2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
  [3] Orca Q4_K_S — Strata 0.1.45 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted
  [4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.45 | native 262K | --spec 3 + pipeline-windows 2 | CPU vision | one slot

  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit
  [12] Set next Hive miner - Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
  [13] Set next Hive miner - Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
  [14] Set next Hive miner - Orca Q4_K_S .45 (auto prefill, trimmed split; windows-2 override omitted)
  [15] Set next Hive miner - SC117 GSQ-RCO IQ3_S (abliterated) .45 unqualified trial (MTP)
```

## Start hosting

```bash
cd /home/user/LocalLLM
./HostLLM.sh
# Select 1–4. [3] starts the Orca .45 candidate; [4] starts only the isolated SC117 .45 CPU-vision trial.
# Hive actions [12]–[15] safely switch an owned Hive LLM; no manual miner stop is needed.
# No manual miner stop is needed when no live Qwen listener is present and readiness passes.
```

**Starting hosting automatically pauses OctaSpace (`osn.service`) and stops the Hive miner.** This restores the
intended hosting behavior; the earlier install-only update incorrectly removed it. Docker must be inspectable and
empty **before any stop**, and occupied GPUs must be identifiable as the selected Hive miner. A rental container,
unidentified workload, existing LLM or busy API port blocks a new start without stopping that workload.

A persistent pause lease in `~/.local/state/hostllm/pause.json` remembers the previous service/miner state. Hive's
`miner stop/start` consumes `MINER_STOP`, so the controller reasserts that marker after stop while hosting is active.
No driver, clock, watchdog, flight-sheet or Hive configuration changes are made. A live Hive-managed Qwen listener is identified by its process, command-line port, listener PID, and private state; HostLLM checks the actual port for current clients but deliberately refuses to stop it, because it cannot prevent a new client arriving during shutdown. Stop Qwen through its owning launcher/Hive workflow, confirm it is gone, then retry. A manually launched or unverifiable LLM is never stopped as a miner.

- **Strata:** stays in the foreground. Root menus automatically hand options **[3]/[4]** to the unprivileged
  operator (`user` by default); the inference server never runs as root. Ctrl+C stops its identity-verified
  frontend/engine/vision helper, then restores the previous miner/OctaSpace state once the GPUs have actually cleared.
- **Swift/Hauhau:** the dashboard can leave the server running. Leaving the menu keeps mining/OctaSpace paused while
  that server is active. Use **[9]** to stop it and restore the previous state. Stop before switching models.
- The pause survives closing/reopening the menu; **[9]** also performs safe recovery after a failed/interrupted start.
- Restoration fails closed while a rental, unknown GPU process, or incomplete GPU teardown remains. No broad
  `pkill`, `killall`, or reusable-PID sudo-kill fallback is used.
- Reopen the menu after an update; an already-running shell retains its old menu functions.

## LAN web UI and APIs

For the standard Hive Orca `.45` trusted-LAN profile (unauthenticated; use only on a trusted network), these are the `.69` endpoints:

- **Web UI:** <http://192.168.1.69:8080/>
- **OpenAI API:** `http://192.168.1.69:8080/v1` (`/chat/completions`, `/models`)
- **Anthropic API:** `http://192.168.1.69:8080/v1/messages`
- **Orca .45:** HostLLM [3] and Hive [14] bind the trusted LAN on `0.0.0.0:8080` by default. Set `STRATA_HOST=127.0.0.1` before opening HostLLM for local-only Orca. The lower-level Q4 CLI still defaults to loopback.
- **Immediate loading:** [3]/[4] and the Hive profiles selected by [14]/[15] load the model and vision encoder during startup, before opening the API. They do not wait for a chat request, and automatic idle unloading is disabled. “Model loaded and API ready” requires `loaded=true` and the expected native runtime. Loading typically takes a minute or two; keep the launcher open. [Real eager-start/LAN checks for all four routes](results/strata-eager-lan-20261017.json) recorded loaded GPUs before the first chat.

Orca `.45` is the standard Strata route; generic Orca `.39`, `.41`, and `.44.1` remain explicit rollback profiles. SC117 is **fixed** to trusted-LAN `0.0.0.0:8080`, uses its dedicated trial state, and does not permit other binds. Its API is unauthenticated: use only on a trusted, firewalled LAN (for `.69`, `http://192.168.1.69:8080/`). SC117 and Orca are mutually exclusive on port 8080. SC117 remains unqualified and must not be selected for normal hosting; the legacy SC117 `.39` launch path is retired. The original IQ3_S model ID is `qwen3.8-flash-next-iq3_s-strata`; SC117 has its own profile/model ID. Swift/Hauhau retain their existing llama.cpp web UI/API behavior.

Other Strata profiles remain unauthenticated by default: **do not expose them publicly**. HostLLM accepts an optional key at its secure prompt for routes that support LAN binding and forwards it only through stdin; exported `STRATA_API_KEY` is refused. SC117 is an explicit exception: its isolated, unqualified trial binds the trusted LAN on `0.0.0.0:8080` without authentication. Firewall it and never expose it to an untrusted network. Its dedicated state remains isolated, and no normal/qualified SC117 hosting is enabled. No global Pi configuration is edited.

## Retained models and normal settings

| Main choice | Profile | Normal settings on four RTX 3090s |
| --- | --- | --- |
| [1] Swift | `swift15u-q8` | Q8_K_XL weights, shared BF16 vision, embedded native MTP depth 3, Q8 K/V, xhigh, two 262144-token slots |
| [2] Hauhau | `hauhau-q8-fastmtp-q4kv-xhigh` | Q8_K_P weights, BF16 vision, matching FastMTP sidecar depth 4, Q4_0 K/V, xhigh, three 262144-token slots |
| [3] Orca .45 | `orca-q4_k_s` / runtime `0.1.45` | Standard machine-specific profile; native 262K, CPU vision, 8 GiB/four-slot cache, auto prefill, trimmed split 14,24,36, one serving slot. Pipeline-windows-2 override omitted. No .45-specific quality/speed claim. See [ORCA_Q4KS_V01441.md](ORCA_Q4KS_V01441.md) |
| [4] SC117 trial | `iq3_s-sc117-abliterated` | Unqualified Strata .45 trial, CPU vision, native 262144, `--spec 3` + pipeline-windows 2, INT8 KV, one slot/group, trusted-LAN `0.0.0.0:8080` (unauthenticated), dedicated state; no qualification or production authorization |
| Strata standard | `orca-q4_k_s-v01441` | Q4_K_S .45, native 262K, CPU vision, auto prefill, trimmed split 14,24,36, one slot |
| Rollback | `orca-q4_k_s` | Generic Q4 .39 profile, native 262K, CPU vision, one serving slot |
| Rollback files | Orca runtime `0.1.41` | Separate one-slot and experimental 2-slot/2-group profiles remain available by explicit profile name. See [ORCA_Q4KS_V0141.md](ORCA_Q4KS_V0141.md) |

Swift remains the default; Hauhau's previously verified Q4-KV/xhigh preset remains the fallback. Dense Qwen runtime
pins, weights and normal flags are unchanged. Hauhau's alternative Q8-KV preset shares those retained weights and
remains CLI-only.

Strata is a separate **Flash-Next** engine, not an engine for the retained dense 27B Swift/Hauhau GGUFs. The original base IQ3_S CLI profile and rollback assets remain unchanged; HostLLM [4] is the separate SC117 abliterated `.45` CPU-vision trial and remains unqualified. The IQ3_XXS weights/pack were removed. Q4_K_S uses a separately pinned CPU-vision helper; one generated synthetic image passed, but broader vision quality is unqualified. Profiles default to one request at a time
and expose thinking `none/low/medium/high` (**not `xhigh`**).
The former `.69` baseline used **0.1.39 single-slot**; that profile remains rollback-only. The standard Strata choice is now Orca Q4_K_S `.45`. Guarded two-slot
batching is opt-in; see [STRATA_V0139.md](STRATA_V0139.md) for historical speed/latency and discovered restrictions.
**0.1.41 was prepared/tested separately, but not promoted:** its new four-way auto placement regressed uncached prefill
by about 24% for IQ3_S and 37% for IQ3_XXS. See [STRATA_V0141.md](STRATA_V0141.md) for real results and optional use.
No context extension or experimental CVec/speed projection is enabled in the current IQ3_S profile. Its configured
2048 MiB reserve and native INT8 streaming policy are retained; sampled auto-prefill free VRAM can be lower than the
configured reserve. Historical IQ3_XXS tuning results remain documented, but that model/pack is not runnable on `.69`.

## Orca Q4_K_S: bounded live probes recorded

See [ORCA_Q4KS.md](ORCA_Q4KS.md). The generic Orca `.39`, `.41`, and `.44.1` routes remain explicit rollbacks; `.45` is the standard Strata choice.
The separate Q4 .41 candidate remains available explicitly through the Hive `orca-q4_k_s-v0141*` profile names; it has no visible HostLLM menu row. Its
source/build/config/parity/evidence paths are separate. It starts at 32K text-only and cannot migrate context or enable
CPU vision until its own identity-bound 32K validator succeeds. Start that isolated candidate with
`./HostLLM.sh --run-q4ks141-32k-validation`, then run `./v1strata.sh --validate-orca-q4ks-141-32k` in a second terminal;
the startup remains loopback-only and uses the standard rental/idle and lease gates. The validator resends full A/B
conversation histories and requires its own `/metrics` deltas to show a park, restore, and reused tokens under the
8192 MiB/four-cache-slot budget. Four is a slot cap, not proof that four 262K snapshots fit in 8 GiB; large-snapshot
and multi-user capacity remain unmeasured. After gated native-context/CPU-vision migration, `./v1strata.sh --validate-orca-q4ks-141-native`
provides a bounded synthetic-image and near-full-context tail-marker probe with RAM/GPU safeguards. The candidate
presets retain one serving slot even though the conversation-cache configuration has four slots. The Hive profile explicitly requests trusted-LAN binding. See
[ORCA_Q4KS_V0141.md](ORCA_Q4KS_V0141.md). The .39 32K text-only health/arithmetic/strict-JSON/tool/memory/MTP gate passed
(38 MTP drafts, 36 accepted); after the identity-bound record, its profile was migrated to native 262144 / INT8 /
32768 resident and CPU vision. One synthetic magenta-square image was identified correctly, and a 260,996-token
prompt (99.56% of the configured window) returned its tail marker. Separately, on 2026-10-11 the .41 candidate passed
its own 32K gate (35 MTP drafts, 31 accepted) and one bounded native/CPU-vision run: a synthetic magenta image passed,
and a counted 261,000-token prompt (99.56%) returned its tail marker in 264.6 s. Minimum available RAM was 43,615 MiB;
per-GPU peak usage was 23,740 / 19,502 / 22,410 / 23,316 MiB. These single-run checks do not establish general
context/vision quality, snapshot capacity, worst-case memory, throughput, or sustained/multi-user capacity. The raw reports remain on `.69` under the Q4 profile/staging
paths. Readiness runs before HostLLM pauses anything; the menu leaves an already-running Strata instance alone.
No generic runtime selection is changed.

The standard machine-specific Q4 .45 profile is selected with HostLLM [3], Q4 runtime `auto`, or Hive profile
`orca-q4_k_s-v01441`. It checks the pinned .45 source/build/runtime artifacts and exact effective config before the
HostLLM pause or Hive pointer change. The model preset/evidence derive from the prior .44.1 route; no .45-specific
inference, quality, or speed result is claimed. The .39/.41/.44.1 routes remain explicit rollbacks. Paired screening supported `--prefill auto`
and `--trim-stage-weights` with split `14,24,36`; the corrected 95% interval for pipeline-windows 2 crosses zero.
The pinned capability report used that exploratory override, but the launcher removes it from the effective serving
config; this is not a separate performance qualification of the resulting default-window run. The candidate's bounded
evidence and limitations are documented in [ORCA_Q4KS_V01441.md](ORCA_Q4KS_V01441.md); they are not a broad speed,
quality, or sustained-load qualification.

The IQ3_XXS model and pack were removed from `.69` to reclaim disk and are not a runnable rollback. The previously verified Swift/Qwen Q8 profile remains the explicit fallback. Do not treat retained IQ3_XXS metadata as a usable rollback unless its assets are restored and readiness passes.

## Pi context metadata

The legacy `llamacpp-model-sync` extension must read the server's **`meta.n_ctx`** (served context), not only
`meta.n_ctx_train`. Otherwise Strata/Orca alias entries fall back to 128000 in Pi despite a larger server window.
The corresponding Pi-profile fix gives the live per-sequence allocation precedence over stale static/training
metadata. Reload Pi/reselect the model after installing it. No global model/settings edits are necessary.

## Strata preparation and validation

See [HOWTOTUNE.md](HOWTOTUNE.md) for the reusable Strata measurement/safety protocol and historical SC117 `.44.1` trial results. Those measurements are not a `.45` speed comparison. The measured `--pipeline-windows 2 --spec 3` settings are retained for SC117 HostLLM [4]/Hive [15] only; they remain unqualified and do not affect other profiles.

See [STRATA_IQ3S.md](STRATA_IQ3S.md) for the original profile's source/model pins, full-file checksums, installed paths and validation scope, and [STRATA_SC117_IQ3S.md](STRATA_SC117_IQ3S.md) for the isolated abliterated model pins and hard qualification gate.
The preserved Strata 0.1.38 baseline is pinned to `99f3dbd0b21d1401b3769e0c0d963913607f380b`; the side-by-side
0.1.39 runtime is pinned to `6f32ec070f23ced9f50e704d854d775da52591ab` (see [upgrade/benchmark report](STRATA_V0139.md)).
Optional 0.1.41 is pinned to `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`; explicit `--prepare-runtime --runtime 0.1.41`
is build-only and does not alter the selected runtime or an active engine. Strata 0.1.45 is pinned to
`477ac13cbe8e582c20484871ce459d20d82df09d`, restricted to Orca Q4_K_S and the unqualified SC117 trial, and is
never a generic/default runtime choice. Its explicit `--prepare-runtime --runtime 0.1.45` build records CUDA sm_86,
Q4/native-expert options, and a separate native-CPU vision helper; it does not run inference or change an active service.
The IQ3_S publisher revision is
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. Native engine and GPU vision helper were built for `sm_86` using existing
CUDA 12.9 and a private Python/CMake environment, without replacing system tools/drivers.

Real four-GPU startup and bounded arithmetic, strict JSON, exactly-one tool call, color-image vision and high-thinking
responses have now passed. Compilation/install-only checks alone were not adequate runtime validation. **No
full-context generation was run**, and populated-262K quality/worst-case peaks remain unmeasured. Tiny-request timing
is not a full-context benchmark or a general speed promise.

Ordinary starts never download/update Strata. Explicit install-only preparation on another compatible four-3090
Linux host uses `./v1strata.sh --prepare`; it never starts inference/calibration or installs system build tools. This
requires git, Python 3.10+ with venv/ensurepip, gcc/g++, and existing `/usr/local/cuda-12.9/bin/nvcc`.

The port gate permits TCP TIME_WAIT after an owned server stops (SO_REUSEADDR), but still refuses a live listener.
The currently configured Hive miner is resumed, not a hard-coded or historical Wildrig PID/bundle.

Raw `v1strata.sh --quickstart` is a low-level launcher, not the hosting pause controller: use **HostLLM** for automatic
miner/OctaSpace management. `v1strata.sh --status`, `--check-ready`, and `--stop` inspect/control only the owned runtime.
Logs are separate timestamped `server-*.log` and `server-*-engine.log` files under the Strata data root. Stopping an
owned CUDA task recognizes kernel `PF_EXITING` as terminal, even before `/proc/<pid>/exe` and its zombie state agree.

## Cleanup and historical tools

Only these Qwen bundles remain under `~/.local/share/localllm-qwen38/models/`:

- `hauhau/`: Q8_K_P, BF16 projector, matching FastMTP sidecar;
- `swift15-uncensored/`: Q8_K_XL and its shared BF16 projector.

Strata model/pack/MTP assets are separate under `~/.local/share/localllm-strata/`. Its publisher-identical PLE second
shard and BF16 projector were reused by hard link and fully hashed. Retired weights were removed: both BF16 Swift
comparisons, UkisAI Swift 27B Q8, TURBO, DFlash2, GSQ Q2 and GLM EXL3. **275.23 GiB reclaimed physically; about
655.30 GiB free** after preparation. Retained Qwen file identities/ownership were unchanged.

Archived comparison profiles remain explicit CLI-only and can re-download removed weights. Do not use the legacy
`--speed-test-all` for this curated collection. Profile-specific smoke/speed modes use 4K allocation and reasoning
off, not normal native-context performance. Legacy `v1llama_cpp.sh`/`v1glm53.sh` are not shared-host menu choices;
their old stop/benchmark paths are not covered by the maintained identity-bound guards.

The optional [Hive custom LLM miner](hive-llm-miner/README.md) is separate from the HostLLM engine picker. HostLLM actions [12]–[15] select Swift, Hauhau, Orca `.45`, and the isolated SC117 `.45` CPU-vision trial. Orca `.45` is the standard Hive Strata selection; `.39`, `.41`, and `.44.1` are explicit rollback routes. If the old SC117 `.39` or generic `.39` pointer exists, installation migrates it to standard `.45` before replacing the SC117 profile directory. SC117 remains unqualified and uses trusted-LAN `0.0.0.0:8080` without authentication, with dedicated state; it shares port 8080 exclusively with Orca. The low-level root-only profile menu changes one atomic pointer after idle/readiness checks and never manages services itself. HostLLM wraps [12]–[15] in a serialized switch: verify that all LLM/GPU processes belong to the installed Hive wrapper, pause OctaSpace before stopping Hive, hold `MINER_STOP` during readiness/package refresh, change the pointer, then restore the prior miner/OctaSpace running state. A current unchanged profile returns immediately without stopping or reloading. Rentals, unrelated GPU jobs, and manual HostLLM sessions are refused before any stop. A private `hive-profile-switch.json` recovery journal is retained if safe restoration cannot complete; recover with `sudo python3 hive_profile_switch.py --recover`. Do not run a switch during a rental.

## Checks and references

```bash
bash -n HostLLM.sh v1qwen38.sh v1strata.sh hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh
bash tests/retained-menu-test.sh
bash tests/swift15-profile-test.sh
bash tests/swift15-uncensored-profile-test.sh
bash tests/gsq-profile-test.sh
bash tests/hive-miner-profile-menu-test.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Linux tests use CPU-only fixtures and mocked system/miner boundaries; live GPU/API/LAN checks are documented
separately. [Hive and root-menu smoke results](results/hive-hostllm-live-smoke-20261017.json) record real short
prompt/reply tests for [3], [4], [14], and [15], not only readiness or `/health` checks. Historical tuning: [Swift two-slot](SWIFT15U_2SLOT_2026-09-26.md), [Swift source/MTP](SWIFT15.md),
[retired GSQ Q2](GSQ_TUNING_RESULTS.md), [September speed research](SPEED_REPORT_2026-09-24.md).
Strata is MIT-licensed; model weights retain their publisher licenses, including Swift Open License terms.
