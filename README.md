# LocalLLM — Swift, Hauhau, Strata and Orca

Four directly selectable models on `.69` (4× RTX 3090, 128 GB RAM); Orca is experimental:

```text
  [1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
  [2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
  [3] Strata IQ3_S — BF16 GPU vision | MTP | high | INT8 KV | native 262K | four-GPU split
  [4] Orca Uncensored IQ3_XXS (Strata) — experimental | MTP | high | native 262K | INT8 streaming KV | F16 GPU vision

  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit
```

## Start hosting

```bash
cd /home/user/LocalLLM
./HostLLM.sh
# Select 1, 2, 3 or 4. No manual miner stop is required.
```

**Starting hosting automatically pauses OctaSpace (`osn.service`) and stops the Hive miner.** This restores the
intended hosting behavior; the earlier install-only update incorrectly removed it. Docker must be inspectable and
empty **before any stop**, and occupied GPUs must be identifiable as the selected Hive miner. A rental container,
unidentified workload, existing LLM or busy API port blocks a new start without stopping that workload.

A persistent pause lease in `~/.local/state/hostllm/pause.json` remembers the previous service/miner state. Hive's
`miner stop/start` consumes `MINER_STOP`, so the controller reasserts that marker after stop while hosting is active.
No driver, clock, watchdog, flight-sheet or Hive configuration changes are made. The configured Hive miner can
itself be Swift LLM hosting: its live `llm-hosting/h-run.sh` supervisor and matching private state prove ownership,
so it is safely paused too. A manually launched LLM is not mistaken for that miner.

- **Strata:** stays in the foreground. Ctrl+C stops its identity-verified frontend/engine/vision helper, then restores
  the previous miner/OctaSpace state once the GPUs have actually cleared.
- **Swift/Hauhau:** the dashboard can leave the server running. Leaving the menu keeps mining/OctaSpace paused while
  that server is active. Use **[9]** to stop it and restore the previous state. Stop before switching models.
- The pause survives closing/reopening the menu; **[9]** also performs safe recovery after a failed/interrupted start.
- Restoration fails closed while a rental, unknown GPU process, or incomplete GPU teardown remains. No broad
  `pkill`, `killall`, or reusable-PID sudo-kill fallback is used.
- Reopen the menu after an update; an already-running shell retains its old menu functions.

## LAN web UI and APIs

After **[3] Strata** reports ready:

- **Web UI:** <http://192.168.1.69:8080/>
- **OpenAI API:** `http://192.168.1.69:8080/v1` (`/chat/completions`, `/models`)
- **Anthropic API:** `http://192.168.1.69:8080/v1/messages`

Strata binds **`0.0.0.0:8080`** by default, not loopback, so other LAN devices can reach it. Like llama.cpp, one engine
owns port 8080 at a time. Its configured model ID is `qwen3.8-flash-next-iq3_s-strata`; `qwen38` and `strata` are aliases.
Swift/Hauhau retain their existing LAN binding and llama.cpp web UI/API.

The default trusted-LAN API has no authentication: **do not expose it publicly**. Set `STRATA_API_KEY` before starting
for authenticated access, or `STRATA_HOST=127.0.0.1` for local/tunnel-only access. `STRATA_PORT` overrides Strata's port.
Those settings belong to the launched process; no global Pi configuration is edited.

## Retained models and normal settings

| Main choice | Profile | Normal settings on four RTX 3090s |
| --- | --- | --- |
| [1] Swift | `swift15u-q8` | Q8_K_XL weights, shared BF16 vision, embedded native MTP depth 3, Q8 K/V, xhigh, two 262144-token slots |
| [2] Hauhau | `hauhau-q8-fastmtp-q4kv-xhigh` | Q8_K_P weights, BF16 vision, matching FastMTP sidecar depth 4, Q4_0 K/V, xhigh, three 262144-token slots |
| [3] Strata | Original Flash-Next GSQ-RCO IQ3_S | BF16 GPU vision, appropriate Flash-Next MTP `--spec 4`, INT8 KV, high reasoning, native 262144, automatic contiguous four-GPU layer split |
| [4] Orca | `orca-iq3_xxs` | Separate uncensored Flash-Next IQ3_XXS pack/tokenizer, original Flash-Next MTP, INT8 streaming KV (32768 resident), high, native 262144, four-GPU split; F16 vision configured |

Swift remains the default; Hauhau's previously verified Q4-KV/xhigh preset remains the fallback. Dense Qwen runtime
pins, weights and normal flags are unchanged. Hauhau's alternative Q8-KV preset shares those retained weights and
remains CLI-only.

Strata is a separate **Flash-Next** engine, not an engine for the retained dense 27B Swift/Hauhau GGUFs. It defaults to
one request at a time, supports **images but not video**, and exposes thinking `none/low/medium/high` (**not `xhigh`**).
The prepared `.69` node now selects **0.1.39 single-slot**, with the original 0.1.38 rollback retained. Guarded two-slot
batching is opt-in; see [STRATA_V0139.md](STRATA_V0139.md) for measured speed/latency and discovered restrictions.
No context extension or experimental CVec/speed projection is enabled. The configured 2048 MiB reserve and native
INT8 streaming policy are retained; sampled auto-prefill free VRAM can be lower than the configured reserve.

## Orca: prepared separately, no live model switch

See [ORCA_IQ3XXS.md](ORCA_IQ3XXS.md). **[4] never automatically stops a running Strata instance.** Preparing Orca is
CPU/download-only, uses its own pinned shards/projector and compatibility pack/tokenizer, and neither rebuilds nor
updates the shared runtime. Readiness is checked before any hosting pause; missing preparation cannot stop mining.
Orca uses native 262144 with INT8 streaming KV (32768 resident), and shares Strata's ownership/state/port.
The default remains one FIFO sequence; optional 0.1.39 batching uses slots in this same engine, not duplicate workers. Existing 32K preparations require an explicit cached
`sudo ./v1strata.sh --configure-native-context --profile orca-iq3_xxs` after updating; ordinary launches do not rewrite configs. Native allocation is not a populated-context quality claim. It is not a concurrent extra worker or a replacement for [3].

## Pi context metadata

The legacy `llamacpp-model-sync` extension must read the server's **`meta.n_ctx`** (served context), not only
`meta.n_ctx_train`. Otherwise Strata/Orca alias entries fall back to 128000 in Pi despite a larger server window.
The corresponding Pi-profile fix gives the live per-sequence allocation precedence over stale static/training
metadata. Reload Pi/reselect the model after installing it. No global model/settings edits are necessary.

## Strata preparation and validation

See [STRATA_IQ3S.md](STRATA_IQ3S.md) for source/model pins, full-file checksums, installed paths and validation scope.
The preserved Strata 0.1.38 baseline is pinned to `99f3dbd0b21d1401b3769e0c0d963913607f380b`; the side-by-side
0.1.39 runtime is pinned to `6f32ec070f23ced9f50e704d854d775da52591ab` (see [upgrade/benchmark report](STRATA_V0139.md)).
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

The optional [Hive custom LLM miner](hive-llm-miner/README.md) remains separate: its direct Qwen launcher still leaves
`osn.service` running for Hive/OctaSpace rental handoffs. This update does not install it or change the selected miner.

## Checks and references

```bash
bash -n HostLLM.sh v1qwen38.sh v1strata.sh
bash tests/retained-menu-test.sh
bash tests/swift15-profile-test.sh
bash tests/swift15-uncensored-profile-test.sh
bash tests/gsq-profile-test.sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Linux tests use CPU-only fixtures and mocked system/miner boundaries; live GPU/API/LAN checks are documented
separately. Historical tuning: [Swift two-slot](SWIFT15U_2SLOT_2026-09-26.md), [Swift source/MTP](SWIFT15.md),
[retired GSQ Q2](GSQ_TUNING_RESULTS.md), [September speed research](SPEED_REPORT_2026-09-24.md).
Strata is MIT-licensed; model weights retain their publisher licenses, including Swift Open License terms.
