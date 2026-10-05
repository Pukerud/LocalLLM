# Strata IQ3_S — preparation and live hosting follow-up on `.69`, 2026-10-03

**2026-10-05 runtime follow-up:** [STRATA_V0139.md](STRATA_V0139.md) records the side-by-side 0.1.39 promotion,
real single/two-slot measurements and guarded batching restrictions. The 0.1.38 preparation below remains the
unchanged model/config/runtime rollback baseline; its historical evidence was not rewritten.

## Status and scope

**Prepared and checksum-verified; real four-GPU startup and bounded model/API/vision tests now pass.**
The initial installation/cleanup left mining running. The user subsequently authorized miner/OctaSpace pausing for
actual inference testing and requested restoration of automatic hosting pause behavior. HostLLM now presents
**[1] Swift, [2] Hauhau, [3] Strata**, with a persistent automatic miner/OctaSpace pause lease and safe restoration.
No driver, clock, watchdog, Hive configuration or renter/container changes were made. Swift remains the default;
Hauhau Q8_K_P + BF16 vision + FastMTP remains its preserved fallback.

## Pins and installed layout

- Engine: [Niko1221/Strata 0.1.38](https://github.com/Niko1221/Strata/tree/99f3dbd0b21d1401b3769e0c0d963913607f380b),
  source commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
- GGUF publisher: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09),
  revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, original model, **IQ3_S**.
- Data/runtime root: `/home/user/.local/share/localllm-strata`.
- Source/private environment: `source-99f3dbd0b21d/` and its `.venv/`.
- Prepared config: `source-99f3dbd0b21d/strata-iq3_s.json`.
- Models: `data/models/IQ3_S/`; projector: `data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf`.
- Packed model: `data/packs/iq3_s/`; original Flash-Next MTP: `data/mtp/`, packed MTP: `data/mtp/rt/`.
- Readiness evidence: `prepared.json`; preparation exit status `prepare.exit` was **0**.
- Logs: `logs/prepare.log`, timestamped `server-*.log` frontend and `server-*-engine.log` native-engine logs.
- `prepared.json` records install-only checks; the later live follow-up does not rewrite that historical manifest.
- Runtime state: `~/.local/state/locallm-strata/` (intentional single-l `locallm`, like the existing Qwen state path).

Both native executables (Strata and GPU vision helper) were compiled in Release mode for `sm_86` using the existing
CUDA **12.9.86**, GCC **11.4** and private-environment CMake. The system CMake 3.22 was not replaced. Compilation used
at most four workers, low CPU/I/O priority. Build/setup callbacks blocked system build-tool installation,
calibration and any automatic inference start. The repository's explicit `v1strata.sh --prepare` provides the same
install-only flags and guards; ordinary start never downloads or updates anything.

Full-file SHA-256 checks succeeded on the installed GGUF files, not just publisher metadata:

| Asset | Exact bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |

The second shard is the publisher-identical PLE data; it and the BF16 projector were reused from the old GSQ bundle
through **hard links**, then fully hashed in their new location. Removing the old GSQ directory did not remove either
retained Strata asset. The original Flash-Next MTP download also passed the pinned tools' offline `mtp_fetch.py verify`.
The compiled native executables have recorded SHA-256 hashes in `prepared.json`.

## Prepared runtime settings

- GPU list `[0,1,2,3]`, **automatic contiguous layer split**; no tensor-parallel assumption or P2P requirement.
- Native maximum context **262144**; no rope-scaled extension.
- **INT8 KV**, automatic KV streaming (setup reports approximately 3.6 GB host KV for this context).
- BF16 projector, **GPU vision**, image support; no video support.
- Model-appropriate Flash-Next MTP, Strata `--spec 4`; not the dense Hauhau/Swift head or DFlash draft.
- Low-RAM mode off; 2048 MiB VRAM reserve; experimental speed projection/CVec off.
- Default API binding `0.0.0.0:8080`, maximum supported thinking level **high**.
- One serial request at a time. Normal sampling: temperature 1.0, top-p 0.95, top-k 20.

Existing dense 27B Swift/Hauhau GGUFs are **not Strata-compatible**. Strata's optional Swift 1.5 **Flash-Next** model is a
different fine-tune from the retained Swift 1.5 **27B** model. No other Strata quantizations or families were downloaded.

## Native context and Pi detection — 2026-10-04

Option [3] still enforces **262144**, INT8 KV and **32768 resident**; the 32K GPU working window is not a 32K total
context. The observed 128000 Pi window came from the legacy sync extension reading only `meta.n_ctx_train`, while
Strata publishes the effective limit as `meta.n_ctx`. The Pi-profile extension now gives the served per-sequence
allocation precedence. Reload Pi/reselect the model after updating the extension; no global model/settings edit is
required. Native context includes prompt/history, tools, image tokens, reasoning and answer tokens together.

Both IQ3_S and the separate Orca profile were rechecked live at 262144 with bounded arithmetic/JSON/tool/swapped-image
requests and real API-to-Pi-registration metadata tests. The source, original IQ3_S config/assets and runtime flags
were not changed to achieve this. See [ORCA_IQ3XXS.md](ORCA_IQ3XXS.md) for the Orca native-context migration and scope.
No full-context generation or context beyond native was performed.

## Automatic hosting start

```bash
cd /home/user/LocalLLM
./HostLLM.sh                  # choose [3] Strata; miner/OctaSpace pause automatically
```

The menu checks rentals/unknown workloads before changing any services. A persistent lease remembers previous
miner/service state; `MINER_STOP` is reasserted after Hive consumes it. Raw `v1strata.sh --quickstart` is a low-level
launcher without that pause controller; use HostLLM for normal hosting.

Wait for health readiness, then use **http://192.168.1.69:8080/** or `/v1/chat/completions`. The default binding is
`0.0.0.0:8080`, not localhost; the web UI, health and model list have been verified from a different LAN machine. The model ID is
`qwen3.8-flash-next-iq3_s-strata`; `qwen38` and `strata` are aliases. API thinking values are `none/low/medium/high`:
do not send `xhigh`. Port/host/API-key overrides are `STRATA_PORT`, `STRATA_HOST`, `STRATA_API_KEY`.
Set `STRATA_API_KEY` or bind loopback before exposing it beyond a trusted LAN; the default has no API authentication.

Ctrl+C invokes identity-bound teardown, not a process-name-wide kill. A second terminal may use `./v1strata.sh --stop`
from the same user/root privilege level. The menu restores the previous miner/OctaSpace state after confirmed GPU
teardown; [9] also recovers the persistent lease if the menu was closed/interrupted.

Before launch, Docker must be inspectable/empty, GPUs idle and the selected port free. During the foreground lifetime,
the supervisor polls for rental containers, Docker uncertainty and new/unidentified GPU workloads and yields **only
Strata**. Only the separate hosting controller pauses the selected Hive miner/OctaSpace; neither component stops
renters/unknown workloads. Unknown process-family identities block signalling rather than guessing. `PF_EXITING`
CUDA tasks are recognized as terminal before zombie state; external stop is serialized against the runtime watcher,
so draining CUDA PIDs are not misreported as a new workload. Frontend and native-engine logs are separate.

## Initial preparation validation — CPU-only

1. Native Strata and GPU vision helper compilation succeeded on the actual host; native engine `--help` succeeded on
   its early-exit path without model inference.
2. Both installed model shards and the projector passed full-file SHA-256; original MTP tensors passed offline verify.
3. Readiness checks validated source revision, compiled-runtime hashes, model/projector metadata, native context,
   KV/MTP flags, four-GPU automatic split and configured GPU vision.
4. The initial **25 Linux safety/lifecycle tests passed as the ordinary user and again as root**. The follow-up suite
   is now **43 tests** including mocked miner/OctaSpace pause/restore boundaries and shutdown-race regressions. These exercise real procfs/pidfds,
   terminal/zombie handling, refused mismatched PID/start-time/model identities, Docker/GPU/port gates, private config,
   external stop and foreground reaping. The fake `llama-server` and HTTP fixtures are CPU-only.
5. The **pinned upstream frontend in `--engine mock` mode** passed a bounded known API response (`ANSWER=42`) and
   `/v1/models`. This tests dependencies/API plumbing, **not the IQ3_S model**.
6. Retained-menu, Swift 1.5, Swift Uncensored and archived GSQ profile regression scripts passed; no model downloads or
   GPU generations were performed by those scripts.
7. The miner's PID/start-time/executable identity remained unchanged during cleanup; the retained Qwen file
   device/inode/size/mtime/UID/GID/mode identities and Strata model assets matched before/after cleanup.
8. `osn.service` remained active and protected Hive configuration hashes matched before/after cleanup.

The old, unfinished Hauhau runtime benchmark/lifecycle matrix was **not reused or declared cleared** by these checks.

## Authorized live GPU follow-up

- Native-context four-3090 startup with BF16 GPU vision and the pinned original MTP succeeded after pausing both
  the miner and OctaSpace. The automatic split was layers 0–12 / 13–24 / 25–35 / 36–47.
- Completed bounded requests passed: `ANSWER=42` arithmetic (6 output tokens), strict JSON (30), exactly one
  `get_weather(city=Oslo)` call (27), red/blue image classification (20), and high reasoning with correct arithmetic
  (57 tokens including reasoning, a 128-token reasoning budget, 512-token total cap).
- Those tiny requests took approximately 0.54–1.08 seconds, with response-reported decode around 61–125 tokens/s.
  These are small completed functional checks, **not a representative speed benchmark or full-context result**.
- The running LAN server returned HTTP 200 for `/`, `/health`, and `/v1/models` from another LAN machine; model
  metadata exposes text/image input and native context 262144.
- The updated three-choice menu successfully paused active OctaSpace and its configured **Swift LLM Hive miner**,
  then started the actual IQ3_S server on LAN port 8080. Live-supervisor/private-state proof distinguishes that miner
  from a manual Qwen hosting session; no Hive flight-sheet/configuration was changed.
- Nine additional bounded LAN checks passed: arithmetic, strict JSON, one tool, yellow/green vision, swapped-image
  vision (same text, different image), high reasoning, Python code, OpenAI SSE and Anthropic messages. A separate
  real-browser chat returned `UI_OK` (3 output tokens); only the harmless upstream missing-favicon 404 was logged.
- After those short requests, observed GPU used/free MiB were 21931/2196, 20655/3472, 20463/3664, 19359/4765.
  This is one post-request observation, **not a peak or populated-context guarantee**.
- Identity-bound external stop completed cleanly without a false new-workload warning. The menu reaped its frontend,
  exited **0**, removed the pause lease, and restored OctaSpace plus the configured Hive miner. A second real startup
  and identity-bound **SIGINT (Ctrl+C equivalent)** to its foreground supervisor likewise exited 0, reaped the
  frontend, cleared GPUs and restored the prior state.
- Port checks now accept TIME_WAIT after owned-server teardown using SO_REUSEADDR, while still blocking live
  listeners. Shutdown preserves original errors, serializes external stop versus the watcher, and retains recorded
  identities while CUDA resources drain. No manual miner stop is required.

### Still unmeasured

Populated 262K histories, worst-case image/memory peaks, sustained representative throughput and full-context quality.
No full-context generation was performed; do not use it as a smoke test. No other-hardware published speed is treated
as a `.69` measurement. The separate old Hauhau benchmark/lifecycle matrix remains uncleared.

## Authorized cleanup result

Only model assets were removed: older Swift BF16, Swift Uncensored BF16 **weights only**, UkisAI Swift 27B Q8,
TURBO Q8, DFlash2, GSQ Q2 and GLM-5.3-Flash EXL3. Explicit targets were checked for live command/open-file/mapping
references, quarantined by rename, rechecked, then deleted. Neither retained Qwen directory/projector was removed.

**275.23 GiB physically reclaimed; 655.30 GiB free** after the already-completed IQ3_S/runtime preparation.
About 302.90 GiB of old paths were removed logically; the PLE/projector bytes remain in Strata by hard link.
The private file-identity/cleanup ledger is `~/.local/share/localllm-strata/cleanup-20261003.json`.
Unused comparison build environments/history were not broadly deleted.

## Primary sources

- [Models / quantization support](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/MODELS.md)
- [Multi-GPU layer split and transfers](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/MULTI_GPU.md)
- [Installation](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/docs/INSTALL.md)
- [Pinned setup code and model revisions](https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/setup.py)
