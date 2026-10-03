# Strata IQ3_S — install-only preparation on `.69`, 2026-10-03

## Status and scope

**Prepared, downloaded, packed and checksum-verified. Actual GPU inference is not yet tested.**
The existing Hive/Wildrig miner continued running on all four RTX 3090s during this work. No `miner stop/start`,
OctaSpace stop/start, Docker stop, driver installation, clock changes, Hive configuration edits or watchdog changes
were performed. Strata is an independent experimental engine: **HostLLM [2]**. Swift 1.5 Uncensored Q8 remains the
Qwen default; Hauhau Q8_K_P + BF16 vision + FastMTP remains the fallback bundle.

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
- Logs: `logs/prepare.log`, plus separate timestamped frontend logs on future launches.
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

## Safe manual start

```bash
cd /home/user/LocalLLM
miner stop                   # manual user action after preparation
./HostLLM.sh                  # choose [2] Strata
# Direct equivalent: ./v1strata.sh --quickstart
```

Wait for health readiness, then use the web UI or `/v1/chat/completions` on port 8080. The model ID is
`qwen3.8-flash-next-iq3_s-strata`; `qwen38` and `strata` are aliases. API thinking values are `none/low/medium/high`:
do not send `xhigh`. Port/host/API-key overrides are `STRATA_PORT`, `STRATA_HOST`, `STRATA_API_KEY`.
Set `STRATA_API_KEY` or bind loopback before exposing it beyond a trusted LAN; the default has no API authentication.

Ctrl+C invokes identity-bound teardown, not a process-name-wide kill. A second terminal may use `./v1strata.sh --stop`
from the same user/root privilege level. Only after stopping the LLM should you manually restart the miner.

Before launch, Docker must be inspectable/empty, GPUs idle and the selected port free. During the foreground lifetime,
the supervisor polls for rental containers, Docker uncertainty and new/unidentified GPU workloads and yields **only
Strata**. It does not stop the miner/renter/container/OctaSpace service. Unknown process-family identities block
signalling rather than guessing. Use the recorded log/state if an abnormal teardown needs inspection.

## Validation completed — CPU-only

1. Native Strata and GPU vision helper compilation succeeded on the actual host; native engine `--help` succeeded on
   its early-exit path without model inference.
2. Both installed model shards and the projector passed full-file SHA-256; original MTP tensors passed offline verify.
3. Readiness checks validated source revision, compiled-runtime hashes, model/projector metadata, native context,
   KV/MTP flags, four-GPU automatic split and configured GPU vision.
4. **25 Linux safety/lifecycle tests passed as the ordinary user and again as root**. These exercise real procfs/pidfds,
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

### Not validated yet

- actual IQ3_S startup/GPU allocation/VRAM peaks on four 3090s;
- native model responses, arithmetic/code, strict JSON/tool calls, image correctness;
- prefill/decode throughput, draft acceptance and completed-task latency;
- populated 262K histories, worst-case memory or full-context generation/quality.

No published RTX 5070/AMD speed or estimated 3090 rate is represented as a `.69` measurement. Keep this engine
experimental until bounded actual-model text/tool/JSON/vision checks succeed after the user's manual miner stop.
Do not use full-context generation as a smoke test.

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
