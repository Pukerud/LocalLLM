# LocalLLM — retained Qwen models and experimental Strata

Local NVIDIA-GPU inference on `.69` (four RTX 3090s, 128 GB RAM).
**The menu does not stop/start Hive miners, change drivers/clocks/watchdog settings, or pause `osn.service`.**
Stop the miner yourself before launching an LLM. Docker uncertainty, running containers, occupied GPUs, or an
occupied API port block a new launch.

## Current menu

```text
HostLLM — Engine Picker
  [1] Qwen profiles — Swift 1.5 Uncensored Q8 (default) / Hauhau Q8 FastMTP
  [2] Strata IQ3_S — EXPERIMENTAL | BF16 GPU vision | native 262K | INT8 KV
  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit
```

The Qwen submenu contains only the two retained, installed bundles:

| Choice | Profile | Normal settings on four RTX 3090s |
| --- | --- | --- |
| Qwen [1] | `swift15u-q8` — Swift 1.5 Uncensored Q8_K_XL | BF16 vision, native MTP depth 3, Q8 K/V, xhigh, two 262,144-token slots |
| Qwen [2] | `hauhau-q8-fastmtp-q4kv-xhigh` — Hauhau Q8_K_P, proven fallback | BF16 vision, matching FastMTP sidecar depth 4, Q4_0 K/V, xhigh, three 262,144-token slots |
| HostLLM [2] | Strata — original Flash-Next GSQ-RCO IQ3_S | BF16 GPU vision, its own Flash-Next MTP runtime, INT8 KV, high reasoning, one serial request, native 262,144 context, four-GPU automatic layer split |

Strata is a **separate engine**, not a new quantization of the dense 27B Swift/Hauhau models. The two Qwen bundles,
their existing runtime pins, vision projectors and normal inference flags are unchanged. The menu selects Hauhau's
previously verified Q4-KV/xhigh preset; its Q8-KV alternative remains CLI-only and shares the same retained weights.

## Start on the prepared node

```bash
cd /home/user/LocalLLM
miner stop                 # your manual action, not an installer/menu side effect
./HostLLM.sh
# Select [2] for Strata, or [1] then a retained Qwen profile.
```

Strata remains in the foreground, shows loading/health progress and writes a persistent log. Open
`http://192.168.1.69:8080` after it becomes healthy. OpenAI base URL: `http://192.168.1.69:8080/v1`.
Anthropic endpoint: `/v1/messages`. The configured model name is `qwen3.8-flash-next-iq3_s-strata`; `qwen38` and
`strata` are accepted aliases.

**Ctrl+C stops only the identity-verified Strata frontend, engine and vision helper**, then returns to the menu.
Start the miner again manually only after the LLM has stopped. Nothing automatically resumes mining.
If an engine was started as root, stop it from the same root shell (or with `sudo`).

The default API binding is `0.0.0.0:8080`, matching the existing private-LAN workflow. It is unauthenticated unless
an API key is set: **do not expose it publicly**. For Strata:

```bash
STRATA_HOST=127.0.0.1 ./v1strata.sh --quickstart    # local/tunnel only
# Or set STRATA_API_KEY in the environment before a trusted-LAN start.
./v1strata.sh --status
./v1strata.sh --check-ready
./v1strata.sh --stop
```

## Strata IQ3_S: preparation and limits

See [STRATA_IQ3S.md](STRATA_IQ3S.md) for pins, checksums, setup provenance and validation boundaries.
The prepared profile uses:

- [Niko1221/Strata](https://github.com/Niko1221/Strata), engine 0.1.38, pinned source `99f3dbd0b21d1401b3769e0c0d963913607f380b`;
- [ISTA-DASLab/Flash-Next GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S), with both GGUF shards and its own BF16 projector;
- the original Flash-Next MTP tensors, fetched/checked and packed by the pinned Strata tools;
- existing CUDA 12.9, `sm_86`, Release builds, at most four low-priority build workers;
- a private Python environment; no global pip/apt installation or driver/toolkit replacement;
- no experimental speed projection, no context extension, INT8 KV with Strata's native-context streaming policy.

**GPU inference is deliberately deferred to the user's first manual launch while the miner remains running during
installation.** Compilation, checksums, configuration and CPU-only API/lifecycle checks are not speed, vision-quality,
VRAM-peak or full-context-generation measurements. No full-context generation was sent.

Differences from the Qwen runtime:

- Strata serves **one request at a time**, not two/three parallel generation slots.
- It supports images; **not video**. Its highest exposed thinking level is `high`, not `xhigh`.
- 262,144 is native. Its optional 384K/512K modes are experimental rope-scaled extensions and are not enabled here.
- Performance depends on CPU, PCIe links, cache residency, draft acceptance and actual prompt length.
  Other machines' published throughput is not a measurement on this host.
- The foreground supervisor yields only its own Strata processes if Docker becomes uncertain, a rental container
  appears, or a new/unidentified GPU workload appears. It never kills that other workload.

To prepare the same profile on another compatible four-3090 Linux host, explicitly run:

```bash
./v1strata.sh --prepare
```

This installs **without starting inference or calibration**, verifies all model/projector SHA-256 hashes and creates
`prepared.json`. It requires Python 3.10+ with venv/ensurepip, git, gcc/g++, and the already-installed
`/usr/local/cuda-12.9/bin/nvcc`. Missing system tools cause failure rather than a system package/driver installation.
Preparation may continue alongside a bare Hive miner; unknown Docker state or rental containers block it.
A fresh copy needs roughly 78 GiB for the model/projector, plus its runtime, private environment and MTP artifacts.

## Model cleanup

The retained Qwen assets are under `~/.local/share/localllm-qwen38/models/`:

- `hauhau/`: Q8_K_P weights, BF16 projector and matching FastMTP sidecar;
- `swift15-uncensored/`: **Q8_K_XL weights and the shared BF16 projector only**.

Strata's IQ3_S, projector, pack and MTP assets live separately under `~/.local/share/localllm-strata/`.
The original PLE second shard and BF16 projector were reused by hard link where publisher SHA-256 hashes matched;
there is no reason to download those bytes again.

The retired downloaded assets are: older Swift BF16, Swift 1.5 Uncensored BF16 weights, UkisAI Swift 1.5 27B Q8,
TURBO Q8, DFlash2 draft, GSQ Q2 and GLM-5.3-Flash EXL3. A guarded cleanup checks active file references and retained
asset identities; it does not delete shared Swift Q8 vision or the retained Hauhau sidecar.
Cleanup reclaimed **275.23 GiB physically**; about **655.30 GiB** remained free after Strata preparation.
The larger logical removed total includes the PLE shard/projector that Strata retains by hard link.
Historical reports and unselected build artifacts are not mistaken for extra downloaded model weights.

Archived comparison profiles remain explicit CLI-only in `v1qwen38.sh`; using their download/start commands can
re-download retired weights. **Do not use the legacy `--speed-test-all` to test this curated collection:** it includes
retired profiles. Profile-specific smoke/speed modes use 4K allocation and reasoning off, not normal native-context
performance.

Legacy `v1llama_cpp.sh` and `v1glm53.sh` remain archival CLI scripts, not shared-host menu choices. Their old
stop/benchmark paths are not covered by the new identity-bound guards; do not use them on a rented/shared host.

## HiveOS LLM custom miner (optional, unchanged)

The repository's optional custom miner still targets `swift15u-q8`, with BF16 vision, Q8 K/V, native MTP and two
native-262K slots on this machine. Its own Hive lifecycle/rental handling remains separate from a manual Strata
experiment. This update does not install it or change the currently selected Hive miner/flight sheet.
See [hive-llm-miner/README.md](hive-llm-miner/README.md).

## Checks

```bash
bash -n HostLLM.sh v1qwen38.sh v1strata.sh
bash tests/retained-menu-test.sh
bash tests/swift15-profile-test.sh
bash tests/swift15-uncensored-profile-test.sh
bash tests/gsq-profile-test.sh                 # archived CLI regression checks, no downloads
python3 -m unittest discover -s tests -p 'test_engine_safety.py' -v
```

The Python safety suite runs on Linux. Its HTTP fixtures and fake `llama-server` are **CPU-only**; they test
identity-bound lifecycle/known responses without starting a GPU model. Stop helpers use pidfds and matching
PID/start-time/executable/command identities; no process-name-wide `pkill`, `killall`, or reusable-PID sudo-kill fallback.

## Historical measurements and licenses

- [Swift 1.5 Uncensored two-slot tuning, 2026-09-26](SWIFT15U_2SLOT_2026-09-26.md)
- [Swift 1.5 source/MTP notes](SWIFT15.md)
- [Retired GSQ Q2 tuning](GSQ_TUNING_RESULTS.md) and [earlier menu notes](GSQ_FLASH_NEXT.md)
- [September 24 speed/update research](SPEED_REPORT_2026-09-24.md)

Historical model/menu numbers are not the current menu. Strata is MIT-licensed; model weights retain their own
publisher licenses, including the Swift Open License terms for Swift fine-tunes. Consult the exact model repository
before commercial use. Model/runtime pins are not silently advanced by a normal launch.
