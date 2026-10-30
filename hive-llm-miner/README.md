# HiveOS LocalLLM custom miner

One Hive custom miner with an atomic next-start profile selector. **Swift remains the default**; Hauhau is retained. Heretic RE is the new Flash-Next Strata route. Orca Q4_K_S and DeepSeek weights/menu routes were retired on explicit operator request (2026-10-29); their historical code/reports are not installed-model rollback guarantees.

## Select a model

Preferred high-level, ownership-checked switch:

```bash
cd /home/user/LocalLLM
./HostLLM.sh
# [12] Swift; [13] Hauhau; [14] Heretic IQ4_NL PLE; [15] SC117 .46 IQ4_NL PLE.
# [16] Heretic Q8_0 PLE; [17] SC117 .46 Q8_0 PLE (bounded smoke passed; quality/speed A/B pending).
```

The low-level root-only selector changes a pointer, **not services**. Use it only when hosting/mining is safely stopped and no rental/manual session is active:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT
[2] Hauhau Q8_K_P
[3] Heretic RE — IQ4_NL n-gram | Strata .46 + isolated BF16-gate compatibility
[4] SC117 .45 — IQ4_NL n-gram | unqualified CPU-vision trial
[5] SC117 .46 — IQ4_NL n-gram | unqualified CPU-vision trial
[6] Heretic RE — Q8_0 n-gram | same .46 | bounded smoke passed; A/B pending
[7] SC117 .46 — Q8_0 n-gram | same runtime/tuning | bounded smoke passed; A/B pending
```

Low-level [3]/[5] correspond to HostLLM Hive **[14]/[15]**; new low-level [6]/[7] correspond to high-level **[16]/[17]**. HostLLM direct [3]/[4] retain the original IQ4_NL PLE; [5]/[6] add Q8_0 PLE. Direct SC117 remains .45, Hive SC117 remains .46. The old high-level SC117 .46 [17] number moved to [15]; the selected profile itself is not changed by this menu update.

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15-heretic-re
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile hauhau-q8-fastmtp-q4kv-xhigh
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated-v0146
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15-heretic-re-ple-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated-ple-q8-v0146
```

Retired Orca/DeepSeek names are rejected by the active selector. The high-level switch refuses another manual/unowned server, unknown GPU work, a maintenance lease, or any possible Docker rental before a stop. An orphaned recorded server is **not automatically adopted as Hive-owned**; the operator must explicitly authorize/control that separate session.

## Heretic configuration

Profile **`swift15-heretic-re`** uses all eight checksum-pinned publisher mixed-IQ4_NL V2-70G-FIX shards and the exact supplied Strata pack. See [pins and BF16 compatibility evidence](../HERETIC_RE.md).

- Genuine native **262144 context**, INT8 KV with 32768 resident; no extension/truncation.
- **BF16 CPU vision**, high thinking, four GPUs, one serving slot/group.
- Auto prefill, trimmed `14,24,36` split, `--spec 3 --pipeline-windows 2`.
- **8192 MiB/four parked conversation histories**, not four serving slots.
- Eager model + vision loading, no idle unloading; dedicated private state/expert profile.
- Existing pinned .46 frontend/vision plus a **Heretic-only BF16 L0-gate engine overlay**. Shared SC117 .46 binary/source and all target weights remain unchanged. The original engine loaded the model but failed the first actual request; the isolated fallback uses existing BF16 kernels rather than requantizing or omitting the sensitive gate.
- Stock/non-abliterated Flash-Next MTP is target-verified, not claimed as a Heretic-trained head. Inherited tuning is not an optimal-performance or broad-quality qualification.

Endpoints are `http://192.168.1.69:8080/` and `/v1`. **No mandatory API key**, dummy token, secret environment or Authorization header. Host/DNS-rebinding and browser Origin checks remain; this is only for a trusted, firewalled LAN. Never expose it publicly. The original .46 no-key patch remains exact/version-pinned. Configured optional auth is not bypassed.

SC117 .45/.46 remain **unqualified trials**, fixed to unauthenticated trusted-LAN `0.0.0.0:8080`, dedicated `sc117-iq3s-trial` state (Q8 variants use `sc117-iq3s-ple-q8-trial`), CPU vision, spec 3/windows 2, four history slots and one serving slot. No normal/production qualification is granted. SC117 now verifies its own inputs plus exact retained preset metadata independently of deleted Orca weight files.

## Q8_0 PLE-only candidates

[PLE_Q8.md](../PLE_Q8.md) pins the shared 54.4-GB original-Qwen Q8_0 table and documents full file/header checks and original-model table identity. These candidates keep expert/dense weights, native 262K, CPU vision, INT8/32768 KV, spec 3/windows 2 and 8192-MiB/four-history cache policy unchanged. They use separate state/adaptive profiles at `swift15-heretic-re-ple-q8` and `sc117-iq3s-ple-q8-trial`; original configs/cache are not rewritten. API model IDs add `-ple-q8_0`, while existing aliases remain accepted.

Initial no-stop publication did not run inference. Later operator-authorized **[5]/[6]/[16]/[17]** live functional smoke passed: key-free text/high thinking, swapped/ordered images, SSE/tool formatting, real four-history cache restoration, owned shutdown and both Hive 259835-token retrievals. [The live receipt](../results/ple-q8-live-menus-20261030.json) records scope and initial corrections; original SC117 .46/IQ4_NL and active OctaSpace were restored. This is **not matched quality/speed A/B or production qualification**. Higher PLE precision is not a guaranteed answer-quality or speed improvement. No Strata upgrade is involved.

## Install and lifecycle

After operator-owned Q8 data preparation, this narrow mode may add **only the two inactive Q8 profiles and selector while the current miner stays running**:

```bash
sudo ./install-hive-llm-miner.sh --add-ple-q8-profiles
```

It holds the existing selection/lifecycle reservations but creates no pause lease, performs no stop/start/switch, and preserves `current`, active profile files, stable entrypoints, miner markers, service state and rig/wallet configuration. Differing existing Q8 profiles, rentals or pending leases/journals cause refusal. **Do not use the full installer below for a live update.**

Prepare the selected model explicitly as the normal operator first; ordinary starts never download/upgrade. See the [main README](../README.md) and [Heretic preparation](../HERETIC_RE.md#preparation).

When idle and rental-free:

```bash
sudo ./install-hive-llm-miner.sh
```

New installs default to Swift. The installer preserves an identifiable active current profile, verifies its pinned selected Strata readiness before publication, and refuses a selected retired model instead of silently switching from missing weights. It does not start/stop mining or OctaSpace itself. It updates only the required Hive custom-miner assignments; unrelated configuration/second-miner fields are retained. A stale previous custom-miner install URL is cleared so Hive does not download the wrong miner.

The high-level switch separately serializes admission, exact Hive process ancestry/identity, owned teardown, package/readiness and atomic pointer publication; then restores the recorded previous running state. If the miner was already stopped, selecting a profile does not pretend it was running. An explicit normal Hive start is separate. Rentals/manual sessions are never stopped to force selection. The private journal is `~/.local/state/hostllm/hive-profile-switch.json`; inspect it before an explicit `sudo python3 hive_profile_switch.py --recover`.

Hive owns the wrapper; inference runs as unprivileged `user`. Launch tokens and recorded PID/start-time/executable/UID/session/parent identities prevent a reused PID or replacement session from being signalled. The native model yields its own processes if a rental/unknown workload appears. No `pkill`, driver reset/unload, watchdog disable, cloud-agent suppression, power or clock changes.

`osn.service` remains available during normal Hive hosting and may admit rentals. Direct HostLLM [3] instead holds its own pause lease, and Ctrl+C releases only that lease after exact owned shutdown. It never releases someone else's maintenance. A failed owned teardown remains paused/journalled for safe recovery.

Telemetry keeps `khs=0` and four zero hash-rate entries; token throughput is visible in the inference Monitor and is never misreported as mining hashrate. User conversation state and protected Pi skills/extensions/settings are not changed by this package.
