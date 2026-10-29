# HiveOS LocalLLM custom miner

One Hive custom miner with an atomic next-start profile selector. **Swift remains the default**; Hauhau is retained. Heretic RE is the new Flash-Next Strata route. Orca Q4_K_S and DeepSeek weights/menu routes were retired on explicit operator request (2026-10-29); their historical code/reports are not installed-model rollback guarantees.

## Select a model

Preferred high-level, ownership-checked switch:

```bash
cd /home/user/LocalLLM
./HostLLM.sh
# [12] Swift; [13] Hauhau; [14] Heretic RE; [17] SC117 .46 trial.
```

The low-level root-only selector changes a pointer, **not services**. Use it only when hosting/mining is safely stopped and no rental/manual session is active:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT
[2] Hauhau Q8_K_P
[3] Swift 1.5 Heretic RE — Strata .46 + isolated BF16-gate compatibility
[4] SC117 .45 unqualified CPU-vision trial
[5] SC117 .46 unqualified CPU-vision trial
```

Low-level [3] corresponds to HostLLM Hive **[14]**; low-level [5] corresponds to HostLLM Hive **[17]**. HostLLM direct [3] starts Heretic's foreground server. SC117 .45 direct [4] remains, but its former high-level Hive [15] row was explicitly removed.

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15-heretic-re
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile hauhau-q8-fastmtp-q4kv-xhigh
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated-v0146
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

SC117 .45/.46 remain **unqualified trials**, fixed to unauthenticated trusted-LAN `0.0.0.0:8080`, dedicated `sc117-iq3s-trial` state, CPU vision, spec 3/windows 2, four history slots and one serving slot. No normal/production qualification is granted. SC117 now verifies its own inputs plus exact retained preset metadata independently of deleted Orca weight files.

## Install and lifecycle

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
