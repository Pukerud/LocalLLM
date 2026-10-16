# HiveOS LocalLLM custom miner

This package installs the `llm-hosting` custom miner with six selectable launch
profiles:

1. **Orca Uncensored Q4_K_S** — Strata 0.1.39, native 262,144 context, CPU
   vision, four RTX 3090s, one serving slot. This remains the install default.
2. **Previous Swift 1.5 Uncensored Q8_K_XL** — the prior Qwen3.8 Hive profile,
   retained for rollback.
3. **Orca Q4_K_S .41 one-slot** — explicit opt-in to Strata 0.1.41, native 262,144
   context, CPU vision, 8 GiB conversation cache (four cache slots), one serving
   slot. Its own 32K validation and migration gates must already pass.
4. **Orca Q4_K_S .41 two-slot** — experimental opt-in to the same .41 config with
   two serving slots/two batch groups. A short test left only 457 MiB VRAM free
   and showed no clear aggregate throughput gain; do not infer long-context or
   sustained-load capacity.
5. **Orca Q4_K_S .44.1** — machine-specific opt-in using auto prefill and the
   trimmed `14,24,36` stage split, one serving slot, native 262K, CPU vision,
   and the 8 GiB/four-slot conversation cache. Its effective serving config omits
   the uncertain pipeline-windows-2 override (corrected decode interval crosses zero).
6. **SC117 abliterated IQ3_S** — readiness-gated Strata 0.1.39, native 262,144 context,
   INT8 KV (32,768 resident), MTP=4 using the published stock/non-abliterated head,
   BF16 GPU vision, one slot/group, auto prefill/split, and 8 GiB/four cache slots.
   It remains unqualified until a trusted identity-bound report exists.

All six profiles use the same Hive custom-miner name and Flight Sheet. Hive continues
to own its normal miner lifecycle; `osn.service` is not changed by installation
or profile selection.

## Select a profile

After installing the package, run the menu as root while the Hive miner and all
LLM/renter workloads are idle:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

HostLLM [17] selects the SC117 profile through this guarded selector; it does not
manage Hive's service lifecycle. Existing [12]–[16] profile routes, .41 launch
presets [5]/[6], and the .44.1 launch [7] remain unchanged.

Choices are:

```text
[1] Orca Uncensored Q4_K_S — Strata 0.1.39, native 262K, CPU vision, one serving slot
[2] Previous Swift 1.5 Uncensored Q8_K_XL profile
[3] Orca Q4_K_S — Strata 0.1.41, native 262K, CPU vision, 8 GiB cache, one serving slot
[4] Orca Q4_K_S — Strata 0.1.41, native 262K, CPU vision, 8 GiB cache, two serving slots/two groups (experimental)
[5] Orca Q4_K_S — Strata 0.1.44.1, native 262K, CPU vision, 8 GiB cache, one serving slot, auto prefill, trimmed 14,24,36 split (pipeline-windows 2 omitted)
[6] SC117 abliterated IQ3_S — Strata 0.1.39, native 262K, BF16 GPU vision, MTP=4, one slot/group, 8 GiB/four-slot cache (exact qualification required)
```

Noninteractive form:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s-v0141
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s-v0141-2slot
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s-v01441
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated
```

The menu atomically changes one `current` symlink; Hive's stable `h-run.sh`,
configuration and metadata paths then resolve to the selected profile. On an
older install with regular root-level Hive files, its first idle selection
identifies the current profile from the exact config and safely bootstraps those
symlinks before switching. It refuses to make a change while an LLM
launcher/server is present, port 8080 is occupied, Docker or GPU compute is
active, a HostLLM pause lease exists, or a safety check cannot be completed.
Hive's `MINER_RUN` marker is informational by itself: it may remain set while
the registered custom-miner slot is idle. The selector checks for an actual
`h-run.sh`/LLM process and active API/GPU/Docker work before switching the
next-start profile.
Before selecting an Orca profile, it runs the non-inference `--check-ready`
gate. The .41 profiles require their exact migrated settings; the two-slot
profile additionally requires the 2-slot/2-group preset. The .44.1 profile
verifies pinned calibration source/build/config/report evidence, preserved
native-context/CPU-vision/cache settings, auto prefill, stage trimming, and
single-slot configuration before changing the pointer. The pinned integrated report was recorded with the exploratory
pipeline-windows-2 calibration config; that override is omitted from the selected serving copy, so the report does not
qualify windows=1 performance. Its machine-bound calibration workspace must exist on this host; otherwise selection
fails closed. SC117 selection also checks its exact .39 config and requires a
root-owned report at `/var/lib/localllm-strata/qualifications/sc117-iq3s.json`.
No trusted evidence runner is installed or pinned, so a report cannot currently
unlock normal starts. The exact SC117 readiness line is:

```text
READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.
```

SC117 defaults to `127.0.0.1`; only explicit `STRATA_HOST=0.0.0.0` enables LAN
binding, and that requires a protected `STRATA_API_KEY` in the Hive miner
configuration. The wrapper unsets the exported key name and forwards its value
through stdin; the launcher keeps it out of argv, environment, and persistent
run-config, using a private tmpfs config removed on stop. HTTP remains unencrypted;
use only a trusted LAN or tunnel. See [STRATA_SC117_IQ3S.md](../STRATA_SC117_IQ3S.md)
for pins, qualification requirements, and the current fail-closed status.

The 32K/runtime validation gates remain required. Hive Orca profiles explicitly
use `STRATA_HOST=0.0.0.0` for trusted-LAN access and do not pass
`STRATA_API_KEY`; their API is therefore unauthenticated. Firewall it and do not
expose it to an untrusted network. Candidate validation defaults to loopback.
The menu **never** runs `miner start`, `miner stop`, `osn.service` controls, or an
inference request. Selection takes effect
on the next normal Hive miner start. Do not run it during a rental; wait until
the miner has stopped and the renter workload has cleared.

## Install

The host must already have both launchers installed and Orca profiles prepared
with their readiness evidence. .44.1 is pinned to the existing `.69` calibration
workspace described in [ORCA_Q4KS_V01441.md](../ORCA_Q4KS_V01441.md); the installer
does not create or copy that machine-specific candidate. This package does not
download models or prepare/change Strata runtimes.

From the repository root, as root, while the host is idle:

```bash
./install-hive-llm-miner.sh
```

The installer checks both launchers, Q4 .39, both .41 presets, the .44.1
`--check-ready` preset, and the idle state; it installs all six profile
directories and the menu. SC117 can be installed before qualification, but its
selector and normal start remain blocked until the exact trusted report exists.
A fresh install selects Orca Q4_K_S .39; updates preserve a recognized existing
`current` profile pointer. SC117 is never selected implicitly. Hive is configured
with `MINER=custom` / `CUSTOM_MINER=llm-hosting`. It backs up
`rig.conf` and `wallet.conf` once with the suffix `.llm-hosting.bak`. It does
not start or stop Hive or OctaSpace services. Start the miner later using
Hive's normal operator workflow, when appropriate.

## Remove

The uninstall script refuses to remove the package, including the SC117 profile,
while Hive, an LLM launcher/server, the API, Docker, a HostLLM lease, or GPU compute is active. It
never stops the miner automatically. Run it only
when the host is idle:

```bash
./uninstall-hive-llm-miner.sh
```

The official `hive-miners-custom` package and `osn.service` are left installed
and running. Hive may report zero hashrate because this custom miner serves
inference rather than mining; GPU telemetry and the custom-miner running state
remain visible in the HiveOS dashboard.

## Flight Sheet

`Qwen3.8-LLM-Hosting.flight-sheet.json` is a template for the shared custom
miner name. Import it through the HiveOS Flight Sheets page if needed, and keep
these settings:

```text
Miner: Custom
Custom miner: llm-hosting
Installation URL: empty (the package is already installed on the worker)
Pool: empty / configure in miner
```

The `LLM` coin and empty wallet are metadata only. If HiveOS requires a wallet,
use a harmless custom wallet. The Flight Sheet does not choose the model; use
`profile-menu.sh` while idle. Do not apply an ordinary crypto-mining Flight
Sheet to this worker.
