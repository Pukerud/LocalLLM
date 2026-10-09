# HiveOS LocalLLM custom miner

This package installs the `llm-hosting` custom miner with two selectable launch
profiles:

1. **Orca Uncensored Q4_K_S** — Strata 0.1.39, native 262,144 context, CPU
   vision, four RTX 3090s, one slot. This is the default profile.
2. **Previous Swift 1.5 Uncensored Q8_K_XL** — the prior Qwen3.8 Hive profile,
   retained for rollback.

Both profiles use the same Hive custom-miner name and Flight Sheet. Hive continues
to own its normal miner lifecycle; `osn.service` is not changed by installation
or profile selection.

## Select a profile

After installing the package, run the menu as root while the Hive miner and all
LLM/renter workloads are idle:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

The HostLLM main menu exposes the same choices as **[12] Orca** and **[13]
previous Swift/Qwen**; those entries call this guarded selector and do not manage
Hive's service lifecycle.

Choices are:

```text
[1] Orca Uncensored Q4_K_S — Strata 0.1.39, native 262K, CPU vision, one slot
[2] Previous Swift 1.5 Uncensored Q8_K_XL profile
```

Noninteractive form:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
```

The menu atomically changes one `current` symlink; Hive's stable `h-run.sh`,
configuration and metadata paths then resolve to the selected profile. It
refuses to make a change while Hive reports a running miner, an LLM
launcher/server is present, port 8080 is occupied, Docker or GPU compute is
active, a HostLLM pause lease exists, or a safety check cannot be completed.
Before selecting Orca, it runs the non-inference `--check-ready` gate. It
**never** runs `miner start`, `miner stop`, `osn.service` controls, or an
inference request. A selection takes effect on the next normal Hive miner
start. Do not run it during a rental; wait until the miner has stopped and the
renter workload has cleared.

## Install

The host must already have both launchers installed and the Orca Q4_K_S
profile prepared with its native-context/CPU-vision readiness evidence. This
package does not download models or prepare/change Strata runtimes.

From the repository root, as root, while the host is idle:

```bash
./install-hive-llm-miner.sh
```

The installer checks both launchers, the Q4 `--check-ready` gate, and the idle
state; it installs both profile directories and the menu, points the default to
Orca Q4_K_S, and configures Hive to use `MINER=custom` / `CUSTOM_MINER=llm-hosting`. It backs up
`rig.conf` and `wallet.conf` once with the suffix `.llm-hosting.bak`. It does
not start or stop Hive or OctaSpace services. Start the miner later using
Hive's normal operator workflow, when appropriate.

## Remove

The uninstall script refuses to remove the package while Hive, an LLM
launcher/server, the API, Docker, a HostLLM lease, or GPU compute is active. It
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
