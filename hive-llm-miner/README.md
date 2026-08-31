# HiveOS LLM custom miner

This custom HiveOS miner exposes the retained Qwen3.8 Hauhau FastMTP server as
an ordinary HiveOS miner. It uses every detected NVIDIA GPU, the official
`hive-miners-custom` control package, and the normal `miner start` / `miner stop`
lifecycle.

The miner starts:

```text
/home/user/LocalLLM/v1qwen38.sh --quickstart --profile hauhau-q8-fastmtp --no-dashboard
```

It deliberately does **not** stop or start `osn.service`. On the current
4x RTX 3090 host, the launcher automatically uses three native-262K slots;
on the original 3-GPU layout it uses two. When OctaSpace rents the node, its
normal HiveOS `miner stop` command reaches the foreground wrapper, which stops
the Qwen server. When the rental ends, `miner start` starts the wrapper again
while `osn.service` remains running. Managed starts skip hashing already-present
assets for fast handoff; newly downloaded assets are still checksum-verified.

The wrapper fails closed if Docker reports any non-HostLLM running container,
or if Docker status cannot be read. This prevents it from consuming GPUs while
a renter workload is active. It also closes its Hive screen when startup exits,
which prevents a stale empty screen from blocking the next `miner start`.

Install from the repository root as root:

```bash
./install-hive-llm-miner.sh
miner start
```

The installer backs up `rig.conf` and `wallet.conf` once with the suffix
`.llm-hosting.bak`. To remove the integration, stop the custom miner and run:

```bash
./uninstall-hive-llm-miner.sh
```

Hive reports zero hashrate because this is an inference host, but the custom
miner name, running state, and GPU telemetry remain visible in the HiveOS
dashboard.

## Optional Flight Sheet import

`Qwen3.8-LLM-Hosting.flight-sheet.json` is a HiveOS Flight Sheet JSON template.
Import it from the HiveOS Flight Sheets page using the clipboard/file importer.
The `LLM` coin and empty wallet are metadata only; this custom miner does not
mine or use a wallet/pool. If HiveOS requires a wallet while importing, create
or select a harmless custom wallet, then keep these settings:

```text
Miner: Custom
Custom miner: llm-hosting
Installation URL: empty (the package is already installed on the worker)
Pool: empty / configure in miner
```

After applying the sheet, verify that the worker still has `MINER=custom` and
`CUSTOM_MINER=llm-hosting`. Do not apply an ordinary crypto-mining Flight Sheet.
