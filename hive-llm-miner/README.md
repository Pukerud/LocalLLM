# HiveOS LLM custom miner

This custom HiveOS miner exposes the retained Qwen3.8 Hauhau FastMTP server as
an ordinary HiveOS miner. It uses the official `hive-miners-custom` control
package and the normal `miner start` / `miner stop` lifecycle.

The miner starts:

```text
/home/user/LocalLLM/v1qwen38.sh --quickstart --profile hauhau-q8-fastmtp --no-dashboard
```

It deliberately does **not** stop or start `osn.service`. When OctaSpace rents
the node, its normal HiveOS `miner stop` command reaches the foreground wrapper,
which stops the Qwen server. When the rental ends, `miner start` starts the
wrapper again while `osn.service` remains running.

The wrapper fails closed if Docker reports any non-HostLLM running container,
or if Docker status cannot be read. This prevents it from consuming GPUs while
a renter workload is active.

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
