# HiveOS LocalLLM custom miner

This package installs one Hive custom miner with an atomic next-start profile selector. Its retained default remains Orca Q4_K_S Strata 0.1.39. The user-facing HostLLM actions are Swift, Hauhau, Orca .44.1, and the isolated SC117 .44.1 CPU-vision trial. The older SC117 .39 selector route is retired; its shared model/native-pack assets are preserved.

## Interactive selector

Run as root only while the miner, inference server, Docker workloads, and GPU compute are idle:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
[2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
[3] Orca Q4_K_S — Strata 0.1.44.1 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted
[4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.44.1 | native 262K | CPU vision | MTP-4 | one serving slot
```

The selector changes only the `current` symlink; it never starts or stops Hive,
Strata, Docker, or `osn.service`. It checks both API ports (8080 and 18080),
processes, Docker, GPU compute, and the HostLLM lease before changing the next
profile. `.39` remains the generic default/explicit rollback, and `.41` rollback
profiles remain available by explicit profile name rather than visible menu rows.

Noninteractive selections include:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile hauhau-q8-fastmtp-q4kv-xhigh
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s-v01441
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated
```

## SC117 trial isolation

SC117 is **unqualified and trial-only**. Its Hive wrapper uses Strata 0.1.44.1,
CPU vision, exactly one serving slot/group, and a dedicated state root:

```text
Host:       127.0.0.1
Port:       18080
State:      /home/user/.local/state/locallm-strata/sc117-iq3s-trial
Runtime:    Strata 0.1.44.1
Vision:     CPU
MTP:        4
Context:    262144 native
KV:         INT8 (32768 resident)
Prefill:    auto
Split:      14,24,36 with trimmed stage weights
Cache:      8192 MiB / four cache slots
```

The route rejects LAN/production binding, uses a private isolated Python cache,
and cannot write or claim a normal qualification report. The selector and
HostLLM require the exact `TRIAL CONFIG` preflight line; this is readiness for
starting the isolated experiment, **not** quality evidence or qualification.
Do not select SC117 for normal/production hosting. No model or native-pack
assets are deleted by the menu cleanup.

## Install and rollback

The installer checks the Orca .39 default, preserved .41 profiles, Orca .44.1,
and the exact SC117 trial preflight before changing Hive configuration. If the
current pointer still targets the obsolete SC117 .39 route, it atomically moves
the next-start pointer to the generic Orca .39 default before replacing that
profile directory with the isolated trial package. Other recognized selections
are preserved. The installer does not start/stop Hive or `osn.service`.

The menu does not expose `.41` rows, but explicit `.41` profile names remain
available for rollback. Keep the generic `.39` default and the `.41`/`.44.1`
profile files; do not delete shared model/native-pack assets.
