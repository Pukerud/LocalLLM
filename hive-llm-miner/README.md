# HiveOS LocalLLM custom miner

This package installs one Hive custom miner with an atomic next-start profile selector. The standard Strata selection is Orca Q4_K_S Strata 0.1.44.1 with auto prefill and trimmed `14,24,36` split; the windows-2 override is omitted. Orca .39 is rollback-only. Hive's standard Orca profile deliberately binds `0.0.0.0:8080` for trusted-LAN serving and is unauthenticated; do not expose it to an untrusted network. HostLLM's Orca route defaults to loopback. The unqualified SC117 `.44.1` trial also binds trusted-LAN `0.0.0.0:8080` without authentication and shares the port exclusively with Orca. The other user-facing actions are Swift, Hauhau, and that SC117 trial. The old SC117 .39 selector route is retired; shared model/native-pack assets are preserved.

## Interactive selector

Run as root only while the miner, inference server, Docker workloads, and GPU compute are idle:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
[2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
[3] Orca Q4_K_S — Strata 0.1.44.1 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted
[4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.44.1 | native 262K | CPU vision | MTP-4 | one serving slot; unqualified trusted-LAN trial on 8080 (no auth)
```

The selector changes only the `current` symlink; it never starts or stops Hive,
Strata, Docker, or `osn.service`. It checks both API ports (8080 and 18080),
processes, Docker, GPU compute, and the HostLLM lease before changing the next
profile. HostLLM's [12]–[15] actions compare the selected installed profile/menu
with this checkout and run the guarded installer first if either is stale. Orca
`.44.1` is the standard Strata default. `.39` is rollback-only, and `.41`
rollback profiles remain available by explicit profile name rather than visible menu rows.

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
Host:       0.0.0.0 (trusted LAN; unauthenticated)
Port:       8080
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

The route intentionally exposes its unauthenticated API on the trusted LAN;
firewall it and never expose it to an untrusted network. For `.69`, open
`http://192.168.1.69:8080/`. Orca also uses port 8080, so it cannot run
concurrently with SC117. The trial uses a private isolated Python cache and
cannot write or claim a normal qualification report.
The selector and HostLLM require the exact `TRIAL CONFIG` preflight line; this
is readiness for starting the isolated experiment, **not** quality evidence or
qualification.
Do not select SC117 for normal/production hosting. No model or native-pack
assets are deleted by the menu cleanup.

## Install and rollback

The installer checks the pinned Orca .44.1 standard profile, the retained .39/.41
rollback profiles, and the exact SC117 trial preflight before changing Hive
configuration. If the current pointer selects generic Orca .39 or the obsolete
SC117 .39 route, it atomically migrates the next-start pointer to standard Orca
.44.1 before replacing the SC117 profile directory with the isolated trial
package. Other recognized selections, including the explicit SC117 .44.1 trial,
are preserved. The installer does not start/stop Hive or `osn.service`.

The menu does not expose `.41` rows, but explicit `.41` profile names remain
available for rollback. Keep Orca `.44.1` as standard, `.39`/`.41` as rollback,
and all shared model/native-pack assets.
