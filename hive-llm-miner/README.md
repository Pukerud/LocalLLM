# HiveOS LocalLLM custom miner

This package installs one Hive custom miner with an atomic next-start profile selector. The standard Strata selection is Orca Q4_K_S Strata 0.1.45 with auto prefill and trimmed `14,24,36` split; the windows-2 override is omitted. Orca .39 is rollback-only; .41 and .44.1 remain explicit rollback runtimes. Strata .46, pinned to source `90767252f420e25223c739d604145ad15911d48b`, is not the standard: HostLLM [5] is an isolated single-slot Q4 test candidate, and the Hive profile-menu [5]/HostLLM [16] select the unqualified SC117 .46 CPU-vision trial. The .46 pinned build and bounded no-key LAN text/image smoke tests passed through HostLLM; see [the patch and results](../STRATA_V0146.md). A follow-up actual root-terminal HostLLM [16] switch and .46 Hive startup also passed no-key web/text/image smoke checks after fixing the system `runuser` lookup; see [the selector results](../results/hive-option16-runuser-20261020.json). This does not qualify SC117. Hive's standard Orca profile deliberately binds `0.0.0.0:8080` for trusted-LAN serving and is unauthenticated; do not expose it to an untrusted network. HostLLM's Orca route also defaults to trusted-LAN serving (explicit `STRATA_HOST=127.0.0.1` retains local-only access). The unqualified SC117 `.45` and `.46` trials also bind trusted-LAN `0.0.0.0:8080` without authentication and share the port exclusively with Orca. The other user-facing actions are Swift, Hauhau, and those SC117 trials. The old SC117 .39 selector route is retired; shared model/native-pack assets are preserved. Strata .45 is build/provenance checked; no Q4-specific speed, quality, or 40% improvement is claimed.

**Standing upgrade rule:** trusted-LAN routes must continue to start and serve requests without an API key after every upgrade. See [the maintenance rules](../AGENTS.md). No required key, dummy key, or client Authorization header may be introduced for normal LAN use.

## Interactive selector

Run as root only while the miner, inference server, Docker workloads, and GPU compute are idle:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh
```

```text
[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots
[2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots
[3] Orca Q4_K_S — Strata 0.1.45 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted
[4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.45 | native 262K | --spec 3 + pipeline-windows 2 | CPU vision | one slot; unqualified trusted-LAN trial on 8080 (no auth)
[5] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.46 | new SM86 defaults enabled | CPU vision | one slot; unqualified trusted-LAN trial on 8080 (no auth)
```

The selector changes only the `current` symlink; it never starts or stops Hive,
Strata, Docker, or `osn.service`. It checks both API ports (8080 and 18080),
processes, Docker, GPU compute, and the HostLLM lease before changing the next
profile. Hive profile-menu [5] selects SC117 .46; this is distinct from HostLLM [5],
which tests Orca Q4 .46. HostLLM's [12]–[16] actions are a higher-level safe switch: they pause
OctaSpace before stopping an identity/ancestry-verified Hive LLM, hold it stopped
through readiness and any guarded package refresh, select the target, and restore
the previous miner/OctaSpace running state. **Do not run `miner stop` first**: its
stop marker is temporary and OctaSpace can immediately restart the old model.
Selecting an already-current, unchanged profile is a fast no-op without reloading.
Manual hosting sessions, rentals, and unrelated GPU work are never stopped.
An interrupted/blocked restoration retains `~user/.local/state/hostllm/hive-profile-switch.json`;
run `sudo python3 /home/user/LocalLLM/hive_profile_switch.py --recover` after the
conflicting workload is safely gone. Orca `.45` is the standard Strata default. `.39`, `.41`, and `.44.1`
rollback profiles remain available by explicit runtime/profile name rather than visible menu rows.

Noninteractive selections include:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile hauhau-q8-fastmtp-q4kv-xhigh
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s-v01441
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated-v0146
```

The existing Orca .45 and SC117 .45 routes load their weights and vision encoder immediately during startup, before the API opens. The launcher requires the model to be loaded before reporting ready; a blank HTTP frontend is not a successful start. Idle unloading is disabled. The .46 candidates also passed bounded eager-load/text/image smoke checks through HostLLM, with no API key. The later [16] selector follow-up exercised the real Hive wrapper as well. Its Python preflight resolves `runuser` from trusted system paths, including `/usr/sbin` on HiveOS, before any service pause.

## SC117 trial isolation

SC117 `.45` and `.46` are **unqualified and trial-only**. Both Hive wrappers use
CPU vision, exactly one serving slot/group, and the same dedicated state root.
The `.46` route is only a pinned test candidate; build and exact readiness checks
must pass before any live test. No .46 qualification or performance claim is made:

```text
Host:       0.0.0.0 (trusted LAN; unauthenticated)
Port:       8080
State:      /home/user/.local/state/locallm-strata/sc117-iq3s-trial
Runtime:    Strata 0.1.45 (Hive menu [4], HostLLM [15])
Vision:     CPU
Spec:       3 (stock/non-abliterated head; maximum head depth 4)
Windows:    2
Context:    262144 native
KV:         INT8 (32768 resident)
Prefill:    auto
Split:      14,24,36 with trimmed stage weights
Cache:      8192 MiB / four cache slots
```

The separate `.46` route uses profile `iq3_s-sc117-abliterated-v0146`, Hive menu [5]/HostLLM [16], and Strata 0.1.46 with the same dedicated state, trusted-LAN bind, and one-slot settings. Both routes intentionally expose unauthenticated APIs on the trusted LAN;
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

The installer checks the pinned Orca .45 build/profile, retained .39/.41/.44.1
rollback routes, and the exact SC117 .45 trial preflight before changing Hive
configuration. If the current pointer already selects SC117 .46, it additionally
requires the exact .46 readiness line before preserving/updating that selection.
Other .46 selections are gated by exact readiness before the menu changes the
pointer or HostLLM pauses services. Strata .46 remains isolated from standard
Orca .45. If the current pointer selects generic Orca .39 or the obsolete
SC117 .39 route, it atomically migrates the next-start pointer to standard Orca
.45 before replacing the SC117 profile directory with the isolated trial
package. Other recognized selections, including an existing SC117 trial, are
preserved. The installer does not start/stop Hive or `osn.service`.

The menu does not expose rollback rows, but `.39`/`.41`/`.44.1` runtime paths
remain available for rollback. Keep Orca `.45` as standard and preserve all
shared model/native-pack assets.
