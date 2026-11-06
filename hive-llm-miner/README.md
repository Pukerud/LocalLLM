# HiveOS LocalLLM custom miner

Atomic next-start selection and identity-bound supervision for the retained **Swift / SC117 two-user** models. Swift remains the installation default.

**2026-11-06 completed recovery:** after the operator’s manual restart, all four original GPUs recovered and **native262K [18] is ACTIVE again**. Its unchanged .46 two-slot policy and UID1000 key-free LAN text/vision/stream/Host-Origin checks passed. Only the exact task fault lease was released after fresh checks; existing owned startup was observed without duplication. OctaSpace stayed active; no renter/admission override or cloud rewrite. No lease/journal/test/controller/reboot pending. See [the report](../TENSORSHARP_SWIFT_20261106.md). TensorSharp remains disabled; do not replay benchmarks or retry it.

## Current choices

| HostLLM direct | HostLLM Hive | Low-level | Profile |
|---:|---:|---:|---|
| 1 | 12 | 1 | `swift15u-q8` — original llama.cpp, BF16 vision, MTP n3, two native-262K slots |
| 7 | 18 | 8 | `iq3_s-sc117-abliterated-ple-q8-2slot-v0146` — .46, two pipelined slots/groups, 16 GiB/eight histories, CPU vision, native262K |

HostLLM **[8]/[19] TensorSharp** are disabled information entries, **not a runnable Hive profile**. Former high-level [2–6]/[13–17] and low-level Hauhau/Heretic/single-slot SC117 targets are rejected. Historical source wrappers are audit evidence, not supported installed choices.

```bash
cd /home/user/LocalLLM
./HostLLM.sh
```

The root-only low-level selector is **idle-only**, never changes services and refuses occupied API ports, launchers, GPU work, any Docker workload or a hosting lease:

```bash
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --list
# When explicitly authorized, genuinely idle and rental-free:
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile swift15u-q8
sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile iq3_s-sc117-abliterated-ple-q8-2slot-v0146
```

Do not use selection/install/recovery to bypass any hardware-fault hold.

## Retained SC117 contract

`.46`, unauthenticated trusted-LAN `0.0.0.0:8080`, native262144 per slot, INT8/32768-resident KV, BF16 CPU vision, split14/24/36, spec3/windows2, **2 slots/2 pipelined groups**, **16384 MiB/eight histories**, isolated `sc117-iq3s-ple-q8-2slot-trial` state. Solo MTP stays active; concurrent windows use no drafts. Long prefill can stall another stream. Still unqualified, no 512K retry or output-budget change.

[SC117_Q8_2SLOT.md](../SC117_Q8_2SLOT.md) and [PLE_Q8.md](../PLE_Q8.md) retain provenance. Q8 applies to PLE only. Both original SC117 GGUF shards, shared Q8 PLE, CPU vision, .46 runtime and required old preset metadata remain installed. Hauhau/Heretic weight trees were unused and deleted; no retained model was requantized or rewritten.

## Package installation

Prepare model/software as unprivileged `user` first. For an already installed active package, use the **no-stop retained-only** publisher, never the full installer:

```bash
sudo ./install-hive-llm-miner.sh --publish-retained-profiles
```

The old `--add-ple-q8-profiles` spelling delegates to the same retained-only publisher. Under both lifecycle/selector reservations it validates exact current and inactive bundle bytes, removes only explicitly retired inactive bundles, and atomically publishes the selector. Active files, current/stable links, miner markers, admission state and rig/wallet bytes remain unchanged. Rentals, leases, differing files or unknown work cause refusal.

The ordinary `sudo ./install-hive-llm-miner.sh` stays **idle-only**, preserves a recognized retained current pointer, rejects retired/unknown layouts instead of guessing a migration, and never starts/stops a process/service. An explicit full install changes only the normal four custom-miner flight-sheet fields; unrelated settings/secrets/secondary-miner fields remain intact. Ordinary live publication never rewrites cloud configuration.

## Ownership and renters

**Renters always win. Admission is not stopped for LLM switching/hosting.** Docker, active GPU work, manual/replacement sessions and exact Hive ancestry are freshly checked. Only the verified owned LLM is stopped/yielded. Signals target pidfds with recorded executable/UID/argv/birth/session/parent provenance; no numeric-PID fallback, unknown-process adoption, `pkill`, GPU reset or driver/clock/power changes.

Normal inference runs as UID1000. Preserve token-bound Hive shutdown and asynchronous startup observation; full hashing/loading is not permission to start a duplicate model. Preserve legitimate shell-quoted/cloud-refreshed miner settings rather than restoring old config hashes.

Use **http://192.168.1.69:8080/** only on a trusted firewalled LAN: no mandatory key/dummy token/Authorization header; retain Host/Origin and explicitly configured optional-auth enforcement. The Strata .46 binary and its exact no-key patch were not changed.

## Completed fault recovery / future guard

This task’s fault hold is **released and completed**, not pending recovery. The generic `hardware_recovery_required` guard remains: future fault holds cannot be cleared by ordinary stop/release without explicit reviewed hardware recovery. Verify new boot/four original GPUs/power/current-boot faults and fresh rental/manual-session/ownership state under reservation before releasing only a proven task lease. Observe asynchronous owned startup; do not create a duplicate. Never stop a renter, retry TensorSharp/512K or replay completed controllers.
