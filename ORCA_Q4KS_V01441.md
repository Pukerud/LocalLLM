# Orca Q4_K_S / Strata 0.1.44.1 opt-in candidate

## Status and scope

This is a machine-specific, opt-in candidate for the `.69` HostLLM/Hive node, not a promotion. The generic Strata runtime default remains 0.1.39, and 0.1.41 remains the separately selected Hive profile/rollback. Neither profile assets nor `runtime-selection.json` are changed by this route. HostLLM [7] starts this candidate; HostLLM [16] selects Hive profile `orca-q4_k_s-v01441` for the next normal Hive start. Both choices require an explicit operator action and a read-only readiness check before a service pause or profile-pointer change.

The workspace is deliberately pinned to these `.69` paths:

- Tuning/effective profile: `/home/user/.local/share/localllm-strata/calibration/orca-q4ks-machine-tuning-v01441-20261015/`
- Pinned source/build workspace: `/home/user/.local/share/localllm-strata/calibration/orca-q4_k_s-v01441-prefill-20261015-retry2/`
- Integrated report: `arms/integration-auto-trim-windows2/results.json` in the tuning workspace.

This is not a portable install or preparation path. On a host without these exact artifacts, the .44.1 readiness check and Hive selector fail closed; the installer does not copy or create the candidate workspace.

## Pins and readiness contract

The read-only candidate gate checks:

- Strata source commit `296b59d2f73c8f04a805df29c2fa97655adace78` and llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
- the source archive (`90a002c9…`), exact extracted-source manifest (`a8de552f…`), and a complete file/directory/byte comparison between the archive and extracted tree (including serving Python files and rejecting extras), plus engine/CPU-vision build caches and exact hashes/sizes of the engine, CPU vision helper, kernel library, and built parity executables;
- the pinned calibration config, input manifest, integrated results report, and recorded GPU/RAM telemetry;
- the already verified .39 and .41 Q4 assets/readiness, the exact Q4 model/projector, and the .41 private Python 3.10.12 interpreter used by the integrated check.

The tuning workspace's `build` and source entries are expected symlinks to their exact audited calibration outputs. Readiness verifies those destinations rather than accepting arbitrary links. The candidate is intentionally not added to the generic runtime pin/selection manager.

The effective candidate profile preserves native context 262,144, CPU vision, the 8-GiB conversation cache/four cache slots, one serving slot, `--prefill auto`, explicit split `14,24,36`, and `--trim-stage-weights`. The checked-in calibration config/report contains `--pipeline-windows 2`, because that was the measured integration arm. **The launcher validates the original pinned config/report, then removes that exploratory override from the private serving copy.** No `--pipeline-windows` option is passed by the selected profile.

The corrected paired decode estimate for pipeline-windows 2 was +14.6% at the median, but its corrected 95% interval crosses zero. It is not a confirmed improvement and is not enabled in this profile. The capability report was run with that calibration override; it does not establish performance for the selected default-window serving copy.

## Recorded evidence and limits

The hash-bound integrated report recorded passing checks for:

- cache-hit/restore behavior (at least three cached-token hits and 16 restore events);
- CPU vision identifying the synthetic red target;
- a fresh 31,509-token prefill and a fresh 256,860-token native-context request;
- clean isolated shutdown and memory telemetry within the candidate's per-GPU/RAM safety bounds.

Paired screening cleared its measured thresholds for `--prefill auto` (+12.7% median prefill) and `--trim-stage-weights` (+14.0% in the tested comparison). These are specific to the `.69` calibration, configuration, and measurements. They do not establish broad speedup, response quality, context/vision quality, long-duration stability, worst-case memory, snapshot capacity, or multi-user throughput. The capability report also predates the serving-copy removal of pipeline-windows 2, so performance of that effective config is not separately benchmark-qualified.

No user traffic or production workload is migrated automatically. The candidate's existence/readiness does not itself select it or alter the .39/.41 state.

## Launch, networking, and rollback

HostLLM [7] invokes `v1strata.sh` as the normal `user`, checks readiness before acquiring the HostLLM pause lease, and is restricted to one slot/one batch group. It defaults the candidate host bind to `0.0.0.0` for the trusted LAN; set `STRATA_HOST=127.0.0.1` to use loopback. Hive's .44.1 `h-run.sh` uses the same single-slot profile and explicit trusted-LAN bind. Direct readiness/validation defaults to loopback. The Q4 launcher refuses root-owned server starts and arbitrary host addresses. The API is unauthenticated unless `STRATA_API_KEY` is supplied; firewall it and do not expose it to untrusted networks.

Hive option [16] only changes its guarded `current` profile symlink after readiness and idle/security checks. It never starts/stops Hive, OctaSpace, or an inference server. The selection takes effect on Hive's next normal miner start. Restore the .39 Hive profile with HostLLM [12] (or choose `orca-q4_k_s` in the profile selector); the separately pinned .41 profiles remain available through HostLLM [14]/[15].

Local CPU-only code/contract tests cover runtime routing, pins, tamper-sensitive config checks, the omitted windows-2 serving override, menu entries, and shell syntax. They do not perform a new live `.69` launch; use the identity-bound readiness/lease flow before any operator-authorized runtime test.
