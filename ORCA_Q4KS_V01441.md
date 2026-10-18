# Orca Q4_K_S / Strata 0.1.45 — standard profile

## Status and scope

This is the standard Orca Q4_K_S Strata route for the `.69` HostLLM/Hive node, now using the pinned Strata 0.1.45 runtime. The Q4 model assets, machine-specific preset, and calibration evidence below originate from the prior `.44.1` candidate; the launcher combines that verified preset with the separately built `.45` engine. The `.45` source/build/helper and effective configuration are provenance-checked, but **no `.45`-specific Q4 inference, speed, quality, or load result is claimed**. The historical bounded evidence is not runtime-upgrade evidence or broad qualification. Orca `.39`, `.41`, and `.44.1` remain explicit rollback routes; the generic runtime-selection manager and its assets are not rewritten by this route. For Q4_K_S, `--runtime auto` resolves directly to `.45` without changing other profiles. HostLLM [3] starts it; HostLLM [14] selects Hive profile `orca-q4_k_s-v01441` for the next normal Hive start. The installer selects this profile by default while preserving other recognized selections, and migrates an old `.39` next-start pointer to `.45`. Readiness is checked before any service pause or profile-pointer change.

The workspace is deliberately pinned to these `.69` paths:

- Tuning/effective profile: `/home/user/.local/share/localllm-strata/calibration/orca-q4ks-machine-tuning-v01441-20261015/`
- Pinned source/build workspace: `/home/user/.local/share/localllm-strata/calibration/orca-q4_k_s-v01441-prefill-20261015-retry2/`
- Integrated report: `arms/integration-auto-trim-windows2/results.json` in the tuning workspace.

This is not a portable install or preparation path. On a host without the pinned `.44.1` Q4 preset assets and prepared `.45` runtime artifacts, readiness and the Hive selector fail closed; the installer does not download/build or create either workspace.

## Pins and readiness contract

The read-only candidate gate checks the retained Q4 preset/evidence plus the new runtime:

- Strata 0.1.45 source commit `477ac13cbe8e582c20484871ce459d20d82df09d` and llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; CUDA sm_86, native experts, Orca Q4_K_S MMQ, K-quant MMQ, and the separate native-CPU vision helper are checked against the prepared build manifest/CMake caches;
- the inherited Q4 preset source commit `296b59d2f73c8f04a805df29c2fa97655adace78` and its historical llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
- the source archive (`90a002c9…`), exact extracted-source manifest (`a8de552f…`), and a complete file/directory/byte comparison between the archive and extracted tree (including serving Python files and rejecting extras), plus engine/CPU-vision build caches and exact hashes/sizes of the engine, CPU vision helper, kernel library, and built parity executables;
- the pinned calibration config, input manifest, integrated results report, and recorded GPU/RAM telemetry;
- the already verified .39 and .41 Q4 assets/readiness, the exact Q4 model/projector, and the .41 private Python 3.10.12 interpreter used by the integrated check.

The tuning workspace's `build` and source entries are expected symlinks to their exact audited calibration outputs. Readiness verifies those destinations rather than accepting arbitrary links. This profile does not change `runtime-selection.json`; Q4-specific `auto` points directly to `.45`, while `.45` is forbidden as a generic/default runtime or global selection.

The standard profile uses native context 262,144, CPU vision, the 8-GiB conversation cache/four cache slots, one serving slot, `--prefill auto`, explicit split `14,24,36`, and `--trim-stage-weights`. The checked-in calibration config/report contains `--pipeline-windows 2`, because that was the measured integration arm. **The launcher validates the original pinned config/report, then removes that exploratory override from the private serving copy.** No `--pipeline-windows` option is passed by the selected profile.

The corrected paired decode estimate for pipeline-windows 2 was +14.6% at the median, but its corrected 95% interval crosses zero. It is not a confirmed improvement and is not enabled in this profile. The capability report and paired screening are historical `.44.1` measurements run with the calibration override; they do not establish performance for the selected default-window serving copy or for Strata `.45`.

## Recorded evidence and limits

The hash-bound integrated report recorded passing checks for:

- cache-hit/restore behavior (at least three cached-token hits and 16 restore events);
- CPU vision identifying the synthetic red target;
- a fresh 31,509-token prefill and a fresh 256,860-token native-context request;
- clean isolated shutdown and memory telemetry within the candidate's per-GPU/RAM safety bounds.

Historical `.44.1` paired screening cleared its measured thresholds for `--prefill auto` (+12.7% median prefill) and `--trim-stage-weights` (+14.0% in the tested comparison). These are specific to the `.69` calibration, configuration, runtime, and measurements; none is evidence of a `.45` speedup. They do not establish broad speedup, response quality, context/vision quality, long-duration stability, worst-case memory, snapshot capacity, or multi-user throughput. The capability report also predates the serving-copy removal of pipeline-windows 2, so performance of that effective config is not separately benchmark-qualified.

The installer/menu do not stop or migrate a running service; the next normal Hive start uses `.45` by default. Readiness and the historical `.44.1` evidence do not constitute `.45` qualification. Orca `.39`/`.41`/`.44.1` remain available only as explicit rollback routes.

## Launch, networking, and rollback

HostLLM [3] invokes `v1strata.sh` as the normal `user`, checks readiness before acquiring the HostLLM pause lease, and is restricted to one slot/one batch group. It defaults to loopback; set `STRATA_HOST=0.0.0.0` explicitly for a trusted LAN, and provide an API key at the secure prompt if desired. Hive's `.45` `h-run.sh` uses the same single-slot profile and explicit trusted-LAN bind. Direct readiness/validation defaults to loopback. The Q4 launcher refuses root-owned server starts and arbitrary host addresses. Without a key, a LAN-bound API is unauthenticated; firewall it and do not expose it to untrusted networks.

HostLLM [14] and Hive menu option [3] only change the guarded `current` profile symlink after readiness and idle/security checks. They never start/stop Hive, OctaSpace, or an inference server. The selection takes effect on Hive's next normal miner start. Restore the `.39` rollback profile explicitly with `sudo /hive/miners/custom/llm-hosting/profile-menu.sh --profile orca-q4_k_s`; `.41` Hive profiles and the Q4 `.44.1` runtime also remain available explicitly.

Local CPU-only code/contract tests cover runtime routing, pins, tamper-sensitive config checks, the omitted windows-2 serving override, menu entries, and shell syntax. They do not perform a new live `.69` launch; use the identity-bound readiness/lease flow before any operator-authorized runtime test.
