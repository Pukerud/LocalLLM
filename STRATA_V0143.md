# Orca Q4_K_S / Strata 0.1.43 isolated candidate

This is an explicit, version-bound Q4 candidate. The generic `strata_runtime.VERSION` stays **0.1.41**, the preserved Q4 default stays **0.1.39**, and no HostLLM/Hive menu route or default is changed. The new Hive profile payloads are deliberately not added to the existing selector.

## Pins and build contract

- Strata source: `7695859aa13afb248bdb4d36f9b2e95707db079d` (v0.1.43).
- llama.cpp: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- The prepared .43 `tools/vision` source contract is `1fb2b3fb797ec402`. It differs from .41's `6f47618c713ce0a2`; .43 builds and validates against its own prepared .43 vision helper. A .41 helper/evidence cannot satisfy the .43 checks.
- Candidate CMake options remain CUDA, native experts, Orca Q4_K_S MMQ and build tests enabled, Release, `sm_86`, and the isolated llama.cpp path. The pinned v0.1.43 CMake source still defines `STRATA_NATIVE_EXPERTS`, `STRATA_ORCA_Q4KS_MMQ`, `native_expert_parity`, and `ple_q5_parity`.
- The source's CLI still accepts `--max-context`, `--conversation-cache-mib`, `--conversation-cache-slots`, and explicit `--batch-groups`; the server exposes the health/status/metrics and token-count APIs used by the copied .41 gates. The .43 `/v1/status` `in_flight` count includes queued requests.

## Runtime behavior and output-affecting defaults

The .43 launcher retains the already-used safeguard environment values: `STRATA_VERIFY_ALL_RESIDENT=1`, while stage pinning, residency-biased routing, embedding reuse accounting, CPU prefill sharing, and batch MTP are explicitly disabled. `STRATA_ROUTE_TAIL_SKIP` is intentionally **not set**: on CUDA, upstream may enable rank-7 route-tail skipping when experts are missed and its runtime conditions permit. It can change generated output slightly; it may be inactive when there are no misses, nearly all experts are resident, or batched slots are used. Do not infer .41-identical answers from the shared model/profile.

The .43 native-pack startup also probes host-to-device PCIe bandwidth and derives the initial PCIe share (base 0.55, scaled below 20 GB/s); `STRATA_PCIE_FRAC_DEFAULT=old` opts out. It is not a new manually tuned split. Two-stage HIP-only auto-refinement remains opt-in and is not enabled here. The candidate keeps the established GPU order and automatic layer split.

Only 1/1 and 2/2 slot/group combinations are accepted. This is launch/readiness routing, not evidence of .43 multi-slot capacity or performance. The independent live 32K gate remains the single-slot text-only gate; native-context migration and CPU vision require that candidate's own evidence.

## Explicit workflow

The .43 generic runtime is prepared only as an isolated build/helper source; it is not eligible for generic runtime selection or promotion. The generic build is a required prerequisite for Q4 candidate preparation and does not switch the runtime selection or replace .39/.41:

```bash
sudo -n ./v1strata.sh --prepare-runtime --runtime 0.1.43
./HostLLM.sh --prepare-q4ks143
./HostLLM.sh --verify-q4ks143-parity
```

Start the initial 32K text-only server with `./HostLLM.sh --run-q4ks143-32k-validation` (it binds loopback only). In another terminal run `./v1strata.sh --validate-orca-q4ks-143-32k`, then stop the server with Ctrl+C. The launcher independently rejects explicit LAN binding while the .43 candidate remains at this 32K stage. Native-context/CPU-vision migration (`--configure-orca-q4ks-143-native-context`, `--enable-orca-q4ks-143-vision`) and the 2/2 route (`./HostLLM.sh --run-q4ks143-two-slot`) are implemented but remain unrun. The functional gate and clock-profile retest do not authorize those steps: .43 remains 32K/text-only and isolated because the bounded retest did not show a meaningful advantage over the saved .41 result.

HostLLM/Hive menu entries remain unchanged until a separate qualification/review.

## Recorded node evidence and decision (2026-10-12)

These are bounded operator-reported results from the 4× RTX 3090 node. They are not cryptographic attestations or general quality/capacity/load qualifications.

- **Exact-shard parity passed:** both `native_expert_parity` and `ple_q5_parity` passed against the pinned Q4 assets. See `results/orca-q4ks-v0143-parity-20261012.json`.
- **32K functional gate passed:** health, arithmetic, strict JSON, tool call, conversation-cache configuration and A/B/A/B reuse, and memory checks passed. MTP drafted 35 tokens and accepted 31. Peak GPU memory was 23,732 / 19,036 / 21,944 / 23,160 MiB (87,872 MiB aggregate). This is single-slot, text-only, 32K evidence only; see `results/orca-q4ks-v0143-validation-32k-20261012.json`.
- **Initial bounded prefill run:** the same three saved .39 fixtures (5,295 input tokens each; exact hashes; `cache_n=0`; 16 output tokens) measured **590.5 / 675.1 / 547.1 tok/s** on .43, median **590.5 tok/s**. The operator reports this was run under the lower-clock mining profile; that run has no GPU clock telemetry. See `results/orca-q4ks-v0143-prefill-20261012.json`.
- **Clock-profile retest:** unchanged .43 runtime/profile (`--prefill 512`, one slot, 32K text-only) measured **932.4 / 933.6 / 919.8 tok/s**, median **932.4 tok/s**. This is **57.90% above** the initial .43 median. One-Hz `nvidia-smi` samples during the retest recorded a 300 W limit on each GPU, 9501 MHz memory clocks on loaded samples, and per-GPU median graphics clocks of 1965 / 1920 / 1935 / 1905 MHz (loaded defined as GPU utilization ≥50%). Initial idle samples were 210 MHz core / 405 MHz memory; those idle readings did not describe clocks during inference. Peak sampled power was 276.85 / 248.15 / 273.92 / 282.19 W; max temperature was 75 °C. The earlier slow run lacks comparable telemetry, so this improvement is associated with the profile change but does **not** establish causation. See `results/orca-q4ks-v0143-prefill-llm-clocks-20261012.json` and raw `results/orca-q4ks-v0143-gpu-telemetry-llm-clocks-20261012.csv`.
- **Comparison with saved .41:** the saved .41 median is **935.1 tok/s** (795.8 / 970.6 / 935.1); the retested .43 median is **0.29% lower**, effectively a tie at this precision, not a measured .43 advantage. The .41 run was about 71 hours earlier, not an interleaved A/B. Both recorded candidates used 32K, text-only, one slot, and the same fixture hashes. The saved .39 median is **956.4 tok/s**; the retested .43 is 2.51% lower, but .39 used 262144 context with CPU vision configured and is not a runtime-only comparison. See `results/orca-q4ks-v0141-prefill-20261009.json`.
- **Pinned .43 calibration:** ran `tools/calibrate.py` directly against this exact 32K/text-only candidate in measurement-only mode (not `setup.py --calibrate`, which by default refreshes the installed engine and persists results into the selected run config and shared settings). It took **727 seconds** and selected no settings (`settings: {}`): no candidate exceeded the tool's >3% improvement threshold. The PCIe confirmation was 121.68 tok/s at the default 0.09 share versus 121.65 at 0.2 (-0.03%); the default 0.5 speculative floor and 5 workers remained preferable, and both tested adaptive tiers were slower. The tool's final-stage 104.4 tok/s is a decode-only calibration result, not comparable to the API prefill benchmark or its shorter sweep samples. The candidate config, pinned tool, shared settings, and automatic .39 selection were verified unchanged; no settings were applied. See `results/orca-q4ks-v0143-calibration-20261012.json`.

The validation report hashes the candidate's profile preset; the prefill reports hash HostLLM's live run-config. The live-candidate guard checked the model/runtime arguments before and after each measured request. .43 continues to leave route-tail-skip at the upstream default; these probes do not establish whether it activated or qualify answer quality.

The README/docs show no Strata GPU-clock control. The pinned calibration tests host/model-specific PCIe share, speculative threshold, CPU workers, and cache-adaptation settings, but it does not tune GPU clocks or prompt/prefill throughput. This candidate's measurement-only run found no >3% decode improvement and persisted nothing. `--prefill auto` (larger, memory-dependent prompt chunks; the Q4_K_S example uses `--prefill 512`) and the broader `tools/strata_bench.py` remain separate, unrun tests.

**Decision: do not promote .43.** The initial low result improved substantially under the LLM-hosting profile, but the corrected-clock run only ties the saved .41 result and is not contemporaneous; the separate calibration found no decode tuning win to change that assessment. Keep .43 isolated; leave .39 as default and .41 as opt-in. Native-context/CPU-vision, multi-slot, LAN, HostLLM-menu, and Hive-menu promotion remain unqualified and unchanged. Any reconsideration should start with an interleaved .41/.43 comparison under matched settings and explicit route-tail-skip control, followed by the separate qualification gates.
