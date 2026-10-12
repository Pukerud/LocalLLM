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

Start the initial 32K text-only server with `./HostLLM.sh --run-q4ks143-32k-validation` (it binds loopback only). In another terminal run `./v1strata.sh --validate-orca-q4ks-143-32k`, then stop the server with Ctrl+C. The launcher independently rejects explicit LAN binding while the .43 candidate remains at this unvalidated 32K stage. Only after its evidence gate passes, migrate and enable CPU vision with `--configure-orca-q4ks-143-native-context` and `--enable-orca-q4ks-143-vision`; validate native behavior before LAN serving via `./HostLLM.sh --run-q4ks143-candidate` and `./v1strata.sh --validate-orca-q4ks-143-native`. The bounded two-slot route is `./HostLLM.sh --run-q4ks143-two-slot` and remains experimental pending its own memory/performance checks.

HostLLM/Hive menu entries remain unchanged until a separate qualification/review.

No .43 build, parity run, GPU inference, or node operation is claimed by this source-only integration.
