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

Start the initial 32K text-only server with `./HostLLM.sh --run-q4ks143-32k-validation` (it binds loopback only). In another terminal run `./v1strata.sh --validate-orca-q4ks-143-32k`, then stop the server with Ctrl+C. The launcher independently rejects explicit LAN binding while the .43 candidate remains at this 32K stage. Native-context/CPU-vision migration (`--configure-orca-q4ks-143-native-context`, `--enable-orca-q4ks-143-vision`) and the 2/2 route (`./HostLLM.sh --run-q4ks143-two-slot`) are implemented but remain unrun. A passing 32K functional gate does not authorize those steps: the bounded prefill comparison below is a negative signal, so keep .43 at 32K/text-only pending a controlled performance rerun and separate review.

HostLLM/Hive menu entries remain unchanged until a separate qualification/review.

## Recorded node evidence and decision (2026-10-12)

These are bounded operator-reported results from the 4× RTX 3090 node. They are not cryptographic attestations or general quality/capacity/load qualifications.

- **Exact-shard parity passed:** both `native_expert_parity` and `ple_q5_parity` passed against the pinned Q4 assets. See `results/orca-q4ks-v0143-parity-20261012.json`.
- **32K functional gate passed:** health, arithmetic, strict JSON, tool call, conversation-cache configuration and A/B/A/B reuse, and memory checks passed. MTP drafted 35 tokens and accepted 31. Peak GPU memory was 23,732 / 19,036 / 21,944 / 23,160 MiB (87,872 MiB aggregate). This is single-slot, text-only, 32K evidence only; see `results/orca-q4ks-v0143-validation-32k-20261012.json`.
- **Bounded prefill result:** the same three saved .39 fixture prompts (5,295 input tokens each; exact hashes; `cache_n=0`; 16 output tokens) measured **590.5 / 675.1 / 547.1 tok/s** on .43, median **590.5 tok/s**. The saved .41 comparison measured **795.8 / 970.6 / 935.1 tok/s**, median **935.1 tok/s**; .43's median is **36.85% lower**, and all three .43 samples were below all three saved .41 samples. See `results/orca-q4ks-v0143-prefill-20261012.json` and the sanitized prior measurement `results/orca-q4ks-v0141-prefill-20261009.json`.

The validation report hashes the candidate's profile preset; the prefill report hashes HostLLM's live run-config. Those hashes intentionally differ only in the host binding (`0.0.0.0` preset vs `127.0.0.1` loopback) and server-log path. The live-candidate guard checked the model/runtime arguments against the exact 32K preset before and after each prefill request.

This is an observed negative signal, **not a causal runtime-version estimate**: according to the operator record, the .41 probe ran on the same node about 71 hours earlier, not as an interleaved A/B run. The .39 reference (median 956.4 tok/s; .43 is 38.26% lower) used 262144 context with CPU vision, unlike the 32768 text-only .43 candidate, so that comparison is especially not runtime-only. The .43 launcher also leaves route-tail-skip at the upstream default; the probe does not establish whether it activated, explain the measured difference, or qualify answer quality. All probes were short and one-slot.

**Decision: do not promote .43.** Keep it isolated; leave .39 as default and .41 as opt-in. No native-context, CPU-vision, multi-slot, LAN, HostLLM-menu, or Hive-menu promotion was performed. Reconsider only after a controlled, contemporaneous comparison and the candidate's separate qualification gates.
