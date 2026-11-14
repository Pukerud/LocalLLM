# Strata0.2.13 / engine0.1.59 — official rollback / historical activation

**Later operator-authorized overnight work,2026-11-14:** separate tuned Hive[23]/low[11] is now active; official[22] below is retained unchanged as rollback. See [the tuned profile](STRATA_V0159_TUNED.md). This historical report is not an instruction to revert the newer selection or replay its completed activation controller.

## Historical operator selection — 2026-11-13

After saving the current .46 baseline, the operator explicitly approved .59 switching/testing, then explicitly asked to **leave the new version up**. **Hive[22] / low[10] / `unsloth-v31-iq4xs-q8-cvec-2slot-v0159` is ACTIVE.** No further switch/restart is pending. Original .46 Hive[21]/low[9], Swift[1]/[12] and SC117[7]/[18] remain retained; installation-default Swift unchanged.

Real LAN text/thinking/vision/projection/tools-format/two occupied slots/eight histories/Host-Origin checks and actual owned stop/drain/restart/new-token/new-process/fresh63 passed. Backend inference family is UID1000, normal LAN remains API-key-free. **Bounded functional trial, NOT a speed-win or broad quality/capacity/stability qualification.**

## Measured comparison

Medians of3 sequential runs,128 actual output tokens,actual native KV reuse0, exact frozen messages and request-local temperature0/top-p1/top-k1/seed719/no-thinking/projectionON:

| Input tokens | .46 prefill | .59 prefill | .46 decode | .59 decode | First token .46 → .59 |
|---|---:|---:|---:|---:|---:|
| 256 | 239.9 | 117.7 | 119.7 | 125.8 | 1.083 →2.190s |
| 131072 (half native) | 1794.1 | 1387.5 | 99.6 | 95.2 | 73.141 →94.552s |

Rates tokens/s. Half-context prefill22.7% slower/decode4.4% slower; short decode5.1% faster but prefill50.9% slower. Short-prefill variation is substantial. Operator chose to keep .59 running after being told about the prefill regression; no tuning was silently applied.

Same frozen fixture SHA256 `e3b1b43331c7fee033c47178571cc622272679d7aa8524aafd358b596b975f78`. Old .46 was warm/adapted; new .59 started its separate generic seed. No state reset/copy, no cold-disk claim. This is a prepared-profile migration comparison, not isolated engine causality. **0/6 capped greedy texts were byte-identical**; same weights do not guarantee identical answers or unchanged intelligence. Simple functional checks are not a broad quality/refusal evaluation.

[Live JSON](results/hostllm-strata-v0213-live-20261113.json) / [baseline JSON](results/hostllm-strata-v0146-baseline-20261113.json).

## Preserved contract and software pins

- Frozen V3.1 IQ4_XS main GGUFs/revision`d96201194f103c906994f283f504209dc8d2a10c`, Q8 PLE, original-Qwen per-layer4–44 projection ON, BF16 CPU vision, stock Q2-expert/Q8-dense MTP3.
- Native262144 **per chat**,2slots/2groups,INT8/32768 resident KV,16384MiB/eight cached histories,auto split/as-given GPU order. Actual .59 split17,20,33 on both starts.
- Original GGUFs/derived compatibility pack unchanged. No live .46 KV/conversations/adaptation copied. .59 uses isolated `unsloth-v31-iq4xs-q8-cvec-2slot-v0159-trial` state.
- No lower main quant, routing bias, DFlash2, smaller context/fewer slots or global output/thinking cap. `n_predict=-1`; benchmark limits were request-local. Eight histories are not eight concurrent full-window chats.
- Scheduling/copy defaults and reviewed `STRATA_IQ4_W4A8=1` / `STRATA_SELECT_TWO_PASS=1`; GDN/HC arithmetic-changing defaults, routing shortcuts, batch-MTP and extra solo-pipeline-with-batch VRAM remain off. HC_Q8 is not the model's Q8 PLE. Declared windows2 does not imply another effective pipeline with batch slots.
- App0.2.13 frontend`d2f096475817d4666c364c820b0ca830f751cfc5`; native build`712fec6b4297+cuda13.0`, separately recorded.
- Official723498702-byte `strata-engine-0.1.59-linux-x86_64-cuda-cuda13.tar.gz`, SHA256`42ab04fa3201e41bef5fac28c05f5c35c909419ec937d879b505bf1516742610`. Ed25519 signature/all payload hashes/SM86 verified. Private runtime/dependencies/packaged CUDA13 libraries; no system CUDA/driver/kernel change.
- Portable-AVX2 CPU-only vision helper built UID1000/nice19/ionice idle/2jobs, GPUs hidden, existing pinned llama.cpp read only. Serving engine directory excludes release split-engine/GPU-vision executables.

Projection changes activations, not original weights; experimental original-Qwen-specific behavior, not permanent abliteration or ordinary-answer equivalence.

## Actual real-LAN validation

UID1000 client, intended .59/model/native262K/two-slot target, no Authorization/x-api-key/dummy token:

- UI/loaded status/health/models;63 completeSSE; high2099 with reasoning.
- Projection ON/OFF telemetry; ordered red-blue and swapped blue-red image answers.
- CEDAR37/MAPLE82 complete independent streams, **two simultaneously occupied slots**.
- Forced emit_number63 formatting only, no external execution; badHost403/foreignOrigin403/sameOrigin200 POSTs.
- **8/8 ~4K histories**, each reused>1000tokens.
- Actual token/UID/exe/argv/birth/parent/session-owned Hive stop and complete drain; normal restart observed without duplicate start, new token/process and fresh63 passed.
- Selector-only publication preserved active server/bundle/links/cloud/admission/markers with zero lifecycle actions; fresh63 afterward.

Final regression **558 Python OK/4skipped +all8 shell suites**. Historical offline543/local,43/upstream targeted CPU, MOCK frontend and real CPU encoder checks remain preparation evidence. Full upstream discovery timed out240s/incomplete, not a pass claim.

## Permission, issues and history

The real separate permission `/var/lib/localllm-strata/approvals/unsloth-v31-iq4xs-v0159.json` is now explicitly granted: root0644/nonsymlink/single-link/safe-root directories/exact config-model-runtime-engine-critical-source hashes. Env/CLI selection/offline receipts/saved PIDs never grant permission. Original offline prepared.json and five-file bundle preparation flags remain historical immutable preparation facts, **not current live permission or validation**. Some preparation readiness log wording still says locked; actual current menu status and real gate validate the separate live policy.

First startup refused **before main-LLM/GPU load** because root mkdir0755 under umask077 produced approvals0700. Corrected only that newly created public permission-data directory to0755, not private state/task directories; exact failed task grant archived and updated exact-policy grant published. Publication now explicitly sets new-directory mode and performs a **real UID1000 policy read before stop/switch**. Existing private directories still require explicit administrator correction. Real restrictive-umask fixture/four new tests pass; no numerical/auth/runtime/model/config change or GPU fault retry.

A list-vs-dict diagnostic was corrected without changing ownership rules. A90s start-observation timeout did not cancel/retry the normal asynchronous hashing/loading. After actual owned stop, normal Hive startup appeared; observed new identity instead of duplicating it. No controller remains pending.

Locked tests now use their own absent permission path rather than borrowing a live deployment grant. Selector publisher now requires actual live preflight for an approved prepared profile instead of misclassifying it as retired; unknown/retired selections and active hot overwrites still refuse. Low selector dynamically displays TRIAL AUTHORIZED or LOCKED, keeping its independent mandatory pre-mutation gate.

No numerical/auth patch. Only checksum-bound opt-in `STRATA_EXTERNAL_LIFECYCLE=1` frontend guard prevents native restart before close/spawn without fresh external safety checks. Orphan adoption disabled/startup retries0. Exact source hashes:

- Original server`398dea98fd319ad3b570a006dd1a4c1907480cb1ab07737368382620d85f2d14`.
- Patched server`1229beb5123d20442e040df1b8a38f2c649615414aea41d1cb8101bbdd795700`.

Full immutable shared data hashes still precede every native spawn. Preserve optional configured auth, Host/Origin and token/pidfd/UID/parent/renter gates.

## Safety / no pending work

Admission remained ACTIVE; no renter/container/unknown workload stopped. Original cloud bytes/four GPUs/driver/kernel/power/boot preserved. No TensorSharp/512K/system/clock/unlock/reset/reboot. During speed run max72C/minimum MemAvailable39.75GiB/minimum free VRAM650/14918/1760/1177MiB; still tight headroom, not full-context capacity proof.

Private `/var/lib/localllm-strata/strata-v0213-activation-20261113/`, baseline task and offline-prep task are completed audit history, **not replay/stop authority**. Normal requested Hive[22] remains up; no test/controller/switch/retry/reboot queued. Renters always win.
