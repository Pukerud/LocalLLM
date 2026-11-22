# Unsloth V3.1 IQ4_XS — tuned .59 Hive trial, 2026-11-14

## Latest — 2026-11-22, read-only hardware-policy repair; [23] restored

The operator's current selection is **tuned .59 Hive[23]/low[11]**, restored through its existing normal Hive supervision after repeated hardware-guard refusals. Actual GPU identities/driver/VRAM were unchanged; power caps had differed from the old exact300/300/300/280W snapshot. The operator explicitly approved **both300/300/300/280W and300/300/300/300W**.

The read-only guard now accepts exactly those two complete power profiles, with clearer field-specific errors. It still rejects different UUID/order/driver/kernel/VRAM, all other cap sets, and current-boot GPU faults. **No power limits, clocks, drivers, Hive/cloud settings, resets or reboots were changed.** Only the shared hardware-policy source and that source hash in both .59 permissions changed; all serving configs, frozen recipe/qualification/native binaries/model/template/tuning/cache/output defaults stayed intact. No manual or duplicate start was issued; no rental/service was stopped.

Actual caps300/300/300/300, fresh Hive-owned token/UID1000/birth/exe/argv/parent proof, key-free LAN63 completeSSE/usage, loaded .59/native262K/two slots and defaultn_predict-1 passed. `undeclared_tool_calls="call"` retained. **620Python OK/4skipped +all8shell**; admission active, no lease/journal/stop marker or pending test/controller/switch. This is startup-repair evidence, not a new quality/speed benchmark. See `results/strata-v0159-power-policy-repair-20261122.json` and matching pi-profile report.

## Historical — 2026-11-15, setting-only update; .46 then stayed running

The operator currently uses **.46 Hive[21]/low[9]**, reporting better real-task/tool reliability than .59, and requested preparing [23] for a later manual test **without disturbing that session**.

Only [23]'s canonical configuration now adds **`"undeclared_tool_calls": "call"`**. Its inactive `config.expected.json` and exact root grant's configuration/profile-source hashes were refreshed; original[22]/[21], frozen recipe/qualification/runtime/binaries/model/template/sampling/features and all Hive bundle/selector bytes remain unchanged. A future normal [23] launch gets this setting; no current run-config/history/adaptive state was rewritten, no switch/restart/generation/controller was run or queued. The previous overnight activation record below is historical, not current selection or replay authority.

The setting lets a **validating client** reject an unknown tool with a structured error, allowing correction/retry; the server does not execute unknown tools or guess names. Actual-config **CPU-only mock HTTP/SSE12requests** passed (unknown call→client error→corrected call→63; no-tools/prose/default-text controls). Real UID1000 readiness/independent permission reads,610Python OK/4skipped and all8shell suites passed. This is offline preparation, **not a demonstrated fix for real model tasks**; the operator will switch/test manually later. The original56/72 results were versus **stock .59**, not .46; forced `emit_number` tests were not full agent-loop qualification.

See `results/strata-v0159-tool-call-compat-20261115.json` and pi-profile `reports/2026-11-15-strata-v0159-tool-call-compat.md`. Upstream context: [#5933](https://github.com/Niko1221/Strata/issues/5933), [#4622](https://github.com/Niko1221/Strata/issues/4622). This does not install experimental parser/template/CUDA patches or remove lifecycle/auth/rental gates.

## Historical serving selection — 2026-11-14

**Hive [23] / low selector [11]** is the separately published tuned profile:
`unsloth-v31-iq4xs-q8-cvec-2slot-v0159-tuned`.
URL: **http://192.168.1.69:8080/**, key-free on the trusted LAN.

Actual normal Hive startup and exact-owned stop/drain/new-token/new-frontend restart passed as UID1000. Full real features passed both before and after restart. The **same-session30-minute soak passed:60cycles/126complete streams**, two occupied slots, vision/high reasoning, no default-cap change; max73C,minGPUfree3826/3048/2802/2853MiB,minRAM39.79GiB. A real **other-LAN-host .188 UID1000** request replied63 without Authorization/x-api-key/dummy credentials. Final live JSON/mirror and regression passed. No lease/journal/stop marker, dated controller or further automatic action remains; [23] is left running.

Original **[22]/low[10] official .59**, **[21]/low[9] .46**, SC117[18] and Swift[12] remain available. Swift stays the installation default. Running/selected labels are separate; selecting the already-selected profile does not force a restart. Old .59 model-ID/profile aliases remain accepted by [23].

## Frozen model/features — no capacity/quality shortcut

- Same four V3.1 IQ4_XS GGUFs at revision `d96201194f103c906994f283f504209dc8d2a10c`; same Q8 PLE and original-Qwen projection ON layers4–44. No weight/header/quant rewrite, rerouting bias or expert skipping.
- Original BF16 **CPU** vision helper/projector; stock Q2-expert/Q8-dense MTP, depth3. Actual solo drafting observed in every measured sample; concurrent windows retain the established no-draft behavior.
- Native **262144 per chat**, two serving slots/two groups, INT8 KV/32768 resident, 16384MiB/eight histories. Eight parked histories are not eight simultaneous/full-window chats.
- **`n_predict=-1`** remains the default. Benchmark/quality caps are request-local. Input/history/images/reasoning/output share the native context.
- Separate permanent state: `~/.local/state/locallm-strata/unsloth-v31-iq4xs-q8-cvec-2slot-v0159-tuned-trial/`. Fresh generic-only seed; no live/trial history or adaptive-state copy/reset.

## Winning recipe

`strata_v0159_tuned_recipe.json` freezes the independently qualified `batch-pipeline` recipe:

| Change | Value |
|---|---|
| Separate native .59 build | Q5_0 MMQ capability ON |
| Layer cuts / original GPU order | **12,24,36** / as-given |
| `STRATA_PF_HC_REGION` | **1** |
| `STRATA_PF_GROUP_SCRATCH` | **1** |
| `STRATA_PW_WITH_BATCH` | **1**, after measured headroom |

The nine scheduling/copy switches remain enabled; IQ4 W4A8=1 and two-pass selection=1 remain enabled. GDN_TC=0, GDN_CONVL2=0 and HC_Q8=0 remain conservative. Routing/CPU-share/batch-MTP precision/behavior opt-ins remain off. Drafting=MTP, split-engine=off, upstream orphan adoption/retry disabled, external lifecycle guard retained.

The fastest fully qualified numerical combination was only about2% ahead on the representative latency score. The predeclared within3% rule therefore chose the faster qualifying memory/layout/scheduling recipe without the GDN/HC precision-changing switches.

Balanced screening left roughly3–4GiB free on each card, versus tight/uneven stock headroom. Only then was the extra solo pipeline allocation tested. Logs confirm an8192-token prompt chunk/96-slot ring and both scratch-reuse paths; group scratch held15872 rows rather than81920. Q8 PLE is **not** HC_Q8 or expert quantization.

## Build provenance / rollback

No running binary, original runtime, system toolkit, driver, kernel, power, clock, unlock or boot configuration was overwritten.

Separate UID1000 builds used final source `d2f096475817d4666c364c820b0ca830f751cfc5`, existing CUDA12.9.86, SM86-real, portable AVX2, two jobs, nice19/ionice-idle, GPUs hidden. Official .59 native rc-build/CUDA13 and final frontend tag are not falsely equated. A matched custom Q5-OFF control separated compiler/tag/build differences from the Q5 feature.

- Q5-ON ELF: `5d9184cef67c6ad898aecb8b2733ba0c3a4e285f13ab0159aaa84adb5eaa0d9e` (88366688bytes).
- Q5-OFF ELF: `45c8e1d5e9518c76a9adbf0d7e7bab540b31ad4ff7ad60a039e183ac7cba9f20` (86351904bytes).
- Both custom `libstrata_kq.so`: `9c3bb058dbac8d893fcda2ae8e2f0fb25fb1fc3d59ba47d7e794e2a7be0b8a1d`.

Despite the IQ4_XS label,33of48 expert layers contain Q5_0 down-projections. The official .59 binary lacked that MMQ instance. The build flag is historically named `STRATA_ORCA_Q4KS_MMQ=ON`; enabling this kernel does **not** restore retired Orca weights/routes. It is not an environment toggle.

Native executable/libraries/build receipt and resolved system dependencies are checksum-bound by the recipe. Original .59 runtime/CPU vision seals and full immutable model hashes still precede every GPU spawn.

## Repeated, cache-free results versus official .59

Six actual samples per workload/arm across fresh generic-only states. All sampling penalties explicit/off, temperature0, top-p1/top-k1/min-p0/seed719, no-thinking/projectionON,128 actual output tokens each, actual KV reuse0. CPU count fitting matched actual generation usage. Complete SSE/native clocks/client monotonic timing retained privately.

Fresh stock controls bracket both candidate blocks; the middle stock control is retained as a drift check. This was not a randomized/exhaustive factorial trial. Short-prompt timings are variable.

| Actual input | Prefill stock→tuned tok/s | Decode stock→tuned tok/s | Request wall stock→tuned |
|---:|---:|---:|---:|
|256|126.35→183.10|119.65→135.20|3.100→2.344s|
|4096|901.10→2001.40|113.40→134.20|5.675→3.006s|
|32768|1355.80→4730.35|110.75→124.95|25.347→8.020s|
|131072|1374.05→6419.85|96.35→113.25|96.802→21.621s|

Half-context prefill **4.67×**, TTFT95.476→20.505s, wall latency4.48× better. Decode improved roughly13–18% across these fixtures. Two independent ~4K/256-output requests occupied both slots: pair wall13.230→7.858s (**1.68×**). The five-workload geometric latency score was2.264×; this is fixture-specific, not a universal throughput multiplier.

Native retrieval actually processed196608 and259000 input tokens, both correct: stock145.37/202.66s versus tuned30.09/41.82s. One populated native-window chat at a time was tested, **not two concurrent full windows**. No512K work occurred.

## Quality and real feature gates

All three finalists matched reference outcomes: **32/48 primary +24/24 holdout =56/72 correct**, with **no new reference-passed-task failures**. The16 existing strict no-thinking arithmetic/logic/code-trace/instruction failures were not hidden or relabeled as new regressions. The holdout covered high reasoning, multilingual JSON, multi-turn recall and treating quoted data as data. Primary checks included coding in an AST-restricted CPU sandbox, extraction, instructions, retrieval, vision and tool formatting. Same weights are not a universal unchanged-intelligence/answer guarantee.

Real UID1000/key-free LAN checks passed on the actual new normal Hive process, then again after exact owned restart: UI/health/models/loaded status,63 complete SSE, high-thinking2099 with real reasoning, projectionON/OFF telemetry, ordered/swapped red/blue vision, independent concurrent tags with **two occupied slots**, forced `emit_number(63)` formatting **without external execution**, badHostPOST403/foreignOriginPOST403/sameOriginPOST200, and8/8 distinct~4K histories reused>1000tokens. Old .59 alias63 passed too.

## Safety / lifecycle / regression

Renters always win; OctaSpace admission stayed active. No renter/container/service was stopped, no unknown workload signalled, no cloud rig/wallet/OC bytes restored from stale snapshots. Full model hashes and exact UID/exe/argv/birth/session/parent/token/pidfd gates remain.

One between-screen pre-spawn guard refused before any gdn-conv runtime/GPU existed. The original .59 was safely restored. Fresh scoped remaining tests waited for natural quiescence; no process filtering/override and no spawned/GPU-failure retry. OctaSpace's legitimate miner-start journal coincided with that refusal, but its precise process cause was not captured. Hardware/current-boot checks remained clear. No reset/reboot/TensorSharp/512K retry.

The new Hive wrapper additionally refuses early on a manual HostLLM lease and checks the exact qualified GPU/driver/power/kernel/current-boot fault inventory before load. It never resets or retries hardware.

Original [22] permission was refreshed only for the reviewed `strata_launcher.py`, `engine_safety.py` and `hive_profile_switch.py` source integration: model/config/official payload contract unchanged, real UID1000 permission read passed. [23] has a separate root0644 checksum-bound grant. Expiring temporary test permission is archived; historical preparation flags/controllers/leases/PIDs/tokens are audit only, never replay/stop authority.

Final regression: **607 Python tests OK,4 skipped; all8 shell suites passed.** The original current bundle was not hot-overwritten: inactive publication had zero lifecycle actions, then a freshly checked normal Hive switch selected[23].

## Evidence / future maintenance

- Frozen qualification: `results/strata-v0159-overnight-qualification-20261114.json`.
- Exact configuration/dependency/build recipe: `strata_v0159_tuned_recipe.json`.
- Final live/soak checkpoint: `results/strata-v0159-overnight-live-20261114.json`.
- Public operator report: pi-profile `reports/2026-11-14-strata-v0159-overnight-tuned.md`.
- `STRATA_OVERNIGHT_20261113.md` records the bounded18-case plan.

The dated controllers are completed audit tools, **not commands to rerun on the active miner**. A new experiment/rebuild needs a fresh explicitly authorized namespace, provenance/policy and current rental/request/ownership/hardware checks. Never replay saved identities or overwrite the sealed active binary. Short soak/functional/quality checks are bounded, not full-context concurrency or indefinite-stability qualification.
