# Menu[24]: 512K context and 64K web-UI output defaults

## Applied and live verified — November24, 2026

The operator explicitly requested these defaults on the existing new **Hive[24]/low[12]** option, then requested **OSN and the miner running when done**. Both are now active. Fresh actual UID1000/token/birth/exe/argv/parent proof, loaded app0.2.20/engine0.1.66,524288 context/two slots and key-free LAN reply63 passed after a normal owned stop and fresh-token/new-process restart.

| Setting | Menu[24] default |
|---|---|
| Effective context per chat | **524288 (512K), YaRN2** |
| Original/trained context |262144; extension is not native training |
| Strata web UI maximum output | **65536 (64K)** |
| API default output | **Unlimited (`n_predict=-1`)** |
| Thinking / streaming | Existing High default / streaming unchanged |
| Model/features | Same IQ4_XS/Q8 PLE/projection ON4–44/BF16 CPU vision/stock MTP3 |
| Scheduling/cache | Same2slots/2groups/16384MiB/eight histories/INT8-32768/conservative .66 tuning |

**Hard-refresh the web UI (Ctrl+F5).** For this model only, old blank/16000/16384 global browser defaults seed a separate model-scoped preference with65536. Global preferences for other options and chat messages are not rewritten. Later explicit custom limits or blank **Until done** remain respected; Reset uses64K for [24]. No automatic POST of shared defaults is introduced. An actual native UI request sent65536 and streamed reply63, while `/props` remained `n_predict=-1`; an explicit131072 API cap also completed a short correct reply, proving64K is not a backend hard clamp.

Input, thinking and final output still share the context. A510000-token prompt leaves roughly14288 tokens before reserve, not another64K. The advice to compact a coding client's history around400K is not a new Strata server auto-compaction feature or an altered global budget.

## Narrow implementation and preservation

- Only .66 receives the exact source-pinned UI patch. Original/patched `app.js` SHA256:
  - `9472c8dd864145567a30abe2b7a576b948f2cab64318276af58e9ee44661ccab`
  - `25099dfb6b55da5864a5b406b330a3d458ad4514ff066a7343279a29af84d867`
- Reproducible/idempotent patch: `strata_v0166_ui_defaults.py`; runtime readiness accepts only the exact lifecycle/UI modifications. Signed engine, libraries, Python, CPU vision and historical offline receipt unchanged. Supported key-free startup, optional configured auth, Host/Origin, rental and all GPU/process gates retained.
- New regular state: `~/.local/state/locallm-strata/unsloth-v31-iq4xs-q8-cvec-2slot-v0166-yarn2-524k-trial`. Generic-only initial routing seed; old `.66 -v0166-trial` native state/config/adaptation preserved byte-for-byte. No existing histories copied/reset. Both old/new paths stay detectable to prevent duplicate/manual adoption.
- New exact root permission binds the approved context/config, source/runtime and UI hashes; the old fault-boot exclusion remains. Official/tuned .59 grants refreshed **only for the reviewed additional-state detection SHA**; their configuration, scope, models, runtimes and recipe unchanged. [21]/[22]/[23]/SC117/Swift untouched.
- Narrow `publish_v0166_defaults.py` permits only the selected, genuinely **stopped** [24] bundle, zero wrappers/LLM/GPU jobs, fresh Docker/hardware checks and both locks. Complete staged directory atomically exchanged; original bundle retained, never deleted. Selector/current/stable links/cloud/markers/service unchanged by publication; no lifecycle action in publisher.

## Validation and final state

**649 Python CPU tests passed/4 skipped; all8 shell suites;14 exact-UI Node scenarios.** Actual UID1000/GPU-hidden native command assembly, real pinned mock frontend HTTP/SSE63/Host-Origin/owned shutdown passed. Real Chromium tested fresh64K, one-time16K migration, non-output preferences/chat preservation, custom/blank/reset, foreign-model isolation, actual POST65536/stream/key-free63, and unchanged unlimited API default. The actual normal-Hive .66 UI repeated those relevant checks and produced63; normal owned stop and fresh-token/new-process restart passed. Final actual UID1000 LAN63 with OSN active passed.

Test setup issues were corrected without weakening gates: root-invoked legacy tests correctly refused until rerun as actualUID1000; the private test supervisor's umask077 made public Hive coordination metadata unreadable, producing a **pre-GPU refusal**. Only the normal Hive command's umask was corrected to022, private artifacts remained0600. Existing installed Chromium was selected explicitly rather than updating protected tools. A mock slider fixture was aligned to the existing step. These are not hardware-fault retries.

Final: **OSN active; normal Hive[24] running; no pause lease/switch journal/stop marker or pending controller/test/reboot**. User's current flight sheet/OC was never rewritten or restored from a snapshot. No assistant OSN stop, GPU power/clock/driver/reset/reboot action or renter/container stop. Hive/operator retain normal control of power; power values are not launch locks. All four original GPU/driver/kernel/VRAM/current-boot fault gates pass.

The earlier [510K synthetic retrieval/cached follow-up](STRATA_V0166_510K_20261124.md) is bounded evidence, not broad IQ/agent parity, two simultaneous full512K windows, eight full histories,64K-long output or soak qualification. The original bus-loss cause remains unknown. Dated private records/identities/old grants/controllers are completed audit only, never replay/start/stop authority.

[Sanitized receipt](results/strata-v0166-524k-ui64k-defaults-20261124.json).
