# LocalLLM maintenance rules

## Standing operator preference: no API key on the trusted LAN

Normal trusted-LAN hosting must work **without an API key**. This applies to HostLLM, Hive profiles, and every future Strata/runtime upgrade. Do not introduce a required key, dummy key, Authorization header, or secret environment variable to make LAN startup work. Optional authentication support may remain for explicitly requested use, but it must not become a requirement/default for these LAN routes.

For every runtime upgrade:

1. Inspect upstream startup/auth changes before installing or switching profiles. If upstream newly requires a key for a non-loopback bind, make a narrow, reproducible, version-pinned local patch to remove only that mandatory startup requirement. Record original/patched hashes and validate the exact patch during preparation/readiness; do not broadly ignore source drift. Preserve earlier versions' exact patch and process-ownership contracts when advancing a candidate version.
2. Preserve Host/DNS-rebinding and browser Origin checks, configured-key enforcement when deliberately enabled, and all rental/GPU/process ownership gates. No public exposure or wildcard browser-origin workaround.
3. Build separately as the unprivileged runtime owner. Keep the current/default runtime and explicit rollbacks until the candidate passes its checks. Do not stop rentals, unidentified GPU work, or someone else's manual hosting session.
4. Test real LAN HTTP requests to the intended port with **no Authorization or x-api-key header**: startup, loaded runtime/version, web UI, models/status, a known-working text request, and vision if the profile supports it. Verify health reports no key. Readiness/compilation alone is not proof.
5. Verify owned shutdown and restore the previous miner/OctaSpace state using the serialized HostLLM lifecycle workflow. Record actual scope/results; do not infer speed, quality, or qualification from a short smoke test.

## Current profile boundaries

- Orca Q4 Strata .45 remains the standard route; .46 is an isolated explicit test candidate. Q4 .39/.41/.44.1 remain explicit rollback routes.
- SC117 .45/.46 are **unqualified trials only**, unauthenticated on trusted-LAN `0.0.0.0:8080`. Never expose them publicly or turn trial smoke results into qualification.
- Preserve SC117 `--spec 3 --pipeline-windows 2`, CPU vision, one serving slot/group, 8192 MiB/four conversation-cache slots, native context 262144, INT8 KV/32768 resident, and dedicated `sc117-iq3s-trial` state. Four conversation-cache slots are not four serving slots.

See `STRATA_V0146.md` for the exact .46 no-key startup patch and version-specific lifecycle compatibility notes.
