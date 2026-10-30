# Strata 0.1.46: isolated trusted-LAN test

**Current scope (2026-10-30):** Orca Q4/DeepSeek weights and menu routes were retired on 2026-10-29; they are not installed-model rollbacks. The pinned .46 runtime below remains unchanged for SC117/Heretic. Original IQ4_NL PLE routes are Heretic [3]/Hive [14] and SC117 .45 [4]/.46 Hive [15]; opt-in Q8_0 PLE adds [5]/[6]/Hive [16]/[17] with separate state/cache, not a runtime upgrade. See [PLE_Q8.md](PLE_Q8.md) and [the later authorized Q8 live-menu smoke](results/ple-q8-live-menus-20261030.json). All four new routes passed bounded functional checks; original SC117 was restored. SC117 remains an **unqualified trial**. Earlier Orca-standard/menu/rollback statements below describe dated historical tests, not current installation policy. Keep the retained runtimes/preset evidence without silently redownloading retired weights.

## Pins and build

- Strata source: `90767252f420e25223c739d604145ad15911d48b` (engine version 0.1.46).
- llama.cpp: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Separate runtime: `/home/user/.local/share/localllm-strata/runtimes/0.1.46`.
- Existing CUDA 12.9, `sm_86`, private Python dependencies; no driver/system-package replacement.
- Inherits the .45 native-expert/Q4 build options and enables `STRATA_MMQ_TQ_OVERLAY=ON`.
- Separate GPU and native-CPU vision helpers; these must pass recorded artifact/build checks.

Preparation is build-only, as the unprivileged runtime owner, with the miner safely paused through the HostLLM lifecycle controller:

```bash
./v1strata.sh --prepare-runtime --runtime 0.1.46
```

Do not bypass Docker/GPU/session ownership checks. Preparation must not replace executables of an active candidate. Both .45 and .46 remain profile-bound and cannot be globally selected for other models.

## Optional API key: exact local patch

Upstream .46 refuses a non-loopback bind without an inference API key. At the operator's request, preparation removes **only these three lines** from this isolated .46 checkout's `serve/server.py`:

```python
    if a.host not in ("127.0.0.1", "localhost", "::1") and not preflight_api_key:
        print("[strata] a non-loopback bind requires an inference API key", file=sys.stderr, flush=True)
        return 2
```

The original full-file SHA-256 is `f15efbe6a1e3a97d58ddbc0fa9a68ce544c90f6c1b03cbf6cecf50c625a5a5f8`; the patched full-file SHA-256 is `36f7a63d02e2c9fc4106904ec597d9a161b21f6b588f70d90a10fd8d92d6269f`.

`prepared.json` records this local patch separately from the upstream source pin. Preparation accepts only the original or exact patched bytes; readiness checks require the patched digest and matching manifest record. Other tracked source edits are refused. The .45/rollback checkouts are not patched.

Configured authentication, API-key validation, DNS-rebinding Host checks, and browser Origin checks remain unchanged. An empty optional HostLLM key means **no authentication**. Use only on a trusted, firewalled LAN; never forward the port publicly.

## Routes and isolation

- **HostLLM [5]:** single-slot Orca Q4 .46 test. Leave the optional API-key prompt empty for no authentication. HostLLM defaults to `0.0.0.0:8080`; `STRATA_HOST=127.0.0.1` retains local-only Q4 access.
- **HostLLM [17] / Hive profile-menu [5]:** `iq3_s-sc117-abliterated-v0146`, an unqualified SC117 .46 trial. These menu numbers are different from HostLLM [5].
- SC117 is fixed to unauthenticated trusted-LAN `0.0.0.0:8080`, CPU vision, native context 262144, INT8 KV with 32768 resident, `--spec 3 --pipeline-windows 2`, one serving slot/group, and an 8192 MiB/four-conversation-cache-slot budget.
- SC117 state remains `/home/user/.local/state/locallm-strata/sc117-iq3s-trial`; four cache slots do **not** mean four concurrent serving slots or prove capacity for four full-context snapshots.
- Q4 and SC117 share port 8080 exclusively. Models and vision load before the launcher reports API-ready; idle unloading is disabled.
- Exact candidate readiness precedes a HostLLM service pause or Hive pointer change. Inference runs as `user`, never root. Rentals, unknown GPU jobs, and pre-existing manual sessions are never stopped.

A short successful request is only a smoke test. No .46 speed gain, broad quality result, full-context capacity, sustained load, or SC117 qualification follows from compilation/readiness or another GPU's release benchmarks.

## Standing preference for future upgrades

The operator explicitly requires **API-key-free trusted-LAN hosting on every future upgrade**, not just .46. This is recorded in [AGENTS.md](AGENTS.md). Inspect new upstream auth/startup changes, retain previous version-specific patch/ownership contracts, and prove a real LAN request without Authorization/x-api-key headers before claiming an upgrade works. Do not substitute a dummy token or silently add mandatory auth. Keep public exposure disabled and preserve Host/Origin and workload safety checks.

**Menu numbering follow-up (2026-10-25):** DeepSeek now uses HostLLM [6]/Hive [16]. The unchanged SC117 .46 profile moves to HostLLM [17], retaining low-level Hive [5]. The [16] tests below are historical evidence from before that renumbering.

## Initial direct-HostLLM live results: 2026-10-20

The pinned runtime/build and exact patch checks passed. Real HTTP requests used `http://192.168.1.69:8080`, with no API-key environment input and no Authorization/x-api-key headers. Both .46 frontends/native engines ran as UID 1000, eagerly loaded the expected runtime, and reported `api_key=false` and one serving slot.

| Route | Web UI / models | Text reply (`7 * 9`) | CPU synthetic-image reply | Owned shutdown / lease recovery |
| --- | --- | --- | --- | --- |
| Root HostLLM [5], Q4 .46 | HTTP 200, no key | `63` | `Magenta` | Passed |
| Root HostLLM explicit SC117 .46 trial | HTTP 200, no key | `63` | `Magenta` | Passed |
| Root HostLLM explicit SC117 .45 rollback trial | HTTP 200, no key | `63` | Not rerun | Passed |

Same-origin browser-style POSTs without a key returned HTTP 200. Untrusted Host and Origin requests still returned HTTP 403. These checks used this host's LAN address, not an independent remote-device/firewall validation. That initial run did not exercise HostLLM [16] or the .46 Hive wrapper/profile-pointer switch; the installed Hive selection was preserved. The follow-up below covers the actual menu/Hive path.

### .46 process-group compatibility discovered during the first start

The initial start failed the old same-PGID ownership gate: upstream `serve/lifeline.py` now gives the native engine and vision helper their own PGIDs while retaining the frontend's session. That failed start is excluded and preserved in the raw artifacts. The launcher now accepts this only for .46, only for exact allowed non-Python executables, matching UID/session, and an own-PGID direct child of the verified live frontend. An unchanged recorded identity is required after frontend exit. Unknown executables/parents/users/groups and reused identities remain refused; .45/older grouping rules are unchanged.

The first test's exact owned family was safely stopped using the corrected identity gate, its lease was released, and fresh Q4/SC117 .46 starts subsequently passed requests and clean shutdown. CPU regression tests cover both the new child-group layout and refusal cases, including real fake-frontend startup/request/external-stop/reap.

### Initial test final state and scope

The prior Wildrig miner and OctaSpace service were restored, `MINER_STOP` was absent, there was no HostLLM pause lease or candidate listener, and 45 retained runtime/profile/Hive-config hashes were unchanged. Orca .45 remains standard; SC117 remains unqualified. The .45 rollback request also passed after the launcher changes.

Validation: 350 Python tests (one skipped), all seven shell tests, Python compilation, shell syntax, and `git diff --check` passed.

[Machine-readable results](results/strata-v0146-noauth-20261020.json). Raw build, readiness, response, startup-failure, and lifecycle artifacts: `/tmp/localllm-v0146-noauth-20261020/` on the tested host. No throughput/speed comparison, full-context generation, sustained/multi-user load, or general quality qualification was performed.

## Follow-up: actual HostLLM [16] / Hive .46 switch

The operator found a missed selector bug: Python preflight hard-coded `/usr/bin/runuser`, but HiveOS provides `/usr/sbin/runuser`. The helper now resolves it only through trusted system directories (`/usr/sbin:/usr/bin:/sbin:/bin`), never through an inherited user-writable PATH. If unavailable, it still refuses before pausing services or changing the pointer. Exact trial readiness and key-free LAN settings remain mandatory.

The corrected path was tested in a real root controlling terminal: select **16**, wait for the serialized package/profile switch, press Enter, and exit. The existing .45 server was proved Hive-owned before any stop. Preflight passed, OctaSpace was paused, only that owned Hive session was stopped, the .46 package/pointer was selected, and Hive/OctaSpace running state was restored. The new .46 Hive frontend/native engine ran as UID 1000, eagerly loaded, reported one serving slot and `api_key=false`, and served the web UI/models plus text (`63`) and CPU-image (`Magenta`) requests on `192.168.1.69:8080` without authentication headers. A same-origin browser-style image POST also passed.

The explicitly requested SC117 .46 trial was left **selected and running**, with OctaSpace active, `MINER_STOP` absent, and no pause lease or switch recovery journal. This differs from the initial smoke run, which restored Wildrig. The guarded package refresh also set Hive's custom-miner config to `llm-hosting` and its stable config path; `rig.conf` and the generic runtime-selection hash were unchanged. SC117 remains unqualified, and the separate standard Orca route remains .45.

Regression checks: **357 Python tests (one skipped), seven shell tests**, compilation, shell syntax, and clean diff checks. Coverage now executes the preflight command rather than only mocking the whole preflight function, including trusted/sbin lookup, missing-command refusal before mutations, and exact-line rejection.

[Actual menu/Hive results](results/hive-option16-runuser-20261020.json). Raw PTY transcript, request replies, and snapshots: `/tmp/hive-option16-runuser-20261020/`. These remain bounded smoke checks, not full-context, sustained-load, or quality qualification.
