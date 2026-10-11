# Orca Q4_K_S / Strata 0.1.41 — isolated candidate

**Status: opt-in candidate path only; Q4 .39 remains the default and rollback.** The recorded Q4 .39 prefill baseline is `956.4` input tokens/s median over three uncached 5,295-token prompts, capped at 16 output tokens. No .41 speed or generalized-quality claim is made; any .41 result is a bounded operator self-attestation, not qualification.

The candidate pins Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` and is independent of `strata_runtime.py`'s default-selection manager. It reuses the already prepared .39 Q4 shards/compatibility pack as read-only shared inputs, and requires the exact prepared generic .41 llama.cpp commit and vision-helper hash to pass manifest/source/hash checks. The candidate records its own full llama.cpp source-tree hash without modifying the generic .41 manifest. It has a distinct Q4 source checkout and CMake cache with CUDA, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_ORCA_Q4KS_MMQ=ON`, and `STRATA_BUILD_TESTS=ON`.

Separate candidate files live below:

```text
runtimes/orca-q4_k_s-v0141/source/                 # exact .41 checkout; distinct Q4 build tree/cache
runtimes/orca-q4_k_s-v0141/source/build-orca-q4ks-v0141/
profiles/orca-q4_k_s-v0141/config.json
profiles/orca-q4_k_s-v0141/prepared.json
profiles/orca-q4_k_s-v0141/parity.json
profiles/orca-q4_k_s-v0141/evidence.json
profiles/orca-q4_k_s-v0141/validation-32k.json     # only after separate live validation
profiles/orca-q4_k_s-v0141/validation-native.json # only after native-context/CPU-vision probes
```

The staged candidate config is **32K, one serving slot, text-only**, with an 8192 MiB conversation-cache budget and four cache slots. The same cache flags are retained in gated native-context and CPU-vision configs. Four is the configured parked-snapshot cap, not a promise that four large or 262K conversations fit in 8 GiB; only the short 32K A/B/A/B cache probe is required by this candidate gate. Full-context snapshot sizes, four retained 262K histories, and multi-user concurrency remain unmeasured. Preparation does not run inference, parity, or change the selected runtime. It fails closed if the .39 Q4 assets/profile or pinned .41 runtime are missing/modified, if the .41 llama/helper source/hash contracts disagree, or if the separate candidate source/build tree already exists in an ambiguous state.

## Explicit preparation, parity, and use

Run as the normal operator; do not use `sudo`. Wait for any rental, Docker workload, and GPU compute to clear. The explicit HostLLM preparation command checks idle state, acquires a fresh pause lease, builds/stages only the isolated Q4 .41 candidate, then releases its own lease; it does not start a model or change runtime selection. The generic .41 runtime must already be prepared and manifest-verified. HostLLM refuses preparation while an LLM is active; stop Qwen through its owning launcher first. The candidate does not auto-refresh the generic .41 runtime. Parity and normal inference go through HostLLM's readiness/lease/identity lifecycle:

```bash
./HostLLM.sh --prepare-q4ks141
./HostLLM.sh --verify-q4ks141-parity
./HostLLM.sh --run-q4ks141-32k-validation
# In a second terminal, run the pinned validator; then Ctrl+C in the first terminal.
./v1strata.sh --validate-orca-q4ks-141-32k
# After a successful report and stopped server, apply the separately gated migrations:
./v1strata.sh --configure-q4ks141-native-context --profile orca-q4_k_s
./v1strata.sh --enable-q4ks141-vision --profile orca-q4_k_s
./v1strata.sh --check-ready --profile orca-q4_k_s --runtime 0.1.41

# Start the final profile with HostLLM [5], then run its bounded native/vision probe in another terminal:
./HostLLM.sh                         # choose [5]
./v1strata.sh --validate-orca-q4ks-141-native
```

The native/vision runner requires an already-running identity-verified HostLLM .41 lease; it never starts/stops the server. It checks one synthetic image and one near-full-context tail marker, with fresh `/proc`/`nvidia-smi` telemetry, a 12 GiB available-RAM floor, a 24,300 MiB per-GPU ceiling, and a 30-minute request limit. If a limit trips or the client disconnects, it waits at most 120 seconds for server-side idle confirmation and otherwise exits blocked without claiming the request stopped. This is a single bounded observation, not a general quality, snapshot-capacity, or load test.

HostLLM runs `--check-ready` before acquiring a pause lease. The candidate parity command requires a new HostLLM-owned pause lease, refuses an already-active or unmanaged server, verifies exact pinned shard hashes, and uses identity-bound child cleanup (pidfd only; no numeric-PID fallback). Direct candidate parity as root is refused. Do not launch its parity binaries directly.

The candidate does not change `runtime-selection.json`, the default Q4 .39 profile/config/runtime, the recorded baseline result, or the separate IQ3_XXS asset-cleanup state. HostLLM [4] stays pinned to Q4 .39; [5] explicitly routes Q4 to the .41 one-slot candidate, and [6] routes only to the separately gated .41 two-slot/two-group experiment. Hive options [14] and [15] select the .41 one-slot and two-slot profiles respectively. The normal IQ3 runtime selection behavior is unchanged; neither .41 profile changes the Q4 .39 default.

## Candidate-specific 32K gate

Parity is kernel validation, not inference validation. Before migration, start the initial 32K text-only candidate through this guarded HostLLM path in terminal 1; it binds loopback only and uses the normal idle/rental and HostLLM lease checks:

```bash
./HostLLM.sh --run-q4ks141-32k-validation
```

While it is serving, run the pinned validation from terminal 2, then press Ctrl+C in terminal 1 after it completes:

```bash
./v1strata.sh --validate-orca-q4ks-141-32k
```

The runner checks candidate engine/shard/config hashes, health, arithmetic, strict JSON, tool calls, memory, original MTP drafted/accepted tokens, and the live server's process and HostLLM lease identities. Its A/B/A/B conversation probe resends full histories and requires `/metrics` counter deltas from that probe itself: a park, a restore, reused requests/tokens, and the enabled 8192 MiB/four-cache-slot configuration. It accepts no caller-supplied report and writes its record only after all checks pass. Normal .41 starts permit only loopback or explicit `STRATA_HOST=0.0.0.0`; this LAN mode is unauthenticated unless `STRATA_API_KEY` is provided, so firewall it and use only on a trusted network. The Hive profile does not pass an API key. A per-run lock excludes another validator, and an exclusive HostLLM lifecycle reservation covers startup through durable process-identity recording. Hive launch checks `MINER_RUN`, `cur_miner=custom`, absence of a HostLLM lease or stop marker, and a per-run 48-hex token; after launch, `h-run.sh` binds supervision/stop to the recorded process identities and token. The start token is an operational coordination gate, not cryptographic proof of root Hive-supervisor origin: a local process running as the trusted operator can synthesize one. Treat the local operator account as trusted; this is not a boundary against concurrent or malicious local processes. The validation sends seven bounded functional requests.

**Evidence trust boundary:** parity and live-validation JSON are operator self-attestations, not cryptographic attestations or tamper-proof records. Hashes bind the report to the pinned inputs and runner; they do not prove that the operator did not edit the report. The normal runner only records after its tests succeed, but an operator with filesystem access can fabricate or alter these files. The explicit migration command is therefore an operator approval based on reviewed self-attested results—not an anti-forgery security gate. A fresh successful live run is required before migration; do not describe .41 as generally qualified. CPU vision may be enabled only after that gate and migration; it reuses only the .41 manifest/hash-verified helper and a separately recorded, read-only same-revision projector hardlink:

```bash
./v1strata.sh --configure-q4ks141-native-context --profile orca-q4_k_s
./v1strata.sh --enable-q4ks141-vision --profile orca-q4_k_s
./v1strata.sh --rollback-q4ks141-vision --profile orca-q4_k_s
```

These commands affect future Q4 .41 candidate launches only. They do not change the .39 config, active private run-config, generic runtime selection, or rollback profile. Reports are bounded operator observations—not independent attestations, general quality proof, sustained-load qualification, or evidence of .41 speed gain.

## Bounded live observations on .69 — 2026-10-11

The isolated loopback 32K text-only gate passed health, arithmetic, strict JSON, tool call, conversation-cache configuration/reuse, and memory checks; MTP drafted 35 tokens and accepted 31. Only after that report was recorded did the candidate migrate to native context and enable CPU vision.

The bounded native/CPU-vision run also passed: text smoke answered `437`, one generated synthetic image was identified as `Magenta`, and the short token-count control reconciled 7 uncached + 20 cache-read tokens with the API's 27-token total. The near-full prompt's count and response usage matched exactly at 261,000 / 262,144 tokens (99.56%); it retrieved `Q4-141-NATIVE-CTX-39B7` in 264.6 seconds. Minimum available RAM was 43,615 MiB. Per-GPU peaks were 23,740 / 19,502 / 22,410 / 23,316 MiB (aggregate sum 88,968 MiB); 260 sampler samples had no errors. This is one controlled tail-marker retrieval and one synthetic CPU-vision probe, not general context/image quality, worst-case memory, snapshot capacity, throughput, or multi-user/sustained-load evidence.

The identity-bound reports remain private on `.69` at `profiles/orca-q4_k_s-v0141/validation-32k.json` and `profiles/orca-q4_k_s-v0141/validation-native.json` (mode `0600`); they are not committed. An initial native attempt was retained separately as `validation-native-first-run-failed.json` after its short control exposed an uncached-vs-total token accounting bug; only the subsequent passing report is counted. The `.39` default, Hive miner, and `osn.service` were restored after the earlier migration checks; no inference request was sent to the restored `.39` service.

## Experimental .41 two-slot observation — 2026-10-11

A temporary test-only override was used to start the native/CPU-vision .41 config with `parallel=2`, `batch-groups=2`; the runtime reported two serving slots. One 4,537-token prompt returned the expected answer at 904.2 prompt tok/s and 64.7 generated tok/s. Two simultaneous 4,537-token prompts both returned correct answers; their reported prefill rates were 964.4 and 969.6 tok/s, with 9,071 combined prompt tokens over 9.49 seconds (about 955.7 aggregate tok/s). This short sample establishes that this exact 2/2 config started and served two requests, but shows no clear aggregate throughput gain over the solo sample and is not a sustained-load or long-context test.

During startup Strata loaded the 68.26 GiB expert pool into RAM and populated an 18.3 GiB GPU expert cache; its log reported only 457 MiB VRAM free with everything loaded. The 2-slot preset is therefore explicitly experimental and has very little headroom. Options [6]/[15] use only the tested 2-slot/2-group combination. The launcher rejects counts above two; three/four slots remain unsupported and untested. The temporary test override was reverted before normal .41 one-slot service was restored.
