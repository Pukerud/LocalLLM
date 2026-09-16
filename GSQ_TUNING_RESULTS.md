# GSQ Q2_0 tuning on .69 — 2026-09-16

Completed direct testing, no subagents. Current menu option6 automatically uses the measured settings on at least two selected RTX3090 24GB GPUs, absent explicit GPU selection/split overrides:

- Q2_0 weights unchanged, BF16 vision, xhigh reasoning enabled/preserved.
- 262144-token capacity, one slot, Q8 K/V, lazy PLE, mmap.
- First two GPUs only; layer split23:25 (slightly fewer layers on projector/input GPU).
- No MTP, clock changes, driver changes, or Hive/osn configuration changes.
- Other GSQ quantizations remain conservative; Hauhau production menu default4 unchanged.

## Measurements

Warmup excluded; three identical short prompts per arm, fixed seed1234, temperature1, maximum512 output tokens. Some responses deliberately hit the throughput cap; these are NOT answer-quality passes. Different GPU placement sometimes changes generated lengths, so results are indicative steady decoding measurements, not bit-identical task completion comparisons.

| Arm | Decode tok/s, three samples |
|---|---|
| Original4GPU32K, before |62.60 /63.16 /62.67|
| Original4GPU32K, after |63.37 /63.30 /63.43|
| CPU-resident PLE4GPU32K |60.74 /60.73 /60.90|
| Lazy2GPU32K |69.54 /69.43 /69.85|
| Lazy2GPU128K allocation |69.65 /69.45 /69.71|
| Lazy2GPU262K allocation, equal split |69.60 /69.63 /69.75|
| Lazy3GPU262K allocation |63.97 /64.02 /64.22|
| Final2GPU262K23:25, trial |69.63 /69.43 /69.63|
| Final live repeat |69.51 /69.60 /69.51|

Approximately10% better short-prompt decode and8x allocated context versus original menu settings. This is NOT a70tok/s claim at full context. A capped9.6K-prompt trial measured about59 decode tok/s. Native262K allocation is verified, but no full-context generation/quality test was performed.

The original user request had32346 GENERATED tokens,660.19s decode,48.99tok/s, filling almost the32K slot despite its short initial prompt. That historical end-to-end result is not directly comparable to the short samples above.

## Validation and limitations

Final configuration and deployed live API both passed normal-xhigh exact arithmetic/JSON, exactly-one parsed weather tool call, red/blue vision, and exact last-record retrieval from an8026-token synthetic reference (8078 prompt tokens after formatting). Final retrieval returned normally; no length-truncated output accepted as quality success. Earlier repetitive-inventory summaries hit512/2048 caps and caused automatic rollback; that ambiguous fixture was replaced with objective record retrieval, also verified on the original baseline before promotion. This does not resolve arbitrary long-thinking behavior or prove all application workloads terminate quickly.

Observed post-request allocation: GPU0 22480MiB used /1648MiB free; GPU1 21838MiB used /2290MiB free; GPU2/3 idle. These are snapshots, not measured worst-case peaks. Large images/near-full-context active workloads remain untested. Use `QWEN38_GSQ_CTX=131072` for more safety margin, or `QWEN38_GSQ_TUNED=0` to revert to original32K/all-GPU settings. Explicit GPU selection/split bypasses auto tuning.

Shell syntax, configuration/argument regression tests passed locally and on Linux. All candidate processes stopped; final API8080 is healthy with262144 context and one slot. Private per-arm timings/logs remain under `/tmp/gsq-{tuning,context,final-v3}-20260916` on the node. No raw user prompts published.
