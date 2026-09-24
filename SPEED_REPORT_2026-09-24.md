# Speed/update report — 2026-09-24

Direct read-only primary-source research, no subagents. Status observed at review time; unmerged PRs can change. No production runtime upgrade, restart, new-model inference, or new benchmark was performed for this report.

## Actual node state

Node .69 is currently back on Hauhau Q8 FastMTP n4 / Q4_0 KV / xhigh / vision /3x262144 slots, runtime b10454-4df29be4f. osn active, Docker inventory empty, API healthy. Do not assume GSQ is still running merely because it was last tuned in this conversation.

Previously measured GSQ Q2_0: about63tok/s on4GPUs vs69.5tok/s on2GPUs at short prompts, with262K allocation and balanced23:25 split. That remains historical evidence, not a new result today and not full-context throughput. CPU-residentPLE tested slower; don't repeat it as an established optimization. See GSQ_TUNING_RESULTS.md.

## News and applicability

| Candidate/source | Verified status | What it could improve | Limits/recommendation |
|---|---|---|---|
| [Swift1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF) | Model repository updated today, revision a1614465cfa35d04d3e8575d713fa779662b5eab | Total task time via fewer reasoning tokens | Most practical next comparison. Publisher GPQA xhigh mean-token reduction41.9%, median58.5%; LCB mean24.5%, Terminal-Bench mean16.3%. Not a decode-rate multiplier. |
| [llama.cpp v0.5.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0) | Released September23 | Newer backend/model/server fixes; CUDA conv2d implicitGEMM highlight may help applicable vision work | Stage separately; not a verified drop-in performance upgrade. Preserve/validate the Hauhau FastMTP patch before changing its runtime. |
| [#29030 direct lazy-row reads](https://github.com/ggml-org/llama.cpp/pull/29030) | Open/unmerged; supersedes28136 | Flash-Next qwen4exp PLE prompt processing | Highest-priority GSQ runtime experiment. Author reports+65–121% PP on StrixHalo. NOT3090 decode evidence. Requires isolated build; existing --lazy-mode on does not acquire proposed on-direct behavior automatically. |
| [#28243 Flash-Next MTP](https://github.com/ggml-org/llama.cpp/pull/28243) | Open/unmerged; builds on27836 | Potential speculative decode improvement | Author claims1.3–2x. Requires compatible draft/modules, extra memory and validation; no ready matching GSQ draft established. Current twoGPU262K headroom is tight. Do not reuse Hauhau's sidecar. |
| [#29353 chunkedGDN CUDA/HIP](https://github.com/ggml-org/llama.cpp/pull/29353) | Open/unmerged, updated today | qwen35 prompt processing | Author reports roughly11–12% Qwen27B PP gains on Blackwell/GB10; no3090 result established. Not proof of faster decode. |
| [#29393 RMS_NORM+SCALE fusion](https://github.com/ggml-org/llama.cpp/pull/29393) | Open/unmerged, updated today | Reduce kernel launches on qwen35/qwen4exp | Reported4–5% recovery in one prefill scenario on1080Ti/Windows without CUDA graphs. It repairs a newer-build regression; not a guaranteed gain over our older b10454/b10803 builds. |
| [#28785 skip empty CPU threadpool](https://github.com/ggml-org/llama.cpp/pull/28785) | Open/unmerged | Lower host overhead for fully offloaded graphs | Published decode gain is on0.8B/AppleM3, not27B/3090. Low-confidence for this node; lazyPLE also means CPU work can remain. |

Earlier tracked [#27140](https://github.com/ggml-org/llama.cpp/pull/27140) small-KV prefill fix and [#28473](https://github.com/ggml-org/llama.cpp/pull/28473) per-sequenceMTP draft-limit fix remain open. No new local result justifies promoting either. Historical aborted Hauhau experiments are not successful benchmark evidence.

## Ranked plan

1. Compare Swift1.5 Q8 first against current Hauhau on a small coding/tool/reasoning set at normalxhigh. Measure complete-task latency, final correctness, thinking/output tokens, PP and decode separately. Then compare Q4_K_M for weight-bandwidth savings; account for fidelity loss.
2. For GSQ, separately benchmark29030 on same base/settings with bounded1K/8K/16K prompts. No patch stacking; keep the old runtime rollback.
3. For denseQwen/Hauhau, test29353 against an identical parent build; consider29393 if that parent includes the regression. Check MTP, JSON, tool, vision and multi-slot behavior before promotion.
4. Only afterward explore Flash-NextMTP. Verify matching draft and actual accepted tokens/s vs verification cost; memory may require a different split or more GPUs.
5. Batch/microbatch and prefix-cache experiments can improve time-to-first-token/agent reuse without changing weights. They should not be sold as guaranteed single-user decode improvements. Keep current thinking enabled, native context allocation and rental/lifecycle safety.

## Swift1.5 caveats

The9.18x headline is a specific game-building demo (104.6 vs11.39minutes), NOT9.18x tok/s. Main benchmarks use BF16/vLLM rather than these GGUFs. Publisher results also show tradeoffs: AIME96.00 vs98.67, IFBench72.07 vs73.53, while LiveCodeBench81.71 vs76.76 and Terminal-Bench72.13 vs69.21. Fewer tokens does not prove universally equal quality.

Model metadata is qwen35/native262144; it is NOT qwen4exp and does not use the GSQ Flash-Next PLE optimization. Its supplied vision projector is F16. Swift Open License v1.0 contains commercial revenue/affiliate restrictions; check it before commercial hosting above the stated US$1million threshold.

Menu9/10 and CLI profiles swift15-q8/swift15-q4 use pinned checksummed assets, full262K/one slot, F16vision, Q8KV and xhigh. Follow-up correction: native MTP is now ON at draft depth3, with --no-spec fallback. The official source card documents MTP, and both pinned GGUF headers actually contain nextn_predict_layers=1 and the complete embedded head. Initial no-MTP configuration was unnecessarily conservative. No separate Hauhau sidecar is used. Tests cover native flags, draft depth, fallback, menu/default preservation, download manifest and incompatible external-draft rejection. Download and actual Swift inference remain user-invoked; no measured new-model speedup or full-context validation is claimed.
