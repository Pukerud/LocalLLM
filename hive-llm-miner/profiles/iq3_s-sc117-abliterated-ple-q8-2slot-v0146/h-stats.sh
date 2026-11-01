#!/usr/bin/env bash
# Inference telemetry only; no mining hashrate or unmeasured quality claim.
khs=0
version='SC117 Q8 two-user .46 (UNQUALIFIED); 2 pipelined slots; 16 GiB / 8 histories; solo MTP; long-prefill pauses possible'
if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" '. + {algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" '{algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' 2>/dev/null || printf '{"algo":"llm","ver":"%s","hs":[0,0,0,0],"hs_units":"hs"}' "$version")"
fi
