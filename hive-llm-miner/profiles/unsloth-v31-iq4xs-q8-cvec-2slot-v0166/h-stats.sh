#!/usr/bin/env bash
# Inference telemetry only, never mining hashrate or an intelligence guarantee.
khs=0
version='Unsloth V3.1 IQ4_XS Q8 tuned .66 (bounded trial); native262K; CPU vision; 2 slots/groups; 16 GiB/8 histories; stock solo MTP'
if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" '. + {algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" '{algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' 2>/dev/null || printf '{"algo":"llm","ver":"%s","hs":[0,0,0,0],"hs_units":"hs"}' "$version")"
fi
