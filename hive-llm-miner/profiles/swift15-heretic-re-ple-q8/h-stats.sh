#!/usr/bin/env bash
# Inference telemetry only; tokens/sec are never mining hashrate.
khs=0
version='Heretic RE Q8_0 PLE candidate; Strata 0.1.46; bounded smoke passed; matched quality/speed A/B pending'
if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" '. + {algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" '{algo:"llm", ver:$version, hs:[0,0,0,0], hs_units:"hs"}' 2>/dev/null || printf '{"algo":"llm","ver":"%s","hs":[0,0,0,0],"hs_units":"hs"}' "$version")"
fi
