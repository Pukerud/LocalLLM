#!/usr/bin/env bash

# This is an inference host rather than a hash-rate miner. Report zero
# hashrate plus the normal GPU telemetry so HiveOS shows the custom miner as
# running without inventing a mining result.
khs=0
version='Qwen3.8 HauhauCS FastMTP'

if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 \
    && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" \
        '. + {algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" \
        '{algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        2>/dev/null || printf '%s\n' '{"algo":"llm","ver":"Qwen3.8 HauhauCS FastMTP","hs":[0,0,0],"hs_units":"hs"}')"
fi
