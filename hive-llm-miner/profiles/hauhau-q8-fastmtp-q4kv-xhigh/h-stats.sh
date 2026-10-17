#!/usr/bin/env bash

# Inference host: report zero hashrate while preserving Hive GPU telemetry.
khs=0
version='Hauhau Q8_K_P BF16 vision FastMTP n4 Q4 KV xhigh three native-262K slots'

if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 \
    && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" \
        '. + {algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" \
        '{algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        2>/dev/null || printf '{\"algo\":\"llm\",\"ver\":\"%s\",\"hs\":[0,0,0],\"hs_units\":\"hs\"}' "$version")"
fi
