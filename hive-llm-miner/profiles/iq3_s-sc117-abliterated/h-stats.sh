#!/usr/bin/env bash

# Isolated unqualified inference trial: report telemetry without a hash-rate result.
khs=0
version='SC117 IQ3_S Strata 0.1.44.1 CPU-vision trusted-LAN trial (UNQUALIFIED)'

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
