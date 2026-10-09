#!/usr/bin/env bash

# Inference host: Hive has no hash-rate result to report.
khs=0
version='Orca Uncensored Q4_K_S Strata 0.1.39 native 262K CPU vision'

if [[ -n "${gpu_stats:-}" ]] && command -v jq >/dev/null 2>&1 \
    && jq -e . >/dev/null 2>&1 <<< "$gpu_stats"; then
    stats="$(jq -c --arg version "$version" \
        '. + {algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        <<< "$gpu_stats")"
else
    stats="$(jq -cn --arg version "$version" \
        '{algo:"llm", ver:$version, hs:[0,0,0], hs_units:"hs"}' \
        2>/dev/null || printf '{"algo":"llm","ver":"%s","hs":[0,0,0],"hs_units":"hs"}' "$version")"
fi
