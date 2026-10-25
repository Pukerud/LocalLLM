#!/usr/bin/env bash

# Inference telemetry only; never report token throughput as a mining hash rate.
khs=0
version='DeepSeek 4.1 Flash 1017 Abiray IQ2_XXS; engine 0.1.51; native 1M; BF16 GPU vision'

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
