#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export QWEN38_DATA_ROOT="$scratch/data" QWEN38_STATE_ROOT="$scratch/state"
source <(head -n -1 "$root/v1qwen38.sh")
GPU_COUNT=4
GPU_INDICES=(0 1 2 3)
GPU_NAMES=('NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090')
GPU_MEMORY_MIB=(24576 24576 24576 24576)
GPU_DEVICE_LIST=CUDA0,CUDA1,CUDA2,CUDA3
GPU_TENSOR_SPLIT=1,1,1,1
download_file() { printf '%s|%s|%s\n' "$1" "$2" "${3:-}" >> "$scratch/downloads"; }

for profile in swift15u-q8 swift15u-bf16; do
    PROFILE="$profile"
    SPEC_OVERRIDE=""
    SMOKE=0
    configure_profile
    make_server_args
    [[ "$RUNTIME_KIND" == swift15 && "$SPEC_MODE" == native && -z "$DRAFT_PATH" ]]
    [[ "$SERVER_CTX" == 262144 && "$PARALLEL" == 1 && "$KV_TYPE" == q8_0 ]]
    [[ "$MMPROJ_PATH" == *swift15-uncensored/mmproj-BF16.gguf ]]
    args=" ${SERVER_ARGS[*]} "
    [[ "$args" == *' --spec-type draft-mtp '* && "$args" == *' --spec-draft-n-max 3 '* ]]
    [[ "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --mmproj '* ]]
    [[ "$args" != *' --spec-draft-model '* ]]
    : > "$scratch/downloads"
    ensure_swift15_assets
    [[ "$(wc -l < "$scratch/downloads")" -eq 2 ]]
    grep -q "$SWIFT15U_REV" "$scratch/downloads"
    grep -q '19acd7fcf4ee504328566eddf0f8e603ef853a205b4830209d2a4908213abdea' "$scratch/downloads"
    if [[ "$profile" == swift15u-q8 ]]; then
        [[ "$MODEL_PATH" == *"$SWIFT15U_Q8" ]]
        grep -q '9076f99a2b1c0d232d5c169bb117e919ca21bafbe092be8fc8d9c4d645ed09c9' "$scratch/downloads"
    else
        [[ "$MODEL_PATH" == *"$SWIFT15U_BF16" ]]
        grep -q 'f4bcbcc27fbb3f3560e51a4fbcea5fea346e76d89965fe21901bf43f70154b8f' "$scratch/downloads"
    fi
    SPEC_OVERRIDE=none
    configure_profile
    make_server_args
    [[ " ${SERVER_ARGS[*]} " != *' --spec-type '* ]]
done
choose_profile <<< 11 > "$scratch/menu"
[[ "$PROFILE" == swift15u-q8 ]]
grep -q 'Swift 1.5 Uncensored Q8_K_XL' "$scratch/menu"
echo 'Swift 1.5 uncensored Q8/BF16 profile assets, MTP and no-spec: PASS'
