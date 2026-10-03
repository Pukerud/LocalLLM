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
# Even if an archived comparison is downloaded later, it is not reintroduced into the curated menu.
for PROFILE in swift15u-q8 hauhau-q8-fastmtp turbo-q8-mtp swift15-q8 swift15u-bf16; do
    configure_profile
    mkdir -p "$(dirname "$MODEL_PATH")" "$(dirname "$MMPROJ_PATH")"
    touch "$MODEL_PATH" "$MMPROJ_PATH"
    if [[ -n "$DRAFT_PATH" ]]; then touch "$DRAFT_PATH"; fi
done
choose_profile <<< 1 > "$scratch/menu"
[[ "$PROFILE" == swift15u-q8 ]]
configure_profile; make_server_args
args=" ${SERVER_ARGS[*]} "
[[ "$args" == *' --ctx-size 524288 '* && "$args" == *' --parallel 2 '* ]]
[[ "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --cache-type-k q8_0 '* ]]
[[ "$args" == *' --spec-draft-n-max 3 '* && "$args" == *' --mmproj '* ]]
choose_profile <<< 2 > "$scratch/menu"
[[ "$PROFILE" == hauhau-q8-fastmtp-q4kv-xhigh ]]
configure_profile; make_server_args
args=" ${SERVER_ARGS[*]} "
[[ "$args" == *' --ctx-size 786432 '* && "$args" == *' --parallel 3 '* ]]
[[ "$args" == *' --spec-draft-n-max 4 '* && "$args" == *' --spec-draft-model '* ]]
[[ "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --cache-type-v q4_0 '* ]]
[[ "$(grep -c '^  \[[12]\]' "$scratch/menu")" == 2 ]]
! grep -qE 'TURBO|UkisAI|Uncensored BF16|\[3\] Swift|GSQ-RCO' "$scratch/menu"
grep -q 'HostLLM \[2\]' "$scratch/menu"
source "$root/HostLLM.sh"
[[ "$(declare -f stop_owned_engines)" != *pkill* ]]
[[ "$(declare -f run_selected)" == *'safety preflight'* ]]
! grep -qE '^[^#]*(systemctl[[:space:]]+(stop|start)|/hive/bin/miner|^[[:space:]]*(pkill|killall)[[:space:]])' "$root/HostLLM.sh"
echo 'Retained menu, Q8 profile settings and service-free launch gates: PASS'
