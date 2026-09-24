#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export QWEN38_DATA_ROOT="$scratch/data" QWEN38_STATE_ROOT="$scratch/state"
source <(head -n -1 "$root/v1qwen38.sh")
[[ "$PROFILE" == hauhau-q8-fastmtp-q4kv-xhigh ]]
GPU_COUNT=4
GPU_INDICES=(0 1 2 3)
GPU_NAMES=('NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090')
GPU_MEMORY_MIB=(24576 24576 24576 24576)
GPU_DEVICE_LIST=CUDA0,CUDA1,CUDA2,CUDA3
GPU_TENSOR_SPLIT=1,1,1,1
# Mock downloads: assert exact pins, file paths and digests without network.
download_file() { printf '%s|%s|%s\n' "$1" "$2" "${3:-}" >> "$scratch/downloads"; }
for p in swift15-q8 swift15-q4; do
 PROFILE="$p";SMOKE=0;SPEC_OVERRIDE="";configure_profile;make_server_args
 [[ "$FULL_CTX" == 262144 && "$SERVER_CTX" == 262144 && "$PARALLEL" == 1 ]]
 [[ "$RUNTIME_KIND" == swift15 && "$SPEC_MODE" == none && -z "$DRAFT_PATH" ]]
 [[ "$MODEL_PATH" == *Swift-1.5-Qwen3.8-27B-*.gguf && "$MMPROJ_PATH" == *mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf ]]
 args=" ${SERVER_ARGS[*]} "
 [[ "$args" == *' --ctx-size 262144 '* && "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --reasoning on '* ]]
 [[ "$args" == *' --mmproj '* && "$args" == *' --load-mode mmap '* && "$args" == *' --cache-type-k q8_0 '* ]]
 [[ "$args" != *' --spec-type '* && "$args" != *' --lazy-mode '* ]]
 : > "$scratch/downloads";ensure_swift15_assets
 [[ "$(wc -l < "$scratch/downloads")" -eq 3 ]]
 grep -q "$SWIFT15_REV" "$scratch/downloads"
 grep -q 'daa1116c9422fa390cc8688495da0e91781f92841dfc3b31a378ff252571745a' "$scratch/downloads"
 if [[ "$p" == swift15-q8 ]]; then grep -q '0b3b4aa0e367c620756de4f9db77fc18f94fbeb256db208281cbb84c48e2a101' "$scratch/downloads"; else grep -q '2ebba0ff1e63c1ac3fadd4e83efcea189f47f33ec72c91877af94de6ebe30590' "$scratch/downloads"; fi
 if (SPEC_OVERRIDE=fast;configure_profile); then echo 'Wrong-model sidecar accepted' >&2;exit 1;fi
 SPEC_OVERRIDE=none;configure_profile
 [[ "$SPEC_MODE" == none ]]
done
SPEC_OVERRIDE=""
choose_profile <<< 9 > "$scratch/menu"
[[ "$PROFILE" == swift15-q8 ]]
choose_profile <<< 10 > "$scratch/menu"
[[ "$PROFILE" == swift15-q4 ]]
choose_profile <<< '' > "$scratch/menu"
[[ "$PROFILE" == hauhau-q8-fastmtp-q4kv-xhigh ]]
configure_profile
[[ "$SPEC_MODE" == fast && "$KV_TYPE" == q4_0 ]]
echo 'Swift 1.5 menu, full-context vision arguments, pinned downloads, MTP rejection and production default: PASS'
