#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export QWEN38_DATA_ROOT="$scratch/data" QWEN38_STATE_ROOT="$scratch/state"
source <(head -n -1 "$root/v1qwen38.sh")
[[ "$PROFILE" == swift15u-q8 ]]
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
 [[ "$RUNTIME_KIND" == swift15 && "$SPEC_MODE" == native && -z "$DRAFT_PATH" ]]
 [[ "$MODEL_PATH" == *Swift-1.5-Qwen3.8-27B-*.gguf && "$MMPROJ_PATH" == *mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf ]]
 args=" ${SERVER_ARGS[*]} "
 [[ "$args" == *' --ctx-size 262144 '* && "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --reasoning on '* ]]
 [[ "$args" == *' --mmproj '* && "$args" == *' --load-mode mmap '* && "$args" == *' --cache-type-k q8_0 '* ]]
 [[ "$args" == *' --spec-type draft-mtp '* && "$args" == *' --spec-draft-n-max 3 '* ]]
 [[ "$args" != *' --spec-draft-model '* && "$args" != *' --lazy-mode '* ]]
 : > "$scratch/downloads";ensure_swift15_assets
 [[ "$(wc -l < "$scratch/downloads")" -eq 3 ]]
 grep -q "$SWIFT15_REV" "$scratch/downloads"
 grep -q 'daa1116c9422fa390cc8688495da0e91781f92841dfc3b31a378ff252571745a' "$scratch/downloads"
 if [[ "$p" == swift15-q8 ]]; then grep -q '0b3b4aa0e367c620756de4f9db77fc18f94fbeb256db208281cbb84c48e2a101' "$scratch/downloads"; else grep -q '2ebba0ff1e63c1ac3fadd4e83efcea189f47f33ec72c91877af94de6ebe30590' "$scratch/downloads"; fi
 if (SPEC_OVERRIDE=fast;configure_profile); then echo 'Wrong-model sidecar accepted' >&2;exit 1;fi
 if (SPEC_OVERRIDE=dflash2;configure_profile); then echo 'Unrelated DFlash accepted' >&2;exit 1;fi
 SPEC_OVERRIDE=native;configure_profile;make_server_args
 [[ " ${SERVER_ARGS[*]} " == *' --spec-type draft-mtp '* ]]
 SWIFT15_MTP_N_MAX=2;configure_profile;make_server_args
 [[ " ${SERVER_ARGS[*]} " == *' --spec-draft-n-max 2 '* ]]
 for invalid in 0 8 abc; do
  if (SWIFT15_MTP_N_MAX="$invalid";configure_profile); then echo 'Invalid MTP depth accepted' >&2;exit 1;fi
 done
 SWIFT15_MTP_N_MAX=3
 SPEC_OVERRIDE=none;configure_profile;make_server_args
 [[ "$SPEC_MODE" == none && "$PROFILE_LABEL" == *'MTP disabled'* ]]
 [[ " ${SERVER_ARGS[*]} " != *' --spec-type '* && " ${SERVER_ARGS[*]} " == *' --ctx-size 262144 '* ]]
done
SPEC_OVERRIDE=""
mkdir -p "$MODEL_ROOT/swift15-uncensored"
touch "$MODEL_ROOT/swift15-uncensored/$SWIFT15U_Q8" "$MODEL_ROOT/swift15-uncensored/$SWIFT15U_MMPROJ"
choose_profile <<< '' > "$scratch/menu"
[[ "$PROFILE" == swift15u-q8 ]]
configure_profile
make_server_args
[[ "$SPEC_MODE" == native && "$KV_TYPE" == q8_0 ]]
[[ "$PARALLEL" == 2 && "$SERVER_CTX" == 524288 ]]
[[ " ${SERVER_ARGS[*]} " == *' --ctx-size 524288 '* && " ${SERVER_ARGS[*]} " == *' --parallel 2 '* && " ${SERVER_ARGS[*]} " == *' --spec-draft-n-max 3 '* ]]
grep -q 'Swift 1.5 Uncensored Q8_K_XL | DEFAULT' "$scratch/menu"
! grep -q 'GSQ IQ3_XXS' "$scratch/menu"
QWEN38_SWIFT15U_SLOTS=1;configure_profile
[[ "$PARALLEL" == 1 && "$SERVER_CTX" == 262144 ]]
QWEN38_SWIFT15U_SLOTS=4
if (configure_profile); then echo 'Unvalidated four Q8-KV slots accepted' >&2;exit 1;fi
echo 'Swift 1.5 native MTP, pinned downloads, two-slot Q8 default and installed-only menu: PASS'
