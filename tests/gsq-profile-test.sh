#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Load definitions without executing main (no downloads/GPU operations).
source <(head -n -1 "$root/v1qwen38.sh")
GPU_DEVICE_LIST=CUDA0,CUDA1,CUDA2,CUDA3
GPU_TENSOR_SPLIT=1,1,1,1
for p in gsq-q2 gsq-iq2 gsq-iq3; do
    PROFILE="$p"
    configure_profile
    [[ "$RUNTIME_KIND" == gsq && "$SPEC_MODE" == none && "$PARALLEL" == 1 && "$SERVER_CTX" == 32768 ]]
    [[ "$MODEL_PATH" == *-00001-of-00002.gguf && "$MMPROJ_PATH" == *BF16.gguf ]]
    make_server_args
    args=" ${SERVER_ARGS[*]} "
    [[ "$args" == *' --lazy-mode on '* && "$args" == *' --load-mode mmap '* ]]
    [[ "$args" == *' --reasoning-effort xhigh '* && "$args" != *' --spec-type '* ]]
done
GPU_COUNT=4
GPU_INDICES=(0 1 2 3)
GPU_NAMES=('NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090')
GPU_MEMORY_MIB=(24576 24576 24576 24576)
PROFILE=gsq-q2
configure_profile
make_server_args
args=" ${SERVER_ARGS[*]} "
[[ "$SERVER_CTX" == 262144 && "$GSQ_TUNED" == 1 ]]
[[ "$args" == *' --device CUDA0,CUDA1 '* && "$args" == *' --tensor-split 23,25 '* ]]
[[ "$args" == *' --ctx-size 262144 '* && "$args" == *' --reasoning-effort xhigh '* ]]
QWEN38_GSQ_TUNED=0 configure_profile
[[ "$GSQ_TUNED" == 0 && "$SERVER_CTX" == 32768 ]]
QWEN38_GPU_INDICES=0,1,2,3 configure_profile
[[ "$GSQ_TUNED" == 0 ]]
QWEN38_GSQ_CTX=131072 configure_profile
[[ "$SERVER_CTX" == 131072 ]]
PROFILE=gsq-iq3
configure_profile
[[ "$GSQ_TUNED" == 0 && "$SERVER_CTX" == 32768 ]]
PROFILE=hauhau-q8-fastmtp-q4kv-xhigh
configure_profile
[[ "$RUNTIME_KIND" == hauhau && "$KV_TYPE" == q4_0 && "$SPEC_MODE" == fast ]]
if (PROFILE=gsq-q2; SPEC_OVERRIDE=fast; configure_profile); then
    echo 'Unexpected GSQ speculative override acceptance' >&2; exit 1
fi
if (PROFILE=gsq-q2; QWEN38_GSQ_CTX=999999; configure_profile); then
    echo 'Unexpected context acceptance' >&2; exit 1
fi
echo 'GSQ profiles and production regression checks passed'
