#!/usr/bin/env bash

# SPEED DEMON — isolated vLLM + Qwen3.8 AWQ + DFlash2 launcher.
# This script deliberately does not touch Hive, watchdog, miner, or driver
# settings. HostLLM.sh is responsible for pausing/resuming OctaSpace around
# this engine when launched from the main menu.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A HiveOS login shell may automatically enter sudo -s. Keep the model,
# container cache, logs, and state below the invoking user's home so root and
# user shells share one SPEED DEMON installation.
speed_home="${HOME}"
if [[ "${EUID}" -eq 0 ]]; then
    speed_owner="${SPEED_DEMON_OWNER_USER:-${SUDO_USER:-user}}"
    speed_resolved_home="$(getent passwd "$speed_owner" 2>/dev/null | awk -F: 'NR == 1 {print $6}')"
    if [[ -n "$speed_resolved_home" && "$speed_resolved_home" != "/root" ]]; then
        speed_home="$speed_resolved_home"
    elif [[ -d /home/user ]]; then
        speed_home="/home/user"
    fi
fi

DATA_ROOT="${SPEED_DEMON_DATA_ROOT:-${speed_home}/.local/share/localllm-speed-demon}"
STATE_ROOT="${SPEED_DEMON_STATE_ROOT:-${speed_home}/.local/state/locallm-speed-demon}"
MODEL_ROOT="${DATA_ROOT}/models"
TARGET_DIR="${MODEL_ROOT}/qwen38-awq-int4"
BF16_DRAFT_DIR="${MODEL_ROOT}/dflash2"
FP8_DRAFT_DIR="${MODEL_ROOT}/dflash2-fp8-vllm"
CACHE_ROOT="${DATA_ROOT}/vllm-cache"
LOG_ROOT="${DATA_ROOT}/logs"
SERVER_INFO="${STATE_ROOT}/server.info"

PORT="${SPEED_DEMON_PORT:-8080}"
BIND_HOST="${SPEED_DEMON_HOST:-0.0.0.0}"
CONTAINER_NAME="${SPEED_DEMON_CONTAINER_NAME:-vllm-speed-demon}"
MAX_CONTEXT="${SPEED_DEMON_MAX_CONTEXT:-262144}"
DRAFT_TOKENS="${SPEED_DEMON_DRAFT_TOKENS:-7}"
DRAFT_MODE="${SPEED_DEMON_DRAFT_MODE:-fp8}"
BASE_IMAGE="vllm/vllm-openai:v0.28.0"
BF16_IMAGE="localllm/speed-demon:vllm-0.28.0-flashinfer-50885"
FP8_IMAGE="${SPEED_DEMON_FP8_IMAGE:-localllm/speed-demon:vllm-0.28.0-flashinfer-50885-fp8-53122}"
TARGET_REPO="cyankiwi/Qwen3.8-27B-AWQ-INT4"
BF16_DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"
FP8_DRAFT_REPO="TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm"
MODE="help"
SMOKE=0
SERVER_LOG=""

case "$DRAFT_MODE" in
    fp8)
        DRAFT_DIR="$FP8_DRAFT_DIR"
        DRAFT_REPO="$FP8_DRAFT_REPO"
        SPEED_DEMON_IMAGE="$FP8_IMAGE"
        DOCKERFILE_DIR="$SCRIPT_DIR/speed-demon-fp8"
        SPEED_LABEL="SPEED DEMON — Qwen3.8-27B AWQ INT4 + FP8 DFlash2"
        SPEED_RESULT="~123 tok/s coding* | ~67 tok/s tools | ~70 tok/s vision"
        VISION_LABEL="target vision input ON; FP8 DFlash2 draft text-only; video unvalidated"
        ;;
    bf16)
        DRAFT_DIR="$BF16_DRAFT_DIR"
        DRAFT_REPO="$BF16_DRAFT_REPO"
        SPEED_DEMON_IMAGE="$BF16_IMAGE"
        DOCKERFILE_DIR="$SCRIPT_DIR/speed-demon"
        SPEED_LABEL="SPEED DEMON — Qwen3.8-27B AWQ INT4 + BF16 DFlash2"
        SPEED_RESULT="~144 tok/s coding | ~97 tok/s agent | ~58 tok/s prose"
        VISION_LABEL="target vision input ON; BF16 DFlash2 draft text-only; video unvalidated"
        ;;
    *)
        printf 'ERROR: SPEED_DEMON_DRAFT_MODE must be fp8 or bf16\n' >&2
        exit 1
        ;;
esac

readonly SPEED_LABEL SPEED_RESULT VISION_LABEL DRAFT_DIR DRAFT_REPO SPEED_DEMON_IMAGE DOCKERFILE_DIR

mkdir -p "$DATA_ROOT" "$STATE_ROOT" "$MODEL_ROOT" "$CACHE_ROOT" "$LOG_ROOT"

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  v1speeddemon.sh --quickstart
  v1speeddemon.sh --smoke
  v1speeddemon.sh --status
  v1speeddemon.sh --dashboard
  v1speeddemon.sh --download
  v1speeddemon.sh --stop

SPEED DEMON uses vLLM 0.28.0, Qwen3.8-27B AWQ INT4, and the Qwen3.8
DFlash2 drafter on two RTX 3090 GPUs. The default drafter is the tested FP8
candidate; set SPEED_DEMON_DRAFT_MODE=bf16 for the BF16 fallback. It is a
text/code-first profile. The target accepts image input, but DFlash2 drafts
from text only; video has not been validated. Automatic tool choice is enabled
with vLLM's `qwen3_xml` parser for Qwen's XML tool-call format. Qwen reasoning
is enabled by default and parsed with `qwen3`; clients may explicitly override
the thinking setting. No LMCache is used.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quickstart|--start)
                MODE="start"
                ;;
            --smoke)
                MODE="smoke"
                SMOKE=1
                ;;
            --status)
                MODE="status"
                ;;
            --dashboard)
                MODE="dashboard"
                ;;
            --download|--install)
                MODE="download"
                ;;
            --stop)
                MODE="stop"
                ;;
            --port)
                [[ $# -ge 2 ]] || die "--port requires a value"
                PORT="$2"
                shift
                ;;
            --port=*)
                PORT="${1#*=}"
                ;;
            --max-context)
                [[ $# -ge 2 ]] || die "--max-context requires a value"
                MAX_CONTEXT="$2"
                shift
                ;;
            --max-context=*)
                MAX_CONTEXT="${1#*=}"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown argument: $1 (use --help)"
                ;;
        esac
        shift
    done
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"
}

validate_settings() {
    [[ "$PORT" =~ ^[0-9]+$ ]] || die "SPEED_DEMON_PORT must be numeric"
    (( PORT >= 1 && PORT <= 65535 )) || die "SPEED_DEMON_PORT must be 1..65535"
    [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] || die "SPEED_DEMON_MAX_CONTEXT must be numeric"
    (( MAX_CONTEXT >= 1024 )) || die "SPEED_DEMON_MAX_CONTEXT must be at least 1024"
    [[ "$DRAFT_TOKENS" =~ ^[1-7]$ ]] || die "SPEED_DEMON_DRAFT_TOKENS must be 1..7"
    [[ "$DRAFT_MODE" == fp8 || "$DRAFT_MODE" == bf16 ]] || die "SPEED_DEMON_DRAFT_MODE must be fp8 or bf16"
}

docker_ready() {
    need_command docker
    docker info >/dev/null 2>&1 || die "Docker is not available; start Docker before SPEED DEMON"
}

container_exists() {
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
    container_exists || return 1
    [[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]
}

port_in_use() {
    command -v ss >/dev/null 2>&1 || return 1
    ss -ltn 2>/dev/null | awk -v port=":${PORT}" '$4 ~ (port "$") { found = 1 } END { exit !found }'
}

ensure_image() {
    if docker image inspect "$SPEED_DEMON_IMAGE" >/dev/null 2>&1; then
        return 0
    fi
    say "Building isolated SPEED DEMON image from ${BASE_IMAGE}..."
    docker build --tag "$SPEED_DEMON_IMAGE" "$DOCKERFILE_DIR"
}

download_snapshot() {
    local repo="$1" dest="$2"
    mkdir -p "$dest"
    say "Downloading model snapshot: $repo"
    docker run --rm \
        --network host \
        --entrypoint python3 \
        --user "$(id -u):$(id -g)" \
        -e "HF_REPO=$repo" \
        -e HF_HOME=/download/.cache/huggingface \
        -v "$dest:/download" \
        "$BASE_IMAGE" \
        -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["HF_REPO"], local_dir="/download")'
}

target_assets_ready() {
    [[ -f "$TARGET_DIR/config.json" ]] || return 1
    [[ -f "$TARGET_DIR/model.safetensors.index.json" ]] || return 1
    [[ -f "$TARGET_DIR/tokenizer.json" ]] || return 1
    local shard_count
    shard_count="$(find "$TARGET_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' -printf x 2>/dev/null | wc -c)"
    (( shard_count >= 5 ))
}

draft_assets_ready() {
    [[ -f "$DRAFT_DIR/config.json" && -f "$DRAFT_DIR/model.safetensors" ]]
}

ensure_assets() {
    if ! target_assets_ready; then
        download_snapshot "$TARGET_REPO" "$TARGET_DIR"
    else
        say "Target model already present: $TARGET_REPO"
    fi
    if ! draft_assets_ready; then
        download_snapshot "$DRAFT_REPO" "$DRAFT_DIR"
    else
        say "DFlash2 draft already present: $DRAFT_REPO"
    fi
    target_assets_ready || die "target model snapshot is incomplete: $TARGET_DIR"
    draft_assets_ready || die "DFlash2 snapshot is incomplete: $DRAFT_DIR"
}

server_log_from_info() {
    [[ -r "$SERVER_INFO" ]] || return 1
    awk -F= '$1 == "log" { print substr($0, index($0, "=") + 1); exit }' "$SERVER_INFO"
}

save_container_logs() {
    local path="${1:-}"
    if [[ -z "$path" ]]; then
        path="${LOG_ROOT}/speed-demon-$(date +%Y%m%d-%H%M%S).log"
    fi
    mkdir -p "$(dirname -- "$path")"
    if container_exists; then
        docker logs "$CONTAINER_NAME" > "$path" 2>&1 || true
        say "Container log saved: $path"
    fi
}

write_server_info() {
    local host_pid=""
    host_pid="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    umask 022
    cat > "$SERVER_INFO" <<EOF
engine=speed-demon
profile=speed-demon
label=$SPEED_LABEL
container=$CONTAINER_NAME
container_pid=$host_pid
image=$SPEED_DEMON_IMAGE
port=$PORT
host=$BIND_HOST
context=$([[ "$SMOKE" -eq 1 ]] && echo 4096 || echo "$MAX_CONTEXT")
model=$TARGET_REPO
model_path=$TARGET_DIR
draft_mode=$DRAFT_MODE
draft=$DRAFT_REPO
draft_path=$DRAFT_DIR
runtime=vLLM 0.28.0 + FlashInfer full decode graph overlay PR #50885
speculation=DFlash2 n=$DRAFT_TOKENS
kv=FP8
cache=none (no LMCache)
tools=automatic function calling (qwen3_xml parser)
default_thinking=on (client may override)
gpus=CUDA0,CUDA1 (2x RTX 3090); CUDA2 unused
speed=$SPEED_RESULT
vision=$VISION_LABEL
log=$SERVER_LOG
EOF
}

remove_stale_container() {
    if container_exists && ! container_running; then
        local stale_log="${LOG_ROOT}/speed-demon-stale-$(date +%Y%m%d-%H%M%S).log"
        save_container_logs "$stale_log"
        docker rm "$CONTAINER_NAME" >/dev/null
    fi
}

wait_for_health() {
    local timeout="${SPEED_DEMON_HEALTH_TIMEOUT:-600}" response elapsed=0
    say "Waiting for SPEED DEMON health (up to ${timeout}s; no long-context request will be sent)"
    for _ in $(seq 1 "$timeout"); do
        if response="$(curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
            say "Health OK: $response"
            return 0
        fi
        if ! container_running; then
            warn "SPEED DEMON container exited before becoming healthy"
            save_container_logs "$SERVER_LOG"
            return 1
        fi
        elapsed=$((elapsed + 1))
        if (( elapsed % 10 == 0 )); then
            say "  Still loading... ${elapsed}s elapsed (container log: $(basename -- "$SERVER_LOG"))"
        fi
        sleep 1
    done
    warn "SPEED DEMON did not become healthy"
    save_container_logs "$SERVER_LOG"
    return 1
}

start_server() {
    local context="$MAX_CONTEXT"
    local spec_json
    local -a docker_args vllm_args

    docker_ready
    validate_settings
    if container_running; then
        say "SPEED DEMON is already running in container $CONTAINER_NAME"
        return 0
    fi
    remove_stale_container
    if port_in_use; then
        die "TCP port $PORT is already in use; stop the existing engine first"
    fi
    ensure_image
    ensure_assets
    (( SMOKE )) && context=4096

    SERVER_LOG="${LOG_ROOT}/speed-demon-$(date +%Y%m%d-%H%M%S).log"
    if [[ "$DRAFT_MODE" == fp8 ]]; then
        spec_json="{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":${DRAFT_TOKENS},\"quantization\":\"compressed-tensors\"}"
    else
        spec_json="{\"method\":\"dflash\",\"model\":\"/models/dflash2\",\"num_speculative_tokens\":${DRAFT_TOKENS}}"
    fi
    vllm_args=(
        /models/target
        --served-model-name speed-demon
        --tensor-parallel-size 2
        --dtype bfloat16
        --kv-cache-dtype fp8
        --mamba-ssm-cache-dtype bfloat16
        --mamba-cache-mode align
        --speculative-config "$spec_json"
        --gpu-memory-utilization 0.90
        --max-model-len "$context"
        --max-num-seqs 1
        --max-num-batched-tokens 2048
        --enable-prefix-caching
        --enable-chunked-prefill
        --enable-auto-tool-choice
        --tool-call-parser qwen3_xml
        --reasoning-parser qwen3
        --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
        --attention-backend FLASHINFER
        --performance-mode balanced
        --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
        --disable-custom-all-reduce
        --trust-remote-code
        --generation-config vllm
        --host "$BIND_HOST"
        --port "$PORT"
    )
    docker_args=(
        run -d
        --name "$CONTAINER_NAME"
        --label "com.pukerud.localllm.engine=speed-demon"
        --restart=no
        --gpus '"device=0,1"'
        --ipc=host
        --shm-size=16g
        --network host
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID
        -e CUDA_VISIBLE_DEVICES=0,1
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_DISABLE=1
        -e OMP_NUM_THREADS=1
        -v "$TARGET_DIR:/models/target:ro"
        -v "$DRAFT_DIR:/models/dflash2:ro"
        -v "$CACHE_ROOT:/root/.cache/vllm"
    )

    say "Starting: $SPEED_LABEL"
    say "  target: $TARGET_REPO"
    say "  draft:  $DRAFT_REPO (DFlash2 n=${DRAFT_TOKENS})"
    say "  context: $context (native configured context: $MAX_CONTEXT)"
    say "  GPUs: CUDA0,CUDA1 (2x RTX 3090); CUDA2 unused"
    say "  speed: $SPEED_RESULT"
    say "  vision: $VISION_LABEL"
    say "  log: $SERVER_LOG"

    docker_id="$(docker "${docker_args[@]}" "$SPEED_DEMON_IMAGE" "${vllm_args[@]}")"
    [[ -n "$docker_id" ]] || die "Docker did not return a container ID"
    write_server_info
    if ! wait_for_health; then
        stop_server
        return 1
    fi
    say "SPEED DEMON is ready on http://${BIND_HOST}:${PORT}"
    say "API base: http://${BIND_HOST}:${PORT}/v1"
}

stop_server() {
    docker_ready
    local log_path="${SERVER_LOG:-}"
    if [[ -z "$log_path" ]]; then
        log_path="$(server_log_from_info || true)"
    fi
    if container_exists; then
        say "Stopping SPEED DEMON container $CONTAINER_NAME"
        docker stop -t 30 "$CONTAINER_NAME" >/dev/null 2>&1 || docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
        save_container_logs "$log_path"
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    rm -f "$SERVER_INFO"
}

response_text() {
    local file="$1"
    python3 - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
if data.get("error"):
    raise SystemExit(json.dumps(data["error"], ensure_ascii=False))
try:
    message = data["choices"][0]["message"]
except Exception as exc:
    raise SystemExit(f"response has no choices[0].message: {exc}")
value = message.get("content") or message.get("reasoning_content") or ""
if isinstance(value, list):
    value = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in value)
value = str(value).strip()
if not value:
    raise SystemExit("response content is empty")
print(value)
PY
}

run_text_smoke() {
    local out="${STATE_ROOT}/smoke-text.json"
    say "Smoke 1/2: short thinking-enabled text request (max 64 tokens)"
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"speed-demon","messages":[{"role":"user","content":"Reply with a short confirmation that SPEED DEMON is ready."}],"max_tokens":64,"temperature":0.2,"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}' \
        -o "$out"
    response_text "$out" | head -c 500
    printf '\n'
}

make_test_png() {
    local path="$1"
    python3 - "$path" <<'PY'
import struct
import sys
import zlib

path = sys.argv[1]
w = h = 64
rows = []
for _ in range(h):
    rows.append(b"\x00" + bytes([255, 0, 0] * (w // 2) + [0, 0, 255] * (w // 2)))
raw = b"".join(rows)
def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(raw, 9))
png += chunk(b"IEND", b"")
open(path, "wb").write(png)
PY
}

run_vision_smoke() {
    local img="${STATE_ROOT}/smoke-red-blue.png"
    local b64 payload out="${STATE_ROOT}/smoke-vision.json"
    make_test_png "$img"
    b64="$(base64 -w0 "$img")"
    payload="${STATE_ROOT}/smoke-vision-request.json"
    python3 - "$b64" "$payload" <<'PY'
import json
import sys

b64, path = sys.argv[1:]
body = {
    "model": "speed-demon",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "In one short sentence, which color is on the left side of the image?"},
    ]}],
    "max_tokens": 64,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
}
open(path, "w", encoding="utf-8").write(json.dumps(body))
PY
    say "Smoke 2/2: one small 64x64 red/blue image request (max 32 tokens)"
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload" \
        -o "$out"
    response_text "$out" | head -c 500
    printf '\n'
}

run_smoke() {
    docker_ready
    if container_running; then
        die "stop the running SPEED DEMON container before running --smoke"
    fi
    start_server
    local rc=0
    run_text_smoke || rc=1
    run_vision_smoke || rc=1
    say "GPU snapshot after smoke:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || true
    stop_server
    if (( rc == 0 )); then
        say "SHORT SMOKE PASSED: text + image target vision"
    else
        warn "SHORT SMOKE FAILED; container log retained at ${SERVER_LOG:-$LOG_ROOT}"
    fi
    return "$rc"
}

show_status() {
    docker_ready
    say "$SPEED_LABEL"
    say "Target model: $TARGET_REPO"
    say "DFlash2 draft: $DRAFT_REPO ($DRAFT_MODE, n=${DRAFT_TOKENS})"
    say "Speed reference: $SPEED_RESULT"
    say "Vision: $VISION_LABEL"
    say "Context: $MAX_CONTEXT | KV: FP8 | LMCache: OFF"
    say "GPUs: CUDA0,CUDA1 (2x RTX 3090); CUDA2 unused"
    say "Container: $CONTAINER_NAME"
    if container_running; then
        say "State: RUNNING"
        if [[ -r "$SERVER_INFO" ]]; then
            cat "$SERVER_INFO"
        fi
        if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
            printf '\n'
        else
            say "Health: not responding"
        fi
        say "Recent container log:"
        docker logs --tail 20 "$CONTAINER_NAME" 2>&1 || true
    else
        say "State: STOPPED"
        [[ -r "$SERVER_INFO" ]] && say "Stale state file: $SERVER_INFO"
    fi
    nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || true
}

show_dashboard() {
    local choice
    docker_ready
    if ! container_running; then
        warn "SPEED DEMON is not running"
        return 1
    fi
    while true; do
        clear 2>/dev/null || true
        echo "=================================================================="
        echo "  SPEED DEMON RUNNING"
        echo "=================================================================="
        say "  Model:    $TARGET_REPO"
        say "  Draft:    $DRAFT_REPO ($DRAFT_MODE DFlash2 n=${DRAFT_TOKENS}; text-only draft)"
        say "  Context:  $MAX_CONTEXT | KV: FP8 | LMCache: OFF"
        say "  Speed:    $SPEED_RESULT"
        say "  Vision:   target image input ON; draft text-only; video unvalidated"
        say "  Tools:    automatic function calling (qwen3_xml parser)"
        say "  Thinking: ON by default (qwen3 parser; client may override)"
        say "  GPUs:     CUDA0,CUDA1 (2x RTX 3090); CUDA2 unused"
        echo ""
        say "  API:      http://${BIND_HOST}:${PORT}/v1"
        say "  Model ID: speed-demon"
        echo "=================================================================="
        printf '  Health: '
        curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null || printf 'not responding'
        printf '\n'
        say "  [1] Stop SPEED DEMON and return to menu"
        say "  [2] Return to menu (keep server running)"
        say "  [r] Refresh"
        echo ""
        read -r -p "  Select [1/2/r]: " choice
        case "$choice" in
            1)
                stop_server
                say "  SPEED DEMON stopped."
                sleep 1
                return 0
                ;;
            2) return 0 ;;
            r|R) ;;
            *) ;;
        esac
        container_running || return 0
    done
}

main() {
    parse_args "$@"
    validate_settings
    case "$MODE" in
        help)
            usage
            ;;
        stop)
            stop_server
            ;;
        download)
            docker_ready
            ensure_image
            ensure_assets
            say "SPEED DEMON assets and runtime image are ready."
            ;;
        start)
            start_server
            ;;
        smoke)
            run_smoke
            ;;
        status)
            show_status
            ;;
        dashboard)
            show_dashboard
            ;;
        *)
            die "internal error: unsupported mode '$MODE'"
            ;;
    esac
}

main "$@"
