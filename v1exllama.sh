#!/usr/bin/env bash

# Qwen3.8-27B EXL3 vision launcher using ExLlamaV3 + TabbyAPI.
# This is an isolated, reversible option. It does not touch Hive, watchdog,
# miner, driver, or the existing llama.cpp/vLLM runtimes.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# HiveOS may enter a root shell automatically via `sudo -s`. Keep the model,
# image build cache, logs, and state below the invoking user's home in that
# case, just like the other LocalLLM launchers.
exllama_home="${HOME}"
if [[ "${EUID}" -eq 0 ]]; then
    exllama_owner="${EXLLAMA_OWNER_USER:-${SUDO_USER:-user}}"
    exllama_resolved_home="$(getent passwd "$exllama_owner" 2>/dev/null | awk -F: 'NR == 1 {print $6}')"
    if [[ -n "$exllama_resolved_home" && "$exllama_resolved_home" != "/root" ]]; then
        exllama_home="$exllama_resolved_home"
    elif [[ -d /home/user ]]; then
        exllama_home="/home/user"
    fi
fi

DATA_ROOT="${EXLLAMA_DATA_ROOT:-${exllama_home}/.local/share/locallm-exllama}"
STATE_ROOT="${EXLLAMA_STATE_ROOT:-${exllama_home}/.local/state/locallm-exllama}"
MODEL_ROOT="${DATA_ROOT}/models"
MODEL_DIR="${MODEL_ROOT}/qwen38-exl3-sc6-h6-v6"
CACHE_ROOT="${DATA_ROOT}/cache"
LOG_ROOT="${DATA_ROOT}/logs"
MODEL_COMPLETE="${MODEL_DIR}/.complete"
CONFIG_FILE="${STATE_ROOT}/config.yml"
SERVER_INFO="${STATE_ROOT}/server.info"
SPEED_CACHE="${STATE_ROOT}/speed-results.tsv"

PORT="${EXLLAMA_PORT:-8080}"
BIND_HOST="${EXLLAMA_HOST:-0.0.0.0}"
CONTAINER_NAME="${EXLLAMA_CONTAINER_NAME:-tabbyapi-exllama}"
MAX_CONTEXT="${EXLLAMA_MAX_CONTEXT:-262144}"
CACHE_MODE="${EXLLAMA_CACHE_MODE:-8,8}"
VISION_OFFLOAD="${EXLLAMA_VISION_OFFLOAD:-0}"
IMAGE="${EXLLAMA_IMAGE:-localllm/qwen38-exllama:tabbyapi-1.4.4}"
MODEL_ID="qwen38-exl3-sc6-h6-v6"
MODEL_REPO="turboderp/Qwen3.8-27B-exl3"
MODEL_REVISION="SC_6.00bpw_H6_V6"
MODEL_BASE="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}"

# The three large files are checked against the LFS SHA-256 values published
# by Hugging Face. Small metadata files are checked for presence and are
# fetched only from the pinned revision above.
MODEL_ASSETS=(
    ".gitattributes|"
    "LICENSE|"
    "README.md|"
    "chat_template.jinja|"
    "chat_template.jinja.fixed|"
    "config.json|"
    "generation_config.json|"
    "merges.txt|"
    "model-00001-of-00003.safetensors|39b6523bc82ce685be2184634a1f74c1ee6990cbcdb5647370baeb2ccab0536b"
    "model-00002-of-00003.safetensors|8d99c95b913e23feb0727e885e4a6e1f2e8cce38fddbfb5f57249c69d9887ca6"
    "model-00003-of-00003.safetensors|0dcbda9f02884f3b7cfb6dcd85fa64482dc9e3fa31f531b0a71d70417bd4bfe1"
    "model.safetensors.index.json|"
    "preprocessor_config.json|"
    "quantization_config.json|"
    "tokenizer.json|"
    "tokenizer_config.json|"
    "video_preprocessor_config.json|"
    "vocab.json|"
)

MODE="help"
SMOKE=0
SERVER_LOG=""

mkdir -p "$DATA_ROOT" "$STATE_ROOT" "$MODEL_ROOT" "$CACHE_ROOT" "$LOG_ROOT"

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  v1exllama.sh --quickstart
  v1exllama.sh --smoke
  v1exllama.sh --speed-test
  v1exllama.sh --status
  v1exllama.sh --dashboard
  v1exllama.sh --download
  v1exllama.sh --build
  v1exllama.sh --stop

The profile is turboderp/Qwen3.8-27B-exl3 revision
SC_6.00bpw_H6_V6: 6-bit EXL3 text plus 6-bit vision, native 262K context.
It uses ExLlamaV3 1.4.4 through TabbyAPI, autosplits across available RTX
30-series GPUs, uses an 8-bit K/V cache by default, and has no draft model.
Use --speed-test for a quick two-prompt 4096-token decode measurement; the
result is cached for the HostLLM menu. Set EXLLAMA_CACHE_MODE=Q6 or Q4 if
more cache headroom is needed. The short smoke test uses a 4096-token cache;
--quickstart uses the configured native context. No Hive/miner/watchdog
settings are changed.
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
            --speed-test)
                MODE="speed"
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
            --build)
                MODE="build"
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
            --cache-mode)
                [[ $# -ge 2 ]] || die "--cache-mode requires a value"
                CACHE_MODE="$2"
                shift
                ;;
            --cache-mode=*)
                CACHE_MODE="${1#*=}"
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

validate_settings() {
    [[ "$PORT" =~ ^[0-9]+$ ]] || die "EXLLAMA_PORT must be numeric"
    (( PORT >= 1 && PORT <= 65535 )) || die "EXLLAMA_PORT must be 1..65535"
    [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]] || die "EXLLAMA_MAX_CONTEXT must be numeric"
    (( MAX_CONTEXT >= 1024 )) || die "EXLLAMA_MAX_CONTEXT must be at least 1024"
    (( MAX_CONTEXT % 256 == 0 )) || die "EXLLAMA_MAX_CONTEXT must be a multiple of 256"
    case "$CACHE_MODE" in
        FP16|Q8|Q6|Q4) ;;
        [2-8],[2-8]) ;;
        *) die "EXLLAMA_CACHE_MODE must be FP16, Q8, Q6, Q4, or K,V bits from 2-8" ;;
    esac
    [[ "$VISION_OFFLOAD" == 0 || "$VISION_OFFLOAD" == 1 ]] || die "EXLLAMA_VISION_OFFLOAD must be 0 or 1"
}

docker_ready() {
    command -v docker >/dev/null 2>&1 || die "missing command 'docker'"
    docker info >/dev/null 2>&1 || die "Docker is not available; start Docker before ExLlamaV3"
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

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

verify_asset() {
    local file="$1" expected="${2:-}" actual
    [[ -s "$file" ]] || return 1
    [[ -n "$expected" ]] || return 0
    actual="$(sha256_file "$file")"
    if [[ "$actual" != "$expected" ]]; then
        warn "checksum mismatch for $file"
        warn "expected: $expected"
        warn "actual:   $actual"
        return 1
    fi
    return 0
}

download_file() {
    local file="$1" expected="${2:-}" dest="${MODEL_DIR}/${1}" tmp
    mkdir -p "$MODEL_DIR"

    if [[ -f "$dest" ]] && verify_asset "$dest" "$expected"; then
        say "Already present: $file"
        return 0
    fi
    if [[ -f "$dest" ]]; then
        rm -f -- "$dest"
    fi

    tmp="${dest}.part"
    say "Downloading: $file"
    wget --continue --tries=20 --waitretry=10 --timeout=90 \
        --progress=dot:giga -O "$tmp" "$MODEL_BASE/$file?download=true"
    mv -f -- "$tmp" "$dest"
    verify_asset "$dest" "$expected" || die "checksum verification failed after download: $dest"
    say "Verified: $file"
}

model_assets_ready() {
    local spec file expected
    [[ -f "$MODEL_COMPLETE" ]] || return 1
    # The completion marker means the large shard checksums have already been
    # verified. Recheck presence on every start without hashing 22 GiB again.
    for spec in "${MODEL_ASSETS[@]}"; do
        IFS='|' read -r file expected <<< "$spec"
        [[ -s "${MODEL_DIR}/${file}" ]] || return 1
    done
}

ensure_assets() {
    local spec file expected
    if model_assets_ready; then
        say "Model assets ready: ${MODEL_REPO} @ ${MODEL_REVISION}"
        return 0
    fi

    say "Checking/downloading EXL3 model assets for ${MODEL_REVISION}..."
    for spec in "${MODEL_ASSETS[@]}"; do
        IFS='|' read -r file expected <<< "$spec"
        download_file "$file" "$expected"
    done
    : > "$MODEL_COMPLETE"
    say "Model assets ready: ${MODEL_DIR}"
}

ensure_image() {
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        return 0
    fi
    say "Building isolated ExLlamaV3 image: $IMAGE"
    docker build --network host --tag "$IMAGE" "${SCRIPT_DIR}/exllama-v3"
}

server_log_from_info() {
    [[ -r "$SERVER_INFO" ]] || return 1
    awk -F= '$1 == "log" { print substr($0, index($0, "=") + 1); exit }' "$SERVER_INFO"
}

save_container_logs() {
    local path="${1:-}"
    if [[ -z "$path" ]]; then
        path="${LOG_ROOT}/exllama-$(date +%Y%m%d-%H%M%S).log"
    fi
    mkdir -p "$(dirname -- "$path")"
    if container_exists; then
        docker logs "$CONTAINER_NAME" > "$path" 2>&1 || true
        say "Container log saved: $path"
    fi
}

remove_stale_container() {
    if container_exists && ! container_running; then
        local stale_log="${LOG_ROOT}/exllama-stale-$(date +%Y%m%d-%H%M%S).log"
        save_container_logs "$stale_log"
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}

write_config() {
    local context="$MAX_CONTEXT" vision_offload=false
    (( SMOKE )) && context=4096
    [[ "$VISION_OFFLOAD" == 1 ]] && vision_offload=true

    umask 022
    cat > "$CONFIG_FILE" <<EOF
network:
  host: ${BIND_HOST}
  port: ${PORT}
  disable_auth: true
  api_servers: ["OAI"]
model:
  model_dir: /app/models
  model_name: ${MODEL_ID}
  backend: exllamav3
  max_seq_len: ${context}
  cache_size: ${context}
  cache_mode: "${CACHE_MODE}"
  tensor_parallel: false
  tensor_parallel_backend: native
  gpu_split_auto: true
  autosplit_reserve: [512, 512, 512]
  chunk_size: 2048
  output_chunking: true
  max_batch_size: 1
  vision: true
  vision_offload: ${vision_offload}
  template_vars_default:
    enable_thinking: true
    preserve_thinking: true
  reasoning: true
  tool_format: qwen3_5
draft_model:
  draft_mode: disabled
sampling:
  override_preset: safe_defaults
memory:
  sysmem_recurrent_cache: 4096
  cuda_malloc_async: true
EOF
}

write_server_info() {
    local host_pid
    host_pid="$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    umask 022
    cat > "$SERVER_INFO" <<EOF
engine=exllama-v3
profile=exllama-qwen38-sc6-h6-v6
label=Qwen3.8-27B EXL3 SC_6.00bpw_H6_V6 / vision / native 262K
container=$CONTAINER_NAME
container_pid=$host_pid
image=$IMAGE
port=$PORT
host=$BIND_HOST
context=$MAX_CONTEXT
cache=$CACHE_MODE
model=$MODEL_REPO
revision=$MODEL_REVISION
model_path=$MODEL_DIR
runtime=ExLlamaV3 1.4.4 + TabbyAPI
vision=6-bit quantized vision tower; image path tested; video config present but not long-video tested
tools=automatic function calling (qwen3_5 parser)
default_thinking=on (client may override)
gpus=autosplit across visible RTX 30-series GPUs
log=$SERVER_LOG
EOF
}

wait_for_health() {
    local timeout="${EXLLAMA_HEALTH_TIMEOUT:-600}" response elapsed=0
    say "Waiting for ExLlamaV3 health (up to ${timeout}s)..."
    for _ in $(seq 1 "$timeout"); do
        if response="$(curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
            say "Health OK: $response"
            return 0
        fi
        if ! container_running; then
            warn "ExLlamaV3 container exited before becoming healthy"
            save_container_logs "$SERVER_LOG"
            return 1
        fi
        elapsed=$((elapsed + 1))
        if (( elapsed % 10 == 0 )); then
            say "  Still loading... ${elapsed}s elapsed (container log: $(basename -- "$SERVER_LOG"))"
        fi
        sleep 1
    done
    warn "ExLlamaV3 did not become healthy"
    save_container_logs "$SERVER_LOG"
    return 1
}

start_server() {
    local context="$MAX_CONTEXT" docker_id
    local -a docker_args

    docker_ready
    validate_settings
    if container_running; then
        say "ExLlamaV3 is already running in container $CONTAINER_NAME"
        return 0
    fi
    remove_stale_container
    if port_in_use; then
        die "TCP port $PORT is already in use; stop the existing engine first"
    fi
    ensure_image
    ensure_assets
    write_config
    (( SMOKE )) && context=4096

    SERVER_LOG="${LOG_ROOT}/exllama-$(date +%Y%m%d-%H%M%S).log"
    docker_args=(
        run -d
        --name "$CONTAINER_NAME"
        --label "com.pukerud.localllm.engine=exllama-v3"
        --restart=no
        --gpus all
        --entrypoint /opt/venv/bin/python
        --ipc=host
        --shm-size=8g
        --network host
        --ulimit memlock=-1
        --ulimit nofile=1048576
        -e NVIDIA_VISIBLE_DEVICES=all
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID
        -e TRITON_CACHE_DIR=/root/.cache/triton
        -v "$MODEL_ROOT:/app/models:ro"
        -v "$CONFIG_FILE:/app/config.yml:ro"
        -v "$CACHE_ROOT:/root/.cache"
    )

    say "Starting: Qwen3.8-27B EXL3 SC_6.00bpw_H6_V6"
    say "  runtime: ExLlamaV3 1.4.4 + TabbyAPI"
    say "  model:  $MODEL_REPO @ $MODEL_REVISION"
    say "  context: $context (native configured context: $MAX_CONTEXT)"
    say "  cache:   $CACHE_MODE"
    say "  vision:  6-bit quantized vision tower; image input ON"
    say "  GPUs:    autosplit across visible RTX 30-series GPUs"
    say "  log:     $SERVER_LOG"

    docker_id="$(docker "${docker_args[@]}" "$IMAGE" main.py --config /app/config.yml)"
    [[ -n "$docker_id" ]] || die "Docker did not return a container ID"
    write_server_info
    if ! wait_for_health; then
        stop_server
        return 1
    fi
    say "ExLlamaV3 is ready on http://${BIND_HOST}:${PORT}"
    say "API base: http://${BIND_HOST}:${PORT}/v1"
}

stop_server() {
    docker_ready
    local log_path="${SERVER_LOG:-}"
    if [[ -z "$log_path" ]]; then
        log_path="$(server_log_from_info || true)"
    fi
    if container_exists; then
        say "Stopping ExLlamaV3 container $CONTAINER_NAME"
        docker stop -t 45 "$CONTAINER_NAME" >/dev/null 2>&1 || docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
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

speed_tps_from_logs() {
    # TabbyAPI reports prompt-ingestion speed before the generation speed:
    # "... at 177.78 T/s, Generate: 32.12 T/s". Use the value after
    # Generate:, not the earlier prompt-processing number.
    docker logs "$CONTAINER_NAME" 2>&1 \
        | awk '/Generate:/ { for (i = 1; i <= NF; i++) if ($i == "Generate:") { value = $(i + 1); sub(/[^0-9.].*$/, "", value); print value } }' \
        | tail -1
}

speed_cache_row() {
    local row
    [[ -r "$SPEED_CACHE" ]] || return 1
    row="$(awk -F'|' '$1 == "exllama-qwen38-sc6-h6-v6" { row = $0 } END { print row }' "$SPEED_CACHE")"
    [[ -n "$row" ]] || return 1
    printf '%s\n' "$row"
}

speed_detail() {
    local row date context coding story average
    row="$(speed_cache_row || true)"
    if [[ -z "$row" ]]; then
        printf 'not benchmarked'
        return 0
    fi
    IFS='|' read -r _ date context coding story average <<< "$row"
    printf 'avg %s tok/s | coding %s | story %s | %s' "$average" "$coding" "$story" "$date"
}

record_speed_result() {
    local coding="$1" story="$2" average="$3" tmp
    tmp="${SPEED_CACHE}.tmp.$$"
    umask 022
    if [[ -f "$SPEED_CACHE" ]]; then
        awk -F'|' '$1 != "exllama-qwen38-sc6-h6-v6"' "$SPEED_CACHE" > "$tmp"
    else
        : > "$tmp"
    fi
    printf 'exllama-qwen38-sc6-h6-v6|%s|4096|%s|%s|%s\n' \
        "$(date +%Y-%m-%d)" "$coding" "$story" "$average" >> "$tmp"
    mv -f -- "$tmp" "$SPEED_CACHE"
}

speed_request() {
    local kind="$1" prompt="$2" max_tokens="$3" payload out
    payload="${STATE_ROOT}/speed-${kind}.request.json"
    out="${STATE_ROOT}/speed-${kind}.response.json"
    python3 - "$prompt" "$max_tokens" "$payload" <<'PY'
import json
import sys

prompt, max_tokens, path = sys.argv[1:]
body = {
    "model": "qwen38-exl3-sc6-h6-v6",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": 0.2,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(body, handle)
PY
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload" \
        -o "$out"
    response_text "$out" >/dev/null
}

run_speed_test() {
    local coding_tps story_tps average
    docker_ready
    if container_running; then
        die "stop the running ExLlamaV3 container before running --speed-test"
    fi
    SMOKE=1
    say "Speed test: ExLlamaV3 (4096-token context; two short prompts)"
    ensure_image
    start_server || return 1

    say "  Warm-up request..."
    if ! speed_request warmup 'Reply with one word: ready.' 16; then
        stop_server
        return 1
    fi
    say "  Coding prompt..."
    if ! speed_request coding 'Write a concise Python function that merges overlapping intervals. Include type hints, one example, and a brief time-complexity note. Keep the answer under 120 words.' 192; then
        stop_server
        return 1
    fi
    coding_tps="$(speed_tps_from_logs)"
    [[ "$coding_tps" =~ ^[0-9]+([.][0-9]+)?$ ]] || { stop_server; die "could not read coding generation speed from TabbyAPI logs"; }
    say "    coding: ${coding_tps} tok/s"

    say "  Story prompt..."
    if ! speed_request story 'Write a short story in around 100 words about a night-shift engineer who receives a radio message from tomorrow. Give it a clear ending.' 192; then
        stop_server
        return 1
    fi
    story_tps="$(speed_tps_from_logs)"
    [[ "$story_tps" =~ ^[0-9]+([.][0-9]+)?$ ]] || { stop_server; die "could not read story generation speed from TabbyAPI logs"; }
    say "    story: ${story_tps} tok/s"

    average="$(python3 - "$coding_tps" "$story_tps" <<'PY'
import sys
values = [float(value) for value in sys.argv[1:]]
print(f"{sum(values) / len(values):.2f}")
PY
)"
    record_speed_result "$coding_tps" "$story_tps" "$average"
    stop_server
    say "  Speed result: avg ${average} tok/s (cached for the HostLLM menu)"
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

run_text_smoke() {
    local out="${STATE_ROOT}/smoke-text.json"
    say "Smoke 1/3: short text request with thinking explicitly disabled"
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen38-exl3-sc6-h6-v6","messages":[{"role":"user","content":"Reply with exactly: EXL3 text pass"}],"max_tokens":64,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' \
        -o "$out"
    response_text "$out" | head -c 500
    printf '\n'
}

run_vision_smoke() {
    local img="${STATE_ROOT}/smoke-red-blue.png" b64 payload out="${STATE_ROOT}/smoke-vision.json"
    make_test_png "$img"
    b64="$(base64 -w0 "$img")"
    payload="${STATE_ROOT}/smoke-vision-request.json"
    python3 - "$b64" "$payload" <<'PY'
import json
import sys

b64, path = sys.argv[1:]
body = {
    "model": "qwen38-exl3-sc6-h6-v6",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Inspect this image. What color is on the left and what color is on the right? Reply exactly: left=<color>, right=<color>"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    ]}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
open(path, "w", encoding="utf-8").write(json.dumps(body))
PY
    say "Smoke 2/3: one small 64x64 red/blue image request"
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload" \
        -o "$out"
    response_text "$out" | head -c 500
    printf '\n'
}

run_tool_smoke() {
    local out="${STATE_ROOT}/smoke-tool.json"
    cat > "${STATE_ROOT}/smoke-tool-request.json" <<'JSON'
{
  "model":"qwen38-exl3-sc6-h6-v6",
  "messages":[{"role":"user","content":"Use the calculator tool to calculate 2 + 2."}],
  "tools":[{"type":"function","function":{"name":"calculator","description":"Calculate an arithmetic expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
  "tool_choice":"auto",
  "max_tokens":128,
  "temperature":0,
  "stream":false,
  "chat_template_kwargs":{"enable_thinking":false}
}
JSON
    say "Smoke 3/3: automatic calculator tool call"
    curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@${STATE_ROOT}/smoke-tool-request.json" \
        -o "$out"
    python3 - "$out" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
message = data["choices"][0]["message"]
tools = message.get("tool_calls") or []
if not tools or tools[0].get("function", {}).get("name") != "calculator":
    raise SystemExit(f"calculator tool call missing: {data}")
print(json.dumps(tools[0], ensure_ascii=False))
PY
}

run_smoke() {
    docker_ready
    if container_running; then
        die "stop the running ExLlamaV3 container before running --smoke"
    fi
    start_server
    local rc=0
    run_text_smoke || rc=1
    run_vision_smoke || rc=1
    run_tool_smoke || rc=1
    say "GPU snapshot after smoke:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || true
    stop_server
    if (( rc == 0 )); then
        say "SHORT SMOKE PASSED: text + image + tool call"
    else
        warn "SHORT SMOKE FAILED; container log retained at ${SERVER_LOG:-$LOG_ROOT}"
    fi
    return "$rc"
}

show_status() {
    docker_ready
    say "Qwen3.8-27B EXL3 SC_6.00bpw_H6_V6"
    say "Repository: $MODEL_REPO @ $MODEL_REVISION"
    say "Runtime: ExLlamaV3 1.4.4 + TabbyAPI"
    say "Context: $MAX_CONTEXT | KV cache: $CACHE_MODE"
    say "Vision: 6-bit quantized tower enabled"
    say "Container: $CONTAINER_NAME"
    if container_running; then
        say "State: RUNNING"
        [[ -r "$SERVER_INFO" ]] && cat "$SERVER_INFO"
        if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
            printf '\n'
        else
            say "Health: not responding"
        fi
        say "Recent container log:"
        docker logs --tail 25 "$CONTAINER_NAME" 2>&1 || true
    else
        say "State: STOPPED"
    fi
    nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || true
}

display_ip() {
    local ip="${EXLLAMA_DISPLAY_IP:-}"
    if [[ -z "$ip" ]]; then
        ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    printf '%s' "${ip:-127.0.0.1}"
}

cpu_percent() {
    local _ user1 nice1 system1 idle1 iowait1 irq1 softirq1 steal1 guest1 guest_nice1
    local user2 nice2 system2 idle2 iowait2 irq2 softirq2 steal2 guest2 guest_nice2
    local total1 idle_total1 total2 idle_total2 total_delta idle_delta busy_delta
    read -r _ user1 nice1 system1 idle1 iowait1 irq1 softirq1 steal1 guest1 guest_nice1 < /proc/stat
    total1=$((user1 + nice1 + system1 + idle1 + iowait1 + irq1 + softirq1 + steal1))
    idle_total1=$((idle1 + iowait1))
    sleep 0.1
    read -r _ user2 nice2 system2 idle2 iowait2 irq2 softirq2 steal2 guest2 guest_nice2 < /proc/stat
    total2=$((user2 + nice2 + system2 + idle2 + iowait2 + irq2 + softirq2 + steal2))
    idle_total2=$((idle2 + iowait2))
    total_delta=$((total2 - total1))
    idle_delta=$((idle_total2 - idle_total1))
    busy_delta=$((total_delta - idle_delta))
    if (( total_delta > 0 )); then
        printf '%d' $((busy_delta * 100 / total_delta))
    else
        printf '0'
    fi
}

gpu_dashboard() {
    local index util used total temp util_display used_display total_display percent_display
    local total_used=0 total_mem=0 count=0 percent used_gb total_gb
    while IFS=',' read -r index util used total temp; do
        index="${index// /}"
        util="${util// /}"
        used="${used// /}"
        total="${total// /}"
        temp="${temp// /}"
        [[ -n "$index" ]] || continue
        if [[ "$util" =~ ^[0-9]+$ ]]; then util_display="$util"; else util_display="--"; fi
        if [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$total" -gt 0 ]]; then
            used_gb="$(awk -v value="$used" 'BEGIN { printf "%.1f", value / 1024 }')"
            total_gb="$(awk -v value="$total" 'BEGIN { printf "%.1f", value / 1024 }')"
            percent=$((used * 100 / total))
            used_display="$used_gb"; total_display="$total_gb"; percent_display="$percent"
            total_used=$((total_used + used)); total_mem=$((total_mem + total))
        else
            used_display="?"; total_display="?"; percent_display="?"
        fi
        printf '  GPU %-2s: %3s%% | VRAM: %s GB / %s GB (%s%%) | Temp: %s degC\n' \
            "$index" "$util_display" "$used_display" "$total_display" "$percent_display" "${temp:-?}"
        count=$((count + 1))
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)

    if (( count > 0 && total_mem > 0 )); then
        used_gb="$(awk -v value="$total_used" 'BEGIN { printf "%.1f", value / 1024 }')"
        total_gb="$(awk -v value="$total_mem" 'BEGIN { printf "%.1f", value / 1024 }')"
        percent=$((total_used * 100 / total_mem))
        printf '  TOTAL: VRAM: %s GB / %s GB (%s%%) | GPUs: %s\n' \
            "$used_gb" "$total_gb" "$percent" "$count"
    else
        printf '  TOTAL: GPU metrics unavailable\n'
    fi
}

show_dashboard() {
    local choice ip health
    docker_ready
    if ! container_running; then
        warn "ExLlamaV3 is not running"
        return 1
    fi
    while true; do
        clear 2>/dev/null || true
        ip="$(display_ip)"
        if health="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then :; else health="not responding"; fi

        echo "=================================================================="
        echo "  EXLLAMAV3 SERVER RUNNING"
        echo "=================================================================="
        printf '  Profile:  Qwen3.8-27B EXL3 SC_6.00bpw_H6_V6 / vision / native 262K\n'
        printf '  Model:    %s\n' "$MODEL_REPO"
        printf '  Context:  %s  |  KV: %s  |  Speculation: off\n' "$MAX_CONTEXT" "$CACHE_MODE"
        printf '  Vision:   ON (6-bit quantized vision tower)\n'
        printf '  GPUs:     3x RTX 3090 visible (autosplit; CUDA2 may remain unused)\n'
        printf '  Reasoning: ON\n'
        echo ""
        echo "  Connect from any device on your network:"
        echo ""
        printf '  API Base:      http://%s:%s/v1\n' "$ip" "$PORT"
        printf '  Anthropic:      http://%s:%s/v1/messages\n' "$ip" "$PORT"
        echo ""
        echo "  API Key: any string or blank (not required)"
        echo ""
        printf '  OpenWebUI:      OpenAI base URL → http://%s:%s/v1\n' "$ip" "$PORT"
        printf '  Pi / Codex:     OPENAI_API_BASE=http://%s:%s/v1\n' "$ip" "$PORT"
        printf '  Cline / Continue: OpenAI compatible → http://%s:%s/v1\n' "$ip" "$PORT"
        printf '  Anthropic SDK:   base_url → http://%s:%s/v1\n' "$ip" "$PORT"
        echo "=================================================================="
        echo ""
        printf '  Health: %s\n' "$health"
        printf '  Speed:  %s\n' "$(speed_detail)"
        printf '  CPU: %s%%\n' "$(cpu_percent)"
        gpu_dashboard
        echo ""
        echo "  [1] Stop server and return to menu"
        echo "  [2] Return to menu (keep server running)"
        echo "  [r] Refresh"
        echo ""
        read -r -p "  Select [1/2/r]: " choice
        case "$choice" in
            1)
                stop_server
                say "  ExLlamaV3 stopped."
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
            ensure_assets
            say "ExLlamaV3 model assets are ready."
            ;;
        build)
            docker_ready
            ensure_image
            say "ExLlamaV3 image is ready: $IMAGE"
            ;;
        speed)
            run_speed_test
            ;;
        start)
            start_server
            say "ExLlamaV3 is running. State: $SERVER_INFO"
            if [[ -t 0 && -t 1 ]]; then
                show_dashboard
            fi
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
