#!/usr/bin/env bash

# Qwen3.8 / Qwen3.8-Flash-Next isolated LocalLLM launcher.
# This script deliberately does not touch Hive, watchdog, or miner settings.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A HiveOS login shell may automatically enter `sudo -s`. Keep Qwen3.8's
# models, runtimes, logs, and state in the invoking user's home in that case.
qwen38_home="${HOME}"
if [[ "${EUID}" -eq 0 ]]; then
    qwen38_owner="${QWEN38_OWNER_USER:-${SUDO_USER:-user}}"
    qwen38_resolved_home="$(getent passwd "$qwen38_owner" 2>/dev/null | awk -F: 'NR == 1 {print $6}')"
    if [[ -n "$qwen38_resolved_home" && "$qwen38_resolved_home" != "/root" ]]; then
        qwen38_home="$qwen38_resolved_home"
    elif [[ -d /home/user ]]; then
        qwen38_home="/home/user"
    fi
fi

DATA_ROOT="${QWEN38_DATA_ROOT:-${qwen38_home}/.local/share/localllm-qwen38}"
STATE_ROOT="${QWEN38_STATE_ROOT:-${qwen38_home}/.local/state/locallm-qwen38}"
RUNTIME_ROOT="${DATA_ROOT}/runtimes"
MODEL_ROOT="${DATA_ROOT}/models"
LOG_ROOT="${DATA_ROOT}/logs"

# CUDA packages install nvcc under /usr/local/cuda/bin but Hive's login PATH
# does not always include it. Do not alter the NVIDIA driver; only expose the
# already-installed compiler to this process.
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi

PORT="${QWEN38_PORT:-8080}"
BIND_HOST="${QWEN38_HOST:-0.0.0.0}"
PROFILE="hauhau-q8"
MODE="menu"
PROFILE_EXPLICIT=0
SPEC_OVERRIDE=""
SMOKE=0
SERVER_PID=""
SERVER_LOG=""

# Pinned/provenance data. Re-check PR head before intentionally updating it.
readonly HAUHAU_REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
readonly HAUHAU_COMMIT="4df29be4f4c3673f428170fda944a5b19f743bb8"
readonly HAUHAU_MODEL="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
readonly HAUHAU_MMPROJ="mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
readonly HAUHAU_DRAFT="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
readonly HAUHAU_MODEL_SHA="4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc2"
readonly HAUHAU_MMPROJ_SHA="5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142"
readonly HAUHAU_DRAFT_SHA="115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b"
readonly QWEN4EXP_COMMIT="af1ffaf37f1e44edb62e87ab8ddb9bb6840849bc"
readonly FLASH_REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
readonly FLASH_MMPROJ="mmproj-F16.gguf"
readonly FLASH_MMPROJ_SHA="1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980"

# Set by configure_profile.
PROFILE_LABEL=""
RUNTIME_KIND=""
MODEL_PATH=""
MMPROJ_PATH=""
DRAFT_PATH=""
FULL_CTX=262144
SPEC_MODE="none"
RUNTIME_DIR=""

mkdir -p "$DATA_ROOT" "$STATE_ROOT" "$RUNTIME_ROOT" "$MODEL_ROOT" "$LOG_ROOT"
SPEED_CACHE="${STATE_ROOT}/speed-results.tsv"

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  v1qwen38.sh --quickstart [--profile PROFILE]
  v1qwen38.sh --smoke [--profile PROFILE] [--no-spec|--spec MODE]
  v1qwen38.sh --speed-test [--profile PROFILE]
  v1qwen38.sh --speed-test-all
  v1qwen38.sh --download [--profile PROFILE]
  v1qwen38.sh --build [--profile PROFILE]
  v1qwen38.sh --status [--profile PROFILE]
  v1qwen38.sh --dashboard [--profile PROFILE]
  v1qwen38.sh --stop

Profiles:
  hauhau-q8          HauhauCS Q8_K_P + BF16 vision, native 262K, embedded MTP
  hauhau-q8-fastmtp  HauhauCS Q8_K_P + BF16 vision, FastMTP sidecar, 262K on 3x3090
  flash-iq3          Flash-Next UD-IQ3_XXS + F16 vision, experimental PR #27742
  flash-iq4          Flash-Next UD-IQ4_XS + F16 vision + Q8 KV/auto-fit, experimental PR #27742

--smoke uses a 4096-token context, one short text request, and one small PNG
request. It never sends a long-context prompt. --quickstart uses the profile's
configured native context and leaves the server running.
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
            --speed-test-all)
                MODE="speed-all"
                SMOKE=1
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
            --status)
                MODE="status"
                ;;
            --stop)
                MODE="stop"
                ;;
            --profile)
                [[ $# -ge 2 ]] || die "--profile requires a value"
                PROFILE="$2"
                PROFILE_EXPLICIT=1
                shift
                ;;
            --profile=*)
                PROFILE="${1#*=}"
                PROFILE_EXPLICIT=1
                ;;
            --no-spec)
                SPEC_OVERRIDE="none"
                ;;
            --spec)
                [[ $# -ge 2 ]] || die "--spec requires none, native, fast, or ngram"
                SPEC_OVERRIDE="$2"
                shift
                ;;
            --spec=*)
                SPEC_OVERRIDE="${1#*=}"
                ;;
            --port)
                [[ $# -ge 2 ]] || die "--port requires a value"
                PORT="$2"
                shift
                ;;
            --host)
                [[ $# -ge 2 ]] || die "--host requires a value"
                BIND_HOST="$2"
                shift
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

configure_profile() {
    local qdir
    case "$PROFILE" in
        hauhau-q8)
            PROFILE_LABEL="Qwen3.8-27B HauhauCS Q8_K_P / vision / native 262K"
            RUNTIME_KIND="hauhau"
            RUNTIME_DIR="${RUNTIME_ROOT}/llama-qwen38-hauhau"
            MODEL_PATH="${MODEL_ROOT}/hauhau/${HAUHAU_MODEL}"
            MMPROJ_PATH="${MODEL_ROOT}/hauhau/${HAUHAU_MMPROJ}"
            DRAFT_PATH=""
            FULL_CTX=262144
            SPEC_MODE="native"
            ;;
        hauhau-q8-fastmtp)
            PROFILE_LABEL="Qwen3.8-27B HauhauCS Q8_K_P / vision / FastMTP / 262K"
            RUNTIME_KIND="hauhau"
            RUNTIME_DIR="${RUNTIME_ROOT}/llama-qwen38-hauhau"
            MODEL_PATH="${MODEL_ROOT}/hauhau/${HAUHAU_MODEL}"
            MMPROJ_PATH="${MODEL_ROOT}/hauhau/${HAUHAU_MMPROJ}"
            DRAFT_PATH="${MODEL_ROOT}/hauhau/${HAUHAU_DRAFT}"
            FULL_CTX=262144
            SPEC_MODE="fast"
            ;;
        flash-iq3)
            qdir="UD-IQ3_XXS"
            PROFILE_LABEL="Qwen3.8-Flash-Next ${qdir} / vision / PR #27742"
            RUNTIME_KIND="flash"
            RUNTIME_DIR="${RUNTIME_ROOT}/llama-qwen4exp-pr27742"
            MODEL_PATH="${MODEL_ROOT}/flash/${qdir}/Qwen3.8-Flash-Next-${qdir}-00001-of-00003.gguf"
            MMPROJ_PATH="${MODEL_ROOT}/flash/${FLASH_MMPROJ}"
            DRAFT_PATH=""
            FULL_CTX=262144
            SPEC_MODE="none"
            ;;
        flash-iq4)
            qdir="UD-IQ4_XS"
            PROFILE_LABEL="Qwen3.8-Flash-Next ${qdir} / vision / Q8 KV / auto-fit / PR #27742"
            RUNTIME_KIND="flash"
            RUNTIME_DIR="${RUNTIME_ROOT}/llama-qwen4exp-pr27742"
            MODEL_PATH="${MODEL_ROOT}/flash/${qdir}/Qwen3.8-Flash-Next-${qdir}-00001-of-00003.gguf"
            MMPROJ_PATH="${MODEL_ROOT}/flash/${FLASH_MMPROJ}"
            DRAFT_PATH=""
            FULL_CTX=262144
            SPEC_MODE="none"
            ;;
        *)
            die "unknown profile '$PROFILE'"
            ;;
    esac

    if [[ -n "$SPEC_OVERRIDE" ]]; then
        case "$SPEC_OVERRIDE" in
            none|native|fast|ngram) SPEC_MODE="$SPEC_OVERRIDE" ;;
            *) die "invalid --spec '$SPEC_OVERRIDE'" ;;
        esac
    fi
}

sha256_file() {
    # Keep large-file verification visibly alive. The digest remains the only
    # stdout value so callers can safely use command substitution.
    python3 - "$1" <<'PY'
import hashlib
import os
import sys
import time

path = sys.argv[1]
size = os.path.getsize(path)
show_progress = size >= 64 * 1024 * 1024
chunk_size = 64 * 1024 * 1024
hasher = hashlib.sha256()
read = 0
started = time.monotonic()
last_report = started - 1.0

if show_progress:
    print(f"\rVerifying checksum: {os.path.basename(path)} 0.00%", end="", file=sys.stderr, flush=True)

with open(path, "rb") as handle:
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
        read += len(chunk)
        now = time.monotonic()
        if show_progress and (now - last_report >= 0.5 or read == size):
            elapsed = max(now - started, 0.001)
            rate = read / elapsed / (1024 * 1024)
            percent = 100.0 * read / size if size else 100.0
            eta = (size - read) / (rate * 1024 * 1024) if rate > 0 else 0.0
            print(
                f"\rVerifying checksum: {os.path.basename(path)} "
                f"{percent:6.2f}% {rate:7.1f} MiB/s ETA {eta:5.1f}s",
                end="",
                file=sys.stderr,
                flush=True,
            )
            last_report = now

if show_progress:
    print(file=sys.stderr)
print(hasher.hexdigest())
PY
}

verify_asset() {
    local file="$1" expected="$2" actual
    [[ -f "$file" ]] || return 1
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
    local url="$1" dest="$2" expected="${3:-}" dir tmp
    dir="$(dirname -- "$dest")"
    mkdir -p "$dir"

    if [[ -f "$dest" ]] && verify_asset "$dest" "$expected"; then
        say "Already present: $dest"
        return 0
    fi
    if [[ -f "$dest" ]]; then
        rm -f -- "$dest"
    fi

    tmp="${dest}.part"
    say "Downloading: $(basename -- "$dest")"
    say "  URL: $url"
    wget --continue --tries=10 --waitretry=5 --timeout=60 \
        --progress=dot:giga -O "$tmp" "$url"
    mv -f -- "$tmp" "$dest"

    if ! verify_asset "$dest" "$expected"; then
        rm -f -- "$dest"
        die "checksum verification failed after download: $dest"
    fi
    say "Verified: $dest"
}

ensure_hauhau_assets() {
    local dir="${MODEL_ROOT}/hauhau"
    local base="https://huggingface.co/${HAUHAU_REPO}/resolve/main"
    download_file "$base/$HAUHAU_MODEL" "$dir/$HAUHAU_MODEL" "$HAUHAU_MODEL_SHA"
    download_file "$base/$HAUHAU_MMPROJ" "$dir/$HAUHAU_MMPROJ" "$HAUHAU_MMPROJ_SHA"
    # Keep the publisher manifest/checksum list beside the downloaded files.
    download_file "$base/SHA256SUMS" "$dir/SHA256SUMS"
    download_file "$base/HauhauCS-RELEASE-MANIFEST.json" "$dir/HauhauCS-RELEASE-MANIFEST.json"

    if [[ "$SPEC_MODE" == "fast" ]]; then
        download_file "$base/$HAUHAU_DRAFT" "$dir/$HAUHAU_DRAFT" "$HAUHAU_DRAFT_SHA"
    fi
}

flash_quant_files() {
    local quant="$1"
    case "$quant" in
        UD-IQ3_XXS)
            printf '%s\n' \
                "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf|268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8" \
                "Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf|cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f" \
                "Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf|f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250"
            ;;
        UD-IQ4_XS)
            printf '%s\n' \
                "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf|5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4" \
                "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf|577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7" \
                "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf|d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"
            ;;
        *) die "unsupported Flash-Next quant '$quant'" ;;
    esac
}

ensure_flash_assets() {
    local quant="$1" line file expected dir
    local base
    dir="${MODEL_ROOT}/flash/${quant}"
    base="https://huggingface.co/${FLASH_REPO}/resolve/main/${quant}"
    mkdir -p "$dir"

    while IFS='|' read -r file expected; do
        [[ -n "$file" ]] || continue
        download_file "$base/$file" "$dir/$file" "$expected"
    done < <(flash_quant_files "$quant")

    download_file \
        "https://huggingface.co/${FLASH_REPO}/resolve/main/${FLASH_MMPROJ}" \
        "${MODEL_ROOT}/flash/${FLASH_MMPROJ}" "$FLASH_MMPROJ_SHA"
}

ensure_assets() {
    say "Checking model assets for ${PROFILE} (checksum progress will be shown)..."
    case "$RUNTIME_KIND" in
        hauhau) ensure_hauhau_assets ;;
        flash)
            if [[ "$PROFILE" == "flash-iq3" ]]; then
                ensure_flash_assets UD-IQ3_XXS
            else
                ensure_flash_assets UD-IQ4_XS
            fi
            ;;
        *) die "internal error: unknown runtime kind '$RUNTIME_KIND'" ;;
    esac
    say "Model assets ready for ${PROFILE}."
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "missing command '$1'. Install build dependencies first."
}

ensure_build_deps() {
    need_command git
    need_command cmake
    need_command gcc
    need_command g++
    need_command nvcc
}

build_runtime() {
    local source="$RUNTIME_DIR/source" build_jobs patch_file
    build_jobs="${QWEN38_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
    [[ "$build_jobs" =~ ^[0-9]+$ ]] || build_jobs=4
    (( build_jobs > 12 )) && build_jobs=12

    if [[ -x "$RUNTIME_DIR/source/build/bin/llama-server" ]]; then
        say "Runtime already built: $RUNTIME_DIR/source/build/bin/llama-server"
        return 0
    fi

    ensure_build_deps
    mkdir -p "$RUNTIME_DIR"

    if [[ ! -d "$source/.git" ]]; then
        say "Cloning isolated runtime source into $source"
        git clone --no-tags https://github.com/ggml-org/llama.cpp.git "$source"
    fi

    pushd "$source" >/dev/null
    if [[ "$RUNTIME_KIND" == "hauhau" ]]; then
        git fetch --quiet origin "$HAUHAU_COMMIT"
        git reset --hard --quiet "$HAUHAU_COMMIT"
        git clean -fdx >/dev/null
        patch_file="$source/HauhauCS-FastMTP-llama.cpp.patch"
        download_file \
            "https://huggingface.co/${HAUHAU_REPO}/resolve/main/HauhauCS-FastMTP-llama.cpp.patch" \
            "$patch_file"
        git apply --check "$patch_file"
        git apply "$patch_file"
        say "Applied HauhauCS FastMTP patch to pinned qwen35 runtime"
    else
        git fetch --quiet origin "pull/27742/head:refs/remotes/origin/pr-27742"
        git reset --hard --quiet "$QWEN4EXP_COMMIT"
        git clean -fdx >/dev/null
        say "Using qwen4exp PR #27742 commit $QWEN4EXP_COMMIT"
    fi

    cmake -S . -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        2>&1 | tee "${LOG_ROOT}/build-${RUNTIME_KIND}.log"
    cmake --build build --config Release --target llama-server -j"$build_jobs" \
        2>&1 | tee -a "${LOG_ROOT}/build-${RUNTIME_KIND}.log"
    popd >/dev/null

    [[ -x "$RUNTIME_DIR/source/build/bin/llama-server" ]] || die "llama-server build did not produce an executable"
    say "Built: $RUNTIME_DIR/source/build/bin/llama-server"
}

write_server_info() {
    # State is shared with the original HiveOS login user when this launcher
    # is started from the automatic root shell; it contains no secrets.
    umask 022
    cat > "${STATE_ROOT}/server.info" <<EOF
profile=$PROFILE
label=$PROFILE_LABEL
pid=$SERVER_PID
port=$PORT
host=$BIND_HOST
model=$MODEL_PATH
mmproj=$MMPROJ_PATH
context=$FULL_CTX
speculation=$SPEC_MODE
log=$SERVER_LOG
EOF
    printf '%s\n' "$SERVER_PID" > "${STATE_ROOT}/server.pid"
}

running_pid() {
    local pid
    [[ -s "${STATE_ROOT}/server.pid" ]] || return 1
    pid="$(cat "${STATE_ROOT}/server.pid" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    # kill -0 fails when the server belongs to the other shared-shell user;
    # /proc plus the command line still lets us reject stale/reused PIDs.
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "$cmdline" == *llama-server* ]] || return 1
    printf '%s\n' "$pid"
}

pid_exists() {
    [[ -r "/proc/$1/cmdline" ]]
}

signal_pid() {
    local signal="$1" pid="$2"
    if kill "-$signal" "$pid" 2>/dev/null; then
        return 0
    fi
    if [[ "${EUID}" -ne 0 ]]; then
        sudo kill "-$signal" "$pid" 2>/dev/null
    else
        return 1
    fi
}

stop_server() {
    local pid=""
    if pid="$(running_pid)"; then
        say "Stopping Qwen3.8 server PID $pid"
        signal_pid TERM "$pid" || true
        for _ in $(seq 1 20); do
            pid_exists "$pid" || break
            sleep 1
        done
        if pid_exists "$pid"; then
            warn "server did not stop gracefully; sending SIGKILL to PID $pid"
            signal_pid KILL "$pid" || true
        fi
    fi
    rm -f "${STATE_ROOT}/server.pid" "${STATE_ROOT}/server.info"
}

check_port_free() {
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
        die "TCP port $PORT is already in use; stop the existing engine first"
    fi
}

make_server_args() {
    local ctx="$FULL_CTX" batch=512 ubatch=128
    if (( SMOKE )); then
        ctx=4096
        batch=256
        ubatch=64
    fi

    SERVER_ARGS=(
        --model "$MODEL_PATH"
        --mmproj "$MMPROJ_PATH"
        --ctx-size "$ctx"
        --n-gpu-layers "$([[ "$RUNTIME_KIND" == "flash" ]] && echo auto || echo all)"
        --split-mode layer
        --flash-attn on
        --batch-size "$batch"
        --ubatch-size "$ubatch"
        --parallel 1
        --jinja
        --temp 1.0
        --top-k 20
        --top-p 0.95
        --min-p 0
        --presence-penalty 0
        --repeat-penalty 1.0
        --image-min-tokens 1024
        --host "$BIND_HOST"
        --port "$PORT"
    )

    if [[ "$PROFILE" == "flash-iq4" ]]; then
        # The larger IQ4 model needs quantized KV plus llama.cpp's automatic
        # layer fitting to leave enough room for native 262K context on 3x24GB.
        SERVER_ARGS+=(--cache-type-k q8_0 --cache-type-v q8_0)
    else
        SERVER_ARGS+=(--tensor-split 1,1,1 --cache-type-k f16 --cache-type-v f16)
    fi

    if (( SMOKE )); then
        # Keep the verification request short and deterministic enough to finish quickly.
        SERVER_ARGS+=(--reasoning off)
    else
        SERVER_ARGS+=(
            --reasoning on
            --reasoning-effort xhigh
            --reasoning-preserve
            --reasoning-format deepseek
        )
    fi

    # The dense model card uses no-mmap for its FastMTP reference. Flash-Next
    # must remain mmap-backed because its PLE/n-gram tables are much larger.
    # Use the non-deprecated spelling supported by both pinned runtimes.
    if [[ "$SPEC_MODE" == "fast" ]]; then
        SERVER_ARGS+=(--load-mode none)
    else
        SERVER_ARGS+=(--load-mode mmap)
    fi

    if [[ "${QWEN38_CPU_MMPROJ:-0}" == "1" ]]; then
        SERVER_ARGS+=(--no-mmproj-offload)
    fi

    case "$SPEC_MODE" in
        none)
            ;;
        native)
            SERVER_ARGS+=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0)
            ;;
        fast)
            [[ -n "$DRAFT_PATH" ]] || die "FastMTP profile has no draft path"
            SERVER_ARGS+=(
                --spec-draft-model "$DRAFT_PATH"
                --spec-draft-ngl all
                --spec-type draft-mtp
                --spec-draft-n-max 3
                --spec-draft-p-min 0
            )
            ;;
        ngram)
            SERVER_ARGS+=(--spec-type ngram-mod)
            ;;
        *) die "internal error: unsupported speculation mode '$SPEC_MODE'" ;;
    esac
}

wait_for_health() {
    local timeout="${QWEN38_HEALTH_TIMEOUT:-300}" response elapsed=0
    say "Waiting for server health (up to ${timeout}s; no long-context request will be sent)"
    for _ in $(seq 1 "$timeout"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            warn "server exited before becoming healthy"
            tail -80 "$SERVER_LOG" >&2 || true
            return 1
        fi
        if response="$(curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
            say "Health OK: $response"
            return 0
        fi
        elapsed=$((elapsed + 1))
        if (( elapsed % 10 == 0 )); then
            say "  Still loading... ${elapsed}s elapsed (model log: $(basename -- "$SERVER_LOG"))"
        fi
        sleep 1
    done
    warn "server did not become healthy"
    tail -100 "$SERVER_LOG" >&2 || true
    return 1
}

start_server() {
    local bin="$RUNTIME_DIR/source/build/bin/llama-server"
    [[ -x "$bin" ]] || die "runtime is not built: $bin"
    [[ -f "$MODEL_PATH" ]] || die "model is missing: $MODEL_PATH"
    [[ -f "$MMPROJ_PATH" ]] || die "vision projector is missing: $MMPROJ_PATH"
    if [[ "$SPEC_MODE" == "fast" && ! -f "$DRAFT_PATH" ]]; then
        die "FastMTP sidecar is missing: $DRAFT_PATH"
    fi

    if SERVER_PID="$(running_pid)"; then
        say "Qwen3.8 server is already running (PID $SERVER_PID)"
        return 0
    fi
    rm -f "${STATE_ROOT}/server.pid" "${STATE_ROOT}/server.info"
    check_port_free
    make_server_args

    SERVER_LOG="${LOG_ROOT}/${PROFILE}-$(date +%Y%m%d-%H%M%S).log"
    say "Starting: $PROFILE_LABEL"
    say "  context: $FULL_CTX (smoke context: $([[ $SMOKE -eq 1 ]] && echo 4096 || echo no))"
    if [[ "$PROFILE" == "flash-iq4" ]]; then
        say "  GPUs: 3x RTX 3090, layer split auto-fit, Q8 KV"
    else
        say "  GPUs: 3x RTX 3090, layer split 1,1,1, F16 KV"
    fi
    say "  log: $SERVER_LOG"
    printf 'Command:' > "$SERVER_LOG"
    printf ' %q' "$bin" "${SERVER_ARGS[@]}" >> "$SERVER_LOG"
    printf '\n\n' >> "$SERVER_LOG"
    nohup "$bin" "${SERVER_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    write_server_info
    if ! wait_for_health; then
        stop_server
        return 1
    fi
}

response_text() {
    local file="$1"
    python3 - "$file" <<'PY'
import json, sys
p = sys.argv[1]
try:
    data = json.load(open(p, encoding='utf-8'))
except Exception as exc:
    print(f"invalid JSON response: {exc}", file=sys.stderr)
    sys.exit(1)
if data.get("error"):
    print(json.dumps(data["error"], ensure_ascii=False), file=sys.stderr)
    sys.exit(1)
try:
    message = data["choices"][0]["message"]
except Exception:
    print("response has no choices[0].message", file=sys.stderr)
    sys.exit(1)
value = message.get("content") or message.get("reasoning_content") or ""
if isinstance(value, list):
    value = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in value)
value = str(value).strip()
if not value:
    print("response content is empty", file=sys.stderr)
    sys.exit(1)
print(value)
PY
}

run_text_smoke() {
    local out="${STATE_ROOT}/smoke-text.json"
    say "Smoke 1/2: short text request (max 32 tokens)"
    if ! curl -fsS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"qwen38","messages":[{"role":"user","content":"Reply with a short confirmation that Qwen3.8 is ready."}],"max_tokens":32,"temperature":0.2}' \
        -o "$out"; then
        warn "text request failed"
        return 1
    fi
    response_text "$out" | head -c 500
    printf '\n'
}

make_test_png() {
    local path="$1"
    python3 - "$path" <<'PY'
import struct, sys, zlib
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
import json, sys
b64, path = sys.argv[1:]
body = {
  "model": "qwen38",
  "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
      {"type": "text", "text": "In one short sentence, which color is on the left side of the image?"}
  ]}],
  "max_tokens": 32,
  "temperature": 0.2
}
open(path, "w", encoding="utf-8").write(json.dumps(body))
PY
    say "Smoke 2/2: one small 64x64 red/blue image request (max 32 tokens)"
    if ! curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload" -o "$out"; then
        warn "vision request failed"
        return 1
    fi
    response_text "$out" | head -c 500
    printf '\n'
}

run_smoke() {
    local rc=0
    start_server || return 1
    run_text_smoke || rc=1
    run_vision_smoke || rc=1
    say "GPU snapshot after smoke:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || true
    stop_server
    if (( rc == 0 )); then
        say "SHORT SMOKE PASSED: text + vision"
    else
        warn "SHORT SMOKE FAILED; server log retained at $SERVER_LOG"
    fi
    return "$rc"
}

speed_cache_row() {
    local wanted="$1"
    [[ -r "$SPEED_CACHE" ]] || return 1
    awk -F'|' -v wanted="$wanted" '$1 == wanted { row = $0 } END { if (row != "") print row }' "$SPEED_CACHE"
}

speed_display() {
    local row date context coding story average
    row="$(speed_cache_row "$1" || true)"
    if [[ -z "$row" ]]; then
        printf 'not tested'
        return 0
    fi
    IFS='|' read -r _ date context coding story average <<< "$row"
    if [[ "$date" == "$(date +%Y-%m-%d)" ]]; then
        printf '%s tok/s' "$average"
    else
        printf '%s tok/s (%s)' "$average" "$date"
    fi
}

speed_detail() {
    local row date context coding story average
    row="$(speed_cache_row "$1" || true)"
    if [[ -z "$row" ]]; then
        printf 'not tested'
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
        awk -F'|' -v profile="$PROFILE" '$1 != profile' "$SPEED_CACHE" > "$tmp"
    else
        : > "$tmp"
    fi
    printf '%s|%s|4096|%s|%s|%s\n' \
        "$PROFILE" "$(date +%Y-%m-%d)" "$coding" "$story" "$average" >> "$tmp"
    mv -f -- "$tmp" "$SPEED_CACHE"
}

speed_request() {
    local kind="$1" prompt="$2" max_tokens="$3" payload out
    payload="${STATE_ROOT}/speed-${PROFILE}-${kind}.request.json"
    out="${STATE_ROOT}/speed-${PROFILE}-${kind}.response.json"
    python3 - "$prompt" "$max_tokens" "$payload" <<'PY'
import json
import sys

prompt, max_tokens, path = sys.argv[1:]
body = {
    "model": "qwen38",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": 0.2,
    "stream": False,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(body, handle)
PY
    if ! curl -fsS --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$payload" -o "$out"; then
        warn "${kind} speed request failed"
        return 1
    fi
    python3 - "$out" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
if data.get("error"):
    raise SystemExit(json.dumps(data["error"], ensure_ascii=False))
timings = data.get("timings", {})
tps = timings.get("predicted_per_second")
if tps is None:
    tokens = timings.get("predicted_n", 0)
    millis = timings.get("predicted_ms", 0)
    tps = tokens * 1000.0 / millis if millis else 0.0
if not tps or tps <= 0:
    raise SystemExit("response did not include a valid generation speed")
print(f"{tps:.2f}")
PY
}

run_speed_test() {
    local coding_tps story_tps average warmup_prompt
    if SERVER_PID="$(running_pid)"; then
        warn "stop the running Qwen3.8 server before starting a speed test (PID $SERVER_PID)"
        return 1
    fi

    SMOKE=1
    say "Speed test: ${PROFILE} (4096-token context; two short prompts)"
    ensure_assets
    build_runtime
    start_server || return 1

    warmup_prompt='Reply with one word: ready.'
    say "  Warm-up request..."
    if ! speed_request warmup "$warmup_prompt" 16 >/dev/null; then
        stop_server
        return 1
    fi

    say "  Coding prompt..."
    if ! coding_tps="$(speed_request coding 'Write a concise Python function that merges overlapping intervals. Include type hints, one example, and a brief time-complexity note. Keep the answer under 120 words.' 192)"; then
        stop_server
        return 1
    fi
    say "    coding: ${coding_tps} tok/s"

    say "  Story prompt..."
    if ! story_tps="$(speed_request story 'Write a short story in around 100 words about a night-shift engineer who receives a radio message from tomorrow. Give it a clear ending.' 192)"; then
        stop_server
        return 1
    fi
    say "    story: ${story_tps} tok/s"

    average="$(python3 - "$coding_tps" "$story_tps" <<'PY'
import sys
values = [float(value) for value in sys.argv[1:]]
print(f"{sum(values) / len(values):.2f}")
PY
)"
    record_speed_result "$coding_tps" "$story_tps" "$average"
    stop_server
    say "  Speed result: avg ${average} tok/s (cached for the menu)"
}

run_speed_test_all() {
    local original_profile="$PROFILE" original_smoke="$SMOKE" rc=0 profile
    for profile in hauhau-q8 hauhau-q8-fastmtp flash-iq3 flash-iq4; do
        PROFILE="$profile"
        SPEC_OVERRIDE=""
        SMOKE=1
        configure_profile
        if ! run_speed_test; then
            warn "speed test failed for ${PROFILE}"
            rc=1
        fi
    done
    PROFILE="$original_profile"
    SMOKE="$original_smoke"
    configure_profile
    return "$rc"
}

show_status() {
    configure_profile
    say "Profile: $PROFILE_LABEL"
    say "Data root: $DATA_ROOT"
    say "Runtime: $RUNTIME_DIR/source/build/bin/llama-server"
    say "Model: $MODEL_PATH"
    say "Projector: $MMPROJ_PATH"
    [[ -n "$DRAFT_PATH" ]] && say "Draft: $DRAFT_PATH"
    for f in "$MODEL_PATH" "$MMPROJ_PATH" "$DRAFT_PATH" "$RUNTIME_DIR/source/build/bin/llama-server"; do
        [[ -n "$f" && -e "$f" ]] && ls -lh "$f"
    done
    if SERVER_PID="$(running_pid)"; then
        say "Server: running PID $SERVER_PID"
        [[ -f "${STATE_ROOT}/server.info" ]] && cat "${STATE_ROOT}/server.info"
    else
        say "Server: stopped"
    fi
    nvidia-smi --query-gpu=index,name,compute_cap,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || true
}

load_server_profile() {
    local saved_profile=""
    if [[ -r "${STATE_ROOT}/server.info" ]]; then
        saved_profile="$(awk -F= '$1 == "profile" { print substr($0, index($0, "=") + 1); exit }' "${STATE_ROOT}/server.info")"
    fi
    if [[ "$saved_profile" =~ ^[a-z0-9-]+$ ]]; then
        PROFILE="$saved_profile"
    fi
}

display_ip() {
    local ip="${QWEN38_DISPLAY_IP:-}"
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
        if [[ "$util" =~ ^[0-9]+$ ]]; then
            util_display="$util"
        else
            util_display="--"
        fi
        if [[ "$used" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ && "$total" -gt 0 ]]; then
            used_gb="$(awk -v value="$used" 'BEGIN { printf "%.1f", value / 1024 }')"
            total_gb="$(awk -v value="$total" 'BEGIN { printf "%.1f", value / 1024 }')"
            percent=$((used * 100 / total))
            used_display="$used_gb"
            total_display="$total_gb"
            percent_display="$percent"
            total_used=$((total_used + used))
            total_mem=$((total_mem + total))
        else
            used_display="?"
            total_display="?"
            percent_display="?"
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
    local choice ip health spec_label
    load_server_profile
    configure_profile
    if ! SERVER_PID="$(running_pid)"; then
        warn "Qwen3.8 server is not running"
        return 1
    fi

    while true; do
        clear 2>/dev/null || true
        ip="$(display_ip)"
        if health="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
            :
        else
            health="not responding"
        fi
        case "$SPEC_MODE" in
            native) spec_label="native MTP" ;;
            fast) spec_label="FastMTP (3-token draft)" ;;
            ngram) spec_label="n-gram speculation" ;;
            *) spec_label="off" ;;
        esac

        echo "=================================================================="
        echo "  QWEN3.8 SERVER RUNNING"
        echo "=================================================================="
        printf '  Profile:  %s\n' "$PROFILE_LABEL"
        printf '  Model:    %s\n' "$(basename -- "$MODEL_PATH")"
        if [[ "$PROFILE" == "flash-iq4" ]]; then
            printf '  Context:  %s  |  KV: Q8  |  Speculation: %s\n' "$FULL_CTX" "$spec_label"
        else
            printf '  Context:  %s  |  KV: F16  |  Speculation: %s\n' "$FULL_CTX" "$spec_label"
        fi
        if [[ "$RUNTIME_KIND" == "hauhau" ]]; then
            printf '  Vision:   ON (BF16 projector)\n'
        else
            printf '  Vision:   ON (F16 projector)\n'
        fi
        echo "  GPUs:     3x RTX 3090 (24 GB each)"
        printf '  Reasoning: ON\n'
        echo ""
        echo "  Connect from any device on your network:"
        echo ""
        printf '  Chat UI:       http://%s:%s\n' "$ip" "$PORT"
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
        printf '  Speed:  %s\n' "$(speed_detail "$PROFILE")"
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
                echo "  Server stopped."
                sleep 1
                return 0
                ;;
            2) return 0 ;;
            r|R) ;;
            *) ;;
        esac
    done
}

choose_profile() {
    say ""
    say "Qwen3.8 Quick Start"
    say "  [1] HauhauCS Q8_K_P + BF16 vision + native MTP + 262K  |  speed: $(speed_display hauhau-q8)"
    say "  [2] HauhauCS Q8_K_P + BF16 vision + FastMTP + 262K (3x3090 profile)  |  speed: $(speed_display hauhau-q8-fastmtp)"
    say "  [3] Flash-Next UD-IQ3_XXS + F16 vision + PR #27742 (experimental)  |  speed: $(speed_display flash-iq3)"
    say "  [4] Flash-Next UD-IQ4_XS + F16 vision + Q8 KV/auto-fit + PR #27742 (experimental, larger)  |  speed: $(speed_display flash-iq4)"
    say "  [s] Run short speed tests for all profiles"
    say "  [q] Cancel"
    read -r -p "Select: " choice
    case "$choice" in
        1) PROFILE="hauhau-q8" ;;
        2) PROFILE="hauhau-q8-fastmtp" ;;
        3) PROFILE="flash-iq3" ;;
        4) PROFILE="flash-iq4" ;;
        s|S)
            PROFILE="hauhau-q8"
            MODE="speed-all"
            SMOKE=1
            ;;
        *) say "Cancelled."; exit 0 ;;
    esac
}

main() {
    parse_args "$@"
    if [[ "$MODE" == "stop" ]]; then
        stop_server
        exit 0
    fi
    if [[ "$MODE" == "status" ]]; then
        configure_profile
        show_status
        exit 0
    fi
    if [[ "$MODE" == "dashboard" ]]; then
        load_server_profile
        configure_profile
        show_dashboard
        exit $?
    fi
    if [[ "$MODE" == "menu" || ( "$MODE" == "start" && "$PROFILE_EXPLICIT" -eq 0 ) || ( "$MODE" == "speed" && "$PROFILE_EXPLICIT" -eq 0 ) ]]; then
        choose_profile
    fi
    configure_profile

    case "$MODE" in
        download)
            ensure_assets
            ;;
        build)
            build_runtime
            ;;
        start)
            ensure_assets
            build_runtime
            start_server
            say "Server is running. State: ${STATE_ROOT}/server.info"
            if [[ -t 0 && -t 1 ]]; then
                show_dashboard
            fi
            ;;
        smoke)
            ensure_assets
            build_runtime
            run_smoke
            ;;
        speed)
            run_speed_test
            ;;
        speed-all)
            run_speed_test_all
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"
