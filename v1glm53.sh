#!/usr/bin/env bash

# GLM-5.3-Flash local launcher.
# Uses the retained EXL3 2.05bpw model through TabbyAPI/ExLlamaV3.
# This is intentionally separate from the llama.cpp/Qwen launcher.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the state and model environment attached to the normal user when HiveOS
# invokes HostLLM from a root shell.
glm_home="${HOME}"
if [[ "${EUID}" -eq 0 ]]; then
    glm_owner="${GLM53_OWNER_USER:-${SUDO_USER:-user}}"
    glm_resolved_home="$(getent passwd "$glm_owner" 2>/dev/null | awk -F: 'NR == 1 {print $6}')"
    if [[ -n "$glm_resolved_home" && "$glm_resolved_home" != "/root" ]]; then
        glm_home="$glm_resolved_home"
    elif [[ -d /home/user ]]; then
        glm_home="/home/user"
    fi
fi

GLM53_ROOT="${GLM53_ROOT:-${glm_home}/glm53-tabby}"
TABBY_ROOT="${GLM53_TABBY_ROOT:-${GLM53_ROOT}/tabbyAPI}"
PYTHON="${GLM53_PYTHON:-${glm_home}/glm205-exllama-test/venv2/bin/python}"
CONFIG="${GLM53_CONFIG:-${SCRIPT_DIR}/glm53-tabby-config.yml}"
MODEL_DIR="${GLM53_MODEL_DIR:-${glm_home}/GLM-5.3-Flash-exl3-2.05bpw}"
STATE_ROOT="${GLM53_STATE_ROOT:-${glm_home}/.local/state/locallm-glm53}"
LOG_ROOT="${GLM53_LOG_ROOT:-${GLM53_ROOT}/logs}"
PORT="${GLM53_PORT:-8080}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
PID_FILE="${STATE_ROOT}/server.pid"
INFO_FILE="${STATE_ROOT}/server.info"
LOG_FILE="${LOG_ROOT}/server.log"
MAIN="${TABBY_ROOT}/main.py"

mkdir -p "$STATE_ROOT" "$LOG_ROOT"
if [[ "${EUID}" -eq 0 && -n "${glm_owner:-}" && "$glm_owner" != root ]]; then
    glm_group="$(id -gn "$glm_owner" 2>/dev/null || printf '%s' "$glm_owner")"
    chown "$glm_owner:$glm_group" "$STATE_ROOT" "$LOG_ROOT" 2>/dev/null || true
fi

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

pid_is_glm() {
    local pid="${1:-}"
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]] || return 1
    # Match a complete argv entry, not an arbitrary shell command containing
    # the path. This avoids killing HostLLM's own shell during [9].
    tr '\0' '\n' < "/proc/${pid}/cmdline" 2>/dev/null |
        grep -Fqx -- "${TABBY_ROOT}/main.py"
}

read_pid() {
    local pid=""
    [[ -r "$PID_FILE" ]] && pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] && printf '%s' "$pid"
}

all_glm_pids() {
    local proc pid
    for proc in /proc/[0-9]*; do
        pid="${proc##*/}"
        if pid_is_glm "$pid"; then
            printf '%s\n' "$pid"
        fi
    done
}

glm_pid() {
    local pid="$(read_pid)"
    if pid_is_glm "$pid"; then
        printf '%s' "$pid"
        return 0
    fi
    # Also discover orphaned/manual TabbyAPI processes so status and [9]
    # still work when the launcher PID file is stale or missing.
    while read -r pid; do
        if [[ -n "$pid" ]]; then
            printf '%s' "$pid"
            return 0
        fi
    done < <(all_glm_pids)
    return 1
}

send_glm_signal() {
    local signal="$1" pid="$2"
    kill "-${signal}" "$pid" 2>/dev/null || \
        sudo -n kill "-${signal}" "$pid" 2>/dev/null || true
}

health_ok() {
    curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null | grep -q '"status":"healthy"'
}

write_info() {
    cat > "$INFO_FILE" <<EOF_INFO
GLM-5.3-Flash EXL3 2.05bpw | TabbyAPI/ExLlamaV3 | port ${PORT} | 262K configured context | vision enabled (vision tower offloaded to system RAM)
EOF_INFO
}

check_install() {
    [[ -x "$PYTHON" ]] || die "GLM Python runtime not found: $PYTHON"
    [[ -f "$MAIN" ]] || die "TabbyAPI not found: $MAIN"
    [[ -f "$CONFIG" ]] || die "GLM config not found: $CONFIG"
    [[ -d "$MODEL_DIR" ]] || die "GLM model not found: $MODEL_DIR"
}

status() {
    local pid=""
    if pid="$(glm_pid)"; then
        if health_ok; then
            say "GLM-5.3-Flash server RUNNING (PID $pid)"
        else
            say "GLM-5.3-Flash server LOADING (PID $pid)"
        fi
        say "  $(cat "$INFO_FILE" 2>/dev/null || true)"
        say "  Log: $LOG_FILE"
        return 0
    fi
    say "GLM-5.3-Flash server stopped"
    return 1
}

stop_server() {
    local pid="" deadline=$((SECONDS + 45)) remaining=0
    local -a pids=()
    local seen=" "

    # Stop every TabbyAPI process for this GLM installation, including
    # manually-started/orphaned servers that do not have our PID file.
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        [[ "$seen" == *" $pid "* ]] && continue
        pids+=("$pid")
        seen+="$pid "
    done < <(all_glm_pids)

    for pid in "${pids[@]}"; do
        say "Stopping GLM-5.3-Flash server PID $pid"
        send_glm_signal TERM "$pid"
    done

    while (( SECONDS < deadline )); do
        remaining=0
        for pid in "${pids[@]}"; do
            if pid_is_glm "$pid"; then
                remaining=1
                break
            fi
        done
        if (( remaining == 0 )); then
            break
        fi
        sleep 1
    done

    for pid in "${pids[@]}"; do
        if pid_is_glm "$pid"; then
            warn "GLM server PID $pid did not stop cleanly; sending SIGKILL"
            send_glm_signal KILL "$pid"
        fi
    done
    rm -f "$PID_FILE" "$INFO_FILE"
}

start_server() {
    local pid="" deadline=$((SECONDS + 900))
    check_install

    if pid="$(glm_pid)"; then
        if health_ok; then
            write_info
            say "GLM-5.3-Flash is already running (PID $pid)."
            return 0
        fi
        say "GLM-5.3-Flash is already loading (PID $pid); waiting for health..."
    else
        rm -f "$PID_FILE" "$INFO_FILE"
        say "Starting GLM-5.3-Flash 2.05bpw with vision and 262K context..."
        (
            cd "$TABBY_ROOT"
            exec env PYTHONUNBUFFERED=1 "$PYTHON" "$MAIN" --config "$CONFIG"
        ) >>"$LOG_FILE" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$PID_FILE"
    fi

    while (( SECONDS < deadline )); do
        if health_ok; then
            write_info
            say "GLM-5.3-Flash is healthy (PID $pid)."
            say "  $(cat "$INFO_FILE")"
            return 0
        fi
        if ! pid_is_glm "$pid"; then
            warn "GLM server exited during startup. Last log lines:"
            tail -30 "$LOG_FILE" 2>/dev/null || true
            rm -f "$PID_FILE"
            return 1
        fi
        sleep 2
    done
    warn "Timed out waiting for GLM health. It may still be loading; see $LOG_FILE"
    return 1
}

case "${1:---status}" in
    --quickstart|--start)
        start_server
        ;;
    --stop)
        stop_server
        ;;
    --status)
        status || true
        ;;
    --no-dashboard)
        start_server
        ;;
    *)
        die "Usage: $0 --quickstart|--start|--stop|--status"
        ;;
esac
