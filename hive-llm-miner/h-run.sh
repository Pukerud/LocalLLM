#!/usr/bin/env bash

# HiveOS runs this file from its miner screen. It intentionally starts the
# direct Qwen launcher rather than HostLLM.sh: osn.service stays running so
# OctaSpace can issue its normal miner stop/start handoff.
set -Eeuo pipefail

LLM_ROOT="${LLM_ROOT:-/home/user/LocalLLM}"
LAUNCHER="${LLM_ROOT}/v1qwen38.sh"
# Hive starts custom miners as root. Pin both roots explicitly so the launcher
# cannot resolve root's HOME to the unrelated /home/octa service account.
export HOME="${QWEN38_HOME:-/home/user}"
export QWEN38_OWNER_USER="${QWEN38_OWNER_USER:-user}"
export QWEN38_DATA_ROOT="${QWEN38_DATA_ROOT:-/home/user/.local/share/localllm-qwen38}"
export QWEN38_STATE_ROOT="${QWEN38_STATE_ROOT:-/home/user/.local/state/locallm-qwen38}"
STATE_ROOT="$QWEN38_STATE_ROOT"
PROFILE="${QWEN38_HIVE_PROFILE:-hauhau-q8-fastmtp}"
PORT="${QWEN38_PORT:-8080}"
STOP_FILE="${MINER_STOP:-/run/hive/MINER_STOP}"
PID_FILE="${STATE_ROOT}/server.pid"

say() { printf '[llm-hosting] %s\n' "$*"; }

stop_llm() {
    [[ -x "$LAUNCHER" ]] || return 0
    "$LAUNCHER" --stop >/dev/null 2>&1 || true
}

on_signal() {
    stop_llm
    exit 0
}

trap on_signal INT TERM HUP
trap stop_llm EXIT

[[ -x "$LAUNCHER" ]] || { say "missing launcher: $LAUNCHER"; exit 1; }

# A running non-HostLLM Docker container is treated as a possible OctaSpace
# rental. Fail closed before using any GPU, including if Docker cannot be
# queried. HostLLM-owned containers are excluded because they are handled by
# HostLLM's own engine arbitration.
docker_workload_info() {
    local lines name image status
    command -v docker >/dev/null 2>&1 || return 2
    if ! lines="$(docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null)"; then
        return 2
    fi
    while IFS=$'\t' read -r name image status; do
        [[ -n "$name" ]] || continue
        case "$name" in
            vllm-speed-demon|vllm-hostllm) continue ;;
        esac
        printf '%s (%s; %s)\n' "$name" "$image" "$status"
        return 0
    done <<< "$lines"
    return 1
}

if workload="$(docker_workload_info)"; then
    say "refusing to start: possible OctaSpace Docker workload is running: $workload"
    mkdir -p "$(dirname -- "$STOP_FILE")"
    printf '1\n' > "$STOP_FILE"
    exit 1
else
    workload_rc=$?
    if [[ "$workload_rc" -eq 2 ]]; then
        say "refusing to start: Docker workload status is unavailable"
        mkdir -p "$(dirname -- "$STOP_FILE")"
        printf '1\n' > "$STOP_FILE"
        exit 1
    fi
fi

say "starting $PROFILE (osn.service is left running)"
QWEN38_SKIP_EXISTING_VERIFY=1 "$LAUNCHER" --quickstart --profile "$PROFILE" --no-dashboard

# v1qwen38.sh returns after the server becomes healthy in non-interactive
# mode. Keep this Hive miner process in the foreground so miner stop reaches
# this wrapper and so miner-run does not immediately restart it.
while true; do
    [[ -f "$STOP_FILE" ]] && exit 0

    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        sleep 5
        continue
    fi

    pid=''
    if [[ -r "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    fi
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        sleep 2
        continue
    fi

    say 'Qwen3.8 server stopped unexpectedly'
    exit 1
 done
