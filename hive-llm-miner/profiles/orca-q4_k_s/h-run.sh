#!/usr/bin/env bash
# HiveOS custom miner: Orca Uncensored Q4_K_S on Strata 0.1.39.
# Hive supervises this wrapper; the model launcher runs as unprivileged user.
# Keep osn.service running as the prior Qwen Hive miner does.
set -Eeuo pipefail

LLM_ROOT='/home/user/LocalLLM'
STRATA_LAUNCHER="${LLM_ROOT}/v1strata.sh"
STRATA_OWNER_USER='user'
STRATA_HOME='/home/user'
STRATA_DATA_ROOT='/home/user/.local/share/localllm-strata'
STRATA_STATE_ROOT='/home/user/.local/state/locallm-strata'
PROFILE='orca-q4_k_s'
RUNTIME='0.1.39'
PORT='8080'
STOP_FILE="${MINER_STOP:-/run/hive/MINER_STOP}"
STATE_FILE="${STRATA_STATE_ROOT}/server.json"
STRATA_LAUNCH_TOKEN=''
START_STATE_IDENTITY=''
LAUNCH_JOB_PID=''
STOPPER_ARMED=0
STOPPING=0
STOP_REQUESTED=0
IDENTITY_REVOKED=0
OWNED_STOP_COMPLETE=0
KEEP_HIVE_SCREEN=0
LAST_STOP_ATTEMPT=0
LAST_OPERATOR_NOTICE=0

# Root-side safety checks must resolve the operator's state, not /root.
export HOME="$STRATA_HOME" QWEN38_OWNER_USER="$STRATA_OWNER_USER"
export STRATA_DATA_ROOT STRATA_STATE_ROOT
export PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV CDPATH || true

say() { printf '[llm-hosting] %s\n' "$*"; }

request_hive_stop() {
    mkdir -p "$(dirname -- "$STOP_FILE")"
    printf '1\n' > "$STOP_FILE"
}

abort_if_stopping() {
    if [[ "$STOP_REQUESTED" -eq 1 || -f "$STOP_FILE" ]]; then
        say 'stop requested before Strata launch; leaving existing services untouched'
        exit 0
    fi
}

active_engine() {
    local value
    if ! value="$(python3 "${LLM_ROOT}/engine_safety.py" active 2>/dev/null)"; then
        say 'refusing to start: identity-safe engine inspection failed'
        request_hive_stop
        exit 1
    fi
    printf '%s' "$value"
}

check_port_free() {
    local listeners
    if ! listeners="$(ss -Hltpn "( sport = :${PORT} )" 2>/dev/null)"; then
        say 'refusing to start: cannot inspect API port state'
        request_hive_stop
        exit 1
    fi
    if [[ -n "$listeners" ]]; then
        say "refusing to start: port ${PORT} is already occupied"
        request_hive_stop
        exit 1
    fi
}

run_strata_user() {
    runuser -u "$STRATA_OWNER_USER" -- env -i \
        HOME="$STRATA_HOME" USER="$STRATA_OWNER_USER" LOGNAME="$STRATA_OWNER_USER" \
        PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
        STRATA_DATA_ROOT="$STRATA_DATA_ROOT" \
        STRATA_STATE_ROOT="$STRATA_STATE_ROOT" \
        STRATA_RUNTIME="$RUNTIME" \
        STRATA_PARALLEL=1 \
        STRATA_BATCH_GROUPS=1 \
        STRATA_PORT="$PORT" \
        HIVE_STRATA_LAUNCH_ID="$STRATA_LAUNCH_TOKEN" \
        "$@"
}

owned_state_identity() {
    [[ -n "$STRATA_LAUNCH_TOKEN" && -r "$STATE_FILE" ]] || return 1
    python3 - "$STATE_FILE" "$STRATA_LAUNCH_TOKEN" "$PROFILE" "$RUNTIME" "$PORT" <<'PY'
import json,os,pwd,sys,time
from pathlib import Path
path,token,profile,runtime,port=sys.argv[1:]
def snapshot(pid):
    p=Path('/proc')/str(pid)
    for attempt in range(3):
        try:
            a=(p/'stat').read_text().rsplit(')',1)[1].split()
            if a[0] in ('Z','X','x') or bool(int(a[6])&0x4): return None
            exe=os.readlink(p/'exe')
            cmd=(p/'cmdline').read_bytes().decode('utf-8','strict').rstrip('\0').split('\0')
            uid=(p/'status').stat().st_uid
            b=(p/'stat').read_text().rsplit(')',1)[1].split()
            exe2=os.readlink(p/'exe')
            cmd2=(p/'cmdline').read_bytes().decode('utf-8','strict').rstrip('\0').split('\0')
            uid2=(p/'status').stat().st_uid
            if a[19]==b[19] and a[1:4]==b[1:4] and exe==exe2 and cmd==cmd2 and uid==uid2:
                return {'pid':int(pid),'start_ticks':int(b[19]),'exe':exe,'cmd':cmd,'uid':uid,
                        'ppid':int(b[1]),'pgid':int(b[2]),'sid':int(b[3])}
        except (FileNotFoundError,PermissionError,UnicodeError,IndexError,ValueError,OSError):
            return None
        time.sleep(.02*(attempt+1))
    return None
def carries_token(pid):
    values=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    return b'HIVE_STRATA_LAUNCH_ID='+token.encode() in values
try:
    state=json.loads(Path(path).read_text())
    identity=state['identity']; pid=int(identity['pid']); ppid=int(identity['ppid'])
    expected_uid=pwd.getpwnam('user').pw_uid
    live=snapshot(pid)
    if (state.get('profile')!=profile or state.get('runtime_version')!=runtime or
        state.get('port')!=int(port) or identity.get('uid')!=expected_uid or not live):
        raise SystemExit(1)
    if any(identity.get(k)!=live.get(k) for k in ('pid','start_ticks','exe','cmd','uid','ppid','pgid','sid')):
        raise SystemExit(1)
    if 'serve.server' not in '\0'.join(identity.get('cmd',[])):
        raise SystemExit(1)
    parent=snapshot(ppid)
    if not parent or parent['uid']!=expected_uid or not any('strata_launcher.py' in x for x in parent['cmd']):
        raise SystemExit(1)
    if not carries_token(pid) or not carries_token(ppid):
        raise SystemExit(1)
    if '--port' not in identity['cmd'] or identity['cmd'][identity['cmd'].index('--port')+1]!=port:
        raise SystemExit(1)
    if '--engine' not in identity['cmd'] or identity['cmd'][identity['cmd'].index('--engine')+1]!='strata':
        raise SystemExit(1)
    print(f'{pid}:{identity["start_ticks"]}:{ppid}:{profile}:{runtime}')
except (OSError,KeyError,ValueError,IndexError,TypeError):
    raise SystemExit(1)
PY
}

stop_llm() {
    [[ "$STOPPER_ARMED" -eq 1 && "$STOPPING" -eq 0 ]] || return 0
    local current
    current="$(owned_state_identity 2>/dev/null || true)"
    if [[ -z "$current" || "$current" != "$START_STATE_IDENTITY" ]]; then
        say 'Strata identity no longer matches this Hive launch; refusing to stop another session'
        return 1
    fi
    STOPPING=1
    say 'stopping only this Hive launch through the token- and identity-verified Strata launcher'
    if ! run_strata_user "$STRATA_LAUNCHER" --stop --profile "$PROFILE" --runtime "$RUNTIME"; then
        STOPPING=0
        say 'Strata stop was not confirmed; keeping Hive supervision armed'
        return 1
    fi
    local engine listeners
    for _ in {1..30}; do
        current="$(owned_state_identity 2>/dev/null || true)"
        if [[ "$current" != "$START_STATE_IDENTITY" ]] && \
           engine="$(python3 "${LLM_ROOT}/engine_safety.py" active 2>/dev/null)" && \
           [[ "$engine" == none ]] && \
           listeners="$(ss -Hltpn "( sport = :${PORT} )" 2>/dev/null)" && \
           [[ -z "$listeners" ]]; then
            STOPPER_ARMED=0
            IDENTITY_REVOKED=1
            OWNED_STOP_COMPLETE=1
            STOPPING=0
            say 'owned Strata identity, engine, and API listener are gone'
            return 0
        fi
        sleep 1
    done
    STOPPING=0
    say 'Strata stop returned, but the owned identity/listener did not clear; keeping Hive supervision armed'
    return 1
}

on_signal() {
    STOP_REQUESTED=1
    request_hive_stop
}

close_hive_screen() {
    [[ "$KEEP_HIVE_SCREEN" -eq 0 ]] || { say 'retaining Hive screen for operator recovery'; return 0; }
    local session="${STY:-}"
    [[ -n "$session" ]] || return 0
    session="${session##*/}"
    command -v screen >/dev/null 2>&1 || return 0
    screen -S "$session" -X quit >/dev/null 2>&1 || true
}

trap on_signal INT TERM HUP
trap 'if [[ "$STOPPER_ARMED" -eq 1 ]]; then stop_llm || true; fi; close_hive_screen' EXIT

[[ -x "$STRATA_LAUNCHER" ]] || { say "missing launcher: $STRATA_LAUNCHER"; request_hive_stop; exit 1; }
[[ -f "$STOP_FILE" ]] && { say 'Hive stop marker is present; not starting Strata'; exit 0; }
command -v runuser >/dev/null 2>&1 || { say 'runuser is unavailable'; request_hive_stop; exit 1; }

# Refuse stale sessions/listeners instead of letting cleanup touch them.
abort_if_stopping
active="$(active_engine)"
if [[ "$active" != none || -e "$STATE_FILE" ]]; then
    say "refusing to start: existing or unmanaged LLM state ($active)"
    request_hive_stop
    exit 1
fi
check_port_free
abort_if_stopping

# Treat any running container as a possible OctaSpace rental; fail closed.
docker_workload_info() {
    local lines name image status
    command -v docker >/dev/null 2>&1 || return 2
    if ! lines="$(docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null)"; then
        return 2
    fi
    while IFS=$'\t' read -r name image status; do
        [[ -n "$name" ]] || continue
        printf '%s (%s; %s)\n' "$name" "$image" "$status"
        return 0
    done <<< "$lines"
    return 1
}
if workload="$(docker_workload_info)"; then
    say "refusing to start: possible OctaSpace Docker workload is running: $workload"
    request_hive_stop
    exit 1
else
    workload_rc=$?
    if [[ "$workload_rc" -eq 2 ]]; then
        say 'refusing to start: Docker workload status is unavailable'
        request_hive_stop
        exit 1
    fi
fi
abort_if_stopping

# The prior near-full-context probe peaked near 23.5 GiB on its fullest RTX 3090.
gpu_free_raw="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)" || {
    say 'refusing to start: GPU memory inventory is unavailable'
    request_hive_stop
    exit 1
}
readarray -t gpu_free <<< "$gpu_free_raw"
if [[ "${#gpu_free[@]}" -ne 4 ]]; then
    say "refusing to start: expected four GPUs, found ${#gpu_free[@]}"
    request_hive_stop
    exit 1
fi
for free_mib in "${gpu_free[@]}"; do
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 23700 )); then
        say "refusing to start: insufficient per-GPU free VRAM (${gpu_free[*]} MiB; require 23700 MiB each)"
        request_hive_stop
        exit 1
    fi
done
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
available_mib=$((available_kib / 1024))
if (( available_mib < 12288 )); then
    say "refusing to start: only ${available_mib} MiB RAM available; require 12288 MiB"
    request_hive_stop
    exit 1
fi
abort_if_stopping

say 'checking pinned Q4_K_S profile (native context 262144, CPU vision, one slot, Strata 0.1.39)'
if ! run_strata_user "$STRATA_LAUNCHER" --check-ready --profile "$PROFILE" \
        --runtime "$RUNTIME" --parallel 1 --batch-groups 1; then
    say 'Q4_K_S readiness failed; no existing service was stopped'
    request_hive_stop
    exit 1
fi
# Recheck after readiness; Strata repeats the Docker/GPU/port gate at launch.
abort_if_stopping
active="$(active_engine)"
if [[ "$active" != none || -e "$STATE_FILE" ]]; then
    say 'refusing to start: an LLM/state appeared during readiness'
    request_hive_stop
    exit 1
fi
check_port_free
abort_if_stopping
if workload="$(docker_workload_info)"; then
    say "refusing to start: possible Docker workload appeared: $workload"
    request_hive_stop
    exit 1
else
    workload_rc=$?
    if [[ "$workload_rc" -eq 2 ]]; then
        say 'refusing to start: Docker status became unavailable'
        request_hive_stop
        exit 1
    fi
fi
abort_if_stopping

STRATA_LAUNCH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
say 'starting Orca Uncensored Q4_K_S through Hive custom miner (four GPUs, one slot; osn.service remains running)'
run_strata_user "$STRATA_LAUNCHER" --quickstart --profile "$PROFILE" \
    --runtime "$RUNTIME" --parallel 1 --batch-groups 1 &
LAUNCH_JOB_PID=$!

while true; do
    current="$(owned_state_identity 2>/dev/null || true)"
    if [[ -n "$current" && "$IDENTITY_REVOKED" -eq 0 ]]; then
        if [[ -z "$START_STATE_IDENTITY" ]]; then
            START_STATE_IDENTITY="$current"
            STOPPER_ARMED=1
            say 'Hive-owned Orca server identity verified; supervision armed'
        elif [[ "$current" != "$START_STATE_IDENTITY" ]]; then
            say 'Strata identity changed; refusing to manage the replacement session'
            STOPPER_ARMED=0
            IDENTITY_REVOKED=1
        fi
    fi

    if [[ "$STOP_REQUESTED" -eq 1 || -f "$STOP_FILE" ]]; then
        if [[ "$STOPPER_ARMED" -eq 1 ]]; then
            if (( SECONDS - LAST_STOP_ATTEMPT >= 5 )); then
                LAST_STOP_ATTEMPT=$SECONDS
                stop_llm || true
            fi
        elif (( SECONDS - LAST_OPERATOR_NOTICE >= 30 )); then
            LAST_OPERATOR_NOTICE=$SECONDS
            say 'stop pending: waiting for this launch to exit or expose its token-verified server identity; no PID fallback'
        fi
    fi

    if ! jobs -pr | grep -qx "$LAUNCH_JOB_PID"; then
        rc=0
        wait "$LAUNCH_JOB_PID" || rc=$?
        if [[ "$STOPPER_ARMED" -eq 1 ]]; then stop_llm || true; fi
        if [[ "$OWNED_STOP_COMPLETE" -eq 1 ]]; then
            if [[ "$STOP_REQUESTED" -eq 1 || -f "$STOP_FILE" ]]; then exit 0; fi
            request_hive_stop
            say 'owned Strata session exited unexpectedly; shutdown verified and Hive restart disabled'
            exit 1
        fi
        if [[ "$IDENTITY_REVOKED" -eq 1 || "$STOPPER_ARMED" -eq 1 ]]; then
            request_hive_stop
            KEEP_HIVE_SCREEN=1
            say 'Hive wrapper cannot prove the remaining Strata identity is its own; refusing to stop it and retaining the screen for manual identity-based recovery'
            exit 1
        fi
        if [[ "$STOP_REQUESTED" -eq 1 || -f "$STOP_FILE" ]]; then
            engine="$(python3 "${LLM_ROOT}/engine_safety.py" active 2>/dev/null || printf 'unknown')"
            listeners="$(ss -Hltpn "( sport = :${PORT} )" 2>/dev/null || printf 'unknown')"
            if [[ "$engine" == none && -z "$listeners" ]]; then exit 0; fi
            request_hive_stop
            KEEP_HIVE_SCREEN=1
            say 'Hive stop ended before this wrapper could verify shutdown; refusing to signal an unidentified service and retaining the screen'
            exit 1
        fi
        say "Strata launcher exited unexpectedly (rc=$rc); requesting fail-closed Hive stop"
        request_hive_stop
        exit "$rc"
    fi
    sleep 1
done
