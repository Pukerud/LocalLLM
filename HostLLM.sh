#!/usr/bin/env bash
# HostLLM: retained Qwen bundles and the prepared Strata experiment.
# Hosting pauses the Hive miner and OctaSpace, then restores their previous state.
# Rental containers/unidentified GPU workloads are never stopped.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

safety() { python3 "$SCRIPT_DIR/engine_safety.py" "$@"; }
hosting() { python3 "$SCRIPT_DIR/hosting_lifecycle.py" "$@"; }
owned_stop() {
    if [[ "$EUID" -eq 0 ]]; then "$@"; else sudo -n "$@"; fi
}
detect_engine() { safety active 2>/dev/null || printf 'unknown\n'; }

host_gpu_summary() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || printf 'GPU inventory unavailable\n'
}

show_workload_status() {
    local docker_result gpu_result octa
    if ! docker_result="$(docker ps --format '{{.ID}}' 2>/dev/null)"; then
        if [[ "$EUID" -ne 0 ]]; then
            docker_result="$(sudo -n docker ps --format '{{.ID}}' 2>/dev/null)" || docker_result=unknown
        else
            docker_result=unknown
        fi
    fi
    if [[ "$docker_result" == unknown ]]; then
        echo '  Docker: UNKNOWN; all starts fail closed'
    elif [[ -n "$docker_result" ]]; then
        echo '  Docker: workload/rental present; all starts blocked'
    else
        echo '  Docker: no running containers'
    fi
    gpu_result="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)" || gpu_result=unknown
    [[ -z "$gpu_result" ]] || printf '  Current GPU workload: %s\n' "$gpu_result"
    octa="$(systemctl is-active osn.service 2>/dev/null || true)"
    printf '  OctaSpace: %s (automatically paused while hosting)\n' "${octa:-unknown}"
}

check_update() {
    local -a git_cmd=(git -c "safe.directory=$SCRIPT_DIR" -C "$SCRIPT_DIR")
    if [[ "$EUID" -ne 0 && ! -w "$SCRIPT_DIR/.git" ]]; then
        git_cmd=(sudo -n git -c "safe.directory=$SCRIPT_DIR" -C "$SCRIPT_DIR")
    fi
    if [[ -n "$("${git_cmd[@]}" status --porcelain 2>/dev/null)" ]]; then
        echo 'Update refused: local changes are present. No files were overwritten.'
        return 1
    fi
    "${git_cmd[@]}" fetch origin && "${git_cmd[@]}" merge --ff-only "origin/$("${git_cmd[@]}" branch --show-current)"
}

stop_owned_engines() {
    # Each helper validates PID/start time/executable and uses pidfds. No pkill/killall.
    local rc=0
    owned_stop "$SCRIPT_DIR/v1strata.sh" --stop || rc=1
    owned_stop "$SCRIPT_DIR/v1qwen38.sh" --stop || rc=1
    if [[ "$(detect_engine)" == unmanaged ]]; then
        echo 'An unmanaged server remains. Stop it in its own terminal/supervisor; no unknown process was killed.'
        rc=1
    fi
    if [[ "$rc" -eq 0 ]]; then hosting release || rc=1; fi
    return "$rc"
}

run_selected() {
    local launcher="$1"
    shift
    if [[ ! -x "$SCRIPT_DIR/$launcher" ]]; then
        echo "$launcher is missing or not executable."
        return 1
    fi
    if [[ "$launcher" == v1strata.sh ]]; then
        local arg previous=''
        local -a check_args=(--check-ready)
        for arg in "$@"; do
            [[ "$previous" != --profile ]] || check_args+=(--profile "$arg")
            previous="$arg"
        done
        "$SCRIPT_DIR/$launcher" "${check_args[@]}" || return 1
    fi
    local port="${QWEN38_PORT:-8080}"
    [[ "$launcher" == v1strata.sh ]] && port="${STRATA_PORT:-8080}"
    hosting begin --port "$port" || return 1
    local rc=0
    "$SCRIPT_DIR/$launcher" "$@" || rc=$?
    hosting release || { [[ "$rc" -ne 0 ]] || rc=1; }
    return "$rc"
}

main() {
    local active choice profile
    trap 'hosting release || true' EXIT
    while true; do
        [[ -t 1 ]] && clear 2>/dev/null || true
        active="$(detect_engine)"
        echo '=========================================================='
        echo '  HostLLM — Engine Picker'
        echo '=========================================================='
        printf '  LLM status: %s\n' "$active"
        show_workload_status
        host_gpu_summary
        echo ''
        echo '  [1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots'
        echo '  [2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots'
        echo '  [3] Strata IQ3_S — BF16 GPU vision | MTP | high | INT8 KV | native 262K | four-GPU split'
        echo '  [4] Orca Uncensored IQ3_XXS (Strata) — EXPERIMENTAL | MTP | high | initial 32K | F16 vision UNTESTED'
        echo ''
        echo '  Starting hosting pauses the miner and OctaSpace automatically (never an active rental).'
        echo '  Strata LAN UI/API: http://192.168.1.69:8080/ — wait for readiness.'
        echo '  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit'
        read -r -p '  Select: ' choice || return 0
        choice="${choice//[[:space:]]/}"
        case "$choice" in
            1|2|q|Q)
                profile=swift15u-q8
                [[ "$choice" == 2 ]] && profile=hauhau-q8-fastmtp-q4kv-xhigh
                if [[ "$active" == qwen38 ]] && ! hosting managed-qwen; then
                    echo 'A manual Qwen hosting session is already running; stop it before switching.'
                    "$SCRIPT_DIR/v1qwen38.sh" --dashboard
                elif [[ "$active" != none && "$active" != qwen38 ]]; then
                    echo 'Another LLM is running. Stop it first with [9].'
                else
                    run_selected v1qwen38.sh --quickstart --profile "$profile"
                fi
                ;;
            3|4|s|S|o|O)
                profile=iq3_s
                [[ "$choice" != 4 && "$choice" != o && "$choice" != O ]] || profile=orca-iq3_xxs
                if [[ "$active" == strata ]]; then
                    "$SCRIPT_DIR/v1strata.sh" --status
                    echo 'The running Strata model was left alone. Use [9] only when you are ready to switch.'
                elif [[ "$active" != none && "$active" != qwen38 ]]; then
                    echo 'Another LLM is running. Stop it first with [9].'
                else
                    run_selected v1strata.sh --quickstart --profile "$profile"
                fi
                ;;
            9) stop_owned_engines ;;
            10) check_update ;;
            11|x|X) return 0 ;;
        esac
        if [[ -t 0 ]]; then
            read -r -p 'Press Enter to return to the menu...' _ || return 0
        fi
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
