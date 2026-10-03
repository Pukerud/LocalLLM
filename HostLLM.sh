#!/usr/bin/env bash
# HostLLM: retained Qwen bundles and the prepared Strata experiment.
# This menu never starts/stops Hive miners, osn.service, or renter containers.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

safety() { python3 "$SCRIPT_DIR/engine_safety.py" "$@"; }
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
    printf '  OctaSpace: %s (never changed by this menu)\n' "${octa:-unknown}"
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
    "$SCRIPT_DIR/v1strata.sh" --stop || rc=1
    safety stop-qwen || rc=1
    if [[ "$(detect_engine)" == unmanaged ]]; then
        echo 'An unmanaged server remains. Stop it in its own terminal/supervisor; no unknown process was killed.'
        rc=1
    fi
    return "$rc"
}

run_selected() {
    local launcher="$1"
    shift
    if [[ ! -x "$SCRIPT_DIR/$launcher" ]]; then
        echo "$launcher is missing or not executable."
        return 1
    fi
    safety preflight || return 1
    "$SCRIPT_DIR/$launcher" "$@"
}

main() {
    local active choice
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
        echo '  [1] Qwen profiles — Swift 1.5 Uncensored Q8 (default) / Hauhau Q8 FastMTP'
        echo '      Both retain BF16 vision and native 262K context per slot'
        echo '  [2] Strata IQ3_S — EXPERIMENTAL | BF16 GPU vision | native 262K | INT8 KV'
        echo '      Four-GPU auto layer split + MTP; high reasoning; one request at a time'
        echo ''
        echo '  Manually run miner stop before starting an LLM. This menu will not do it for you.'
        echo '  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit'
        read -r -p '  Select: ' choice || return 0
        case "${choice//[[:space:]]/}" in
            1|q|Q)
                if [[ "$active" == qwen38 ]]; then
                    "$SCRIPT_DIR/v1qwen38.sh" --dashboard
                elif [[ "$active" != none ]]; then
                    echo 'Another LLM is running. Stop it first.'
                else
                    run_selected v1qwen38.sh --quickstart
                fi
                ;;
            2|3|s|S)
                if [[ "$active" == strata ]]; then
                    "$SCRIPT_DIR/v1strata.sh" --status
                elif [[ "$active" != none ]]; then
                    echo 'Another LLM is running. Stop it first.'
                else
                    run_selected v1strata.sh --quickstart
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
