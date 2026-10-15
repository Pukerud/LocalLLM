#!/usr/bin/env bash
# HostLLM: retained Qwen bundles and guarded Strata experiments.
# Hosting pauses the Hive miner and OctaSpace, then restores their previous state.
# Rental containers/unidentified GPU workloads are never stopped.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOSTLLM_ACTIVE_LEASE_ID=''

safety() { python3 "$SCRIPT_DIR/engine_safety.py" "$@"; }
hosting() { python3 "$SCRIPT_DIR/hosting_lifecycle.py" "$@"; }
release_owned_lease() {
    local lease_id="$1"
    if hosting release --lease-id "$lease_id"; then
        [[ "$HOSTLLM_ACTIVE_LEASE_ID" != "$lease_id" ]] || HOSTLLM_ACTIVE_LEASE_ID=''
    else
        return 1
    fi
}
release_menu_lease_on_exit() {
    [[ -n "$HOSTLLM_ACTIVE_LEASE_ID" ]] || return 0
    release_owned_lease "$HOSTLLM_ACTIVE_LEASE_ID"
}
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
    if [[ "$launcher" == v1strata.sh && "$EUID" -eq 0 ]]; then
        echo 'Run Strata as the unprivileged operator; its Q4 profiles refuse root-owned servers.'
        return 1
    fi
    if [[ ! -x "$SCRIPT_DIR/$launcher" ]]; then
        echo "$launcher is missing or not executable."
        return 1
    fi
    if [[ "$launcher" == v1strata.sh ]]; then
        local arg previous='' i candidate_041=0 candidate_0441=0 requested_parallel=1 requested_groups=1
        local -a check_args=(--check-ready) candidate_env=()
        for arg in "$@"; do
            case "$previous" in
                --profile|--runtime|--parallel|--batch-groups) check_args+=("$previous" "$arg") ;;
            esac
            previous="$arg"
        done
        for ((i=0; i<${#check_args[@]}; i++)); do
            case "${check_args[i]}" in
                --runtime)
                    [[ "${check_args[i+1]:-}" == 0.1.41 ]] && candidate_041=1
                    [[ "${check_args[i+1]:-}" == 0.1.44.1 ]] && candidate_0441=1
                    ;;
                --parallel) requested_parallel="${check_args[i+1]:-1}" ;;
                --batch-groups) requested_groups="${check_args[i+1]:-1}" ;;
            esac
        done
        if [[ "$candidate_041" -eq 1 ]]; then
            local requested_host="${STRATA_HOST:-127.0.0.1}" readiness expected_readiness
            case "$requested_host" in
                127.0.0.1|0.0.0.0) ;;
                *) echo 'Q4 .41 only permits loopback or explicit 0.0.0.0 binding; refusing before service pause.' >&2; return 1 ;;
            esac
            if [[ "${Q4KS_V0141_VALIDATION_ONLY:-0}" == 1 ]]; then
                [[ "$requested_host" == 127.0.0.1 && "$requested_parallel" == 1 && "$requested_groups" == 1 ]] || {
                    echo 'The initial .41 validation server is loopback-only and single-slot; refusing before service pause.' >&2
                    return 1
                }
                expected_readiness='READY CONFIG: pinned Q4_K_S; future profile context=32768; text-only; single-slot; conversation cache=8192 MiB / 4 cache slots.'
            elif [[ "$requested_parallel" == 2 && "$requested_groups" == 2 ]]; then
                expected_readiness='READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; 2 serving slots; batch groups=2; conversation cache=8192 MiB / 4 cache slots.'
            elif [[ "$requested_parallel" == 1 && "$requested_groups" == 1 ]]; then
                expected_readiness='READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; single-slot; conversation cache=8192 MiB / 4 cache slots.'
            else
                echo 'Q4 .41 supports only the single-slot profile or the tested 2-slot/2-group profile.' >&2
                return 1
            fi
            readiness="$("$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            [[ "$readiness" == *"$expected_readiness"* ]] || {
                echo 'Q4 .41 readiness is incomplete for the selected stage; migration gates remain in force.' >&2
                return 1
            }
        elif [[ "$candidate_0441" -eq 1 ]]; then
            local requested_host="${STRATA_HOST:-127.0.0.1}" readiness expected_readiness actual_readiness
            case "$requested_host" in
                127.0.0.1|0.0.0.0) ;;
                *) echo 'Q4 .44.1 only permits loopback or explicit 0.0.0.0 binding; refusing before service pause.' >&2; return 1 ;;
            esac
            if [[ "$requested_parallel" != 1 || "$requested_groups" != 1 ]]; then
                echo 'Q4 .44.1 supports only its validated single-slot/single-group profile.' >&2
                return 1
            fi
            candidate_env=(
                "HOME=${HOME:-/home/user}" "USER=${USER:-user}" "LOGNAME=${LOGNAME:-${USER:-user}}"
                'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' 'PYTHONDONTWRITEBYTECODE=1'
                "QWEN38_OWNER_USER=${QWEN38_OWNER_USER:-${USER:-user}}"
                "STRATA_DATA_ROOT=${STRATA_DATA_ROOT:-${HOME:-/home/user}/.local/share/localllm-strata}"
                "STRATA_STATE_ROOT=${STRATA_STATE_ROOT:-${HOME:-/home/user}/.local/state/locallm-strata}"
                'STRATA_RUNTIME=0.1.44.1' 'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'
                "STRATA_PORT=${STRATA_PORT:-8080}" "STRATA_HOST=$requested_host"
            )
            [[ -z "${STRATA_API_KEY:-}" ]] || candidate_env+=("STRATA_API_KEY=$STRATA_API_KEY")
            [[ -z "${HOSTLLM_STATE_ROOT:-}" ]] || candidate_env+=("HOSTLLM_STATE_ROOT=$HOSTLLM_STATE_ROOT")
            readiness="$(/usr/bin/env -i "${candidate_env[@]}" "$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
            actual_readiness="$(grep '^READY CONFIG:' <<< "$readiness" || true)"
            [[ "$actual_readiness" == "$expected_readiness" ]] || {
                echo 'Q4 .44.1 readiness is incomplete; no HostLLM service pause was started.' >&2
                return 1
            }
        else
            "$SCRIPT_DIR/$launcher" "${check_args[@]}" || return 1
        fi
    fi
    local port="${QWEN38_PORT:-8080}"
    [[ "$launcher" == v1strata.sh ]] && port="${STRATA_PORT:-8080}"
    local lease_id host_state_root launch_fd='' rc=0
    lease_id="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" || return 1
    hosting begin --require-new-lease --lease-id "$lease_id" --port "$port" || return 1
    HOSTLLM_ACTIVE_LEASE_ID="$lease_id"
    host_state_root="${HOSTLLM_STATE_ROOT:-${HOME}/.local/state/hostllm}"
    if [[ -z "${HOSTLLM_STATE_ROOT:-}" && "$EUID" -eq 0 ]]; then
        local state_owner="${QWEN38_OWNER_USER:-${SUDO_USER:-user}}" owner_home
        owner_home="$(getent passwd "$state_owner" | cut -d: -f6)" || owner_home=''
        [[ -n "$owner_home" ]] || { release_owned_lease "$lease_id" || true; return 1; }
        host_state_root="$owner_home/.local/state/hostllm"
    fi
    if ! exec {launch_fd}<"$host_state_root/lifecycle.lock"; then
        echo 'Cannot reserve the HostLLM lifecycle for launch; releasing only this new lease.' >&2
        release_owned_lease "$lease_id" || true
        return 1
    fi
    export HOSTLLM_LAUNCH_LOCK_FD="$launch_fd" HOSTLLM_LAUNCH_LEASE_ID="$lease_id"
    if ! flock -x "$launch_fd" || ! hosting verify-lease --lease-id "$lease_id"; then
        flock -u "$launch_fd" || true
        exec {launch_fd}<&-
        unset HOSTLLM_LAUNCH_LOCK_FD HOSTLLM_LAUNCH_LEASE_ID
        release_owned_lease "$lease_id" || true
        return 1
    fi
    if [[ "${candidate_0441:-0}" -eq 1 ]]; then
        /usr/bin/env -i "${candidate_env[@]}" \
            "HOSTLLM_LAUNCH_LOCK_FD=$launch_fd" "HOSTLLM_LAUNCH_LEASE_ID=$lease_id" \
            "$SCRIPT_DIR/$launcher" "$@" || rc=$?
    else
        "$SCRIPT_DIR/$launcher" "$@" || rc=$?
    fi
    flock -u "$launch_fd" || true
    exec {launch_fd}<&-
    unset HOSTLLM_LAUNCH_LOCK_FD HOSTLLM_LAUNCH_LEASE_ID
    release_owned_lease "$lease_id" || { [[ "$rc" -ne 0 ]] || rc=1; }
    return "$rc"
}

Q4_PARITY_LEASE_ID=''
Q4_SETUP_LEASE_ID=''

run_q4ks141_prepare() {
    local active pause_status rc=0 lease_id port
    if [[ "$EUID" -eq 0 ]]; then
        echo 'Run HostLLM as the unprivileged operator; only its lifecycle helper uses sudo.'
        return 1
    fi
    active="$(detect_engine)" || return 1
    if [[ "$active" != none ]]; then
        echo "Q4 .41 candidate preparation refused while an LLM session is active ($active); nothing was stopped."
        return 1
    fi
    pause_status="$(hosting status 2>/dev/null)" || return 1
    if [[ "$pause_status" != 'not paused by HostLLM' ]]; then
        echo 'Q4 .41 preparation refused because an existing HostLLM pause lease must be recovered separately.'
        return 1
    fi
    port="${STRATA_PORT:-8080}"
    lease_id="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" || return 1
    hosting begin --require-new-lease --lease-id "$lease_id" --port "$port" || return 1
    HOSTLLM_ACTIVE_LEASE_ID="$lease_id"
    Q4_SETUP_LEASE_ID="$lease_id"
    "$SCRIPT_DIR/v1strata.sh" --prepare-orca-q4ks-141 || rc=$?
    if release_owned_lease "$lease_id"; then
        Q4_SETUP_LEASE_ID=''
    else
        rc=1
    fi
    return "$rc"
}

run_q4ks_parity() {
    local active pause_status rc=0 lease_id port
    if [[ "$EUID" -eq 0 ]]; then
        echo 'Run HostLLM as the unprivileged operator; only its lifecycle helper uses sudo.'
        return 1
    fi
    port="${STRATA_PORT:-8080}"
    active="$(detect_engine)"
    if [[ "$active" != none ]] && ! hosting managed-qwen; then
        echo "Q4 parity refused while an unmanaged/other LLM session is active ($active); no session was stopped."
        return 1
    fi
    pause_status="$(hosting status 2>/dev/null)" || return 1
    if [[ "$pause_status" != 'not paused by HostLLM' ]]; then
        echo 'Q4 parity refused because an existing HostLLM pause lease must be recovered separately.'
        return 1
    fi
    lease_id="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" || return 1
    hosting begin --require-new-lease --lease-id "$lease_id" --port "$port" || return 1
    HOSTLLM_ACTIVE_LEASE_ID="$lease_id"
    Q4_PARITY_LEASE_ID="$lease_id"
    active="$(detect_engine)"
    if [[ "$active" != none ]]; then
        echo "Q4 parity cancelled because the managed server remains active ($active); pause lease retained for explicit recovery."
        return 1
    fi
    "$SCRIPT_DIR/v1strata.sh" --verify-orca-q4ks-parity --lease-id "$lease_id" --port "$port" || rc=$?
    if release_owned_lease "$lease_id"; then
        Q4_PARITY_LEASE_ID=''
    else
        rc=1
    fi
    return "$rc"
}

run_q4ks141_parity() {
    local active pause_status rc=0 lease_id port
    if [[ "$EUID" -eq 0 ]]; then
        echo 'Run HostLLM as the unprivileged operator; only its lifecycle helper uses sudo.'
        return 1
    fi
    port="${STRATA_PORT:-8080}"
    active="$(detect_engine)"
    if [[ "$active" != none ]] && ! hosting managed-qwen; then
        echo "Q4 .41 parity refused while an unmanaged/other LLM session is active ($active); no session was stopped."
        return 1
    fi
    pause_status="$(hosting status 2>/dev/null)" || return 1
    if [[ "$pause_status" != 'not paused by HostLLM' ]]; then
        echo 'Q4 .41 parity refused because an existing HostLLM pause lease must be recovered separately.'
        return 1
    fi
    lease_id="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" || return 1
    hosting begin --require-new-lease --lease-id "$lease_id" --port "$port" || return 1
    HOSTLLM_ACTIVE_LEASE_ID="$lease_id"
    Q4_PARITY_LEASE_ID="$lease_id"
    active="$(detect_engine)"
    if [[ "$active" != none ]]; then
        echo "Q4 .41 parity cancelled because the managed server remains active ($active); pause lease retained for explicit recovery."
        return 1
    fi
    "$SCRIPT_DIR/v1strata.sh" --verify-orca-q4ks-141-parity --lease-id "$lease_id" --port "$port" || rc=$?
    if release_owned_lease "$lease_id"; then
        Q4_PARITY_LEASE_ID=''
    else
        rc=1
    fi
    return "$rc"
}

main() {
    local active choice profile hive_profile
    trap 'release_menu_lease_on_exit || true' EXIT
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
        echo '  [4] Orca Q4_K_S — Strata 0.1.39 recorded default | native 262K | CPU vision | one serving slot'
        echo '  [5] Orca Q4_K_S — Strata 0.1.41 | native 262K | CPU vision | 8 GiB conversation cache | one serving slot'
        echo '  [6] Orca Q4_K_S — Strata 0.1.41 | two serving slots / two batch groups | experimental, very low VRAM headroom'
        echo '  [7] Orca Q4_K_S — Strata 0.1.44.1 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted'
        echo ''
        printf '  Strata slots: %s (STRATA_PARALLEL; default 1), batch groups: %s. Runtime override: %s.\n' "${STRATA_PARALLEL:-1}" "${STRATA_BATCH_GROUPS:-1}" "${STRATA_RUNTIME:-selected default}"
        echo '  Starting hosting pauses the miner and OctaSpace automatically (never an active rental).'
        echo '  Strata LAN UI/API: http://192.168.1.69:8080/ — wait for readiness.'
        echo '  API is unauthenticated by default; provide STRATA_API_KEY and firewall access beyond a trusted LAN.'
        echo '  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit'
        echo '  [12] Set next Hive miner profile to Orca Q4_K_S'
        echo '  [13] Restore previous Swift 1.5 Q8 Hive miner profile'
        echo '  [14] Set next Hive miner profile to Orca Q4_K_S .41 (native 262K, CPU vision, one slot)'
        echo '  [15] Set next Hive miner profile to Orca Q4_K_S .41 (native 262K, two slots / two groups)'
        echo '  [16] Set next Hive miner profile to Orca Q4_K_S .44.1 (auto prefill, trimmed split; windows-2 override omitted)'
        echo '  Hive profile changes are idle-only; they never start/stop a miner or osn.service.'
        read -r -p '  Select: ' choice || return 0
        choice="${choice//[[:space:]]/}"
        case "$choice" in
            1|2|q|Q)
                profile=swift15u-q8
                [[ "$choice" == 2 ]] && profile=hauhau-q8-fastmtp-q4kv-xhigh
                if [[ "$active" == qwen38 ]] && ! hosting managed-qwen; then
                    echo 'A manual Qwen hosting session is already running; stop it before switching.'
                    "$SCRIPT_DIR/v1qwen38.sh" --dashboard
                elif [[ "$active" != none ]] && ! hosting managed-qwen; then
                    echo 'Another/unmanaged LLM is running. Stop it in its own verified lifecycle first.'
                else
                    run_selected v1qwen38.sh --quickstart --profile "$profile"
                fi
                ;;
            3|4|5|6|7|s|S|o|O)
                profile=iq3_s
                [[ "$choice" != 4 && "$choice" != o && "$choice" != O && "$choice" != 5 && "$choice" != 6 && "$choice" != 7 ]] || profile=orca-q4_k_s
                if [[ "$active" == strata ]]; then
                    "$SCRIPT_DIR/v1strata.sh" --status
                    echo 'The running Strata model was left alone. Use [9] only when you are ready to switch.'
                elif [[ "$active" == qwen38 ]] && ! hosting managed-qwen; then
                    echo 'A manual Qwen hosting session is already running; stop it before switching.'
                elif [[ "$active" != none ]] && ! hosting managed-qwen; then
                    echo 'Another/unmanaged LLM is running. Stop it in its own verified lifecycle first.'
                elif [[ "$choice" == 5 ]]; then
                    Q4KS_V0141_VALIDATION_ONLY=0 STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.41 --parallel 1 --batch-groups 1
                elif [[ "$choice" == 6 ]]; then
                    Q4KS_V0141_VALIDATION_ONLY=0 STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.41 --parallel 2 --batch-groups 2
                elif [[ "$choice" == 7 ]]; then
                    STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.44.1 --parallel 1 --batch-groups 1
                elif [[ "$profile" == orca-q4_k_s ]]; then
                    run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.39
                else
                    run_selected v1strata.sh --quickstart --profile "$profile"
                fi
                ;;
            9) stop_owned_engines ;;
            10) check_update ;;
            11|x|X) return 0 ;;
            12|13|14|15|16)
                hive_profile=orca-q4_k_s
                [[ "$choice" == 13 ]] && hive_profile=swift15u-q8
                [[ "$choice" == 14 ]] && hive_profile=orca-q4_k_s-v0141
                [[ "$choice" == 15 ]] && hive_profile=orca-q4_k_s-v0141-2slot
                [[ "$choice" == 16 ]] && hive_profile=orca-q4_k_s-v01441
                if [[ ! -x /hive/miners/custom/llm-hosting/profile-menu.sh ]]; then
                    echo 'Hive profile menu is not installed. Install the Hive custom miner package first.'
                elif ! command -v sudo >/dev/null 2>&1; then
                    echo 'sudo is unavailable; run the Hive profile menu as root when the miner is idle.'
                else
                    sudo -- /hive/miners/custom/llm-hosting/profile-menu.sh --profile "$hive_profile"
                fi
                ;;
        esac
        if [[ -t 0 ]]; then
            read -r -p 'Press Enter to return to the menu...' _ || return 0
        fi
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    trap 'release_menu_lease_on_exit || true' EXIT
    if [[ "${1:-}" == --prepare-q4ks141 ]]; then
        [[ "$#" -eq 1 ]] || { echo '--prepare-q4ks141 takes no extra flags' >&2; exit 2; }
        run_q4ks141_prepare
        exit "$?"
    fi
    if [[ "${1:-}" == --run-q4ks141-32k-validation ]]; then
        [[ "$#" -eq 1 ]] || { echo '--run-q4ks141-32k-validation takes no extra flags' >&2; exit 2; }
        echo 'Starting only the loopback 32K .41 candidate under HostLLM; in another terminal run ./v1strata.sh --validate-orca-q4ks-141-32k, then Ctrl+C here.'
        Q4KS_V0141_VALIDATION_ONLY=1 STRATA_HOST=127.0.0.1 \
            run_selected v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.41
        exit "$?"
    fi
    if [[ "${1:-}" == --verify-q4ks-parity ]]; then
        [[ "$#" -eq 1 ]] || { echo '--verify-q4ks-parity takes no extra flags' >&2; exit 2; }
        run_q4ks_parity
        exit "$?"
    fi
    if [[ "${1:-}" == --verify-q4ks141-parity ]]; then
        [[ "$#" -eq 1 ]] || { echo '--verify-q4ks141-parity takes no extra flags' >&2; exit 2; }
        run_q4ks141_parity
        exit "$?"
    fi
    main "$@"
fi
