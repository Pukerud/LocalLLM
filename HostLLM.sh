#!/usr/bin/env bash
# HostLLM: retained Qwen bundles and guarded Strata experiments.
# Hosting pauses the Hive miner and OctaSpace, then restores their previous state.
# Rental containers/unidentified GPU workloads are never stopped.
set -uo pipefail
# Never let inherited bash xtrace print credentials during the interactive handoff.
set +x
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HIVE_PROFILE_SOURCE="$SCRIPT_DIR/hive-llm-miner"
HIVE_PROFILE_MENU='/hive/miners/custom/llm-hosting/profile-menu.sh'
HIVE_PROFILE_ROOT='/hive/miners/custom/llm-hosting/profiles'
HIVE_PROFILE_INSTALLER="$SCRIPT_DIR/install-hive-llm-miner.sh"
HOSTLLM_ACTIVE_LEASE_ID=''
unset HOSTLLM_STRATA_API_KEY
if [[ -n "${STRATA_API_KEY:-}" ]]; then
    echo 'Refusing exported STRATA_API_KEY: restart HostLLM without it and enter credentials at the interactive prompt.' >&2
    exit 2
fi

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
run_as_root() {
    if [[ "$EUID" -eq 0 ]]; then "$@"; else sudo -- "$@"; fi
}
hive_profile_package_is_current() {
    local profile="$1" file source_file installed_file
    grep -qx 'MINER=custom' /hive-config/rig.conf 2>/dev/null || return 1
    grep -qx 'CUSTOM_MINER=llm-hosting' /hive-config/wallet.conf 2>/dev/null || return 1
    grep -qx 'CUSTOM_CONFIG_FILENAME=/hive/miners/custom/llm-hosting/llm-hosting.conf' /hive-config/wallet.conf 2>/dev/null || return 1
    grep -qx 'CUSTOM_INSTALL_URL=' /hive-config/wallet.conf 2>/dev/null || return 1
    [[ -f "$HIVE_PROFILE_SOURCE/profile-menu.sh" && -x "$HIVE_PROFILE_MENU" ]] || return 1
    cmp -s "$HIVE_PROFILE_SOURCE/profile-menu.sh" "$HIVE_PROFILE_MENU" || return 1
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        source_file="$HIVE_PROFILE_SOURCE/profiles/$profile/$file"
        installed_file="$HIVE_PROFILE_ROOT/$profile/$file"
        [[ -f "$source_file" && -f "$installed_file" && ! -L "$installed_file" ]] || return 1
        cmp -s "$source_file" "$installed_file" || return 1
    done
}
set_next_hive_profile() {
    local profile="$1" helper="$SCRIPT_DIR/hive_profile_switch.py"
    if [[ ! -f "$helper" || -L "$helper" ]]; then
        echo 'Hive profile-switch helper is missing or unsafe; no miner was stopped.' >&2
        return 1
    fi
    run_as_root /usr/bin/python3 "$helper" --profile "$profile"
}

apply_next_hive_profile() {
    local profile="$1"
    if [[ ! -x "$HIVE_PROFILE_MENU" ]]; then
        echo 'Hive profile menu is not installed. Install the Hive custom miner package first.'
    elif [[ "$EUID" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
        echo 'sudo is unavailable; run the Hive profile menu as root when the miner is idle.'
    else
        if ! hive_profile_package_is_current "$profile"; then
            if [[ ! -x "$HIVE_PROFILE_INSTALLER" ]]; then
                echo 'Installed Hive profile is outdated; the LocalLLM installer is missing.'
                return 1
            fi
            echo 'Refreshing the outdated Hive profile package; miner and services remain untouched.'
            if ! run_as_root "$HIVE_PROFILE_INSTALLER"; then
                echo 'Hive profile refresh failed; no profile selection was attempted.'
                return 1
            fi
            if ! hive_profile_package_is_current "$profile"; then
                echo 'Hive profile refresh did not match this LocalLLM version; no profile selection was attempted.'
                return 1
            fi
        fi
        if ! run_as_root "$HIVE_PROFILE_MENU" --profile "$profile"; then
            echo "Hive profile selection failed for '$profile'; see the error above."
            return 1
        fi
    fi
}
detect_engine() { safety active 2>/dev/null || printf 'unknown\n'; }

run_strata_as_operator() {
    local owner_user owner_home owner_uid state_root
    owner_user="${QWEN38_OWNER_USER:-${SUDO_USER:-user}}"
    owner_uid="$(id -u "$owner_user" 2>/dev/null)" || return 1
    owner_home="$(getent passwd "$owner_user" | cut -d: -f6)"
    if [[ "$owner_uid" == 0 || -z "$owner_home" || "$owner_home" == /root ]]; then
        echo 'Cannot hand Strata to an unprivileged operator; set QWEN38_OWNER_USER=user.' >&2
        return 1
    fi
    command -v runuser >/dev/null 2>&1 || { echo 'runuser is unavailable; Strata was not started.' >&2; return 1; }
    state_root="$owner_home/.local/state/locallm-strata"
    [[ "${STRATA_SC117_IQ3S_TRIAL:-0}" != 1 ]] || state_root+='/sc117-iq3s-trial'
    local previous='' arg
    for arg in "$@"; do
        if [[ ( "$previous" == --profile && "$arg" == deepseek41-abiray-iq2_xxs ) || "$arg" == --profile=deepseek41-abiray-iq2_xxs ]]; then
            state_root="$owner_home/.local/state/locallm-strata/deepseek41-abiray"
        fi
        if [[ ( "$previous" == --profile && "$arg" == swift15-heretic-re ) || "$arg" == --profile=swift15-heretic-re ]]; then
            state_root="$owner_home/.local/state/locallm-strata/swift15-heretic-re"
        fi
        previous="$arg"
    done
    echo "Handing Strata hosting to $owner_user; the server will not run as root."
    runuser -u "$owner_user" -- /usr/bin/env -i \
        "HOME=$owner_home" "USER=$owner_user" "LOGNAME=$owner_user" \
        'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' 'PYTHONDONTWRITEBYTECODE=1' \
        "QWEN38_OWNER_USER=$owner_user" "HOSTLLM_STATE_ROOT=${HOSTLLM_STATE_ROOT:-$owner_home/.local/state/hostllm}" \
        "QWEN38_STATE_ROOT=$owner_home/.local/state/locallm-qwen38" \
        "STRATA_DATA_ROOT=$owner_home/.local/share/localllm-strata" "STRATA_STATE_ROOT=$state_root" \
        "STRATA_HOST=${STRATA_HOST:-0.0.0.0}" "STRATA_PORT=${STRATA_PORT:-8080}" \
        "STRATA_SC117_IQ3S_TRIAL=${STRATA_SC117_IQ3S_TRIAL:-0}" \
        "$SCRIPT_DIR/HostLLM.sh" --run-strata "$@"
}

sc117_trial_command() {
    local owner_user owner_home trial_state
    local -a trial_env=()
    if [[ "$EUID" -eq 0 ]]; then
        owner_user="${QWEN38_OWNER_USER:-${SUDO_USER:-user}}"
        owner_home="$(getent passwd "$owner_user" | cut -d: -f6)"
        [[ -n "$owner_home" ]] || { echo 'Cannot resolve the SC117 trial owner home.' >&2; return 1; }
        command -v runuser >/dev/null 2>&1 || { echo 'runuser is unavailable; refusing SC117 trial control.' >&2; return 1; }
    else
        owner_user="${USER:-$(id -un)}"
        owner_home="${HOME:-}"
        [[ -n "$owner_home" ]] || { echo 'Cannot resolve the SC117 trial owner home.' >&2; return 1; }
    fi
    trial_state="$owner_home/.local/state/locallm-strata/sc117-iq3s-trial"
    trial_env=(
        "HOME=$owner_home" "USER=$owner_user" "LOGNAME=$owner_user"
        'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin'
        'PYTHONDONTWRITEBYTECODE=1' "PYTHONPYCACHEPREFIX=$trial_state/pycache" 'SETUPTOOLS_USE_DISTUTILS=stdlib'
        "QWEN38_OWNER_USER=$owner_user" "STRATA_DATA_ROOT=$owner_home/.local/share/localllm-strata"
        "STRATA_STATE_ROOT=$trial_state" 'STRATA_RUNTIME=0.1.45' 'STRATA_PARALLEL=1'
        'STRATA_BATCH_GROUPS=1' 'STRATA_PORT=8080' 'STRATA_HOST=0.0.0.0'
        'STRATA_SC117_IQ3S_TRIAL=1'
    )
    # Omit --runtime on control commands so an already-running older trial can still be stopped safely.
    if [[ "$EUID" -eq 0 ]]; then
        runuser -u "$owner_user" -- /usr/bin/env -i "${trial_env[@]}" "$SCRIPT_DIR/v1strata.sh" "$@" --profile iq3_s-sc117-abliterated --parallel 1 --batch-groups 1
    else
        /usr/bin/env -i "${trial_env[@]}" "$SCRIPT_DIR/v1strata.sh" "$@" --profile iq3_s-sc117-abliterated --parallel 1 --batch-groups 1
    fi
}

deepseek_profile_command() {
    local owner_user="${QWEN38_OWNER_USER:-${SUDO_USER:-${USER:-user}}}" owner_home
    owner_home="$(getent passwd "$owner_user" | cut -d: -f6)" || return 1
    [[ -n "$owner_home" && "$owner_home" != /root ]] || return 1
    local -a control=()
    [[ "$EUID" -ne 0 ]] || control=(runuser -u "$owner_user" --)
    "${control[@]}" /usr/bin/env -i "HOME=$owner_home" "USER=$owner_user" "LOGNAME=$owner_user" \
        'PATH=/usr/local/bin:/usr/bin:/bin' PYTHONDONTWRITEBYTECODE=1 \
        "QWEN38_OWNER_USER=$owner_user" "STRATA_DATA_ROOT=$owner_home/.local/share/localllm-strata" \
        "STRATA_STATE_ROOT=$owner_home/.local/state/locallm-strata/deepseek41-abiray" \
        "$SCRIPT_DIR/v1strata.sh" "$@" --profile deepseek41-abiray-iq2_xxs --runtime 0.1.51 --parallel 1 --batch-groups 1
}

heretic_profile_command() {
    local owner_user="${QWEN38_OWNER_USER:-${SUDO_USER:-${USER:-user}}}" owner_home
    owner_home="$(getent passwd "$owner_user" | cut -d: -f6)" || return 1
    [[ -n "$owner_home" && "$owner_home" != /root ]] || return 1
    local -a control=()
    [[ "$EUID" -ne 0 ]] || control=(runuser -u "$owner_user" --)
    "${control[@]}" /usr/bin/env -i "HOME=$owner_home" "USER=$owner_user" "LOGNAME=$owner_user" \
        'PATH=/usr/local/bin:/usr/bin:/bin' PYTHONDONTWRITEBYTECODE=1 \
        "QWEN38_OWNER_USER=$owner_user" "STRATA_DATA_ROOT=$owner_home/.local/share/localllm-strata" \
        "STRATA_STATE_ROOT=$owner_home/.local/state/locallm-strata/swift15-heretic-re" \
        "$SCRIPT_DIR/v1strata.sh" "$@" --profile swift15-heretic-re --runtime 0.1.46 --parallel 1 --batch-groups 1
}

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
    deepseek_profile_command --stop || rc=1
    heretic_profile_command --stop || rc=1
    sc117_trial_command --stop || rc=1
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
    local HOSTLLM_STRATA_API_KEY=''
    shift
    if [[ "$launcher" == v1strata.sh && "$EUID" -eq 0 ]]; then
        run_strata_as_operator "$@"
        return "$?"
    fi
    if [[ ! -x "$SCRIPT_DIR/$launcher" ]]; then
        echo "$launcher is missing or not executable."
        return 1
    fi
    if [[ "$launcher" == v1strata.sh ]]; then
        local arg previous='' i candidate_041=0 candidate_0441=0 candidate_045=0 candidate_046=0 candidate_sc117_iq3s=0 candidate_sc117_trial=0 candidate_deepseek=0 candidate_heretic=0
        local requested_profile='' requested_runtime='' requested_parallel=1 requested_groups=1
        local requested_host readiness expected_readiness actual_readiness
        local requested_profile_count=0 requested_runtime_count=0 requested_parallel_count=0 requested_groups_count=0
        local -a check_args=(--check-ready) candidate_env=()
        for arg in "$@"; do
            case "$previous" in
                --profile|--runtime|--parallel|--batch-groups) check_args+=("$previous" "$arg") ;;
            esac
            case "$arg" in
                --profile=*|--runtime=*|--parallel=*|--batch-groups=*) check_args+=("$arg") ;;
            esac
            previous="$arg"
        done
        for ((i=0; i<${#check_args[@]}; i++)); do
            case "${check_args[i]}" in
                --profile)
                    requested_profile="${check_args[i+1]:-}"
                    requested_profile_count=$((requested_profile_count + 1))
                    [[ "${check_args[i+1]:-}" == iq3_s-sc117-abliterated ]] && candidate_sc117_iq3s=1
                    [[ "${check_args[i+1]:-}" == deepseek41-abiray-iq2_xxs ]] && candidate_deepseek=1
                    [[ "${check_args[i+1]:-}" == swift15-heretic-re ]] && candidate_heretic=1
                    ;;
                --profile=*)
                    requested_profile="${check_args[i]#*=}"
                    requested_profile_count=$((requested_profile_count + 1))
                    [[ "$requested_profile" == iq3_s-sc117-abliterated ]] && candidate_sc117_iq3s=1
                    [[ "$requested_profile" == deepseek41-abiray-iq2_xxs ]] && candidate_deepseek=1
                    [[ "$requested_profile" == swift15-heretic-re ]] && candidate_heretic=1
                    ;;
                --runtime)
                    requested_runtime="${check_args[i+1]:-}"
                    requested_runtime_count=$((requested_runtime_count + 1))
                    [[ "${check_args[i+1]:-}" == 0.1.41 ]] && candidate_041=1
                    [[ "${check_args[i+1]:-}" == 0.1.44.1 ]] && candidate_0441=1
                    [[ "${check_args[i+1]:-}" == 0.1.45 ]] && candidate_045=1
                    [[ "${check_args[i+1]:-}" == 0.1.46 ]] && candidate_046=1
                    ;;
                --runtime=*)
                    requested_runtime="${check_args[i]#*=}"
                    requested_runtime_count=$((requested_runtime_count + 1))
                    [[ "$requested_runtime" == 0.1.41 ]] && candidate_041=1
                    [[ "$requested_runtime" == 0.1.44.1 ]] && candidate_0441=1
                    [[ "$requested_runtime" == 0.1.45 ]] && candidate_045=1
                    [[ "$requested_runtime" == 0.1.46 ]] && candidate_046=1
                    ;;
                --parallel)
                    requested_parallel="${check_args[i+1]:-1}"
                    requested_parallel_count=$((requested_parallel_count + 1))
                    ;;
                --parallel=*)
                    requested_parallel="${check_args[i]#*=}"
                    requested_parallel_count=$((requested_parallel_count + 1))
                    ;;
                --batch-groups)
                    requested_groups="${check_args[i+1]:-1}"
                    requested_groups_count=$((requested_groups_count + 1))
                    ;;
                --batch-groups=*)
                    requested_groups="${check_args[i]#*=}"
                    requested_groups_count=$((requested_groups_count + 1))
                    ;;
            esac
        done
        if [[ "$candidate_sc117_iq3s" -eq 1 ]]; then
            if [[ "$requested_profile_count" -ne 1 || "$requested_profile" != iq3_s-sc117-abliterated || \
                  "$requested_runtime_count" -ne 1 || \
                  ( "$requested_runtime" != 0.1.45 && "$requested_runtime" != 0.1.46 ) || \
                  ( "$requested_runtime" == 0.1.45 && "$candidate_045" -ne 1 ) || \
                  ( "$requested_runtime" == 0.1.46 && "$candidate_046" -ne 1 ) || \
                  "$requested_parallel_count" -ne 1 || "$requested_groups_count" -ne 1 || \
                  "$candidate_041" -ne 0 || "$requested_parallel" != 1 || "$requested_groups" != 1 ]]; then
                echo 'SC117 is unqualified and trial-only: require explicit Strata .45/.46, one slot/group, and its isolated trusted-LAN trial environment; .39 hosting is retired.' >&2
                return 1
            fi
            if [[ "${STRATA_SC117_IQ3S_TRIAL:-0}" != 1 || "${STRATA_HOST:-}" != 0.0.0.0 || "${STRATA_PORT:-}" != 8080 ]]; then
                echo "SC117 $requested_runtime is trial-only and requires explicit trusted-LAN 0.0.0.0:8080 with its trial marker; refusing before service pause." >&2
                return 1
            fi
            candidate_sc117_trial=1
        fi
        if [[ "$candidate_heretic" -eq 1 ]]; then
            if [[ "$requested_profile_count" -ne 1 || "$requested_runtime_count" -ne 1 || "$requested_runtime" != 0.1.46 ||
                  "$requested_parallel_count" -ne 1 || "$requested_groups_count" -ne 1 ||
                  "$requested_parallel" != 1 || "$requested_groups" != 1 ]]; then
                echo 'Heretic requires its pinned .46 profile and one slot/group; no services paused.' >&2; return 1
            fi
            candidate_env=(
                "HOME=${HOME:-/home/user}" "USER=${USER:-user}" "LOGNAME=${LOGNAME:-${USER:-user}}"
                'PATH=/usr/local/bin:/usr/bin:/bin' 'PYTHONDONTWRITEBYTECODE=1' 'SETUPTOOLS_USE_DISTUTILS=stdlib'
                "QWEN38_OWNER_USER=${QWEN38_OWNER_USER:-${USER:-user}}"
                "STRATA_DATA_ROOT=${HOME:-/home/user}/.local/share/localllm-strata"
                "STRATA_STATE_ROOT=${HOME:-/home/user}/.local/state/locallm-strata/swift15-heretic-re"
                'STRATA_RUNTIME=0.1.46' 'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'
                "STRATA_HOST=${STRATA_HOST:-0.0.0.0}" "STRATA_PORT=${STRATA_PORT:-8080}"
            )
            [[ -z "${HOSTLLM_STATE_ROOT:-}" ]] || candidate_env+=("HOSTLLM_STATE_ROOT=$HOSTLLM_STATE_ROOT")
            readiness="$(/usr/bin/env -i "${candidate_env[@]}" "$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            expected_readiness='READY CONFIG: Swift 1.5 Heretic RE; Strata=0.1.46; native context=262144; BF16 CPU vision; one serving slot/group; --spec 3; prefill auto; split 14,24,36; trim-stage-weights; pipeline-windows 2; conversation cache=8192 MiB / 4 slots; thinking high; trusted-LAN 0.0.0.0:8080 without mandatory API key. Bounded test, not qualification.'
            [[ "$(grep '^READY CONFIG:' <<< "$readiness" || true)" == "$expected_readiness" ]] || {
                echo 'Heretic exact readiness failed before service pause.' >&2; return 1;
            }
        elif [[ "$candidate_deepseek" -eq 1 ]]; then
            if [[ "$requested_profile_count" -ne 1 || "$requested_runtime_count" -ne 1 || "$requested_runtime" != 0.1.51 ||
                  "$requested_parallel_count" -ne 1 || "$requested_groups_count" -ne 1 ||
                  "$requested_parallel" != 1 || "$requested_groups" != 1 || "$candidate_sc117_iq3s" -ne 0 ]]; then
                echo 'DeepSeek requires its explicit .51 profile and one slot/group; no services paused.' >&2
                return 1
            fi
            candidate_env=(
                "HOME=${HOME:-/home/user}" "USER=${USER:-user}" "LOGNAME=${LOGNAME:-${USER:-user}}"
                'PATH=/usr/local/bin:/usr/bin:/bin' 'PYTHONDONTWRITEBYTECODE=1'
                "QWEN38_OWNER_USER=${QWEN38_OWNER_USER:-${USER:-user}}"
                "STRATA_DATA_ROOT=${HOME:-/home/user}/.local/share/localllm-strata"
                "STRATA_STATE_ROOT=${HOME:-/home/user}/.local/state/locallm-strata/deepseek41-abiray"
                'STRATA_RUNTIME=0.1.51' 'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'
                "STRATA_HOST=${STRATA_HOST:-0.0.0.0}" "STRATA_PORT=${STRATA_PORT:-8080}"
            )
            [[ -z "${HOSTLLM_STATE_ROOT:-}" ]] || candidate_env+=("HOSTLLM_STATE_ROOT=$HOSTLLM_STATE_ROOT")
            readiness="$(/usr/bin/env -i "${candidate_env[@]}" "$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            expected_readiness='READY CONFIG: DeepSeek V4.1 Flash 1017 Abiray IQ2_XXS; Strata DeepSeek=0.1.51 (app 0.2.6); native context=1048576; native BF16 GPU vision (DSI1 bridge); one serving slot; six experts/token; DSpark pinned preset; four GPUs; trusted-LAN 0.0.0.0:8080 without mandatory API key. Configured native window, not a populated-1M or sustained-load qualification.'
            [[ "$(grep '^READY CONFIG:' <<< "$readiness" || true)" == "$expected_readiness" ]] || {
                echo 'DeepSeek exact readiness failed before service pause.' >&2; return 1;
            }
        elif [[ "$candidate_sc117_trial" -eq 1 ]]; then
            local trial_home="${HOME:-/home/user}" trial_state expected_readiness
            trial_state="$trial_home/.local/state/locallm-strata/sc117-iq3s-trial"
            candidate_env=(
                "HOME=$trial_home" "USER=${USER:-user}" "LOGNAME=${LOGNAME:-${USER:-user}}"
                'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin'
                'PYTHONDONTWRITEBYTECODE=1' "PYTHONPYCACHEPREFIX=$trial_state/pycache" 'SETUPTOOLS_USE_DISTUTILS=stdlib'
                "QWEN38_OWNER_USER=${QWEN38_OWNER_USER:-${USER:-user}}"
                "STRATA_DATA_ROOT=$trial_home/.local/share/localllm-strata" "STRATA_STATE_ROOT=$trial_state"
                "STRATA_RUNTIME=$requested_runtime" 'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'
                'STRATA_PORT=8080' 'STRATA_HOST=0.0.0.0' 'STRATA_SC117_IQ3S_TRIAL=1'
            )
            [[ -z "${HOSTLLM_STATE_ROOT:-}" ]] || candidate_env+=("HOSTLLM_STATE_ROOT=$HOSTLLM_STATE_ROOT")
            readiness="$(/usr/bin/env -i "${candidate_env[@]}" "$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            expected_readiness='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.'
            if [[ "$requested_runtime" == 0.1.46 ]]; then
                expected_readiness='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.46; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2; new SM86 release defaults enabled. This is not qualification.'
            fi
            actual_readiness="$(grep '^TRIAL CONFIG:' <<< "$readiness" || true)"
            [[ "$actual_readiness" == "$expected_readiness" ]] || {
                echo "SC117 $requested_runtime trial preflight is incomplete or mismatched; no service pause was started." >&2
                return 1
            }
        elif [[ "$candidate_041" -eq 1 ]]; then
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
        elif [[ "$candidate_0441" -eq 1 || "$candidate_045" -eq 1 || "$candidate_046" -eq 1 ]]; then
            requested_host="${STRATA_HOST:-127.0.0.1}"
            case "$requested_host" in
                127.0.0.1|0.0.0.0) ;;
                *) echo "Q4 $requested_runtime only permits loopback or explicit 0.0.0.0 binding; refusing before service pause." >&2; return 1 ;;
            esac
            if [[ "$requested_parallel" != 1 || "$requested_groups" != 1 ]]; then
                echo "Q4 $requested_runtime supports only its single-slot/single-group profile." >&2
                return 1
            fi
            candidate_env=(
                "HOME=${HOME:-/home/user}" "USER=${USER:-user}" "LOGNAME=${LOGNAME:-${USER:-user}}"
                'PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' 'PYTHONDONTWRITEBYTECODE=1'
                "QWEN38_OWNER_USER=${QWEN38_OWNER_USER:-${USER:-user}}"
                "STRATA_DATA_ROOT=${STRATA_DATA_ROOT:-${HOME:-/home/user}/.local/share/localllm-strata}"
                "STRATA_STATE_ROOT=${STRATA_STATE_ROOT:-${HOME:-/home/user}/.local/state/locallm-strata}"
                "STRATA_RUNTIME=$requested_runtime" 'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'
                "STRATA_PORT=${STRATA_PORT:-8080}" "STRATA_HOST=$requested_host"
            )
            [[ -z "${HOSTLLM_STATE_ROOT:-}" ]] || candidate_env+=("HOSTLLM_STATE_ROOT=$HOSTLLM_STATE_ROOT")
            readiness="$(/usr/bin/env -i "${candidate_env[@]}" "$SCRIPT_DIR/$launcher" "${check_args[@]}")" || return 1
            printf '%s\n' "$readiness"
            expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
            if [[ "$requested_runtime" == 0.1.45 ]]; then
                expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.45; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
            elif [[ "$requested_runtime" == 0.1.46 ]]; then
                expected_readiness='READY CONFIG: unvalidated Q4_K_S Strata 0.1.46 test candidate; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted; new SM86 release defaults enabled. Readiness only; no Q4 inference or qualification claim.'
            fi
            actual_readiness="$(grep '^READY CONFIG:' <<< "$readiness" || true)"
            [[ "$actual_readiness" == "$expected_readiness" ]] || {
                echo "Q4 $requested_runtime readiness is incomplete; no HostLLM service pause was started." >&2
                return 1
            }
        else
            "$SCRIPT_DIR/$launcher" "${check_args[@]}" || return 1
        fi
    fi
    if [[ "$launcher" == v1strata.sh && -z "$HOSTLLM_STRATA_API_KEY" && -t 0 && -r /dev/tty &&
          "${candidate_sc117_trial:-0}" -ne 1 ]]; then
        read -r -s -p 'Optional Strata API key (Enter for no authentication): ' HOSTLLM_STRATA_API_KEY < /dev/tty
        printf '\n' > /dev/tty
        export -n HOSTLLM_STRATA_API_KEY
    fi
    if [[ "$launcher" == v1strata.sh && -n "$HOSTLLM_STRATA_API_KEY" &&
          ( ! -f "$SCRIPT_DIR/exec_with_stdin_api_key.py" || -L "$SCRIPT_DIR/exec_with_stdin_api_key.py" ) ]]; then
        HOSTLLM_STRATA_API_KEY=''
        echo 'Strata API-key stdin helper is missing or unsafe; no service pause was started.' >&2
        return 1
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
    if [[ "$launcher" == v1strata.sh && -n "$HOSTLLM_STRATA_API_KEY" ]]; then
        if [[ "${candidate_0441:-0}" -eq 1 || "${candidate_045:-0}" -eq 1 || "${candidate_046:-0}" -eq 1 || "${candidate_sc117_iq3s:-0}" -eq 1 || "${candidate_deepseek:-0}" -eq 1 || "${candidate_heretic:-0}" -eq 1 ]]; then
            printf '%s' "$HOSTLLM_STRATA_API_KEY" | /usr/bin/env -i "${candidate_env[@]}" \
                "HOSTLLM_LAUNCH_LOCK_FD=$launch_fd" "HOSTLLM_LAUNCH_LEASE_ID=$lease_id" \
                /usr/bin/python3 "$SCRIPT_DIR/exec_with_stdin_api_key.py" "$SCRIPT_DIR/$launcher" "$@" || rc=$?
        else
            printf '%s' "$HOSTLLM_STRATA_API_KEY" | /usr/bin/python3 \
                "$SCRIPT_DIR/exec_with_stdin_api_key.py" "$SCRIPT_DIR/$launcher" "$@" || rc=$?
        fi
    elif [[ "${candidate_0441:-0}" -eq 1 || "${candidate_045:-0}" -eq 1 || "${candidate_046:-0}" -eq 1 || "${candidate_sc117_iq3s:-0}" -eq 1 || "${candidate_deepseek:-0}" -eq 1 || "${candidate_heretic:-0}" -eq 1 ]]; then
        /usr/bin/env -i "${candidate_env[@]}" \
            "HOSTLLM_LAUNCH_LOCK_FD=$launch_fd" "HOSTLLM_LAUNCH_LEASE_ID=$lease_id" \
            "$SCRIPT_DIR/$launcher" "$@" || rc=$?
    else
        "$SCRIPT_DIR/$launcher" "$@" || rc=$?
    fi
    HOSTLLM_STRATA_API_KEY=''
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
        echo '  [3] Swift 1.5 Heretic RE — Strata 0.1.46 | native 262K | BF16 CPU vision | spec 3 + windows 2 | 8 GiB / 4-history cache | one slot'
        echo '  [4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.45 | native 262K | --spec 3 + pipeline-windows 2 | CPU vision | one slot | LAN 8080'
        echo ''
        printf '  Strata slots: %s (STRATA_PARALLEL; default 1), batch groups: %s. Runtime override: %s.\n' "${STRATA_PARALLEL:-1}" "${STRATA_BATCH_GROUPS:-1}" "${STRATA_RUNTIME:-selected default}"
        echo '  Starting hosting pauses the miner and OctaSpace automatically (never an active rental).'
        echo '  Heretic and SC117 serve trusted-LAN 0.0.0.0:8080; Heretic permits explicit loopback hosting.'
        echo '  SC117 .45/.46 are unqualified trials with dedicated state; neither is qualified for normal/production hosting.'
        echo '  Orca Q4_K_S and DeepSeek routes/weights retired by operator request; Swift remains the default.'
        echo '  LAN access is unauthenticated unless an optional hosting key is configured: use only on a trusted, firewalled network.'
        echo '  [9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit'
        echo '  [12] Set next Hive miner - Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots'
        echo '  [13] Set next Hive miner - Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots'
        echo '  [14] Set next Hive miner - Swift 1.5 Heretic RE | Strata 0.1.46 | native 262K | BF16 CPU vision | spec 3 + windows 2 | 8 GiB / 4-history cache | one slot'
        echo '  [17] Set next Hive miner - SC117 IQ3_S — Strata 0.1.46 unqualified CPU-vision trial | --spec 3 + pipeline-windows 2 | one slot | trusted-LAN 8080'
        echo '  Hive selections safely pause OctaSpace, switch the owned Hive LLM, then restore its running state.'
        echo '  No manual miner stop is needed; rentals and manual hosting sessions are never stopped.'
        read -r -p '  Select: ' choice || return 0
        choice="${choice//[[:space:]]/}"
        case "$choice" in
            1|2)
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
            3|4)
                profile=swift15-heretic-re
                [[ "$choice" == 4 ]] && profile=iq3_s-sc117-abliterated
                if [[ "$active" == strata ]]; then
                    [[ "$choice" == 4 ]] && sc117_trial_command --status
                    "$SCRIPT_DIR/v1strata.sh" --status
                    echo 'The running Strata model was left alone. Use [9] only when you are ready to switch.'
                elif [[ "$active" == qwen38 ]] && ! hosting managed-qwen; then
                    echo 'A manual Qwen hosting session is already running; stop it before switching.'
                elif [[ "$active" != none ]] && ! hosting managed-qwen; then
                    echo 'Another/unmanaged LLM is running. Stop it in its own verified lifecycle first.'
                elif [[ "$choice" == 3 ]]; then
                    STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.46 --parallel 1 --batch-groups 1
                else
                    STRATA_SC117_IQ3S_TRIAL=1 STRATA_RUNTIME=0.1.45 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 \
                        STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
                        run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.45 --parallel 1 --batch-groups 1
                fi
                ;;
            q|Q) return 0 ;;
            9) stop_owned_engines ;;
            10) check_update ;;
            11|x|X) return 0 ;;
            12|13|14|17)
                hive_profile=swift15u-q8
                [[ "$choice" == 13 ]] && hive_profile=hauhau-q8-fastmtp-q4kv-xhigh
                [[ "$choice" == 14 ]] && hive_profile=swift15-heretic-re
                [[ "$choice" == 17 ]] && hive_profile=iq3_s-sc117-abliterated-v0146
                set_next_hive_profile "$hive_profile"
                ;;
        esac
        if [[ -t 0 ]]; then
            read -r -p 'Press Enter to return to the menu...' _ || return 0
        fi
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    trap 'release_menu_lease_on_exit || true' EXIT
    if [[ "${1:-}" == --apply-hive-profile ]]; then
        [[ "$EUID" -eq 0 && "$#" -eq 2 ]] || { echo '--apply-hive-profile requires root and one profile' >&2; exit 2; }
        case "$2" in
            swift15u-q8|hauhau-q8-fastmtp-q4kv-xhigh|swift15-heretic-re|iq3_s-sc117-abliterated|iq3_s-sc117-abliterated-v0146) ;;
            *) echo 'Unknown Hive profile; no files changed.' >&2; exit 2 ;;
        esac
        apply_next_hive_profile "$2"
        exit "$?"
    fi
    if [[ "${1:-}" == --run-strata ]]; then
        shift
        run_selected v1strata.sh "$@"
        exit "$?"
    fi
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
