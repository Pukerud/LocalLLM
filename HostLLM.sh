#!/bin/bash

# =========================================================================
# HostLLM.sh — current engine picker
# Launches the SPEED DEMON, Qwen3.8, ExLlamaV3, or general llama.cpp profile.
# Only one engine can run at a time (shared GPU + port 8080).
# =========================================================================

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# HiveOS may enter a root shell automatically via `sudo -s`. Share the
# Qwen3.8 state with the original login user so both shells see one server.
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
QWEN38_STATE_ROOT="${QWEN38_STATE_ROOT:-${qwen38_home}/.local/state/locallm-qwen38}"
SPEED_DEMON_STATE_ROOT="${SPEED_DEMON_STATE_ROOT:-${qwen38_home}/.local/state/locallm-speed-demon}"
EXLLAMA_STATE_ROOT="${EXLLAMA_STATE_ROOT:-${qwen38_home}/.local/state/locallm-exllama}"

# OctaSpace uses the osn.service unit and shares the same three GPUs. HostLLM
# temporarily pauses it while an inference engine is being started, then
# resumes it after that engine has stopped. The marker survives leaving this
# menu so [9] can restore OctaSpace even after HostLLM is reopened.
OCTA_SERVICE="osn.service"
OCTA_STOP_MARKER="${qwen38_home}/.local/state/hostllm/octaspace-stopped"
OCTA_WAS_PAUSED=0
mkdir -p "$(dirname -- "$OCTA_STOP_MARKER")" 2>/dev/null || true

systemctl_hostllm() {
    if [[ "${EUID}" -eq 0 ]]; then
        systemctl "$@"
    else
        sudo systemctl "$@"
    fi
}

octaspace_exists() {
    command -v systemctl >/dev/null 2>&1 || return 1
    [[ -f "/etc/systemd/system/${OCTA_SERVICE}" || -f "/lib/systemd/system/${OCTA_SERVICE}" ]]
}

octaspace_active() {
    octaspace_exists || return 1
    systemctl_hostllm is-active --quiet "$OCTA_SERVICE" >/dev/null 2>&1
}

pause_octaspace() {
    if ! octaspace_exists; then
        rm -f "$OCTA_STOP_MARKER"
        return 0
    fi
    if ! octaspace_active; then
        return 0
    fi

    echo " Stopping OctaSpace (${OCTA_SERVICE}) before starting HostLLM..."
    if ! systemctl_hostllm stop "$OCTA_SERVICE" || octaspace_active; then
        echo -e "  ${RED}Could not stop OctaSpace safely; engine start cancelled.${RESET}"
        return 1
    fi
    if ! : > "$OCTA_STOP_MARKER"; then
        echo -e "  ${RED}Could not record OctaSpace pause state; restarting it now.${RESET}"
        systemctl_hostllm start "$OCTA_SERVICE" >/dev/null 2>&1 || true
        return 1
    fi
    OCTA_WAS_PAUSED=1
    echo -e "  ${GREEN}OctaSpace paused.${RESET}"
}

resume_octaspace() {
    if ! octaspace_exists; then
        rm -f "$OCTA_STOP_MARKER"
        OCTA_WAS_PAUSED=0
        return 0
    fi
    if [[ ! -f "$OCTA_STOP_MARKER" && "$OCTA_WAS_PAUSED" -ne 1 ]]; then
        return 0
    fi
    if octaspace_active; then
        rm -f "$OCTA_STOP_MARKER"
        OCTA_WAS_PAUSED=0
        echo -e "  ${GREEN}OctaSpace is already running.${RESET}"
        return 0
    fi

    echo " Starting OctaSpace (${OCTA_SERVICE})..."
    if systemctl_hostllm start "$OCTA_SERVICE" && octaspace_active; then
        rm -f "$OCTA_STOP_MARKER"
        OCTA_WAS_PAUSED=0
        echo -e "  ${GREEN}OctaSpace resumed.${RESET}"
        return 0
    fi
    echo -e "  ${RED}OctaSpace failed to start; check: systemctl status ${OCTA_SERVICE}${RESET}"
    return 1
}

run_engine_with_octaspace() {
    local rc=0
    if ! pause_octaspace; then
        sleep 2
        return 1
    fi

    "$@" || rc=$?

    # A launcher may return to this menu while deliberately leaving its
    # server running. Keep OctaSpace paused in that case; [9] resumes it
    # after the server is actually stopped.
    if [[ "$(detect_engine)" == "none" ]]; then
        if ! resume_octaspace && [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
    else
        echo " OctaSpace remains paused while a HostLLM engine is running."
    fi
    return "$rc"
}

qwen_server_pid_running() {
    local pid="$1" cmdline=""
    # kill -0 fails when Qwen is running under the other shared-shell user.
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "$cmdline" == *llama-server* ]]
}

detect_engine() {
    local qwen_pid=""
    if [[ -s "${QWEN38_STATE_ROOT}/server.pid" ]]; then
        qwen_pid=$(cat "${QWEN38_STATE_ROOT}/server.pid" 2>/dev/null || true)
    fi
    if [[ "$qwen_pid" =~ ^[0-9]+$ ]] && qwen_server_pid_running "$qwen_pid"; then
        echo "qwen38"
    elif pgrep -f "llama-server" > /dev/null 2>&1; then
        echo "llamacpp"
    elif docker ps --filter "name=^/tabbyapi-exllama$" --format '{{.Names}}' 2>/dev/null | grep -q "^tabbyapi-exllama$"; then
        echo "exllama"
    elif docker ps --filter "name=^/vllm-speed-demon$" --format '{{.Names}}' 2>/dev/null | grep -q "^vllm-speed-demon$"; then
        echo "speeddemon"
    elif docker ps --filter "name=vllm-hostllm" --format '{{.Names}}' 2>/dev/null | grep -q "vllm-hostllm"; then
        echo "vllm"
    else
        echo "none"
    fi
}

exllama_speed_display() {
    local cache="${EXLLAMA_STATE_ROOT}/speed-results.tsv" row date context coding story average
    if [[ ! -r "$cache" ]]; then
        printf 'speed not tested'
        return 0
    fi
    row="$(awk -F'|' '$1 == "exllama-qwen38-sc6-h6-v6" { row = $0 } END { print row }' "$cache")"
    if [[ -z "$row" ]]; then
        printf 'speed not tested'
        return 0
    fi
    IFS='|' read -r _ date context coding story average <<< "$row"
    if [[ "$date" == "$(date +%Y-%m-%d)" ]]; then
        printf '~%s tok/s' "$average"
    else
        printf '~%s tok/s (%s)' "$average" "$date"
    fi
}

get_server_info() {
    case "$(detect_engine)" in
        qwen38)
            [[ -f "${QWEN38_STATE_ROOT}/server.info" ]] && cat "${QWEN38_STATE_ROOT}/server.info"
            ;;
        speeddemon)
            [[ -f "${SPEED_DEMON_STATE_ROOT}/server.info" ]] && cat "${SPEED_DEMON_STATE_ROOT}/server.info"
            ;;
        exllama)
            [[ -f "${EXLLAMA_STATE_ROOT}/server.info" ]] && cat "${EXLLAMA_STATE_ROOT}/server.info"
            ;;
        llamacpp|vllm)
            [[ -f "${SCRIPT_DIR}/.server_info" ]] && cat "${SCRIPT_DIR}/.server_info"
            ;;
        *)
            echo ""
            ;;
    esac
}

check_update() {
    echo ""
    echo "  Checking for updates..."
    cd "${SCRIPT_DIR}"

    local -a GIT_CMD=(git -c "safe.directory=${SCRIPT_DIR}")
    if [[ "${EUID}" -ne 0 && ! -w "${SCRIPT_DIR}/.git" ]]; then
        GIT_CMD=(sudo git -c "safe.directory=${SCRIPT_DIR}")
    fi

    if ! "${GIT_CMD[@]}" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo -e "  ${RED}Not a git repository. Cannot check for updates.${RESET}"
        sleep 2
        return
    fi

    LOCAL_HASH=$("${GIT_CMD[@]}" rev-parse HEAD)
    BRANCH=$("${GIT_CMD[@]}" rev-parse --abbrev-ref HEAD)

    if ! "${GIT_CMD[@]}" fetch origin; then
        echo -e "  ${RED}Failed to fetch from remote.${RESET}"
        sleep 2
        return
    fi

    REMOTE_HASH=$("${GIT_CMD[@]}" rev-parse "origin/${BRANCH}" 2>/dev/null)
    if [[ -z "$REMOTE_HASH" ]]; then
        echo -e "  ${RED}Could not determine remote branch.${RESET}"
        sleep 2
        return
    fi

    if [[ "$LOCAL_HASH" == "$REMOTE_HASH" ]]; then
        echo -e "  ${GREEN}Already up to date.${RESET}"
        sleep 2
        return
    fi

    echo -e "  ${YELLOW}New commits found on origin/${BRANCH}:${RESET}"
    echo ""
    "${GIT_CMD[@]}" log --oneline "${LOCAL_HASH}..${REMOTE_HASH}"
    echo ""
    read -p "  Update now? (y/N): " upd
    upd=$(echo "$upd" | tr -d '[:space:]')
    if [[ "$upd" == "y" || "$upd" == "Y" ]]; then
        if "${GIT_CMD[@]}" pull origin "$BRANCH"; then
            echo -e "  ${GREEN}Updated. Restarting...${RESET}"
            sleep 1
            exec "$0"
        else
            echo -e "  ${RED}Update failed. Check for conflicts.${RESET}"
            sleep 2
        fi
    fi
}

stop_all() {
    local rc=0
    echo ""
    if [[ -x "${SCRIPT_DIR}/v1speeddemon.sh" ]]; then
        "${SCRIPT_DIR}/v1speeddemon.sh" --stop >/dev/null 2>&1 || true
    fi
    if [[ -x "${SCRIPT_DIR}/v1exllama.sh" ]]; then
        "${SCRIPT_DIR}/v1exllama.sh" --stop >/dev/null 2>&1 || true
    fi
    if [[ -x "${SCRIPT_DIR}/v1qwen38.sh" ]]; then
        "${SCRIPT_DIR}/v1qwen38.sh" --stop >/dev/null 2>&1 || true
    fi
    echo " Stopping llama-server..."
    pkill -f "llama-server" 2>/dev/null && echo "   llama-server killed." || echo "   (not running)"
    echo " Cleaning up any old vLLM container..."
    docker rm -f vllm-hostllm 2>/dev/null || true
    echo "   vLLM cleanup complete."
    rm -f "${SCRIPT_DIR}/.server_info" "${SCRIPT_DIR}/.server_compose"
    if [[ "$(detect_engine)" == "none" ]]; then
        resume_octaspace || rc=1
    fi
    echo ""
    echo -e " ${GREEN}All engines stopped.${RESET}"
    sleep 1
    return "$rc"
}

while true; do
    clear
    active=$(detect_engine)
    info=$(get_server_info)

    echo "=========================================================="
    echo "  HostLLM — Engine Picker"
    echo "=========================================================="
    echo ""

    if [[ "$active" == "qwen38" ]]; then
        echo -e "  Status:  ${GREEN}Qwen3.8 server RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "llamacpp" ]]; then
        echo -e "  Status:  ${GREEN}llama.cpp RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "speeddemon" ]]; then
        echo -e "  Status:  ${GREEN}SPEED DEMON RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "exllama" ]]; then
        echo -e "  Status:  ${GREEN}ExLlamaV3 RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "vllm" ]]; then
        echo -e "  Status:  ${GREEN}external vLLM container RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    else
        echo -e "  Status:  ${YELLOW}No engine running${RESET}"
    fi

    echo ""
    echo "  Quick Start:"
    echo "  ────────────"
    echo -e "  ${BOLD}[1]${RESET} SPEED DEMON  ⚡ Qwen3.8 AWQ INT4 + FP8 DFlash2 │ ~123 code* / ~67 tools / ~62 prose tok/s"
    echo -e "      Native 262K │ 2x RTX 3090 │ target image input ON; FP8 draft text-only; video unvalidated"
    echo -e "  ${BOLD}[Q]${RESET} Qwen3.8-27B  ⚡ vision │ native 262K │ FastMTP"
    echo -e "      HauhauCS and Flash-Next profiles with cached speed results"
    echo -e "  ${BOLD}[2]${RESET} ExLlamaV3   ⚡ 6bpw EXL3 vision │ native 262K │ TabbyAPI │ $(exllama_speed_display)"
    echo -e "      SC_6.00bpw_H6_V6 │ image input ON │ autosplit RTX 30-series GPUs"
    echo ""
    echo "  Engines (manual):"
    echo "  ─────────────────"
    echo -e "  ${BOLD}[3]${RESET} llama.cpp  ik_llama.cpp — general GGUF fallback"
    echo ""
    echo "  Removed Qwen3.6-era engines and tests are documented in README.md."
    echo ""
    echo "  ─────────────────────────"
    echo -e "  ${BOLD}[9]${RESET} Kill All          ${BOLD}[10]${RESET} Update          ${BOLD}[11]${RESET} Exit"
    echo ""

    read -p "  Select: " choice
    choice=$(echo "$choice" | tr -d '[:space:]')

    case $choice in
        1)
            if [[ "$active" == "speeddemon" ]]; then
                cd "${SCRIPT_DIR}"
                ./v1speeddemon.sh --dashboard
                if [[ "$(detect_engine)" == "none" ]]; then
                    resume_octaspace
                fi
            elif [[ "$active" != "none" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            else
                if [[ ! -x "${SCRIPT_DIR}/v1speeddemon.sh" ]]; then
                    echo ""
                    echo -e "  ${RED}v1speeddemon.sh not found or not executable.${RESET}"
                    sleep 2
                    continue
                fi
                cd "${SCRIPT_DIR}"
                run_engine_with_octaspace "${SCRIPT_DIR}/v1speeddemon.sh" --quickstart
                speed_rc=$?
                [[ "$speed_rc" -eq 42 ]] && exit 0
            fi
            ;;
        q|Q)
            if [[ "$active" == "qwen38" ]]; then
                cd "${SCRIPT_DIR}"
                ./v1qwen38.sh --dashboard
                if [[ "$(detect_engine)" == "none" ]]; then
                    resume_octaspace
                fi
            elif [[ "$active" != "none" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            else
                if [[ ! -x "${SCRIPT_DIR}/v1qwen38.sh" ]]; then
                    echo ""
                    echo -e "  ${RED}v1qwen38.sh not found or not executable.${RESET}"
                    sleep 2
                    continue
                fi
                cd "${SCRIPT_DIR}"
                run_engine_with_octaspace "${SCRIPT_DIR}/v1qwen38.sh" --quickstart
                qwen_rc=$?
                [[ "$qwen_rc" -eq 42 ]] && exit 0
            fi
            ;;
        2)
            if [[ "$active" == "exllama" ]]; then
                cd "${SCRIPT_DIR}"
                ./v1exllama.sh --dashboard
                if [[ "$(detect_engine)" == "none" ]]; then
                    resume_octaspace
                fi
            elif [[ "$active" != "none" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            else
                if [[ ! -x "${SCRIPT_DIR}/v1exllama.sh" ]]; then
                    echo ""
                    echo -e "  ${RED}v1exllama.sh not found or not executable.${RESET}"
                    sleep 2
                    continue
                fi
                cd "${SCRIPT_DIR}"
                run_engine_with_octaspace "${SCRIPT_DIR}/v1exllama.sh" --quickstart
                exllama_rc=$?
                [[ "$exllama_rc" -eq 42 ]] && exit 0
            fi
            ;;
        3)
            if [[ "$active" != "none" && "$active" != "llamacpp" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1llama_cpp.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1llama_cpp.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            run_engine_with_octaspace "${SCRIPT_DIR}/v1llama_cpp.sh"
            llama_rc=$?
            [[ "$llama_rc" -eq 42 ]] && exit 0
            ;;
        9)
            stop_all
            ;;
        10)
            check_update
            ;;
        11)
            exit 0
            ;;
        *)
            ;;
    esac
done
