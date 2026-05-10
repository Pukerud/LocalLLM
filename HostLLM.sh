#!/bin/bash

# =========================================================================
# HostLLM.sh — Top-level engine picker
# Launches either llama.cpp dashboard or vLLM dashboard.
# Only one engine can run at a time (shared GPU + port 8080).
# =========================================================================

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

detect_engine() {
    if pgrep -f "llama-server" > /dev/null 2>&1; then
        if [[ -f "${SCRIPT_DIR}/.server_info_mtp" ]]; then
            echo "mtp"
        elif [[ -f "${SCRIPT_DIR}/.server_info_beellama" ]]; then
            echo "beellama"
        elif [[ -f "${SCRIPT_DIR}/.server_info_dflash" ]]; then
            echo "dflash"
        else
            echo "llamacpp"
        fi
    elif pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" > /dev/null 2>&1; then
        echo "zaya"
    elif docker ps --filter "name=vllm-hostllm" --format '{{.Names}}' 2>/dev/null | grep -q "vllm-hostllm"; then
        echo "vllm"
    else
        echo "none"
    fi
}

get_server_info() {
    if [[ -f "${SCRIPT_DIR}/.server_info_mtp" ]]; then
        cat "${SCRIPT_DIR}/.server_info_mtp"
    elif [[ -f "${SCRIPT_DIR}/.server_info_beellama" ]]; then
        cat "${SCRIPT_DIR}/.server_info_beellama"
    elif [[ -f "${SCRIPT_DIR}/.server_info_zaya" ]]; then
        cat "${SCRIPT_DIR}/.server_info_zaya"
    elif [[ -f "${SCRIPT_DIR}/.server_info" ]]; then
        cat "${SCRIPT_DIR}/.server_info"
    elif [[ -f "${SCRIPT_DIR}/.server_info_dflash" ]]; then
        cat "${SCRIPT_DIR}/.server_info_dflash"
    else
        echo ""
    fi
}

check_update() {
    echo ""
    echo "  Checking for updates..."
    cd "${SCRIPT_DIR}"

    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo -e "  ${RED}Not a git repository. Cannot check for updates.${RESET}"
        sleep 2
        return
    fi

    LOCAL_HASH=$(git rev-parse HEAD)
    BRANCH=$(git rev-parse --abbrev-ref HEAD)

    git fetch origin 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo -e "  ${RED}Failed to fetch from remote.${RESET}"
        sleep 2
        return
    fi

    REMOTE_HASH=$(git rev-parse "origin/${BRANCH}" 2>/dev/null)
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
    git log --oneline "${LOCAL_HASH}..${REMOTE_HASH}"
    echo ""
    read -p "  Update now? (y/N): " upd
    upd=$(echo "$upd" | tr -d '[:space:]')
    if [[ "$upd" == "y" || "$upd" == "Y" ]]; then
        git pull origin "$BRANCH"
        if [[ $? -eq 0 ]]; then
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
    echo ""
    echo " Stopping llama-server..."
    pkill -f "llama-server" 2>/dev/null && echo "   llama-server killed." || echo "   (not running)"
    echo " Stopping vLLM container..."
    local compose_used=""
    if [[ -f "${SCRIPT_DIR}/.server_compose" ]]; then
        compose_used=$(cat "${SCRIPT_DIR}/.server_compose")
    fi
    if [[ -n "$compose_used" && -f "${SCRIPT_DIR}/vllm_models/compose/${compose_used}" ]]; then
        cd "${SCRIPT_DIR}/vllm_models/compose" && docker compose -f "$compose_used" down 2>/dev/null && cd "${SCRIPT_DIR}"
    else
        docker rm -f vllm-hostllm 2>/dev/null || true
    fi
    echo "   vLLM stopped."
    rm -f "${SCRIPT_DIR}/.server_info" "${SCRIPT_DIR}/.server_info_dflash" "${SCRIPT_DIR}/.server_info_mtp" "${SCRIPT_DIR}/.server_info_beellama" "${SCRIPT_DIR}/.server_info_zaya"
    # Also kill ZAYA vLLM process if running
    pids=$(pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" 2>/dev/null)
    if [[ -n "$pids" ]]; then
        for pid in $pids; do kill "$pid" 2>/dev/null; done
        sleep 1
        pids=$(pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" 2>/dev/null)
        if [[ -n "$pids" ]]; then
            for pid in $pids; do kill -9 "$pid" 2>/dev/null; done
        fi
        echo "   ZAYA vLLM stopped."
    fi
    echo ""
    echo -e " ${GREEN}All engines stopped.${RESET}"
    sleep 1
}

while true; do
    clear
    active=$(detect_engine)
    info=$(get_server_info)

    echo "=========================================================="
    echo "  HostLLM — Engine Picker"
    echo "=========================================================="
    echo ""

    if [[ "$active" == "llamacpp" ]]; then
        echo -e "  Status:  ${GREEN}llama.cpp RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "dflash" ]]; then
        echo -e "  Status:  ${GREEN}DFlash llama.cpp RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "mtp" ]]; then
        echo -e "  Status:  ${GREEN}llama.cpp MTP RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "beellama" ]]; then
        echo -e "  Status:  ${GREEN}BeeLlama DFlash RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "zaya" ]]; then
        echo -e "  Status:  ${GREEN}ZAYA1-8B RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    elif [[ "$active" == "vllm" ]]; then
        echo -e "  Status:  ${GREEN}vLLM RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    else
        echo -e "  Status:  ${YELLOW}No engine running${RESET}"
    fi

    echo ""
    echo "  Quick Start (one-click):"
    echo "  ─────────────────────────"
    echo -e "  ${BOLD}[0]${RESET} BeeLlama DFlash  ⚡ up to 105 tok/s │ vision │ reasoning"
    echo -e "      Best on single GPU │ dual GPU runs without draft (Tested on 3090)"
    echo ""
    echo -e "  ${BOLD}[1]${RESET} MTP (Legacy)      ⚡ up to 100 tok/s │ no vision"
    echo -e "      Best on dual GPU │ also works on single GPU (Tested on 4090)"
    echo ""
    echo "  Engines (manual):"
    echo "  ─────────────────"
    echo -e "  ${BOLD}[2]${RESET} llama.cpp         ik_llama.cpp — max context (262K), all GGUF models"
    echo -e "  ${BOLD}[3]${RESET} DFlash            buun-llama-cpp — DFlash speculative decoding"
    echo -e "  ${BOLD}[4]${RESET} vLLM              Docker — max throughput, tool calls"
    echo -e "  ${BOLD}[5]${RESET} Lucebox           lucebox-hub — DDTree speculative decoding"
    echo -e "  ${BOLD}[6]${RESET} MTP               ggml-org/llama.cpp — native MTP speculative decoding"
    echo -e "  ${BOLD}[7]${RESET} BeeLlama          Anbeeld/beellama.cpp — full dashboard (manual control)"
    echo -e "  ${BOLD}[8]${RESET} ZAYA1-8B          ${YELLOW}${BOLD}⚠ EXPERIMENTAL${RESET} — Zyphra vLLM, 8B MoE, can crash on install"
    echo ""
    echo "  ─────────────────────────"
    echo -e "  ${BOLD}[9]${RESET} Kill All          ${BOLD}[10]${RESET} Update          ${BOLD}[11]${RESET} Exit"
    echo ""

    read -p "  Select: " choice
    choice=$(echo "$choice" | tr -d '[:space:]')

    case $choice in
        0)
            if [[ "$active" == "beellama" ]]; then
                # Already running — re-enter full dashboard
                cd "${SCRIPT_DIR}"
                ./v1beellama.sh
                [[ $? -eq 42 ]] && exit 0
            elif [[ "$active" != "none" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            else
                if [[ ! -x "${SCRIPT_DIR}/v1beellama.sh" ]]; then
                    echo ""
                    echo -e "  ${RED}v1beellama.sh not found or not executable.${RESET}"
                    sleep 2
                    continue
                fi
                cd "${SCRIPT_DIR}"
                ./v1beellama.sh --quickstart
                [[ $? -eq 42 ]] && exit 0
            fi
            ;;
        1)
            if [[ "$active" == "mtp" ]]; then
                # Already running — re-enter full dashboard
                cd "${SCRIPT_DIR}"
                ./v1llama_mtp.sh
                [[ $? -eq 42 ]] && exit 0
            elif [[ "$active" != "none" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            else
                if [[ ! -x "${SCRIPT_DIR}/v1llama_mtp.sh" ]]; then
                    echo ""
                    echo -e "  ${RED}v1llama_mtp.sh not found or not executable.${RESET}"
                    sleep 2
                    continue
                fi
                cd "${SCRIPT_DIR}"
                ./v1llama_mtp.sh --quickstart
                [[ $? -eq 42 ]] && exit 0
            fi
            ;;
        2)
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
            ./v1llama_cpp.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        3)
            if [[ "$active" != "none" && "$active" != "dflash" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1dflash_llama_cpp.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1dflash_llama_cpp.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1dflash_llama_cpp.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        4)
            if [[ "$active" != "none" && "$active" != "vllm" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1_vllm.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1_vllm.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1_vllm.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        5)
            if [[ "$active" != "none" && "$active" != "lucebox" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1lucebox.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1lucebox.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1lucebox.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        6)
            if [[ "$active" != "none" && "$active" != "mtp" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1llama_mtp.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1llama_mtp.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1llama_mtp.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        7)
            if [[ "$active" != "none" && "$active" != "beellama" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1beellama.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1beellama.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1beellama.sh
            [[ $? -eq 42 ]] && exit 0
            ;;
        8)
            if [[ "$active" != "none" && "$active" != "zaya" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [9].${RESET}"
                sleep 2
                continue
            fi
            if [[ ! -x "${SCRIPT_DIR}/v1zaya.sh" ]]; then
                echo ""
                echo -e "  ${RED}v1zaya.sh not found or not executable.${RESET}"
                sleep 2
                continue
            fi
            echo ""
            echo -e "  ${YELLOW}${BOLD}⚠ WARNING — EXPERIMENTAL${RESET}"
            echo -e "  ${YELLOW}Zyphra's vLLM fork can HARD-CRASH the machine during install.${RESET}"
            echo -e "  ${YELLOW}Use at your own risk. Consider a disposable VM or container.${RESET}"
            echo ""
            read -p "  Continue anyway? (y/N): " confirm_zaya
            if [[ "$confirm_zaya" != "y" && "$confirm_zaya" != "Y" ]]; then
                echo "  Cancelled."
                sleep 1
                continue
            fi
            cd "${SCRIPT_DIR}"
            ./v1zaya.sh
            [[ $? -eq 42 ]] && exit 0
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
