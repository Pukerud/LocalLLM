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
        # Check which build is running by looking at server info files
        if [[ -f "${SCRIPT_DIR}/.server_info_dflash" ]]; then
            echo "dflash"
        else
            echo "llamacpp"
        fi
    elif docker ps --filter "name=vllm-hostllm" --format '{{.Names}}' 2>/dev/null | grep -q "vllm-hostllm"; then
        echo "vllm"
    else
        echo "none"
    fi
}

get_server_info() {
    if [[ -f "${SCRIPT_DIR}/.server_info" ]]; then
        cat "${SCRIPT_DIR}/.server_info"
    elif [[ -f "${SCRIPT_DIR}/.server_info_dflash" ]]; then
        cat "${SCRIPT_DIR}/.server_info_dflash"
    else
        echo ""
    fi
}

stop_all() {
    echo ""
    echo " Stopping llama-server..."
    pkill -f "llama-server" 2>/dev/null && echo "   llama-server killed." || echo "   (not running)"
    echo " Stopping vLLM container..."
    # Try compose down with saved compose file, fallback to container name
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
    rm -f "${SCRIPT_DIR}/.server_info" "${SCRIPT_DIR}/.server_info_dflash"
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
    elif [[ "$active" == "vllm" ]]; then
        echo -e "  Status:  ${GREEN}vLLM RUNNING${RESET}"
        if [[ -n "$info" ]]; then echo "  Server:  $info"; fi
    else
        echo -e "  Status:  ${YELLOW}No engine running${RESET}"
    fi

    echo ""
    echo "  Engines:"
    echo "  -------"
    echo -e "  ${BOLD}[1]${RESET} llama.cpp       ik_llama.cpp — max context (262K), all GGUF models"
    echo -e "  ${BOLD}[2]${RESET} DFlash llama.cpp buun-llama-cpp — DFlash speculative decoding"
    echo -e "  ${BOLD}[3]${RESET} vLLM            Docker — max throughput (50-127 TPS), tool calls"
    echo ""
    echo "  Controls:"
    echo "  ---------"
    echo -e "  ${BOLD}[4]${RESET} Kill All    Stop whatever is running"
    echo -e "  ${BOLD}[5]${RESET} Exit"
    echo ""

    read -p "  Select: " choice
    choice=$(echo "$choice" | tr -d '[:space:]')

    case $choice in
        1)
            if [[ "$active" != "none" && "$active" != "llamacpp" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [4].${RESET}"
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
            exec ./v1llama_cpp.sh
            ;;
        2)
            if [[ "$active" != "none" && "$active" != "dflash" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [4].${RESET}"
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
            exec ./v1dflash_llama_cpp.sh
            ;;
        3)
            if [[ "$active" != "none" && "$active" != "vllm" ]]; then
                echo ""
                echo -e "  ${RED}${active} is running on port 8080. Stop it first with [4].${RESET}"
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
            exec ./v1_vllm.sh
            ;;
        4)
            stop_all
            ;;
        5)
            exit 0
            ;;
        *)
            ;;
    esac
done
