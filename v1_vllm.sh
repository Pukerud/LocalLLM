#!/bin/bash

# =========================================================================
# vLLM DASHBOARD v1.0
# Runs Qwen3.6-27B via vLLM Docker with Genesis patches.
# Based on club-3090 tuned configs for RTX 3090/4090 (24 GB).
# =========================================================================

set +m

VLLM_MODELS_DIR="vllm_models"
COMPOSE_DIR="${VLLM_MODELS_DIR}/compose"
CONTAINER_NAME="vllm-hostllm"
MODEL_NAME="qwen3.6-27b-autoround"

# Colors
GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BLUE=$(tput setaf 4); BOLD=$(tput bold); RESET=$(tput sgr0)

mkdir -p "$VLLM_MODELS_DIR"
mkdir -p "$COMPOSE_DIR"

# --- Dependency checks ---
for cmd in curl jq docker; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd"
    fi
done

if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Start it first: sudo systemctl start docker"
    sleep 3
    exit 1
fi

# --- Terminal handling ---
MONITOR_PID=""

cleanup() {
    kill -9 "$MONITOR_PID" > /dev/null 2>&1
    wait "$MONITOR_PID" > /dev/null 2>&1
    tput csr 0 "$(tput lines)"
    tput cnorm
    echo ""
    exit
}
trap cleanup INT TERM EXIT

# --- GPU / system stats ---
get_cpu_usage() {
    read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
    cpu_active_prev=$((user+nice+system+irq+softirq+steal))
    cpu_total_prev=$((user+nice+system+idle+iowait+irq+softirq+steal))
    sleep 0.5
    read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
    cpu_active_cur=$((user+nice+system+irq+softirq+steal))
    cpu_total_cur=$((user+nice+system+idle+iowait+irq+softirq+steal))
    cpu_total_diff=$((cpu_total_cur - cpu_total_prev))
    cpu_active_diff=$((cpu_active_cur - cpu_active_prev))
    if [[ "$cpu_total_diff" -eq 0 ]]; then echo "0"; else echo $(( (cpu_active_diff * 100) / cpu_total_diff )); fi
}

update_dashboard_stats() {
    if command -v nvidia-smi > /dev/null 2>&1; then
        stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
        IFS=',' read -r gpu_load vram_used vram_total gpu_temp <<< "$stats"
        gpu_load=$(echo "$gpu_load" | tr -d ' ')
        vram_used=$(echo "$vram_used" | tr -d ' ')
        vram_total=$(echo "$vram_total" | tr -d ' ')
        gpu_temp=$(echo "$gpu_temp" | tr -d ' ')
        if [[ "$vram_total" -gt 0 ]]; then vram_pct=$(( (vram_used * 100) / vram_total )); else vram_pct=0; fi
        vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
        vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
        if [[ "$vram_pct" -ge 90 ]]; then c_vram="\033[1;31m"; elif [[ "$vram_pct" -ge 50 ]]; then c_vram="\033[1;33m"; else c_vram="\033[1;32m"; fi
    else
        gpu_load="N/A"; gpu_temp="-"; vram_used_gb="0"; vram_total_gb="0"; vram_pct="0"; c_vram="\033[0m"
    fi

    cpu_pct=$(get_cpu_usage)
    if [[ "$cpu_pct" -ge 80 ]]; then c_cpu="\033[1;31m"; elif [[ "$cpu_pct" -ge 50 ]]; then c_cpu="\033[1;33m"; else c_cpu="\033[1;32m"; fi

    # Check vLLM container status
    container_status=$(docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Status}}' 2>/dev/null)
    if [[ -n "$container_status" ]]; then
        if [[ -f ".server_info" ]]; then
            ACTIVE_INFO=$(cat .server_info)
            SERVER_STATUS="\033[1;32mRUNNING: ${ACTIVE_INFO}\033[0m"
        else
            SERVER_STATUS="\033[1;32mRUNNING (${container_status})\033[0m"
        fi
    else
        SERVER_STATUS="\033[1;31mSTOPPED\033[0m"
        # Also check if llama-server is running (from the other engine)
        if pgrep -f "llama-server" > /dev/null 2>&1; then
            SERVER_STATUS="\033[1;33mSTOPPED (llama.cpp is running!)\033[0m"
        fi
    fi

    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${BOLD}vLLM (Docker)${RESET}    |  SERVER: ${SERVER_STATUS}\033[K"
    tput cup 3 0
    echo -e "   CPU: ${c_cpu}${cpu_pct}%${RESET}   |   GPU: ${gpu_load}%   |   Temp: ${gpu_temp}°C\033[K"
    tput cup 4 0
    echo -e "   VRAM: ${c_vram}${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)${RESET}\033[K"
    tput rc
}

monitor_loop() {
    tput civis
    while true; do
        update_dashboard_stats 2>/dev/null
    done
}

setup_scroll_region() {
    clear
    echo "==========================================================================================================================="
    echo "   vLLM DASHBOARD v1.0"
    echo "==========================================================================================================================="
    tput cup 5 0
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

# --- vLLM server management ---

is_server_running() {
    docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER_NAME"
}

is_llamacpp_running() {
    pgrep -f "llama-server" > /dev/null 2>&1
}

start_vllm_server() {
    local compose_file="$1"
    local preset_name="$2"
    local preset_info="$3"

    echo ""
    echo -e " ${CYAN}>>> START vLLM SERVER — ${preset_name} <<<${RESET}"

    if is_llamacpp_running; then
        echo -e " ${RED}llama.cpp is running! Stop it first (use HostLLM [3] Kill All).${RESET}"
        sleep 3
        return
    fi

    if is_server_running; then
        echo " vLLM server is already running! Stop it first [5]."
        sleep 2
        return
    fi

    # Check prerequisites
    if [[ ! -f "${COMPOSE_DIR}/${compose_file}" ]]; then
        echo -e " ${RED}Error: Compose file not found: ${COMPOSE_DIR}/${compose_file}${RESET}"
        echo " Run Install/Update [0] first."
        sleep 3
        return
    fi

    if [[ ! -f "${VLLM_MODELS_DIR}/qwen3.6-27b-autoround-int4/config.json" ]]; then
        echo -e " ${RED}Error: Model not found in ${VLLM_MODELS_DIR}/qwen3.6-27b-autoround-int4/${RESET}"
        echo " Run Download Model [7] first."
        sleep 3
        return
    fi

    if [[ ! -d "${VLLM_MODELS_DIR}/genesis/vllm/_genesis" ]]; then
        echo -e " ${YELLOW}Warning: Genesis patches not found. Run Install/Update [0] for best results.${RESET}"
        echo " Starting without Genesis (may be slower / less stable)."
        sleep 2
    fi

    echo ""
    echo " Starting vLLM Docker container..."
    echo "   Preset:   ${preset_name}"
    echo "   Info:     ${preset_info}"
    echo "   Port:     8080"
    echo "   Model:    ${MODEL_NAME}"
    echo ""

    cd "$COMPOSE_DIR"
    docker compose -f "$compose_file" up -d 2>&1
    local rc=$?
    cd - > /dev/null

    if [[ $rc -ne 0 ]]; then
        echo -e " ${RED}Docker compose failed (exit code $rc).${RESET}"
        echo " Check: docker logs ${CONTAINER_NAME}"
        sleep 3
        return
    fi

    echo ""
    echo " Waiting for server to load (up to 180s)..."

    local loaded=0
    for i in $(seq 1 180); do
        # Check if container is still running
        if ! docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER_NAME"; then
            echo ""
            echo -e " ${RED}Container exited unexpectedly. Logs:${RESET}"
            docker logs "$CONTAINER_NAME" 2>&1 | tail -n 30
            sleep 3
            return
        fi

        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8080/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            loaded=1
            break
        fi

        printf "\r   Waiting... %ds" "$i"
        sleep 1
    done

    echo ""

    if [[ "$loaded" == "1" ]]; then
        echo "$preset_info" > .server_info
        echo "$compose_file" > .server_compose
        echo ""
        echo -e " ${GREEN}Server is loaded and health check returned OK.${RESET}"
        echo "   API: http://localhost:8080/v1/chat/completions"
        echo "   Model name: ${MODEL_NAME}"
        echo ""
        echo " Quick test:"
        echo "   curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":30}'"
    else
        echo ""
        echo -e " ${YELLOW}Server is still loading. It may need more time.${RESET}"
        echo " Check status with: docker logs ${CONTAINER_NAME}"
        echo "$preset_info" > .server_info
        echo "$compose_file" > .server_compose
    fi

    echo ""
    echo " Last 20 lines of container log:"
    echo "------------------------------------------------------------"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -n 20
    echo "------------------------------------------------------------"
    sleep 3
}

stop_vllm_server() {
    echo ""
    echo -e " ${CYAN}>>> STOPPING vLLM SERVER <<<${RESET}"

    if is_server_running; then
        # Determine which compose file was used
        local compose_used=""
        if [[ -f ".server_compose" ]]; then
            compose_used=$(cat .server_compose)
        fi

        if [[ -n "$compose_used" && -f "${COMPOSE_DIR}/${compose_used}" ]]; then
            cd "$COMPOSE_DIR"
            docker compose -f "$compose_used" down 2>&1
            cd - > /dev/null
        else
            # Fallback: stop by container name
            docker stop "${CONTAINER_NAME}" 2>&1
            docker rm "${CONTAINER_NAME}" 2>&1
        fi
        rm -f .server_info .server_compose
        echo " vLLM server stopped."
    else
        echo " vLLM server is not running."
        rm -f .server_info .server_compose
    fi
    sleep 1
}

# --- Benchmark ---

run_benchmark() {
    echo ""
    echo -e " ${CYAN}>>> vLLM QUICK BENCHMARK <<<${RESET}"

    if ! is_server_running; then
        echo " Server is not running! Start it first."
        sleep 2
        return
    fi

    local n_runs=3
    local max_tokens=128
    local prompt="Write a short story about a robot learning to paint. Be creative."

    echo ""
    echo " Running ${n_runs}x benchmark (${max_tokens} tokens each)..."
    echo "-------------------------------------------------------"
    printf " %-6s | %-12s | %-12s | %s\n" "RUN" "TOKENS" "TIME (ms)" "TPS"
    echo "-------------------------------------------------------"

    local total_tps=0
    local success_count=0

    for run in $(seq 1 $n_runs); do
        local start_ns end_ns response tokens elapsed_ms tps

        start_ns=$(date +%s%N)
        response=$(curl -s --max-time 120 http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens},\"stream\":false}" 2>/dev/null)
        end_ns=$(date +%s%N)

        tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

        if [[ -n "$tokens" && "$tokens" -gt 0 && "$elapsed_ms" -gt 0 ]]; then
            tps=$(awk "BEGIN {printf \"%.1f\", $tokens / ($elapsed_ms / 1000)}")
            printf " %-6s | %-12s | %-12s | ${GREEN}%s t/s${RESET}\n" "$run" "$tokens" "$elapsed_ms" "$tps"
            total_tps=$(awk "BEGIN {print $total_tps + $tps}")
            success_count=$((success_count + 1))
        else
            local err_msg
            err_msg=$(echo "$response" | jq -r '.error.message // "unknown"' 2>/dev/null)
            printf " %-6s | %-12s | %-12s | ${RED}FAIL: %s${RESET}\n" "$run" "?" "?" "$err_msg"
        fi
        sleep 1
    done

    echo "-------------------------------------------------------"
    if [[ "$success_count" -gt 0 ]]; then
        local avg_tps
        avg_tps=$(awk "BEGIN {printf \"%.1f\", $total_tps / $success_count}")
        echo -e " Average: ${GREEN}${avg_tps} t/s${RESET} over ${success_count} successful runs"
    else
        echo -e " ${RED}All benchmark runs failed. Check server logs [8].${RESET}"
    fi
    echo ""
    echo " Note: This measures TOTAL time (prefill + decode)."
    echo " Pure decode TPS is higher. For streaming benchmarks,"
    echo " use an external tool like vLLM's benchmark_serving.py."
    echo "-------------------------------------------------------"
}

# --- Model listing ---

list_models() {
    local models=()
    for d in "$VLLM_MODELS_DIR"/*/; do
        [[ ! -d "$d" ]] && continue
        local name
        name=$(basename "$d")
        [[ "$name" == "compose" || "$name" == "genesis" ]] && continue
        [[ -f "$d/config.json" ]] || continue
        models+=("$name")
    done

    if [[ ${#models[@]} -eq 0 ]]; then
        echo -e "   ${YELLOW}(No vLLM models found in ./${VLLM_MODELS_DIR}/)${RESET}"
        echo "   Run Download Model [7] to get started."
        return 1
    fi

    printf "   %-3s %-44s %-10s %s\n" "NR" "MODEL NAME" "SIZE" "STATUS"
    echo "   ------------------------------------------------------------------------------"
    for i in "${!models[@]}"; do
        local m="${models[$i]}"
        local dir="${VLLM_MODELS_DIR}/${m}"
        local size
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        local m_trunc=$(echo "$m" | cut -c1-44)
        local status="Ready"
        if is_server_running; then
            # Check if this model is currently loaded
            if [[ -f ".server_info" ]] && grep -q "$m" .server_info 2>/dev/null; then
                status="${GREEN}Active${RESET}"
            fi
        fi
        printf "   %2d) %-44s [%-6s]  %b\n" "$((i+1))" "$m_trunc" "$size" "$status"
    done
    return 0
}

# --- Download model ---

download_model() {
    echo ""
    echo -e " ${CYAN}>>> DOWNLOAD MODEL <<<${RESET}"

    local target_dir="${VLLM_MODELS_DIR}/qwen3.6-27b-autoround-int4"

    if [[ -f "${target_dir}/config.json" ]]; then
        echo " Model already exists at: ${target_dir}/"
        local cur_size
        cur_size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
        echo " Size: ${cur_size}"
        echo ""
        read -p " Re-download / update? (y/N): " redl
        redl=$(echo "$redl" | tr -d '[:space:]')
        [[ "$redl" == "y" || "$redl" == "Y" ]] || return
    fi

    echo ""
    echo " Downloading Qwen3.6-27B AutoRound INT4 from HuggingFace..."
    echo " Repository: Lorbus/Qwen3.6-27B-int4-AutoRound"
    echo " Size: ~20 GB"
    echo " Target: ${target_dir}/"
    echo ""

    # Check for huggingface-cli
    if ! command -v huggingface-cli > /dev/null 2>&1; then
        echo " Installing huggingface-cli..."
        pip install -U huggingface_hub 2>&1 | tail -n 3
    fi

    echo " Starting download..."
    huggingface-cli download Lorbus/Qwen3.6-27B-int4-AutoRound --local-dir "$target_dir"

    if [[ $? -eq 0 && -f "${target_dir}/config.json" ]]; then
        local final_size
        final_size=$(du -sh "$target_dir" 2>/dev/null | cut -f1)
        echo ""
        echo -e " ${GREEN}Download complete! Size: ${final_size}${RESET}"
    else
        echo ""
        echo -e " ${RED}Download failed. Check your internet connection and try again.${RESET}"
    fi

    echo ""
    read -p " Press Enter to return to menu..."
}

# --- Install / Update ---

install_update() {
    echo ""
    echo -e " ${CYAN}>>> INSTALL / UPDATE vLLM <<<${RESET}"
    echo ""

    # --- Docker image ---
    echo "--- Docker Image ---"
    local image="vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08"
    local has_image
    has_image=$(docker images -q "$image" 2>/dev/null)

    if [[ -n "$has_image" ]]; then
        echo " Image already pulled: ${image}"
        echo -n " Image size: "
        docker images "$image" --format "{{.Size}}" | head -n 1
        read -p " Pull latest / update? (y/N): " pull_choice
        pull_choice=$(echo "$pull_choice" | tr -d '[:space:]')
    else
        echo " vLLM Docker image not found."
        read -p " Pull it now? (~10 GB download) (Y/n): " pull_choice
        pull_choice=$(echo "$pull_choice" | tr -d '[:space:]')
        if [[ -z "$pull_choice" ]]; then pull_choice="Y"; fi
    fi

    if [[ "$pull_choice" == "y" || "$pull_choice" == "Y" ]]; then
        echo " Pulling ${image}..."
        docker pull "$image"
        if [[ $? -eq 0 ]]; then
            echo -e " ${GREEN}Docker image pulled successfully.${RESET}"
        else
            echo -e " ${RED}Docker pull failed.${RESET}"
        fi
    fi

    echo ""

    # --- Genesis patches ---
    echo "--- Genesis Patches ---"
    local genesis_dir="${VLLM_MODELS_DIR}/genesis"

    if [[ -d "${genesis_dir}/vllm/_genesis" ]]; then
        echo " Genesis patches found at: ${genesis_dir}/"
        read -p " Update? (y/N): " gen_choice
        gen_choice=$(echo "$gen_choice" | tr -d '[:space:]')
        if [[ "$gen_choice" == "y" || "$gen_choice" == "Y" ]]; then
            cd "$genesis_dir" && git pull && cd - > /dev/null
        fi
    else
        echo " Genesis patches not found."
        read -p " Clone now? (Y/n): " gen_choice
        gen_choice=$(echo "$gen_choice" | tr -d '[:space:]')
        if [[ -z "$gen_choice" ]]; then gen_choice="Y"; fi

        if [[ "$gen_choice" == "y" || "$gen_choice" == "Y" ]]; then
            rm -rf "$genesis_dir" 2>/dev/null
            git clone https://github.com/Sandermage/genesis-vllm-patches.git "$genesis_dir"
            if [[ $? -eq 0 ]]; then
                echo -e " ${GREEN}Genesis patches cloned successfully.${RESET}"
            else
                echo -e " ${RED}Git clone failed.${RESET}"
            fi
        fi
    fi

    echo ""

    # --- tolist cudagraph patch ---
    echo "--- Cudagraph Patch ---"
    local patch_file="${VLLM_MODELS_DIR}/patch_tolist_cudagraph.py"

    if [[ -f "$patch_file" ]]; then
        echo " Patch already exists."
        read -p " Re-download? (y/N): " patch_choice
        patch_choice=$(echo "$patch_choice" | tr -d '[:space:]')
    else
        echo " Patch not found."
        read -p " Download now? (Y/n): " patch_choice
        patch_choice=$(echo "$patch_choice" | tr -d '[:space:]')
        if [[ -z "$patch_choice" ]]; then patch_choice="Y"; fi
    fi

    if [[ "$patch_choice" == "y" || "$patch_choice" == "Y" ]]; then
        curl -sL -o "$patch_file" \
            "https://raw.githubusercontent.com/noonghunna/club-3090/master/models/qwen3.6-27b/vllm/patches/patch_tolist_cudagraph.py"
        if [[ -f "$patch_file" && -s "$patch_file" ]]; then
            echo -e " ${GREEN}Patch downloaded successfully.${RESET}"
        else
            echo -e " ${RED}Patch download failed.${RESET}"
            rm -f "$patch_file"
        fi
    fi

    echo ""
    echo "-------------------------------------------------------"
    echo -e " ${GREEN}Install/Update complete.${RESET}"
    echo "-------------------------------------------------------"
    echo ""
    read -p " Press Enter to return to menu..."
}

# --- Setup status ---

show_setup_status() {
    echo ""
    echo -e " ${CYAN}>>> SETUP STATUS <<<${RESET}"
    echo "-------------------------------------------------------"

    # Docker
    if docker info > /dev/null 2>&1; then
        echo -e " Docker:              ${GREEN}Running${RESET}"
    else
        echo -e " Docker:              ${RED}Not running${RESET}"
    fi

    # NVIDIA toolkit
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo -e " NVIDIA Toolkit:      ${GREEN}Installed${RESET}"
    else
        echo -e " NVIDIA Toolkit:      ${RED}Not found${RESET}"
    fi

    # vLLM image
    local image="vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08"
    if docker images -q "$image" 2>/dev/null | grep -q .; then
        local img_size
        img_size=$(docker images "$image" --format "{{.Size}}" | head -n 1)
        echo -e " vLLM Image:          ${GREEN}Pulled (${img_size})${RESET}"
    else
        echo -e " vLLM Image:          ${RED}Not pulled${RESET}"
    fi

    # Model
    if [[ -f "${VLLM_MODELS_DIR}/qwen3.6-27b-autoround-int4/config.json" ]]; then
        local model_size
        model_size=$(du -sh "${VLLM_MODELS_DIR}/qwen3.6-27b-autoround-int4" 2>/dev/null | cut -f1)
        echo -e " Model:               ${GREEN}${model_size}${RESET}"
    else
        echo -e " Model:               ${RED}Not downloaded${RESET}"
    fi

    # Genesis
    if [[ -d "${VLLM_MODELS_DIR}/genesis/vllm/_genesis" ]]; then
        echo -e " Genesis Patches:     ${GREEN}Installed${RESET}"
    else
        echo -e " Genesis Patches:     ${RED}Not installed${RESET}"
    fi

    # Cudagraph patch
    if [[ -f "${VLLM_MODELS_DIR}/patch_tolist_cudagraph.py" ]]; then
        echo -e " Cudagraph Patch:     ${GREEN}Installed${RESET}"
    else
        echo -e " Cudagraph Patch:     ${RED}Not installed${RESET}"
    fi

    # Compose files
    local compose_count
    compose_count=$(ls -1 "${COMPOSE_DIR}"/docker-compose.*.yml 2>/dev/null | wc -l)
    if [[ "$compose_count" -gt 0 ]]; then
        echo -e " Compose Files:       ${GREEN}${compose_count} presets${RESET}"
    else
        echo -e " Compose Files:       ${RED}None${RESET}"
    fi

    echo "-------------------------------------------------------"
    echo ""
    read -p " Press Enter to return to menu..."
}

# --- View logs ---

view_logs() {
    echo ""
    echo -e " ${CYAN}>>> vLLM SERVER LOGS <<<${RESET}"
    echo -e " ${YELLOW}Press [Ctrl+C] to return to menu.${RESET}"

    if is_server_running; then
        docker logs -f "$CONTAINER_NAME"
    else
        echo " Server is not running. No logs to show."
        sleep 2
    fi
}

# ========================================================================
# MAIN LOOP
# ========================================================================

echo "Loading vLLM Dashboard..."
setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""

    # Show model table
    list_models

    echo ""
    echo -e " ${CYAN}--- SERVER PRESETS ---${RESET}"
    echo " [1] General Chat     (48K ctx, vision, TQ3 KV, ~50/68 TPS) — safe default"
    echo " [2] IDE/Tools Agent  (75K ctx, no vision, fp8 KV, ~51/65 TPS) — Cline/Cursor"
    echo " [3] Long Vision      (192K ctx, vision, TQ3 KV, ~50/68 TPS) — long docs + images"
    echo " [4] Long Text Only   (205K ctx, no vision, TQ3 KV, ~50/66 TPS) — max context"
    echo " [5] STOP SERVER"
    echo ""
    echo -e " ${CYAN}--- BENCHMARKING ---${RESET}"
    echo " [6] Quick Benchmark (localhost:8080)"
    echo ""
    echo -e " ${CYAN}--- MANAGEMENT ---${RESET}"
    echo " [7] Download Model (Qwen3.6-27B AutoRound INT4)"
    echo " [8] View Server Logs"
    echo " [9] Setup Status"
    echo " [0] INSTALL / UPDATE (Docker image + Genesis patches)"
    echo " [99] Exit"
    echo ""

    tput cnorm
    read -p " Select Action: " action
    action=$(echo "$action" | tr -d '[:space:]')

    case $action in
        1)
            start_vllm_server "docker-compose.default.yml" \
                "General Chat" \
                "VLLM: qwen3.6-27b [default/48K/TQ3+Vis]"
            ;;
        2)
            start_vllm_server "docker-compose.tools-text.yml" \
                "IDE/Tools Agent" \
                "VLLM: qwen3.6-27b [tools-text/75K/fp8]"
            ;;
        3)
            start_vllm_server "docker-compose.long-vision.yml" \
                "Long Vision" \
                "VLLM: qwen3.6-27b [long-vision/192K/TQ3+Vis]"
            ;;
        4)
            start_vllm_server "docker-compose.long-text.yml" \
                "Long Text Only" \
                "VLLM: qwen3.6-27b [long-text/205K/TQ3]"
            ;;
        5)
            stop_vllm_server
            ;;
        6)
            run_benchmark
            read -p " Press Enter to return to menu..."
            ;;
        7)
            download_model
            ;;
        8)
            view_logs
            ;;
        9)
            show_setup_status
            ;;
        0)
            install_update
            ;;
        99)
            cleanup
            ;;
        *)
            ;;
    esac
done
