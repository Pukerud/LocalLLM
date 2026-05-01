#!/bin/bash

# =========================================================================
# Lucebox DFlash Dashboard v1.1
# Uses Luce-Org/lucebox-hub — DDTree speculative decoding with safetensors draft
# https://github.com/Luce-Org/lucebox-hub
#
# ~104 tok/s on RTX 4090 with Qwen3.6-27B IQ4_XS + matched DFlash draft
# OpenAI-compatible server on port 8080
# =========================================================================

set +m
MODELS_DIR="llama_models"
LUCE_DIR="lucebox-hub/dflash"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

declare -A speed_cache
SELECTED_MODEL=""
mkdir -p "$MODELS_DIR"

# ── Dependencies ─────────────────────────────────────────────────────────

for cmd in curl jq python3 cmake; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd"
    fi
done

if ! command -v /usr/local/cuda/bin/nvcc > /dev/null 2>&1; then
    echo "Warning: CUDA toolkit (12+) not found at /usr/local/cuda/bin/nvcc."
    sleep 2
fi

for pkg in fastapi uvicorn transformers; do
    python3 -c "import $pkg" 2>/dev/null || {
        echo "Installing Python dependency: $pkg..."
        pip install "$pkg" 2>/dev/null
    }
done

# ── Build check ──────────────────────────────────────────────────────────

check_build() {
    if [[ ! -x "${SCRIPT_DIR}/${LUCE_DIR}/build/test_dflash" ]]; then
        echo ""
        echo "  Lucebox binary not found. Building..."
        echo ""
        export PATH="/usr/local/cuda/bin:$PATH"
        cd "${SCRIPT_DIR}/${LUCE_DIR}" || { echo "  Error: ${LUCE_DIR} not found. Did you clone the repo?"; sleep 3; return 1; }
        cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CXX_FLAGS="-I/usr/local/cuda/include" 2>&1 | tail -3
        cmake --build build --target test_dflash test_generate -j$(nproc) 2>&1 | tail -5
        if [[ ! -x "build/test_dflash" ]]; then
            echo "  Build failed. Check compile errors above."
            sleep 3
            return 1
        fi
        echo "  Build successful!"
        cd "${SCRIPT_DIR}"
    fi
}

# ── Draft model check ───────────────────────────────────────────────────

check_draft() {
    local draft_file="${SCRIPT_DIR}/${LUCE_DIR}/models/draft/model.safetensors"
    if [[ ! -f "$draft_file" ]]; then
        echo ""
        echo "  DFlash draft model not found. Downloading z-lab/Qwen3.6-27B-DFlash (~3.3GB)..."
        echo ""
        mkdir -p "${SCRIPT_DIR}/${LUCE_DIR}/models/draft"
        python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('z-lab/Qwen3.6-27B-DFlash', 'model.safetensors', local_dir='${SCRIPT_DIR}/${LUCE_DIR}/models/draft/')" 2>&1
        if [[ ! -f "$draft_file" ]]; then
            echo "  Download failed. You may need to set HF_TOKEN for gated access."
            sleep 3
            return 1
        fi
    fi
}

# ── Dashboard monitoring ─────────────────────────────────────────────────

get_cpu_usage() {
    read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
    local active=$((user+nice+system+irq+softirq+steal))
    local total=$((user+nice+system+idle+iowait+irq+softirq+steal))
    sleep 0.5
    read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
    local active2=$((user+nice+system+irq+softirq+steal))
    local total2=$((user+nice+system+idle+iowait+irq+softirq+steal))
    local td=$((total2 - total))
    local ad=$((active2 - active))
    if [[ "$td" -eq 0 ]]; then echo "0"; else echo $(( (ad * 100) / td )); fi
}

cleanup() {
    kill -9 "$MONITOR_PID" > /dev/null 2>&1
    wait "$MONITOR_PID" > /dev/null 2>&1
    echo ""
    exit
}
trap cleanup INT TERM EXIT

update_dashboard() {
    local info_file="${SCRIPT_DIR}/.server_info_lucebox"
    local server_status="  ${YELLOW}STOPPED${RESET}"
    local active_model=""
    local active_port="8080"
    local active_ctx=""

    if [[ -f "$info_file" ]]; then
        source "$info_file"
        if pgrep -f "scripts/server.py.*--port ${PORT:-8080}" > /dev/null 2>&1; then
            server_status="  ${GREEN}RUNNING: ${MODEL} on port ${PORT:-8080}${RESET}"
            active_model="$MODEL"
            active_port="${PORT:-8080}"
            active_ctx="${MAX_CTX}"
        else
            server_status="  ${RED}CRASHED${RESET}"
        fi
    fi

    # GPU stats
    if command -v nvidia-smi > /dev/null 2>&1; then
        stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
        IFS=',' read -r gpu_load vram_used vram_total gpu_temp <<< "$stats"
        gpu_load=$(echo "$gpu_load" | tr -d ' ')
        vram_used=$(echo "$vram_used" | tr -d ' ')
        vram_total=$(echo "$vram_total" | tr -d ' ')
        gpu_temp=$(echo "$gpu_temp" | tr -d ' ')
        vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
        vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
        if [[ "$vram_total" -gt 0 ]]; then vram_pct=$(( (vram_used * 100) / vram_total )); else vram_pct=0; fi
        if [[ "$vram_pct" -ge 90 ]]; then c_vram="${RED}"; elif [[ "$vram_pct" -ge 50 ]]; then c_vram="${YELLOW}"; else c_vram="${GREEN}"; fi
    else
        gpu_load="N/A"; gpu_temp="-"; vram_used_gb="0"; vram_total_gb="0"; vram_pct=0; c_vram=""
    fi

    cpu_pct=$(get_cpu_usage)
    if [[ "$cpu_pct" -ge 80 ]]; then c_cpu="${RED}"; elif [[ "$cpu_pct" -ge 50 ]]; then c_cpu="${YELLOW}"; else c_cpu="${GREEN}"; fi

    # RAM stats
    ram_line=$(free -g | awk '/^Mem:/{printf "%d/%d GB", $3, $2}')

    echo "  ──────────────────────────────────────────────────────────"
    echo -e "  Server:  $server_status"
    if [[ -n "$active_ctx" ]]; then
        echo -e "  Context: ${active_ctx} tokens"
    fi
    echo -e "  GPU: ${c_cpu}${gpu_load}%${RESET} load | ${c_vram}VRAM ${vram_used_gb}/${vram_total_gb} GB (${vram_pct}%)${RESET} | Temp: ${gpu_temp}°C"
    echo -e "  CPU: ${c_cpu}${cpu_pct}%${RESET} | RAM: ${ram_line}"
    if [[ -n "$active_model" ]]; then
        echo -e "  ${CYAN}Endpoint: http://localhost:${active_port}/v1/chat/completions${RESET}"
    fi
    echo "  ──────────────────────────────────────────────────────────"
}

# ── Server control ───────────────────────────────────────────────────────

start_server() {
    local model_path="$1"
    local port="${2:-8080}"
    local max_ctx="$3"

    if pgrep -f "scripts/server.py.*--port ${port}" > /dev/null 2>&1; then
        echo "  Server already running on port ${port}. Stop it first."
        sleep 2
        return
    fi

    # Kill any stale test_dflash processes holding GPU
    pkill -f "test_dflash" 2>/dev/null
    pkill -f "test_generate" 2>/dev/null
    sleep 1

    local draft_path="${SCRIPT_DIR}/${LUCE_DIR}/models/draft"
    local bin_path="${SCRIPT_DIR}/${LUCE_DIR}/build/test_dflash"
    local server_script="${SCRIPT_DIR}/${LUCE_DIR}/scripts/server.py"

    if [[ ! -f "$server_script" ]]; then
        echo "  Error: server.py not found at ${server_script}"
        sleep 2
        return
    fi

    # Save server info
    cat > "${SCRIPT_DIR}/.server_info_lucebox" <<EOF
MODEL=${model_path##*/}
PORT=${port}
MAX_CTX=${max_ctx}
EOF

    echo "  Starting Lucebox DFlash server..."
    echo "  Model: ${model_path##*/}"
    echo "  Draft: ${draft_path}/model.safetensors"
    echo "  Port:  ${port}"
    echo "  Max context: ${max_ctx}"
    echo ""

    # Start server in background
    (
        cd "${SCRIPT_DIR}/${LUCE_DIR}"
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="build/deps/llama.cpp/ggml/src:build/deps/llama.cpp/ggml/src/ggml-cuda"
        python3 scripts/server.py \
            --target "${model_path}" \
            --draft "${draft_path}" \
            --bin "${bin_path}" \
            --port "${port}" \
            --max-ctx "${max_ctx}" \
            --daemon \
            > "${SCRIPT_DIR}/server_lucebox.log" 2>&1
    ) &

    echo "  Waiting for server..."
    local retries=0
    while ! curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; do
        sleep 2
        retries=$((retries + 1))
        if [[ $retries -gt 30 ]]; then
            echo "  Server didn't start within 60s. Check server_lucebox.log"
            sleep 3
            return
        fi
    done
    echo "  Server ready!"
    sleep 1
}

stop_server() {
    echo ""
    echo "  Stopping Lucebox DFlash server..."
    pkill -f "scripts/server.py" 2>/dev/null && echo "  Server killed." || echo "  (not running)"
    pkill -f "test_dflash" 2>/dev/null
    pkill -f "test_generate" 2>/dev/null
    rm -f "${SCRIPT_DIR}/.server_info_lucebox"
    sleep 1
}

# ── Model selection ──────────────────────────────────────────────────────

select_model() {
    echo ""
    echo "  Select target model:"
    echo "  ────────────────────────────────────────"

    local models=()
    local i=1

    for f in "${SCRIPT_DIR}/${MODELS_DIR}"/Qwen3.6-27B-*.gguf; do
        if [[ -f "$f" ]]; then
            local name=$(basename "$f")
            local size=$(du -h "$f" | cut -f1)
            local size_bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
            local size_gb=$((size_bytes / 1073741824))
            if [[ $size_gb -le 17 ]]; then
                echo -e "  ${BOLD}[${i}]${RESET} ${name}  (${size})  ${GREEN}✓ fits${RESET}"
            else
                echo -e "  ${BOLD}[${i}]${RESET} ${name}  (${size})  ${YELLOW}⚠ may OOM with draft${RESET}"
            fi
            models+=("$f")
            i=$((i + 1))
        fi
    done

    if [[ ${#models[@]} -eq 0 ]]; then
        echo "  No Qwen3.6-27B models found in ${MODELS_DIR}/"
        sleep 2
        return 1
    fi

    echo ""
    read -p "  Select [1-${#models[@]}]: " choice
    if [[ "$choice" -ge 1 && "$choice" -le "${#models[@]}" ]] 2>/dev/null; then
        SELECTED_MODEL="${models[$((choice-1))]}"
    else
        echo "  Invalid selection."
        sleep 1
        return 1
    fi
}

# ── Context size selection ───────────────────────────────────────────────

select_ctx() {
    echo ""
    echo "  Select max context size:"
    echo "  ────────────────────────────────────────"
    echo -e "  ${BOLD}[1]${RESET}    512    Token-gen only (fastest decode)"
    echo -e "  ${BOLD}[2]${RESET}   2048    Short prompt + completion"
    echo -e "  ${BOLD}[3]${RESET}   4096    Short chat / code completion"
    echo -e "  ${BOLD}[4]${RESET}   8192    Standard chat"
    echo -e "  ${BOLD}[5]${RESET}  16384   Long documents  ${GREEN}(default)${RESET}"
    echo -e "  ${BOLD}[6]${RESET}  32768   Very long context  ${YELLOW}(Q4_0 KV)${RESET}"
    echo -e "  ${BOLD}[7]${RESET}  65536   64K context  ${YELLOW}(TQ3_0 KV)${RESET}"
    echo -e "  ${BOLD}[8]${RESET} 131072   128K context  ${YELLOW}(TQ3_0 KV)${RESET}"
    echo -e "  ${BOLD}[9]${RESET} 262144   256K context  ${YELLOW}(TQ3_0 KV, max for 24GB)${RESET}"
    echo -e "  ${BOLD}[0]${RESET}  Custom"
    echo ""
    read -p "  Select [1-0, default=5]: " ctx_choice
    ctx_choice=${ctx_choice:-5}
    case $ctx_choice in
        1) SELECTED_CTX=512 ;;
        2) SELECTED_CTX=2048 ;;
        3) SELECTED_CTX=4096 ;;
        4) SELECTED_CTX=8192 ;;
        5) SELECTED_CTX=16384 ;;
        6) SELECTED_CTX=32768 ;;
        7) SELECTED_CTX=65536 ;;
        8) SELECTED_CTX=131072 ;;
        9) SELECTED_CTX=262144 ;;
        0)
            read -p "  Enter context size: " SELECTED_CTX
            SELECTED_CTX=${SELECTED_CTX:-16384}
            ;;
        *) SELECTED_CTX=16384 ;;
    esac
}

# ── Quick benchmark ─────────────────────────────────────────────────────

run_bench() {
    local info_file="${SCRIPT_DIR}/.server_info_lucebox"
    local port=8080
    if [[ -f "$info_file" ]]; then
        source "$info_file"
        port="${PORT:-8080}"
    fi

    if ! pgrep -f "scripts/server.py.*--port ${port}" > /dev/null 2>&1; then
        echo ""
        echo -e "  ${RED}Server not running. Start it first with [1].${RESET}"
        sleep 2
        return
    fi

    local n_gen="${1:-128}"
    echo ""
    echo "  Benchmarking running server on port ${port} (10 prompts, ${n_gen} gen tokens)..."
    echo ""

    python3 "${SCRIPT_DIR}/v1lucebox_bench.py" --port ${port} --n-gen ${n_gen} 2>&1

    echo ""
    read -p "  Press Enter to continue..."
}

# ── Main menu ────────────────────────────────────────────────────────────

# First run checks
check_build
check_draft

while true; do
    clear
    echo "=========================================================="
    echo "  Lucebox DFlash — DDTree Speculative Decoding"
    echo "  Luce-Org/lucebox-hub  |  ~104 tok/s on RTX 4090"
    echo "=========================================================="

    update_dashboard

    echo ""
    echo "  Actions:"
    echo "  -------"
    echo -e "  ${BOLD}[1]${RESET} Select model + Start server"
    echo -e "  ${BOLD}[2]${RESET} Stop server"
    echo -e "  ${BOLD}[3]${RESET} Quick benchmark (HumanEval 10 prompts)"
    echo -e "  ${BOLD}[4]${RESET} Test with curl"
    echo -e "  ${BOLD}[5]${RESET} Tail server log"
    echo -e "  ${BOLD}[6]${RESET} Rebuild binary"
    echo -e "  ${BOLD}[0]${RESET} Back to engine picker"
    echo ""

    read -p "  Select: " choice
    choice=$(echo "$choice" | tr -d '[:space:]')

    case $choice in
        1)
            if select_model; then
                select_ctx
                start_server "$SELECTED_MODEL" 8080 "$SELECTED_CTX"
            fi
            ;;
        2)
            stop_server
            ;;
        3)
            if [[ -z "$SELECTED_MODEL" ]]; then
                if ! select_model; then continue; fi
            fi
            read -p "  Gen tokens per prompt [128]: " n_gen
            n_gen=${n_gen:-128}
            run_bench "$n_gen"
            ;;
        4)
            echo ""
            echo "  Testing with curl..."
            curl -s http://localhost:8080/v1/chat/completions \
                -H 'Content-Type: application/json' \
                -d '{"model":"luce-dflash","messages":[{"role":"user","content":"def fibonacci(n):"}],"max_tokens":64,"stream":false}' | python3 -m json.tool 2>/dev/null || echo "  Server not responding."
            echo ""
            read -p "  Press Enter to continue..."
            ;;
        5)
            tail -50 "${SCRIPT_DIR}/server_lucebox.log" 2>/dev/null || echo "  No log file found."
            read -p "  Press Enter to continue..."
            ;;
        6)
            export PATH="/usr/local/cuda/bin:$PATH"
            cd "${SCRIPT_DIR}/${LUCE_DIR}"
            cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CXX_FLAGS="-I/usr/local/cuda/include" 2>&1 | tail -3
            cmake --build build --target test_dflash test_generate -j$(nproc) 2>&1 | tail -5
            cd "${SCRIPT_DIR}"
            echo ""
            read -p "  Press Enter to continue..."
            ;;
        0)
            exit 0
            ;;
        *)
            ;;
    esac
done
