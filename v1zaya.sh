#!/bin/bash

# =========================================================================
# ZAYA1-8B DASHBOARD v1.0  ⚠️ EXPERIMENTAL
# Runs Zyphra/ZAYA1-8B via Zyphra's vLLM fork.
# Novel MoE architecture: 8B total / 760M active params.
# Requires: pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr"
#
# ⚠️ WARNING: Zyphra's vLLM fork can HARD-CRASH the machine during install.
#   Use at your own risk. Consider a disposable VM or container.
# =========================================================================

set +m

ZAYA_DIR="zaya_models"
MODEL_ID="Zyphra/ZAYA1-8B"
LOCAL_MODEL_DIR="${ZAYA_DIR}/ZAYA1-8B"
LOG_FILE="server_zaya.log"
INFO_FILE=".server_info_zaya"
PORT=8080

# Colors
GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BLUE=$(tput setaf 4); BOLD=$(tput bold); RESET=$(tput sgr0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$ZAYA_DIR"

# --- Dependency checks ---
for cmd in curl jq python3 pip; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd"
    fi
done

# --- Terminal handling ---
MONITOR_PID=""

cleanup() {
    local exit_code=$?
    kill -9 "$MONITOR_PID" > /dev/null 2>&1
    wait "$MONITOR_PID" > /dev/null 2>&1
    tput csr 0 "$(tput lines)"
    tput cnorm
    echo ""
    exit $exit_code
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

get_multi_gpu_stats() {
    local gpu_idx=0
    local total_vram=0
    local total_used=0
    local result=""
    while IFS=',' read -r gpu_load vram_used vram_total gpu_temp; do
        gpu_load=$(echo "$gpu_load" | tr -d ' ')
        vram_used=$(echo "$vram_used" | tr -d ' ')
        vram_total=$(echo "$vram_total" | tr -d ' ')
        gpu_temp=$(echo "$gpu_temp" | tr -d ' ')
        vram_pct=$(( (vram_used * 100) / vram_total ))
        vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
        vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
        result+="GPU ${gpu_idx}: ${gpu_load}%   |   VRAM: ${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)   |   Temp: ${gpu_temp}°C\n"
        total_vram=$((total_vram + vram_total))
        total_used=$((total_used + vram_used))
        gpu_idx=$((gpu_idx + 1))
    done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)

    echo "$gpu_idx"     # GPU count
    echo -e "$result"   # Per-GPU lines
    echo "$total_used"  # Total VRAM used (MiB)
    echo "$total_vram"  # Total VRAM (MiB)
}

update_dashboard_stats() {
    local gpu_data
    gpu_data=$(get_multi_gpu_stats)

    exec 3>&1
    read -r gpu_count < <(echo "$gpu_data" | sed -n '1p')
    local gpu_lines
    gpu_lines=$(echo "$gpu_data" | sed -n '2p')
    read -r total_used total_vram < <(echo "$gpu_data" | sed -n '3p;4p' | tr '\n' ' ')

    cpu_pct=$(get_cpu_usage)
    if [[ "$cpu_pct" -ge 80 ]]; then c_cpu="\033[1;31m"; elif [[ "$cpu_pct" -ge 50 ]]; then c_cpu="\033[1;33m"; else c_cpu="\033[1;32m"; fi

    local total_pct=0
    if [[ "$total_vram" -gt 0 ]]; then total_pct=$(( (total_used * 100) / total_vram )); fi
    local total_used_gb=$(awk "BEGIN {printf \"%.1f\", $total_used/1024}")
    local total_total_gb=$(awk "BEGIN {printf \"%.0f\", $total_vram/1024}")
    if [[ "$total_pct" -ge 90 ]]; then c_vram="\033[1;31m"; elif [[ "$total_pct" -ge 50 ]]; then c_vram="\033[1;33m"; else c_vram="\033[1;32m"; fi

    # Check if ZAYA server is running
    local server_status
    if is_server_running; then
        local info=""
        if [[ -f "${INFO_FILE}" ]]; then info=$(cat "${INFO_FILE}"); fi
        server_status="\033[1;32mRUNNING${RESET}  ${info}"
    else
        server_status="\033[1;31mSTOPPED${RESET}"
    fi

    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${BOLD}ZAYA1-8B (Zyphra vLLM)${RESET}    |  SERVER: ${server_status}\033[K"
    tput cup 3 0
    echo -e "   CPU: ${c_cpu}${cpu_pct}%${RESET}   |   ${gpu_lines}\033[K"
    tput cup 4 0
    echo -e "   TOTAL: VRAM: ${c_vram}${total_used_gb} GB / ${total_total_gb} GB (${total_pct}%)${RESET}   |   GPUs: ${gpu_count}\033[K"
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
    echo "   ZAYA1-8B DASHBOARD v1.0"
    echo "==========================================================================================================================="
    tput cup 5 0
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

# --- Server management ---

is_server_running() {
    # Check for vllm process serving ZAYA
    pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" > /dev/null 2>&1
}

is_llamacpp_running() {
    pgrep -f "llama-server" > /dev/null 2>&1
}

get_vllm_path() {
    # Find the vllm binary - could be in a venv or global
    local vllm_bin
    # Check common locations
    for path in "$(python3 -c 'import shutil; print(shutil.which("vllm"))' 2>/dev/null)" \
                "/usr/local/bin/vllm" \
                "$HOME/.local/bin/vllm" \
                "$(find /opt -name vllm -type f 2>/dev/null | head -1)"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

check_vllm_zaya() {
    # Check if Zyphra's vLLM fork is installed
    python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        # Check if it supports ZAYA
        python3 -c "from vllm.model_executor.models.zaya import ZayaForCausalLM" 2>/dev/null
        return $?
    fi
    return 1
}

# --- Install / Update ---

install_update() {
    echo ""
    echo -e " ${CYAN}>>> INSTALL / UPDATE ZAYA vLLM <<<${RESET}"
    echo ""
    echo " This installs Zyphra's vLLM fork with ZAYA architecture support."
    echo " It builds vLLM from source (may take 10-20 minutes)."
    echo ""
    read -p "  Continue? (y/N): " confirm
    confirm=$(echo "$confirm" | tr -d '[:space:]')
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo " Cancelled."
        sleep 1
        return
    fi

    echo ""
    echo " Installing build dependencies..."

    # --- pip bootstrap ---
    if ! python3 -m pip --version > /dev/null 2>&1; then
        echo " Installing pip..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3 2>&1 | tail -3
    fi

    # --- system packages ---
    echo " Installing system build tools (ninja, g++, python3-dev)..."
    apt-get install -y ninja-build g++ python3-dev 2>&1 | tail -3

    # --- setuptools version compatible with torch ---
    echo " Installing setuptools <82 (torch compatibility)..."
    python3 -m pip install -U "setuptools<82" wheel packaging 2>&1 | tail -3

    # --- cmake 3.26+ required (system-wide) ---
    local cmake_ver
    cmake_ver=$(cmake --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
    local cmake_major cmake_minor
    cmake_major=$(echo "$cmake_ver" | cut -d. -f1)
    cmake_minor=$(echo "$cmake_ver" | cut -d. -f2)
    local need_cmake=0
    if [[ -z "$cmake_ver" ]]; then
        need_cmake=1
    elif [[ "$cmake_major" -lt 3 ]] || { [[ "$cmake_major" -eq 3 ]] && [[ "$cmake_minor" -lt 26 ]]; }; then
        need_cmake=1
    fi
    if [[ "$need_cmake" == 1 ]]; then
        echo " cmake ${cmake_ver:-N/A} found — vLLM requires 3.26+. Installing system-wide..."
        local cmake_url="https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh"
        curl -sL "$cmake_url" -o /tmp/cmake-install.sh
        bash /tmp/cmake-install.sh --prefix=/usr/local --skip-license 2>&1 | tail -3
        rm -f /tmp/cmake-install.sh
        echo " cmake now: $(cmake --version 2>/dev/null | head -1)"
    else
        echo " cmake $(cmake --version 2>/dev/null | head -1) — OK"
    fi

    # --- CUDA in PATH ---
    if ! nvcc --version > /dev/null 2>&1; then
        if [[ -x /usr/local/cuda/bin/nvcc ]]; then
            echo " Adding CUDA to PATH..."
            export PATH="/usr/local/cuda/bin:${PATH}"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        else
            echo -e " ${RED}CUDA nvcc not found! Install CUDA Toolkit first.${RESET}"
            sleep 3
            return
        fi
    fi
    echo " nvcc: $(nvcc --version 2>/dev/null | tail -1)"

    # --- PyTorch (build dependency for vLLM) ---
    # IMPORTANT: vLLM's pip install may pull in a newer torch with wrong CUDA version.
    # We must pin torch to match the DRIVER's CUDA capability, not nvcc's version.
    local driver_cuda
    driver_cuda=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || echo "12.4")
    echo " Driver CUDA version: ${driver_cuda}"
    # Map driver CUDA to available PyTorch wheel tags
    # cu121, cu124, cu126, cu128 — pick the highest that driver supports
    local torch_index
    case "$driver_cuda" in
        13.*) torch_index="cu130" ;;
        12.8|12.9) torch_index="cu128" ;;
        12.6|12.7) torch_index="cu126" ;;
        12.4|12.5) torch_index="cu124" ;;
        12.*) torch_index="cu121" ;;
        *) torch_index="cu124" ;;
    esac
    echo " Using PyTorch wheel index: ${torch_index}"

    local torch_installed
    torch_installed=$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "")
    if [[ -n "$torch_installed" ]]; then
        # Check if installed torch CUDA matches driver
        local torch_cu_major torch_cu_minor
        torch_cu_major=$(echo "$torch_installed" | cut -d. -f1)
        torch_cu_minor=$(echo "$torch_installed" | cut -d. -f2)
        local driver_major driver_minor
        driver_major=$(echo "$driver_cuda" | cut -d. -f1)
        driver_minor=$(echo "$driver_cuda" | cut -d. -f2)
        if [[ "$torch_cu_major" -gt "$driver_major" ]] || { [[ "$torch_cu_major" -eq "$driver_major" ]] && [[ "$torch_cu_minor" -gt "$driver_minor" ]]; }; then
            echo -e " ${YELLOW}⚠ Installed torch CUDA ${torch_installed} > driver CUDA ${driver_cuda}${RESET}"
            echo " Reinstalling PyTorch with matching CUDA version (force-reinstall)..."
            python3 -m pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${torch_index}" 2>&1 | tail -5
        else
            echo " PyTorch $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null) CUDA ${torch_installed} — OK"
        fi
    else
        echo " Installing PyTorch (required for vLLM build, force-reinstall)..."
        python3 -m pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${torch_index}" 2>&1 | tail -5
    fi

    echo ""
    echo " Installing Zyphra vLLM fork (zaya1-pr branch)..."
    echo " This will build from source — please wait (10-30 min)..."
    echo " Full build log: ${SCRIPT_DIR}/zaya_build.log"
    echo ""

    # --- Memory safety: pre-flight check ---
    local mem_avail_kb
    mem_avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local mem_avail_gb=$((mem_avail_kb / 1024 / 1024))
    local swap_total_kb
    swap_total_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
    local swap_total_gb=$((swap_total_kb / 1024 / 1024))
    local need_gb=30  # vLLM build needs ~30-40GB peak

    echo -e " ${CYAN}Memory pre-flight check:${RESET}"
    echo "   Available RAM: ${mem_avail_gb} GB"
    echo "   Current swap:  ${swap_total_gb} GB"
    echo "   Recommended:   ${need_gb} GB free (RAM + swap)"
    echo ""

    local ZAYA_SWAPFILE=""
    local total_usable=$((mem_avail_gb + swap_total_gb))

    if [[ ${total_usable} -lt ${need_gb} ]]; then
        local deficit=$(( need_gb - total_usable ))
        local swap_add=$(( deficit + 8 ))  # add 8GB extra padding
        echo -e " ${YELLOW}⚠ Low memory! ${mem_avail_gb}GB RAM + ${swap_total_gb}GB swap = ${total_usable}GB (need ${need_gb}GB)${RESET}"
        echo -e " ${YELLOW}  Without intervention, the build WILL likely OOM and may hard-lock the machine.${RESET}"
        echo ""
        echo " Options:"
        echo "  [1] Auto-create ${swap_add}GB swap file (recommended, cleaned up after build)"
        echo "  [2] Continue anyway (DANGEROUS — may crash)"
        echo "  [3] Abort"
        echo ""
        read -p "  Choose [1/2/3]: " mem_choice
        case "$mem_choice" in
            1)
                # Find a place for swap file
                ZAYA_SWAPFILE="/zaya_swap.tmp"
                if [[ ! -w / ]]; then
                    ZAYA_SWAPFILE="/tmp/zaya_swap.tmp"
                fi
                echo ""
                echo " Creating ${swap_add}GB swap file at ${ZAYA_SWAPFILE}..."
                dd if=/dev/zero of="${ZAYA_SWAPFILE}" bs=1M count=$((swap_add * 1024)) status=progress 2>&1 | tail -3
                chmod 600 "${ZAYA_SWAPFILE}"
                mkswap "${ZAYA_SWAPFILE}" 2>&1 | tail -1
                swapon "${ZAYA_SWAPFILE}" 2>&1 | tail -1
                local new_swap_gb
                new_swap_gb=$(awk '/SwapTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
                echo -e " ${GREEN}Swap active: ${new_swap_gb} GB total swap now available.${RESET}"
                echo ""
                ;;
            2)
                echo -e " ${YELLOW}Continuing at your own risk...${RESET}"
                ;;
            *)
                echo " Aborted."
                sleep 1
                return
                ;;
        esac
    else
        echo -e " ${GREEN}Memory OK — ${total_usable}GB usable (RAM + swap).${RESET}"
        echo ""
    fi

    # --- Limit MAX_JOBS to reduce peak RAM ---
    # Default ninja uses all CPU cores, each ~2-4GB RAM.
    # Cap at half the available cores, max 4.
    local nproc_total
    nproc_total=$(nproc 2>/dev/null || echo 4)
    local max_jobs=$(( nproc_total / 2 ))
    if [[ ${max_jobs} -gt 4 ]]; then max_jobs=4; fi
    if [[ ${max_jobs} -lt 1 ]]; then max_jobs=1; fi
    export MAX_JOBS=${max_jobs}
    echo " Build parallelism: MAX_JOBS=${MAX_JOBS} (${nproc_total} CPUs detected, capped for RAM safety)"
    echo ""

    # vLLM (Zyphra fork) requires torch==2.11.0, which is only on cu126+.
    # Pre-install the correct torch so pip doesn't try to pull a conflicting version.
    local vllm_torch_ver="2.11.0"
    # Map driver CUDA to the best torch wheel index for the required version
    local vllm_torch_index
    case "$driver_cuda" in
        13.*) vllm_torch_index="cu130" ;;
        12.8|12.9) vllm_torch_index="cu126" ;;
        12.6|12.7) vllm_torch_index="cu126" ;;
        12.4|12.5) vllm_torch_index="cu124" ;;
        12.*) vllm_torch_index="cu121" ;;
        *) vllm_torch_index="cu126" ;;
    esac

    # Check if the mapped index has the required torch version; if not, fall back to cu126
    local has_torch_ver
    has_torch_ver=$(pip install --dry-run "torch==${vllm_torch_ver}" --index-url "https://download.pytorch.org/whl/${vllm_torch_index}" 2>&1)
    if echo "$has_torch_ver" | grep -q "Could not find a version"; then
        echo " torch==${vllm_torch_ver} not available on ${vllm_torch_index}, trying cu126..."
        vllm_torch_index="cu126"
        # Verify cu126 also has it (it should, but double-check)
        has_torch_ver=$(pip install --dry-run "torch==${vllm_torch_ver}" --index-url "https://download.pytorch.org/whl/${vllm_torch_index}" 2>&1)
        if echo "$has_torch_ver" | grep -q "Could not find a version"; then
            echo -e " ${RED}torch==${vllm_torch_ver} is not available for your CUDA version.${RESET}"
            echo -e " ${RED}Zyphra vLLM requires torch ${vllm_torch_ver}, which needs CUDA driver >= 12.6.${RESET}"
            echo -e " ${RED}Your driver CUDA: ${driver_cuda}. Cannot proceed.${RESET}"
            echo ""
            read -p " Press Enter to return to menu..."
            return
        fi
    fi

    echo " Pre-installing torch==${vllm_torch_ver} from ${vllm_torch_index} index (required by Zyphra vLLM)..."
    python3 -m pip install --force-reinstall "torch==${vllm_torch_ver}" --index-url "https://download.pytorch.org/whl/${vllm_torch_index}" 2>&1 | tail -5
    echo ""

    # torchvision must match torch's CUDA wheel. If a CPU or older torchvision remains,
    # importing vLLM can fail with: RuntimeError: operator torchvision::nms does not exist
    echo " Installing matching torchvision==0.26.0 from ${vllm_torch_index} index..."
    python3 -m pip install --force-reinstall --no-deps "torchvision==0.26.0" --index-url "https://download.pytorch.org/whl/${vllm_torch_index}" 2>&1 | tail -5
    echo ""

    echo " Installing Zyphra vLLM fork (no torch constraint — let it use the pre-installed torch)"
    python3 -m pip install -U "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr" 2>&1 | tee "${SCRIPT_DIR}/zaya_build.log" | tail -30
    local rc=${PIPESTATUS[0]}

    # --- Cleanup temporary swap ---
    if [[ -n "${ZAYA_SWAPFILE}" && -f "${ZAYA_SWAPFILE}" ]]; then
        echo ""
        echo " Cleaning up temporary swap file..."
        swapoff "${ZAYA_SWAPFILE}" 2>/dev/null
        rm -f "${ZAYA_SWAPFILE}"
        echo -e " ${GREEN}Swap file removed.${RESET}"
    fi

    if [[ "$rc" -eq 0 ]]; then
        echo ""
        echo -e " ${GREEN}vLLM (Zyphra fork) installed successfully!${RESET}"
        python3 -m vllm.entrypoints.openai.api_server --version 2>/dev/null || true
    else
        echo ""
        echo -e " ${RED}Installation failed (exit code $rc). Check errors above.${RESET}"
        echo ""
        echo " Common fixes:"
        echo "  - Python 3.10+ required (current: $(python3 --version))"
        echo "  - CUDA 12.x required (current: $(nvcc --version 2>/dev/null | tail -1))"
        echo "  - cmake 3.26+ required (current: $(cmake --version 2>/dev/null | head -1))"
        echo "  - 20+ GB free disk space for build"
    fi
    echo ""
    read -p " Press Enter to return to menu..."
}

# --- Download model ---

download_model() {
    echo ""
    echo -e " ${CYAN}>>> DOWNLOAD ZAYA1-8B <<<${RESET}"
    echo ""
    echo " Model: ${MODEL_ID}"
    echo " Size:  ~16.5 GB (bf16, 4 safetensor shards)"
    echo " Save:  ${LOCAL_MODEL_DIR}/"
    echo ""

    if [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
        echo -e " ${YELLOW}Model already exists at ${LOCAL_MODEL_DIR}/${RESET}"
        echo ""
        read -p " Re-download? (y/N): " redl
        redl=$(echo "$redl" | tr -d '[:space:]')
        if [[ "$redl" != "y" && "$redl" != "Y" ]]; then
            echo " Skipping."
            sleep 1
            return
        fi
        rm -rf "${LOCAL_MODEL_DIR}"
    fi

    # Install huggingface hub tools if needed
    if ! command -v hf > /dev/null 2>&1 && ! command -v huggingface-cli > /dev/null 2>&1; then
        echo " Installing huggingface_hub..."
        python3 -m pip install -U huggingface_hub 2>&1 | tail -3
    fi

    mkdir -p "${LOCAL_MODEL_DIR}"
    echo ""
    echo " Downloading... (this will take a while for 16.5 GB)"
    echo ""

    # Use new 'hf' CLI if available, fall back to deprecated huggingface-cli
    if command -v hf > /dev/null 2>&1; then
        hf download "${MODEL_ID}" --local-dir "${LOCAL_MODEL_DIR}"
    else
        huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_MODEL_DIR}"
    fi

    if [[ $? -eq 0 ]]; then
        echo ""
        echo -e " ${GREEN}Download complete!${RESET}"
        du -sh "${LOCAL_MODEL_DIR}"
    else
        echo ""
        echo -e " ${RED}Download failed. Check your internet connection.${RESET}"
    fi
    echo ""
    read -p " Press Enter to return to menu..."
}

# --- Delete model ---

delete_model() {
    echo ""
    echo -e " ${CYAN}>>> DELETE ZAYA1-8B MODEL <<<${RESET}"
    echo ""

    if [[ ! -d "${LOCAL_MODEL_DIR}" ]]; then
        echo -e " ${YELLOW}No model found at ${LOCAL_MODEL_DIR}/${RESET}"
        sleep 2
        return
    fi

    du -sh "${LOCAL_MODEL_DIR}"
    echo ""
    read -p "  Delete this model? (y/N): " confirm
    confirm=$(echo "$confirm" | tr -d '[:space:]')
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        rm -rf "${LOCAL_MODEL_DIR}"
        echo -e " ${GREEN}Model deleted.${RESET}"
    else
        echo " Cancelled."
    fi
    sleep 1
}

# --- Start server ---

start_server() {
    local use_local="$1"  # "local" = use downloaded model, "" = from HF hub
    local model_path

    echo ""
    echo -e " ${CYAN}>>> START ZAYA1-8B SERVER <<<${RESET}"

    # Prerequisite checks
    if is_llamacpp_running; then
        echo -e " ${RED}llama.cpp is running! Stop it first (use HostLLM [8] Kill All).${RESET}"
        sleep 3
        return
    fi

    if is_server_running; then
        echo -e " ${YELLOW}ZAYA server is already running! Stop it first [6].${RESET}"
        sleep 2
        return
    fi

    # Check vLLM is installed
    if ! check_vllm_zaya; then
        echo -e " ${RED}Zyphra vLLM fork not detected! Run Install [0] first.${RESET}"
        echo ""
        echo " Install with:"
        echo '   pip install "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr"'
        sleep 4
        return
    fi

    # Determine model path
    if [[ "$use_local" == "local" ]]; then
        if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
            echo -e " ${RED}Model not downloaded! Run Download [5] first, or use option [2] for online mode.${RESET}"
            sleep 3
            return
        fi
        model_path="${SCRIPT_DIR}/${LOCAL_MODEL_DIR}"
    else
        model_path="${MODEL_ID}"
    fi

    # Detect GPU count
    local gpu_count=0
    if command -v nvidia-smi > /dev/null 2>&1; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    fi

    echo ""
    echo "   Model:     ZAYA1-8B (8B total / 760M active, MoE 16 experts)"
    echo "   Source:    $([ "$use_local" == "local" ] && echo "Local (${LOCAL_MODEL_DIR})" || echo "HuggingFace Hub (auto-download)")"
    echo "   Port:      ${PORT}"
    echo "   GPUs:      ${gpu_count}"
    echo ""

    # Build the vllm command
    local cmd="vllm serve ${model_path}"
    cmd+=" --served-model-name ZAYA1-8B"
    cmd+=" --port ${PORT}"
    cmd+=" --mamba-cache-dtype float32"
    cmd+=" --dtype bfloat16"

    # ZAYA advertises 131K context, but that does not fit on a 24GB GPU after
    # weights + CUDA graphs. vLLM will fail during KV-cache init unless max length
    # is reduced. 32K is a safe default for RTX 3090/4090-class single GPUs.
    local max_model_len="32768"
    if [[ "$gpu_count" -ge 2 ]]; then
        max_model_len="65536"
    fi
    cmd+=" --max-model-len ${max_model_len}"
    cmd+=" --gpu-memory-utilization 0.92"

    cmd+=" --reasoning-parser qwen3"

    # Disabled by default: current Zyphra fork's zaya_xml parser is incompatible
    # with this vLLM parser API and can fail with:
    #   ZayaXMLToolParser.__init__() takes 2 positional arguments but 3 were given
    # Basic OpenAI-compatible chat works without auto tool choice.

    # Multi-GPU: recommend DP+EP for 2+ GPUs
    if [[ "$gpu_count" -ge 2 ]]; then
        echo -e " ${YELLOW}Note: For multi-GPU, Zyphra recommends DP+EP (not TP).${RESET}"
        echo " Adding --data-parallel-size ${gpu_count} --enable-expert-parallel"
        cmd+=" --data-parallel-size ${gpu_count} --enable-expert-parallel"
    fi

    echo ""
    echo " Starting server..."
    echo " Command: ${cmd}"
    echo ""

    # Start in background, log to file
    nohup $cmd > "${LOG_FILE}" 2>&1 &
    local server_pid=$!

    echo " PID: ${server_pid}"
    echo " Log: ${LOG_FILE}"
    echo ""
    echo " Waiting for server to load (up to 300s)..."

    local loaded=0
    for i in $(seq 1 300); do
        # Check if process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo ""
            echo -e " ${RED}Server process died! Check log:${RESET}"
            tail -n 30 "${LOG_FILE}"
            sleep 3
            return
        fi

        local code
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            loaded=1
            break
        fi

        printf "\r   Waiting... %ds" "$i"
        sleep 1
    done

    echo ""

    if [[ "$loaded" == "1" ]]; then
        echo "ZAYA: zaya1-8b [bf16/moe16/760m-active] port ${PORT}" > "${INFO_FILE}"
        echo ""
        echo -e " ${GREEN}Server is loaded and health check returned OK!${RESET}"
        echo ""
        echo "================================================================"
        echo "  ZAYA1-8B SERVER RUNNING"
        echo "================================================================"
        echo ""
        echo "  Model:   ZAYA1-8B (8B total / 760M active)"
        echo "  Port:    ${PORT}"
        echo "  Log:     ${LOG_FILE}"
        echo ""
        echo "  API:         http://localhost:${PORT}/v1/chat/completions"
        echo "  Models:      http://localhost:${PORT}/v1/models"
        echo "  Health:      http://localhost:${PORT}/health"
        echo ""
        echo "  Recommended: temperature 1.0, top-p 0.95, top-k -1"
        echo "  For code:    temperature 0.6, top-p 0.95, top-k -1"
        echo ""
        echo "  Quick test:"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"Zyphra/ZAYA1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'"
        echo ""
        echo "================================================================"
    else
        echo ""
        echo -e " ${YELLOW}Server is still loading. It may need more time.${RESET}"
        echo " Check log: tail -f ${LOG_FILE}"
        echo "ZAYA: zaya1-8b [loading...] port ${PORT}" > "${INFO_FILE}"
    fi

    sleep 3
}

# --- Stop server ---

stop_server() {
    echo ""
    echo -e " ${CYAN}>>> STOPPING ZAYA1-8B SERVER <<<${RESET}"

    if is_server_running; then
        # Find and kill the vllm process serving ZAYA
        local pids
        pids=$(pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" 2>/dev/null)
        if [[ -n "$pids" ]]; then
            for pid in $pids; do
                kill "$pid" 2>/dev/null
            done
            sleep 2
            # Force kill if still running
            pids=$(pgrep -f "vllm.*ZAYA1-8B\|vllm.*Zyphra" 2>/dev/null)
            if [[ -n "$pids" ]]; then
                for pid in $pids; do
                    kill -9 "$pid" 2>/dev/null
                done
            fi
        fi
        rm -f "${INFO_FILE}"
        echo -e " ${GREEN}ZAYA server stopped.${RESET}"
    else
        echo " Server is not running."
        rm -f "${INFO_FILE}"
    fi
    sleep 1
}

# --- Quick benchmark ---

run_benchmark() {
    echo ""
    echo -e " ${CYAN}>>> ZAYA1-8B QUICK BENCHMARK <<<${RESET}"

    if ! is_server_running; then
        echo " Server is not running! Start it first."
        sleep 2
        return
    fi

    local n_runs=3
    local max_tokens=128
    local prompt="Write a short story about a robot learning to paint. Be creative and use vivid imagery."

    echo ""
    echo " Running ${n_runs}x benchmark (${max_tokens} tokens each)..."
    echo ""

    local total_tps=0
    local valid=0

    for run in $(seq 1 $n_runs); do
        local start_time
        start_time=$(date +%s%N)

        local response
        response=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens},\"stream\":false}" 2>/dev/null)

        local end_time
        end_time=$(date +%s%N)
        local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

        local tokens
        tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)

        if [[ -n "$tokens" && "$tokens" -gt 0 ]]; then
            local tps
            tps=$(awk "BEGIN {printf \"%.1f\", ($tokens / $elapsed_ms) * 1000}")
            echo "   Run ${run}: ${tokens} tokens in ${elapsed_ms}ms = ${tps} tok/s"
            total_tps=$(awk "BEGIN {printf \"%.1f\", $total_tps + $tps}")
            valid=$((valid + 1))
        else
            echo "   Run ${run}: Error or empty response"
            echo "$response" | jq -r '.error.message // "Unknown error"' 2>/dev/null
        fi
    done

    if [[ "$valid" -gt 0 ]]; then
        local avg_tps
        avg_tps=$(awk "BEGIN {printf \"%.1f\", $total_tps / $valid}")
        echo ""
        echo -e " ${GREEN}Average: ${avg_tps} tok/s (${valid} runs)${RESET}"
    fi
    echo ""
    read -p " Press Enter to return to menu..."
}

# --- View logs ---

view_logs() {
    echo ""
    echo -e " ${CYAN}>>> ZAYA1-8B SERVER LOGS <<<${RESET}"
    echo -e " ${YELLOW}Press [Ctrl+C] to return to menu.${RESET}"

    if [[ -f "${LOG_FILE}" ]]; then
        tail -f "${LOG_FILE}"
    else
        echo " No log file found (${LOG_FILE})."
        sleep 2
    fi
}

# --- Setup status ---

show_setup_status() {
    echo ""
    echo -e " ${CYAN}>>> SETUP STATUS <<<${RESET}"
    echo ""
    echo " 1. Zyphra vLLM fork:"
    if check_vllm_zaya; then
        local ver
        ver=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        echo -e "    ${GREEN}Installed${RESET} (v${ver})"
        local vllm_bin
        vllm_bin=$(get_vllm_path 2>/dev/null)
        echo "    Path: ${vllm_bin}"
    else
        echo -e "    ${RED}Not installed${RESET}"
        echo "    Install with: pip install \"vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr\""
    fi

    echo ""
    echo " 2. ZAYA1-8B model:"
    if [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
        local size
        size=$(du -sh "${LOCAL_MODEL_DIR}" 2>/dev/null | cut -f1)
        echo -e "    ${GREEN}Downloaded${RESET} (${size})"
        echo "    Path: ${LOCAL_MODEL_DIR}/"
    else
        echo -e "    ${YELLOW}Not downloaded${RESET} (will auto-download from HuggingFace Hub on first start)"
        echo "    Or run Download [5] to pre-download."
    fi

    echo ""
    echo " 3. GPU:"
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name mem; do
            echo "    GPU ${idx}: ${name} (${mem})"
        done
    else
        echo -e "    ${RED}nvidia-smi not found${RESET}"
    fi

    echo ""
    echo " 4. Server:"
    if is_server_running; then
        local info=""
        [[ -f "${INFO_FILE}" ]] && info=$(cat "${INFO_FILE}")
        echo -e "    ${GREEN}Running${RESET} — ${info}"
    else
        echo -e "    ${YELLOW}Stopped${RESET}"
    fi

    echo ""
    read -p " Press Enter to return to menu..."
}

# ========================================================================
# QUICK START
# ========================================================================

quickstart() {
    echo ""
    echo -e " ${CYAN}==========================================================${RESET}"
    echo -e " ${CYAN}   ZAYA1-8B QUICK START${RESET}"
    echo -e " ${CYAN}==========================================================${RESET}"
    echo ""

    # Step 1: Check vLLM
    echo " [1/3] Checking Zyphra vLLM fork..."
    if ! check_vllm_zaya; then
        echo " Not installed. Installing now..."
        python3 -m pip install -U "vllm @ git+https://github.com/Zyphra/vllm.git@zaya1-pr" 2>&1 | tail -10
        if [[ $? -ne 0 ]]; then
            echo -e " ${RED}Installation failed!${RESET}"
            sleep 3
            return
        fi
        echo -e " ${GREEN}Installed!${RESET}"
    else
        echo -e " ${GREEN}Already installed.${RESET}"
    fi
    echo ""

    # Step 2: Check model
    echo " [2/3] Checking ZAYA1-8B model..."
    if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
        echo " Not downloaded. Will use HuggingFace Hub (auto-download on start)."
        echo " Alternatively, you can pre-download with menu option [5]."
    else
        local size
        size=$(du -sh "${LOCAL_MODEL_DIR}" 2>/dev/null | cut -f1)
        echo -e " ${GREEN}Already downloaded (${size}).${RESET}"
    fi
    echo ""

    # Step 3: Start server
    echo " [3/3] Starting ZAYA1-8B server..."
    if is_llamacpp_running; then
        echo -e " ${RED}llama.cpp is running! Stop it first.${RESET}"
        sleep 3
        return
    fi

    if is_server_running; then
        echo -e " ${YELLOW}Already running!${RESET}"
        sleep 2
        return
    fi

    # Use local model if available, otherwise HF hub
    if [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
        start_server "local"
    else
        start_server ""
    fi
}

# ========================================================================
# MAIN LOOP
# ========================================================================

# Handle --quickstart flag
if [[ "${1:-}" == "--quickstart" ]]; then
    quickstart
    exit 0
fi

echo "Loading ZAYA1-8B Dashboard..."
setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""
    echo -e " ${CYAN}─── ZAYA1-8B ── Zyphra vLLM ── 8B total / 760M active ── MoE 16 experts ──${RESET}"
    echo ""
    echo " [1] Start Server (local model)"
    echo " [2] Start Server (from HuggingFace Hub — auto-downloads if missing)"
    echo ""
    echo -e " ${CYAN}─── SETUP ────────────────────────────────────────────────────────${RESET}"
    echo " [0] Install / Update Zyphra vLLM fork"
    echo " [5] Download Model (~16.5 GB)"
    echo " [D] Delete Model"
    echo " [s] Setup Status"
    echo ""
    echo -e " ${CYAN}─── CONTROLS ─────────────────────────────────────────────────────${RESET}"
    echo " [6] Stop Server"
    echo " [7] Quick Benchmark"
    echo " [8] View Server Logs"
    echo " [99] Back to Main Menu"
    echo " [98] Exit"
    echo ""

    tput cnorm
    read -p " Select: " action
    action=$(echo "$action" | tr -d '[:space:]')

    case $action in
        0)
            install_update
            ;;
        1)
            start_server "local"
            ;;
        2)
            start_server ""
            ;;
        5)
            download_model
            ;;
        d|D)
            delete_model
            ;;
        6)
            stop_server
            ;;
        7)
            run_benchmark
            ;;
        8)
            view_logs
            ;;
        s|S)
            show_setup_status
            ;;
        98)
            exit 42
            ;;
        99)
            exit 0
            ;;
        *)
            ;;
    esac
done
