#!/bin/bash

# =========================================================================
# BeeLlama.cpp DFlash Dashboard v1.0
# Uses Anbeeld/beellama.cpp — DFlash + TurboQuant/TCQ + Vision + Reasoning
# https://github.com/Anbeeld/beellama.cpp
# https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md
#
# Advantages over buun-llama-cpp (engine 2):
#   - Vision works with flat DFlash (--mmproj + --no-mmproj-offload)
#   - Reasoning ON supported (--reasoning on + preserve_thinking)
#   - TurboQuant + TCQ KV cache (turbo4/turbo3_tcq — massive VRAM savings)
#   - Adaptive draft depth (profit/fringe controllers)
#   - Reasoning loop guard
#   - Sampled DFlash (--spec-draft-temp)
#   - 128-value TurboQuant blocks (faster than buun's 32-value)
# =========================================================================

set +m

BEELLAMA_DIR="beellama-cpp"
MODELS_DIR="llama_models"
DEBUG_LOG="beellama_compile_debug.log"
SERVER_LOG="server_beellama.log"

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BLUE=$(tput setaf 4); BOLD=$(tput bold); RESET=$(tput sgr0)

declare -A speed_cache
mkdir -p "$MODELS_DIR"

for cmd in curl jq wget git cmake gcc g++; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd" build-essential
    fi
done

if ! command -v /usr/local/cuda/bin/nvcc > /dev/null 2>&1; then
    echo ""
    MAX_GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' .' | sort -n | tail -1)
    if [[ "$MAX_GPU_ARCH" -ge 120 ]]; then
        CUDA_PKG="cuda-toolkit-12-8"
        echo "CUDA Toolkit not found. Blackwell GPU detected (sm_${MAX_GPU_ARCH}), installing ${CUDA_PKG}..."
    else
        CUDA_PKG="cuda-toolkit-12-4"
        echo "CUDA Toolkit not found. Installing ${CUDA_PKG}..."
    fi

    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    case "$UBUNTU_CODENAME" in
        focal|groovy|hirsute|impish) REPO_DISTRO="ubuntu2004" ;;
        *) REPO_DISTRO="ubuntu2204" ;;
    esac

    sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/keyrings/cuda-keyring.gpg

    TMPDIR=$(mktemp -d)
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
        -O "${TMPDIR}/cuda-keyring.deb"
    sudo dpkg -i "${TMPDIR}/cuda-keyring.deb"
    rm -rf "${TMPDIR}"

    sudo apt update
    sudo apt install -y "$CUDA_PKG"

    if ! command -v /usr/local/cuda/bin/nvcc > /dev/null 2>&1; then
        echo ""
        echo " CUDA Toolkit install failed. Cannot compile with GPU support."
        echo " Install manually: sudo apt install ${CUDA_PKG}"
        read -p " Press Enter to exit..."
        exit 1
    fi
    echo "${CUDA_PKG} installed successfully."
fi

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

# ── Quickstart mode ───────────────────────────────────────────────────────


# ── Dashboard monitoring ─────────────────────────────────────────────────

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
    reset="\033[0m"; bold="\033[1m"
    if command -v nvidia-smi > /dev/null 2>&1; then
        gpu_load=0; vram_used=0; vram_total=0; gpu_temp_max=0; gpu_count=0
        gpu_lines=()
        while IFS=',' read -r load used total temp; do
            load=$(echo "$load" | tr -d ' ')
            used=$(echo "$used" | tr -d ' ')
            total=$(echo "$total" | tr -d ' ')
            temp=$(echo "$temp" | tr -d ' ')
            pct=0; [[ "$total" -gt 0 ]] && pct=$(( (used * 100) / total ))
            u_gb=$(awk "BEGIN {printf \"%.1f\", $used/1024}")
            t_gb=$(awk "BEGIN {printf \"%.0f\", $total/1024}")
            if [[ "$pct" -ge 90 ]]; then c="\033[1;31m"; elif [[ "$pct" -ge 50 ]]; then c="\033[1;33m"; else c="\033[1;32m"; fi
            gpu_lines+=("   GPU ${gpu_count}:  ${load}%   |   VRAM: ${c}${u_gb} GB / ${t_gb} GB (${pct}%)${reset}   |   Temp: ${temp} degC")
            gpu_load=$((gpu_load + load))
            vram_used=$((vram_used + used))
            vram_total=$((vram_total + total))
            [[ "$temp" -gt "$gpu_temp_max" ]] && gpu_temp_max=$temp
            gpu_count=$((gpu_count + 1))
        done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
        [[ "$gpu_count" -gt 0 ]] && gpu_load_avg=$((gpu_load / gpu_count)) || gpu_load_avg=0
        if [[ "$vram_total" -gt 0 ]]; then vram_pct=$(( (vram_used * 100) / vram_total )); else vram_pct=0; fi
        vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
        vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
        if [[ "$vram_pct" -ge 90 ]]; then c_vram="\033[1;31m"; elif [[ "$vram_pct" -ge 50 ]]; then c_vram="\033[1;33m"; else c_vram="\033[1;32m"; fi
        total_line="   TOTAL: VRAM: ${c_vram}${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)${reset}   |   GPUs: ${gpu_count}"
    else
        gpu_lines=("   GPU: N/A")
        total_line="   TOTAL: N/A"
        gpu_load_avg="N/A"; gpu_temp_max="-"
    fi

    cpu_pct=$(get_cpu_usage)
    if [[ "$cpu_pct" -ge 80 ]]; then c_cpu="\033[1;31m"; elif [[ "$cpu_pct" -ge 50 ]]; then c_cpu="\033[1;33m"; else c_cpu="\033[1;32m"; fi

    SERVER_PID=$(pgrep -f "llama-server" | head -n 1)
    if [[ -n "$SERVER_PID" ]]; then
        if [[ -f ".server_info_beellama" ]]; then
            ACTIVE_INFO=$(cat .server_info_beellama)
            SERVER_STATUS="\033[1;32mRUNNING: ${ACTIVE_INFO}\033[0m"
        else
            SERVER_STATUS="\033[1;32mRUNNING (PID: $SERVER_PID)\033[0m"
        fi
    else
        SERVER_STATUS="\033[1;31mSTOPPED\033[0m"
        rm -f .server_info_beellama 2>/dev/null
    fi

    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${bold}BeeLlama.cpp (DFlash + TurboQuant)${reset}\033[K"
    tput cup 3 0
    echo -e "   SERVER: ${SERVER_STATUS}\033[K"
    tput cup 4 0
    echo -e "   CPU: ${c_cpu}${cpu_pct}%${reset}\033[K"
    row=5
    for line in "${gpu_lines[@]}"; do
        tput cup $row 0
        echo -e "${line}\033[K"
        row=$((row + 1))
    done
    tput cup $row 0
    echo -e "${total_line}\033[K"
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
    tput csr 8 "$(tput lines)"
    tput cup 0 0
    echo "==========================================================================================================================="
    echo "   BEELLAMA.CPP DFLASH DASHBOARD v1.0  --  Anbeeld/beellama.cpp"
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "==========================================================================================================================="
    tput cup 7 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

# ── Argument probe helper ────────────────────────────────────────────────

arg_probe_valid() {
    local server_bin="$1"
    shift
    local probe_log=".beellama_arg_probe.log"
    local dummy_model=".beellama_arg_probe_dummy.gguf"
    : > "$dummy_model"
    local -a test_cmd
    test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18099)
    timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true
    rm -f "$dummy_model" 2>/dev/null
    if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
        return 1
    fi
    return 0
}

# ── GPU architecture detection ───────────────────────────────────────────

detect_gpu_arch() {
    # Returns the CUDA architecture number for cmake
    # e.g. 86 for RTX 3090, 89 for RTX 4090, 120 for RTX 5090
    local arch_str
    arch_str=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' .' | sort -n | tail -1)
    echo "${arch_str:-86}"
}

# ── Model helpers ────────────────────────────────────────────────────────

get_dflash_drafts() {
    mapfile -t dflash_drafts < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*dflash*.gguf' -printf '%f\n' 2>/dev/null | sort)
    if [[ ${#dflash_drafts[@]} -eq 0 ]]; then
        mapfile -t dflash_drafts < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*draft*.gguf' -printf '%f\n' 2>/dev/null | sort)
    fi
}

get_mmproj_files() {
    mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*mmproj*.gguf' 2>/dev/null | sort)
}

# ── Install / Update ─────────────────────────────────────────────────────

install_beellama() {
    echo ""
    echo -e " ${CYAN}>>> INSTALL / UPDATE beellama.cpp (Anbeeld/beellama.cpp) <<<${RESET}"
    echo ""

    if [[ ! -d "$BEELLAMA_DIR" ]]; then
        echo " Cloning Anbeeld/beellama.cpp..."
        git clone https://github.com/Anbeeld/beellama.cpp.git "$BEELLAMA_DIR"
    else
        echo " Pulling latest beellama.cpp..."
        cd "$BEELLAMA_DIR"
        git stash --include-untracked 2>/dev/null || true
        git checkout main 2>/dev/null || true
        git reset --hard origin/main 2>/dev/null || git pull --ff-only
        OLD_HASH=$(git rev-parse HEAD)
        cd ..
    fi

    local gpu_arch=$(detect_gpu_arch)

    # Verify we got the latest code
    NEW_HASH=$(cd "$BEELLAMA_DIR" && git rev-parse HEAD)
    if [[ -n "$OLD_HASH" && "$OLD_HASH" != "$NEW_HASH" ]]; then
        echo -e " ${GREEN}Updated: ${OLD_HASH:0:9} → ${NEW_HASH:0:9}${RESET}"
    elif [[ -n "$OLD_HASH" ]]; then
        echo -e " ${YELLOW}Already up to date (${NEW_HASH:0:9})${RESET}"
    fi

    echo ""
    echo " Compiling for sm_${gpu_arch} with CUDA + Flash Attention + TurboQuant/TCQ..."
    echo "   -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON"
    echo ""

    export CC=gcc
    export CXX=g++

    cd "$BEELLAMA_DIR"
    rm -rf build

    echo "--- CMAKE CONFIGURE ---" > "../$DEBUG_LOG"
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${gpu_arch}" \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=ON \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_HOST_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee -a "../$DEBUG_LOG"

    # ── Patch known build issues in beellama.cpp ──
    echo ""
    echo " Patching known build issues..."

    # Fix 1: llama-context.cpp — auto* can't deduce between nullptr and function pointer
    if grep -q 'const auto \* cb_eval_new = dflash_graph_hidden_ready' src/llama-context.cpp 2>/dev/null; then
        sed -i 's|const auto \* cb_eval_new = dflash_graph_hidden_ready ? nullptr : dflash_eval_callback;|ggml_backend_sched_eval_callback cb_eval_new = dflash_graph_hidden_ready ? nullptr : dflash_eval_callback;|' src/llama-context.cpp
        echo "   [llama-context.cpp] Fixed auto* type deduction for cb_eval_new"
    fi

    # Fix 2: server-context.cpp — missing #include <cfloat> for FLT_MAX
    if grep -q 'FLT_MAX' tools/server/server-context.cpp 2>/dev/null && ! grep -q '#include <cfloat>' tools/server/server-context.cpp 2>/dev/null; then
        sed -i '/#include <cmath>/i #include <cfloat>' tools/server/server-context.cpp
        echo "   [server-context.cpp] Added missing #include <cfloat>"
    fi

    # Fix 3: test-reasoning-budget.cpp — missing #include <climits> for INT_MAX
    if grep -q 'INT_MAX' tests/test-reasoning-budget.cpp 2>/dev/null && ! grep -q '#include <climits>' tests/test-reasoning-budget.cpp 2>/dev/null; then
        sed -i '1i #include <climits>' tests/test-reasoning-budget.cpp
        echo "   [test-reasoning-budget.cpp] Added missing #include <climits>"
    fi

    echo "--- CMAKE BUILD (llama-server only) ---" >> "../$DEBUG_LOG"
    cmake --build build --config Release -j$(nproc) --target llama-server 2>&1 | tee -a "../$DEBUG_LOG"
    BUILD_STATUS=${PIPESTATUS[0]}
    cd ..

    if [ $BUILD_STATUS -ne 0 ]; then
        echo -e "\n ${RED}[!] COMPILE FAILED.${RESET}"
        echo " Raw error logs: $DEBUG_LOG"
        read -p " Press Enter to return to menu..."
    else
        echo -e "\n ${GREEN}Build Complete!${RESET}"
        sleep 2
    fi
}

# ── Start BeeLlama DFlash server ────────────────────────────────────────

start_beellama_server() {
    echo ""
    echo -e " ${CYAN}>>> START BEELLAMA DFLASH SERVER <<<${RESET}"
    echo ""

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo " Server is already running! Please stop it first [4]."
        sleep 2
        return
    fi

    server_bin="./${BEELLAMA_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo " Error: llama-server not found at $server_bin"
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return..."
        return
    fi

    # ── Config preset selection ──
    echo " Select config preset:"
    echo ""
    echo "   [1] Precision    Q5_K_S target + Q4_K_M draft + turbo4/turbo3_tcq"
    echo "                    120K ctx, reasoning ON, vision ON (CPU-offload)"
    echo "                    Best quality, tuned for 24GB VRAM"
    echo "   [2] Speed/VRAM   Q4_K_M target + Q4_K_M draft + turbo3_tcq/turbo3_tcq"
    echo "                    More VRAM headroom, higher ctx or -ub possible"
    echo "   [3] Custom       Pick everything manually"
    read -p " Choice (1-3, default 1): " preset_choice
    preset_choice=$(echo "$preset_choice" | tr -d '[:space:]')
    [[ -z "$preset_choice" ]] && preset_choice="1"

    # ── Draft model selection ──
    echo ""
    get_dflash_drafts
    if [[ ${#dflash_drafts[@]} -eq 0 ]]; then
        echo -e " ${RED}No DFlash draft models found!${RESET}"
        echo " You need a DFlash draft model (e.g. Qwen3.6-27B-DFlash-Q4_K_M.gguf)"
        echo " Download from: https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF"
        echo "         or:   https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF"
        read -p " Press Enter to return..."
        return
    fi

    draft_model=""
    if [[ ${#dflash_drafts[@]} -eq 1 ]]; then
        draft_model="${dflash_drafts[0]}"
        echo " Auto-selected draft model: $draft_model"
    else
        echo " Select DFlash Draft Model:"
        for i in "${!dflash_drafts[@]}"; do
            local d="${dflash_drafts[$i]}"
            local d_low=$(echo "$d" | tr '[:upper:]' '[:lower:]')
            local tag=""
            if [[ "$d_low" == *"q4_k_m"* ]]; then tag="  <-- recommended"; fi
            echo "   [$((i+1))] $d$tag"
        done
        read -p " Choice (default 1): " draft_choice
        draft_choice=$(echo "$draft_choice" | tr -d '[:space:]')
        [[ -z "$draft_choice" ]] && draft_choice="1"
        local didx=$((draft_choice - 1))
        draft_model="${dflash_drafts[$didx]:-${dflash_drafts[0]}}"
    fi
    echo " Draft: $draft_model"

    # Extract compatibility tokens from selected draft (Qwen version like 3.6)
    draft_tokens=()
    d_low=$(echo "$draft_model" | tr '[:upper:]' '[:lower:]')
    # Extract version numbers like 3.6 from tokens like "qwen3.6" or plain "3.6"
    for tok in $(echo "$d_low" | sed 's/[-_]/ /g'); do
        ver=$(echo "$tok" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$ver" ]]; then
            draft_tokens+=("$ver")
        fi
    done

    # ── List compatible target models ──
    raw_data=()
    compat_data=()
    incompt_data=()

    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            [[ "$name" == *"mmproj"* ]] && continue
            local_name_lower=$(echo "$name" | tr '[:upper:]' '[:lower:]')
            [[ "$local_name_lower" == *"dflash"* || "$local_name_lower" == *"draft"* ]] && continue
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo " No target models found in $MODELS_DIR/"
        read -p " Press Enter to return..."
        return
    fi

    for entry in "${raw_data[@]}"; do
        m_name="${entry%%|*}"
        m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
        matched=false
        for tok in "${draft_tokens[@]}"; do
            if [[ "$m_low" == *"qwen${tok}"* ]] || [[ "$m_low" == *"qwen"*"${tok}"* ]]; then
                matched=true
                break
            fi
        done
        if $matched; then
            compat_data+=("$entry")
        else
            incompt_data+=("$entry")
        fi
    done

    echo ""
    if [[ ${#compat_data[@]} -gt 0 ]]; then
        echo " Compatible target models (matched to $draft_model):"
        printf "   %-3s %-64s %-6s %s\n" "NR" "MODEL NAME" "SIZE" "MATCH"
        echo "   ----------------------------------------------------------------------"
        for i in "${!compat_data[@]}"; do
            IFS="|" read -r m_name m_size <<< "${compat_data[$i]}"
            m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
            match_reason=""
            for tok in "${draft_tokens[@]}"; do
                if [[ "$m_low" == *"qwen${tok}"* ]] || [[ "$m_low" == *"qwen"*"${tok}"* ]]; then
                    match_reason="qwen${tok}"
                    break
                fi
            done
            printf "   %2d) %-64s [%-5s] ${GREEN}%s${RESET}\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size" "$match_reason"
        done
    fi

    if [[ ${#incompt_data[@]} -gt 0 ]]; then
        echo ""
        echo " Incompatible models (wrong family for this draft):"
        echo "   ----------------------------------------------------------------------"
        for i in "${!incompt_data[@]}"; do
            IFS="|" read -r m_name m_size <<< "${incompt_data[$i]}"
            printf "       \033[2m%-64s [%-5s]\033[0m\n" "$(echo "$m_name" | cut -c1-64)" "$m_size"
        done
    fi

    # ── Auto-select target based on preset ──
    local auto_target=""
    if [[ "$preset_choice" == "1" ]]; then
        # Precision: prefer Q5_K_S
        for entry in "${compat_data[@]}"; do
            m_name="${entry%%|*}"
            m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
            if [[ "$m_low" == *"q5_k_s"* ]]; then
                auto_target="$m_name"
                break
            fi
        done
    elif [[ "$preset_choice" == "2" ]]; then
        # Speed: prefer Q4_K_M
        for entry in "${compat_data[@]}"; do
            m_name="${entry%%|*}"
            m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
            if [[ "$m_low" == *"q4_k_m"* ]]; then
                auto_target="$m_name"
                break
            fi
        done
    fi

    echo ""
    if [[ -n "$auto_target" ]]; then
        echo -e " Preset auto-selected: ${GREEN}${auto_target}${RESET}"
        read -p " Use this model? (Y/n): " use_auto
        use_auto=$(echo "$use_auto" | tr -d '[:space:]')
        if [[ "$use_auto" == "n" || "$use_auto" == "N" ]]; then
            auto_target=""
        fi
    fi

    if [[ -z "$auto_target" ]]; then
        if [[ ${#compat_data[@]} -gt 0 ]]; then
            read -p " Model NR: " n
        else
            echo -e " ${YELLOW}No compatible models found for $draft_model${RESET}"
            echo " You can still select from incompatible models at your own risk."
            read -p " Enter NR from full list (shown greyed), or Enter to cancel: " n
            compat_data=("${raw_data[@]}")
        fi
        n=$(echo "$n" | tr -d '[:space:]')
        local idx=$(( n - 1 ))
        local entry=${compat_data[$idx]}
        if [[ -z "$entry" ]]; then
            echo " Invalid model number."
            sleep 2
            return
        fi
        target="${entry%%|*}"
    else
        target="$auto_target"
    fi

    # ── Vision (mmproj) ──
    mmproj_model=""
    get_mmproj_files
    if [[ ${#mmproj_files[@]} -gt 0 ]]; then
        # Use the first mmproj found
        mmproj_model=$(basename "${mmproj_files[0]}")
        echo ""
        echo -e " ${GREEN}Found mmproj: ${mmproj_model}${RESET}"
        echo " Vision will be enabled with --no-mmproj-offload (runs on CPU, saves VRAM)."
        read -p " Enable vision? (Y/n): " use_vision
        use_vision=$(echo "$use_vision" | tr -d '[:space:]')
        if [[ "$use_vision" == "n" || "$use_vision" == "N" ]]; then
            mmproj_model=""
        fi
    else
        echo ""
        echo " No mmproj found. Vision disabled."
        echo " Download mmproj-BF16.gguf from unsloth/Qwen3.6-27B-GGUF for vision support."
    fi

    # ── Context size ──
    local model_size_bytes
    model_size_bytes=$(stat -c%s "${MODELS_DIR}/${target}" 2>/dev/null || echo 0)
    local model_size_gb=$(awk "BEGIN{printf \"%.1f\", ${model_size_bytes}/1024/1024/1024}")

    local default_ctx="122800"
    if [[ "$preset_choice" == "1" ]]; then
        default_ctx="122800"
    elif [[ "$preset_choice" == "2" ]]; then
        default_ctx="131072"
    fi

    echo ""
    echo " Model size: ~${model_size_gb} GB"
    echo ""
    echo " Select Context Size:"
    echo "   [1] 32768    (32K — conservative, lots of VRAM headroom)"
    echo "   [2] 65536    (64K — medium)"
    echo "   [3] 122800   (~120K — recommended for 24GB, precision preset)"
    echo "   [4] 131072   (128K — speed/VRAM preset default)"
    echo "   [5] 200000   (~200K — may need clean system, no extra apps)"
    echo "   [6] 262144   (256K — max, may OOM on 24GB)"
    echo "   [7] Custom"
    read -p " Choice (1-7, default 3): " ctx_choice
    ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')
    [[ -z "$ctx_choice" ]] && ctx_choice="3"

    case "$ctx_choice" in
        1) ctx="32768" ;;
        2) ctx="65536" ;;
        3) ctx="122800" ;;
        4) ctx="131072" ;;
        5) ctx="200000" ;;
        6) ctx="262144" ;;
        7)
            read -p " Enter context size: " ctx
            ctx=$(echo "$ctx" | tr -d '[:space:]')
            ;;
        *) ctx="122800" ;;
    esac

    if [[ ! "$ctx" =~ ^[0-9]+$ ]]; then
        echo " Invalid context size."
        sleep 2
        return
    fi

    # ── KV cache types ──
    local cache_k="turbo4"
    local cache_v="turbo3_tcq"

    if [[ "$preset_choice" == "2" ]]; then
        cache_k="turbo3_tcq"
        cache_v="turbo3_tcq"
    fi

    if [[ "$preset_choice" == "3" ]]; then
        echo ""
        echo " Select K cache type:"
        echo "   [1] turbo4      (4.125 bpv — best precision)"
        echo "   [2] turbo3_tcq  (3.25 bpv — good balance, CUDA only)"
        echo "   [3] turbo3      (3.125 bpv — no TCQ)"
        echo "   [4] turbo2_tcq  (2.25 bpv — max compression, CUDA only)"
        echo "   [5] q4_0        (fallback)"
        echo "   [6] q8_0        (legacy fallback)"
        read -p " Choice (1-6, default 1): " ck_choice
        ck_choice=$(echo "$ck_choice" | tr -d '[:space:]')
        [[ -z "$ck_choice" ]] && ck_choice="1"
        case "$ck_choice" in
            1) cache_k="turbo4" ;;
            2) cache_k="turbo3_tcq" ;;
            3) cache_k="turbo3" ;;
            4) cache_k="turbo2_tcq" ;;
            5) cache_k="q4_0" ;;
            6) cache_k="q8_0" ;;
        esac

        echo ""
        echo " Select V cache type:"
        echo "   [1] turbo3_tcq  (3.25 bpv — good balance, CUDA only)"
        echo "   [2] turbo4      (4.125 bpv — best precision)"
        echo "   [3] turbo3      (3.125 bpv — no TCQ)"
        echo "   [4] turbo2_tcq  (2.25 bpv — max compression, CUDA only)"
        echo "   [5] q4_0        (fallback)"
        echo "   [6] q8_0        (legacy fallback)"
        read -p " Choice (1-6, default 1): " cv_choice
        cv_choice=$(echo "$cv_choice" | tr -d '[:space:]')
        [[ -z "$cv_choice" ]] && cv_choice="1"
        case "$cv_choice" in
            1) cache_v="turbo3_tcq" ;;
            2) cache_v="turbo4" ;;
            3) cache_v="turbo3" ;;
            4) cache_v="turbo2_tcq" ;;
            5) cache_v="q4_0" ;;
            6) cache_v="q8_0" ;;
        esac
    fi

    # ── DFlash cross-ctx ──
    local cross_ctx="1024"
    if [[ "$preset_choice" == "3" ]]; then
        echo ""
        echo " DFlash cross-attention context (hidden state window for drafter):"
        echo "   [1] 512   (default — lower VRAM)"
        echo "   [2] 1024  (recommended for longer context)"
        echo "   [3] 2048  (experimental — more VRAM)"
        read -p " Choice (1-3, default 2): " cc_choice
        cc_choice=$(echo "$cc_choice" | tr -d '[:space:]')
        [[ -z "$cc_choice" ]] && cc_choice="2"
        case "$cc_choice" in
            1) cross_ctx="512" ;;
            2) cross_ctx="1024" ;;
            3) cross_ctx="2048" ;;
        esac
    fi

    # ── Reasoning ──
    local reasoning="on"
    if [[ "$preset_choice" == "3" ]]; then
        echo ""
        echo " Reasoning mode (thinking tokens give drafter richer context):"
        echo "   [1] on   (recommended — improves DFlash acceptance rates)"
        echo "   [2] off  (if task doesn't benefit from reasoning)"
        read -p " Choice (1-2, default 1): " r_choice
        r_choice=$(echo "$r_choice" | tr -d '[:space:]')
        [[ -z "$r_choice" ]] && r_choice="1"
        case "$r_choice" in
            2) reasoning="off" ;;
        esac
    fi

    # ── Microbatch size ──
    local ub_size="256"
    if [[ "$preset_choice" == "3" ]]; then
        echo ""
        echo " Physical microbatch size (-ub):"
        echo "   [1] 256   (safe default)"
        echo "   [2] 512   (faster prefill if VRAM allows)"
        echo "   [3] 1024  (aggressive — needs spare VRAM)"
        read -p " Choice (1-3, default 1): " ub_choice
        ub_choice=$(echo "$ub_choice" | tr -d '[:space:]')
        [[ -z "$ub_choice" ]] && ub_choice="1"
        case "$ub_choice" in
            2) ub_size="512" ;;
            3) ub_size="1024" ;;
        esac
    fi

    # ── Build command ──
    cmd=("$server_bin"
        -m "${MODELS_DIR}/${target}"
        --spec-draft-model "${MODELS_DIR}/${draft_model}"
        --spec-type dflash
        --spec-dflash-cross-ctx "$cross_ctx"
        -np 1
        --kv-unified
        -ngl all
        --spec-draft-ngl all
        -b 2048 -ub "$ub_size"
        --ctx-size "$ctx"
        --cache-type-k "$cache_k"
        --cache-type-v "$cache_v"
        --flash-attn on
        --cache-ram 0
        --jinja
        --no-mmap --mlock
        --no-host --metrics
        --log-timestamps --log-prefix --log-colors off
        --reasoning "$reasoning"
        --temp 0.6 --top-k 20 --min-p 0.0
        --host 0.0.0.0
        --port 8080
    )

    # Add mmproj if selected
    if [[ -n "$mmproj_model" ]]; then
        cmd+=(
            --mmproj "${MODELS_DIR}/${mmproj_model}"
            --no-mmproj-offload
        )
    fi

    # Add chat-template-kwargs if reasoning is on
    if [[ "$reasoning" == "on" ]]; then
        cmd+=(--chat-template-kwargs '{"preserve_thinking":true}')
    fi

    # Note: --spec-draft-temp is not in the reference quickstart and may affect quality.
    # Uncomment below only if you know you want sampled DFlash.
    # if arg_probe_valid "$server_bin" --spec-draft-temp auto; then
    #     cmd+=(--spec-draft-temp auto)
    # fi

    # State file
    echo "BEELLAMA: ${target} + Draft(${draft_model}) [ctx=${ctx}, K=${cache_k}, V=${cache_v}]" > .server_info_beellama

    echo ""
    echo " Starting BeeLlama DFlash server:"
    echo "   Main:       $target"
    echo "   Draft:      $draft_model"
    echo "   Context:    $ctx"
    echo "   K cache:    $cache_k"
    echo "   V cache:    $cache_v"
    echo "   Cross-ctx:  $cross_ctx"
    echo "   Reasoning:  $reasoning"
    echo "   Vision:     ${mmproj_model:-disabled}"
    echo "   -ub:        $ub_size"
    echo ""
    echo " Command:"
    printf ' %q' "${cmd[@]}"
    echo ""
    echo ""

    {
        echo "COMMAND PROFILE: beellama-dflash"
        printf '%q ' "${cmd[@]}"
        echo ""
        echo "----- llama-server output -----"
    } > "$SERVER_LOG"

    nohup "${cmd[@]}" >> "$SERVER_LOG" 2>&1 &
    server_pid=$!

    failed=0
    loaded=0

    for i in $(seq 1 180); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            failed=1
            break
        fi

        if grep -Eqi 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$SERVER_LOG"; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" >/dev/null 2>&1 || true
            failed=1
            break
        fi

        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            loaded=1
            break
        fi

        sleep 1
    done

    if [[ "$failed" == "0" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
        echo ""
        if [[ "$loaded" == "1" ]]; then
            echo " Server loaded and health check returned OK."
        else
            echo " Server process is running. It may still be loading."
        fi

        echo ""
        echo " Last 35 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 35 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        sleep 3
    else
        rm -f .server_info_beellama
        echo ""
        echo -e " ${RED}Server failed during startup.${RESET}"
        echo " Last 220 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 220 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        read -p " Press Enter to return to menu..."
    fi
}

if [[ "${1:-}" == "--quickstart" ]]; then

    echo ""
    echo -e " ${BOLD}${CYAN}=============================================${RESET}"
    echo -e " ${BOLD}${CYAN} HostLLM — Quick Start (BeeLlama DFlash)${RESET}"
    echo -e " ${BOLD}${CYAN}=============================================${RESET}"
    echo ""

    # -- Step 0: Pick model ---------------------------------------------
    # Model catalog: name|file|size_gb|min_vram|cache_k|cache_v|ctx_per_gb|description
    QS_MODELS=(
        "NEO-CODE IQ4_XS|Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf|https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF/resolve/main/Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf|15|16|turbo3_tcq|turbo3_tcq|38000|⚡ Speed king — 262K ctx, up to 105 tok/s"
        "NEO-CODE Q5_K_M|Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q5_K_M.gguf|https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF/resolve/main/Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q5_K_M.gguf|19|20|turbo4|turbo3_tcq|71000|✨ Best quality — 200K ctx, up to 95 tok/s"
        "Qwen3.6 Q4_K_M|Qwen3.6-27B-Q4_K_M.gguf|https://huggingface.co/bartowski/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf|16|17|turbo3_tcq|turbo3_tcq|42000|⚖️ Balanced — 240K ctx, up to 100 tok/s"
        "Heretic v2 MTP Q4_K_S|Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf|https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/resolve/main/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf|16|17|turbo3_tcq|turbo3_tcq|40000|🔥 Uncensored heretic — 240K ctx, MTP preserved"
        "HauhauCS Aggressive IQ4_XS|Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf|https://huggingface.co/spiritbuun/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-GGUF/resolve/main/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf|15|16|turbo3_tcq|turbo3_tcq|38000|💀 Uncensored aggressive — 262K ctx, up to 105 tok/s"
    )

    echo -e " Pick your model:"
    echo ""
    for i in "${!QS_MODELS[@]}"; do
        IFS='|' read -r name file url size_gb min_vram ck cv ctx_pg desc <<< "${QS_MODELS[$i]}"
        local_marker=""
        if [[ -f "${MODELS_DIR}/${file}" ]]; then
            local_marker=" ${GREEN}[cached]${RESET}"
        fi
        echo -e "  ${BOLD}[$((i+1))]${RESET} ${desc}  (~${size_gb} GB)${local_marker}"
    done
    echo ""
    read -p " Select [1-${#QS_MODELS[@]}] (default=1): " model_pick
    model_pick=$(echo "$model_pick" | tr -d '[:space:]')
    [[ -z "$model_pick" ]] && model_pick=1

    # Validate selection
    if ! [[ "$model_pick" =~ ^[0-9]+$ ]] || [[ "$model_pick" -lt 1 ]] || [[ "$model_pick" -gt ${#QS_MODELS[@]} ]]; then
        echo -e " ${RED}Invalid selection. Using model 1.${RESET}"
        model_pick=1
    fi

    model_idx=$((model_pick - 1))
    IFS='|' read -r QS_TARGET_LABEL QS_TARGET QS_TARGET_URL QS_MODEL_GB QS_MIN_VRAM QS_CACHE_K QS_CACHE_V QS_CTX_PER_GB QS_DESC <<< "${QS_MODELS[$model_idx]}"

    QS_DRAFT="Qwen3.6-27B-DFlash-Q5_K_M.gguf"
    QS_DRAFT_URL="https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q5_K_M.gguf"
    QS_MMPROJ="mmproj-BF16.gguf"
    QS_MMPROJ_URL="https://huggingface.co/DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF/resolve/main/mmproj-BF16.gguf"

    echo -e " Selected: ${GREEN}${QS_TARGET_LABEL}${RESET} (~${QS_MODEL_GB} GB)"
    echo ""

    # -- Step 1: Build binary if missing ----------------------------------
    server_bin="./${BEELLAMA_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo -e " ${YELLOW}[1/4]${RESET} Binary not found. Building BeeLlama.cpp..."
        echo ""
        install_beellama
        if [[ ! -x "$server_bin" ]]; then
            echo -e " ${RED}Build failed. Cannot continue.${RESET}"
            read -p " Press Enter to exit..."
            exit 1
        fi
    else
        echo -e " ${GREEN}[1/4]${RESET} Binary ready."
    fi

    # -- Step 2: Download models if missing --------------------------------
    missing=0
    [[ ! -f "${MODELS_DIR}/${QS_TARGET}" ]] && missing=$((missing + 1))
    [[ ! -f "${MODELS_DIR}/${QS_DRAFT}" ]]  && missing=$((missing + 1))
    [[ ! -f "${MODELS_DIR}/${QS_MMPROJ}" ]] && missing=$((missing + 1))

    if [[ $missing -gt 0 ]]; then
        echo ""
        echo -e " ${YELLOW}[2/4]${RESET} Downloading ${missing} model(s)..."
        total_dl_gb=0
        [[ ! -f "${MODELS_DIR}/${QS_TARGET}" ]] && total_dl_gb=$((total_dl_gb + QS_MODEL_GB))
        [[ ! -f "${MODELS_DIR}/${QS_DRAFT}" ]]  && total_dl_gb=$((total_dl_gb + 2))
        [[ ! -f "${MODELS_DIR}/${QS_MMPROJ}" ]] && total_dl_gb=$((total_dl_gb + 1))
        echo "   Total download: ~${total_dl_gb} GB (${QS_TARGET_LABEL})"
        echo ""

        if [[ ! -f "${MODELS_DIR}/${QS_TARGET}" ]]; then
            echo "   Downloading target: ${QS_TARGET}..."
            wget --show-progress -O "${MODELS_DIR}/${QS_TARGET}" "$QS_TARGET_URL"
            if [[ $? -ne 0 ]]; then
                echo -e " ${RED}Target model download failed.${RESET}"
                rm -f "${MODELS_DIR}/${QS_TARGET}"
                read -p " Press Enter to exit..."
                exit 1
            fi
        fi

        if [[ ! -f "${MODELS_DIR}/${QS_DRAFT}" ]]; then
            echo "   Downloading draft: ${QS_DRAFT} (~1.2 GB)..."
            wget --show-progress -O "${MODELS_DIR}/${QS_DRAFT}" "$QS_DRAFT_URL"
            if [[ $? -ne 0 ]]; then
                echo -e " ${RED}Draft model download failed.${RESET}"
                rm -f "${MODELS_DIR}/${QS_DRAFT}"
                read -p " Press Enter to exit..."
                exit 1
            fi
        fi

        if [[ ! -f "${MODELS_DIR}/${QS_MMPROJ}" ]]; then
            echo "   Downloading mmproj: ${QS_MMPROJ} (~0.9 GB)..."
            wget --show-progress -O "${MODELS_DIR}/${QS_MMPROJ}" "$QS_MMPROJ_URL"
            if [[ $? -ne 0 ]]; then
                echo -e " ${YELLOW}mmproj download failed. Continuing without vision.${RESET}"
                rm -f "${MODELS_DIR}/${QS_MMPROJ}"
            fi
        fi
    else
        echo -e " ${GREEN}[2/4]${RESET} All models ready."
    fi

    # -- Step 3: Detect GPUs, calculate context -----------------------------
    echo ""
    echo -e " ${YELLOW}[3/4]${RESET} Detecting hardware..."

    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ "$GPU_COUNT" -eq 0 ]]; then
        echo -e " ${RED}No NVIDIA GPUs detected.${RESET}"
        read -p " Press Enter to exit..."
        exit 1
    fi

    TOTAL_VRAM_MB=0
    while read -r vram; do
        TOTAL_VRAM_MB=$((TOTAL_VRAM_MB + vram))
    done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
    TOTAL_VRAM_GB=$((TOTAL_VRAM_MB / 1024))

    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' '|' | sed 's/|$//')

    echo "   GPUs:        ${GPU_NAMES}"
    echo "   Total VRAM:  ${TOTAL_VRAM_GB} GB (${GPU_COUNT} GPU(s))"

    if [[ "$TOTAL_VRAM_GB" -lt "$QS_MIN_VRAM" ]]; then
        echo ""
        echo -e " ${RED}Not enough VRAM. Need at least ${QS_MIN_VRAM} GB total for ${QS_TARGET_LABEL}, found ${TOTAL_VRAM_GB} GB.${RESET}"
        read -p " Press Enter to exit..."
        exit 1
    fi

    # Context calculation for BeeLlama with TurboQuant
    # Calibrated from real 3090 (24GB) data points:
    #   IQ4_XS: 15+2=17GB used → 6.8GB free → 262K ctx (turbo3_tcq/turbo3_tcq)  ≈ 38K/GB
    #   Q5_K_M: 19+2=21GB used → 2.8GB free → 200K ctx (turbo4/turbo3_tcq)      ≈ 71K/GB
    # Multi-GPU: no draft model (DFlash off), so more VRAM available
    DRAFT_GB=0
    [[ "$GPU_COUNT" -eq 1 ]] && DRAFT_GB=2
    OVERHEAD_GB=1

    if [[ "$GPU_COUNT" -gt 1 ]]; then
        PER_GPU_GB=$(( TOTAL_VRAM_GB / GPU_COUNT ))
        AVAIL_GB=$(( PER_GPU_GB - (QS_MODEL_GB / GPU_COUNT) - (DRAFT_GB / GPU_COUNT) - OVERHEAD_GB ))
    else
        AVAIL_GB=$(( TOTAL_VRAM_GB - QS_MODEL_GB - DRAFT_GB - OVERHEAD_GB ))
    fi

    if [[ "$AVAIL_GB" -lt 1 ]]; then
        echo -e " ${YELLOW}Warning: Very tight VRAM. Context will be minimal (8K).${RESET}"
        AVAIL_GB=0
    fi

    CTX=$(( AVAIL_GB * QS_CTX_PER_GB ))

    # Multi-GPU safety: context not reduced (crash was upstream DFlash bug, not ctx size)

    [[ $CTX -lt 8192 ]]   && CTX=8192
    [[ $CTX -gt 262144 ]] && CTX=262144

    echo "   Target:      ${QS_TARGET_LABEL}"
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        echo "   DFlash:      OFF (multi-GPU upstream bug)"
    else
        echo "   Draft:       Q5_K_M (~1.2 GB)"
    fi
    echo "   Vision:      mmproj-BF16 (CPU offload)"
    echo "   KV cache:    K=${QS_CACHE_K}, V=${QS_CACHE_V}"
    echo "   Context:     ${CTX}"
    echo "   Reasoning:   ON"
    echo ""

    # -- Step 4: Launch server --------------------------------------------
    echo -e " ${YELLOW}[4/4]${RESET} Starting BeeLlama DFlash server..."
    echo ""

    # Multi-GPU workaround: beellama.cpp DFlash crashes with multiple GPUs.
    # The target model's hidden states span two GPUs, and dflash_kv_cache_update_gpu
    # can't gather them — CUDA error: illegal memory access (upstream bug).
    # Fix: disable DFlash for multi-GPU, run without speculative decoding.
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        USE_DFLASH=false
        echo -e " ${YELLOW}Note:${RESET} Multi-GPU detected — DFlash disabled (upstream multi-GPU bug)"
        echo -e "         Server will run without speculative decoding (slower but stable)."
        echo ""
    else
        USE_DFLASH=true
    fi

    # -- Template setup -----------------------------------------------
    BUNDLED_TEMPLATE="chat_templates/qwen3.6-chat_template-v9.jinja"
    TEMPLATE_FLAGS=()
    if [[ -f "$BUNDLED_TEMPLATE" ]]; then
        if arg_probe_valid "$server_bin" --chat-template-file "$BUNDLED_TEMPLATE"; then
            TEMPLATE_FLAGS=(--chat-template-file "$BUNDLED_TEMPLATE")
            echo -e " ${GREEN}Template:${RESET} Using bundled Qwen 3.6 v9 (fixed jinja)"
        else
            echo -e " ${YELLOW}Template:${RESET} --chat-template-file not supported, using built-in"
        fi
    else
        echo -e " ${YELLOW}Template:${RESET} Bundled template not found, using built-in"
    fi

    if [[ "$USE_DFLASH" == true ]]; then
        launch_cmd=("$server_bin"
            -m "${MODELS_DIR}/${QS_TARGET}"
            --spec-draft-model "${MODELS_DIR}/${QS_DRAFT}"
            --spec-type dflash
            --spec-dflash-cross-ctx 1024
            -np 1 --kv-unified
            -ngl all --spec-draft-ngl all
            -b 2048 -ub 256
            --ctx-size "$CTX"
            --cache-type-k ${QS_CACHE_K} --cache-type-v ${QS_CACHE_V}
            --flash-attn on
            --cache-ram 0 --jinja
            --no-mmap --mlock
            --no-host --metrics
            --log-timestamps --log-prefix --log-colors off
            --reasoning on
            --temp 0.6 --top-k 20 --min-p 0.0
            ${TEMPLATE_FLAGS[@]:-}
            --mmproj "${MODELS_DIR}/${QS_MMPROJ}"
            --no-mmproj-offload
            --chat-template-kwargs '{"preserve_thinking":true}'
            --host 0.0.0.0 --port 8080
        )
    else
        launch_cmd=("$server_bin"
            -m "${MODELS_DIR}/${QS_TARGET}"
            -np 1 --kv-unified
            -ngl all
            -b 2048 -ub 256
            --ctx-size "$CTX"
            --cache-type-k ${QS_CACHE_K} --cache-type-v ${QS_CACHE_V}
            --flash-attn on
            --cache-ram 0 --jinja
            --no-mmap --mlock
            --no-host --metrics
            --log-timestamps --log-prefix --log-colors off
            --reasoning on
            --temp 0.6 --top-k 20 --min-p 0.0
            ${TEMPLATE_FLAGS[@]:-}
            --mmproj "${MODELS_DIR}/${QS_MMPROJ}"
            --no-mmproj-offload
            --chat-template-kwargs '{"preserve_thinking":true}'
            --host 0.0.0.0 --port 8080
        )
    fi

    # Only add mmproj if the file exists
    if [[ ! -f "${MODELS_DIR}/${QS_MMPROJ}" ]]; then
        # Remove mmproj flags
        launch_cmd=($(printf '%s\n' "${launch_cmd[@]}" | grep -v 'mmproj' | grep -v 'no-mmproj'))
    fi

    echo " Launch command:"
    printf '  %q' "${launch_cmd[@]}"
    echo ""
    echo ""

    if [[ "$USE_DFLASH" == true ]]; then
        echo "BEELLAMA-QuickStart: ${QS_TARGET} + ${QS_DRAFT} [ctx=${CTX}/${QS_CACHE_K}/${QS_CACHE_V}/GPUs=${GPU_COUNT}]" > .server_info_beellama
    else
        echo "BEELLAMA-QuickStart: ${QS_TARGET} (no DFlash) [ctx=${CTX}/${QS_CACHE_K}/${QS_CACHE_V}/GPUs=${GPU_COUNT}]" > .server_info_beellama
    fi

    {
        echo "COMMAND PROFILE: quickstart-beellama"
        printf '%q ' "${launch_cmd[@]}"
        echo ""
        echo ""
        echo "----- llama-server output -----"
    } > "$SERVER_LOG"

    nohup "${launch_cmd[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    FAILED=0
    LOADED=0

    echo " Waiting for server to load model..."
    for i in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
            FAILED=1; break
        fi
        if grep -Eqi 'unknown argument|unrecognized option|invalid option|error:.*argument|usage:' "$SERVER_LOG" 2>/dev/null; then
            kill "$SERVER_PID" >/dev/null 2>&1 || true
            FAILED=1; break
        fi
        if grep -Eqi 'out of memory|failed to allocate|CUDA error' "$SERVER_LOG" 2>/dev/null; then
            kill "$SERVER_PID" >/dev/null 2>&1 || true
            FAILED=1; break
        fi
        CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health" 2>/dev/null || true)
        if [[ "$CODE" == "200" ]]; then
            LOADED=1; break
        fi
        sleep 1
    done

    if [[ "$FAILED" -eq 1 ]] || ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        rm -f .server_info_beellama
        echo ""
        echo -e " ${RED}Server failed during startup.${RESET}"
        echo ""
        echo " Last 50 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 50 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        read -p " Press Enter to exit..."
        exit 1
    fi

    # -- Detect local IP --------------------------------------------------
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -z "$LOCAL_IP" ]] && LOCAL_IP="localhost"

    # -- Show running dashboard -------------------------------------------
    clear
    echo "=================================================================="
    echo -e "  ${GREEN}${BOLD}BEELLAMA DFLASH SERVER RUNNING${RESET}"
    echo "=================================================================="
    echo ""
    echo "  Model:   ${QS_TARGET}"
    if [[ "$USE_DFLASH" == true ]]; then
        echo "  Draft:   ${QS_DRAFT}"
        echo "  Context: ${CTX}  |  KV: K=${QS_CACHE_K}, V=${QS_CACHE_V}  |  DFlash: cross-ctx 1024"
    else
        echo "  Context: ${CTX}  |  KV: K=${QS_CACHE_K}, V=${QS_CACHE_V}  |  DFlash: OFF (multi-GPU)"
    fi
    echo "  Vision:  ON (CPU offload)  |  Reasoning: ON"
    echo "  GPUs:    ${GPU_COUNT}x (${TOTAL_VRAM_GB} GB total)"
    echo ""
    echo -e "  ${CYAN}${BOLD}Connect from any device on your network:${RESET}"
    echo ""
    echo -e "  ${BOLD}Chat UI:${RESET}       http://${LOCAL_IP}:8080"
    echo -e "  ${BOLD}API Base:${RESET}      http://${LOCAL_IP}:8080/v1"
    echo -e "  ${BOLD}Anthropic:${RESET}     http://${LOCAL_IP}:8080/v1/messages"
    echo ""
    echo -e "  ${YELLOW}API Key:${RESET} any string (e.g. sk-1234) or leave blank"
    echo ""
    echo -e "  ${BOLD}OpenWebUI:${RESET}    OpenAI base URL → http://${LOCAL_IP}:8080/v1"
    echo -e "  ${BOLD}Pi / Codex:${RESET}    OPENAI_API_BASE=http://${LOCAL_IP}:8080/v1"
    echo -e "  ${BOLD}Cline / Continue:${RESET} OpenAI compatible → http://${LOCAL_IP}:8080/v1"
    echo -e "  ${BOLD}Anthropic SDK:${RESET}  base_url → http://${LOCAL_IP}:8080/v1"
    echo "=================================================================="
    echo ""

    LIVE_START=22

    while kill -0 "$SERVER_PID" >/dev/null 2>&1; do
        if command -v nvidia-smi > /dev/null 2>&1; then
            gpu_load_sum=0; vram_used=0; vram_total=0; gpu_temp_max=0; gpu_count=0
            gpu_lines=()
            while IFS=',' read -r load used total temp; do
                load=$(echo "$load" | tr -d ' ')
                used=$(echo "$used" | tr -d ' ')
                total=$(echo "$total" | tr -d ' ')
                temp=$(echo "$temp" | tr -d ' ')
                pct=0; [[ "$total" -gt 0 ]] && pct=$(( (used * 100) / total ))
                u_gb=$(awk "BEGIN {printf \"%.1f\", $used/1024}")
                t_gb=$(awk "BEGIN {printf \"%.0f\", $total/1024}")
                if [[ "$pct" -ge 90 ]]; then c="\033[1;31m"; elif [[ "$pct" -ge 50 ]]; then c="\033[1;33m"; else c="\033[1;32m"; fi
                gpu_lines+=("  GPU ${gpu_count}:  ${load}%   |   VRAM: ${c}${u_gb} GB / ${t_gb} GB (${pct}%)${RESET}   |   Temp: ${temp} degC")
                gpu_load_sum=$((gpu_load_sum + load))
                vram_used=$((vram_used + used))
                vram_total=$((vram_total + total))
                [[ "$temp" -gt "$gpu_temp_max" ]] && gpu_temp_max=$temp
                gpu_count=$((gpu_count + 1))
            done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
            [[ "$gpu_count" -gt 0 ]] && vram_pct=$(( (vram_used * 100) / vram_total )) || vram_pct=0
            vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
            vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
            if [[ "$vram_pct" -ge 90 ]]; then c_vram="\033[1;31m"; elif [[ "$vram_pct" -ge 50 ]]; then c_vram="\033[1;33m"; else c_vram="\033[1;32m"; fi
            total_line="  TOTAL: VRAM: ${c_vram}${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)${RESET}   |   GPUs: ${gpu_count}"
        else
            gpu_lines=("  GPU: N/A")
            total_line="  TOTAL: N/A"
        fi

        read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
        cpu_ap=$((user+nice+system+irq+softirq+steal))
        cpu_tp=$((user+nice+system+idle+iowait+irq+softirq+steal))
        sleep 0.5
        read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
        cpu_ac=$((user+nice+system+irq+softirq+steal))
        cpu_tc=$((user+nice+system+idle+iowait+irq+softirq+steal))
        cpu_diff=$((cpu_tc - cpu_tp))
        cpu_adiff=$((cpu_ac - cpu_ap))
        if [[ "$cpu_diff" -gt 0 ]]; then cpu_pct=$(( (cpu_adiff * 100) / cpu_diff )); else cpu_pct=0; fi

        row=$LIVE_START
        tput sc

        tput cup $row 0; echo -e "  CPU: ${cpu_pct}%\033[K"
        row=$((row + 1))
        for line in "${gpu_lines[@]}"; do
            tput cup $row 0; echo -e "${line}\033[K"
            row=$((row + 1))
        done
        tput cup $row 0; echo -e "${total_line}\033[K"
        row=$((row + 1))
        tput cup $row 0; echo -e "\033[K"
        row=$((row + 1))
        tput cup $row 0; echo -e "  [1] Stop server and return to menu\033[K"
        row=$((row + 1))
        tput cup $row 0; echo -e "  [2] Return to menu (keep server running)\033[K"
        row=$((row + 1))
        tput cup $row 0; echo -e "\033[K"
        row=$((row + 1))
        tput cup $row 0; echo -n "  Select [1/2]: "
        tput rc

        read -t 3 -n 1 qs_choice 2>/dev/null || continue
        echo ""
        case "$qs_choice" in
            1)
                echo ""
                echo -e "  ${YELLOW}Stopping server...${RESET}"
                kill "$SERVER_PID" >/dev/null 2>&1 || true
                wait "$SERVER_PID" >/dev/null 2>&1 || true
                rm -f .server_info_beellama 2>/dev/null
                echo -e "  ${GREEN}Server stopped.${RESET}"
                sleep 1
                exit 0
                ;;
            2)
                exit 0
                ;;
        esac
    done

    # Server died
    echo -e " ${RED}Server process exited unexpectedly.${RESET}"
    echo " Check ${SERVER_LOG} for details."
    rm -f .server_info_beellama 2>/dev/null
    read -p " Press Enter to exit..."
    exit 1
fi
# ── Main menu ────────────────────────────────────────────────────────────

setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""

    # List models
    raw_data=()
    draft_data=()
    compatible_data=()
    incompatible_data=()

    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            local_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')

            if [[ "$local_name" == *"mmproj"* ]]; then
                continue
            fi

            local_name_lower=$(echo "$local_name" | tr '[:upper:]' '[:lower:]')
            if [[ "$local_name_lower" == *"dflash"* || "$local_name_lower" == *"draft"* ]]; then
                size=$(du -h "$f" | cut -f1)
                draft_data+=("${name}|${size}")
                continue
            fi

            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    # Build compatibility tokens from draft model filenames
    draft_tokens=()
    if [[ ${#draft_data[@]} -gt 0 ]]; then
        for d_entry in "${draft_data[@]}"; do
            d_name="${d_entry%%|*}"
            d_low=$(echo "$d_name" | tr '[:upper:]' '[:lower:]')
            # Extract version numbers like 3.6 from tokens like "qwen3.6" or plain "3.6"
            for tok in $(echo "$d_low" | sed 's/[-_]/ /g'); do
                ver=$(echo "$tok" | grep -oE '[0-9]+\.[0-9]+' | head -1)
                if [[ -n "$ver" ]]; then
                    draft_tokens+=("$ver")
                fi
            done
        done
        mapfile -t draft_tokens < <(printf '%s\n' "${draft_tokens[@]}" | sort -u)
    fi

    # Classify target models
    if [[ ${#draft_data[@]} -gt 0 ]] && [[ ${#raw_data[@]} -gt 0 ]]; then
        for entry in "${raw_data[@]}"; do
            m_name="${entry%%|*}"
            m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
            matched=false
            for tok in "${draft_tokens[@]}"; do
                if [[ "$tok" =~ ^[0-9]+\.[0-9]+$ ]]; then
                    if [[ "$m_low" == *"qwen${tok}"* ]] || [[ "$m_low" == *"qwen"*"${tok}"* ]]; then
                        matched=true
                        break
                    fi
                else
                    matched=true
                    break
                fi
            done
            if $matched; then
                compatible_data+=("$entry")
            else
                incompatible_data+=("$entry")
            fi
        done
    else
        compatible_data=("${raw_data[@]}")
    fi

    # Display models
    if [[ ${#compatible_data[@]} -eq 0 && ${#incompatible_data[@]} -eq 0 ]]; then
        echo "   (No .gguf models found in ./$MODELS_DIR/)"
    else
        if [[ ${#compatible_data[@]} -gt 0 ]]; then
            echo ""
            if [[ ${#draft_data[@]} -gt 0 ]]; then
                printf "   %-3s %-64s %-6s %s\n" "NR" "COMPATIBLE TARGET MODELS" "SIZE" "MATCH"
            else
                printf "   %-3s %-64s %s\n" "NR" "TARGET MODELS" "SIZE"
            fi
            echo "   ----------------------------------------------------------------------"
            for i in "${!compatible_data[@]}"; do
                IFS="|" read -r m_name m_size <<< "${compatible_data[$i]}"
                m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
                match_reason=""
                if [[ ${#draft_data[@]} -gt 0 ]]; then
                    for tok in "${draft_tokens[@]}"; do
                        if [[ "$tok" =~ ^[0-9]+\.[0-9]+$ ]]; then
                            if [[ "$m_low" == *"qwen${tok}"* ]] || [[ "$m_low" == *"qwen"*"${tok}"* ]]; then
                                match_reason="qwen${tok}"
                                break
                            fi
                        fi
                    done
                fi
                if [[ -n "$match_reason" ]]; then
                    printf "   %2d) %-64s [%-5s] ${GREEN}%s${RESET}\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size" "$match_reason"
                else
                    printf "   %2d) %-64s [%-5s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
                fi
            done
        fi

        if [[ ${#incompatible_data[@]} -gt 0 ]]; then
            echo ""
            printf "   %-64s %s\n" "OTHER MODELS (not matched to draft)" "SIZE"
            echo "   ----------------------------------------------------------------------"
            for i in "${!incompatible_data[@]}"; do
                IFS="|" read -r m_name m_size <<< "${incompatible_data[$i]}"
                printf "       \033[2m%-64s [%-5s]\033[0m\n" "$(echo "$m_name" | cut -c1-64)" "$m_size"
            done
        fi
    fi

    if [[ ${#draft_data[@]} -gt 0 ]]; then
        echo ""
        printf "   %-64s %s\n" "DRAFT MODELS" "SIZE"
        echo "   ----------------------------------------------------------------------"
        for i in "${!draft_data[@]}"; do
            IFS="|" read -r d_name d_size <<< "${draft_data[$i]}"
            printf "       %-64s [%s]\n" "$(echo "$d_name" | cut -c1-64)" "$d_size"
        done
    fi

    # Check mmproj
    mmproj_count=$(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*mmproj*.gguf' 2>/dev/null | wc -l)

    # Check build status
    server_bin="./${BEELLAMA_DIR}/build/bin/llama-server"
    if [[ -x "$server_bin" ]]; then
        echo ""
        echo -e "   Binary: ${GREEN}${server_bin}${RESET}"
    else
        echo ""
        echo -e "   Binary: ${RED}${server_bin} (NOT BUILT — run [0] first)${RESET}"
    fi

    if [[ "$mmproj_count" -gt 0 ]]; then
        echo -e "   Vision: ${GREEN}mmproj found (vision available)${RESET}"
    else
        echo -e "   Vision: ${YELLOW}no mmproj (download for vision support)${RESET}"
    fi

    echo ""
    echo -e " ${CYAN}--- SETUP ---${RESET}"
    echo " [0] Install / Update beellama.cpp (Anbeeld/beellama.cpp)"
    echo ""
    echo -e " ${CYAN}--- BEELLAMA DFLASH SERVER ---${RESET}"
    echo " [1] Start DFlash Server (Precision preset — Q5_K_S + turbo4 + reasoning ON)"
    echo " [2] Start DFlash Server (Speed preset — Q4_K_M + turbo3_tcq)"
    echo " [3] Start DFlash Server (Custom — pick everything)"
    echo " [4] Stop Server"
    echo ""
    echo -e " ${CYAN}--- MANAGEMENT ---${RESET}"
    echo " [5] Download Model (.gguf URL)"
    echo " [6] Download DFlash Draft Model"
    echo " [7] Download mmproj (vision projector)"
    echo " [8] Delete Model"
    echo " [99] Back to Main Menu"
    echo " [98] Exit"
    echo ""

    tput cnorm
    read -p " Select Action: " action
    action=$(echo "$action" | tr -d '[:space:]')

    case $action in
        0)
            kill -9 "$MONITOR_PID" 2>/dev/null
            wait "$MONITOR_PID" 2>/dev/null
            install_beellama
            setup_scroll_region
            monitor_loop &
            MONITOR_PID=$!
            ;;
        1)
            preset_choice="1"
            start_beellama_server
            ;;
        2)
            preset_choice="2"
            start_beellama_server
            ;;
        3)
            preset_choice="3"
            start_beellama_server
            ;;
        4)
            echo ""
            echo -e " ${CYAN}>>> STOPPING SERVER <<<${RESET}"
            pkill -f "llama-server"
            rm -f .server_info_beellama
            echo " Server stopped."
            sleep 1
            ;;
        5)
            echo ""
            echo -e " ${CYAN}>>> DOWNLOAD TARGET MODEL <<<${RESET}"
            echo " Paste the direct download URL for a .gguf file."
            echo ""
            echo " Q5_K_S (precision):"
            echo "   https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q5_K_S.gguf"
            echo " Q4_K_M (speed/VRAM):"
            echo "   https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf"
            echo " IQ4_XS (extreme VRAM):"
            echo "   https://huggingface.co/cHunter789/Qwen3.6-27B-i1-IQ4_XS-GGUF/resolve/main/Qwen3.6-27B.i1-IQ4_XS.gguf"
            echo ""
            read -p " URL: " url
            url=$(echo "$url" | tr -d '[:space:]')
            if [[ -n "$url" ]]; then
                url=$(echo "$url" | sed 's|/blob/|/resolve/|')
                filename=$(basename "${url%%\?*}")
                echo " Downloading $filename..."
                wget --show-progress -O "${MODELS_DIR}/${filename}" "$url"
            fi
            ;;
        6)
            echo ""
            echo -e " ${CYAN}>>> DOWNLOAD DFLASH DRAFT MODEL <<<${RESET}"
            echo ""
            echo " Available DFlash draft models:"
            echo ""
            echo " [1] Q4_K_M  (~2.7 GB, recommended)"
            echo "   https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q4_K_M.gguf"
            echo ""
            echo " [2] Q5_K_M  (~3.1 GB, more precision)"
            echo "   https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q5_K_M.gguf"
            echo ""
            echo " [3] Q8_0    (~4.8 GB, highest draft precision but slower)"
            echo "   https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q8_0.gguf"
            echo ""
            echo " [4] IQ4_XS  (~1.5 GB, smallest, speed/VRAM combo)"
            echo "   https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
            echo ""
            echo " [5] Custom URL"
            echo ""
            read -p " Choice (1-5): " draft_dl
            draft_dl=$(echo "$draft_dl" | tr -d '[:space:]')

            case "$draft_dl" in
                1)
                    url="https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q4_K_M.gguf"
                    filename="Qwen3.6-27B-DFlash-Q4_K_M.gguf"
                    ;;
                2)
                    url="https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q5_K_M.gguf"
                    filename="Qwen3.6-27B-DFlash-Q5_K_M.gguf"
                    ;;
                3)
                    url="https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-Q8_0.gguf"
                    filename="Qwen3.6-27B-DFlash-Q8_0.gguf"
                    ;;
                4)
                    url="https://huggingface.co/Ardenzard/Qwen3.6-27B-DFlash-GGUF/resolve/main/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
                    filename="Qwen3.6-27B-DFlash-IQ4_XS.gguf"
                    ;;
                5)
                    read -p " URL: " url
                    url=$(echo "$url" | tr -d '[:space:]')
                    url=$(echo "$url" | sed 's|/blob/|/resolve/|')
                    filename=$(basename "${url%%\?*}")
                    ;;
                *) continue ;;
            esac

            if [[ -n "$url" ]]; then
                echo " Downloading $filename..."
                wget --show-progress -O "${MODELS_DIR}/${filename}" "$url"
            fi
            ;;
        7)
            echo ""
            echo -e " ${CYAN}>>> DOWNLOAD MMPROJ (VISION PROJECTOR) <<<${RESET}"
            echo ""
            url="https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/mmproj-BF16.gguf"
            echo " Downloading mmproj-BF16.gguf from unsloth/Qwen3.6-27B-GGUF..."
            wget --show-progress -O "${MODELS_DIR}/mmproj-BF16.gguf" "$url"
            ;;
        8)
            echo ""
            echo -e " ${CYAN}>>> DELETE MODEL <<<${RESET}"

            all_models=()
            if [[ -d "$MODELS_DIR" ]]; then
                for f in "$MODELS_DIR"/*.gguf; do
                    [[ -e "$f" ]] || continue
                    all_models+=("$(basename "$f")")
                done
            fi

            if [[ ${#all_models[@]} -eq 0 ]]; then
                echo " No models found."
                sleep 2
                continue
            fi

            for i in "${!all_models[@]}"; do
                printf "   %2d) %s\n" "$((i+1))" "${all_models[$i]}"
            done

            read -p " Select model NR to delete: " n
            n=$(echo "$n" | tr -d '[:space:]')
            local_idx=$((n - 1))
            del_target="${all_models[$local_idx]}"
            if [[ -n "$del_target" ]]; then
                read -p " Delete '${del_target}'? (y/N): " confirm
                confirm=$(echo "$confirm" | tr -d '[:space:]')
                if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
                    rm "${MODELS_DIR}/${del_target}"
                    echo " Deleted $del_target"
                else
                    echo " Canceled."
                fi
                sleep 1
            else
                echo " Invalid model number."
                sleep 2
            fi
            ;;
        98) exit 42 ;;
        99) exit 0 ;;
        *) ;;
    esac
done
