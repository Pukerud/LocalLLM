#!/bin/bash

# =========================================================================
# LLAMA.CPP NATIVE MTP DASHBOARD v1.0
# Uses ggml-org/llama.cpp with MTP PR #22673
# Native Multi-Token Prediction for Qwen3.6-27B (no separate draft model)
# https://www.reddit.com/r/LocalLLaMA/comments/1t57xuu/
# Fixed chat templates: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
# =========================================================================

set +m

MTP_DIR="llama_cpp_mtp"
MODELS_DIR="llama_models"
TEMPLATES_DIR="${MTP_DIR}/chat_templates"
DEBUG_LOG="mtp_compile_debug.log"
SERVER_LOG="server_mtp.log"
HF_CACHE="hf_models"

declare -A speed_cache

mkdir -p "$MODELS_DIR"
mkdir -p "$TEMPLATES_DIR"
mkdir -p "$HF_CACHE"

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BLUE=$(tput setaf 4); BOLD=$(tput bold); RESET=$(tput sgr0)

for cmd in curl jq wget git cmake gcc g++; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd" build-essential
    fi
done

if ! command -v /usr/local/cuda/bin/nvcc > /dev/null 2>&1; then
    echo ""
    # Detect highest GPU arch to pick the right CUDA version
    MAX_GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' .' | sort -n | tail -1)
    if [[ "$MAX_GPU_ARCH" -ge 120 ]]; then
        CUDA_PKG="cuda-toolkit-12-8"
        echo "CUDA Toolkit not found. Blackwell GPU detected (sm_${MAX_GPU_ARCH}), installing ${CUDA_PKG}..."
    else
        CUDA_PKG="cuda-toolkit-12-4"
        echo "CUDA Toolkit not found. Installing ${CUDA_PKG}..."
    fi

    # Detect Ubuntu codename for the correct repo URL
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "jammy")
    case "$UBUNTU_CODENAME" in
        focal|groovy|hirsute|impish) REPO_DISTRO="ubuntu2004" ;;
        *) REPO_DISTRO="ubuntu2204" ;;
    esac

    # Remove any stale manual repo/keyring from previous attempts
    sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/keyrings/cuda-keyring.gpg

    # Use NVIDIA's .deb keyring package (handles GPG + repo setup cleanly)
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
        echo " Or check /etc/apt/sources.list.d/"
        read -p " Press Enter to exit..."
        exit 1
    fi

    echo "${CUDA_PKG} installed successfully."
fi

# Variables for quickstart model (used by both quickstart and main menu)
QUICKSTART_MODEL_URL="https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/resolve/main/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf"
QUICKSTART_MODEL="Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_S.gguf"

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

# -- Dashboard Monitoring -------------------------------------------------

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

    SERVER_PID=$(pgrep -f "llama-server" | head -n 1)
    if [[ -n "$SERVER_PID" ]]; then
        if [[ -f ".server_info_mtp" ]]; then
            ACTIVE_INFO=$(cat .server_info_mtp)
            SERVER_STATUS="\033[1;32mRUNNING: ${ACTIVE_INFO}\033[0m"
        else
            SERVER_STATUS="\033[1;32mRUNNING (PID: $SERVER_PID)\033[0m"
        fi
    else
        SERVER_STATUS="\033[1;31mSTOPPED\033[0m"
        rm -f .server_info_mtp 2>/dev/null
    fi

    reset="\033[0m"; bold="\033[1m"

    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${bold}llama.cpp (MTP PR #22673)${reset}    |  SERVER: ${SERVER_STATUS}\033[K"
    tput cup 3 0
    echo -e "   CPU: ${c_cpu}${cpu_pct}%${reset}   |   GPU: ${gpu_load}%   |   Temp: ${gpu_temp} degC\033[K"
    tput cup 4 0
    echo -e "   VRAM: ${c_vram}${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)${reset}\033[K"
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
    tput csr 7 "$(tput lines)"
    tput cup 0 0
    echo "==========================================================================================================================="
    echo "   LLAMA.CPP NATIVE MTP DASHBOARD v1.0  --  ggml-org/llama.cpp PR #22673"
    echo "==========================================================================================================================="
    tput cup 5 0
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

# -- Argument Probe --------------------------------------------------------

arg_probe_valid() {
    local server_bin="$1"
    shift
    local probe_log=".mtp_arg_probe.log"
    local dummy_model=".mtp_arg_probe_dummy.gguf"
    : > "$dummy_model"
    local -a test_cmd
    test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18097)
    timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true
    rm -f "$dummy_model" 2>/dev/null
    if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
        return 1
    fi
    return 0
}

# -- Install / Update -----------------------------------------------------

install_mtp() {
    echo ""
    echo -e " \033[1;36m>>> INSTALL / UPDATE llama.cpp (MTP PR #22673) <<<\033[0m"
    echo ""
    echo " This fetches ggml-org/llama.cpp with MTP speculative decoding support."
    echo " PR: https://github.com/ggml-org/llama.cpp/pull/22673"
    echo ""

    local EXPECTED_REMOTE="https://github.com/ggml-org/llama.cpp"
    local NEED_CLONE=false

    if [[ ! -d "$MTP_DIR" ]]; then
        NEED_CLONE=true
    else
        # Verify the remote points to ggml-org/llama.cpp, not some other repo
        local current_remote
        current_remote=$(cd "$MTP_DIR" && git remote get-url origin 2>/dev/null || echo "")
        if [[ "$current_remote" != *"ggml-org/llama.cpp"* ]] && [[ "$current_remote" != *"llama.cpp"* ]]; then
            echo -e " \033[1;33mWarning: ${MTP_DIR}/ exists but origin points to ${current_remote}\033[0m"
            echo " Expected: ${EXPECTED_REMOTE}"
            echo " Removing stale directory and re-cloning..."
            rm -rf "$MTP_DIR"
            NEED_CLONE=true
        fi
    fi

    if $NEED_CLONE; then
        echo " Cloning ggml-org/llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "$MTP_DIR"
    else
        echo " Updating llama.cpp..."
        cd "$MTP_DIR" && git fetch --all && cd ..
    fi

    echo ""
    echo " Fetching MTP PR #22673..."
    cd "$MTP_DIR"

    # Switch off mtp-pr before deleting it (can't delete current branch)
    git checkout master 2>/dev/null || git checkout main 2>/dev/null
    git branch -D mtp-pr 2>/dev/null || true

    if ! git fetch origin pull/22673/head:mtp-pr 2>&1; then
        echo ""
        echo -e " \033[1;33mDirect PR fetch failed. Trying unshallow...\033[0m"
        git fetch --unshallow 2>/dev/null
        if ! git fetch origin pull/22673/head:mtp-pr 2>&1; then
            echo -e " \033[1;31mFailed to fetch PR #22673. Cannot build MTP binary.\033[0m"
            echo " The PR may have been merged, closed, or network issues."
            echo " Check: https://github.com/ggml-org/llama.cpp/pull/22673"
            cd ..
            read -p " Press Enter to return to menu..."
            return
        else
            git checkout mtp-pr 2>&1
            cd ..
        fi
    else
        git checkout mtp-pr 2>&1
        cd ..
    fi

    echo ""
    # Auto-detect all GPU compute capabilities
    CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' .' | sort -u | tr '\n' ';' | sed 's/;$//')
    if [[ -z "$CUDA_ARCHS" ]]; then
        CUDA_ARCHS="89"
        echo -e " \033[1;33mCould not detect GPU arch. Defaulting to sm_89 (RTX 4090).\033[0m"
    fi
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ' | sed 's/, $//')
    echo " Compiling for ${GPU_NAMES} (sm_${CUDA_ARCHS}) with CUDA..."
    echo "   -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
    echo ""

    export CC=gcc
    export CXX=g++

    cd "$MTP_DIR"
    rm -rf build

    echo "--- CMAKE CONFIGURE ---" > "../$DEBUG_LOG"
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_HOST_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        2>&1 | tee -a "../$DEBUG_LOG"

    echo "--- CMAKE BUILD ---" >> "../$DEBUG_LOG"
    cmake --build build --config Release -j$(nproc) --target llama-cli llama-server llama-quantize 2>&1 | tee -a "../$DEBUG_LOG"
    BUILD_STATUS=${PIPESTATUS[0]}

    echo ""
    echo " Checking Python for GGUF conversion tools..."
    if command -v python3 > /dev/null 2>&1; then
        echo " Python3 found. Installing conversion dependencies..."
        pip install -q torch transformers sentencepiece gguf 2>/dev/null || \
        pip3 install -q torch transformers sentencepiece gguf 2>/dev/null || \
        echo " Warning: pip install failed. Convert step may need manual setup."
    else
        echo " Python3 not found. You'll need it for model conversion (step [1])."
    fi

    cd ..

    if [ $BUILD_STATUS -ne 0 ]; then
        echo -e "\n \033[1;31m[!] COMPILE FAILED.\033[0m"
        echo " Raw error logs: $DEBUG_LOG"
        read -p " Press Enter to return to menu..."
    else
        # Verify MTP support in the built binary
        local built_bin="./${MTP_DIR}/build/bin/llama-server"
        if [[ -x "$built_bin" ]]; then
            if $built_bin -h 2>&1 | grep -q 'spec-type.*mtp'; then
                echo -e "\n \033[1;32mBuild Complete! MTP support verified.\033[0m"
            else
                echo -e "\n \033[1;31m[!] Build succeeded but MTP support NOT found in binary!\033[0m"
                echo " The PR may not have been checked out correctly."
                echo " Try running Install/Update again."
            fi
        else
            echo -e "\n \033[1;32mBuild Complete!\033[0m"
        fi
        sleep 2
    fi
}

# -- Convert Model ---------------------------------------------------------

convert_model() {
    echo ""
    echo -e " \033[1;36m>>> CONVERT HF MODEL -> GGUF (with MTP layers) <<<\033[0m"
    echo ""
    echo -e " \033[1;33mImportant:\033[0m Existing GGUF files do NOT have MTP layers."
    echo " You must convert from the original HuggingFace model weights."
    echo " This requires ~54 GB free disk space for the HF download + ~54 GB for F16 GGUF."
    echo ""

    local convert_script="${MTP_DIR}/convert_hf_to_gguf.py"
    local quantize_bin="${MTP_DIR}/build/bin/llama-quantize"

    if [[ ! -f "$convert_script" ]]; then
        echo " Error: ${convert_script} not found."
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return..."
        return
    fi

    if ! command -v python3 > /dev/null 2>&1; then
        echo " Error: python3 is required for conversion."
        read -p " Press Enter to return..."
        return
    fi

    echo " HuggingFace model source:"
    echo "   [1] Download from HuggingFace (Qwen/Qwen3.6-27B, ~54 GB)"
    echo "   [2] Use local directory (specify path)"
    read -p " Choice (1-2): " source_choice
    source_choice=$(echo "$source_choice" | tr -d '[:space:]')

    local hf_model_dir=""

    case "$source_choice" in
        2)
            read -p " Path to local HF model directory: " hf_model_dir
            hf_model_dir=$(echo "$hf_model_dir" | tr -d '[:space:]')
            if [[ ! -d "$hf_model_dir" ]]; then
                echo " Directory not found: $hf_model_dir"
                read -p " Press Enter to return..."
                return
            fi
            ;;
        *)
            local hf_repo="Qwen/Qwen3.6-27B"
            hf_model_dir="${HF_CACHE}/Qwen3.6-27B"

            echo ""
            echo " Downloading ${hf_repo} to ${hf_model_dir}..."
            echo " This is a large model (~54 GB). It may take a while."
            echo ""

            if ! command -v huggingface-cli > /dev/null 2>&1; then
                echo " Installing huggingface-cli..."
                pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null
            fi

            huggingface-cli download "$hf_repo" --local-dir "$hf_model_dir"

            if [[ $? -ne 0 ]]; then
                echo " Download failed. Check your internet connection and disk space."
                read -p " Press Enter to return..."
                return
            fi
            ;;
    esac

    echo ""
    echo " Quantization level:"
    echo "   [1] Q5_K_M  (~18.5 GB -- recommended, 180K context on 4090)"
    echo "   [2] Q4_K_M  (~15.5 GB -- more VRAM headroom, longer context)"
    echo "   [3] Q6_K    (~21 GB -- higher quality, shorter context)"
    echo "   [4] Q8_0    (~27 GB -- too large for 24GB VRAM)"
    echo "   [5] F16     (~54 GB -- too large for 24GB VRAM, no quantization)"
    echo "   [6] Custom"
    read -p " Choice (1-6, default 1): " quant_choice
    quant_choice=$(echo "$quant_choice" | tr -d '[:space:]')

    local quant_type="Q5_K_M"
    local quant_tag="Q5_K_M"
    case "$quant_choice" in
        2) quant_type="Q4_K_M"; quant_tag="Q4_K_M" ;;
        3) quant_type="Q6_K"; quant_tag="Q6_K" ;;
        4) quant_type="Q8_0"; quant_tag="Q8_0" ;;
        5) quant_type="F16"; quant_tag="F16" ;;
        6)
            read -p " Enter quantization type (e.g. Q5_K_M, Q4_K_M, Q3_K_M): " quant_type
            quant_type=$(echo "$quant_type" | tr -d '[:space:]')
            quant_tag="$quant_type"
            ;;
        *) quant_type="Q5_K_M"; quant_tag="Q5_K_M" ;;
    esac

    local f16_out="${MODELS_DIR}/Qwen3.6-27B-f16-mtp.gguf"
    local final_out="${MODELS_DIR}/Qwen3.6-27B-${quant_tag}-mtp.gguf"

    echo ""
    echo " Step 1: Converting ${hf_model_dir} -> F16 GGUF with MTP layers..."
    echo " Output: ${f16_out}"
    echo ""
    echo -e " \033[1;33mNote:\033[0m The MTP PR's converter should auto-detect MTP layers in the model."
    echo " If conversion fails, make sure you're on the mtp-pr branch (run [0] to reinstall)."
    echo ""

    python3 "$convert_script" "$hf_model_dir" --outfile "$f16_out" --outtype f16 2>&1

    if [[ $? -ne 0 ]]; then
        echo ""
        echo -e " \033[1;31mConversion failed.\033[0m"
        echo " Common fixes:"
        echo "   - Install deps: pip install torch transformers sentencepiece gguf"
        echo "   - Ensure enough disk space (~54 GB for F16 GGUF)"
        read -p " Press Enter to return..."
        return
    fi

    if [[ "$quant_type" == "F16" ]]; then
        mv "$f16_out" "$final_out"
        echo ""
        echo -e " \033[1;32mDone! F16 MTP GGUF saved to: ${final_out}\033[0m"
    else
        echo ""
        echo " Step 2: Quantizing to ${quant_type}..."
        echo " Output: ${final_out}"
        echo ""

        if [[ ! -x "$quantize_bin" ]]; then
            echo " Error: llama-quantize not found at ${quantize_bin}"
            echo " Run Install/Update [0] first. F16 file kept at: ${f16_out}"
            read -p " Press Enter to return..."
            return
        fi

        "$quantize_bin" "$f16_out" "$final_out" "$quant_type" 2>&1

        if [[ $? -eq 0 ]]; then
            echo ""
            echo " Cleaning up intermediate F16 file..."
            rm -f "$f16_out"
            echo -e " \033[1;32mDone! MTP GGUF saved to: ${final_out}\033[0m"
        else
            echo -e " \033[1;31mQuantization failed.\033[0m"
            echo " F16 file kept at: ${f16_out}"
            read -p " Press Enter to return..."
            return
        fi
    fi

    local final_size
    final_size=$(du -h "$final_out" | cut -f1)
    echo ""
    echo " Model: ${final_out} (${final_size})"
    echo " This GGUF includes MTP layers for native speculative decoding."
    echo " No separate draft model needed!"
    sleep 3
}

# -- Download Chat Template ------------------------------------------------

download_template() {
    echo ""
    echo -e " \033[1;36m>>> DOWNLOAD FIXED CHAT TEMPLATE <<<\033[0m"
    echo ""
    echo " Downloads froggeric's fixed Qwen chat templates."
    echo " Fixes 7 jinja issues from vLLM-specific workarounds that break in llama.cpp."
    echo " Repo: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates"
    echo ""

    # Try to list available templates via HuggingFace API
    local api_url="https://huggingface.co/api/models/froggeric/Qwen-Fixed-Chat-Templates"
    local files_json
    files_json=$(curl -s "$api_url" 2>/dev/null)

    local template_files=()
    if [[ -n "$files_json" ]]; then
        while IFS= read -r f; do
            [[ -n "$f" ]] && template_files+=("$f")
        done < <(echo "$files_json" | jq -r '.siblings[].rfilename' 2>/dev/null | grep -iE '\.jinja|template|qwen' | sort)
    fi

    if [[ ${#template_files[@]} -gt 0 ]]; then
        echo " Available templates:"
        for i in "${!template_files[@]}"; do
            echo "   [$((i+1))] ${template_files[$i]}"
        done
        echo ""
        echo "   [0] Download ALL .jinja files"
        read -p " Select template NR (default: 0): " tmpl_choice
        tmpl_choice=$(echo "$tmpl_choice" | tr -d '[:space:]')
        [[ -z "$tmpl_choice" ]] && tmpl_choice="0"

        if [[ "$tmpl_choice" == "0" ]]; then
            # Download all jinja files
            local count=0
            mkdir -p "$TEMPLATES_DIR"
            for f in "${template_files[@]}"; do
                if [[ "$f" == *.jinja ]]; then
                    local url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/${f}"
                    # Flatten subdirs into unique filename: qwen3.6/chat_template.jinja -> qwen3.6-chat_template.jinja
                    local flat_name=$(echo "$f" | tr '/' '-')
                    local dest="${TEMPLATES_DIR}/${flat_name}"
                    echo " Downloading ${f}..."
                    if wget -q -O "$dest" "$url" 2>/dev/null; then
                        echo -e "   \033[1;32mOK: ${dest}\033[0m"
                        count=$((count+1))
                    else
                        rm -f "$dest"
                        echo -e "   \033[1;31mFAIL: ${f}\033[0m"
                    fi
                fi
            done
            echo ""
            echo " Downloaded ${count} template(s)."
        else
            local idx=$((tmpl_choice - 1))
            local selected="${template_files[$idx]}"
            if [[ -n "$selected" ]]; then
                mkdir -p "$TEMPLATES_DIR"
                local url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/${selected}"
                local flat_name=$(echo "$selected" | tr '/' '-')
                local dest="${TEMPLATES_DIR}/${flat_name}"
                echo " Downloading ${selected}..."
                if wget -O "$dest" "$url"; then
                    echo -e " \033[1;32mSaved to: ${dest}\033[0m"
                else
                    rm -f "$dest"
                    echo -e " \033[1;31mDownload failed.\033[0m"
                fi
            fi
        fi
    else
        echo " Could not list repo files via API. Trying direct download..."
        echo ""
        mkdir -p "$TEMPLATES_DIR"
        local tried_files=(
            "qwen3.6/chat_template.jinja"
            "qwen3.5/chat_template.jinja"
            "qwen3-chat.jinja"
            "qwen3.jinja"
            "chat_template.jinja"
        )
        local found=0
        for f in "${tried_files[@]}"; do
            local url="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/${f}"
            local flat_name=$(echo "$f" | tr '/' '-')
            local dest="${TEMPLATES_DIR}/${flat_name}"
            echo " Trying: ${f}..."
            if wget -q -O "$dest" "$url" 2>/dev/null; then
                echo -e "   \033[1;32mOK: ${dest}\033[0m"
                found=$((found+1))
            else
                rm -f "$dest"
                echo "   Not found at this path."
            fi
        done

        if [[ $found -eq 0 ]]; then
            echo ""
            echo -e " \033[1;33mCould not auto-download templates.\033[0m"
            echo " Please visit https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates"
            echo " and manually download the .jinja file(s) to: ${TEMPLATES_DIR}/"
        else
            echo ""
            echo " Downloaded ${found} template(s)."
        fi
    fi

    echo ""
    echo " Templates in ${TEMPLATES_DIR}/:"
    ls -la "${TEMPLATES_DIR}/"*.jinja 2>/dev/null || echo "  (no .jinja files found)"
    echo ""
    read -p " Press Enter to return..."
}

# -- Start MTP Server ------------------------------------------------------

start_mtp_server() {
    echo ""
    echo -e " \033[1;36m>>> START MTP SERVER <<<\033[0m"
    echo ""
    echo -e " \033[1;33mNote:\033[0m MTP speculative decoding uses the model's built-in MTP tensor layers."
    echo " The GGUF must be converted with this PR's converter (run [1] to convert)."
    echo " No separate draft model needed."

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo ""
        echo -e " \033[1;31mServer is already running! Stop it first [4].\033[0m"
        sleep 2
        return
    fi

    local server_bin="./${MTP_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo " Error: llama-server not found at $server_bin"
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return..."
        return
    fi

    if ! $server_bin -h 2>&1 | grep -q 'spec-type.*mtp'; then
        echo -e " \033[1;31mError: Binary does not support MTP. Re-run Install/Update [0].\033[0m"
        read -p " Press Enter to return..."
        return
    fi

    # -- Model selection (MTP models only) --
    raw_data=()
    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            [[ "$name" == *"mmproj"* ]] && continue
            name_low=$(echo "$name" | tr '[:upper:]' '[:lower:]')
            [[ "$name_low" == *"-mtp"* ]] || continue
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo ""
        echo -e " \033[1;31mNo MTP-enabled GGUF models found!\033[0m"
        echo " MTP models must be converted with the PR #22673 converter."
        echo " Regular GGUF files do not have MTP layers."
        echo ""
        echo " Run [1] to convert Qwen3.6-27B from HuggingFace (preserves MTP layers)"
        echo " Or [5] to download a pre-converted MTP GGUF."
        read -p " Press Enter to return..."
        return
    fi

    echo ""
    echo " MTP-enabled models:"
    printf "   %-3s %-64s %-7s\n" "NR" "MODEL NAME" "SIZE"
    echo "   ----------------------------------------------------------------------"
    for i in "${!raw_data[@]}"; do
        IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
        printf "   %2d) %-64s [%-5s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
    done

    echo ""
    read -p " Select Model NR: " n
    n=$(echo "$n" | tr -d '[:space:]')
    local idx=$(( n - 1 ))
    local entry=${raw_data[$idx]}
    if [[ -z "$entry" ]]; then
        echo " Invalid model number."
        sleep 2
        return
    fi

    target="${entry%%|*}"
    target_low=$(echo "$target" | tr '[:upper:]' '[:lower:]')

    if [[ "$target_low" != *"-mtp"* ]]; then
        echo ""
        echo -e " \033[1;33mWarning:\033[0m This GGUF was likely converted WITHOUT MTP support."
        echo " MTP speculative decoding requires a GGUF converted with the MTP PR's converter."
        echo " Use [1] to convert the model first."
        echo ""
        read -p " Continue anyway? (y/N): " mtp_warn
        mtp_warn=$(echo "$mtp_warn" | tr -d '[:space:]')
        if [[ "$mtp_warn" != "y" && "$mtp_warn" != "Y" ]]; then
            echo " Canceled."
            sleep 1
            return
        fi
    fi

    # -- Context / KV --
    echo ""
    echo " Select Context Size / KV Cache:"
    echo "   [1] 184320  (180K, q4_0 KV -- recommended for Q5_K_M on 4090)"
    echo "   [2] 131072  (128K, q5_0 KV -- higher quality, shorter context)"
    echo "   [3] 131072  (128K, q4_0 KV -- more VRAM headroom)"
    echo "   [4] 65536   (64K, q8_0 KV -- short context, highest quality KV)"
    echo "   [5] 262144  (256K, q4_0 KV -- max context, may OOM)"
    echo "   [6] Custom"
    read -p " Choice (1-6, default 1): " ctx_choice
    ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

    case "$ctx_choice" in
        1) ctx="184320"; cache_type="q4_0" ;;
        2) ctx="131072"; cache_type="q5_0" ;;
        3) ctx="131072"; cache_type="q4_0" ;;
        4) ctx="65536";  cache_type="q8_0" ;;
        5) ctx="262144"; cache_type="q4_0" ;;
        6)
            read -p " Enter context size: " ctx
            ctx=$(echo "$ctx" | tr -d '[:space:]')
            read -p " Enter KV cache type (q4_0/q5_0/q5_1/q8_0/f16) [q4_0]: " cache_type
            cache_type=$(echo "$cache_type" | tr -d '[:space:]')
            [[ -z "$cache_type" ]] && cache_type="q4_0"
            ;;
        *) ctx="184320"; cache_type="q4_0" ;;
    esac

    if [[ ! "$ctx" =~ ^[0-9]+$ ]]; then
        echo " Invalid context size."
        sleep 2
        return
    fi

    # -- MTP speculative tokens --
    echo ""
    echo " MTP speculative tokens (how many tokens the model predicts ahead):"
    echo "   [1] 5  (recommended -- best speed/quality tradeoff)"
    echo "   [2] 3  (conservative -- higher acceptance rate)"
    echo "   [3] 1  (minimal -- single token lookahead)"
    echo "   [4] 8  (aggressive -- may waste compute on rejections)"
    echo "   [5] Custom"
    read -p " Choice (1-5, default 1): " mtp_choice
    mtp_choice=$(echo "$mtp_choice" | tr -d '[:space:]')

    case "$mtp_choice" in
        2) mtp_tokens="3" ;;
        3) mtp_tokens="1" ;;
        4) mtp_tokens="8" ;;
        5)
            read -p " Enter number of speculative tokens: " mtp_tokens
            mtp_tokens=$(echo "$mtp_tokens" | tr -d '[:space:]')
            ;;
        *) mtp_tokens="5" ;;
    esac

    # -- Chat template --
    echo ""
    echo " Chat template:"
    echo "   [1] Fixed template (froggeric -- recommended, fixes 7 jinja bugs)"
    echo "   [2] Built-in model template"
    echo "   [3] Custom template file path"
    read -p " Choice (1-3, default 1): " tmpl_choice
    tmpl_choice=$(echo "$tmpl_choice" | tr -d '[:space:]')

    template_flags=()
    template_text="Built-in"

    case "$tmpl_choice" in
        1)
            # Find available fixed templates
            local jinja_files=()
            if [[ -d "$TEMPLATES_DIR" ]]; then
                mapfile -t jinja_files < <(find "$TEMPLATES_DIR" -name '*.jinja' -type f 2>/dev/null | sort)
            fi

            if [[ ${#jinja_files[@]} -eq 0 ]]; then
                echo ""
                echo -e " \033[1;33mNo fixed templates downloaded yet. Run [2] to download first.\033[0m"
                echo " Falling back to built-in template."
                template_text="Built-in (no fixed template downloaded)"
            elif [[ ${#jinja_files[@]} -eq 1 ]]; then
                if arg_probe_valid "$server_bin" --chat-template-file "${jinja_files[0]}"; then
                    template_flags=(--chat-template-file "${jinja_files[0]}")
                    template_text="Fixed: $(basename "${jinja_files[0]}")"
                else
                    echo -e " \033[1;33m--chat-template-file not accepted by this binary. Using built-in.\033[0m"
                    template_text="Built-in (--chat-template-file not supported)"
                fi
            else
                echo ""
                echo " Available fixed templates:"
                for i in "${!jinja_files[@]}"; do
                    echo "   [$((i+1))] $(basename "${jinja_files[$i]}")"
                done
                read -p " Select template NR: " jt_choice
                jt_choice=$(echo "$jt_choice" | tr -d '[:space:]')
                local jt_idx=$((jt_choice - 1))
                local selected_tmpl="${jinja_files[$jt_idx]:-${jinja_files[0]}}"

                if arg_probe_valid "$server_bin" --chat-template-file "$selected_tmpl"; then
                    template_flags=(--chat-template-file "$selected_tmpl")
                    template_text="Fixed: $(basename "$selected_tmpl")"
                else
                    echo -e " \033[1;33m--chat-template-file not accepted. Using built-in.\033[0m"
                    template_text="Built-in (flag not supported)"
                fi
            fi
            ;;
        2)
            template_text="Built-in"
            ;;
        3)
            read -p " Path to template file: " custom_tmpl
            custom_tmpl=$(echo "$custom_tmpl" | tr -d '[:space:]')
            if [[ -f "$custom_tmpl" ]]; then
                if arg_probe_valid "$server_bin" --chat-template-file "$custom_tmpl"; then
                    template_flags=(--chat-template-file "$custom_tmpl")
                    template_text="Custom: $(basename "$custom_tmpl")"
                else
                    echo -e " \033[1;33m--chat-template-file not accepted. Using built-in.\033[0m"
                    template_text="Built-in (flag not supported)"
                fi
            else
                echo " File not found: $custom_tmpl"
                template_text="Built-in (file not found)"
            fi
            ;;
        *)
            template_text="Built-in"
            ;;
    esac

    # -- Thinking mode --
    echo ""
    echo " Thinking mode:"
    echo "   [1] Thinking OFF (recommended for speed)"
    echo "   [2] Thinking ON (model shows reasoning)"
    echo "   [3] Client decides (default)"
    read -p " Choice (1-3, default 1): " thinking_choice
    thinking_choice=$(echo "$thinking_choice" | tr -d '[:space:]')
    [[ -z "$thinking_choice" ]] && thinking_choice="1"

    jinja_flags=()
    thinking_text="ThinkOff"
    think_on_json='{"enable_thinking":true}'
    think_off_json='{"enable_thinking":false}'

    # Always try --jinja for proper template handling
    if arg_probe_valid "$server_bin" --jinja; then
        jinja_flags+=(--jinja)
    fi

    case "$thinking_choice" in
        2)
            thinking_text="ThinkOn"
            if arg_probe_valid "$server_bin" --chat-template-kwargs "$think_on_json"; then
                jinja_flags+=(--chat-template-kwargs "$think_on_json")
            fi
            ;;
        3)
            thinking_text="ClientDefault"
            ;;
        *)
            thinking_text="ThinkOff"
            if arg_probe_valid "$server_bin" --chat-template-kwargs "$think_off_json"; then
                jinja_flags+=(--chat-template-kwargs "$think_off_json")
            elif arg_probe_valid "$server_bin" --reasoning off; then
                jinja_flags+=(--reasoning off)
            fi
            ;;
    esac

    # -- Build flags via probing --
    flag_summary=()
    skipped_summary=()

    fa_flags=()
    cache_flags=()
    mtp_flags=()
    batch_flags=()
    parallel_flags=()

    # Flash attention
    if arg_probe_valid "$server_bin" -fa on; then
        fa_flags=(-fa on)
        flag_summary+=("flash:-fa on")
    elif arg_probe_valid "$server_bin" -fa; then
        fa_flags=(-fa)
        flag_summary+=("flash:-fa")
    else
        skipped_summary+=("flash attention not accepted")
    fi

    # KV cache
    if arg_probe_valid "$server_bin" -ctk "$cache_type" -ctv "$cache_type"; then
        cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
        flag_summary+=("kv:-ctk/-ctv ${cache_type}")
    else
        skipped_summary+=("KV cache flags not accepted")
    fi

    # MTP / speculative decoding flags
    # PR #22673 uses --spec-type mtp with --spec-draft-n-max N
    mtp_flag_found=false
    if arg_probe_valid "$server_bin" --spec-type mtp; then
        mtp_flags=(--spec-type mtp --spec-draft-n-max "$mtp_tokens")
        flag_summary+=("mtp:--spec-type mtp --spec-draft-n-max ${mtp_tokens}")
        mtp_flag_found=true
    elif arg_probe_valid "$server_bin" --mtp "$mtp_tokens"; then
        mtp_flags=(--mtp "$mtp_tokens")
        flag_summary+=("mtp:--mtp ${mtp_tokens}")
        mtp_flag_found=true
    fi

    if ! $mtp_flag_found; then
        skipped_summary+=("MTP flag not found via probe")
        echo ""
        echo -e " \033[1;33mNote:\033[0m Could not probe an MTP flag."
    fi

    # -- Build command --
    # Matches the Reddit PR author's recommended launch:
    #   llama-server -m <model> --spec-type mtp --spec-draft-n-max 5
    #     --cache-type-k q4_0 --cache-type-v q4_0
    #     -np 1 -c 262144 --temp 0.7 --top-k 20 -ngl 99 --port 8081
    #
    # KEY: -ngl 99 (NOT 999) -- with 999, the MTP head loads as a second
    # model with 66 layers AGAIN, overflowing VRAM and spilling to CPU.
    # With 99, auto-fitting works correctly for both main + MTP head.
    # No -fa, no -b/-ub, no --parallel, no -t -- keep it simple like the PR.
    cmd=("$server_bin"
        -m "${MODELS_DIR}/${target}"
        --spec-type mtp
        --spec-draft-n-max "$mtp_tokens"
        --cache-type-k "$cache_type"
        --cache-type-v "$cache_type"
        -np 1
        -c "$ctx"
        --temp 0.7
        --top-k 20
        -ngl 99
        "${template_flags[@]}"
        --host 0.0.0.0
        --port 8080
    )

    # Add chat template kwargs if present
    if [[ ${#jinja_flags[@]} -gt 0 ]]; then
        cmd+=("${jinja_flags[@]}")
    fi

    local target_short="$target"
    echo "MTP: ${target_short} [${ctx}/${cache_type}/mtp=${mtp_tokens}/${thinking_text}]" > .server_info_mtp

    echo ""
    echo " Starting MTP server (matching Reddit PR author's recommended flags):"
    echo "   Model:      $target"
    echo "   Context:    $ctx"
    echo "   KV cache:   $cache_type"
    echo "   MTP tokens: ${mtp_tokens} (speculative)"
    echo "   Template:   $template_text"
    echo "   Thinking:   $thinking_text"
    echo "   Vision:     NO (text-only for max context)"
    echo "   -ngl 99 (NOT 999 -- prevents MTP head double-loading)"
    echo "   --temp 0.7 --top-k 20"
    echo "   -np 1 (single sequence)"
    echo ""
    echo " Accepted flags:"
    for x in "${flag_summary[@]}"; do echo "   + $x"; done

    if [[ ${#skipped_summary[@]} -gt 0 ]]; then
        echo ""
        echo " Skipped / notes:"
        for x in "${skipped_summary[@]}"; do echo "   - $x"; done
    fi

    echo ""
    echo " Command:"
    printf ' %q' "${cmd[@]}"
    echo ""
    echo ""

    {
        echo "COMMAND PROFILE: llama-mtp-speculative"
        printf '%q ' "${cmd[@]}"
        echo ""
        echo "----- llama-server output -----"
    } > "$SERVER_LOG"

    nohup "${cmd[@]}" >> "$SERVER_LOG" 2>&1 &
    server_pid=$!

    failed=0
    loaded=0

    for i in $(seq 1 120); do
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
        rm -f .server_info_mtp
        echo ""
        echo -e " \033[1;31mServer failed during startup.\033[0m"
        echo " Last 220 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 220 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        read -p " Press Enter to return to menu..."
    fi
}

# -- Quick Start MTP (Reddit PR author's exact command) ---------------

quick_start_mtp() {
    echo ""
    echo -e " \033[1;36m>>> QUICK START MTP (Reddit PR #22673 author's params) <<<\033[0m"
    echo ""
    echo " Uses the exact command from the PR author:"
    echo "   llama-server -m <model> --spec-type mtp --spec-draft-n-max 5"
    echo "     --cache-type-k q4_0 --cache-type-v q4_0"
    echo "     -np 1 -c 262144 --temp 0.7 --top-k 20 -ngl 99"
    echo ""
    echo " You only pick the model. Everything else is fixed."

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo ""
        echo -e " \033[1;31mServer is already running! Stop it first [4].\033[0m"
        sleep 2
        return
    fi

    local server_bin="./${MTP_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo " Error: llama-server not found at $server_bin"
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return..."
        return
    fi

    if ! $server_bin -h 2>&1 | grep -q 'spec-type.*mtp'; then
        echo -e " \033[1;31mError: Binary does not support MTP. Re-run Install/Update [0].\033[0m"
        read -p " Press Enter to return..."
        return
    fi

    # List MTP models only
    raw_data=()
    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            [[ "$name" == *"mmproj"* ]] && continue
            name_low=$(echo "$name" | tr '[:upper:]' '[:lower:]')
            [[ "$name_low" == *"-mtp"* ]] || continue
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo ""
        echo -e " \033[1;31mNo MTP-enabled GGUF models found!\033[0m"
        echo " Run [1] to convert one from HuggingFace, or [5] to download."
        read -p " Press Enter to return..."
        return
    fi

    echo ""
    printf "   %-3s %-64s %-7s\n" "NR" "MODEL NAME" "SIZE"
    echo "   ----------------------------------------------------------------------"
    for i in "${!raw_data[@]}"; do
        IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
        printf "   %2d) %-64s [%-5s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
    done

    echo ""
    read -p " Select Model NR: " n
    n=$(echo "$n" | tr -d '[:space:]')
    local idx=$(( n - 1 ))
    local entry=${raw_data[$idx]}
    if [[ -z "$entry" ]]; then
        echo " Invalid model number."
        sleep 2
        return
    fi

    local target="${entry%%|*}"

    echo ""
    echo " Context size:"
    echo "   [1] 262144 (256K -- Reddit poster's default)"
    echo "   [2] 131072 (128K)"
    echo "   [3]  65536 (64K)"
    read -p " Choice (1-3, default 1): " ctx_choice
    ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')
    case "$ctx_choice" in
        2) local ctx=131072 ;;
        3) local ctx=65536 ;;
        *) local ctx=262144 ;;
    esac

    local cache_type="q4_0"
    echo ""
    echo " KV cache type:"
    echo "   [1] q4_0  (best context, Reddit poster's default)"
    echo "   [2] q5_0  (better quality, less context)"
    echo "   [3] q8_0  (highest quality, shortest context)"
    echo "   [4] q5_1  (alternative quality)"
    read -p " Choice (1-4, default 1): " kv_choice
    kv_choice=$(echo "$kv_choice" | tr -d '[:space:]')
    case "$kv_choice" in
        2) cache_type="q5_0" ;;
        3) cache_type="q8_0" ;;
        4) cache_type="q5_1" ;;
        *) cache_type="q4_0" ;;
    esac

    # The exact command from the Reddit PR author
    local cmd=("$server_bin"
        -m "${MODELS_DIR}/${target}"
        --spec-type mtp
        --spec-draft-n-max 5
        --cache-type-k "$cache_type"
        --cache-type-v "$cache_type"
        -np 1
        -c "$ctx"
        --temp 0.7
        --top-k 20
        -ngl 99
        --host 0.0.0.0
        --port 8080
    )

    echo "MTP: ${target} [${ctx}/${cache_type}/mtp=5/quickstart]" > .server_info_mtp

    echo ""
    echo " Launching with Reddit PR author's exact parameters:"
    echo "   Model:    $target"
    echo "   Command:  llama-server -m <model> --spec-type mtp --spec-draft-n-max 5"
    echo "             --cache-type-k ${cache_type} --cache-type-v ${cache_type}"
    echo "             -np 1 -c ${ctx} --temp 0.7 --top-k 20 -ngl 99"
    echo ""
    echo " Full command:"
    printf ' %q' "${cmd[@]}"
    echo ""
    echo ""

    {
        echo "COMMAND PROFILE: quick-start-mtp-reddit"
        printf '%q ' "${cmd[@]}"
        echo ""
        echo "----- llama-server output -----"
    } > "$SERVER_LOG"

    nohup "${cmd[@]}" >> "$SERVER_LOG" 2>&1 &
    local server_pid=$!

    local failed=0
    local loaded=0

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

        # Check for OOM or allocation failure
        if grep -Eqi 'out of memory|failed to allocate|CUDA error' "$SERVER_LOG"; then
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
        echo " Last 50 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 50 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        sleep 3
    else
        rm -f .server_info_mtp
        echo ""
        echo -e " \033[1;31mServer failed during startup.\033[0m"
        echo " Last 250 lines of ${SERVER_LOG}:"
        echo "------------------------------------------------------------"
        tail -n 250 "$SERVER_LOG"
        echo "------------------------------------------------------------"
        read -p " Press Enter to return to menu..."
    fi
}

# =========================================================================
# QUICKSTART MODE — one-shot: build, download model, start server
# =========================================================================
if [[ "${1:-}" == "--quickstart" ]]; then
    echo ""
    echo -e " ${BOLD}${CYAN}=============================================${RESET}"
    echo -e " ${BOLD}${CYAN} HostLLM — Quick Start (one-click MTP)${RESET}"
    echo -e " ${BOLD}${CYAN}=============================================${RESET}"
    echo ""

    # -- Step 1: Build binary if missing --------------------------------
    server_bin="./${MTP_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo -e " ${YELLOW}[1/3]${RESET} Binary not found. Building llama.cpp (MTP)..."
        echo ""
        install_mtp
        if [[ ! -x "$server_bin" ]]; then
            echo -e " ${RED}Build failed. Cannot continue.${RESET}"
            read -p " Press Enter to exit..."
            exit 1
        fi
    fi

    # Verify MTP support before proceeding
    if ! $server_bin -h 2>&1 | grep -q 'spec-type.*mtp'; then
        echo -e " ${RED}Binary does not support MTP speculative decoding.${RESET}"
        echo -e " The PR may not have been checked out. Re-run install from the menu."
        read -p " Press Enter to exit..."
        exit 1
    fi

    echo -e " ${GREEN}[1/3]${RESET} Binary ready (MTP verified)."

    # -- Step 2: Download model if missing ------------------------------
    model_path="${MODELS_DIR}/${QUICKSTART_MODEL}"
    if [[ ! -f "$model_path" ]]; then
        echo ""
        echo -e " ${YELLOW}[2/3]${RESET} Downloading default model..."
        echo "   ${QUICKSTART_MODEL}"
        echo "   This is ~16 GB and may take a while."
        echo ""
        wget --show-progress -O "$model_path" "$QUICKSTART_MODEL_URL"
        if [[ $? -ne 0 ]]; then
            echo -e " ${RED}Download failed. Check internet connection.${RESET}"
            rm -f "$model_path"
            read -p " Press Enter to exit..."
            exit 1
        fi
    else
        echo -e " ${GREEN}[2/3]${RESET} Model ready."
    fi

    # -- Step 3: Detect GPUs, calculate context -------------------------
    echo ""
    echo -e " ${YELLOW}[3/3]${RESET} Detecting hardware..."

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

    # GPU names for display
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' '|' | sed 's/|$//')

    echo "   GPUs:        ${GPU_NAMES}"
    echo "   Total VRAM:  ${TOTAL_VRAM_GB} GB (${GPU_COUNT} GPU(s))"

    # Minimum VRAM check — model is ~16 GB
    if [[ "$TOTAL_VRAM_GB" -lt 16 ]]; then
        echo ""
        echo -e " ${RED}Not enough VRAM. Need at least 16 GB total, found ${TOTAL_VRAM_GB} GB.${RESET}"
        echo " Consider a smaller model or adding more GPUs."
        read -p " Press Enter to exit..."
        exit 1
    fi

    # Model ~16 GB, reserve 1 GB overhead per GPU
    AVAIL_GB=$((TOTAL_VRAM_GB - 16 - GPU_COUNT))

    if [[ "$AVAIL_GB" -lt 1 ]]; then
        echo -e " ${YELLOW}Warning: Very tight VRAM. Context will be minimal (8K).${RESET}"
        AVAIL_GB=0
    fi

    # ~25K context per available GB at q4_0 KV
    CTX=$((AVAIL_GB * 25000))
    [[ $CTX -lt 8192 ]]   && CTX=8192
    [[ $CTX -gt 262144 ]] && CTX=262144

    echo "   Model size:  ~16 GB (Q4_K_S)"
    echo "   KV cache:    q4_0"
    echo "   Context:     ${CTX}"
    echo "   MTP tokens:  5"
    echo ""
    echo -e " ${GREEN}Starting server on port 8080...${RESET}"
    echo ""

    echo "MTP-QuickStart: ${QUICKSTART_MODEL} [${CTX}/q4_0/mtp=5/GPUs=${GPU_COUNT}]" > .server_info_mtp

    # -- Launch server in background ------------------------------------
    {
        echo "COMMAND PROFILE: quickstart-mtp"
        echo ""
        echo "----- llama-server output -----"
    } > "$SERVER_LOG"

    nohup "$server_bin" \
        -m "$model_path" \
        --spec-type mtp --spec-draft-n-max 5 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        -np 1 -c "$CTX" \
        --temp 0.7 --top-k 20 \
        -ngl 99 \
        --host 0.0.0.0 --port 8080 \
        >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    # -- Wait for health check ------------------------------------------
    FAILED=0
    LOADED=0

    echo " Waiting for server to load model..."
    for i in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
            FAILED=1
            break
        fi

        if grep -Eqi 'unknown argument|unrecognized option|invalid option|error:.*argument|usage:' "$SERVER_LOG" 2>/dev/null; then
            kill "$SERVER_PID" >/dev/null 2>&1 || true
            FAILED=1
            break
        fi

        if grep -Eqi 'out of memory|failed to allocate|CUDA error' "$SERVER_LOG" 2>/dev/null; then
            kill "$SERVER_PID" >/dev/null 2>&1 || true
            FAILED=1
            break
        fi

        CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health" 2>/dev/null || true)
        if [[ "$CODE" == "200" ]]; then
            LOADED=1
            break
        fi

        sleep 1
    done

    if [[ "$FAILED" -eq 1 ]] || ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        rm -f .server_info_mtp
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

    # -- Detect local IP for clickable URL ------------------------------
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -z "$LOCAL_IP" ]] && LOCAL_IP="localhost"

    # -- Show running dashboard -----------------------------------------
    clear
    echo "=================================================================="
    echo -e "  ${GREEN}${BOLD}MTP SERVER RUNNING${RESET}"
    echo "=================================================================="
    echo ""
    echo -e "  ${BOLD}Chat URL:${RESET}  http://${LOCAL_IP}:8080"
    echo ""
    echo "  Model:   ${QUICKSTART_MODEL}"
    echo "  Context: ${CTX}  |  KV: q4_0  |  MTP: 5"
    echo "  GPUs:    ${GPU_COUNT}x (${TOTAL_VRAM_GB} GB total)"
    echo ""
    echo "  Press Ctrl+C to stop server"
    echo "=================================================================="
    echo ""

    # Live stats loop
    tput civis
    while kill -0 "$SERVER_PID" >/dev/null 2>&1; do
        if command -v nvidia-smi > /dev/null 2>&1; then
            stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
            IFS=',' read -r gpu_load vram_used vram_total gpu_temp <<< "$stats"
            gpu_load=$(echo "$gpu_load" | tr -d ' ')
            vram_used=$(echo "$vram_used" | tr -d ' ')
            vram_total=$(echo "$vram_total" | tr -d ' ')
            gpu_temp=$(echo "$gpu_temp" | tr -d ' ')
            if [[ "$vram_total" -gt 0 ]]; then vram_pct=$(( (vram_used * 100) / vram_total )); else vram_pct=0; fi
            vram_used_gb=$(awk "BEGIN {printf \"%.1f\", $vram_used/1024}")
            vram_total_gb=$(awk "BEGIN {printf \"%.0f\", $vram_total/1024}")
        else
            gpu_load="N/A"; gpu_temp="-"; vram_used_gb="0"; vram_total_gb="0"; vram_pct="0"
        fi

        read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
        cpu_ap=$((user+nice+system+irq+softirq+steal))
        cpu_tp=$((user+nice+system+idle+iowait+irq+softirq+steal))
        sleep 1
        read cpu user nice system idle iowait irq softirq steal guest < /proc/stat
        cpu_ac=$((user+nice+system+irq+softirq+steal))
        cpu_tc=$((user+nice+system+idle+iowait+irq+softirq+steal))
        cpu_diff=$((cpu_tc - cpu_tp))
        cpu_adiff=$((cpu_ac - cpu_ap))
        if [[ "$cpu_diff" -gt 0 ]]; then cpu_pct=$(( (cpu_adiff * 100) / cpu_diff )); else cpu_pct=0; fi

        tput sc
        tput cup 11 0
        echo -e "  CPU: ${cpu_pct}%   |   GPU: ${gpu_load}%   |   Temp: ${gpu_temp} degC"
        echo -e "  VRAM: ${vram_used_gb} GB / ${vram_total_gb} GB (${vram_pct}%)"
        echo ""
        echo -e "  ${BOLD}Chat URL:${RESET}  http://${LOCAL_IP}:8080"
        tput rc
        tput cup 17 0
    done

    tput cnorm
    echo ""
    echo -e " ${RED}Server stopped.${RESET}"
    rm -f .server_info_mtp 2>/dev/null
    read -p " Press Enter to exit..."
    exit 0
fi

# -- Main Menu -------------------------------------------------------------

setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""

    # List models
    raw_data=()
    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            [[ "$name" == *"mmproj"* ]] && continue
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo "   (No .gguf models found in ./$MODELS_DIR/)"
    else
        printf "   %-3s %-64s %-7s %s\n" "NR" "MODEL NAME" "SIZE" "MTP?"
        echo "   ----------------------------------------------------------------------"
        for i in "${!raw_data[@]}"; do
            IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
            m_low=$(echo "$m_name" | tr '[:upper:]' '[:lower:]')
            if [[ "$m_low" == *"-mtp"* ]]; then
                printf "   %2d) %-64s [%-5s] \033[1;32mMTP\033[0m\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
            else
                printf "   %2d) %-64s [%-5s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
            fi
        done
    fi

    # Build status
    server_bin="./${MTP_DIR}/build/bin/llama-server"
    if [[ -x "$server_bin" ]]; then
        echo ""
        echo -e "   Binary: \033[1;32m${server_bin}\033[0m"
    else
        echo ""
        echo -e "   Binary: \033[1;31m${server_bin} (NOT BUILT -- run [0] first)\033[0m"
    fi

    # Template status
    local_tmpl_count=0
    if [[ -d "$TEMPLATES_DIR" ]]; then
        local_tmpl_count=$(find "$TEMPLATES_DIR" -name '*.jinja' -type f 2>/dev/null | wc -l)
    fi
    if [[ "$local_tmpl_count" -gt 0 ]]; then
        echo -e "   Templates: \033[1;32m${local_tmpl_count} fixed template(s) downloaded\033[0m"
    else
        echo -e "   Templates: \033[1;33mNo fixed templates (run [2] to download)\033[0m"
    fi

    echo ""
    echo -e " \033[1;36m--- MTP SERVER ---\033[0m"
    echo " [3] Start MTP Server (configure context, KV, MTP tokens)"
    echo " [7] Quick Start MTP   (Reddit PR params -- just pick model)"
    echo " [4] Stop Server"
    echo ""
    echo -e " \033[1;36m--- SETUP ---\033[0m"
    echo " [0] Install / Update llama.cpp (MTP PR #22673)"
    echo " [1] Convert HF Model -> GGUF (preserve MTP layers + quantize)"
    echo " [2] Download Fixed Chat Template (froggeric)"
    echo ""
    echo -e " \033[1;36m--- MANAGEMENT ---\033[0m"
    echo " [5] Download Model (.gguf URL)"
    echo " [6] Delete Model"
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
            install_mtp
            setup_scroll_region
            monitor_loop &
            MONITOR_PID=$!
            ;;
        1)
            convert_model
            ;;
        2)
            download_template
            ;;
        3)
            start_mtp_server
            ;;
        7)
            quick_start_mtp
            ;;
        4)
            echo ""
            echo -e " \033[1;36m>>> STOPPING SERVER <<<\033[0m"
            pkill -f "llama-server"
            rm -f .server_info_mtp
            echo " Server stopped."
            sleep 1
            ;;
        5)
            echo ""
            echo -e " \033[1;36m>>> DOWNLOAD MODEL <<<\033[0m"
            echo " Paste the direct download URL for a .gguf file."
            echo ""
            echo " Tip: Pre-converted MTP GGUF files should have '-mtp' in the name."
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
            echo -e " \033[1;36m>>> DELETE MODEL <<<\033[0m"

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
                rm "${MODELS_DIR}/${del_target}"
                echo " Deleted $del_target"
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
