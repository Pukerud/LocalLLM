#!/bin/bash

# =========================================================================
# DFlash llama.cpp Dashboard v1.0
# Uses spiritbuun/buun-llama-cpp fork for DFlash speculative decoding.
# https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF
# =========================================================================

set +m
MODELS_DIR="llama_models"
DFLASH_DIR="buun-llama-cpp"
DEBUG_LOG="dflash_compile_debug.log"

declare -A speed_cache
mkdir -p "$MODELS_DIR"

for cmd in curl jq wget git cmake gcc g++; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
        echo "Missing dependency: $cmd. Installing..."
        sudo apt update && sudo apt install -y "$cmd" build-essential
    fi
done

if ! command -v /usr/local/cuda/bin/nvcc > /dev/null 2>&1; then
    echo "Warning: Modern CUDA toolkit (12+) not found at /usr/local/cuda/bin/nvcc."
    sleep 2
fi

cleanup() {
    kill -9 "$MONITOR_PID" > /dev/null 2>&1
    wait "$MONITOR_PID" > /dev/null 2>&1
    tput csr 0 "$(tput lines)"
    tput cnorm
    echo ""
    exit
}
trap cleanup INT TERM EXIT

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
        if [[ -f ".server_info_dflash" ]]; then
            ACTIVE_INFO=$(cat .server_info_dflash)
            SERVER_STATUS="\033[1;32mRUNNING: ${ACTIVE_INFO}\033[0m"
        else
            SERVER_STATUS="\033[1;32mRUNNING (PID: $SERVER_PID)\033[0m"
        fi
    else
        SERVER_STATUS="\033[1;31mSTOPPED\033[0m"
        rm -f .server_info_dflash 2>/dev/null
    fi

    reset="\033[0m"; bold="\033[1m"
    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${bold}buun-llama-cpp (DFlash)${reset}    |  SERVER: ${SERVER_STATUS}\033[K"
    tput cup 3 0
    echo -e "   CPU: ${c_cpu}${cpu_pct}%${reset}   |   GPU: ${gpu_load}%   |   Temp: ${gpu_temp}°C\033[K"
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
    echo "   DFlash llama.cpp DASHBOARD v1.0  —  spiritbuun/buun-llama-cpp"
    echo "==========================================================================================================================="
    tput cup 5 0
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

# ── Model helpers ────────────────────────────────────────────────────────

get_dflash_drafts() {
    # Find DFlash draft models (contain "dflash" or "draft" in name)
    mapfile -t dflash_drafts < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*dflash*.gguf' -printf '%f\n' 2>/dev/null | sort)
    # Also look for generic draft models if no dflash-specific ones
    if [[ ${#dflash_drafts[@]} -eq 0 ]]; then
        mapfile -t dflash_drafts < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*draft*.gguf' -printf '%f\n' 2>/dev/null | sort)
    fi
}

get_mmproj_files() {
    mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*mmproj*.gguf' 2>/dev/null | sort)
}

# ── Install / Update ─────────────────────────────────────────────────────

install_dflash() {
    echo ""
    echo -e " \033[1;36m>>> INSTALL / UPDATE buun-llama-cpp (DFlash fork) <<<\033[0m"
    echo ""

    if [[ ! -d "$DFLASH_DIR" ]]; then
        echo " Cloning spiritbuun/buun-llama-cpp..."
        git clone https://github.com/spiritbuun/buun-llama-cpp.git "$DFLASH_DIR"
    else
        echo " Pulling latest buun-llama-cpp..."
        cd "$DFLASH_DIR" && git pull && cd ..
    fi

    echo ""
    echo " Compiling for RTX 4090 (sm_89) with DFlash flags..."
    echo "   -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON"
    echo ""

    export CC=gcc
    export CXX=g++

    cd "$DFLASH_DIR"
    rm -rf build

    echo "--- CMAKE CONFIGURE ---" > "../$DEBUG_LOG"
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_HOST_COMPILER=g++ \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        2>&1 | tee -a "../$DEBUG_LOG"

    echo "--- CMAKE BUILD ---" >> "../$DEBUG_LOG"
    cmake --build build --config Release -j$(nproc) 2>&1 | tee -a "../$DEBUG_LOG"
    BUILD_STATUS=${PIPESTATUS[0]}
    cd ..

    if [ $BUILD_STATUS -ne 0 ]; then
        echo -e "\n \033[1;31m[!] COMPILE FAILED.\033[0m"
        echo " Raw error logs: $DEBUG_LOG"
        read -p " Press Enter to return to menu..."
    else
        echo -e "\n \033[1;32mBuild Complete!\033[0m"
        sleep 2
    fi
}

# ── Argument probe helper ────────────────────────────────────────────────

arg_probe_valid() {
    local server_bin="$1"
    shift
    local probe_log=".dflash_arg_probe.log"
    local dummy_model=".dflash_arg_probe_dummy.gguf"
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

# ── Start DFlash server ─────────────────────────────────────────────────

start_dflash_server() {
    local vision_mode="$1"
    local vision_title=""
    if [[ "$vision_mode" == "vision" ]]; then
        vision_title=" + Vision"
    elif [[ "$vision_mode" == "vision-cpu" ]]; then
        vision_title=" + Vision (CPU mmproj)"
    fi

    echo ""
    echo -e " \033[1;36m>>> START DFLASH SERVER${vision_title} <<<\033[0m"
    echo ""

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo " Server is already running! Please stop it first [5]."
        sleep 2
        return
    fi

    server_bin="./${DFLASH_DIR}/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo " Error: llama-server not found at $server_bin"
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return..."
        return
    fi

    # ── List models ──
    raw_data=()
    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            [[ "$name" == *"mmproj"* ]] && continue
            [[ "$name" == *"dflash"* || "$name" == *"draft"* ]] && continue
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo " No target models found in $MODELS_DIR/"
        echo " (Draft/mmproj models are filtered out)"
        read -p " Press Enter to return..."
        return
    fi

    echo " Select MAIN (target) Model:"
    printf "   %-3s %-64s %s\n" "NR" "MODEL NAME" "SIZE"
    echo "   ----------------------------------------------------------------------"
    for i in "${!raw_data[@]}"; do
        IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
        printf "   %2d) %-64s [%s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
    done
    echo ""
    read -p " Model NR: " n
    n=$(echo "$n" | tr -d '[:space:]')

    local idx=$(( n - 1 ))
    local entry=${raw_data[$idx]}
    if [[ -z "$entry" ]]; then
        echo " Invalid model number."
        sleep 2
        return
    fi
    target="${entry%%|*}"

    # ── Draft model selection ──
    echo ""
    get_dflash_drafts
    if [[ ${#dflash_drafts[@]} -eq 0 ]]; then
        echo -e " \033[1;31mNo DFlash draft models found!\033[0m"
        echo " You need a DFlash draft model (e.g. dflash-draft-3.6-q8_0.gguf)"
        echo " Download from: https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF"
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
            if [[ "$d_low" == *"q8_0"* ]]; then tag="  <-- recommended"; fi
            echo "   [$((i+1))] $d$tag"
        done
        read -p " Choice (default 1): " draft_choice
        draft_choice=$(echo "$draft_choice" | tr -d '[:space:]')
        if [[ -z "$draft_choice" ]]; then draft_choice="1"; fi
        local didx=$((draft_choice - 1))
        draft_model="${dflash_drafts[$didx]:-${dflash_drafts[0]}}"
    fi

    echo " Draft: $draft_model"

    # ── Context / KV ──
    echo ""
    echo -e " \033[1;33mNote:\033[0m DFlash uses ~1.5 GB extra for tree verify buffers on top of"
    echo " normal VRAM. Use smaller context / lower KV cache than ik_llama.cpp."
    echo ""
    echo " Select Context Size / KV Cache:"
    echo "   [1] 4096    (4K, q4_0 KV - smallest, safest)"
    echo "   [2] 8192    (8K, q4_0 KV - short chat, safe on 24GB)"
    echo "   [3] 32768   (32K, q4_0 KV - normal chat, safe on 24GB)"
    echo "   [4] 65536   (64K, q4_0 KV - may OOM on 24GB with 27B models)"
    echo "   [5] 131072  (128K, q4_0 KV - likely OOM on 24GB)"
    echo "   [6] Custom"
    read -p " Choice (1-6, default 3): " ctx_choice
    ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

    case "$ctx_choice" in
        1) ctx="4096";   cache_type="q4_0" ;;
        2) ctx="8192";   cache_type="q4_0" ;;
        4) ctx="65536";  cache_type="q4_0" ;;
        5) ctx="131072"; cache_type="q4_0" ;;
        6)
            read -p " Enter context size: " ctx
            ctx=$(echo "$ctx" | tr -d '[:space:]')
            read -p " Enter KV cache type (q8_0/q5_0/q4_0/f16) [q4_0]: " cache_type
            cache_type=$(echo "$cache_type" | tr -d '[:space:]')
            if [[ -z "$cache_type" ]]; then cache_type="q4_0"; fi
            ;;
        *) ctx="32768"; cache_type="q4_0" ;;
    esac

    if [[ ! "$ctx" =~ ^[0-9]+$ ]]; then
        echo " Invalid context size."
        sleep 2
        return
    fi

    # ── Draft context ──
    echo ""
    echo " Draft model context (DFlash recommends small values):"
    echo "   [1] 256    (recommended — fastest draft)"
    echo "   [2] 512"
    echo "   [3] 1024"
    echo "   [4] Same as main: $ctx"
    read -p " Choice (1-4, default 1): " draft_ctx_choice
    draft_ctx_choice=$(echo "$draft_ctx_choice" | tr -d '[:space:]')

    case "$draft_ctx_choice" in
        2) draft_ctx="512" ;;
        3) draft_ctx="1024" ;;
        4) draft_ctx="$ctx" ;;
        *) draft_ctx="256" ;;
    esac

    # ── Vision ──
    mmproj_file=""
    vis_text=""
    cpu_vision="0"

    if [[ "$vision_mode" == "vision" || "$vision_mode" == "vision-cpu" ]]; then
        if [[ "$vision_mode" == "vision-cpu" ]]; then
            cpu_vision="1"
        fi
        get_mmproj_files
        if [[ ${#mmproj_files[@]} -eq 0 ]]; then
            echo ""
            echo -e " \033[1;31mNo vision projector (*mmproj*.gguf) found in $MODELS_DIR/\033[0m"
            read -p " Continue without vision? (Y/n): " cont_no_vis
            cont_no_vis=$(echo "$cont_no_vis" | tr -d '[:space:]')
            if [[ "$cont_no_vis" == "n" || "$cont_no_vis" == "N" ]]; then
                return
            fi
        else
            echo ""
            echo " Select Vision Projector:"
            for i in "${!mmproj_files[@]}"; do
                echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
            done
            read -p " Choice (default 1): " vis_choice
            vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
            if [[ -z "$vis_choice" ]]; then vis_choice="1"; fi
            local vidx=$((vis_choice - 1))
            mmproj_file="${mmproj_files[$vidx]:-${mmproj_files[0]}}"
            echo " Vision projector: $(basename "$mmproj_file")"

            if [[ "$cpu_vision" == "1" ]]; then
                vis_text="+ VisCPU "
            else
                vis_text="+ Vis "
            fi
        fi
    else
        # Ask if user wants vision
        get_mmproj_files
        if [[ ${#mmproj_files[@]} -gt 0 ]]; then
            echo ""
            read -p " Enable vision? (y/N): " load_vis
            load_vis=$(echo "$load_vis" | tr -d '[:space:]')
            if [[ "$load_vis" == "y" || "$load_vis" == "Y" ]]; then
                echo " Select Vision Projector:"
                for i in "${!mmproj_files[@]}"; do
                    echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
                done
                read -p " Choice (default 1): " vis_choice
                vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
                if [[ -z "$vis_choice" ]]; then vis_choice="1"; fi
                local vidx=$((vis_choice - 1))
                mmproj_file="${mmproj_files[$vidx]:-${mmproj_files[0]}}"
                echo " Vision projector: $(basename "$mmproj_file")"
                vis_text="+ Vis "

                echo ""
                read -p " Offload vision to CPU? (--no-mmproj-offload) (y/N): " cpu_vis_choice
                cpu_vis_choice=$(echo "$cpu_vis_choice" | tr -d '[:space:]')
                if [[ "$cpu_vis_choice" == "y" || "$cpu_vis_choice" == "Y" ]]; then
                    cpu_vision="1"
                    vis_text="+ VisCPU "
                fi
            fi
        fi
    fi

    # ── Build flags ──
    flag_summary=()
    skipped_summary=()

    fa_flags=()
    cache_flags=()
    vision_flags=()
    dflash_flags=()
    batch_flags=()

    # KV cache
    if arg_probe_valid "$server_bin" -ctk "$cache_type" -ctv "$cache_type"; then
        cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
        flag_summary+=("kv:-ctk/-ctv ${cache_type}")
    else
        skipped_summary+=("KV cache flags not accepted")
    fi

    # Flash attention
    if arg_probe_valid "$server_bin" -fa on; then
        fa_flags=(-fa on)
        flag_summary+=("flash:-fa on")
    elif arg_probe_valid "$server_bin" -fa; then
        fa_flags=(-fa)
        flag_summary+=("flash:-fa")
    else
        skipped_summary+=("flash attention flag not accepted")
    fi

    # DFlash spec-type + draft model
    draft_path="${MODELS_DIR}/${draft_model}"
    if arg_probe_valid "$server_bin" -md "$draft_path" --spec-type dflash; then
        dflash_flags=(-md "$draft_path" --spec-type dflash)
        flag_summary+=("dflash:--spec-type dflash")
        flag_summary+=("draft:-md $(basename "$draft_path")")
    elif arg_probe_valid "$server_bin" -md "$draft_path"; then
        dflash_flags=(-md "$draft_path")
        flag_summary+=("draft:-md $(basename "$draft_model")")
        skipped_summary+=("--spec-type dflash not accepted (may need newer buun-llama-cpp)")
    else
        echo -e " \033[1;31mError: draft model flag -md not accepted.\033[0m"
        read -p " Press Enter to return..."
        return
    fi

    # Draft GPU offload (use 99, not 999 — buun fork convention)
    if [[ ${#dflash_flags[@]} -gt 0 ]]; then
        if arg_probe_valid "$server_bin" "${dflash_flags[@]}" -ngld 99; then
            dflash_flags+=(-ngld 99)
            flag_summary+=("draft:-ngld 99")
        fi
    fi

    # Draft context
    if [[ ${#dflash_flags[@]} -gt 0 ]]; then
        if arg_probe_valid "$server_bin" "${dflash_flags[@]}" -cd "$draft_ctx"; then
            dflash_flags+=(-cd "$draft_ctx")
            flag_summary+=("draft:-cd ${draft_ctx}")
        fi
    fi

    # Batch (DFlash works best with small batches)
    if arg_probe_valid "$server_bin" -b 256 -ub 64; then
        batch_flags=(-b 256 -ub 64)
        flag_summary+=("batch:-b 256 -ub 64")
    elif arg_probe_valid "$server_bin" -b 512 -ub 128; then
        batch_flags=(-b 512 -ub 128)
        flag_summary+=("batch:-b 512 -ub 128")
    else
        skipped_summary+=("batch flags not accepted")
    fi

    # Vision
    if [[ -n "$mmproj_file" ]]; then
        if arg_probe_valid "$server_bin" --mmproj "$mmproj_file"; then
            vision_flags=(--mmproj "$mmproj_file")
            flag_summary+=("vision:--mmproj $(basename "$mmproj_file")")

            if [[ "$cpu_vision" == "1" ]]; then
                if arg_probe_valid "$server_bin" --no-mmproj-offload; then
                    vision_flags+=(--no-mmproj-offload)
                    flag_summary+=("vision-cpu:--no-mmproj-offload")
                else
                    skipped_summary+=("--no-mmproj-offload not accepted by this binary")
                fi
            fi
        else
            mmproj_file=""
            vis_text=""
            skipped_summary+=("--mmproj not accepted")
        fi
    fi

    # ── Build command ──
    # DFlash requires thinking OFF for good acceptance rates.
    # Always use --jinja + thinking=false.
    # Use -ngl 99 (not 999) — buun fork convention, avoids over-offloading.
    # Use -np 1 (not --parallel) — buun fork uses -np.
    cmd=("$server_bin"
        -m "${MODELS_DIR}/${target}"
        -md "${MODELS_DIR}/${draft_model}"
        --spec-type dflash
        -ngl 99
        -ngld 99
        -cd "$draft_ctx"
        -np 1
        -c "$ctx"
        "${fa_flags[@]}"
        "${cache_flags[@]}"
        "${batch_flags[@]}"
        --jinja
        --chat-template-kwargs '{"enable_thinking":false}'
        --host 0.0.0.0
        --port 8080
        "${vision_flags[@]}"
    )

    local target_short="$target"
    echo "DFLASH: ${target_short} ${vis_text}+ Draft(${draft_model}) [${ctx}/${cache_type}]" > .server_info_dflash

    echo ""
    echo " Starting DFlash server:"
    echo "   Main:      $target"
    echo "   Draft:     $draft_model"
    echo "   Draft ctx: $draft_ctx"
    echo "   Context:   $ctx"
    echo "   KV cache:  $cache_type"
    echo "   Vision:    ${vis_text:-off}"
    echo "   Thinking:  OFF (required for DFlash acceptance)"
    echo "   -ngl 99 -ngld 99 -np 1 (buun fork conventions)"
    echo ""
    echo " Accepted flags:"
    for x in "${flag_summary[@]}"; do echo "   + $x"; done

    if [[ ${#skipped_summary[@]} -gt 0 ]]; then
        echo ""
        echo " Skipped flags:"
        for x in "${skipped_summary[@]}"; do echo "   - $x"; done
    fi

    echo ""
    echo " Command:"
    printf ' %q' "${cmd[@]}"
    echo ""
    echo ""

    echo -e " \033[1;33mNote:\033[0m If you get OOM/crash on first message, try a smaller context or q4_0 KV."
    echo " DFlash tree verify buffers need ~1.5 GB extra VRAM beyond normal model usage."
    echo ""

    {
        echo "COMMAND PROFILE: dflash-speculative"
        printf '%q ' "${cmd[@]}"
        echo ""
        echo "----- llama-server output -----"
    } > server_dflash.log

    nohup "${cmd[@]}" >> server_dflash.log 2>&1 &
    server_pid=$!

    failed=0
    loaded=0

    for i in $(seq 1 120); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            failed=1
            break
        fi

        if grep -Eqi 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' server_dflash.log; then
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
        echo " Last 35 lines of server_dflash.log:"
        echo "------------------------------------------------------------"
        tail -n 35 server_dflash.log
        echo "------------------------------------------------------------"
        sleep 3
    else
        rm -f .server_info_dflash
        echo ""
        echo -e " \033[1;31mServer failed during startup.\033[0m"
        echo " Last 220 lines of server_dflash.log:"
        echo "------------------------------------------------------------"
        tail -n 220 server_dflash.log
        echo "------------------------------------------------------------"
        read -p " Press Enter to return to menu..."
    fi
}

# ── Main menu ────────────────────────────────────────────────────────────

setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""

    # List models (target models = not mmproj, not draft/dflash)
    raw_data=()
    draft_data=()

    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            local_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')

            if [[ "$local_name" == *"mmproj"* ]]; then
                continue
            fi

            if [[ "$local_name" == *"dflash"* || "$local_name" == *"draft"* ]]; then
                size=$(du -h "$f" | cut -f1)
                draft_data+=("${name}|${size}")
                continue
            fi

            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo "   (No target .gguf models found in ./$MODELS_DIR/)"
    else
        printf "   %-3s %-64s %s\n" "NR" "TARGET MODELS" "SIZE"
        echo "   ----------------------------------------------------------------------"
        for i in "${!raw_data[@]}"; do
            IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
            printf "   %2d) %-64s [%s]\n" "$((i+1))" "$(echo "$m_name" | cut -c1-64)" "$m_size"
        done
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

    # Check build status
    server_bin="./${DFLASH_DIR}/build/bin/llama-server"
    if [[ -x "$server_bin" ]]; then
        echo ""
        echo -e "   Binary: \033[1;32m${server_bin}\033[0m"
    else
        echo ""
        echo -e "   Binary: \033[1;31m${server_bin} (NOT BUILT — run [0] first)\033[0m"
    fi

    echo ""
    echo -e " \033[1;36m--- DFLASH SERVER ---\033[0m"
    echo " [1] START DFlash Server (main + draft, thinking OFF)"
    echo " [2] START DFlash Server + Vision"
    echo " [3] START DFlash Server + Vision (CPU mmproj / --no-mmproj-offload)"
    echo " [4] STOP SERVER"
    echo ""
    echo -e " \033[1;36m--- MANAGEMENT ---\033[0m"
    echo " [5] Download Model (.gguf URL)"
    echo " [6] Delete Model"
    echo " [0] INSTALL / UPDATE buun-llama-cpp (DFlash fork)"
    echo " [99] Exit"
    echo ""

    tput cnorm
    read -p " Select Action: " action
    action=$(echo "$action" | tr -d '[:space:]')

    case $action in
        0)
            kill -9 "$MONITOR_PID" 2>/dev/null
            wait "$MONITOR_PID" 2>/dev/null
            install_dflash
            setup_scroll_region
            monitor_loop &
            MONITOR_PID=$!
            ;;
        1)
            start_dflash_server "text"
            ;;
        2)
            start_dflash_server "vision"
            ;;
        3)
            start_dflash_server "vision-cpu"
            ;;
        4)
            echo ""
            echo -e " \033[1;36m>>> STOPPING SERVER <<<\033[0m"
            pkill -f "llama-server"
            rm -f .server_info_dflash
            echo " Server stopped."
            sleep 1
            ;;
        5)
            echo ""
            echo -e " \033[1;36m>>> DOWNLOAD MODEL <<<\033[0m"
            echo " Paste the direct download URL for a .gguf file."
            echo ""
            echo " DFlash draft models:"
            echo "   https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/dflash-draft-3.6-q8_0.gguf"
            echo "   https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF/resolve/main/dflash-draft-3.6-q4_k_m.gguf"
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
        99) cleanup ;;
        *) ;;
    esac
done
