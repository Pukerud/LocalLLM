#!/bin/bash

# =========================================================================
# LLAMA.CPP ULTIMATE DASHBOARD v1.55
# Change: Option [4] is now Long Context OpenClaw with NO speculative decoding.
# Context menus for OpenClaw now use 64K / 128K / 256K.
# =========================================================================

set +m
MODELS_DIR="llama_models"
DB_FILE="llamacpp_benchmark_db.txt"
KV_DB_FILE="llamacpp_kv_benchmark_db.tsv"
KV_EVENT_FILE="llamacpp_kv_benchmark_events.tsv"
DEBUG_LOG="compile_debug.log"
declare -A speed_cache
declare -A kv_speed_cache
declare -A kv_pp_cache
declare -A kv_status_cache
declare -A kv_time_cache

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

load_benchmarks() {
    speed_cache=()
    kv_speed_cache=()
    kv_status_cache=()
    kv_time_cache=()

    if [[ -f "$DB_FILE" ]]; then
        while IFS="=" read -r key speed; do
            [[ -z "$key" ]] && continue
            speed_cache["$key"]="$speed"
        done < "$DB_FILE"
    fi

    if [[ -f "$KV_DB_FILE" ]]; then
        while IFS="|" read -r model ctx kv status speed ts; do
            [[ -z "$model" || "$model" == "model" ]] && continue
            key="${model}:${ctx}:${kv}"
            kv_speed_cache["$key"]="$speed"
            kv_status_cache["$key"]="$status"
            kv_time_cache["$key"]="$ts"

            if [[ "$status" == "OK" && "$speed" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                base="${model}:${ctx}"
                cur="${speed_cache[$base]}"
                if [[ -z "$cur" || "$cur" == "Error" ]]; then
                    speed_cache["$base"]="$speed"
                elif [[ "$cur" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                    faster=$(awk -v a="$speed" -v b="$cur" 'BEGIN{print (a>b)?1:0}')
                    if [[ "$faster" == "1" ]]; then
                        speed_cache["$base"]="$speed"
                    fi
                fi
            fi
        done < "$KV_DB_FILE"
    fi
}

save_benchmark() {
    local model=$1
    local ctx=$2
    local speed=$3
    
    speed_cache["${model}:${ctx}"]="$speed"
    
    if [[ -f "$DB_FILE" ]]; then
        grep -v "^${model}:${ctx}=" "$DB_FILE" > "${DB_FILE}.tmp" 2>/dev/null
        mv "${DB_FILE}.tmp" "$DB_FILE"
    fi
    echo "${model}:${ctx}=${speed}" >> "$DB_FILE"
}

save_kv_benchmark() {
    local model="$1"
    local ctx="$2"
    local kv="$3"
    local status="$4"
    local speed="$5"
    local ts
    ts=$(date '+%Y-%m-%d_%H:%M:%S')

    if [[ ! -f "$KV_DB_FILE" ]]; then
        echo "model|ctx|kv|status|speed|timestamp" > "$KV_DB_FILE"
    fi

    local tmp="${KV_DB_FILE}.tmp.$$"
    awk -F'|' -v m="$model" -v c="$ctx" -v k="$kv" 'BEGIN{OFS=FS} NR==1 && $1=="model" {print; next} !($1==m && $2==c && $3==k) {print}' "$KV_DB_FILE" > "$tmp" 2>/dev/null || true
    mv "$tmp" "$KV_DB_FILE"

    echo "${model}|${ctx}|${kv}|${status}|${speed}|${ts}" >> "$KV_DB_FILE"

    kv_speed_cache["${model}:${ctx}:${kv}"]="$speed"
    kv_status_cache["${model}:${ctx}:${kv}"]="$status"
    kv_time_cache["${model}:${ctx}:${kv}"]="$ts"

    if [[ "$status" == "OK" && "$speed" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        local base="${model}:${ctx}"
        local cur="${speed_cache[$base]}"
        if [[ -z "$cur" || "$cur" == "Error" ]]; then
            speed_cache["$base"]="$speed"
        elif [[ "$cur" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            local faster
            faster=$(awk -v a="$speed" -v b="$cur" 'BEGIN{print (a>b)?1:0}')
            if [[ "$faster" == "1" ]]; then
                speed_cache["$base"]="$speed"
            fi
        fi
    fi
}

kv_quality_rank() {
    case "$1" in
        f16|bf16) echo 70 ;;
        q8_0) echo 60 ;;
        q6_K|q6_k) echo 55 ;;
        q5_1) echo 50 ;;
        q5_0) echo 48 ;;
        q4_1) echo 34 ;;
        q4_0) echo 32 ;;
        q3_K|q3_k|q3_*) echo 20 ;;
        *) echo 25 ;;
    esac
}

recommend_model_settings() {
    local model="$1"
    rec_ctx=""
    rec_kv=""
    rec_speed=""
    rec_reason=""

    [[ -f "$KV_DB_FILE" ]] || return 1

    local best_score=-999999
    local m ctx kv status speed ts ctx_score kv_score speed_int above_bonus score

    while IFS='|' read -r m ctx kv status speed ts; do
        [[ "$m" == "model" || -z "$m" ]] && continue
        [[ "$m" == "$model" ]] || continue
        [[ "$status" == "OK" ]] || continue
        [[ "$ctx" =~ ^[0-9]+$ ]] || continue
        [[ "$speed" =~ ^[0-9]+([.][0-9]+)?$ ]] || continue

        case "$ctx" in
            131072) ctx_score=400 ;;
            65536)  ctx_score=250 ;;
            262144) ctx_score=200 ;;
            *)      ctx_score=$(( ctx / 1024 )) ;;
        esac

        kv_score=$(kv_quality_rank "$kv")
        above_bonus=0
        case "$kv" in
            f16|bf16|q8_0|q6_K|q6_k|q5_1|q5_0) above_bonus=90 ;;
        esac

        speed_int=$(awk -v s="$speed" 'BEGIN{printf "%d", s}')
        score=$(( ctx_score + kv_score + above_bonus + speed_int / 20 ))

        if (( score > best_score )); then
            best_score=$score
            rec_ctx="$ctx"
            rec_kv="$kv"
            rec_speed="$speed"
            if [[ "$ctx" == "131072" ]]; then
                rec_reason="best balanced long-context result"
            elif [[ "$ctx" == "65536" ]]; then
                rec_reason="best high-quality safe result found"
            elif [[ "$ctx" == "262144" ]]; then
                rec_reason="max-context result found; may need vision disabled if VRAM is tight"
            else
                rec_reason="best benchmark result found"
            fi
        fi
    done < "$KV_DB_FILE"

    [[ -n "$rec_ctx" ]]
}

gguf_norm_name() {
    basename "$1" .gguf | tr '[:upper:]' '[:lower:]'
}

gguf_first_size_token() {
    gguf_norm_name "$1" | grep -oE '[0-9]+([.][0-9]+)?b' | head -n 1
}

gguf_first_active_token() {
    gguf_norm_name "$1" | grep -oE 'a[0-9]+([.][0-9]+)?b' | head -n 1
}

gguf_family_token() {
    gguf_norm_name "$1" | grep -oE 'qwen[0-9]+([.][0-9]+)?|llama[0-9]*|gemma[0-9]*|mistral|mixtral|glm|phi[0-9]*|internvl|minicpm|llava' | head -n 1
}

score_mmproj_for_model() {
    local model="$1"
    local mp="$2"
    local mn pn ms ps ma pa mf pf score token

    mn=$(gguf_norm_name "$model")
    pn=$(gguf_norm_name "$mp")
    ms=$(gguf_first_size_token "$model")
    ps=$(gguf_first_size_token "$mp")
    ma=$(gguf_first_active_token "$model")
    pa=$(gguf_first_active_token "$mp")
    mf=$(gguf_family_token "$model")
    pf=$(gguf_family_token "$mp")
    score=0

    if [[ -n "$ms" && -n "$ps" ]]; then
        if [[ "$ms" == "$ps" ]]; then
            score=$((score + 300))
        else
            score=$((score - 500))
        fi
    fi

    if [[ -n "$ma" || -n "$pa" ]]; then
        if [[ -n "$ma" && -n "$pa" && "$ma" == "$pa" ]]; then
            score=$((score + 100))
        elif [[ -n "$ma" && -n "$pa" && "$ma" != "$pa" ]]; then
            score=$((score - 150))
        fi
    fi

    if [[ -n "$mf" && -n "$pf" ]]; then
        if [[ "$mf" == "$pf" ]]; then
            score=$((score + 70))
        else
            score=$((score - 30))
        fi
    fi

    local mbase pbase
    mbase=$(echo "$mn" | sed -E 's/[-_.]q[0-9].*$//; s/[-_.](f16|fp16|bf16).*$//')
    pbase=$(echo "$pn" | sed -E 's/^mmproj[-_.]?//; s/[-_.](f16|fp16|bf16|q[0-9].*)$//')
    if [[ -n "$mbase" && -n "$pbase" ]]; then
        if [[ "$pbase" == *"$mbase"* || "$mbase" == *"$pbase"* ]]; then
            score=$((score + 100))
        fi
    fi

    for token in $(echo "$mn" | sed -E 's/[._-]+/ /g'); do
        [[ ${#token} -lt 3 ]] && continue
        case "$token" in
            gguf|mmproj|uncensored|instruct|chat|aggressive|q4|q5|q6|q8|k|m|s|xs|xl|f16|fp16|bf16) continue ;;
        esac
        if [[ "$pn" == *"$token"* ]]; then
            score=$((score + 5))
        fi
    done

    echo "$score"
}

auto_select_mmproj() {
    local target="$1"
    shift
    SELECTED_MMPROJ=""
    SELECTED_MMPROJ_SCORE=-999999

    local mp score
    for mp in "$@"; do
        [[ -f "$mp" ]] || continue
        score=$(score_mmproj_for_model "$target" "$mp")
        if (( score > SELECTED_MMPROJ_SCORE )); then
            SELECTED_MMPROJ_SCORE=$score
            SELECTED_MMPROJ="$mp"
        fi
    done
}

print_mmproj_scores() {
    local target="$1"
    shift
    local i=1 mp score
    for mp in "$@"; do
        score=$(score_mmproj_for_model "$target" "$mp")
        printf "   [%d] score=%s  %s\n" "$i" "$score" "$(basename "$mp")"
        i=$((i+1))
    done
}

choose_vision_projector() {
    local target="$1"
    local -a mmproj_files
    mmproj_file=""
    vis_text=""

    mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*mmproj*.gguf' 2>/dev/null)
    [[ ${#mmproj_files[@]} -gt 0 ]] || return 0

    read -p " Vision projector(s) detected. Enable vision? (y/n): " load_vis
    load_vis=$(echo "$load_vis" | tr -d '[:space:]')
    if [[ "$load_vis" != "y" && "$load_vis" != "Y" ]]; then
        return 0
    fi

    auto_select_mmproj "$target" "${mmproj_files[@]}"

    if [[ -n "$SELECTED_MMPROJ" && "$SELECTED_MMPROJ_SCORE" -ge 100 ]]; then
        mmproj_file="$SELECTED_MMPROJ"
        vis_text="+ Vis "
        echo " Auto-matched vision model: $(basename "$mmproj_file") (score $SELECTED_MMPROJ_SCORE)"
        return 0
    fi

    echo ""
    echo -e " \033[1;33mNo safe vision projector match found by filename.\033[0m"
    echo " The wrong mmproj can fail with n_embd mismatch, like 5120 text vs 2048 projector."
    echo " Available projector scores for this model:"
    print_mmproj_scores "$target" "${mmproj_files[@]}"
    echo ""
    read -p " Enter projector number to force-load anyway, or press Enter for NO vision: " vis_choice
    vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')

    if [[ "$vis_choice" =~ ^[0-9]+$ ]]; then
        local idx=$((vis_choice-1))
        if [[ -n "${mmproj_files[$idx]}" ]]; then
            mmproj_file="${mmproj_files[$idx]}"
            vis_text="+ VisManual "
            echo " Manual vision projector selected: $(basename "$mmproj_file")"
        fi
    fi
}

run_kv_matrix_benchmark() {
    local bench_path="./ik_llama.cpp/build/bin/llama-bench"
    if [[ ! -x "$bench_path" ]]; then
        echo " Error: llama-bench not found at $bench_path. Run Install/Update [0] first."
        sleep 2
        return
    fi

    echo ""
    echo -e " \033[1;36m>>> FULL KV/CONTEXT BENCHMARK MATRIX <<<\033[0m"
    echo " This benchmarks every non-mmproj GGUF with multiple context sizes and KV cache types."
    echo " Results are saved to: $KV_DB_FILE"
    echo ""

    read -p " Continue? This can run for hours. (y/n): " go
    go=$(echo "$go" | tr -d '[:space:]')
    [[ "$go" == "y" || "$go" == "Y" ]] || return

    local stop_after
    read -p " When finished, make sure llama-server is stopped? (Y/n): " stop_after
    stop_after=$(echo "$stop_after" | tr -d '[:space:]')
    if [[ -z "$stop_after" ]]; then stop_after="Y"; fi

    local ctx_line kv_line n_tokens skip_existing
    read -p " Contexts to test [default: 65536 131072 262144]: " ctx_line
    if [[ -z "$ctx_line" ]]; then ctx_line="65536 131072 262144"; fi

    read -p " KV caches to test [default: q8_0 q5_1 q5_0 q4_0]: " kv_line
    if [[ -z "$kv_line" ]]; then kv_line="q8_0 q5_1 q5_0 q4_0"; fi

    read -p " Tokens per test [default: 128]: " n_tokens
    n_tokens=$(echo "$n_tokens" | tr -d '[:space:]')
    if [[ -z "$n_tokens" ]]; then n_tokens="128"; fi

    read -p " Skip tests already successful in DB? (Y/n): " skip_existing
    skip_existing=$(echo "$skip_existing" | tr -d '[:space:]')
    if [[ -z "$skip_existing" ]]; then skip_existing="Y"; fi

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo " Stopping active llama-server to free VRAM for benchmark..."
        pkill -f "llama-server"
        rm -f .server_info
        sleep 3
    fi

    mapfile -t bench_models < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.gguf' ! -name '*mmproj*.gguf' -printf '%f\n' 2>/dev/null | sort)
    if [[ ${#bench_models[@]} -eq 0 ]]; then
        echo " No GGUF models found in $MODELS_DIR."
        return
    fi

    local total=0 done_count=0 model ctx kv key old_status bench_out speed status timeout_s
    for model in "${bench_models[@]}"; do
        for ctx in $ctx_line; do
            for kv in $kv_line; do
                total=$((total+1))
            done
        done
    done

    echo " Total tests queued: $total"
    echo "-------------------------------------------------------"
    printf " %-3s %-60s %-8s %-7s %-12s %s\n" "NR" "MODEL" "CTX" "KV" "SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for model in "${bench_models[@]}"; do
        for ctx in $ctx_line; do
            for kv in $kv_line; do
                done_count=$((done_count+1))
                key="${model}:${ctx}:${kv}"
                old_status="${kv_status_cache[$key]}"
                if [[ ("$skip_existing" == "Y" || "$skip_existing" == "y") && "$old_status" == "OK" ]]; then
                    printf " %-3s %-60s %-8s %-7s %-12s %s\n" "$done_count" "$(echo "$model" | cut -c1-60)" "$ctx" "$kv" "${kv_speed_cache[$key]}" "SKIP"
                    continue
                fi

                timeout_s=900
                bench_out=$(timeout "$timeout_s" "$bench_path" -m "${MODELS_DIR}/${model}" -p "$ctx" -n "$n_tokens" -ngl 999 -fa 1 -ctk "$kv" -ctv "$kv" 2>&1)
                speed=$(echo "$bench_out" | awk -F'|' '/tg/ {print $(NF-1); exit}' | grep -oE '[0-9]+([.][0-9]+)?' | head -n 1)

                if [[ -n "$speed" ]]; then
                    status="OK"
                    save_kv_benchmark "$model" "$ctx" "$kv" "$status" "$speed"
                    printf " %-3s %-60s %-8s %-7s \033[1;32m%-12s\033[0m %s\n" "$done_count" "$(echo "$model" | cut -c1-60)" "$ctx" "$kv" "${speed} t/s" "OK"
                else
                    status="Error"
                    save_kv_benchmark "$model" "$ctx" "$kv" "$status" "Error"
                    printf " %-3s %-60s %-8s %-7s \033[1;31m%-12s\033[0m %s\n" "$done_count" "$(echo "$model" | cut -c1-60)" "$ctx" "$kv" "ERR" "FAIL"
                    echo "   Diagnostic tail:"
                    echo "$bench_out" | tail -n 4 | sed 's/^/     /'
                fi
                sleep 1
            done
        done
    done

    echo "-------------------------------------------------------"
    echo " Benchmark matrix complete. Results saved to $KV_DB_FILE"

    if [[ "$stop_after" == "Y" || "$stop_after" == "y" ]]; then
        pkill -f "llama-server" 2>/dev/null || true
        rm -f .server_info
        echo " llama-server is stopped."
    fi
}

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
        if [[ -f ".server_info" ]]; then
            ACTIVE_INFO=$(cat .server_info)
            SERVER_STATUS="\033[1;32mRUNNING: ${ACTIVE_INFO}\033[0m"
        else
            SERVER_STATUS="\033[1;32mRUNNING (PID: $SERVER_PID)\033[0m"
        fi
    else
        SERVER_STATUS="\033[1;31mSTOPPED\033[0m"
        rm -f .server_info 2>/dev/null
    fi

    reset="\033[0m"; bold="\033[1m"

    tput sc
    tput cup 2 0
    echo -e "   ENGINE: ${bold}ik_llama.cpp${reset}    |  SERVER: ${SERVER_STATUS}\033[K"
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
    echo "   LLAMA.CPP ULTIMATE DASHBOARD v1.55"
    echo "==========================================================================================================================="
    tput cup 5 0
    echo "==========================================================================================================================="
    tput cup 6 0
    echo "   LOG OUTPUT:"
    echo "---------------------------------------------------------------------------------------------------------------------------"
}

format_speed_col() {
    local val=$1
    local width=$2
    local padded
    if [[ -z "$val" ]]; then 
        printf -v padded "%-${width}s" "-"
        echo -ne "$padded"
    elif [[ "$val" == "Error" ]]; then 
        printf -v padded "%-${width}s" "ERR"
        echo -ne "\033[0;31m${padded}\033[0m"
    else 
        printf -v padded "%-${width}s" "${val}"
        echo -ne "\033[0;32m${padded}\033[0m"
    fi
}

run_benchmark() {
    local model_file=$1
    local is_experimental=$2
    
    if [[ "$is_experimental" == "true" ]]; then
        echo "-------------------------------------------------------"
        echo " EXPERIMENTAL Q8 CACHE TEST: $model_file"
        echo "-------------------------------------------------------"
    else
        echo "-------------------------------------------------------"
        echo " STANDARD SCALING TEST: $model_file"
        echo "-------------------------------------------------------"
    fi
    
    local bench_path="./ik_llama.cpp/build/bin/llama-bench"
    if [[ ! -f "$bench_path" ]]; then
        echo " Error: llama-bench not found at $bench_path. Run Install/Update [0] first."
        sleep 2
        return
    fi
    
    printf " %-10s | %-15s | %s\n" "CONTEXT" "SPEED (tg)" "STATUS"
    echo "-------------------------------------------------------"

    local contexts
    if [[ "$is_experimental" == "true" ]]; then
        contexts=(4096 65536 131072 262144)
    else
        contexts=(4096 8192 16384 32768 65536 131072 262144)
    fi

    for ctx in "${contexts[@]}"; do
        
        if [[ "$is_experimental" == "true" ]]; then
            bench_out=$("$bench_path" -m "${MODELS_DIR}/${model_file}" -p "$ctx" -n 128 -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 2>&1)
        else
            bench_out=$("$bench_path" -m "${MODELS_DIR}/${model_file}" -p "$ctx" -n 128 -ngl 999 -fa 1 2>&1)
        fi
        
        speed=$(echo "$bench_out" | grep "tg" | awk -F'|' '{print $(NF-1)}' | grep -oE '[0-9]+\.[0-9]+' | head -n 1)

        if [[ -n "$speed" ]]; then
            printf " %-10s | \033[1;32m%-15s\033[0m | %s\n" "${ctx}" "${speed} t/s" "Success"
            save_benchmark "$model_file" "$ctx" "$speed"
        else
            printf " %-10s | %-15s | \033[1;31m%s\033[0m\n" "${ctx}" "CRASHED/OOM" "Failed"
            echo ""
            echo -e "\033[1;33m[!] CRASH DIAGNOSTIC:\033[0m"
            echo "$bench_out" | tail -n 8
            echo ""
            echo " Stopping test: Available VRAM limit reached."
            save_benchmark "$model_file" "$ctx" "Error"
            break
        fi
        sleep 1
    done
    echo "-------------------------------------------------------"
}


# === ADAPTIVE GENERAL SERVER HELPERS v1.55 START ===
start_adaptive_general_server() {
    local server_mode="$1"
    local mode_title="TEXT ONLY"
    local cpu_vision="0"
    if [[ "$server_mode" == "vision" ]]; then
        mode_title="TEXT + VISION"
    elif [[ "$server_mode" == "vision-cpu" ]]; then
        mode_title="TEXT + VISION (CPU mmproj)"
        cpu_vision="1"
    fi

    echo ""
    echo -e " \033[1;36m>>> START SERVER (${mode_title} / ADAPTIVE) <<<\033[0m"

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo " Server is already running! Please stop it first [5]."
        sleep 2
        return
    fi

    read -p " Select Model NR: " n
    n=$(echo "$n" | tr -d '[:space:]')
    target=$(get_model_name "$n")

    if [[ -z "$target" ]]; then
        echo " Invalid model number. Canceling."
        sleep 2
        return
    fi

    server_bin="./ik_llama.cpp/build/bin/llama-server"
    if [[ ! -x "$server_bin" ]]; then
        echo " Error: llama-server not found at $server_bin"
        echo " Run Install/Update [0] first."
        read -p " Press Enter to return to menu..."
        return
    fi

    rec_ctx=""
    rec_kv=""
    rec_speed=""
    rec_reason=""

    if type recommend_model_settings >/dev/null 2>&1 && recommend_model_settings "$target"; then
        echo ""
        echo -e " \033[1;32mBenchmark recommendation found:\033[0m"
        echo "   Context:  $rec_ctx"
        echo "   KV cache: $rec_kv"
        echo "   Speed:    ${rec_speed} t/s"
        echo "   Reason:   $rec_reason"
        echo ""
        read -p " Use this recommendation? (Y/n): " use_rec
        use_rec=$(echo "$use_rec" | tr -d '[:space:]')
        if [[ -z "$use_rec" || "$use_rec" == "y" || "$use_rec" == "Y" ]]; then
            ctx="$rec_ctx"
            cache_type="$rec_kv"
        else
            ctx=""
        fi
    else
        ctx=""
    fi

    if [[ -z "$ctx" ]]; then
        echo " Select Context Size / KV Cache:"
        echo "   [1] 8192   (8K, q8_0 KV - small/fast)"
        echo "   [2] 32768  (32K, q8_0 KV - normal chat)"
        echo "   [3] 65536  (64K, q8_0 KV - quality long context)"
        echo "   [4] 65536  (64K, q5_0 KV - lower VRAM)"
        echo "   [5] 131072 (128K, q4_0 KV - safe long context)"
        echo "   [6] 131072 (128K, q5_0 KV - higher quality, may OOM with vision)"
        echo "   [7] 262144 (256K, q4_0 KV - max context, may OOM)"
        echo "   [8] Custom"
        if [[ "$server_mode" == "vision" ]]; then
            read -p " Choice (1-8, default 5): " ctx_choice
        else
            read -p " Choice (1-8, default 2): " ctx_choice
        fi
        ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

        case $ctx_choice in
            1) ctx="8192";   cache_type="q8_0" ;;
            3) ctx="65536";  cache_type="q8_0" ;;
            4) ctx="65536";  cache_type="q5_0" ;;
            5) ctx="131072"; cache_type="q4_0" ;;
            6) ctx="131072"; cache_type="q5_0" ;;
            7) ctx="262144"; cache_type="q4_0" ;;
            8)
                read -p " Enter context size: " ctx
                ctx=$(echo "$ctx" | tr -d '[:space:]')
                read -p " Enter KV cache type (q8_0/q5_0/q4_0/f16) [q4_0]: " cache_type
                cache_type=$(echo "$cache_type" | tr -d '[:space:]')
                if [[ -z "$cache_type" ]]; then cache_type="q4_0"; fi
                ;;
            2) ctx="32768";  cache_type="q8_0" ;;
            *)
                if [[ "$server_mode" == "vision" ]]; then
                    ctx="131072"; cache_type="q4_0"
                else
                    ctx="32768"; cache_type="q8_0"
                fi
                ;;
        esac
    fi

    if [[ ! "$ctx" =~ ^[0-9]+$ ]]; then
        echo " Invalid context size. Canceling."
        sleep 2
        return
    fi

    echo ""
    echo " Select Chat / Thinking Mode:"
    echo "   [1] Default/client controls (no startup thinking flag)"
    echo "   [2] Jinja + Thinking ON"
    echo "   [3] Jinja + Thinking OFF"
    echo "   [4] Jinja only (model/client decides)"
    echo "   [5] Raw completion style (no Jinja/thinking flags)"
    read -p " Choice (1-5, default 2): " thinking_choice
    thinking_choice=$(echo "$thinking_choice" | tr -d '[:space:]')
    if [[ -z "$thinking_choice" ]]; then thinking_choice="2"; fi

    want_jinja="0"
    thinking_json=""
    thinking_text="Default"

    case $thinking_choice in
        2)
            want_jinja="1"
            thinking_json='{"enable_thinking":true}'
            thinking_text="ThinkOn"
            ;;
        3)
            want_jinja="1"
            thinking_json='{"enable_thinking":false}'
            thinking_text="ThinkOff"
            ;;
        4)
            echo ""
            echo -e " [1;36m>>> START OPENCLAW (LONG CONTEXT / VISION / SIMPLE DRAFT) <<<[0m"

            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Server is already running! Please stop it first [5]."
                sleep 2
                continue
            fi

            read -p " Select MAIN Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")

            if [[ -z "$target" ]]; then
                echo " Invalid model number. Canceling."
                sleep 2
                continue
            fi

            server_bin="./ik_llama.cpp/build/bin/llama-server"
            if [[ ! -x "$server_bin" ]]; then
                echo " Error: llama-server not found at $server_bin"
                echo " Run Install/Update [0] first."
                read -p " Press Enter to return to menu..."
                continue
            fi

            echo ""
            echo " Select Context Size / KV Cache:"
            echo "   [1] 65536  (64K, q8_0 KV)"
            echo "   [2] 65536  (64K, q5_0 KV)"
            echo "   [3] 131072 (128K, q4_0 KV - recommended)"
            echo "   [4] 131072 (128K, q5_0 KV)"
            echo "   [5] 262144 (256K, q4_0 KV)"
            read -p " Choice (1-5, default 3): " ctx_choice
            ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

            case "$ctx_choice" in
                1) ctx="65536";  cache_type="q8_0" ;;
                2) ctx="65536";  cache_type="q5_0" ;;
                4) ctx="131072"; cache_type="q5_0" ;;
                5) ctx="262144"; cache_type="q4_0" ;;
                *) ctx="131072"; cache_type="q4_0" ;;
            esac

            echo ""
            echo " Select Launch Profile:"
            echo "   [1] Full: flash + jinja + thinking off if accepted"
            echo "   [2] Stable: flash + parallel, no jinja extras"
            echo "   [3] Raw safe: context + KV + host/port only"
            read -p " Choice (1-3, default 1): " launch_profile
            launch_profile=$(echo "$launch_profile" | tr -d '[:space:]')
            [[ -z "$launch_profile" ]] && launch_profile="1"

            echo ""
            echo " Select Prompt Batch Mode:"
            echo "   [1] Default"
            echo "   [2] Safe:  -b 2048 -ub 512"
            echo "   [3] Turbo: -b 4096 -ub 1024"
            read -p " Choice (1-3, default 1): " batch_choice
            batch_choice=$(echo "$batch_choice" | tr -d '[:space:]')
            [[ -z "$batch_choice" ]] && batch_choice="1"

            mmproj_file=""
            vis_text=""

            mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*mmproj*.gguf' 2>/dev/null | sort)

            if [[ ${#mmproj_files[@]} -gt 0 ]]; then
                read -p " Enable vision? (y/N): " load_vis
                load_vis=$(echo "$load_vis" | tr -d '[:space:]')

                if [[ "$load_vis" == "y" || "$load_vis" == "Y" ]]; then
                    if type auto_select_mmproj >/dev/null 2>&1; then
                        auto_select_mmproj "$target" "${mmproj_files[@]}"
                        if [[ -n "$SELECTED_MMPROJ" && "$SELECTED_MMPROJ_SCORE" -ge 100 ]]; then
                            mmproj_file="$SELECTED_MMPROJ"
                            echo " Auto-matched vision model: $(basename "$mmproj_file")"
                        fi
                    fi

                    if [[ -z "$mmproj_file" ]]; then
                        echo " Select Vision Projector:"
                        for i in "${!mmproj_files[@]}"; do
                            echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
                        done
                        read -p " Choice: " vis_choice
                        vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
                        idx=$((vis_choice-1))
                        mmproj_file="${mmproj_files[$idx]:-${mmproj_files[0]}}"
                    fi

                    vis_text="+ Vis "
                fi
            fi

            draft_path=""
            draft_model=""
            draft_text=""
            draft_ctx="$ctx"
            draft_cache_type="$cache_type"
            draft_max="8"
            draft_min="0"
            draft_p_min="0.45"

            echo ""
            read -p " Enable draft/speculative decoding? (y/N): " enable_draft
            enable_draft=$(echo "$enable_draft" | tr -d '[:space:]')

            if [[ "$enable_draft" == "y" || "$enable_draft" == "Y" ]]; then
                mapfile -t draft_candidates < <(
                    find "$MODELS_DIR" -maxdepth 1 -type f -iname '*.gguf' ! -iname '*mmproj*.gguf' -printf '%f
' 2>/dev/null | sort
                )

                filtered_drafts=()
                for d in "${draft_candidates[@]}"; do
                    [[ "$d" == "$target" ]] && continue
                    filtered_drafts+=("$d")
                done

                if [[ ${#filtered_drafts[@]} -eq 0 ]]; then
                    echo " No draft candidates found."
                else
                    echo ""
                    echo " Select Draft Model:"
                    for i in "${!filtered_drafts[@]}"; do
                        d="${filtered_drafts[$i]}"
                        d_low=$(echo "$d" | tr '[:upper:]' '[:lower:]')
                        tag=""
                        if [[ "$d_low" == *"dflash"* ]]; then tag="  <-- DFlash"; fi
                        if [[ "$d_low" == *"draft"* && -z "$tag" ]]; then tag="  <-- draft"; fi
                        echo "   [$((i+1))] $d$tag"
                    done

                    read -p " Choice: " draft_choice
                    draft_choice=$(echo "$draft_choice" | tr -d '[:space:]')

                    if [[ "$draft_choice" =~ ^[0-9]+$ ]]; then
                        draft_idx=$((draft_choice-1))
                        draft_model="${filtered_drafts[$draft_idx]}"
                    else
                        draft_model="${filtered_drafts[0]}"
                    fi

                    if [[ -n "$draft_model" ]]; then
                        draft_path="${MODELS_DIR}/${draft_model}"
                    fi
                fi

                if [[ -n "$draft_path" && -f "$draft_path" ]]; then
                    echo ""
                    echo " Selected draft: $(basename "$draft_path")"
                    echo ""
                    echo " Draft tuning:"
                    echo "   [1] Conservative: max 4,  p-min 0.60"
                    echo "   [2] Balanced:     max 8,  p-min 0.45"
                    echo "   [3] Aggressive:   max 12, p-min 0.35"
                    read -p " Choice (1-3, default 2): " draft_tune
                    draft_tune=$(echo "$draft_tune" | tr -d '[:space:]')

                    case "$draft_tune" in
                        1) draft_max="4";  draft_p_min="0.60" ;;
                        3) draft_max="12"; draft_p_min="0.35" ;;
                        *) draft_max="8";  draft_p_min="0.45" ;;
                    esac

                    echo ""
                    echo " Draft context:"
                    echo "   [1] Same as main: $ctx"
                    echo "   [2] 65536"
                    echo "   [3] 32768"
                    echo "   [4] 16384"
                    read -p " Choice (1-4, default 1): " draft_ctx_choice
                    draft_ctx_choice=$(echo "$draft_ctx_choice" | tr -d '[:space:]')

                    case "$draft_ctx_choice" in
                        2) draft_ctx="65536" ;;
                        3) draft_ctx="32768" ;;
                        4) draft_ctx="16384" ;;
                        *) draft_ctx="$ctx" ;;
                    esac

                    draft_text="+ Draft "
                fi
            fi

            probe_log=".llama_arg_probe_option4.log"
            dummy_model=".llama_arg_probe_dummy.gguf"
            : > "$dummy_model"

            arg_combo_valid() {
                local -a test_cmd
                test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18083)
                timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true

                if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
                    return 1
                fi
                return 0
            }

            flag_summary=()
            skipped_summary=()

            fa_flags=()
            cache_flags=()
            parallel_flags=()
            jinja_flags=()
            batch_flags=()
            vision_flags=()
            draft_flags=()

            if arg_combo_valid -ctk "$cache_type" -ctv "$cache_type"; then
                cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
                flag_summary+=("main-kv:-ctk/-ctv ${cache_type}")
            else
                skipped_summary+=("main KV flags not accepted")
            fi

            if [[ "$launch_profile" == "1" || "$launch_profile" == "2" ]]; then
                if arg_combo_valid -fa on; then
                    fa_flags=(-fa on)
                    flag_summary+=("flash:-fa on")
                elif arg_combo_valid -fa; then
                    fa_flags=(-fa)
                    flag_summary+=("flash:-fa")
                else
                    skipped_summary+=("flash flag not accepted")
                fi

                if arg_combo_valid --parallel 1; then
                    parallel_flags=(--parallel 1)
                    flag_summary+=("parallel:--parallel 1")
                elif arg_combo_valid -np 1; then
                    parallel_flags=(-np 1)
                    flag_summary+=("parallel:-np 1")
                else
                    skipped_summary+=("parallel flag not accepted")
                fi
            fi

            if [[ "$launch_profile" == "1" ]]; then
                if arg_combo_valid --jinja; then
                    jinja_flags+=(--jinja)
                    flag_summary+=("chat:--jinja")
                fi

                if arg_combo_valid --chat-template-kwargs '{"enable_thinking":false}'; then
                    jinja_flags+=(--chat-template-kwargs '{"enable_thinking":false}')
                    flag_summary+=("chat:thinking=false")
                fi
            fi

            case "$batch_choice" in
                2) try_b="2048"; try_ub="512" ;;
                3) try_b="4096"; try_ub="1024" ;;
                *) try_b=""; try_ub="" ;;
            esac

            if [[ -n "$try_b" && "$launch_profile" != "3" ]]; then
                if arg_combo_valid -b "$try_b" -ub "$try_ub"; then
                    batch_flags=(-b "$try_b" -ub "$try_ub")
                    flag_summary+=("batch:-b ${try_b} -ub ${try_ub}")
                else
                    skipped_summary+=("batch flags not accepted")
                fi
            fi

            if [[ -n "$mmproj_file" ]]; then
                if arg_combo_valid --mmproj "$mmproj_file"; then
                    vision_flags=(--mmproj "$mmproj_file")
                    flag_summary+=("vision:--mmproj $(basename "$mmproj_file")")
                else
                    mmproj_file=""
                    vis_text=""
                    skipped_summary+=("--mmproj not accepted")
                fi
            fi

            if [[ -n "$draft_path" && -f "$draft_path" ]]; then
                if arg_combo_valid -md "$draft_path"; then
                    draft_flags+=(-md "$draft_path")
                    flag_summary+=("draft:-md $(basename "$draft_path")")

                    if arg_combo_valid -md "$draft_path" -ngld 999; then
                        draft_flags+=(-ngld 999)
                        flag_summary+=("draft:-ngld 999")
                    fi

                    if arg_combo_valid -md "$draft_path" -cd "$draft_ctx"; then
                        draft_flags+=(-cd "$draft_ctx")
                        flag_summary+=("draft:-cd ${draft_ctx}")
                    fi

                    if arg_combo_valid -md "$draft_path" -ctkd "$draft_cache_type" -ctvd "$draft_cache_type"; then
                        draft_flags+=(-ctkd "$draft_cache_type" -ctvd "$draft_cache_type")
                        flag_summary+=("draft-kv:-ctkd/-ctvd ${draft_cache_type}")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-max "$draft_max"; then
                        draft_flags+=(--draft-max "$draft_max")
                        flag_summary+=("draft:--draft-max ${draft_max}")
                    elif arg_combo_valid -md "$draft_path" --draft "$draft_max"; then
                        draft_flags+=(--draft "$draft_max")
                        flag_summary+=("draft:--draft ${draft_max}")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-min "$draft_min"; then
                        draft_flags+=(--draft-min "$draft_min")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-p-min "$draft_p_min"; then
                        draft_flags+=(--draft-p-min "$draft_p_min")
                    fi
                else
                    draft_flags=()
                    draft_text=""
                    skipped_summary+=("-md draft model not accepted")
                fi
            fi

            rm -f "$dummy_model" 2>/dev/null

            cmd=("$server_bin"
                -m "${MODELS_DIR}/${target}"
                -c "$ctx"
                -ngl 999
                "${fa_flags[@]}"
                "${cache_flags[@]}"
                "${batch_flags[@]}"
                "${jinja_flags[@]}"
                "${parallel_flags[@]}"
                --host 0.0.0.0
                --port 8080
                "${draft_flags[@]}"
                "${vision_flags[@]}"
            )

            target_short="$target"
            echo "LC: ${target_short} ${vis_text}${draft_text}[${ctx}/${cache_type}]" > .server_info

            echo ""
            echo " Starting OpenClaw:"
            echo "   Main:    $target"
            echo "   Context: $ctx"
            echo "   KV:      $cache_type"
            echo "   Vision:  ${vis_text:-off}"
            if [[ -n "$draft_text" ]]; then
                echo "   Draft:   $(basename "$draft_path")"
                echo "   Draft settings: ctx=${draft_ctx}, max=${draft_max}, min=${draft_min}, p-min=${draft_p_min}"
            else
                echo "   Draft:   off"
            fi

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

            {
                echo "COMMAND PROFILE: option4-simple-draft"
                printf '%q ' "${cmd[@]}"
                echo ""
                echo "----- llama-server output -----"
            } > server.log

            nohup "${cmd[@]}" >> server.log 2>&1 &
            server_pid=$!

            failed=0
            loaded=0

            for i in $(seq 1 120); do
                if ! kill -0 "$server_pid" >/dev/null 2>&1; then
                    failed=1
                    break
                fi

                if grep -Eqi 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' server.log; then
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

                echo " Last 35 lines of server.log:"
                echo "------------------------------------------------------------"
                tail -n 35 server.log
                echo "------------------------------------------------------------"
                sleep 3
                continue
            fi

            rm -f .server_info

            echo ""
            echo -e " [1;31mServer failed during startup.[0m"
            echo " Last 220 lines of server.log:"
            echo "------------------------------------------------------------"
            tail -n 220 server.log
            echo "------------------------------------------------------------"
            read -p " Press Enter to return to menu..."
            ;;
        5)
            want_jinja="0"
            thinking_json=""
            thinking_text="Raw"
            ;;
        *)
            want_jinja="0"
            thinking_json=""
            thinking_text="Default"
            ;;
    esac

    echo ""
    echo " Select Launch Feature Profile:"
    echo "   [1] Full adaptive: flash + parallel + selected chat/thinking flags"
    echo "   [2] Stable: parallel + selected chat/thinking flags, no flash"
    echo "   [3] Raw safe: context + KV + host/port only"
    read -p " Choice (1-3, default 1): " launch_profile
    launch_profile=$(echo "$launch_profile" | tr -d '[:space:]')
    if [[ -z "$launch_profile" ]]; then launch_profile="1"; fi

    echo ""
    echo " Select Prompt Batch Mode:"
    echo "   [1] Default server batch settings"
    echo "   [2] Safe:  -b 2048 -ub 512"
    echo "   [3] Turbo: -b 4096 -ub 1024"
    echo "   [4] Max:   -b 8192 -ub 1024"
    read -p " Choice (1-4, default 1): " batch_choice
    batch_choice=$(echo "$batch_choice" | tr -d '[:space:]')
    if [[ -z "$batch_choice" ]]; then batch_choice="1"; fi

    if [[ ( "$server_mode" == "vision" || "$server_mode" == "vision-cpu" ) && ( "$batch_choice" == "3" || "$batch_choice" == "4" ) ]]; then
        echo ""
        echo -e " \033[1;33mWarning:\033[0m Turbo/Max batch can start with vision but hang on the first image request."
        read -p " Downgrade to Default batch for this vision run? (Y/n): " batch_downgrade
        batch_downgrade=$(echo "$batch_downgrade" | tr -d '[:space:]')
        if [[ -z "$batch_downgrade" || "$batch_downgrade" == "y" || "$batch_downgrade" == "Y" ]]; then
            batch_choice="1"
        fi
    fi

    mmproj_file=""
    vis_text=""

    if [[ "$server_mode" == "vision" || "$server_mode" == "vision-cpu" ]]; then
        mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*mmproj*.gguf' 2>/dev/null)

        if [[ ${#mmproj_files[@]} -eq 0 ]]; then
            echo " Error: No vision projector (*mmproj*.gguf) found in $MODELS_DIR/"
            read -p " Press Enter to return to menu..."
            return
        fi

        if type auto_select_mmproj >/dev/null 2>&1; then
            auto_select_mmproj "$target" "${mmproj_files[@]}"
            if [[ -n "$SELECTED_MMPROJ" && "$SELECTED_MMPROJ_SCORE" -ge 100 ]]; then
                mmproj_file="$SELECTED_MMPROJ"
                echo " Auto-matched vision model: $(basename "$mmproj_file") (score $SELECTED_MMPROJ_SCORE)"
            else
                echo ""
                echo -e " \033[1;33mNo safe vision projector match found by filename.\033[0m"
                echo " Available projector scores:"
                if type print_mmproj_scores >/dev/null 2>&1; then
                    print_mmproj_scores "$target" "${mmproj_files[@]}"
                else
                    for i in "${!mmproj_files[@]}"; do
                        echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
                    done
                fi
                echo ""
                read -p " Enter projector number to force-load, or press Enter to cancel vision server: " vis_choice
                vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
                if [[ "$vis_choice" =~ ^[0-9]+$ ]]; then
                    idx=$((vis_choice-1))
                    mmproj_file="${mmproj_files[$idx]}"
                fi
            fi
        else
            echo " Select Vision Projector:"
            for i in "${!mmproj_files[@]}"; do
                echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
            done
            read -p " Choice: " vis_choice
            vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
            if [[ "$vis_choice" =~ ^[0-9]+$ ]]; then
                idx=$((vis_choice-1))
                mmproj_file="${mmproj_files[$idx]}"
            else
                mmproj_file="${mmproj_files[0]}"
            fi
        fi

        if [[ -z "$mmproj_file" ]]; then
            echo " Vision server canceled because no projector was selected."
            sleep 2
            return
        fi

        vis_text="+ Vis "
        if [[ "$cpu_vision" == "1" ]]; then
            vis_text="+ VisCPU "
        fi

        if [[ "$ctx" -ge 131072 && "$cache_type" != "q4_0" ]]; then
            echo ""
            echo -e " \033[1;33mWarning:\033[0m ${ctx} + vision + ${cache_type} KV may OOM or hang on 24GB."
            read -p " Downgrade KV cache to q4_0 for this vision run? (Y/n): " downgrade_kv
            downgrade_kv=$(echo "$downgrade_kv" | tr -d '[:space:]')
            if [[ -z "$downgrade_kv" || "$downgrade_kv" == "y" || "$downgrade_kv" == "Y" ]]; then
                cache_type="q4_0"
            fi
        fi
    fi

    probe_log=".llama_arg_probe_general.log"
    dummy_model=".llama_arg_probe_dummy.gguf"
    : > "$dummy_model"

    arg_combo_valid() {
        local -a test_cmd
        test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18082)
        timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true

        if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
            return 1
        fi
        return 0
    }

    flag_summary=()
    skipped_summary=()

    fa_flags=()
    cache_flags=()
    parallel_flags=()
    jinja_flags=()
    batch_flags=()
    vision_flags=()

    if arg_combo_valid -ctk "$cache_type" -ctv "$cache_type"; then
        cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
        flag_summary+=("kv:-ctk/-ctv ${cache_type}")
    elif arg_combo_valid --cache-type-k "$cache_type" --cache-type-v "$cache_type"; then
        cache_flags=(--cache-type-k "$cache_type" --cache-type-v "$cache_type")
        flag_summary+=("kv:--cache-type-k/v ${cache_type}")
    else
        skipped_summary+=("KV cache flags not accepted")
    fi

    if [[ "$launch_profile" == "1" ]]; then
        if arg_combo_valid -fa on; then
            fa_flags=(-fa on)
            flag_summary+=("flash:-fa on")
        elif arg_combo_valid --flash-attn on; then
            fa_flags=(--flash-attn on)
            flag_summary+=("flash:--flash-attn on")
        elif arg_combo_valid -fa; then
            fa_flags=(-fa)
            flag_summary+=("flash:-fa")
        elif arg_combo_valid --flash-attn; then
            fa_flags=(--flash-attn)
            flag_summary+=("flash:--flash-attn")
        else
            skipped_summary+=("flash attention flag not accepted")
        fi
    fi

    if [[ "$launch_profile" == "1" || "$launch_profile" == "2" ]]; then
        if arg_combo_valid --parallel 1; then
            parallel_flags=(--parallel 1)
            flag_summary+=("parallel:--parallel 1")
        elif arg_combo_valid -np 1; then
            parallel_flags=(-np 1)
            flag_summary+=("parallel:-np 1")
        else
            skipped_summary+=("parallel flag not accepted")
        fi
    fi

    if [[ "$launch_profile" != "3" && "$want_jinja" == "1" ]]; then
        if arg_combo_valid --jinja; then
            jinja_flags+=(--jinja)
            flag_summary+=("chat:--jinja")
        else
            skipped_summary+=("--jinja not accepted")
        fi

        if [[ -n "$thinking_json" ]]; then
            if arg_combo_valid --chat-template-kwargs "$thinking_json"; then
                jinja_flags+=(--chat-template-kwargs "$thinking_json")
                flag_summary+=("thinking:${thinking_json}")
            else
                skipped_summary+=("--chat-template-kwargs not accepted as startup flag; send chat_template_kwargs per request")
            fi
        fi
    elif [[ "$launch_profile" == "3" && "$thinking_choice" != "1" ]]; then
        skipped_summary+=("thinking/Jinja skipped because Raw safe profile was selected")
    fi

    case "$batch_choice" in
        2) try_b="2048"; try_ub="512" ;;
        3) try_b="4096"; try_ub="1024" ;;
        4) try_b="8192"; try_ub="1024" ;;
        *) try_b=""; try_ub="" ;;
    esac

    if [[ "$launch_profile" != "3" && -n "$try_b" ]]; then
        if arg_combo_valid -b "$try_b" -ub "$try_ub"; then
            batch_flags=(-b "$try_b" -ub "$try_ub")
            flag_summary+=("batch:-b ${try_b} -ub ${try_ub}")
        elif arg_combo_valid --batch-size "$try_b" --ubatch-size "$try_ub"; then
            batch_flags=(--batch-size "$try_b" --ubatch-size "$try_ub")
            flag_summary+=("batch:--batch-size ${try_b} --ubatch-size ${try_ub}")
        else
            skipped_summary+=("batch flags not accepted")
        fi
    fi

    if [[ -n "$mmproj_file" ]]; then
        if arg_combo_valid --mmproj "$mmproj_file"; then
            vision_flags=(--mmproj "$mmproj_file")
            flag_summary+=("vision:--mmproj $(basename "$mmproj_file")")

            if [[ "$cpu_vision" == "1" ]]; then
                if arg_combo_valid --no-mmproj-offload; then
                    vision_flags+=(--no-mmproj-offload)
                    flag_summary+=("vision-cpu:--no-mmproj-offload")
                else
                    skipped_summary+=("--no-mmproj-offload not accepted by this binary")
                fi
            fi
        else
            rm -f "$dummy_model" 2>/dev/null
            echo ""
            echo -e " \033[1;31mThis llama-server binary rejected --mmproj during argument probing.\033[0m"
            echo " Cannot start a vision server with this binary."
            read -p " Press Enter to return to menu..."
            return
        fi
    fi

    rm -f "$dummy_model" 2>/dev/null

    cmd=("$server_bin"
        -m "${MODELS_DIR}/${target}"
        -c "$ctx"
        -ngl 999
        "${fa_flags[@]}"
        "${cache_flags[@]}"
        "${batch_flags[@]}"
        "${jinja_flags[@]}"
        "${parallel_flags[@]}"
        --host 0.0.0.0
        --port 8080
        "${vision_flags[@]}"
    )

    target_short="$target"
    server_info_text="${mode_title}: ${target_short} ${vis_text}[${ctx}/${cache_type}/${thinking_text}]"

    echo ""
    echo " Starting adaptive server:"
    echo "   Mode:        $mode_title"
    echo "   Model:       $target"
    echo "   Context:     $ctx"
    echo "   KV cache:    $cache_type"
    echo "   Thinking:    $thinking_text"
    echo "   Vision:      ${vis_text:-off}"
    echo "   Draft/spec:  OFF"
    echo ""
    echo " Accepted feature flags:"
    if [[ ${#flag_summary[@]} -gt 0 ]]; then
        for x in "${flag_summary[@]}"; do echo "   + $x"; done
    else
        echo "   none"
    fi

    if [[ ${#skipped_summary[@]} -gt 0 ]]; then
        echo ""
        echo " Skipped unsupported flags:"
        for x in "${skipped_summary[@]}"; do echo "   - $x"; done
    fi

    echo ""
    echo " Command:"
    printf ' %q' "${cmd[@]}"
    echo ""
    echo ""

    {
        echo "COMMAND PROFILE: adaptive-general-${server_mode}"
        printf '%q ' "${cmd[@]}"
        echo ""
        echo "----- accepted feature flags -----"
        for x in "${flag_summary[@]}"; do echo "+ $x"; done
        echo "----- skipped feature flags -----"
        for x in "${skipped_summary[@]}"; do echo "- $x"; done
        echo "----- llama-server output -----"
    } > server.log

    nohup "${cmd[@]}" >> server.log 2>&1 &
    server_pid=$!

    launched=0
    failed=0

    for i in $(seq 1 120); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            failed=1
            break
        fi

        if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' server.log; then
            kill "$server_pid" >/dev/null 2>&1 || true
            wait "$server_pid" >/dev/null 2>&1 || true
            failed=1
            break
        fi

        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8080/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            launched=1
            break
        fi

        sleep 1
    done

    if [[ "$failed" == "0" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "$server_info_text" > .server_info
        echo ""
        if [[ "$launched" == "1" ]]; then
            echo " Server is loaded and health check returned OK."
        else
            echo " Server process is running. It may still be loading the model."
        fi
        echo " Last 35 lines of server.log:"
        echo "------------------------------------------------------------"
        tail -n 35 server.log
        echo "------------------------------------------------------------"
        sleep 3
        return
    fi

    rm -f .server_info

    echo ""
    echo -e " \033[1;31mServer failed during startup.\033[0m"
    echo " Last 220 lines of server.log:"
    echo "------------------------------------------------------------"
    tail -n 220 server.log
    echo "------------------------------------------------------------"

    if [[ -n "$mmproj_file" ]]; then
        echo ""
        echo " Vision was enabled. The projector may be incompatible, VRAM may be tight, or the image path may need safer batch settings."
        echo " Chosen mmproj: $(basename "$mmproj_file")"
        echo ""
        read -p " Start text-only fallback with same settings now? (Y/n): " text_fb
        text_fb=$(echo "$text_fb" | tr -d '[:space:]')

        if [[ -z "$text_fb" || "$text_fb" == "y" || "$text_fb" == "Y" ]]; then
            cmd_text=("$server_bin"
                -m "${MODELS_DIR}/${target}"
                -c "$ctx"
                -ngl 999
                "${fa_flags[@]}"
                "${cache_flags[@]}"
                "${batch_flags[@]}"
                "${jinja_flags[@]}"
                "${parallel_flags[@]}"
                --host 0.0.0.0
                --port 8080
            )

            {
                echo "COMMAND PROFILE: adaptive-general-text-fallback"
                printf '%q ' "${cmd_text[@]}"
                echo ""
                echo "----- llama-server output -----"
            } > server.log

            nohup "${cmd_text[@]}" >> server.log 2>&1 &
            server_pid=$!
            sleep 8

            if kill -0 "$server_pid" >/dev/null 2>&1; then
                echo "TEXT: ${target_short} [${ctx}/${cache_type}/${thinking_text}]" > .server_info
                echo " Text-only fallback server is running."
                sleep 3
                return
            else
                echo " Text-only fallback also failed. Last 160 lines:"
                tail -n 160 server.log
            fi
        fi
    fi

    read -p " Press Enter to return to menu..."
}
# === ADAPTIVE GENERAL SERVER HELPERS v1.55 END ===


# === BENCHMARK REWORK v1.55 START ===
BENCH_STATE_FILE=".llamacpp_benchmark_state"

kv_short() {
    case "$1" in
        q4_0) echo "q4" ;;
        q5_0) echo "q5" ;;
        q5_1) echo "q51" ;;
        q8_0) echo "q8" ;;
        default) echo "def" ;;
        *) echo "$1" ;;
    esac
}

kv_quality_rank() {
    case "$1" in
        q8_0) echo 80 ;;
        q5_1) echo 60 ;;
        q5_0) echo 55 ;;
        q4_0) echo 40 ;;
        default) echo 20 ;;
        *) echo 10 ;;
    esac
}

save_kv_result() {
    local model="$1"
    local ctx="$2"
    local kv="$3"
    local status="$4"
    local pp="$5"
    local tg="$6"
    local notes="$7"
    local ts
    ts=$(date '+%Y-%m-%d_%H:%M:%S')

    if [[ ! -f "$KV_DB_FILE" ]]; then
        echo "model|ctx|kv|status|pp_tps|tg_tps|timestamp|notes" > "$KV_DB_FILE"
    fi

    local tmp="${KV_DB_FILE}.tmp.$$"
    awk -F'|' -v m="$model" -v c="$ctx" -v k="$kv" 'BEGIN{OFS=FS} NR==1 && $1=="model" {print; next} !($1==m && $2==c && $3==k) {print}' "$KV_DB_FILE" > "$tmp" 2>/dev/null || true
    mv "$tmp" "$KV_DB_FILE"

    echo "${model}|${ctx}|${kv}|${status}|${pp}|${tg}|${ts}|${notes}" >> "$KV_DB_FILE"

    kv_speed_cache["${model}:${ctx}:${kv}"]="$tg"
    kv_pp_cache["${model}:${ctx}:${kv}"]="$pp"
    kv_status_cache["${model}:${ctx}:${kv}"]="$status"
    kv_time_cache["${model}:${ctx}:${kv}"]="$ts"

    if [[ "$status" == "OK" && "$tg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        speed_cache["${model}:${ctx}"]="$tg"
    fi
}

mark_interrupted_benchmark() {
    [[ -f "$BENCH_STATE_FILE" ]] || return 0

    local status model ctx kv started cmd
    status=""
    model=""
    ctx=""
    kv=""
    started=""
    cmd=""

    while IFS='=' read -r k v; do
        case "$k" in
            status) status="$v" ;;
            model) model="$v" ;;
            ctx) ctx="$v" ;;
            kv) kv="$v" ;;
            started) started="$v" ;;
            cmd) cmd="$v" ;;
        esac
    done < "$BENCH_STATE_FILE"

    if [[ "$status" == "RUNNING" && -n "$model" && -n "$ctx" && -n "$kv" ]]; then
        echo ""
        echo -e " \033[1;33mDetected interrupted benchmark from previous run.\033[0m"
        echo " Marking as REBOOT:"
        echo "   Model: $model"
        echo "   Ctx:   $ctx"
        echo "   KV:    $kv"
        save_kv_result "$model" "$ctx" "$kv" "REBOOT" "" "" "detected stale RUNNING state from $started"
    fi

    rm -f "$BENCH_STATE_FILE" 2>/dev/null
}

load_benchmarks() {
    speed_cache=()
    kv_speed_cache=()
    kv_pp_cache=()
    kv_status_cache=()
    kv_time_cache=()

    mark_interrupted_benchmark

    if [[ -f "$DB_FILE" ]]; then
        while IFS="=" read -r key speed; do
            [[ -z "$key" ]] && continue
            speed_cache["$key"]="$speed"
        done < "$DB_FILE"
    fi

    if [[ -f "$KV_DB_FILE" ]]; then
        while IFS="|" read -r model ctx kv status pp tg ts notes; do
            [[ -z "$model" || "$model" == "model" ]] && continue
            local key="${model}:${ctx}:${kv}"
            kv_speed_cache["$key"]="$tg"
            kv_pp_cache["$key"]="$pp"
            kv_status_cache["$key"]="$status"
            kv_time_cache["$key"]="$ts"
        done < "$KV_DB_FILE"
    fi
}

best_kv_cell() {
    local model="$1"
    local ctx="$2"
    local kv status tg pp best_kv best_tg best_pp best_rank rank reboot_kv err_kv
    best_kv=""
    best_tg=""
    best_pp=""
    best_rank=-1
    reboot_kv=""
    err_kv=""

    for kv in q8_0 q5_1 q5_0 q4_0; do
        status="${kv_status_cache["${model}:${ctx}:${kv}"]}"
        tg="${kv_speed_cache["${model}:${ctx}:${kv}"]}"
        pp="${kv_pp_cache["${model}:${ctx}:${kv}"]}"

        if [[ "$status" == "OK" && "$tg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            rank=$(kv_quality_rank "$kv")
            if (( rank > best_rank )); then
                best_rank=$rank
                best_kv="$kv"
                best_tg="$tg"
                best_pp="$pp"
            fi
        elif [[ "$status" == "REBOOT" && -z "$reboot_kv" ]]; then
            reboot_kv="$kv"
        elif [[ "$status" == "Error" && -z "$err_kv" ]]; then
            err_kv="$kv"
        elif [[ "$status" == "TIMEOUT" && -z "$err_kv" ]]; then
            err_kv="$kv"
        fi
    done

    if [[ -n "$best_kv" ]]; then
        if [[ "$best_pp" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            printf "%.0f/%.0f/%s" "$best_tg" "$best_pp" "$(kv_short "$best_kv")"
        else
            printf "%.0f/?/%s" "$best_tg" "$(kv_short "$best_kv")"
        fi
    elif [[ -n "$reboot_kv" ]]; then
        echo "RBT/$(kv_short "$reboot_kv")"
    elif [[ -n "$err_kv" ]]; then
        echo "ERR/$(kv_short "$err_kv")"
    else
        echo "-"
    fi
}

format_bench_cell() {
    local val="$1"
    local width="$2"
    local padded
    printf -v padded "%-${width}s" "$val"
    case "$val" in
        RBT*|ERR*) echo -ne "\033[0;31m${padded}\033[0m" ;;
        -) echo -ne "$padded" ;;
        *) echo -ne "\033[0;32m${padded}\033[0m" ;;
    esac
}

bench_models_list() {
    find "$MODELS_DIR" -maxdepth 1 -type f -name '*.gguf' ! -name '*mmproj*.gguf' -printf '%f\n' 2>/dev/null | sort
}

run_one_kv_benchmark() {
    local model="$1"
    local ctx="$2"
    local kv="$3"
    local n_tokens="${4:-128}"
    local bench_path="./ik_llama.cpp/build/bin/llama-bench"
    local cmd bench_out pp tg status notes started

    if [[ ! -x "$bench_path" ]]; then
        echo " Error: llama-bench not found at $bench_path. Run Install/Update [0] first."
        return 2
    fi

    cmd=("$bench_path" -m "${MODELS_DIR}/${model}" -p "$ctx" -n "$n_tokens" -ngl 999 -fa 1)

    if [[ "$kv" != "default" ]]; then
        cmd+=(-ctk "$kv" -ctv "$kv")
    fi

    started=$(date '+%Y-%m-%d_%H:%M:%S')

    {
        echo "status=RUNNING"
        echo "model=$model"
        echo "ctx=$ctx"
        echo "kv=$kv"
        echo "started=$started"
        printf 'cmd='
        printf '%q ' "${cmd[@]}"
        echo ""
    } > "$BENCH_STATE_FILE"
    sync

    bench_out=$(timeout 1800 "${cmd[@]}" 2>&1)
    rc=$?

    rm -f "$BENCH_STATE_FILE" 2>/dev/null
    sync

    pp=$(echo "$bench_out" | awk -F'|' '/pp[0-9]+/ {print $(NF-1); exit}' | grep -oE '[0-9]+([.][0-9]+)?' | head -n 1)
    tg=$(echo "$bench_out" | awk -F'|' '/tg[0-9]+/ {print $(NF-1); exit}' | grep -oE '[0-9]+([.][0-9]+)?' | head -n 1)

    if [[ -n "$tg" ]]; then
        status="OK"
        notes="ok"
        save_kv_result "$model" "$ctx" "$kv" "$status" "$pp" "$tg" "$notes"
        printf " %-48s %-8s %-7s \033[1;32m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "${tg} tg/s" "${pp:-?} pp/s" "OK"
        return 0
    else
        if [[ "$rc" == "124" ]]; then
            status="TIMEOUT"
            notes="timeout"
        else
            status="Error"
            notes=$(echo "$bench_out" | tail -n 2 | tr '\n' ' ' | sed 's/|//g' | cut -c1-120)
        fi

        save_kv_result "$model" "$ctx" "$kv" "$status" "" "" "$notes"
        printf " %-48s %-8s %-7s \033[1;31m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "$status" "-" "FAIL"
        echo "$bench_out" | tail -n 6 | sed 's/^/     /'
        return 1
    fi
}

run_practical_q4_benchmark() {
    local model="$1"
    local contexts=(4096 65536 131072 262144)
    local ctx

    echo "-------------------------------------------------------"
    echo " PRACTICAL BENCHMARK: q4_0 KV SAFE BASELINE"
    echo " MODEL: $model"
    echo "-------------------------------------------------------"
    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "MODEL" "CTX" "KV" "TG SPEED" "PP SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for ctx in "${contexts[@]}"; do
        run_one_kv_benchmark "$model" "$ctx" "q4_0" "128"
        sleep 1
    done

    echo "-------------------------------------------------------"
}

run_selected_kv_quality_benchmark() {
    local model="$1"
    local ctx_line kv_line n_tokens ctx kv

    echo ""
    echo " KV Quality Benchmark for: $model"
    read -p " Contexts [default: 4096 65536 131072 262144]: " ctx_line
    [[ -z "$ctx_line" ]] && ctx_line="4096 65536 131072 262144"

    read -p " KV order [default: q4_0 q5_0 q5_1 q8_0]: " kv_line
    [[ -z "$kv_line" ]] && kv_line="q4_0 q5_0 q5_1 q8_0"

    read -p " Tokens per test [default: 128]: " n_tokens
    n_tokens=$(echo "$n_tokens" | tr -d '[:space:]')
    [[ -z "$n_tokens" ]] && n_tokens="128"

    echo "-------------------------------------------------------"
    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "MODEL" "CTX" "KV" "TG SPEED" "PP SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for kv in $kv_line; do
        echo ""
        echo " Phase: KV=$kv"
        for ctx in $ctx_line; do
            run_one_kv_benchmark "$model" "$ctx" "$kv" "$n_tokens"
            sleep 1
        done
    done

    echo "-------------------------------------------------------"
}

run_all_models_kv_matrix() {
    local ctx_line kv_line n_tokens stop_after skip_existing skip_reboot model ctx kv key st
    local models=()

    echo ""
    echo -e " \033[1;36m>>> ALL MODELS PHASE ORDERED KV MATRIX <<<\033[0m"
    echo " This runs q4_0 for all models first, then q5_0, q5_1, q8_0."
    echo " Results are saved to: $KV_DB_FILE"
    echo " If a reboot happens, start the menu again and the interrupted test is marked REBOOT."
    echo ""

    read -p " Continue? (y/n): " go
    go=$(echo "$go" | tr -d '[:space:]')
    [[ "$go" == "y" || "$go" == "Y" ]] || return

    read -p " Stop llama-server when benchmark is done? (Y/n): " stop_after
    stop_after=$(echo "$stop_after" | tr -d '[:space:]')
    [[ -z "$stop_after" ]] && stop_after="Y"

    read -p " Contexts [default: 4096 65536 131072 262144]: " ctx_line
    [[ -z "$ctx_line" ]] && ctx_line="4096 65536 131072 262144"

    read -p " KV phases [default: q4_0 q5_0 q5_1 q8_0]: " kv_line
    [[ -z "$kv_line" ]] && kv_line="q4_0 q5_0 q5_1 q8_0"

    read -p " Tokens per test [default: 128]: " n_tokens
    n_tokens=$(echo "$n_tokens" | tr -d '[:space:]')
    [[ -z "$n_tokens" ]] && n_tokens="128"

    read -p " Skip existing OK results? (Y/n): " skip_existing
    skip_existing=$(echo "$skip_existing" | tr -d '[:space:]')
    [[ -z "$skip_existing" ]] && skip_existing="Y"

    read -p " Skip tests already marked REBOOT? (Y/n): " skip_reboot
    skip_reboot=$(echo "$skip_reboot" | tr -d '[:space:]')
    [[ -z "$skip_reboot" ]] && skip_reboot="Y"

    if [[ -n $(pgrep -f "llama-server") ]]; then
        echo " Stopping active llama-server to free VRAM..."
        pkill -f "llama-server"
        rm -f .server_info
        sleep 3
    fi

    mapfile -t models < <(bench_models_list)

    if [[ ${#models[@]} -eq 0 ]]; then
        echo " No GGUF models found in $MODELS_DIR."
        return
    fi

    echo "-------------------------------------------------------"
    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "MODEL" "CTX" "KV" "TG SPEED" "PP SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for kv in $kv_line; do
        echo ""
        echo "======================================================="
        echo " PHASE: $kv for all models"
        echo "======================================================="

        for model in "${models[@]}"; do
            for ctx in $ctx_line; do
                key="${model}:${ctx}:${kv}"
                st="${kv_status_cache[$key]}"

                if [[ ("$skip_existing" == "Y" || "$skip_existing" == "y") && "$st" == "OK" ]]; then
                    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "${kv_speed_cache[$key]}" "${kv_pp_cache[$key]:--}" "SKIP OK"
                    continue
                fi

                if [[ ("$skip_reboot" == "Y" || "$skip_reboot" == "y") && "$st" == "REBOOT" ]]; then
                    printf " %-48s %-8s %-7s \033[1;31m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "REBOOT" "-" "SKIP RBT"
                    continue
                fi

                run_one_kv_benchmark "$model" "$ctx" "$kv" "$n_tokens"
                sleep 1
            done
        done
    done

    echo "-------------------------------------------------------"
    echo " All-model KV matrix complete."

    if [[ "$stop_after" == "Y" || "$stop_after" == "y" ]]; then
        pkill -f "llama-server" 2>/dev/null || true
        rm -f .server_info
        echo " llama-server is stopped."
    fi
}
# === BENCHMARK REWORK v1.55 END ===



# === BENCHMARK SKIP/PRESERVE UPDATE v1.55 START ===
save_kv_event() {
    local model="$1"
    local ctx="$2"
    local kv="$3"
    local status="$4"
    local pp="$5"
    local tg="$6"
    local notes="$7"
    local ts
    ts=$(date '+%Y-%m-%d_%H:%M:%S')

    if [[ ! -f "$KV_EVENT_FILE" ]]; then
        echo "model|ctx|kv|status|pp_tps|tg_tps|timestamp|notes" > "$KV_EVENT_FILE"
    fi

    echo "${model}|${ctx}|${kv}|${status}|${pp}|${tg}|${ts}|${notes}" >> "$KV_EVENT_FILE"
}

get_existing_kv_status() {
    local model="$1"
    local ctx="$2"
    local kv="$3"

    if [[ -f "$KV_DB_FILE" ]]; then
        awk -F'|' -v m="$model" -v c="$ctx" -v k="$kv" '
            NR > 1 && $1 == m && $2 == c && $3 == k {
                status = $4
            }
            END {
                if (status != "") print status
            }
        ' "$KV_DB_FILE"
    fi
}

save_kv_result() {
    local model="$1"
    local ctx="$2"
    local kv="$3"
    local status="$4"
    local pp="$5"
    local tg="$6"
    local notes="$7"
    local ts existing_status
    ts=$(date '+%Y-%m-%d_%H:%M:%S')

    if [[ ! -f "$KV_DB_FILE" ]]; then
        echo "model|ctx|kv|status|pp_tps|tg_tps|timestamp|notes" > "$KV_DB_FILE"
    fi

    existing_status=$(get_existing_kv_status "$model" "$ctx" "$kv")

    # Preserve good benchmark data. A later reboot/error should not erase a previous OK result.
    if [[ "$existing_status" == "OK" && "$status" != "OK" ]]; then
        save_kv_event "$model" "$ctx" "$kv" "$status" "$pp" "$tg" "preserved existing OK; new event: $notes"
        return 0
    fi

    # Do not replace a REBOOT marker with a generic Error/TIMEOUT unless the new run succeeds.
    if [[ "$existing_status" == "REBOOT" && "$status" != "OK" && "$status" != "REBOOT" ]]; then
        save_kv_event "$model" "$ctx" "$kv" "$status" "$pp" "$tg" "preserved existing REBOOT; new event: $notes"
        return 0
    fi

    local tmp="${KV_DB_FILE}.tmp.$$"
    awk -F'|' -v m="$model" -v c="$ctx" -v k="$kv" 'BEGIN{OFS=FS} NR==1 && $1=="model" {print; next} !($1==m && $2==c && $3==k) {print}' "$KV_DB_FILE" > "$tmp" 2>/dev/null || true
    mv "$tmp" "$KV_DB_FILE"

    echo "${model}|${ctx}|${kv}|${status}|${pp}|${tg}|${ts}|${notes}" >> "$KV_DB_FILE"
    save_kv_event "$model" "$ctx" "$kv" "$status" "$pp" "$tg" "$notes"

    kv_speed_cache["${model}:${ctx}:${kv}"]="$tg"
    kv_pp_cache["${model}:${ctx}:${kv}"]="$pp"
    kv_status_cache["${model}:${ctx}:${kv}"]="$status"
    kv_time_cache["${model}:${ctx}:${kv}"]="$ts"

    if [[ "$status" == "OK" && "$tg" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        speed_cache["${model}:${ctx}"]="$tg"
    fi
}

run_selected_kv_quality_benchmark() {
    local model="$1"
    local ctx_line kv_line n_tokens skip_existing skip_reboot ctx kv key st

    echo ""
    echo " KV Quality Benchmark for: $model"
    echo " Existing OK results are skipped by default so [7] does not rerun q4 rows from [6]."

    read -p " Contexts [default: 4096 65536 131072 262144]: " ctx_line
    [[ -z "$ctx_line" ]] && ctx_line="4096 65536 131072 262144"

    read -p " KV order [default: q4_0 q5_0 q5_1 q8_0]: " kv_line
    [[ -z "$kv_line" ]] && kv_line="q4_0 q5_0 q5_1 q8_0"

    read -p " Tokens per test [default: 128]: " n_tokens
    n_tokens=$(echo "$n_tokens" | tr -d '[:space:]')
    [[ -z "$n_tokens" ]] && n_tokens="128"

    read -p " Skip existing OK results? (Y/n): " skip_existing
    skip_existing=$(echo "$skip_existing" | tr -d '[:space:]')
    [[ -z "$skip_existing" ]] && skip_existing="Y"

    read -p " Skip tests already marked REBOOT? (Y/n): " skip_reboot
    skip_reboot=$(echo "$skip_reboot" | tr -d '[:space:]')
    [[ -z "$skip_reboot" ]] && skip_reboot="Y"

    echo "-------------------------------------------------------"
    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "MODEL" "CTX" "KV" "TG SPEED" "PP SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for kv in $kv_line; do
        echo ""
        echo " Phase: KV=$kv"
        for ctx in $ctx_line; do
            key="${model}:${ctx}:${kv}"
            st="${kv_status_cache[$key]}"

            if [[ ("$skip_existing" == "Y" || "$skip_existing" == "y") && "$st" == "OK" ]]; then
                printf " %-48s %-8s %-7s \033[1;32m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "${kv_speed_cache[$key]}" "${kv_pp_cache[$key]:--}" "SKIP OK"
                continue
            fi

            if [[ ("$skip_reboot" == "Y" || "$skip_reboot" == "y") && "$st" == "REBOOT" ]]; then
                printf " %-48s %-8s %-7s \033[1;31m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "$kv" "REBOOT" "-" "SKIP RBT"
                continue
            fi

            run_one_kv_benchmark "$model" "$ctx" "$kv" "$n_tokens"
            sleep 1
        done
    done

    echo "-------------------------------------------------------"
}

run_practical_q4_benchmark() {
    local model="$1"
    local contexts=(4096 65536 131072 262144)
    local ctx key st skip_existing

    echo "-------------------------------------------------------"
    echo " PRACTICAL BENCHMARK: q4_0 KV SAFE BASELINE"
    echo " MODEL: $model"
    echo "-------------------------------------------------------"

    read -p " Skip existing OK q4_0 results? (Y/n): " skip_existing
    skip_existing=$(echo "$skip_existing" | tr -d '[:space:]')
    [[ -z "$skip_existing" ]] && skip_existing="Y"

    printf " %-48s %-8s %-7s %-12s %-12s %s\n" "MODEL" "CTX" "KV" "TG SPEED" "PP SPEED" "STATUS"
    echo "-------------------------------------------------------"

    for ctx in "${contexts[@]}"; do
        key="${model}:${ctx}:q4_0"
        st="${kv_status_cache[$key]}"

        if [[ ("$skip_existing" == "Y" || "$skip_existing" == "y") && "$st" == "OK" ]]; then
            printf " %-48s %-8s %-7s \033[1;32m%-12s\033[0m %-12s %s\n" "$(echo "$model" | cut -c1-48)" "$ctx" "q4_0" "${kv_speed_cache[$key]}" "${kv_pp_cache[$key]:--}" "SKIP OK"
            continue
        fi

        run_one_kv_benchmark "$model" "$ctx" "q4_0" "128"
        sleep 1
    done

    echo "-------------------------------------------------------"
}
# === BENCHMARK SKIP/PRESERVE UPDATE v1.55 END ===

echo "Loading Dashboard..."
load_benchmarks
setup_scroll_region
monitor_loop &
MONITOR_PID=$!

while true; do
    echo ""
    
    raw_data=()
    if [[ -d "$MODELS_DIR" ]]; then
        for f in "$MODELS_DIR"/*.gguf; do
            [[ -e "$f" ]] || continue
            name=$(basename "$f")
            
            if [[ "$name" == *"mmproj"* ]]; then
                continue
            fi
            
            size=$(du -h "$f" | cut -f1)
            raw_data+=("${name}|${size}")
        done
    fi

    if [[ ${#raw_data[@]} -eq 0 ]]; then
        echo "   (No standard .gguf models found in ./$MODELS_DIR/)"
    else
        printf "   %-3s %-64s %-7s %-14s %-14s %-14s %-14s\n" "NR" "MODEL NAME" "SIZE" "4K T/P/KV" "64K T/P/KV" "128K T/P/KV" "256K T/P/KV"
        echo "   --------------------------------------------------------------------------------------------------------------------------------------------"
        for i in "${!raw_data[@]}"; do
            IFS="|" read -r m_name m_size <<< "${raw_data[$i]}"
            m_name_trunc=$(echo "$m_name" | cut -c 1-64)

            c_4k=$(best_kv_cell "$m_name" 4096)
            c_64k=$(best_kv_cell "$m_name" 65536)
            c_128k=$(best_kv_cell "$m_name" 131072)
            c_256k=$(best_kv_cell "$m_name" 262144)

            s_4k=$(format_bench_cell "$c_4k" 14)
            s_64k=$(format_bench_cell "$c_64k" 14)
            s_128k=$(format_bench_cell "$c_128k" 14)
            s_256k=$(format_bench_cell "$c_256k" 14)

            printf "   %2d) %-64s [%-5s] %s %s %s %s\n" "$((i+1))" "$m_name_trunc" "$m_size" "$s_4k" "$s_64k" "$s_128k" "$s_256k"
        done
    fi

    echo ""
    echo -e " \033[1;36m--- SERVER CONTROLS ---\033[0m"
    echo " [1] START SERVER (Adaptive Text + Thinking)"
    echo " [2] START SERVER (Adaptive Vision + Thinking)"
    echo " [12] START SERVER (Vision + CPU Offload / --no-mmproj-offload)"
    echo " [3] START OPENCLAW (Standard 64K/128K/256K)"
    echo " [4] START OPENCLAW (Long Context + Vision + Simple Draft)"
    echo " [5] STOP SERVER"
    echo " [11] START COWORK SERVER (Anthropic / Claude-compatible API)"
    echo ""
    echo -e " \033[1;36m--- BENCHMARKING ---\033[0m"
    echo " [6] Practical Benchmark (q4_0 KV Safe Baseline)"
    echo " [7] KV Quality Benchmark (selected model, skip OK by default)"
    echo " [10] Benchmark ALL models (phase ordered KV matrix)"
    echo ""
    echo -e " \033[1;36m--- MANAGEMENT ---\033[0m"
    echo " [8] Download Model (.gguf URL)"
    echo " [9] Delete Model"
    echo " [0] INSTALL / UPDATE ik_llama.cpp (High-Performance Fork)"
    echo " [99] Exit"
    echo ""
    
    tput cnorm
    read -p " Select Action: " action
    action=$(echo "$action" | tr -d '[:space:]')

    get_model_name() {
        local idx=$(( $1 - 1 ))
        local entry=${raw_data[$idx]}
        echo "${entry%%|*}"
    }

    case $action in
        0)
            kill -9 "$MONITOR_PID" 2>/dev/null
            wait "$MONITOR_PID" 2>/dev/null
            tput csr 0 "$(tput lines)"
            clear
            echo " Pulling latest ik_llama.cpp from GitHub..."
            if [[ ! -d "ik_llama.cpp" ]]; then
                git clone https://github.com/ikawrakow/ik_llama.cpp.git
            else
                cd ik_llama.cpp && git pull && cd ..
            fi
            
            echo " Compiling natively for RTX 4090 architecture (89) using modern GCC..."
            export CC=gcc
            export CXX=g++
            
            cd ik_llama.cpp
            rm -rf build 
            
            echo "--- CMAKE CONFIGURE STAGE ---" > "../$DEBUG_LOG"
            cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc 2>&1 | tee -a "../$DEBUG_LOG"
            
            echo "--- CMAKE BUILD STAGE ---" >> "../$DEBUG_LOG"
            cmake --build build --config Release -j$(nproc) 2>&1 | tee -a "../$DEBUG_LOG"
            BUILD_STATUS=${PIPESTATUS[0]}
            cd ..
            
            if [ $BUILD_STATUS -ne 0 ]; then
                echo -e "\n \033[1;31m[!] COMPILE FAILED.\033[0m"
                echo " The raw error logs have been saved to: $DEBUG_LOG"
                echo ""
                read -p " Press Enter to return to menu..."
            else
                echo -e "\n \033[1;32mBuild Complete!\033[0m"
                sleep 2
            fi
            
            setup_scroll_region
            monitor_loop &
            MONITOR_PID=$!
            ;;
        1)
            start_adaptive_general_server "text"
            ;;
        2)
            start_adaptive_general_server "vision"
            ;;
        12)
            start_adaptive_general_server "vision-cpu"
            ;;
        3)
            echo ""
            echo -e " \033[1;36m>>> START OPENCLAW (STANDARD) <<<\033[0m"
            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Server is already running! Please stop it first [5]."
                sleep 2
                continue
            fi
            read -p " Select MAIN Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")
            if [[ -n "$target" ]]; then
                echo " Select Context Size:"
                echo "   [1] 65536  (64K - Default)"
                echo "   [2] 131072 (128K)"
                echo "   [3] 262144 (256K)"
                read -p " Choice (1-3): " ctx_choice
                ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')
                
                case $ctx_choice in
                    2) ctx="131072" ;;
                    3) ctx="262144" ;;
                    *) ctx="65536" ;;
                esac
                
                mmproj_flag=""
                vis_text=""
                mmproj_files=($(ls "${MODELS_DIR}"/*mmproj*.gguf 2>/dev/null))
                if [[ ${#mmproj_files[@]} -gt 0 ]]; then
                    read -p " Vision projector(s) detected. Enable vision? (y/n): " load_vis
                    load_vis=$(echo "$load_vis" | tr -d '[:space:]')
                    if [[ "$load_vis" == "y" || "$load_vis" == "Y" ]]; then
                        target_prefix=$(echo "$target" | awk -F'[-_]' '{print tolower($1)}')
                        smart_mmproj=""
                        for mp in "${mmproj_files[@]}"; do
                            mp_name=$(basename "$mp" | tr '[:upper:]' '[:lower:]')
                            if [[ "$mp_name" == *"$target_prefix"* ]]; then
                                smart_mmproj="$mp"
                                break
                            fi
                        done

                        if [[ -n "$smart_mmproj" ]]; then
                            mmproj_file="$smart_mmproj"
                            echo " Auto-matched vision model: $(basename "$mmproj_file")"
                        elif [[ ${#mmproj_files[@]} -eq 1 ]]; then
                            mmproj_file="${mmproj_files[0]}"
                        else
                            echo " Select Vision Projector:"
                            for i in "${!mmproj_files[@]}"; do
                                echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
                            done
                            read -p " Choice: " vis_choice
                            idx=$((vis_choice-1))
                            mmproj_file="${mmproj_files[$idx]:-${mmproj_files[0]}}"
                        fi
                        mmproj_flag="--mmproj ${mmproj_file}"
                        vis_text="+ Vis "
                    fi
                fi
                
                target_short="$target"
                echo "API: ${target_short} ${vis_text}[${ctx}]" > .server_info
                
                echo " Starting llama-server for OpenClaw (Standard ${ctx})..."
                nohup ./ik_llama.cpp/build/bin/llama-server -m "${MODELS_DIR}/${target}" -c "$ctx" -ngl 999 -fa on $mmproj_flag --jinja --chat-template-kwargs '{"enable_thinking":false}' --host 0.0.0.0 --port 8080 > server.log 2>&1 &
                sleep 2
            else
                echo " Invalid model number. Canceling."
                sleep 2
            fi
            ;;
        4)
            echo ""
            echo -e " [1;36m>>> START OPENCLAW (LONG CONTEXT / VISION / SIMPLE DRAFT) <<<[0m"

            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Server is already running! Please stop it first [5]."
                sleep 2
                continue
            fi

            read -p " Select MAIN Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")

            if [[ -z "$target" ]]; then
                echo " Invalid model number. Canceling."
                sleep 2
                continue
            fi

            server_bin="./ik_llama.cpp/build/bin/llama-server"
            if [[ ! -x "$server_bin" ]]; then
                echo " Error: llama-server not found at $server_bin"
                echo " Run Install/Update [0] first."
                read -p " Press Enter to return to menu..."
                continue
            fi

            echo ""
            echo " Select Context Size / KV Cache:"
            echo "   [1] 65536  (64K, q8_0 KV)"
            echo "   [2] 65536  (64K, q5_0 KV)"
            echo "   [3] 131072 (128K, q4_0 KV - recommended)"
            echo "   [4] 131072 (128K, q5_0 KV)"
            echo "   [5] 262144 (256K, q4_0 KV)"
            read -p " Choice (1-5, default 3): " ctx_choice
            ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

            case "$ctx_choice" in
                1) ctx="65536";  cache_type="q8_0" ;;
                2) ctx="65536";  cache_type="q5_0" ;;
                4) ctx="131072"; cache_type="q5_0" ;;
                5) ctx="262144"; cache_type="q4_0" ;;
                *) ctx="131072"; cache_type="q4_0" ;;
            esac

            echo ""
            echo " Select Launch Profile:"
            echo "   [1] Full: flash + jinja + thinking off if accepted"
            echo "   [2] Stable: flash + parallel, no jinja extras"
            echo "   [3] Raw safe: context + KV + host/port only"
            read -p " Choice (1-3, default 1): " launch_profile
            launch_profile=$(echo "$launch_profile" | tr -d '[:space:]')
            [[ -z "$launch_profile" ]] && launch_profile="1"

            mmproj_file=""
            vis_text=""

            mapfile -t mmproj_files < <(find "$MODELS_DIR" -maxdepth 1 -type f -iname '*mmproj*.gguf' 2>/dev/null | sort)

            if [[ ${#mmproj_files[@]} -gt 0 ]]; then
                read -p " Enable vision? (y/N): " load_vis
                load_vis=$(echo "$load_vis" | tr -d '[:space:]')

                if [[ "$load_vis" == "y" || "$load_vis" == "Y" ]]; then
                    if type auto_select_mmproj >/dev/null 2>&1; then
                        auto_select_mmproj "$target" "${mmproj_files[@]}"
                        if [[ -n "$SELECTED_MMPROJ" && "$SELECTED_MMPROJ_SCORE" -ge 100 ]]; then
                            mmproj_file="$SELECTED_MMPROJ"
                            echo " Auto-matched vision model: $(basename "$mmproj_file")"
                        fi
                    fi

                    if [[ -z "$mmproj_file" ]]; then
                        echo " Select Vision Projector:"
                        for i in "${!mmproj_files[@]}"; do
                            echo "   [$((i+1))] $(basename "${mmproj_files[$i]}")"
                        done
                        read -p " Choice: " vis_choice
                        vis_choice=$(echo "$vis_choice" | tr -d '[:space:]')
                        idx=$((vis_choice-1))
                        mmproj_file="${mmproj_files[$idx]:-${mmproj_files[0]}}"
                    fi

                    vis_text="+ Vis "
                fi
            fi

            draft_path=""
            draft_model=""
            draft_text=""
            draft_ctx="$ctx"
            draft_cache_type="$cache_type"
            draft_max="8"
            draft_min="0"
            draft_p_min="0.45"

            echo ""
            read -p " Enable draft/speculative decoding? (y/N): " enable_draft
            enable_draft=$(echo "$enable_draft" | tr -d '[:space:]')

            if [[ "$enable_draft" == "y" || "$enable_draft" == "Y" ]]; then
                mapfile -t all_draft_models < <(
                    find "$MODELS_DIR" -maxdepth 1 -type f -iname '*.gguf' ! -iname '*mmproj*.gguf' -printf '%f
' 2>/dev/null | sort
                )

                draft_candidates=()
                for d in "${all_draft_models[@]}"; do
                    [[ "$d" == "$target" ]] && continue
                    draft_candidates+=("$d")
                done

                if [[ ${#draft_candidates[@]} -eq 0 ]]; then
                    echo " No draft candidates found."
                else
                    echo ""
                    echo " Select Draft Model:"
                    for i in "${!draft_candidates[@]}"; do
                        d="${draft_candidates[$i]}"
                        d_low=$(echo "$d" | tr '[:upper:]' '[:lower:]')
                        tag=""
                        if [[ "$d_low" == *"dflash"* ]]; then tag="  <-- DFlash"; fi
                        if [[ "$d_low" == *"draft"* && -z "$tag" ]]; then tag="  <-- draft"; fi
                        echo "   [$((i+1))] $d$tag"
                    done

                    read -p " Choice: " draft_choice
                    draft_choice=$(echo "$draft_choice" | tr -d '[:space:]')

                    if [[ "$draft_choice" =~ ^[0-9]+$ ]]; then
                        draft_idx=$((draft_choice-1))
                        draft_model="${draft_candidates[$draft_idx]}"
                    else
                        draft_model="${draft_candidates[0]}"
                    fi

                    if [[ -n "$draft_model" ]]; then
                        draft_path="${MODELS_DIR}/${draft_model}"
                    fi
                fi

                if [[ -n "$draft_path" && -f "$draft_path" ]]; then
                    echo ""
                    echo " Selected draft: $(basename "$draft_path")"
                    echo ""
                    echo " Draft tuning:"
                    echo "   [1] Conservative: max 4,  p-min 0.60"
                    echo "   [2] Balanced:     max 8,  p-min 0.45"
                    echo "   [3] Aggressive:   max 12, p-min 0.35"
                    read -p " Choice (1-3, default 2): " draft_tune
                    draft_tune=$(echo "$draft_tune" | tr -d '[:space:]')

                    case "$draft_tune" in
                        1) draft_max="4";  draft_p_min="0.60" ;;
                        3) draft_max="12"; draft_p_min="0.35" ;;
                        *) draft_max="8";  draft_p_min="0.45" ;;
                    esac

                    echo ""
                    echo " Draft context:"
                    echo "   [1] Same as main: $ctx"
                    echo "   [2] 65536"
                    echo "   [3] 32768"
                    echo "   [4] 16384"
                    read -p " Choice (1-4, default 1): " draft_ctx_choice
                    draft_ctx_choice=$(echo "$draft_ctx_choice" | tr -d '[:space:]')

                    case "$draft_ctx_choice" in
                        2) draft_ctx="65536" ;;
                        3) draft_ctx="32768" ;;
                        4) draft_ctx="16384" ;;
                        *) draft_ctx="$ctx" ;;
                    esac

                    draft_text="+ Draft "
                fi
            fi

            probe_log=".llama_arg_probe_option4.log"
            dummy_model=".llama_arg_probe_dummy.gguf"
            : > "$dummy_model"

            arg_combo_valid() {
                local -a test_cmd
                test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18083)
                timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true

                if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
                    return 1
                fi
                return 0
            }

            flag_summary=()
            skipped_summary=()

            fa_flags=()
            cache_flags=()
            parallel_flags=()
            jinja_flags=()
            vision_flags=()
            draft_flags=()

            if arg_combo_valid -ctk "$cache_type" -ctv "$cache_type"; then
                cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
                flag_summary+=("main-kv:-ctk/-ctv ${cache_type}")
            else
                skipped_summary+=("main KV flags not accepted")
            fi

            if [[ "$launch_profile" == "1" || "$launch_profile" == "2" ]]; then
                if arg_combo_valid -fa on; then
                    fa_flags=(-fa on)
                    flag_summary+=("flash:-fa on")
                elif arg_combo_valid -fa; then
                    fa_flags=(-fa)
                    flag_summary+=("flash:-fa")
                else
                    skipped_summary+=("flash flag not accepted")
                fi

                if arg_combo_valid --parallel 1; then
                    parallel_flags=(--parallel 1)
                    flag_summary+=("parallel:--parallel 1")
                elif arg_combo_valid -np 1; then
                    parallel_flags=(-np 1)
                    flag_summary+=("parallel:-np 1")
                else
                    skipped_summary+=("parallel flag not accepted")
                fi
            fi

            if [[ "$launch_profile" == "1" ]]; then
                if arg_combo_valid --jinja; then
                    jinja_flags+=(--jinja)
                    flag_summary+=("chat:--jinja")
                fi

                if arg_combo_valid --chat-template-kwargs '{"enable_thinking":false}'; then
                    jinja_flags+=(--chat-template-kwargs '{"enable_thinking":false}')
                    flag_summary+=("chat:thinking=false")
                fi
            fi

            if [[ -n "$mmproj_file" ]]; then
                if arg_combo_valid --mmproj "$mmproj_file"; then
                    vision_flags=(--mmproj "$mmproj_file")
                    flag_summary+=("vision:--mmproj $(basename "$mmproj_file")")
                else
                    mmproj_file=""
                    vis_text=""
                    skipped_summary+=("--mmproj not accepted")
                fi
            fi

            if [[ -n "$draft_path" && -f "$draft_path" ]]; then
                if arg_combo_valid -md "$draft_path"; then
                    draft_flags+=(-md "$draft_path")
                    flag_summary+=("draft:-md $(basename "$draft_path")")

                    if arg_combo_valid -md "$draft_path" -ngld 999; then
                        draft_flags+=(-ngld 999)
                        flag_summary+=("draft:-ngld 999")
                    fi

                    if arg_combo_valid -md "$draft_path" -cd "$draft_ctx"; then
                        draft_flags+=(-cd "$draft_ctx")
                        flag_summary+=("draft:-cd ${draft_ctx}")
                    fi

                    if arg_combo_valid -md "$draft_path" -ctkd "$draft_cache_type" -ctvd "$draft_cache_type"; then
                        draft_flags+=(-ctkd "$draft_cache_type" -ctvd "$draft_cache_type")
                        flag_summary+=("draft-kv:-ctkd/-ctvd ${draft_cache_type}")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-max "$draft_max"; then
                        draft_flags+=(--draft-max "$draft_max")
                        flag_summary+=("draft:--draft-max ${draft_max}")
                    elif arg_combo_valid -md "$draft_path" --draft "$draft_max"; then
                        draft_flags+=(--draft "$draft_max")
                        flag_summary+=("draft:--draft ${draft_max}")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-min "$draft_min"; then
                        draft_flags+=(--draft-min "$draft_min")
                    fi

                    if arg_combo_valid -md "$draft_path" --draft-p-min "$draft_p_min"; then
                        draft_flags+=(--draft-p-min "$draft_p_min")
                    fi
                else
                    draft_flags=()
                    draft_text=""
                    skipped_summary+=("-md draft model not accepted")
                fi
            fi

            rm -f "$dummy_model" 2>/dev/null

            cmd=("$server_bin"
                -m "${MODELS_DIR}/${target}"
                -c "$ctx"
                -ngl 999
                "${fa_flags[@]}"
                "${cache_flags[@]}"
                "${jinja_flags[@]}"
                "${parallel_flags[@]}"
                --host 0.0.0.0
                --port 8080
                "${draft_flags[@]}"
                "${vision_flags[@]}"
            )

            target_short="$target"
            echo "LC: ${target_short} ${vis_text}${draft_text}[${ctx}/${cache_type}]" > .server_info

            echo ""
            echo " Starting OpenClaw:"
            echo "   Main:    $target"
            echo "   Context: $ctx"
            echo "   KV:      $cache_type"
            echo "   Vision:  ${vis_text:-off}"
            if [[ -n "$draft_text" ]]; then
                echo "   Draft:   $(basename "$draft_path")"
                echo "   Draft settings: ctx=${draft_ctx}, max=${draft_max}, min=${draft_min}, p-min=${draft_p_min}"
            else
                echo "   Draft:   off"
            fi

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

            {
                echo "COMMAND PROFILE: option4-real-simple-draft"
                printf '%q ' "${cmd[@]}"
                echo ""
                echo "----- llama-server output -----"
            } > server.log

            nohup "${cmd[@]}" >> server.log 2>&1 &
            server_pid=$!

            failed=0
            loaded=0

            for i in $(seq 1 120); do
                if ! kill -0 "$server_pid" >/dev/null 2>&1; then
                    failed=1
                    break
                fi

                if grep -Eqi 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' server.log; then
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

                echo " Last 35 lines of server.log:"
                echo "------------------------------------------------------------"
                tail -n 35 server.log
                echo "------------------------------------------------------------"
                sleep 3
                continue
            fi

            rm -f .server_info

            echo ""
            echo -e " [1;31mServer failed during startup.[0m"
            echo " Last 220 lines of server.log:"
            echo "------------------------------------------------------------"
            tail -n 220 server.log
            echo "------------------------------------------------------------"
            read -p " Press Enter to return to menu..."
            ;;
        5)
            echo ""
            echo -e " \033[1;36m>>> STOPPING SERVER <<<\033[0m"
            pkill -f "llama-server"
            rm -f .server_info
            echo " Server stopped."
            sleep 1
            ;;
        6)
            echo ""
            echo -e " [1;36m>>> PRACTICAL BENCHMARK (q4_0 KV SAFE BASELINE) <<<[0m"
            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Stopping active server to free VRAM for benchmark..."
                pkill -f "llama-server"
                rm -f .server_info
                sleep 2
            fi
            read -p " Select Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")
            if [[ -n "$target" ]]; then
                run_practical_q4_benchmark "$target"
                tput cnorm
                read -p " Press Enter to return to menu..."
            else
                echo " Invalid model number. Canceling."
                sleep 2
            fi
            ;;
        7)
            echo ""
            echo -e " [1;36m>>> KV QUALITY BENCHMARK (SELECTED MODEL) <<<[0m"
            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Stopping active server to free VRAM for benchmark..."
                pkill -f "llama-server"
                rm -f .server_info
                sleep 2
            fi
            read -p " Select Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")
            if [[ -n "$target" ]]; then
                run_selected_kv_quality_benchmark "$target"
                tput cnorm
                read -p " Press Enter to return to menu..."
            else
                echo " Invalid model number. Canceling."
                sleep 2
            fi
            ;;
        8)
            echo ""
            echo -e " \033[1;36m>>> DOWNLOAD MODEL <<<\033[0m"
            echo " Paste the direct download URL for a .gguf file."
            read -p " URL: " url
            url=$(echo "$url" | tr -d '[:space:]')
            if [[ -n "$url" ]]; then
                url=$(echo "$url" | sed 's|/blob/|/resolve/|')
                filename=$(basename "${url%%\?*}")
                echo " Downloading $filename..."
                wget --show-progress -O "${MODELS_DIR}/${filename}" "$url"
            fi
            ;;
        9)
            echo ""
            echo -e " \033[1;36m>>> DELETE MODEL <<<\033[0m"
            read -p " Select model NR to delete: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")
            if [[ -n "$target" ]]; then
                rm "${MODELS_DIR}/${target}"
                echo " Deleted $target"
                sleep 1
            else
                echo " Invalid model number. Canceling."
                sleep 2
            fi
            ;;
        10)
            echo ""
            echo -e " [1;36m>>> BENCHMARK ALL MODELS / PHASE ORDERED KV MATRIX <<<[0m"
            run_all_models_kv_matrix
            tput cnorm
            read -p " Press Enter to return to menu..."
            ;;
        11)
            echo ""
            echo -e " [1;36m>>> START COWORK SERVER (ANTHROPIC / CLAUDE-COMPATIBLE API) <<<[0m"

            if [[ -n $(pgrep -f "llama-server") ]]; then
                echo " Server is already running! Please stop it first [5]."
                sleep 2
                continue
            fi

            read -p " Select MAIN Model NR: " n
            n=$(echo "$n" | tr -d '[:space:]')
            target=$(get_model_name "$n")

            if [[ -z "$target" ]]; then
                echo " Invalid model number. Canceling."
                sleep 2
                continue
            fi

            server_bin="./ik_llama.cpp/build/bin/llama-server"
            if [[ ! -x "$server_bin" ]]; then
                echo " Error: llama-server not found at $server_bin"
                echo " Run Install/Update [0] first."
                read -p " Press Enter to return to menu..."
                continue
            fi

            echo ""
            echo " Select Context Size / KV Cache:"
            echo "   [1] 32768  (32K, q8_0 KV - safer for coding tools)"
            echo "   [2] 65536  (64K, q8_0 KV)"
            echo "   [3] 131072 (128K, q4_0 KV - recommended long context)"
            echo "   [4] 262144 (256K, q4_0 KV - high memory pressure)"
            read -p " Choice (1-4, default 3): " ctx_choice
            ctx_choice=$(echo "$ctx_choice" | tr -d '[:space:]')

            case "$ctx_choice" in
                1) ctx="32768";  cache_type="q8_0" ;;
                2) ctx="65536";  cache_type="q8_0" ;;
                4) ctx="262144"; cache_type="q4_0" ;;
                *) ctx="131072"; cache_type="q4_0" ;;
            esac

            echo ""
            read -p " Model alias for Cowork/Claude client [claude-local]: " model_alias
            model_alias=$(echo "$model_alias" | tr -d '[:space:]')
            [[ -z "$model_alias" ]] && model_alias="claude-local"

            echo ""
            echo " Thinking mode:"
            echo "   [1] Disable thinking in template if supported"
            echo "   [2] Leave thinking controlled by client/model"
            echo "   [3] Enable thinking in template if supported"
            read -p " Choice (1-3, default 1): " thinking_choice
            thinking_choice=$(echo "$thinking_choice" | tr -d '[:space:]')
            [[ -z "$thinking_choice" ]] && thinking_choice="1"

            case "$thinking_choice" in
                3) thinking_json='{"enable_thinking":true}'; thinking_text="ThinkOn" ;;
                2) thinking_json=""; thinking_text="ClientDefault" ;;
                *) thinking_json='{"enable_thinking":false}'; thinking_text="ThinkOff" ;;
            esac

            echo ""
            echo " Select Launch Profile:"
            echo "   [1] Cowork full: flash + jinja + alias + parallel"
            echo "   [2] Cowork stable: jinja + alias + parallel, no flash"
            echo "   [3] Raw safe: context + KV + alias only"
            read -p " Choice (1-3, default 1): " launch_profile
            launch_profile=$(echo "$launch_profile" | tr -d '[:space:]')
            [[ -z "$launch_profile" ]] && launch_profile="1"

            probe_log=".llama_arg_probe_cowork.log"
            dummy_model=".llama_arg_probe_dummy.gguf"
            : > "$dummy_model"

            arg_combo_valid() {
                local -a test_cmd
                test_cmd=("$server_bin" -m "$dummy_model" -c 16 -ngl 0 "$@" --host 127.0.0.1 --port 18084)
                timeout 8 "${test_cmd[@]}" > "$probe_log" 2>&1 || true

                if grep -Eiq 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' "$probe_log"; then
                    return 1
                fi
                return 0
            }

            flag_summary=()
            skipped_summary=()

            fa_flags=()
            cache_flags=()
            parallel_flags=()
            jinja_flags=()
            alias_flags=()
            sampling_flags=()

            if arg_combo_valid -ctk "$cache_type" -ctv "$cache_type"; then
                cache_flags=(-ctk "$cache_type" -ctv "$cache_type")
                flag_summary+=("kv:-ctk/-ctv ${cache_type}")
            else
                skipped_summary+=("KV cache flags not accepted")
            fi

            if [[ "$launch_profile" == "1" ]]; then
                if arg_combo_valid -fa on; then
                    fa_flags=(-fa on)
                    flag_summary+=("flash:-fa on")
                elif arg_combo_valid -fa; then
                    fa_flags=(-fa)
                    flag_summary+=("flash:-fa")
                else
                    skipped_summary+=("flash flag not accepted")
                fi
            fi

            if [[ "$launch_profile" == "1" || "$launch_profile" == "2" ]]; then
                if arg_combo_valid --parallel 1; then
                    parallel_flags=(--parallel 1)
                    flag_summary+=("parallel:--parallel 1")
                elif arg_combo_valid -np 1; then
                    parallel_flags=(-np 1)
                    flag_summary+=("parallel:-np 1")
                else
                    skipped_summary+=("parallel flag not accepted")
                fi
            fi

            if [[ "$launch_profile" != "3" ]]; then
                if arg_combo_valid --jinja; then
                    jinja_flags+=(--jinja)
                    flag_summary+=("chat:--jinja")
                else
                    skipped_summary+=("--jinja not accepted")
                fi

                if [[ -n "$thinking_json" ]]; then
                    if arg_combo_valid --chat-template-kwargs "$thinking_json"; then
                        jinja_flags+=(--chat-template-kwargs "$thinking_json")
                        flag_summary+=("thinking:${thinking_json}")
                    else
                        skipped_summary+=("--chat-template-kwargs not accepted as startup flag")
                    fi
                fi
            fi

            if arg_combo_valid --alias "$model_alias"; then
                alias_flags=(--alias "$model_alias")
                flag_summary+=("alias:${model_alias}")
            else
                skipped_summary+=("--alias not accepted")
            fi

            # Coding-agent friendly sampling, only if the binary accepts these flags.
            if arg_combo_valid --temp 0.2; then
                sampling_flags+=(--temp 0.2)
                flag_summary+=("sampling:temp=0.2")
            fi
            if arg_combo_valid --top-p 0.95; then
                sampling_flags+=(--top-p 0.95)
                flag_summary+=("sampling:top-p=0.95")
            fi

            rm -f "$dummy_model" 2>/dev/null

            cmd=("$server_bin"
                -m "${MODELS_DIR}/${target}"
                -c "$ctx"
                -ngl 999
                "${fa_flags[@]}"
                "${cache_flags[@]}"
                "${jinja_flags[@]}"
                "${parallel_flags[@]}"
                "${alias_flags[@]}"
                "${sampling_flags[@]}"
                --host 0.0.0.0
                --port 8080
            )

            target_short="$target"
            echo "COWORK: ${target_short} [${ctx}/${cache_type}/${thinking_text}/alias=${model_alias}]" > .server_info

            cat > cowork_env.sh <<EOF
# Source this before starting Claude Cowork / Claude-compatible client.
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080/v1"
export ANTHROPIC_API_KEY="local-dummy-key"
export ANTHROPIC_MODEL="${model_alias}"

# Some Claude-compatible clients use config instead of env.
# In Cowork config, set:
# wire_api = "responses"
EOF

            echo ""
            echo " Starting Cowork-compatible llama-server:"
            echo "   Main model:  $target"
            echo "   Context:     $ctx"
            echo "   KV cache:    $cache_type"
            echo "   Alias:       $model_alias"
            echo "   Endpoint:    http://127.0.0.1:8080/v1/messages"
            echo "   Env file:    ./cowork_env.sh"
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

            {
                echo "COMMAND PROFILE: cowork-anthropic-api"
                printf '%q ' "${cmd[@]}"
                echo ""
                echo "----- llama-server output -----"
            } > server.log

            nohup "${cmd[@]}" >> server.log 2>&1 &
            server_pid=$!

            failed=0
            loaded=0

            for i in $(seq 1 120); do
                if ! kill -0 "$server_pid" >/dev/null 2>&1; then
                    failed=1
                    break
                fi

                if grep -Eqi 'unknown argument|unrecognized option|invalid option|invalid argument|error:.*argument|usage:' server.log; then
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
                echo " Cowork client setup:"
                echo "   source ./cowork_env.sh"
                echo "   ANTHROPIC_BASE_URL=http://127.0.0.1:8080/v1"
                echo "   model alias: ${model_alias}"
                echo ""
                echo " Quick Anthropic Messages API test:"
                echo "   curl http://127.0.0.1:8080/v1/messages -H 'Content-Type: application/json' -d '{\"model\":\"${model_alias}\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}'"
                echo ""
                echo " Last 35 lines of server.log:"
                echo "------------------------------------------------------------"
                tail -n 35 server.log
                echo "------------------------------------------------------------"
                sleep 3
                continue
            fi

            rm -f .server_info

            echo ""
            echo -e " [1;31mCowork server failed during startup.[0m"
            echo " Last 220 lines of server.log:"
            echo "------------------------------------------------------------"
            tail -n 220 server.log
            echo "------------------------------------------------------------"
            read -p " Press Enter to return to menu..."
            ;;
        99) cleanup ;;
        *) ;;
    esac
done
