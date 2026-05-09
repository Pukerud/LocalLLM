#!/bin/bash
# =========================================================================
# BeeLlama DFlash Benchmark — all target × draft combos
# =========================================================================

CTX=100000
PORT=8081
PROMPT="Write a Python function that finds the longest increasing subsequence in a list of integers. Include proper type hints, docstring, and a few test cases with assertions."

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

SERVER_BIN="./beellama-cpp/build/bin/llama-server"
MODELS_DIR="llama_models"
LOG_DIR="bench_logs"
RESULTS_FILE="bench_results.txt"

TARGETS=(
    "Qwen3.6-27B-Q4_K_M.gguf"
    "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q5_K_M.gguf"
)
DRAFTS=(
    "Qwen3.6-27B-DFlash-IQ4_XS.gguf"
    "Qwen3.6-27B-DFlash-Q5_K_M.gguf"
    "Qwen3.6-27B-DFlash-Q6_K.gguf"
    "Qwen3.6-27B-DFlash-Q8_0.gguf"
)

mkdir -p "$LOG_DIR"
: > "$RESULTS_FILE"

kill_server() {
    pkill -f "llama-server" 2>/dev/null
    sleep 2
    pkill -9 -f "llama-server" 2>/dev/null
    sleep 1
}

run_bench() {
    local target="$1"
    local draft="$2"
    local label="$3"
    local log_file="$4"

    echo ""
    echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
    echo -e " ${BOLD}  BENCHMARK: ${label}${RESET}"
    echo -e " ${BOLD}  Target: ${target}${RESET}"
    echo -e " ${BOLD}  Draft:  ${draft}${RESET}"
    echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"

    # Check files exist
    if [[ ! -f "${MODELS_DIR}/${target}" ]]; then
        echo -e " ${RED}Target not found: ${MODELS_DIR}/${target}${RESET}"
        echo "SKIP: ${label} — target missing" >> "$RESULTS_FILE"
        return 1
    fi
    if [[ ! -f "${MODELS_DIR}/${draft}" ]]; then
        echo -e " ${RED}Draft not found: ${MODELS_DIR}/${draft}${RESET}"
        echo "SKIP: ${label} — draft missing" >> "$RESULTS_FILE"
        return 1
    fi

    kill_server

    echo " Starting server..."
    local cmd=(
        "$SERVER_BIN"
        -m "${MODELS_DIR}/${target}"
        --spec-draft-model "${MODELS_DIR}/${draft}"
        --spec-type dflash
        --spec-dflash-cross-ctx 1024
        -np 1 --kv-unified
        -ngl all --spec-draft-ngl all
        -b 2048 -ub 256
        --ctx-size "$CTX"
        --cache-type-k turbo3_tcq
        --cache-type-v turbo3_tcq
        --flash-attn on
        --cache-ram 0 --jinja
        --no-mmap --mlock
        --no-host --metrics
        --log-timestamps --log-prefix --log-colors off
        --reasoning on
        --temp 0.6 --top-k 20 --min-p 0.0
        --host 127.0.0.1 --port "$PORT"
    )

    : > "$log_file"
    nohup "${cmd[@]}" >> "$log_file" 2>&1 &
    local srv_pid=$!

    # Wait for server to be ready
    local ready=0
    for i in $(seq 1 120); do
        if ! kill -0 "$srv_pid" 2>/dev/null; then
            echo -e " ${RED}Server crashed during startup.${RESET}"
            tail -20 "$log_file"
            echo "FAIL: ${label} — server crashed" >> "$RESULTS_FILE"
            return 1
        fi
        local code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
        if [[ "$code" == "200" ]]; then
            ready=1
            break
        fi
        sleep 1
    done

    if [[ "$ready" == "0" ]]; then
        echo -e " ${RED}Server did not become ready in 120s.${RESET}"
        kill_server
        echo "FAIL: ${label} — timeout" >> "$RESULTS_FILE"
        return 1
    fi

    echo -e " ${GREEN}Server ready (PID: $srv_pid)${RESET}"

    # Extract server startup info
    local dflash_info=$(grep -E 'dflash: GPU cross ring|dflash: block_size' "$log_file" | tail -2)
    echo " DFlash: $dflash_info"

    # Send benchmark prompt
    echo " Sending benchmark prompt..."
    local t_start=$(date +%s%N)

    local response=$(curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
            \"max_tokens\": 1024,
            \"temperature\": 0.6,
            \"top_k\": 20
        }" 2>/dev/null)

    local t_end=$(date +%s%N)
    local elapsed_ms=$(( (t_end - t_start) / 1000000 ))

    if [[ -z "$response" ]]; then
        echo -e " ${RED}No response from server.${RESET}"
        kill_server
        echo "FAIL: ${label} — no response" >> "$RESULTS_FILE"
        return 1
    fi

    # Extract metrics from response
    local usage=$(echo "$response" | jq -r '.usage // empty' 2>/dev/null)
    local prompt_tokens=$(echo "$usage" | jq -r '.prompt_tokens // 0' 2>/dev/null)
    local completion_tokens=$(echo "$usage" | jq -r '.completion_tokens // 0' 2>/dev/null)
    local total_tokens=$(echo "$usage" | jq -r '.total_tokens // 0' 2>/dev/null)

    # Calculate tok/s
    local tok_per_sec="0"
    if [[ "$elapsed_ms" -gt 0 && "$completion_tokens" -gt 0 ]]; then
        tok_per_sec=$(awk "BEGIN {printf \"%.1f\", ($completion_tokens / $elapsed_ms) * 1000}")
    fi

    # Extract speculative decoding stats from server log
    # Get spec cycles that occurred during our request
    # Extract DFlash acceptance rate from slot result line
    local accept_line=$(grep 'draft acceptance rate' "$log_file" | tail -1)
    local accept_rate=$(echo "$accept_line" | grep -oP 'rate = \K[0-9.]+')
    local accept_detail=$(echo "$accept_line" | grep -oP '\(.*\)')
    local n_accepted=$(echo "$accept_detail" | grep -oP '\d+(?= accepted)')
    local n_generated=$(echo "$accept_detail" | grep -oP '\d+(?= generated)')
    [[ -z "$accept_rate" ]] && accept_rate="N/A"
    [[ -z "$n_accepted" ]] && n_accepted="?"
    [[ -z "$n_generated" ]] && n_generated="?"

    # Extract prompt eval speed
    local prompt_eval_line=$(grep 'prompt eval time' "$log_file" | tail -1)
    local prompt_tps=$(echo "$prompt_eval_line" | grep -oP '[0-9.]+(?= tokens per second)' | tail -1)
    [[ -z "$prompt_tps" ]] && prompt_tps="?"

    # Extract generation speed from server log
    local eval_line=$(grep 'eval time' "$log_file" | grep -v 'prompt' | tail -1)
    local gen_tps=$(echo "$eval_line" | grep -oP '[0-9.]+(?= tokens per second)' | tail -1)
    [[ -z "$gen_tps" ]] && gen_tps="?"

    local spec_lines=$(grep 'spec cycle' "$log_file" | tail -50)
    local n_cycles=$(echo "$spec_lines" | wc -l)

    local avg_draft_ms="0"
    local avg_verify_ms="0"
    local avg_accept_ms="0"
    local avg_total_ms="0"

    if [[ "$n_cycles" -gt 0 ]]; then
        local sum_draft=0 sum_verify=0 sum_accept=0 sum_total=0
        while IFS= read -r line; do
            d=$(echo "$line" | grep -oP 'draft=\K[0-9.]+')
            v=$(echo "$line" | grep -oP 'verify=\K[0-9.]+')
            a=$(echo "$line" | grep -oP 'accept=\K[0-9.]+')
            t=$(echo "$line" | grep -oP 'total=\K[0-9.]+')
            sum_draft=$(awk "BEGIN {printf \"%.1f\", $sum_draft + $d}")
            sum_verify=$(awk "BEGIN {printf \"%.1f\", $sum_verify + $v}")
            sum_accept=$(awk "BEGIN {printf \"%.1f\", $sum_accept + $a}")
            sum_total=$(awk "BEGIN {printf \"%.1f\", $sum_total + $t}")
        done <<< "$spec_lines"

        avg_draft_ms=$(awk "BEGIN {printf \"%.1f\", $sum_draft / $n_cycles}")
        avg_verify_ms=$(awk "BEGIN {printf \"%.1f\", $sum_verify / $n_cycles}")
        avg_accept_ms=$(awk "BEGIN {printf \"%.1f\", $sum_accept / $n_cycles}")
        avg_total_ms=$(awk "BEGIN {printf \"%.1f\", $sum_total / $n_cycles}")
    fi

    # Get the content (strip think blocks for display)
    local content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null | head -20)

    # Print results
    echo ""
    echo -e " ${BOLD}─── RESULTS ───${RESET}"
    echo "  Prompt tokens:     $prompt_tokens"
    echo "  Completion tokens: $completion_tokens"
    echo "  Total tokens:      $total_tokens"
    echo "  Wall time:         ${elapsed_ms}ms"
    echo -e "  Speed:             ${BOLD}${GREEN}${tok_per_sec} tok/s${RESET}"
    echo ""
    echo "  Speed (wall):      ${BOLD}${GREEN}${tok_per_sec} tok/s${RESET}"
    echo "  Speed (server):    ${BOLD}${gen_tps} tok/s${RESET}"
    echo "  Prompt speed:      ${prompt_tps} tok/s"
    echo ""
    echo "  Acceptance rate:   ${BOLD}${YELLOW}${accept_rate}${RESET} (${n_accepted} accepted / ${n_generated} generated)"
    echo "  Spec cycles:       $n_cycles"
    echo "  Avg draft:         ${avg_draft_ms}ms"
    echo "  Avg verify:        ${avg_verify_ms}ms"
    echo "  Avg accept:        ${avg_accept_ms}ms"
    echo "  Avg total cycle:   ${avg_total_ms}ms"
    echo ""
    echo "  First 20 lines of output:"
    echo "$content" | sed 's/^/    /'
    echo ""

    # Save to results file
    echo "╔══════════════════════════════════════════════════════════╗" >> "$RESULTS_FILE"
    echo "║ ${label}" >> "$RESULTS_FILE"
    echo "║ Target: ${target}" >> "$RESULTS_FILE"
    echo "║ Draft:  ${draft}" >> "$RESULTS_FILE"
    echo "║ Ctx:    ${CTX} | K: turbo3_tcq | V: turbo3_tcq" >> "$RESULTS_FILE"
    echo "║" >> "$RESULTS_FILE"
    echo "║ Prompt tokens:     $prompt_tokens" >> "$RESULTS_FILE"
    echo "║ Completion tokens: $completion_tokens" >> "$RESULTS_FILE"
    echo "║ Wall time:         ${elapsed_ms}ms" >> "$RESULTS_FILE"
    echo "║ Speed (wall):      ${tok_per_sec} tok/s" >> "$RESULTS_FILE"
    echo "║ Speed (server):    ${gen_tps} tok/s" >> "$RESULTS_FILE"
    echo "║ Prompt speed:      ${prompt_tps} tok/s" >> "$RESULTS_FILE"
    echo "║" >> "$RESULTS_FILE"
    echo "║ Acceptance rate:   ${accept_rate} (${n_accepted}/${n_generated})" >> "$RESULTS_FILE"
    echo "║ Spec cycles:       $n_cycles" >> "$RESULTS_FILE"
    echo "║ Avg draft:         ${avg_draft_ms}ms" >> "$RESULTS_FILE"
    echo "║ Avg verify:        ${avg_verify_ms}ms" >> "$RESULTS_FILE"
    echo "║ Avg accept:        ${avg_accept_ms}ms" >> "$RESULTS_FILE"
    echo "║ Avg total cycle:   ${avg_total_ms}ms" >> "$RESULTS_FILE"
    echo "╚══════════════════════════════════════════════════════════╝" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    kill_server
    echo " Server stopped. Cooling down 5s..."
    sleep 5
}

# ── Main ─────────────────────────────────────────────────────────────────

echo ""
echo -e " ${BOLD}${CYAN}BeeLlama DFlash Benchmark Suite${RESET}"
echo -e " ${BOLD}Targets: ${#TARGETS[@]} | Drafts: ${#DRAFTS[@]} | Combos: $((${#TARGETS[@]} * ${#DRAFTS[@]}))${RESET}"
echo -e " Context: ${CTX} | Port: ${PORT}"
echo ""

# Check deps
if ! command -v jq &>/dev/null; then
    echo "Installing jq..."
    sudo apt install -y jq
fi

if [[ ! -x "$SERVER_BIN" ]]; then
    echo -e " ${RED}Server binary not found: $SERVER_BIN${RESET}"
    exit 1
fi

for target in "${TARGETS[@]}"; do
    for draft in "${DRAFTS[@]}"; do
        t_short=$(echo "$target" | sed 's/Qwen3.6-27B-//;s/\.gguf//')
        d_short=$(echo "$draft" | sed 's/Qwen3.6-27B-DFlash-//;s/\.gguf//')
        label="Target=${t_short} | Draft=${d_short}"
        log_file="${LOG_DIR}/bench_${t_short}__${d_short}.log"

        run_bench "$target" "$draft" "$label" "$log_file"
    done
done

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}${GREEN}  BENCHMARK COMPLETE — ALL RESULTS${RESET}"
echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo ""
cat "$RESULTS_FILE"

echo ""
echo " Full server logs saved in: ${LOG_DIR}/"
echo " Summary saved to: ${RESULTS_FILE}"
