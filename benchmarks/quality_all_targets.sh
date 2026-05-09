#!/bin/bash
# Run from repo root: cd /path/to/LocalLLM && bash benchmarks/quality_all_targets.sh

# Resolve repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../HostLLM.sh" ]]; then
    cd "${SCRIPT_DIR}/.."
fi

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

SERVER_BIN="./beellama-cpp/build/bin/llama-server"
DRAFT="Qwen3.6-27B-DFlash-IQ4_XS.gguf"
PORT=8081
CTX=100000

TARGETS=(
    "Qwen3.6-27B-Q4_K_M.gguf"
    "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q5_K_M.gguf"
    "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-IQ4_XS.gguf"
)

SCORES=()
LABELS=()

echo ""
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}  Quality Benchmark — All Targets (beellama DFlash)${RESET}"
echo -e " ${BOLD}  Draft: ${DRAFT} (quality-independent)${RESET}"
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo ""

for target in "${TARGETS[@]}"; do
    t_short=$(echo "$target" | sed 's/Qwen3.6-27B-//;s/\.gguf//')
    echo -e " ${BOLD}${YELLOW}═══ ${t_short} ═══${RESET}"

    pkill -f llama-server 2>/dev/null; sleep 2; pkill -9 -f llama-server 2>/dev/null; sleep 1

    nohup "$SERVER_BIN" \
        -m "llama_models/${target}" \
        --spec-draft-model "llama_models/${DRAFT}" \
        --spec-type dflash --spec-dflash-cross-ctx 1024 \
        -np 1 --kv-unified -ngl all --spec-draft-ngl all \
        -b 2048 -ub 256 --ctx-size $CTX \
        --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq \
        --flash-attn on --cache-ram 0 --jinja \
        --no-mmap --mlock --no-host --metrics \
        --log-timestamps --log-prefix --log-colors off \
        --reasoning on --temp 0.6 --top-k 20 --min-p 0.0 \
        --host 127.0.0.1 --port $PORT \
        > /tmp/beellama_quality.log 2>&1 &

    echo " Waiting for server..."
    ready=0
    for i in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
        [[ "$code" == "200" ]] && ready=1 && break
        sleep 1
    done

    if [[ "$ready" == "0" ]]; then
        echo -e " ${RED}Server failed!${RESET}"
        LABELS+=("$t_short"); SCORES+=("FAIL")
        echo ""; continue
    fi

    echo -e " ${GREEN}Ready${RESET} — running quality benchmark..."
    echo ""

    output=$(bash bench_quality.sh $PORT 127.0.0.1 2>&1)
    echo "$output"
    echo ""

    score=$(echo "$output" | grep 'TOTAL:' | head -1 | grep -oP '\d+(?=/40)')
    [[ -z "$score" ]] && score="?"

    LABELS+=("$t_short")
    SCORES+=("$score")

    pkill -f llama-server 2>/dev/null; sleep 3
done

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}${GREEN}  QUALITY BENCHMARK SUMMARY${RESET}"
echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Draft: ${DRAFT} (quality-independent)"
echo "  Context: ${CTX} | KV: turbo3_tcq | Reasoning: ON"
echo ""
printf "  %-45s %s\n" "Model" "Score"
echo "  -------------------------------------------------------"
for i in "${!LABELS[@]}"; do
    printf "  %-45s %s/40\n" "${LABELS[$i]}" "${SCORES[$i]}"
done
echo ""
