#!/bin/bash
# =========================================================================
# Quality Benchmark — Automated Model IQ Test v2
#
# 4 prompt classes, fully auto-scored, HARD prompts:
#   1. Code:        merge intervals with 10 nasty edge-case assertions
#   2. Reasoning:   Einstein logic puzzle (5 houses, 15 clues)
#   3. Instruction: 8 constraints including negatives + hidden gotcha
#   4. Knowledge:   CAP theorem with 12 specific technical concepts
#
# Total: 40 points (10 per class)
# Draft models do NOT affect quality — only the target model matters.
# =========================================================================

GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
RED=$(tput setaf 1); BOLD=$(tput bold); RESET=$(tput sgr0)

PORT=${1:-8080}
HOST=${2:-127.0.0.1}
URL="http://${HOST}:${PORT}/v1/chat/completions"

RESULTS_DIR="bench_quality"
mkdir -p "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────

send_prompt() {
    local prompt="$1"
    local max_tokens="${2:-2048}"
    curl -s --max-time 300 "$URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | jq -Rs .)}],
            \"max_tokens\": $max_tokens,
            \"temperature\": 0.6,
            \"top_k\": 20
        }" 2>/dev/null
}

get_content() {
    echo "$1" | jq -r '.choices[0].message.content // empty' 2>/dev/null
}

# ── TEST 1: Code — merge intervals with edge cases (10 pts) ──────────────

test_code() {
    local label="$1"
    local content="$2"
    echo -ne "  [Code]          "

    local code=$(echo "$content" | sed -n '/^```python/,/^```/p' | sed '1d;$d')
    [[ -z "$code" ]] && code=$(echo "$content" | sed -n '/^def \|^    /p')
    if [[ -z "$code" ]]; then
        echo -e "${RED}0/10${RESET}  (no code found)"
        echo "0" > /tmp/_bench_score; return
    fi

    cat > "${RESULTS_DIR}/${label}_code.py" << 'PYEOF'
import sys
PYEOF
    echo "$code" >> "${RESULTS_DIR}/${label}_code.py"
    cat >> "${RESULTS_DIR}/${label}_code.py" << 'PYEOF'

# --- Auto-generated assertions ---
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]], "basic overlap"
assert merge_intervals([[1,4]]) == [[1,4]], "single interval"
assert merge_intervals([]) == [], "empty input"
assert merge_intervals([[1,2],[3,4],[5,6]]) == [[1,2],[3,4],[5,6]], "no overlaps"
assert merge_intervals([[1,4],[4,5]]) == [[1,5]], "touching intervals"
assert merge_intervals([[1,10],[2,5],[3,4]]) == [[1,10]], "nested intervals"
assert merge_intervals([[5,8],[1,3],[2,6]]) == [[1,8]], "unsorted input"
assert merge_intervals([[1,4],[1,4]]) == [[1,4]], "duplicates"
assert merge_intervals([[1,4],[2,5],[3,6],[5,8]]) == [[1,8]], "all merge into one"
assert merge_intervals([[6,8],[3,5],[1,2]]) == [[1,2],[3,5],[6,8]], "reverse sorted"
print("ALL_ASSERTIONS_PASSED")
PYEOF

    local output=$(python3 "${RESULTS_DIR}/${label}_code.py" 2>&1)

    if echo "$output" | grep -q "ALL_ASSERTIONS_PASSED"; then
        echo -e "${GREEN}10/10${RESET}  (all 10 assertions passed)"
        echo "10" > /tmp/_bench_score; return
    fi

    # Count individual passes
    local test_cases=(
        'merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]'
        'merge_intervals([[1,4]])==[[1,4]]'
        'merge_intervals([])==[]'
        'merge_intervals([[1,2],[3,4],[5,6]])==[[1,2],[3,4],[5,6]]'
        'merge_intervals([[1,4],[4,5]])==[[1,5]]'
        'merge_intervals([[1,10],[2,5],[3,4]])==[[1,10]]'
        'merge_intervals([[5,8],[1,3],[2,6]])==[[1,8]]'
        'merge_intervals([[1,4],[1,4]])==[[1,4]]'
        'merge_intervals([[1,4],[2,5],[3,6],[5,8]])==[[1,8]]'
        'merge_intervals([[6,8],[3,5],[1,2]])==[[1,2],[3,5],[6,8]]'
    )
    local passed=0
    for tc in "${test_cases[@]}"; do
        echo -e "import sys\n${code}\nassert ${tc}" > "${RESULTS_DIR}/${label}_single.py"
        python3 "${RESULTS_DIR}/${label}_single.py" &>/dev/null && passed=$((passed + 1))
    done
    local score=$(( (passed * 10) / 10 ))
    if [[ $score -eq 0 ]]; then
        echo -e "${RED}0/10${RESET}  (code error: $(echo "$output" | tail -1 | cut -c1-60))"
    else
        echo -e "${YELLOW}${score}/10${RESET}  (${passed}/10 assertions passed)"
    fi
    echo "$score" > /tmp/_bench_score
}

# ── TEST 2: Reasoning — Einstein puzzle (10 pts) ─────────────────────────

test_reasoning() {
    local label="$1"
    local content="$2"
    echo -ne "  [Reasoning]     "

    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    local correct_claims=0
    local method_score=0

    # Check 7 specific correct facts from the solution (4 pts)
    echo "$c_lower" | grep -qE 'green.*(coffee|drink)|coffee.*green' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'british.*(red)|red.*(british)' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'norwegian.*(first|leftmost|house 1|house #1|1st)' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'white.*(right|next|after).*green|green.*(left|before).*white' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'yellow.*(dunhill)|dunhill.*(yellow)' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'pall.?mall.*(bird|birds)|bird.*(pall.?mall)' && correct_claims=$((correct_claims + 1))
    echo "$c_lower" | grep -qE 'german.*(prince)|prince.*(german)' && correct_claims=$((correct_claims + 1))

    local answer_score=0
    [[ $correct_claims -ge 5 ]] && answer_score=4
    [[ $correct_claims -eq 4 ]] && answer_score=3
    [[ $correct_claims -ge 2 ]] && [[ $answer_score -eq 0 ]] && answer_score=2
    [[ $correct_claims -ge 1 ]] && [[ $answer_score -eq 0 ]] && answer_score=1

    # Method: organized reasoning (6 pts)
    if echo "$c_lower" | grep -qE 'house.*(1|2|3|4|5)|first|second|third|fourth|fifth'; then method_score=$((method_score + 1)); fi
    if echo "$c_lower" | grep -qE 'therefore|so |must |cannot |only |eliminat|deduc'; then method_score=$((method_score + 1)); fi
    if echo "$c_lower" | grep -qE 'norwegian|british|german|swedish|danish'; then method_score=$((method_score + 1)); fi
    if echo "$content" | grep -qE '\|.*\|.*\|' || echo "$content" | grep -qE '\-\-\-.*\-\-\-.*\-\-\-'; then method_score=$((method_score + 2)); fi
    if echo "$c_lower" | grep -qE 'clue|hint|constraint|given|from'; then method_score=$((method_score + 1)); fi
    [[ $method_score -gt 6 ]] && method_score=6

    local total=$((answer_score + method_score))
    [[ $total -gt 10 ]] && total=10

    if [[ $total -ge 8 ]]; then echo -e "${GREEN}${total}/10${RESET}  (facts=${answer_score}/4, method=${method_score}/6)"
    elif [[ $total -ge 5 ]]; then echo -e "${YELLOW}${total}/10${RESET}  (facts=${answer_score}/4, method=${method_score}/6)"
    else echo -e "${RED}${total}/10${RESET}  (facts=${answer_score}/4, method=${method_score}/6)"
    fi
    echo "$total" > /tmp/_bench_score
}

# ── TEST 3: Instruction — 8 constraints with negatives + gotcha (10 pts) ─

test_instruction() {
    local label="$1"
    local content="$2"
    echo -ne "  [Instruction]   "

    local met=0
    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')

    # 1. Exactly 5 paragraphs
    local para_count=$(echo "$content" | awk 'BEGIN{p=0} NF{if(!in_p){p++; in_p=1} next} /^$/{in_p=0} END{print p+0}')
    [[ "$para_count" -ge 4 && "$para_count" -le 6 ]] && met=$((met + 1))

    # 2. Each paragraph starts with a different letter
    local first_letters=$(echo "$content" | awk 'NF && !done{print toupper(substr($0,1,1)); done=1} /^$/{done=0}' | sort -u | wc -l)
    [[ "$first_letters" -ge 4 ]] && met=$((met + 1))

    # 3. Exact phrase "quantum entanglement"
    echo "$c_lower" | grep -q 'quantum entanglement' && met=$((met + 1))

    # 4. No "basically" (negative constraint)
    echo "$c_lower" | grep -qw 'basically' || met=$((met + 1))

    # 5. No "literally" (negative constraint)
    echo "$c_lower" | grep -qw 'literally' || met=$((met + 1))

    # 6. "serendipity" appears at least twice
    local ser_count=$(echo "$c_lower" | grep -o 'serendipity' | wc -l)
    [[ "$ser_count" -ge 2 ]] && met=$((met + 1))

    # 7. Year 1927 mentioned
    echo "$content" | grep -q '1927' && met=$((met + 1))

    # 8. Hidden gotcha: 3rd word of response must be "the"
    local third_word=$(echo "$content" | head -1 | sed 's/^[[:space:]]*//' | tr -s ' ' | cut -d' ' -f3 | tr -d '.,;:!?"'"'" | tr '[:upper:]' '[:lower:]')
    [[ "$third_word" == "the" ]] && met=$((met + 1))

    local score=$(( (met * 10) / 8 ))
    if [[ $score -ge 8 ]]; then echo -e "${GREEN}${score}/10${RESET}  (${met}/8 constraints met)"
    elif [[ $score -ge 5 ]]; then echo -e "${YELLOW}${score}/10${RESET}  (${met}/8 constraints met)"
    else echo -e "${RED}${score}/10${RESET}  (${met}/8 constraints met)"
    fi
    echo "$score" > /tmp/_bench_score
}

# ── TEST 4: Knowledge — CAP theorem with 12 concepts (10 pts) ────────────

test_knowledge() {
    local label="$1"
    local content="$2"
    echo -ne "  [Knowledge]     "

    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    local found=0

    echo "$c_lower" | grep -qE 'consistency|consistent'         && found=$((found + 1))
    echo "$c_lower" | grep -qE 'availability|available'         && found=$((found + 1))
    echo "$c_lower" | grep -qE 'partition.?tolerance|partition.?tolerant|network.?partition|partition.?fault' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'choose.*two|pick.*two|select.*two|only.*two.*three|two.*of.*three|at most two' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'brewer'                         && found=$((found + 1))
    echo "$c_lower" | grep -qE 'trade.?off'                     && found=$((found + 1))
    echo "$c_lower" | grep -qE 'distributed.?system|distributed.?database|distributed.?comput' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'network.?fail|message.?loss|communicat.*fail|node.*fail|crash' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'eventual.*consisten'            && found=$((found + 1))
    echo "$c_lower" | grep -qE 'cassandra|dynamodb|riak|mongodb|redis' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'acid|base.*model|strong.*consisten|linearizab' && found=$((found + 1))
    echo "$c_lower" | grep -qE 'cp|ap|ca' | grep -qE 'system|database|example' && found=$((found + 1))

    [[ $found -gt 12 ]] && found=12
    local score=$(( (found * 10) / 12 ))

    if [[ $score -ge 8 ]]; then echo -e "${GREEN}${score}/10${RESET}  (${found}/12 concepts found)"
    elif [[ $score -ge 5 ]]; then echo -e "${YELLOW}${score}/10${RESET}  (${found}/12 concepts found)"
    else echo -e "${RED}${score}/10${RESET}  (${found}/12 concepts found)"
    fi
    echo "$score" > /tmp/_bench_score
}

# ── Main ──────────────────────────────────────────────────────────────────

echo ""
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}  Quality Benchmark v2 — Hard Prompts${RESET}"
echo -e " ${BOLD}  4 classes × 10 points = 40 total${RESET}"
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo ""

health=$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null || true)
if [[ "$health" != "200" ]]; then
    echo -e " ${RED}Server not responding at http://${HOST}:${PORT}/health${RESET}"
    echo " Start a server first, then run: $0 [port] [host]"
    exit 1
fi

model_info=$(curl -s "http://${HOST}:${PORT}/v1/models" 2>/dev/null | jq -r '.data[0].id // "unknown"' 2>/dev/null)
echo -e " Model: ${BOLD}${model_info}${RESET}"
echo ""

# ── Prompts ───────────────────────────────────────────────────────────────

CODE_PROMPT='Write a Python function called merge_intervals that takes a list of intervals (each interval is a list of two integers [start, end]) and merges all overlapping intervals, returning a list of merged intervals sorted by start value. Intervals that touch (e.g. [1,4] and [4,5]) should be merged. The input may be unsorted, contain duplicates, or be empty. Include a docstring and type hints. Output ONLY the function inside a ```python code block, nothing else. Do NOT include any test code.'

REASONING_PROMPT='Five people live in five houses in a row, each painted a different color: green, red, white, blue, yellow. Each person has a different nationality: British, Swedish, Danish, Norwegian, German. Each drinks a different beverage: tea, coffee, milk, beer, water. Each smokes a different brand: Pall Mall, Dunhill, Blends, Blue Master, Prince. Each keeps a different pet: dogs, birds, cats, horses, fish.

Clues:
1. The British person lives in the red house.
2. The Swedish person keeps dogs.
3. The Danish person drinks tea.
4. The green house is immediately left of the white house.
5. The green house owner drinks coffee.
6. The person who smokes Pall Mall keeps birds.
7. The yellow house owner smokes Dunhill.
8. The person in the center house drinks milk.
9. The Norwegian lives in the first house.
10. The person who smokes Blends lives next to the one who keeps cats.
11. The person who keeps horses lives next to the Dunhill smoker.
12. The person who smokes Blue Master drinks beer.
13. The German smokes Prince.
14. The Norwegian lives next to the blue house.
15. The person who smokes Blends has a neighbor who drinks water.

Solve the puzzle. For each house (1-5 from left to right), list the color, nationality, drink, smoke, and pet. Show your reasoning. End with: ANSWER: [the complete solution]'

INSTRUCTION_PROMPT='Write a short essay about a scientist who discovers something unexpected. Your response MUST follow ALL of these rules:
1. The essay must be exactly 5 paragraphs
2. Each paragraph must start with a different letter of the alphabet
3. The phrase "quantum entanglement" must appear exactly as written (those two words together)
4. Do NOT use the word "basically" anywhere in your response
5. Do NOT use the word "literally" anywhere in your response
6. The word "serendipity" must appear at least twice
7. The year 1927 must be mentioned somewhere
8. The third word of your very first sentence must be "the" (e.g. "In the morning..." works, "Once upon..." does not)'

KNOWLEDGE_PROMPT='Explain the CAP theorem in distributed computing systems. Your explanation must cover: what CAP stands for, why you can only have two of the three properties simultaneously, who formulated it and when, what a network partition is, the trade-offs involved, what eventual consistency means, and give at least one real-world database example for each of the three CAP combinations (CP, AP, CA). Be precise and technical.'

# ── Run tests ─────────────────────────────────────────────────────────────

total_score=0

echo -e " ${CYAN}Test 1: Code Generation (merge intervals)${RESET}"
echo " Sending code prompt..."
response=$(send_prompt "$CODE_PROMPT" 4096)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/code_response.txt"
test_code "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 2: Reasoning (Einstein puzzle — 15 clues)${RESET}"
echo " Sending reasoning prompt..."
response=$(send_prompt "$REASONING_PROMPT" 4096)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/reasoning_response.txt"
test_reasoning "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 3: Instruction Following (8 constraints)${RESET}"
echo " Sending instruction prompt..."
response=$(send_prompt "$INSTRUCTION_PROMPT" 4096)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/instruction_response.txt"
test_instruction "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 4: Knowledge (CAP theorem — 12 concepts)${RESET}"
echo " Sending knowledge prompt..."
response=$(send_prompt "$KNOWLEDGE_PROMPT" 4096)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/knowledge_response.txt"
test_knowledge "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

# ── Summary ───────────────────────────────────────────────────────────────

echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}${GREEN}  TOTAL: ${total_score}/40${RESET}"
echo -e " ${BOLD}${GREEN}═══════════════════════════════════════════════════════════${RESET}"
echo ""
echo " Model: ${model_info}"
echo " Responses saved to: ${RESULTS_DIR}/"
