#!/bin/bash
# =========================================================================
# Quality Benchmark — Automated Model IQ Test
#
# 4 prompt classes, fully auto-scored:
#   1. Code:        generate a function, run it, check assertions
#   2. Reasoning:   logic puzzle with known answer
#   3. Instruction: count constraints followed out of N
#   4. Knowledge:   check required key concepts are mentioned
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

# ── Send prompt and get response ─────────────────────────────────────────

send_prompt() {
    local prompt="$1"
    local max_tokens="${2:-2048}"
    local response=$(curl -s --max-time 300 "$URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | jq -Rs .)}],
            \"max_tokens\": $max_tokens,
            \"temperature\": 0.6,
            \"top_k\": 20
        }" 2>/dev/null)
    echo "$response"
}

# ── Extract text content from API response ────────────────────────────────

get_content() {
    echo "$1" | jq -r '.choices[0].message.content // empty' 2>/dev/null
}

# ── TEST 1: Code (10 points) ─────────────────────────────────────────────
# Generate a function, run it, check if assertions pass.

test_code() {
    local label="$1"
    local content="$2"

    echo -ne "  [Code]          "

    # Extract python code from markdown code blocks
    local code=$(echo "$content" | sed -n '/^```python/,/^```/p' | sed '1d;$d')

    # If no markdown blocks, try to extract the function directly
    if [[ -z "$code" ]]; then
        code=$(echo "$content" | sed -n '/^def \|^    /p')
    fi

    if [[ -z "$code" ]]; then
        echo -e "${RED}0/10${RESET}  (no code found)"
        echo "0" > /tmp/_bench_score
        return
    fi

    # Add test assertions
    local test_code='
# --- Auto-generated assertions ---
assert is_valid_parentheses("()") == True, "test 1 failed"
assert is_valid_parentheses("()[]{}") == True, "test 2 failed"
assert is_valid_parentheses("(]") == False, "test 3 failed"
assert is_valid_parentheses("([)]") == False, "test 4 failed"
assert is_valid_parentheses("{[]}") == True, "test 5 failed"
assert is_valid_parentheses("") == True, "test 6 failed"
assert is_valid_parentheses("(((") == False, "test 7 failed"
assert is_valid_parentheses("}") == False, "test 8 failed"
print("ALL_ASSERTIONS_PASSED")
'

    echo "${code}${test_code}" > "${RESULTS_DIR}/${label}_code.py"

    local output=$(python3 "${RESULTS_DIR}/${label}_code.py" 2>&1)
    local exit_code=$?

    if echo "$output" | grep -q "ALL_ASSERTIONS_PASSED"; then
        echo -e "${GREEN}10/10${RESET}  (all assertions passed)"
        echo "10" > /tmp/_bench_score
    elif [[ $exit_code -ne 0 ]] && echo "$output" | grep -q "AssertionError\|assert"; then
        local passed=0
        local total=8
        for i in $(seq 1 $total); do
            local single_assert=""
            case $i in
                1) single_assert='assert is_valid_parentheses("()") == True' ;;
                2) single_assert='assert is_valid_parentheses("()[]{}") == True' ;;
                3) single_assert='assert is_valid_parentheses("(]") == False' ;;
                4) single_assert='assert is_valid_parentheses("([)]") == False' ;;
                5) single_assert='assert is_valid_parentheses("{[]}") == True' ;;
                6) single_assert='assert is_valid_parentheses("") == True' ;;
                7) single_assert='assert is_valid_parentheses("(((") == False' ;;
                8) single_assert='assert is_valid_parentheses("}") == False' ;;
            esac
            echo "${code}\n${single_assert}" > "${RESULTS_DIR}/${label}_code_single.py"
            python3 "${RESULTS_DIR}/${label}_code_single.py" &>/dev/null && passed=$((passed + 1))
        done
        local score=$(( (passed * 10) / total ))
        echo -e "${YELLOW}${score}/10${RESET}  (${passed}/${total} assertions passed)"
        echo "$score" > /tmp/_bench_score
    else
        echo -e "${RED}0/10${RESET}  (code error: $(echo "$output" | tail -1 | cut -c1-60))"
        echo "0" > /tmp/_bench_score
    fi
}

# ── TEST 2: Reasoning (10 points) ────────────────────────────────────────
# Logic puzzle with known answer: 1/5

test_reasoning() {
    local label="$1"
    local content="$2"

    echo -ne "  [Reasoning]     "

    # The correct answer is 1/5
    # Check for various forms of the answer
    local answer_score=0
    local steps_score=0

    # Check for correct answer (4 points)
    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    if echo "$c_lower" | grep -qE '1\s*/\s*5|one.?fifth|0\.2[^0-9]|20\s*%'; then
        answer_score=4
    fi

    # Check for step-by-step work (3 points)
    if echo "$c_lower" | grep -qE 'day 1|first day|after day 1|1/2.*remain'; then
        steps_score=$((steps_score + 1))
    fi
    if echo "$c_lower" | grep -qE 'day 2|second day|1/3.*remain|1/2.*2/3|1/3'; then
        steps_score=$((steps_score + 1))
    fi
    if echo "$c_lower" | grep -qE 'day 3|third day|1/4.*remain|1/3.*3/4'; then
        steps_score=$((steps_score + 1))
    fi

    # Check for correct method / formula (3 points)
    if echo "$c_lower" | grep -qE 'multiply|×|\*|fraction of|what remains'; then
        steps_score=$((steps_score + 1))
    fi
    if [[ $steps_score -gt 3 ]]; then steps_score=3; fi

    local total=$((answer_score + steps_score))

    if [[ $total -ge 8 ]]; then
        echo -e "${GREEN}${total}/10${RESET}  (answer=${answer_score}/4, steps=${steps_score}/3, method=$(( total - answer_score - steps_score ))/3)"
    elif [[ $total -ge 5 ]]; then
        echo -e "${YELLOW}${total}/10${RESET}  (answer=${answer_score}/4, steps=${steps_score}/3, method=$(( total - answer_score - steps_score ))/3)"
    else
        echo -e "${RED}${total}/10${RESET}  (answer=${answer_score}/4, steps=${steps_score}/3)"
    fi

    echo "$total" > /tmp/_bench_score
}

# ── TEST 3: Instruction Following (10 points) ────────────────────────────
# Count how many of 6 constraints are followed.

test_instruction() {
    local label="$1"
    local content="$2"

    echo -ne "  [Instruction]   "

    local met=0
    local total=6

    # 1. Exactly 4 stanzas (count blank-line-separated blocks)
    local stanzas=$(echo "$content" | sed '/^$/,$!d' | awk 'NF' | sed '/^$/d' | awk 'BEGIN{c=0} /^$/{c++; next} {lines[c]++} END{for(i in lines) if(lines[i]>0) n++; print n+0}')
    # Simpler: count blocks separated by empty lines
    local stanza_count=$(echo "$content" | awk 'BEGIN{c=1} /^$/{c++; next} {} END{print c}')
    if [[ "$stanza_count" -eq 4 ]] || [[ "$stanza_count" -ge 4 && "$stanza_count" -le 5 ]]; then
        met=$((met + 1))
    fi

    # 2. Each stanza is ~4 lines (check total line count is ~16)
    local line_count=$(echo "$content" | grep -c '.' 2>/dev/null || echo 0)
    if [[ "$line_count" -ge 12 && "$line_count" -le 24 ]]; then
        met=$((met + 1))
    fi

    # 3. Word "colors" appears
    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    if echo "$c_lower" | grep -q 'colors'; then
        met=$((met + 1))
    fi

    # 4. Robot named "Rust"
    if echo "$c_lower" | grep -q 'rust'; then
        met=$((met + 1))
    fi

    # 5. Last line ends with "?"
    local last_line=$(echo "$content" | sed '/^$/d' | tail -1 | sed 's/[[:space:]]*$//')
    if echo "$last_line" | grep -qE '\?$'; then
        met=$((met + 1))
    fi

    # 6. Word "canvas" appears
    if echo "$c_lower" | grep -q 'canvas'; then
        met=$((met + 1))
    fi

    local score=$(( (met * 10) / total ))

    if [[ $score -ge 8 ]]; then
        echo -e "${GREEN}${score}/10${RESET}  (${met}/${total} constraints met)"
    elif [[ $score -ge 5 ]]; then
        echo -e "${YELLOW}${score}/10${RESET}  (${met}/${total} constraints met)"
    else
        echo -e "${RED}${score}/10${RESET}  (${met}/${total} constraints met)"
    fi

    echo "$score" > /tmp/_bench_score
}

# ── TEST 4: Knowledge (10 points) ────────────────────────────────────────
# Check for 10 required key concepts.

test_knowledge() {
    local label="$1"
    local content="$2"

    echo -ne "  [Knowledge]     "

    local c_lower=$(echo "$content" | tr '[:upper:]' '[:lower:]')
    local found=0
    local total=10

    # Required concepts for 4-stroke engine explanation
    echo "$c_lower" | grep -q 'intake\|induction\|suction'     && found=$((found + 1))
    echo "$c_lower" | grep -q 'compression\|compress'          && found=$((found + 1))
    echo "$c_lower" | grep -q 'power\|combustion\|ignition\|explosion' && found=$((found + 1))
    echo "$c_lower" | grep -q 'exhaust'                        && found=$((found + 1))
    echo "$c_lower" | grep -q 'piston'                         && found=$((found + 1))
    echo "$c_lower" | grep -q 'valve'                          && found=$((found + 1))
    echo "$c_lower" | grep -q 'fuel.*air\|air.*fuel\|mixture\|air-fuel' && found=$((found + 1))
    echo "$c_lower" | grep -q 'spark\|ignite\|combust\|explode' && found=$((found + 1))
    echo "$c_lower" | grep -q 'cylinder'                       && found=$((found + 1))
    echo "$c_lower" | grep -q 'crankshaft\|crank shaft'        && found=$((found + 1))

    local score=$(( (found * 10) / total ))

    if [[ $score -ge 8 ]]; then
        echo -e "${GREEN}${score}/10${RESET}  (${found}/${total} key concepts found)"
    elif [[ $score -ge 5 ]]; then
        echo -e "${YELLOW}${score}/10${RESET}  (${found}/${total} key concepts found)"
    else
        echo -e "${RED}${score}/10${RESET}  (${found}/${total} key concepts found)"
    fi

    echo "$score" > /tmp/_bench_score
}

# ── Main ──────────────────────────────────────────────────────────────────

echo ""
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo -e " ${BOLD}  Quality Benchmark — Automated Model IQ Test${RESET}"
echo -e " ${BOLD}  4 classes × 10 points = 40 total${RESET}"
echo -e " ${BOLD}${CYAN}═══════════════════════════════════════════════════════════${RESET}"
echo ""

# Check server is up
health=$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null || true)
if [[ "$health" != "200" ]]; then
    echo -e " ${RED}Server not responding at http://${HOST}:${PORT}/health${RESET}"
    echo " Start a server first, then run: $0 [port] [host]"
    exit 1
fi

# Get model info
model_info=$(curl -s "http://${HOST}:${PORT}/v1/models" 2>/dev/null | jq -r '.data[0].id // "unknown"' 2>/dev/null)
echo -e " Model: ${BOLD}${model_info}${RESET}"
echo ""

# ── Define prompts ────────────────────────────────────────────────────────

CODE_PROMPT='Write a Python function called is_valid_parentheses that takes a string containing only parentheses, brackets, and braces — (, ), {, }, [, ] — and returns True if they are properly matched and nested, False otherwise. Include a docstring and type hints. Output ONLY the function inside a ```python code block, nothing else. Do NOT include any test code.'

REASONING_PROMPT='I have a barrel of wine. On day 1, I drink exactly 1/2 of the barrel. On day 2, I drink exactly 1/3 of what remains. On day 3, I drink exactly 1/4 of what remains. On day 4, I drink exactly 1/5 of what remains. What fraction of the original wine is left in the barrel? Show your work step by step for each day. End with: ANSWER: X/Y'

INSTRUCTION_PROMPT='Write a poem about a robot learning to paint. Your response MUST follow ALL of these rules:
1. The poem must be exactly 4 stanzas
2. Each stanza must be exactly 4 lines
3. The word "colors" must appear at least once
4. The robot must be named "Rust"
5. The last line must end with a question mark
6. Include the word "canvas" somewhere in the poem'

KNOWLEDGE_PROMPT='Explain how a 4-stroke internal combustion engine works. For each of the four strokes, describe: what happens to the piston, which valves are open or closed, and what happens to the fuel-air mixture. Be thorough and technical.'

# ── Run tests ─────────────────────────────────────────────────────────────

total_score=0

echo -e " ${CYAN}Test 1: Code Generation${RESET}"
echo " Sending code prompt..."
response=$(send_prompt "$CODE_PROMPT" 2048)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/code_response.txt"
test_code "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 2: Reasoning${RESET}"
echo " Sending reasoning prompt..."
response=$(send_prompt "$REASONING_PROMPT" 4096)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/reasoning_response.txt"
test_reasoning "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 3: Instruction Following${RESET}"
echo " Sending instruction prompt..."
response=$(send_prompt "$INSTRUCTION_PROMPT" 2048)
content=$(get_content "$response")
echo "$content" > "${RESULTS_DIR}/instruction_response.txt"
test_instruction "bench" "$content"
score=$(cat /tmp/_bench_score)
total_score=$((total_score + score))
echo ""

echo -e " ${CYAN}Test 4: Knowledge${RESET}"
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
