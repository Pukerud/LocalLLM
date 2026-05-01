#!/usr/bin/env python3
"""
Benchmark the running Lucebox DFlash server via its OpenAI-compatible API.

Tests the actual running server with whatever model/context/settings you
configured — no separate test_dflash process launched.

Usage:
    python3 v1lucebox_bench.py                    # HumanEval 10 prompts, 128 gen tokens
    python3 v1lucebox_bench.py --port 8080         # custom port
    python3 v1lucebox_bench.py --n-gen 256         # more tokens per prompt
    python3 v1lucebox_bench.py --url http://...    # remote server
"""
import argparse
import json
import time
import sys
import requests

# Same 10 HumanEval prompts used by bench_he.py
PROMPTS = [
    ("has_close_elements",
     "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
     '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
     "    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
     "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n"
     '    """\n    for'),
    ("separate_paren_groups",
     "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n"
     '    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n'
     "    separate those group into separate strings and return the list of those.\n"
     "    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n"
     "    Ignore any spaces in the input string.\n"
     "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
     "    ['()', '(())', '(()())']\n"
     '    """\n    result'),
    ("truncate_number",
     "def truncate_number(number: float) -> float:\n"
     '    """ Given a floating point number, return its decimal part.\n'
     "    >>> truncate_number(3.5)\n    0.5\n"
     '    """\n    return'),
    ("below_zero",
     "from typing import List\n\ndef below_zero(operations: List[float]) -> bool:\n"
     '    """ Given a list of operation deposits and withdrawals, check if at any point the account balance falls below zero.\n'
     "    >>> below_zero([1, 2, 3])\n    False\n"
     "    >>> below_zero([1, 2, -4, 5])\n    True\n"
     '    """\n    balance'),
    ("mean_absolute_deviation",
     "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n"
     '    """ Given a list of numbers, return the mean absolute deviation.\n'
     "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n"
     '    """\n    mean'),
    ("intersperse",
     "from typing import List\n\ndef intersperse(numbers: List[int], delimiter: int) -> List[int]:\n"
     '    """ Insert a number delimiter between every two consecutive elements of the list numbers.\n'
     "    >>> intersperse([], 4)\n    []\n"
     "    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n"
     '    """\n    result'),
    ("parse_nested_parens",
     "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n"
     '    """ Input to this function is a string containing multiple groups of nested parentheses.\n'
     "    Your goal is to parse those groups and return the maximum depth of nesting in each group.\n"
     "    >>> parse_nested_parens('(()()) ((()) ())')\n    [2, 3, 1]\n"
     '    """\n    depths'),
    ("filter_by_substring",
     "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
     '    """ Filter a list of strings to include only those containing the given substring.\n'
     "    >>> filter_by_substring([], 'a')\n    []\n"
     "    >>> filter_by_substring(['abc', 'bac', 'cde', 'xyz'], 'bc')\n    ['abc', 'bac']\n"
     '    """\n    return'),
    ("sum_product",
     "from typing import List\n\ndef sum_product(numbers: List[int]) -> tuple:\n"
     '    """ Return a tuple (sum, product) of the numbers in the list.\n'
     "    >>> sum_product([])\n    (0, 1)\n"
     "    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n"
     '    """\n    sum_val'),
    ("rolling_max",
     "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n"
     '    """ Return a list where each element is the running maximum up to that index.\n'
     "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n"
     '    """\n    result'),
]


def bench_one(url: str, prompt_text: str, n_gen: int, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": n_gen,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
    wall = time.time() - t0
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    data = r.json()
    comp = data.get("usage", {}).get("completion_tokens", 0)
    prompt = data.get("usage", {}).get("prompt_tokens", 0)
    if comp == 0 or wall == 0:
        return {"error": "zero tokens or zero time", "prompt": prompt, "comp": comp, "wall": wall}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "wall_s": wall,
        "tok_s": comp / wall,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark running Lucebox DFlash server")
    ap.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    ap.add_argument("--port", type=int, default=None, help="Port (shorthand for --url http://localhost:PORT)")
    ap.add_argument("--n-gen", type=int, default=128, help="Max tokens to generate per prompt")
    ap.add_argument("--model", default="luce-dflash", help="Model name to send in requests")
    ap.add_argument("--prompts", type=int, default=10, help="Number of prompts to run (1-10)")
    args = ap.parse_args()

    if args.port:
        args.url = f"http://localhost:{args.port}"

    # Check server is up
    try:
        r = requests.get(f"{args.url}/v1/models", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"[!] Server not responding at {args.url}: {e}")
        sys.exit(1)

    print(f"[bench] url = {args.url}")
    print(f"[bench] n_gen = {args.n_gen}")
    print(f"[bench] server models: {r.json()}")
    print()

    n = min(args.prompts, len(PROMPTS))
    results = []
    print(f"{'prompt':<30} {'prompt_tok':>10} {'comp_tok':>10} {'wall_s':>8} {'tok/s':>8}")
    print("-" * 72)

    for i in range(n):
        name, text = PROMPTS[i]
        res = bench_one(args.url, text, args.n_gen, args.model)
        if "error" in res:
            print(f"  {name:<28} ERROR: {res['error']}")
            continue
        print(f"  {name:<28} {res['prompt_tokens']:>10} {res['completion_tokens']:>10} "
              f"{res['wall_s']:>8.2f} {res['tok_s']:>8.2f}")
        results.append(res)

    if results:
        print("-" * 72)
        avg_tps = sum(r["tok_s"] for r in results) / len(results)
        avg_wall = sum(r["wall_s"] for r in results) / len(results)
        avg_comp = sum(r["completion_tokens"] for r in results) / len(results)
        print(f"  {'MEAN':<28} {'':>10} {avg_comp:>10.0f} {avg_wall:>8.2f} {avg_tps:>8.2f}")
        print()
        print(f"  Average: {avg_tps:.2f} tok/s  ({len(results)} prompts)")


if __name__ == "__main__":
    main()
