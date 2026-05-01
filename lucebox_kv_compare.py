#!/bin/env python3
"""
Compare Lucebox decode speed with different KV cache settings at the same context size.
Restarts the server for each config, runs the benchmark, reports results.
"""
import json, os, sys, time, subprocess, signal, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LUCE_DIR = os.path.join(SCRIPT_DIR, "lucebox-hub", "dflash")
SERVER_LOG = os.path.join(SCRIPT_DIR, "lucebox_kv_test.log")
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "v1lucebox_bench.py")
MODEL = os.path.join(SCRIPT_DIR, "llama_models", "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf")
DRAFT = os.path.join(LUCE_DIR, "models", "draft")
BIN = os.path.join(LUCE_DIR, "build", "test_dflash")
PORT = 8080

# Configs to test: (label, ctx, ctk, ctv)
CONFIGS = [
    # Same context, different KV — isolate KV effect
    ("q8_0 / q8_0  ctx=32768",  32768,  "q8_0", "q8_0"),
    ("q4_0 / q4_0  ctx=32768",  32768,  "q4_0", "q4_0"),
    ("tq3_0/ q4_0  ctx=32768",  32768,  "tq3_0","q4_0"),
    ("tq3_0/ tq3_0 ctx=32768",  32768,  "tq3_0","tq3_0"),
    # Now same KV, different context — isolate context effect
    ("q4_0 / q4_0  ctx=16384",  16384,  "q4_0", "q4_0"),
    ("q4_0 / q4_0  ctx=65536",  65536,  "q4_0", "q4_0"),
    ("tq3_0/ tq3_0 ctx=131072", 131072, "tq3_0","tq3_0"),
]

def stop_server():
    subprocess.run(["pkill", "-f", "scripts/server.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "test_dflash"], capture_output=True)
    time.sleep(2)

def start_server(ctx, ctk, ctv):
    stop_server()
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        os.path.join(LUCE_DIR, "build/deps/llama.cpp/ggml/src") + ":" +
        os.path.join(LUCE_DIR, "build/deps/llama.cpp/ggml/src/ggml-cuda")
    )
    cmd = [
        sys.executable, os.path.join(LUCE_DIR, "scripts/server.py"),
        "--target", MODEL,
        "--draft", DRAFT,
        "--bin", BIN,
        "--port", str(PORT),
        "--max-ctx", str(ctx),
        "--ctk", ctk,
        "--ctv", ctv,
        "--daemon",
    ]
    log = open(SERVER_LOG, "w")
    proc = subprocess.Popen(cmd, cwd=LUCE_DIR, env=env, stdout=log, stderr=log)
    # Wait for server ready
    for i in range(45):
        try:
            r = requests.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.ok:
                return proc
        except:
            pass
        time.sleep(2)
    print(f"  ERROR: Server didn't start after 90s. Check {SERVER_LOG}")
    return None

def run_bench():
    """Run 3 quick prompts and return average tok/s"""
    prompts = [
        "def fibonacci(n):",
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for",
        "Write a short poem about GPUs and parallel computation.",
    ]
    speeds = []
    for p in prompts:
        payload = {
            "model": "luce-dflash",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 128,
            "stream": False,
        }
        try:
            t0 = time.time()
            r = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json=payload, timeout=120)
            wall = time.time() - t0
            data = r.json()
            comp = data.get("usage", {}).get("completion_tokens", 0)
            if comp > 0 and wall > 0:
                speeds.append(comp / wall)
        except Exception as e:
            print(f"  Request failed: {e}")
    if speeds:
        return sum(speeds) / len(speeds)
    return 0

def main():
    print(f"{'Config':<35} {'avg tok/s':>10}")
    print("-" * 48)

    results = []
    for label, ctx, ctk, ctv in CONFIGS:
        print(f"  Testing: {label} ...", end="", flush=True)
        proc = start_server(ctx, ctk, ctv)
        if proc is None:
            print(f"  SKIP (server failed)")
            continue
        tps = run_bench()
        print(f"  {tps:>8.1f}")
        results.append((label, tps))
        stop_server()

    print()
    print("=" * 48)
    print(f"{'Config':<35} {'avg tok/s':>10}")
    print("-" * 48)
    for label, tps in results:
        print(f"  {label:<33} {tps:>8.1f}")
    print()

    # Summary
    if len(results) >= 2:
        fastest = max(results, key=lambda x: x[1])
        slowest = min(results, key=lambda x: x[1])
        print(f"  Fastest: {fastest[0]} — {fastest[1]:.1f} tok/s")
        print(f"  Slowest: {slowest[0]} — {slowest[1]:.1f} tok/s")

if __name__ == "__main__":
    main()
