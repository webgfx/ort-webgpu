"""
llama.cpp (Vulkan) Gemma4 E2B Benchmark — uses llama-server for model persistence,
extracts server-side timings for accurate measurements (no HTTP overhead in metrics).

The script auto-starts llama-server, runs benchmarks, and stops it when done.

Usage:
    python benchmark_llamacpp_gemma4.py
    python benchmark_llamacpp_gemma4.py --main-gpu 0  (for different GPU)
"""

import argparse
import atexit
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import httpx

LLAMA_SERVER_EXE = "E:/workspace/llama.cpp-bin/vulkan/llama-server.exe"
GGUF_MODEL = "E:/workspace/models/gemma4-e2b-gguf/google_gemma-4-E2B-it-Q4_K_M.gguf"
SERVER_URL = "http://127.0.0.1:8080"

SPEED_PROMPTS = [
    "Explain how a combustion engine works in detail.",
    "Write a short story about a robot learning to paint.",
    "List and explain the top 10 programming languages in 2026.",
    "Describe the process of photosynthesis step by step.",
    "What are the main differences between TCP and UDP protocols?",
]

TTFT_PROMPTS = [
    "Write a detailed explanation of quantum entanglement.",
    "Explain the history of the Internet in three paragraphs.",
    "Describe how neural networks learn from data.",
]

MULTITURN_PROMPTS = [
    "What is machine learning?",
    "How does it differ from traditional programming?",
    "Give an example of supervised learning.",
    "What about unsupervised learning?",
    "How is deep learning different?",
    "What are transformers in ML?",
    "Explain attention mechanism briefly.",
    "What is transfer learning?",
    "How do LLMs work?",
    "What is RLHF?",
]

QUALITY_TESTS = [
    {"prompt": "What is 15 * 37? Reply with just the number.", "expected": "555", "category": "Math"},
    {"prompt": "What is the capital of Australia? Reply with just the city name.", "expected": "Canberra", "category": "Factual"},
    {"prompt": "Translate 'hello' to French. Reply with just the French word.", "expected": "bonjour", "category": "Translation"},
    {"prompt": "Complete the sequence: 2, 4, 8, 16, __ Reply with just the number.", "expected": "32", "category": "Pattern"},
    {"prompt": "Is a whale a fish or a mammal? Reply with one word.", "expected": "mammal", "category": "Classification"},
    {"prompt": "What day comes after Monday? Reply with the day name.", "expected": "Tuesday", "category": "Common Sense"},
    {"prompt": "Fix the grammar: 'She don't like apples.' Reply with the corrected sentence only.", "expected": "doesn't", "category": "Grammar"},
    {"prompt": "What is the largest planet in our solar system? Reply with just the name.", "expected": "Jupiter", "category": "Factual 2"},
    {"prompt": "What is the chemical formula for water? Reply with just the formula in plain text.", "expected": "H2O", "category": "Science"},
    {"prompt": "How many legs does a spider have? Reply with just the number.", "expected": "8", "category": "Common Sense 2"},
]


# --------------- GPU monitor (background thread) ---------------

gpu_samples = []
gpu_monitor_running = False


def query_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=timestamp,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 9:
            return None
        return {
            "timestamp": parts[0],
            "gpu_name": parts[1],
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "memory_free_mb": float(parts[4]),
            "gpu_utilization_pct": float(parts[5]),
            "memory_utilization_pct": float(parts[6]),
            "temperature_c": float(parts[7]),
            "power_draw_w": float(parts[8]),
        }
    except Exception:
        return None


def gpu_monitor_loop(interval=0.5):
    global gpu_monitor_running
    while gpu_monitor_running:
        sample = query_nvidia_smi()
        if sample:
            sample["elapsed_sec"] = len(gpu_samples) * interval
            gpu_samples.append(sample)
        time.sleep(interval)


def start_gpu_monitor():
    global gpu_monitor_running, gpu_samples
    gpu_samples = []
    gpu_monitor_running = True
    t = threading.Thread(target=gpu_monitor_loop, daemon=True)
    t.start()
    return t


def stop_gpu_monitor():
    global gpu_monitor_running
    gpu_monitor_running = False
    time.sleep(0.6)


def summarize_gpu():
    if not gpu_samples:
        return {}
    mem = [s["memory_used_mb"] for s in gpu_samples]
    util = [s["gpu_utilization_pct"] for s in gpu_samples]
    power = [s["power_draw_w"] for s in gpu_samples]
    return {
        "gpu_name": gpu_samples[0]["gpu_name"],
        "vram_total_mb": gpu_samples[0]["memory_total_mb"],
        "samples": len(gpu_samples),
        "vram_mb": {
            "baseline": mem[0],
            "peak": max(mem),
            "average": round(sum(mem) / len(mem), 1),
            "min": min(mem),
            "delta": round(max(mem) - mem[0], 1),
        },
        "gpu_utilization_pct": {
            "average": round(sum(util) / len(util), 1),
            "peak": max(util),
        },
        "power_watts": {
            "average": round(sum(power) / len(power), 1),
            "peak": max(power),
        },
    }


# --------------- Server management ---------------

server_process = None


def start_llama_server(exe_path, model_path, port=8080, main_gpu=0, cache_prompt=True):
    """Start llama-server as a subprocess and wait until healthy."""
    global server_process
    cmd = [
        exe_path,
        "-m", model_path,
        "--port", str(port),
        "-ngl", "99",
        "-fa", "on",
        "--main-gpu", str(main_gpu),
        "--reasoning", "off",
    ]
    if not cache_prompt:
        cmd.append("--no-cache-prompt")
    print(f"Starting llama-server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    atexit.register(stop_llama_server)

    url = f"http://127.0.0.1:{port}/health"
    for i in range(120):
        if server_process.poll() is not None:
            print(f"ERROR: llama-server exited with code {server_process.returncode}")
            sys.exit(1)
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    print(f"llama-server ready after {i+1}s")
                    return
        except Exception:
            pass
        time.sleep(1)
    print("ERROR: llama-server did not become healthy within 120s")
    stop_llama_server()
    sys.exit(1)


def stop_llama_server():
    global server_process
    if server_process and server_process.poll() is None:
        print("Stopping llama-server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("llama-server stopped.")
    server_process = None


# --------------- API helpers ---------------

def chat_completion(server_url, messages, max_tokens=2048, temperature=0.7):
    """Non-streaming chat completion. Returns response with server-side timings."""
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{server_url}/v1/chat/completions", json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        })
        resp.raise_for_status()
        return resp.json()


# --------------- Benchmark routines ---------------

def bench_speed(server_url):
    print("\n=== Speed Test (5 long-form prompts) ===")
    sys_msg = "You are a helpful assistant. Respond in detail."
    results = []
    for i, prompt in enumerate(SPEED_PROMPTS):
        print(f"  [{i+1}/5] {prompt[:50]}...")
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        resp = chat_completion(server_url, messages, max_tokens=2048, temperature=0.7)

        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        timings = resp.get("timings", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)
        tok_per_sec = timings.get("predicted_per_second", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        results.append({
            "prompt": prompt,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "timeMs": round(predicted_ms),
            "tokPerSec": round(tok_per_sec, 1),
            "msPerToken": round(timings.get("predicted_per_token_ms", 0), 1),
            "fullResponse": text,
        })
        print(f"         {output_tokens} tokens, {tok_per_sec:.1f} tok/s (server-side)")
    return results


def bench_ttft(server_url):
    print("\n=== TTFT Test (3 prompts) ===")
    results = []
    for i, prompt in enumerate(TTFT_PROMPTS):
        print(f"  [{i+1}/3] {prompt[:50]}...")
        messages = [{"role": "user", "content": prompt}]
        resp = chat_completion(server_url, messages, max_tokens=2048, temperature=0.7)

        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        prompt_ms = timings.get("prompt_ms", 0)
        output_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = timings.get("predicted_per_second", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        results.append({
            "prompt": prompt,
            "ttftMs": round(prompt_ms),
            "totalMs": round(prompt_ms + predicted_ms),
            "outputTokens": output_tokens,
            "decodeSpeed": round(tok_per_sec, 1),
        })
        print(f"         TTFT={round(prompt_ms)}ms (server-side prompt eval), {tok_per_sec:.1f} tok/s")
    return results


def bench_multiturn(server_url):
    print("\n=== Multi-Turn Test (10 turns) ===")
    sys_msg = "You are a helpful assistant. Keep responses concise (2-3 sentences)."
    messages = [{"role": "system", "content": sys_msg}]
    results = []

    for i, prompt in enumerate(MULTITURN_PROMPTS):
        print(f"  [Turn {i+1}/10] {prompt}")
        messages.append({"role": "user", "content": prompt})

        resp = chat_completion(server_url, messages, max_tokens=256, temperature=0.7)

        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        timings = resp.get("timings", {})
        output_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = timings.get("predicted_per_second", 0)
        predicted_ms = timings.get("predicted_ms", 0)

        messages.append({"role": "assistant", "content": text})

        results.append({
            "turn": i + 1,
            "prompt": prompt,
            "timeMs": round(predicted_ms),
            "outputTokens": output_tokens,
            "tokPerSec": round(tok_per_sec, 1),
        })
        print(f"         {output_tokens} tokens, {tok_per_sec:.1f} tok/s (server-side)")

    return results


def bench_quality(server_url):
    print("\n=== Quality Test (10 factual prompts, greedy) ===")
    results = []
    for i, test in enumerate(QUALITY_TESTS):
        messages = [{"role": "user", "content": test["prompt"]}]
        resp = chat_completion(server_url, messages, max_tokens=128, temperature=0.0)
        text = resp["choices"][0]["message"]["content"]
        passed = test["expected"].lower() in text.lower()
        status = "PASS" if passed else "FAIL"
        results.append({
            "category": test["category"],
            "prompt": test["prompt"],
            "expected": test["expected"],
            "response": text,
            "pass": passed,
        })
        print(f"  [{status}] {test['category']}: {text[:60]}")
    score = sum(1 for r in results if r["pass"])
    print(f"  Score: {score}/{len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="llama.cpp Gemma4 E2B Benchmark")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--llama-server", default=LLAMA_SERVER_EXE)
    parser.add_argument("--model-path", default=GGUF_MODEL)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--no-auto-server", action="store_true", help="Don't auto-start llama-server")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()

    print("=" * 60)
    print("  llama.cpp Gemma4 E2B Benchmark")
    print("  Metrics from server-side timings (no HTTP overhead)")
    print("=" * 60)

    # Check if server is already running, otherwise start it
    server_already_running = False
    try:
        with httpx.Client(timeout=2.0) as client:
            health = client.get(f"{args.server_url}/health")
            health.raise_for_status()
            server_already_running = True
            print(f"\nllama-server already running at {args.server_url}")
    except Exception:
        pass

    if not server_already_running:
        if args.no_auto_server:
            print(f"\nERROR: llama-server not running at {args.server_url}")
            print("Start it manually or remove --no-auto-server")
            sys.exit(1)
        port = int(args.server_url.rsplit(":", 1)[-1])
        start_llama_server(args.llama_server, args.model_path, port=port, main_gpu=args.main_gpu)

    # Warmup
    print("\nWarming up...")
    warmup_prompts = [
        "Hello, how are you?",
        "Explain gravity in one paragraph.",
        "Write a haiku about the ocean.",
        "What is 2 + 2? Explain your reasoning.",
        "Describe the color blue.",
    ]
    for i, wp in enumerate(warmup_prompts):
        chat_completion(args.server_url, [{"role": "user", "content": wp}], max_tokens=128, temperature=0.7)
        print(f"  Warmup {i+1}/{len(warmup_prompts)} done.")
    print("Warmup complete.")

    # Start GPU monitoring
    start_gpu_monitor()
    time.sleep(1)

    # Run benchmarks
    speed_results = bench_speed(args.server_url)
    ttft_results = bench_ttft(args.server_url)
    multiturn_results = bench_multiturn(args.server_url)
    quality_results = bench_quality(args.server_url)

    # Stop GPU monitoring
    stop_gpu_monitor()
    gpu_summary = summarize_gpu()

    # Stop server if we started it
    if not server_already_running:
        stop_llama_server()

    # Build report
    report = {
        "speed": speed_results,
        "ttft": ttft_results,
        "multiturn": multiturn_results,
        "quality": quality_results,
        "gpu": gpu_summary,
        "meta": {
            "engine": "llama.cpp",
            "label": args.label,
            "llama_server": args.llama_server,
            "server_url": args.server_url,
            "model": args.model_path,
            "quantization": "Q4_K_M",
            "backend": "CUDA" if "cuda" in args.llama_server.lower() else "Vulkan",
            "gpu_offload": "all layers (-ngl 99)",
            "flash_attention": True,
            "mtp_enabled": False,
            "speculative_decoding": "none",
            "measurement": "server-side timings (no HTTP overhead)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    filename = args.output or f"llamacpp_benchmark_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    avg_tps = sum(r["tokPerSec"] for r in speed_results) / len(speed_results)
    avg_ttft = sum(r["ttftMs"] for r in ttft_results) / len(ttft_results)
    mt_avg_tps = sum(r["tokPerSec"] for r in multiturn_results) / len(multiturn_results)
    q_score = sum(1 for r in quality_results if r["pass"])

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Avg decode speed (long-form):  {avg_tps:.1f} tok/s")
    print(f"  Avg TTFT (prompt eval):        {avg_ttft:.0f} ms")
    print(f"  Avg decode speed (multi-turn): {mt_avg_tps:.1f} tok/s")
    print(f"  Quality score:                 {q_score}/{len(quality_results)}")
    if gpu_summary:
        print(f"  VRAM peak:                     {gpu_summary['vram_mb']['peak']:.0f} MB")
        print(f"  VRAM delta:                    +{gpu_summary['vram_mb']['delta']:.0f} MB")
        print(f"  GPU util avg:                  {gpu_summary['gpu_utilization_pct']['average']:.1f}%")
    print(f"\n  Metrics: server-side timings (no HTTP overhead)")
    print(f"  Results saved to: {filename}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
