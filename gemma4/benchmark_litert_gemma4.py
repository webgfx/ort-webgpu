"""
LiteRT-LM Gemma4 E2B Benchmark — comparable to ORT and Chrome benchmarks.
Runs the same prompts and collects: TTFT, decode tok/s, quality, GPU usage.

Usage:
    python benchmark_litert_gemma4.py
    python benchmark_litert_gemma4.py --model-path path/to/model.litertlm
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import litert_lm

MODEL_PATH = "E:/workspace/models/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm"

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


# --------------- Helper: count tokens ---------------

def count_tokens(engine, text):
    """Count tokens using the engine's tokenizer."""
    return len(engine.tokenize(text))


# --------------- Benchmark routines ---------------

def generate_response(engine, prompt, system_msg=None, max_tokens=2048, greedy=False):
    """Send a message and collect response with timing."""
    sampler = litert_lm.SamplerConfig(
        temperature=0.0 if greedy else 0.7,
        top_k=1 if greedy else 40,
    )
    with engine.create_conversation(
        system_message=system_msg,
        sampler_config=sampler,
        max_output_tokens=max_tokens,
    ) as conversation:
        t0 = time.perf_counter()
        response = conversation.send_message(prompt)
        total_time = time.perf_counter() - t0

    text = ""
    if response and "content" in response:
        for item in response["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
            elif isinstance(item, dict) and "text" in item:
                text = item["text"]

    output_tokens = count_tokens(engine, text)
    input_tokens = count_tokens(engine, prompt)
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": total_time,
        "tok_per_sec": output_tokens / total_time if total_time > 0 else 0,
    }


def generate_streaming(engine, prompt, system_msg=None, max_tokens=2048, greedy=False):
    """Send a message with streaming to measure TTFT and pure decode speed."""
    sampler = litert_lm.SamplerConfig(
        temperature=0.0 if greedy else 0.7,
        top_k=1 if greedy else 40,
    )
    with engine.create_conversation(
        system_message=system_msg,
        sampler_config=sampler,
        max_output_tokens=max_tokens,
    ) as conversation:
        t0 = time.perf_counter()
        ttft = None
        t_after_first = None
        text_parts = []

        for chunk in conversation.send_message_async(prompt):
            if ttft is None:
                ttft = time.perf_counter() - t0
                t_after_first = time.perf_counter()
            if "content" in chunk:
                for item in chunk["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])

        total_time = time.perf_counter() - t0
        decode_time = time.perf_counter() - t_after_first if t_after_first else total_time

    text = "".join(text_parts)
    output_tokens = count_tokens(engine, text)
    input_tokens = count_tokens(engine, prompt)
    decode_tokens = max(output_tokens - 1, 1)
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_s": total_time,
        "decode_time_s": decode_time,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
    }


def bench_speed(engine):
    print("\n=== Speed Test (5 long-form prompts) ===")
    sys_msg = "You are a helpful assistant. Respond in detail."
    results = []
    for i, prompt in enumerate(SPEED_PROMPTS):
        print(f"  [{i+1}/5] {prompt[:50]}...")
        r = generate_streaming(engine, prompt, system_msg=sys_msg)
        results.append({
            "prompt": prompt,
            "inputTokens": r["input_tokens"],
            "outputTokens": r["output_tokens"],
            "timeMs": round(r["total_time_s"] * 1000),
            "decodeMs": round(r["decode_time_s"] * 1000),
            "ttftMs": round(r["ttft_s"] * 1000) if r["ttft_s"] else 0,
            "tokPerSec": round(r["decode_tps"], 1),
            "msPerToken": round(1000 / r["decode_tps"], 1) if r["decode_tps"] > 0 else 0,
            "fullResponse": r["text"],
        })
        print(f"         {r['output_tokens']} tokens, {r['decode_tps']:.1f} tok/s (decode only), TTFT={round(r['ttft_s']*1000) if r['ttft_s'] else 0}ms")
    return results


def bench_ttft(engine):
    print("\n=== TTFT Test (3 prompts) ===")
    results = []
    for i, prompt in enumerate(TTFT_PROMPTS):
        print(f"  [{i+1}/3] {prompt[:50]}...")
        r = generate_streaming(engine, prompt)
        results.append({
            "prompt": prompt,
            "ttftMs": round(r["ttft_s"] * 1000) if r["ttft_s"] else 0,
            "totalMs": round(r["total_time_s"] * 1000),
            "outputTokens": r["output_tokens"],
            "decodeSpeed": round(r["decode_tps"], 1),
        })
        print(f"         TTFT={results[-1]['ttftMs']}ms, {r['decode_tps']:.1f} tok/s (decode only)")
    return results


def bench_multiturn(engine):
    print("\n=== Multi-Turn Test (10 turns) ===")
    sys_msg = "You are a helpful assistant. Keep responses concise (2-3 sentences)."
    sampler = litert_lm.SamplerConfig(temperature=0.7, top_k=40)
    results = []

    with engine.create_conversation(
        system_message=sys_msg,
        sampler_config=sampler,
        max_output_tokens=256,
    ) as conversation:
        for i, prompt in enumerate(MULTITURN_PROMPTS):
            print(f"  [Turn {i+1}/10] {prompt}")
            t0 = time.perf_counter()
            response = conversation.send_message(prompt)
            elapsed = time.perf_counter() - t0

            text = ""
            if response and "content" in response:
                for item in response["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                    elif isinstance(item, dict) and "text" in item:
                        text = item["text"]

            output_tokens = count_tokens(engine, text)
            tok_per_sec = output_tokens / elapsed if elapsed > 0 else 0

            results.append({
                "turn": i + 1,
                "prompt": prompt,
                "timeMs": round(elapsed * 1000),
                "outputTokens": output_tokens,
                "tokPerSec": round(tok_per_sec, 1),
            })
            print(f"         ~{output_tokens} tokens, {tok_per_sec:.1f} tok/s")

    return results


def bench_quality(engine):
    print("\n=== Quality Test (10 factual prompts, greedy) ===")
    results = []
    for i, test in enumerate(QUALITY_TESTS):
        r = generate_response(engine, test["prompt"], max_tokens=128, greedy=True)
        passed = test["expected"].lower() in r["text"].lower()
        status = "PASS" if passed else "FAIL"
        results.append({
            "category": test["category"],
            "prompt": test["prompt"],
            "expected": test["expected"],
            "response": r["text"],
            "pass": passed,
        })
        print(f"  [{status}] {test['category']}: {r['text'][:60]}")
    score = sum(1 for r in results if r["pass"])
    print(f"  Score: {score}/{len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LiteRT-LM Gemma4 E2B Benchmark")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--no-mtp", action="store_true", help="Disable MTP speculative decoding")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()

    print("=" * 60)
    print("  LiteRT-LM Gemma4 E2B Benchmark")
    print("=" * 60)

    print(f"\nLoading model from: {args.model_path}")
    t_load = time.perf_counter()
    engine_kwargs = {}
    if not args.no_mtp:
        engine_kwargs["enable_speculative_decoding"] = True
    engine = litert_lm.Engine(
        args.model_path,
        backend=litert_lm.Backend.GPU(),
        max_num_tokens=8192,
        **engine_kwargs,
    )
    load_time = time.perf_counter() - t_load
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup (multiple passes to let MTP drafter compile shaders)
    print("\nWarming up...")
    warmup_prompts = [
        "Hello, how are you?",
        "Explain gravity in one paragraph.",
        "Write a haiku about the ocean.",
        "What is 2 + 2? Explain your reasoning.",
        "Describe the color blue.",
    ]
    for i, wp in enumerate(warmup_prompts):
        with engine.create_conversation(max_output_tokens=128) as conv:
            conv.send_message(wp)
        print(f"  Warmup {i+1}/{len(warmup_prompts)} done.")
    print("Warmup complete.")

    # Start GPU monitoring
    start_gpu_monitor()
    time.sleep(1)

    # Run benchmarks
    speed_results = bench_speed(engine)
    ttft_results = bench_ttft(engine)
    multiturn_results = bench_multiturn(engine)
    quality_results = bench_quality(engine)

    # Stop GPU monitoring
    stop_gpu_monitor()
    gpu_summary = summarize_gpu()

    # Cleanup (Engine uses context manager, no explicit delete)
    del engine

    # Build report
    report = {
        "speed": speed_results,
        "ttft": ttft_results,
        "multiturn": multiturn_results,
        "quality": quality_results,
        "gpu": gpu_summary,
        "meta": {
            "engine": "litert-lm (Python API)",
            "engine_version": importlib.metadata.version("litert-lm-api"),
            "label": args.label,
            "model_path": args.model_path,
            "load_time_s": round(load_time, 2),
            "backend": "GPU",
            "mtp_enabled": not args.no_mtp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    mtp_label = "mtp-on" if not args.no_mtp else "mtp-off"
    filename = args.output or f"litert_benchmark_{mtp_label}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
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
    print(f"  Avg TTFT:                      {avg_ttft:.0f} ms")
    print(f"  Avg decode speed (multi-turn): {mt_avg_tps:.1f} tok/s")
    print(f"  Quality score:                 {q_score}/{len(quality_results)}")
    if gpu_summary:
        print(f"  VRAM peak:                     {gpu_summary['vram_mb']['peak']:.0f} MB")
        print(f"  VRAM delta:                    +{gpu_summary['vram_mb']['delta']:.0f} MB")
        print(f"  GPU util avg:                  {gpu_summary['gpu_utilization_pct']['average']:.1f}%")
    print(f"\n  Results saved to: {filename}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
