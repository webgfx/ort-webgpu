"""
ORT Gemma4 INT4 Benchmark — comparable to Chrome on-device AI benchmark.
Runs the same prompts and collects: TTFT, decode tok/s, quality, GPU usage.

Usage:
    python benchmark_ort_gemma4.py
    python benchmark_ort_gemma4.py --model-path path/to/model
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import onnxruntime_genai as og

MODEL_PATH = "E:/workspace/models/gemma4-webgpu-int4"

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


def format_prompt(user_msg, system_msg=None):
    parts = ["<bos>"]
    if system_msg:
        parts.append(f"<|turn>system\n{system_msg}<turn|>\n")
    parts.append(f"<|turn>user\n{user_msg}<turn|>\n<|turn>model\n")
    return "".join(parts)


def generate(model, tokenizer, prompt_text, max_tokens=2048, greedy=False, t0=None):
    input_ids = tokenizer.encode(prompt_text)
    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(input_ids) + max_tokens,
        do_sample=not greedy,
        top_k=1 if greedy else 40,
        temperature=1.0 if greedy else 0.7,
        repetition_penalty=1.1,
    )
    gen = og.Generator(model, params)
    if t0 is None:
        t0 = time.perf_counter()
    gen.append_tokens(input_ids)

    tokens = []
    ttft = None
    t_after_first = None
    for _ in range(max_tokens):
        if gen.is_done():
            break
        gen.generate_next_token()
        if ttft is None:
            ttft = time.perf_counter() - t0
            t_after_first = time.perf_counter()
        tokens.append(gen.get_next_tokens()[0])

    total_time = time.perf_counter() - t0
    decode_elapsed = time.perf_counter() - t_after_first if t_after_first else 0
    decode_tokens = len(tokens) - 1
    decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 and decode_tokens > 0 else 0.0

    output_text = tokenizer.decode(tokens).split("<turn|>")[0].strip()
    return {
        "tokens": tokens,
        "text": output_text,
        "input_token_count": len(input_ids),
        "output_token_count": len(tokens),
        "ttft_s": ttft,
        "total_time_s": total_time,
        "decode_tps": decode_tps,
    }


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


# --------------- Benchmark routines ---------------

def bench_speed(model, tokenizer):
    print("\n=== Speed Test (5 long-form prompts) ===")
    results = []
    for i, prompt in enumerate(SPEED_PROMPTS):
        print(f"  [{i+1}/5] {prompt[:50]}...")
        prompt_text = format_prompt(prompt, system_msg="You are a helpful assistant. Respond in detail.")
        r = generate(model, tokenizer, prompt_text)
        results.append({
            "prompt": prompt,
            "inputTokens": r["input_token_count"],
            "outputTokens": r["output_token_count"],
            "timeMs": round(r["total_time_s"] * 1000),
            "tokPerSec": round(r["decode_tps"], 1),
            "msPerToken": round(1000 / r["decode_tps"], 1) if r["decode_tps"] > 0 else 0,
            "fullResponse": r["text"],
        })
        print(f"         {r['output_token_count']} tokens, {r['decode_tps']:.1f} tok/s")
    return results


def bench_ttft(model, tokenizer):
    print("\n=== TTFT Test (3 prompts) ===")
    results = []
    for i, prompt in enumerate(TTFT_PROMPTS):
        print(f"  [{i+1}/3] {prompt[:50]}...")
        prompt_text = format_prompt(prompt)
        r = generate(model, tokenizer, prompt_text)
        results.append({
            "prompt": prompt,
            "ttftMs": round(r["ttft_s"] * 1000) if r["ttft_s"] else 0,
            "totalMs": round(r["total_time_s"] * 1000),
            "outputTokens": r["output_token_count"],
            "decodeSpeed": round(r["decode_tps"], 1),
        })
        print(f"         TTFT={results[-1]['ttftMs']}ms, {r['decode_tps']:.1f} tok/s")
    return results


def bench_multiturn(model, tokenizer):
    print("\n=== Multi-Turn Test (10 turns) ===")
    sys_msg = "You are a helpful assistant. Keep responses concise (2-3 sentences)."
    results = []
    history = ""
    for i, prompt in enumerate(MULTITURN_PROMPTS):
        print(f"  [Turn {i+1}/10] {prompt}")
        # Build multi-turn context
        if i == 0:
            prompt_text = format_prompt(prompt, system_msg=sys_msg)
        else:
            prompt_text = history + f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"

        r = generate(model, tokenizer, prompt_text, max_tokens=256)
        # Append to history for next turn
        if i == 0:
            history = format_prompt(prompt, system_msg=sys_msg)
        else:
            history = history + f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
        history += r["text"] + "<turn|>\n"

        results.append({
            "turn": i + 1,
            "prompt": prompt,
            "timeMs": round(r["total_time_s"] * 1000),
            "outputTokens": r["output_token_count"],
            "tokPerSec": round(r["decode_tps"], 1),
        })
        print(f"         {r['output_token_count']} tokens, {r['decode_tps']:.1f} tok/s")
    return results


def bench_quality(model, tokenizer):
    print("\n=== Quality Test (10 factual prompts, greedy) ===")
    results = []
    for i, test in enumerate(QUALITY_TESTS):
        prompt_text = format_prompt(test["prompt"])
        r = generate(model, tokenizer, prompt_text, max_tokens=128, greedy=True)
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
    parser = argparse.ArgumentParser(description="ORT Gemma4 INT4 Benchmark")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output", help="Output JSON path (default: timestamped file)")
    parser.add_argument("--label", default="candidate", help="Human-readable run label")
    args = parser.parse_args()

    print("=" * 60)
    print("  ORT Gemma4 INT4 Benchmark")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    t_load = time.perf_counter()
    model = og.Model(args.model_path)
    tokenizer = og.Tokenizer(model)
    load_time = time.perf_counter() - t_load
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup
    print("\nWarming up...")
    warmup_text = format_prompt("Hello, how are you?")
    generate(model, tokenizer, warmup_text, max_tokens=32)
    print("Warmup done.")

    # Start GPU monitoring
    start_gpu_monitor()
    time.sleep(1)  # baseline sample

    # Run benchmarks
    speed_results = bench_speed(model, tokenizer)
    ttft_results = bench_ttft(model, tokenizer)
    multiturn_results = bench_multiturn(model, tokenizer)
    quality_results = bench_quality(model, tokenizer)

    # Stop GPU monitoring
    stop_gpu_monitor()
    gpu_summary = summarize_gpu()

    # Build report
    decoder_path = __import__("pathlib").Path(args.model_path) / "decoder" / "model.onnx"
    decoder_sha256 = None
    if decoder_path.is_file():
        digest = hashlib.sha256()
        with decoder_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        decoder_sha256 = digest.hexdigest()

    report = {
        "speed": speed_results,
        "ttft": ttft_results,
        "multiturn": multiturn_results,
        "quality": quality_results,
        "gpu": gpu_summary,
        "meta": {
            "engine": "onnxruntime-genai",
            "engine_version": importlib.metadata.version("onnxruntime-genai"),
            "label": args.label,
            "model_path": args.model_path,
            "decoder_model_sha256": decoder_sha256,
            "load_time_s": round(load_time, 2),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    filename = args.output or f"ort_benchmark_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
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
