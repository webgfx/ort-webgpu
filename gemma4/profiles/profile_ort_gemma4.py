"""Run a bounded Gemma 4 ORT GenAI WebGPU profiling workload."""
import argparse
import gc
import json
import time
from pathlib import Path

import onnxruntime_genai as og


def format_prompt(text):
    return f"<bos><|turn>user\n{text}<turn|>\n<|turn>model\n"


def run(model, tokenizer, text, max_output_tokens):
    ids = tokenizer.encode(format_prompt(text))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + max_output_tokens, do_sample=False, top_k=1)
    generator = og.Generator(model, params)
    start = time.perf_counter()
    generator.append_tokens(ids)
    first = None
    output = []
    while not generator.is_done() and len(output) < max_output_tokens:
        generator.generate_next_token()
        if first is None:
            first = time.perf_counter()
        output.append(generator.get_next_tokens()[0])
    end = time.perf_counter()
    del generator
    return {
        "input_tokens": len(ids),
        "output_tokens": len(output),
        "ttft_ms": round((first - start) * 1000, 3) if first else None,
        "elapsed_ms": round((end - start) * 1000, 3),
        "decode_tokens_per_second": round((len(output) - 1) / (end - first), 3) if first and len(output) > 1 else 0.0,
        "response": tokenizer.decode(output).split("<turn|>")[0].strip(),
    }


def make_exact_token_prompt(tokenizer, target_tokens):
    seed = "Explain how transformer attention processes a prompt, including queries, keys, values, masking, and contextual representations. "
    text = seed
    while len(tokenizer.encode(format_prompt(text))) < target_tokens:
        text += seed
    ids = tokenizer.encode(format_prompt(text))
    while len(ids) > target_tokens:
        text = text[:-1]
        ids = tokenizer.encode(format_prompt(text))
    if len(ids) != target_tokens:
        raise RuntimeError(f"Unable to construct exactly {target_tokens} input tokens; got {len(ids)}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-tokens", type=int, default=570)
    parser.add_argument("--generation-tokens", type=int, default=128)
    args = parser.parse_args()
    model = og.Model(args.model_path)
    tokenizer = og.Tokenizer(model)
    warmup = run(model, tokenizer, "Briefly explain photosynthesis.", 16)
    prompt = make_exact_token_prompt(tokenizer, args.input_tokens)
    measured = run(model, tokenizer, prompt, args.generation_tokens)
    result = {"workload": {"warmup": warmup, "measured": measured}}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    del tokenizer, model
    gc.collect()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
