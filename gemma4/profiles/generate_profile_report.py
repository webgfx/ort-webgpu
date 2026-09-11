"""Summarize an ONNX Runtime Chrome-trace profile for Gemma 4."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    events = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))["workload"]["measured"]
    nodes = [event for event in events if event.get("cat") == "Node" and isinstance(event.get("dur"), (int, float))]
    by_op = defaultdict(lambda: [0.0, 0])
    providers = Counter()
    for event in nodes:
        event_args = event.get("args", {})
        op = event_args.get("op_name") or event.get("name", "unknown")
        by_op[op][0] += event["dur"] / 1000.0
        by_op[op][1] += 1
        providers[event_args.get("provider", "unspecified")] += 1
    top = sorted(by_op.items(), key=lambda item: item[1][0], reverse=True)[:15]
    total_node_ms = sum(value[0] for value in by_op.values())
    lines = [
        "# Gemma 4 ONNX Runtime WebGPU Profiling Report", "",
        "- Runtime: **ONNX Runtime WebGPU 1.30.0 + ONNX Runtime GenAI 0.16.0-dev**",
        "- Hardware: **NVIDIA GeForce RTX 4070 12 GB**, driver **591.44**, Windows 11",
        "- Model: **Gemma 4 E2B INT4 ONNX**, decoder WebGPU graph capture enabled",
        f"- Trace: `{Path(args.profile).name}`",
        f"- Input/output tokens: **{workload['input_tokens']} / {workload['output_tokens']}**",
        f"- TTFT: **{workload['ttft_ms']:.1f} ms**",
        f"- Decode throughput: **{workload['decode_tokens_per_second']:.1f} tok/s**",
        f"- End-to-end generation: **{workload['elapsed_ms']:.1f} ms**",
        f"- Profile events / node events: **{len(events)} / {len(nodes)}**", "",
        "Profiling was enabled on `model.decoder.session_options` with an absolute trace prefix:", "",
        "```json",
        f'"enable_profiling": "{str(Path(args.profile).with_name(Path(args.profile).stem.rsplit("_2026-", 1)[0])).replace(chr(92), "/")}"',
        "```", "",
        "## Provider Coverage", "", "| Provider | Node events |", "|---|---:|",
    ]
    lines += [f"| {provider} | {count} |" for provider, count in providers.most_common()]
    lines += ["", "## Top Operators by Accumulated Profile Duration", "",
              "Durations are accumulated ORT node-event time. They can overlap and include profiling overhead; they are not additive wall time.", "",
              "| Operator | Accumulated duration | Calls | Share of node duration |", "|---|---:|---:|---:|"]
    lines += [f"| {op} | {values[0]:.3f} ms | {values[1]} | {values[0] / total_node_ms * 100:.1f}% |" for op, values in top]
    lines += ["", "## Interpretation", "",
              "This is an instrumented diagnostic run, not a benchmark result. Profiling changes execution overhead. Open the trace in Perfetto or `chrome://tracing` for timeline analysis.", ""]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
