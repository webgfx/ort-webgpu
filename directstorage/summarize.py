"""Summarize benchmark JSON and verify greedy token consistency."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--phases", nargs="+", default=["baseline", "tuning", "cache"])
    args = parser.parse_args()
    summaries = []
    references = {}
    mismatches = []
    for phase in args.phases:
        path = args.root / phase / "suite.json"
        if not path.exists():
            continue
        groups = collections.defaultdict(list)
        suite = json.loads(path.read_text())
        for run in suite["runs"]:
            if run["returncode"]:
                continue
            data = json.loads(Path(run["output"]).read_text())
            key = (data["prompt_tokens"], data["input_ids_sha256"])
            for request in data["requests"]:
                output = request["output_tokens"]
                reference = references.setdefault(key, output)
                if output != reference:
                    mismatches.append(
                        {
                            "phase": phase,
                            "run": run["output"],
                            "request": request["index"],
                            "expected": reference,
                            "actual": output,
                        }
                    )
            groups[(run["name"], run["prompt_tokens"])].append((run, data))
        for (name, prompt), samples in sorted(groups.items()):
            metrics = {
                "model_ms": [data["model_ms"] for _, data in samples],
                "tokenizer_ms": [data["tokenizer_ms"] for _, data in samples],
                "startup_ttft_ms": [
                    data["launch_to_first_token_ms"] for _, data in samples
                ],
                "first_request_ms": [
                    data["requests"][0]["ttft_ms"] for _, data in samples
                ],
                "warm_request_ms": [
                    r["ttft_ms"] for _, data in samples for r in data["requests"][1:]
                ],
                "peak_private_mib": [
                    data["memory"]["peak_pagefile"] / 1048576 for _, data in samples
                ],
                "peak_working_set_mib": [
                    data["memory"]["peak_wset"] / 1048576 for _, data in samples
                ],
            }
            row = {
                "phase": phase,
                "name": name,
                "prompt_tokens": prompt,
                "processes": len(samples),
            }
            for metric_name, values in metrics.items():
                if values:
                    row[metric_name] = {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                    }
            row["cache_stats"] = [
                stats for run, _ in samples for stats in run.get("cache_stats", [])
            ]
            summaries.append(row)
    result = {
        "summaries": summaries,
        "token_mismatches": mismatches,
        "reference_tokens": {str(key[0]): value for key, value in references.items()},
    }
    (args.root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(
            row["phase"],
            row["name"],
            row["prompt_tokens"],
            "n=",
            row["processes"],
            {
                name: round(value["median"], 2)
                for name, value in row.items()
                if isinstance(value, dict)
            },
        )
    print("TOKEN MISMATCHES:", len(mismatches))
    if mismatches:
        print(json.dumps(mismatches[:8], indent=2))


if __name__ == "__main__":
    main()
