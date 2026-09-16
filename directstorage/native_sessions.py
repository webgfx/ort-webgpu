"""Measure the PR's native ORT loading path without GenAI's bootstrap session."""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prime-weights", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (repeat, mode)
        for repeat in range(args.repetitions)
        for mode in ("off", "required", "required-pipelined")
    ]
    random.Random(32444).shuffle(jobs)
    results = []
    for index, (repeat, mode) in enumerate(jobs):
        command = [str(args.runtime / "ort_session_probe.exe"), str(args.model), mode]
        log_path = args.output / f"{index:02d}-{mode}-r{repeat}.log"
        if args.prime_weights:
            buffer = bytearray(16 * 1024 * 1024)
            for weight_file in args.model.parent.glob("*.data"):
                with weight_file.open("rb", buffering=0) as stream:
                    while stream.readinto(buffer):
                        pass
            del buffer
        start = time.perf_counter()
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False
            )
        elapsed = (time.perf_counter() - start) * 1000
        text = (
            log_path.read_bytes().decode("utf-8", errors="replace").replace("\x00", "")
        )
        if completed.returncode:
            raise RuntimeError(text)
        data = next(
            json.loads(line.removeprefix("RESULT "))
            for line in text.splitlines()
            if line.startswith("RESULT ")
        )
        data.update(repeat=repeat, process_ms=elapsed, log=str(log_path))
        results.append(data)
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
