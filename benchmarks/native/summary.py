"""Validate and display native benchmark data without rerunning inference."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text('utf-8-sig'))
    points = validate(data)
    print(f"{'Model':26} {'Input':>6} {'TTFT throughput':>17} {'Decode TPS':>12}")
    for (model, length), row in points.items():
        ttft = statistics.mean(s['ttftMs'] for s in row['samples'])
        decode = statistics.mean(s['e2eMs'] - s['ttftMs'] for s in row['samples'])
        print(f"{model:26} {length:6} {length*1000/ttft:17.2f} {(data['request']['generationLength']-1)*1000/decode:12.2f}")
    print('Rates are tokens/s from mean elapsed time. All requested samples validated.')


if __name__ == '__main__':
    main()
