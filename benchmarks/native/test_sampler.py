"""Native GPU tests for the shared Web/native WGSL, including buffer reuse."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def fixtures(dtype):
    np_type = np.float16 if dtype == 'float16' else np.float32
    bits_type = np.uint16 if dtype == 'float16' else np.uint32
    cases = []
    def add(name, values):
        values = np.asarray(values, dtype=np_type)
        finite = bool(np.isfinite(values).all())
        expected = values.copy()
        expected[0] = -np.inf
        cases.append(dict(name=name, bits=values.view(bits_type).tolist(), valid=int(finite), token=int(np.argmax(expected))))
    n = 1025
    add('masked-max', [10, -2, 1, 0] + [-3]*(n-4))
    add('negative-tie', [0, -2, -2] + [-3]*(n-3))
    add('signed-zero-tie', [1, -0., 0.] + [-1]*(n-3))
    add('subnormal', [0, np.nextafter(np_type(0), np_type(1))] + [-1]*(n-2))
    for name, value in [('nan', np.nan), ('positive-inf', np.inf), ('negative-inf', -np.inf)]:
        add('suppressed-'+name, [value, 2, 1] + [-1]*(n-3))
        add(name, [0, value, 1] + [-1]*(n-3))
    rng = np.random.default_rng(42)
    for i in range(8):
        add('random-'+str(i), rng.standard_normal(n))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for dtype in ('float16', 'float32'):
        config = args.output_dir / (dtype+'.json')
        config.write_text(json.dumps(dict(probe=True, type=dtype, cases=fixtures(dtype))), encoding='utf-8')
        output = args.output_dir / (dtype+'-result.json')
        subprocess.run([str(args.executable.resolve()), str(config.resolve()), str(output.resolve())], check=True)
        print(output.read_text('utf-8'))
    # Every finite FP16 encoding participates in a real GPU reduction.
    values = np.arange(65536, dtype=np.uint16)
    values = values[np.isfinite(values.view(np.float16))]
    config = args.output_dir / 'all-finite-half.json'
    config.write_text(json.dumps(dict(probe=True, type='float16', cases=[dict(name='all-finite-half', bits=values.tolist(), valid=1, token=int(np.argmax(values.view(np.float16))))])), encoding='utf-8')
    subprocess.run([str(args.executable.resolve()), str(config.resolve()), str(args.output_dir / 'all-finite-half-result.json')], check=True)


if __name__ == '__main__':
    main()
