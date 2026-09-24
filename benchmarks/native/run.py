"""Run each native model in an isolated process; preserve artifacts and failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.run import verify_model
from web.model_manifest import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-sampling', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--no-capture', action='store_true')
    parser.add_argument('--capture-diagnostics', action='store_true')
    parser.add_argument('--max-pending', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_pending <= 64:
        parser.error('Output exists or pending limit is invalid')
    request = json.loads(args.request.read_text('utf-8-sig'))
    staging = args.output.with_suffix('.artifacts')
    staging.mkdir(parents=True, exist_ok=False)
    result = dict(runtime='ort-native-optimized', success=False, models=[], errors=[], request=request,
                  requestSha256=hashlib.sha256(args.request.read_bytes()).hexdigest(),
                  artifacts={p.name: fingerprint(p) for p in [args.executable, *args.executable.parent.glob('*.dll')]})
    metadata = args.executable.parent / 'build-metadata.json'
    if metadata.exists():
        result['build'] = json.loads(metadata.read_text('utf-8-sig'))
    benchmark_build = args.executable.parent / 'benchmark-build.json'
    if benchmark_build.exists():
        result['benchmarkBuild'] = json.loads(benchmark_build.read_text('utf-8-sig'))
        if fingerprint(args.executable.parent / 'onnxruntime.dll')['sha256'] != result['benchmarkBuild']['runtimeSha256']:
            raise ValueError('Native runtime no longer matches its build provenance')
    for entry in request['models']:
        verify_model(entry)
        folder = staging / entry['name']
        folder.mkdir()
        settings = dict(capture=not args.no_capture, diagnostics=args.capture_diagnostics, gpuSampling=args.gpu_sampling,
                        maxPending=args.max_pending)
        result['benchmarkOptions'] = {key: value for key, value in settings.items() if key != 'sampler'}
        config = folder / 'config.json'
        output = folder / 'result.json'
        config.write_text(json.dumps(dict(model=entry, options=settings, generationLength=request['generationLength'],
                                         repetitions=request['repetitions'])), encoding='utf-8')
        with (folder / 'run.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run([str(args.executable.resolve()), str(config.resolve()), str(output.resolve())],
                                         cwd=args.executable.parent, stdout=log, stderr=log, timeout=args.timeout)
                model = json.loads(output.read_text('utf-8')) if output.exists() else dict(name=entry['name'], success=False, errors=['Process produced no result'])
                if process.returncode:
                    model['success'] = False
                    model.setdefault('errors', []).append(f'Process exit {process.returncode}')
            except subprocess.TimeoutExpired:
                model = dict(name=entry['name'], success=False, errors=['Process timeout'])
        result['models'].append(model)
        print(entry['name'], 'PASS' if model.get('success') else model.get('errors'), flush=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    result['success'] = len(result['models']) == len(request['models']) and all(m.get('success') for m in result['models'])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
