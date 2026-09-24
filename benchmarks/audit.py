"""Audit a complete native/Web comparison against the delivered source and binary."""
import argparse
import json
from pathlib import Path

from compare import compare, validate
from web.model_manifest import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--web', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    native, control, web = [json.loads(p.read_text('utf-8-sig')) for p in (args.native, args.control, args.web)]
    native_comparison = compare(control, native)
    cross_comparison = compare(native, web)
    if not all(row['outputsMatch'] for row in native_comparison):
        raise ValueError('Native optimization changed outputs')
    if not all(row['outputsMatch'] for row in cross_comparison):
        raise ValueError('Cross-runtime divergence; disclose it instead of accepting this audit')
    root = Path(__file__).resolve().parent
    for name, expected in native['benchmarkBuild']['sourceFiles'].items():
        if fingerprint(root / 'native' / name)['sha256'] != expected:
            raise ValueError('Delivered native source differs from tested build: ' + name)
    for name in ('host.mjs', 'runner.mjs', 'generator.mjs', 'capture-generator.mjs', 'gpu-greedy.mjs',
                 'gpu-pipeline.mjs', 'runtime-evidence.mjs', 'device.mjs', 'gpu_failure.mjs'):
        if fingerprint(root / 'web' / name) != web['sourceArtifacts'][name]:
            raise ValueError('Delivered Web source differs from tested runner: ' + name)
    if fingerprint(args.executable) != native['artifacts'][args.executable.name]:
        raise ValueError('Delivered executable differs from tested binary')
    if fingerprint(args.request)['sha256'] != native['requestSha256']:
        raise ValueError('Saved request bytes differ')
    timed = 0
    for data in (native, control, web):
        points = validate(data)
        timed += sum(len(row['samples']) for row in points.values())
        expected_runs = len(data['request']['promptLengths']) * (data['request']['repetitions'] + 1) * (data['request']['generationLength'] - 1)
        for model in data['models']:
            if model['executionEvidence']['decodeRuns'] != expected_runs:
                raise ValueError('Decode execution count differs from workload')
    for model in web['models']:
        runtime = model['executionEvidence']['runtime']
        if runtime['wasmVariant'] != 'jspi' or len(runtime['observedArtifacts']) != 2:
            raise ValueError('Actual JSPI loads not verified')
        for row in model['rows']:
            reference = row['validation']['reference']
            if not reference['passed'] or reference['comparedTokens'] != web['request']['generationLength']:
                raise ValueError('Web full-output reference validation is absent')
    result = dict(success=True, cases=len(cross_comparison), timedSamplesIncludingControl=timed,
                  allFullOutputsMatch=True, deliveredSourcesAndExecutableMatch=True,
                  cpuVisibleTimingValidated=True, actualJspiLoadsVerified=True,
                  requestSha256=native['requestSha256'], nativeBuild=native['benchmarkBuild'],
                  webBuild=web['build']['version'], browser=web['browser']['version'], comparison=cross_comparison,
                  inputArtifacts={str(p): fingerprint(p) for p in (args.native, args.control, args.web, args.request)})
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"PASS: {len(cross_comparison)} cases, {timed} timed samples including CPU control; exact outputs, CPU delivery and delivered artifacts verified")


if __name__ == '__main__':
    main()
