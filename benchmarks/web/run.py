"""Run the portable Web benchmark using an explicit local runtime and workload."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.model_manifest import fingerprint, contained_path


def verify_model(entry):
    root, manifest = entry['root'], entry['manifest']
    artifacts = [manifest['configArtifact']]
    for session in manifest['sessions'].values():
        artifacts.extend([session, *session['externalData']])
    for artifact in artifacts:
        actual = fingerprint(contained_path(root, artifact['file']))
        if any(actual[key] != artifact[key] for key in ('size', 'sha256')):
            raise ValueError('Model changed after preparing workload: ' + artifact['file'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', choices=['edge', 'edge-canary'], default='edge')
    parser.add_argument('--gpu-sampling', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--no-capture', action='store_true')
    parser.add_argument('--capture-diagnostics', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new result filename')
    request = json.loads(args.request.read_text('utf-8-sig'))
    build = json.loads((args.runtime / 'build-metadata.json').read_text('utf-8-sig'))
    required = {'ort.all.min.js', 'ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm'}
    if (build.get('runtime') != 'ort-web' or build.get('builtFromSource') is not True
            or build.get('wasmVariant') != 'jspi' or build.get('webgpuImplementation') != 'cpp'
            or build.get('entrypoint') != 'ort.all.min.js' or set(build.get('artifacts', {})) != required):
        raise ValueError('A pinned source-built C++ WebGPU/JSPI runtime is required')
    for name, expected in build['artifacts'].items():
        if fingerprint(contained_path(args.runtime, name)) != expected:
            raise ValueError('Runtime integrity failure: ' + name)
    if not args.no_capture and build.get('capabilities', {}).get('capturedExternalIoBinding') is not True:
        raise ValueError('Runtime lacks verified captured external I/O binding support')
    for entry in request['models']:
        verify_model(entry)
    config = dict(models=[m['manifest'] for m in request['models']], modelRoots=[m['root'] for m in request['models']],
                  prompts={m['name']: {str(c['pl']): c['prompt'] for c in m['cases']} for m in request['models']},
                  build=build, artifactRoot=str(args.runtime.resolve()), browser=args.browser,
                  headless=args.headless, timeout=args.timeout, wasmThreads=1, seed=request['seed'],
                  promptLengths=request['promptLengths'], generationLength=request['generationLength'],
                  repetitions=request['repetitions'], prefillChunkSize=0, graphCapture=not args.no_capture,
                  ioBinding=True, referenceTokens=request['generationLength'], captureDiagnostics=args.capture_diagnostics,
                  gpuSampling=args.gpu_sampling)
    with tempfile.TemporaryDirectory(prefix='ort-web-benchmark-') as temporary:
        config_file = Path(temporary) / 'config.json'
        config_file.write_text(json.dumps(config), encoding='utf-8')
        process = subprocess.run(['node', str(Path(__file__).with_name('host.mjs')), str(config_file)],
                                 stdout=subprocess.PIPE, text=True, timeout=args.timeout * len(request['models']) + 60)
        result = json.loads(process.stdout)
    devices = {(item.get('vendorId'), item.get('deviceId')) for item in result.get('gpu', {}).get('devices', [])}
    for model in result.get('models', []):
        if not model.get('success'):
            continue
        vendor = {'nvidia': 0x10de, 'amd': 0x1002, 'intel': 0x8086}.get(model.get('adapter', {}).get('vendor', '').lower())
        candidates = [pair for pair in devices if vendor and pair[0] == vendor and pair[1]]
        if len(candidates) != 1:
            model['success'] = False
            model.setdefault('errors', []).append('Cannot unambiguously attribute browser adapter to a hardware GPU')
            result['success'] = False
        else:
            model['executionEvidence']['device'] = dict(vendorId=candidates[0][0], deviceId=candidates[0][1], evidence='unique-selected-vendor-in-CDP-inventory')
    result.update(runtime='ort-web', build=build, request=request,
                  requestSha256=hashlib.sha256(args.request.read_bytes()).hexdigest(),
                  benchmarkOptions={key: config[key] for key in ('gpuSampling', 'graphCapture', 'wasmThreads', 'prefillChunkSize', 'referenceTokens')})
    result['sourceArtifacts'] = {p.name: fingerprint(p) for p in Path(__file__).parent.glob('*.mjs')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if process.returncode or not result.get('success'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
