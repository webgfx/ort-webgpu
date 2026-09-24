"""Run stock GenAI and a matched public-API harness without modifying source models."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.run import verify_model
from web.model_manifest import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stock', action='store_true')
    parser.add_argument('--validation', choices=['basic', 'disabled'], default='basic')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve existing result file')
    request = json.loads(args.request.read_text('utf-8-sig'))
    folder = args.output.with_suffix('.artifacts'); folder.mkdir(parents=True, exist_ok=False)
    build = json.loads((args.binary_dir / 'genai-build.json').read_text('utf-8-sig'))
    result = dict(runtime='genai-stock' if args.stock else 'genai-matched', success=False, models=[], errors=[], request=request,
                  requestSha256=fingerprint(args.request)['sha256'], genaiBuild=build,
                  artifacts={p.name: fingerprint(p) for p in args.binary_dir.iterdir() if p.suffix in ('.dll', '.exe')},
                  policy=dict(capture='Aion only, others disabled as in daily effective policy', validation=args.validation,
                              robustness='requested enabled; GenAI initial-device option forwarding must be verified' if args.validation == 'basic' else 'release-default',
                              prompt='stock unseeded random IDs 0..99' if args.stock else 'exact saved request IDs'))
    for entry in request['models']:
        verify_model(entry)
        source = Path(entry['root']); staged = folder / entry['name']; staged.mkdir()
        # Hardlinks share immutable weights; only the new configuration is written.
        for file in source.rglob('*'):
            if file.is_file() and file.name != 'genai_config.json':
                dest = staged / file.relative_to(source); dest.parent.mkdir(parents=True, exist_ok=True); os.link(file, dest)
        config = json.loads((source / 'genai_config.json').read_text('utf-8-sig'))
        config['search'].update(max_length=8192, do_sample=False, num_beams=1, repetition_penalty=1)
        def visit(section):
            if not isinstance(section, dict): return
            for provider in section.get('session_options', {}).get('provider_options', []):
                if 'webgpu' in provider:
                    provider['webgpu'].update(dawnBackendType='D3D12', powerPreference='high-performance', validationMode=args.validation,
                                             enableGraphCapture='1' if section is config['model']['decoder'] and entry['name'] == 'aion' else '0')
                    if args.validation == 'basic': provider['webgpu']['enableRobustness'] = '1'
            for key, value in section.items():
                if key != 'session_options': visit(value)
        visit(config['model'])
        (staged / 'genai_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        model = dict(name=entry['name'], success=False, rows=[], errors=[], effectiveConfig=config)
        if not args.stock:
            config_file = staged / 'benchmark-request.json'; output = staged / 'result.json'
            config_file.write_text(json.dumps(dict(modelPath=str(staged.resolve()), cases=entry['cases'],
                                                   generationLength=request['generationLength'], repetitions=request['repetitions'])), encoding='utf-8')
            command = [str((args.binary_dir / 'genai_reference.exe').resolve()), str(config_file.resolve()), str(output.resolve())]
            with (staged / 'run.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, stdout=log, stderr=log, cwd=args.binary_dir, timeout=1800)
            if output.exists(): model.update(json.loads(output.read_text('utf-8')))
            model['success'] = model.get('success', False) and process.returncode == 0
        else:
            for case in entry['cases']:
                command = [str((args.binary_dir / 'model_benchmark.exe').resolve()), '-i', str(staged.resolve()), '-l', str(case['pl']),
                           '--use_random_tokens', '-g', str(request['generationLength']), '-r', str(request['repetitions']), '-w', '1', '--reuse_generator', '-ml', '8192']
                process = subprocess.run(command, capture_output=True, text=True, cwd=args.binary_dir, timeout=1800)
                (staged / f"stock-{case['pl']}.log").write_text(process.stdout + process.stderr, encoding='utf-8')
                row = dict(pl=case['pl'], command=command, exitCode=process.returncode)
                for key, pattern in dict(prefillTps=r'Prompt processing.*?avg \(tokens/s\):\s*([\d.e+\-]+)',
                                         decodeTps=r'Token generation:.*?avg \(tokens/s\):\s*([\d.e+\-]+)').items():
                    match = re.search(pattern, process.stdout, re.DOTALL)
                    if match: row[key] = float(match[1])
                model['rows'].append(row)
            model['success'] = all(r['exitCode'] == 0 and r.get('decodeTps', 0) > 0 and r.get('prefillTps', 0) > 0 for r in model['rows'])
        result['models'].append(model); print(entry['name'], 'PASS' if model['success'] else model.get('errors'), flush=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    result['success'] = all(model['success'] for model in result['models'])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['success']: raise SystemExit(1)


if __name__ == '__main__': main()
