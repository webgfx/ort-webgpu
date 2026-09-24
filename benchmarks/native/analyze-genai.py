"""Audit completed PTL comparison data and generate an evidence-backed report."""
import argparse
import json
from pathlib import Path
import statistics

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import compare
from web.model_manifest import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validated-only', action='store_true', help='Report only models that passed full GenAI conformance; disclose excluded models')
    parser.add_argument('--include-reverse', action='store_true', help='Aggregate an additional reverse-order block while retaining both block summaries')
    args = parser.parse_args()
    read = lambda name: json.loads((args.results / name).read_text('utf-8-sig'))
    cpu, gpu, reference = [read('correctness-' + arm + '.json') for arm in ('cpu', 'gpu', 'genai')]
    if not all(row['outputsMatch'] for row in compare(cpu, gpu)): raise ValueError('CPU/GPU correctness failed')
    conformance = compare(reference, gpu)
    excluded = [row for row in conformance if not row['outputsMatch']]
    if excluded and not args.validated_only: raise ValueError('GenAI conformance failed; use --validated-only to disclose and exclude divergent models')
    suffix = '-validated' if args.validated_only else ''
    native, genai, stock = [read(f'perf-{arm}{suffix}.json') for arm in ('native', 'genai', 'genai-stock')]
    rows = compare(genai, native)
    if not all(row['outputsMatch'] for row in rows): raise ValueError('Performance output mismatch')
    if any(row['model'] in {e['model'] for e in excluded} for row in rows): raise ValueError('An unvalidated model was included in performance claims')
    if native['artifacts']['onnxruntime.dll']['sha256'] != genai['artifacts']['onnxruntime.dll']['sha256']:
        raise ValueError('ORT DLLs differ')
    if stock['artifacts']['onnxruntime.dll'] != genai['artifacts']['onnxruntime.dll']:
        raise ValueError('Stock ORT DLL differs')
    if not stock['success']: raise ValueError('Stock comparison did not finish successfully')
    for model in native['models']:
        if model['executionEvidence']['device']['deviceId'] != 0xb080: raise ValueError('Not the designated Panther Lake GPU')
        for row in model['rows']:
            if len(row['samples']) != native['request']['repetitions']: raise ValueError('Incomplete repetitions')
    extra_inputs = []
    if args.include_reverse:
        native_reverse, genai_reverse = read('perf-native-reverse.json'), read('perf-genai-reverse.json')
        reverse_rows = compare(genai_reverse, native_reverse)
        for agreement in (compare(native, native_reverse), compare(genai, genai_reverse), reverse_rows):
            if not all(row['outputsMatch'] for row in agreement): raise ValueError('Reverse-order outputs differ')
        def summarize(blocks, model, length):
            samples = [sample for block in blocks for entry in block['models'] if entry['name'] == model
                       for point in entry['rows'] if point['pl'] == length for sample in point['samples']]
            return dict(ttftThroughput=length*1000/statistics.mean(s['ttftMs'] for s in samples),
                        decodeTps=(native['request']['generationLength']-1)*1000/statistics.mean(s['e2eMs']-s['ttftMs'] for s in samples),
                        decodeTpsMin=min(s['decodeTps'] for s in samples), decodeTpsMax=max(s['decodeTps'] for s in samples), samples=len(samples))
        for row in rows:
            reverse = next(r for r in reverse_rows if (r['model'], r['input']) == (row['model'], row['input']))
            row['initialPass'] = dict(left=row['left'], right=row['right'], decodeChangePercent=row['decodeChangePercent'])
            row['reversePass'] = dict(left=reverse['left'], right=reverse['right'], decodeChangePercent=reverse['decodeChangePercent'])
            row['left'] = summarize([genai, genai_reverse], row['model'], row['input'])
            row['right'] = summarize([native, native_reverse], row['model'], row['input'])
            row['decodeChangePercent'] = (row['right']['decodeTps']/row['left']['decodeTps']-1)*100
        extra_inputs = ['perf-native-reverse.json', 'perf-genai-reverse.json']
    for row in rows:
        stock_model = next(model for model in stock['models'] if model['name'] == row['model'])
        sample = next(sample for sample in stock_model['rows'] if sample['pl'] == row['input'])
        row['stockGenai'] = {key: sample[key] for key in ('prefillTps', 'decodeTps')}
        row['decodeChangeAgainstStockPercent'] = (row['right']['decodeTps'] / sample['decodeTps'] - 1) * 100
    result = dict(success=True, device=read('environment.json'), correctness=dict(modelsTested=len(conformance), modelsAccepted=len(rows), generatedTokens=128,
                  cpuGpuExact=True, genaiNativeExact=not excluded, excluded=excluded), ortBuild=native['build'], genaiBuild=genai['genaiBuild'],
                  nativeArtifacts=native['artifacts'], genaiArtifacts=genai['artifacts'], comparison=rows,
                  requestSha256=native['requestSha256'],
                  inputs={name: fingerprint(args.results / name) for name in ('correctness-cpu.json', 'correctness-gpu.json', 'correctness-genai.json', *[f'perf-{arm}{suffix}.json' for arm in ('native', 'genai', 'genai-stock')], *extra_inputs)},
                  limitations=['One device, warmed runs; small changes need repeated confirmation.',
                               'Matched means identical weights/runtime/prompts/timing, not identical generation implementations.',
                               'Native capture all models; GenAI Aion only. Native robustness enabled; GenAI initialization forwarding omission remains.',
                               'Stock GenAI uses different random prompts and excludes first sampling from its TTFT metric.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise ValueError('Preserve existing analysis')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('| Model | GenAI aligned TTFT TPS | Native TTFT TPS | GenAI aligned decode | Native decode | Decode change | Stock GenAI decode |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        print(f"| {row['model']} | {row['left']['ttftThroughput']:.2f} | {row['right']['ttftThroughput']:.2f} | {row['left']['decodeTps']:.2f} | {row['right']['decodeTps']:.2f} | {row['decodeChangePercent']:+.1f}% | {row['stockGenai']['decodeTps']:.2f} |")


if __name__ == '__main__': main()
