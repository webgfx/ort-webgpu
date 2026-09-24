"""Independently validate paired raw samples, then compare mean elapsed time.

An output difference is reported explicitly, never hidden behind a throughput
average. Missing coverage, changed workloads and invalid timing fail the report.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def validate(data):
    if data.get('success') is not True:
        raise ValueError('Benchmark failed: ' + str(data.get('errors')))
    request = data['request']
    models = {m['name']: m for m in data['models']}
    if len(models) != len(data['models']) or set(models) != {m['name'] for m in request['models']}:
        raise ValueError('Incomplete or duplicate model coverage')
    points = {}
    for entry in request['models']:
        model = models[entry['name']]
        if not model.get('success'):
            raise ValueError('Failed model: ' + entry['name'])
        rows = {r['pl']: r for r in model['rows']}
        if len(rows) != len(model['rows']) or set(rows) != {c['pl'] for c in entry['cases']}:
            raise ValueError('Missing or duplicate input lengths')
        for case in entry['cases']:
            row = rows[case['pl']]
            if row['prompt'] != case['prompt'] or len(row['samples']) != request['repetitions']:
                raise ValueError('Prompt or repetitions mismatch')
            expected = row['warmup']['generated']
            for sample in [row['warmup'], *row['samples']]:
                if len(sample['generated']) != request['generationLength'] or sample['generated'] != expected:
                    raise ValueError('Nonrepeatable or incomplete generation')
                if any(not isinstance(sample[k], (float, int)) or not math.isfinite(sample[k]) or sample[k] <= 0
                       for k in ('ttftMs', 'e2eMs', 'prefillTps', 'decodeTps')):
                    raise ValueError('Invalid timing')
                decode = sample['e2eMs'] - sample['ttftMs']
                if decode <= 0 or not math.isclose(sample['decodeTps'], (request['generationLength']-1)*1000/decode, rel_tol=1e-6):
                    raise ValueError('Decode timing/count mismatch')
                if not math.isclose(sample['prefillTps'], case['pl']*1000/sample['ttftMs'], rel_tol=1e-6):
                    raise ValueError('TTFT timing/count mismatch')
                delivery = sample.get('tokenDeliveryMs')
                if model.get('executionEvidence', {}).get('samplingDevice') == 'gpu' and delivery is None:
                    raise ValueError('GPU sample lacks CPU delivery evidence')
                if delivery is not None:
                    if (len(delivery) != request['generationLength'] or delivery[0] != sample['ttftMs']
                            or any(not math.isfinite(t) or t <= 0 for t in delivery)
                            or any(a > b for a, b in zip(delivery, delivery[1:])) or delivery[-1] > sample['e2eMs']):
                        raise ValueError('Invalid CPU delivery timing')
            points[(entry['name'], case['pl'])] = row
    return points


def compare(left, right):
    if left['requestSha256'] != right['requestSha256'] or left['request'] != right['request']:
        raise ValueError('Requests, models, or prompts differ')
    a, b = validate(left), validate(right)
    def revision(data):
        return data.get('build', {}).get('repositories', {}).get('onnxruntime', {}).get('commit')
    if revision(left) and revision(right) and revision(left) != revision(right):
        raise ValueError('ORT source revisions differ; this is not a matched runtime comparison')
    for ma, mb in zip(sorted(left['models'], key=lambda m: m['name']), sorted(right['models'], key=lambda m: m['name'])):
        da, db = [m.get('executionEvidence', {}).get('device') for m in (ma, mb)]
        if da and db and (da['vendorId'], da['deviceId']) != (db['vendorId'], db['deviceId']):
            raise ValueError('Actual GPU selections differ')
    rows = []
    for key in a:
        first = next((i for i, (x, y) in enumerate(zip(a[key]['warmup']['generated'], b[key]['warmup']['generated'])) if x != y), None)
        values = {}
        for label, row in [('left', a[key]), ('right', b[key])]:
            decode = [s['e2eMs']-s['ttftMs'] for s in row['samples']]
            values[label] = dict(ttftThroughput=key[1]*1000/statistics.mean(s['ttftMs'] for s in row['samples']),
                                 decodeTps=(left['request']['generationLength']-1)*1000/statistics.mean(decode),
                                 decodeMsMin=min(decode), decodeMsMax=max(decode))
        rows.append(dict(model=key[0], input=key[1], outputsMatch=first is None,
                         firstDifferentToken=first, **values,
                         decodeChangePercent=(values['right']['decodeTps']/values['left']['decodeTps']-1)*100))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    left, right = [json.loads(p.read_text('utf-8-sig')) for p in (args.left, args.right)]
    rows = compare(left, right)
    report = dict(left=left['runtime'], right=right['runtime'], requestSha256=left['requestSha256'],
                  allOutputsMatch=all(r['outputsMatch'] for r in rows), rows=rows)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('| Model | Input | Left TTFT TPS | Right TTFT TPS | Left decode TPS | Right decode TPS | Decode change | Outputs |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for row in rows:
        print(f"| {row['model']} | {row['input']} | {row['left']['ttftThroughput']:.2f} | {row['right']['ttftThroughput']:.2f} | "
              f"{row['left']['decodeTps']:.2f} | {row['right']['decodeTps']:.2f} | {row['decodeChangePercent']:+.1f}% | "
              f"{'match' if row['outputsMatch'] else 'DIFFER at '+str(row['firstDifferentToken'])} |")


if __name__ == '__main__':
    main()
