"""Create a shared, hashed workload; never download or modify original models."""
import argparse
import json
from pathlib import Path
from web.model_manifest import describe_model


def random_prompt(config, length, seed=42):
    model = config['model']
    forbidden = set()
    for key, value in model.items():
        if key.endswith('_token_id'):
            forbidden.update(value if isinstance(value, list) else [value])
    vocabulary = model['vocab_size']
    if vocabulary <= len(forbidden):
        raise ValueError('No usable vocabulary')
    state, tokens = seed & 0xffffffff, []
    while len(tokens) < length:
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state = state or 0x9e3779b9
        token = state % vocabulary
        if token not in forbidden:
            tokens.append(token)
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--models', default='aion,Phi-4-mini-instruct,Qwen3.5-2B,Qwen3.5-4B,gemma-4-e2b-it')
    parser.add_argument('--prompt-lengths', default='128,512,1024')
    parser.add_argument('--generation-length', type=int, default=128)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lengths = list(map(int, args.prompt_lengths.split(',')))
    names = args.models.split(',')
    if (len(set(names)) != len(names) or len(set(lengths)) != len(lengths)
            or min(lengths) <= 0 or args.generation_length < 2 or args.repetitions < 1
            or max(lengths) + args.generation_length > args.max_length):
        parser.error('Invalid or duplicate workload dimensions')
    if args.output.exists():
        parser.error('Output already exists; preserve previous experiment requests')
    data = dict(schemaVersion=1, models=[], promptLengths=lengths, generationLength=args.generation_length,
                repetitions=args.repetitions, maxLength=args.max_length, seed=args.seed,
                prefillChunkSize=0, batchSize=1, sampling='greedy', suppressEos=True)
    for name in names:
        root = (args.models_root / name / 'onnx-webgpu').resolve()
        manifest = describe_model(root, name, args.max_length)
        data['models'].append(dict(name=name, root=str(root), manifest=manifest,
                                   cases=[dict(pl=pl, prompt=random_prompt(manifest['config'], pl, args.seed)) for pl in lengths]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
