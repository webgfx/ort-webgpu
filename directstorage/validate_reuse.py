"""Check RewindTo(0) with different prompt contents and sequence lengths."""

import argparse
import ctypes as C
import json
from pathlib import Path

from phi4_ttft import GenAI, prompt_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = GenAI(args.runtime)
    handles = []

    def create(name, *values):
        handle = api.create(name, *values)
        handles.append((name, handle))
        return handle

    try:
        model = create("Model", str(args.model).encode())
        tokenizer = create("Tokenizer", model)
        first_ids = prompt_ids(api, tokenizer, 128)
        second_ids = api.encode(
            tokenizer,
            "<|user|>\nExplain how rain forms in detail.<|end|>\n<|assistant|>\n",
        )

        def new_generator():
            params = create("GeneratorParams", model)
            api.check(
                api.OgaGeneratorParamsSetSearchNumber(params, b"max_length", 8192)
            )
            api.check(api.OgaGeneratorParamsSetSearchBool(params, b"do_sample", False))
            return create("Generator", model, params)

        def generate(generator, ids):
            values = (C.c_int32 * len(ids))(*ids)
            api.check(api.OgaGenerator_AppendTokens(generator, values, len(ids)))
            tokens = []
            for _ in range(8):
                api.check(api.OgaGenerator_GenerateNextToken(generator))
                count = api.OgaGenerator_GetSequenceCount(generator, 0)
                tokens.append(
                    int(api.OgaGenerator_GetSequenceData(generator, 0)[count - 1])
                )
            return tokens

        reused = new_generator()
        first = generate(reused, first_ids)
        api.check(api.OgaGenerator_RewindTo(reused, 0))
        assert api.OgaGenerator_GetSequenceCount(reused, 0) == 0
        second_reused = generate(reused, second_ids)
        second_fresh = generate(new_generator(), second_ids)
        assert second_reused == second_fresh, (second_reused, second_fresh)
        api.check(api.OgaGenerator_RewindTo(reused, 0))
        first_reused = generate(reused, first_ids)
        assert first_reused == first, (first_reused, first)
        result = {
            "passed": True,
            "prompt_lengths": [len(first_ids), len(second_ids)],
            "first_tokens": first,
            "second_reused_tokens": second_reused,
            "second_fresh_tokens": second_fresh,
            "first_reused_tokens": first_reused,
            "first_text": api.decode(tokenizer, first),
            "second_text": api.decode(tokenizer, second_reused),
        }
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    finally:
        for name, handle in reversed(handles):
            getattr(api, "OgaDestroy" + name)(handle)
        api.OgaShutdown()


if __name__ == "__main__":
    main()
