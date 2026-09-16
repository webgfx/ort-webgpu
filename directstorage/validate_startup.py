"""Exercise error propagation and lifetimes for the startup-overlap prototype."""

import argparse
import ctypes as C
import json
import os
import subprocess
import sys
from pathlib import Path

from phi4_ttft import GenAI, P, prompt_ids


def worker(args):
    api = GenAI(args.runtime)
    handles = []
    details = {}

    def create(name, *values):
        handle = api.create(name, *values)
        handles.append((name, handle))
        return handle

    def destroy(name, handle):
        getattr(api, "OgaDestroy" + name)(handle)
        handles.remove((name, handle))

    def expect_error(operation):
        try:
            operation()
        except RuntimeError as error:
            return str(error)
        raise AssertionError("Expected a failure")

    def generate(model, tokenizer, release_model=False):
        ids = prompt_ids(api, tokenizer, 128)
        params = create("GeneratorParams", model)
        api.check(api.OgaGeneratorParamsSetSearchNumber(params, b"max_length", 8192))
        api.check(api.OgaGeneratorParamsSetSearchBool(params, b"do_sample", False))
        generator = create("Generator", model, params)
        if release_model:
            destroy("Model", model)
        array = (C.c_int32 * len(ids))(*ids)
        api.check(api.OgaGenerator_AppendTokens(generator, array, len(ids)))
        tokens = []
        for _ in range(8):
            api.check(api.OgaGenerator_GenerateNextToken(generator))
            count = api.OgaGenerator_GetSequenceCount(generator, 0)
            tokens.append(
                int(api.OgaGenerator_GetSequenceData(generator, 0)[count - 1])
            )
        assert tokens == [976, 5030, 11282, 382, 2494, 591, 220, 24], tokens
        destroy("Generator", generator)
        destroy("GeneratorParams", params)
        return tokens

    try:
        if args.case == "tokenizer-independence":
            model = create("Model", str(args.model).encode())
            first = create("Tokenizer", model)
            second = create("Tokenizer", model)
            text = "Hello 世界 café 😀 <|end|>"
            ids = api.encode(first, text)
            assert api.encode(second, text) == ids
            initial = api.decode(second, ids)
            update = api.dll.OgaUpdateTokenizerOptions
            update.argtypes = [
                P,
                C.POINTER(C.c_char_p),
                C.POINTER(C.c_char_p),
                C.c_size_t,
            ]
            update.restype = P
            keys = (C.c_char_p * 1)(b"skip_special_tokens")
            values = (C.c_char_p * 1)(b"false")
            api.check(update(first, keys, values, 1))
            changed = api.decode(first, ids)
            assert changed != initial and "<|end|>" in changed
            assert api.decode(second, ids) == initial
            destroy("Model", model)
            assert api.encode(first, text) == ids
            assert api.decode(second, ids) == initial
            details.update(
                ids=ids,
                default_decode=initial,
                modified_decode=changed,
                tokenizer_survives_model=True,
            )
        elif args.case == "generator-survives-model":
            model = create("Model", str(args.model).encode())
            tokenizer = create("Tokenizer", model)
            details["tokens_after_model_release"] = generate(
                model, tokenizer, release_model=True
            )
        elif args.case == "unused-tokenizer":
            model = create("Model", str(args.model).encode())
            destroy("Model", model)
            # A second model exercises the lifetime of the retained global device allocator.
            model = create("Model", str(args.model).encode())
            tokenizer = create("Tokenizer", model)
            details["recovery_tokens"] = generate(model, tokenizer)
        elif args.case == "missing-tokenizer":
            model = create("Model", str(args.bad_model).encode())
            details["first_error"] = expect_error(
                lambda: api.create("Tokenizer", model)
            )
            details["second_error"] = expect_error(
                lambda: api.create("Tokenizer", model)
            )
            destroy("Model", model)
            model = create("Model", str(args.model).encode())
            tokenizer = create("Tokenizer", model)
            details["recovery_tokens"] = generate(model, tokenizer)
        elif args.case in ("missing-model", "missing-weights"):
            details["error"] = expect_error(
                lambda: api.create("Model", str(args.bad_model).encode())
            )
            model = create("Model", str(args.model).encode())
            tokenizer = create("Tokenizer", model)
            details["recovery_tokens"] = generate(model, tokenizer)
        else:
            raise ValueError(args.case)
    finally:
        for name, handle in reversed(handles):
            getattr(api, "OgaDestroy" + name)(handle)
        api.OgaShutdown()
    result = {"case": args.case, "passed": True, "details": details}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


def suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for enabled in (False, True):
        for case in (
            "tokenizer-independence",
            "unused-tokenizer",
            "generator-survives-model",
            "missing-tokenizer",
            "missing-model",
            "missing-weights",
        ):
            stem = ("both" if enabled else "control") + "-" + case
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--runtime",
                str(args.runtime),
                "--model",
                str(args.model),
                "--case",
                case,
                "--output",
                str(args.output / (stem + ".json")),
            ]
            if case.startswith("missing"):
                command += ["--bad-model", str(args.bad_models / case)]
            env = os.environ.copy()
            for name in (
                "ORT_GENAI_BENCH_PRELOAD_TOKENIZER",
                "ORT_GENAI_BENCH_DEFER_DEVICE_INIT",
            ):
                env[name] = "1" if enabled else "0"
            with (args.output / (stem + ".log")).open("wb") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=90,
                    check=False,
                )
            row = {"case": case, "enabled": enabled, "returncode": process.returncode}
            results.append(row)
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(json.dumps(row), flush=True)
            if process.returncode:
                raise RuntimeError(f"Validation failed: {stem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("worker", "suite"):
        p = subparsers.add_parser(action)
        p.add_argument("--runtime", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if action == "worker":
            p.add_argument("--case", required=True)
            p.add_argument("--bad-model", type=Path)
            p.set_defaults(func=worker)
        else:
            p.add_argument("--bad-models", type=Path, required=True)
            p.set_defaults(func=suite)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
