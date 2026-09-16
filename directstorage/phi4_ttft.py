"""Fresh-process Phi-4 TTFT measurements using the native ORT GenAI C API.

No Python ORT/GenAI wheel is imported: --runtime selects the exact native DLLs.
Model staging hard-links immutable assets and writes a separate GenAI config.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

P = C.c_void_p
PP = C.POINTER(P)
I32P = C.POINTER(C.c_int32)
ROOT = Path(__file__).resolve().parent


class GenAI:
    def __init__(self, runtime):
        self.dll_directory = os.add_dll_directory(str(runtime.resolve()))
        # DirectStorage's default core discovery is relative to the host executable.
        core_path = runtime / "dstoragecore.dll"
        self.dstorage_core = C.WinDLL(str(core_path)) if core_path.exists() else None
        self.ort = C.WinDLL(str(runtime / "onnxruntime.dll"))
        self.dll = C.WinDLL(str(runtime / "onnxruntime-genai.dll"))
        declarations = {
            "OgaResultGetError": (C.c_char_p, [P]),
            "OgaDestroyResult": (None, [P]),
            "OgaCreateModel": (P, [C.c_char_p, PP]),
            "OgaCreateTokenizer": (P, [P, PP]),
            "OgaCreateSequences": (P, [PP]),
            "OgaTokenizerEncode": (P, [P, C.c_char_p, P]),
            "OgaSequencesGetSequenceCount": (C.c_size_t, [P, C.c_size_t]),
            "OgaSequencesGetSequenceData": (I32P, [P, C.c_size_t]),
            "OgaCreateGeneratorParams": (P, [P, PP]),
            "OgaGeneratorParamsSetSearchNumber": (P, [P, C.c_char_p, C.c_double]),
            "OgaGeneratorParamsSetSearchBool": (P, [P, C.c_char_p, C.c_bool]),
            "OgaCreateGenerator": (P, [P, P, PP]),
            "OgaGenerator_AppendTokens": (P, [P, I32P, C.c_size_t]),
            "OgaGenerator_GenerateNextToken": (P, [P]),
            "OgaGenerator_RewindTo": (P, [P, C.c_size_t]),
            "OgaGenerator_GetSequenceCount": (C.c_size_t, [P, C.c_size_t]),
            "OgaGenerator_GetSequenceData": (I32P, [P, C.c_size_t]),
            "OgaTokenizerDecode": (P, [P, I32P, C.c_size_t, PP]),
            "OgaShutdown": (None, []),
        }
        for name in (
            "Model",
            "Tokenizer",
            "Sequences",
            "GeneratorParams",
            "Generator",
            "String",
        ):
            declarations["OgaDestroy" + name] = (None, [P])
        for name, (restype, argtypes) in declarations.items():
            function = getattr(self.dll, name)
            function.restype = restype
            function.argtypes = argtypes
            setattr(self, name, function)

    def check(self, result):
        if result:
            message = self.OgaResultGetError(result).decode("utf-8", errors="replace")
            self.OgaDestroyResult(result)
            raise RuntimeError(message)

    def create(self, name, *args):
        value = P()
        self.check(getattr(self, "OgaCreate" + name)(*args, C.byref(value)))
        return value

    def encode(self, tokenizer, text):
        sequences = self.create("Sequences")
        try:
            self.check(self.OgaTokenizerEncode(tokenizer, text.encode(), sequences))
            count = self.OgaSequencesGetSequenceCount(sequences, 0)
            return list(self.OgaSequencesGetSequenceData(sequences, 0)[:count])
        finally:
            self.OgaDestroySequences(sequences)

    def decode(self, tokenizer, tokens):
        data = (C.c_int32 * len(tokens))(*tokens)
        value = P()
        self.check(
            self.OgaTokenizerDecode(tokenizer, data, len(tokens), C.byref(value))
        )
        try:
            return C.string_at(value).decode("utf-8", errors="replace")
        finally:
            self.OgaDestroyString(value)


def prompt_ids(api, tokenizer, length):
    if length == 1:
        return api.encode(tokenizer, "Hello")[:1]
    prefix = api.encode(tokenizer, "<|user|>\n")
    suffix = api.encode(
        tokenizer, "\nSummarize the information above.<|end|>\n<|assistant|>\n"
    )
    body = api.encode(
        tokenizer,
        (
            "The city library opens at nine in the morning and closes at six in the evening. "
            "It offers books, quiet study rooms, computer access, and weekly science workshops. "
        )
        * (length // 16 + 2),
    )
    if length < len(prefix) + len(suffix):
        return (prefix + body)[:length]
    return prefix + body[: length - len(prefix) - len(suffix)] + suffix


def loaded_modules():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [C.c_wchar_p]
    kernel.GetModuleHandleW.restype = P
    kernel.GetModuleFileNameW.argtypes = [P, C.c_wchar_p, C.c_uint32]
    kernel.GetModuleFileNameW.restype = C.c_uint32
    paths = {}
    for name in (
        "onnxruntime.dll",
        "onnxruntime-genai.dll",
        "dstorage.dll",
        "dstoragecore.dll",
        "dxcompiler.dll",
        "dxil.dll",
        "d3d12.dll",
        "D3D12Core.dll",
    ):
        handle = kernel.GetModuleHandleW(name)
        if handle:
            buffer = C.create_unicode_buffer(32768)
            if kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
                paths[name] = buffer.value
    return paths


def worker(args):
    worker_start = time.perf_counter()
    api = GenAI(args.runtime)
    dll_end = time.perf_counter()
    model = tokenizer = params = generator = None
    result = {
        "runtime": str(args.runtime),
        "model": str(args.model),
        "prompt_tokens": args.prompt_tokens,
        "max_length": args.max_length,
        "pid": os.getpid(),
        "reuse_generator": args.reuse_generator,
        "startup_flags": {
            name: os.environ.get(name)
            for name in (
                "ORT_GENAI_BENCH_PRELOAD_TOKENIZER",
                "ORT_GENAI_BENCH_DEFER_DEVICE_INIT",
            )
        },
        "requests": [],
    }
    try:
        model_start = time.perf_counter()
        model = api.create("Model", str(args.model).encode())
        model_end = time.perf_counter()
        tokenizer = api.create("Tokenizer", model)
        tokenizer_end = time.perf_counter()
        ids = prompt_ids(api, tokenizer, args.prompt_tokens)
        assert len(ids) == args.prompt_tokens, len(ids)
        ids_array = (C.c_int32 * len(ids))(*ids)
        encode_end = time.perf_counter()
        result.update(
            dll_ms=(dll_end - worker_start) * 1000,
            model_ms=(model_end - model_start) * 1000,
            tokenizer_ms=(tokenizer_end - model_end) * 1000,
            encode_ms=(encode_end - tokenizer_end) * 1000,
            input_ids_sha256=hashlib.sha256(bytes(ids_array)).hexdigest(),
        )
        for request_index in range(args.warm_requests + 1):
            request_start = time.perf_counter()
            try:
                if generator is None:
                    params = api.create("GeneratorParams", model)
                    api.check(
                        api.OgaGeneratorParamsSetSearchNumber(
                            params, b"max_length", args.max_length
                        )
                    )
                    api.check(
                        api.OgaGeneratorParamsSetSearchBool(params, b"do_sample", False)
                    )
                    generator = api.create("Generator", model, params)
                else:
                    api.check(api.OgaGenerator_RewindTo(generator, 0))
                    assert api.OgaGenerator_GetSequenceCount(generator, 0) == 0
                generator_end = time.perf_counter()
                api.check(api.OgaGenerator_AppendTokens(generator, ids_array, len(ids)))
                append_end = time.perf_counter()
                api.check(api.OgaGenerator_GenerateNextToken(generator))
                generate_end = time.perf_counter()
                count = api.OgaGenerator_GetSequenceCount(generator, 0)
                tokens = api.OgaGenerator_GetSequenceData(generator, 0)
                first_token = int(tokens[count - 1])
                token_end = time.perf_counter()
                first_text = api.decode(tokenizer, [first_token])
                text_end = time.perf_counter()
                if request_index == 0:
                    result["worker_to_first_token_ms"] = (
                        token_end - worker_start
                    ) * 1000
                    result["launch_to_first_token_ms"] = (
                        (token_end - args.launch_qpc) * 1000
                        if args.launch_qpc
                        else None
                    )
                    result["launch_to_first_text_ms"] = (
                        (text_end - args.launch_qpc) * 1000 if args.launch_qpc else None
                    )
                    print(
                        "FIRST_TOKEN "
                        + json.dumps(
                            {
                                "token": first_token,
                                "text": first_text,
                                "launch_to_first_token_ms": result[
                                    "launch_to_first_token_ms"
                                ],
                            }
                        ),
                        flush=True,
                    )
                output = [first_token]
                for _ in range(args.generate_tokens - 1):
                    api.check(api.OgaGenerator_GenerateNextToken(generator))
                    count = api.OgaGenerator_GetSequenceCount(generator, 0)
                    tokens = api.OgaGenerator_GetSequenceData(generator, 0)
                    output.append(int(tokens[count - 1]))
                request = {
                    "index": request_index,
                    "generator_ms": (generator_end - request_start) * 1000,
                    "append_ms": (append_end - generator_end) * 1000,
                    "generate_first_ms": (generate_end - append_end) * 1000,
                    "read_first_ms": (token_end - generate_end) * 1000,
                    "ttft_ms": (token_end - request_start) * 1000,
                    "output_tokens": output,
                    "output_text": api.decode(tokenizer, output),
                }
                result["requests"].append(request)
            finally:
                if not args.reuse_generator:
                    if generator:
                        api.OgaDestroyGenerator(generator)
                        generator = None
                    if params:
                        api.OgaDestroyGeneratorParams(params)
                        params = None
        # Import monitoring code after the measured requests.
        import psutil

        process = psutil.Process()
        result["memory"] = process.memory_info()._asdict()
        result["cpu_times"] = process.cpu_times()._asdict()
        result["loaded_modules"] = loaded_modules()
        result["configuration"] = json.loads(
            (args.model / "genai_config.json").read_text()
        )
    finally:
        if generator:
            api.OgaDestroyGenerator(generator)
        if params:
            api.OgaDestroyGeneratorParams(params)
        if tokenizer:
            api.OgaDestroyTokenizer(tokenizer)
        if model:
            api.OgaDestroyModel(model)
        api.OgaShutdown()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)


def stage(args):
    args.destination.mkdir(parents=True, exist_ok=True)
    for source in args.source.iterdir():
        if source.is_file() and source.name not in (
            "genai_config.json",
            "model-metadata.json",
        ):
            destination = args.destination / source.name
            if not destination.exists():
                os.link(source, destination)
    config = json.loads((args.source / "genai_config.json").read_text(encoding="utf-8"))
    session = config["model"]["decoder"]["session_options"]
    provider = next(
        options[key]
        for options in session["provider_options"]
        for key in options
        if key.lower() == "webgpu"
    )
    provider.update(
        weightLoadAcceleration=args.mode,
        powerPreference="high-performance",
        dawnBackendType="D3D12",
    )
    provider["enableGraphCapture"] = str(args.graph_capture)
    provider["validationMode"] = args.validation
    session["optimization.disable_specified_optimizers"] = "ConstantFolding"
    session["log_severity_level"] = args.log_severity
    if args.optimization:
        session["graph_optimization_level"] = args.optimization
    if args.threads:
        session["intra_op_num_threads"] = args.threads
    if args.spin is not None:
        session["session.intra_op.allow_spinning"] = str(args.spin)
    if args.profile:
        session["enable_profiling"] = str(args.profile)
    (args.destination / "genai_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(args.destination)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=True)
    variants = [item.split("=", 1) for item in args.variant]
    cache_variants = dict(item.split("=", 1) for item in args.cache_variant)
    max_lengths = dict(item.split("=", 1) for item in args.max_length_variant)
    runtimes = dict(item.split("=", 1) for item in args.runtime_variant)
    env_variants = {}
    for value in args.env_variant:
        name, key, setting = value.split("=", 2)
        env_variants.setdefault(name, {})[key] = setting
    jobs = [
        (repeat, name, Path(model), length)
        for repeat in range(args.repetitions)
        for name, model in variants
        for length in args.prompt_lengths
    ]
    random.Random(args.seed).shuffle(jobs)
    metadata = {
        "seed": args.seed,
        "variants": variants,
        "runtime": str(args.runtime),
        "cache_variants": cache_variants,
        "runtime_variants": runtimes,
        "environment_variants": env_variants,
        "argv": sys.argv,
        "runs": [],
    }
    for index, (repeat, name, model, length) in enumerate(jobs):
        stem = f"{index:03d}-{name}-p{length}-r{repeat}"
        output = args.output / (stem + ".json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--runtime",
            str(runtimes.get(name, args.runtime)),
            "--model",
            str(model),
            "--prompt-tokens",
            str(length),
            "--max-length",
            str(max_lengths.get(name, args.max_length)),
            "--warm-requests",
            str(args.warm_requests),
            "--generate-tokens",
            str(args.generate_tokens),
            "--output",
            str(output),
        ]
        if name in args.reuse_variant:
            command.append("--reuse-generator")
        environment = os.environ.copy()
        environment.pop("ORT_WEBGPU_BENCH_CACHE_DIR", None)
        for key in (
            "ORT_GENAI_BENCH_PRELOAD_TOKENIZER",
            "ORT_GENAI_BENCH_DEFER_DEVICE_INIT",
            "ORT_GENAI_BENCH_STARTUP_TRACE",
        ):
            environment.pop(key, None)
        environment.update(env_variants.get(name, {}))
        cache_directory = Path(cache_variants[name]) if name in cache_variants else None
        previous_stats = set()
        if cache_directory:
            cache_directory.mkdir(parents=True, exist_ok=True)
            previous_stats = set(cache_directory.glob("stats-*.json"))
            environment["ORT_WEBGPU_BENCH_CACHE_DIR"] = str(cache_directory)
        primed_bytes = 0
        prime_start = time.perf_counter()
        if args.prime_weights:
            buffer = bytearray(16 * 1024 * 1024)
            for weight_file in model.glob("*.data"):
                with weight_file.open("rb", buffering=0) as stream:
                    while count := stream.readinto(buffer):
                        primed_bytes += count
            del buffer
        prime_ms = (time.perf_counter() - prime_start) * 1000
        with (args.output / (stem + ".log")).open("w", encoding="utf-8") as log:
            launch = time.perf_counter()
            command += ["--launch-qpc", str(launch)]
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
                env=environment,
            )
        run = {
            "name": name,
            "prompt_tokens": length,
            "repeat": repeat,
            "returncode": completed.returncode,
            "process_ms": (time.perf_counter() - launch) * 1000,
            "output": str(output),
            "runtime": str(runtimes.get(name, args.runtime)),
            "primed_bytes": primed_bytes,
            "weight_priming_ms": prime_ms,
        }
        if completed.returncode == 0 and output.exists():
            data = json.loads(output.read_text())
            run.update(
                model_ms=data["model_ms"],
                launch_to_first_token_ms=data["launch_to_first_token_ms"],
                cold_ttft_ms=data["requests"][0]["ttft_ms"],
                warm_ttft_ms=statistics.median(
                    x["ttft_ms"] for x in data["requests"][1:]
                )
                if args.warm_requests
                else None,
            )
        if cache_directory:
            run["cache_stats"] = [
                json.loads(path.read_text())
                for path in sorted(
                    set(cache_directory.glob("stats-*.json")) - previous_stats
                )
            ]
        metadata["runs"].append(run)
        (args.output / "suite.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(run), flush=True)
        if completed.returncode:
            raise RuntimeError(f"Benchmark failed; inspect {stem}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("stage")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--mode", default="off")
    p.add_argument(
        "--optimization",
        choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_ALL"],
    )
    p.add_argument("--threads", type=int)
    p.add_argument("--spin", type=int, choices=[0, 1])
    p.add_argument("--graph-capture", type=int, default=0, choices=[0, 1])
    p.add_argument("--validation", default="basic")
    p.add_argument("--log-severity", type=int, default=2)
    p.add_argument("--profile", type=Path)
    p.set_defaults(func=stage)
    for name in ("worker", "suite"):
        p = commands.add_parser(name)
        p.add_argument("--runtime", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--max-length", type=int, default=8192)
        p.add_argument("--warm-requests", type=int, default=2)
        p.add_argument("--generate-tokens", type=int, default=8)
        if name == "worker":
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--prompt-tokens", type=int, default=128)
            p.add_argument("--launch-qpc", type=float, default=0)
            p.add_argument("--reuse-generator", action="store_true")
            p.set_defaults(func=worker)
        else:
            p.add_argument(
                "--variant", action="append", required=True, help="NAME=MODEL_DIRECTORY"
            )
            p.add_argument(
                "--cache-variant",
                action="append",
                default=[],
                help="NAME=CACHE_DIRECTORY",
            )
            p.add_argument(
                "--prompt-lengths", type=int, nargs="+", default=[1, 128, 1024]
            )
            p.add_argument("--repetitions", type=int, default=5)
            p.add_argument("--seed", type=int, default=32444)
            p.add_argument("--timeout", type=int, default=180)
            p.add_argument("--prime-weights", action="store_true")
            p.add_argument(
                "--max-length-variant",
                action="append",
                default=[],
                help="NAME=MAX_LENGTH",
            )
            p.add_argument("--reuse-variant", action="append", default=[])
            p.add_argument(
                "--runtime-variant",
                action="append",
                default=[],
                help="NAME=RUNTIME_DIRECTORY",
            )
            p.add_argument(
                "--env-variant", action="append", default=[], help="NAME=KEY=VALUE"
            )
            p.set_defaults(func=suite)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
