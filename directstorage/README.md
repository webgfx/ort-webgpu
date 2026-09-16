# Phi-4 DirectStorage and TTFT experiment

This experiment tests ONNX Runtime PR #32444 on the archived Phi-4-mini-instruct
INT4 model from `perf-trend`. See [REPORT.md](REPORT.md) for measurements and
their interpretation.

The [startup follow-up](STARTUP_REPORT.md) adds parallel tokenizer loading and
deferred GenAI allocator bootstrap, with independent timing and regression tests.

The original repositories and historical model packages are preserved. The
experiment uses detached worktrees at `E:\ort-wt\ds32444` and
`E:\ort-wt\ds-genai`; runtime copies, staged configurations, raw logs, and JSON
measurements are under:

```text
E:\workspace\project\ort-webgpu\gitignore\directstorage-phi4-20260906
```

## Runtime provenance

- ORT: `bebc6b807101a464b6455f40502b8fb1e0b66f4c`, the reviewed PR head.
- GenAI: `aa8fec1a9a6b8f9412bad5eadeaf53bbcec7cfb8`, with
  [genai-weight-load-forwarding.patch](genai-weight-load-forwarding.patch).
- `runtime/pr` contains the original ORT PR build.
- `runtime/cache-prototype` contains that ORT build plus the optional
  [Dawn cache patch](dawn-cache-prototype.patch). All final TTFT cases use the
  same patched GenAI binary. DLL hashes are recorded in `manifest.json`.

ORT was built as a Release shared library with native WebGPU, the D3D12 backend,
and `onnxruntime_ENABLE_WEBGPU_DIRECT_STORAGE=ON`. The exact build command is in
`manifest.json`. GenAI was built against that ORT SDK with `USE_WEBGPU=ON`.
Stage the complete public ORT session header bundle, including `.inc` files;
GenAI's SDK layout expects both `onnxruntime.lib` and `onnxruntime.dll` in `lib`.

The Python harness preloads `dstoragecore.dll` by absolute path. Merely adding
the runtime folder with `os.add_dll_directory()` did not let DirectStorage find
its core DLL when Python was the host executable. This behavior can be checked
with [directstorage_probe.cc](directstorage_probe.cc).

## Run the benchmark

The harness uses the native GenAI C API through `ctypes`, so it does not import
an ORT or GenAI Python wheel. Its only nonstandard runtime dependency is
`psutil`. Use the experiment's isolated virtual environment:

```powershell
$experimentRoot = 'E:\workspace\project\ort-webgpu\gitignore\directstorage-phi4-20260906'
$benchPython = "$experimentRoot\.venv\Scripts\python.exe"
$sourceModel = 'E:\backup\ort\perf-trend\model\aligned\standard-fused-rope\Phi-4-mini-instruct'

foreach ($case in @(@('off', 'off'), @('required', 'required'), @('pipelined', 'required-pipelined'))) {
    & $benchPython .\phi4_ttft.py stage --source $sourceModel `
        --destination "$experimentRoot\models\$($case[0])" --mode $case[1]
}

& $benchPython .\phi4_ttft.py suite `
    --runtime "$experimentRoot\runtime\pr" `
    --variant "off=$experimentRoot\models\off" `
    --variant "required=$experimentRoot\models\required" `
    --variant "pipelined=$experimentRoot\models\pipelined" `
    --prompt-lengths 1 128 1024 --repetitions 5 --warm-requests 2 `
    --prime-weights --output "$experimentRoot\rerun-baseline"
```

Staging hard-links immutable model assets and writes a new `genai_config.json`.
Do not edit the hard-linked weight files. The staged config disables Constant
Folding, selects the high-performance D3D12 adapter, and otherwise preserves the
archive's settings. Graph capture is off and validation is basic by default.

Each trial starts a fresh process, creates the model and tokenizer, runs a first
request with no inference warmup, and then runs two additional requests. The
default total sequence budget is 8192 tokens. Eight greedy output tokens are
retained for consistency checks. Trial order is randomized with a recorded seed.

`--prime-weights` reads each `.data` file immediately before launching the timed
process. That read is excluded from startup timing. This controls the preceding
file-access pattern; it is not a test with a purged OS or driver cache. The
separate `baseline` archive preserves the initial run without per-trial reads.

Startup TTFT uses the parent launch timestamp and the child's timestamp after
the generated token is accessible on CPU. The loaded-model request timer starts
before generator creation or reset, using an already encoded prompt. It includes
prefill, sampling, and token readback. Tokenization and model creation are
reported separately.

## Additional experiments

- `--reuse-variant NAME` keeps a generator and calls `RewindTo(0)` before later
  requests. The original 8K capacity is retained and the prompt is recomputed.
- `--max-length-variant NAME=2048` tests a smaller total sequence budget.
- `--cache-variant NAME=DIRECTORY` enables the prototype only for that case.
  Populate the cache in a separate priming run before measuring cache hits.
- `stage --threads`, `--spin`, `--optimization`, and `--graph-capture` create
  configuration variants. The archived `tuning-pilot` contains exploratory
  single-process results for these settings.
- [native_sessions.py](native_sessions.py) runs
  [ort_session_probe.cc](ort_session_probe.cc) to measure native ORT model loading
  without GenAI's bootstrap session. This probe uses one CPU thread and disables
  runtime graph optimization. Its inclusive `setup_and_session_ms` includes
  environment and EP setup as well as session creation.

To reproduce the cache build, run `apply_cache_prototype.py --ort <isolated-ORT-checkout>
--patch-output <output.patch>`, rebuild the `onnxruntime` target, and copy its DLL
to a separate runtime folder with the matching dependencies. The cache defaults
to disabled and is enabled by `ORT_WEBGPU_BENCH_CACHE_DIR`. It is a benchmark
prototype; production work should add cache-size limits, eviction, and stronger
multi-process coordination.

## Verification and evidence

- [summarize.py](summarize.py) computes medians/ranges and checks generated tokens
  across cases and repeated requests.
- [validate_reuse.py](validate_reuse.py) alternates a 128-token prompt and a
  different 10-token prompt, comparing reused and fresh generators.
- [dawn_disk_cache_test.cc](dawn_disk_cache_test.cc) tests misses, roundtrips,
  wrong keys/sizes, corruption, truncation, repair, and concurrent distinct keys.
  Build it against the same Dawn public and generated include directories.
- `native-pipelined-diagnostic.log` records the selected GPU, actual transferred
  bytes, DirectStorage request counts, and preparation/transfer/import timings.
- `manifest.json` records hardware, model provenance, verified model hashes,
  runtime hashes, and integration adjustments.
- Each measurement phase has per-process `.log` and `.json` files and a
  `suite.json` containing the exact invocation and trial order.

The experiment runs full-model generation and the targeted cache tests. It does
not claim that the full ORT or GenAI unit-test suites were run.
