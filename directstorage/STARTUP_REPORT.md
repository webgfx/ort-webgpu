# Phi-4 startup overlap: follow-up results

The new GenAI prototype reduced startup-to-first-token from **1.561 s to
1.137 s** for a 128-token prompt: **27.2% lower**, on top of
the previously tested DirectStorage and persistent shader cache. It overlaps
tokenizer construction with model loading and defers GenAI's allocator bootstrap
until the real decoder session has loaded.

This continues the [initial DirectStorage study](REPORT.md). The same
Phi-4-mini-instruct INT4 export, 2.49 GB of external weights, RTX 5080, driver
616.56, NVMe drive, and Windows 11 system were used. The model files and ORT
binary did not change in this follow-up. The GenAI baseline is the previous
allocator-option-forwarding build; the new patch is
[genai-startup-overlap.patch](genai-startup-overlap.patch).

## Independent and combined effects

The table reports five fresh processes per cell, in randomized order. All four
variants use the same candidate GenAI DLL and the same populated Dawn cache.
Each process has an 8K sequence budget and performs its first inference without
an in-process warmup. Weights are read immediately before each timed launch;
that pre-read is outside the measured interval. OS/driver caches are preserved.

| Prompt tokens | Control | Parallel tokenizer | Deferred bootstrap | Both |
|---:|---:|---:|---:|---:|
| 1 | 1.518 s | 1.146 s | 1.486 s | 1.078 s |
| 128 | 1.561 s | 1.168 s | 1.519 s | 1.137 s |
| 1024 | 1.735 s | 1.363 s | 1.713 s | 1.311 s |

These are complete process-start-to-CPU-token-availability measurements. They
include all tokenizer work needed for the request: the first tokenizer call
consumes the future and waits if necessary. Moving work to a thread does not
remove it from the measured interval.

For 128 prompt tokens, parallel tokenizer loading contributed approximately
400 ms of improvement. Deferring the bootstrap contributed roughly 40 to 60 ms.
Warm-request latency remained approximately 68 ms with a new generator.

The primary control-versus-combined ranges were:

| Prompt tokens | Variant | Observed startup range |
|---:|---|---:|
| 1 | control | 1.513 to 1.544 s |
| 1 | both | 1.071 to 1.155 s |
| 128 | control | 1.547 to 1.569 s |
| 128 | both | 1.112 to 1.151 s |
| 1024 | control | 1.729 to 1.752 s |
| 1024 | both | 1.307 to 1.337 s |

## What changed

**Tokenizer loading.** A worker starts after session options are prepared and
loads a tokenizer from an independent configuration snapshot. The tokenizer
directory is resolved on the caller thread so a package resolver is not shared
concurrently with model setup. The first `CreateTokenizer` call consumes the
preloaded object; subsequent calls continue creating independent tokenizer
objects, preserving their mutable option semantics. Exceptions are delivered
when the tokenizer is requested, and future lifetime management joins the
worker on failure or teardown.

The caller's tokenizer wait fell from roughly 400 ms to 0.04 ms because parsing
finished during model loading. Parsing itself still takes CPU time. The
diagnostic trace showed tokenizer loading from approximately 0.5 to 553 ms
relative to the GenAI base constructor, with the decoder ready around 828 ms.

**Allocator bootstrap.** GenAI normally creates a trivial session first to obtain
a device allocator with process-wide lifetime. The prototype keeps that session
and its ownership contract, but creates it after the real decoder session.
This lets the real model initiate DirectStorage loading while Dawn initializes.
In the diagnostic trace, the retained bootstrap followed decoder readiness and
took about 3.4 ms. Only the decoder-only WebGPU path opts into this deferral.

Both behaviors are opt-in benchmark flags, disabled by default. Public C API
signatures remain unchanged.

## Compatibility with the previous optimizations

A separate 25-process randomized comparison used a 128-token prompt and five
processes per variant. It included the previous GenAI binary, disabled prototype
flags, generator reuse, and runs with the explicit Dawn disk cache disabled.

| Variant | Startup to first token | Later request |
|---|---:|---:|
| Previous GenAI binary, shader cache | 1.603 s | 68.5 ms |
| New binary, startup flags off, shader cache | 1.586 s | 68.5 ms |
| Both startup changes + shader cache + generator reuse | 1.118 s | 40.1 ms |
| Startup flags off, no Dawn disk cache | 1.708 s | 69.1 ms |
| Both startup changes, no Dawn disk cache | 1.238 s | 69.4 ms |

The startup improvements also worked without the shader cache. Reusing a
generator with `RewindTo(0)` retained the prior approximately 40 ms warm-request
latency and the 8K capacity. An additional mixed-prompt test alternated 128-token
and 10-token inputs and matched fresh-generator outputs.

## Validation and tradeoffs

There were **85 timed processes and 255 requests** in this follow-up, with no
tokenization or greedy-output mismatches against the original study. Loaded DLL
paths and hashes were checked, and all 75 cache-enabled processes reported
expected cache hits with zero misses. The remaining ten processes intentionally
disabled the explicit Dawn disk cache.

Twelve control/candidate regression processes checked:

- independent tokenizer options and tokenizers surviving model-handle release;
- model destruction without requesting the speculative tokenizer;
- missing tokenizer files, including repeated errors and subsequent recovery;
- missing ONNX files and external weights, followed by successful model loading;
- generators continuing to run after their model handle is released.

The primary combined case increased peak process working set by approximately
**65.8 MiB** at 128 prompt tokens. Overlap changes when memory and CPU
work are used; it does not eliminate tokenizer parsing. These are native Windows
Phi-4 measurements, and the explicit shader cache was already populated. This is
not a purged-cache measurement or a general model-quality evaluation. Full
ORT/GenAI test suites were not run.

The patch is experimental. Production work should make tokenizer preloading an
explicit application intent, handle worker-launch resource failures gracefully,
and extend coverage to other models, platforms, and package configurations.

## Reproduce

Apply [genai-weight-load-forwarding.patch](genai-weight-load-forwarding.patch)
and then [genai-startup-overlap.patch](genai-startup-overlap.patch) to GenAI commit
`aa8fec1a9a6b8f9412bad5eadeaf53bbcec7cfb8`. Rebuild against the ORT PR runtime
described in [README.md](README.md). The tested candidate DLL is archived under
`../gitignore/directstorage-phi4-20260906/startup-followup/runtime/`.

Enable the independent flags only in the candidate process:

```text
ORT_GENAI_BENCH_PRELOAD_TOKENIZER=1
ORT_GENAI_BENCH_DEFER_DEVICE_INIT=1
```

The benchmark runner supports per-variant settings through
`--env-variant NAME=KEY=VALUE`, and different binaries through
`--runtime-variant NAME=RUNTIME_DIRECTORY`. It clears startup flags between
variants. `--reuse-variant NAME` enables generator reuse.

For example, with `$experimentRoot` and `$benchPython` defined as in the README:

```powershell
& $benchPython .\phi4_ttft.py suite `
    --runtime "$experimentRoot\startup-followup\runtime" `
    --variant "control=$experimentRoot\models\pipelined" `
    --variant "both=$experimentRoot\models\pipelined" `
    --cache-variant "control=$experimentRoot\shader-cache" `
    --cache-variant "both=$experimentRoot\shader-cache" `
    --env-variant "both=ORT_GENAI_BENCH_PRELOAD_TOKENIZER=1" `
    --env-variant "both=ORT_GENAI_BENCH_DEFER_DEVICE_INIT=1" `
    --prompt-lengths 1 128 1024 --repetitions 5 --warm-requests 2 `
    --prime-weights --output "$experimentRoot\startup-followup\rerun"
```

The `startup-followup` evidence directory contains `manifest.json`, `summary.json`,
`timing/suite.json`, `integration/suite.json`, raw per-process logs, `*.trace`
diagnostics, and regression results under `validation/`. Optional
`ORT_GENAI_BENCH_STARTUP_TRACE=<file>` records the C++ constructor timeline; it
was disabled for timing runs. [validate_startup.py](validate_startup.py) contains
the error and lifetime tests. The original runtimes and initial study results
remain archived separately.
