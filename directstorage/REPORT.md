# Phi-4 DirectStorage and TTFT results

A [follow-up study](STARTUP_REPORT.md) implements and measures overlapping
tokenizer loading and deferring GenAI's bootstrap. This report retains the
initial study and its original measurements.

Measured on 2026-09-06. DirectStorage reduced Phi-4 model creation from
**2.021 s to 1.022 s (49.4% lower)**.
For a 128-token prompt, application startup through the first CPU-accessible
generated token fell from **2.710 s to 1.696 s
(37.4% lower)**. Loaded-model request latency stayed
approximately 70 ms. Generator reuse and persistent shader caching address
different portions of the remaining latency.

The model is **Phi-4-mini-instruct**, using the archived fused-RoPE INT4 export:
389 external initializers, 2,491,299,840 tensor bytes in a 2,491,416,576-byte
weight file, and 398 DirectStorage requests at the PR's 64 MiB request limit.
The graph has 160 four-bit MatMulNBits nodes and one eight-bit MatMulNBits node.
All archived model file hashes were verified before testing.

Hardware: NVIDIA GeForce RTX 5080, driver 616.56; Ryzen 9 9950X; approximately
64 GB RAM; Samsung SSD 990 EVO Plus 1TB on E: (NTFS); Windows 11 build 26200.
This is a native Dawn/D3D12 test.

## Startup results

Each cell is the median of five fresh processes. Each process performs its first
inference without an inference warmup. Before each timed launch, the parent reads
the weight file; this pre-read is excluded from timing. OS and driver caches are
preserved, and preliminary runs exercised the prompt shapes. These numbers do
not represent a cold-disk or purged-driver-cache test.

| Prompt tokens | Acceleration off | DirectStorage required | DirectStorage required-pipelined |
|---:|---:|---:|---:|
| 1 | 2.623 s | 1.693 s | 1.648 s |
| 128 | 2.710 s | 1.771 s | 1.696 s |
| 1024 | 2.875 s | 1.939 s | 1.873 s |

The measured interval includes process startup, native DLL loading, GenAI model
creation, tokenizer creation, prompt encoding, generator creation, prefill,
sampling, and CPU token readback. It ends before process teardown. The timestamps
come from the same run; these are not sums of independent medians.

The sequence budget is 8192 tokens, batch size is one, graph capture is off,
validation is basic, sampling is greedy, and Constant Folding is disabled.
Other runtime graph optimizations and the default GenAI CPU thread count remain
enabled for the primary comparison. Each run saves eight generated tokens and
then makes two additional requests using the same loaded model and tokenizer.

The first 45-process comparison omitted the per-trial pre-read and showed a
larger, less stable advantage: at 128 prompt tokens, startup was 4.226 s with
acceleration off and 1.605 s with pipelined DirectStorage. File-access history
materially affects the size of the gain; the table above uses the more controlled
comparison. The original `baseline` logs are retained.

Observed startup ranges for the primary comparison:

| Prompt tokens | Mode | Minimum to maximum |
|---:|---|---:|
| 1 | off | 2.561 to 2.663 s |
| 1 | required | 1.679 to 1.727 s |
| 1 | pipelined | 1.626 to 1.658 s |
| 128 | off | 2.604 to 2.743 s |
| 128 | required | 1.755 to 1.805 s |
| 128 | pipelined | 1.686 to 1.740 s |
| 1024 | off | 2.818 to 2.985 s |
| 1024 | required | 1.936 to 1.948 s |
| 1024 | pipelined | 1.862 to 1.917 s |

## Additional optimizations tested

This was a separate randomized 75-process comparison: five fresh processes for
each of five variants and three prompt lengths. All variants use the same cache
prototype DLL, with its cache disabled for the controls. The original PR DLL
and prototype DLL are archived separately.

Results below use a 128-token prompt. First-request and warm-request timers start
with an already encoded prompt and include generator creation or reset, prefill,
sampling, and readback. Warm medians use ten requests per variant.

| Variant | Startup to first token | First request after loading | Later request |
|---|---:|---:|---:|
| DirectStorage pipelined, new generator | 1.697 s | 197.3 ms | 68.2 ms |
| Reuse generator, reset to zero | 1.686 s | 195.2 ms | 40.0 ms |
| New generator, 2K sequence budget | 1.695 s | 187.1 ms | 41.3 ms |
| Persistent shader cache, new generator | 1.575 s | 79.4 ms | 69.7 ms |
| Persistent shader cache + generator reuse | 1.593 s | 80.1 ms | 40.1 ms |

**Generator reuse** is the clearest improvement for an already running service.
Calling `RewindTo(0)` preserves allocations while recomputing the prompt. It keeps
the 8K sequence budget and avoids approximately 29 ms of generator allocation
work per later request. An additional test alternated different 128-token and
10-token prompts and matched the outputs from fresh generators.

| Prompt tokens | New generator | Reused generator | Reduction |
|---:|---:|---:|---:|
| 1 | 34.50 ms | 5.26 ms | 84.8% |
| 128 | 68.19 ms | 39.96 ms | 41.4% |
| 1024 | 261.44 ms | 230.83 ms | 11.7% |

The smaller 2K budget also reduces allocation work, but limits the total prompt
plus generated sequence to 2048 tokens. Reuse achieved similar or better request
latency while retaining the original capacity.

**Persistent shader caching** reduces repeated compilation across processes. The
prototype implements Dawn's caching interface and preserves the complete cache
key. The cache contained 63 entries totaling 519,034 bytes. Its first population
recorded 63 misses and 63 stores; measured subsequent cache-enabled runs recorded
42 or 63 hits and zero misses/stores. A deliberately corrupted entry was detected,
recompiled, and restored byte-for-byte, with 62 hits, one miss, one store, and
unchanged generated tokens. The empty-cache priming run is excluded from the
cache-hit comparison. The initial cache population has its own cost.

The prototype needs cache-size limits, eviction, and stronger multi-process
coordination before production use. Its cache helps later application launches;
keeping the model, tokenizer, and reusable generators resident avoids the larger
startup costs altogether.

The exploratory `tuning-pilot` tested CPU thread counts of one/four, disabled
spinning, disabled runtime graph optimization, and enabled graph capture. These
single-run pilots did not identify a clear additional TTFT winner. Graph capture
raised the pilot's first-request time while leaving later-request TTFT similar;
this experiment does not evaluate its benefit for long decoding runs.

## Integration findings and native loading

Two integration adjustments were necessary:

1. With Python hosting ORT, `DStorageGetFactory` returned `0x80004001` when the
   core DLL was only in an added DLL directory. A C++ executable beside the DLLs
   succeeded, and explicitly preloading `dstoragecore.dll` fixed the Python case.
   The harness applies that preload consistently to every mode.
2. The archived GenAI runtime initialized its bootstrap WebGPU context without
   forwarding `weightLoadAcceleration`. Required-pipelined mode then correctly
   rejected the context initialized with acceleration off. The companion GenAI
   patch forwards the option. Every final TTFT case uses the same patched GenAI
   binary. That bootstrap session still finishes device initialization before
   loading the real model, so GenAI does not expose the PR's full opportunity to
   overlap real-model weight I/O with device initialization. A further GenAI
   initialization refactor could recover that overlap; it was not implemented here.

A separate native ORT probe bypassed GenAI's bootstrap. It used the same model,
one CPU thread, disabled runtime graph optimization, a pre-read before each
trial, and five randomized processes per mode. Its inclusive measurement covers
environment/EP setup and session construction, and does not measure inference:

| Mode | Native setup + model loading |
|---|---:|
| off | 1864.0 ms |
| required | 1014.8 ms |
| required-pipelined | 941.2 ms |

The diagnostic trace confirmed the RTX 5080 and actual DirectStorage transfers
of all 389 external tensors. The PR's `required` modes were used to make
unsupported acceleration fail instead of silently falling back.

## Evidence and scope

The main comparisons and pilot comprise 174 fresh-process TTFT runs and 522
requests, with no greedy-token mismatches across loading modes, cache variants,
sequence budgets, or generator reuse. There were also 15 native loading trials,
separate priming/diagnostic runs, a mixed-prompt reuse check, and cache
roundtrip/corruption/truncation/concurrency checks. Full ORT and GenAI test suites
were not run. Token agreement is a regression check for these prompts, not a
general model-quality evaluation.

Reproduction instructions and source files are in [README.md](README.md).
Raw logs, per-process JSON, complete configuration snapshots, runtime hashes,
model hashes, and summaries are in
`../gitignore/directstorage-phi4-20260906/`. Important files are `manifest.json`,
`summary.json`, the phase-specific `suite.json` files, `reuse-validation.json`,
and `cache-corruption-validation.json`. The repository's original checkouts and
historical benchmark archives were preserved.
