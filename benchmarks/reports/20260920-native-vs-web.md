# Standalone native vs. Web — September 20, 2026

Device: **webgfx-104, NVIDIA RTX 5080**, driver `32.0.16.1664`, Intel Core Ultra
9 285K, existing Balanced power plan. No driver, model, power or browser-security
changes were made. Experiments are separate from daily performance history.

## Implementations

The Web runner lives in `benchmarks/web`. The new C++ runner lives in
`benchmarks/native`; it uses native ORT directly, not GenAI's CPU-search loop.
Both compile the **same exact-bit WGSL greedy selection shader** from
`web/gpu-greedy.mjs`. The chosen token feeds the next decoder invocation on GPU;
up to 16 readbacks deliver CPU-readable IDs in order. Every autoregressive step
is executed; batch size remains one. The first token uses CPU selection.

Gemma keeps CPU sampling because of its separate embedding session. Its CPU
token inputs are rebound after each update, with GPU embedding output and a
captured GPU decoder. Native CPU greedy selection uses integer IEEE ordering,
including signed-zero ties and nonfinite rejection, without scalar FP16
conversion or per-vocabulary-entry error-message allocations.

ORT source for **both** builds:
`09dfa6ad06ed8072b2fbe57687d4f71c7914b025`.

- Native: x64 Release ORT from the existing source build, linked with that exact
  build's Dawn/Tint. ORT DLL SHA-256:
  `e9f8a85dba5463eb1cf76177a0c2643d67dafd8fe3006e23d087718755f9c193`.
  Dawn dependency: `v20260828.013844`. Compiler: MSVC `19.51.36257.0`.
- Web: newly built `benchmark-20260920-09dfa6ad`, Release WASM SIMD, JSPI,
  C++ WebGPU EP, one WASM thread. Artifact hashes and compatibility patches are
  recorded in its `build-metadata.json`. The browser uses its own Dawn build;
  matching the ORT source does not make the entire platform/compiler identical.
- Both use D3D12, basic WebGPU validation, decoder graph capture, GPU I/O and
  an 8,192-token static KV cache. Neither disables GPU robustness. Native records
  enabled Dawn toggles and rejects disabled robustness/validation.
- Edge version: `153.0.4234.46`. Both runners request shader-f16, subgroups,
  subgroup-size-control and timestamp-query; the browser does not expose an
  extra subgroup-matrix feature in this run.

## Workload and validation

Five unchanged models, input lengths 128/512/1024, output length 128, full-prompt
prefill, seed-42 saved prompt IDs, greedy selection, EOS suppressed. Each point
has one warm-up and five timed repetitions. Model/config/weight hashes and exact
prompts come from the same immutable request, SHA-256:
`50299db6ca2b268f07bb8d13363fa739375542b5af8ba619426dd7c990e11620`.

TTFT ends with the first CPU-readable selected token. Decode counts the remaining
127 tokens and ends only after **all** tokens are CPU-readable. Comparison rates
use mean elapsed time; no GPU-submission-only timings or averaged reciprocal
rates are used. Loading, reference execution, reset and warm-up are excluded.

The CPU control is this standalone harness, **not the original GenAI benchmark**.
Its purpose is to check the new GPU feedback path. Do not interpret its improvement
percentage as a gain over the previously published native GenAI data.

Validation includes:

- Full 128-token repeatability and native CPU/GPU agreement at every point.
- Web checks its complete output against the uncaptured same-browser reference.
- An independent comparator checks coverage, timing equations, ordered CPU
  delivery, model/request equality, source revisions and actual adapter identity.
- Native selector tests cover FP16/FP32, signed zero, ties, subnormals, masked
  maxima and NaN/Infinity (including suppressed entries), plus a GPU reduction
  containing all 63,488 finite FP16 encodings. Native and Web share the shader.
- Web CPU tests cover all 65,536 FP16 encodings. Web GPU tests include 85 cases
  and 120 autoregressive toy tokens.
- The newly built Web runtime passes all 20 small-window/large-window capture
  probes, with eight observed graph-replay messages. Separate native smoke tests
  observed 13 decoder replays for 14 decode calls on each of the five models.
- The final Web model-level diagnostic also observed 13 decoder replays for 14
  decode calls on **each of the five models**, with verified actual JSPI loads.
- Final audit: **15 cases, 225 timed samples including the CPU control**, exact
  complete outputs, valid CPU-visible timing, and delivered source/executable
  hashes matching the measured artifacts. Node tests: 27 passed; Python tests:
  12 passed. Native CPU/GPU edge-case probes passed on the local T1000 and RTX 5080.

## Results

**The large Web advantage disappears once native uses the comparable generation
strategy.** Aion, Phi-4 and both Qwen models are essentially at decode parity;
the measured differences are below 1%. Gemma is substantially faster natively:
Web decode is 46–49% lower in these measurements. Gemma retains a mixed CPU/GPU
embedding path in both runners; attributing its entire residual difference to a
single component would require additional profiling.

At input 1,024, native TTFT throughput is 0.8–4.6% higher. Short-input TTFT gaps
are larger. These are **first-token throughput** figures, not isolated GPU-only
prefill kernel rates.

Input 1,024 / output 128 summary (tokens/s):

| Model | Native TTFT throughput | Web TTFT throughput | Native decode | Web decode | Web decode vs native |
|---|---:|---:|---:|---:|---:|
| Aion | 6458.75 | 6270.71 | 272.92 | 272.39 | -0.2% |
| Phi-4-mini | 4621.18 | 4523.69 | 223.00 | 221.45 | -0.7% |
| Qwen3.5-2B | 6463.08 | 6340.68 | 257.08 | 256.51 | -0.2% |
| Qwen3.5-4B | 3412.57 | 3386.36 | 155.13 | 155.28 | +0.1% |
| Gemma-4-e2b-it | 3836.40 | 3669.17 | 131.75 | 70.93 | -46.2% |

Full matrix; **all rows match the entire 128-token output**:

| Model | Input | Native TTFT throughput | Web TTFT throughput | Native decode | Web decode |
|---|---:|---:|---:|---:|---:|
| Aion | 128 | 4496.73 | 3713.48 | 278.74 | 276.41 |
| Aion | 512 | 6080.84 | 5783.61 | 274.51 | 274.89 |
| Aion | 1024 | 6458.75 | 6270.71 | 272.92 | 272.39 |
| Phi-4-mini | 128 | 3393.71 | 2985.91 | 228.30 | 226.74 |
| Phi-4-mini | 512 | 4489.86 | 4353.30 | 224.71 | 223.53 |
| Phi-4-mini | 1024 | 4621.18 | 4523.69 | 223.00 | 221.45 |
| Qwen3.5-2B | 128 | 4410.85 | 3544.63 | 259.40 | 258.53 |
| Qwen3.5-2B | 512 | 5918.26 | 5696.55 | 257.75 | 256.96 |
| Qwen3.5-2B | 1024 | 6463.08 | 6340.68 | 257.08 | 256.51 |
| Qwen3.5-4B | 128 | 2474.32 | 2221.95 | 156.17 | 155.70 |
| Qwen3.5-4B | 512 | 3271.73 | 3197.52 | 155.39 | 155.37 |
| Qwen3.5-4B | 1024 | 3412.57 | 3386.36 | 155.13 | 155.28 |
| Gemma-4-e2b-it | 128 | 2063.17 | 1737.50 | 136.90 | 70.93 |
| Gemma-4-e2b-it | 512 | 3430.08 | 3071.72 | 136.28 | 69.87 |
| Gemma-4-e2b-it | 1024 | 3836.40 | 3669.17 | 131.75 | 70.93 |

Against the final native CPU-selection control, GPU feedback improves decode
by 6.7–14.1% for the four supported models. Gemma uses CPU selection in both
arms and its small control/candidate variation is not an optimization gain.
The earlier prototype's slow scalar FP16 conversion/error-string allocation
inflated the apparent CPU-to-GPU improvement; those prototype percentages are
**not** the final optimization claim.

Machine-readable metrics and audit provenance: [20260920-native-vs-web.json](20260920-native-vs-web.json).

## Limits and retained evidence

This is a warmed, fixed-length, deterministic comparison on one device, not a
general model-accuracy certificate or a fleet-wide result. Small percentage
differences can be run-to-run variation; the study is not a randomized multi-day
statistical experiment. Pipelined CPU token-delivery spacing can differ from a
synchronous chat loop. Beam search, penalties, natural EOS and cancellation are
outside this harness's scope.

Raw requests, samples, output tokens, failed prototype diagnostics and build
evidence are retained locally under `benchmarks/results/webgfx104-20260920` and
`gitignore/runtime-comparison`. Remote experiment staging is
`D:/workspace/project/agents/webgfx-china/gitignore/ort-port-20260920` on
webgfx-104. Final data are in `full-final/native-cpu.json`,
`full-final/native-gpu.json` and `full-01/web.json`; earlier native prototypes are
retained separately. Final executable SHA-256:
`8e0a2f598b74abb4222df67330de905fe4c063bfed25f959fc3b660c5323e212`.
Temporary launcher tasks were removed after successful completion. No historical
chart data was overwritten and no changes were pushed.

The rejected ONNX-sampler prototype failed a GPU NaN-validation test. It is not
the implementation used for the accepted comparison. The accepted implementation
uses the shared integer-bit WGSL shader and direct queue readbacks. A failed
Windows launcher attempt produced no measurements; the successful run uses
process-level stdout/stderr redirection so diagnostic output is not treated as
a terminating PowerShell error.
