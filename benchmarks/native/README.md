# Optimized native benchmark

## Run the built benchmark on this machine

From `D:\workspace\project\ort-webgpu`:

```powershell
.\benchmarks\native\start.ps1 -Models aion
```

This prints TTFT throughput and decode TPS and saves the request, raw samples,
token IDs and logs to a new dated folder under `gitignore/benchmarks/results`.
Default inputs are 128/512/1024, output is 128, and there are five timed repetitions.
For all five models, use
`-Models 'aion,Phi-4-mini-instruct,Qwen3.5-2B,Qwen3.5-4B,gemma-4-e2b-it'`.
Use `-ModelsRoot` if the models are not in `D:/workspace/project/agents/ai-models`.

The executable and required DLLs are in
`gitignore/benchmarks/build/native/ort_webgpu_benchmark.exe`. Do not move the EXE
without its DLLs. These are local ignored build artifacts, not tracked binaries.
To inspect saved data without running again:

```powershell
python benchmarks/native/summary.py gitignore/benchmarks/results/<run>/result.json
```

This is a standalone C++ ORT generator, not a patch to GenAI. It owns GPU
buffers, uses phase-specific ORT sessions, and shares its GPU greedy shader with
the Web benchmark. No model export changes or ONNX sampler graph are needed.

## Difference from the daily ORT WebGPU Perf test

[Measured Panther Lake comparison and correctness findings](../reports/20260924-ptl-native-vs-genai.md).

| Area | Current daily native runner | This standalone native runner |
|---|---|---|
| Entry point | `tasks/ai-test.py` calls GenAI `model_benchmark.exe` | `run.py` calls `ort_webgpu_benchmark.exe`, which uses ORT directly |
| Generation loop | GenAI WebGPU currently uses CPU greedy search | Same exact-bit GPU greedy shader as Web, with direct GPU token feedback for supported models |
| Decode capture | Model configuration/compatibility policy; not forced on for all models | Separate fixed-shape decoder session, capture enabled by default |
| GPU buffers | Managed by GenAI | Explicit stable input/output, 8K KV and recurrent-state buffers |
| Gemma | GenAI's mixed embedding/decoder path | Uncaptured embedding with rebound CPU inputs, captured decoder, optimized CPU greedy selection |
| Prompts | Stock `--use_random_tokens` generates unsaved random IDs | Exact prompt IDs saved in the request and reusable by Web/GenAI reference |
| TTFT | Parses stock “Prompt processing” duration, which excludes first-token selection in the inspected GenAI source | Includes first CPU-readable selected token |
| Validation | Validates successful/nonzero summary metrics | Full generated IDs, deterministic repeatability and CPU-delivery timestamps; paired runs check exact output equality |
| Scope | Daily testing and chart ingestion | Controlled benchmark experiments; no automatic daily-chart replacement |

Both use the same unmodified ONNX exports, D3D12 and an 8K KV limit. This runner
explicitly uses basic validation and default enabled Dawn robustness. The daily
wrapper does not request those same settings by default. Model/capture policy,
runtime revision, prompt IDs and timing boundaries must be disclosed in comparisons.

`genai-reference.cpp` is a small public-API harness to align saved prompts and
CPU-visible timing without altering GenAI's internals. `run-genai.py --stock`
also runs the original `model_benchmark.exe`, but those stock numbers retain its
random-prompt and TTFT-definition differences. The pinned GenAI revision's
initial-device option forwarding omits `enableRobustness`, so a model-config
request alone is not evidence of matched effective robustness.

## Build

Requires Visual Studio x64 C++ tools, Windows SDK, CMake/Ninja, Node.js, and an
existing native ORT WebGPU build. The script reuses that build's Dawn/Tint,
Abseil and nlohmann headers/libraries. **The ORT DLL must come from that exact
build**, because the externally supplied Dawn procedure table has a versioned
ABI. The build script checks the DLL hash rather than accepting a mismatched
archived package.

```powershell
.\native\build.ps1 `
  -OrtSource D:/path/to/onnxruntime `
  -OrtBuild D:/path/to/ort-build/Release `
  -Runtime D:/path/to/ort-build/Release

python native/run.py --request ../gitignore/benchmarks/results/request.json `
  --executable ../gitignore/benchmarks/build/native/ort_webgpu_benchmark.exe `
  --output ../gitignore/benchmarks/results/native.json
```

Each model runs in a separate process. The result includes runtime/executable
hashes, build source hashes, actual adapter identity, enabled Dawn toggles,
effective provider options, generated IDs and ordered CPU-delivery timestamps.
Default D3D12 hardware selection uses high-performance preference; software
adapters and disabled robustness/validation toggles are rejected.

A standalone run checks repeatability. Use `run-comparison.ps1` to additionally
check the native CPU control and the full cross-runtime token sequences.

## Implementation

- `main.cpp`: model sessions, fixed GPU I/O, KV/recurrent state, CPU reference
  sampling, generation and timing. CPU selection also uses exact integer-bit
  ordering, avoiding scalar FP16 conversion and per-score error-string allocation.
  CPU embedding inputs are rebound after
  updates: native I/O binding can otherwise reuse previously staged input IDs.
- `gpu.h`: Dawn device, GPU queue operations and bounded sampler readbacks.
- `emit-shader.mjs`: embeds `web/gpu-greedy.mjs`'s shader verbatim into generated
  C++. The CPU does not inspect the vocabulary between GPU-selected tokens.
- `probe.h` / `test_sampler.py`: real native GPU edge-case checks.
- `run.py`: model integrity, isolated processes, timeouts and result provenance.

Application-owned buffers avoid interference between ORT allocator initialization
and external queue writes. Uploads, state copies and readbacks use the same
external Dawn device/queue. Each result waits for all token readbacks; timings
are not GPU submission times. Capture binds stable tensor addresses throughout.

Controls: `--gpu-sampling cpu`, `--max-pending 1`, `--no-capture`.
`--capture-diagnostics` verifies replay separately from the timed comparison.
The first token and Gemma's auxiliary-session generation retain CPU sampling.
This benchmark intentionally implements batch-one greedy, fixed-length generation;
it does not claim full GenAI search, beam, penalty, cancellation or chat semantics.

See [common setup and comparison rules](../README.md).
