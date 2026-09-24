# ORT WebGPU benchmarks

[Validated RTX 5080 comparison and complete measurements](reports/20260920-native-vs-web.md).

[Panther Lake native vs. GenAI: correctness, performance, and Qwen2B caveat](reports/20260924-ptl-native-vs-genai.md).

Standalone benchmarks for the same unmodified ONNX model exports:

- [Web](web/README.md): browser JavaScript generator, source-built ORT Web
  C++ WebGPU/JSPI runtime, and WebGPU GPU token selection.
- [Native](native/README.md): C++ generator, native ORT WebGPU, and the **same
  WGSL selection shader**, compiled directly into the executable from the Web
  source. This is not the existing ORT GenAI CPU-search benchmark.

For an interactive browser UI, simply open **[web/index.html](web/index.html)**.
The setup commands below are only for automated/native runs and development.
All intermediate builds and trial results belong under the repository's ignored
`gitignore/benchmarks` folder. The older `benchmarks/build` and
`benchmarks/results` contents were moved there on September 24, 2026; historical
reports retain their original run identities.

The Web implementation was ported from `webgfx-china/tasks/ort_web` on September
20, 2026. It does not require the dashboard, task server, database, network model
storage or `onnxruntime-node`. The dashboard's original runner remains untouched.

## Workload and measurements

Defaults: batch one, greedy decoding, EOS suppressed, 8K KV capacity, complete
128/512/1024-token prompts, 128 generated tokens, one warm-up and five timed
repetitions. Models: Aion, Phi-4-mini-instruct, Qwen3.5-2B, Qwen3.5-4B and
Gemma-4-e2b-it. A shared request contains exact prompt IDs and hashes of every
model/config/weight artifact. Neither runner converts or uploads models.

- TTFT includes full-prompt prefill and the first selected token available to CPU.
  TTFT throughput = input tokens / TTFT.
- Decode excludes that first token: `(output tokens - 1) / (end - first token)`.
  End is recorded only after every token is CPU-readable.
- Decode capture uses fixed GPU inputs/outputs and shared static KV buffers.
  Hybrid recurrent states use fixed-buffer GPU copies.
- For single-graph models, GPU selection writes the next token directly into the
  decoder's input buffer. At most 16 small readbacks are outstanding. This is
  **not** a batch of 16 independent tokens or speculative decoding.
- Gemma keeps the auxiliary embedding session uncaptured and CPU token inputs
  are rebound on every step. Decoder capture remains enabled; sampling is CPU.
- These are warmed, fixed-length benchmarks, not natural-EOS chat APIs. Ordered
  delivery is recorded, but GPU pipelining can change inter-token delivery gaps.
  Tokenization, loading, shader warm-up and KV reset are outside measured runs.

## Set up and run

Use an interactive Windows desktop, a D3D12 GPU with shader-f16, Node.js 22 or
newer and Python with `onnx==1.22.0` (including its NumPy dependency).

```powershell
cd D:\workspace\project\ort-webgpu\benchmarks
python -m venv ../gitignore/benchmarks/venv
..\gitignore\benchmarks\venv\Scripts\Activate.ps1
python -m pip install -r web/requirements.txt
npm ci

python prepare.py --models-root D:/workspace/project/agents/ai-models `
  --output ../gitignore/benchmarks/results/request.json

# First build the native executable as described in native/README.md.
.\run-comparison.ps1 -Request ../gitignore/benchmarks/results/request.json `
  -OutputDirectory ../gitignore/benchmarks/results/comparison-01 `
  -WebRuntime D:/path/to/verified/ort-web-runtime
```

`run-comparison.ps1` refuses an existing result directory or a competing benchmark.
It runs native CPU sampling, optimized native, then Web **sequentially**. Native
GPU output must match native CPU output before the Web comparison proceeds.
Web independently checks all generated tokens against its uncaptured reference.
`compare.py` validates coverage, repeatability, timing arithmetic and final CPU
delivery, and explicitly lists any cross-runtime output differences.

```powershell
python compare.py ../gitignore/benchmarks/results/comparison-01/native-gpu.json `
  ../gitignore/benchmarks/results/comparison-01/web.json --output ../gitignore/benchmarks/results/comparison-01/recheck.json
```

Compare rates computed from mean elapsed time, not the average of instantaneous
rates. Raw samples, output IDs, model hashes, runtime hashes, build provenance,
device/driver information and option choices remain available for review.
Use the same ORT source revision for both builds. Browser/Dawn versions and
platform-specific runtime patches must still be disclosed; a shared revision is
not proof that every compiler/runtime detail is identical.

## Tests

```powershell
npm test
npm run test:browser
python -m unittest test_benchmarks.py test_manifest.py
python native/test_sampler.py --executable ../gitignore/benchmarks/build/native/ort_webgpu_benchmark.exe `
  --output-dir ../gitignore/benchmarks/results/native-sampler-check
```

Native GPU tests cover FP16/FP32, ties, signed zero, subnormals, masked maxima,
nonfinite values including suppressed entries, and all finite FP16 encodings.
Full model testing is required in addition to these small tests.

For capture evidence, run a separate short request with
`--capture-diagnostics`; verbose diagnostic timings are not normal perf data.
Original daily data and models are not replaced by these experiments.
