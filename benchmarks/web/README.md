# Browser benchmark

## Open and run directly in your browser

Open **[index.html](index.html)** in a current Edge or Chrome. You can double-click
the file or open `D:\workspace\project\ort-webgpu\benchmarks\web\index.html`.
It is self-contained and works from `file://`: **no server, Node.js or Python is
needed to run the page**.

1. Select the trusted ORT Web runtime folder. It must contain the build metadata,
   JavaScript bundle, JSPI module and WASM file described below.
2. Select a model folder (containing `onnx-webgpu/genai_config.json`) or its
   `onnx-webgpu` folder directly. A parent folder can contain multiple models.
3. Choose input lengths, output length, repetitions and token selection; click
   **Run benchmark**. Model/runtime hashes are verified before timing begins.
4. Review the live table and download JSON (all samples/token IDs/provenance) or CSV.

The runtime and weights are loaded from selected local files; they are not
uploaded. Use only trusted runtime code. Large weights are hashed incrementally
and passed to JSPI as File/Blob ranges, not copied wholesale into JavaScript RAM.
Each model runs in an isolated iframe using the same generator/selector as the
automated runner. Stop destroys the active iframe and marks the result incomplete.

The default uses one WASM thread and therefore does not need cross-origin
isolation or a service worker for the supported JSPI build. WebGPU and JSPI must
be available in the selected browser. Keep the page visible and avoid overlapping
GPU workloads. This is not a chat UI: prompts are deterministic token IDs and
generation is greedy/fixed-length with EOS suppressed.

For exact comparison with native, select its saved `request.json` under
**Diagnostics & native comparison** after selecting the same model folders.
The page checks the model hashes and reuses its exact prompt IDs and workload.
Browser privacy can hide the precise GPU device ID; standalone exports report
what the adapter exposes, not a guessed inventory ID. JSPI evidence records
verified local files and successful inference, not the CDP network-response
observations supplied by the automated runner.

Maintainers rebuild `index.html` after editing page sources with
`npm run build:web-page` from `benchmarks`. The generated HTML is a deliverable;
build binaries, screenshots and trial results live under `gitignore/benchmarks`.
`page.html`, `page-entry.mjs`, `page-runner.mjs`, `local-files.mjs` and
`onnx-metadata.mjs` are the unbundled sources for review.

## Automated command-line runner

`run.py` validates local artifacts, starts a loopback-only host and launches Edge.
Inference runs in the browser, not in a Node native binding.

```powershell
python web/run.py --request ../gitignore/benchmarks/results/request.json `
  --runtime D:/path/to/verified/ort-web-runtime --output ../gitignore/benchmarks/results/web.json
```

The runtime directory must contain `build-metadata.json`, `ort.all.min.js`,
`ort-wasm-simd-threaded.jspi.mjs` and `ort-wasm-simd-threaded.jspi.wasm` with
matching hashes. Both JSPI loads are observed in the actual browser. No CDN or
alternate browser/runtime fallback is used. Capture requires a build supporting
external captured GPU bindings and the model's attention paths.

Key files for review:

- `generator.mjs`: reference generator, CPU argmax, prompt generation and provider options.
- `capture-generator.mjs`: phase separation, static GPU I/O and state reset.
- `gpu-greedy.mjs`: exact FP16/FP32 integer-bit two-pass WGSL reduction.
- `gpu-pipeline.mjs`: GPU token feedback, ordered bounded CPU delivery and timing.
- `runner.mjs`: full reference comparison, warm-up and repetitions.
- `host.mjs`: local artifact server, browser control and observed runtime evidence.
- `model_manifest.py`: reads graph I/O and actual state shapes without converting models.

`--gpu-sampling cpu` is the control. `auto` selects GPU for compatible captured
single-graph models and CPU for Gemma; `gpu` rejects unsupported models. Use
`--no-capture` for a separate noncapture control, not as a silent recovery path.
`--capture-diagnostics` requires observed ORT graph-replay log messages.

See [common setup and measurement rules](../README.md).
