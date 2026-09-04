# Gemma 4 E2B Model and Runtime Benchmark

Reproducible comparison of Gemma 4 E2B model packages running with ONNX Runtime GenAI, native LiteRT-LM, and llama.cpp.

## Required scripts

| File | Purpose |
|---|---|
| `benchmark_ort_gemma4.py` | ONNX Runtime GenAI performance, memory, and quality benchmark |
| `benchmark_litert_gemma4.py` | Native LiteRT-LM benchmark with MTP on or off |
| `benchmark_llamacpp_gemma4.py` | llama.cpp server benchmark using Vulkan |
| `benchmark_prefill_scaling.py` | TTFT benchmark at 128–4,096 input tokens |
| `generate_runtime_comparison.py` | Generate the Markdown report |
| `generate_runtime_comparison_html.py` | Generate the standalone HTML report |

The report generators read benchmark JSON files produced by the four benchmark scripts. Keep those data files in the same directory when regenerating reports, but they do not need to be linked from the published reports.

## Dependencies

- Python 3.13
- NVIDIA driver with `nvidia-smi`
- ONNX Runtime WebGPU and ONNX Runtime GenAI for the ONNX benchmark
- `litert-lm-api` for native LiteRT-LM
- `httpx` for llama.cpp server requests
- A Vulkan-enabled `llama-server` binary

Install ordinary Python dependencies with:

```powershell
python -m pip install httpx litert-lm-api
```

Install ONNX Runtime WebGPU and ONNX Runtime GenAI from compatible wheels. Verify the environment before benchmarking:

```powershell
python -c "import onnxruntime as ort, onnxruntime_genai as og; print(ort.__version__, og.__version__, ort.get_available_providers())"
```

`WebGpuExecutionProvider` must be present.

## Model paths

Every benchmark accepts model/runtime paths through command-line options. Defaults reflect the original Windows test machine; override them in another environment.

## Run benchmarks

```powershell
python benchmark_ort_gemma4.py --model-path <onnx-model-dir> --output ort.json
python benchmark_litert_gemma4.py --model-path <model.litertlm> --no-mtp --output litert-off.json
python benchmark_litert_gemma4.py --model-path <model.litertlm> --output litert-on.json
python benchmark_llamacpp_gemma4.py --llama-server <llama-server.exe> --model-path <model.gguf> --main-gpu 0 --output llama.json

python benchmark_prefill_scaling.py ort --output prefill-ort.json
python benchmark_prefill_scaling.py litert-off --output prefill-litert-off.json
python benchmark_prefill_scaling.py litert-on --output prefill-litert-on.json
python benchmark_prefill_scaling.py llama --output prefill-llama.json
```

The prefill script currently keeps its model/runtime defaults near the top of the file; update those constants for a different machine.

## Generate reports

The generators currently expect the canonical result filenames declared near the top of each script. Adjust those filenames when using newly generated data.

```powershell
python generate_runtime_comparison.py
python generate_runtime_comparison_html.py
```

Outputs:

- `gemma4_runtime_comparison_2026-09-03.md`
- `gemma4_runtime_comparison_2026-09-03.html`

## Published models

The tested model packages are available at:

<https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it>
