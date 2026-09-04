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

Official resources:

- [Python downloads](https://www.python.org/downloads/)
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
- [ONNX Runtime source](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime GenAI source](https://github.com/microsoft/onnxruntime-genai)
- [ONNX Runtime build documentation](https://onnxruntime.ai/docs/build/)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- [llama.cpp source](https://github.com/ggml-org/llama.cpp)

### Create an environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install common Python dependencies

```powershell
python -m pip install httpx litert-lm-api
```

`litert-lm-api` installs the official native LiteRT-LM Python bindings and `litert-lm.dll`.

### Install ONNX Runtime WebGPU and GenAI

Use WebGPU and GenAI wheels built for the same Python version and architecture. The benchmark recorded in this repository used these locally built CPython 3.13 wheels:

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-webgpu onnxruntime-genai onnxruntime-genai-cuda
python -m pip install "<path-to-wheels>\onnxruntime_webgpu-1.30.0-cp313-cp313-win_amd64.whl"
python -m pip install "<path-to-wheels>\onnxruntime_genai-0.16.0.dev0-cp313-cp313-win_amd64.whl"
```

These development wheels are not assumed to be available from PyPI. Build newer wheels from the official [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) repositories, following their build documentation, or replace the paths above with compatible published wheels. Do not mix CPU, CUDA, and WebGPU wheels in the same environment because they install overlapping modules.

Verify the environment before benchmarking:

```powershell
python -c "import onnxruntime as ort, onnxruntime_genai as og; print(ort.__version__, og.__version__, ort.get_available_providers())"
```

`WebGpuExecutionProvider` must be present.

### Install llama.cpp with Vulkan

Download a Windows Vulkan build containing `llama-server.exe` from the official [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), when available. Extract the archive and pass the executable path with `--llama-server`.

If a suitable prebuilt archive is unavailable, build from source. Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home), Visual Studio C++ Build Tools, Git, and CMake, then run:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_CURL=OFF
cmake --build build-vulkan --config Release --target llama-server -j 12
```

The resulting executable is normally under `build-vulkan\bin\Release\llama-server.exe`. Verify it with:

```powershell
.\build-vulkan\bin\Release\llama-server.exe --version
.\build-vulkan\bin\Release\llama-server.exe --list-devices
```

The benchmark uses Vulkan with all model layers offloaded to the selected GPU.

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

Direct folders:

- [ONNX WebGPU INT4](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/onnx-webgpu)
- [Native LiteRT-LM](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/litert-lm)
- [GGUF Q4_K_M](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/gguf)
