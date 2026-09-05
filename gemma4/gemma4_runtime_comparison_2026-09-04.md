# Gemma 4 E2B Model + Inference Runtime Comparison

> Hardware: NVIDIA RTX 4070 12 GB, driver 591.44, Windows 11. ONNX Runtime uses the WebGPU EP; native LiteRT-LM uses WebGPU over D3D12; llama.cpp uses Vulkan. Measurements collected 2026-09-03 and 2026-09-04.

## Executive Summary

- llama.cpp Vulkan leads decode throughput at **164.8 tok/s** without MTP.
- LiteRT MTP raises long-form decode from **94.5 to 110.9 tok/s**.
- ONNX prefill-prefix pruning reduces length-controlled TTFT by **66.4–80.2%** versus the previous unpruned artifact.
- All stacks score **10/10** on core capability tests; the ONNX stack retains a long-form coherence limitation.

## What Models and Artifacts Were Tested?

All stacks use Gemma 4 E2B-IT (5.1B total / approximately 2.3B effective parameters, 35 text layers, GQA, per-layer embeddings, 262,144-token vocabulary). They are different conversions and quantization formats, so results describe complete model-plus-runtime stacks.

**Model context:** Per-Layer Embeddings account for much of the total parameter count while only part of the model is active in each inference layer, explaining the 2.3B effective size. The ONNX and LiteRT-LM packages include text, vision, and audio components, whereas the tested GGUF contains only the text model. Prefill-prefix pruning reduces prompt-processing work, while MTP speculative decoding targets token generation; they optimize different phases.

| Artifact | Size | Format and contents | Published model |
|---|---:|---|---|
| Gemma 4 ONNX WebGPU | 3.47 GB | INT4 decoder, embeddings, vision/audio graphs; prefill-prefix pruning | [onnx-webgpu](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/onnx-webgpu) |
| Gemma 4 LITERTLM | 2.59 GB | Native INT4 multimodal container with MTP drafter | [litert-lm](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/litert-lm) |
| Gemma 4 GGUF | 3.46 GB | Text Q4_K_M target model; no MTP-head GGUF installed | [gguf](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it/gguf) |

[Open the complete Gemma 4 E2B model collection on Hugging Face](https://huggingface.co/webai-community/ai-models/tree/main/gemma-4-E2B-it)

## Overall Model + Runtime Stack Results

| Model + runtime stack | Format/backend | Long decode | Multi-turn | Core Capability Accuracy | Avg / peak VRAM |
|---|---|---:|---:|---:|---:|
| ONNX Runtime GenAI · Prefill Optimization | INT4 ONNX / WebGPU | 100.4 tok/s | 98.7 tok/s | 10/10 | 4597 / 4694 MB |
| LiteRT-LM Native · MTP off | INT4 LITERTLM / native WebGPU | 94.5 tok/s | 94.3 tok/s | 10/10 | 1958 / 1958 MB |
| LiteRT-LM Native · MTP on | INT4 LITERTLM / native WebGPU | 110.9 tok/s | 94.9 tok/s | 10/10 | 2031 / 2031 MB |
| llama.cpp | Q4_K_M GGUF / Vulkan | 164.8 tok/s | 167.4 tok/s | 10/10 | 2782 / 2822 MB |

### TTFT by Input Length

Each request allows up to **128 output tokens**; TTFT stops at the first emitted content token. Each length has one warmup and three timed requests. Values below are medians. Model loading is excluded; llama.cpp prompt caching is disabled.

| Stack | 128 tokens | 512 tokens | 1,024 tokens | 2,048 tokens | 4,096 tokens |
|---|---:|---:|---:|---:|---:|
| ONNX Runtime WebGPU · Prefill Optimization | 35.6 ms | 80.0 ms | 155.3 ms | 347.4 ms | 899.2 ms |
| LiteRT-LM Native WebGPU · MTP off | 34.9 ms | 107.3 ms | 116.5 ms | 217.5 ms | 444.5 ms |
| LiteRT-LM Native WebGPU · MTP on | 54.1 ms | 128.4 ms | 137.6 ms | 244.6 ms | 472.8 ms |
| llama.cpp Vulkan | 216.2 ms | 249.6 ms | 308.4 ms | 443.6 ms | 703.7 ms |

## Prefill Optimization: Before vs After

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Short-prompt TTFT | 45.3 ms | 24.0 ms | −47.0% |
| 128-token TTFT | 106.0 ms | 35.6 ms | −66.4% |
| 512-token TTFT | 385.2 ms | 80.0 ms | −79.2% |
| 1,024-token TTFT | 785.9 ms | 155.3 ms | −80.2% |
| 2,048-token TTFT | 1,677.9 ms | 347.4 ms | −79.3% |
| 4,096-token TTFT | 4,523.3 ms | 899.2 ms | −80.1% |
| Long decode | 93.4 tok/s | 100.4 tok/s | +7.6% |
| Core Capability Accuracy | 10/10 | 10/10 | Unchanged |

**Current ONNX runtime:** WebGPU 1.30.0 with ONNX Runtime GenAI 0.16.0-dev.

**Graph proof:** logits changed from `[batch, sequence_len, 262144]` to `[batch, 1, 262144]`; the build manifest records `prune-prefill-prefix`.

## Generation Quality Assessment

| Stack | Core Capability Accuracy | Long-form coherence |
|---|---:|---|
| ONNX Runtime GenAI · Prefill Optimization | 10/10 | 1/5 clean; four malformed, repetitive, or multilingual tails |
| LiteRT-LM Native · MTP off | 10/10 | 5/5 coherent completions |
| LiteRT-LM Native · MTP on | 10/10 | 5/5 coherent completions |
| llama.cpp | 10/10 | 5/5 coherent completions |

Core capability tests cover arithmetic, factual knowledge, translation, pattern completion, classification, grammar, basic science, and common-sense reasoning under greedy decoding. Long-form generation uses temperature 0.7. Deterministic source-model output/logit comparison is recommended to isolate the ONNX long-form issue.

## Model Conversion and Runtime Optimization Analysis

### INT4 ONNX + ONNX Runtime GenAI

- WebGPU graph capture enabled.
- Prefill-prefix pruning enabled and verified.
- Decoder logits are `[batch, 1, 262144]`.
- 278 decoder `MatMulNBits` operations, 35 block-quantized per-layer embeddings, and one block-quantized main embedding.
- No MTP drafter.

### INT4 LITERTLM + LiteRT-LM Native

- Official Google AI Edge native LiteRT-LM library (`litert-lm.dll`) invoked through Python bindings.
- Native WebGPU execution over D3D12.
- Dedicated `prefill_128`, `prefill_1024`, `decode`, and `verify` graphs.
- Bundled MTP drafter.
- MTP benchmark acceptance: 37.3%; long-form decode improves 17.4%, with increased TTFT.

### Q4_K_M GGUF + llama.cpp

- Vulkan, full GPU offload, and flash attention.
- `draft-mtp` requires a separate compatible MTP-head GGUF; none was installed.

## Methodology and Caveats

- Decode excludes prefill.
- LiteRT token counts are approximate; ORT and llama.cpp use runtime/tokenizer counts.
- llama.cpp TTFT includes local HTTP/SSE delivery, so compare it directionally with in-process ORT/LiteRT.
- Absolute VRAM includes runtime state and other GPU allocations.
- One throughput run per stack does not provide confidence intervals; TTFT-by-length uses three measured samples per point.

## Reproducibility

Run the benchmark scripts first, keep their JSON outputs beside the generator, then generate the Markdown report.

| Script | Purpose |
|---|---|
| `benchmark_ort_gemma4.py` | ONNX Runtime GenAI performance, memory, and quality |
| `benchmark_litert_gemma4.py` | Native LiteRT-LM benchmark with MTP on or off |
| `benchmark_llamacpp_gemma4.py` | llama.cpp Vulkan server benchmark |
| `benchmark_prefill_scaling.py` | TTFT at 128–4,096 input tokens |
| `generate_runtime_comparison.py` | Generate the Markdown report |

See `README.md` for dependencies, model-path options, and exact commands.
