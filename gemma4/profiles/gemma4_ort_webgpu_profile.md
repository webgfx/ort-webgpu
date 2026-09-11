# Gemma 4 ONNX Runtime WebGPU Profiling Report

- Runtime: **ONNX Runtime WebGPU 1.30.0 + ONNX Runtime GenAI 0.16.0-dev**
- Hardware: **NVIDIA GeForce RTX 4070 12 GB**, driver **591.44**, Windows 11
- Model: **Gemma 4 E2B INT4 ONNX**, decoder WebGPU graph capture enabled
- Trace: `gemma4_decoder_webgpu.json`
- Input/output tokens: **1024 / 256**
- TTFT: **206.8 ms**
- Decode throughput: **86.1 tok/s**
- End-to-end generation: **3170.1 ms**
- Profile events / node events: **316619 / 4916**

Profiling was enabled on `model.decoder.session_options`

## Provider Coverage

| Provider | Node events |
|---|---:|
| WebGpuExecutionProvider | 4916 |

## Top Operators by Accumulated Profile Duration

Durations are accumulated ORT node-event time. They can overlap and include profiling overhead; they are not additive wall time.

| Operator | Accumulated duration | Calls | Share of node duration |
|---|---:|---:|---:|
| RMSNormalization | 245.198 ms | 772 | 29.1% |
| MatMulNBits | 159.736 ms | 1112 | 19.0% |
| GroupQueryAttention | 111.859 ms | 140 | 13.3% |
| Gelu | 104.540 ms | 280 | 12.4% |
| Mul | 46.341 ms | 640 | 5.5% |
| Add | 43.704 ms | 348 | 5.2% |
| Gather | 41.918 ms | 236 | 5.0% |
| ReduceMean | 24.070 ms | 60 | 2.9% |
| SkipSimplifiedLayerNormalization | 10.266 ms | 140 | 1.2% |
| Reshape | 9.400 ms | 528 | 1.1% |
| GatherBlockQuantized | 9.280 ms | 140 | 1.1% |
| Concat | 7.122 ms | 8 | 0.8% |
| Cast | 6.682 ms | 128 | 0.8% |
| Where | 6.261 ms | 8 | 0.7% |
| Div | 4.572 ms | 64 | 0.5% |

## Interpretation

This is an instrumented diagnostic run, not a benchmark result. Profiling changes execution overhead. Open the trace in Perfetto or `chrome://tracing` for timeline analysis.
