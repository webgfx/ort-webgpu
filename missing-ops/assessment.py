"""Human-reviewed importance and limitations, separate from mechanical inventory."""

from analyze import CORE, CONTRIB


# Priorities are engineering judgments for inference / Chrome WebNN, not measured prevalence.
GROUPS = {
    "webnn": ("P1", "Chrome WebNN / portable graphs", "Chromium directly emits this ONNX operator. Missing native coverage can cause fallback or failure with CPU fallback disabled.", "Implement the exact WebNN datatype semantics, or validate an equivalent lowering. Run WebNN conformance with verified WebGPU placement."),
    "quant": ("P1", "INT8 / activation-quantized inference", "Blocks this quantized graph representation; weight-only MatMulNBits does not replace arbitrary QLinear, QDQ or integer operators.", "Prefer a validated supported export or deliberate dequantization to floating point; account for memory, precision and speed changes. Add quantized kernels where target models justify them."),
    "dynamic": ("P1", "Detection / dynamic selection", "Data-dependent output sizes or detection postprocessing can split a GPU graph and require readback. Especially important if called every frame/token.", "Implement scan/compaction or detection kernels; alternatively keep bounded postprocessing on CPU and measure transfer overhead. Constant folding only helps constant inputs."),
    "composition": ("P1", "General model portability", "Small missing building blocks can prevent otherwise-supported exported graphs from staying on WebGPU.", "Implement a kernel or a semantics-preserving graph rewrite; test broadcasting, integers, negative values, NaN/Inf and edge shapes."),
    "control": ("P2", "Dynamic control flow / containers", "Relevant to ONNX models with loops, sequence or optional values, but not a direct Chrome WebNN graph-building primitive. GPU kernels inside a host-controlled loop may still be useful.", "Use host orchestration, unroll bounded loops, or add executor/container support. Do not call a model fully GPU-resident simply because its loop body uses WebGPU."),
    "shape": ("P2", "Shape / allocation plumbing", "Often small or constant-foldable, so absence is not automatically a high-value GPU-kernel task. Dynamic device data can still introduce synchronization.", "Inspect the optimized graph; fold constants or keep metadata on host where safe. Implement only if the dynamic path is frequent or no-fallback deployment requires it."),
    "vision": ("P2", "Specialized vision / segmentation", "Important to the specific architecture using ROI, deformable, unpooling or tensor-to-image operations; not required by every CNN or LLM.", "Prioritize using target-model node inventories. Re-export or use a validated primitive decomposition where available; benchmark intermediate memory and copies."),
    "audio": ("P2", "Audio / signal preprocessing", "Important when preprocessing is embedded in ASR, speech or audio models. The existing DFT kernel does not provide all windowing, STFT or complex-transform operators.", "Precompute static windows/filterbanks, run an explicit host preprocessing stage, or add GPU signal-processing coverage. Validate normalization and complex-number conventions."),
    "recurrent": ("P2", "Legacy recurrent networks", "Simple RNN or legacy fused recurrent exports need this path; GRU and LSTM are already registered and should not be counted missing by name.", "Export a supported GRU/LSTM representation only when mathematically equivalent, or lower the recurrence to supported ops plus orchestration."),
    "random": ("P2", "Sampling / diffusion / stochastic models", "Useful for model-internal sampling and latent/noise generation. Many applications already generate randomness in their host pipeline.", "Supply host-generated inputs when appropriate; otherwise implement documented distributions, seeds and state/reproducibility without promising cross-backend bit equality."),
    "function": ("P2", "ONNX function / lowering", "No dedicated native registration, but the schema has a function body. Absence alone does not prove the model cannot run.", "Verify function expansion and every resulting operator/datatype on the actual ORT build; measure extra dispatches/materialization. Fused kernels are a performance/coverage choice, not automatically mandatory."),
    "llm_function": ("P1", "Modern standard ONNX transformer exports", "New standard attention/stateful operators have no dedicated native kernel here. Similarly named com.microsoft kernels are different contracts, not automatic substitutes.", "Validate function expansion, state/KV semantics and memory cost, or implement an explicit verified lowering/fusion to supported kernels. Do not change only the domain string."),
    "llm_fusion": ("P2", "Transformer fusion / export portability", "An exporter or optimizer may emit this specialized fusion even though much of its primitive computation is already available.", "Disable the incompatible fusion or re-export/decompose with correct masks, layout and state semantics. Add fusion support when end-to-end profiling justifies it."),
    "moe": ("P1", "Mixture-of-experts models", "com.microsoft.MoE is not registered (the registry entry is commented out). Quantized QMoE is registered but is not a drop-in replacement for floating-point MoE.", "Choose a supported quantized MoE export with validated accuracy, or implement floating-point routing/expert execution. Test routing determinism, top-k, capacity and empty experts."),
    "search": ("P2", "Autoregressive generation orchestration", "Fused BeamSearch/GreedySearch/Sampling nodes are optional export choices; host/ORT GenAI generation does not require these exact kernels.", "Keep the generation loop in a supported host/GenAI pipeline, or implement the fused operator only for models that embed it. Preserve beam/KV reorder and stopping semantics."),
    "sparse": ("P2", "Sparse / long-context / ragged models", "Large impact for selected sparse, packed or long-context architectures, but not a universal requirement for dense inference.", "Use a correct dense/padded fallback with explicit memory/latency cost, or add specialized kernels for target architectures."),
    "precision": ("P2", "FP8 / FP4 / alternate quantization", "These representation-specific kernels are not covered by float16/float32 or the existing integer weight-only path.", "Convert/requantize models with accuracy checks and documented storage cost, or implement the exact format. Do not conflate packed INT4 with floating-point FP4."),
    "string": ("P3", "Text / string preprocessing", "WGSL tensor math is not a natural fit for strings, tokenization, regex or image decoding. These may be essential to a whole app but not to its GPU numerical core.", "Keep an explicit CPU/WASM preprocessing stage, then pass numeric tensors to WebGPU. No-fallback all-GPU graphs need restructuring, not a silent success claim."),
    "tabular": ("P3", "Traditional ML / tabular", "Low priority for neural-network/LLM WebGPU work; potentially high importance if the product specifically targets tree ensembles, SVMs or feature pipelines.", "Keep CPU execution or select a dedicated tabular GPU strategy. Assess batch size and conversion cost before implementing all ai.onnx.ml kernels."),
    "training": ("P3", "Training / optimizer / dropout", "Not a requirement for inference-only Chrome WebNN. Inference Dropout can often be removed, but training masks, losses and gradients are separate contracts.", "Export an inference graph and verify elimination. Retain explicit unsupported status if training behavior is requested."),
    "legacy": ("P3", "Deprecated / compatibility exports", "Old or provider-specific export dialect; generally lower value than the modern standard replacement.", "Re-export or version-convert to modern ONNX and validate equivalence. Do not assume a current kernel accepts obsolete opset attributes."),
    "internal": ("P3", "Provider-specific / internal contracts", "Layout, fused, serialized-context or framework escape-hatch operators are tied to another backend or integration, not a general missing WGSL primitive.", "Use an unoptimized portable model or the matching target backend. EPContext needs integration-specific compile/load validation rather than an ordinary arithmetic kernel."),
    "distributed": ("P3", "Distributed / multi-device inference", "Collectives and sharded model execution are not typical single-device browser workloads and need orchestration/communication beyond a local kernel.", "Use an appropriate distributed backend or export a single-device model; do not prioritize these ahead of browser semantics and common inference gaps."),
    "special": ("P2", "Specialized model / export contract", "Relevant when a target model contains this exact operation; no workload-frequency evidence was collected for this audit.", "Inspect the schema and real model use before implementation. Evaluate a verified standard-op decomposition or explicit host boundary."),
}

STANDARD_GROUPS = {
    "webnn": "Or Xor IsInf IsNaN Round Sign Softsign",
    "quant": "QuantizeLinear DynamicQuantizeLinear QLinearConv QLinearMatMul ConvInteger MatMulInteger",
    "dynamic": "NonMaxSuppression NonZero Unique Compress",
    "composition": "Mean Sum Mod OneHot BitCast BitShift BitwiseAnd BitwiseOr BitwiseNot BitwiseXor CumProd TensorScatter",
    "control": "Loop Scan Optional OptionalGetElement OptionalHasElement SequenceAt SequenceConstruct SequenceEmpty SequenceErase SequenceInsert SequenceLength SequenceMap SplitToSequence ConcatFromSequence",
    "shape": "ConstantOfShape Size EyeLike",
    "vision": "DeformConv RoiAlign MaxRoiPool MaxUnpool Col2Im SpaceToDepth GlobalLpPool LpPool LRN",
    "audio": "STFT MelWeightMatrix BlackmanWindow HammingWindow HannWindow",
    "recurrent": "RNN ReverseSequence",
    "random": "RandomNormal RandomNormalLike RandomUniform RandomUniformLike Multinomial Bernoulli",
    "llm_function": "Attention LinearAttention CausalConvWithState",
    "string": "ImageDecoder StringConcat StringSplit StringNormalizer RegexFullMatch TfIdfVectorizer",
    "training": "Dropout NegativeLogLikelihoodLoss SoftmaxCrossEntropyLoss",
    "legacy": "Affine Crop DynamicSlice GroupNorm ImageScaler ParametricSoftplus Scale ScaledTanh Scatter Upsample",
    "special": "Det Hardmax",
}

CONTRIB_GROUPS = {
    "llm_fusion": "BiasSoftmax EmbedLayerNormalization FusedGemm FusedMatMul FusedMatMulActivation GemmFastGelu GatedRelativePositionBias RelativePositionBias DecoderAttention DecoderMaskedMultiHeadAttention DecoderMaskedSelfAttention GemmaRotaryEmbedding TorchEmbedding TransposeMatMul WordConvEmbedding",
    "search": "BeamSearch GreedySearch Sampling WhisperBeamSearch NGramRepeatBlock BifurcationDetector",
    "sparse": "LongformerAttention SparseAttention PackedAttention PackedMultiHeadAttention RemovePadding RestorePadding SparseToDenseMatMul ShrunkenGather",
    "moe": "MoE",
    "precision": "DequantizeBFP QuantizeBFP DequantizeWithOrder QuantizeWithOrder GemmFloat8 MatMulBlockQuantizedFp4Weight MatMulBlockQuantizedFp8Weight MatMulFpQ4",
    "quant": "DynamicQuantizeLSTM DynamicQuantizeMatMul MatMulInteger16 MatMulIntegerToFloat MulInteger ReduceSumInteger",
    "vision": "ConvTransposeWithDynamicPads CropAndResize MaxpoolWithMask UnfoldTensor",
    "audio": "ComplexMul ComplexMulConj Rfft Irfft DynamicTimeWarping",
    "string": "Tokenizer MurmurHash3",
    "training": "BiasDropout BitmaskBiasDropout BitmaskDropout IsAllFinite",
    "legacy": "AttnLSTM ExpandDims GatherND GridSample Pad Range Trilu Unique QuantizeLinear DequantizeLinear",
    "internal": "EPContext Snpe SampleOp NhwcConv NhwcFusedConv NhwcMaxPool",
    "special": "Inverse CDist",
}


def assessment(row):
    name, domain = row["name"], row["domain"]
    if domain.startswith("ai.onnx.preview.training"):
        group = "training"
    elif domain == "ai.onnx.ml":
        group = "tabular"
    elif domain == "ai.onnx.preview":
        group = "function"
    elif domain not in {"ai.onnx", "com.microsoft"} or name.endswith("_TRT"):
        group = "internal"
    elif domain == "com.microsoft" and (name.startswith("Distributed") or name in {"AllGather", "AllReduce", "AllToAll", "ShardedMoE"}):
        group = "distributed"
    elif domain == "com.microsoft" and name.startswith(("QLinear", "QOrdered", "QAttention", "QEmbed", "QGemm")):
        group = "quant"
    else:
        groups = STANDARD_GROUPS if domain == "ai.onnx" else CONTRIB_GROUPS
        group = next((key for key, names in groups.items() if name in names.split()), None)
        if group is None and row["functionSchema"]:
            group = "function"
        if group is None:
            raise ValueError(f"Unreviewed absent operator: {domain}::{name}")
    priority, workload, importance, mitigation = GROUPS[group]
    return {"priority": priority, "group": group, "workload": workload,
            "importance": importance, "mitigation": mitigation}


def limitation(identifier, ops, priority, title, finding, impact, test, path, needle):
    return {"id": identifier, "ops": ops.split(), "priority": priority, "title": title,
            "finding": finding, "impact": impact, "test": test, "path": path, "needle": needle}


LIMITATIONS = [
    limitation("int64", "Add Sub Mul Div Max Min Equal Greater Less GreaterOrEqual LessOrEqual", "P0",
               "Int64 registration is not full-width arithmetic",
               "enableInt64 adds registrations, but the binary implementation explicitly documents low-32-bit i32 arithmetic and incorrect results outside the int32 range. Even small inputs can overflow intermediate results. This does not imply all int64 movement or Cast paths are truncated.",
               "Silent numerical errors are more serious than an unsupported-node error; full WebNN int64 semantics cannot be inferred from the option.",
               "Test ±2^31, ±2^32, int64 extrema, unsigned-looking low words, comparisons and overflowing products against a reference; gate capability or implement full semantics.",
               CORE + "math/binary_elementwise_ops.cc", "NOTE: int64 in the WebGPU shader"),
    limitation("types", "MatMul Gemm Conv ConvTranspose LSTM GRU Softmax", "P1", "Limited numeric types",
               "The common floating-point kernel type list is float32/float16, not float64, bfloat16 or general FP8. Common numeric lists add int32/uint32; uint8, bool and int64 support is operator-specific, not universal.",
               "A supported name can still fail on a legal ONNX/WebNN dtype; blanket type support statements are unsafe.",
               "Build a per-op dtype matrix; test permitted FP16 hardware/features, unsupported dtype rejection and accuracy before conversion.",
               CORE + "webgpu_supported_types.h", "using SupportedFloats"),
    limitation("cast", "Cast", "P1", "Cast destination coverage",
               "The Cast shader has cases for float16, float32, int32, uint32, uint8, bool and int64. int8, uint16/int16, uint64, float64, bfloat16 and FP8 are not general destinations here. Current float→int64 code has a separate full-width conversion path; do not apply the binary low-32-bit statement indiscriminately.",
               "Chromium inserts Cast nodes, including signed/narrow quantization conversions. WebNN logical uint8↔bool conversions also need exact semantics.",
               "Test every exposed source/destination pair, boundary values, NaN/Inf and packed outputs; inspect kernel type matching and runtime destination checks.",
               CORE + "tensor/cast.cc", "switch (to_)"),
    limitation("pow", "Pow", "P1", "No int64 Pow registration",
               "The binary registration code deliberately excludes int64 from Pow, including its exponent type constraint.",
               "Mathematical and exported graphs can fail even with enableInt64=1.",
               "Test base/exponent type combinations and negative/large exponents; do not use float conversion as an exact integer substitute.",
               CORE + "math/binary_elementwise_ops.cc", "TODO: the ONNX Pow schema"),
    limitation("gridsample", "GridSample", "P1", "GridSample opset and dimensionality gap",
               "Only opsets 16–19 are registered. The implementation accepts some newer mode spellings but requires 4-D NCHW input/grid with final grid dimension 2. That is not opset-20+ or volumetric support.",
               "Recent vision exporters can miss the kernel despite a familiar operator name.",
               "Test opsets 16/19/20/22, 2-D versus 3-D, all interpolation/padding modes and empty dimensions; add explicit version/shape coverage.",
               CORE + "tensor/grid_sample.cc", "X_shape.NumDimensions() == 4"),
    limitation("conv3d", "Conv FusedConv", "P2", "Conv3D exists, but grouped Conv3D does not",
               "The native path supports 1-D/2-D/3-D convolution. Rank above five is rejected; the 3-D path requires group=1.",
               "Grouped/depthwise volumetric or video models remain a gap. The old JS documentation's blanket 'conv3d unsupported' statement is stale for native WebGPU.",
               "Test group=1 and >1 in rank-five models; benchmark the existing naive path separately from correctness.",
               CORE + "nn/conv.cc", "if (rank > 5)"),
    limitation("convtranspose", "ConvTranspose", "P2", "ConvTranspose3D absent",
               "The native implementation rejects ranks other than the 1-D/2-D paths.",
               "Volumetric decoders and segmentation upsampling can fail; ordinary 2-D transposed convolution is present.",
               "Test rank-five input, groups, output_padding and output_shape; verify any rewrite is mathematically equivalent.",
               CORE + "nn/conv_transpose.cc", "Only Conv2d or Conv1d"),
    limitation("pool", "MaxPool", "P2", "MaxPool indices / storage order",
               "Pool implementation requires storage_order=0 and one output; MaxPool's optional Indices output is rejected.",
               "Pooling/unpooling segmentation pipelines need indices; values-only inference is not the same contract.",
               "Test both outputs, column-major storage, ties, dilations and ceil/padding rules.",
               CORE + "nn/pool.cc", "pool_attrs_.storage_order == 0"),
    limitation("batchnorm", "BatchNormalization", "P3", "BatchNormalization training mode",
               "trainingMode is explicitly rejected; inference BatchNormalization is registered.",
               "Not a blocker for a properly exported inference graph, but not training support.",
               "Verify training_mode=0 export and optional output behavior; explicitly reject training requests.",
               CORE + "nn/batch_norm.cc", "trainingMode is not supported"),
    limitation("rnnactivations", "GRU LSTM", "P1", "Recurrent activation subset",
               "Native GRU and LSTM map sigmoid, tanh and relu activations to WGSL; other valid ONNX recurrent activation names throw.",
               "Default GRU/LSTM models have kernels, while non-default activation exports can fail. Double precision is also outside the common float type list.",
               "Test supported directions/layouts/state/sequence lengths and each requested activation, including alpha/beta attributes where relevant.",
               CORE + "rnn/gru.cc", "Unsupported GRU activation"),
    limitation("scatter", "ScatterND ScatterElements", "P1", "Scatter reduction datatype restrictions",
               "Reduction paths accept int32, uint32 and float32; other registered data types such as float16 are rejected when reduction is not none.",
               "Accumulating indexed updates can fail even when a no-reduction scatter works; important to graph manipulation and some WebNN workloads.",
               "Exercise add/mul/min/max, duplicate indices, fp16 versus fp32 and negative indices; validate atomic numerical behavior.",
               CORE + "tensor/scatter_nd.cc", "if (reduction_ != ScatterNDReduction::None && !reducible)"),
    limitation("dequant", "DequantizeLinear", "P1", "Dequantization does not imply quantization completeness",
               "Native DequantizeLinear registers int8/uint8/int32/int4/uint4 input formats and float16/float32 scales/output in newer opsets. It does not register all newer ONNX FP8/FP4/bfloat formats. QuantizeLinear itself has no native registration.",
               "QDQ exports must be audited in both directions; weight-only decompression support does not make an activation-quantized graph supported.",
               "Test per-tensor, per-axis, blocked scales, omitted/asymmetric zero-points, signed packed values and zero/odd sizes.",
               CORE + "quantization/quantize_linear.cc", "DequantizeLinearConstraints()"),
    limitation("attention", "com.microsoft::Attention", "P1", "Legacy Attention masks and cache",
               "Built-in and plugin capability checks reject mask_index, past, past_sequence_length, present output and past_present_share_buffer for com.microsoft.Attention.",
               "A fused encoder/decoder model may fall back despite the kernel registration; this is distinct from standard ai.onnx.Attention.",
               "Test optional inputs/outputs and actual node placement with CPU fallback disabled; choose MHA/GQA only with a validated semantic conversion.",
               CORE + "webgpu_execution_provider.cc", "Current implementation does not support mask_index"),
    limitation("mha", "com.microsoft::MultiHeadAttention", "P1", "MultiHeadAttention packed inputs / padding mask",
               "Packed 5-D QKV/KV and key_padding_mask explicitly throw. The source does use past_key/past_value and supports causal execution; a blanket 'no past/present' claim is outdated.",
               "Transformer export layout and mask representation determine support.",
               "Test separate versus packed Q/K/V, padding and additive attention bias, causal/noncausal, cache growth and output shapes.",
               CONTRIB + "bert/multihead_attention.cc", "Packed QKV of shape"),
    limitation("gqa", "com.microsoft::GroupQueryAttention", "P1", "GQA noncausal / sliding-window cache",
               "causal=0 and sliding_window_cache=1 are rejected. Sliding-window attention via local_window_size is not the same as CUDA's compacting sliding_window_cache mode.",
               "Bidirectional/vision attention and certain long-context cache exports cannot assume the decoder GQA kernel covers them.",
               "Validate chosen cache/layout/causal attributes and shared-KV special cases; do not silently drop unsupported attributes.",
               CONTRIB + "bert/group_query_attention.h", "causal=0 is not implemented"),
    limitation("paged", "com.microsoft::PagedAttention", "P1", "PagedAttention feature subset",
               "Requires causal operation. Guards reject nonzero softcap, non-SEPARATE layout, unequal V head size, rotary offset, smooth softmax, quantized/non-float16 cache, slot_mapping, q/k norm weights and k/v scales.",
               "A registered PagedAttention name is not broad serving-model compatibility.",
               "Test every requested cache format, optional input and paging boundary, including multi-request isolation; use the source-supported subset only.",
               CONTRIB + "bert/paged_attention.cc", "Feature guards for combinations"),
    limitation("nbits", "com.microsoft::MatMulNBits", "P1", "Weight-only MatMulNBits variants",
               "The constructor allows 2/4/8-bit weights, but group_idx is rejected and explicit zero-points must be packed uint8. Support is not limited to INT4, nor does it cover all quantization formats.",
               "Group-index/activation-order and alternate zero-point exports can fail; relevant to LLM model builders.",
               "Test bit width, block size, odd dimensions, bias, zero-points, accuracy settings and the exact exported weight packing.",
               CONTRIB + "quantization/matmul_nbits.cc", "group_idx as input"),
    limitation("bnb", "com.microsoft::MatMulBnb4", "P2", "MatMulBnb4 transpose constraint",
               "Built-in GetCapability rejects transB=0. The plugin capability implementation inspected here does not mirror this specific guard; registration parity does not guarantee rejection/fallback parity.",
               "Export options and built-in versus plugin builds can behave differently.",
               "Run transB=0/1 models on both deployment paths; verify error/fallback rather than assuming the guard is shared.",
               CORE + "webgpu_execution_provider.cc", "Current implementation only supports the forward case"),
    limitation("linear", "com.microsoft::LinearAttention", "P2", "LinearAttention state window",
               "state_window > 0 is explicitly unsupported (CUDA-only path); ordinary registered LinearAttention is not every stateful variant.",
               "Hybrid/state-space model exports can depend on this feature.",
               "Test initialization, update and reset of state, window length and the exact model's export attributes.",
               CONTRIB + "bert/linear_attention.cc", "does not support state_window"),
    limitation("delta", "com.microsoft::GatedDeltaNet", "P2", "GatedDeltaNet capture/state-update variants",
               "The native kernel rejects positive state_update_capacity and a capture_count input.",
               "Speculative/state-history variants can be missing despite basic hybrid-model support.",
               "Exercise decode/prefill and every optional state-management input in the target export.",
               CONTRIB + "bert/gated_delta_net.cc", "does not support state_update_capacity"),
    limitation("mrope", "com.microsoft::MRotaryEmbedding", "P2", "MRotaryEmbedding cache updates",
               "Updating cos_cache and sin_cache is not implemented.",
               "Multimodal models requiring runtime cache growth need another export or cache management strategy.",
               "Precompute adequate caches where valid; test offset/length boundaries and optional updated-cache outputs.",
               CONTRIB + "bert/mrotary_embedding.cc", "Updating cos_cache and sin_cache"),
    limitation("ngram", "com.microsoft::NGramHashMapping", "P2", "NGram token IDs",
               "Native NGramHashMapping requires int32 IDs.",
               "An exporter using int64 IDs cannot infer compatibility from registration alone.",
               "Range-check before converting IDs; verify hashing equivalence and overflow behavior.",
               CONTRIB + "bert/ngram_hash_mapping.cc", "only supports int32 ids"),
]
