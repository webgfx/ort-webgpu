# Qwen3.5-2B correctness investigation

## Conclusion and scope

The direct-ORT benchmark and GenAI do not use identical GPU math settings.
On the local NVIDIA T1000, changing only Dawn's
`d3d_disable_ieee_strictness` toggle reproduces the entire observed logit
difference. With matching strict device settings, all logits agree bit-for-bit
through complete 128-token generations. The production benchmark's captured,
GPU-sampling path also matches GenAI on the tested natural-language prompt.

This is strong evidence for an execution-policy mismatch, not a token-selector
or cache-update bug in these local tests. **The original Panther Lake
second-token failure remains unverified under aligned settings.** WinRM
authentication on webgfx-32 is currently failing (`0x80090304`); it is not
appropriate to certify that device or restore Qwen2B performance claims yet.
No model, main benchmark implementation, driver or daily performance data was
changed by this investigation.

## Pinned artifacts

- ORT: `09dfa6ad06ed8072b2fbe57687d4f71c7914b025`.
- GenAI: `a7b5804bde01a7c9f4ef2c9edc038a024b22355f`.
- Same ORT DLL as the original comparison; SHA-256
  `e9f8a85dba5463eb1cf76177a0c2643d67dafd8fe3006e23d087718755f9c193`.
- Local GPU: NVIDIA T1000, PCI `1FB0`, driver `32.0.15.9671`, D3D12.
- Qwen model ONNX graph, external weights and `genai_config.json` were each
  SHA-256 checked against the original saved request. All match.
- Batch 1, greedy selection, repetition penalty 1, 8192-token KV capacity,
  EOS suppression for the requested 128 output tokens. These are equivalence
  workloads, not a model-accuracy evaluation.

## What was isolated

ORT's `WebGpuContext::GetEnabledDeviceToggles()` unconditionally enables
`d3d_disable_ieee_strictness`. The benchmark's externally supplied `NativeGpu`
device leaves it disabled. Dawn's D3D12 compute-pipeline compiler passes this
setting into shader compilation. This changes floating-point evaluation rules;
the observed differences already begin during prefill and accumulate through
the recurrent layers, before output-token feedback.

For the saved 1024-token synthetic prompt on T1000:

| Experiment | Prefill logits versus new native | First decode logits versus new native |
|---|---:|---:|
| GenAI, normal internal ORT device | 247,330 values differ; max absolute delta 0.51171875 | 245,798 differ; max delta 0.22265625 |
| GenAI, externally supplied strict device matching the benchmark | Exact, all 248,320 values | Exact, all 248,320 values |
| Same external device policy, changing only `d3d_disable_ieee_strictness` to enabled | Exactly reproduces normal GenAI logits | Exactly reproduces normal GenAI logits |

The last experiment holds the GPU, model, ORT/GenAI binaries, input, requested
features, validation and buffer logic constant. It is a causal toggle A/B on
T1000, not just a comparison of two unrelated implementations. Matrix-feature
availability may additionally matter on Panther Lake and has not been ruled out
there.

Additional controls:

- Removing native decoder shape specialization leaves local logits unchanged.
- Using a growing attention-mask shape instead of the padded 8192-element
  decoder mask leaves local logits unchanged.
- Explicit GenAI `enableInt64=1` leaves the default local GenAI logits unchanged.
- GenAI intra-op thread count 1 also leaves those logits unchanged.
- Prefill token IDs, position IDs, attention mask and zero-initialized recurrent
  inputs match. Shared KV inputs were not compared as immutable before/after
  snapshots: GenAI's public output accessor observes them after the run.
- The original Panther Lake capture-on/off diagnostics already agreed; this
  investigation does not attribute the issue to capture replay.

## Full-generation checks

1. Saved synthetic input1024, output128: strict direct ORT and strict GenAI
   agree on **all 31,784,960 logits** and all 128 selected tokens.
2. Natural-language chat prompt, input23, output128, two generations using reset
   / rewind: strict direct ORT and strict GenAI agree on every logit and every
   token in both generations. Resetting the native state also reproduces the
   first generation exactly. Prompt: “Explain the difference between a CPU and
   a GPU in three short paragraphs.” The model's reasoning is included in the
   output budget; this is not a claim that the answer completed within 128 tokens.
3. The existing **production** `ort_webgpu_benchmark.exe`, unchanged, was then
   run with graph capture and GPU sampling enabled on that natural prompt.
   Warm-up and measured repetition both match all 128 strict GenAI tokens.
   Execution evidence records GPU sampling, KV8192, 254 decode calls and
   **253 actual capture-replay messages**. Diagnostic logging was enabled;
   no performance numbers from these runs are published.

The synthetic sequence is repetitive and cannot by itself establish semantic
quality. The extra natural prompt and reset checks improve coverage, but do not
certify all prompts, lengths, hardware or model accuracy.

## Diagnostic configuration pitfall

The pinned GenAI configuration setter can create duplicate provider entries
when called with lowercase `webgpu` after provider names have been normalized
to `WebGPU`. The first entry is subsequently selected, potentially ignoring
new options. Initial lowercase-setter experiments under `local-initial` and
`local-options` are superseded wherever provider options were varied.

The decisive experiments under `local-canonical`, `local-full` and
`local-natural` use canonical **`WebGPU`**, with the external-device warning and
actual Dawn toggles recorded. The webgfx-china conformance reference helper now
uses the canonical spelling and the correct `dawnBackendType` option so these
controls are not silently lost. The original performance comparison used staged
configuration files, not these diagnostic setter calls.

## Evidence and remaining work

All diagnostic code, binaries, requests, full logits and state dumps are under
`gitignore/benchmarks/qwen2b-audit-20260924/`:

- `diagnostic.cpp`, `run.py`, `build.ps1`: isolated experiments.
- `local-canonical/`: decisive device-toggle A/B.
- `local-full/`: all 128 synthetic-prompt logit arrays.
- `local-natural/`: natural-prompt arrays across two resets.
- `production-natural/summary.json`: unchanged production benchmark check.

Once remote authentication is restored, repeat the canonical device-toggle A/B
on webgfx-32 with the original saved input, then rerun full-sequence checks there.
Until then, the earlier report's Qwen3.5-2B exclusion remains in force. No
performance speedup should be attributed solely to generator optimizations
without accounting for these different device math/feature policies.
