# Panther Lake: optimized native benchmark vs. GenAI

September 24, 2026. Device **webgfx-32**, Intel Core Ultra X7 358H,
**Intel Arc B390 (PCI B080)**, driver `32.0.101.8991`. Existing Balanced power
plan; AC connected, battery 100% when checked. No model, driver or power changes.

## Correctness before performance

The new benchmark's CPU and GPU paths match **all 128 generated tokens for all
five models** at input length 1,024. GPU selector tests also pass on Panther Lake:
54 FP16 checks, 54 FP32 checks, and three reductions covering all 63,488 finite
FP16 encodings. Checks include ties, signed zero, masked maxima, and rejection of
NaN/Infinity even at suppressed tokens.

Independent GenAI public-API comparison:

| Model | Complete 128-token agreement |
|---|---|
| Aion | Pass |
| Phi-4-mini-instruct | Pass |
| Qwen3.5-4B | Pass |
| Gemma-4-e2b-it | Pass |
| Qwen3.5-2B | **Differs at token index 1; excluded from accepted performance comparison** |

This is benchmark/output equivalence on the tested inputs, not a general model
accuracy certification. The accepted models also retain exact output agreement
across every repetition in both performance passes.

### Qwen3.5-2B discrepancy

Follow-up: [Qwen2B correctness investigation](20260924-qwen2b-correctness.md)
isolates Dawn floating-point strictness as the logit difference on both local
T1000 and Panther Lake, with full-token/logit agreement under matching settings.
The original Panther Lake discrepancy is reproduced exactly and explained.
The performance exclusion below is unchanged: no new matched-policy Qwen2B
timing experiment was performed.

Both implementations emit first token `12434`. The second-token logits differ:

| Candidate token | New native (capture on) | New native (capture off) | GenAI |
|---|---:|---:|---:|
| 63643 | 8.937500 | 8.937500 | 8.906250 |
| 12434 | 8.890625 | 8.890625 | 8.906250 |
| 220 | 8.8515625 | 8.8515625 | 8.843750 |

Native correctly selects `63643`. GenAI has an exact tie and correctly selects
the earlier index `12434`. The score changes are a few FP16 representable steps;
this is **not a wrong argmax or lost GPU-feedback token**. Turning native graph
capture off leaves these logits unchanged, so capture replay alone does not
explain it. The precise numerical execution-path difference between the direct
ORT harness and GenAI was not yet localized during this timing experiment; the
follow-up above now isolates the device floating-point strictness policy.
Original model files are identical.
We do not claim correctness-qualified Qwen2B speedups while full output differs.

## Performance comparison

Both paths load the same model files and **the same ORT DLL**. The aligned GenAI
harness uses the same saved prompt IDs, batch one, greedy decoding, repetition
penalty 1, EOS suppression, 8K KV capacity and generator reuse. TTFT includes the
first CPU-readable selected token; decode ends after all remaining 127 tokens
are CPU-readable.

The stock GenAI executable does not accept saved token IDs: `--use_random_tokens`
uses unseeded IDs in 0–99. Its “Prompt processing (time to first token)” excludes
first-token sampling. Therefore the main comparison uses a small harness around
the **unmodified GenAI public API**; stock executable numbers appear separately.

Input **1,024**, output **128**. Two blocks, one warm-up plus three timed
repetitions per model per block (**six timed samples per runtime/model**).
Order was GenAI → native, then native → GenAI. Models/runtimes were never run
concurrently on this GPU. Rates use mean elapsed time, not mean reciprocal rates.

| Model | Aligned GenAI decode TPS | New native decode TPS | Difference |
|---|---:|---:|---:|
| Aion | 50.43 | 49.70 | -1.5% |
| Phi-4-mini | 29.35 | 33.31 | +13.5% |
| Qwen3.5-4B | 22.31 | 28.98 | +29.9% |
| Gemma-4-e2b-it | 24.55 | 46.27 | +88.5% |

**Conclusion:** Gemma and Qwen4B show the clearest decode gains. Phi improves in
aggregate but is variable. Aion has no demonstrated benefit in this experiment.
This is not a universal native speedup, nor a guarantee of these exact percentages.

Timing variation must not be hidden:

| Model | Native sample range, TPS | GenAI sample range, TPS | First / reverse block gain |
|---|---:|---:|---:|
| Aion | 40.88–52.82 | 48.97–51.10 | -7.9% / +5.8% |
| Phi-4-mini | 28.24–37.73 | 24.21–31.99 | -0.1% / +30.7% |
| Qwen3.5-4B | 27.65–29.71 | 20.09–24.26 | +20.3% / +39.9% |
| Gemma-4-e2b-it | 45.87–46.68 | 23.26–25.57 | +85.1% / +91.9% |

TTFT throughput (tokens/s; higher is better), with the aligned timing boundary:

| Model | GenAI | New native | Difference |
|---|---:|---:|---:|
| Aion | 1647.94 | 1478.83 | -10.3% |
| Phi-4-mini | 1050.06 | 926.55 | -11.8% |
| Qwen3.5-4B | 657.51 | 787.35 | +19.7% |
| Gemma-4-e2b-it | 3359.10 | 2830.27 | -15.7% |

TTFT also varies between blocks. The decode improvements do not imply a
consistent prefill/TTFT improvement; native is slower in aggregate for three
models at this input length.

### Original GenAI executable, daily-style options

One warm-up, three timed repetitions, separate run with stock random prompts
and validation disabled as in the daily default. These are contextual numbers,
not an identical-workload TTFT comparison:

| Model | Stock GenAI decode TPS | New native aggregate TPS | Difference |
|---|---:|---:|---:|
| Aion | 50.46 | 49.70 | -1.5% |
| Phi-4-mini | 30.04 | 33.31 | +10.9% |
| Qwen3.5-4B | 24.25 | 28.98 | +19.5% |
| Gemma-4-e2b-it | 26.35 | 46.27 | +75.6% |

```text
model_benchmark.exe -i <staged-model> -l 1024 --use_random_tokens
  -g 128 -r 3 -w 1 --reuse_generator -ml 8192
```

## What differs in implementation

- Daily-style native uses GenAI's CPU search and generation loop. The new
  benchmark uses ORT directly with explicit static buffers and GPU feedback.
- The new harness captures all decoder sessions. GenAI retains the daily
  effective policy: Aion capture on, the other models off. This is an end-to-end
  implementation comparison, not a pure isolated argmax experiment.
- Gemma keeps CPU sampling in the new harness; its improvement comes from the
  overall session/buffer/host path, not GPU greedy selection. We have not
  attributed the entire improvement to a single operation.
- Both aligned paths request basic validation. New native robustness is enabled
  and its actual Dawn toggles are recorded. GenAI's initial shared-device option
  forwarding omits `enableRobustness`; logs confirm a later model-session request
  cannot change the supplied device. Effective robustness is **not proven equal**.
- Stock GenAI's prompt distribution and timing boundary differ as noted above.

## Reproduction and evidence

Pinned ORT: `09dfa6ad06ed8072b2fbe57687d4f71c7914b025`.
Pinned GenAI: `a7b5804bde01a7c9f4ef2c9edc038a024b22355f`.
ORT DLL SHA-256 on both paths:
`e9f8a85dba5463eb1cf76177a0c2643d67dafd8fe3006e23d087718755f9c193`.
This is a pinned comparison build, not a claim about the latest daily package.

Source: `native/genai-reference.cpp`, `native/run-genai.py`,
`native/diagnose-logits.cpp`, `native/diagnose-genai.cpp`,
`native/analyze-genai.py`. Machine-readable audit:
[20260924-ptl-native-vs-genai.json](20260924-ptl-native-vs-genai.json).

Local raw data, logs, configurations and hashes:
`gitignore/benchmarks/results/ptl-20260924`.
Remote staging (including isolated Python dependencies and model hardlinks):
`D:/workspace/project/ort-webgpu/gitignore/benchmarks/ptl-20260924` on webgfx-32.
Original models were not rewritten. No chart data import, model upload, driver
change, power change, or push was performed.

```powershell
python -B benchmarks/native/analyze-genai.py `
  gitignore/benchmarks/results/ptl-20260924 --validated-only --include-reverse `
  --output gitignore/benchmarks/results/ptl-20260924/recheck.json
```
