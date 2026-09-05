---
name: perf-trend
description: "Collect and archive ORT WebGPU performance data for new features that improve prefill, TTFT, decode, or token generation, including controlled baseline-versus-feature benchmarks. Also use for Phi-4 milestone trends, historical runtime builds, builder/model compatibility, regression analysis, and report updates. Trigger when the user says a new feature improves prefill or decode and asks to collect data, benchmark it, compare performance, or add it to the trend."
---

# ORT WebGPU Phi-4 Performance Trend

Build one reproducible WebGPU runtime snapshot per month from December 2, 2024,
benchmark Phi-4 with a fixed workload, preserve the binaries and raw logs, and
render a self-contained HTML trend report.

Run commands from the directory containing this `SKILL.md`. Resolve that
directory from the loaded skill path rather than assuming the user's current
working directory.

Read [README.md](README.md) first. It is the operator runbook for configuring a
new machine, regenerating the report, rerunning benchmarks, rebuilding exact
runtime snapshots, and executing the verification gate.

The milestone runtime configuration builds exact ORT/ORT GenAI pairs for
performance events that fall between the monthly snapshots.

## Safety contract

- Treat the configured ORT and ORT GenAI paths as source repositories. Never
  reset, clean, stash, switch, or build in their working trees.
- Use detached Git worktrees below the configured short worktree root. Existing
  untracked build/install files in the source repositories must remain untouched.
- Resolve each monthly snapshot to the latest commit at or before 23:59:59 UTC
  on the second day of that month. Record both exact commits.
- Pair ORT and GenAI by the same monthly cutoff. Do not substitute release tags
  or later dependencies silently.
- Use the configured CMake policy minimum for legacy dependency compatibility;
  do not patch historical source trees merely to satisfy modern CMake.
- Use the configured Visual Studio 2022 CMake 3.x executable. CMake 4 cannot
  configure the oldest Dawn/DXC dependency snapshot.
- Suppress only the configured historical MSVC warning needed to build the 2024
  GenAI tree; retain `/WX` for every other warning.
- Allow only the documented valueless-macro compatibility edit in the temporary
  fetched Extensions tree; record it in build metadata.
- Accept a regenerated Eigen archive hash only after verifying that its ZIP root
  identifies the exact commit pinned by the historical URL; record the new hash.
- Use the configured bounded build parallelism to avoid Windows/CMake file races.
- Preserve failed build and benchmark logs. Historical incompatibility is data,
  not a reason to rewrite old source or report a synthetic result.
- Run all comparable points on the same physical GPU, driver, power mode, build
  configuration, model files, and benchmark matrix.

Read [references/methodology.md](references/methodology.md) before changing the
snapshot rule, benchmark matrix, compatibility policy, or report interpretation.

## Reference-machine inputs

- ORT: `E:\workspace\project\agents\onnxruntime`
- ORT GenAI: `E:\workspace\project\agents\onnxruntime-genai`
- Phi-4:
  `E:\backup\ort\perf-trend\model\aligned\standard-unfused-rope\Phi-4-mini-instruct`
- Data archive: `E:\backup\ort\perf-trend`
- Build backups: `E:\backup\ort\perf-trend\runtime`
- Workload: batch size 1; prompt length 1024; generation length 128; ORT GenAI
  maximum KV-cache length 8192 (`-ml 8192` for C++ `model_benchmark.exe`,
  `-m 8192` for Python `benchmark.py`); five measured repetitions after one warmup

Create `..\gitignore\perf-trend\config\config.local.json` from
[config.example.json](config.example.json) for each machine. Local configs are
kept in the categorized, Git-ignored root folder. Use environment variables or
paths relative to the config file; do not commit machine-specific drive paths.

## Workflow

### Collect data for a new feature

When the user says a new feature improves prefill or decode and asks to collect
data, treat that as a request to run a controlled feature measurement, not just
to explain the existing report.

1. Determine from the request and repository history:
   - feature name and whether it targets prefill, decode, or both;
   - every owning repository (ORT and/or ORT GenAI), candidate commits,
     branches, and PRs, plus the intended parent/control commit (normally each
     candidate's first parent unless the user names another comparison);
   - whether the change is runtime-only or also requires a new model export,
     builder option, or GenAI change.
   Ask only for inputs that cannot be established safely from the repositories.
2. Keep the comparison isolated. For runtime-only changes, benchmark the exact
   same archived model with baseline and candidate runtimes. If the feature
   requires a model change, archive and identify both model packages and pin
   their builder commit, source revision, options, sizes, and hashes. Hold all
   unrelated runtime, model, workload, GPU, driver, and power variables fixed.
3. Add explicit baseline and candidate snapshots to a local copy of
   `milestone-runtime-config.example.json`. Build both through
   `ort-webgpu-perf-trend.py` so detached worktrees, build metadata, binaries,
   and logs are preserved. Never build in or switch the configured source
   repositories.
4. Run the fixed batch-1/prompt-1024/generation-128 workload with an 8K maximum
   KV cache (`-ml 8192` in C++ or `-m 8192` in Python), one warmup, and five
   measured repetitions for both snapshots. Detect historical C++ CLI support
   before passing `-ml`; if unavailable, record the effective prompt + generation
   length and do not claim the run used 8K. Preserve failed runs and raw logs;
   rerun both sides when a result is noisy or surprising.
5. Report prefill TPS for a prefill feature and decode TPS for a decode feature.
   Continue measuring and archiving TTFT, but do not add a separate TTFT chart.
   Calculate the baseline-to-candidate throughput delta. Include exact commits,
   model identity, device fingerprint, raw-log paths, and any compatibility
   adjustment.
6. Add the candidate as a new `milestones.json` stage and to only the impacted
   optimization group(s) when it belongs to the maintained cumulative Phi-4
   RTX 5080 series. Otherwise keep it as a labeled experiment and do not mix it
   into the published trend. Render and verify after any plan change.

Do not claim that the feature caused an improvement if baseline and candidate
differ in uncontrolled ways. Clearly label missing, failed, incomparable, or
statistically ambiguous data.

### Milestone-aligned trend

Use [milestones.json](milestones.json) and
[milestone_benchmark.py](milestone_benchmark.py) for the primary report. It
aligns exact ORT/GenAI commits, historical builder commits, cumulative Phi-4
model features, and RTX 5080 measurements:

```powershell
python .\milestone_benchmark.py run
```

The milestone report is written to `index.html` beside the script by default.
The published series contains only measured major performance improvements.
Keep parent/control measurements in the archive as attribution evidence, but do
not render them as chart or timeline nodes and do not label any published point
as a baseline.
Re-run only rendering and verification with:

```powershell
python .\milestone_benchmark.py render
python .\milestone_benchmark.py verify
```

The report organizes two intentionally cumulative event series.

Prefill:

1. 2025-01 subgroup MatMulNBits
2. 2025-01 DP4A MatMulNBits
3. 2025-02 GQA FlashAttention
4. 2025-08 shared-memory broadcasting
5. 2026-03 LM-head prune
6. 2026-05 larger dynamic FlashAttention max K-step

Decode:

1. 2025-04 Flash Decoding
2. 2025-11 fused RoPE
3. WebGPU graph capture
4. MatMulNBits MLP fusion
5. FlashAttention decode workgroup/K-tile tuning

Use prompt length 1024 for every point. Never mix the attached Alder Lake
reference values into the RTX 5080 series; keep it as milestone context only.
Each unique model backup contains `model-metadata.json` with its cumulative
options, builder/source revisions, mapped milestones, file sizes, and SHA-256
hashes.

Use exactly two maintained historical Phi-4 model packages: separate RoPE and fused RoPE.
New comparable exports use full INT4 with `accuracy_level=4`,
`is_symmetric=true`, `block_size=32`, `algo_config=rtn_last`, `enable_webgpu_graph=true`, and
`shared_embeddings=true`. Current post-prune exports also use
`prune_lm_head=true`; pre-prune controls keep it false or omitted. Pinned historical builders may require the deprecated
`int4_` aliases. Keep graph capture disabled in the stored `genai_config.json`.
Do not add model variants for runtime-only milestones.
When a milestone inherently changes the exported graph, keep its matched
control/candidate exports under `experiments` and do not present them as a
third maintained historical model. Later cumulative points may reuse the
candidate export when needed to retain that model-side optimization.
Trailing empty optional GQA inputs may be removed deterministically for old
schema compatibility because that does not change any real input or operator
attribute.

Every published milestone must explicitly record all landed PRs required to
enable the measured path. Trace umbrella PR cross-references and implementation
history; do not stop at a single headline PR when coordinated ORT and ORT GenAI
changes, model-builder support, or follow-up kernel fusions are part of the
measured milestone. Record the GitHub PR author as its owner. Keep unmerged
discussion/umbrella PRs out of the required list. Render direct links, owner,
merge date, merge commit, and role in reverse chronological order in the merged
milestone table. Use the date when the
complete required path was available, not merely the date of its first PR.
For each repository with required PRs, the tested runtime must use the exact
merge commit of the newest required PR. Never substitute a later monthly
snapshot. For a repository without a required PR, prefer a compatible same-date
commit. If no same-date commit can build against the required exact merge,
select the newest compatible earlier commit and explain that pairing in the
report.

Every published milestone must also record its contributors and a short
applicability/conditions note. Render these in the merged milestone table
together with the introduction, prefill/decode performance area, all required
PR links and roles, and measured impact
from an archived controlled parent-to-candidate run when available. Otherwise,
use the preceding published point in the same cumulative series and label that
comparison as non-isolated. Show the multiplier and raw throughput transition
without appending a redundant percentage.

Maintain a model-builder option registry in `milestones.json`. It must cover
the standard profile and model-changing experiment options, including current
canonical names, historical aliases, the value used for new comparable builds,
the current default, whether it is mandatory, conditional, or optional now,
and a newest-first history of default or naming changes with exact ORT GenAI
PRs and commits. Render it as a responsive report table and concatenate both
the currently required settings and full pinned profile above it. Keep a
separate responsive registry of WebGPU-specific `genai_config.json` provider
settings, including requirement, profile value, and current default.

Show subgroup, DP4A, shared-memory broadcast, larger K-step, and LM-head prune
only on the prefill chart. They are not decode
milestones.
The published report charts prefill throughput and decode throughput. Keep TTFT
in the archived raw measurements and validation data, but do not render a
separate TTFT chart.
Keep all primary plan entries, measurements, logs, and report sections
Phi-4-only. Preserve a known improvement with an explicit missing status when
no isolated runtime exists; never substitute a cumulative or synthetic value.

State the report boundary explicitly: Qualcomm-specific prefill/decode
optimizations, Whisper, Phi-4 multimodal, GPT-OSS, and other model or hardware
optimization work are outside this Phi-4 text-only RTX 5080 series and are not
represented in its charts or timeline.

### 1. Check prerequisites

```powershell
python .\ort-webgpu-perf-trend.py doctor
```

Require Git, CMake, Visual Studio 2022 C++ build tools, the two complete
repositories, and both ONNX WebGPU model directories. A dirty source checkout is
allowed because the pipeline uses worktrees. A shallow repository is not allowed
because old commits may be unavailable.

### 2. Plan monthly commits

```powershell
python .\ort-webgpu-perf-trend.py plan --fetch
```

This updates remote refs without changing either working tree and writes
`manifest.json` below the output root. Inspect the first and last entries and the
snapshot count before building.

Use `plan` without `--fetch` for an offline, local-ref-only plan.

### 3. Prove one historical snapshot

```powershell
python .\ort-webgpu-perf-trend.py run --snapshot 2024-12
```

This creates isolated worktrees, builds ORT WebGPU and GenAI's
`model_benchmark`, copies the required runtime files into
`E:\backup\ort\perf-trend\runtime\2024-12\`, runs the models, saves raw logs,
and updates the report.

Do not start the full range until this first snapshot either succeeds or fails
for a clearly recorded historical compatibility reason.

### 4. Run or resume the complete range

```powershell
python .\ort-webgpu-perf-trend.py run --all-pending
```

The manifest is the durable state machine. Re-running skips successful work and
resumes pending or failed snapshots only when `--retry-failed` is supplied:

```powershell
python .\ort-webgpu-perf-trend.py run `
  --all-pending --retry-failed
```

Use `--limit 1` while diagnosing the pipeline. Use `--keep-worktrees` only when
a failed build needs direct inspection; otherwise worktrees are removed after
each snapshot while archived binaries and logs are retained.

### 5. Re-run benchmarks without rebuilding

```powershell
python .\ort-webgpu-perf-trend.py benchmark `
  --snapshot 2025-06
```

Use this after a controlled benchmark rerun on the same device. Existing raw
results are overwritten only for the selected snapshot.

### 6. Refresh the report

```powershell
python .\ort-webgpu-perf-trend.py render
```

Open:

```text
.\index.html
```

The report shows prefill TPS and decode TPS by model and prompt length, TTFT in
the raw table, exact ORT and ORT GenAI source revisions and commit dates, and
build/benchmark gaps. A model's trend begins at its first successful compatible
snapshot; failed earlier points remain visible in the timeline.

## Output contract

Keep:

- `milestones.json` — primary milestone/runtime/model alignment
- `milestone-results.json` — RTX 5080 milestone measurements
- `manifest.json` — monthly commits and build/benchmark status
- `E:\backup\ort\perf-trend\runtime\<YYYY-MM>\` — backed-up runtime binaries and
  `build-metadata.json`
- `E:\backup\ort\perf-trend\model\<model>\` — complete benchmark model backups
- `results\<YYYY-MM>\results.json` — parsed metrics and device fingerprint
- `logs\<YYYY-MM>\` — complete command and benchmark output
- project `index.html` — self-contained milestone trend report

Never delete successful historical outputs during a refresh. If commit
resolution changes after a fetch, treat it as a new plan requiring explicit
review; do not attach old results to new commits.

## Validation

Before reporting completion:

```powershell
python -m unittest discover .\tests -v
python .\ort-webgpu-perf-trend.py doctor
python .\ort-webgpu-perf-trend.py plan --dry-run
python .\milestone_benchmark.py verify
```

For a real trend, also verify:

- every successful backup contains `onnxruntime.dll` and
  `model_benchmark.exe`, plus `onnxruntime-genai.dll` when that snapshot's
  benchmark links GenAI dynamically;
- every plotted point has a raw log, model identity, prompt/generation lengths,
  device fingerprint, and exact ORT/GenAI commits;
- no failed or missing month is converted to zero;
- the milestone plan and primary results contain only Phi-4;
- subgroup, DP4A, shared-memory broadcast, larger K-step, and LM-head prune are
  absent from the decode chart;
- parent/control measurements are absent from the charts and published
  timeline;
- no published point is named or described as a baseline;
- missing isolated measurements remain visible with a reason and are not
  converted to zero;
- the report and manifest agree on successful, failed, and pending counts.
