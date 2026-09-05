# Methodology

## Snapshot definition

Create one point per calendar month. For month `YYYY-MM`, use the latest commit
reachable from the configured ref whose commit timestamp is no later than
`YYYY-MM-02T23:59:59Z`. Resolve ORT and ORT GenAI independently against the same
cutoff and record both hashes.

This is a source snapshot trend, not a release trend. Do not replace monthly
commits with tags because release cadence and source age differ between the two
repositories.

## Isolation and backups

Build detached worktrees below the configured worktree root. The primary
checkouts may contain valuable untracked build products and must not be modified.

Use a short worktree root (`E:\ort-wt` by default) for temporary source/build
trees. Recent Dawn/DXC snapshots exceed Windows compiler path limits when nested
below the longer archive output path. Keep only worktrees there; archives, logs,
results, and the report remain under `outputRoot`.

Store the self-contained trend backup under the configured `outputRoot`
(`E:\backup\ort\perf-trend` on the reference machine). Put monthly binaries under `runtime`, complete ONNX
WebGPU model packages under `model`, and keep manifests, raw logs, benchmark
results, experiments, and download cache at the same archive root. Generate the
maintained `index.html` beside the project scripts. The configured model paths
must point to the backed-up model copies.

Pass `CMAKE_POLICY_VERSION_MINIMUM=3.5` to old dependency trees when using a
modern CMake release. This is a build-system compatibility policy, not a source
patch; record it with every build command.

Use the Visual Studio 2022 bundled CMake 3.31 executable for the historical
range. CMake 4 removes OLD policy behavior required by the December 2024
Dawn/DXC dependency tree. Start each snapshot from clean build directories so a
retry never mixes generators or CMake versions.

Enable standard MSVC C++ exception unwind semantics (`/EHsc`) and suppress
warning C4834 for GenAI through the configured C++ flags. The 2024 tree promotes
warnings to errors, uses exception handlers without declaring unwind semantics,
and discards one `[[nodiscard]]` return value. Keep all other warnings enabled
and do not edit the source snapshot.

Some 2024 ONNX Runtime Extensions code uses a valueless
`OCOS_ENABLE_VENDOR_IMAGE_CODECS` macro and nested `WIN32` macro in `#if`.
Current MSVC rejects that syntax. After FetchContent materializes the temporary
dependency, replace only those conditions with `#if defined(...)`. This
preserves their meaning, does not modify either repository, and must be recorded
in `build-metadata.json`.

Archive the runtime needed to reproduce each benchmark:

- `onnxruntime.dll`
- `model_benchmark.exe`
- `onnxruntime-genai.dll` for dynamically linked snapshots, or the static GenAI
  library as supporting build evidence for statically linked historical snapshots
- `dxcompiler.dll` and `dxil.dll` when produced
- source hashes, commit dates, subjects, commands, host, and build timestamps

The full object/build tree is not the backup. It is temporary and can be removed
after the required binaries and logs are verified.

The December 2024 `model_benchmark` links `onnxruntime-genai-static`; later
versions link the shared GenAI runtime. Detect this from that snapshot's
`benchmark/c/CMakeLists.txt`, record `genaiLinkage`, and require only the runtime
files appropriate to that linkage.

Historical GenAI `build.py` versions also disagree on the CMake build root.
After configuration, locate the directory that actually contains
`CMakeCache.txt`; it may be `build\Windows` or `build\Windows\Release`. Build
`model_benchmark` from that detected root rather than assuming one layout.

GitLab may regenerate ZIP bytes for an unchanged Eigen commit, invalidating an
old archive SHA1. Before changing a hash, download the configured URL, require
the ZIP root directory to contain the exact 40-character commit pinned by the
URL, compute the current SHA1, and record the replacement in build metadata.
Never accept an archive whose root does not identify that commit.

Cap Windows builds at four parallel jobs. Higher concurrency produced CMake
timestamp access races and compiler-generated-file failures on the reference
machine; throughput is less important than consistent monthly builds.

For the August 2024 DML baseline, prepopulate the exact
`Microsoft.AI.DirectML` version declared by that snapshot's `packages.config`.
Its bundled NuGet 5.3 client cannot negotiate TLS with current nuget.org.
Download the pinned NUPKG through the current client, validate the archive path,
extract it into the build's expected packages directory, and record the
adjustment in build metadata.

The August 2024 `ort_mutex.h` uses `std::chrono` while relying on a transitive
MSVC include. Current MSVC no longer provides it. Add a direct `<chrono>` include
only in the temporary worktree and record the compatibility adjustment.

DirectML 1.15.1 uses the Windows-provided `D3D12Core.dll`; the NuGet package does
not contain a copy. Stage and archive the exact `System32\D3D12Core.dll` used by
the benchmark so the milestone folder contains every DLL expected by the
historical GenAI build.

## Benchmark matrix

Use only the milestone-aligned Phi-4 ONNX WebGPU exports backed up below
`model\aligned`. Each export must retain `model-metadata.json` with its builder
commit/date, source revision, cumulative builder options, mapped stages, file
sizes, and SHA-256 hashes.

Run batch size 1 and prompt length 1024 with generation length 128. Request an
8192-token maximum sequence/KV-cache length with `-ml 8192` for the C++ runner
or `-m 8192` for the Python runner. Detect whether historical C++ binaries expose
`-ml`; if they do not, record the effective prompt + generation length instead
of claiming an 8K run. Use one warmup and five measured repetitions. Let each
model's `genai_config.json` select WebGPU; do not pass a different execution
provider.

Record:

- prompt processing throughput (prefill TPS)
- token generation throughput (decode TPS)
- prompt processing latency (TTFT)
- end-to-end latency when emitted
- peak working set when emitted

Do not compare runs across different GPUs, GPU drivers, power modes, model
exports, graph-capture settings, prompt/generation lengths, warmup counts, or
repetition counts as one continuous series.

## Historical compatibility

The current model exports were generated by a later ORT GenAI builder. Older
runtimes may reject a newer graph/config or lack a required operator. Phi-4 also
postdates the first requested snapshot.

Apply these rules:

1. Build the exact monthly pair without source patches.
2. Run the unchanged model export and fixed workload.
3. If build or load fails, retain the complete log and mark the point failed.
4. Start the plotted line at the first successful point.
5. Keep failed earlier months visible in the status timeline.
6. Patch or regenerate a historical model only as a separate experiment with a
   new series identity.

### Phi-4 milestone alignment

Phi-4 postdates the first two runtime optimizations. Use only two model
identities: separate GQA RoPE and fused GQA RoPE. Generate both as full INT4
with `is_symmetric=true`, `accuracy_level=4`, `block_size=32`, `algo_config=rtn_last`,
`enable_webgpu_graph=true`, and `shared_embeddings=true`, while keeping graph
capture disabled in the stored GenAI config. Current post-prune exports also
use `prune_lm_head=true`; pre-prune controls keep it false or omitted. Use the deprecated `int4_` aliases
only when a pinned historical builder predates the canonical names. A
graph-capture measurement may enable capture on a temporary config copy; it
must not create a third model identity. Runtime-only changes, including LM-head
pruning support, do not create additional model variants.

Treat `shared_embeddings=true` as mandatory for Phi-4 because its source
configuration ties the input embedding and LM head. It remains invalid for
models whose source configuration declares untied embeddings.

Keep the report's builder-option registry in `milestones.json` synchronized
with the standard profile. For every tracked option, record its concise
description, whether it is mandatory, conditional, or optional now, current
default, historical aliases, and every dated default or naming change with the
exact ORT GenAI PR and commit. Keep history newest first. Show the currently
required settings and the full five-option pinned profile as copy-friendly
lines above the detailed table; do not include experiment-only options there.

Keep a separate `genai_config.json` registry for WebGPU-specific provider
settings only. Record the profile value, current default, requirement, and
concise behavior. Stored model configs keep graph capture disabled;
controlled graph-capture measurements may enable it only on a temporary copy.

For old GQA schemas, trimming only trailing empty optional inputs is an allowed
compatibility normalization: no real tensor input, attribute, or graph
operation changes. Record the normalized ONNX hash. Do not apply semantic model
rewrites to make a historical runtime load.

If a milestone itself changes the exported graph, archive a matched
control/candidate pair under `experiments` and keep it separate from the two
maintained historical packages. The pair must differ only in the milestone
option. A later cumulative point may reuse the candidate artifact when the
model-side change is required to preserve the optimization.

## Interpreting trends

Use median-like aggregated output emitted by `model_benchmark`; do not graph a
failed point as zero. Treat a single monthly change as a signal to investigate,
not proof of causality. When a regression appears:

1. repeat both adjacent snapshots on the same machine;
2. check driver, power, thermal, and background-load changes;
3. compare exact ORT and GenAI commit ranges;
4. narrow with additional commits only after the monthly result reproduces.

The trend measures the combined ORT + ORT GenAI source stack. It does not isolate
which repository caused a change.
