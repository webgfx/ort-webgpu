# ORT WebGPU Phi-4 performance trend

This project rebuilds historical ONNX Runtime (ORT) and ORT GenAI WebGPU
runtimes, benchmarks milestone-aligned Phi-4 models, and generates the
self-contained [report](index.html).

The primary published series uses an NVIDIA GeForce RTX 5080, batch size 1,
prompt length 1024, generation length 128, one warmup, and five measured
repetitions. New and rerun measurements also use an 8192-token maximum KV cache
(`-ml 8192` in C++ `model_benchmark.exe`, or `-m 8192` in Python
`benchmark.py`). Do not compare new points with the published series unless
the GPU, driver, power mode, model files, runtime configuration, and workload
all match. Older archived rows do not record the max-length argument independently
and must not be relabeled as 8K measurements without a rerun.

## Reproducibility levels

1. **Regenerate the report** from an existing archive: seconds; no build tools
   or model execution required.
2. **Rerun the benchmark** from archived runtimes and model packages: requires
   the target GPU and driver.
3. **Rebuild historical runtimes and rerun**: requires full ORT and ORT GenAI
   Git repositories plus Visual Studio build tools.

The two maintained Phi-4 model packages (separate RoPE and fused RoPE) are
controlled input artifacts. New comparable exports use full INT4 with
`accuracy_level=4`, `is_symmetric=true`, `block_size=32`, `algo_config=rtn_last`,
`enable_webgpu_graph=true`, and `shared_embeddings=true`; graph capture remains
disabled in the stored GenAI config. Current post-prune exports also use
`prune_lm_head=true`; pre-prune controls keep it false or omitted. Historical builder commits use the
equivalent `int4_`-prefixed option names when required. Each package must contain
`model-metadata.json` with its source revision, builder commit/date, builder
options, file sizes, and SHA-256 hashes. Bit-for-bit reproduction requires the
same archived model files; generating a new export creates a new series
identity.

`shared_embeddings=true` is mandatory for this Phi-4 profile because Phi-4
uses tied input and output embeddings. Do not apply it to an untied model.

The LM-head pruning milestone uses matched control and candidate exports below
`experiments`. These are measurement evidence, not a third maintained
historical package; both sides must pin the same builder, source revision,
quantization options, and runtime. Current post-prune exports retain the
candidate behavior.

## Requirements

- Windows 11
- Python 3.11 or newer; the scripts use only the standard library
- Git with complete, non-shallow ORT and ORT GenAI clones
- Visual Studio 2022 C++ build tools
- Visual Studio's bundled CMake 3.x (CMake 4 is not compatible with the oldest
  dependency snapshots)
- Sufficient disk space for detached worktrees, runtime archives, and Phi-4
  model packages
- For comparable numbers: NVIDIA GeForce RTX 5080 and the recorded driver

The scripts never build, switch, reset, clean, or stash the primary source
checkouts. Builds run in detached worktrees below the configured short
worktree root.

## Configure

Create local configs from the templates:

```powershell
New-Item -ItemType Directory -Force ..\gitignore\perf-trend\config
Copy-Item .\config.example.json ..\gitignore\perf-trend\config\config.local.json
Copy-Item .\milestone-runtime-config.example.json ..\gitignore\perf-trend\config\milestone-runtime-config.local.json
```

Set the paths used by the templates for the current PowerShell session:

```powershell
$env:ORT_REPO = "E:\workspace\project\agents\onnxruntime"
$env:ORT_GENAI_REPO = "E:\workspace\project\agents\onnxruntime-genai"
$env:ORT_WEBGPU_PERF_ROOT = "E:\backup\ort\perf-trend"
$env:ORT_WEBGPU_WORKTREE_ROOT = "E:\ort-wt"
$env:ORT_CMAKE = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
```

Environment variables and `~` are expanded in configs. Relative paths are
resolved from the config file's directory. Local configs are ignored by Git.

Expected archive layout:

```text
<ORT_WEBGPU_PERF_ROOT>/
  milestone-results.json
  milestones.json
  logs/milestone-runs/
  model/aligned/<model-variant>/Phi-4-mini-instruct/
  runtime/<YYYY-MM>/
  runtime-milestones/<milestone>/
```

## Fast path: regenerate and verify

With an existing archive:

```powershell
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json render
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify
```

`render` writes `index.html` beside the script unless `--report` specifies
another destination, and mirrors the same report to `<outputRoot>/index.html`
so the archive never retains a stale report. `verify` checks workload settings, GPU identity, runtime
binaries and commits, model files, raw logs, positive metrics, archived-plan
agreement, and report presence.

Before publishing or transferring an archive, verify every archived runtime and
model file against its recorded SHA-256. This reads all model weights and can
take several minutes:

```powershell
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify --verify-hashes
```

You can override the data archive without editing a config:

```powershell
python .\milestone_benchmark.py --output-root E:\backup\ort\perf-trend render
```

## Rerun milestone benchmarks

Rerun one point first:

```powershell
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json run --stage 02-subgroup-prefill
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify
```

Rerun all points only on a controlled machine:

```powershell
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json run --force
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify
```

Every successful point must retain its raw log under
`logs/milestone-runs/`. Failed points remain failed or explicitly missing;
never replace them with zero or an estimate.

Every published timeline row must link every landed PR required for the
measured path, including coordinated ORT and ORT GenAI changes. Each link must
state its owner, merge date, merge commit, and role. PRs are displayed in
reverse chronological order. Unmerged umbrella/discussion PRs are evidence for
finding the landed implementation pieces, but are not listed as required changes.
For each repository with required PRs, build and test the exact merge commit of
the newest required PR; do not substitute a later monthly snapshot. When the
other repository has no milestone PR, prefer a compatible same-date commit. If
that is impossible because its API already requires later code, use the newest
compatible earlier commit and state the reason in the report.
Runtime-only milestones must reuse one of the two standard model packages
rather than create another model variant.

The report contains one merged milestone table with a concise introduction,
explicit prefill/decode performance area, every required PR and its role,
contributors, measured series impact, and the hardware/model/workload
conditions under which each optimization applies. A separate model-builder
option registry records whether each option is mandatory, conditional, or
optional now, the current profile, current defaults, legacy names, and dated
default or naming history with source PRs and commits. It starts with a
copy-friendly required-now command and the full pinned profile.
The report lists the important `genai_config.json` settings first, including
the benchmark-time `enableGraphCapture` and `validationMode` overrides and the
session-level `optimization.disable_specified_optimizers=ConstantFolding`
setting. It then inventories WebGPU-specific provider settings while
distinguishing stored configuration from the temporary graph-capture
measurement configuration.
ORT GenAI benchmark arguments are documented separately; C++ uses `-ml 8192`
while Python uses `-m 8192`. Historical C++ binaries that lack `-ml` are marked
with their effective prompt + generation length rather than being mislabeled as
8K runs.
It prefers archived controlled parent-to-candidate measurements and otherwise
uses the adjacent cumulative-series transition with that limitation stated.

## Collect on AMD and Intel through WinRM

Additional devices use independent series, reusing the archived runtime and model
files for all 11 milestones. Both charts start with a measurement collected on
that device using the December 2, 2024 runtime. NVIDIA reference values and
controlled comparisons are never copied to another GPU. NVIDIA-specific kernel
changes may have no benefit on AMD or Intel; failed measurements remain gaps.

Prepare a transfer manifest (validates every source artifact against its archived
hash and records the exact runtime commits):

```powershell
python .\cross_device_benchmark.py prepare --archive-root E:\backup\ort\perf-trend --manifest ..\gitignore\perf-trend\remote\transfer-manifest.json
```

The manifest covers 12 runs: the 11 milestones and one starting measurement. It
includes both maintained models plus the two model-changing experiment packages
already used by the NVIDIA series. No new model export is required.

Probe each host, then deploy and start one collector per host. The host requires
Python 3.11+, enough disk space for the manifest, and working WebGPU/D3D12 drivers.
Use the reachable fully qualified hostname when short-name DNS is unavailable.
Credentials can be passed with `-Credential` when the current Windows identity
does not have WinRM access.

```powershell
.\remote_benchmark.ps1 -Action Probe -ComputerName webgfx-30
.\remote_benchmark.ps1 -Action Deploy -ComputerName webgfx-30 -ArchiveRoot E:\backup\ort\perf-trend -Manifest ..\gitignore\perf-trend\remote\transfer-manifest.json
.\remote_benchmark.ps1 -Action Start -ComputerName webgfx-30 -Vendor AMD
.\remote_benchmark.ps1 -Action Status -ComputerName webgfx-30
.\remote_benchmark.ps1 -Action Collect -ComputerName webgfx-30 -LocalResults ..\gitignore\perf-trend\remote\webgfx-30
```

Repeat with `webgfx-32` and `-Vendor Intel`. The default remote archive is
`C:\ort-perf-trend`; override it consistently with `-RemoteRoot` if needed.
Deployment skips files whose sizes and hashes already match and verifies all
transferred files. Identical artifacts already present on the remote disk are
copied locally instead of transferred again. For a reachable artifact server,
`-ArtifactBaseUrl` optionally downloads files under WinRM control and verifies
their hashes; interrupted downloads resume with HTTP Range and retry from zero
if the completed file fails verification. The server should expose only the
manifest-listed files to the intended devices. Start creates a disconnected WinRM session that survives the
client invocation. Keep its returned session name/ID. Status reports live process
IDs and saved result rows; inspect both before restarting interrupted work.
The collector resumes completed stages only when the manifest, host, GPU, driver
and power scheme match. It refuses ambiguous physical GPU selection.

Each row retains the command, effective max length, actual session options,
runtime/model identifiers, timestamps, AC power state, and raw output. Unique log filenames
preserve failed attempts and reruns. The workload remains batch 1, prompt 1024,
generation 128, one warmup and five measurements; `-ml 8192` is used where the
historical executable supports it. Graph-capture overrides use a temporary model
config. Historical binaries that lack `-ml` are explicitly marked.
The March 4 GenAI benchmark (`b2a4ecc603`) counts an additional one-token seed
when generating a prompt with an explicit max length. For that exact binary,
the collector uses `-l 1023` and verifies that the raw output reports 1024 prompt
tokens. The invocation records this adjustment. A shorter diagnostic for failed
historical runtimes is retained separately and never substituted in the trend.

After collection, publish the validated dataset and regenerate the shared report:

```powershell
python .\cross_device_benchmark.py publish --archive-root E:\backup\ort\perf-trend --manifest ..\gitignore\perf-trend\remote\transfer-manifest.json --results ..\gitignore\perf-trend\remote\webgfx-30\results.json --output .\data\devices\webgfx-30.json
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json render
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify
```

Repeat publication for `webgfx-32`. Files in `data/devices/` are tracked evidence,
including raw benchmark output and artifact hashes. Large model/binary copies,
transfer manifests, and working logs remain in the local archives or `gitignore/`.
Rendering automatically loads those datasets; `--device-results` can explicitly
select additional datasets. Publication and report verification reject missing
milestones, wrong commits, changed device/driver/power state, and throughput
values that differ from the raw output.

## Rebuild monthly runtimes

Check prerequisites and review the plan before writing it:

```powershell
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json doctor
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json plan --dry-run
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json plan --fetch
```

Prove one historical snapshot before a full run:

```powershell
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json run --snapshot 2024-12
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json run --all-pending
```

Use `--retry-failed` only after reviewing the preserved failure log.

## Rebuild exact milestone runtimes

`..\gitignore\perf-trend\config\milestone-runtime-config.local.json` declares exact ORT/GenAI commit pairs for
runtime-only milestones such as shared-memory prefill and subgroup MatMulNBits.
No manifest hand editing is required:

```powershell
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\milestone-runtime-config.local.json doctor
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\milestone-runtime-config.local.json plan --dry-run
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\milestone-runtime-config.local.json plan
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\milestone-runtime-config.local.json build --all-pending
```

After rebuilding, run the primary milestone benchmark and verification commands
from the previous section.

## Validation

```powershell
python -m unittest discover .\tests -v
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json doctor
python .\ort-webgpu-perf-trend.py --config ..\gitignore\perf-trend\config\config.local.json plan --dry-run
python .\milestone_benchmark.py --config ..\gitignore\perf-trend\config\config.local.json verify
```

See [methodology](references/methodology.md) for snapshot rules, historical
compatibility adjustments, artifact requirements, and interpretation limits.
