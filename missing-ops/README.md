# Missing ORT WebGPU operators

Open [index.html](index.html) for the detailed, self-contained report. It includes
all domain-qualified entries in the declared corpus, importance assessments,
mitigations, Chrome WebNN lowering gaps, known partial-support restrictions and
exact source links. Search and domain/status/priority filters work offline.

This is **source analysis, not an operator test run**. No ORT/Chromium build or
GPU inference was performed. A missing native registration does not prove an
entire model fails: function expansion, optimization, host orchestration and
CPU fallback are different paths. Conversely, registration is not conformance.

## Snapshot

- ORT: `30b145bf8c72464a76e5b0c9b199543bb26e1208`, local source at
  `E:\workspace\project\agents\onnxruntime`.
- ONNX: `2bb50465112feca9003e1ed654d77f01ff1415ca`, version 1.22.0 from
  ORT's `cmake/external/onnx` checkout, matching the dependency tag.
- Chromium: `94bce48ea49c758081ec5c23d967f047df6194d9`, at `D:\r\cr\src`.
- 413 domain-qualified entries: 238 without a native registration, 24 function
  schemas without a native fused kernel, 150 registered, and one load-time
  Constant entry. Counts include legacy, ML, preview and internal contracts;
  they are not model-success rates.
- Referenced source files were checked against their exact commits. Unrelated
  local ORT test/build edits are recorded in the provenance and were not changed.
- 22 manually reviewed limitation/correctness findings. These are not an
  exhaustive datatype × shape × attribute conformance matrix.

The name corpus is the union of ONNX schema docs, ORT contrib schema docs/source,
published CPU/CUDA/DML kernel docs, CPU source registrations, and native WebGPU
registrations. Arbitrary custom/third-party ops and the separate ORT training
gradient/optimizer catalog are outside scope. Standard ONNX preview-training
schemas and inference-tree distributed schemas remain visible at low priority.

## Reproduce

Python standard library only; no installed ONNX or ORT package is required:

```powershell
python -B .\missing-ops\analyze.py `
  --ort E:\workspace\project\agents\onnxruntime `
  --chromium D:\r\cr\src
python -B .\missing-ops\render.py
python -B -m unittest discover -s .\missing-ops -p 'test_*.py' -v
```

Use checkouts at the exact revisions for a reproduction. On a new revision,
review extraction patterns and priorities before publishing. Unknown missing
operators and changed limitation anchors fail extraction instead of receiving
an invented assessment. Kernel table macros, typed/versioned registrations,
dynamic registration helpers and commented-out entries are handled explicitly;
this is not a C++ preprocessor or a compiled registry dump.

`inventory.json` is the tracked audit evidence. `assessment.py` contains human
judgments and source-backed conditions. `render.py` creates `index.html` without
modifying evidence. Intermediate drafts belong in `../gitignore/missing-ops/`.

Existing results under `tests/` and `perf-trend/` are not changed by these scripts.
