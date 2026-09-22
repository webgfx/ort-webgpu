"""Reproducible source inventory for native ORT WebGPU (no inference is run)."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess


CORE = "onnxruntime/core/providers/webgpu/"
CONTRIB = "onnxruntime/contrib_ops/webgpu/"
DOMAINS = {"kOnnxDomain": "ai.onnx", "kMSDomain": "com.microsoft",
           "kMSInternalNHWCDomain": "com.ms.internal.nhwc", "kMLDomain": "ai.onnx.ml",
           "kPytorchAtenDomain": "org.pytorch.aten", "kMSNchwcDomain": "com.microsoft.nchwc",
           "kMSDmlDomain": "com.microsoft.dml", "kMSExperimentalDomain": "com.microsoft.experimental"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, encoding="utf-8").strip()


def strip_comments(text):
    # Preserve strings (including URLs) and line offsets while ignoring disabled registrations.
    pattern = r'R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m[0]) if m[0].startswith(("//", "/*")) else m[0], text)


def ref(path, text, offset=0, repo="ort"):
    return {"repo": repo, "path": path, "line": text.count("\n", 0, offset) + 1}


def plain(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text))).strip()


def schema_docs(root, relative, default_domain, repo):
    text = (root / relative).read_text(encoding="utf-8")
    headings = list(re.finditer(r'^### (?:<sub>experimental</sub> )?<a name="([^"]+)"', text, re.M))
    table = text[:headings[0].start()] if headings else text
    metadata = {}
    for line in table.splitlines():
        match = re.search(r'<a href="#[^"]+">([^<]+)</a>', line)
        if not match:
            continue
        versions = sorted(set(map(int, re.findall(r'Changelog[^#]*#[^"\s]*-(\d+)"', line))))
        cells = line.split("|")
        function = len(cells) > 3 and bool(re.fullmatch(r"[\d, ]+", cells[3]))
        metadata[match[1]] = (versions, function)
    rows = {}
    for index, match in enumerate(headings):
        qualified = match[1]
        domain, name = qualified.rsplit(".", 1) if "." in qualified else (default_domain, qualified)
        body = text[text.find("\n", match.start()) + 1:headings[index + 1].start() if index + 1 < len(headings) else len(text)]
        introduction = re.split(r"\n\s*\n|\n####", body.strip(), maxsplit=1)[0]
        versions, function = metadata.get(qualified, ([], False))
        version = re.search(r"available since version (\d+)", body)
        if version:
            versions = sorted(set(versions + [int(version[1])]))
        rows[(domain, name)] = {"domain": domain, "name": name, "schemaVersions": versions,
                               "functionSchema": function, "introduction": plain(introduction)[:900],
                               "catalog": "ONNX schema" if repo == "onnx" else "ORT contrib schema documentation",
                               "schemaRefs": [ref(relative, text, match.start(), repo)]}
    return rows


def native_registry(root):
    result = {}

    def add(domain, name, low, high, source):
        key = (domain, name)
        record = result.setdefault(key, {"ranges": [], "refs": []})
        interval = [int(low), int(high) if high is not None else None]
        if interval not in record["ranges"]:
            record["ranges"].append(interval)
        if source not in record["refs"]:
            record["refs"].append(source)

    for path in (CORE + "webgpu_execution_provider.cc", CONTRIB + "webgpu_contrib_kernels.cc"):
        original = (root / path).read_text(encoding="utf-8")
        text = strip_comments(original)
        start = text.index("build_kernel_create_info_function_table[]")
        end = text.index("};", start)
        table = text[start:end]
        for match in re.finditer(r'ONNX_OPERATOR_\w*?KERNEL_CLASS_NAME\(([^)]+)\)', table):
            args = [arg.strip() for arg in match[1].split(",")]
            if args[0] != "kWebGpuExecutionProvider":
                continue
            versioned = "VERSIONED" in match[0]
            add(DOMAINS[args[1]], args[-1], args[2], args[3] if versioned else None,
                ref(path, original, start + match.start()))
        for match in re.finditer(r'KERNEL_CREATE_INFO(_VERSIONED|_TYPED)?\(([^)]+)\)', table):
            args = [arg.strip() for arg in match[2].split(",")]
            add("ai.onnx", args[-1], args[0], args[1] if match[1] == "_VERSIONED" else None,
                ref(path, original, start + match.start()))
        for match in re.finditer(r'Create(\w+?)(Versioned)?KernelInfo\((\d+)(?:,\s*(\d+))?', text[end:]):
            add("ai.onnx", match[1], match[3], match[4] if match[2] else None,
                ref(path, original, end + match.start()))
        if "CreateGroupQueryAttentionKernelInfo(enable_graph_capture)" in text:
            offset = text.index("CreateGroupQueryAttentionKernelInfo(enable_graph_capture)")
            add("com.microsoft", "GroupQueryAttention", 1, None, ref(path, original, offset))
    path = CORE + "math/binary_elementwise_ops.cc"
    original = (root / path).read_text(encoding="utf-8")
    for match in re.finditer(r'RegisterBinaryOp\(kernel_registry,\s*"(\w+)"([^;]+);', strip_comments(original)):
        for version in re.finditer(r'\{(\d+),\s*(\d+|std::nullopt)\}', match[2]):
            add("ai.onnx", match[1], version[1], None if version[2] == "std::nullopt" else version[2],
                ref(path, original, match.start()))
    path = CORE + "generator/range.cc"
    original = (root / path).read_text(encoding="utf-8")
    if "RegisterRangeKernels(*kernel_registry, enable_int64)" not in (root / (CORE + "webgpu_execution_provider.cc")).read_text(encoding="utf-8"):
        raise ValueError("Range registration path changed; review extractor")
    match = re.search(r'\.SinceVersion\((\d+)\)', original)
    if not match:
        raise ValueError("Range registration not found")
    add("ai.onnx", "Range", match[1], None, ref(path, original, match.start()))
    return result


def collect(root, chromium):
    onnx = root / "cmake/external/onnx"
    rows = schema_docs(onnx, "docs/Operators.md", "ai.onnx", "onnx")
    rows.update(schema_docs(onnx, "docs/Operators-ml.md", "ai.onnx.ml", "onnx"))
    rows.update(schema_docs(root, "docs/ContribOperators.md", "com.microsoft", "ort"))
    # Include new source schemas even when generated ContribOperators.md lags.
    for path in sorted((root / "onnxruntime/core/graph/contrib_ops").glob("*.cc")):
        original = path.read_text(encoding="utf-8")
        text = strip_comments(original)
        for match in re.finditer(r'ONNX_MS_OPERATOR_SET_SCHEMA\(\s*(\w+),\s*(\d+)', text):
            key = ("com.microsoft", match[1])
            row = rows.setdefault(key, {"domain": key[0], "name": key[1], "schemaVersions": [],
                                       "functionSchema": False, "introduction": "ORT contrib schema; see source definition.", "catalog": "ORT source schema", "schemaRefs": []})
            row["schemaVersions"] = sorted(set(row["schemaVersions"] + [int(match[2])]))
            row["schemaRefs"].append(ref(path.relative_to(root).as_posix(), original, match.start()))
        for match in re.finditer(r'ONNX_CONTRIB_OPERATOR_SCHEMA\((\w+)\)\s*\.SetDomain\((\w+)\)\s*\.SinceVersion\((\d+)\)', text):
            key = (DOMAINS[match[2]], match[1])
            row = rows.setdefault(key, {"domain": key[0], "name": key[1], "schemaVersions": [],
                                       "functionSchema": False, "introduction": "ORT contrib schema; see source definition.", "catalog": "ORT source schema", "schemaRefs": []})
            row["schemaVersions"] = sorted(set(row["schemaVersions"] + [int(match[3])]))
            row["schemaRefs"].append(ref(path.relative_to(root).as_posix(), original, match.start()))
    # Published CPU/CUDA/DML kernel documentation is corroboration, not runtime proof.
    kernel_doc = (root / "docs/OperatorKernels.md").read_text(encoding="utf-8")
    domain, provider = None, None
    for line in kernel_doc.splitlines():
        ep = re.match(r'## Operators implemented by (\w+)', line)
        if ep:
            provider = ep[1]
        dom = re.search(r'\*\*Operator Domain:\*\* \*([^*]+)\*', line)
        if dom:
            domain = dom[1]
        op = re.match(r'\|([\w]+)\|', line)
        if op and domain and provider:
            key = (domain, op[1])
            row = rows.setdefault(key, {"domain": domain, "name": op[1], "schemaVersions": [],
                                       "functionSchema": False, "introduction": "ORT-published provider kernel; see kernel documentation.",
                                       "catalog": "ORT kernel documentation (legacy/internal included)",
                                       "schemaRefs": [{"repo": "ort", "path": "docs/OperatorKernels.md", "line": 1}]})
            row.setdefault("otherProviders", [])
            if provider not in row["otherProviders"]:
                row["otherProviders"].append(provider)
    registry = native_registry(root)
    for key, registration in registry.items():
        row = rows.setdefault(key, {"domain": key[0], "name": key[1], "schemaVersions": [], "functionSchema": False,
                                   "introduction": "ORT native WebGPU registration.", "schemaRefs": []})
        row["registration"] = registration
    # Corroborate CPU fallback candidates from source because the published kernel
    # document can omit kernels. This is still build-conditional, not a runtime check.
    for path in ["onnxruntime/core/providers/cpu/cpu_execution_provider.cc",
                 "onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc"]:
        original = (root / path).read_text(encoding="utf-8")
        for match in re.finditer(r'BuildKernelCreateInfo<\s*(?:class\s+)?ONNX_OPERATOR_\w*?KERNEL_CLASS_NAME\(([^)]+)\)', strip_comments(original)):
            args = [arg.strip() for arg in match[1].split(",")]
            if args[0] != "kCpuExecutionProvider" or args[1] not in DOMAINS or not args[2].isdigit():
                continue
            key = (DOMAINS[args[1]], args[-1])
            if key not in rows:
                rows[key] = {"domain": key[0], "name": key[1], "schemaVersions": [], "functionSchema": False,
                             "introduction": "ORT CPU source registration; see source for exact contract.",
                             "catalog": "ORT CPU source registration", "schemaRefs": []}
            rows[key].setdefault("cpuSourceRefs", []).append(ref(path, original, match.start()))
    webnn_path = "services/webnn/ort/graph_builder_ort.cc"
    webnn_text = (chromium / webnn_path).read_text(encoding="utf-8")
    webnn_ops = {}
    for match in re.finditer(r'constexpr base::cstring_view kOpType\w+\s*=\s*"(\w+)"', webnn_text):
        webnn_ops[match[1]] = ref(webnn_path, webnn_text, match.start(), "chromium")
    for key, row in rows.items():
        row.setdefault("catalog", "Native WebGPU registration")
        row["webnnRef"] = webnn_ops.get(key[1]) if key[0] == "ai.onnx" else None
        if row.get("registration"):
            row["status"] = "registered"
        elif key == ("ai.onnx", "Constant"):
            row["status"] = "load_time"
        elif row["functionSchema"]:
            row["status"] = "function"
        else:
            row["status"] = "missing"
    corpus_paths = [CORE + "webgpu_execution_provider.cc", CONTRIB + "webgpu_contrib_kernels.cc",
                    CORE + "math/binary_elementwise_ops.cc", "docs/ContribOperators.md", "docs/OperatorKernels.md", "cmake/deps.txt"]
    return {"generatedAt": datetime.now(timezone.utc).isoformat(),
            "method": "Static source inventory; no ORT or Chrome inference executed.",
            "ort": {"root": str(root), "commit": git(root, "rev-parse", "HEAD"),
                    "date": git(root, "show", "-s", "--format=%cI"), "trackedChanges": git(root, "status", "--short", "--untracked-files=no")},
            "onnx": {"root": str(onnx), "commit": git(onnx, "rev-parse", "HEAD"),
                     "version": (onnx / "VERSION_NUMBER").read_text().strip(), "trackedChanges": git(onnx, "status", "--short", "--untracked-files=no")},
            "chromium": {"root": str(chromium), "commit": git(chromium, "rev-parse", "HEAD"),
                         "trackedChanges": git(chromium, "status", "--short", "--untracked-files=no")},
            "sourceHashes": {"ort:" + path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in corpus_paths},
            "rows": sorted(rows.values(), key=lambda row: (row["domain"], row["name"]))}


def enrich(data):
    from assessment import assessment, LIMITATIONS
    root = Path(data["ort"]["root"])
    limitations = []
    for item in LIMITATIONS:
        item = dict(item)
        content = (root / item["path"]).read_text(encoding="utf-8")
        if item["needle"] not in content:
            raise ValueError(f"Limitation source changed: {item['id']}")
        item["source"] = ref(item["path"], content, content.index(item["needle"]))
        limitations.append(item)
    for row in data["rows"]:
        identifier = (row["name"] if row["domain"] == "ai.onnx" else row["domain"] + "::" + row["name"])
        row["limitations"] = [item["id"] for item in limitations if identifier in item["ops"]]
        if row["status"] in {"missing", "function"}:
            row["assessment"] = assessment(row)
            if row["webnnRef"]:
                row["assessment"]["priority"] = "P1"
                row["assessment"]["importance"] += " Direct Chromium WebNN lowering is also present in the pinned source."
        elif row["status"] == "load_time":
            row["assessment"] = {"priority": "P3", "workload": "Model loading", "group": "host",
                                 "importance": "Constant values are moved into initializers by ORT; a dedicated shader is not required.",
                                 "mitigation": "Verify loading/constant folding on the deployed build. Do not count Constant as an ordinary missing GPU kernel."}
        else:
            applicable = [item for item in limitations if item["id"] in row["limitations"]]
            row["assessment"] = {"priority": min((item["priority"] for item in applicable), default="—"),
                                 "workload": "Registered kernel (not full conformance proof)", "group": "registered",
                                 "importance": "Kernel registration found. Actual support still depends on opset, dtype, attributes, shape and device.",
                                 "mitigation": "Use the exact registration and relevant limitations below, then validate real placement and outputs."}
            row["uncoveredSchemaVersions"] = [version for version in row["schemaVersions"]
                if not any(low <= version and (high is None or version <= high) for low, high in row["registration"]["ranges"])]
    data["limitations"] = limitations
    # Hash every referenced source, not only the two registration tables.
    for item in data["rows"]:
        for source in item["schemaRefs"] + item.get("cpuSourceRefs", []) + item.get("registration", {}).get("refs", []) + ([item["webnnRef"]] if item["webnnRef"] else []):
            source_path = Path(data[source["repo"]]["root"]) / source["path"]
            key = source["repo"] + ":" + source["path"]
            if key not in data["sourceHashes"]:
                data["sourceHashes"][key] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for item in limitations:
        data["sourceHashes"]["ort:" + item["path"]] = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
    # Reject a mixed or locally modified source snapshot. Unrelated worktree edits
    # are recorded, but must not silently invalidate commit-pinned citations.
    for key in data["sourceHashes"]:
        repo, path = key.split(":", 1)
        source_root = Path(data[repo]["root"])
        working = (source_root / path).read_text(encoding="utf-8")
        committed = subprocess.check_output(
            ["git", "-C", str(source_root), "show", data[repo]["commit"] + ":" + path],
            text=True, encoding="utf-8")
        if working != committed:
            raise ValueError(f"Audited source differs from pinned commit: {key}")
    data["referencedSourcesMatchCommits"] = True
    data["counts"] = dict(Counter(row["status"] for row in data["rows"]))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort", type=Path, default=Path(r"E:\workspace\project\agents\onnxruntime"))
    parser.add_argument("--chromium", type=Path, default=Path(r"D:\r\cr\src"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("inventory.json"))
    args = parser.parse_args()
    roots = {"ort": args.ort, "onnx": args.ort / "cmake/external/onnx", "chromium": args.chromium}
    before = {repo: git(root, "rev-parse", "HEAD") for repo, root in roots.items()}
    data = enrich(collect(args.ort, args.chromium))
    if any(data[repo]["commit"] != before[repo] or git(root, "rev-parse", "HEAD") != before[repo]
           for repo, root in roots.items()):
        raise ValueError("A source checkout moved during extraction; rerun for a consistent snapshot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(Counter((row["domain"], row["status"]).__str__() for row in data["rows"]), indent=2))
    print("Missing/function operators:")
    for domain in sorted({row["domain"] for row in data["rows"]}):
        print(domain + ": " + ", ".join(row["name"] + (" [function]" if row["status"] == "function" else "")
                                        for row in data["rows"] if row["domain"] == domain and row["status"] in {"missing", "function"}))


if __name__ == "__main__":
    main()
