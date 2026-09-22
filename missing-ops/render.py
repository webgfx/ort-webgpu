"""Render the pinned operator inventory as an offline, self-contained report."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from urllib.parse import quote


LABELS = {"missing": "No native registration", "function": "No fused kernel · function schema",
          "registered": "Registered · conditional coverage", "load_time": "Handled at model load"}


def esc(value):
    return html.escape(str(value), quote=True)


def source_link(data, source, label=None):
    base = {"ort": "https://github.com/microsoft/onnxruntime/blob/",
            "onnx": "https://github.com/onnx/onnx/blob/",
            "chromium": "https://chromium.googlesource.com/chromium/src/+/"}[source["repo"]]
    suffix = "#" if source["repo"] == "chromium" else "#L"
    url = base + data[source["repo"]]["commit"] + "/" + quote(source["path"], safe="/") + suffix + str(source["line"])
    return f'<a href="{esc(url)}">{esc(label or source["path"] + ":" + str(source["line"]))}</a>'


def citation(data, repo, path, line, label):
    return source_link(data, {"repo": repo, "path": path, "line": line}, label)


def ranges(intervals):
    return ", ".join(str(low) + ("+" if high is None else "" if high == low else "–" + str(high))
                     for low, high in sorted(intervals, key=lambda pair: (pair[0], pair[1] or 9999)))


def badge(label, css=""):
    return f'<span class="badge {esc(css)}">{esc(label)}</span>'


STYLE = """
:root{color-scheme:light;--ink:#192c40;--muted:#587086;--line:#dce5ed;--accent:#17687d;--paper:#fff;--bg:#f2f5f8}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:14px/1.6 'Segoe UI',system-ui,sans-serif}
.page{width:calc(100% - clamp(24px,5vw,96px));margin:auto}header{padding:38px 0 24px}h1{font-size:clamp(29px,3.6vw,49px);line-height:1.15;margin:12px 0}
h2{font-size:25px;line-height:1.3;margin:0 0 18px}h3{font-size:16px;margin:0 0 8px}p{margin:9px 0 14px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:700;color:var(--accent)}
a{color:#16677f;text-underline-offset:3px}a:hover{color:#103d4b}.muted,small{color:var(--muted)}small{display:block;font-size:12px;margin:5px 0}
code{font:12px/1.55 Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:16px;border-radius:8px}
section{background:white;border:1px solid var(--line);border-radius:14px;padding:clamp(18px,2vw,30px);margin:0 0 24px;scroll-margin-top:18px}
.lead{font-size:17px;max-width:100ch}.notice{border-left:4px solid #d9873b;padding:12px 18px;background:#fff6eb}.critical{border-left-color:#b33c3c;background:#fff2f2}
.cards{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:15px;margin:24px 0}.card{border:1px solid var(--line);padding:20px;border-radius:10px;background:white}.card strong{display:block;font-size:36px;line-height:1.2}.card span{color:var(--muted);font-size:13px}
.badge{display:inline-block;font-size:11px;font-weight:700;line-height:1.5;padding:4px 8px;border-radius:5px;background:#edf2f7;color:#3c566d}.P0,.missing{color:#923230;background:#fce9e8}.P1{color:#815011;background:#fff0d6}.P2,.function{color:#205e8c;background:#eaf3fb}.P3{color:#536679;background:#edf1f4}.registered{color:#176c58;background:#e9f6f0}
details{padding:11px 14px;border:1px solid var(--line);border-radius:8px;margin:12px 0}summary{cursor:pointer;font-weight:600;color:var(--accent)}nav{display:flex;flex-wrap:wrap;gap:8px 24px;padding-top:12px}
table{width:100%;border-collapse:collapse;font-size:13px;text-align:left}th,td{padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#edf3f7;color:#496075;font-size:11px;letter-spacing:.04em;text-transform:uppercase}tbody th{font-weight:400}tbody tr:last-child>*{border-bottom:0}.scroll{overflow-x:auto}
.inventory{table-layout:fixed;min-width:1100px}.inventory th:nth-child(1){width:20%}.inventory th:nth-child(2){width:19%}.inventory th:nth-child(3){width:24%}.inventory td p{margin:5px 0 10px}.inventory h3{font-size:15px}.refs{font-size:11px;overflow-wrap:anywhere}.compact{min-width:750px}.limits{table-layout:fixed;min-width:1000px}.limits th:first-child{width:18%}
.toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:12px;padding:16px;background:#eef4f7;border-radius:10px;margin:18px 0}.toolbar label{display:flex;flex-direction:column;font-size:12px;gap:5px}.search{flex:1;min-width:240px}input,select,button{font:inherit;border:1px solid #b4c5d1;border-radius:6px;padding:9px;background:white;color:var(--ink)}button{cursor:pointer}.two{display:grid;grid-template-columns:1fr 1fr;gap:26px}li{margin:6px 0}.metadata{display:grid;grid-template-columns:160px minmax(0,1fr);gap:6px 16px}.metadata dt{color:var(--muted)}.metadata dd{margin:0;overflow-wrap:anywhere}
:focus-visible{outline:3px solid #409caf;outline-offset:3px}[hidden]{display:none!important}.skip{position:absolute;top:-100px;background:white;padding:10px}.skip:focus{top:10px}footer{padding:0 0 32px;color:var(--muted);font-size:12px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(120px,1fr))}.two{grid-template-columns:1fr}.metadata{grid-template-columns:1fr}.metadata dd{margin-bottom:10px}}
@media print{.toolbar,header details,.skip{display:none}.page{width:100%}body{background:white}section{border:0;padding:12px 0}.scroll{overflow:visible}table.inventory,table.limits,table.compact{min-width:0;font-size:8pt}a{color:inherit}}
"""

SCRIPT = """
(()=>{const rows=[...document.querySelectorAll('#inventory tbody tr')];
const search=document.getElementById('search'),status=document.getElementById('status'),domain=document.getElementById('domain'),priority=document.getElementById('priority'),webnn=document.getElementById('webnn-only');
function filter(){let count=0;for(const row of rows){const state=status.value;row.hidden=!(row.dataset.search.includes(search.value.toLowerCase().trim())&&(!domain.value||domain.value===row.dataset.domain)&&(!priority.value||priority.value===row.dataset.priority)&&(!webnn.checked||row.dataset.webnn==='1')&&(!state||state===row.dataset.status||(state==='gaps'&&row.dataset.gap==='1')||(state==='limited'&&row.dataset.limited==='1')));if(!row.hidden)count++;}document.getElementById('count').textContent=`${count} of ${rows.length} operator entries shown`;}
for(const control of [search,status,domain,priority,webnn])control.addEventListener('input',filter);
document.getElementById('clear').addEventListener('click',()=>{search.value='';status.value='';domain.value='';priority.value='';webnn.checked=false;filter();});
document.getElementById('filters').hidden=false;filter();})();
"""


def render(data):
    rows = data["rows"]
    if len({(row["domain"], row["name"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate operator key")
    counts = Counter(row["status"] for row in rows)
    domain_rows = []
    for domain in sorted({row["domain"] for row in rows}):
        subset = [row for row in rows if row["domain"] == domain]
        tally = Counter(row["status"] for row in subset)
        domain_rows.append(f'<tr><th><code>{esc(domain)}</code></th><td>{len(subset)}</td><td>{tally["missing"]}</td>'
                           f'<td>{tally["function"]}</td><td>{tally["registered"]}</td><td>{tally["load_time"]}</td></tr>')
    cards = "".join(f'<div class="card"><strong>{number}</strong><span>{esc(label)}</span></div>' for number, label in [
        (counts["missing"], "Entries without a native registration"), (counts["function"], "Function schemas without a fused kernel"),
        (len(data["limitations"]), "Reviewed partial-support / correctness findings"), (len(rows), "Domain-qualified entries inventoried")])
    limits = []
    for item in sorted(data["limitations"], key=lambda item: item["priority"]):
        limits.append(f'<tr id="limit-{esc(item["id"])}"><th>{badge(item["priority"],item["priority"])}<h3>{esc(item["title"])}</h3>'
                      f'<small>{esc(", ".join(item["ops"]))}</small></th><td>{esc(item["finding"])}</td>'
                      f'<td>{esc(item["impact"])}</td><td>{esc(item["test"])}<small class="refs">{source_link(data,item["source"])}</small></td></tr>')
    all_rows = []
    for row in sorted(rows, key=lambda row: (row["assessment"]["priority"], row["domain"], row["name"])):
        assessment = row["assessment"]
        priority = assessment["priority"]
        identifier = row["domain"] + "::" + row["name"]
        is_gap = row["status"] in {"missing", "function"} or bool(row["limitations"])
        refs = row["schemaRefs"][:2] + row.get("registration", {}).get("refs", [])[:1]
        source_html = "<br>".join(source_link(data, source) for source in refs)
        versions = ", ".join(map(str, row["schemaVersions"])) or "See source"
        registry_note = ("Native range: " + ranges(row["registration"]["ranges"])) if "registration" in row else "No native registration found"
        if row["status"] == "load_time":
            registry_note = "No GPU shader required for Constant loading"
        unsupported_versions = row.get("uncoveredSchemaVersions", [])
        if unsupported_versions:
            registry_note += "; schema revision points outside ranges: " + ", ".join(map(str, unsupported_versions))
        webnn = '<small>' + source_link(data, row["webnnRef"], "Chromium emits this operator") + '</small>' if row["webnnRef"] else ""
        limit_links = " ".join(f'<a href="#limit-{esc(key)}">{esc(key)}</a>' for key in row["limitations"])
        limit_html = f'<p><strong>Reviewed limits:</strong> {limit_links}</p>' if limit_links else ""
        cpu = row.get("cpuSourceRefs", [])
        other = (source_link(data, cpu[0], "CPU source registration exists") if cpu else "CPU source registration not captured")
        documented = ", ".join(row.get("otherProviders", [])) or "None captured in published kernel table"
        detail = (f'<details><summary>Schema / registration / fallback evidence</summary><p>{esc(row["catalog"])}. '
                  f'Schema versions: {esc(versions)}.</p><p>{esc(registry_note)}.</p><p class="refs">{source_html}</p>'
                  f'<p>{other}. Registration presence does not validate dtype/opset/build-specific fallback.</p>'
                  f'<small>Other-provider documentation: {esc(documented)}</small></details>')
        search_text = " ".join([identifier, assessment["workload"], row["introduction"], assessment["importance"], assessment["mitigation"]]).lower()
        all_rows.append(f'<tr data-key="{esc(identifier)}" data-status="{row["status"]}" data-domain="{esc(row["domain"])}" '
                        f'data-priority="{esc(priority)}" data-webnn="{int(bool(row["webnnRef"]))}" data-limited="{int(bool(row["limitations"]))}" '
                        f'data-gap="{int(is_gap)}" data-search="{esc(search_text)}"><th><h3>{esc(row["name"])}</h3>'
                        f'<code>{esc(row["domain"])}</code><p>{esc(row["introduction"])}</p>{webnn}{detail}</th>'
                        f'<td>{badge(LABELS[row["status"]],row["status"])}<p>{esc(registry_note)}</p>{limit_html}</td>'
                        f'<td>{badge(priority,priority)} <strong>{esc(assessment["workload"])}</strong><p>{esc(assessment["importance"])}</p></td>'
                        f'<td>{esc(assessment["mitigation"])}</td></tr>')
    webnn_rows = []
    for row in rows:
        if row["webnnRef"] and row["status"] in {"missing", "function"}:
            operation = {"LpPool": "l2Pool2d (p=2)", "Softsign": "softsign", "Round": "roundEven", "IsInf": "isInfinite", "IsNaN": "isNaN", "Or": "logicalOr", "Xor": "logicalXor", "Sign": "sign", "QuantizeLinear": "quantizeLinear"}.get(row["name"], row["name"])
            webnn_rows.append(f'<tr><th>{esc(operation)}</th><td><code>{esc(row["name"])}</code></td>'
                              f'<td>{badge(LABELS[row["status"]],row["status"])}</td><td>{source_link(data,row["webnnRef"],"Lowering source")}</td></tr>')
    sources = "".join(f'<tr><th><code>{esc(path)}</code></th><td><code>{esc(digest)}</code></td></tr>' for path,digest in sorted(data["sourceHashes"].items()))
    metadata = "".join(f'<dt>{esc(repo.upper())}</dt><dd><code>{esc(data[repo]["commit"])}</code> · <code>{esc(data[repo]["root"])}</code>'
                       f'<small>Tracked source changes: {esc(data[repo]["trackedChanges"] or "none")}</small></dd>' for repo in ("ort","onnx","chromium"))
    priorities = [
        ("P0", "Correctness before coverage", "Int64 arithmetic/comparison semantics", "Prevent silent wrong outputs: implement full-width behavior or restrict/reject the unsupported range. Also test overflow of intermediates."),
        ("P1", "Chrome WebNN operator coverage", "Or, Xor, IsNaN, IsInf, Round, Sign, QuantizeLinear, LpPool; Softsign lowering", "Close eight absent primitive registrations and validate Softsign expansion at the actual Chromium opset. Include Cast/narrow dtypes and fp16 scatter reductions."),
        ("P1", "Portable quantized / modern transformer graphs", "QLinear/MatMulInteger/ConvInteger; ai.onnx.Attention; TensorScatter; floating-point MoE", "Pick target export formats; verify decompositions before adding duplicate fusions. Keep standard and com.microsoft domains distinct."),
        ("P1", "Detection / dynamic tensor graphs", "NonMaxSuppression, NonZero, Unique, Compress; GridSample opset 20+", "Prioritize frequent device-resident use. A CPU tail may be acceptable if outputs are small and transfers are measured."),
        ("P2", "Architecture-dependent coverage", "RNN, control flow/containers, ROI/deformable/unpool, STFT, random/sparse/packed ops", "Promote to P1 when a required model actually needs them. Host orchestration and static rewrites may be the better boundary."),
        ("P3", "Do not chase name-count completeness", "Training, strings/tokenization, ai.onnx.ml, legacy ops, DML/NCHWc/TRT dialects, collectives", "Keep explicit host or matching-backend boundaries. These can matter to specific products, but not every neural-network browser deployment needs GPU kernels for them."),
    ]
    priority_rows = "".join(f'<tr><th>{badge(p,p)}<h3>{esc(t)}</h3></th><td>{esc(o)}</td><td>{esc(a)}</td></tr>' for p,t,o,a in priorities)
    domains = "".join(f'<option value="{esc(domain)}">{esc(domain)}</option>' for domain in sorted({r["domain"] for r in rows}))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>ORT WebGPU — missing operators and importance</title><style>{STYLE}</style></head><body>'
            '<a class="skip" href="#catalog">Skip to operator inventory</a><div class="page"><header><div class="eyebrow">Source audit · inference coverage · Chrome WebNN</div>'
            '<h1>Missing operators.<br>Prioritized by what they block.</h1><p class="lead">Native ORT WebGPU coverage, functional gaps and implementation priorities—'
            'with domain-qualified names, source evidence and practical alternatives.</p>'
            f'<p class="muted">ORT <code>{esc(data["ort"]["commit"])}</code> · ONNX {esc(data["onnx"]["version"])} · '
            f'Review generated {esc(data["generatedAt"][:19].replace("T"," "))}</p>'
            '<details><summary>Menu</summary><nav><a href="#summary">Executive summary</a><a href="#scope">Scope and counting rules</a>'
            '<a href="#priorities">Implementation priorities</a><a href="#webnn">Chrome WebNN gaps</a><a href="#limits">Partial support and correctness</a>'
            '<a href="#catalog">Complete operator inventory</a><a href="#verify">Validation plan</a><a href="#sources">Source provenance</a></nav></details></header>'
            '<section id="summary"><h2>What matters most</h2><p class="notice critical"><strong>Registration is not full support.</strong> '
            'Some int64 binary kernels explicitly document incorrect results outside the int32 range. Treat this as a correctness requirement, not merely another missing operator.</p>'
            + cards + '<div class="two"><div><h3>First close semantic and browser gaps</h3><p>Eight operators emitted by the pinned Chromium ORT builder have no native registration; '
            'Softsign has a function schema but no dedicated kernel. Other important gaps include quantized operator families, dynamic selection/detection, '
            'standard ONNX Attention, and GridSample’s newer opsets.</p></div><div><h3>Avoid false missing-operator claims</h3><p>Native WebGPU already registers '
            'GRU, LSTM, RMSNormalization, standard RotaryEmbedding, DFT, TopK, ScatterND/ScatterElements, QMoE and many LLM fusions. '
            'Conv3D exists for group=1. Constant is handled at load time. Function expansion and supported primitive rewrites can avoid a new fused kernel.</p></div></div>'
            '<p><strong>Evidence level:</strong> static source inspection only. No ORT rebuild, operator execution, Chrome conformance run, model-frequency survey or performance measurement was performed for this report. '
            'Priorities are workload-based engineering judgments, not measured speedups or prevalence.</p></section>'
            '<section id="scope"><h2>Scope and counting rules</h2><p>The inventory is the union of ONNX 1.22 schema documentation (including ML and preview), '
            'ORT contrib source/schema documentation, published CPU/CUDA/DML operator lists, CPU source registration tables and native WebGPU registrations. '
            'Keys are <code>domain::operator</code>; opset variants are not counted as separate operators. This includes legacy and backend-internal entries so they are visible, not mistaken for equally valuable GPU work.</p>'
            '<p>“No native registration” means absent from this native registry—not that ORT has no implementation anywhere. '
            '“Function schema” means a lowering may exist, not that it was verified. A CPU fallback requires an applicable compiled CPU kernel; '
            'unsupported shapes may fail inside an already-selected GPU kernel instead of falling back. No-fallback deployments can fail at initialization.</p>'
            '<p>Assumes an unreduced inference build with contrib operators enabled. Datatype options, adapter capabilities and build flags can reduce support. '
            'Arbitrary third-party/custom ops and the separate ORT training gradient/optimizer catalog are outside the exhaustive name inventory. '
            'The 22 detailed limitation findings are a reviewed set, not an exhaustive Cartesian proof for every datatype/shape/attribute combination.</p>'
            '<div class="scroll"><table class="compact"><thead><tr><th>Domain</th><th>Entries</th><th>No registration</th><th>Function only</th><th>Registered</th><th>Load-time</th></tr></thead><tbody>'
            + "".join(domain_rows) + '</tbody></table></div><p>Do not interpret these counts as a model-success percentage. '
            'Internal layout aliases, classic ML and training entries can dominate a name count without affecting an LLM.</p>'
            '<p>The old web operator document is generated from <code>js_execution_provider.cc</code>, not this native EP: '
            + citation(data,"ort","js/web/script/generate-webgpu-operator-md.ts",47,"generator source") + '. '
            'Constant handling: ' + citation(data,"ort","onnxruntime/core/graph/graph.cc",1278,"initializer conversion") + '. Function expansion: '
            + citation(data,"ort","onnxruntime/core/framework/graph_partitioner.cc",1310,"partition/inlining source") + '.</p></section>'
            '<section id="priorities"><h2>Implementation priorities</h2><p>P0 = correctness risk; P1 = common inference/browser coverage; '
            'P2 = target-model dependent; P3 = usually host-side, legacy or outside the inference focus. A P3 operator can still be mandatory for a specific product.</p>'
            '<div class="scroll"><table class="compact"><thead><tr><th>Priority / objective</th><th>Operators / families</th><th>Recommended action</th></tr></thead><tbody>'
            + priority_rows + '</tbody></table></div></section>'
            '<section id="webnn"><h2>Chrome WebNN: nine direct lowering gaps</h2><p>Eight missing registrations and one function-only operator are referenced by Chromium’s '
            'ORT graph builder at the pinned revision. The builder currently imports ONNX opset 21 ('
            + citation(data,"chromium","services/webnn/ort/model_editor.cc",44,"model-editor source") + '). '
            'The table establishes a source-level coverage concern, not nine demonstrated Chrome failures. Constant folding, function inlining, build configuration and CPU fallback can change actual execution.</p>'
            '<div class="scroll"><table class="compact"><thead><tr><th>WebNN operation</th><th>ONNX target</th><th>Native status</th><th>Evidence</th></tr></thead><tbody>'
            + "".join(webnn_rows) + '</tbody></table></div><p>Also validate int64 arithmetic/comparisons, Cast pairs, fp16 scatter reductions, recurrent activation choices and '
            'operator-specific integer coverage. Enabling <code>enableInt64=1</code> is necessary for some registrations but not sufficient for correct full-width semantics. '
            'Full Chrome support additionally requires ordering, lifetime, interop, security and recovery tests; see the separate <a href="../tests/index.html#webnn">concurrency/WebNN report</a>.</p></section>'
            '<section id="limits"><h2>Registered operators with important limitations</h2><p>These are positive source findings. '
            'They are separate from missing-name counts. Source checks should be turned into focused positive/negative tests before any support claim.</p>'
            '<div class="scroll"><table class="limits"><thead><tr><th>Priority / affected operators</th><th>Exact gap / condition</th><th>Importance</th><th>Recommended verification / source</th></tr></thead><tbody>'
            + "".join(limits) + '</tbody></table></div></section>'
            '<section id="catalog"><h2>Complete domain-qualified operator inventory</h2><p>Every absent registration has an importance assessment and mitigation. '
            'Registered entries remain available to check false positives, opset ranges and reviewed limitations. All rows are in this file; JavaScript only filters them.</p>'
            '<div id="filters" class="toolbar" hidden><label class="search">Search operator / workload<input id="search" type="search" placeholder="Quantize, WebNN, attention, detection…"></label>'
            '<label>Coverage<select id="status"><option value="gaps">Missing / function / reviewed limits</option><option value="">All entries</option>'
            '<option value="missing">No native registration</option><option value="function">Function only</option><option value="limited">Reviewed limitations</option><option value="registered">Registered</option><option value="load_time">Load-time</option></select></label>'
            '<label>Domain<select id="domain"><option value="">All domains</option>' + domains + '</select></label>'
            '<label>Priority<select id="priority"><option value="">All priorities</option><option>P0</option><option>P1</option><option>P2</option><option>P3</option></select></label>'
            '<label><span>Chromium lowering</span><input id="webnn-only" type="checkbox" aria-label="Only operators emitted by Chromium"></label><button id="clear">Show all / clear</button></div>'
            f'<p id="count" role="status" aria-live="polite">{len(rows)} of {len(rows)} operator entries shown</p><div class="scroll"><table id="inventory" class="inventory"><thead><tr>'
            '<th>Operator / purpose / evidence</th><th>Native coverage / versions</th><th>Importance / affected workload</th><th>Alternative / next action</th></tr></thead><tbody>'
            + "".join(all_rows) + '</tbody></table></div></section>'
            '<section id="verify"><h2>Validation plan before implementation or support claims</h2><ol>'
            '<li><strong>Identify actual demand.</strong> Collect domain, opset, dtype, attributes, dynamic shapes and frequency from target models before and after optimization. Rank barriers on the critical path and bytes crossing CPU/GPU boundaries.</li>'
            '<li><strong>Build two attribution profiles.</strong> Use the exact native built-in or plugin DLL intended for deployment; pin hashes. Run with CPU fallback disabled to expose gaps, and separately with the intended shipping fallback policy. Check actual node placement and optimized function bodies.</li>'
            '<li><strong>Test semantics, not just session creation.</strong> Compare outputs with a reference across dtypes, overflow, NaN/Inf, zero-sized/odd dimensions, broadcasting, optional outputs, opset boundaries and unsupported combinations. Test the registered cases adjacent to every gap.</li>'
            '<li><strong>Validate the Chrome path.</strong> Force and verify ORT WebGPU, run the WebNN WPT operator/type/validation matrix at Chromium’s actual opset, and keep robust buffer access enabled. A generic GPU test result can come from another backend.</li>'
            '<li><strong>Measure end-to-end impact.</strong> Record fallback copies/synchronization, peak memory, startup/compile time, prefill/decode or per-frame throughput, on NVIDIA/AMD/Intel as applicable. Decompositions may be correct but slower or more memory-hungry.</li>'
            '<li><strong>Promote support only with evidence.</strong> A new registry entry or a successful small model is not full operator conformance. Keep untested combinations explicit and reject unsafe semantics rather than silently computing incorrect data.</li></ol>'
            '<p>This report does not modify ORT, Chromium, their default options or existing test results.</p></section>'
            '<section id="sources"><h2>Source provenance and reproducibility</h2><dl class="metadata">' + metadata + '</dl>'
            '<p><a href="inventory.json">Machine-readable inventory</a> · <a href="README.md">Method and regeneration</a> · '
            '<a href="analyze.py">Extractor</a> · <a href="assessment.py">Importance / limitation rules</a> · <a href="render.py">Renderer</a></p>'
            '<p>All source links use exact commits. Referenced inventory and limitation files were verified against those commits (normalizing line endings). '
            'Unrelated local changes shown above were left untouched. The local ONNX checkout reports version 1.22.0, matching the ORT dependency tag. '
            'No installed Python ONNX package was used to enumerate schemas. Native registration extraction ignores comments (including the commented MoE entry), '
            'handles versioned/typed table macros and dynamic int64-aware registration helpers. Hashes cover referenced inventory and limitation sources. Generated timestamps use one time base (UTC).</p>'
            '<details><summary>Source SHA-256 manifest</summary><div class="scroll"><table><thead><tr><th>Source</th><th>SHA-256</th></tr></thead><tbody>'
            + sources + '</tbody></table></div></details></section><footer>Static source audit. Kernel absence, optional lowering, runtime conformance and measured performance are different claims.</footer></div>'
            f'<script>{SCRIPT}</script></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path(__file__).with_name("inventory.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("index.html"))
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    args.output.write_text(render(data), encoding="utf-8")
    print(f"Rendered {len(data['rows'])} operator entries and {len(data['limitations'])} limitations to {args.output}")


if __name__ == "__main__":
    main()
