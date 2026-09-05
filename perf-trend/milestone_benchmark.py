#!/usr/bin/env python3
"""Run and render the RTX 5080 ORT/GenAI milestone benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = SKILL_ROOT / "milestones.json"
LOCAL_CONFIG = (
    SKILL_ROOT.parent / "gitignore" / "perf-trend" / "config" / "config.local.json"
)
DEFAULT_CONFIG = (
    LOCAL_CONFIG
    if LOCAL_CONFIG.is_file()
    else SKILL_ROOT / "config.local.json"
    if (SKILL_ROOT / "config.local.json").is_file()
    else SKILL_ROOT / "config.example.json"
)
DEFAULT_REPORT = SKILL_ROOT / "index.html"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def force_all_stages(force: bool, stages: list[str] | None) -> bool:
    """Keep --force scoped to --stage when both options are supplied."""
    return force and not stages


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_output_root(explicit: Path | None, config_path: Path) -> Path:
    if explicit is not None:
        return resolve_path(explicit)
    if environment_root := os.environ.get("ORT_WEBGPU_PERF_ROOT"):
        return resolve_path(environment_root)
    if config_path.is_file():
        config = read_json(config_path)
        if output_root := config.get("outputRoot"):
            return resolve_path(output_root, config_path.resolve().parent)
    raise ValueError(
        "output root is required: pass --output-root, set ORT_WEBGPU_PERF_ROOT, "
        "or provide outputRoot in the selected config file"
    )


def resolve_plan_path(output_root: Path, value: str) -> Path:
    return resolve_path(value, output_root)


def parse_output(text: str) -> dict[str, float]:
    patterns = {
        "ttftMs": (r"Prompt processing.*?avg \(us\):\s+([\d.e+\-]+)", 0.001),
        "prefillTps": (r"Prompt processing.*?avg \(tokens/s\):\s+([\d.e+\-]+)", 1.0),
        "decodeTps": (r"Token generation:.*?avg \(tokens/s\):\s+([\d.e+\-]+)", 1.0),
        "e2eMs": (r"E2E generation.*?avg \(ms\):\s+([\d.e+\-]+)", 1.0),
    }
    metrics = {}
    for name, (pattern, factor) in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            metrics[name] = round(float(match.group(1)) * factor, 3)
    return metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_date(value: str) -> str:
    """Format an ISO date or timestamp as a UTC calendar date."""
    if len(value) == 10:
        return value
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).date().isoformat()


def milestone_utc_date(stage: dict[str, Any]) -> str:
    """Use the date when the last required PR for a milestone merged."""
    merge_dates = [
        pull_request["mergeDate"]
        for pull_request in stage.get("relatedPrs", [])
        if pull_request.get("mergeDate")
    ]
    return max(merge_dates) if merge_dates else utc_date(stage["date"])


def write_model_metadata(plan: dict[str, Any], output_root: Path) -> None:
    by_path: dict[Path, dict[str, Any]] = {}
    for stage in plan["stages"]:
        for model in stage["models"]:
            path = resolve_plan_path(output_root, model["path"])
            entry = by_path.setdefault(
                path,
                {
                    "schemaVersion": 1,
                    "model": model["name"],
                    "path": str(path),
                    "builderCommit": model["builderCommit"],
                    "builderDate": model["builderDate"],
                    "sourceRevision": model["sourceRevision"],
                    "modelOptions": stage.get("modelOptions", []),
                    "chartMetrics": stage.get(
                        "chartMetrics", ["prefillTps", "decodeTps"]
                    ),
                    "stages": [],
                },
            )
            if entry["modelOptions"] != stage.get("modelOptions", []):
                raise ValueError(f"conflicting model options for shared model path: {path}")
            for metric in stage.get("chartMetrics", ["prefillTps", "decodeTps"]):
                if metric not in entry["chartMetrics"]:
                    entry["chartMetrics"].append(metric)
            entry["stages"].append(
                {"id": stage["id"], "label": stage["label"], "date": stage["date"]}
            )
    for path, metadata in by_path.items():
        if not path.is_dir():
            raise FileNotFoundError(path)
        files = []
        for file in sorted(path.iterdir()):
            if file.is_file() and file.name != "model-metadata.json":
                files.append(
                    {
                        "name": file.name,
                        "bytes": file.stat().st_size,
                        "sha256": file_sha256(file),
                    }
                )
        metadata["createdAt"] = now()
        metadata["files"] = files
        write_json(path / "model-metadata.json", metadata)


@contextmanager
def model_path_for_stage(
    model_path: Path, stage: dict[str, Any], output_root: Path
):
    if not stage.get("enableGraphCapture"):
        yield model_path
        return

    temporary_root = output_root / "experiments" / "temporary-model-configs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{stage['id']}-", dir=temporary_root) as name:
        staged_path = Path(name)
        for source in model_path.iterdir():
            if not source.is_file():
                continue
            destination = staged_path / source.name
            if source.name == "genai_config.json":
                destination.write_bytes(source.read_bytes())
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    destination.write_bytes(source.read_bytes())
        config_path = staged_path / "genai_config.json"
        config = read_json(config_path)
        provider_options = config["model"]["decoder"]["session_options"].setdefault(
            "provider_options", [{"webgpu": {}}]
        )
        webgpu = next(
            (item["webgpu"] for item in provider_options if "webgpu" in item), None
        )
        if webgpu is None:
            webgpu = {}
            provider_options.append({"webgpu": webgpu})
        webgpu["enableGraphCapture"] = "1"
        webgpu["validationMode"] = "disabled"
        webgpu.setdefault("multiRotaryCacheConcatOffset", "4096")
        write_json(config_path, config)
        yield staged_path


def device() -> dict[str, Any]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,pci.bus_id,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "nvidiaSmi": process.stdout.strip(),
    }


def benchmark_capabilities(executable: Path) -> dict[str, bool]:
    """Inspect a historical C++ benchmark without assuming its current CLI."""
    process = subprocess.run(
        [str(executable), "-h"],
        cwd=executable.parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    help_text = process.stdout or ""
    return {
        "batchSize": bool(re.search(r"(?:^|\s)-b(?:,|\s).*--batch_size", help_text)),
        "maxLength": bool(re.search(r"(?:^|\s)-ml(?:,|\s).*--max_length", help_text)),
    }


def benchmark_metadata(plan: dict[str, Any], command: list[str], capabilities: dict[str, bool]) -> dict[str, Any]:
    max_length_supported = capabilities["maxLength"]
    return {
        "command": subprocess.list2cmdline(command),
        "tool": "C++ model_benchmark.exe",
        "batchSize": plan["batchSize"],
        "requestedMaxKvCacheLength": plan["maxKvCacheLength"],
        "effectiveMaxLength": (
            plan["maxKvCacheLength"]
            if max_length_supported
            else plan["promptLength"] + plan["generationLength"]
        ),
        "maxLengthOption": "-ml" if max_length_supported else None,
        "supportsRequestedMaxLength": max_length_supported,
    }


def run(plan: dict[str, Any], output_root: Path, force: bool) -> dict[str, Any]:
    write_model_metadata(plan, output_root)
    results_path = output_root / "milestone-results.json"
    previous = read_json(results_path) if results_path.is_file() else {"rows": []}
    existing = {
        (row["stageId"], row["model"]): row
        for row in previous.get("rows", [])
        if row.get("status") == "success"
    }
    rows = []
    log_dir = output_root / "logs" / "milestone-runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for stage in plan["stages"]:
        runtime = resolve_plan_path(output_root, stage["runtime"])
        executable = runtime / "model_benchmark.exe"
        capabilities = benchmark_capabilities(executable)
        if stage.get("measurementStatus") == "missing":
            for model in stage["models"]:
                rows.append(
                    {
                        "stageId": stage["id"],
                        "stage": stage["label"],
                        "chartLabel": stage.get("chartLabel", stage["label"]),
                        "date": stage["date"],
                        "backend": stage["backend"],
                        "model": model["name"],
                        "modelPath": str(resolve_plan_path(output_root, model["path"])),
                        "runtimePath": str(runtime),
                        "status": "missing",
                        "error": stage["measurementReason"],
                    }
                )
            continue
        metadata = read_json(runtime / "build-metadata.json")
        for model in stage["models"]:
            key = (stage["id"], model["name"])
            cached = existing.get(key)
            if (
                not force
                and cached
                and cached.get("runtimePath") == str(runtime)
                and cached.get("modelPath") == str(resolve_plan_path(output_root, model["path"]))
            ):
                cached = dict(cached)
                cached.update(
                    {
                        "stage": stage["label"],
                        "chartLabel": stage.get("chartLabel", stage["label"]),
                        "date": stage["date"],
                        "features": stage["features"],
                        "modelOptions": stage.get("modelOptions", []),
                    }
                )
                rows.append(cached)
                continue
            log = log_dir / f"{stage['id']}-{model['name']}.log"
            model_path = resolve_plan_path(output_root, model["path"])
            with model_path_for_stage(model_path, stage, output_root) as run_model_path:
                command = [
                    str(executable),
                    "-i",
                    str(run_model_path),
                    "-b",
                    str(plan["batchSize"]),
                    "-l",
                    str(plan["promptLength"]),
                    "-g",
                    str(plan["generationLength"]),
                    "-r",
                    str(plan["repetitions"]),
                    "-w",
                    str(plan["warmup"]),
                ]
                if capabilities["maxLength"]:
                    command.extend(("-ml", str(plan["maxKvCacheLength"])))
                print(subprocess.list2cmdline(command))
                try:
                    process = subprocess.run(
                        command,
                        cwd=runtime,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=1800,
                        check=False,
                    )
                    output = process.stdout
                    error_message = None
                except subprocess.TimeoutExpired as error:
                    process = None
                    output = error.stdout or ""
                    if isinstance(output, bytes):
                        output = output.decode("utf-8", errors="replace")
                    error_message = "timed out after 1800 seconds"
                    output += f"\nERROR: {error_message}\n"
            log.write_text(output, encoding="utf-8")
            row = {
                "stageId": stage["id"],
                "stage": stage["label"],
                "chartLabel": stage.get("chartLabel", stage["label"]),
                "date": stage["date"],
                "backend": stage["backend"],
                "model": model["name"],
                "modelPath": str(resolve_plan_path(output_root, model["path"])),
                "runtimePath": str(runtime),
                "artifacts": {
                    "model": [
                        str(model_path / name)
                        for name in ("model.onnx", "model.onnx.data", "genai_config.json")
                        if (model_path / name).is_file()
                    ],
                    "runtime": [
                        str(runtime / name)
                        for name in ("model_benchmark.exe", "onnxruntime.dll", "onnxruntime-genai.dll")
                        if (runtime / name).is_file()
                    ],
                },
                "benchmarkInvocation": benchmark_metadata(plan, command, capabilities),
                "ort": metadata["ort"],
                "genai": metadata["genai"],
                "builderCommit": model["builderCommit"],
                "builderDate": model["builderDate"],
                "sourceRevision": model["sourceRevision"],
                "features": stage["features"],
                "modelOptions": stage.get("modelOptions", []),
                "log": str(log),
                "status": "success" if process and process.returncode == 0 else "failed",
            }
            if process and process.returncode == 0:
                row.update(parse_output(output))
            else:
                row["error"] = error_message or f"exit code {process.returncode}"
            rows.append(row)
            write_json(
                results_path,
                {
                    "schemaVersion": 1,
                    "updatedAt": now(),
                    "device": device(),
                    "benchmark": {
                        key: plan[key]
                        for key in ("promptLength", "generationLength", "batchSize", "maxKvCacheLength", "repetitions", "warmup", "targetGpu")
                    },
                    "rows": rows,
                },
            )
    payload = {
        "schemaVersion": 1,
        "updatedAt": now(),
        "device": device(),
        "benchmark": {
            key: plan[key]
            for key in ("promptLength", "generationLength", "batchSize", "maxKvCacheLength", "repetitions", "warmup", "targetGpu")
        },
        "rows": rows,
    }
    write_json(results_path, payload)
    return payload


def svg_chart(
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    lower: bool,
    note: str | None = None,
) -> str:
    eligible = [
        row
        for row in rows
        if metric in row.get("chartMetrics", ["ttftMs", "prefillTps", "decodeTps"])
    ]
    successful = [
        row
        for row in eligible
        if row.get("status") == "success"
        and metric in row
    ]
    width, height, left, right, top, bottom = 1000, 330, 72, 120, 35, 92
    if not successful:
        return f"<div class='chart'><h3>{html.escape(title)}</h3><p>No data</p></div>"
    values = [float(row[metric]) for row in successful]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.12, high * 0.03, 1)
    low = max(0, low - padding)
    high += padding
    xspan, yspan = width - left - right, height - top - bottom
    coords = []
    segments = []
    segment = []
    missing = []
    for index, row in enumerate(eligible):
        x = left + xspan * index / max(len(eligible) - 1, 1)
        if row.get("status") == "success" and metric in row:
            y = top + yspan * (high - float(row[metric])) / max(high - low, 1)
            coords.append((row, x, y))
            segment.append((x, y))
        else:
            if segment:
                segments.append(segment)
                segment = []
            missing.append((row, x))
    if segment:
        segments.append(segment)
    polylines = "".join(
        f"<polyline points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in points)}'/>"
        for points in segments
    )
    dots = "".join(
        f"<g class='{'reference-point' if row.get('isReference') else 'milestone-point'}'>"
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5'><title>{html.escape(row['date'])} · {html.escape(row['stage'])}: {row[metric]}</title></circle></g>"
        for row, x, y in coords
    )
    value_labels = "".join(
        f"<text class='point-value' x='{x:.1f}' y='{(y + 19 if y < top + 22 else y - 10):.1f}' "
        f"text-anchor='middle'>{float(row[metric]):.1f}</text>"
        for row, x, y in coords
    )
    missing_markers = "".join(
        f"<g class='missing-point'><line x1='{x-5:.1f}' y1='{height-bottom-11}' x2='{x+5:.1f}' y2='{height-bottom-1}'/>"
        f"<line x1='{x+5:.1f}' y1='{height-bottom-11}' x2='{x-5:.1f}' y2='{height-bottom-1}'/>"
        f"<text x='{x:.1f}' y='{height-bottom-17}' text-anchor='middle'>N/A</text>"
        f"<title>{html.escape(row['date'])} · {html.escape(row['stage'])}: {html.escape(row.get('error', 'No measurement'))}</title></g>"
        for row, x in missing
    )
    labels = "".join(
        f"<text transform='translate({x:.1f},{height-bottom+20}) rotate(30)'>{html.escape(row['date'][:7])} · {html.escape(row['chartLabel'])}</text>"
        for index, row in enumerate(eligible)
        for x in [left + xspan * index / max(len(eligible) - 1, 1)]
    )
    direction = "lower is better" if lower else "higher is better"
    first, last = float(successful[0][metric]), float(successful[-1][metric])
    ratio = first / last if lower else last / first
    gain_label = "Overall gain" if ratio >= 1 else "Overall change"
    gain = f"{gain_label}: {ratio:.2f}×"
    chart_note = (
        f"<p class='chart-note'>{html.escape(note)}</p>" if note else ""
    )
    return f"""<div class="chart"><div class="chart-heading"><h3>{html.escape(title)} <small>{direction}</small></h3><span class="gain-badge">{gain}</span></div>
    <svg viewBox="0 0 {width} {height}"><line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
    <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
    <text x="8" y="{top+6}">{high:.1f}</text><text x="8" y="{height-bottom+5}">{low:.1f}</text>
    {polylines}<g>{dots}{missing_markers}</g>{value_labels}{labels}</svg>{chart_note}</div>"""


def optimization_impacts(plan: dict[str, Any]) -> dict[str, list[str]]:
    impacts: dict[str, list[str]] = {}
    for group in plan.get("optimizationGroups", []):
        for item in group["stages"]:
            impacts.setdefault(item["id"], []).append(group["id"])
    return impacts


def chart_metrics_by_stage(plan: dict[str, Any]) -> dict[str, list[str]]:
    metrics: dict[str, list[str]] = {}
    for stage_id, impacts in optimization_impacts(plan).items():
        stage_metrics = []
        if "prefill" in impacts:
            stage_metrics.append("prefillTps")
        if "decode" in impacts:
            stage_metrics.append("decodeTps")
        metrics[stage_id] = stage_metrics
    return metrics


def rows_for_chart(
    plan: dict[str, Any], payload: dict[str, Any], model: str, metric: str
) -> list[dict[str, Any]]:
    group = next(
        group for group in plan.get("optimizationGroups", []) if group["metric"] == metric
    )
    stage_items = {item["id"]: item for item in group["stages"]}
    stages = {stage["id"]: stage for stage in plan["stages"]}
    reference = group.get("reference")
    selected = []
    if reference and (measurement := reference.get("measurement")):
        value = measurement.get(metric)
        if isinstance(value, (int, float)) and value > 0:
            selected.append(
                {
                    "stageId": reference.get("stageId", f"{group['id']}-reference"),
                    "stage": reference["label"],
                    "chartLabel": reference["chartLabel"],
                    "date": reference["date"],
                    "model": model,
                    "status": "success",
                    "isReference": True,
                    "chartMetrics": [metric],
                    metric: value,
                }
            )
    for source in payload.get("rows", []):
        if source.get("model") != model:
            continue
        stage_id = source.get("stageId")
        if stage_id in stage_items:
            row = dict(source)
            row["date"] = milestone_utc_date(stages[stage_id])
            row["stage"] = stages[stage_id]["label"]
            row["chartLabel"] = stages[stage_id].get(
                "chartLabel", stages[stage_id]["label"]
            )
            if stages[stage_id].get("measurementStatus") == "missing":
                row["status"] = "missing"
                row["error"] = stages[stage_id]["measurementReason"]
        elif reference and stage_id == reference.get("stageId"):
            row = dict(source)
            if stage_id in stages:
                row["date"] = milestone_utc_date(stages[stage_id])
            row["stage"] = reference["label"]
            row["chartLabel"] = reference["chartLabel"]
            row["isReference"] = True
        else:
            continue
        row["chartMetrics"] = [metric]
        selected.append(row)
    return selected


def optimization_summaries(plan: dict[str, Any]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for group in plan.get("optimizationGroups", []):
        for item in group["stages"]:
            summaries.setdefault(item["id"], item["summary"])
    return summaries


def impact_badges(impacts: list[str]) -> str:
    return (
        "<div class='impact-badges'>"
        + "".join(
            f"<span class='impact-badge impact-{html.escape(impact)}'>"
            f"{html.escape(impact.title())}</span>"
            for impact in impacts
        )
        + "</div>"
    )


def builder_cell(models: list[dict[str, Any]]) -> str:
    builders = sorted(
        {(model["builderCommit"][:10], model["builderDate"][:10]) for model in models}
    )
    return "".join(
        f"<div class='builder-entry'><code>{html.escape(commit)}</code><br>"
        f"<small>{html.escape(builder_date)}</small></div>"
        for commit, builder_date in builders
    )


def archive_relative(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path.resolve())


def local_artifacts_cell(
    stage: dict[str, Any], payload: dict[str, Any], output_root: Path
) -> str:
    model_entries = []
    for model in stage["models"]:
        model_path = resolve_plan_path(output_root, model["path"])
        files = [
            archive_relative(model_path / name, output_root)
            for name in ("model.onnx", "model.onnx.data", "genai_config.json")
            if (model_path / name).is_file()
        ]
        model_entries.extend(files)
    runtime = resolve_plan_path(output_root, stage["runtime"])
    runtime_entries = [
        archive_relative(runtime / name, output_root)
        for name in ("model_benchmark.exe", "onnxruntime.dll", "onnxruntime-genai.dll")
        if (runtime / name).is_file()
    ]
    result_rows = [
        row for row in payload.get("rows", []) if row.get("stageId") == stage["id"]
    ]
    invocations = [
        row["benchmarkInvocation"] for row in result_rows if row.get("benchmarkInvocation")
    ]
    invocation_html = ""
    if invocations:
        invocation = invocations[0]
        max_length = (
            f"{invocation['effectiveMaxLength']} via {invocation['maxLengthOption']}"
            if invocation["supportsRequestedMaxLength"]
            else f"{invocation['effectiveMaxLength']} (8K option unavailable)"
        )
        invocation_html = (
            "<br><strong>Run</strong><br>"
            f"<small>batch {invocation['batchSize']} · effective max length {html.escape(max_length)}</small>"
        )
    def file_list(items: list[str]) -> str:
        return "<br>".join(f"<code>{html.escape(item)}</code>" for item in items)
    return (
        "<div class='artifact-list'><strong>Model</strong><br>"
        f"{file_list(model_entries)}<br><strong>Runtime</strong><br>"
        f"{file_list(runtime_entries)}{invocation_html}</div>"
    )


def builder_option_history_cell(option: dict[str, Any]) -> str:
    entries = []
    for change in sorted(
        option["defaultHistory"], key=lambda item: item["date"], reverse=True
    ):
        commit = change["commit"]
        pr = int(change["pr"])
        entries.append(
            "<div class='option-history-entry'>"
            f"<strong>{html.escape(change['date'])}</strong> · "
            f"{html.escape(change['value'])}<br>"
            f'<small><a href="https://github.com/microsoft/onnxruntime-genai/pull/{pr}">'
            f"ORT GenAI #{pr}</a> · "
            f'<a href="https://github.com/microsoft/onnxruntime-genai/commit/{html.escape(commit)}">'
            f"<code>{html.escape(commit[:10])}</code></a></small></div>"
        )
    return "".join(entries)


def builder_options_table(plan: dict[str, Any]) -> str:
    rows = []
    options = plan.get("builderOptions", [])
    for option in options:
        requirement_level = option["requirement"].split(maxsplit=1)[0].rstrip(";:").lower()
        aliases = option.get("legacyNames", [])
        alias_html = (
            "<br><small>Formerly "
            + ", ".join(f"<code>{html.escape(alias)}</code>" for alias in aliases)
            + "</small>"
            if aliases
            else ""
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(option['name'])}</code>{alias_html}</td>"
            f"<td>{html.escape(option['description'])}</td>"
            f"<td><span class='requirement-badge requirement-{html.escape(requirement_level)}'>"
            f"{html.escape(option['requirement'])}</span></td>"
            f"<td><code>{html.escape(option['seriesValue'])}</code></td>"
            f"<td>{html.escape(option['currentDefault'])}</td>"
            f"<td>{builder_option_history_cell(option)}</td></tr>"
        )
    standard_profile = " ".join(
        f"{option['name']}={option['seriesValue']}"
        for option in options
        if option.get("profile") == "standard"
    )
    required_profile = " ".join(
        f"{option['name']}={option['seriesValue']}"
        for option in options
        if option.get("profile") == "standard"
        and option.get("requirement", "").startswith("Mandatory")
    )
    return (
        "<section id='model-builder-options'><h2>Model-builder options</h2>"
        "<p><strong>Required now</strong><br>"
        f"<code class='profile-command'>{html.escape(required_profile)}</code><br>"
        "<small>Full explicitly pinned comparable profile:</small><br>"
        f"<code class='profile-command'>{html.escape(standard_profile)}</code></p>"
        "<p class='muted'>Current canonical option names are shown below. The current profile "
        "is the setting to use for new comparable model builds; archived measurements retain "
        "the exact names and values supported by their pinned historical builder commits.</p>"
        "<div class='table'><table class='builder-options-table'><thead><tr>"
        "<th>Option</th><th>Description</th><th>Requirement now</th><th>Current profile</th>"
        "<th>Current default</th><th>Default / naming history</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    ) + genai_config_options_table(plan) + ort_genai_benchmark_options_table(plan) + (
        "<span id='milestones' class='anchor-target'></span>"
    )


def ort_genai_benchmark_options_table(plan: dict[str, Any]) -> str:
    rows = []
    for option in plan.get("ortGenaiBenchmarkOptions", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(option['tool'])}</td>"
            f"<td><code>{html.escape(option['option'])}</code></td>"
            f"<td><code>{html.escape(option['value'])}</code></td>"
            f"<td>{html.escape(option['description'])}</td></tr>"
        )
    command = (
        f"-i &lt;model directory&gt; -b {plan['batchSize']} -l {plan['promptLength']} "
        f"-g {plan['generationLength']} -ml {plan['maxKvCacheLength']} "
        f"-r {plan['repetitions']} -w {plan['warmup']}"
    )
    return (
        "<section id='ort-genai-options'><h2>ORT GenAI benchmark options</h2>"
        "<p><strong>Required benchmark arguments</strong><br>"
        f"<code class='profile-command'>{command}</code></p>"
        "<p class='muted'>These are ORT GenAI benchmark arguments, separate from "
        "the WebGPU execution-provider settings in <code>genai_config.json</code>. "
        "The C++ runner uses <code>-ml 8192</code>; the Python runner uses <code>-m 8192</code>. "
        "Both fix the maximum KV-cache capacity at 8K tokens. "
        "Historical C++ binaries without <code>-ml</code> are retained with their effective prompt + generation length and are explicitly marked as not supporting an 8K request.</p>"
        "<div class='table'><table class='benchmark-options-table'><thead><tr>"
        "<th>Tool</th><th>Option</th><th>Profile value</th><th>Description</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def genai_config_options_table(plan: dict[str, Any]) -> str:
    important_rows = []
    for option in plan.get("importantGenaiConfigOptions", []):
        display_name = option["path"].removeprefix("provider_options.webgpu.")
        important_rows.append(
            "<tr>"
            f"<td><code>{html.escape(display_name)}</code></td>"
            f"<td>{html.escape(option['location'])}</td>"
            f"<td>{html.escape(option['profile'])}</td>"
            f"<td>{html.escape(option['description'])}</td></tr>"
        )
    rows = []
    for option in plan.get("genaiConfigOptions", []):
        display_name = option["path"].removeprefix("provider_options.webgpu.")
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(display_name)}</code></td>"
            f"<td>{html.escape(option['area'])}</td>"
            f"<td><span class='requirement-badge requirement-{html.escape(option['requirement'].lower())}'>"
            f"{html.escape(option['requirement'])}</span></td>"
            f"<td>{html.escape(option['profile'])}</td>"
            f"<td>{html.escape(option['currentDefault'])}</td>"
            f"<td>{html.escape(option['description'])}</td></tr>"
        )
    return (
        "<section id='webgpu-options'><h2>WebGPU genai_config.json options</h2>"
        "<h3>Important settings</h3>"
        "<p class='muted'>These settings are listed first because they directly affect the "
        "WebGPU performance configuration. <code>optimization.disable_specified_optimizers</code> "
        "is an ORT session setting under <code>model.decoder.session_options</code>; the other "
        "entries shown here belong inside the <code>webgpu</code> provider object.</p>"
        "<div class='table'><table class='important-genai-options-table'><thead><tr>"
        "<th>Option</th><th>Location</th><th>Profile value</th><th>Why it matters</th>"
        f"</tr></thead><tbody>{''.join(important_rows)}</tbody></table></div>"
        "<h3>All WebGPU provider settings</h3>"
        "<p class='muted'>Only WebGPU-specific provider settings are listed below. Stored "
        "model configs keep graph capture disabled; only the controlled graph-capture run "
        "enables it on a temporary copy.</p>"
        "<div class='table'><table class='genai-options-table'><thead><tr>"
        "<th>Option</th><th>Area</th><th>Requirement now</th>"
        "<th>Current profile</th><th>Current default</th><th>Description</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def report_index() -> str:
    links = (
        ("Performance charts", "performance-charts"),
        ("Model-builder options", "model-builder-options"),
        ("WebGPU genai_config.json options", "webgpu-options"),
        ("ORT GenAI benchmark options", "ort-genai-options"),
        ("Milestones", "milestones"),
        ("Scope", "scope"),
        ("Method", "method"),
    )
    items = "".join(
        f"<li><a href='#{anchor}'>{html.escape(label)}</a></li>"
        for label, anchor in links
    )
    return (
        "<nav id='report-index' class='report-index' aria-label='Report navigation' hidden>"
        "<div class='report-index-heading'>"
        "<button id='menu-hide' class='menu-toggle' type='button' "
        "aria-controls='report-index' aria-expanded='false'>Close menu</button></div>"
        f"<ol>{items}</ol></nav>"
    )


def related_pull_requests_cell(
    stage: dict[str, Any], repository_filter: str | None = None
) -> str:
    pull_requests = stage.get("relatedPrs") or [
        {
            "repository": stage.get("milestoneRepo", "onnxruntime"),
            "number": stage["milestonePr"],
        }
    ]
    if repository_filter:
        pull_requests = [
            pull_request
            for pull_request in pull_requests
            if pull_request.get("repository", "onnxruntime") == repository_filter
        ]
    pull_requests = sorted(
        pull_requests,
        key=lambda pull_request: pull_request.get("mergeDate", ""),
        reverse=True,
    )
    links = []
    for pull_request in pull_requests:
        repository = pull_request.get("repository", "onnxruntime")
        number = int(pull_request["number"])
        url = f"https://github.com/microsoft/{repository}/pull/{number}"
        role = pull_request.get("role")
        merge_date = pull_request.get("mergeDate")
        merge_commit = pull_request.get("mergeCommit")
        owner = pull_request.get("owner")
        merge_date_html = (
            f"<br><small>Merged {html.escape(merge_date)}</small>" if merge_date else ""
        )
        merge_commit_html = (
            f'<br><small>Merge commit <a href="https://github.com/microsoft/{repository}/commit/{html.escape(merge_commit)}">'
            f"<code>{html.escape(merge_commit[:10])}</code></a></small>"
            if merge_commit
            else ""
        )
        owner_html = (
            f'<br><small>Owner <a href="https://github.com/{html.escape(owner)}">'
            f"@{html.escape(owner)}</a></small>"
            if owner
            else ""
        )
        role_html = f"<br><small>{html.escape(role)}</small>" if role else ""
        links.append(
            f'<div class="pr-entry"><a href="{url}">{html.escape(repository)} #{number}</a>'
            f"{merge_date_html}{merge_commit_html}{owner_html}{role_html}</div>"
        )
    return "".join(links)


def runtime_and_pull_requests_cell(
    runtime: dict[str, Any], stage: dict[str, Any], repository: str
) -> str:
    pull_requests = related_pull_requests_cell(stage, repository)
    required = (
        f"<div class='runtime-prs'><strong>Required PRs</strong>{pull_requests}</div>"
        if pull_requests
        else "<div class='runtime-prs'><small>No milestone PR required in this repository.</small></div>"
    )
    note = stage.get("runtimeNotes", {}).get(repository)
    note_html = (
        f"<div class='runtime-note'><small>{html.escape(note)}</small></div>"
        if note
        else ""
    )
    return (
        "<div class='tested-commit'><strong>Tested commit</strong><br>"
        f"<code>{html.escape(runtime['commit'][:10])}</code><br>"
        f"<small>{html.escape(utc_date(runtime['commitDate']))}</small></div>"
        f"{note_html}{required}"
    )


def measured_impact_cell(
    stage: dict[str, Any], plan: dict[str, Any], payload: dict[str, Any]
) -> str:
    if stage.get("measurementStatus") == "missing":
        reported = stage.get("reportedImpact")
        reported_html = (
            f"<br><small>{html.escape(reported)}</small>" if reported else ""
        )
        return (
            "<div class='impact-measurement'><strong>Not measured in this series</strong><br>"
            f"<small>{html.escape(stage['measurementReason'])}</small>{reported_html}</div>"
        )
    if measurement := stage.get("controlledMeasurement"):
        before = float(measurement["parent"])
        after = float(measurement["candidate"])
        metric_label = html.escape(measurement["metricLabel"])
        return (
            "<div class='impact-measurement'>"
            f"<strong>{after / before:.2f}×</strong><br>"
            f"<small>{before:.1f} → {after:.1f} {metric_label}</small><br>"
            f"<small>{html.escape(measurement.get('comparisonLabel', 'Controlled parent → candidate measurement'))}</small></div>"
        )

    stage_id = stage["id"]
    rows = {
        row.get("stageId"): row
        for row in payload.get("rows", [])
        if row.get("status") == "success"
    }
    parts = []
    for group in plan.get("optimizationGroups", []):
        stage_ids = [item["id"] for item in group["stages"]]
        if stage_id not in stage_ids:
            continue
        metric = group["metric"]
        current = rows.get(stage_id, {}).get(metric)
        if not isinstance(current, (int, float)) or current <= 0:
            parts.append(
                f"<strong>{html.escape(group['title'])}</strong><br>Measurement pending"
            )
            continue
        prior = None
        for prior_id in reversed(stage_ids[: stage_ids.index(stage_id)]):
            value = rows.get(prior_id, {}).get(metric)
            if isinstance(value, (int, float)) and value > 0:
                prior = value
                break
        if prior is None and group.get("reference"):
            reference = group["reference"]
            reference_id = reference.get("stageId")
            value = rows.get(reference_id, {}).get(metric) if reference_id else None
            if value is None:
                value = reference.get("measurement", {}).get(metric)
            if isinstance(value, (int, float)) and value > 0:
                prior = value
        metric_label = html.escape(group["metricLabel"])
        if prior is None:
            parts.append(
                f"<strong>{current:.1f} {metric_label}</strong><br>"
                "<small>First measured milestone in this series</small>"
            )
        elif current >= prior:
            parts.append(
                f"<strong>{current / prior:.2f}×</strong><br>"
                f"<small>{prior:.1f} → {current:.1f} {metric_label}</small>"
            )
        else:
            reported = stage.get("reportedImpact")
            reported_html = (
                f"<br><small>{html.escape(reported)}</small>" if reported else ""
            )
            parts.append(
                f"<strong>{current:.1f} {metric_label}</strong><br>"
                "<small>The adjacent historical snapshots include other stack changes; "
                f"this transition is not an isolated milestone impact.</small>{reported_html}"
            )
    return "<div class='impact-measurement'>" + "<hr>".join(parts) + "</div>"


def render(
    plan: dict[str, Any],
    payload: dict[str, Any],
    output_root: Path,
    report_path: Path,
) -> None:
    model_sections = []
    model_names = []
    for stage in plan["stages"]:
        for model in stage["models"]:
            if model["name"] not in model_names:
                model_names.append(model["name"])
    for model in model_names:
        prefill_group = next(
            group for group in plan["optimizationGroups"] if group["metric"] == "prefillTps"
        )
        decode_group = next(
            group for group in plan["optimizationGroups"] if group["metric"] == "decodeTps"
        )
        prefill_note = " ".join(
            note
            for note in (
                prefill_group.get("reference", {}).get("note"),
                prefill_group.get("chartNote"),
            )
            if note
        )
        section_id = "performance-charts" if not model_sections else f"performance-{len(model_sections) + 1}"
        model_sections.append(
            f"<section id='{section_id}'><h2>{html.escape(model)}</h2>"
            + svg_chart(
                rows_for_chart(plan, payload, model, "prefillTps"),
                "prefillTps",
                "Prefill throughput",
                False,
                prefill_note,
            )
            + svg_chart(
                rows_for_chart(plan, payload, model, "decodeTps"),
                "decodeTps",
                "Decode throughput",
                False,
                decode_group.get("chartNote"),
            )
            + "</section>"
        )
    milestones = []
    impacts = optimization_impacts(plan)
    summaries = optimization_summaries(plan)
    for stage in sorted(plan["stages"], key=lambda item: item["date"], reverse=True):
        runtime = read_json(
            resolve_plan_path(output_root, stage["runtime"]) / "build-metadata.json"
        )
        milestones.append(
            "<tr>"
            f"<td>{html.escape(milestone_utc_date(stage))}</td>"
            f"<td><strong>{html.escape(stage['label'])}</strong><br>"
            f"<span class='detail-intro'>{html.escape(summaries[stage['id']])}</span><br>"
            f"<small class='applicability'>{html.escape(stage['applicability'])}</small></td>"
            f"<td>{'<br>'.join(html.escape(name) for name in stage['majorContributors'])}</td>"
            f"<td>{impact_badges(impacts.get(stage['id'], []))}</td>"
            f"<td>{measured_impact_cell(stage, plan, payload)}</td>"
            f"<td>{runtime_and_pull_requests_cell(runtime['ort'], stage, 'onnxruntime')}</td>"
            f"<td>{runtime_and_pull_requests_cell(runtime['genai'], stage, 'onnxruntime-genai')}</td>"
            f"<td>{builder_cell(stage['models'])}</td>"
            f"<td>{local_artifacts_cell(stage, payload, output_root)}</td>"
            f"<td>{html.escape(' · '.join(stage['features']))}<br><small>{html.escape(' '.join(stage.get('modelOptions', [])))}</small></td></tr>"
        )
    report = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>ORT WebGPU Phi-4 Major Performance Improvements</title><style>
    :root{{--ink:#172033;--muted:#667085;--line:#d8e0ec;--bg:#eef4ff;--panel:#fff;--blue:#2563eb;--teal:#0f766e;--prefill:#b45309;--prefill-bg:#fff7ed;--decode:#047857;--decode-bg:#ecfdf5}}
    *{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:linear-gradient(#e7f0ff,#f5f7fb 42rem);color:var(--ink);font:15px/1.55 Aptos,"Segoe UI",sans-serif}}
    .page-shell{{display:grid;grid-template-columns:minmax(190px,250px) minmax(0,1fr);gap:clamp(18px,2.5vw,42px);width:min(2400px,calc(100% - clamp(24px,4vw,80px)));margin:auto}}.report-content{{min-width:0}}.wrap{{width:100%;margin:auto}}header{{padding:clamp(38px,5vw,72px) 0 28px}}h1,h2,h3,h4{{font-family:Bahnschrift,Aptos,"Segoe UI",sans-serif}}h1{{font-size:clamp(38px,5vw,76px);line-height:1;margin:8px 0 18px;letter-spacing:-.045em}}h2{{font-size:clamp(25px,2vw,32px);line-height:1.15}}h3{{margin:0;font-size:22px}}small,.muted{{color:var(--muted)}}.eyebrow{{color:var(--blue);font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.chart-heading{{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}}.gain-badge{{display:inline-flex;padding:5px 10px;border-radius:999px;background:#eff6ff;border:1px solid #93c5fd;color:#1d4ed8;font-size:12px;font-weight:800;white-space:nowrap}}
    section,.chart,.report-index{{background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 32px rgba(40,65,100,.08)}}section{{padding:clamp(18px,2vw,32px);margin:20px 0}}.anchor-target{{display:block;scroll-margin-top:16px}}section{{scroll-margin-top:16px}}.report-index{{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;padding:20px;margin:20px 0}}.report-index h2{{font-size:20px;margin:0 0 14px}}.report-index ol{{display:grid;grid-template-columns:1fr;gap:8px;margin:0;padding:0;list-style:none}}.report-index a{{display:block;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#f8faff;color:var(--blue);font-weight:700;text-decoration:none}}.report-index a:hover,.report-index a:focus-visible{{border-color:#93c5fd;background:#eff6ff}}.chart{{padding:clamp(14px,1.5vw,22px);margin:16px 0;box-shadow:none;overflow-x:auto}}.chart svg{{display:block;width:100%;min-width:760px;max-width:1500px;margin:auto}}svg text{{fill:var(--muted);font-size:11px}}svg text.point-value{{fill:var(--ink);font-size:11px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}}.axis{{stroke:var(--line)}}polyline{{fill:none;stroke:var(--blue);stroke-width:3}}circle{{fill:var(--teal)}}.reference-point circle{{fill:#fff;stroke:var(--blue);stroke-width:3}}.missing-point line{{stroke:#b42318;stroke-width:2}}.missing-point text{{fill:#b42318;font-weight:800}}.chart-note{{margin:4px 0 0;color:var(--muted);font-size:12px}}
    table{{width:100%;border-collapse:collapse}}.milestone-table{{min-width:1750px}}.builder-options-table,.genai-options-table{{min-width:1150px}}.important-genai-options-table{{min-width:820px}}th,td{{padding:12px clamp(9px,1vw,16px);border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:11px;text-transform:uppercase;color:var(--muted)}}.table{{width:100%;overflow-x:auto;overscroll-behavior-inline:contain}}.requirement-badge{{display:inline-flex;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800;line-height:1.35}}.requirement-mandatory{{background:#fef2f2;color:#b42318;border:1px solid #fca5a5}}.requirement-conditional{{background:#fffbeb;color:#a15c00;border:1px solid #fcd34d}}.requirement-optional{{background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd}}.optimization-intro{{display:inline-block;max-width:320px;margin-top:4px;color:var(--muted);font-size:12px;line-height:1.4}}.detail-intro{{display:inline-block;min-width:240px;max-width:420px}}.applicability{{display:inline-block;max-width:420px;margin-top:5px}}.impact-badges{{display:flex;flex-wrap:wrap;gap:5px}}.impact-badge{{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}}.impact-prefill{{background:var(--prefill-bg);border:1px solid #fdba74;color:var(--prefill)}}.impact-decode{{background:var(--decode-bg);border:1px solid #6ee7b7;color:var(--decode)}}.impact-measurement{{min-width:170px}}.impact-measurement hr{{border:0;border-top:1px solid var(--line);margin:8px 0}}.builder-entry+.builder-entry,.pr-entry+.pr-entry,.option-history-entry+.option-history-entry{{margin-top:8px}}.option-history-entry{{min-width:230px}}.pr-entry{{min-width:190px}}.artifact-list{{min-width:260px;font-size:11px}}.artifact-list code{{overflow-wrap:anywhere}}.runtime-prs{{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}}.runtime-prs>.pr-entry{{margin-top:7px}}code{{font-family:Consolas,monospace}}footer{{padding:32px;text-align:center;color:var(--muted)}}
    @media (min-width:1500px){{.milestone-table{{min-width:100%;table-layout:fixed}}.builder-options-table,.genai-options-table,.important-genai-options-table{{min-width:100%;table-layout:fixed}}.builder-options-table th:nth-child(1){{width:13%}}.builder-options-table th:nth-child(2){{width:22%}}.builder-options-table th:nth-child(3){{width:13%}}.builder-options-table th:nth-child(4){{width:12%}}.builder-options-table th:nth-child(5){{width:15%}}.builder-options-table th:nth-child(6){{width:25%}}}}
    @media (max-width:900px){{.page-shell{{grid-template-columns:1fr;width:calc(100% - 20px);gap:0}}.report-index{{position:static;max-height:none;margin:12px 0 0}}.report-index ol{{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}}header{{padding-top:30px}}section{{border-radius:14px;margin:12px 0}}.chart{{border-radius:12px}}footer{{padding:24px 12px}}}}
    .benchmark-options-table{{min-width:680px}}.page-shell{{display:block;width:min(2400px,calc(100% - clamp(24px,4vw,80px)));margin:auto}}.report-index{{position:fixed;top:62px;left:clamp(12px,2vw,28px);z-index:30;width:min(380px,calc(100vw - 24px));max-height:calc(100vh - 76px);overflow:auto;padding:20px;margin:0;box-shadow:0 20px 60px rgba(23,32,51,.24)}}.report-index[hidden]{{display:none}}.report-index-heading{{display:flex;justify-content:flex-end;margin-bottom:14px}}.menu-toggle{{border:1px solid #93c5fd;border-radius:9px;background:#eff6ff;color:#1d4ed8;padding:7px 10px;font:inherit;font-size:12px;font-weight:800;cursor:pointer}}.menu-toggle:hover,.menu-toggle:focus-visible{{background:#dbeafe}}.menu-show{{display:inline-flex;position:fixed;top:12px;left:clamp(12px,2vw,28px);z-index:30;box-shadow:0 8px 22px rgba(23,32,51,.16)}}.page-shell.menu-open .menu-show{{display:none}}@media (max-width:900px){{.page-shell{{width:calc(100% - 20px)}}.report-index{{position:fixed;top:58px;left:10px;width:calc(100vw - 20px);max-height:calc(100vh - 68px);margin:0}}.report-index ol{{grid-template-columns:1fr}}}}
    </style></head><body><div class="page-shell">{report_index()}<div class="report-content"><button id="menu-show" class="menu-toggle menu-show" type="button" aria-controls="report-index" aria-expanded="false" aria-haspopup="true">Menu</button><header><div class="wrap"><div class="eyebrow">RTX 5080 · Prompt {plan['promptLength']} · Target max KV cache {plan['maxKvCacheLength']} · Milestone-aligned models</div>
    <h1>ORT WebGPU Phi-4 Major Performance Improvements</h1><p class="muted">Prefill and decode are independent, cumulative optimization series. Only measured performance improvements are published as nodes.</p></div></header>
    <main class="wrap">{''.join(model_sections)}{builder_options_table(plan)}<section><h2>Milestones</h2><p class="muted">Measured impact uses an archived controlled parent-to-candidate run when available; otherwise it compares with the preceding published point in the same cumulative series and should not be read as isolated attribution. Performance area identifies whether the milestone improves prompt processing (prefill) or token generation (decode). Dates use one YYYY-MM-DD convention, and a milestone date is when its last required PR merged. The ORT and ORT GenAI columns each show the cumulative commit used for the test and that repository's required PRs; those hashes match only when the test runtime is pinned to the exact PR merge commit. Local test artifacts are archive-relative paths.</p><div class="table"><table class="milestone-table"><thead><tr><th>Date</th><th>Milestone / introduction</th><th>Contributors</th><th>Performance area</th><th>Measured impact</th><th>ORT</th><th>ORT GenAI</th><th>Builder / date</th><th>Local test artifacts</th><th>Model/runtime features</th></tr></thead><tbody>{''.join(milestones)}</tbody></table></div></section>
    <section id="scope"><h2>Scope</h2><p>This report covers the selected Phi-4 text-only prefill and decode milestones on RTX 5080. Other ORT WebGPU work, including Qualcomm-specific prefill and decode optimizations, Whisper, Phi-4 multimodal, GPT-OSS, and additional model or hardware optimizations, is outside this series and is not represented here.</p></section>
    <section id="method"><h2>Method</h2><p>All measurements run on {html.escape(plan['targetGpu'])}. New and rerun measurements use batch {plan['batchSize']}, prompt {plan['promptLength']}, generation {plan['generationLength']}, max KV cache {plan['maxKvCacheLength']} (<code>-ml {plan['maxKvCacheLength']}</code> in C++ or <code>-m {plan['maxKvCacheLength']}</code> in Python), and {plan['repetitions']} measured repetitions after {plan['warmup']} warmup. Historical C++ runners without <code>-ml</code> cannot request 8K and are explicitly labeled with their effective prompt + generation length. The historical series keeps two pinned accuracy-level 4, graph-ready models (fused or separate RoPE). A model-changing milestone uses an archived matched control/candidate experiment; graph capture remains disabled in stored configs and is enabled only on a temporary copy for its measurement.</p></section></main>
    <footer>Results updated {html.escape(payload['updatedAt'])} · {html.escape(payload['device']['nvidiaSmi'])}</footer></div></div><script>(()=>{{const shell=document.querySelector('.page-shell');const menu=document.getElementById('report-index');const close=document.getElementById('menu-hide');const show=document.getElementById('menu-show');const setOpen=(open,moveFocus=true)=>{{menu.hidden=!open;shell.classList.toggle('menu-open',open);close.setAttribute('aria-expanded',String(open));show.setAttribute('aria-expanded',String(open));if(moveFocus)(open?close:show).focus();}};show.addEventListener('click',()=>setOpen(true));close.addEventListener('click',()=>setOpen(false));menu.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>setOpen(false,false)));document.addEventListener('keydown',event=>{{if(event.key==='Escape'&&!menu.hidden){{event.preventDefault();setOpen(false);}}}});document.addEventListener('pointerdown',event=>{{if(!menu.hidden&&!menu.contains(event.target)&&event.target!==show)setOpen(false,false);}});}})();</script></body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")
    archived_report_path = output_root / "index.html"
    if archived_report_path.resolve() != report_path.resolve():
        archived_report_path.write_text(report, encoding="utf-8")
        print(f"wrote {archived_report_path}")


def verify(
    plan: dict[str, Any],
    payload: dict[str, Any],
    output_root: Path,
    report_path: Path,
    verify_hashes: bool,
) -> None:
    issues = []
    hash_cache: dict[Path, str] = {}

    def check_hash(path: Path, expected: str, label: str) -> None:
        if not path.is_file():
            return
        actual = hash_cache.setdefault(path, file_sha256(path))
        if actual != expected:
            issues.append(f"{label}: SHA-256 mismatch for {path}")

    benchmark = payload.get("benchmark", {})
    for key in ("promptLength", "generationLength", "batchSize", "maxKvCacheLength", "repetitions", "warmup", "targetGpu"):
        if benchmark.get(key) != plan.get(key):
            issues.append(f"benchmark {key} does not match the plan")
    if plan.get("maxKvCacheLength") != 8192:
        issues.append("ORT GenAI benchmark maxKvCacheLength must be 8192")
    benchmark_options = {
        option.get("option"): option
        for option in plan.get("ortGenaiBenchmarkOptions", [])
    }
    if benchmark_options.get("-m / --max_lengths", {}).get("value") != "8192":
        issues.append("Python ORT GenAI benchmark options must include -m 8192")
    if benchmark_options.get("-ml / --max_length", {}).get("value") != "8192":
        issues.append("C++ ORT GenAI benchmark options must include -ml 8192")
    if benchmark_options.get("-b / --batch_size", {}).get("value") != "1":
        issues.append("ORT GenAI benchmark options must include batch size 1")
    device_name = payload.get("device", {}).get("nvidiaSmi", "")
    if plan.get("targetGpu") not in device_name:
        issues.append(f"result device does not contain target GPU: {plan.get('targetGpu')}")

    builder_options = plan.get("builderOptions", [])
    option_names = [option.get("name") for option in builder_options]
    required_options = {
        "accuracy_level",
        "is_symmetric",
        "algo_config",
        "enable_webgpu_graph",
        "shared_embeddings",
        "prune_lm_head",
    }
    if len(option_names) != len(set(option_names)):
        issues.append("builder option names must be unique")
    if missing_options := required_options - set(option_names):
        issues.append(
            "missing tracked builder options: " + ", ".join(sorted(missing_options))
        )
    for option in builder_options:
        name = option.get("name", "<unnamed>")
        for key in ("description", "requirement", "seriesValue", "currentDefault"):
            if not option.get(key):
                issues.append(f"builder option {name}: missing {key}")
        if option.get("requirement", "").split(maxsplit=1)[0].rstrip(";:") not in {
            "Mandatory",
            "Conditional",
            "Optional",
        }:
            issues.append(f"builder option {name}: invalid requirement")
        history = option.get("defaultHistory")
        if not isinstance(history, list) or not history:
            issues.append(f"builder option {name}: missing default history")
            continue
        dates = [change.get("date", "") for change in history]
        if dates != sorted(dates, reverse=True):
            issues.append(f"builder option {name}: history must be newest first")
        for change in history:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", change.get("date", "")):
                issues.append(f"builder option {name}: invalid history date")
            if not change.get("value"):
                issues.append(f"builder option {name}: missing historical value")
            if not re.fullmatch(r"[0-9a-f]{40}", change.get("commit", "")):
                issues.append(f"builder option {name}: invalid history commit")
            if not isinstance(change.get("pr"), int) or change["pr"] <= 0:
                issues.append(f"builder option {name}: invalid history PR")

    genai_options = plan.get("genaiConfigOptions", [])
    genai_paths = [option.get("path") for option in genai_options]
    if len(genai_paths) != len(set(genai_paths)):
        issues.append("genai_config option paths must be unique")
    for option in genai_options:
        path = option.get("path", "<unnamed>")
        if not path.startswith("provider_options.webgpu."):
            issues.append(f"genai_config option {path}: not WebGPU-provider-specific")
        for key in ("area", "requirement", "profile", "currentDefault", "description"):
            if not option.get(key):
                issues.append(f"genai_config option {path}: missing {key}")
        if option.get("requirement") not in {"Mandatory", "Conditional", "Optional"}:
            issues.append(f"genai_config option {path}: invalid requirement")
        history = option.get("history", [])
        dates = [change.get("date", "") for change in history]
        if dates != sorted(dates, reverse=True):
            issues.append(f"genai_config option {path}: history must be newest first")

    important_genai_options = plan.get("importantGenaiConfigOptions", [])
    important_genai_paths = [option.get("path") for option in important_genai_options]
    if len(important_genai_paths) != len(set(important_genai_paths)):
        issues.append("important genai_config option paths must be unique")
    required_important_genai_paths = {
        "provider_options.webgpu.enableGraphCapture",
        "provider_options.webgpu.validationMode",
        "optimization.disable_specified_optimizers",
    }
    if missing_options := required_important_genai_paths - set(important_genai_paths):
        issues.append(
            "missing important genai_config options: "
            + ", ".join(sorted(missing_options))
        )
    for option in important_genai_options:
        path = option.get("path", "<unnamed>")
        for key in ("location", "profile", "description"):
            if not option.get(key):
                issues.append(f"important genai_config option {path}: missing {key}")

    stages = plan.get("stages", [])
    stage_ids = [stage.get("id") for stage in stages]
    if len(stage_ids) != len(set(stage_ids)):
        issues.append("stage ids must be unique")
    model_paths = {
        model["path"] for stage in stages for model in stage.get("models", [])
    }
    expected_model_paths = {
        "model\\aligned\\standard-unfused-rope\\Phi-4-mini-instruct",
        "model\\aligned\\standard-fused-rope\\Phi-4-mini-instruct",
    }
    unexpected_model_paths = {
        path
        for path in model_paths - expected_model_paths
        if not path.startswith("experiments\\")
    }
    if not expected_model_paths.issubset(model_paths) or unexpected_model_paths:
        issues.append(
            "the milestone series must retain both standard RoPE models; "
            "model-changing evidence must remain under experiments"
        )
    rows = {
        (row.get("stageId"), row.get("model")): row
        for row in payload.get("rows", [])
    }
    impacts = optimization_impacts(plan)
    successful = 0
    missing = 0
    for stage in stages:
        if not isinstance(stage.get("milestonePr"), int):
            issues.append(f"{stage.get('id')}: missing exact enabling PR number")
        pull_requests = stage.get("relatedPrs")
        if not isinstance(pull_requests, list) or not pull_requests:
            issues.append(f"{stage.get('id')}: missing explicit required PR list")
            pull_requests = []
        seen_pull_requests = set()
        for pull_request in pull_requests:
            repository = pull_request.get("repository")
            number = pull_request.get("number")
            if repository not in {"onnxruntime", "onnxruntime-genai"}:
                issues.append(
                    f"{stage.get('id')}: invalid PR repository {repository}"
                )
            if not isinstance(number, int) or number <= 0:
                issues.append(f"{stage.get('id')}: invalid PR number {number}")
            if not pull_request.get("role"):
                issues.append(
                    f"{stage.get('id')}: required PR {repository} #{number} has no role"
                )
            merge_date = pull_request.get("mergeDate", "")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", merge_date):
                issues.append(
                    f"{stage.get('id')}: required PR {repository} #{number} has no merge date"
                )
            merge_commit = pull_request.get("mergeCommit", "")
            if not re.fullmatch(r"[0-9a-f]{40}", merge_commit):
                issues.append(
                    f"{stage.get('id')}: required PR {repository} #{number} has no merge commit"
                )
            if not pull_request.get("owner"):
                issues.append(
                    f"{stage.get('id')}: required PR {repository} #{number} has no owner"
                )
            key = (repository, number)
            if key in seen_pull_requests:
                issues.append(
                    f"{stage.get('id')}: duplicate required PR {repository} #{number}"
                )
            seen_pull_requests.add(key)
        primary_repository = stage.get("milestoneRepo", "onnxruntime")
        if (primary_repository, stage.get("milestonePr")) not in seen_pull_requests:
            issues.append(
                f"{stage.get('id')}: primary milestone PR is absent from required PR list"
            )
        if not stage.get("majorContributors"):
            issues.append(f"{stage.get('id')}: missing contributors")
        if not stage.get("applicability"):
            issues.append(f"{stage.get('id')}: missing applicability/conditions")
        if measurement := stage.get("controlledMeasurement"):
            if measurement.get("candidateCommit") and measurement.get(
                "candidateCommit"
            ) != stage.get("milestoneCommit"):
                issues.append(
                    f"{stage.get('id')}: controlled candidate commit does not match milestone"
                )
            for key in ("parent", "candidate"):
                if not isinstance(measurement.get(key), (int, float)) or measurement[key] <= 0:
                    issues.append(
                        f"{stage.get('id')}: invalid controlled measurement {key}"
                    )
        runtime = resolve_plan_path(output_root, stage["runtime"])
        metadata_path = runtime / "build-metadata.json"
        if not metadata_path.is_file():
            issues.append(f"{stage['id']}: missing {metadata_path}")
            metadata = {}
        else:
            metadata = read_json(metadata_path)
        for repository, metadata_key in (
            ("onnxruntime", "ort"),
            ("onnxruntime-genai", "genai"),
        ):
            newest_required_pr = next(
                (
                    pull_request
                    for pull_request in pull_requests
                    if pull_request.get("repository") == repository
                ),
                None,
            )
            if (
                newest_required_pr
                and metadata.get(metadata_key, {}).get("commit")
                != newest_required_pr.get("mergeCommit")
            ):
                issues.append(
                    f"{stage['id']}: tested {repository} commit is not the newest required PR merge commit"
                )
        for binary in ("onnxruntime.dll", "model_benchmark.exe"):
            if not (runtime / binary).is_file():
                issues.append(f"{stage['id']}: missing runtime binary {binary}")
        if verify_hashes:
            for filename, file_metadata in metadata.get("files", {}).items():
                if expected_hash := file_metadata.get("sha256"):
                    check_hash(runtime / filename, expected_hash, stage["id"])
        for model in stage["models"]:
            model_path = resolve_plan_path(output_root, model["path"])
            for model_file in ("model.onnx", "model.onnx.data", "genai_config.json"):
                if not (model_path / model_file).is_file():
                    issues.append(f"{stage['id']}: missing model file {model_path / model_file}")
            config_path = model_path / "genai_config.json"
            if config_path.is_file():
                config_text = config_path.read_text(encoding="utf-8")
                if re.search(r'"enableGraphCapture"\s*:\s*"?1"?', config_text):
                    issues.append(f"{stage['id']}: stored model config enables graph capture")
            model_metadata_path = model_path / "model-metadata.json"
            if not model_metadata_path.is_file():
                issues.append(f"{stage['id']}: missing model metadata {model_metadata_path}")
            else:
                model_metadata = read_json(model_metadata_path)
                for key in ("builderCommit", "builderDate", "sourceRevision"):
                    if model_metadata.get(key) != model.get(key):
                        issues.append(f"{stage['id']}: model metadata {key} does not match plan")
                if model_metadata.get("modelOptions") != stage.get("modelOptions", []):
                    issues.append(f"{stage['id']}: model metadata options do not match plan")
                if verify_hashes:
                    for file_metadata in model_metadata.get("files", []):
                        if expected_hash := file_metadata.get("sha256"):
                            check_hash(
                                model_path / file_metadata["name"],
                                expected_hash,
                                stage["id"],
                            )
            row = rows.get((stage["id"], model["name"]))
            if row is None:
                issues.append(f"{stage['id']}: missing result row for {model['name']}")
                continue
            if row.get("date") != stage.get("date"):
                issues.append(f"{stage['id']}: result date does not match plan")
            if row.get("status") == "success":
                successful += 1
                invocation = row.get("benchmarkInvocation")
                if invocation:
                    if invocation.get("batchSize") != plan.get("batchSize"):
                        issues.append(f"{stage['id']}: benchmark batch size does not match plan")
                    if invocation.get("supportsRequestedMaxLength"):
                        if invocation.get("effectiveMaxLength") != plan.get("maxKvCacheLength"):
                            issues.append(f"{stage['id']}: benchmark max length does not match plan")
                        if invocation.get("maxLengthOption") != "-ml":
                            issues.append(f"{stage['id']}: C++ benchmark must use -ml for max length")
                required_metrics = []
                if "prefill" in impacts.get(stage["id"], []):
                    required_metrics.extend(("ttftMs", "prefillTps"))
                if "decode" in impacts.get(stage["id"], []):
                    required_metrics.append("decodeTps")
                for metric in required_metrics:
                    if not isinstance(row.get(metric), (int, float)) or row[metric] <= 0:
                        issues.append(f"{stage['id']}: missing positive {metric}")
                log = resolve_path(row.get("log", ""), output_root)
                if not log.is_file():
                    issues.append(f"{stage['id']}: missing raw log {log}")
                for repository in ("ort", "genai"):
                    if metadata.get(repository, {}).get("commit") != row.get(repository, {}).get("commit"):
                        issues.append(f"{stage['id']}: {repository} result commit does not match runtime")
            elif row.get("status") == "missing" and row.get("error"):
                missing += 1
            else:
                issues.append(f"{stage['id']}: result status is {row.get('status', 'absent')}")

    archived_plan = output_root / "milestones.json"
    if not archived_plan.is_file() or read_json(archived_plan) != plan:
        issues.append(f"archived plan does not match {DEFAULT_PLAN}")
    if not report_path.is_file():
        issues.append(f"report is missing: {report_path}")
    if issues:
        raise ValueError("verification failed:\n- " + "\n- ".join(issues))
    print(
        f"verified {len(stages)} stages: {successful} successful, {missing} explicitly missing; "
        f"device={device_name}; hashes={'verified' if verify_hashes else 'skipped'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "render", "verify"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", action="append", help="force only selected stage id; repeatable")
    args = parser.parse_args()
    plan = read_json(args.plan)
    output_root = resolve_output_root(args.output_root, args.config)
    report_path = resolve_path(args.report)
    results_path = output_root / "milestone-results.json"
    if args.command == "verify":
        verify(plan, read_json(results_path), output_root, report_path, args.verify_hashes)
        return 0
    write_json(output_root / "milestones.json", plan)
    if args.stage:
        known = {stage["id"] for stage in plan["stages"]}
        unknown = set(args.stage) - known
        if unknown:
            raise ValueError(f"unknown stage(s): {', '.join(sorted(unknown))}")
    if args.command == "run" and args.stage:
        previous = read_json(results_path)
        previous["rows"] = [
            row for row in previous["rows"] if row["stageId"] not in set(args.stage)
        ]
        write_json(results_path, previous)
    payload = (
        run(plan, output_root, force_all_stages(args.force, args.stage))
        if args.command == "run"
        else read_json(results_path)
    )
    render(plan, payload, output_root, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
