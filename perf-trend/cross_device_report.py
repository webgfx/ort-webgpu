"""Validate and render separately measured GPU series in the shared report."""

from __future__ import annotations

import copy
import hashlib
import html
import math
import re

import milestone_benchmark as benchmark


def validate(payload: dict, plan: dict) -> None:
    device = payload["device"]
    for key in ("hostname", "gpu", "vendor", "driver", "adapterId", "powerScheme"):
        if not device.get(key):
            raise ValueError(f"Additional device is missing {key}")
    if device["vendor"] not in {"AMD", "Intel", "NVIDIA"}:
        raise ValueError("Unrecognized GPU vendor")
    after = payload.get("deviceAfter", {})
    for key in ("hostname", "adapterId", "driver", "powerScheme"):
        if after.get(key) != device[key]:
            raise ValueError(f"Device {key} changed or was not checked after collection")
    if device.get("acLineStatus") == 0 or after.get("acLineStatus") != device.get("acLineStatus"):
        raise ValueError("AC power status changed or device was on battery")
    for key in ("batchSize", "promptLength", "generationLength", "maxKvCacheLength", "repetitions", "warmup"):
        if payload["benchmark"].get(key) != plan[key]:
            raise ValueError(f"Additional device workload differs: {key}")
    stages = {stage["id"]: stage for stage in plan["stages"]}
    reference_id = "reference-2024-12-02"
    rows = {row["stageId"]: row for row in payload["rows"]}
    if set(rows) != set(stages) | {reference_id} or len(rows) != len(payload["rows"]):
        raise ValueError("Additional device results must cover each milestone and its own starting measurement exactly once")
    artifacts = {item["path"].replace("\\", "/"): item for item in payload["artifacts"]}
    for stage_id, row in rows.items():
        stage = stages.get(stage_id)
        if stage:
            if row["runtimePath"] != stage["runtime"] or row["modelPath"] != stage["models"][0]["path"]:
                raise ValueError(f"{stage_id}: artifacts differ from the milestone plan")
            if row["graphCapture"] != bool(stage.get("enableGraphCapture")):
                raise ValueError(f"{stage_id}: graph-capture configuration differs")
            for repo, key in (("onnxruntime", "ort"), ("onnxruntime-genai", "genai")):
                required = sorted((pr for pr in stage["relatedPrs"] if pr["repository"] == repo),
                                  key=lambda pr: pr["mergeDate"], reverse=True)
                if required and row[key]["commit"] != required[0]["mergeCommit"]:
                    raise ValueError(f"{stage_id}: {key} differs from required PR merge")
        else:
            reference = plan["optimizationGroups"][0]["reference"]
            if row["date"] != "2024-12-02":
                raise ValueError("Additional device starting measurement must use December 2, 2024")
            for key in ("ort", "genai"):
                if row[key]["commit"] != reference[f"{key}Commit"]:
                    raise ValueError(f"Additional device starting {key} commit differs")
        for directory, filenames in ((row["runtimePath"], ("onnxruntime.dll", "model_benchmark.exe")),
                                     (row["modelPath"], ("model.onnx", "model.onnx.data", "genai_config.json"))):
            for filename in filenames:
                key = directory.replace("\\", "/") + "/" + filename
                if not artifacts.get(key, {}).get("sha256"):
                    raise ValueError(f"Missing artifact hash: {key}")
        output = row.get("rawOutput", "")
        if hashlib.sha256(output.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest() != row.get("rawOutputSha256"):
            raise ValueError(f"{stage_id}: embedded raw log hash mismatch")
        if row["status"] == "failed" and row.get("error"):
            continue
        if row["status"] != "success":
            raise ValueError(f"{stage_id}: incomplete collection")
        parsed = benchmark.parse_output(output)
        for metric in ("prefillTps", "decodeTps", "ttftMs"):
            value = row.get(metric, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or value != parsed.get(metric):
                raise ValueError(f"{stage_id}: metric differs from raw log: {metric}")
        invocation = row["benchmarkInvocation"]
        command = invocation.get("command", "")
        prompt_argument = invocation.get("promptLengthArgument", plan["promptLength"])
        if prompt_argument != plan["promptLength"]:
            if not (row["genai"]["commit"] == "b2a4ecc6039b2f7bf46276f0fc9e0ca7bef5244e"
                    and prompt_argument == plan["promptLength"] - 1 and invocation.get("promptLengthAdjustment")):
                raise ValueError(f"{stage_id}: unsupported prompt-length adjustment")
        if not re.search(rf"(?:^|\s)-l\s+{prompt_argument}(?=\s|$)", command):
            raise ValueError(f"{stage_id}: command does not match promptLength")
        for flag, key in (("b", "batchSize"), ("g", "generationLength"),
                          ("r", "repetitions"), ("w", "warmup")):
            if not re.search(rf"(?:^|\s)-{flag}\s+{plan[key]}(?=\s|$)", command):
                raise ValueError(f"{stage_id}: command does not match {key}")
        header = f"Batch size: {plan['batchSize']}, prompt tokens: {plan['promptLength']}, tokens to generate: {plan['generationLength']}"
        samples = re.search(r"Prompt processing.*?n:\s+(\d+)\s*\*\s*(\d+) token", output, re.DOTALL)
        if header not in output or not samples or tuple(map(int, samples.groups())) != (plan["repetitions"], plan["promptLength"]):
            raise ValueError(f"{stage_id}: raw log workload or measured sample count differs")
        if invocation["batchSize"] != plan["batchSize"] or invocation["requestedMaxKvCacheLength"] != plan["maxKvCacheLength"]:
            raise ValueError(f"{stage_id}: wrong invocation workload")
        expected_length = plan["maxKvCacheLength"] if invocation["supportsRequestedMaxLength"] else plan["promptLength"] + plan["generationLength"]
        if invocation["effectiveMaxLength"] != expected_length:
            raise ValueError(f"{stage_id}: incorrect effective max length")


def charts_plan(plan: dict) -> dict:
    """Never use an NVIDIA starting value or controlled comparison for a new device."""
    result = copy.deepcopy(plan)
    for group in result["optimizationGroups"]:
        group["reference"] = {"stageId": "reference-2024-12-02", "date": "2024-12-02",
                              "label": "December 2, 2024 starting measurement", "chartLabel": "Starting point"}
    for stage in result["stages"]:
        stage.pop("controlledMeasurement", None)
    return result


def render_sections(plan: dict, payloads: list[dict]) -> str:
    sections = []
    for payload in payloads:
        validate(payload, plan)
        device = payload["device"]
        series_plan = charts_plan(plan)
        model = plan["stages"][0]["models"][0]["name"]
        metadata = " · ".join(str(device[key]) for key in ("hostname", "gpu", "driver", "powerScheme"))
        diagnostic_note = ""
        reference = plan["optimizationGroups"][0]["reference"]
        if any(item.get("exitCode") == 0 and item.get("promptLength") == 128
               and item.get("generationLength") == 16
               and item.get("runtime", {}).get("ort", {}).get("commit") == reference["ortCommit"]
               and item.get("runtime", {}).get("genai", {}).get("commit") == reference["genaiCommit"]
               for item in payload.get("diagnostics", [])):
            diagnostic_note = (
                "<p class='muted'>The December runtime also completed a separate 128-prompt/16-generation diagnostic on this host. "
                "Its 1,024-prompt trend run failed with GPU device loss. The shorter diagnostic is excluded from the charts.</p>"
            )
        ordered_rows = [dict(row) for row in sorted(payload["rows"], key=lambda row: row["date"])]
        for row in ordered_rows:
            if row["status"] == "failed" and "DXGI_ERROR_DEVICE_HUNG" in row.get("rawOutput", ""):
                row["error"] = "GPU device hung (DXGI_ERROR_DEVICE_HUNG)"
        ordered_payload = {**payload, "rows": ordered_rows}
        chart_parts = []
        for metric, title in (("prefillTps", "Prefill throughput"), ("decodeTps", "Decode throughput")):
            chart_rows = benchmark.rows_for_chart(series_plan, ordered_payload, model, metric)
            successful = [row for row in chart_rows if row["status"] == "success"]
            gain_note = ""
            if successful:
                first, last = successful[0], successful[-1]
                gain_note = f"Overall gain spans {first['date']} ({first['stage']}) to {last['date']} ({last['stage']}). "
            chart_parts.append(benchmark.svg_chart(
                chart_rows, metric, title, False,
                "Measured on this device. Failed runs remain gaps and are excluded from the gain calculation. "
                + gain_note + "NVIDIA-specific changes may have no effect here; adjacent changes are not isolated PR attribution.",
            ))
        charts = "".join(chart_parts)
        rows = []
        for row in reversed(ordered_rows):
            invocation = row.get("benchmarkInvocation", {})
            length = str(invocation.get("effectiveMaxLength", "Unknown"))
            if not invocation.get("supportsRequestedMaxLength"):
                length += " (8K option unavailable)"
            metrics = [f"{row[metric]:.1f}" if row["status"] == "success" else "N/A"
                       for metric in ("prefillTps", "decodeTps")]
            runtime_files = [item["path"] for item in payload["artifacts"]
                             if item["path"].startswith(row["runtimePath"].replace("\\", "/") + "/")
                             and item["path"].endswith((".dll", ".exe"))]
            model_files = [row["modelPath"].replace("\\", "/") + "/" + name
                           for name in ("model.onnx", "model.onnx.data", "genai_config.json")]
            files = "<br>".join(f"<code>{html.escape(path)}</code>" for path in model_files + runtime_files)
            status = row["status"] + (": " + row["error"] if row.get("error") else "")
            rows.append(
                f"<tr><td>{html.escape(row['date'])}</td><td>{html.escape(row['stage'])}</td>"
                f"<td>{metrics[0]}</td><td>{metrics[1]}</td><td>{html.escape(length)}</td>"
                f"<td>{'Enabled' if row.get('graphCapture') else 'Disabled'}</td>"
                f"<td><code>{html.escape(row['ort']['commit'][:10])}</code></td>"
                f"<td><code>{html.escape(row['genai']['commit'][:10])}</code></td>"
                f"<td><details><summary>Model, binaries and log</summary>{files}<br>"
                f"<code>{html.escape(row['log'])}</code><br>"
                f"<code>{html.escape(invocation.get('command', ''))}</code><br>"
                f"{html.escape(invocation.get('promptLengthAdjustment', ''))}</details></td><td>{html.escape(status)}</td></tr>"
            )
        sections.append(
            f"<section id='{benchmark.device_section_id(device)}'><h2>{html.escape(device['vendor'])} · {html.escape(model)}</h2>"
            f"<p class='muted'>{html.escape(metadata)}</p>{diagnostic_note}{charts}"
            f"<p>Artifact paths are relative to <code>{html.escape(payload['artifactRoot'])}</code> on {html.escape(device['hostname'])}. "
            "Every transferred artifact is SHA-256 verified before execution. Both throughput metrics are retained for each snapshot.</p>"
            "<div class='table'><table class='milestone-table'><thead><tr><th>Date</th><th>Milestone</th>"
            "<th>Prefill tok/s</th><th>Decode tok/s</th><th>Effective max length</th><th>Graph capture</th>"
            "<th>ORT</th><th>ORT GenAI</th><th>Local test artifacts</th><th>Status</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class='muted'>Collected {html.escape(payload['updatedAt'])}</p></section>"
        )
    return "".join(sections)
