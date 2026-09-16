#!/usr/bin/env python3
"""Transfer-manifest and resumable milestone collection for additional Windows GPUs."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess

import milestone_benchmark as benchmark


WORKLOAD_KEYS = (
    "batchSize", "promptLength", "generationLength", "maxKvCacheLength",
    "repetitions", "warmup",
)
VENDOR_IDS = {"AMD": "1002", "Intel": "8086", "NVIDIA": "10DE"}
# This historical benchmark generates N tokens after a one-token seed when -ml
# overrides its stop length. Account for the seed, then verify the emitted count.
SEEDED_PROMPT_COMMIT = "b2a4ecc6039b2f7bf46276f0fc9e0ca7bef5244e"


def archive_path(root: Path, relative: str) -> Path:
    path = (root / Path(relative.replace("\\", "/"))).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes archive: {relative}")
    return path


def prepare(plan: dict, archive_root: Path, destination: Path) -> dict:
    """Hash the exact source artifacts without changing their archived metadata."""
    stages = copy.deepcopy(plan["stages"])
    reference = plan["optimizationGroups"][0]["reference"]
    starting_model = copy.deepcopy(stages[0]["models"][0])
    stages.insert(0, {
        "id": "reference-2024-12-02", "label": "December 2, 2024 starting measurement",
        "chartLabel": "Starting point", "date": "2024-12-02", "isReference": True,
        "runtime": "runtime/2024-12", "models": [starting_model],
        "expectedCommits": {"ort": reference["ortCommit"], "genai": reference["genaiCommit"]},
    })
    files = {}
    for stage in stages:
        stage.pop("controlledMeasurement", None)  # NVIDIA measurements cannot transfer to another GPU.
        runtime = archive_path(archive_root, stage["runtime"])
        metadata = benchmark.read_json(runtime / "build-metadata.json")
        stage["ort"] = metadata["ort"]
        stage["genai"] = metadata["genai"]
        expected_commits = stage.get("expectedCommits", {})
        for repo, key in (("onnxruntime", "ort"), ("onnxruntime-genai", "genai")):
            required = sorted(
                (pr for pr in stage.get("relatedPrs", []) if pr["repository"] == repo),
                key=lambda pr: pr["mergeDate"], reverse=True,
            )
            expected = required[0]["mergeCommit"] if required else expected_commits.get(key)
            if expected and metadata[key]["commit"] != expected:
                raise ValueError(f"{stage['id']}: {key} is not the required commit")
        required_binaries = ["model_benchmark.exe", "onnxruntime.dll"]
        if metadata.get("genaiLinkage") != "static":
            required_binaries.append("onnxruntime-genai.dll")
        for name in required_binaries:
            if not (runtime / name).is_file():
                raise FileNotFoundError(runtime / name)
        directories = [(runtime, metadata.get("files", {}))]
        for model in stage["models"]:
            model_path = archive_path(archive_root, model["path"])
            model_meta = benchmark.read_json(model_path / "model-metadata.json")
            for key in ("builderCommit", "builderDate", "sourceRevision"):
                if model_meta[key] != model[key]:
                    raise ValueError(f"{stage['id']}: model {key} mismatch")
            for name in ("model.onnx", "model.onnx.data", "genai_config.json"):
                if not (model_path / name).is_file():
                    raise FileNotFoundError(model_path / name)
            directories.append((model_path, {entry["name"]: entry for entry in model_meta["files"]}))
        for directory, expected_files in directories:
            for source in sorted(directory.iterdir()):
                if not source.is_file():
                    continue
                relative = source.relative_to(archive_root).as_posix()
                if relative in files:
                    continue
                digest = benchmark.file_sha256(source)
                expected = expected_files.get(source.name, {}).get("sha256")
                if expected and digest.lower() != expected.lower():
                    raise ValueError(f"Archived SHA-256 mismatch: {source}")
                files[relative] = {"path": relative, "bytes": source.stat().st_size, "sha256": digest}
    manifest = {
        "schemaVersion": 1, "createdAt": benchmark.now(),
        "benchmark": {key: plan[key] for key in WORKLOAD_KEYS},
        "stages": stages, "files": list(files.values()),
        "referenceNote": "Each additional GPU uses its own December 2, 2024 measurement for both charts.",
    }
    benchmark.write_json(destination, manifest)
    print(f"Prepared {len(stages)} runs, {len(files)} files, {sum(f['bytes'] for f in files.values()) / 2**30:.2f} GiB", flush=True)
    return manifest


def verify_files(manifest: dict, root: Path) -> None:
    for item in manifest["files"]:
        path = archive_path(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or truncated artifact: {path}")
        if benchmark.file_sha256(path).lower() != item["sha256"].lower():
            raise ValueError(f"Artifact SHA-256 mismatch: {path}")


def probe_device(expected_vendor: str) -> dict:
    script = """
    $ErrorActionPreference = 'Stop'
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
    [pscustomobject]@{
        gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID)
        powerScheme = (& powercfg /getactivescheme | Out-String).Trim()
        cpu = @(Get-CimInstance Win32_Processor | Select-Object Name)
        memoryBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    } | ConvertTo-Json -Depth 5 -Compress
    """
    process = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script], check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    details = json.loads(process.stdout)
    physical = [gpu for gpu in details["gpus"] if gpu["PNPDeviceID"].upper().startswith("PCI\\")]
    if len(physical) != 1:
        raise ValueError(f"Need unambiguous physical GPU selection before benchmarking: {physical}")
    selected = physical[0]
    if f"VEN_{VENDOR_IDS[expected_vendor]}" not in selected["PNPDeviceID"].upper():
        raise ValueError(f"Expected {expected_vendor}, detected {selected['Name']}")
    class PowerStatus(ctypes.Structure):
        _fields_ = [("acLine", ctypes.c_ubyte), ("batteryFlag", ctypes.c_ubyte),
                    ("batteryPercent", ctypes.c_ubyte), ("systemStatus", ctypes.c_ubyte),
                    ("batteryLifeSeconds", ctypes.c_uint32), ("batteryFullLifeSeconds", ctypes.c_uint32)]
    power = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
        raise OSError("Cannot read AC power status")
    if power.acLine == 0:
        raise ValueError("Benchmark device must be connected to AC power")
    return {**details, "hostname": socket.gethostname(), "platform": platform.platform(),
            "vendor": expected_vendor, "gpu": selected["Name"], "driver": selected["DriverVersion"],
            "adapterId": selected["PNPDeviceID"], "selection": "single physical PCI GPU",
            "acLineStatus": int(power.acLine)}


def device_identity(device: dict) -> tuple:
    return tuple(device.get(key) for key in ("hostname", "adapterId", "driver", "powerScheme", "acLineStatus"))


def run(manifest_path: Path, root: Path, results: Path, vendor: str, stage_ids: list[str] | None) -> dict:
    manifest = benchmark.read_json(manifest_path)
    manifest_hash = benchmark.file_sha256(manifest_path)
    device = probe_device(vendor)
    verify_files(manifest, root)
    print(f"Verified all transferred files; GPU={device['gpu']}, driver={device['driver']}", flush=True)
    plan = manifest["benchmark"]
    rows = []
    if results.is_file():
        previous = benchmark.read_json(results)
        if previous["manifestSha256"] != manifest_hash or device_identity(previous["device"]) != device_identity(device):
            raise ValueError("Existing results belong to different artifacts, GPU, driver, host, or power mode; use a new results path")
        rows = previous["rows"]
    known = {stage["id"] for stage in manifest["stages"]}
    if stage_ids and set(stage_ids) - known:
        raise ValueError(f"Unknown stages: {set(stage_ids) - known}")
    payload = {"schemaVersion": 1, "device": device, "benchmark": plan,
               "manifestSha256": manifest_hash, "artifactRoot": str(root),
               "artifacts": manifest["files"], "updatedAt": benchmark.now(), "rows": rows,
               "runnerPid": os.getpid()}
    log_dir = results.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for stage in manifest["stages"]:
        if stage_ids and stage["id"] not in stage_ids:
            continue
        if not stage_ids and any(r["stageId"] == stage["id"] and r["status"] == "success" for r in rows):
            continue
        runtime = archive_path(root, stage["runtime"])
        executable = runtime / "model_benchmark.exe"
        model = stage["models"][0]
        model_path = archive_path(root, model["path"])
        run_id = benchmark.now().replace(":", "-")
        log_path = log_dir / f"{stage['id']}-{run_id}.log"
        row = {"stageId": stage["id"], "stage": stage["label"],
               "date": benchmark.milestone_utc_date(stage), "chartLabel": stage.get("chartLabel", stage["label"]),
               "isReference": stage.get("isReference", False), "model": model["name"],
               "modelPath": model["path"], "runtimePath": stage["runtime"],
               "ort": stage["ort"], "genai": stage["genai"],
               "builderCommit": model["builderCommit"], "sourceRevision": model["sourceRevision"],
               "graphCapture": bool(stage.get("enableGraphCapture")),
               "startedAt": benchmark.now(), "status": "running", "log": str(log_path)}
        rows[:] = [r for r in rows if r["stageId"] != stage["id"]] + [row]
        benchmark.write_json(results, payload)
        print(f"Running {stage['id']}", flush=True)
        output = ""
        try:
            capabilities = benchmark.benchmark_capabilities(executable)
            prompt_argument = plan["promptLength"]
            if stage["genai"]["commit"] == SEEDED_PROMPT_COMMIT and capabilities["maxLength"]:
                prompt_argument -= 1
            with benchmark.model_path_for_stage(model_path, stage, root) as run_model:
                config = benchmark.read_json(run_model / "genai_config.json")
                row["sessionOptions"] = config["model"]["decoder"]["session_options"]
                row["configSha256"] = benchmark.file_sha256(run_model / "genai_config.json")
                command = [str(executable), "-i", str(run_model), "-b", str(plan["batchSize"]),
                           "-l", str(prompt_argument), "-g", str(plan["generationLength"]),
                           "-r", str(plan["repetitions"]), "-w", str(plan["warmup"])]
                if capabilities["maxLength"]:
                    command.extend(["-ml", str(plan["maxKvCacheLength"])])
                row["benchmarkInvocation"] = benchmark.benchmark_metadata(plan, command, capabilities)
                row["benchmarkInvocation"]["promptLengthArgument"] = prompt_argument
                if prompt_argument != plan["promptLength"]:
                    row["benchmarkInvocation"]["promptLengthAdjustment"] = "GeneratePrompt adds the one-token A seed with an explicit -ml; use -l 1023 for 1024 measured prompt tokens."
                process = subprocess.run(command, cwd=runtime, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
                output = process.stdout
                row["exitCode"] = process.returncode
                if process.returncode:
                    raise ValueError(f"Benchmark exited with {process.returncode}")
                metrics = benchmark.parse_output(output)
                if any(not math.isfinite(metrics.get(key, 0)) or metrics.get(key, 0) <= 0
                       for key in ("prefillTps", "decodeTps", "ttftMs")):
                    raise ValueError("Benchmark did not emit positive finite prefill, decode and TTFT measurements")
                row.update(metrics)
                row["status"] = "success"
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            row["status"] = "failed"
            row["error"] = str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                output = error.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
            output += f"\nERROR: {error}\n"
        log_path.write_text(output, encoding="utf-8")
        row["rawOutput"] = output
        row["rawOutputSha256"] = hashlib.sha256(output.replace("\r\n", "\n").encode("utf-8")).hexdigest()
        row["logSha256"] = benchmark.file_sha256(log_path)
        row["finishedAt"] = benchmark.now()
        payload["updatedAt"] = benchmark.now()
        benchmark.write_json(results, payload)
        print(f"{stage['id']}: {row['status']} prefill={row.get('prefillTps')} decode={row.get('decodeTps')}", flush=True)
    final_device = probe_device(vendor)
    if device_identity(final_device) != device_identity(device):
        raise ValueError("GPU, driver, or power scheme changed during collection")
    payload["deviceAfter"] = final_device
    benchmark.write_json(results, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-files", "run", "publish"))
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=benchmark.DEFAULT_PLAN)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vendor", choices=tuple(VENDOR_IDS))
    parser.add_argument("--stage", action="append")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    if args.command == "prepare":
        prepare(benchmark.read_json(args.plan), root, args.manifest)
    elif args.command == "verify-files":
        verify_files(benchmark.read_json(args.manifest), root)
    elif args.command == "publish":
        if not args.results or not args.output:
            parser.error("publish requires --results and --output")
        from cross_device_report import validate
        result = benchmark.read_json(args.results)
        manifest = benchmark.read_json(args.manifest)
        if result["manifestSha256"] != benchmark.file_sha256(args.manifest) or result["artifacts"] != manifest["files"]:
            raise ValueError("Collection does not match the transfer manifest")
        validate(result, benchmark.read_json(args.plan))
        diagnostic_files = sorted((args.results.parent / "diagnostics").glob("*.json"))
        if diagnostic_files:
            result["diagnostics"] = [benchmark.read_json(path) for path in diagnostic_files]
        benchmark.write_json(args.output, result)
        print(f"Published verified device results to {args.output}")
    else:
        if not args.results or not args.vendor:
            parser.error("run requires --results and --vendor")
        result = run(args.manifest, root, args.results, args.vendor, args.stage)
        if any(row["status"] != "success" for row in result["rows"]):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
