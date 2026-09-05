#!/usr/bin/env python3
"""Build, archive, benchmark, and chart monthly ORT WebGPU snapshots."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from typing import Any
import urllib.request
import zipfile
import xml.etree.ElementTree as ET


SKILL_ROOT = Path(__file__).resolve().parent
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
SCHEMA_VERSION = 1
BASE_REQUIRED_BINARIES = ("onnxruntime.dll", "model_benchmark.exe")
OPTIONAL_BINARIES = (
    "onnxruntime-genai.dll",
    "onnxruntime-genai-static.lib",
    "DirectML.dll",
    "D3D12Core.dll",
    "dxcompiler.dll",
    "dxil.dll",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    required_strings = (
        "ortRepo",
        "genaiRepo",
        "ortRef",
        "genaiRef",
        "outputRoot",
        "backupRoot",
        "worktreeRoot",
        "startDate",
        "buildConfig",
        "cmakeGenerator",
        "cmakePath",
        "cmakePolicyVersionMinimum",
        "genaiCxxFlags",
        "runtimeBackend",
        "dawnBackend",
    )
    for key in required_strings:
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in (
        "anchorDay",
        "generationLength",
        "repetitions",
        "warmup",
        "buildTimeoutSeconds",
        "benchmarkTimeoutSeconds",
        "parallelJobs",
    ):
        if not isinstance(config.get(key), int) or config[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"{key} must be a valid integer")
    if config["dawnBackend"] not in ("d3d12", "vulkan"):
        raise ValueError("dawnBackend must be d3d12 or vulkan")
    if config["runtimeBackend"] not in ("webgpu", "dml"):
        raise ValueError("runtimeBackend must be webgpu or dml")
    if not isinstance(config.get("patchLegacyExtensionsMacro"), bool):
        raise ValueError("patchLegacyExtensionsMacro must be boolean")
    if not isinstance(config.get("refreshPinnedArchiveHashes"), bool):
        raise ValueError("refreshPinnedArchiveHashes must be boolean")
    prompts = config.get("promptLengths")
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(value, int) and value > 0 for value in prompts
    ):
        raise ValueError("promptLengths must be a non-empty list of positive integers")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    for model in models:
        if not isinstance(model, dict) or not all(
            isinstance(model.get(key), str) and model[key].strip() for key in ("name", "path")
        ):
            raise ValueError("each model must have non-empty name and path strings")
    config_dir = path.resolve().parent
    for key in ("ortRepo", "genaiRepo", "outputRoot", "backupRoot", "worktreeRoot", "cmakePath"):
        config[key] = expand_config_path(config[key], config_dir, key)
    for model in models:
        model["path"] = expand_config_path(model["path"], config_dir, f"{model['name']} path")
    snapshots = config.get("snapshots")
    if snapshots is not None:
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("snapshots must be a non-empty list when provided")
        keys = []
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or not all(
                isinstance(snapshot.get(key), str) and snapshot[key].strip()
                for key in ("key", "anchorDate", "ortCommit", "genaiCommit")
            ):
                raise ValueError(
                    "each explicit snapshot must have key, anchorDate, ortCommit, and genaiCommit"
                )
            date.fromisoformat(snapshot["anchorDate"])
            if snapshot.get("runtimeBackend", config["runtimeBackend"]) not in ("webgpu", "dml"):
                raise ValueError("snapshot runtimeBackend must be webgpu or dml")
            keys.append(snapshot["key"])
        if len(keys) != len(set(keys)):
            raise ValueError("explicit snapshot keys must be unique")
    date.fromisoformat(config["startDate"])
    return config


def expand_config_path(value: str, config_dir: Path, label: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if re.search(r"\$\{[^}]+\}|%[^%]+%", expanded):
        raise ValueError(f"{label} contains an unresolved environment variable: {value}")
    path = Path(expanded)
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    log_path: Path | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.list2cmdline(command)
    prefix = f"[cwd={cwd}] " if cwd else ""
    print(f"{prefix}{rendered}")
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")

    lines = [f"$ {rendered}", f"cwd: {cwd or Path.cwd()}", ""]
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = process.stdout or ""
        lines.append(output)
        lines.append(f"\nexit_code: {process.returncode}\n")
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") if isinstance(error.stdout, str) else ""
        lines.extend((output, f"\nTIMEOUT after {timeout} seconds\n"))
        if log_path:
            atomic_write_text(log_path, "\n".join(lines))
        raise RuntimeError(f"command timed out after {timeout}s: {rendered}") from error

    if log_path:
        atomic_write_text(log_path, "\n".join(lines))
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {rendered}")
    return process


def git_output(repo: Path, arguments: list[str], timeout: int = 120) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repo}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def monthly_anchors(start: date, through: date, anchor_day: int) -> list[date]:
    anchors: list[date] = []
    year, month = start.year, start.month
    while True:
        day = min(anchor_day, calendar.monthrange(year, month)[1])
        current = date(year, month, day)
        if current > through:
            break
        if current >= start:
            anchors.append(current)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return anchors


def resolve_commit(repo: Path, ref: str, anchor: date) -> dict[str, str]:
    cutoff = f"{anchor.isoformat()}T23:59:59Z"
    commit = git_output(repo, ["rev-list", "-1", f"--before={cutoff}", ref])
    if not commit:
        raise RuntimeError(f"no commit in {repo} at or before {cutoff} from {ref}")
    details = git_output(repo, ["show", "-s", "--format=%H%n%cI%n%s", commit]).splitlines()
    return {
        "commit": details[0],
        "commitDate": details[1],
        "subject": details[2] if len(details) > 2 else "",
    }


def resolve_exact_commit(repo: Path, commit: str) -> dict[str, str]:
    resolved = git_output(repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    details = git_output(repo, ["show", "-s", "--format=%H%n%cI%n%s", resolved]).splitlines()
    return {
        "commit": details[0],
        "commitDate": details[1],
        "subject": details[2] if len(details) > 2 else "",
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        key: config[key]
        for key in (
            "startDate",
            "anchorDay",
            "ortRef",
            "genaiRef",
            "models",
            "promptLengths",
            "generationLength",
            "repetitions",
            "warmup",
            "buildConfig",
            "cmakePath",
            "cmakePolicyVersionMinimum",
            "genaiCxxFlags",
            "dawnBackend",
        )
    }
    relevant["snapshots"] = config.get("snapshots")
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": [model["name"] for model in config["models"]],
        "promptLengths": config["promptLengths"],
        "generationLength": config["generationLength"],
        "repetitions": config["repetitions"],
        "warmup": config["warmup"],
    }


def manifest_path(config: dict[str, Any]) -> Path:
    return Path(config["outputRoot"]) / "manifest.json"


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run plan first")
    manifest = read_json(path)
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema in {path}")
    return manifest


def benchmark_result_covers(config: dict[str, Any], snapshot_key: str) -> bool:
    result_path = (
        Path(config["outputRoot"]) / "results" / snapshot_key / "results.json"
    )
    if not result_path.is_file():
        return False
    payload = read_json(result_path)
    actual_config = payload.get("benchmarkConfig", {})
    for key in ("generationLength", "repetitions", "warmup"):
        if actual_config.get(key) != config[key]:
            return False
    expected = {
        (model["name"], prompt_length)
        for model in config["models"]
        for prompt_length in config["promptLengths"]
    }
    actual = {
        (row.get("model"), row.get("promptLength"))
        for row in payload.get("rows", [])
    }
    return expected.issubset(actual)


def command_doctor(config: dict[str, Any]) -> int:
    issues: list[str] = []
    warnings: list[str] = []
    for executable in ("git",):
        resolved = shutil.which(executable)
        print(f"{executable}: {resolved or 'NOT FOUND'}")
        if not resolved:
            issues.append(f"{executable} is not available")
    cmake_path = Path(config["cmakePath"])
    if cmake_path.is_file():
        process = subprocess.run(
            [str(cmake_path), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        version = process.stdout.splitlines()[0] if process.returncode == 0 else "unavailable"
        print(f"cmake: {cmake_path} ({version})")
        if process.returncode != 0:
            issues.append(f"configured CMake could not run: {cmake_path}")
    else:
        print(f"cmake: NOT FOUND ({cmake_path})")
        issues.append(f"configured CMake does not exist: {cmake_path}")

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if vswhere.is_file():
        generator_match = re.search(r"Visual Studio (\d+)", config["cmakeGenerator"])
        version_arguments = (
            ["-version", f"[{generator_match.group(1)}.0,{int(generator_match.group(1)) + 1}.0)"]
            if generator_match
            else []
        )
        process = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                *version_arguments,
                "-property",
                "installationPath",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        value = process.stdout.strip()
        print(f"{config['cmakeGenerator']}: {value or 'NOT FOUND'}")
        if not value:
            issues.append(f"{config['cmakeGenerator']} C++ tools were not found")
    else:
        warnings.append(f"vswhere not found at {vswhere}; Visual Studio was not verified")

    for label, path_key, ref_key in (
        ("ORT", "ortRepo", "ortRef"),
        ("ORT GenAI", "genaiRepo", "genaiRef"),
    ):
        repo = Path(config[path_key])
        if not (repo / ".git").exists():
            issues.append(f"{label} repository missing: {repo}")
            continue
        shallow = git_output(repo, ["rev-parse", "--is-shallow-repository"])
        status = git_output(repo, ["status", "--short"])
        print(f"{label}: {repo}")
        print(f"  ref: {config[ref_key]}")
        print(f"  shallow: {shallow}")
        print(f"  working tree changes: {'yes (safe; worktrees are used)' if status else 'no'}")
        if shallow == "true":
            issues.append(f"{label} is shallow; historical commits may be missing")
        try:
            git_output(repo, ["rev-parse", "--verify", config[ref_key]])
        except RuntimeError as error:
            issues.append(str(error))

    for model in config["models"]:
        path = Path(model["path"])
        required = ("genai_config.json", "model.onnx")
        missing = [name for name in required if not (path / name).is_file()]
        print(f"model {model['name']}: {path}")
        if missing:
            issues.append(f"{model['name']} missing: {', '.join(missing)}")

    print(f"output root: {config['outputRoot']}")
    print(f"backup root: {config['backupRoot']}")
    print(f"worktree root: {config['worktreeRoot']}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for issue in issues:
        print(f"ERROR: {issue}")
    return 1 if issues else 0


def command_plan(
    config: dict[str, Any], *, fetch: bool, dry_run: bool, through: date | None
) -> dict[str, Any]:
    repos = (Path(config["ortRepo"]), Path(config["genaiRepo"]))
    if fetch:
        for repo in repos:
            run_command(["git", "-C", str(repo), "fetch", "--prune", "origin"], timeout=1800)

    through = through or datetime.now(timezone.utc).date()
    previous: dict[str, Any] = {}
    path = manifest_path(config)
    if path.is_file():
        previous = read_json(path)
    old_by_key = {
        item["key"]: item for item in previous.get("snapshots", []) if isinstance(item, dict)
    }
    fingerprint = config_fingerprint(config)

    snapshots = []
    explicit_snapshots = config.get("snapshots")
    if explicit_snapshots:
        definitions = [
            snapshot
            for snapshot in explicit_snapshots
            if date.fromisoformat(snapshot["anchorDate"]) <= through
        ]
        snapshot_rule = "explicit commit pairs declared in configuration"
    else:
        start = date.fromisoformat(config["startDate"])
        definitions = [
            {"key": anchor.strftime("%Y-%m"), "anchorDate": anchor.isoformat()}
            for anchor in monthly_anchors(start, through, config["anchorDay"])
        ]
        snapshot_rule = "latest commit at or before monthly day 2, 23:59:59 UTC"
    for definition in definitions:
        key = definition["key"]
        anchor = date.fromisoformat(definition["anchorDate"])
        if explicit_snapshots:
            ort = resolve_exact_commit(Path(config["ortRepo"]), definition["ortCommit"])
            genai = resolve_exact_commit(Path(config["genaiRepo"]), definition["genaiCommit"])
        else:
            ort = resolve_commit(Path(config["ortRepo"]), config["ortRef"], anchor)
            genai = resolve_commit(Path(config["genaiRepo"]), config["genaiRef"], anchor)
        old = old_by_key.get(key)
        unchanged = bool(
            old
            and old.get("ort", {}).get("commit") == ort["commit"]
            and old.get("genai", {}).get("commit") == genai["commit"]
        )
        benchmark_reusable = unchanged and benchmark_result_covers(config, key)
        snapshot = {
            "key": key,
            "anchorDate": anchor.isoformat(),
            "ort": ort,
            "genai": genai,
            "build": old.get("build", {"status": "pending"}) if unchanged else {"status": "pending"},
            "benchmark": (
                old.get("benchmark", {"status": "pending"})
                if benchmark_reusable
                else {"status": "pending"}
            ),
        }
        if explicit_snapshots:
            snapshot["runtimeBackend"] = definition.get(
                "runtimeBackend", config["runtimeBackend"]
            )
        else:
            snapshot["cutoffUtc"] = f"{anchor.isoformat()}T23:59:59Z"
        snapshots.append(snapshot)
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": utc_now(),
        "configFingerprint": fingerprint,
        "benchmarkConfig": benchmark_config(config),
        "snapshotRule": snapshot_rule,
        "snapshots": snapshots,
    }
    print(f"planned snapshots: {len(snapshots)}")
    if snapshots:
        print(
            f"range: {snapshots[0]['key']} ({snapshots[0]['ort']['commit'][:8]} / "
            f"{snapshots[0]['genai']['commit'][:8]}) -> "
            f"{snapshots[-1]['key']} ({snapshots[-1]['ort']['commit'][:8]} / "
            f"{snapshots[-1]['genai']['commit'][:8]})"
        )
    if dry_run:
        print(f"dry run: would write {path}")
    else:
        atomic_write_json(path, manifest)
        print(f"wrote {path}")
    return manifest


def is_child(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def remove_tree(path: Path) -> None:
    def make_writable_and_retry(function: Any, failed_path: str, _: Any) -> None:
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    shutil.rmtree(path, onexc=make_writable_and_retry)


def add_worktree(repo: Path, target: Path, commit: str, log_path: Path) -> None:
    if target.exists():
        try:
            if git_output(target, ["rev-parse", "HEAD"]) == commit:
                return
        except RuntimeError:
            pass
        raise RuntimeError(f"worktree path exists at a different revision: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(target), commit],
        timeout=600,
        log_path=log_path,
    )


def remove_worktree(repo: Path, target: Path, worktree_root: Path) -> None:
    if not target.exists():
        return
    if not is_child(target, worktree_root):
        raise RuntimeError(f"refusing to remove worktree outside {worktree_root}: {target}")
    process = subprocess.run(
        ["git", "-C", str(repo), "worktree", "remove", "--force", str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or f"failed to remove {target}")


def locate_file(root: Path, name: str) -> Path | None:
    matches = [
        path
        for path in root.rglob(name)
        if "_deps" not in {part.lower() for part in path.parts}
        and "test" not in path.name.lower()
    ]
    if not matches:
        return None
    matches.sort(key=lambda path: (len(path.parts), -path.stat().st_size, str(path)))
    return matches[0]


def genai_linkage(genai_worktree: Path) -> str:
    cmake_file = genai_worktree / "benchmark" / "c" / "CMakeLists.txt"
    content = cmake_file.read_text(encoding="utf-8", errors="replace")
    return "static" if "onnxruntime-genai-static" in content else "shared"


def required_binaries(linkage: str, runtime_backend: str = "webgpu") -> tuple[str, ...]:
    binaries = list(BASE_REQUIRED_BINARIES)
    if linkage == "shared":
        binaries.append("onnxruntime-genai.dll")
    if runtime_backend == "dml":
        binaries.extend(("DirectML.dll", "D3D12Core.dll"))
    return tuple(binaries)


def stage_ort(
    ort_worktree: Path,
    ort_build: Path,
    destination: Path,
    runtime_backend: str,
) -> None:
    if destination.exists():
        remove_tree(destination)
    (destination / "include").mkdir(parents=True)
    (destination / "lib").mkdir()
    include_roots = (
        ort_worktree / "include" / "onnxruntime" / "core" / "session",
        ort_worktree / "include" / "onnxruntime" / "core" / "providers",
    )
    for include_root in include_roots:
        if not include_root.is_dir():
            continue
        for source in include_root.rglob("*"):
            if source.is_file() and source.suffix in (".h", ".inc"):
                shutil.copy2(source, destination / "include" / source.name)
    for name in ("onnxruntime.dll", "onnxruntime.lib"):
        source = locate_file(ort_build, name)
        if not source:
            raise RuntimeError(f"{name} was not produced below {ort_build}")
        shutil.copy2(source, destination / "lib" / name)
    if runtime_backend == "dml":
        for name in ("DirectML.dll", "D3D12Core.dll"):
            source = locate_file(ort_build, name)
            if not source and name == "D3D12Core.dll":
                system_source = (
                    Path(os.environ.get("SystemRoot", r"C:\Windows"))
                    / "System32"
                    / name
                )
                source = system_source if system_source.is_file() else None
            if not source:
                raise RuntimeError(f"{name} was not produced below {ort_build}")
            shutil.copy2(source, destination / "lib" / name)


def genai_configure_command(
    genai_worktree: Path,
    build_dir: Path,
    ort_home: Path,
    config: dict[str, Any],
    runtime_backend: str,
) -> list[str]:
    source = (genai_worktree / "build.py").read_text(encoding="utf-8", errors="replace")
    command = [
        sys.executable,
        str(genai_worktree / "build.py"),
        "--build_dir",
        str(build_dir),
        "--config",
        config["buildConfig"],
        "--update",
        "--skip_tests",
        "--skip_wheel",
        "--ort_home",
        str(ort_home),
        "--cmake_path",
        config["cmakePath"],
    ]
    if "--skip_examples" in source:
        command.append("--skip_examples")
    if runtime_backend == "dml":
        if "--use_dml" not in source:
            raise RuntimeError("selected GenAI snapshot does not support --use_dml")
        command.append("--use_dml")
        command.extend(
            [
                "--cmake_extra_defines",
                f"CMAKE_POLICY_VERSION_MINIMUM={config['cmakePolicyVersionMinimum']}",
                f"CMAKE_CXX_FLAGS={config['genaiCxxFlags']}",
            ]
        )
    elif "--use_webgpu" in source:
        command.append("--use_webgpu")
        command.extend(
            [
                "--cmake_extra_defines",
                f"CMAKE_POLICY_VERSION_MINIMUM={config['cmakePolicyVersionMinimum']}",
                f"CMAKE_CXX_FLAGS={config['genaiCxxFlags']}",
            ]
        )
    else:
        command.extend(
            [
                "--cmake_extra_defines",
                "USE_WEBGPU=ON",
                f"CMAKE_POLICY_VERSION_MINIMUM={config['cmakePolicyVersionMinimum']}",
                f"CMAKE_CXX_FLAGS={config['genaiCxxFlags']}",
            ]
        )
    return command


def resolve_genai_cmake_root(build_dir: Path, build_config: str) -> Path:
    candidates = (build_dir / build_config, build_dir)
    for candidate in candidates:
        if (candidate / "CMakeCache.txt").is_file():
            return candidate
    raise RuntimeError(f"GenAI CMake cache was not produced below {build_dir}")


def patch_legacy_extensions_macro(genai_build: Path) -> list[dict[str, str]]:
    sources = [
        path
        for path in genai_build.rglob("image_processor.cc")
        if "onnxruntime_extensions-src" in path.parts
        and path.parent.name == "api"
        and path.parent.parent.name == "shared"
    ]
    if not sources:
        return []
    source = sources[0]
    original = source.read_text(encoding="utf-8")
    replacements = {
        "#if OCOS_ENABLE_VENDOR_IMAGE_CODECS": "#if defined(OCOS_ENABLE_VENDOR_IMAGE_CODECS)",
        "#if WIN32": "#if defined(WIN32)",
    }
    applicable = [(old, new) for old, new in replacements.items() if old in original]
    if not applicable:
        return []
    updated = original
    for old, new in applicable:
        updated = updated.replace(old, new)
    source.write_text(updated, encoding="utf-8", newline="\n")
    return [
        {
            "file": str(source),
            "change": f"{old} -> {new}",
            "reason": "current MSVC rejects a valueless macro in #if; semantics are unchanged",
        }
        for old, new in applicable
    ]


def patch_legacy_ort_headers(ort_worktree: Path) -> list[dict[str, str]]:
    header = ort_worktree / "include" / "onnxruntime" / "core" / "platform" / "ort_mutex.h"
    if not header.is_file():
        return []
    content = header.read_text(encoding="utf-8")
    if "#include <chrono>" in content:
        return []
    marker = "#include <Windows.h>\n"
    if marker not in content:
        return []
    header.write_text(
        content.replace(marker, marker + "#include <chrono>\n", 1),
        encoding="utf-8",
        newline="\n",
    )
    return [
        {
            "file": str(header),
            "change": "added direct #include <chrono>",
            "reason": "current MSVC no longer supplies the historical transitive chrono include",
        }
    ]


def patch_pinned_archive_hashes(
    ort_worktree: Path, cache_root: Path
) -> list[dict[str, str]]:
    deps_path = ort_worktree / "cmake" / "deps.txt"
    if not deps_path.is_file():
        return []
    archive_pattern = re.compile(
        r"^(eigen3?);"
        r"(https://gitlab\.com/libeigen/eigen/-/archive/"
        r"([0-9a-f]{40})/[^;]+\.zip);([0-9a-f]{40})$"
    )
    lines = deps_path.read_text(encoding="utf-8").splitlines()
    adjustments: list[dict[str, str]] = []
    updated_lines = []
    cache_root.mkdir(parents=True, exist_ok=True)
    for line in lines:
        match = archive_pattern.match(line.strip())
        if not match:
            updated_lines.append(line)
            continue
        dependency, url, commit, expected = match.groups()
        archive = cache_root / f"eigen-{commit}.zip"
        if not archive.is_file():
            print(f"downloading pinned archive for verification: {url}")
            urllib.request.urlretrieve(url, archive)
        actual = hashlib.sha1(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
        if not names or commit not in names[0]:
            raise RuntimeError(
                f"downloaded Eigen archive does not identify pinned commit {commit}: {archive}"
            )
        if actual == expected:
            updated_lines.append(line)
            continue
        updated_lines.append(f"{dependency};{url};{actual}")
        adjustments.append(
            {
                "file": str(deps_path),
                "change": f"{dependency} SHA1 {expected} -> {actual}",
                "reason": (
                    "GitLab regenerated the archive bytes; the archive root still "
                    f"identifies pinned commit {commit}"
                ),
            }
        )
    if adjustments:
        deps_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8", newline="\n")
    return adjustments


def prepare_dml_packages(
    ort_worktree: Path, cache_root: Path
) -> list[dict[str, str]]:
    packages_config = ort_worktree / "packages.config"
    if not packages_config.is_file():
        raise RuntimeError(f"DML packages config is missing: {packages_config}")
    document = ET.parse(packages_config)
    packages = [
        (element.attrib.get("id"), element.attrib.get("version"))
        for element in document.getroot().findall("package")
    ]
    if not any(package_id == "Microsoft.AI.DirectML" for package_id, _ in packages):
        raise RuntimeError("Microsoft.AI.DirectML version is missing from packages.config")
    cache_root.mkdir(parents=True, exist_ok=True)
    adjustments = []
    for package_id, version in packages:
        if not package_id or not version:
            raise RuntimeError("NuGet package id/version is missing from packages.config")
        normalized_id = package_id.lower()
        archive = cache_root / f"{normalized_id}.{version}.nupkg"
        if not archive.is_file():
            url = (
                f"https://api.nuget.org/v3-flatcontainer/{normalized_id}/{version}/"
                f"{normalized_id}.{version}.nupkg"
            )
            print(f"downloading NuGet package: {url}")
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            if not bundle.namelist():
                raise RuntimeError(f"NuGet package is empty: {archive}")
        adjustments.append(
            {
                "file": str(archive),
                "change": f"cached {package_id} {version}",
                "reason": "historical NuGet 5.3 cannot negotiate TLS with nuget.org",
            }
        )
    nuget_config = ort_worktree / "NuGet.config"
    nuget_config.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<configuration>\n"
        "  <packageSources>\n"
        "    <clear />\n"
        f'    <add key="Local milestone cache" value="{cache_root}" />\n'
        "  </packageSources>\n"
        "</configuration>\n",
        encoding="utf-8",
        newline="\n",
    )
    adjustments.append(
        {
            "file": str(nuget_config),
            "change": f"set package source to {cache_root}",
            "reason": "restore pinned packages without contacting nuget.org",
        }
    )
    return adjustments


def device_fingerprint() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    if os.name != "nt":
        return result
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    if process.returncode == 0 and process.stdout.strip():
        try:
            value = json.loads(process.stdout)
            result["gpus"] = value if isinstance(value, list) else [value]
        except json.JSONDecodeError:
            result["gpuDetectionError"] = "PowerShell returned invalid JSON"
    else:
        result["gpuDetectionError"] = process.stderr.strip() or "GPU detection failed"
    return result


def update_snapshot(manifest: dict[str, Any], snapshot: dict[str, Any], field: str, value: dict[str, Any], path: Path) -> None:
    snapshot[field] = value
    manifest["updatedAt"] = utc_now()
    atomic_write_json(path, manifest)


def build_snapshot(
    config: dict[str, Any],
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    keep_worktrees: bool,
) -> None:
    output_root = Path(config["outputRoot"])
    runtime_backend = snapshot.get("runtimeBackend", config["runtimeBackend"])
    key = snapshot["key"]
    worktrees_parent = Path(config["worktreeRoot"])
    worktree_root = worktrees_parent / key
    ort_worktree = worktree_root / "onnxruntime"
    genai_worktree = worktree_root / "onnxruntime-genai"
    logs = output_root / "logs" / key
    logs.mkdir(parents=True, exist_ok=True)
    path = manifest_path(config)
    started = utc_now()
    update_snapshot(manifest, snapshot, "build", {"status": "running", "startedAt": started}, path)
    commands: list[list[str]] = []
    compatibility_adjustments: list[dict[str, str]] = []
    success = False
    try:
        add_worktree(
            Path(config["ortRepo"]),
            ort_worktree,
            snapshot["ort"]["commit"],
            logs / "01-ort-worktree.log",
        )
        add_worktree(
            Path(config["genaiRepo"]),
            genai_worktree,
            snapshot["genai"]["commit"],
            logs / "02-genai-worktree.log",
        )
        ort_build = ort_worktree / "build" / "Windows"
        genai_build = genai_worktree / "build" / "Windows"
        staging = output_root / "staging" / key / "ort"
        for build_dir in (ort_build, genai_build):
            if build_dir.exists():
                if not is_child(build_dir, worktree_root):
                    raise RuntimeError(
                        f"refusing to clear build directory outside {worktree_root}: {build_dir}"
                    )
                remove_tree(build_dir)
        compatibility_adjustments.extend(patch_legacy_ort_headers(ort_worktree))
        if config["refreshPinnedArchiveHashes"]:
            compatibility_adjustments.extend(
                patch_pinned_archive_hashes(
                    ort_worktree, output_root / "download-cache"
                )
            )
        if runtime_backend == "dml":
            compatibility_adjustments.extend(
                prepare_dml_packages(
                    ort_worktree, output_root / "download-cache" / "nuget"
                )
            )
        if runtime_backend == "webgpu":
            backend_defines = (
                [
                    "onnxruntime_ENABLE_DAWN_BACKEND_D3D12=ON",
                    "onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=OFF",
                ]
                if config["dawnBackend"] == "d3d12"
                else [
                    "onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF",
                    "onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON",
                    "DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON",
                ]
            )
        else:
            backend_defines = []
        backend_defines.append(
            f"CMAKE_POLICY_VERSION_MINIMUM={config['cmakePolicyVersionMinimum']}"
        )
        ort_command = [
            sys.executable,
            str(ort_worktree / "tools" / "ci_build" / "build.py"),
            "--build_dir",
            str(ort_build),
            "--config",
            config["buildConfig"],
            "--update",
            "--build",
            "--parallel",
            str(config["parallelJobs"]),
            "--skip_tests",
            "--build_shared_lib",
            "--compile_no_warning_as_error",
            "--cmake_generator",
            config["cmakeGenerator"],
            "--cmake_path",
            config["cmakePath"],
            "--cmake_extra_defines",
            *backend_defines,
        ]
        ort_command.insert(
            ort_command.index("--compile_no_warning_as_error"),
            "--use_webgpu" if runtime_backend == "webgpu" else "--use_dml",
        )
        commands.append(ort_command)
        run_command(
            ort_command,
            cwd=ort_worktree,
            timeout=config["buildTimeoutSeconds"],
            log_path=logs / "03-build-ort.log",
        )
        stage_ort(ort_worktree, ort_build, staging, runtime_backend)

        genai_command = genai_configure_command(
            genai_worktree, genai_build, staging, config, runtime_backend
        )
        commands.append(genai_command)
        run_command(
            genai_command,
            cwd=genai_worktree,
            timeout=config["buildTimeoutSeconds"],
            log_path=logs / "04-configure-genai.log",
        )
        if config["patchLegacyExtensionsMacro"]:
            compatibility_adjustments.extend(patch_legacy_extensions_macro(genai_build))
        genai_cmake_root = resolve_genai_cmake_root(
            genai_build, config["buildConfig"]
        )
        benchmark_command = [
            config["cmakePath"],
            "--build",
            str(genai_cmake_root),
            "--config",
            config["buildConfig"],
            "--parallel",
            str(config["parallelJobs"]),
            "--target",
            "model_benchmark",
        ]
        commands.append(benchmark_command)
        run_command(
            benchmark_command,
            cwd=genai_worktree,
            timeout=config["buildTimeoutSeconds"],
            log_path=logs / "05-build-model-benchmark.log",
        )

        backup = Path(config["backupRoot"]) / key
        backup.mkdir(parents=True, exist_ok=True)
        linkage = genai_linkage(genai_worktree)
        required = required_binaries(linkage, runtime_backend)
        produced = {}
        for name in dict.fromkeys((*required, *OPTIONAL_BINARIES)):
            roots = (ort_build, genai_build, staging)
            source = next((found for root in roots if (found := locate_file(root, name))), None)
            if source:
                shutil.copy2(source, backup / name)
                produced[name] = {
                    "source": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            elif name in required:
                raise RuntimeError(f"required binary not produced: {name}")

        metadata = {
            "schemaVersion": SCHEMA_VERSION,
            "snapshot": key,
            "anchorDate": snapshot["anchorDate"],
            "ort": snapshot["ort"],
            "genai": snapshot["genai"],
            "buildConfig": config["buildConfig"],
            "dawnBackend": config["dawnBackend"],
            "runtimeBackend": runtime_backend,
            "genaiLinkage": linkage,
            "builtAt": utc_now(),
            "host": device_fingerprint(),
            "commands": [subprocess.list2cmdline(command) for command in commands],
            "compatibilityAdjustments": compatibility_adjustments,
            "files": produced,
        }
        atomic_write_json(backup / "build-metadata.json", metadata)
        update_snapshot(
            manifest,
            snapshot,
            "build",
            {
                "status": "success",
                "startedAt": started,
                "finishedAt": utc_now(),
                "backup": str(backup),
            },
            path,
        )
        success = True
    except Exception as error:
        update_snapshot(
            manifest,
            snapshot,
            "build",
            {
                "status": "failed",
                "startedAt": started,
                "finishedAt": utc_now(),
                "error": str(error),
            },
            path,
        )
        update_snapshot(
            manifest,
            snapshot,
            "benchmark",
            {
                "status": "blocked",
                "finishedAt": utc_now(),
                "reason": "build failed",
            },
            path,
        )
        raise
    finally:
        if success or not keep_worktrees:
            errors = []
            for repo, worktree in (
                (Path(config["genaiRepo"]), genai_worktree),
                (Path(config["ortRepo"]), ort_worktree),
            ):
                try:
                    remove_worktree(repo, worktree, worktrees_parent)
                except Exception as error:
                    errors.append(str(error))
            if errors:
                print("WARNING: worktree cleanup: " + "; ".join(errors), file=sys.stderr)


def parse_benchmark_output(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    patterns = (
        ("ttftMs", r"Prompt processing.*?avg \(us\):\s+([\d.e+\-]+)", 0.001),
        ("prefillTps", r"Prompt processing.*?avg \(tokens/s\):\s+([\d.e+\-]+)", 1.0),
        ("decodeTps", r"Token generation:.*?avg \(tokens/s\):\s+([\d.e+\-]+)", 1.0),
        ("e2eMs", r"E2E generation.*?avg \(ms\):\s+([\d.e+\-]+)", 1.0),
    )
    for name, pattern, factor in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            metrics[name] = round(float(match.group(1)) * factor, 3)
    memory = re.search(r"Peak working set size \(bytes\):\s+(\d+)", text, re.IGNORECASE)
    if memory:
        metrics["peakMemoryBytes"] = int(memory.group(1))
    return metrics


def benchmark_snapshot(
    config: dict[str, Any], manifest: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    if snapshot.get("build", {}).get("status") != "success":
        raise RuntimeError(f"{snapshot['key']} has no successful build")
    output_root = Path(config["outputRoot"])
    key = snapshot["key"]
    backup = Path(config["backupRoot"]) / key
    metadata_path = backup / "build-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"{metadata_path} is missing")
    metadata = read_json(metadata_path)
    for name in required_binaries(
        metadata.get("genaiLinkage", "shared"),
        metadata.get("runtimeBackend", "webgpu"),
    ):
        if not (backup / name).is_file():
            raise RuntimeError(f"{backup / name} is missing")
    result_dir = output_root / "results" / key
    result_dir.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs" / key
    started = utc_now()
    path = manifest_path(config)
    update_snapshot(
        manifest, snapshot, "benchmark", {"status": "running", "startedAt": started}, path
    )
    rows = []
    try:
        for model in config["models"]:
            for prompt_length in config["promptLengths"]:
                log_path = logs / f"benchmark-{model['name']}-pl{prompt_length}.log"
                command = [
                    str(backup / "model_benchmark.exe"),
                    "-i",
                    model["path"],
                    "-l",
                    str(prompt_length),
                    "-g",
                    str(config["generationLength"]),
                    "-r",
                    str(config["repetitions"]),
                    "-w",
                    str(config["warmup"]),
                ]
                row: dict[str, Any] = {
                    "snapshot": key,
                    "model": model["name"],
                    "modelPath": model["path"],
                    "promptLength": prompt_length,
                    "generationLength": config["generationLength"],
                    "repetitions": config["repetitions"],
                    "warmup": config["warmup"],
                    "log": str(log_path),
                }
                try:
                    process = run_command(
                        command,
                        cwd=backup,
                        timeout=config["benchmarkTimeoutSeconds"],
                        log_path=log_path,
                    )
                    metrics = parse_benchmark_output(process.stdout or "")
                    if "prefillTps" not in metrics or "decodeTps" not in metrics:
                        raise RuntimeError("benchmark output did not contain both TPS metrics")
                    row.update({"status": "success", **metrics})
                except Exception as error:
                    row.update({"status": "failed", "error": str(error)})
                rows.append(row)
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "snapshot": key,
            "anchorDate": snapshot["anchorDate"],
            "ort": snapshot["ort"],
            "genai": snapshot["genai"],
            "device": device_fingerprint(),
            "benchmarkConfig": {
                "promptLengths": config["promptLengths"],
                "generationLength": config["generationLength"],
                "repetitions": config["repetitions"],
                "warmup": config["warmup"],
            },
            "completedAt": utc_now(),
            "rows": rows,
        }
        atomic_write_json(result_dir / "results.json", payload)
        successful = sum(row["status"] == "success" for row in rows)
        status = "success" if successful == len(rows) else ("partial" if successful else "failed")
        update_snapshot(
            manifest,
            snapshot,
            "benchmark",
            {
                "status": status,
                "startedAt": started,
                "finishedAt": utc_now(),
                "successfulRows": successful,
                "totalRows": len(rows),
                "results": str(result_dir / "results.json"),
            },
            path,
        )
    except Exception as error:
        update_snapshot(
            manifest,
            snapshot,
            "benchmark",
            {
                "status": "failed",
                "startedAt": started,
                "finishedAt": utc_now(),
                "error": str(error),
            },
            path,
        )
        raise


def select_snapshots(
    manifest: dict[str, Any],
    *,
    snapshot_key: str | None,
    all_pending: bool,
    retry_failed: bool,
    operation: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    snapshots = manifest["snapshots"]
    if snapshot_key:
        selected = [item for item in snapshots if item["key"] == snapshot_key]
        if not selected:
            raise ValueError(f"snapshot not found: {snapshot_key}")
        return selected
    if not all_pending:
        raise ValueError("specify --snapshot YYYY-MM or --all-pending")
    statuses = {"pending"}
    if retry_failed:
        statuses.update(("failed", "partial"))
    field = "build" if operation == "build" else "benchmark"
    selected = [item for item in snapshots if item.get(field, {}).get("status", "pending") in statuses]
    return selected[:limit] if limit else selected


def select_run_snapshots(
    manifest: dict[str, Any],
    *,
    snapshot_key: str | None,
    retry_failed: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    if snapshot_key:
        selected = [
            item for item in manifest["snapshots"] if item["key"] == snapshot_key
        ]
        if not selected:
            raise ValueError(f"snapshot not found: {snapshot_key}")
        return selected

    build_statuses = {"pending"}
    benchmark_statuses = {"pending"}
    if retry_failed:
        build_statuses.add("failed")
        benchmark_statuses.update(("failed", "partial"))
    selected = [
        item
        for item in manifest["snapshots"]
        if item.get("build", {}).get("status", "pending") in build_statuses
        or (
            item.get("build", {}).get("status") == "success"
            and item.get("benchmark", {}).get("status", "pending")
            in benchmark_statuses
        )
    ]
    return selected[:limit] if limit else selected


def chart_svg(points: list[tuple[str, float]], title: str, color: str) -> str:
    width, height = 540, 230
    left, right, top, bottom = 52, 18, 28, 42
    if not points:
        return (
            f'<div class="chart"><h3>{html.escape(title)}</h3>'
            '<div class="empty">No successful points</div></div>'
        )
    values = [value for _, value in points]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.12, high * 0.03, 1.0)
    low = max(0.0, low - padding)
    high += padding
    span = max(high - low, 1.0)
    x_span = width - left - right
    y_span = height - top - bottom
    coordinates = []
    for index, (_, value) in enumerate(points):
        x = left + (x_span * index / max(len(points) - 1, 1))
        y = top + y_span * (high - value) / span
        coordinates.append((x, y))
    segments: list[list[tuple[float, float]]] = []
    current_segment: list[tuple[float, float]] = []
    previous_month: int | None = None
    for (label, _), coordinate in zip(points, coordinates):
        match = re.fullmatch(r"(\d{4})-(\d{2})", label)
        month_index = (
            int(match.group(1)) * 12 + int(match.group(2)) if match else None
        )
        if (
            previous_month is not None
            and month_index is not None
            and month_index != previous_month + 1
        ):
            segments.append(current_segment)
            current_segment = []
        current_segment.append(coordinate)
        previous_month = month_index
    if current_segment:
        segments.append(current_segment)
    polylines = "".join(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in segment)}" '
        f'style="stroke:{color}"/>'
        for segment in segments
        if len(segment) > 1
    )
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"><title>{html.escape(label)}: {value:.2f}</title></circle>'
        for (label, value), (x, y) in zip(points, coordinates)
    )
    labels = "".join(
        f'<text x="{coordinates[index][0]:.1f}" y="{height - 13}" text-anchor="middle">'
        f"{html.escape(label[2:] if re.fullmatch(r'\d{4}-\d{2}', label) else label)}</text>"
        for index, (label, _) in enumerate(points)
        if len(points) <= 12 or index % 2 == 0
    )
    return f"""
    <div class="chart">
      <h3>{html.escape(title)}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>
        <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
        <text x="8" y="{top+5}">{high:.1f}</text>
        <text x="8" y="{height-bottom+5}">{low:.1f}</text>
        {polylines}
        <g style="fill:{color}">{circles}</g>
        {labels}
      </svg>
    </div>
    """


def render_report(config: dict[str, Any], manifest: dict[str, Any]) -> Path:
    output_root = Path(config["outputRoot"])
    payloads: dict[str, dict[str, Any]] = {}
    for snapshot in manifest["snapshots"]:
        result_path = output_root / "results" / snapshot["key"] / "results.json"
        if result_path.is_file():
            payloads[snapshot["key"]] = read_json(result_path)

    summary = {"build_success": 0, "benchmark_success": 0, "gaps": 0}
    timeline_rows = []
    for snapshot in manifest["snapshots"]:
        build_status = snapshot.get("build", {}).get("status", "pending")
        benchmark_status = snapshot.get("benchmark", {}).get("status", "pending")
        if build_status == "failed" and benchmark_status == "pending":
            benchmark_status = "blocked"
        ort_timestamp = str(snapshot["ort"].get("commitDate", ""))
        genai_timestamp = str(snapshot["genai"].get("commitDate", ""))
        ort_date = ort_timestamp[:10] or "unknown"
        genai_date = genai_timestamp[:10] or "unknown"
        summary["build_success"] += build_status == "success"
        summary["benchmark_success"] += benchmark_status == "success"
        summary["gaps"] += benchmark_status not in ("success",)
        timeline_rows.append(
            "<tr>"
            f"<td>{snapshot['key']}</td>"
            f'<td title="{html.escape(ort_timestamp)}">{html.escape(ort_date)}</td>'
            f"<td><code>{snapshot['ort']['commit'][:10]}</code></td>"
            f'<td title="{html.escape(genai_timestamp)}">{html.escape(genai_date)}</td>'
            f"<td><code>{snapshot['genai']['commit'][:10]}</code></td>"
            f'<td><span class="pill {build_status}">{html.escape(build_status)}</span></td>'
            f'<td><span class="pill {benchmark_status}">{html.escape(benchmark_status)}</span></td>'
            "</tr>"
        )

    model_sections = []
    raw_rows = []
    colors = {"prefillTps": "#2563eb", "decodeTps": "#0f766e"}
    for model in config["models"]:
        charts = []
        for prompt_length in config["promptLengths"]:
            for metric, label in (("prefillTps", "Prefill TPS"), ("decodeTps", "Decode TPS")):
                points = []
                for snapshot in manifest["snapshots"]:
                    payload = payloads.get(snapshot["key"])
                    if not payload:
                        continue
                    row = next(
                        (
                            item
                            for item in payload.get("rows", [])
                            if item.get("model") == model["name"]
                            and item.get("promptLength") == prompt_length
                            and item.get("status") == "success"
                            and metric in item
                        ),
                        None,
                    )
                    if row:
                        points.append((snapshot["key"], float(row[metric])))
                charts.append(
                    chart_svg(
                        points,
                        f"{label} · prompt {prompt_length}",
                        colors[metric],
                    )
                )
        model_sections.append(
            f'<section><h2>{html.escape(model["name"])}</h2>'
            f'<div class="charts">{"".join(charts)}</div></section>'
        )

    for snapshot in manifest["snapshots"]:
        payload = payloads.get(snapshot["key"])
        if not payload:
            continue
        for row in payload.get("rows", []):
            if row.get("promptLength") not in config["promptLengths"]:
                continue
            raw_rows.append(
                "<tr>"
                f"<td>{snapshot['key']}</td><td>{html.escape(str(row.get('model', '')))}</td>"
                f"<td>{row.get('promptLength', '')}</td><td>{row.get('prefillTps', '')}</td>"
                f"<td>{row.get('decodeTps', '')}</td><td>{row.get('ttftMs', '')}</td>"
                f"<td>{html.escape(str(row.get('status', '')))}</td>"
                f"<td>{html.escape(str(row.get('error', '')))}</td>"
                "</tr>"
            )

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORT WebGPU Performance Trend</title>
<style>
:root{{--ink:#172033;--muted:#657184;--line:#d9e1ec;--bg:#f4f7fb;--panel:#fff;--blue:#2563eb;--teal:#0f766e;--red:#b42318;--amber:#a15c00}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#e8f1ff,#f4f7fb 32rem);color:var(--ink);font:15px/1.55 "Segoe UI",sans-serif}}
.wrap{{width:min(1240px,calc(100% - 32px));margin:auto}}header{{padding:60px 0 28px}}h1{{font-size:clamp(38px,6vw,72px);line-height:1;margin:8px 0 18px;letter-spacing:-.045em}}h2{{font-size:30px;margin:0 0 18px}}h3{{font-size:16px;margin:0 0 10px}}.eyebrow{{color:var(--blue);font-weight:800;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}
.lead{{max-width:900px;color:var(--muted);font-size:19px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,.chart,section{{background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 34px rgba(40,65,100,.08)}}.card{{padding:22px}}.big{{font-size:42px;font-weight:800;color:var(--blue)}}section{{padding:26px;margin:18px 0}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.chart{{padding:18px;box-shadow:none}}svg{{width:100%;height:auto}}svg text{{fill:var(--muted);font-size:11px}}.axis{{stroke:var(--line)}}polyline{{fill:none;stroke-width:3}}.empty{{height:190px;display:grid;place-items:center;color:var(--muted)}}table{{width:100%;border-collapse:collapse;min-width:780px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}}th{{font-size:11px;text-transform:uppercase;color:var(--muted)}}.table{{overflow:auto}}.pill{{display:inline-block;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;background:#eef2f7}}.pill.success{{background:#dcfce7;color:#166534}}.pill.failed{{background:#fee2e2;color:var(--red)}}.pill.blocked{{background:#e5e7eb;color:#4b5563}}.pill.partial{{background:#fef3c7;color:var(--amber)}}code{{font-family:Consolas,monospace}}footer{{padding:30px;color:var(--muted);text-align:center}}@media(max-width:850px){{.charts,.cards{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="wrap"><div class="eyebrow">Historical source snapshots · WebGPU</div><h1>ORT WebGPU Performance Trend</h1><p class="lead">Monthly ONNX Runtime + ONNX Runtime GenAI snapshots, benchmarked with fixed Phi model exports and workload settings. Missing or incompatible snapshots remain gaps, never zeroes.</p>
<div class="cards"><div class="card"><div class="big">{len(manifest['snapshots'])}</div>planned months</div><div class="card"><div class="big">{summary['benchmark_success']}</div>complete months</div><div class="card"><div class="big">{summary['gaps']}</div>pending / failed gaps</div></div></div></header>
<main class="wrap"><section><h2>Trend workload</h2><p>Prompt length: <strong>{", ".join(str(value) for value in config["promptLengths"])}</strong> tokens · Generation length: <strong>{config["generationLength"]}</strong> tokens · <strong>{config["repetitions"]}</strong> measured repetitions after <strong>{config["warmup"]}</strong> warmup.</p></section>{''.join(model_sections)}
<section><h2>Snapshot timeline</h2><p>Dates are commit dates recorded by each repository; hover a date for the full timestamp and timezone.</p><div class="table"><table><thead><tr><th>Month</th><th>ORT date</th><th>ORT commit</th><th>ORT GenAI date</th><th>ORT GenAI commit</th><th>Build</th><th>Benchmark</th></tr></thead><tbody>{''.join(timeline_rows)}</tbody></table></div></section>
<section><h2>Raw measurements</h2><div class="table"><table><thead><tr><th>Month</th><th>Model</th><th>Prompt</th><th>Prefill TPS</th><th>Decode TPS</th><th>TTFT ms</th><th>Status</th><th>Error</th></tr></thead><tbody>{''.join(raw_rows) or '<tr><td colspan="8">No benchmark results yet.</td></tr>'}</tbody></table></div></section>
</main><footer>Generated {html.escape(utc_now())} · Config fingerprint {html.escape(manifest.get('configFingerprint','')[:12])}</footer></body></html>"""
    destination = output_root / "index.html"
    atomic_write_text(destination, report)
    print(f"wrote {destination}")
    return destination


def command_operation(config: dict[str, Any], args: argparse.Namespace, operation: str) -> None:
    manifest = load_manifest(config)
    selected = select_snapshots(
        manifest,
        snapshot_key=args.snapshot,
        all_pending=args.all_pending,
        retry_failed=args.retry_failed,
        operation=operation,
        limit=args.limit,
    )
    if not selected:
        print("no matching snapshots")
        return
    for snapshot in selected:
        print(f"=== {operation} {snapshot['key']} ===")
        try:
            if operation == "build":
                build_snapshot(
                    config, manifest, snapshot, keep_worktrees=args.keep_worktrees
                )
            else:
                benchmark_snapshot(config, manifest, snapshot)
        except Exception as error:
            print(f"ERROR {snapshot['key']}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    render_report(config, manifest)


def command_run(config: dict[str, Any], args: argparse.Namespace) -> None:
    manifest = load_manifest(config)
    selected = select_run_snapshots(
        manifest,
        snapshot_key=args.snapshot,
        retry_failed=args.retry_failed,
        limit=args.limit,
    )
    if not selected:
        print("no matching snapshots")
        return
    for snapshot in selected:
        try:
            if snapshot.get("build", {}).get("status") != "success":
                build_snapshot(
                    config, manifest, snapshot, keep_worktrees=args.keep_worktrees
                )
            benchmark_status = snapshot.get("benchmark", {}).get("status", "pending")
            if benchmark_status != "success" and (
                benchmark_status == "pending" or args.retry_failed or args.snapshot
            ):
                benchmark_snapshot(config, manifest, snapshot)
        except Exception as error:
            print(f"ERROR {snapshot['key']}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    render_report(config, manifest)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", help="single snapshot key, for example 2024-12")
    group.add_argument("--all-pending", action="store_true", help="process pending snapshots")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--fetch", action="store_true")
    plan_parser.add_argument("--dry-run", action="store_true")
    plan_parser.add_argument("--through", type=date.fromisoformat)
    for name in ("build", "benchmark", "run"):
        child = subparsers.add_parser(name)
        add_selection_arguments(child)
    subparsers.add_parser("render")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        return command_doctor(config)
    if args.command == "plan":
        command_plan(config, fetch=args.fetch, dry_run=args.dry_run, through=args.through)
        return 0
    if args.command in ("build", "benchmark"):
        command_operation(config, args, args.command)
        return 0
    if args.command == "run":
        command_run(config, args)
        return 0
    if args.command == "render":
        render_report(config, load_manifest(config))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
