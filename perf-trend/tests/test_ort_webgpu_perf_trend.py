import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile


MODULE_PATH = Path(__file__).resolve().parents[1] / "ort-webgpu-perf-trend.py"
SPEC = importlib.util.spec_from_file_location("ort_webgpu_perf_trend", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TrendTests(unittest.TestCase):
    def test_load_config_expands_environment_and_relative_paths(self):
        source = json.loads(
            (MODULE_PATH.parent / "config.example.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source.update(
                {
                    "ortRepo": "repos/onnxruntime",
                    "genaiRepo": "repos/onnxruntime-genai",
                    "outputRoot": "${TEST_PERF_ROOT}",
                    "backupRoot": "archive/runtime",
                    "worktreeRoot": "~/ort-test-worktrees",
                    "cmakePath": "tools/cmake.exe",
                }
            )
            source["models"][0]["path"] = "archive/model/Phi-4"
            source["snapshots"] = [
                {
                    "key": "subgroup",
                    "anchorDate": "2025-01-14",
                    "ortCommit": "a" * 40,
                    "genaiCommit": "b" * 40,
                }
            ]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")

            with mock.patch.dict(os.environ, {"TEST_PERF_ROOT": str(root / "data")}):
                config = MODULE.load_config(config_path)

            self.assertEqual(config["ortRepo"], str((root / "repos/onnxruntime").resolve()))
            self.assertEqual(config["outputRoot"], str((root / "data").resolve()))
            self.assertEqual(config["backupRoot"], str((root / "archive/runtime").resolve()))
            self.assertEqual(
                config["models"][0]["path"], str((root / "archive/model/Phi-4").resolve())
            )
            self.assertEqual(config["snapshots"][0]["key"], "subgroup")

    def test_monthly_anchors_start_on_requested_day(self):
        anchors = MODULE.monthly_anchors(
            MODULE.date(2024, 12, 2), MODULE.date(2025, 3, 1), 2
        )
        self.assertEqual(
            [value.isoformat() for value in anchors],
            ["2024-12-02", "2025-01-02", "2025-02-02"],
        )

    def test_parse_benchmark_output(self):
        output = """
Prompt processing (time to first token)
  avg (us): 250000
  avg (tokens/s): 512.25
Token generation:
  avg (tokens/s): 42.5
E2E generation
  avg (ms): 3500
Peak working set size (bytes): 123456
"""
        self.assertEqual(
            MODULE.parse_benchmark_output(output),
            {
                "ttftMs": 250.0,
                "prefillTps": 512.25,
                "decodeTps": 42.5,
                "e2eMs": 3500.0,
                "peakMemoryBytes": 123456,
            },
        )

    def test_remove_tree_handles_read_only_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "build"
            target.mkdir()
            read_only = target / "pack.idx"
            read_only.write_text("data", encoding="utf-8")
            os.chmod(read_only, stat.S_IREAD)
            MODULE.remove_tree(target)
            self.assertFalse(target.exists())

    def test_patch_legacy_extensions_macro(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = (
                Path(temporary)
                / "Release"
                / "_deps"
                / "onnxruntime_extensions-src"
                / "shared"
                / "api"
                / "image_processor.cc"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                "#if OCOS_ENABLE_VENDOR_IMAGE_CODECS\n#if WIN32\n#endif\n#endif\n",
                encoding="utf-8",
            )
            adjustments = MODULE.patch_legacy_extensions_macro(Path(temporary))
            self.assertEqual(len(adjustments), 2)
            updated = source.read_text(encoding="utf-8")
            self.assertIn(
                "#if defined(OCOS_ENABLE_VENDOR_IMAGE_CODECS)",
                updated,
            )
            self.assertIn("#if defined(WIN32)", updated)

    def test_patch_legacy_ort_headers_adds_chrono_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            header = (
                Path(temporary)
                / "include"
                / "onnxruntime"
                / "core"
                / "platform"
                / "ort_mutex.h"
            )
            header.parent.mkdir(parents=True)
            header.write_text(
                "#pragma once\n#ifdef _WIN32\n#include <Windows.h>\n#include <mutex>\n",
                encoding="utf-8",
            )
            first = MODULE.patch_legacy_ort_headers(Path(temporary))
            second = MODULE.patch_legacy_ort_headers(Path(temporary))
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertEqual(
                header.read_text(encoding="utf-8").count("#include <chrono>"), 1
            )

    def test_run_selection_includes_built_snapshot_awaiting_benchmark(self):
        manifest = {
            "snapshots": [
                {
                    "key": "2024-12",
                    "build": {"status": "success"},
                    "benchmark": {"status": "pending"},
                },
                {
                    "key": "2025-01",
                    "build": {"status": "success"},
                    "benchmark": {"status": "success"},
                },
            ]
        }
        selected = MODULE.select_run_snapshots(
            manifest, snapshot_key=None, retry_failed=False, limit=None
        )
        self.assertEqual([item["key"] for item in selected], ["2024-12"])

    def test_existing_result_can_cover_narrower_prompt_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "results" / "2026-08"
            result_dir.mkdir(parents=True)
            (result_dir / "results.json").write_text(
                json.dumps(
                    {
                        "benchmarkConfig": {
                            "promptLengths": [128, 512, 1024],
                            "generationLength": 128,
                            "repetitions": 5,
                            "warmup": 1,
                        },
                        "rows": [
                            {
                                "model": model,
                                "promptLength": prompt,
                                "status": "success",
                            }
                            for model in ("Phi-4",)
                            for prompt in (128, 512, 1024)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "outputRoot": str(root),
                "models": [{"name": "Phi-4"}],
                "promptLengths": [1024],
                "generationLength": 128,
                "repetitions": 5,
                "warmup": 1,
            }
            self.assertTrue(MODULE.benchmark_result_covers(config, "2026-08"))

    def test_required_binaries_follow_genai_linkage(self):
        self.assertNotIn(
            "onnxruntime-genai.dll", MODULE.required_binaries("static")
        )
        self.assertIn(
            "onnxruntime-genai.dll", MODULE.required_binaries("shared")
        )
        self.assertIn(
            "DirectML.dll", MODULE.required_binaries("shared", "dml")
        )
        self.assertIn(
            "D3D12Core.dll", MODULE.required_binaries("shared", "dml")
        )

    def test_genai_cmake_root_supports_both_historical_layouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / "build"
            build.mkdir()
            (build / "CMakeCache.txt").touch()
            self.assertEqual(MODULE.resolve_genai_cmake_root(build, "Release"), build)

            (build / "CMakeCache.txt").unlink()
            release = build / "Release"
            release.mkdir()
            (release / "CMakeCache.txt").touch()
            self.assertEqual(
                MODULE.resolve_genai_cmake_root(build, "Release"), release
            )

    def test_patch_pinned_archive_hash_verifies_commit_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            cache = root / "cache"
            (worktree / "cmake").mkdir(parents=True)
            cache.mkdir()
            commit = "a" * 40
            archive = cache / f"eigen-{commit}.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(f"eigen-{commit}/README", "pinned content")
            (worktree / "cmake" / "deps.txt").write_text(
                f"eigen;https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.zip;"
                f"{'0' * 40}\n",
                encoding="utf-8",
            )
            adjustments = MODULE.patch_pinned_archive_hashes(worktree, cache)
            actual = MODULE.hashlib.sha1(archive.read_bytes()).hexdigest()
            self.assertEqual(len(adjustments), 1)
            self.assertTrue(
                (worktree / "cmake" / "deps.txt")
                .read_text(encoding="utf-8")
                .rstrip()
                .endswith(actual)
            )

    def test_prepare_dml_packages_use_local_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worktree = root / "worktree"
            build = root / "build"
            cache = root / "cache"
            worktree.mkdir()
            cache.mkdir()
            (worktree / "packages.config").write_text(
                '<packages><package id="Microsoft.AI.DirectML" version="1.2.3" /></packages>',
                encoding="utf-8",
            )
            (worktree / "packages.config").write_text(
                '<packages><package id="Microsoft.AI.DirectML" version="1.2.3" />'
                '<package id="Example.Tools" version="4.5.6" /></packages>',
                encoding="utf-8",
            )
            for name in (
                "microsoft.ai.directml.1.2.3.nupkg",
                "example.tools.4.5.6.nupkg",
            ):
                with zipfile.ZipFile(cache / name, "w") as bundle:
                    bundle.writestr("content.txt", "package")
            adjustments = MODULE.prepare_dml_packages(worktree, cache)
            self.assertEqual(len(adjustments), 3)
            nuget_config = (worktree / "NuGet.config").read_text(encoding="utf-8")
            self.assertIn(str(cache), nuget_config)
            self.assertNotIn("nuget.org", nuget_config)

    def test_render_keeps_missing_month_as_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "outputRoot": str(root),
                "models": [{"name": "Phi", "path": "unused"}],
                "promptLengths": [128],
                "generationLength": 128,
                "repetitions": 5,
                "warmup": 1,
            }
            manifest = {
                "configFingerprint": "abc",
                "snapshots": [
                    {
                        "key": "2024-12",
                        "ort": {
                            "commit": "a" * 40,
                            "commitDate": "2024-12-02T13:57:30-08:00",
                        },
                        "genai": {
                            "commit": "b" * 40,
                            "commitDate": "2024-11-25T16:55:30-08:00",
                        },
                        "build": {"status": "failed"},
                        "benchmark": {"status": "failed"},
                    },
                    {
                        "key": "2025-01",
                        "ort": {"commit": "c" * 40},
                        "genai": {"commit": "d" * 40},
                        "build": {"status": "success"},
                        "benchmark": {"status": "success"},
                    },
                ],
            }
            results = root / "results" / "2025-01"
            results.mkdir(parents=True)
            (results / "results.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "model": "Phi",
                                "promptLength": 128,
                                "status": "success",
                                "prefillTps": 100.0,
                                "decodeTps": 20.0,
                                "ttftMs": 10.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.render_report(config, manifest).read_text(encoding="utf-8")
            self.assertIn("2024-12", report)
            self.assertIn("ORT date", report)
            self.assertIn("2024-12-02", report)
            self.assertIn("2024-11-25", report)
            self.assertIn("failed", report)
            self.assertIn("2025-01", report)
            self.assertNotIn(">0.0<", report)

    def test_chart_breaks_line_across_missing_month(self):
        chart = MODULE.chart_svg(
            [("2026-04", 10.0), ("2026-05", 11.0), ("2026-07", 13.0)],
            "Trend",
            "#000",
        )
        self.assertEqual(chart.count("<polyline "), 1)
        self.assertEqual(chart.count("<circle "), 3)

    def test_chart_accepts_named_milestone_labels(self):
        chart = MODULE.chart_svg(
            [("02-subgroup-prefill", 10.0), ("03-dp4a-prefill", 20.0)],
            "Milestones",
            "#000",
        )
        self.assertEqual(chart.count("<polyline "), 1)
        self.assertIn("03-dp4a-prefill", chart)


if __name__ == "__main__":
    unittest.main()
