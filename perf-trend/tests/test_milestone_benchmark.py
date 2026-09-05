import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "milestone_benchmark.py"
SPEC = importlib.util.spec_from_file_location("milestone_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class MilestoneBenchmarkTests(unittest.TestCase):
    def test_report_index_links_all_major_sections(self):
        index = MODULE.report_index()
        self.assertNotIn("Report menu", index)
        self.assertIn("aria-label='Report navigation'", index)
        for anchor in (
            "performance-charts",
            "model-builder-options",
            "ort-genai-options",
            "webgpu-options",
            "milestones",
            "scope",
            "method",
        ):
            self.assertIn(f"href='#{anchor}'", index)
        self.assertIn("ORT GenAI benchmark options", index)
        self.assertIn("WebGPU genai_config.json options", index)
        self.assertLess(
            index.index("WebGPU genai_config.json options"),
            index.index("ORT GenAI benchmark options"),
        )
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("class='report-index' aria-label='Report navigation' hidden", index)
        self.assertIn(".page-shell{{display:block", source)
        self.assertIn(".report-index{{position:fixed", source)
        self.assertIn("@media (max-width:900px)", source)
        self.assertIn("id='menu-hide'", source)
        self.assertIn('id="menu-show"', source)
        self.assertIn(".page-shell.menu-open", source)
        self.assertIn("setOpen(true)", source)
        self.assertIn("setOpen(false)", source)
        self.assertIn("event.key==='Escape'", source)
        self.assertIn("document.addEventListener('pointerdown'", source)

    def test_builder_option_registry_tracks_profile_defaults_and_history(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        options = {option["name"]: option for option in plan["builderOptions"]}

        self.assertEqual(
            set(options),
            {
                "accuracy_level",
                "is_symmetric",
                "block_size",
                "algo_config",
                "enable_webgpu_graph",
                "shared_embeddings",
                "prune_lm_head",
            },
        )
        self.assertEqual(options["accuracy_level"]["seriesValue"], "4")
        self.assertEqual(options["is_symmetric"]["seriesValue"], "true")
        self.assertEqual(options["block_size"]["seriesValue"], "32")
        self.assertEqual(options["algo_config"]["seriesValue"], "rtn_last")
        self.assertEqual(options["shared_embeddings"]["seriesValue"], "true")
        self.assertIn("int4_accuracy_level", options["accuracy_level"]["legacyNames"])
        self.assertIn("int4_is_symmetric", options["is_symmetric"]["legacyNames"])
        self.assertIn("int4_block_size", options["block_size"]["legacyNames"])
        self.assertIn("int4_algo_config", options["algo_config"]["legacyNames"])
        for option in options.values():
            self.assertTrue(option["description"])
            self.assertTrue(option["requirement"])
            self.assertTrue(option["currentDefault"])
            history = option["defaultHistory"]
            self.assertTrue(history)
            self.assertEqual(
                [change["date"] for change in history],
                sorted((change["date"] for change in history), reverse=True),
            )
            for change in history:
                self.assertEqual(len(change["commit"]), 40)
                self.assertGreater(change["pr"], 0)

        table = MODULE.builder_options_table(plan)
        self.assertIn("<h2>Model-builder options</h2>", table)
        self.assertIn("Current profile", table)
        self.assertIn("Requirement now", table)
        self.assertIn("Current default", table)
        self.assertIn("Default / naming history", table)
        self.assertIn("algo_config", table)
        self.assertIn("int4_algo_config", table)
        self.assertIn("rtn_last", table)
        self.assertIn("shared_embeddings", table)
        self.assertIn(
            "<strong>Required now</strong><br><code class='profile-command'>"
            "algo_config=rtn_last enable_webgpu_graph=true shared_embeddings=true "
            "prune_lm_head=true</code>",
            table,
        )
        self.assertIn(
            "accuracy_level=4 is_symmetric=true block_size=32 algo_config=rtn_last "
            "enable_webgpu_graph=true shared_embeddings=true prune_lm_head=true",
            table,
        )
        self.assertLess(
            table.index("WebGPU genai_config.json options"),
            table.index("ORT GenAI benchmark options"),
        )
        self.assertLess(
            table.index("ORT GenAI benchmark options"),
            table.index("id='milestones'"),
        )
        self.assertLess(table.index("2026-07-20"), table.index("2025-05-13"))

    def test_genai_config_registry_covers_webgpu_runtime_options(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        options = {option["path"]: option for option in plan["genaiConfigOptions"]}
        for path in (
            "provider_options.webgpu.enableGraphCapture",
            "provider_options.webgpu.validationMode",
            "provider_options.webgpu.multiRotaryCacheConcatOffset",
            "provider_options.webgpu.kvCacheQuantizationBits",
            "provider_options.webgpu.maxNumPendingDispatches",
        ):
            self.assertIn(path, options)
        self.assertTrue(
            all(path.startswith("provider_options.webgpu.") for path in options)
        )
        self.assertEqual(
            options["provider_options.webgpu.multiRotaryCacheConcatOffset"]["requirement"],
            "Conditional",
        )
        self.assertIn(
            '"0" stored; "1" only in the controlled graph run',
            options["provider_options.webgpu.enableGraphCapture"]["profile"],
        )
        table = MODULE.genai_config_options_table(plan)
        self.assertIn("WebGPU genai_config.json options", table)
        self.assertIn("Important settings", table)
        self.assertIn("All WebGPU provider settings", table)
        self.assertIn("Requirement now", table)
        self.assertIn("multiRotaryCacheConcatOffset", table)
        self.assertNotIn("provider_options.webgpu.", table)
        self.assertLess(table.index("Important settings"), table.index("Requirement now"))

        important = {
            option["path"]: option
            for option in plan["importantGenaiConfigOptions"]
        }
        for path in (
            "provider_options.webgpu.enableGraphCapture",
            "provider_options.webgpu.validationMode",
            "optimization.disable_specified_optimizers",
        ):
            self.assertIn(path, important)
        self.assertIn(
            '"ConstantFolding"',
            important["optimization.disable_specified_optimizers"]["profile"],
        )
        self.assertIn("decoder session_options", table)
        self.assertIn("optimization.disable_specified_optimizers", table)

    def test_ort_genai_benchmark_options_include_8k_kv_cache(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["maxKvCacheLength"], 8192)
        options = {
            option["option"]: option for option in plan["ortGenaiBenchmarkOptions"]
        }
        self.assertEqual(options["-m / --max_lengths"]["value"], "8192")
        self.assertEqual(options["-ml / --max_length"]["value"], "8192")
        self.assertEqual(options["-b / --batch_size"]["value"], "1")
        table = MODULE.ort_genai_benchmark_options_table(plan)
        self.assertIn("ORT GenAI benchmark options", table)
        self.assertIn("-m 8192", table)
        self.assertIn("-ml 8192", table)
        self.assertIn("maximum KV-cache capacity at 8K tokens", table)
        self.assertNotIn("provider_options.webgpu.-m", table)

    def test_force_with_stage_does_not_force_all_stages(self):
        self.assertFalse(MODULE.force_all_stages(True, ["03-dp4a-prefill"]))
        self.assertTrue(MODULE.force_all_stages(True, None))

    def test_cpp_benchmark_capabilities_and_metadata_use_ml(self):
        help_output = "-b,--batch_size <number>\n-ml,--max_length <number>\n"
        with mock.patch.object(
            MODULE.subprocess,
            "run",
            return_value=mock.Mock(stdout=help_output),
        ):
            capabilities = MODULE.benchmark_capabilities(Path("model_benchmark.exe"))

        self.assertEqual(capabilities, {"batchSize": True, "maxLength": True})
        plan = {
            "batchSize": 1,
            "promptLength": 1024,
            "generationLength": 128,
            "maxKvCacheLength": 8192,
        }
        metadata = MODULE.benchmark_metadata(
            plan,
            ["model_benchmark.exe", "-ml", "8192"],
            capabilities,
        )
        self.assertEqual(metadata["maxLengthOption"], "-ml")
        self.assertEqual(metadata["effectiveMaxLength"], 8192)

    def test_old_cpp_benchmark_records_effective_length_without_8k_claim(self):
        plan = {
            "batchSize": 1,
            "promptLength": 1024,
            "generationLength": 128,
            "maxKvCacheLength": 8192,
        }
        metadata = MODULE.benchmark_metadata(
            plan,
            ["model_benchmark.exe", "-l", "1024", "-g", "128"],
            {"batchSize": True, "maxLength": False},
        )
        self.assertFalse(metadata["supportsRequestedMaxLength"])
        self.assertIsNone(metadata["maxLengthOption"])
        self.assertEqual(metadata["effectiveMaxLength"], 1152)

    def test_benchmark_timeout_is_recorded_without_aborting_later_stages(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("except subprocess.TimeoutExpired as error:", source)
        self.assertIn('error_message = "timed out after 1800 seconds"', source)

    def test_output_root_and_plan_paths_are_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"outputRoot": "archive"}), encoding="utf-8")

            self.assertEqual(
                MODULE.resolve_output_root(None, config), (root / "archive").resolve()
            )
            with mock.patch.dict(os.environ, {"ORT_WEBGPU_PERF_ROOT": str(root / "env")}):
                self.assertEqual(
                    MODULE.resolve_output_root(None, config), (root / "env").resolve()
                )
            self.assertEqual(
                MODULE.resolve_output_root(root / "explicit", config),
                (root / "explicit").resolve(),
            )
            self.assertEqual(
                MODULE.resolve_plan_path(root, "runtime/2025-01"),
                (root / "runtime/2025-01").resolve(),
            )

    def test_plan_is_phi4_only_and_matches_requested_event_series(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        self.assertEqual(len(plan["stages"]), 11)
        self.assertEqual(
            {
                model["name"]
                for stage in plan["stages"]
                for model in stage["models"]
            },
            {"Phi-4-mini-instruct"},
        )
        groups = {group["id"]: group for group in plan["optimizationGroups"]}
        self.assertEqual(
            [stage["id"] for stage in groups["prefill"]["stages"]],
            [
                "02-subgroup-prefill",
                "03-dp4a-prefill",
                "03-gqa-flash-prefill",
                "06-shared-memory-broadcast",
                "07-prune",
                "10-flashattention-max-k-step",
            ],
        )
        self.assertEqual(
            [stage["id"] for stage in groups["decode"]["stages"]],
            [
                "04-gqa-flash-decode",
                "06-gqa-rope",
                "08-graph-capture",
                "09-matmulnbits-mlp",
                "11-flashattention-decode-tuning",
            ],
        )
        by_id = {stage["id"]: stage for stage in plan["stages"]}
        self.assertNotIn("05-matmul8-no-rope", by_id)
        for stage_id in (
            "02-subgroup-prefill",
            "03-dp4a-prefill",
            "06-shared-memory-broadcast",
            "07-prune",
            "10-flashattention-max-k-step",
        ):
            self.assertEqual(
                by_id[stage_id]["chartMetrics"], ["prefillTps"]
            )
        self.assertNotIn("01-shared-memory-prefill", by_id)
        self.assertNotIn("measurementStatus", by_id["02-subgroup-prefill"])
        self.assertIn(
            "ModelPackage API",
            by_id["11-flashattention-decode-tuning"]["runtimeNotes"]["onnxruntime-genai"],
        )
        self.assertIn(
            "runtime-milestones\\02-subgroup-prefill",
            by_id["02-subgroup-prefill"]["runtime"],
        )
        dp4a = by_id["03-dp4a-prefill"]
        self.assertEqual(
            dp4a["runtime"], "runtime-milestones\\03-dp4a-prefill"
        )
        self.assertEqual(
            dp4a["milestoneCommit"],
            "58c29d34b2f29d2d3836e15c45e05a1bde0c8ad6",
        )
        self.assertIn("int4_accuracy_level=4", dp4a["modelOptions"])
        self.assertIn("standard-unfused-rope", dp4a["models"][0]["path"])
        shm = by_id["06-shared-memory-broadcast"]
        self.assertEqual(shm["runtime"], "runtime-milestones\\shm-broadcast-candidate")
        self.assertNotIn("05-flashattention-causal-prefill", by_id)
        self.assertNotIn("05-matmulnbits-widetile", by_id)
        model_paths = {stage["models"][0]["path"] for stage in plan["stages"]}
        self.assertIn(
            "model\\aligned\\standard-unfused-rope\\Phi-4-mini-instruct",
            model_paths,
        )
        self.assertIn(
            "model\\aligned\\standard-fused-rope\\Phi-4-mini-instruct",
            model_paths,
        )
        self.assertEqual(
            by_id["07-prune"]["models"][0]["path"],
            "experiments\\prune-accuracy4\\candidate\\Phi-4-mini-instruct",
        )
        expected_options = {
            "int4_is_symmetric=true",
            "int4_accuracy_level=4",
            "enable_webgpu_graph=true",
        }
        for stage in plan["stages"]:
            self.assertTrue(expected_options.issubset(set(stage["modelOptions"])))
            if stage["id"] == "07-prune":
                self.assertIn("prune_lm_head=true", stage["modelOptions"])
            self.assertIsInstance(stage["milestonePr"], int)
            self.assertTrue(stage["relatedPrs"])
            for pull_request in stage["relatedPrs"]:
                self.assertIn(
                    pull_request["repository"],
                    {"onnxruntime", "onnxruntime-genai"},
                )
                self.assertIsInstance(pull_request["number"], int)
                self.assertTrue(pull_request["role"])
                self.assertTrue(pull_request["owner"])
                self.assertRegex(pull_request["mergeDate"], r"^\d{4}-\d{2}-\d{2}$")
                self.assertRegex(pull_request["mergeCommit"], r"^[0-9a-f]{40}$")
            merge_dates = [pull_request["mergeDate"] for pull_request in stage["relatedPrs"]]
            self.assertEqual(merge_dates, sorted(merge_dates, reverse=True))
        self.assertEqual(by_id["06-gqa-rope"]["date"][:7], "2025-11")
        self.assertEqual(by_id["10-flashattention-max-k-step"]["date"][:7], "2026-05")
        self.assertEqual(
            {pull_request["repository"] for pull_request in by_id["06-gqa-rope"]["relatedPrs"]},
            {"onnxruntime", "onnxruntime-genai"},
        )
        self.assertEqual(len(by_id["06-gqa-rope"]["relatedPrs"]), 5)
        self.assertEqual(
            {pull_request["repository"] for pull_request in by_id["08-graph-capture"]["relatedPrs"]},
            {"onnxruntime", "onnxruntime-genai"},
        )
        self.assertGreaterEqual(len(by_id["08-graph-capture"]["relatedPrs"]), 10)
        self.assertEqual(
            {pull_request["number"] for pull_request in by_id["10-flashattention-max-k-step"]["relatedPrs"]},
            {27780, 28511},
        )
        self.assertEqual(
            [pull_request["number"] for pull_request in by_id["10-flashattention-max-k-step"]["relatedPrs"]],
            [28511, 27780],
        )
        self.assertNotIn("05-pre-gqa-rope", by_id)
        self.assertEqual(
            plan["optimizationGroups"][0]["reference"]["date"], "2024-12-02"
        )

    def test_prefill_only_milestones_are_excluded_from_decode_chart(self):
        rows = [
            {
                "status": "success",
                "date": "2024-12-10",
                "stage": "Shared-memory tiled broadcast",
                "chartLabel": "SHM Broadcast",
                "decodeTps": 10.0,
                "chartMetrics": ["ttftMs", "prefillTps"],
            },
            {
                "status": "success",
                "date": "2025-04-08",
                "stage": "Flash Decoding",
                "chartLabel": "Flash Decoding",
                "decodeTps": 20.0,
                "chartMetrics": ["ttftMs", "prefillTps", "decodeTps"],
            },
            {
                "status": "success",
                "date": "2026-03-04",
                "stage": "LM-head prune",
                "chartLabel": "LM-head Prune",
                "decodeTps": 30.0,
                "chartMetrics": ["ttftMs", "prefillTps"],
            },
        ]

        chart = MODULE.svg_chart(rows, "decodeTps", "Decode", False)

        self.assertNotIn("SHM Broadcast", chart)
        self.assertNotIn("LM-head Prune", chart)
        self.assertIn("Flash Decoding", chart)
        self.assertEqual(chart.count("<circle "), 1)
        self.assertIn(">20.0</text>", chart)

    def test_chart_metrics_follow_optimization_groups(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        metrics = MODULE.chart_metrics_by_stage(plan)

        self.assertNotIn("01-shared-memory-prefill", metrics)
        self.assertEqual(metrics["03-gqa-flash-prefill"], ["prefillTps"])
        self.assertNotIn("05-flashattention-causal-prefill", metrics)
        self.assertNotIn("05-matmul8-no-rope", metrics)
        self.assertEqual(metrics["04-gqa-flash-decode"], ["decodeTps"])
        self.assertEqual(metrics["06-gqa-rope"], ["decodeTps"])
        self.assertEqual(metrics["08-graph-capture"], ["decodeTps"])

    def test_missing_prefill_milestone_remains_visible_without_a_value(self):
        rows = [
            {
                "status": "success",
                "date": "2024-12-10",
                "stage": "Starting measurement",
                "chartLabel": "Starting point",
                "prefillTps": 278.264,
                "chartMetrics": ["ttftMs", "prefillTps", "decodeTps"],
            },
            {
                "status": "missing",
                "date": "2025-01-14",
                "stage": "Unmeasured prefill optimization",
                "chartLabel": "Unmeasured",
                "error": "No isolated runtime",
                "chartMetrics": ["ttftMs", "prefillTps"],
            },
            {
                "status": "success",
                "date": "2025-01-21",
                "stage": "DP4A MatMulNBits",
                "chartLabel": "DP4A",
                "prefillTps": 538.1,
                "chartMetrics": ["ttftMs", "prefillTps"],
            },
        ]

        chart = MODULE.svg_chart(rows, "prefillTps", "Prefill", False)

        self.assertIn("2025-01 · Unmeasured", chart)
        self.assertIn("class='missing-point'", chart)
        self.assertIn(">N/A</text>", chart)
        self.assertIn("No isolated runtime", chart)
        self.assertEqual(chart.count("<circle "), 2)
        self.assertEqual(chart.count("<polyline "), 2)

    def test_charts_show_overall_gain_from_first_to_last_valid_point(self):
        rows = [
            {
                "status": "success",
                "date": "2025-01-01",
                "stage": "First optimization",
                "chartLabel": "First",
                "ttftMs": 100.0,
                "prefillTps": 50.0,
            },
            {
                "status": "success",
                "date": "2025-02-01",
                "stage": "Optimized",
                "chartLabel": "Optimized",
                "ttftMs": 50.0,
                "prefillTps": 100.0,
            },
        ]

        ttft = MODULE.svg_chart(rows, "ttftMs", "TTFT", True)
        throughput = MODULE.svg_chart(rows, "prefillTps", "Prefill", False)

        self.assertIn("Overall gain: 2.00×", ttft)
        self.assertIn("Overall gain: 2.00×", throughput)
        self.assertNotIn("% lower", ttft)
        self.assertNotIn("+100.0%", throughput)

    def test_timeline_does_not_render_ep_column(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('svg_chart(rows, "ttftMs"', source)
        self.assertIn('rows_for_chart(plan, payload, model, "prefillTps")', source)
        self.assertIn('rows_for_chart(plan, payload, model, "decodeTps")', source)
        self.assertNotIn("<th>EP</th>", source)
        self.assertNotIn("stage['backend'].upper()", source)
        self.assertNotIn("Open the original monthly-runtime report", source)
        self.assertNotIn("Open the original Alder Lake milestone reference", source)
        for removed_card in ("successful measurements", "runs per point", 'class="cards"'):
            self.assertNotIn(removed_card, source)
        for excluded_work in ("Qualcomm-specific", "Whisper", "Phi-4 multimodal", "GPT-OSS"):
            self.assertIn(excluded_work, source)

    def test_timeline_impacts_match_optimization_groups(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        impacts = MODULE.optimization_impacts(plan)

        self.assertNotIn("01-shared-memory-prefill", impacts)
        self.assertEqual(impacts["02-subgroup-prefill"], ["prefill"])
        self.assertEqual(impacts["04-gqa-flash-decode"], ["decode"])
        self.assertEqual(impacts["06-gqa-rope"], ["decode"])
        self.assertEqual(impacts["07-prune"], ["prefill"])
        self.assertEqual(impacts["11-flashattention-decode-tuning"], ["decode"])
        self.assertIn("impact-prefill", MODULE.impact_badges(["prefill"]))
        self.assertIn("impact-decode", MODULE.impact_badges(["decode"]))
        cell = MODULE.impact_badges(["prefill", "decode"])
        self.assertIn(">Prefill</span>", cell)
        self.assertIn(">Decode</span>", cell)
        self.assertNotIn("tok/s", cell)

    def test_builder_cell_includes_commit_and_date(self):
        cell = MODULE.builder_cell(
            [
                {
                    "builderCommit": "ab3dd3d84ca64ecfccac52fd520f9f00414cba82",
                    "builderDate": "2025-04-08",
                }
            ]
        )

        self.assertIn("<code>ab3dd3d84c</code>", cell)
        self.assertIn("<small>2025-04-08</small>", cell)

    def test_git_dates_are_normalized_to_utc(self):
        self.assertEqual(
            MODULE.utc_date("2024-12-10T17:07:11-08:00"), "2024-12-11"
        )
        stage = {
            "date": "2024-12-10",
            "relatedPrs": [
                {"mergeDate": "2024-12-11"},
                {"mergeDate": "2024-12-13"},
            ],
        }
        self.assertEqual(MODULE.milestone_utc_date(stage), "2024-12-13")

    def test_pr_links_include_all_repositories_and_roles(self):
        links = MODULE.related_pull_requests_cell(
            {
                "milestonePr": 1848,
                "milestoneRepo": "onnxruntime-genai",
                "relatedPrs": [
                    {
                        "repository": "onnxruntime-genai",
                        "number": 1848,
                        "mergeDate": "2025-11-25",
                        "mergeCommit": "32d101d997c9f73b85ee16faac55b0ab67e82f45",
                        "owner": "qjia7",
                        "role": "Graph capture integration.",
                    },
                    {
                        "repository": "onnxruntime",
                        "number": 26450,
                        "mergeDate": "2025-12-15",
                        "mergeCommit": "0aebe823cd7acd46251ad1721e4579b37f6c6e79",
                        "owner": "qjia7",
                        "role": "Required data transfer support.",
                    },
                ],
            }
        )

        self.assertIn("microsoft/onnxruntime/pull/26450", links)
        self.assertIn("microsoft/onnxruntime-genai/pull/1848", links)
        self.assertIn("Required data transfer support.", links)
        self.assertIn("Graph capture integration.", links)
        self.assertIn("Merged 2025-12-15", links)
        self.assertIn("Merged 2025-11-25", links)
        self.assertIn("Merge commit", links)
        self.assertIn('Owner <a href="https://github.com/qjia7">@qjia7</a>', links)
        self.assertIn("<code>0aebe823cd</code>", links)
        self.assertIn("<code>32d101d997</code>", links)
        self.assertLess(links.index("onnxruntime/pull/26450"), links.index("onnxruntime-genai/pull/1848"))

        ort_links = MODULE.related_pull_requests_cell(
            {
                "milestonePr": 1848,
                "milestoneRepo": "onnxruntime-genai",
                "relatedPrs": [
                    {"repository": "onnxruntime", "number": 26450, "owner": "qjia7"},
                    {"repository": "onnxruntime-genai", "number": 1848, "owner": "qjia7"},
                ],
            },
            "onnxruntime",
        )
        self.assertIn("onnxruntime/pull/26450", ort_links)
        self.assertNotIn("onnxruntime-genai/pull/1848", ort_links)

    def test_runtime_and_prs_share_repository_specific_columns(self):
        stage = {
            "milestonePr": 1848,
            "milestoneRepo": "onnxruntime-genai",
            "relatedPrs": [
                {
                    "repository": "onnxruntime",
                    "number": 26450,
                    "mergeDate": "2025-12-15",
                    "owner": "qjia7",
                    "role": "ORT support.",
                },
                {
                    "repository": "onnxruntime-genai",
                    "number": 1848,
                    "mergeDate": "2025-11-25",
                    "owner": "qjia7",
                    "role": "GenAI integration.",
                },
            ],
        }
        runtime = {"commit": "abcdef1234567890", "commitDate": "2026-05-01T00:00:00Z"}
        ort = MODULE.runtime_and_pull_requests_cell(runtime, stage, "onnxruntime")
        genai = MODULE.runtime_and_pull_requests_cell(
            runtime, stage, "onnxruntime-genai"
        )
        self.assertIn("Tested commit", ort)
        self.assertIn("<code>abcdef1234</code>", ort)
        self.assertIn("onnxruntime/pull/26450", ort)
        self.assertNotIn("onnxruntime-genai/pull/1848", ort)
        self.assertIn("onnxruntime-genai/pull/1848", genai)
        self.assertNotIn("onnxruntime/pull/26450", genai)

    def test_detailed_milestone_table_fields_and_measured_impact(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        for stage in plan["stages"]:
            self.assertTrue(stage["majorContributors"])
            self.assertTrue(stage["applicability"])

        links = MODULE.related_pull_requests_cell(
            {
                "milestonePr": 1848,
                "milestoneRepo": "onnxruntime-genai",
            }
        )
        self.assertIn("microsoft/onnxruntime-genai/pull/1848", links)

        sample_plan = {
            "optimizationGroups": [
                {
                    "id": "prefill",
                    "title": "Prefill optimizations",
                    "metric": "prefillTps",
                    "metricLabel": "prefill tok/s",
                    "stages": [{"id": "one"}, {"id": "two"}],
                }
            ]
        }
        sample_payload = {
            "rows": [
                {"stageId": "one", "status": "success", "prefillTps": 50.0},
                {"stageId": "two", "status": "success", "prefillTps": 100.0},
            ]
        }
        impact = MODULE.measured_impact_cell({"id": "two"}, sample_plan, sample_payload)
        self.assertIn("2.00×", impact)
        self.assertIn("50.0 → 100.0 prefill tok/s", impact)
        self.assertNotIn("%", impact)

        controlled = MODULE.measured_impact_cell(
            {
                "id": "two",
                "controlledMeasurement": {
                    "parent": 90.0,
                    "candidate": 100.0,
                    "metricLabel": "prefill tok/s",
                },
            },
            sample_plan,
            sample_payload,
        )
        self.assertIn("1.11×", controlled)
        self.assertIn("Controlled parent → candidate measurement", controlled)

        referenced_plan = {
            "optimizationGroups": [
                {
                    "id": "decode",
                    "title": "Decode optimizations",
                    "metric": "decodeTps",
                    "metricLabel": "decode tok/s",
                    "reference": {"stageId": "start"},
                    "stages": [{"id": "feature"}],
                }
            ]
        }
        referenced_payload = {
            "rows": [
                {"stageId": "start", "status": "success", "decodeTps": 70.0},
                {"stageId": "feature", "status": "success", "decodeTps": 84.0},
            ]
        }
        referenced = MODULE.measured_impact_cell(
            {"id": "feature"}, referenced_plan, referenced_payload
        )
        self.assertIn("1.20×", referenced)
        self.assertIn("70.0 → 84.0 decode tok/s", referenced)

        source = MODULE_PATH.read_text(encoding="utf-8")
        for heading in (
            "<h2>Milestones</h2>",
            "Milestone / introduction",
            "Performance area",
            "Required PRs",
            "Contributors",
            "Measured impact",
            "<th>ORT</th>",
            "<th>ORT GenAI</th>",
            "Builder / date",
            "Local test artifacts",
            "Model/runtime features",
        ):
            self.assertIn(heading, source)
        self.assertNotIn("<h2>Aligned timeline</h2>", source)
        self.assertNotIn("<h2>Milestone details</h2>", source)
        self.assertNotIn("<th>Applicability / conditions</th>", source)
        self.assertIn(
            "<th>Contributors</th><th>Performance area</th>", source
        )
        self.assertIn("<th>Date</th>", source)
        self.assertNotIn("(UTC)", source)
        self.assertIn(
            'for stage in sorted(plan["stages"], key=lambda item: item["date"], reverse=True):',
            source,
        )
        self.assertIn("width:min(2400px", source)
        self.assertIn("@media (min-width:1500px)", source)
        self.assertIn("@media (max-width:900px)", source)
        self.assertIn('archived_report_path = output_root / "index.html"', source)
        self.assertNotIn("milestone_details", source)

    def test_prefill_chart_can_use_an_inline_dated_baseline(self):
        plan = {
            "optimizationGroups": [
                {
                    "id": "prefill",
                    "metric": "prefillTps",
                    "metricLabel": "prefill tok/s",
                    "reference": {
                        "date": "2024-12-02",
                        "label": "December 2, 2024 baseline",
                        "chartLabel": "Baseline",
                        "measurement": {"prefillTps": 50.0},
                    },
                    "stages": [{"id": "new"}],
                }
            ],
            "stages": [
                {
                    "id": "new",
                    "label": "Prefill optimization",
                    "chartLabel": "Optimized",
                    "date": "2024-12-10",
                }
            ],
        }
        payload = {
            "rows": [
                {
                    "stageId": "new",
                    "model": "Phi-4",
                    "status": "success",
                    "date": "2024-12-10",
                    "prefillTps": 100.0,
                }
            ]
        }

        rows = MODULE.rows_for_chart(plan, payload, "Phi-4", "prefillTps")
        self.assertEqual([row["date"] for row in rows], ["2024-12-02", "2024-12-10"])
        self.assertEqual([row["chartLabel"] for row in rows], ["Baseline", "Optimized"])
        self.assertTrue(rows[0]["isReference"])
        chart = MODULE.svg_chart(rows, "prefillTps", "Prefill", False)
        self.assertIn("Overall gain: 2.00×", chart)
        impact = MODULE.measured_impact_cell({"id": "new"}, plan, payload)
        self.assertIn("50.0 → 100.0 prefill tok/s", impact)

    def test_decode_chart_can_use_an_older_non_milestone_starting_measurement(self):
        plan = {
            "optimizationGroups": [
                {
                    "id": "decode",
                    "metric": "decodeTps",
                    "reference": {
                        "stageId": "old",
                        "label": "December 2024 starting measurement",
                        "chartLabel": "Starting point",
                    },
                    "stages": [{"id": "new"}],
                }
            ],
            "stages": [
                {"id": "old", "label": "Prefill-only milestone", "date": "2024-12-10"},
                {
                    "id": "new",
                    "label": "Decode optimization",
                    "chartLabel": "Decode",
                    "date": "2025-04-08",
                },
            ],
        }
        payload = {
            "rows": [
                {
                    "stageId": "old",
                    "model": "Phi-4",
                    "status": "success",
                    "date": "2024-12-10",
                    "decodeTps": 70.0,
                },
                {
                    "stageId": "new",
                    "model": "Phi-4",
                    "status": "success",
                    "date": "2025-04-08",
                    "decodeTps": 140.0,
                },
            ]
        }

        rows = MODULE.rows_for_chart(plan, payload, "Phi-4", "decodeTps")
        self.assertEqual([row["chartLabel"] for row in rows], ["Starting point", "Decode"])
        self.assertTrue(rows[0]["isReference"])
        chart = MODULE.svg_chart(rows, "decodeTps", "Decode", False)
        self.assertIn("Overall gain: 2.00×", chart)
        self.assertIn("reference-point", chart)

    def test_graph_capture_uses_a_temporary_config_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            original = {
                "model": {
                    "decoder": {
                        "session_options": {"provider_options": [{"webgpu": {}}]}
                    }
                }
            }
            config = model / "genai_config.json"
            config.write_text(json.dumps(original), encoding="utf-8")
            (model / "model.onnx").write_bytes(b"model")

            with MODULE.model_path_for_stage(
                model, {"id": "graph", "enableGraphCapture": True}, root
            ) as staged:
                staged_config = json.loads(
                    (staged / "genai_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    staged_config["model"]["decoder"]["session_options"]
                    ["provider_options"][0]["webgpu"]["enableGraphCapture"],
                    "1",
                )
                self.assertEqual(
                    staged_config["model"]["decoder"]["session_options"]
                    ["provider_options"][0]["webgpu"]["validationMode"],
                    "disabled",
                )
                self.assertEqual(
                    staged_config["model"]["decoder"]["session_options"]
                    ["provider_options"][0]["webgpu"]
                    ["multiRotaryCacheConcatOffset"],
                    "4096",
                )
                self.assertEqual((staged / "model.onnx").read_bytes(), b"model")

            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), original)

    def test_every_timeline_stage_has_a_brief_optimization_summary(self):
        plan = json.loads((ROOT / "milestones.json").read_text(encoding="utf-8"))
        summaries = MODULE.optimization_summaries(plan)

        self.assertEqual(set(summaries), {stage["id"] for stage in plan["stages"]})
        self.assertIn("subgroup operations", summaries["02-subgroup-prefill"])
        self.assertIn("one-token generation", summaries["04-gqa-flash-decode"])
        self.assertIn("final prompt token", summaries["07-prune"])
        self.assertIn("CPU submission overhead", summaries["08-graph-capture"])

if __name__ == "__main__":
    unittest.main()
