"""Offline extraction, completeness and rendering regression checks; no GPU used."""

import copy
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest

from analyze import native_registry, schema_docs, strip_comments
from assessment import assessment
from render import render, source_link


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.targets, self.operators, self.assets = [], [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if attrs.get("href", "").startswith("#"):
            self.targets.append(attrs["href"][1:])
        if tag == "tr" and "data-key" in attrs:
            self.operators.append(attrs["data-key"])
            assert "hidden" not in attrs
        if tag in {"script", "img", "iframe"} and attrs.get("src"):
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.assets.append(attrs.get("href"))


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(Path(__file__).with_name("inventory.json").read_text(encoding="utf-8"))
        cls.rows = {(row["domain"], row["name"]): row for row in cls.data["rows"]}

    def test_counts_and_domain_qualified_uniqueness(self):
        self.assertEqual(len(self.rows), len(self.data["rows"]))
        self.assertEqual(dict(Counter(row["status"] for row in self.rows.values())), self.data["counts"])
        self.assertEqual(self.rows["ai.onnx", "Attention"]["status"], "function")
        self.assertEqual(self.rows["com.microsoft", "Attention"]["status"], "registered")

    def test_every_absent_operator_has_reviewed_importance(self):
        for row in self.rows.values():
            if row["status"] in {"missing", "function"}:
                result = assessment(row)
                self.assertIn(result["priority"], {"P1", "P2", "P3"})
                self.assertGreater(len(result["importance"]), 30)
                self.assertGreater(len(result["mitigation"]), 30)

    def test_comments_are_not_registrations_and_strings_survive(self):
        original = '// fake\n"https://example.test" /* skip\nthis */ R"tag(//literal)tag"'
        result = strip_comments(original)
        self.assertNotIn("fake", result)
        self.assertNotIn("skip", result)
        self.assertIn("https://example.test", result)
        self.assertIn("//literal", result)
        self.assertEqual(original.count("\n"), result.count("\n"))

    def test_known_false_positives_are_absent(self):
        for name in ["GRU", "LSTM", "RMSNormalization", "RotaryEmbedding", "TopK", "DFT", "ScatterND", "ScatterElements", "Add", "Range"]:
            self.assertEqual(self.rows["ai.onnx", name]["status"], "registered")
        self.assertEqual(self.rows["com.microsoft", "MoE"]["status"], "missing")
        self.assertEqual(self.rows["com.microsoft", "QMoE"]["status"], "registered")
        self.assertEqual(self.rows["ai.onnx", "Constant"]["status"], "load_time")

    def test_grid_sample_opset_bounds_remain_visible(self):
        row = self.rows["ai.onnx", "GridSample"]
        self.assertEqual(row["registration"]["ranges"], [[16, 19]])
        self.assertEqual(row["uncoveredSchemaVersions"], [20, 22])

    def test_chromium_gaps_not_confused_with_runtime_failures(self):
        gaps = {row["name"]: row["status"] for row in self.rows.values()
                if row["webnnRef"] and row["status"] in {"missing", "function"}}
        self.assertEqual(set(gaps), {"Or", "Xor", "IsInf", "IsNaN", "Round", "Sign", "QuantizeLinear", "LpPool", "Softsign"})
        self.assertEqual(gaps["Softsign"], "function")
        self.assertIn("not nine demonstrated Chrome failures", render(self.data))

    def test_complete_offline_html_and_unique_navigation(self):
        page = Page()
        page.feed(render(self.data))
        self.assertEqual(set(page.operators), {domain + "::" + name for domain, name in self.rows})
        self.assertEqual(len(page.ids), len(set(page.ids)))
        self.assertFalse(set(page.targets) - set(page.ids))
        self.assertFalse(page.assets)

    def test_unknown_new_operator_requires_review(self):
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            assessment({"domain": "ai.onnx", "name": "FutureUnknownOp", "functionSchema": False})

    def test_report_escapes_source_text_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.data)
        modified = copy.deepcopy(self.data)
        attack = '<img src=x onerror="alert(1)">'
        modified["rows"][0]["introduction"] = attack
        document = render(modified)
        self.assertNotIn(attack, document)
        self.assertIn("&lt;img", document)
        self.assertEqual(before, self.data)

    def test_source_links_are_pinned(self):
        self.assertTrue(self.data["referencedSourcesMatchCommits"])
        for repo in ("ort", "onnx", "chromium"):
            self.assertRegex(self.data[repo]["commit"], r"^[a-f0-9]{40}$")
        for item in self.data["limitations"]:
            link = source_link(self.data, item["source"])
            self.assertIn(self.data["ort"]["commit"], link)
            self.assertNotIn("/main/", link)

    def test_registry_matches_pinned_local_sources_when_available(self):
        root = Path(self.data["ort"]["root"])
        if not root.exists():
            self.skipTest("Source checkout is not available")
        from analyze import git
        if git(root, "rev-parse", "HEAD") != self.data["ort"]["commit"]:
            self.skipTest("Local source moved; use pinned worktree to refresh")
        parsed = native_registry(root)
        recorded = {key: row["registration"] for key, row in self.rows.items() if "registration" in row}
        self.assertEqual(parsed, recorded)

    def test_onnx_schema_tables_have_no_dropped_names(self):
        root = Path(self.data["onnx"]["root"])
        if not root.exists():
            self.skipTest("ONNX source is not available")
        for path, domain in [("docs/Operators.md", "ai.onnx"), ("docs/Operators-ml.md", "ai.onnx.ml")]:
            parsed = schema_docs(root, path, domain, "onnx")
            text = (root / path).read_text(encoding="utf-8")
            table = text.split("\n## ai.onnx", 1)[0]
            qualified_names = re.findall(r'\|(?:<sub>experimental</sub> )?<a href="#[^"]+">([^<]+)</a>', table)
            expected = {(name.rsplit(".", 1)[0], name.rsplit(".", 1)[1]) if "." in name else (domain, name)
                        for name in qualified_names}
            self.assertEqual(set(parsed), expected)
            self.assertFalse(set(parsed) - set(self.rows))


if __name__ == "__main__":
    unittest.main()
