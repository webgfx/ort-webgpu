import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cross_device_benchmark as collector
import cross_device_report as report
import milestone_benchmark as benchmark


class CrossDeviceTests(unittest.TestCase):
    def test_archive_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(collector.archive_path(root, 'model\\weights.data'), root / 'model' / 'weights.data')
            with self.assertRaisesRegex(ValueError, 'escapes archive'):
                collector.archive_path(root, '../outside.data')

    def test_transferred_file_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifact').write_bytes(b'changed')
            manifest = {'files': [{'path': 'artifact', 'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}]}
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                collector.verify_files(manifest, root)

    def test_additional_gpu_never_inherits_nvidia_measurements(self):
        plan = json.loads((ROOT / 'milestones.json').read_text(encoding='utf-8'))
        own_plan = report.charts_plan(plan)
        self.assertTrue(any('controlledMeasurement' in stage for stage in plan['stages']))
        self.assertFalse(any('controlledMeasurement' in stage for stage in own_plan['stages']))
        self.assertFalse(any('measurement' in group['reference'] for group in own_plan['optimizationGroups']))
        payload = {'rows': [{'stageId': 'reference-2024-12-02', 'model': 'Phi-4-mini-instruct',
                             'date': '2024-12-02', 'status': 'success', 'prefillTps': 100, 'decodeTps': 10}]}
        rows = benchmark.rows_for_chart(own_plan, payload, 'Phi-4-mini-instruct', 'prefillTps')
        self.assertEqual([row['prefillTps'] for row in rows], [100])
        self.assertEqual(benchmark.rows_for_chart(own_plan, {'rows': []}, 'Phi-4-mini-instruct', 'prefillTps'), [])

    def fixture(self):
        plan = json.loads((ROOT / 'milestones.json').read_text(encoding='utf-8'))
        manifest = {'stages': copy.deepcopy(plan['stages'])}
        ref = plan['optimizationGroups'][0]['reference']
        manifest['stages'].append({'id': 'reference-2024-12-02', 'label': 'Starting point',
                                  'date': '2024-12-02', 'runtime': 'runtime/2024-12',
                                  'models': plan['stages'][0]['models']})
        raw = ('Batch size: 1, prompt tokens: 1024, tokens to generate: 128\n'
               'Prompt processing (time to first token):\n avg (us): 10240000\n avg (tokens/s): 100\n n: 5 * 1024 token(s)\n'
               'Token generation:\n avg (tokens/s): 10\n')
        device = {'hostname': 'test-host', 'gpu': 'Test AMD GPU', 'vendor': 'AMD',
                  'driver': 'test-driver', 'adapterId': 'PCI\\VEN_1002', 'powerScheme': 'test-power'}
        rows, artifacts = [], {}
        for stage in manifest['stages']:
            commits = {key: {'commit': ref[f'{key}Commit']} for key in ('ort', 'genai')}
            for repo, key in (('onnxruntime', 'ort'), ('onnxruntime-genai', 'genai')):
                prs = sorted((pr for pr in stage.get('relatedPrs', []) if pr['repository'] == repo),
                             key=lambda pr: pr['mergeDate'], reverse=True)
                if prs:
                    commits[key] = {'commit': prs[0]['mergeCommit']}
            row = {'stageId': stage['id'], 'stage': stage['label'], 'date': benchmark.milestone_utc_date(stage),
                   'chartLabel': stage['label'], 'model': 'Phi-4-mini-instruct',
                   'runtimePath': stage['runtime'], 'modelPath': stage['models'][0]['path'],
                   'graphCapture': bool(stage.get('enableGraphCapture')), 'status': 'success',
                   'rawOutput': raw, 'rawOutputSha256': hashlib.sha256(raw.encode()).hexdigest(),
                   'prefillTps': 100.0, 'decodeTps': 10.0, 'ttftMs': 10240.0,
                   'log': 'test/log', **commits,
                   'benchmarkInvocation': {'batchSize': 1, 'requestedMaxKvCacheLength': 8192,
                                           'command': 'model_benchmark.exe -b 1 -l 1024 -g 128 -r 5 -w 1 -ml 8192',
                                           'effectiveMaxLength': 8192, 'supportsRequestedMaxLength': True}}
            rows.append(row)
            for directory, names in ((row['runtimePath'], ('onnxruntime.dll', 'model_benchmark.exe')),
                                     (row['modelPath'], ('model.onnx', 'model.onnx.data', 'genai_config.json'))):
                for name in names:
                    path = directory.replace('\\', '/') + '/' + name
                    artifacts[path] = {'path': path, 'sha256': 'a' * 64}
        return plan, {'device': device, 'deviceAfter': dict(device), 'rows': rows, 'artifactRoot': 'test/archive',
                      'artifacts': list(artifacts.values()), 'updatedAt': '2026-09-14',
                      'benchmark': {key: plan[key] for key in collector.WORKLOAD_KEYS}}

    def test_results_require_all_milestones_and_own_reference(self):
        plan, payload = self.fixture()
        report.validate(payload, plan)
        payload['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'each milestone'):
            report.validate(payload, plan)

    def test_results_reject_metric_or_log_tampering(self):
        plan, payload = self.fixture()
        payload['rows'][0]['decodeTps'] = 200
        with self.assertRaisesRegex(ValueError, 'metric differs'):
            report.validate(payload, plan)
        payload['rows'][0]['decodeTps'] = 10
        payload['rows'][0]['rawOutput'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'raw log hash mismatch'):
            report.validate(payload, plan)

    def test_results_reject_driver_change_and_wrong_runtime(self):
        plan, payload = self.fixture()
        payload['deviceAfter']['driver'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'driver changed'):
            report.validate(payload, plan)
        payload['deviceAfter']['driver'] = payload['device']['driver']
        payload['rows'][0]['ort']['commit'] = '0' * 40
        with self.assertRaisesRegex(ValueError, 'required PR merge'):
            report.validate(payload, plan)

    def test_results_reject_wrong_warmup_and_sample_count(self):
        plan, payload = self.fixture()
        payload['rows'][0]['benchmarkInvocation']['command'] = 'model_benchmark.exe -b 1 -l 1024 -g 128 -r 5 -w 0'
        with self.assertRaisesRegex(ValueError, 'warmup'):
            report.validate(payload, plan)
        plan, payload = self.fixture()
        row = payload['rows'][0]
        row['rawOutput'] = row['rawOutput'].replace('5 * 1024', '1 * 1024')
        row['rawOutputSha256'] = hashlib.sha256(row['rawOutput'].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'sample count'):
            report.validate(payload, plan)
    def test_failed_measurements_remain_gaps(self):
        plan, payload = self.fixture()
        payload['rows'][0].update(status='failed', error='kernel unsupported')
        result = report.render_sections(plan, [payload])
        self.assertIn('kernel unsupported', result)
        self.assertIn('N/A', result)

    def test_retry_order_does_not_reorder_charts(self):
        plan, payload = self.fixture()
        payload['rows'].reverse()
        with mock.patch.object(benchmark, 'svg_chart', return_value='chart') as chart:
            report.render_sections(plan, [payload])
        for call in chart.call_args_list:
            dates = [row['date'] for row in call.args[0]]
            self.assertEqual(dates, sorted(dates))


if __name__ == '__main__':
    unittest.main()
