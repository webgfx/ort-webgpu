import copy
import unittest

from prepare import random_prompt
from compare import compare, validate


class BenchmarkTests(unittest.TestCase):
    def sample_result(self):
        sample = dict(ttftMs=10., e2eMs=30., prefillTps=200., decodeTps=100., generated=[4, 5, 6], tokenDeliveryMs=[10., 20., 30.])
        entry = dict(name='model', cases=[dict(pl=2, prompt=[7, 8])])
        return dict(success=True, requestSha256='same', runtime='test',
                    request=dict(models=[entry], repetitions=1, generationLength=3),
                    models=[dict(name='model', success=True, executionEvidence=dict(samplingDevice='gpu'), rows=[dict(pl=2, prompt=[7, 8], warmup=sample, samples=[copy.deepcopy(sample)])])])

    def test_valid_comparison(self):
        data = self.sample_result()
        row = compare(data, copy.deepcopy(data))[0]
        self.assertTrue(row['outputsMatch'])
        self.assertEqual(row['decodeChangePercent'], 0)

    def test_late_delivery_rejected(self):
        data = self.sample_result()
        data['models'][0]['rows'][0]['samples'][0]['tokenDeliveryMs'][-1] = 31
        with self.assertRaises(ValueError): validate(data)

    def test_missing_coverage_rejected(self):
        data = self.sample_result()
        data['models'][0]['rows'] = []
        with self.assertRaises(ValueError): validate(data)

    def test_changed_tokens_reported(self):
        a, b = self.sample_result(), self.sample_result()
        for s in [b['models'][0]['rows'][0]['warmup'], *b['models'][0]['rows'][0]['samples']]: s['generated'][1] = 9
        self.assertEqual(compare(a, b)[0]['firstDifferentToken'], 1)

    def test_changed_request_rejected(self):
        a, b = self.sample_result(), self.sample_result()
        b['requestSha256'] = 'different'
        with self.assertRaises(ValueError): compare(a, b)

    def test_bad_timing_rejected(self):
        for value in (0, float('nan'), float('inf'), -10):
            data = self.sample_result()
            data['models'][0]['rows'][0]['samples'][0]['decodeTps'] = value
            with self.assertRaises(ValueError): validate(data)

    def test_prompts(self):
        config = dict(model=dict(vocab_size=100, eos_token_id=[0, 1]))
        self.assertEqual(random_prompt(config, 4), [32, 48, 59, 16])
        self.assertTrue(all(t not in (0, 1) for t in random_prompt(config, 1024, 0)))

    def test_missing_delivery_rejected(self):
        data = self.sample_result()
        del data['models'][0]['rows'][0]['samples'][0]['tokenDeliveryMs']
        with self.assertRaises(ValueError): validate(data)


if __name__ == '__main__':
    unittest.main()
