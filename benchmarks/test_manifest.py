from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from web.model_manifest import cache_dimensions, contained_path, state_bindings


class ModelManifestTests(unittest.TestCase):
    def test_sparse_hybrid_layers_use_the_actual_graph_inputs(self):
        decoder = {'head_size': 256,
                   'inputs': {'past_key_names': 'past_key_values.%d.key', 'past_conv_names': 'past.%d.conv'},
                   'outputs': {'present_key_names': 'present.%d.key', 'present_conv_names': 'present.%d.conv'}}
        inputs = [{'name': 'past_key_values.3.key', 'type': 'float16',
                   'dims': ['batch_size', 2, 'past_sequence_length', 'kv_cache_dim']},
                  {'name': 'past.0.conv', 'type': 'float16', 'dims': ['batch_size', 6144, 3]}]
        outputs = [{'name': 'present.3.key'}, {'name': 'present.0.conv'}]
        states = state_bindings(decoder, inputs, outputs, 8192)
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]['dims'], [1, 2, 8192, 256])
        self.assertTrue(states[0]['shared'])
        self.assertEqual(states[1]['dims'], [1, 6144, 3])
        self.assertFalse(states[1]['shared'])

    def test_gemma_static_head_dimensions_override_config_default(self):
        tensor = {'name': 'past_key_values.4.key', 'dims': ['batch', 1, 'past_sequence_len', 512]}
        self.assertEqual(cache_dimensions(tensor, 256, 8192, 'key'), [1, 1, 8192, 512])

    def test_unknown_recurrent_dimensions_are_not_guessed(self):
        tensor = {'name': 'past.0.recurrent', 'dims': ['batch_size', 'unknown_heads', 128, 128]}
        with self.assertRaisesRegex(ValueError, 'Unresolved cache dimension'):
            cache_dimensions(tensor, 256, 8192, 'recurrent')

    def test_model_files_cannot_escape_the_model_directory(self):
        with tempfile.TemporaryDirectory() as root:
            for file in ('../secrets', '/secrets', r'C:\secrets', r'..\secrets', 'https://host/model'):
                with self.subTest(file=file), self.assertRaises(ValueError):
                    contained_path(root, file)
            self.assertEqual(contained_path(root, 'decoder/model.onnx'), Path(root) / 'decoder' / 'model.onnx')


if __name__ == '__main__':
    unittest.main()
