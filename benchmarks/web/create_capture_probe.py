"""Generate a tiny, weight-free GQA regression graph; never modify benchmark models."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_manifest import load_onnx


def create(output, window=4):
    if Path(output).exists():
        raise ValueError('Refusing to overwrite an existing file with the test graph')
    if not isinstance(window, int) or window <= 0:
        raise ValueError('Window must be positive')
    onnx = load_onnx()
    helper, types = onnx.helper, onnx.TensorProto
    inputs = [helper.make_tensor_value_info(name, types.FLOAT, shape) for name, shape in [
        ('query', [1, 1, 8]), ('key', [1, 1, 8]), ('value', [1, 1, 8]),
        ('past_key', [1, 1, 16, 8]), ('past_value', [1, 1, 16, 8])]]
    inputs += [helper.make_tensor_value_info('seqlens_k', types.INT32, [1]),
               helper.make_tensor_value_info('total_sequence_length', types.INT32, [1])]
    outputs = [helper.make_tensor_value_info(name, types.FLOAT, shape) for name, shape in [
        ('output', [1, 1, 8]), ('present_key', [1, 1, 16, 8]), ('present_value', [1, 1, 16, 8])]]
    node = helper.make_node('GroupQueryAttention', [i.name for i in inputs], [o.name for o in outputs],
                           domain='com.microsoft', num_heads=1, kv_num_heads=1, local_window_size=window, scale=1.0)
    model = helper.make_model(helper.make_graph([node], 'capture-sliding-window', inputs, outputs),
                              opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--window', type=int, default=4)
    args = parser.parse_args()
    create(args.output, args.window)
