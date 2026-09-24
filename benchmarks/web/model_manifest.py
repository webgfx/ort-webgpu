"""Describe the unmodified native ONNX exports for the browser generator.

Read graph IO, not architecture-wide guesses: Qwen has hybrid layers and Gemma
uses different key dimensions and fewer cache inputs than hidden layers.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


TYPE_NAMES = {1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32', 7: 'int64',
              9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32', 13: 'uint64'}


def load_onnx():
    if importlib.util.find_spec('onnx') is None:
        # Native-only benchmark devices may not have ONNX's Python parser.
        # Install in this task's ignored cache, never upgrade a user's global
        # environment or their model-conversion dependencies.
        cache = Path(__file__).resolve().parents[2] / 'gitignore' / f'ort-web-python-{sys.version_info.major}.{sys.version_info.minor}'
        sys.path.insert(0, str(cache))
        importlib.invalidate_caches()
        if importlib.util.find_spec('onnx') is None:
            print('[ort-web] Installing the model parser into the task-local dependency cache', file=sys.stderr, flush=True)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
                            '--target', str(cache), '-r', str(Path(__file__).with_name('requirements.txt'))],
                           stdout=sys.stderr, stderr=sys.stderr, timeout=600, check=True)
            importlib.invalidate_caches()
    return importlib.import_module('onnx')


def contained_path(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace('\\', '/')
    if (not relative or PurePosixPath(relative).is_absolute() or ':' in relative
            or '..' in PurePosixPath(relative).parts):
        raise ValueError(f'Unsafe model file path: {relative}')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Model artifact is outside its model directory')
    return resolved


def fingerprint(filename):
    filename = Path(filename)
    digest = hashlib.sha256()
    with filename.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'size': filename.stat().st_size, 'sha256': digest.hexdigest()}


def tensor_metadata(value):
    tensor = value.type.tensor_type
    if tensor.elem_type not in TYPE_NAMES:
        raise ValueError(f'Unsupported browser tensor type for {value.name}: {tensor.elem_type}')
    return {'name': value.name, 'type': TYPE_NAMES[tensor.elem_type],
            'dims': [dim.dim_value if dim.HasField('dim_value') else dim.dim_param
                     for dim in tensor.shape.dim]}


def cache_dimensions(tensor, head_size, max_length, kind):
    dims = tensor['dims'][:]
    for index, dim in enumerate(dims):
        if isinstance(dim, int) and dim > 0:
            continue
        if index == 0:
            dims[index] = 1
        elif kind in ('key', 'value') and index == 2:
            dims[index] = max_length
        elif kind in ('key', 'value') and index == 3 and dim == 'kv_cache_dim':
            dims[index] = head_size
        else:
            raise ValueError(f'Unresolved cache dimension {dim!r} for {tensor["name"]}')
    if kind in ('key', 'value') and (len(dims) != 4 or dims[2] != max_length):
        raise ValueError(f'KV cache {tensor["name"]} does not support the requested {max_length}-token capacity')
    return dims


def state_bindings(decoder, graph_inputs, graph_outputs, max_length):
    outputs = {item['name'] for item in graph_outputs}
    bindings = []
    for kind in ('key', 'value', 'conv', 'recurrent'):
        input_template = decoder['inputs'].get(f'past_{kind}_names')
        output_template = decoder['outputs'].get(f'present_{kind}_names')
        if not input_template and not output_template:
            continue
        if not input_template or not output_template or input_template.count('%d') != 1:
            raise ValueError(f'Invalid {kind} state templates in genai_config.json')
        pattern = '^' + re.escape(input_template).replace('%d', r'(\d+)') + '$'
        for tensor in graph_inputs:
            match = re.match(pattern, tensor['name'])
            if not match:
                continue
            output = output_template.replace('%d', match[1])
            if output not in outputs:
                raise ValueError(f'Missing state output {output}')
            bindings.append({'input': tensor['name'], 'output': output, 'kind': kind,
                             'type': tensor['type'],
                             'dims': cache_dimensions(tensor, decoder['head_size'], max_length, kind),
                             'shared': kind in ('key', 'value')})
    if not bindings:
        raise ValueError('Model has no recognized autoregressive state inputs')
    return bindings


def describe_graph(model_root, filename):
    # Imported lazily: unit tests for bindings do not require an ONNX install.
    onnx = load_onnx()
    model_file = contained_path(model_root, filename)
    graph = onnx.load(str(model_file), load_external_data=False).graph
    # External tensors can also occur in Constant nodes/subgraphs. Use ONNX's
    # tensor traversal rather than inspecting only top-level initializers.
    from onnx.external_data_helper import _get_all_tensors
    wrapper = onnx.helper.make_model(graph)
    locations = sorted({entry.value for tensor in _get_all_tensors(wrapper)
                        for entry in tensor.external_data if entry.key == 'location'})
    external = []
    for location in locations:
        artifact = contained_path(model_file.parent, location)
        if not artifact.is_relative_to(Path(model_root).resolve()):
            raise ValueError('External weights are outside their model directory')
        external.append({'path': location, 'file': artifact.relative_to(model_root).as_posix(),
                         **fingerprint(artifact)})
    windows = sorted({attr.i for node in graph.node if node.op_type == 'GroupQueryAttention'
                      for attr in node.attribute if attr.name == 'local_window_size' and attr.i > 0})
    features = {'slidingWindowAttention': bool(windows), 'localAttentionWindows': windows}
    return {'file': model_file.relative_to(model_root).as_posix(), **fingerprint(model_file), 'features': features,
            'inputs': [tensor_metadata(value) for value in graph.input],
            'outputs': [tensor_metadata(value) for value in graph.output], 'externalData': external}


def describe_model(model_root, name, max_length=8192):
    model_root = Path(model_root).resolve()
    config_file = model_root / 'genai_config.json'
    config = json.loads(config_file.read_text('utf-8-sig'))
    model = config['model']
    if not isinstance(max_length, int) or max_length <= 0 or max_length > model['context_length']:
        raise ValueError('Invalid KV cache capacity for this model')
    if not config['search'].get('past_present_share_buffer'):
        raise ValueError('Browser benchmark currently requires the shared KV cache export')
    sessions = {name: describe_graph(model_root, model[name]['filename'])
                for name in ('decoder', 'embedding') if name in model}
    decoder = model['decoder']
    graph = sessions['decoder']
    states = state_bindings(decoder, graph['inputs'], graph['outputs'], max_length)
    recognized = {state['input'] for state in states}
    recognized.update(decoder['inputs'].get(name) for name in (
        'input_ids', 'attention_mask', 'position_ids', 'inputs_embeds'))
    unknown = [value['name'] for value in graph['inputs'] if value['name'] not in recognized]
    if unknown:
        raise ValueError(f'Unsupported decoder inputs: {unknown}')
    if decoder['outputs']['logits'] not in {value['name'] for value in graph['outputs']}:
        raise ValueError('Missing logits output')
    return {'schemaVersion': 1, 'name': name, 'maxLength': max_length,
            'config': config, 'configArtifact': {'file': 'genai_config.json', **fingerprint(config_file)},
            'sessions': sessions, 'states': states}
