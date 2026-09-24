// Read ONNX protobuf metadata only. Tensor payloads are skipped, never decoded.
// This lets the standalone page describe existing exports without Python/ONNX.
const decoder = new TextDecoder('utf-8', { fatal: true });
const types = { 1: 'float32', 2: 'uint8', 3: 'int8', 6: 'int32', 7: 'int64', 9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32', 13: 'uint64' };

export function fields(bytes) {
  let offset = 0;
  const result = [];
  const varint = () => {
    let value = 0n;
    for (let i = 0; i < 10; ++i) {
      if (offset >= bytes.length) throw Error('Truncated ONNX protobuf');
      const byte = bytes[offset++];
      if (i === 9 && byte > 1) throw Error('Invalid ONNX varint');
      value |= BigInt(byte & 127) << BigInt(i * 7);
      if (!(byte & 128)) return value;
    }
    throw Error('Invalid ONNX varint');
  };
  while (offset < bytes.length) {
    const tag = varint();
    const id = Number(tag >> 3n), wire = Number(tag & 7n);
    if (!id || id > 0x1fffffff) throw Error('Invalid ONNX field');
    if (wire === 0) result.push({ id, wire, value: varint() });
    else {
      const length = wire === 2 ? Number(varint()) : wire === 1 ? 8 : wire === 5 ? 4 : NaN;
      if (!Number.isSafeInteger(length) || length < 0 || length > bytes.length - offset) throw Error('Invalid ONNX field length');
      result.push({ id, wire, bytes: bytes.subarray(offset, offset + length) });
      offset += length;
    }
  }
  return result;
}
const all = (items, id) => items.filter(item => item.id === id);
const one = (items, id) => items.find(item => item.id === id);
const nested = item => fields(item?.bytes || new Uint8Array());
const text = item => item ? decoder.decode(item.bytes) : '';
const integer = item => {
  if (item?.wire !== 0) throw Error('Missing ONNX integer');
  const value = Number(BigInt.asIntN(64, item.value));
  if (!Number.isSafeInteger(value)) throw Error('ONNX dimension exceeds safe integer range');
  return value;
};

function tensorInfo(item) {
  const value = nested(item);
  const tensor = nested(one(nested(one(value, 2)), 1));
  const type = types[integer(one(tensor, 1))];
  if (!type) throw Error('Unsupported ONNX tensor type');
  const dims = all(nested(one(tensor, 2)), 1).map(dim => {
    const info = nested(dim);
    return one(info, 1) ? integer(one(info, 1)) : text(one(info, 2));
  });
  return { name: text(one(value, 1)), type, dims };
}

export function describeOnnx(bytes) {
  const graphField = one(fields(bytes), 7);
  if (!graphField) throw Error('ONNX model has no graph');
  const graph = nested(graphField), external = new Set(), windows = new Set();
  const tensor = item => {
    for (const entry of all(nested(item), 13)) {
      const pair = nested(entry);
      if (text(one(pair, 1)) === 'location') external.add(text(one(pair, 2)));
    }
  };
  const sparse = item => { const data = nested(item); for (const id of [1, 2]) if (one(data, id)) tensor(one(data, id)); };
  function walk(items, depth = 0) {
    if (depth > 32) throw Error('ONNX subgraph nesting limit exceeded');
    for (const item of all(items, 5)) tensor(item);
    for (const item of all(items, 15)) sparse(item);
    for (const node of all(items, 1)) {
      const data = nested(node);
      for (const attribute of all(data, 5)) {
        const attr = nested(attribute);
        if (text(one(data, 4)) === 'GroupQueryAttention' && text(one(attr, 1)) === 'local_window_size') {
          const window = integer(one(attr, 3));
          if (window > 0) windows.add(window);
        }
        for (const id of [5, 10]) for (const item of all(attr, id)) tensor(item);
        for (const id of [6, 11]) for (const item of all(attr, id)) walk(nested(item), depth + 1);
        for (const id of [22, 23]) for (const item of all(attr, id)) sparse(item);
      }
    }
  }
  walk(graph);
  return { inputs: all(graph, 11).map(tensorInfo), outputs: all(graph, 12).map(tensorInfo),
    external: [...external].sort(), features: { slidingWindowAttention: windows.size > 0, localAttentionWindows: [...windows].sort((a, b) => a - b) } };
}

export function stateBindings(decoderConfig, inputs, outputs, maxLength) {
  const states = [], names = new Set(outputs.map(output => output.name));
  for (const kind of ['key', 'value', 'conv', 'recurrent']) {
    const inputTemplate = decoderConfig.inputs[`past_${kind}_names`], outputTemplate = decoderConfig.outputs[`present_${kind}_names`];
    if (!inputTemplate && !outputTemplate) continue;
    if (!inputTemplate || !outputTemplate || inputTemplate.split('%d').length !== 2 || outputTemplate.split('%d').length !== 2) throw Error(`Invalid ${kind} state templates`);
    const pattern = new RegExp('^' + inputTemplate.split('%d').map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('(\\d+)') + '$');
    for (const input of inputs) {
      const match = input.name.match(pattern);
      if (!match) continue;
      const output = outputTemplate.replace('%d', match[1]);
      if (!names.has(output)) throw Error(`Missing state output ${output}`);
      const shared = kind === 'key' || kind === 'value';
      const dims = input.dims.map((dim, i) => {
        if (Number.isSafeInteger(dim) && dim > 0) return dim;
        if (i === 0) return 1;
        if (shared && i === 2) return maxLength;
        if (shared && i === 3 && dim === 'kv_cache_dim') return decoderConfig.head_size;
        throw Error(`Unresolved state dimension ${dim} for ${input.name}`);
      });
      if (shared && (dims.length !== 4 || dims[2] !== maxLength)) throw Error(`State ${input.name} cannot use the requested KV capacity`);
      states.push({ input: input.name, output, kind, type: input.type, dims, shared });
    }
  }
  if (!states.length) throw Error('No recognized autoregressive state inputs');
  return states;
}
