import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cpuInputs, greedyToken, halfToFloat, randomPrompt, TextGenerator, modelProviderOptions } from './generator.mjs';

test('greedy selection decodes half floats, picks the last logits row, and rejects non-finite data', () => {
  assert.equal(halfToFloat(0x3c00), 1);
  assert.equal(halfToFloat(0xc000), -2);
  assert.equal(greedyToken(new Uint16Array([0x4000, 0x3c00, 0x3c00, 0x4200]), 'float16', 2), 1);
  assert.equal(greedyToken(new Float32Array([8, 8]), 'float32', 2), 0);
  assert.equal(greedyToken(new Float32Array([1, 10, 20, 2]), 'float32', 4, [1, 2]), 3);
  assert.throws(() => greedyToken(new Float32Array([NaN, 1]), 'float32', 2), /Non-finite/);
  assert.throws(() => greedyToken(new Uint16Array([0x7c00, 0x3c00]), 'float16', 2), /Non-finite/);
  assert.throws(() => greedyToken(new Float32Array([1, 2, 3]), 'float32', 2), /complete vocabulary/);
});

function withFloat16Constructor(constructor, run) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Float16Array');
  Object.defineProperty(globalThis, 'Float16Array', { configurable: true, writable: true, value: constructor });
  try { return run(); }
  finally {
    if (descriptor) Object.defineProperty(globalThis, 'Float16Array', descriptor);
    else delete globalThis.Float16Array;
  }
}

test('native FP16 conversion preserves the buffer subview and does not mutate logits', () => {
  const calls = [];
  // Node versions without Float16Array still exercise the constructor branch.
  // Real native-view behavior is covered by generator.browser.cjs in Edge.
  class Float16View extends Float32Array {
    constructor(buffer, byteOffset, length) {
      calls.push({ buffer, byteOffset, length });
      super(Array.from(new Uint16Array(buffer, byteOffset, length), halfToFloat));
    }
  }
  const storage = new Uint16Array([0x7c00, 0x4200, 0x3c00, 0xbc00, 0x0001, 0x7c00]);
  const view = storage.subarray(1, 5), before = [...storage];
  withFloat16Constructor(Float16View, () => {
    assert.equal(greedyToken(view, 'float16', 2), 1);
    assert.equal(greedyToken(view, 'float16', 2, [1]), 0);
    assert.equal(greedyToken(new Float32Array([2, 3]), 'float32', 2), 1);
  });
  assert.equal(calls.length, 2);
  for (const call of calls) {
    assert.equal(call.buffer, view.buffer);
    assert.equal(call.byteOffset, view.byteOffset);
    assert.equal(call.length, view.length);
  }
  assert.deepEqual([...storage], before);
});

test('fallback preserves FP16 signs, ties, subnormals, EOS suppression and validation', () => {
  withFloat16Constructor(undefined, () => {
    assert.equal(greedyToken(new Uint16Array([0xbc00, 0xc000]), 'float16', 2), 0);
    assert.equal(greedyToken(new Uint16Array([0x8000, 0x0000]), 'float16', 2), 0);
    assert.equal(greedyToken(new Uint16Array([0x0000, 0x0001]), 'float16', 2), 1);
    assert.equal(greedyToken(new Uint16Array([0x4000, 0x3c00]), 'float16', 2, [0]), 1);
    assert.throws(() => greedyToken(new Uint16Array([0x3c00]), 'float16', 1, [0]), /No valid next token/);
    for (const value of [0x7c00, 0xfc00, 0x7e00]) {
      assert.throws(() => greedyToken(new Uint16Array([value, 0]), 'float16', 2, [0]), /Non-finite/);
    }
    // Only the last row contributes, so an earlier non-finite row is ignored.
    assert.equal(greedyToken(new Uint16Array([0x7c00, 0xfc00, 0, 1]), 'float16', 2), 1);
    for (const size of [0, -1, 1.5, 3]) {
      assert.throws(() => greedyToken(new Uint16Array([0, 1]), 'float16', size), /complete vocabulary/);
    }
  });
});

test('random prompts are deterministic and exclude padding and modality tokens', () => {
  const config = { model: { vocab_size: 12, pad_token_id: 0, eos_token_id: [1, 2], image_token_id: 3 } };
  const tokens = randomPrompt(config, 200);
  assert.deepEqual(tokens, randomPrompt(config, 200));
  assert.ok(tokens.every(token => token >= 4 && token < 12));
  assert.equal(randomPrompt(config, 4, 0).length, 4);
});

test('candidate-only suppression preserves dense ties, masked maxima and complete validation', () => {
  for (let seed = 0; seed < 100; seed++) {
    const values = Float32Array.from({ length: 257 }, (_, i) => ((i * 71 + seed * 37) % 41) - 20);
    const suppressed = Array.from({ length: 30 }, (_, i) => (i * 17 + seed) % values.length);
    let expected = -1, best = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (!suppressed.includes(i) && values[i] > best) { best = values[i]; expected = i; }
    }
    assert.equal(greedyToken(values, 'float32', values.length, suppressed), expected);
  }
  assert.equal(greedyToken(new Float32Array([100, 100, 8, 8]), 'float32', 4, [0, 1]), 2);
  for (const bad of [NaN, Infinity, -Infinity]) {
    assert.throws(() => greedyToken(new Float32Array([100, bad]), 'float32', 2, [1]), /Non-finite/);
  }
});

class Tensor {
  constructor(type, data, dims) { Object.assign(this, { type, data, dims, disposed: false }); }
  async getData() { return this.data; }
  dispose() { this.disposed = true; this.onDispose?.(); }
  static fromGpuBuffer(gpuBuffer, { dataType, dims, dispose }) {
    const tensor = new Tensor(dataType, null, dims);
    tensor.gpuBuffer = gpuBuffer;
    tensor.onDispose = dispose;
    return tensor;
  }
}
const ort = { Tensor };

test('preserves model-specific WebGPU options while capture and validation remain explicit', () => {
  const manifest = { config: { model: { decoder: { session_options: { provider_options: [{ webgpu: {
    multiRotaryCacheConcatOffset: '4096', enableGraphCapture: '0', validationMode: 'disabled',
  } }] } } } } };
  assert.deepEqual(modelProviderOptions(manifest, 'decoder'), { multiRotaryCacheConcatOffset: '4096', validationMode: 'basic' });
  assert.equal(manifest.config.model.decoder.session_options.provider_options[0].webgpu.enableGraphCapture, '0');
});

test('Qwen text positions replicate all three MRoPE axes with the accumulated offset', () => {
  const section = { inputs: { input_ids: 'ids', position_ids: 'pos', attention_mask: 'mask' } };
  const graph = { inputs: [{ name: 'ids', type: 'int64', dims: ['batch', 'seq'] },
    { name: 'pos', type: 'int64', dims: [3, 'batch', 'seq'] },
    { name: 'mask', type: 'int64', dims: ['batch', 'total'] }] };
  const { feeds } = cpuInputs(ort, section, graph, [8, 9], 3);
  assert.deepEqual(feeds.pos.dims, [3, 1, 2]);
  assert.deepEqual([...feeds.pos.data], [3n, 4n, 3n, 4n, 3n, 4n]);
  assert.deepEqual(feeds.mask.dims, [1, 5]);
  assert.deepEqual([...feeds.mask.data], [1n, 1n, 1n, 1n, 1n]);
});

test('Gemma text embedding uses empty image/audio tensors and forwards its actual embedding output', () => {
  const embedding = { inputs: { input_ids: 'ids', image_features: 'image', audio_features: 'audio' } };
  const graph = { inputs: [{ name: 'ids', type: 'int64', dims: ['batch', 'seq'] },
    { name: 'image', type: 'float16', dims: ['images', 1536] },
    { name: 'audio', type: 'float16', dims: ['audio', 1536] }] };
  const prepared = cpuInputs(ort, embedding, graph, [2, 5], 0);
  assert.deepEqual(prepared.feeds.image.dims, [0, 1536]);
  assert.deepEqual(prepared.feeds.audio.dims, [0, 1536]);
  const output = new Tensor('float16', new Uint16Array(3072), [1, 2, 1536]);
  const decoder = cpuInputs(ort, { inputs: { inputs_embeds: 'embeds' } },
    { inputs: [{ name: 'embeds', type: 'float16', dims: ['batch', 'seq', 1536] }] }, [2, 5], 0, output);
  assert.equal(decoder.feeds.embeds, output);
  assert.deepEqual(decoder.temporary, []);
});

function fixture() {
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 };
  const cleared = [];
  const device = { limits: { maxBufferSize: 1e9, maxStorageBufferBindingSize: 1e9 },
    createBuffer: ({ size }) => ({ size, destroyed: false, destroy() { this.destroyed = true; } }),
    createCommandEncoder: () => ({ clearBuffer: buffer => cleared.push(buffer), finish: () => ({}) }),
    queue: { submit() {}, async onSubmittedWorkDone() {} } };
  const manifest = { maxLength: 8192, config: { model: { vocab_size: 4,
    decoder: { inputs: { input_ids: 'ids', attention_mask: 'mask' }, outputs: { logits: 'logits' } } } },
    sessions: { decoder: { inputs: [{ name: 'ids', type: 'int64', dims: ['batch', 'seq'] },
      { name: 'mask', type: 'int64', dims: ['batch', 'total'] }] } },
    states: [{ input: 'past.3.key', output: 'present.3.key', type: 'float16', dims: [1, 2, 8192, 8], shared: true },
      { input: 'past.0.conv', output: 'present.0.conv', type: 'float16', dims: [1, 32, 3], shared: false }] };
  const calls = [];
  const decoder = { async run(feeds, fetches) {
    calls.push({ feeds: { ...feeds }, fetches: { ...fetches } });
    return { ...fetches, logits: new Tensor('float32', new Float32Array([0, 1, 3, 2]), [1, 1, 4]) };
  }, async release() {} };
  let tick = 0;
  const generator = new TextGenerator(ort, device, manifest, { decoder }, () => (tick += 10));
  return { generator, device, calls, cleared };
}

test('generation feeds previous output tokens back, shares KV, and ping-pongs recurrent state', async () => {
  const { generator, calls, cleared } = fixture();
  const sample = await generator.generate([1, 3], 3);
  assert.deepEqual(sample.generated, [2, 2, 2]);
  assert.equal(calls.length, 3);
  assert.deepEqual([...calls[0].feeds.ids.data], [1n, 3n]);
  assert.deepEqual([...calls[1].feeds.ids.data], [2n]);
  assert.deepEqual(calls[2].feeds.mask.dims, [1, 4]);
  assert.equal(calls[0].feeds['past.3.key'], calls[0].fetches['present.3.key']);
  assert.notEqual(calls[0].feeds['past.0.conv'], calls[0].fetches['present.0.conv']);
  assert.equal(calls[1].feeds['past.0.conv'], calls[0].fetches['present.0.conv']);
  assert.equal(calls[1].fetches['present.0.conv'], calls[0].feeds['past.0.conv']);
  assert.equal(sample.ttftMs, 10);
  assert.equal(sample.prefillTps, 200);
  assert.equal(sample.decodeTps, 200); // two decoded tokens, not three
  assert.equal(sample.e2eMs, 20);
  assert.equal(cleared.length, 2);
  assert.equal(calls[0].feeds['past.3.key'].disposed, false);
  assert.equal(calls[0].feeds.ids.disposed, true);
  await generator.generate([1, 3], 3);
  assert.equal(cleared.length, 4);
  assert.deepEqual(calls[3].feeds.mask.dims, [1, 2]);
  const buffers = generator.states.flatMap(state => [state.past.gpuBuffer, state.present.gpuBuffer]);
  await generator.dispose();
  assert.ok(buffers.every(buffer => buffer.destroyed));
});

test('cache overflow and missing measurements cannot become successful samples', async () => {
  const { generator } = fixture();
  await assert.rejects(generator.generate(Array(8190).fill(1), 3), /KV cache/);
  await assert.rejects(generator.generate([1], 1), /Generation length/);
  generator.now = () => 0;
  await assert.rejects(generator.generate([1], 3), /Invalid measured/);
  await generator.dispose();
});

test('fixed-length generation masks every model EOS ID as native min_length does', async () => {
  const { generator } = fixture();
  generator.manifest.config.model.eos_token_id = [1, 2];
  const sample = await generator.generate([1, 3], 3);
  assert.deepEqual(sample.generated, [3, 3, 3]);
  await generator.dispose();
});

test('chunked prefill consumes the entire prompt without emitting intermediate predictions', async () => {
  const { generator, calls, cleared } = fixture();
  generator.manifest.config.model.decoder.inputs.position_ids = 'pos';
  generator.manifest.sessions.decoder.inputs.push({ name: 'pos', type: 'int64', dims: [3, 'batch', 'seq'] });
  const sample = await generator.generate([1, 3, 1, 3, 1], 3, { prefillChunkSize: 2 });
  assert.deepEqual(sample.generated, [2, 2, 2]);
  assert.equal(calls.length, 5); // three prefill chunks, then two decode calls
  assert.deepEqual(calls.map(call => [...call.feeds.ids.data]), [[1n, 3n], [1n, 3n], [1n], [2n], [2n]]);
  assert.deepEqual(calls.map(call => call.feeds.mask.dims[1]), [2, 4, 5, 6, 7]);
  assert.deepEqual([...calls[1].feeds.pos.data], [2n, 3n, 2n, 3n, 2n, 3n]);
  assert.equal(generator.pastLength, 7);
  assert.equal(cleared.length, 2); // reset once, not between chunks
  assert.equal(sample.prefillTps, 500); // all five prompt tokens are counted
  await generator.dispose();
});
