import { test } from 'node:test';
import assert from 'node:assert/strict';
import { allocateGpuTensor, CaptureGenerator, staticInputPlan, decodeLogitsShape } from './capture-generator.mjs';

class Tensor {
  constructor(type, data, dims) { Object.assign(this, { type, data, dims, location: 'cpu' }); }
  dispose() { this.disposed = true; this.onDispose?.(); }
  static fromGpuBuffer(gpuBuffer, { dataType: type, dims, dispose }) {
    const tensor = new Tensor(type, undefined, dims);
    Object.assign(tensor, { gpuBuffer, location: 'gpu-buffer', onDispose: dispose });
    return tensor;
  }
}
const ort = { Tensor };
function fakeDevice() {
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };
  globalThis.GPUMapMode = { READ: 1 };
  const buffers = [];
  return { buffers, limits: { maxBufferSize: 1e9, maxStorageBufferBindingSize: 1e9 },
    createBuffer({ size }) {
      const buffer = { size, bytes: new ArrayBuffer(size), destroy() { this.destroyed = true; }, async mapAsync() {}, getMappedRange() { return this.bytes; }, unmap() {} };
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      const operations = [];
      return { clearBuffer: buffer => operations.push(() => new Uint8Array(buffer.bytes).fill(0)),
        copyBufferToBuffer: (source, offset, destination, start, size) => operations.push(() => new Uint8Array(destination.bytes, start, size).set(new Uint8Array(source.bytes, offset, size))),
        finish: () => operations };
    },
    queue: { submit: commands => commands.flat().forEach(operation => operation()), async onSubmittedWorkDone() {},
      writeBuffer(buffer, offset, data, start = 0, size) {
        const source = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        new Uint8Array(buffer.bytes, offset, size ?? source.byteLength - start).set(source.subarray(start, size === undefined ? undefined : start + size));
      } },
  };
}
function manifest() {
  return { maxLength: 8192, config: { model: { vocab_size: 4,
    decoder: { inputs: { input_ids: 'ids', attention_mask: 'mask', position_ids: 'pos' }, outputs: { logits: 'logits' } } } },
    sessions: { decoder: { inputs: [{ name: 'ids', type: 'int64', dims: ['batch', 'sequence'] },
      { name: 'mask', type: 'int64', dims: ['batch', 'total'] }, { name: 'pos', type: 'int64', dims: [3, 'batch', 'sequence'] },
      { name: 'past.key', type: 'float16', dims: [1, 2, 'capacity', 8] }, { name: 'past.conv', type: 'float32', dims: [1, 1] }],
      outputs: [{ name: 'logits' }, { name: 'present.key' }, { name: 'present.conv' }] } },
    states: [{ input: 'past.key', output: 'present.key', type: 'float16', dims: [1, 2, 8192, 8], shared: true },
      { input: 'past.conv', output: 'present.conv', type: 'float32', dims: [1, 1], shared: false }] };
}
function fixture() {
  const device = fakeDevice(), m = manifest(), calls = [];
  function session(capture) { return { async release() {}, async run(feeds, fetches) {
    assert.ok(Object.values(feeds).every(tensor => tensor.location === 'gpu-buffer'));
    const size = tensor => tensor.dims.reduce((a, b) => a * b, 1);
    const ids = [...new BigInt64Array(feeds.ids.gpuBuffer.bytes, 0, size(feeds.ids))].map(Number);
    const mask = [...new BigInt64Array(feeds.mask.gpuBuffer.bytes, 0, size(feeds.mask))].map(Number);
    const past = new Float32Array(feeds['past.conv'].gpuBuffer.bytes)[0];
    calls.push({ capture, feeds: { ...feeds }, fetches: { ...fetches }, ids, activeMask: mask.reduce((a, b) => a + b, 0), past,
      positions: [...new BigInt64Array(feeds.pos.gpuBuffer.bytes, 0, size(feeds.pos))] });
    const logits = fetches.logits || allocateGpuTensor(ort, device, 'float32', [1, 1, 4], 'prefill-logits');
    const values = new Float32Array(logits.gpuBuffer.bytes); values.fill(-1); values[(ids.at(-1) + 1) % 4] = 10;
    new Float32Array(fetches['present.conv'].gpuBuffer.bytes)[0] = past + 1;
    return { ...fetches, logits };
  } }; }
  let now = 0;
  const generator = new CaptureGenerator(ort, device, m, { decoder: session(false), decodeDecoder: session(true) }, { now: () => now += 10 });
  return { generator, device, calls, m };
}

test('fixed decode plan binds 8K masks, actual KV shapes and all Qwen position axes', () => {
  const plan = staticInputPlan(manifest(), 'decoder');
  assert.deepEqual(plan.inputs.mask.dims, [1, 8192]);
  assert.deepEqual(plan.inputs.pos.dims, [3, 1, 1]);
  assert.deepEqual(plan.overrides, { batch: 1, sequence: 1, total: 8192, capacity: 8192 });
  assert.deepEqual(decodeLogitsShape([1, 128, 200029], 200029), [1, 1, 200029]);
  assert.throws(() => decodeLogitsShape([1, 128, 4], 5), /logits/);
});

test('decode reuses exact GPU bindings, copies hybrid state, and reads fresh logits each step', async () => {
  const { generator, calls, device } = fixture();
  const sample = await generator.generate([1, 2], 4);
  assert.deepEqual(sample.generated, [3, 0, 1, 2]);
  assert.deepEqual(calls.map(call => call.capture), [false, true, true, true]);
  assert.deepEqual(calls.map(call => call.activeMask), [2, 3, 4, 5]);
  assert.deepEqual(calls.map(call => call.past), [0, 1, 2, 3]);
  assert.deepEqual(calls[1].positions, [2n, 2n, 2n]);
  for (const [name, tensor] of Object.entries(calls[1].feeds)) assert.equal(calls[2].feeds[name], tensor);
  for (const [name, tensor] of Object.entries(calls[1].fetches)) assert.equal(calls[2].fetches[name], tensor);
  assert.notEqual(calls[1].feeds['past.conv'], calls[1].fetches['present.conv']);
  assert.equal(calls[1].feeds['past.key'], calls[1].fetches['present.key']);
  const again = await generator.generate([2], 4);
  assert.deepEqual(again.generated, [3, 0, 1, 2]);
  assert.deepEqual(calls.slice(4).map(call => call.activeMask), [1, 2, 3, 4]);
  assert.deepEqual(calls.slice(4).map(call => call.past), [0, 1, 2, 3]);
  assert.equal(generator.evidence.decodeRuns, 6);
  await generator.dispose();
  assert.ok(device.buffers.every(buffer => buffer.destroyed));
});

test('chunked prefill stays uncaptured even when its final chunk contains one token', async () => {
  const { generator, calls } = fixture();
  await generator.generate([1, 2, 3, 0, 1], 3, { prefillChunkSize: 2 });
  assert.deepEqual(calls.map(call => call.capture), [false, false, false, true, true]);
  assert.deepEqual(calls.map(call => call.activeMask), [2, 4, 5, 6, 7]);
  await generator.dispose();
});

test('readback handles unaligned final-row offsets for odd FP16 vocabularies', async () => {
  const { generator, device } = fixture();
  generator.manifest.config.model.vocab_size = 3;
  const tensor = allocateGpuTensor(ort, device, 'float16', [1, 2, 3], 'odd-vocab');
  new Uint16Array(tensor.gpuBuffer.bytes).set([1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...await generator.readLogits(tensor)], [4, 5, 6]);
  new Uint16Array(tensor.gpuBuffer.bytes).set([7, 8, 9], 3);
  assert.deepEqual([...await generator.readLogits(tensor)], [7, 8, 9]);
  tensor.dispose(); await generator.dispose();
});

test('empty Gemma modality inputs remain GPU-bound without allocating zero-byte buffers', () => {
  const device = fakeDevice();
  const tensor = allocateGpuTensor(ort, device, 'float16', [0, 1536], 'empty-modality');
  assert.equal(tensor.location, 'gpu-buffer');
  assert.equal(tensor.gpuBuffer.size, 16);
  assert.deepEqual(tensor.dims, [0, 1536]);
  tensor.dispose();
});

test('uncaptured embedding keeps tokens on CPU while decoder embeddings retain stable GPU storage', async () => {
  const device = fakeDevice(), m = manifest();
  m.config.model.embedding = { inputs: { input_ids: 'tokens' }, outputs: { inputs_embeds: 'embeddings' } };
  m.config.model.decoder.inputs.inputs_embeds = 'embeddings';
  m.sessions.embedding = { inputs: [{ name: 'tokens', type: 'int64', dims: ['batch', 'sequence'] }],
    outputs: [{ name: 'embeddings', type: 'float16', dims: ['batch', 'sequence', 8] }] };
  m.sessions.decoder.inputs.push({ name: 'embeddings', type: 'float16', dims: ['batch', 'sequence', 8] });
  const session = { async release() {} };
  const generator = new CaptureGenerator(ort, device, m, { decoder: session, decodeDecoder: session, decodeEmbedding: session });
  const token = generator.embeddingFeeds.tokens, output = generator.embeddingFetches.embeddings;
  assert.equal(token.location, 'cpu');
  assert.equal(output.location, 'gpu-buffer');
  assert.equal(generator.decodeFeeds.embeddings, output);
  generator.updateDecodeInputs([3]);
  assert.deepEqual([...token.data], [3n]);
  generator.updateDecodeInputs([2]);
  assert.equal(generator.embeddingFeeds.tokens, token);
  assert.deepEqual([...token.data], [2n]);
  assert.equal(generator.decodeFeeds.embeddings, output);
  assert.equal(generator.evidence.embeddingCapture, false);
  assert.equal(generator.evidence.embeddingInputLocations.tokens, 'cpu');
  await generator.dispose();
  assert.equal(token.disposed, true);
  assert.ok(device.buffers.every(buffer => buffer.destroyed));
});

test('embedding prefill avoids uploading CPU inputs and releases its temporary tensors', async () => {
  const device = fakeDevice(), m = manifest();
  m.config.model.embedding = { inputs: { input_ids: 'tokens' }, outputs: { inputs_embeds: 'embeddings' } };
  m.config.model.decoder.inputs.inputs_embeds = 'embeddings';
  m.sessions.embedding = { inputs: [{ name: 'tokens', type: 'int64', dims: ['batch', 'sequence'] }],
    outputs: [{ name: 'embeddings', type: 'float16', dims: ['batch', 'sequence', 8] }] };
  m.sessions.decoder.inputs.push({ name: 'embeddings', type: 'float16', dims: ['batch', 'sequence', 8] });
  let input, output;
  const embedding = { async release() {}, async run(feeds) {
    input = feeds.tokens; assert.equal(input.location, 'cpu'); assert.deepEqual([...input.data], [1n, 2n]);
    output = allocateGpuTensor(ort, device, 'float16', [1, 2, 8], 'prefill-embedding');
    return { embeddings: output };
  } };
  const decoder = { async release() {}, async run(feeds, fetches) {
    assert.equal(feeds.embeddings, output); assert.equal(feeds.embeddings.location, 'gpu-buffer');
    const logits = allocateGpuTensor(ort, device, 'float32', [1, 1, 4], 'prefill-logits');
    new Float32Array(logits.gpuBuffer.bytes).set([0, 1, 9, 2]);
    return { ...fetches, logits };
  } };
  const stub = { async release() {} };
  const generator = new CaptureGenerator(ort, device, m, { embedding, decoder, decodeEmbedding: stub, decodeDecoder: stub });
  assert.equal(await generator.step([1, 2], [], { phase: 'prefill' }), 2);
  assert.equal(input.disposed, true); assert.equal(output.disposed, true);
  assert.equal(generator.evidence.embeddingPrefillIoBinding, 'cpu-inputs-gpu-output');
  await generator.dispose(); assert.ok(device.buffers.every(buffer => buffer.destroyed));
});
