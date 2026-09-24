import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSettings, csvResults } from './page-utils.mjs';
import { fields, describeOnnx, stateBindings } from './onnx-metadata.mjs';
import { safePath, directoryFiles, discoverModels, fingerprint, verifyRuntime } from './local-files.mjs';

const defaults = { lengths: '128,512,1024', tokens: '128', repetitions: '5', graphCapture: true, gpuSampling: 'auto' };
test('workload preserves full prompt and 8K capacity, rejecting invalid inputs', () => {
  assert.deepEqual(parseSettings(defaults).promptLengths, [128, 512, 1024]);
  for (const patch of [{ lengths: '128,128' }, { lengths: '0' }, { lengths: '128,' }, { lengths: '8192' }, { tokens: 1 }, { repetitions: 0 }, { repetitions: 1.1 }, { gpuSampling: 'other' }, { graphCapture: false, gpuSampling: 'gpu' }]) {
    assert.throws(() => parseSettings({ ...defaults, ...patch }));
  }
});
test('selected files cannot escape a folder or refer to the network', () => {
  for (const bad of ['../model', '/model', 'C:\\model', 'https://host/model', 'dir/../model', '.', 'a//b']) assert.throws(() => safePath(bad));
  assert.equal(safePath('dir\\weights.bin'), 'dir/weights.bin');
  const files = directoryFiles([{ name: 'genai_config.json', webkitRelativePath: 'models/aion/onnx-webgpu/genai_config.json' }]);
  assert.equal(discoverModels(files)[0].name, 'aion');
  assert(discoverModels(files)[0].files.has('genai_config.json'));
  assert.throws(() => directoryFiles([{ name: 'a' }, { name: 'a' }]), /Duplicate/);
});
test('streamed SHA-256 matches known digest and cancellation is honored', async () => {
  assert.equal((await fingerprint(new Blob(['abc']))).sha256, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  const controller = new AbortController(); controller.abort();
  await assert.rejects(fingerprint(new Blob(['abc']), controller.signal), /abort/i);
});
test('runtime manifest cannot substitute an arbitrary script or runtime', async () => {
  await assert.rejects(verifyRuntime(new Map([['build-metadata.json', new Blob(['{"runtime":"ort-web"}'])]])), /complete/);
});
test('protobuf rejects truncated messages and no-graph models', () => {
  for (const bytes of [[10, 4, 1], [128], [0], [15], Array(12).fill(255)]) assert.throws(() => fields(new Uint8Array(bytes)));
  assert.throws(() => describeOnnx(new Uint8Array()), /no graph/);
});
test('state dimensions come from actual graph IO, including hybrid layers', () => {
  const decoder = { head_size: 256, inputs: { past_key_names: 'past.%d.key', past_conv_names: 'past.%d.conv' }, outputs: { present_key_names: 'present.%d.key', present_conv_names: 'present.%d.conv' } };
  const inputs = [{ name: 'past.3.key', type: 'float16', dims: ['batch', 2, 'past_length', 512] }, { name: 'past.0.conv', type: 'float16', dims: ['batch', 6144, 3] }];
  const outputs = [{ name: 'present.3.key' }, { name: 'present.0.conv' }];
  const states = stateBindings(decoder, inputs, outputs, 8192);
  assert.deepEqual(states.map(s => s.dims), [[1, 2, 8192, 512], [1, 6144, 3]]);
  assert.deepEqual(states.map(s => s.shared), [true, false]);
  assert.throws(() => stateBindings(decoder, inputs, [], 8192), /Missing state/);
});
test('CSV escapes model names and marks incomplete runs', () => {
  const csv = csvResults({ success: false, models: [{ name: '=cmd,"quoted"', rows: [{ pl: 128, tg: 128, plTs: 100, tgTs: 20 }] }] });
  assert(csv.includes('"\'=cmd,""quoted"""')); assert(csv.includes('"false"'));
});
