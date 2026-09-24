// Manual GPU regression test, not a throughput benchmark. Usage:
// node web/check_capture_runtime.mjs D:/path/to/verified/ort-web-runtime
import fs from 'node:fs';
import os from 'node:os';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';

if (!process.argv[2]) throw new Error('Usage: node check_capture_runtime.mjs <pinned-artifact-directory>');
const artifactRoot = path.resolve(process.argv[2]);
const metadata = JSON.parse(fs.readFileSync(path.join(artifactRoot, 'build-metadata.json'), 'utf8'));
for (const [name, expected] of Object.entries(metadata.artifacts)) {
  if (path.basename(name) !== name) throw new Error('Invalid artifact path');
  const bytes = fs.readFileSync(path.join(artifactRoot, name));
  assert.equal(bytes.length, expected.size);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), expected.sha256);
}
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-capture-probe-'));
let browser, server;
try {
  const routes = new Map(Object.keys(metadata.artifacts).map(name => ['/' + name, path.join(artifactRoot, name)]));
  for (const window of [4, 262144]) {
    const modelFile = path.join(temporary, `gqa-${window}.onnx`);
    const python = spawnSync(process.env.PYTHON || 'python', [path.join(path.dirname(fileURLToPath(import.meta.url)), 'create_capture_probe.py'), modelFile, '--window', String(window)], { encoding: 'utf8', timeout: 600000 });
    if (python.status !== 0) throw new Error(python.stderr || python.error?.message || 'Could not create regression graph');
    routes.set(`/model-${window}.onnx`, modelFile);
  }
  server = http.createServer((req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    if (req.url === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><script src="/ort.all.min.js"></script>'); return; }
    const file = routes.get(req.url);
    if (!file) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : /\.(mjs|js)$/.test(file) ? 'text/javascript' : 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const candidates = [process.env.LOCALAPPDATA, process.env.ProgramFiles, process.env['ProgramFiles(x86)']]
    .filter(Boolean).map(root => path.join(root, 'Microsoft/Edge/Application/msedge.exe'));
  const executablePath = candidates.find(file => fs.existsSync(file));
  if (!executablePath) throw new Error('Edge Stable is required for this Windows regression test');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--use-webgpu-adapter=default', '--force-high-performance-gpu'] });
  const page = await browser.newPage();
  const replayMessages = [];
  page.on('console', message => { if (message.text().includes('Replaying the captured WebGpuExecutionProvider graph')) replayMessages.push(message.text()); });
  await page.goto('http://127.0.0.1:' + server.address().port);
  const result = await page.evaluate(async () => {
    ort.env.wasm.wasmPaths = location.origin + '/'; ort.env.wasm.numThreads = 1;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const device = await adapter.requestDevice();
    const errors = [], cases = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    for (const window of [4, 262144]) for (const capture of [false, true]) {
      const session = await ort.InferenceSession.create(`/model-${window}.onnx`, { executionProviders: [{ name: 'webgpu', device, validationMode: 'basic' }],
        enableGraphCapture: capture, preferredOutputLocation: 'gpu-buffer', logSeverityLevel: 1 });
      const owned = [];
      const make = (type, dims) => {
        const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(dims.reduce((a, b) => a * b, 1) * 4 / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const tensor = ort.Tensor.fromGpuBuffer(buffer, { dataType: type, dims }); owned.push(tensor); return tensor;
      };
      const feeds = { query: make('float32', [1, 1, 8]), key: make('float32', [1, 1, 8]), value: make('float32', [1, 1, 8]),
        past_key: make('float32', [1, 1, 16, 8]), past_value: make('float32', [1, 1, 16, 8]), seqlens_k: make('int32', [1]), total_sequence_length: make('int32', [1]) };
      const output = make('float32', [1, 1, 8]);
      const fetches = { output, present_key: feeds.past_key, present_value: feeds.past_value };
      const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      for (const length of [2, 4, 8, 12, 3]) {
        device.queue.writeBuffer(feeds.past_value.gpuBuffer, 0, Float32Array.from({ length: 128 }, (_, i) => Math.floor(i / 8) + 1));
        device.queue.writeBuffer(feeds.value.gpuBuffer, 0, new Float32Array(8).fill(length));
        device.queue.writeBuffer(feeds.seqlens_k.gpuBuffer, 0, new Int32Array([length - 1]));
        device.queue.writeBuffer(feeds.total_sequence_length.gpuBuffer, 0, new Int32Array([length]));
        await session.run(feeds, fetches);
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(output.gpuBuffer, 0, readback, 0, 32); device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ); const actual = Array.from(new Float32Array(readback.getMappedRange().slice(0))); readback.unmap();
        cases.push({ window, capture, length, expected: (length + Math.max(1, length - window + 1)) / 2, actual });
      }
      await session.release(); for (const tensor of owned) tensor.gpuBuffer.destroy(); readback.destroy();
    }
    device.destroy(); return { cases, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
  });
  assert.deepEqual(result.errors, []);
  for (const item of result.cases) assert.ok(item.actual.every(value => Math.abs(value - item.expected) < 1e-4), `Window mismatch: capture=${item.capture}, length=${item.length}`);
  assert.ok(replayMessages.length >= 8, 'Expected explicit ORT graph replay log messages');
  console.log(JSON.stringify({ success: true, version: metadata.version, sourceCommit: metadata.repositories.onnxruntime.commit,
    replayMessages: replayMessages.length, ...result }, null, 2));
} finally {
  if (browser) await browser.close();
  if (server?.listening) await new Promise(resolve => server.close(resolve));
  const resolved = fs.realpathSync(temporary), tempRoot = fs.realpathSync(os.tmpdir());
  if (path.dirname(resolved) !== tempRoot || !path.basename(resolved).startsWith('ort-capture-probe-')) throw new Error('Unexpected temporary test directory');
  fs.rmSync(resolved, { recursive: true, force: true });
}
