// Runs in a disposable iframe so each model gets a fresh WASM instance/device.
import { loadGenerator, randomPrompt } from './generator.mjs';
import { loadCapturedGenerator } from './capture-generator.mjs';
import { configureGpuSampling } from './gpu-pipeline.mjs';
import { ortDeviceDescriptor } from './device.mjs';
import { monitorGpuDevice } from './gpu_failure.mjs';

const send = (kind, value) => parent.postMessage({ ortBenchmark: true, kind, value }, '*');
const progress = message => send('progress', message);
const average = values => values.reduce((a, b) => a + b, 0) / values.length;
let started = false;
addEventListener('message', async event => {
  if (event.source !== parent || !event.data?.start || started) return;
  started = true;
  const { manifest, files: selectedFiles, runtimeFiles, build, settings } = event.data;
  const files = new Map(selectedFiles), runtime = new Map(runtimeFiles), urls = [];
  const result = { name: manifest.name, rows: [], errors: [], success: false, modelProvenance: manifest };
  let device, gpu, generator, replayMessages = 0;
  const originalLog = console.log, originalError = console.error, originalWarn = console.warn;
  const observe = original => (...args) => {
    if (args.some(value => String(value).includes('Replaying the captured WebGpuExecutionProvider graph'))) replayMessages++;
    original.apply(console, args);
  };
  console.log = observe(originalLog); console.error = observe(originalError); console.warn = observe(originalWarn);
  const blobUrl = (file, type) => { const url = URL.createObjectURL(new Blob([file], { type })); urls.push(url); return url; };
  try {
    const script = document.createElement('script');
    script.src = blobUrl(runtime.get('ort.all.min.js'), 'text/javascript');
    await new Promise((resolve, reject) => { script.onload = resolve; script.onerror = () => reject(Error('Could not load selected ORT JavaScript')); document.head.append(script); });
    const ort = globalThis.ort;
    if (!ort) throw Error('Selected JavaScript did not provide ORT');
    // Explicit blobs prevent CDN/default-runtime fallback. numThreads=1 requires
    // no worker or SharedArrayBuffer transfer, including on file:// pages.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = { mjs: blobUrl(runtime.get('ort-wasm-simd-threaded.jspi.mjs'), 'text/javascript'),
      wasm: blobUrl(runtime.get('ort-wasm-simd-threaded.jspi.wasm'), 'application/wasm') };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter || adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) throw Error('A hardware WebGPU adapter is required');
    result.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description };
    result.deviceDescriptor = ortDeviceDescriptor(adapter);
    device = await adapter.requestDevice(result.deviceDescriptor);
    gpu = monitorGpuDevice(device);
    const source = { read: async name => { const file = files.get(name); if (!file) throw Error(`Missing selected model file: ${name}`); return file; } };
    const references = new Map();
    progress(`${manifest.name}: validating with the uncaptured reference`);
    generator = await gpu.race(() => loadGenerator(ort, device, manifest, source, progress));
    for (const pl of settings.promptLengths) {
      const prompt = settings.prompts?.[pl] || randomPrompt(manifest.config, pl, 42);
      references.set(pl, (await gpu.race(() => generator.generate(prompt, settings.generationLength))).generated);
    }
    await gpu.race(() => generator.dispose()); generator = null;
    await device.queue.onSubmittedWorkDone();
    generator = await gpu.race(() => loadCapturedGenerator(ort, device, manifest, source, progress,
      { graphCapture: settings.graphCapture, diagnostics: settings.diagnostics }));
    await configureGpuSampling(generator, settings.gpuSampling);
    for (const pl of settings.promptLengths) {
      const prompt = settings.prompts?.[pl] || randomPrompt(manifest.config, pl, 42);
      progress(`${manifest.name} · ${pl} input tokens: warm-up`);
      const warmup = await gpu.race(() => generator.generate(prompt, settings.generationLength));
      if (JSON.stringify(warmup.generated) !== JSON.stringify(references.get(pl))) throw Error(`Full reference token mismatch at input ${pl}`);
      const samples = [];
      for (let rep = 0; rep < settings.repetitions; ++rep) {
        progress(`${manifest.name} · ${pl} input tokens: run ${rep + 1}/${settings.repetitions}`);
        const sample = await gpu.race(() => generator.generate(prompt, settings.generationLength));
        if (JSON.stringify(sample.generated) !== JSON.stringify(warmup.generated)) throw Error('Generated tokens changed after reset');
        validateSample(sample, pl, settings.generationLength, generator.evidence.samplingDevice);
        samples.push(sample);
      }
      validateSample(warmup, pl, settings.generationLength, generator.evidence.samplingDevice);
      const ttftMs = average(samples.map(sample => sample.ttftMs));
      const decodeMs = average(samples.map(sample => sample.e2eMs - sample.ttftMs));
      const row = { model: manifest.name, pl, prompt, tg: settings.generationLength, warmup, samples,
        ttftMs, e2eMs: ttftMs + decodeMs, plTs: pl * 1000 / ttftMs, tgTs: (settings.generationLength - 1) * 1000 / decodeMs,
        validation: { kind: 'deterministic-repeatability', passed: true,
          reference: { kind: 'same-browser-uncaptured-generator', passed: true, comparedTokens: settings.generationLength } }, testedAt: new Date().toISOString() };
      result.rows.push(row); send('row', row);
    }
    gpu.throwIfFailed();
    result.executionEvidence = { ...generator.evidence, runtimeReplayMessages: replayMessages,
      runtime: { wasmVariant: 'jspi', source: 'user-selected-sha256-verified-files',
        evidence: 'successful-inference-with-explicit-local-module-and-wasm-URLs',
        // No CDP in the standalone page: do not pretend network responses were observed.
        suppliedArtifacts: Object.entries(build.artifacts).map(([file, artifact]) => ({ file, sha256: artifact.sha256 })) } };
    if (settings.diagnostics && settings.graphCapture && !replayMessages) throw Error('No actual decoder replay messages observed');
    result.success = true;
  } catch (error) { result.errors.push(String(error.stack || error)); }
  finally {
    try { if (generator && !gpu?.failed) await gpu.race(() => generator.dispose()); }
    catch (error) { result.success = false; result.errors.push(`Cleanup failed: ${error.message}`); }
    gpu?.close(); device?.destroy();
    for (const url of urls) URL.revokeObjectURL(url);
    console.log = originalLog; console.error = originalError; console.warn = originalWarn;
  }
  send('result', result);
});

function validateSample(sample, pl, tg, samplingDevice) {
  const positive = value => Number.isFinite(value) && value > 0;
  if (![sample.ttftMs, sample.e2eMs, sample.prefillTps, sample.decodeTps].every(positive)
      || sample.generated.length !== tg || sample.e2eMs <= sample.ttftMs) throw Error('Invalid or incomplete measurement');
  const close = (a, b) => Math.abs(a - b) <= Math.max(1, b) * 1e-6;
  if (!close(sample.prefillTps, pl * 1000 / sample.ttftMs) || !close(sample.decodeTps, (tg - 1) * 1000 / (sample.e2eMs - sample.ttftMs))) throw Error('Token counts do not match elapsed time');
  if (samplingDevice === 'gpu') {
    const times = sample.tokenDeliveryMs;
    if (!Array.isArray(times) || times.length !== tg || !times.every(positive) || times[0] !== sample.ttftMs
        || times.at(-1) > sample.e2eMs || times.some((time, i) => i && time < times[i - 1])) throw Error('Incomplete or prematurely timed CPU token delivery');
  }
}
send('ready');
