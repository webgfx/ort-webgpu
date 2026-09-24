import { loadGenerator, randomPrompt } from './generator.mjs';
import { loadCapturedGenerator } from './capture-generator.mjs';
import { ortDeviceDescriptor } from './device.mjs';
import { monitorGpuDevice } from './gpu_failure.mjs';
import { configureGpuSampling } from './gpu-pipeline.mjs';

window.__ortWebPartial = { rows: [], errors: [] };
const progress = message => console.info(`[ort-web] ${message}`);
const average = values => values.reduce((sum, value) => sum + value, 0) / values.length;

async function run() {
  if (!crossOriginIsolated) throw new Error('Browser benchmark requires cross-origin isolation');
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  if (!WebAssembly.Suspending || !WebAssembly.promising) throw new Error('Browser does not support WebAssembly JSPI');
  const response = await fetch('/config');
  if (!response.ok) throw new Error('Benchmark configuration is unavailable');
  const config = await response.json();
  const index = Number(new URL(location.href).searchParams.get('model'));
  const manifest = config.models[index];
  if (!manifest) throw new Error('Requested benchmark model is unavailable');
  const promptFor = pl => config.prompts?.[manifest.name]?.[pl] || randomPrompt(manifest.config, pl, config.seed);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) {
    throw new Error('A hardware WebGPU adapter is required');
  }
  const deviceDescriptor = ortDeviceDescriptor(adapter);
  const device = await adapter.requestDevice(deviceDescriptor);
  window.__ortWebPartial.deviceDescriptor = deviceDescriptor;
  const gpu = monitorGpuDevice(device);
  const info = adapter.info || {};
  window.__ortWebPartial.adapter = { vendor: info.vendor, architecture: info.architecture,
    device: info.device, description: info.description };
  const runtime = globalThis.ort;
  if (!runtime) throw new Error('Pinned ORT Web JavaScript did not load');
  runtime.env.wasm.wasmPaths = new URL('/artifacts/', location.href).href;
  runtime.env.wasm.numThreads = config.wasmThreads;
  let generator;
  try {
    const baseUrl = new URL(`/models/${index}/`, location.href);
    const references = new Map();
    const referenceTokens = Math.min(config.referenceTokens || 0, config.generationLength);
    if (config.ioBinding && referenceTokens >= 2) {
      progress(`${manifest.name}: checking against the uncaptured reference (${referenceTokens} tokens)`);
      generator = await gpu.race(() => loadGenerator(runtime, device, manifest, baseUrl, progress));
      for (const pl of config.promptLengths) {
        const prompt = promptFor(pl);
        const reference = await gpu.race(() => generator.generate(prompt, referenceTokens, { prefillChunkSize: config.prefillChunkSize || 0 }));
        references.set(pl, reference.generated);
      }
      await gpu.race(() => generator.dispose());
      generator = null;
      await device.queue.onSubmittedWorkDone();
    }
    const loadStart = performance.now();
    generator = await gpu.race(() => config.ioBinding
      ? loadCapturedGenerator(runtime, device, manifest, baseUrl, progress, { graphCapture: config.graphCapture, diagnostics: config.captureDiagnostics })
      : loadGenerator(runtime, device, manifest, baseUrl, progress));
    await gpu.race(() => configureGpuSampling(generator, config.gpuSampling || 'cpu'));
    if (generator.evidence) window.__ortWebPartial.executionEvidence = generator.evidence;
    window.__ortWebPartial.providerOptions = generator.providerOptions || generator.evidence?.providerOptions;
    window.__ortWebPartial.sessionLoadMs = performance.now() - loadStart;
    for (const pl of config.promptLengths) {
      const prompt = promptFor(pl);
      progress(`${manifest.name}, prompt ${pl}: warm-up`);
      const options = { prefillChunkSize: config.prefillChunkSize || 0 };
      const warmup = await gpu.race(() => generator.generate(prompt, config.generationLength, options));
      const reference = references.get(pl);
      if (reference && JSON.stringify(warmup.generated.slice(0, referenceTokens)) !== JSON.stringify(reference)) {
        throw new Error(`Optimized/reference token mismatch for ${manifest.name}, prompt ${pl}: expected ${reference}, got ${warmup.generated.slice(0, referenceTokens)}`);
      }
      const samples = [];
      for (let repetition = 0; repetition < config.repetitions; ++repetition) {
        gpu.throwIfFailed();
        progress(`${manifest.name}, prompt ${pl}: repetition ${repetition + 1}/${config.repetitions}`);
        const sample = await gpu.race(() => generator.generate(prompt, config.generationLength, options));
        gpu.throwIfFailed();
        if (JSON.stringify(sample.generated) !== JSON.stringify(warmup.generated)) {
          throw new Error(`Generation is not repeatable after resetting ${manifest.name}'s state`);
        }
        samples.push(sample);
      }
      window.__ortWebPartial.rows.push({ model: manifest.name, pl, prompt, tg: config.generationLength,
        ttftMs: average(samples.map(sample => sample.ttftMs)),
        plTs: average(samples.map(sample => sample.prefillTps)),
        tgTs: average(samples.map(sample => sample.decodeTps)),
        e2eMs: average(samples.map(sample => sample.e2eMs)), samples, warmup,
        validation: { kind: 'deterministic-repeatability', passed: true,
          tokenDelivery: generator.evidence?.samplingDevice === 'gpu' ? 'ordered-cpu-stream' : 'per-step-cpu',
          reference: reference ? { kind: 'same-browser-uncaptured-generator', comparedTokens: referenceTokens, passed: true } : undefined },
        testedAt: new Date().toISOString() });
      const measured = window.__ortWebPartial.rows.at(-1);
      progress(`${manifest.name}, prompt ${pl}: TTFT throughput ${measured.plTs.toFixed(2)} tokens/s; decode ${measured.tgTs.toFixed(2)} TPS`);
    }
    if (generator.evidence) window.__ortWebPartial.executionEvidence = generator.evidence;
    gpu.throwIfFailed();
  } finally {
    // Releasing a session with a lost-device inference can itself wait forever.
    // The host closes this page after receiving the failure, releasing its WASM
    // context and buffers. Completed rows remain available in __ortWebPartial.
    try {
      if (generator && !gpu.failed) await gpu.race(() => generator.dispose());
    } finally {
      gpu.close();
      device.destroy();
    }
  }
}

run().then(() => { window.__ortWebResult = { ...window.__ortWebPartial, success: true }; })
  .catch(error => {
    window.__ortWebResult = { ...window.__ortWebPartial, success: false, errors: [String(error.stack || error)] };
    console.error(error);
  });
