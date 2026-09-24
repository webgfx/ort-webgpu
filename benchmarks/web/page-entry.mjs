import { directoryFiles, discoverModels, verifyRuntime, describeLocalModel, readJson, fingerprint } from './local-files.mjs';
import { randomPrompt } from './generator.mjs';
import { parseSettings, csvResults } from './page-utils.mjs';
// Replaced with the self-contained iframe script by build-page.mjs.
const workerSource = __BENCHMARK_WORKER_SOURCE__;
const $ = id => document.getElementById(id);
let runtimeFiles = new Map(), models = [], result, controller, frame, cancelFrame;
let importedRequest, importedRequestHash;
const log = message => {
  const line = document.createElement('div'); line.textContent = message;
  $('log').append(line); $('log').scrollTop = $('log').scrollHeight;
  $('status').textContent = message;
};
function checkBrowser() {
  const available = isSecureContext && !!navigator.gpu && !!WebAssembly.Suspending && !!WebAssembly.promising;
  $('environment').textContent = available ? `WebGPU + JSPI available · ${crossOriginIsolated ? 'isolated' : 'single-thread local'} mode` : 'Use a current Edge or Chrome with WebGPU and WebAssembly JSPI enabled.';
  $('run').disabled = !available || !runtimeFiles.size || !models.length;
}
function clearRequest() { importedRequest = importedRequestHash = undefined; $('request-info').textContent = 'Optional: reuse exact prompts from a native benchmark request.'; }
function renderModels() {
  $('models').replaceChildren();
  models.forEach((entry, i) => {
    const label = document.createElement('label'); label.className = 'model-choice';
    const box = document.createElement('input'); box.type = 'checkbox'; box.value = i; box.checked = true;
    const name = document.createElement('span'); name.textContent = entry.name;
    label.append(box, name); $('models').append(label);
  });
  checkBrowser();
}
$('runtime-files').addEventListener('change', event => {
  try { runtimeFiles = directoryFiles(event.target.files); $('runtime-info').textContent = `${runtimeFiles.size} files selected. SHA-256 verification runs before inference.`; checkBrowser(); }
  catch (error) { runtimeFiles.clear(); log(error.message); checkBrowser(); }
});
$('model-files').addEventListener('change', async event => {
  try {
    clearRequest();
    const files = directoryFiles(event.target.files);
    const folder = event.target.files[0]?.webkitRelativePath.split('/')[0];
    const entries = discoverModels(files, folder);
    if (!entries.length) throw Error('Select an onnx-webgpu model folder, or a parent folder containing genai_config.json files');
    // A direct onnx-webgpu selection has no parent name; use the optional UI name.
    for (const entry of entries) if (entry.name === 'onnx-webgpu') {
      const config = await readJson(entry.files.get('genai_config.json'));
      entry.name = config.model?.type || 'Selected model';
    }
    models = entries; renderModels(); $('model-info').textContent = `${entries.length} model(s). Files stay on this machine.`;
  } catch (error) { models = []; renderModels(); log(error.message); }
});
$('request-file').addEventListener('change', async event => {
  try {
    const file = event.target.files[0]; if (!file) return;
    const request = await readJson(file);
    if (!Array.isArray(request.models) || request.models.length !== models.length || request.maxLength !== 8192
        || request.batchSize !== 1 || request.sampling !== 'greedy' || request.suppressEos !== true || request.prefillChunkSize !== 0) throw Error('Request must match the selected models and batch-one greedy, full-prompt, 8K benchmark');
    for (const [i, entry] of request.models.entries()) {
      if (!models.some(model => model.name === entry.name)) {
        if (models.length === 1) models[i].name = entry.name;
        else throw Error(`Select the models parent folder to match request model ${entry.name}`);
      }
    }
    importedRequest = request; importedRequestHash = (await fingerprint(file)).sha256;
    $('lengths').value = request.promptLengths.join(','); $('tokens').value = request.generationLength; $('repetitions').value = request.repetitions;
    renderModels(); $('request-info').textContent = `Loaded ${file.name}; model hashes and prompts will be checked before running.`;
  } catch (error) { clearRequest(); log(error.message); }
});
const busy = value => { $('configuration').disabled = value; $('run').disabled = value; $('stop').disabled = !value; if (!value) checkBrowser(); };
function appendRow(row) {
  const tr = document.createElement('tr');
  for (const value of [row.model, row.pl, row.tg, row.plTs.toFixed(2), row.tgTs.toFixed(2), 'Validated']) {
    const td = document.createElement('td'); td.textContent = value; tr.append(td);
  }
  $('rows').append(tr); $('empty').hidden = true;
}
function runModel(entry, manifest, build, settings, partial) {
  return new Promise((resolve, reject) => {
    frame = document.createElement('iframe'); frame.title = 'Isolated benchmark runtime'; frame.hidden = true;
    const current = frame;
    const cleanup = () => { clearTimeout(timer); removeEventListener('message', receive); current.remove(); if (frame === current) frame = null; cancelFrame = null; };
    const timer = setTimeout(() => { cleanup(); reject(Error('Model timed out after 30 minutes')); }, 30 * 60 * 1000);
    cancelFrame = () => { cleanup(); reject(new DOMException('Run cancelled', 'AbortError')); };
    function receive(event) {
      if (event.source !== current.contentWindow || !event.data?.ortBenchmark) return;
      const { kind, value } = event.data;
      if (kind === 'ready') current.contentWindow.postMessage({ start: true, manifest, files: [...entry.files], runtimeFiles: [...runtimeFiles], build, settings }, '*');
      if (kind === 'progress') log(value);
      if (kind === 'row') { partial.rows.push(value); appendRow(value); }
      if (kind === 'result') { cleanup(); resolve(value); }
    }
    addEventListener('message', receive);
    current.srcdoc = '<!doctype html><meta charset="utf-8"><script>' + workerSource.replace(/<\/script/gi, '<\\/script') + '</script>';
    document.body.append(current);
  });
}
$('run').addEventListener('click', async () => {
  controller = new AbortController(); result = null;
  $('rows').replaceChildren(); $('log').replaceChildren(); $('empty').hidden = false;
  $('json').disabled = $('csv').disabled = true; busy(true);
  try {
    const settings = parseSettings({ lengths: $('lengths').value, tokens: $('tokens').value, repetitions: $('repetitions').value,
      graphCapture: $('capture').checked, gpuSampling: $('sampling').value, diagnostics: $('diagnostics').checked });
    const selected = [...$('models').querySelectorAll('input:checked')].map(box => models[Number(box.value)]);
    if (!selected.length) throw Error('Select at least one model');
    if (importedRequest && (selected.length !== importedRequest.models.length || JSON.stringify(settings.promptLengths) !== JSON.stringify(importedRequest.promptLengths)
        || settings.generationLength !== importedRequest.generationLength || settings.repetitions !== importedRequest.repetitions)) throw Error('Workload differs from the imported request. Restore its values or choose the model folder again to start a new workload.');
    const build = await verifyRuntime(runtimeFiles, controller.signal, log);
    if (settings.graphCapture && !build.capabilities?.capturedExternalIoBinding) throw Error('Runtime lacks verified captured external I/O support');
    $('runtime-info').textContent = `${build.version} · ORT ${build.repositories.onnxruntime.commit.slice(0, 12)} · hashes verified`;
    result = { runtime: 'ort-web', interface: 'standalone-browser-v1', success: false, models: [], errors: [], build,
      browser: { userAgent: navigator.userAgent, crossOriginIsolated, locationProtocol: location.protocol },
      benchmarkOptions: { ...settings, wasmThreads: 1, ioBinding: true, maxLength: 8192, prefillChunkSize: 0, referenceTokens: settings.generationLength },
      request: importedRequest || { schemaVersion: 1, models: [], promptLengths: settings.promptLengths, generationLength: settings.generationLength,
        repetitions: settings.repetitions, maxLength: 8192, seed: 42, prefillChunkSize: 0, batchSize: 1, sampling: 'greedy', suppressEos: true },
      requestSha256: importedRequestHash, startedAt: new Date().toISOString() };
    // Verify all selected artifacts before any GPU work. Weight hashing is outside timings.
    const prepared = [];
    for (const entry of selected) {
      const manifest = await describeLocalModel(entry, 8192, controller.signal, log);
      let prompts;
      if (importedRequest) {
        const expected = importedRequest.models.find(item => item.name === entry.name);
        if (expected.manifest.configArtifact.sha256 !== manifest.configArtifact.sha256
            || JSON.stringify(expected.manifest.states) !== JSON.stringify(manifest.states)) throw Error(`Model/config differs from imported request: ${entry.name}`);
        for (const [name, session] of Object.entries(manifest.sessions)) {
          const old = expected.manifest.sessions[name];
          if (old?.sha256 !== session.sha256 || JSON.stringify(old.externalData) !== JSON.stringify(session.externalData)) throw Error(`Model weights differ from request: ${entry.name}`);
        }
        prompts = Object.fromEntries(expected.cases.map(item => [item.pl, item.prompt]));
        for (const pl of settings.promptLengths) if (!Array.isArray(prompts[pl]) || prompts[pl].length !== pl || prompts[pl].some(id => !Number.isSafeInteger(id) || id < 0 || id >= manifest.config.model.vocab_size)) throw Error('Invalid saved prompt IDs');
      } else result.request.models.push({ name: entry.name, root: 'user-selected-local-folder', manifest,
        cases: settings.promptLengths.map(pl => ({ pl, prompt: randomPrompt(manifest.config, pl, 42) })) });
      prepared.push({ entry, manifest, prompts });
    }
    if (!result.requestSha256) result.requestSha256 = (await fingerprint(new Blob([JSON.stringify(result.request)]))).sha256;
    for (const { entry, manifest, prompts } of prepared) {
      controller.signal.throwIfAborted();
      const partial = { name: entry.name, rows: [], success: false, errors: ['Model has not completed'] };
      result.models.push(partial);
      const measured = await runModel(entry, manifest, build, { ...settings, prompts }, partial);
      Object.assign(partial, measured);
      if (!measured.success) throw Error(`${entry.name}: ${measured.errors.join('; ')}`);
    }
    result.success = true;
    log('Complete. Every timed output matches the full reference sequence. Download JSON for all samples and token IDs.');
  } catch (error) {
    const cancelled = controller.signal.aborted;
    log(cancelled ? 'Cancelled. Partial results are not a completed performance run.' : `Failed: ${error.message}`);
    result ||= { runtime: 'ort-web', interface: 'standalone-browser-v1', models: [], errors: [] };
    result.success = false; result.cancelled = cancelled; result.errors.push(String(error));
  } finally {
    frame?.remove(); frame = null; cancelFrame = null; result.finishedAt = new Date().toISOString();
    $('json').disabled = false; $('csv').disabled = !result.models.some(model => model.rows?.length); busy(false);
  }
});
$('stop').addEventListener('click', () => { controller?.abort(); cancelFrame?.(); });
function download(contents, extension, type) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement('a'); anchor.href = url;
  anchor.download = `ort-web-${new Date().toISOString().replace(/[:.]/g, '-')}.${extension}`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('json').addEventListener('click', () => download(JSON.stringify(result, null, 2), 'json', 'application/json'));
$('csv').addEventListener('click', () => download(csvResults(result), 'csv', 'text/csv'));
checkBrowser();
