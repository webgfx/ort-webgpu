import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { jspiLoadEvidence } from './runtime-evidence.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const configuration = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const mime = filename => filename.endsWith('.wasm') ? 'application/wasm'
  : /\.(mjs|js)$/.test(filename) ? 'text/javascript' : 'application/octet-stream';

function browserPath(browser) {
  if (process.platform !== 'win32') throw new Error('This daily benchmark targets the Windows D3D12 browser backend');
  const relative = browser === 'edge-canary' ? 'Microsoft/Edge SxS/Application/msedge.exe' : 'Microsoft/Edge/Application/msedge.exe';
  for (const root of [process.env.LOCALAPPDATA, process.env.ProgramFiles, process.env['ProgramFiles(x86)']]) {
    if (root && fs.existsSync(path.join(root, relative))) return path.join(root, relative);
  }
  throw new Error(`Microsoft ${browser} was not found; no alternate browser will be substituted`);
}

// Only explicitly verified artifacts can be served. No arbitrary filesystem
// paths, directory browsing, remote redirects, or external CDN fallback.
const routes = new Map();
for (const name of Object.keys(configuration.build.artifacts)) routes.set(`/artifacts/${name}`, path.join(configuration.artifactRoot, name));
for (let i = 0; i < configuration.models.length; ++i) {
  const manifest = configuration.models[i];
  for (const session of Object.values(manifest.sessions)) {
    for (const artifact of [session, ...session.externalData]) {
      routes.set(`/models/${i}/${artifact.file}`, path.join(configuration.modelRoots[i], artifact.file));
    }
  }
}
routes.set('/runner.mjs', path.join(here, 'runner.mjs'));
routes.set('/generator.mjs', path.join(here, 'generator.mjs'));
routes.set('/capture-generator.mjs', path.join(here, 'capture-generator.mjs'));
routes.set('/device.mjs', path.join(here, 'device.mjs'));
routes.set('/gpu_failure.mjs', path.join(here, 'gpu_failure.mjs'));
routes.set('/gpu-greedy.mjs', path.join(here, 'gpu-greedy.mjs'));
routes.set('/gpu-pipeline.mjs', path.join(here, 'gpu-pipeline.mjs'));
const publicConfig = Object.fromEntries(['models', 'prompts', 'promptLengths', 'generationLength', 'repetitions', 'wasmThreads', 'seed', 'prefillChunkSize', 'graphCapture', 'ioBinding', 'referenceTokens', 'captureDiagnostics', 'gpuSampling']
  .map(key => [key, configuration[key]]));
const server = http.createServer((request, response) => {
  if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
  if (request.headers.host !== `127.0.0.1:${server.address().port}`) { response.writeHead(403).end(); return; }
  response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  response.setHeader('Cache-Control', 'no-store');
  let requested;
  try { requested = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname); }
  catch { response.writeHead(400).end(); return; }
  if (requested === '/') {
    response.setHeader('Content-Type', 'text/html');
    response.end(`<!doctype html><meta charset="utf-8"><title>ORT WebGPU LLM benchmark</title><script src="/artifacts/${configuration.build.entrypoint}"></script><script type="module" src="/runner.mjs"></script>`);
  } else if (requested === '/config') {
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify(publicConfig));
  } else if (routes.has(requested)) {
    const file = routes.get(requested);
    response.setHeader('Content-Type', mime(file));
    response.setHeader('Content-Length', fs.statSync(file).size);
    if (request.method === 'HEAD') { response.end(); return; }
    const stream = fs.createReadStream(file);
    stream.on('error', error => response.destroy(error));
    response.on('close', () => stream.destroy());
    stream.pipe(response);
  } else response.writeHead(404).end();
});

let browser;
const result = { success: false, models: [], errors: [] };
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const executablePath = browserPath(configuration.browser);
  // A fresh disposable profile isolates shader caches from users' browsers.
  // Do not disable sandboxing, GPU validation, or browser security features.
  // Chromium's Windows "default" WebGPU backend is D3D12; "d3d12" is not a
  // recognized adapter-name switch value (unlike "d3d11" or "swiftshader").
  const args = ['--use-webgpu-adapter=default', '--force-high-performance-gpu'];
  browser = await puppeteer.launch({ executablePath, headless: configuration.headless, args,
    protocolTimeout: configuration.timeout * 1000 });
  const cdp = await browser.target().createCDPSession();
  const system = await cdp.send('SystemInfo.getInfo');
  result.browser = { name: configuration.browser, version: await browser.version(), executablePath, args };
  result.gpu = system.gpu;
  for (let i = 0; i < configuration.models.length; ++i) {
    const page = await browser.newPage();
    let replayMessages = 0;
    const runtimeResponses = [];
    page.on('response', response => {
      const url = new URL(response.url());
      if (url.origin === `http://127.0.0.1:${server.address().port}` && url.pathname.startsWith('/artifacts/')) {
        runtimeResponses.push({ file: path.posix.basename(url.pathname), status: response.status() });
      }
    });
    page.on('console', message => {
      const text = message.text();
      if (text.includes('Replaying the captured WebGpuExecutionProvider graph')) replayMessages++;
      process.stderr.write(`${text}\n`);
    });
    page.on('pageerror', error => process.stderr.write(`[ort-web] page error: ${error.message}\n`));
    try {
      await page.goto(`http://127.0.0.1:${server.address().port}/?model=${i}`, { waitUntil: 'load', timeout: 60000 });
      await page.waitForFunction(() => window.__ortWebResult !== undefined, { timeout: configuration.timeout * 1000 });
      const model = { name: configuration.models[i].name, ...await page.evaluate(() => window.__ortWebResult) };
      if (model.success) {
        try {
          model.executionEvidence = { ...model.executionEvidence, runtime: jspiLoadEvidence(configuration.build, runtimeResponses) };
        } catch (error) {
          model.success = false;
          model.errors = [...(model.errors || []), error.message];
        }
      }
      if (model.executionEvidence) model.executionEvidence.runtimeReplayMessages = replayMessages;
      if (configuration.captureDiagnostics && configuration.graphCapture && model.success && !replayMessages) {
        model.success = false;
        model.errors = [...(model.errors || []), 'Capture diagnostics requested but ORT emitted no graph replay evidence'];
      }
      result.models.push(model);
    } catch (error) {
      let partial = {};
      try { partial = await page.evaluate(() => window.__ortWebPartial || {}); } catch { /* renderer unavailable */ }
      result.models.push({ name: configuration.models[i].name, ...partial, success: false, errors: [String(error)] });
    } finally { await page.close().catch(() => {}); }
  }
  result.success = result.models.length === configuration.models.length && result.models.every(model => model.success);
} catch (error) { result.errors.push(String(error.stack || error)); }
finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
process.stdout.write(JSON.stringify(result));
// Preserve structured partial results. The Python wrapper and server validate
// success independently instead of losing measurements on a nonzero exit.
