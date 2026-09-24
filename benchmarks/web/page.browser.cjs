// Real-browser end-to-end check of opening index.html directly from disk.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');

(async () => {
  const { default: puppeteer } = await import('puppeteer-core');
  const root = path.resolve(__dirname, '../..');
  const output = path.join(root, 'gitignore/benchmarks/results/browser-page-check');
  fs.mkdirSync(output, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: process.env.BENCHMARK_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true, args: ['--force-high-performance-gpu'], protocolTimeout: 180000 });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1340, height: 1080 });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    const consoleMessages = [];
    page.on('console', message => { if (consoleMessages.length < 15000) consoleMessages.push(message.text()); });
    const network = [];
    page.on('request', request => { if (/^https?:/.test(request.url())) network.push(request.url()); });
    await page.goto(pathToFileURL(path.join(__dirname, 'index.html')).href);
    await page.waitForFunction(() => document.querySelector('#environment').textContent.includes('WebGPU + JSPI'));
    await page.screenshot({ path: path.join(output, 'page-desktop.png'), fullPage: true });
    await page.setViewport({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Page overflows on mobile');
    await page.screenshot({ path: path.join(output, 'page-mobile.png'), fullPage: true });
    await page.setViewport({ width: 1340, height: 1080 });
    if (process.argv.includes('--model')) {
      const runtime = process.env.BENCHMARK_RUNTIME || path.join(root, 'gitignore/runtime-comparison/ort-web/benchmark-20260920-09dfa6ad');
      const model = process.env.BENCHMARK_MODEL || 'D:/workspace/project/agents/ai-models/aion';
      await (await page.$('#runtime-files')).uploadFile(runtime);
      await (await page.$('#model-files')).uploadFile(model);
      await page.waitForFunction(() => document.querySelectorAll('#models input').length > 0);
      let tokenCount = 128;
      if (process.env.BENCHMARK_REQUEST) {
        await (await page.$('#request-file')).uploadFile(process.env.BENCHMARK_REQUEST);
        await page.waitForFunction(() => document.querySelector('#request-info').textContent.startsWith('Loaded '));
        tokenCount = JSON.parse(fs.readFileSync(process.env.BENCHMARK_REQUEST, 'utf8')).generationLength;
        await page.evaluate(() => { document.querySelector('#diagnostics').checked = true; });
      } else await page.evaluate(() => { document.querySelector('#lengths').value = '128'; document.querySelector('#tokens').value = '128'; document.querySelector('#repetitions').value = '1'; document.querySelector('#diagnostics').checked = true; });
      await page.click('#run');
      await page.waitForFunction(() => !document.querySelector('#json').disabled, { timeout: 600000 });
      const status = await page.$eval('#status', node => node.textContent);
      fs.writeFileSync(path.join(output, 'browser-console.log'), consoleMessages.join('\n'));
      fs.writeFileSync(path.join(output, 'page.log'), await page.$eval('#log', node => node.textContent));
      assert(status.startsWith('Complete.'), status);
      assert.equal(await page.$$eval('#rows tr', nodes => nodes.length), 1);
      await page.screenshot({ path: path.join(output, 'page-result.png'), fullPage: true });
      // Collect the generated JSON download without writing outside ignored results.
      const client = await page.createCDPSession();
      await client.send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: output });
      await page.click('#json');
      await new Promise(resolve => setTimeout(resolve, 500));
      const name = fs.readdirSync(output).filter(name => name.endsWith('.json')).sort().at(-1);
      const data = JSON.parse(fs.readFileSync(path.join(output, name)));
      assert.equal(data.success, true); assert.equal(data.models[0].rows[0].validation.reference.comparedTokens, tokenCount);
      assert(data.models[0].executionEvidence.runtimeReplayMessages > 0, 'Missing actual capture replay');
      assert.equal(data.models[0].rows[0].samples[0].generated.length, tokenCount);
      if (process.env.BENCHMARK_REQUEST) assert.equal(data.requestSha256, require('node:crypto').createHash('sha256').update(fs.readFileSync(process.env.BENCHMARK_REQUEST)).digest('hex'));
      await page.click('#run'); await page.click('#stop');
      await page.waitForFunction(() => !document.querySelector('#json').disabled);
      assert((await page.$eval('#status', node => node.textContent)).startsWith('Cancelled.'), 'Stop did not mark the run incomplete');
      console.log('PASS: direct-file model run, complete reference output, real decoder replay and downloadable JSON');
    }
    assert.deepEqual(errors, []); assert.deepEqual(network, [], 'Standalone page made external network requests');
    console.log('PASS: file:// page, desktop/mobile layout, no external requests');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
