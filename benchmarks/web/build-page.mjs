// Maintainer-only build. Users open the generated index.html without tooling.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
const here = path.dirname(fileURLToPath(import.meta.url));
const options = { bundle: true, write: false, format: 'iife', platform: 'browser', target: 'es2022', minify: false };
const worker = await build({ ...options, entryPoints: [path.join(here, 'page-runner.mjs')] });
const page = await build({ ...options, entryPoints: [path.join(here, 'page-entry.mjs')],
  define: { __BENCHMARK_WORKER_SOURCE__: JSON.stringify(worker.outputFiles[0].text) } });
const template = fs.readFileSync(path.join(here, 'page.html'), 'utf8');
const script = page.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const notices = fs.readFileSync(path.join(here, 'THIRD_PARTY_NOTICES.txt'), 'utf8').replaceAll('--', '—');
fs.writeFileSync(path.join(here, 'index.html'), template.replace('<!-- BENCHMARK_SCRIPT -->', () => '<!--\n' + notices + '\n-->\n<script>\n' + script + '\n</script>'));
console.log('Built web/index.html — self-contained, no server required');
