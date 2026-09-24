import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex } from '@noble/hashes/utils';
import { describeOnnx, stateBindings } from './onnx-metadata.mjs';

export function safePath(value) {
  const name = String(value).replaceAll('\\', '/');
  if (!name || name.startsWith('/') || name.includes(':') || name.split('/').some(part => !part || part === '..' || part === '.')) throw Error(`Unsafe artifact path: ${value}`);
  return name;
}
export function directoryFiles(files) {
  const result = new Map();
  for (const file of files) {
    const relative = file.webkitRelativePath || file.name;
    const name = safePath(file.webkitRelativePath ? relative.slice(relative.indexOf('/') + 1) : relative);
    if (result.has(name)) throw Error(`Duplicate selected file: ${name}`);
    result.set(name, file);
  }
  return result;
}
export async function fingerprint(file, signal, progress = () => {}) {
  const hash = sha256.create();
  const chunkSize = 8 * 1024 * 1024;
  for (let offset = 0; offset < file.size; offset += chunkSize) {
    signal?.throwIfAborted();
    hash.update(new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer()));
    progress(Math.min(file.size, offset + chunkSize), file.size);
    // Yield between chunks so validation of multi-GB weights does not freeze UI.
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  signal?.throwIfAborted();
  return { size: file.size, sha256: bytesToHex(hash.digest()) };
}
export const readJson = async file => JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
export function requireFile(files, name) {
  const file = files.get(safePath(name));
  if (!file) throw Error(`Missing selected file: ${name}`);
  return file;
}
export async function verifyRuntime(files, signal, progress = () => {}) {
  const build = await readJson(requireFile(files, 'build-metadata.json'));
  const names = ['ort.all.min.js', 'ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm'];
  if (build.runtime !== 'ort-web' || !build.builtFromSource || build.wasmVariant !== 'jspi' || build.webgpuImplementation !== 'cpp' || build.entrypoint !== names[0]
      || !/^[a-f0-9]{40}$/.test(build.repositories?.onnxruntime?.commit || '') || JSON.stringify(Object.keys(build.artifacts || {}).sort()) !== JSON.stringify([...names].sort())) throw Error('Select a complete, source-built ORT C++ WebGPU/JSPI runtime folder');
  for (const name of names) {
    progress(`Verifying ${name}`);
    const actual = await fingerprint(requireFile(files, name), signal);
    if (actual.sha256 !== build.artifacts[name].sha256 || actual.size !== build.artifacts[name].size) throw Error(`Runtime hash mismatch: ${name}`);
  }
  return build;
}
export function discoverModels(files, folderName = 'Selected model') {
  return [...files.keys()].filter(name => name === 'genai_config.json' || name.endsWith('/genai_config.json')).map(configPath => {
    const prefix = configPath.slice(0, -'genai_config.json'.length);
    const parts = prefix.split('/').filter(Boolean);
    const name = parts.at(-1) === 'onnx-webgpu' ? parts.at(-2) || folderName : parts.at(-1) || folderName;
    return { name, configPath, files: new Map([...files].filter(([key]) => key.startsWith(prefix)).map(([key, file]) => [key.slice(prefix.length), file])) };
  });
}
export async function describeLocalModel(entry, maxLength, signal, progress = () => {}) {
  const configFile = requireFile(entry.files, 'genai_config.json'), config = await readJson(configFile);
  const model = config.model;
  if (!model?.decoder || !Number.isSafeInteger(model.vocab_size) || model.vocab_size <= 1 || maxLength > model.context_length || !config.search?.past_present_share_buffer) throw Error('Model needs a supported decoder and shared static KV cache export');
  const sessions = {}, hashes = new Map();
  for (const name of ['decoder', 'embedding']) if (model[name]) {
    const filename = safePath(model[name].filename), file = requireFile(entry.files, filename);
    progress(`Reading ${entry.name}: ${filename}`);
    // Export graphs are small; multi-GB external weights are never read here.
    if (file.size > 512 * 1024 * 1024) throw Error('Use an external-data ONNX export; embedded graph exceeds 512 MB');
    const graph = describeOnnx(new Uint8Array(await file.arrayBuffer()));
    const externalData = [];
    for (const location of graph.external) {
      const relative = filename.includes('/') ? filename.slice(0, filename.lastIndexOf('/') + 1) + safePath(location) : safePath(location);
      const external = requireFile(entry.files, relative);
      if (!hashes.has(relative)) {
        progress(`Hashing ${entry.name}: ${relative} (${(external.size / 1024 ** 3).toFixed(2)} GB)`);
        hashes.set(relative, await fingerprint(external, signal));
      }
      externalData.push({ path: location, file: relative, ...hashes.get(relative) });
    }
    sessions[name] = { file: filename, ...await fingerprint(file, signal), inputs: graph.inputs, outputs: graph.outputs, features: graph.features, externalData };
  }
  const states = stateBindings(model.decoder, sessions.decoder.inputs, sessions.decoder.outputs, maxLength);
  const known = new Set([...states.map(state => state.input), ...['input_ids', 'attention_mask', 'position_ids', 'inputs_embeds'].map(key => model.decoder.inputs[key])]);
  if (sessions.decoder.inputs.some(input => !known.has(input.name))) throw Error('Unsupported decoder input in selected model');
  if (!sessions.decoder.outputs.some(output => output.name === model.decoder.outputs.logits)) throw Error('Missing decoder logits');
  return { schemaVersion: 1, name: entry.name, maxLength, config, configArtifact: { file: 'genai_config.json', ...await fingerprint(configFile, signal) }, sessions, states };
}
