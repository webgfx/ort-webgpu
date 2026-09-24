// Autoregressive text generation over the exact ONNX exports used by native
// GenAI. This file is shared by the browser runner and deterministic unit tests.
export function halfToFloat(value) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 31;
  const fraction = value & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? (1 + fraction / 1024) * 2 ** (exponent - 15) : fraction * 2 ** -24);
}

export function greedyToken(data, type, vocabSize, suppressedTokens = []) {
  if (!Number.isSafeInteger(vocabSize) || vocabSize <= 0 || data.length < vocabSize || data.length % vocabSize) {
    throw new Error('Logits do not contain complete vocabulary rows');
  }
  // GPU readback exposes FP16 bits as Uint16Array. Reinterpret the same bytes
  // without a copy instead of decoding every vocabulary score in JavaScript.
  // Preserve subviews, and keep scalar conversion for older browsers.
  if (type === 'float16' && data instanceof Uint16Array && typeof globalThis.Float16Array === 'function') {
    data = new globalThis.Float16Array(data.buffer, data.byteOffset, data.length);
  }
  const offset = data.length - vocabSize;
  const suppressed = new Set(suppressedTokens);
  const rawHalf = type === 'float16' && data instanceof Uint16Array;
  let best = -Infinity;
  let token = -1;
  for (let i = 0; i < vocabSize; ++i) {
    const score = rawHalf ? halfToFloat(data[offset + i]) : Number(data[offset + i]);
    if (!Number.isFinite(score)) throw new Error(`Non-finite logits at token ${i}`);
    // Only a new maximum can win, so consult the EOS set for candidates rather
    // than for every vocabulary entry. Still validate every score, even masked
    // ones; strict comparison preserves the earliest token on ties.
    if (score > best && !suppressed.has(i)) { best = score; token = i; }
  }
  if (token < 0) throw new Error('No valid next token');
  return token;
}

export function randomPrompt(config, length, seed = 42) {
  if (!Number.isSafeInteger(length) || length <= 0) throw new Error('Prompt length must be positive');
  const model = config.model;
  const forbidden = new Set(Object.entries(model).filter(([key]) => key.endsWith('_token_id'))
    .flatMap(([, value]) => Array.isArray(value) ? value : [value]));
  if (!Number.isSafeInteger(model.vocab_size) || model.vocab_size <= forbidden.size) throw new Error('Invalid vocabulary');
  let state = seed >>> 0;
  const tokens = [];
  while (tokens.length < length) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    // Xorshift has an absorbing zero state. Keep all seeds usable.
    if (!state) state = 0x9e3779b9;
    const token = (state >>> 0) % model.vocab_size;
    if (!forbidden.has(token)) tokens.push(token);
  }
  return tokens;
}

function tensorData(type, values) {
  switch (type) {
    case 'int64': return BigInt64Array.from(values, BigInt);
    case 'int32': return Int32Array.from(values);
    case 'float16': return Uint16Array.from(values);
    case 'float32': return Float32Array.from(values);
    default: throw new Error(`Unsupported generator input type: ${type}`);
  }
}

export function cpuInputs(ort, section, session, tokens, pastLength, embeddings) {
  const feeds = {};
  const temporary = [];
  for (const input of session.inputs) {
    let dims, values;
    if (input.name === section.inputs.input_ids) {
      dims = [1, tokens.length]; values = tokens;
    } else if (input.name === section.inputs.attention_mask) {
      dims = [1, pastLength + tokens.length]; values = Array(dims[1]).fill(1);
    } else if (input.name === section.inputs.position_ids) {
      const sequence = tokens.map((_, index) => pastLength + index);
      if (input.dims.length === 3 && input.dims[0] === 3) {
        // Qwen MRoPE's three axes coincide for text-only inputs.
        dims = [3, 1, tokens.length]; values = [...sequence, ...sequence, ...sequence];
      } else if (input.dims.length === 2) {
        dims = [1, tokens.length]; values = sequence;
      } else throw new Error(`Unsupported position_ids shape for ${input.name}`);
    } else if (input.name === section.inputs.inputs_embeds) {
      if (!embeddings) throw new Error('Decoder requires the embedding session output');
      feeds[input.name] = embeddings;
      continue;
    } else if (input.name === section.inputs.image_features || input.name === section.inputs.audio_features) {
      if (input.dims.length !== 2 || !Number.isSafeInteger(input.dims[1])) {
        throw new Error(`Unsupported empty modality input ${input.name}`);
      }
      dims = [0, input.dims[1]]; values = [];
    } else continue; // State inputs are supplied separately.
    const tensor = new ort.Tensor(input.type, tensorData(input.type, values), dims);
    feeds[input.name] = tensor;
    temporary.push(tensor);
  }
  return { feeds, temporary };
}

export class TextGenerator {
  constructor(ort, device, manifest, sessions, now = () => performance.now()) {
    this.ort = ort;
    this.device = device;
    this.manifest = manifest;
    this.sessions = sessions;
    this.now = now;
    this.pastLength = 0;
    this.states = [];
    try {
      for (const state of manifest.states) {
        const first = this.allocateState(state);
        this.states.push({ ...state, past: first, present: first });
        if (!state.shared) this.states.at(-1).present = this.allocateState(state);
      }
    } catch (error) {
      this.disposeStates();
      throw error;
    }
  }

  allocateState(state) {
    if (!['float16', 'float32'].includes(state.type)) throw new Error(`Unsupported cache type ${state.type}`);
    const bytes = state.dims.reduce((a, b) => a * b, 1) * (state.type === 'float16' ? 2 : 4);
    if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error(`Invalid state size for ${state.input}`);
    if (bytes > this.device.limits.maxStorageBufferBindingSize || bytes > this.device.limits.maxBufferSize) {
      throw new Error(`GPU buffer limit cannot accommodate ${state.input} (${bytes} bytes)`);
    }
    const buffer = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: state.input });
    try {
      return this.ort.Tensor.fromGpuBuffer(buffer, { dataType: state.type, dims: state.dims,
        dispose: () => buffer.destroy() });
    } catch (error) { buffer.destroy(); throw error; }
  }

  async reset() {
    this.pastLength = 0;
    // KV entries beyond the attention mask are unread; prefill overwrites the
    // active span. Hybrid recurrent/conv state must be zeroed on each sequence.
    const encoder = this.device.createCommandEncoder();
    for (const state of this.states) if (!state.shared) {
      encoder.clearBuffer(state.past.gpuBuffer);
      encoder.clearBuffer(state.present.gpuBuffer);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  async step(tokens, suppressedTokens = []) {
    if (!tokens.length || this.pastLength + tokens.length > this.manifest.maxLength) {
      throw new Error(`Sequence exceeds ${this.manifest.maxLength}-token KV cache capacity`);
    }
    const model = this.manifest.config.model;
    for (const token of tokens) if (!Number.isInteger(token) || token < 0 || token >= model.vocab_size) {
      throw new Error(`Token ${token} is outside the model vocabulary`);
    }
    const temporary = [];
    let embeddingOutputs, decoderOutputs;
    try {
      let embedding;
      if (this.sessions.embedding) {
        const prepared = cpuInputs(this.ort, model.embedding, this.manifest.sessions.embedding, tokens, this.pastLength);
        temporary.push(...prepared.temporary);
        embeddingOutputs = await this.sessions.embedding.run(prepared.feeds);
        embedding = embeddingOutputs[model.embedding.outputs.inputs_embeds];
      }
      const prepared = cpuInputs(this.ort, model.decoder, this.manifest.sessions.decoder, tokens, this.pastLength, embedding);
      temporary.push(...prepared.temporary);
      const fetches = { [model.decoder.outputs.logits]: null };
      for (const state of this.states) {
        prepared.feeds[state.input] = state.past;
        fetches[state.output] = state.present;
      }
      decoderOutputs = await this.sessions.decoder.run(prepared.feeds, fetches);
      const logits = decoderOutputs[model.decoder.outputs.logits];
      // This readback synchronizes execution. Never time only queued GPU work.
      const values = await logits.getData();
      const token = greedyToken(values, logits.type, model.vocab_size, suppressedTokens);
      this.pastLength += tokens.length;
      for (const state of this.states) if (!state.shared) [state.past, state.present] = [state.present, state.past];
      return token;
    } finally {
      for (const tensor of temporary) tensor.dispose();
      if (embeddingOutputs) for (const tensor of Object.values(embeddingOutputs)) tensor.dispose();
      // Explicit cache fetches belong to the generator and must not be disposed.
      decoderOutputs?.[model.decoder.outputs.logits]?.dispose();
    }
  }

  async generate(prompt, generationLength, { prefillChunkSize = 0 } = {}) {
    if (!Number.isSafeInteger(generationLength) || generationLength < 2 ||
        prompt.length + generationLength > this.manifest.maxLength) {
      throw new Error('Generation length must be >=2 and fit within the KV cache');
    }
    if (!Number.isSafeInteger(prefillChunkSize) || prefillChunkSize < 0 || prefillChunkSize > this.manifest.maxLength || !prompt.length) {
      throw new Error('Invalid prefill chunk size or empty prompt');
    }
    await this.reset();
    const eos = this.manifest.config.model.eos_token_id;
    const suppressedTokens = eos === undefined ? [] : Array.isArray(eos) ? eos : [eos];
    const start = this.now();
    const chunkSize = prefillChunkSize || prompt.length;
    let firstToken;
    for (let offset = 0; offset < prompt.length; offset += chunkSize) {
      // Intermediate predictions are discarded; every chunk consists solely
      // of the original prompt tokens. Cache and positions advance normally.
      firstToken = await this.step(prompt.slice(offset, offset + chunkSize), suppressedTokens, { phase: 'prefill' });
    }
    const generated = [firstToken];
    const firstTokenAt = this.now();
    // Match native GenAI's min_length = prompt_length + generation_length:
    // suppress EOS scores instead of feeding EOS back as ordinary text tokens.
    for (let index = 1; index < generationLength; ++index) generated.push(await this.step([generated.at(-1)], suppressedTokens, { phase: 'decode' }));
    const end = this.now();
    const ttftMs = firstTokenAt - start;
    const decodeMs = end - firstTokenAt;
    const sample = { ttftMs, prefillTps: prompt.length * 1000 / ttftMs,
      decodeTps: (generationLength - 1) * 1000 / decodeMs, e2eMs: end - start, generated };
    for (const key of ['ttftMs', 'prefillTps', 'decodeTps', 'e2eMs']) {
      if (!(sample[key] > 0 && Number.isFinite(sample[key]))) throw new Error(`Invalid measured ${key}`);
    }
    return sample;
  }

  disposeStates() {
    for (const state of this.states) {
      state.past.dispose();
      if (state.present !== state.past) state.present.dispose();
    }
    this.states = [];
  }

  async dispose() {
    this.disposeStates();
    for (const session of Object.values(this.sessions)) await session.release();
  }
}

export function modelProviderOptions(manifest, name) {
  const providers = manifest.config.model[name].session_options?.provider_options || [];
  const configured = Object.assign({}, ...providers.map(provider => provider.webgpu || {}));
  // Capture is an explicit per-phase policy; validation stays at the benchmark's
  // basic setting. Preserve model-specific options such as Phi's rotary cache
  // concat offset, which is essential to its captured decode fast path.
  delete configured.enableGraphCapture;
  delete configured.device;
  delete configured.name;
  return { ...configured, validationMode: 'basic' };
}

export async function createModelSession(ort, device, manifest, baseUrl, name, options = {}, weights = new Map()) {
  const description = manifest.sessions[name];
  const externalData = [];
  for (const artifact of description.externalData) {
    let blob = weights.get(artifact.file);
    if (!blob) {
      if (typeof baseUrl?.read === 'function') blob = await baseUrl.read(artifact.file);
      else {
        const response = await fetch(new URL(artifact.file, baseUrl));
        if (!response.ok) throw new Error(`Weights download failed: HTTP ${response.status}`);
        blob = await response.blob();
      }
      if (blob.size !== artifact.size) throw new Error(`Weights size mismatch: ${artifact.file}`);
      weights.set(artifact.file, blob);
    }
    // JSPI reads Blob ranges without copying multi-GB weights into WASM memory.
    externalData.push({ path: artifact.path, data: blob });
  }
  const graph = typeof baseUrl?.read === 'function'
    ? new Uint8Array(await (await baseUrl.read(description.file)).arrayBuffer())
    : new URL(description.file, baseUrl).href;
  return ort.InferenceSession.create(graph, {
    executionProviders: [{ ...modelProviderOptions(manifest, name), name: 'webgpu', device }],
    externalData, graphOptimizationLevel: 'all', enableGraphCapture: false,
    preferredOutputLocation: name === 'embedding' ? 'gpu-buffer' : { [manifest.config.model.decoder.outputs.logits]: 'cpu' },
    ...options,
  });
}

export async function loadGenerator(ort, device, manifest, baseUrl, onProgress = () => {}) {
  const sessions = {};
  try {
    const weights = new Map();
    for (const name of Object.keys(manifest.sessions)) {
      onProgress(`Loading ${manifest.name} ${name}`);
      sessions[name] = await createModelSession(ort, device, manifest, baseUrl, name, {}, weights);
    }
    const generator = new TextGenerator(ort, device, manifest, sessions);
    generator.providerOptions = Object.fromEntries(Object.keys(manifest.sessions).map(name => [name, modelProviderOptions(manifest, name)]));
    return generator;
  } catch (error) {
    for (const session of Object.values(sessions)) await session.release();
    throw error;
  }
}
