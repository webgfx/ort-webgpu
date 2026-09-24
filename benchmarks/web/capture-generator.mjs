// Stable GPU I/O for decode graph capture. Prefill is a separate, uncaptured
// session so changing prompt lengths never replace a captured session's bindings.
import { TextGenerator, cpuInputs, greedyToken, createModelSession, modelProviderOptions } from './generator.mjs';

const TYPES = { float16: [2, Uint16Array], float32: [4, Float32Array], int64: [8, BigInt64Array], int32: [4, Int32Array] };

export function allocateGpuTensor(ort, device, type, dims, label) {
  const bytes = TYPES[type]?.[0];
  if (!bytes || dims.some(d => !Number.isSafeInteger(d) || d < 0)) throw new Error(`Unsupported static tensor ${label}`);
  const size = dims.reduce((a, b) => a * b, 1) * bytes;
  // ORT's C++ WebGPU data transfers round to 16 bytes (not only WebGPU's
  // 4-byte copy alignment), including an int64 token copied to a CPU shape op.
  const allocated = Math.max(16, Math.ceil(size / 16) * 16);
  if (!Number.isSafeInteger(size) || allocated > device.limits.maxBufferSize || allocated > device.limits.maxStorageBufferBindingSize) {
    throw new Error(`GPU buffer limit cannot accommodate ${label} (${size} bytes)`);
  }
  const buffer = device.createBuffer({ size: allocated, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label });
  try { return ort.Tensor.fromGpuBuffer(buffer, { dataType: type, dims, dispose: () => buffer.destroy() }); }
  catch (error) { buffer.destroy(); throw error; }
}

export function staticInputPlan(manifest, name) {
  const section = manifest.config.model[name];
  const states = new Map(manifest.states.map(state => [state.input, state]));
  const inputs = {}, overrides = {};
  for (const input of manifest.sessions[name].inputs) {
    let dims;
    if (states.has(input.name)) dims = states.get(input.name).dims;
    else if (input.name === section.inputs.input_ids) dims = [1, 1];
    else if (input.name === section.inputs.attention_mask) dims = [1, manifest.maxLength];
    else if (input.name === section.inputs.position_ids) dims = input.dims.length === 3 ? [3, 1, 1] : [1, 1];
    else if (input.name === section.inputs.inputs_embeds) dims = [1, 1, input.dims[2]];
    else if (input.name === section.inputs.image_features || input.name === section.inputs.audio_features) dims = [0, input.dims[1]];
    else throw new Error(`Unrecognized static input ${name}/${input.name}`);
    if (dims.length !== input.dims.length || dims.some(d => !Number.isSafeInteger(d) || d < 0)) throw new Error(`Cannot resolve static dimensions for ${input.name}`);
    input.dims.forEach((dim, index) => {
      if (typeof dim === 'number') {
        if (dim !== dims[index]) throw new Error(`Static input dimension mismatch: ${input.name}`);
      } else if (dim) {
        if (overrides[dim] !== undefined && overrides[dim] !== dims[index]) throw new Error(`Conflicting static dimension ${dim}`);
        overrides[dim] = dims[index];
      }
    });
    inputs[input.name] = { type: input.type, dims };
  }
  return { inputs, overrides };
}

export function decodeLogitsShape(dims, vocabulary) {
  if (dims.at(-1) !== vocabulary || dims.length < 1 || dims.length > 3 || dims.some(d => !Number.isSafeInteger(d) || d <= 0)) {
    throw new Error(`Unsupported logits shape ${dims}`);
  }
  return dims.map((dim, i) => i === dims.length - 1 ? dim : 1);
}

export class CaptureGenerator extends TextGenerator {
  constructor(ort, device, manifest, sessions, { graphCapture = true, now } = {}) {
    super(ort, device, manifest, sessions, now);
    this.graphCapture = graphCapture;
    this.owned = [];
    this.prefillBuffers = new Map();
    this.decodeFeeds = {};
    this.decodeFetches = {};
    this.embeddingFeeds = {};
    this.embeddingFetches = {};
    this.maskFilled = 0;
    this.recurrentStates = this.states.filter(state => !state.shared);
    this.evidence = { graphCaptureRequested: graphCapture, captureScope: graphCapture ? 'decoder-decode' : 'none', staticKvCache: true,
      embeddingCapture: false, embeddingIoBinding: sessions.decodeEmbedding ? 'cpu-token-input-gpu-output' : 'not-applicable',
      embeddingPrefillIoBinding: sessions.embedding ? 'cpu-inputs-gpu-output' : 'not-applicable',
      kvCapacity: manifest.maxLength, ioBinding: 'gpu-inputs-and-outputs', recurrentStatePolicy: 'fixed-buffers-copy-present-to-past',
      providerOptions: Object.fromEntries(Object.keys(manifest.sessions).map(name => [name, modelProviderOptions(manifest, name)])),
      decodeRuns: 0, decodeInputLocations: {}, decodeInputShapes: {}, decodeOutputShapes: {} };
    try {
      if (sessions.decodeEmbedding) {
        const plan = staticInputPlan(manifest, 'embedding');
        for (const [name, spec] of Object.entries(plan.inputs)) {
          if (name === manifest.config.model.embedding.inputs.input_ids) {
            // This auxiliary session is not captured and can have CPU token-mask
            // operations. Keep the tiny token input on CPU instead of uploading
            // it only for ORT to synchronously download it again. Its output is
            // still the fixed GPU buffer consumed by the captured decoder.
            const ArrayType = TYPES[spec.type][1];
            this.embeddingFeeds[name] = new ort.Tensor(spec.type, new ArrayType(1), spec.dims);
            this.owned.push(this.embeddingFeeds[name]);
          } else this.embeddingFeeds[name] = this.own(spec.type, spec.dims, `decode-embedding/${name}`);
        }
        this.evidence.embeddingInputLocations = Object.fromEntries(Object.entries(this.embeddingFeeds).map(([name, tensor]) => [name, tensor.location]));
        const outputName = manifest.config.model.embedding.outputs.inputs_embeds;
        const spec = manifest.sessions.embedding.outputs.find(output => output.name === outputName);
        const dims = spec.dims.map(dim => typeof dim === 'number' ? dim : plan.overrides[dim]);
        this.embeddingFetches[outputName] = this.own(spec.type, dims, `decode-embedding/${outputName}`);
      }
      const section = manifest.config.model.decoder;
      const plan = staticInputPlan(manifest, 'decoder');
      const states = new Map(this.states.map(state => [state.input, state]));
      for (const [name, spec] of Object.entries(plan.inputs)) {
        this.decodeFeeds[name] = states.has(name) ? states.get(name).past
          : name === section.inputs.inputs_embeds ? this.embeddingFetches[manifest.config.model.embedding.outputs.inputs_embeds]
            : this.own(spec.type, spec.dims, `decode/${name}`);
      }
      const stateOutputs = new Map(this.states.map(state => [state.output, state.present]));
      for (const output of manifest.sessions.decoder.outputs) {
        if (output.name === section.outputs.logits) this.decodeFetches[output.name] = null;
        else if (stateOutputs.has(output.name)) this.decodeFetches[output.name] = stateOutputs.get(output.name);
        else throw new Error(`Unsupported captured decoder output ${output.name}`);
      }
      const mask = this.decodeFeeds[section.inputs.attention_mask];
      if (mask) {
        const ArrayType = TYPES[mask.type][1];
        this.maskOnes = new ArrayType(manifest.maxLength).fill(mask.type === 'int64' ? 1n : 1);
      }
      for (const [name, tensor] of Object.entries(this.decodeFeeds)) {
        this.evidence.decodeInputLocations[name] = tensor.location;
        this.evidence.decodeInputShapes[name] = [...tensor.dims];
      }
    } catch (error) { this.disposeBuffers(); this.disposeStates(); throw error; }
  }

  own(type, dims, name) {
    const tensor = allocateGpuTensor(this.ort, this.device, type, dims, name);
    this.owned.push(tensor);
    return tensor;
  }

  async reset() {
    await super.reset();
    this.maskFilled = 0;
    const mask = this.decodeFeeds[this.manifest.config.model.decoder.inputs.attention_mask];
    if (mask) {
      const encoder = this.device.createCommandEncoder();
      encoder.clearBuffer(mask.gpuBuffer);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    }
  }

  upload(tensor, values) {
    if (!values.length) return;
    const ArrayType = TYPES[tensor.type][1];
    const data = tensor.type === 'int64' ? BigInt64Array.from(values, BigInt) : ArrayType.from(values);
    if (tensor.location === 'cpu') tensor.data.set(data);
    else this.device.queue.writeBuffer(tensor.gpuBuffer, 0, data);
  }

  gpuPrefillInputs(section, description, tokens, embeddings) {
    const prepared = cpuInputs(this.ort, section, description, tokens, this.pastLength, embeddings);
    try {
      for (const [name, tensor] of Object.entries(prepared.feeds)) {
        if (tensor.location === 'gpu-buffer') continue;
        const key = [name, tensor.type, ...tensor.dims].join('|');
        let gpu = this.prefillBuffers.get(key);
        if (!gpu) { gpu = this.own(tensor.type, tensor.dims, `prefill/${name}`); this.prefillBuffers.set(key, gpu); }
        if (tensor.data.byteLength) this.device.queue.writeBuffer(gpu.gpuBuffer, 0, tensor.data);
        prepared.feeds[name] = gpu;
      }
      return prepared.feeds;
    } finally { for (const tensor of prepared.temporary) tensor.dispose(); }
  }

  updateDecodeInputs(tokens) {
    const section = this.manifest.config.model.decoder;
    if (tokens.length !== 1) throw new Error('Captured decode requires exactly one input token');
    if (section.inputs.input_ids) this.upload(this.decodeFeeds[section.inputs.input_ids], tokens);
    if (section.inputs.position_ids) {
      const tensor = this.decodeFeeds[section.inputs.position_ids];
      this.upload(tensor, Array(tensor.dims.reduce((a, b) => a * b, 1)).fill(this.pastLength));
    }
    const mask = this.decodeFeeds[section.inputs.attention_mask];
    const active = this.pastLength + 1;
    if (mask) {
      const bytes = TYPES[mask.type][0];
      this.device.queue.writeBuffer(mask.gpuBuffer, this.maskFilled * bytes,
        this.maskOnes.buffer, this.maskFilled * bytes, (active - this.maskFilled) * bytes);
      this.maskFilled = active;
    }
    if (this.sessions.decodeEmbedding) this.upload(this.embeddingFeeds[this.manifest.config.model.embedding.inputs.input_ids], tokens);
  }

  copyRecurrentState() {
    if (!this.recurrentStates.length) return;
    const encoder = this.device.createCommandEncoder();
    for (const state of this.recurrentStates) encoder.copyBufferToBuffer(state.present.gpuBuffer, 0, state.past.gpuBuffer, 0, state.past.gpuBuffer.size);
    this.device.queue.submit([encoder.finish()]);
  }

  async readLogits(logits) {
    if (logits.location !== 'gpu-buffer' || !['float16', 'float32'].includes(logits.type)) throw new Error('Expected GPU-bound floating-point logits');
    const vocab = this.manifest.config.model.vocab_size;
    const elements = logits.dims.reduce((a, b) => a * b, 1);
    if (elements < vocab || elements % vocab) throw new Error('Incomplete logits vocabulary rows');
    const [bytes, ArrayType] = TYPES[logits.type];
    const offset = (elements - vocab) * bytes;
    const alignedOffset = Math.floor(offset / 4) * 4;
    const padding = offset - alignedOffset;
    const size = Math.ceil((vocab * bytes + padding) / 4) * 4;
    if (!this.readback || this.readback.size < size) {
      this.readback?.destroy();
      this.readback = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: 'logits-readback' });
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(logits.gpuBuffer, alignedOffset, this.readback, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    try { return new ArrayType(this.readback.getMappedRange().slice(padding, padding + vocab * bytes)); }
    finally { this.readback.unmap(); }
  }

  async step(tokens, suppressedTokens = [], { phase = 'prefill' } = {}) {
    if (!tokens.length || this.pastLength + tokens.length > this.manifest.maxLength) throw new Error('Sequence exceeds KV cache capacity');
    const model = this.manifest.config.model;
    if (tokens.some(token => !Number.isSafeInteger(token) || token < 0 || token >= model.vocab_size)) throw new Error('Token outside model vocabulary');
    let logits, embeddingOutputs, decoderOutputs, embeddingPrepared;
    const captured = phase === 'decode';
    try {
      if (captured) {
        for (const [name, tensor] of [...Object.entries(this.decodeFeeds), ...Object.entries(this.decodeFetches)]) {
          if (tensor?.location !== 'gpu-buffer' || !tensor.gpuBuffer) throw new Error(`Captured GPU binding is missing: ${name}`);
        }
        this.updateDecodeInputs(tokens);
        if (this.sessions.decodeEmbedding) await this.sessions.decodeEmbedding.run(this.embeddingFeeds, this.embeddingFetches);
        try { await this.sessions.decodeDecoder.run(this.decodeFeeds, this.decodeFetches); }
        catch (error) { throw new Error(`Captured decoder failed after ${this.evidence.decodeRuns} successful decode calls: ${error.message}`, { cause: error }); }
        logits = this.decodeFetches[model.decoder.outputs.logits];
        this.evidence.decodeRuns++;
      } else {
        let embedding;
        if (this.sessions.embedding) {
          embeddingPrepared = cpuInputs(this.ort, model.embedding, this.manifest.sessions.embedding, tokens, this.pastLength);
          embeddingOutputs = await this.sessions.embedding.run(embeddingPrepared.feeds);
          embedding = embeddingOutputs[model.embedding.outputs.inputs_embeds];
        }
        const feeds = this.gpuPrefillInputs(model.decoder, this.manifest.sessions.decoder, tokens, embedding);
        const fetches = { [model.decoder.outputs.logits]: null };
        for (const state of this.states) { feeds[state.input] = state.past; fetches[state.output] = state.present; }
        decoderOutputs = await this.sessions.decoder.run(feeds, fetches);
        logits = decoderOutputs[model.decoder.outputs.logits];
        if (!this.decodeFetches[model.decoder.outputs.logits]) {
          const dims = decodeLogitsShape(logits.dims, model.vocab_size);
          this.decodeFetches[model.decoder.outputs.logits] = this.own(logits.type, dims, 'decode/logits');
          this.evidence.decodeOutputShapes = Object.fromEntries(Object.entries(this.decodeFetches).map(([name, tensor]) => [name, [...tensor.dims]]));
        }
      }
      // Hybrid state buffers keep the exact addresses bound during capture.
      // Queue copies before logits readback so measured time includes them.
      this.copyRecurrentState();
      const values = await this.readLogits(logits);
      const token = greedyToken(values, logits.type, model.vocab_size, suppressedTokens);
      this.pastLength += tokens.length;
      return token;
    } finally {
      if (embeddingOutputs) for (const tensor of Object.values(embeddingOutputs)) tensor.dispose();
      if (embeddingPrepared) for (const tensor of embeddingPrepared.temporary) tensor.dispose();
      if (!captured) decoderOutputs?.[model.decoder.outputs.logits]?.dispose();
    }
  }

  disposeBuffers() { for (const tensor of this.owned) tensor.dispose(); this.owned = []; this.readback?.destroy(); this.readback = null; }
  async dispose() {
    // Release captured bindings before destroying their externally owned buffers.
    for (const session of Object.values(this.sessions)) await session.release();
    this.disposeBuffers(); this.disposeStates();
  }
}

export async function loadCapturedGenerator(ort, device, manifest, baseUrl, onProgress = () => {}, options = {}) {
  const sessions = {}, weights = new Map();
  try {
    for (const name of Object.keys(manifest.sessions)) {
      onProgress(`Loading ${manifest.name} ${name}: GPU-bound prefill`);
      sessions[name] = await createModelSession(ort, device, manifest, baseUrl, name, { preferredOutputLocation: 'gpu-buffer' }, weights);
      const captureSession = name === 'decoder' && options.graphCapture !== false;
      onProgress(`Loading ${manifest.name} ${name}: static ${captureSession ? 'captured' : 'GPU-bound'} decode`);
      const plan = staticInputPlan(manifest, name);
      sessions[name === 'decoder' ? 'decodeDecoder' : 'decodeEmbedding'] = await createModelSession(ort, device, manifest, baseUrl, name, {
        // Capture the transformer decoder. Auxiliary embedding sessions can
        // retain CPU shape/empty-modality nodes, so run them uncaptured into
        // the same fixed GPU tensor consumed by the captured decoder.
        preferredOutputLocation: 'gpu-buffer', enableGraphCapture: captureSession, freeDimensionOverrides: plan.overrides,
        ...(options.diagnostics ? { logSeverityLevel: 0, logVerbosityLevel: 1 } : {}),
      }, weights);
    }
    return new CaptureGenerator(ort, device, manifest, sessions, options);
  } catch (error) { for (const session of Object.values(sessions)) await session.release(); throw error; }
}
