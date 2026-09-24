import {GpuGreedy} from './gpu-greedy.mjs';

export function gpuPipelineSupport(engine) {
  const model=engine.manifest.config.model,ids=engine.decodeFeeds?.[model.decoder.inputs.input_ids];
  if(!engine.graphCapture)return {supported:false,reason:'uncaptured-decoder'};
  if(engine.sessions.decodeEmbedding)return {supported:false,reason:'auxiliary-embedding-session'};
  if(!ids||ids.location!=='gpu-buffer'||!['int32','int64'].includes(ids.type))return {supported:false,reason:'unsupported-token-binding'};
  const spec=engine.manifest.sessions.decoder.outputs.find(output=>output.name===model.decoder.outputs.logits);
  if(!spec||!['float16','float32'].includes(spec.type))return {supported:false,reason:'unsupported-logits-type'};
  const eos=model.eos_token_id===undefined?[]:Array.isArray(model.eos_token_id)?model.eos_token_id:[model.eos_token_id];
  if(new Set(eos).size>8)return {supported:false,reason:'too-many-eos-ids'};
  return {supported:true};
}

export async function configureGpuSampling(engine, policy) {
  if(!['auto','cpu','gpu'].includes(policy))throw Error('Invalid GPU sampling policy');
  engine.evidence ||= {};
  engine.evidence.requestedSamplingDevice=policy;
  const support=gpuPipelineSupport(engine);
  if(policy==='gpu'&&!support.supported)throw Error('GPU sampling unavailable: '+support.reason);
  if(policy!=='cpu'&&support.supported){
    // Initialization or execution failures are errors, never silent CPU fallbacks.
    await enableGpuPipeline(engine);
  }else{
    engine.evidence.samplingDevice='cpu';
    engine.evidence.cpuSamplingReason=policy==='cpu'?'explicit-cpu-policy':support.reason;
  }
  return engine;
}

// Captured single-graph models can consume the next
// token directly from GPU selection. CPU-readable token delivery remains ordered
// and bounded; TTFT includes the first readable token, and completion awaits all.
export async function enableGpuPipeline(engine, {maxPending = 16} = {}) {
  const model=engine.manifest.config.model,ids=engine.decodeFeeds[model.decoder.inputs.input_ids];
  if(!gpuPipelineSupport(engine).supported) {
    throw Error('GPU token pipeline requires a captured, single-graph text decoder');
  }
  const spec=engine.manifest.sessions.decoder.outputs.find(output=>output.name===model.decoder.outputs.logits);
  const eos=model.eos_token_id===undefined?[]:Array.isArray(model.eos_token_id)?model.eos_token_id:[model.eos_token_id];
  const sampler=await GpuGreedy.create(engine.device,model.vocab_size,spec.type,eos,ids.gpuBuffer,maxPending);
  const upload=engine.upload.bind(engine),dispose=engine.dispose.bind(engine);
  let tokenOnGpu=false;
  engine.upload=(tensor,values)=>{if(tokenOnGpu&&tensor===ids)return;upload(tensor,values);};
  engine.evidence.samplingDevice='gpu';
  engine.evidence.gpuGreedy={algorithm:'exact-ieee-order',maxPendingReadbacks:maxPending,delivery:'ordered-cpu-stream',firstToken:'cpu-greedy'};
  engine.generate=async function(prompt,generationLength,{prefillChunkSize=0,onToken}={}){
    if(!Number.isSafeInteger(generationLength)||generationLength<2||prompt.length+generationLength>this.manifest.maxLength)throw Error('Generation length must fit the KV cache');
    if(!Number.isSafeInteger(prefillChunkSize)||prefillChunkSize<0||prefillChunkSize>this.manifest.maxLength||!prompt.length)throw Error('Invalid prefill chunk size or empty prompt');
    await this.reset();
    const start=this.now(),chunk=prefillChunkSize||prompt.length;
    let first;
    for(let offset=0;offset<prompt.length;offset+=chunk)first=await this.step(prompt.slice(offset,offset+chunk),eos,{phase:'prefill'});
    const firstAt=this.now(),generated=Array(generationLength),ready=Array(generationLength).fill(false),tokenDeliveryMs=Array(generationLength);
    generated[0]=first;ready[0]=true;tokenDeliveryMs[0]=firstAt-start;onToken?.(first,0);
    let nextToDeliver=1;
    let generationError;
    const pending=[];
    // Only the first decode input crosses CPU -> GPU; later IDs are written by
    // the sampler to this same captured binding, in WebGPU queue order.
    upload(ids,[first]);tokenOnGpu=true;
    try{
      for(let index=1;index<generationLength;index++){
        if(generationError)throw generationError;
        for(const [name,tensor]of [...Object.entries(this.decodeFeeds),...Object.entries(this.decodeFetches)]){
          if(tensor?.location!=='gpu-buffer'||!tensor.gpuBuffer)throw Error('Missing captured GPU binding '+name);
        }
        this.updateDecodeInputs([0]); // Upload wrapper preserves the GPU-selected ID.
        await this.sessions.decodeDecoder.run(this.decodeFeeds,this.decodeFetches);
        this.evidence.decodeRuns++;
        this.copyRecurrentState();
        const selected=await sampler.enqueue(this.decodeFetches[model.decoder.outputs.logits]);
        this.pastLength++;
        const delivery=selected.token.then(token=>{
          generated[index]=token;ready[index]=true;
          while(nextToDeliver<generationLength&&ready[nextToDeliver]){
            tokenDeliveryMs[nextToDeliver]=this.now()-start;
            onToken?.(generated[nextToDeliver],nextToDeliver);nextToDeliver++;
          }
        });
        delivery.catch(error=>{generationError ||= error;}); // Fail promptly; still drain queued work.
        pending.push(delivery);
      }
      await Promise.all(pending);
      const end=this.now(),ttftMs=firstAt-start,decodeMs=end-firstAt;
      if(nextToDeliver!==generationLength||!(ttftMs>0&&decodeMs>0))throw Error('Incomplete or invalid GPU generation');
      return {ttftMs,prefillTps:prompt.length*1000/ttftMs,decodeTps:(generationLength-1)*1000/decodeMs,
        e2eMs:end-start,generated,tokenDeliveryMs};
    }finally{
      tokenOnGpu=false;
      await Promise.allSettled(pending);await sampler.drain();
    }
  };
  engine.dispose=async()=>{try{await sampler.drain();await dispose();}finally{sampler.dispose();}};
  return engine;
}
