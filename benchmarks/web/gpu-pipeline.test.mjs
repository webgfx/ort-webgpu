import {test} from 'node:test';
import assert from 'node:assert/strict';
import {gpuPipelineSupport,configureGpuSampling} from './gpu-pipeline.mjs';
function engine(){return {graphCapture:true,sessions:{},decodeFeeds:{ids:{location:'gpu-buffer',type:'int64'}},
  manifest:{config:{model:{vocab_size:20,eos_token_id:[1,2],decoder:{inputs:{input_ids:'ids'},outputs:{logits:'logits'}}}},sessions:{decoder:{outputs:[{name:'logits',type:'float16'}]}}}};}
test('auto sampling only selects the supported fixed captured decoder topology',()=>{
  const value=engine();assert.equal(gpuPipelineSupport(value).supported,true);
  value.sessions.decodeEmbedding={};assert.equal(gpuPipelineSupport(value).reason,'auxiliary-embedding-session');
  delete value.sessions.decodeEmbedding;value.graphCapture=false;assert.equal(gpuPipelineSupport(value).reason,'uncaptured-decoder');
});
test('CPU selection is explicit for auxiliary sessions and forced GPU requests fail',async()=>{
  const value=engine();value.sessions.decodeEmbedding={};await configureGpuSampling(value,'auto');
  assert.equal(value.evidence.samplingDevice,'cpu');assert.equal(value.evidence.cpuSamplingReason,'auxiliary-embedding-session');
  await assert.rejects(()=>configureGpuSampling(value,'gpu'),/GPU sampling unavailable/);
  await configureGpuSampling(value,'cpu');assert.equal(value.evidence.cpuSamplingReason,'explicit-cpu-policy');
  await assert.rejects(()=>configureGpuSampling(value,'bad'),/Invalid/);
});
test('GPU initialization errors never become automatic CPU success',async()=>{
  const value=engine();value.decodeFeeds.ids.gpuBuffer={size:16};value.device={createBuffer(){throw Error('GPU allocation failed');}};
  const previous=globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage={STORAGE:1,COPY_SRC:2,COPY_DST:4,UNIFORM:8,MAP_READ:16};
  try{
    await assert.rejects(()=>configureGpuSampling(value,'auto'),/GPU allocation failed/);
    assert.notEqual(value.evidence.samplingDevice,'cpu');
  }finally{if(previous===undefined)delete globalThis.GPUBufferUsage;else globalThis.GPUBufferUsage=previous;}
});
