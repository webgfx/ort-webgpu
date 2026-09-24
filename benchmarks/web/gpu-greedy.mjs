// Exact GPU greedy selection over one logits row. No softmax or precision
// conversion; all logits, including suppressed entries, must remain finite.
function shader(half) {
  return `
struct Params { dimensions: vec4<u32>, eos: array<vec4<u32>, 2>, };
@group(0) @binding(0) var<storage, read> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> partial: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> next_token: array<u32>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
var<workgroup> values: array<vec4<u32>, 256>;
fn merge(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let take_b = b.x > a.x || (b.x == a.x && b.y < a.y);
  return vec4<u32>(select(a.x, b.x, take_b), select(a.y, b.y, take_b), a.z | b.z, 0u);
}
fn suppressed(token: u32) -> bool {
  for(var i=0u; i<params.dimensions.w; i++) { if(params.eos[i/4u][i%4u] == token) { return true; } }
  return false;
}
fn reduce(lane: u32) {
  workgroupBarrier();
  for(var stride=128u; stride>0u; stride/=2u) {
    if(lane<stride) { values[lane]=merge(values[lane],values[lane+stride]); }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  var best=vec4<u32>(0u,0xffffffffu,0u,0u);
  for(var i=group.x*256u+lane; i<params.dimensions.x; i+=params.dimensions.z*256u) {
    let element=params.dimensions.y+i;
    ${half ? 'let bits=(logits[element/2u] >> ((element%2u)*16u)) & 0xffffu;' : 'let bits=logits[element];'}
    let invalid=${half ? '(bits & 0x7c00u)==0x7c00u' : '(bits & 0x7f800000u)==0x7f800000u'};
    best.z |= select(0u,1u,invalid);
    if(!invalid && !suppressed(i)) {
      let magnitude=bits & ${half ? '0x7fffu' : '0x7fffffffu'};
      let sign=bits & ${half ? '0x8000u' : '0x80000000u'};
      let center=${half ? '0x8000u' : '0x80000000u'};
      let rank=select(center+magnitude,center-magnitude,sign!=0u);
      best=merge(best,vec4<u32>(rank,i,0u,0u));
    }
  }
  values[lane]=best;reduce(lane);
  if(lane==0u) { partial[group.x]=values[0]; }
}
@compute @workgroup_size(256)
fn finish(@builtin(local_invocation_index) lane: u32) {
  var best=vec4<u32>(0u,0xffffffffu,0u,0u);
  for(var i=lane; i<params.dimensions.z; i+=256u) { best=merge(best,partial[i]); }
  values[lane]=best;reduce(lane);
  if(lane==0u) {
    let value=values[0];let missing=value.y==0xffffffffu;
    let token=select(value.y,0u,missing || value.z!=0u);
    next_token[0]=token;next_token[1]=0u;
    result[0]=token;result[1]=value.z | select(0u,2u,missing);result[2]=value.x;result[3]=0u;
  }
}`;
}

export { shader as greedyShader };

export class GpuGreedy {
  static async create(device, vocabulary, type, suppressedTokens, nextTokenBuffer, maxPending = 16) {
    if(!Number.isSafeInteger(vocabulary)||vocabulary<1||vocabulary>0x7fffffff||!['float16','float32'].includes(type)) throw Error('Unsupported GPU logits');
    if(!Number.isSafeInteger(maxPending)||maxPending<1||maxPending>64||nextTokenBuffer.size<8)throw Error('Invalid GPU sampling buffers');
    const eos=[...new Set(suppressedTokens)].filter(token=>Number.isSafeInteger(token)&&token>=0&&token<vocabulary);
    if(eos.length>8) throw Error('GPU greedy currently supports at most eight EOS IDs');
    const instance=new GpuGreedy();
    Object.assign(instance,{device,vocabulary,type,eos,nextTokenBuffer,groups:Math.ceil(vocabulary/1024),maxPending,slots:[],position:0,buffers:[]});
    try{
      const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
      const own=(size,usage,label)=>{const buffer=device.createBuffer({size,usage,label});instance.buffers.push(buffer);return buffer;};
      instance.partial=own(instance.groups*16,storage,'greedy-partial');
      instance.result=own(16,storage,'greedy-result');
      instance.params=own(48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,'greedy-params');
      const module=device.createShaderModule({code:shader(type==='float16'),label:'exact-greedy-selection'});
      const errors=(await module.getCompilationInfo()).messages.filter(message=>message.type==='error');
      if(errors.length)throw Error(errors.map(message=>`${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
      instance.scan=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'scan'}});
      instance.finish=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'finish'}});
      instance.cache=new WeakMap();
      for(let i=0;i<maxPending;i++)instance.slots.push({buffer:own(16,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,'greedy-readback'),pending:Promise.resolve()});
      return instance;
    }catch(error){instance.dispose();throw error;}
  }

  async enqueue(logits) {
    if(logits.type!==this.type||logits.location!=='gpu-buffer'||logits.dims.at(-1)!==this.vocabulary)throw Error('GPU greedy logits mismatch');
    const slot=this.slots[this.position++%this.maxPending];
    await slot.pending; // Bounded streaming readbacks; no full-vocabulary download.
    const elements=logits.dims.reduce((a,b)=>a*b,1);
    if(!Number.isSafeInteger(elements)||elements<this.vocabulary||elements%this.vocabulary)throw Error('Incomplete GPU logits row');
    const params=new Uint32Array(12);params.set([this.vocabulary,elements-this.vocabulary,this.groups,this.eos.length]);params.set(this.eos,4);
    this.device.queue.writeBuffer(this.params,0,params);
    let bindings=this.cache.get(logits.gpuBuffer);
    if(!bindings){
      const entry=(binding,buffer)=>({binding,resource:{buffer}});
      bindings={scan:this.device.createBindGroup({layout:this.scan.getBindGroupLayout(0),entries:[entry(0,logits.gpuBuffer),entry(1,this.partial),entry(2,this.params)]}),
        finish:this.device.createBindGroup({layout:this.finish.getBindGroupLayout(0),entries:[entry(1,this.partial),entry(2,this.params),entry(3,this.nextTokenBuffer),entry(4,this.result)]})};
      this.cache.set(logits.gpuBuffer,bindings);
    }
    const encoder=this.device.createCommandEncoder();
    for(const [pipeline,group,count]of [[this.scan,bindings.scan,this.groups],[this.finish,bindings.finish,1]]){
      const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(count);pass.end();
    }
    encoder.copyBufferToBuffer(this.result,0,slot.buffer,0,16);this.device.queue.submit([encoder.finish()]);
    const token=slot.buffer.mapAsync(GPUMapMode.READ).then(()=>{
      try{
        const value=new Uint32Array(slot.buffer.getMappedRange());
        if(value[1]&1)throw Error('Non-finite GPU logits');
        if(value[1]&2)throw Error('No valid next GPU token');
        if(value[0]>=this.vocabulary)throw Error('GPU token outside vocabulary');
        return value[0];
      }finally{slot.buffer.unmap();}
    });
    // Observe rejections immediately; the caller still receives and checks the
    // original rejecting promise. Do not turn a bad sample into success.
    slot.pending=token.then(()=>{},()=>{});
    return {token};
  }
  async drain(){await Promise.all(this.slots.map(slot=>slot.pending));}
  dispose(){for(const buffer of this.buffers||[])buffer.destroy();this.buffers=[];}
}
