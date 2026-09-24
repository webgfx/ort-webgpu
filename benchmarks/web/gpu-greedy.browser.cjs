const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
(async()=>{
  const {default:puppeteer}=await import('puppeteer-core');
  const source=fs.readFileSync(path.join(__dirname,'gpu-greedy.mjs'));
  const server=http.createServer((req,res)=>{
    res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
    if(req.url==='/gpu-greedy.mjs'){res.setHeader('Content-Type','text/javascript');res.end(source);}
    else if(req.url==='/gpu-pipeline.mjs'){res.setHeader('Content-Type','text/javascript');res.end(fs.readFileSync(path.join(__dirname,'gpu-pipeline.mjs')));}
    else if(req.url==='/favicon.ico'){res.writeHead(204).end();}
    else if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>GPU greedy correctness test</title>');}
    else res.writeHead(404).end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await puppeteer.launch({executablePath:process.env.WEBGFX_UI_TEST_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
    const page=await browser.newPage();page.on('console',message=>{if(message.type()==='error')console.error(message.text());});
    await page.goto('http://127.0.0.1:'+server.address().port+'/');
    const result=await page.evaluate(async()=>{
      const {GpuGreedy}=await import('/gpu-greedy.mjs');
      const adapter=await navigator.gpu.requestAdapter(),device=await adapter.requestDevice();
      const errors=[];device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
      const next=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      let checked=0;
      async function check(values,type,vocabulary,eos,expected,error){
        const data=type==='float16'?new Uint16Array(values):new Float32Array(values);
        const buffer=device.createBuffer({size:Math.max(16,Math.ceil(data.byteLength/16)*16),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
        const padded=new Uint8Array(Math.ceil(data.byteLength/4)*4);padded.set(new Uint8Array(data.buffer));
        device.queue.writeBuffer(buffer,0,padded);
        const sampler=await GpuGreedy.create(device,vocabulary,type,eos,next,2);
        try{
          const logits={type,location:'gpu-buffer',dims:[1,values.length/vocabulary,vocabulary],gpuBuffer:buffer};
          // Exercise both reuse and multiple queued, independently validated results.
          const pending=[];for(let i=0;i<5;i++)pending.push((await sampler.enqueue(logits)).token);
          const actual=await Promise.allSettled(pending);
          for(const item of actual){
            if(error){if(item.status!=='rejected'||!item.reason.message.includes(error))throw Error('Expected '+error);}
            else if(item.status!=='fulfilled'||item.value!==expected)throw Error('GPU selection mismatch '+JSON.stringify(item));
          }
          checked+=5;
        }finally{await sampler.drain();sampler.dispose();buffer.destroy();}
      }
      try{
        await check([0x3c00,0x4000,0x4200],'float16',3,[],2);
        await check([0x7c00,0xfc00,0x7e00,0x4000,0x3c00,0x4200],'float16',3,[2],0);
        await check([0x8000,0,0x8000],'float16',3,[],0);
        await check([0xbc00,0xc000,0x8001],'float16',3,[],2);
        await check([0,1,2,2],'float16',4,[],2);
        await check([0x3c00,0x4000],'float16',2,[0,1],null,'No valid');
        for(const bad of [0x7c00,0xfc00,0x7e00])await check([bad,0],'float16',2,[0],null,'Non-finite');
        const finite=Array.from({length:65536},(_,i)=>i).filter(bits=>(bits&0x7c00)!==0x7c00);
        await check(finite,'float16',finite.length,[],finite.indexOf(0x7bff));
        await check([1,3,3,-2],'float32',4,[],1);
        await check([-0,0,-1],'float32',3,[],0);
        await check([100,2,3],'float32',3,[0],2);
        for(const bad of [NaN,Infinity,-Infinity])await check([1,bad],'float32',2,[1],null,'Non-finite');
        const large=Array.from({length:2051},(_,i)=>i%331-500);large[1024]=3;large[2049]=3;
        await check(large,'float32',large.length,[],1024);
        const {enableGpuPipeline}=await import('/gpu-pipeline.mjs');
        const vocabulary=8;
        const logits=device.createBuffer({size:vocabulary*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
        const ids=device.createBuffer({size:16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
        const module=device.createShaderModule({code:`@group(0) @binding(0) var<storage,read> ids: array<u32>;
          @group(0) @binding(1) var<storage,read_write> scores: array<f32>;
          @compute @workgroup_size(8) fn main(@builtin(local_invocation_index) i:u32) {
            scores[i]=select(-1.0,10.0,i==(ids[0]+1u)%8u);
          }`});
        const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
        const binding=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ids}},{binding:1,resource:{buffer:logits}}]});
        const idTensor={type:'int64',location:'gpu-buffer',dims:[1,1],gpuBuffer:ids};
        const logitsTensor={type:'float32',location:'gpu-buffer',dims:[1,1,vocabulary],gpuBuffer:logits};
        const engine={device,graphCapture:true,pastLength:0,evidence:{decodeRuns:0},
          manifest:{maxLength:8192,config:{model:{vocab_size:vocabulary,eos_token_id:[0],decoder:{inputs:{input_ids:'ids'},outputs:{logits:'logits'}}}},sessions:{decoder:{outputs:[{name:'logits',type:'float32'}]}}},
          decodeFeeds:{ids:idTensor},decodeFetches:{logits:logitsTensor},now:()=>performance.now(),
          upload(tensor,values){device.queue.writeBuffer(tensor.gpuBuffer,0,new BigInt64Array([BigInt(values[0])]));},
          updateDecodeInputs(tokens){this.upload(idTensor,tokens);},copyRecurrentState(){},
          async reset(){this.pastLength=0;await device.queue.onSubmittedWorkDone();},
          async step(prompt){this.pastLength+=prompt.length;device.queue.writeBuffer(logits,0,new Float32Array(vocabulary));await device.queue.onSubmittedWorkDone();return (prompt.at(-1)+1)%8||1;},
          sessions:{decodeDecoder:{async run(){const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,binding);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);}}},
          async dispose(){ids.destroy();logits.destroy();},
        };
        await enableGpuPipeline(engine,{maxPending:3});
        for(const seed of [0,5,7]){
          const delivered=[],sample=await engine.generate([seed],40,{onToken:(token,index)=>delivered.push({token,index})});
          let token=(seed+1)%8||1;const expected=[token];for(let i=1;i<40;i++){token=(token+1)%8||1;expected.push(token);}
          if(JSON.stringify(sample.generated)!==JSON.stringify(expected)||delivered.length!==40)throw Error('Pipelined autoregressive feedback mismatch');
          for(let i=0;i<40;i++)if(delivered[i].index!==i||delivered[i].token!==expected[i]||sample.tokenDeliveryMs[i]<0||(i&&sample.tokenDeliveryMs[i]<sample.tokenDeliveryMs[i-1]))throw Error('Token delivery was not ordered');
          if(sample.e2eMs<sample.tokenDeliveryMs.at(-1)||sample.ttftMs!==sample.tokenDeliveryMs[0])throw Error('Timing completed before CPU token delivery');
        }
        if(engine.evidence.decodeRuns!==117)throw Error('Skipped decoder calls');
        const originalRun=engine.sessions.decodeDecoder.run;
        engine.sessions.decodeDecoder.run=async()=>{device.queue.writeBuffer(logits,0,new Float32Array([NaN,1,2,3,4,5,6,7]));};
        let rejected=false;try{await engine.generate([1],40);}catch(error){rejected=error.message.includes('Non-finite');}
        if(!rejected)throw Error('Pipelined invalid scores were accepted');
        engine.sessions.decodeDecoder.run=originalRun;
        const recovered=await engine.generate([1],5);
        if(JSON.stringify(recovered.generated)!==JSON.stringify([2,3,4,5,6]))throw Error('Readback pool did not recover after failure');
        await engine.dispose();
        await device.queue.onSubmittedWorkDone();if(errors.length)throw Error(errors.join('; '));
        return {checked,finiteHalfPatterns:finite.length,pipelinedTokens:120,adapter:adapter.info.vendor};
      }finally{next.destroy();device.destroy();}
    });
    assert.equal(result.checked,85);console.log('PASS GPU greedy',JSON.stringify(result));
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
