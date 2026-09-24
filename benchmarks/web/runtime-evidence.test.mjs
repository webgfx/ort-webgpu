import {test} from 'node:test';
import assert from 'node:assert/strict';
import {jspiLoadEvidence} from './runtime-evidence.mjs';
const names=['ort.all.min.js','ort-wasm-simd-threaded.jspi.mjs','ort-wasm-simd-threaded.jspi.wasm'];
const build={wasmVariant:'jspi',artifacts:Object.fromEntries(names.map(file=>[file,{sha256:'a'.repeat(64)}]))};
const responses=names.map(file=>({file,status:200}));
test('JSPI evidence requires successful observations of both pinned artifacts',()=>{
  assert.equal(jspiLoadEvidence(build,responses).observedArtifacts.length,2);
  assert.throws(()=>jspiLoadEvidence(build,responses.slice(0,2)),/not successfully loaded/);
  assert.throws(()=>jspiLoadEvidence(build,responses.map(r=>({...r,status:404}))),/not successfully loaded/);
  assert.throws(()=>jspiLoadEvidence({...build,wasmVariant:'asyncify'},responses),/Expected a JSPI/);
  assert.throws(()=>jspiLoadEvidence(build,[...responses,{file:'ort-wasm-simd-threaded.wasm',status:200}]),/Unexpected/);
  assert.throws(()=>jspiLoadEvidence({...build,artifacts:{}},responses),/Unexpected/);
});
