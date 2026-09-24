// No model/GPU benchmark: exercise the production module with Edge's native
// Float16Array as an independent reference, plus browsers without that API.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const {default: puppeteer} = await import('puppeteer-core');
  const browser = await puppeteer.launch({headless: true,
    executablePath: process.env.WEBGFX_UI_TEST_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const page = await browser.newPage();
    const source = fs.readFileSync(path.join(__dirname, 'generator.mjs'), 'utf8');
    const result = await page.evaluate(async source => {
      const moduleUrl = URL.createObjectURL(new Blob([source], {type: 'text/javascript'}));
      const {greedyToken, halfToFloat} = await import(moduleUrl);
      URL.revokeObjectURL(moduleUrl);
      const Native = globalThis.Float16Array;
      if (typeof Native !== 'function') throw Error('This browser must support native Float16Array for the fast-path test');
      const raw = Uint16Array.from({length: 65536}, (_, i) => i), floats = new Native(raw.buffer);
      for (let i = 0; i < raw.length; i++) {
        if (!Object.is(floats[i], halfToFloat(raw[i]))) throw Error('Conversion mismatch at ' + i);
        const pair=new Uint16Array([0, i]);
        if(Number.isFinite(floats[i])) {
          const expected=floats[i]>0?1:0;
          if(greedyToken(pair,'float16',2)!==expected)throw Error('FP16 ordering mismatch at '+i);
          if(greedyToken(pair,'float16',2,[0])!==1)throw Error('FP16 suppression mismatch at '+i);
        } else {
          let rejected=false;try{greedyToken(pair,'float16',2,[1]);}catch(e){rejected=e.message.includes('Non-finite');}
          if(!rejected)throw Error('Non-finite FP16 was accepted at '+i);
        }
      }
      // Cover the complete finite ordering, not only comparison against zero.
      const finite=Array.from(raw).filter(bits=>(bits&0x7c00)!==0x7c00).sort((a,b)=>floats[a]-floats[b]);
      for(let i=1;i<finite.length;i++) {
        const a=finite[i-1],b=finite[i],expected=floats[b]>floats[a]?1:0;
        if(greedyToken(new Uint16Array([a,b]),'float16',2)!==expected)throw Error('Adjacent FP16 ordering mismatch');
      }
      let views = 0;
      globalThis.Float16Array = new Proxy(Native, {construct(target, args) {
        const view = Reflect.construct(target, args);
        if (view.buffer !== args[0] || view.byteOffset !== args[1] || view.length !== args[2]) throw Error('Not a zero-copy subview');
        views++; return view;
      }});
      const scenarios = [
        {bits: [0x4000, 0x3c00, 0xbc00, 1], vocab: 2, expected: 1},
        {bits: [0x4000, 0x3c00], vocab: 2, suppressed: [0], expected: 1},
        {bits: [0x8000, 0], vocab: 2, expected: 0},
        {bits: [0xbc00, 0xc000], vocab: 2, expected: 0},
        {bits: [0, 1], vocab: 2, expected: 1},
        {bits: [0x7c00, 0xfc00, 0, 1], vocab: 2, expected: 1},
        {bits: [0x7e00, 0], vocab: 2, error: 'Non-finite'},
        {bits: [0x7c00, 0], vocab: 2, suppressed: [0], error: 'Non-finite'},
        {bits: [0xfc00, 0], vocab: 2, error: 'Non-finite'},
        {bits: [0, 1], vocab: 2, suppressed: [0, 1], error: 'No valid next token'},
      ];
      function check(scenario) {
        const backing = new Uint16Array([0x7c00, ...scenario.bits, 0x7c00]);
        const data = backing.subarray(1, backing.length - 1), before = [...backing];
        try {
          const token = greedyToken(data, 'float16', scenario.vocab, scenario.suppressed);
          if (scenario.error || token !== scenario.expected) throw Error('Unexpected argmax result');
        } catch (error) {
          if (!scenario.error || !error.message.includes(scenario.error)) throw error;
        }
        if (JSON.stringify([...backing]) !== JSON.stringify(before)) throw Error('Input was mutated');
      }
      try {
        scenarios.forEach(check);
        const nativeViews = views;
        if (nativeViews !== scenarios.length) throw Error('Native view was not used for every FP16 input');
        // Float32 and already-numeric Float16 tensors must not be reinterpreted.
        if (greedyToken(new Float32Array([1, 2]), 'float32', 2) !== 1 ||
            greedyToken(new Native([1, 2]), 'float16', 2) !== 1 || views !== nativeViews) throw Error('Unnecessary reinterpretation');
        globalThis.Float16Array = undefined;
        scenarios.forEach(check);
        return {patternsChecked: raw.length, nativeCases: scenarios.length, fallbackCases: scenarios.length, zeroCopyViews: nativeViews};
      } finally {globalThis.Float16Array = Native;}
    }, source);
    assert.equal(result.patternsChecked, 65536);
    assert.equal(result.zeroCopyViews, result.nativeCases);
    console.log('PASS: native/fallback production FP16 argmax', JSON.stringify(result));
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
