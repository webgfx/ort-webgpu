// Python verifies artifact bytes before launch; the browser response evidence
// proves which of those pinned artifacts actually loaded, not only intent.
export function jspiLoadEvidence(build, responses) {
  if (build.wasmVariant !== 'jspi') throw new Error('Expected a JSPI runtime');
  const names = ['ort-wasm-simd-threaded.jspi.mjs', 'ort-wasm-simd-threaded.jspi.wasm'];
  for (const response of responses) {
    if (!Object.hasOwn(build.artifacts, response.file)) throw new Error(`Unexpected runtime artifact: ${response.file}`);
  }
  return { wasmVariant: 'jspi', observedArtifacts: names.map(file => {
    if (!responses.some(response => response.file === file && response.status === 200)) {
      throw new Error(`JSPI artifact was not successfully loaded: ${file}`);
    }
    const sha256 = build.artifacts[file]?.sha256;
    if (!/^[a-f0-9]{64}$/.test(sha256 || '')) throw new Error(`Missing pinned JSPI hash: ${file}`);
    return { file, sha256 };
  }) };
}
