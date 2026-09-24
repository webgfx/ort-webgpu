export function parseSettings({ lengths, tokens, repetitions, graphCapture, gpuSampling, diagnostics }) {
  const promptLengths = lengths.split(',').map(value => Number(value.trim()));
  const generationLength = Number(tokens), reps = Number(repetitions);
  if (!promptLengths.length || new Set(promptLengths).size !== promptLengths.length || promptLengths.some(n => !Number.isSafeInteger(n) || n < 1)
      || !Number.isSafeInteger(generationLength) || generationLength < 2 || Math.max(...promptLengths) + generationLength > 8192
      || !Number.isSafeInteger(reps) || reps < 1 || reps > 100) throw Error('Use unique positive input lengths, 2+ output tokens, 1–100 repetitions, and at most 8,192 total tokens');
  if (!['auto', 'cpu', 'gpu'].includes(gpuSampling)) throw Error('Invalid sampling policy');
  if (!graphCapture && gpuSampling === 'gpu') throw Error('GPU sampling requires graph capture');
  return { promptLengths, generationLength, repetitions: reps, graphCapture, gpuSampling, diagnostics };
}
export function csvResults(result) {
  const escape = value => '"' + String(value).replaceAll('"', '""').replace(/^[=+@-]/, "'$&") + '"';
  const rows = [['Model', 'Input tokens', 'Output tokens', 'TTFT throughput (tokens/s)', 'Decode (tokens/s)', 'Run complete']];
  for (const model of result.models || []) for (const row of model.rows || []) rows.push([model.name, row.pl, row.tg, row.plTs, row.tgTs, result.success]);
  return rows.map(row => row.map(escape).join(',')).join('\r\n');
}
