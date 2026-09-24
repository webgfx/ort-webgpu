import fs from 'node:fs';
import path from 'node:path';
import { greedyShader } from '../web/gpu-greedy.mjs';
const output = process.argv[2];
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, '#pragma once\n' + [true, false].map(half =>
  `inline constexpr char kGreedy${half ? 'Half' : 'Float'}[] = R"WGSL(${greedyShader(half)})WGSL";\n`).join(''));
