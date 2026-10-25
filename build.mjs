// esbuild script → dist/
import * as esbuild from 'esbuild';
import { cpSync, existsSync, mkdirSync, rmSync, readdirSync } from 'node:fs';

const watch = process.argv.includes('--watch');
const out = 'dist';

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

function copyStatic() {
  cpSync('manifest.json', `${out}/manifest.json`);
  cpSync('src/offscreen/offscreen.html', `${out}/offscreen/offscreen.html`);
  cpSync('src/popup/popup.html', `${out}/popup/popup.html`);
  cpSync('src/popup/popup.css', `${out}/popup/popup.css`);
  cpSync('src/welcome/welcome.html', `${out}/welcome/welcome.html`);
  cpSync('src/welcome/welcome.css', `${out}/welcome/welcome.css`);
  if (existsSync('icons')) cpSync('icons', `${out}/icons`, { recursive: true });

  // ORT loads its .wasm (and JSEP glue .mjs) at runtime from env.wasm.wasmPaths.
  // The webgpu bundle uses the JSEP build for both the WebGPU and WASM EPs.
  const ortDist = 'node_modules/onnxruntime-web/dist';
  mkdirSync(`${out}/ort`, { recursive: true });
  for (const f of readdirSync(ortDist)) {
    if (/^ort-wasm-simd-threaded\.jsep\.(wasm|mjs)$/.test(f)) cpSync(`${ortDist}/${f}`, `${out}/ort/${f}`);
  }

  if (existsSync('models')) {
    mkdirSync(`${out}/models`, { recursive: true });
    for (const f of readdirSync('models')) {
      if (/\.(onnx|json)$/.test(f)) cpSync(`models/${f}`, `${out}/models/${f}`);
    }
  }
}

const common = {
  bundle: true,
  sourcemap: 'inline',
  target: 'chrome120',
  logLevel: 'info',
  loader: { '.css': 'text' },
};

const builds = [
  // Content scripts can't be ES modules → iife.
  { ...common, entryPoints: ['src/content/index.ts'], outfile: `${out}/content/index.js`, format: 'iife' },
  { ...common, entryPoints: ['src/background/sw.ts'], outfile: `${out}/background/sw.js`, format: 'esm' },
  { ...common, entryPoints: ['src/offscreen/infer.ts'], outfile: `${out}/offscreen/infer.js`, format: 'esm', sourcemap: false },
  { ...common, entryPoints: ['src/popup/popup.ts'], outfile: `${out}/popup/popup.js`, format: 'iife' },
  { ...common, entryPoints: ['src/welcome/welcome.ts'], outfile: `${out}/welcome/welcome.js`, format: 'iife' },
];

if (watch) {
  copyStatic();
  for (const b of builds) (await esbuild.context(b)).watch();
} else {
  await Promise.all(builds.map((b) => esbuild.build(b)));
  copyStatic();
}
