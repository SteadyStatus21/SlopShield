// Load ONNX session, preprocess, run, return P(AI). Runs in the offscreen document.
import * as ort from 'onnxruntime-web/webgpu';
import { ENGINE, MODEL, OFFSCREEN_IDLE_MS } from '../config';
import { log, warn } from '../log';
import type { BackendInfo, InferResp, LabelResp, ModelMeta, OffscreenMsg } from '../types';

ort.env.wasm.wasmPaths = chrome.runtime.getURL('ort/');
// Offscreen docs aren't cross-origin isolated, so no SharedArrayBuffer threads.
ort.env.wasm.numThreads = 1;

interface Loaded {
  session: ort.InferenceSession;
  meta: ModelMeta;
  backend: BackendInfo['backend'];
}

let loading: Promise<Loaded> | null = null;
let loadingGpu = false;

async function loadMeta(): Promise<ModelMeta> {
  try {
    const r = await fetch(chrome.runtime.getURL(MODEL.metaPath));
    if (r.ok) return { ...MODEL.fallbackMeta, ...(await r.json()) };
  } catch {
    /* fall through */
  }
  warn('models/model.json missing; using fallback preprocessing');
  return MODEL.fallbackMeta;
}

async function load(gpu: boolean): Promise<Loaded> {
  const meta = await loadMeta();
  const r = await fetch(chrome.runtime.getURL(MODEL.path));
  if (!r.ok) throw new Error(`model not found at ${MODEL.path} — run scripts/fetch_model.sh`);
  const bytes = new Uint8Array(await r.arrayBuffer());

  // WebGPU is opt-in: on weak integrated GPUs a large model can stall the GPU process the
  // whole browser UI depends on. Single-threaded WASM only ever costs one CPU core.
  const hasWebGPU = gpu && 'gpu' in navigator &&!!(await (navigator as any).gpu.requestAdapter().catch(() => null));
  if (hasWebGPU) {
    try {
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
      log('model loaded, backend = webgpu', meta.name ?? '');
      return { session, meta, backend: 'webgpu' };
    } catch (e) {
      warn('webgpu session failed, falling back to wasm', e);
    }
  }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  log('model loaded, backend = wasm', meta.name ?? '');
  return { session, meta, backend: 'wasm' };
}

function getModel(gpu: boolean): Promise<Loaded> {
  if (loading && loadingGpu !== gpu) {
    // Backend setting changed: drop the old session and load on the new backend.
    const old = loading;
    loading = null;
    old.then((m) => m.session.release()).catch(() => {});
  }
  loadingGpu = gpu;
  loading ??= load(gpu).catch((e) => {
    loading = null; // allow a retry on the next request
    throw e;
  });
  return loading;
}

async function toTensor(dataUrl: string, meta: ModelMeta): Promise<ort.Tensor> {
  const S = meta.size;
  const bitmap = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const canvas = new OffscreenCanvas(S, S);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, S, S);
  bitmap.close();
  const px = ctx.getImageData(0, 0, S, S).data;

  // HWC uint8 → NCHW float32, rescale to [0,1], then normalize per the model card.
  const plane = S * S;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (px[i * 4 + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  return new ort.Tensor('float32', out, [1, 3, S, S]);
}

function pAI(logits: Float32Array, aiIndex: number, logitBias = 0): number {
  // Single raw-logit output; logitBias applies the model's published calibration.
  if (logits.length === 1) return 1 / (1 + Math.exp(-(logits[0] + logitBias)));
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps[aiIndex] / sum;
}

// ---------- desktop engine (engine/slopshield_engine.py) ----------
// When the local engine is running, crops are scored there (native OpenVINO on the iGPU, ~3x faster
// than WebGPU). If it isn't running, or fails, everything falls back to the in-browser model below.

let engine: WebSocket | null = null;
let engineDevice: string | null = null; // set once the engine has said hello
let lastEngineAttempt = 0;
// Replies from the engine, by request id (raw message: {type: 'scores' | 'labeled' | 'error', ...}).
const pending = new Map<number, (reply: any) => void>();
let nextId = 1;

function connectEngine() {
  if (engine || Date.now() - lastEngineAttempt < ENGINE.retryMs) return;
  lastEngineAttempt = Date.now();
  let ws: WebSocket;
  try {
    ws = new WebSocket(ENGINE.url);
  } catch {
    return;
  }
  engine = ws;
  ws.onopen = () => ws.send(JSON.stringify({ type: 'hello' }));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(String(ev.data));
    if (msg.type === 'hello') {
      engineDevice = msg.device;
      log('desktop engine connected:', msg.device, msg.model);
    } else if (pending.has(msg.id)) {
      pending.get(msg.id)!(msg);
      pending.delete(msg.id);
    }
  };
  ws.onclose = () => {
    if (engineDevice) log('desktop engine disconnected; using in-browser model');
    engine = null;
    engineDevice = null;
    for (const resolve of pending.values()) resolve({ type: 'error', error: 'engine disconnected' });
    pending.clear();
  };
  ws.onerror = () => {}; // onclose follows; not running is the normal case
}

/** Send one request to the engine and wait for its reply (rejects on timeout or engine error). */
function askEngine(payload: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!(engine?.readyState === WebSocket.OPEN && engineDevice)) return reject(new Error('desktop engine is not running'));
    const id = nextId++;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('engine timeout'));
    }, ENGINE.timeoutMs);
    pending.set(id, (reply) => {
      clearTimeout(timer);
      if (reply.type === 'error') reject(new Error(reply.error ?? 'engine error'));
      else resolve(reply);
    });
    engine.send(JSON.stringify({ ...payload, id }));
  });
}

async function scoreOnEngine(images: string[]): Promise<number[]> {
  return (await askEngine({ type: 'score', images })).scores;
}

/**
 * A freshly created offscreen document gets its first request before the engine socket has said
 * hello. Wait briefly instead of cold-loading the in-browser model on WebGPU, which stalls the GPU
 * (videos flicker) for many seconds when the engine would have answered in ~0.1 s.
 */
async function engineReady(maxMs = 1500): Promise<boolean> {
  const until = Date.now() + maxMs;
  while (engine && !engineDevice && engine.readyState <= WebSocket.OPEN && Date.now() < until) {
    await new Promise((r) => setTimeout(r, 50));
  }
  return engine?.readyState === WebSocket.OPEN && !!engineDevice;
}

async function inferAll(images: string[], gpu: boolean): Promise<InferResp> {
  connectEngine();
  if (await engineReady()) {
    try {
      const t0 = performance.now();
      const scores = await scoreOnEngine(images);
      log(`${images.length} tile(s) on engine: P(AI)=[${scores.map((p) => p.toFixed(2)).join(', ')}] in ${(performance.now() - t0).toFixed(0)}ms`);
      return { ok: true, scores, backend: 'engine', device: engineDevice ?? undefined };
    } catch (e) {
      warn('engine failed, falling back to in-browser model', e);
    }
  }
  try {
    const { session, meta, backend } = await getModel(gpu);
    const inputName = session.inputNames.includes(meta.inputName) ? meta.inputName : session.inputNames[0];
    // One tile per run, always shape [1, 3, S, S]. A varying batch size makes the WebGPU backend
    // allocate buffers/pipelines for every new shape, which grows over a long session; batching
    // gave no speedup anyway (measured: 1 tile 153 ms, 3 tiles 432 ms on Iris Xe).
    const t0 = performance.now();
    const scores: number[] = [];
    for (const img of images) {
      const tensor = await toTensor(img, meta);
      const out = await session.run({ [inputName]: tensor });
      tensor.dispose();
      const t = out[session.outputNames[0]];
      scores.push(pAI(t.data as Float32Array, meta.aiIndex, meta.logitBias));
      t.dispose();
    }
    log(`${images.length} tile(s): P(AI)=[${scores.map((p) => p.toFixed(2)).join(', ')}] in ${(performance.now() - t0).toFixed(0)}ms (${backend})`);
    return { ok: true, scores, backend };
  } catch (e) {
    warn('inference error', e);
    return { ok: false, error: String((e as Error)?.message ?? e) };
  }
}

// Free the model's memory when nobody is scrolling past videos. The service worker
// re-creates this document on the next request.
let idleTimer: ReturnType<typeof setTimeout> | null = null;
function touch() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    log('idle, closing offscreen document');
    window.close();
  }, OFFSCREEN_IDLE_MS);
}
touch();
connectEngine();

chrome.runtime.onMessage.addListener((msg: OffscreenMsg, _sender, sendResponse) => {
  if (msg?.target !== 'offscreen') return false;
  touch();
  if (msg.kind === 'INFER') {
    inferAll(msg.images, msg.gpu).then(sendResponse);
    return true;
  }
  if (msg.kind === 'LABEL') {
    connectEngine();
    askEngine({ type: 'label', label: msg.label, key: msg.key, page: msg.page, images: msg.images })
      .then((r) => sendResponse({ ok: true, saved: r.saved, folder: r.folder } satisfies LabelResp))
      .catch((e) => sendResponse({ ok: false, error: String((e as Error).message ?? e) } satisfies LabelResp));
    return true;
  }
  if (msg.kind === 'STATUS') {
    getModel(loadingGpu)
      .then((m) => sendResponse({ ok: true, backend: m.backend }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }
  return false;
});
