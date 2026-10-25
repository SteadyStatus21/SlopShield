// Message router, capture budget, verdict cache, offscreen lifecycle.
import { MODEL, VISION } from '../config';
import { log, warn } from '../log';
import { getSettings } from '../settings';
import type { BackendInfo, CaptureResp, ContentMsg, InferResp, LabelResp, ModelMeta, OffscreenMsg, UiMsg } from '../types';
import { captureWindow, cropAll, type CropReq } from './capture';

// ---------- model meta (input size for crops) ----------

let metaPromise: Promise<ModelMeta> | null = null;
function getMeta(): Promise<ModelMeta> {
  metaPromise ??= fetch(chrome.runtime.getURL(MODEL.metaPath))
    .then((r) => (r.ok ? r.json() : MODEL.fallbackMeta))
    .catch(() => MODEL.fallbackMeta);
  return metaPromise;
}

// Verdicts are cached per model: a verdict from an older/other model must never be reused after
// the model changes (e.g. a stale 1.00 from a retired model showed up as "cached").
async function cacheKey(key: string): Promise<string> {
  const meta = await getMeta();
  return `v:${meta.name ?? 'model'}:${key}`;
}

// ---------- offscreen document ----------

const OFFSCREEN_URL = 'offscreen/offscreen.html';
let creating: Promise<void> | null = null;

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  if (contexts.length > 0) return;
  creating ??= chrome.offscreen
    .createDocument({
      url: OFFSCREEN_URL,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: 'Run the on-device ONNX model (WASM/WebGPU) on video frames.',
    })
    .finally(() => (creating = null));
  await creating;
}

let backend: BackendInfo = { backend: 'none', modelLoaded: false };

async function infer(images: string[], gpu: boolean): Promise<number[]> {
  await ensureOffscreen();
  const msg: OffscreenMsg = { target: 'offscreen', kind: 'INFER', images, gpu };
  const resp: InferResp = await chrome.runtime.sendMessage(msg);
  if (!resp?.ok) {
    backend = { ...backend, modelLoaded: false, error: resp?.error ?? 'no response' };
    throw new Error(backend.error);
  }
  backend = { backend: resp.backend, device: resp.device, modelLoaded: true };
  return resp.scores;
}

// ---------- capture engine ----------
// Requests from all visible media wait in a small queue. At each slot, one screenshot is taken
// for a window and every waiting request for that window is cropped from it and scored in one
// batch (one GPU dispatch). After each batch the engine rests in proportion to how long inference
// took, so the model uses at most VISION.maxDutyCycle of the GPU/CPU. Slots are also spaced to
// stay inside captureVisibleTab's ~2 screenshots/sec quota, which caps the engine at ~2 fps.
// When the queue is full, callers get 'busy' and retry later, so work never piles up.

interface Pending {
  windowId: number;
  req: CropReq;
  key?: string; // media key, so its crops can be kept for labeling
  snapshot?: boolean; // labeling snapshot: keep the crops, skip inference
  resolve: (r: CaptureResp) => void;
}

// Labeling: the most recent clean crops per media key (captured before any overlay was drawn), so a
// "Not AI" / "Yes, AI" click can send exactly what the model saw to the engine's dataset folders.
const recentCrops = new Map<string, string[]>();
const KEEP_KEYS = 40;
const KEEP_TILES_PER_KEY = 18;

function rememberCrops(key: string, urls: string[]) {
  const tiles = [...(recentCrops.get(key) ?? []), ...urls].slice(-KEEP_TILES_PER_KEY);
  recentCrops.delete(key); // re-insert so this key becomes the newest
  recentCrops.set(key, tiles);
  while (recentCrops.size > KEEP_KEYS) recentCrops.delete(recentCrops.keys().next().value!);
}

const queue: Pending[] = [];
let timer: ReturnType<typeof setTimeout> | null = null;
let inFlight = false;
let nextAllowed = 0;
let lastDebugWrite = 0;

// Jank back-off: content scripts report long main-thread tasks on their page. Each report doubles
// the rest between batches (up to VISION.maxBackoff×); quiet periods halve it again.
let backoff = 1;
let lastJank = 0;

function onJank() {
  backoff = Math.min(VISION.maxBackoff, backoff * 2);
  lastJank = Date.now();
  log('page jank reported, backoff', backoff);
}

function currentBackoff(): number {
  if (backoff > 1 && Date.now() - lastJank > VISION.backoffRecoverMs) {
    backoff = Math.max(1, backoff / 2);
    lastJank = Date.now();
  }
  return backoff;
}

function enqueue(windowId: number, req: CropReq, key?: string, snapshot = false): Promise<CaptureResp> {
  if (queue.length >= VISION.maxQueue) return Promise.resolve({ ok: false, reason: 'busy' });
  return new Promise((resolve) => {
    queue.push({ windowId, req, key, snapshot, resolve });
    schedule();
  });
}

function schedule() {
  if (timer || inFlight || queue.length === 0) return;
  // Short gather window so media that become visible together share one screenshot.
  const wait = Math.max(VISION.batchWindowMs, nextAllowed - Date.now());
  timer = setTimeout(flush, wait);
}

async function flush() {
  timer = null;
  inFlight = true;
  const windowId = queue[0].windowId;
  const batch = queue.filter((p) => p.windowId === windowId);
  queue.splice(0, queue.length, ...queue.filter((p) => p.windowId !== windowId));
  let inferMs = 0;

  try {
    const [meta, bitmap, settings] = await Promise.all([getMeta(), captureWindow(windowId), getSettings()]);
    const crops = await cropAll(
      bitmap,
      batch.map((p) => p.req),
      meta,
    );
    bitmap.close();

    // Every tile of every item goes into one inference batch; `owner` maps tiles back to items.
    const tilesFlat: string[] = [];
    const owner: number[] = [];
    crops.forEach((c, i) => {
      if (!c.ok) return batch[i].resolve({ ok: false, reason: c.reason });
      if (batch[i].key) rememberCrops(batch[i].key!, c.dataUrls);
      if (batch[i].snapshot) return batch[i].resolve({ ok: true, score: NaN }); // stored for labeling only
      for (const url of c.dataUrls) {
        tilesFlat.push(url);
        owner.push(i);
      }
    });
    if (tilesFlat.length > 0) {
      if (settings.debug && Date.now() - lastDebugWrite > 3000) {
        // Throttled: each write serializes ~100s of KB of PNG data URLs.
        lastDebugWrite = Date.now();
        const last = owner[owner.length - 1];
        chrome.storage.session.set({ lastCrops: tilesFlat.filter((_, k) => owner[k] === last) });
      }
      const t0 = Date.now();
      try {
        const scores = await infer(tilesFlat, settings.useGpu);
        // Per item: average the tiles' logits, then back to a probability.
        const sums = new Map<number, { sum: number; n: number }>();
        scores.forEach((p, k) => {
          const q = Math.min(1 - 1e-6, Math.max(1e-6, p));
          const s = sums.get(owner[k]) ?? { sum: 0, n: 0 };
          s.sum += Math.log(q / (1 - q));
          s.n++;
          sums.set(owner[k], s);
        });
        for (const [i, { sum, n }] of sums) batch[i].resolve({ ok: true, score: 1 / (1 + Math.exp(-sum / n)) });
      } catch (e) {
        warn('inference failed', e);
        for (const i of new Set(owner)) batch[i].resolve({ ok: false, reason: 'model', detail: String(e) });
      } finally {
        inferMs = Date.now() - t0;
      }
    }
  } catch (e) {
    const detail = String((e as Error)?.message ?? e);
    const reason = /MAX_CAPTURE|quota|per second/i.test(detail) ? 'busy' : 'error';
    if (reason !== 'busy') warn('capture failed', detail);
    batch.forEach((p) => p.resolve({ ok: false, reason, detail }));
  } finally {
    inFlight = false;
    const rest = inferMs * (1 / VISION.maxDutyCycle - 1);
    nextAllowed = Date.now() + Math.max(VISION.minCaptureIntervalMs, rest) * currentBackoff();
    schedule();
  }
}

// ---------- router ----------

chrome.runtime.onMessage.addListener((msg: ContentMsg | UiMsg | OffscreenMsg, sender, sendResponse) => {
  if ('target' in msg) return false; // meant for the offscreen doc

  const handle = async (): Promise<unknown> => {
    switch (msg.kind) {
      case 'CAPTURE_REQ': {
        const tab = sender.tab;
        if (!tab?.windowId || !tab.active) return { ok: false, reason: 'offscreen' } satisfies CaptureResp;
        return enqueue(tab.windowId, { rect: msg.rect, viewportWidth: msg.viewportWidth, viewportHeight: msg.viewportHeight }, msg.key, msg.snapshot);
      }
      case 'CACHE_GET': {
        const k = await cacheKey(msg.key);
        return (await chrome.storage.session.get(k))[k] ?? null;
      }
      case 'JANK':
        onJank();
        return true;
      case 'LABEL': {
        const images = recentCrops.get(msg.key);
        if (!images?.length) return { ok: false, noFrames: true, error: 'no recent frames for this item' } satisfies LabelResp;
        await ensureOffscreen();
        const fwd: OffscreenMsg = { target: 'offscreen', kind: 'LABEL', label: msg.label, key: msg.key, page: sender.tab?.url ?? '', images };
        return chrome.runtime.sendMessage(fwd);
      }
      case 'CACHE_SET':
        await chrome.storage.session.set({ [await cacheKey(msg.key)]: msg.verdict });
        return true;
      case 'GET_STATUS': {
        const { lastCrops } = await chrome.storage.session.get('lastCrops');
        return { backend, lastCrops: lastCrops ?? [] };
      }
    }
  };

  handle().then(sendResponse, (e) => {
    warn('handler error', msg.kind, e);
    sendResponse(null);
  });
  return true;
});

// After an install/update/reload, tabs that were already open still run the old (now orphaned)
// content script, which shuts itself down. Inject the new one so they keep working without F5.
chrome.runtime.onInstalled.addListener(async () => {
  log('installed');
  const tabs = await chrome.tabs.query({ url: ['http://*/*', 'https://*/*'] });
  for (const tab of tabs) {
    if (tab.id === undefined || tab.discarded) continue;
    chrome.scripting
      .executeScript({ target: { tabId: tab.id }, files: ['content/index.js'] })
      .catch(() => {}); // e.g. the web store or other pages extensions may not script
  }
});
