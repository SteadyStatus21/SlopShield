// captureVisibleTab + crop (OffscreenCanvas). Runs in the service worker.
import { VISION } from '../config';
import type { ModelMeta, Rect } from '../types';

export interface CropReq {
  rect: Rect; // CSS px, relative to viewport, read right before capture
  viewportWidth: number;
  viewportHeight: number;
}

// One media element → one or more model-sized tiles (see tiles()).
export type CropResult = { ok: true; dataUrls: string[] } | { ok: false; reason: 'blank' | 'offscreen' };

export async function captureWindow(windowId: number): Promise<ImageBitmap> {
  const dataUrl = await chrome.tabs.captureVisibleTab(windowId, { format: 'jpeg', quality: 92 });
  const blob = await (await fetch(dataUrl)).blob();
  return createImageBitmap(blob);
}

/** Crop each rect out of one screenshot into model-sized (size×size) tiles. */
export async function cropAll(bitmap: ImageBitmap, reqs: CropReq[], meta: ModelMeta): Promise<CropResult[]> {
  return Promise.all(reqs.map((r) => cropOne(bitmap, r, meta)));
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Square tiles that together cover the media. Square-input models (center-crop or squash) only
 * see the middle of a tall or wide post, so tile squares along the long side (start, middle,
 * end) instead, up to VISION.maxTiles. With a center-crop model each tile keeps its
 * size/resizeShortest zoom, as in training.
 */
function tiles(b: Box, meta: ModelMeta): Box[] {
  const short = Math.min(b.w, b.h);
  const side = Math.round(meta.resizeShortest ? short * (meta.size / meta.resizeShortest) : short);
  const long = Math.max(b.w, b.h);
  const n = Math.min(VISION.maxTiles, Math.max(1, Math.ceil(long / side - 0.15))); // ≥15% uncovered → add a tile
  const out: Box[] = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0.5 : i / (n - 1); // 0 = start, 1 = end of the long side
    const along = Math.round((long - side) * t);
    const across = Math.round((short - side) / 2);
    out.push(
      b.w >= b.h
        ? { x: b.x + along, y: b.y + across, w: side, h: side }
        : { x: b.x + across, y: b.y + along, w: side, h: side },
    );
  }
  return out;
}

/**
 * Shrink a box to its content by dropping flat rows/columns at the edges (near-constant colour).
 * Keep in sync with trim_borders() in scripts/train_from_labels.py.
 */
export function trimBorders(bitmap: ImageBitmap, b: Box): Box {
  const N = 64; // analyse a small copy
  const cw = Math.max(8, Math.round((N * b.w) / Math.max(b.w, b.h)));
  const ch = Math.max(8, Math.round((N * b.h) / Math.max(b.w, b.h)));
  const ctx = new OffscreenCanvas(cw, ch).getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, b.x, b.y, b.w, b.h, 0, 0, cw, ch);
  const px = ctx.getImageData(0, 0, cw, ch).data;
  const luma = (x: number, y: number) => {
    const i = (y * cw + x) * 4;
    return 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  };
  const flat = (vals: number[]) => {
    const m = vals.reduce((a, v) => a + v, 0) / vals.length;
    return Math.sqrt(vals.reduce((a, v) => a + (v - m) * (v - m), 0) / vals.length) < TRIM_FLAT_STD;
  };
  const col = (x: number) => flat(Array.from({ length: ch }, (_, y) => luma(x, y)));
  const row = (y: number) => flat(Array.from({ length: cw }, (_, x) => luma(x, y)));
  let l = 0, r = cw - 1, t = 0, btm = ch - 1;
  while (l < r && col(l)) l++;
  while (r > l && col(r)) r--;
  while (t < btm && row(t)) t++;
  while (btm > t && row(btm)) btm--;
  const kw = (r - l + 1) / cw, kh = (btm - t + 1) / ch;
  if (kw * kh < 0.25 || (kw > 0.97 && kh > 0.97)) return b; // nothing to trim, or (almost) all flat
  return {
    x: b.x + Math.round((l / cw) * b.w),
    y: b.y + Math.round((t / ch) * b.h),
    w: Math.round(kw * b.w),
    h: Math.round(kh * b.h),
  };
}

const TRIM_FLAT_STD = 6;

async function cropOne(bitmap: ImageBitmap, req: CropReq, meta: ModelMeta): Promise<CropResult> {
  const { size } = meta;
  // Derive the scale from the screenshot itself rather than devicePixelRatio alone:
  // it also absorbs page zoom.
  const scale = bitmap.width / req.viewportWidth;
  const x0 = Math.max(0, req.rect.x);
  const y0 = Math.max(0, req.rect.y);
  const x1 = Math.min(req.viewportWidth, req.rect.x + req.rect.width);
  const y1 = Math.min(req.viewportHeight, req.rect.y + req.rect.height);
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < VISION.minVideoPx || h < VISION.minVideoPx) return { ok: false, reason: 'offscreen' };

  const sx = Math.round(x0 * scale);
  const sy = Math.round(y0 * scale);
  const sw = Math.min(bitmap.width - sx, Math.round(w * scale));
  const sh = Math.min(bitmap.height - sy, Math.round(h * scale));
  if (sw <= 0 || sh <= 0) return { ok: false, reason: 'offscreen' };

  // Letterbox / pillarbox / carousel padding: flat bars around the actual picture. The model never saw
  // half-black training images, so they were a source of false positives. Score only the content.
  const content = trimBorders(bitmap, { x: sx, y: sy, w: sw, h: sh });
  const dataUrls: string[] = [];
  for (const t of tiles(content, meta)) {
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bitmap, t.x, t.y, t.w, t.h, 0, 0, size, size);
    // Skip black/loading tiles (DRM video, letterbox bars) rather than the whole item.
    if (lumaVariance(ctx.getImageData(0, 0, size, size).data) < VISION.minVariance) continue;
    dataUrls.push(await blobToDataUrl(await canvas.convertToBlob({ type: 'image/png' })));
  }
  return dataUrls.length ? { ok: true, dataUrls } : { ok: false, reason: 'blank' };
}

function lumaVariance(px: Uint8ClampedArray): number {
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  // Subsample every 4th pixel; plenty for a blank check.
  for (let i = 0; i < px.length; i += 16) {
    const y = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
    sum += y;
    sumSq += y * y;
    n++;
  }
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${blob.type};base64,${btoa(bin)}`;
}
