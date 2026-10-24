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

  const dataUrls: string[] = [];
  for (const t of tiles({ x: sx, y: sy, w: sw, h: sh }, meta)) {
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
