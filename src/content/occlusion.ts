// Is a media element actually visible, or covered by something else (a post viewer, a popup)?
// Our overlays sit above the whole page, so without this check a modal's background grid gets
// badges, blocks and tags painted on top of the modal.

const OURS = new Set(['SLOPSHIELD-OVERLAY', 'SLOPSHIELD-DEBUG']);
// An element on top that shares an ancestor with the media within this many levels belongs to the
// same post (e.g. Instagram's transparent click layer over a video) — that doesn't count as covered.
const SAME_POST_LEVELS = 10;

function parentOf(el: Element): Element | null {
  if (el.parentElement) return el.parentElement;
  const root = el.getRootNode();
  return root instanceof ShadowRoot ? root.host : null;
}

function topElementAt(x: number, y: number): Element | null {
  for (const el of document.elementsFromPoint(x, y)) if (!OURS.has(el.tagName)) return el;
  return null;
}

function belongsTo(media: Element, top: Element): boolean {
  if (top === media || media.contains(top) || top.contains(media)) return true;
  const ancestors = new Map<Element, number>();
  let n: Element | null = media;
  for (let depth = 0; n && depth <= SAME_POST_LEVELS; depth++, n = parentOf(n)) ancestors.set(n, depth);
  for (n = top; n; n = parentOf(n)) {
    if (ancestors.has(n)) return n !== document.body && n !== document.documentElement;
  }
  return false;
}

/** True if most of the element's visible area is covered by unrelated content. */
export function isOccluded(media: Element): boolean {
  const r = media.getBoundingClientRect();
  const x0 = Math.max(r.left, 0), x1 = Math.min(r.right, innerWidth);
  const y0 = Math.max(r.top, 0), y1 = Math.min(r.bottom, innerHeight);
  if (x1 <= x0 || y1 <= y0) return false; // off-screen is handled elsewhere
  const pts: Array<[number, number]> = [
    [(x0 + x1) / 2, (y0 + y1) / 2],
    [x0 + (x1 - x0) * 0.25, y0 + (y1 - y0) * 0.25],
    [x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.75],
  ];
  let covered = 0;
  for (const [x, y] of pts) {
    const top = topElementAt(x, y);
    if (top && !belongsTo(media, top)) covered++;
  }
  return covered >= 2;
}
