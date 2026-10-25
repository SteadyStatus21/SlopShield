// Find videos and images (including inside shadow DOM), observe them, run the pipeline, apply overlays.
import { MEDIA_SELECTOR, VISION } from '../config';
import { log } from '../log';
import { getSettings, onSettingsChanged } from '../settings';
import type { CaptureResp, ContentMsg, Label, LabelResp, Settings, Verdict } from '../types';
import { showBadge, showBlock, showDebugTag, type DebugTag, type Overlay, type TagTone } from './overlay';
import { combine, findCgiContext, findPlatformLabel, gatherText, parentOf, scoreText, trimmedMean } from './signals';

type Media = HTMLVideoElement | HTMLImageElement;

// Vision progress is stored per media *key*, not per element: feeds like Reddit destroy posts
// that scroll out of view and create new elements when you scroll back.
interface Progress {
  scores: number[]; // vision frame scores collected so far
  attempts: number;
}

interface VState {
  key: string;
  status: 'idle' | 'scanning' | 'done';
  visible: boolean;
  prog: Progress;
  verdict?: Verdict;
  overlay?: Overlay;
  tag?: DebugTag; // debug mode: live status next to the element
}

const media = new Map<Media, VState>();
const progress = new Map<string, Progress>();
const verdicts = new Map<string, Verdict>(); // in-page copy of the session cache
const revealed = new Set<string>(); // keys the user clicked "Show anyway" on

function progressFor(key: string): Progress {
  let p = progress.get(key);
  if (!p) progress.set(key, (p = { scores: [], attempts: 0 }));
  return p;
}
let settings: Settings;

// When the extension is reloaded or updated, content scripts already running in open tabs are
// orphaned: every chrome.runtime call throws "Extension context invalidated". Detect that and
// shut this copy down quietly (the new version's content script takes over on the next load).
let orphaned = false;
let repositionTimer: ReturnType<typeof setInterval> | undefined;

function shutdown() {
  if (orphaned) return;
  orphaned = true;
  clearInterval(jankTimer);
  mo.disconnect();
  io.disconnect();
  clearInterval(repositionTimer);
  for (const st of media.values()) {
    st.overlay?.destroy();
    st.tag?.destroy();
  }
  media.clear();
}

/** Debug mode: show what the pipeline is doing with this element. */
function note(m: Media, st: VState, text: string, tone: TagTone = 'info') {
  if (!settings.debug) return;
  st.tag ??= showDebugTag(m, (label) => labelItem(m, st, label));
  st.tag.set(text, tone);
}

/**
 * User feedback: store this item's recent crops as an AI / real example (desktop engine → dataset/).
 * "Not AI" on a flagged item also lifts the flag for the rest of this page view.
 */
async function labelItem(m: Media, st: VState, label: Label): Promise<string> {
  let resp = await send<LabelResp>({ kind: 'LABEL', key: st.key, label });
  if (resp && !resp.ok && resp.noFrames) {
    // Cached verdict (or scanned long ago): no stored crops. Take a clean snapshot now and retry.
    await snapshot(m, st);
    resp = await send<LabelResp>({ kind: 'LABEL', key: st.key, label });
  }
  if (!resp) throw new Error('extension unavailable');
  if (!resp.ok) throw new Error(resp.error);
  if (label === 'real') {
    revealed.add(st.key);
    render(m, st);
  }
  return `saved as ${label === 'ai' ? 'AI' : 'real'} (${resp.saved} crops)`;
}

/**
 * Capture this item's crops for labeling without scoring them. Our overlays, blur and debug tags are
 * removed for the moment of capture (the model must see the page, not our UI), then restored.
 */
async function snapshot(m: Media, st: VState) {
  const hidden: Array<[Media, VState]> = [];
  for (const [o, ost] of media) {
    ost.tag?.hide(true); // never let our UI into a crop
    if (ost.overlay) {
      ost.overlay.destroy();
      ost.overlay = undefined;
      hidden.push([o, ost]);
    }
  }
  try {
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 80))); // let the page repaint
    for (let attempt = 0; attempt < 8; attempt++) {
      const r = m.getBoundingClientRect();
      const resp = await send<CaptureResp>({
        kind: 'CAPTURE_REQ',
        rect: { x: r.left, y: r.top, width: r.width, height: r.height },
        viewportWidth: innerWidth,
        viewportHeight: innerHeight,
        key: st.key,
        snapshot: true,
      });
      if (resp?.ok || (resp && resp.reason !== 'busy')) break;
      await sleep(VISION.busyRetryMs);
    }
  } finally {
    for (const ost of media.values()) ost.tag?.hide(false);
    for (const [o, ost] of hidden) render(o, ost);
  }
}

/** A video in the same spot as this image (Instagram/Reddit put a thumbnail <img> under the <video>). */
function videoOver(m: Media): HTMLVideoElement | undefined {
  return isVideo(m) ? undefined : stackedWith(m).find(isVideo);
}

/**
 * The video is what the user actually sees and gets several frames, so it decides for its spot.
 * Thumbnails under it are never scanned, and if one was judged before the video appeared, that
 * verdict (and any overlay) is dropped.
 */
function yieldThumbnailsTo(video: HTMLVideoElement) {
  for (const o of stackedWith(video)) {
    const ost = media.get(o);
    if (!ost || isVideo(o)) continue;
    ost.overlay?.destroy();
    Object.assign(ost, { status: 'done', verdict: undefined, overlay: undefined });
    note(o, ost, 'skipped: thumbnail of a video', 'skip');
  }
}

/** Message the service worker. Never throws: resolves null if the extension is unreachable. */
async function send<T>(msg: ContentMsg): Promise<T | null> {
  if (orphaned) return null;
  try {
    if (!chrome.runtime?.id) throw new Error('Extension context invalidated');
    return (await chrome.runtime.sendMessage(msg)) ?? null;
  } catch (e) {
    if (/context invalidated/i.test(String(e))) shutdown();
    else log('message failed', msg.kind, e);
    return null;
  }
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const isVideo = (m: Media): m is HTMLVideoElement => m instanceof HTMLVideoElement;

function active(): boolean {
  return !orphaned && settings.mode !== 'off';
}

/** URL without its query string when the path alone identifies the file (CDN resize/signature params change). */
function normUrl(u: string): string {
  try {
    const x = new URL(u, location.href);
    return x.pathname.length > 12 ? x.origin + x.pathname : x.href;
  } catch {
    return u;
  }
}

// Ancestor attributes that identify a post across re-renders.
const POST_ID_ATTRS = ['permalink', 'post-id', 'thingid', 'data-post-id', 'data-video-id', 'data-id', 'content-href'];

function postIdOf(m: Media): string | null {
  let node = parentOf(m);
  for (let i = 0; node && i < 20; i++, node = parentOf(node)) {
    for (const a of POST_ID_ATTRS) {
      const v = node.getAttribute(a);
      if (v) return `${a}=${v}`;
    }
  }
  return null;
}

/** Stable identity for a media item, so a re-created element hits the cache instead of being rescanned. */
function keyFor(m: Media): string {
  const src = m.currentSrc || m.src;
  if (src && !src.startsWith('blob:') && !src.startsWith('data:')) return normUrl(src);
  // MediaSource players (Reddit, YouTube) expose blob: URLs that change per load.
  if (isVideo(m) && m.poster) return `poster:${normUrl(m.poster)}`;
  const post = postIdOf(m);
  if (post) return `${location.host}|${post}`;
  const page = location.href.split('#')[0];
  return `${page}#${[...document.querySelectorAll('video')].indexOf(m as HTMLVideoElement)}`;
}

function isReady(m: Media): boolean {
  return isVideo(m) ? m.readyState >= 2 : m.complete && m.naturalWidth > 0;
}

function isVisibleEnough(m: Media): boolean {
  const r = m.getBoundingClientRect();
  if (r.width < VISION.minVideoPx || r.height < VISION.minVideoPx) return false;
  const visW = Math.min(r.right, innerWidth) - Math.max(r.left, 0);
  const visH = Math.min(r.bottom, innerHeight) - Math.max(r.top, 0);
  if (visW <= 0 || visH <= 0) return false;
  return (visW * visH) / (r.width * r.height) >= VISION.minVisibleRatio && document.visibilityState === 'visible';
}

// ---------- discovery ----------

const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      const m = e.target as Media;
      const st = media.get(m);
      if (!st) continue;
      st.visible = e.intersectionRatio >= VISION.minVisibleRatio;
      if (st.visible) maybeScan(m);
    }
  },
  { threshold: [0, VISION.minVisibleRatio, 1] },
);

function register(m: Media) {
  if (media.has(m)) return;
  const key = keyFor(m);
  media.set(m, { key, status: 'idle', visible: false, prog: progressFor(key) });
  io.observe(m);
  // Players like YouTube reuse one <video> across navigations, and lazy images swap src on load;
  // rescan when the source changes.
  m.addEventListener(isVideo(m) ? 'loadeddata' : 'load', () => maybeScan(m));
  // A video starting to play inside an already-blocked spot must be blocked too.
  if (isVideo(m)) {
    m.addEventListener('play', () => extendBlocksTo(m));
    yieldThumbnailsTo(m);
  }
  extendBlocksTo(m);
}

// Sites like Reddit render media inside custom elements' shadow roots, which
// document.querySelectorAll can't see. Walk into them (closed ones too, via chrome.dom).
const seenRoots = new WeakSet<ShadowRoot>();
const noClosedRoot = new WeakSet<Element>(); // custom elements already checked, so chrome.dom runs once each

function shadowOf(el: Element): ShadowRoot | null {
  if (el.shadowRoot) return el.shadowRoot;
  if (orphaned || !el.tagName.includes('-') || noClosedRoot.has(el)) return null; // closed roots ≈ custom elements only
  let sr: ShadowRoot | null = null;
  try {
    sr = (chrome.dom?.openOrClosedShadowRoot?.(el as HTMLElement) as ShadowRoot | undefined) ?? null;
  } catch (e) {
    // chrome.* APIs throw once the extension has been reloaded under this page.
    if (/context invalidated/i.test(String(e))) shutdown();
    return null;
  }
  if (!sr) noClosedRoot.add(el);
  return sr;
}

function collect(root: Document | ShadowRoot) {
  root.querySelectorAll<Media>(MEDIA_SELECTOR).forEach(register);
  root.querySelectorAll('*').forEach((el) => {
    const sr = shadowOf(el);
    if (!sr) return;
    if (!seenRoots.has(sr)) {
      seenRoots.add(sr);
      mo.observe(sr, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
    }
    collect(sr);
  });
}

function discover() {
  if (orphaned) return; // a discovery sweep can still be scheduled when shutdown happens
  collect(document);
  for (const [m, st] of media) {
    if (!m.isConnected) {
      st.overlay?.destroy();
      st.tag?.destroy();
      io.unobserve(m);
      media.delete(m);
    }
  }
}

// Feeds mutate constantly. Sweep at most once per DISCOVER_MIN_GAP_MS, when the page is idle.
const DISCOVER_MIN_GAP_MS = 1000;
let discoverPending = false;
let lastDiscover = 0;
const mo = new MutationObserver(() => {
  if (discoverPending) return;
  discoverPending = true;
  const wait = Math.max(0, lastDiscover + DISCOVER_MIN_GAP_MS - Date.now());
  setTimeout(() => {
    requestIdleCallback(
      () => {
        discoverPending = false;
        lastDiscover = Date.now();
        discover();
      },
      { timeout: 1000 },
    );
  }, wait);
});

// ---------- pipeline ----------

async function maybeScan(m: Media) {
  const st = media.get(m);
  if (!st || !active()) return;

  const key = keyFor(m);
  if (key !== st.key) {
    st.overlay?.destroy();
    Object.assign(st, { key, status: 'idle', prog: progressFor(key), verdict: undefined, overlay: undefined });
  }
  if (st.status !== 'idle') return;
  // Skip icons/avatars before doing any work (also re-checked right before capture).
  const r = m.getBoundingClientRect();
  if (r.width < VISION.minVideoPx || r.height < VISION.minVideoPx) return;
  if (videoOver(m)) {
    st.status = 'done';
    note(m, st, 'skipped: thumbnail of a video', 'skip');
    return;
  }
  st.status = 'scanning';
  note(m, st, 'checking…');

  const cached = verdicts.get(key) ?? (await send<Verdict | null>({ kind: 'CACHE_GET', key }));
  if (cached) {
    verdicts.set(key, cached);
    finish(m, st, cached, true);
    return;
  }

  // 1. Free signals: platform labels in the DOM.
  const text = gatherText(m);
  const label = findPlatformLabel(text);
  // 2. Text signals.
  const textScore = scoreText(text);
  // Gameplay/CGI: the image model calls every rendered frame "AI", and a down-weighted vision
  // score can't reach the threshold anyway, so don't spend CPU on it.
  const cgi = label ? null : findCgiContext(text);

  let vision: number | null = null;
  if (!label && !cgi && !settings.fastMode) {
    // 3. Vision.
    const res = await sampleFrames(m, st);
    if (res === 'interrupted') {
      st.status = 'idle'; // resume when it becomes visible again
      return;
    }
    vision = res;
  } else if (cgi) {
    note(m, st, `skipped vision: gameplay/CGI ("${cgi}")`, 'skip');
  }

  // 4. Combine.
  const verdict = combine({ label, text: textScore, vision, frames: st.prog.scores.length, threshold: settings.threshold, cgi });
  log('verdict', m.tagName, key, verdict);
  verdicts.set(key, verdict);
  progress.delete(key);
  send({ kind: 'CACHE_SET', key, verdict });
  finish(m, st, verdict);
}

async function sampleFrames(m: Media, st: VState): Promise<number | null | 'interrupted'> {
  const want = isVideo(m) ? VISION.framesPerVideo : VISION.framesPerImage;
  while (st.prog.scores.length < want && st.prog.attempts < VISION.maxAttemptsPerVideo) {
    const why = !m.isConnected || !active() ? 'removed/off' : !isVisibleEnough(m) ? 'not visible' : videoOver(m) ? 'video on top' : null;
    if (why) {
      // (A video appearing on top of an image takes over the spot; see maybeScan.)
      log('scan paused:', why, m.tagName);
      note(m, st, `paused: ${why} (${st.prog.scores.length} frames so far)`, 'skip');
      return 'interrupted';
    }
    if (!isReady(m)) {
      note(m, st, 'waiting for media to load…');
      await sleep(VISION.frameIntervalMs);
      continue;
    }
    const s = st.prog.scores;
    note(m, st, `frame ${s.length + 1}/${want}${s.length ? ` · avg ${trimmedMean(s).toFixed(2)}` : ''}`);

    const r = m.getBoundingClientRect(); // re-read right before capture; the user may have scrolled
    const resp = await send<CaptureResp>({
      kind: 'CAPTURE_REQ',
      rect: { x: r.left, y: r.top, width: r.width, height: r.height },
      viewportWidth: innerWidth,
      viewportHeight: innerHeight,
      key: st.key, // lets the background keep these crops for a later "Not AI" / "Yes, AI" label
    });
    if (resp && !resp.ok && resp.reason === 'busy') {
      // The model is working on something else or resting; wait our turn (doesn't count as an attempt).
      note(m, st, `queued (frame ${s.length + 1}/${want})`);
      await sleep(VISION.busyRetryMs);
      continue;
    }
    if (resp && !resp.ok) note(m, st, `frame failed: ${resp.reason}`, 'skip');
    st.prog.attempts++;
    if (resp?.ok) st.prog.scores.push(resp.score);
    else if (resp?.reason === 'model') break; // no model available; fall back to text signals
    if (st.prog.scores.length >= want) break;

    // Paused videos show the same frame every time; one or two samples are enough.
    if (isVideo(m) && m.paused && st.prog.scores.length >= 2) break;
    if (confidentEarly(st.prog.scores)) break;
    await sleep(VISION.frameIntervalMs);
  }
  return st.prog.scores.length ? trimmedMean(st.prog.scores) : null;
}

// Stop early only when frames agree the media is real. An AI verdict changes what the user
// sees, so it always gets the full frame set spread across the video.
function confidentEarly(scores: number[]): boolean {
  if (scores.length < VISION.earlyStopFrames) return false;
  return scores.every((x) => x < VISION.earlyStopMargin);
}

function finish(m: Media, st: VState, verdict: Verdict, cached = false) {
  st.status = 'done';
  st.verdict = verdict;
  const ai = verdict.score >= settings.threshold;
  const why = verdict.vision !== null ? `vision ${verdict.vision.toFixed(2)} × ${verdict.frames}` : verdict.reasons[0];
  note(m, st, `${ai ? 'AI' : 'real'} ${verdict.score.toFixed(2)} (≥${settings.threshold} = AI) · ${why}${cached ? ' · cached' : ''}`, ai ? 'ai' : 'real');
  render(m, st);
  if (isVideo(m)) yieldThumbnailsTo(m);
}

// ---------- rendering ----------

/** Intersection area as a fraction of the smaller rect. */
function overlapFraction(a: DOMRect, b: DOMRect): number {
  const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
  const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / Math.max(1, Math.min(a.width * a.height, b.width * b.height));
}

function render(m: Media, st: VState) {
  st.overlay?.destroy();
  st.overlay = undefined;
  const verdict = st.verdict;
  if (!verdict || !active() || revealed.has(st.key)) return;
  if (verdict.score < settings.threshold) return; // never touch real/unscored media

  // Posts often stack media in one spot (poster image under a video, blurred backdrop copy).
  // Show one overlay per spot: the largest element wins.
  const area = (r: DOMRect) => r.width * r.height;
  const mine = m.getBoundingClientRect();
  for (const [other, ost] of media) {
    if (other === m || !ost.overlay) continue;
    const theirs = other.getBoundingClientRect();
    if (overlapFraction(mine, theirs) < 0.3) continue;
    if (area(mine) <= area(theirs)) return;
    ost.overlay.destroy();
    ost.overlay = undefined;
  }

  const opts = {
    media: m,
    verdict,
    onReveal: () => {
      revealed.add(st.key);
      st.overlay = undefined;
    },
    onLabel: (label: Label) => labelItem(m, st, label),
    stacked: stackedWith(m),
  };
  st.overlay = settings.mode === 'block' ? showBlock(opts) : showBadge(opts);
}

/** Other media occupying the same spot as m (poster image under a video, etc.). */
function stackedWith(m: Media): Media[] {
  const mine = m.getBoundingClientRect();
  return [...media.keys()].filter((o) => o !== m && overlapFraction(mine, o.getBoundingClientRect()) >= 0.3);
}

/**
 * Players can swap a new <video> into a spot that is already blocked (Instagram Reels replaces the
 * poster when playback starts). Re-apply any block there so the newcomer is blurred and paused too.
 */
function extendBlocksTo(m: Media) {
  if (!active() || settings.mode !== 'block') return;
  const mine = m.getBoundingClientRect();
  for (const [o, ost] of media) {
    // Only a video's verdict extends; a thumbnail's block yields to the video (yieldThumbnailsTo).
    if (o === m || !isVideo(o) || !ost.overlay || ost.overlay.covers?.(m)) continue;
    if (overlapFraction(mine, o.getBoundingClientRect()) >= 0.3) render(o, ost);
  }
}

// ---------- jank back-off ----------
// If this page spends too long in long main-thread tasks (the user feels this as lag), ask the
// engine to back off. Whatever the cause, the extension should yield first.
// Measured as timer lateness: this content script shares the page's main thread, so a 250 ms
// tick that fires late means the page was stalled for that long. (The 'longtask' observer isn't
// reported in every Chromium build.) Only counted while the tab is visible, since hidden tabs
// throttle timers on purpose.
let jankTimer: ReturnType<typeof setInterval> | undefined;
const JANK_TICK_MS = 250;

function watchJank() {
  let expected = performance.now() + JANK_TICK_MS;
  let recent: Array<{ at: number; late: number }> = [];
  let lastReport = 0;
  jankTimer = setInterval(() => {
    const now = performance.now();
    const late = now - expected;
    expected = now + JANK_TICK_MS;
    if (document.visibilityState !== 'visible') return void (recent = []);
    if (late > 50) recent.push({ at: now, late });
    recent = recent.filter((e) => now - e.at < VISION.jankWindowMs);
    const stalled = recent.reduce((sum, e) => sum + e.late, 0);
    if (stalled > VISION.jankLongTaskMs && now - lastReport > VISION.jankWindowMs && active()) {
      lastReport = now;
      send({ kind: 'JANK' });
    }
  }, JANK_TICK_MS);
}

let rafPending = false;
function reposition() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    for (const st of media.values()) {
      st.overlay?.update();
      st.tag?.update();
    }
  });
}

// ---------- init ----------

async function init() {
  settings = await getSettings();
  onSettingsChanged((s) => {
    settings = s;
    for (const [m, st] of media) {
      if (!s.debug && st.tag) {
        st.tag.destroy();
        st.tag = undefined;
      }
      if (st.status === 'done') render(m, st);
      else if (st.visible) maybeScan(m);
    }
  });

  discover();
  mo.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
  addEventListener('scroll', reposition, { capture: true, passive: true });
  addEventListener('resize', reposition, { passive: true });
  repositionTimer = setInterval(reposition, 500); // layout shifts that don't scroll
  watchJank();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') for (const [m, st] of media) if (st.visible) maybeScan(m);
  });
  log('content script ready', location.hostname);
}

init();
