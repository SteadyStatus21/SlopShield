// Blur + badge + "Show anyway".
import css from './overlay.css';
import type { Verdict } from '../types';

export interface Overlay {
  update(): void; // reposition over the media element
  destroy(): void;
  covers?(m: Element): boolean; // block mode: is m one of the elements blurred/paused by this overlay?
}

interface Opts {
  media: HTMLVideoElement | HTMLImageElement;
  verdict: Verdict;
  onReveal: () => void;
  stacked?: Array<HTMLVideoElement | HTMLImageElement>; // other media in the same spot (block mode)
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function mount(media: HTMLElement) {
  const host = document.createElement('slopshield-overlay');
  const root = host.attachShadow({ mode: 'closed' });
  const style = el('style');
  style.textContent = css;
  const frame = el('div', 'frame');
  root.append(style, frame);
  document.documentElement.append(host);

  const position = () => {
    const r = media.getBoundingClientRect();
    const hidden = !media.isConnected || r.width === 0 || r.height === 0;
    frame.style.display = hidden ? 'none' : '';
    frame.style.left = `${r.left}px`;
    frame.style.top = `${r.top}px`;
    frame.style.width = `${r.width}px`;
    frame.style.height = `${r.height}px`;
    frame.classList.toggle('small', r.height < 260);
  };
  position();
  return { host, frame, position };
}

// The score isn't a calibrated probability, so never show certainty.
const pct = (v: Verdict) => `${Math.min(95, Math.round(v.score * 100))}%`;

export function showBadge({ media, verdict }: Opts): Overlay {
  const { host, frame, position } = mount(media);
  const badge = el('div', 'badge', `⚠ Likely AI · ${pct(verdict)}`);
  const tip = el('div', 'tip', 'Why SlopShield flagged this:');
  const ul = el('ul');
  for (const r of verdict.reasons) ul.append(el('li', undefined, r));
  tip.append(ul);
  badge.append(tip);
  frame.append(badge);
  return { update: position, destroy: () => host.remove() };
}

/**
 * Blur (and for videos: mute + pause) one element until the returned function is called.
 *
 * Feeds like Instagram Reels call play() again the moment a video is paused. Re-pausing on every
 * `play` turns that into a tight loop (measured: ~4,500 play events/sec per blocked reel), so after
 * VIDEO_FIGHT_LIMIT re-plays within a second we stop fighting: the video stays muted and is hidden
 * with opacity instead (a CSS blur on a playing video is re-rendered every frame, which is costly
 * on integrated GPUs). The block overlay covers the spot either way.
 */
const VIDEO_FIGHT_LIMIT = 3;

function suppress(m: HTMLVideoElement | HTMLImageElement): () => void {
  const prev = { filter: m.style.filter, opacity: m.style.opacity };
  m.style.filter = `${prev.filter} blur(32px)`.trim();
  if (!(m instanceof HTMLVideoElement)) return () => (m.style.filter = prev.filter);

  const wasMuted = m.muted;
  m.muted = true;
  m.pause();
  let plays: number[] = [];
  const onPlay = () => {
    const now = performance.now();
    plays = plays.filter((t) => now - t < 1000);
    plays.push(now);
    if (plays.length <= VIDEO_FIGHT_LIMIT) {
      m.pause();
      return;
    }
    // The page insists on playing: stop fighting, hide it instead.
    m.removeEventListener('play', onPlay);
    m.muted = true;
    m.style.filter = prev.filter;
    m.style.opacity = '0';
  };
  m.addEventListener('play', onPlay);
  return () => {
    m.removeEventListener('play', onPlay);
    m.style.filter = prev.filter;
    m.style.opacity = prev.opacity;
    m.muted = wasMuted;
  };
}

export function showBlock({ media, verdict, onReveal, stacked = [] }: Opts): Overlay {
  const { host, frame, position } = mount(media);
  // Players often stack a poster <img> and the playing <video> in the same spot; whichever one
  // was flagged, everything in the spot is blurred and paused, or the video plays on top.
  const all = [media, ...stacked];
  const video = all.find((m): m is HTMLVideoElement => m instanceof HTMLVideoElement) ?? null;
  const restores = all.map(suppress);

  const block = el('div', 'block');
  const panel = el('div', 'panel');
  panel.append(
    el('h2', undefined, `⚠ This ${video ? 'video' : 'image'} is likely AI-generated`),
    el(
      'p',
      undefined,
      `SlopShield score: ${pct(verdict)}. Detection can be wrong, so check the reasons below. Be careful with requests for money or personal details.`,
    ),
  );
  const ul = el('ul');
  for (const r of verdict.reasons) ul.append(el('li', undefined, r));
  panel.append(ul);

  const reveal = el('button', undefined, 'Show anyway');
  reveal.addEventListener('click', () => done());
  panel.append(reveal);
  // Clicks on the overlay must not reach the player underneath.
  block.addEventListener('click', (e) => e.stopPropagation());
  block.append(panel);
  frame.append(block);

  const restore = () => {
    for (const r of restores) r();
    host.remove();
  };
  function done() {
    restore();
    onReveal();
  }
  return { update: position, destroy: restore, covers: (m) => all.includes(m as HTMLVideoElement) };
}

// ---------- debug tags ----------

export type TagTone = 'info' | 'real' | 'ai' | 'skip';

export interface DebugTag {
  set(text: string, tone: TagTone): void;
  update(): void;
  destroy(): void;
}

/**
 * Debug mode: a small status label attached to a media element. It sits just *outside* the
 * element (above it, or below it if there's no room) so it never ends up in the screenshot
 * crops the model scores.
 */
export function showDebugTag(media: HTMLElement): DebugTag {
  const host = document.createElement('slopshield-debug');
  const root = host.attachShadow({ mode: 'closed' });
  const style = el('style');
  style.textContent = css;
  const tag = el('div', 'dbg');
  root.append(style, tag);
  document.documentElement.append(host);

  const update = () => {
    const r = media.getBoundingClientRect();
    const hidden = !media.isConnected || r.width === 0 || r.height === 0 || r.bottom < 0 || r.top > innerHeight;
    tag.style.display = hidden ? 'none' : '';
    if (hidden) return;
    const h = tag.offsetHeight || 20;
    const above = r.top - h - 2;
    const below = r.bottom + 2;
    tag.style.left = `${Math.max(0, r.left)}px`;
    tag.style.top = `${above >= 0 ? above : below + h <= innerHeight ? below : Math.max(0, r.bottom - h)}px`;
    tag.style.maxWidth = `${Math.max(160, r.width)}px`;
  };
  return {
    set(text, tone) {
      tag.textContent = `🛡 ${text}`;
      tag.className = `dbg ${tone}`;
      update();
    },
    update,
    destroy: () => host.remove(),
  };
}
