// Platform label + hashtag + caption scoring, and the final combine.
import {
  CGI_CONTEXT,
  CGI_VISION_FACTOR,
  TEXT_DECISIVE,
  HASHTAGS,
  KEYWORDS,
  MEDIA_SELECTOR,
  PLATFORM_LABEL_PATTERNS,
  TEXT,
  VISION,
  WEIGHTS,
} from '../config';
import type { Verdict } from '../types';

// Page-level containers that describe the "main" video on watch pages.
const PRIMARY_DESC_SELECTORS = [
  'ytd-watch-metadata', // YouTube title + description (incl. "Altered or synthetic content")
  '#description',
  'ytd-reel-video-renderer[is-active]', // YouTube Shorts
  '[data-e2e="browse-video-desc"]', // TikTok
  '[data-e2e="video-desc"]',
  '[data-testid="tweetText"]', // X
];

function isPrimary(media: HTMLElement): boolean {
  const r = media.getBoundingClientRect();
  return r.width * r.height > 0.25 * innerWidth * innerHeight;
}

/** Parent element, stepping out of shadow roots to their host. */
export function parentOf(el: Element): HTMLElement | null {
  if (el.parentElement) return el.parentElement;
  const root = el.getRootNode();
  return root instanceof ShadowRoot ? (root.host as HTMLElement) : null;
}

// Elements that wrap exactly one post on common sites (X, Reddit, generic <article> feeds).
const POST_BOUNDARY = 'article, [role="article"], [data-testid="tweet"], shreddit-post, ytd-rich-item-renderer';

/** Another scannable-size image/video under `node` (avatars and icons don't count). */
function hasOtherLargeMedia(node: HTMLElement, self: HTMLElement): boolean {
  for (const el of node.querySelectorAll<HTMLElement>(MEDIA_SELECTOR)) {
    if (el === self) continue;
    const r = el.getBoundingClientRect();
    if (r.width >= VISION.minVideoPx && r.height >= VISION.minVideoPx) return true;
  }
  return false;
}

/** Text near the media: post text, post attributes, aria-labels, page metadata for the main video. */
export function gatherText(media: HTMLElement): string {
  const parts: string[] = [location.pathname];
  let node: HTMLElement | null = parentOf(media);
  let container: HTMLElement | null = null;
  let containerDone = false;
  for (let i = 0; node && i < TEXT.maxAttrLevels; i++, node = parentOf(node)) {
    if (node === document.body || node === document.documentElement) break;
    for (const a of TEXT.attrs) {
      const v = node.getAttribute(a);
      if (v) parts.push(v);
    }
    if (containerDone || i >= TEXT.maxAncestorLevels) continue;
    const len = node.textContent?.length ?? 0;
    // Climbed past this post into the whole feed: too much text, or another post's media inside.
    if (len > TEXT.maxChars || hasOtherLargeMedia(node, media)) {
      containerDone = true;
      continue;
    }
    container = node;
    // The post itself is the boundary: replies, quoted posts and the rest of a thread are other
    // people's words about the media, not a description of it.
    if (node.matches(POST_BOUNDARY)) containerDone = true;
  }
  if (container) {
    parts.push(container.textContent ?? '');
    container.querySelectorAll('[aria-label], [title], img[alt]').forEach((el, i) => {
      if (i > 60) return;
      parts.push(el.getAttribute('aria-label') ?? '', el.getAttribute('title') ?? '', el.getAttribute('alt') ?? '');
    });
  }
  if (isPrimary(media)) {
    parts.push(document.title);
    for (const sel of PRIMARY_DESC_SELECTORS) {
      const el = document.querySelector<HTMLElement>(sel);
      if (el) parts.push((el.textContent ?? '').slice(0, TEXT.maxChars));
    }
  }
  return parts.join(' \n ').replace(/\s+/g, ' ').slice(0, TEXT.maxChars * 2);
}

export function findPlatformLabel(text: string): string | null {
  for (const re of PLATFORM_LABEL_PATTERNS) {
    const m = text.match(re);
    if (m) return m[0];
  }
  return null;
}

export function scoreText(text: string): { score: number; reasons: string[] } {
  const hits: Array<[string, number]> = [];
  const tags = new Set((text.toLowerCase().match(/#[a-z0-9_]+/g) ?? []).map((t) => t.replace(/_/g, '')));
  for (const [tag, w] of Object.entries(HASHTAGS)) if (tags.has(tag)) hits.push([tag, w]);
  const plain = text.replace(/#[\w]+/g, ' '); // don't count a hashtag twice as a keyword
  for (const [re, w] of KEYWORDS) {
    const m = plain.match(re);
    if (m) hits.push([`"${m[0]}"`, w]);
  }
  // Noisy-OR: independent weak signals add up but never exceed 1.
  const score = 1 - hits.reduce((acc, [, w]) => acc * (1 - w), 1);
  return { score, reasons: hits.length ? [`Text cues: ${hits.map(([h]) => h).join(', ')}`] : [] };
}

/** Gameplay / CGI / animation context, where the image model can't tell rendered graphics from AI. */
export function findCgiContext(text: string): string | null {
  for (const re of CGI_CONTEXT) {
    const m = text.match(re);
    if (m) return m[0];
  }
  return null;
}

export function trimmedMean(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const k = Math.floor(s.length * VISION.trimFraction);
  const kept = s.length - 2 * k > 0 ? s.slice(k, s.length - k) : s;
  return kept.reduce((a, b) => a + b, 0) / kept.length;
}

export function combine(opts: {
  label: string | null;
  text: { score: number; reasons: string[] } | null;
  vision: number | null;
  frames: number;
  threshold: number;
  cgi?: string | null;
}): Verdict {
  const { label, text, frames, threshold, cgi } = opts;
  let { vision } = opts;
  const reasons: string[] = [];
  let score: number;
  if (cgi && vision !== null) vision *= CGI_VISION_FACTOR;

  if (label) {
    // A platform label is decisive.
    score = WEIGHTS.label;
    reasons.push(`AI label or disclosure on the post: "${label}"`);
  } else {
    // Missing text cues aren't evidence a video is real, so text only counts when it fired.
    let num = 0;
    let den = 0;
    if (text && text.score > 0) {
      num += WEIGHTS.text * text.score;
      den += WEIGHTS.text;
      reasons.push(...text.reasons);
    }
    if (vision !== null) {
      num += WEIGHTS.vision * vision;
      den += WEIGHTS.vision;
      reasons.push(`Vision score ${vision.toFixed(2)} (${frames} frame${frames === 1 ? '' : 's'})`);
    }
    score = den > 0 ? num / den : 0;
    // A strong caption cue ("made with Seedance", "generado por IA") is the creator disclosing it:
    // treat it like a label instead of letting a confident-but-wrong vision score outvote it.
    if (text && text.score >= TEXT_DECISIVE) score = Math.max(score, text.score);
  }
  if (cgi && !label) {
    const what = opts.vision === null ? 'so it was skipped' : 'so its score was down-weighted';
    reasons.push(`Looks like gameplay/CGI ("${cgi}"): the image model can't tell rendered graphics from AI, ${what}`);
  }
  if (reasons.length === 0) reasons.push('No AI signals found');

  return {
    score: Math.min(1, Math.max(0, score)),
    flagged: score >= threshold,
    reasons,
    label: !!label,
    text: text?.score ?? null,
    vision,
    frames,
    at: Date.now(),
  };
}
