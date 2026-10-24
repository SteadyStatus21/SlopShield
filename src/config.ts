// All tunables live here.

export const DEBUG = true;

export const WEIGHTS = {
  label: 1.0, // platform label is decisive; short-circuits the pipeline
  text: 0.15, // captions are a weak nudge only: they mislabel both ways (see TEXT_DECISIVE)
  vision: 0.75,
};

export const DEFAULT_THRESHOLD = 0.6;

// Media elements that get scanned. Still images need only one capture.
export const MEDIA_SELECTOR = 'video, img';

export const VISION = {
  // 6 frames 0.5 s apart = 3 s of video per verdict at the engine's ~2 fps.
  framesPerVideo: 6,
  framesPerImage: 1,
  // Stop sampling early once this many frames all agree the video is clearly real (saves compute).
  earlyStopFrames: 3,
  earlyStopMargin: 0.15, // stop early only if all scores are below this (clearly real)
  frameIntervalMs: 500,
  minVideoPx: 150, // skip media smaller than this on either side (CSS px): avatars, icons, emoji
  // Tall/wide media are covered by up to this many square tiles along the long side, scored in
  // the same batch and averaged (in logit space), so the model sees the whole post.
  maxTiles: 3,
  minVisibleRatio: 0.5,
  trimFraction: 0.25, // drop this fraction from each end before averaging
  minVariance: 40, // per-channel luminance variance below this = black/loading frame
  maxAttemptsPerVideo: 12, // give up after this many capture attempts
  // Performance budget (see the capture engine in background/sw.ts). After each batch the engine
  // rests so the model uses at most this fraction of the GPU/CPU. Measured on an Intel Iris Xe
  // (i7-12700H): ~160 ms/frame on WebGPU → ~2 fps; ~1.6 s/frame on 1-thread WASM → one frame/4 s.
  maxDutyCycle: 0.4,
  // captureVisibleTab allows ~2 screenshots/sec per extension: the hard fps ceiling.
  minCaptureIntervalMs: 520,
  batchWindowMs: 40, // gather requests that arrive together into one screenshot
  maxQueue: 8, // requests waiting for a screenshot; beyond this callers get 'busy'
  busyRetryMs: 800, // how long a content script waits after a 'busy' reply
  // Jank back-off: if the page is stalled for more than jankLongTaskMs in total within
  // jankWindowMs, the engine's rest period doubles (up to maxBackoff×), recovering by half
  // every backoffRecoverMs without new reports.
  jankWindowMs: 2000,
  jankLongTaskMs: 250,
  maxBackoff: 8,
  backoffRecoverMs: 10_000,
};

// Low resource use: the offscreen doc (and the model in RAM/VRAM) is torn down after
// this long with no inference, and re-created on demand.
export const OFFSCREEN_IDLE_MS = 90_000;

export const MODEL = {
  path: 'models/model.onnx',
  metaPath: 'models/model.json',
  // Used only if models/model.json is missing. Matches the shipped frontier ViT-S + head (scripts/finetune_head.py).
  fallbackMeta: {
    size: 384,
    resizeShortest: 440,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    aiIndex: 0,
    inputName: 'pixel_values',
  },
};

// Strong: platform-applied AI labels. Any match → flagged without inference.
// Only the exact wording of platform-applied badges. A bare "AI-generated" is NOT here: it also
// appears in captions, replies and questions ("is this AI-generated?") on real media.
export const PLATFORM_LABEL_PATTERNS: RegExp[] = [
  /\bAI info\b/i, // Meta (Instagram/Facebook)
  /\bMade with AI\b/i, // Meta (older label)
  /\bAltered or synthetic content\b/i, // YouTube
  /\bCreator labeled as AI-generated\b/i, // TikTok
  /\bAI-generated content label\b/i,
];

// Hashtags: medium-strength.
export const HASHTAGS: Record<string, number> = {
  '#ai': 0.35,
  '#aiart': 0.6,
  '#aivideo': 0.7,
  '#aigenerated': 0.8,
  '#sora': 0.7,
  '#sora2': 0.75,
  '#veo': 0.7,
  '#veo3': 0.75,
  '#midjourney': 0.6,
  '#runway': 0.4,
  '#kling': 0.6,
  '#klingai': 0.7,
  '#deepfake': 0.7,
  '#stablediffusion': 0.6,
  '#generativeai': 0.5,
};

// Free-text keyword hits in title/caption/channel name: weak.
export const KEYWORDS: Array<[RegExp, number]> = [
  [/\bAI[- ]?(generated|made|created|video|animation)\b/i, 0.5],
  // Video generators (2024–2026). Names alone are a moderate cue: they're also discussed in news.
  [/\b(sora ?2?|veo ?3(\.\d)?|kling( ?ai| ?\d(\.\d)?)?|hailuo|minimax|seedance( ?\d(\.\d)?)?|wan ?2(\.\d)?|hunyuan ?video|pika( labs| ?\d(\.\d)?)?|runway( gen-?\d| aleph)?|luma (dream machine|ray ?\d)|vidu|pixverse|higgsfield|midjourney video|grok imagine|dreamina|jimeng)\b/i, 0.5],
  // Same cues in other languages common on social media (es/pt/fr/it: "IA"; de: "KI").
  // "made with/by AI or <generator>" in es/pt/fr/it/de/en.
  [/\b(hech[oa]|cread[oa]|generad[oa]|fei(t|to)|gerad[oa]|criad[oa]|fait|cré[ée]e?|g[ée]n[ée]r[ée]e?|generat[oa]|realizzat[oa]|erstellt|generiert|gemacht|made|created|generated)s? (con|com|por|par|avec|da|dall'|mit|with|using|by|in) (la |l'|a |der |une )?(IA|I\.A\.|KI|AI|sora|veo|kling|seedance|hailuo|minimax|runway|pika|wan|hunyuan|vidu|pixverse|midjourney)\b/i, 0.6],
  [/\b(KI-generiert|mit KI (erstellt|generiert|gemacht))\b/i, 0.6],
  [/\b(video|vídeo|vidéo|clip|imagen|imagem|image|foto|personaje|personagem|personnage) (de |da |par )?(la |l')?IA\b/i, 0.4],
  [/\bmidjourney\b/i, 0.35],
  [/\bdeep ?fake\b/i, 0.5],
  [/\bnot real\b/i, 0.2],
  [/\bsynthetic\b/i, 0.2],
  [/\bAI (art|image|images|artwork|pic|pics|photo)\b/i, 0.5],
  // Seen on viral AI video: "AI mukbang videos are going viral", "AI-generated character".
  [/\bAI[- ](character|influencer|avatar|person|persona|girl|woman|man|baby|mukbang|asmr|slop|clone|actor|actress)\b/i, 0.5],
  // "This AI video…", "my AI short": the poster says the attached media is AI → decisive.
  [/\b(this|these|my|our|here'?s an?|watch this|check out this) (whole |entire |new |short )?(AI|A\.I\.|IA|KI)[- ]?(generated |made |created )?(video|clip|reel|short|film|movie|ad|commercial|music video|trailer|animation|image|photo|pic|art)s?\b/i, 0.5],
  [/\b(stable diffusion|dall-?e ?\d?|flux\.?1|nano banana|imagen ?\d?|firefly|leonardo\.?ai)\b/i, 0.45],
  // Posted in an AI-art community (e.g. Reddit r/aiimages).
  [/\br\/(aiimages|aiart|aiartwork|aiarthub|aigenerated|aivideo|aivideos|midjourney|stablediffusion|dalle|dalle2|weirddalle|sora|soraai|aiphotography|fluxai)\b/i, 0.6],
];

// Counter-signals: rendered graphics (games, CGI, animation) look "synthetic" to an image
// classifier on every frame, so more frames don't help. When the post says it's one of these,
// the vision score is multiplied by CGI_VISION_FACTOR.
export const CGI_CONTEXT: RegExp[] = [
  /\b(gameplay|game ?play|walkthrough|playthrough|let'?s play|speedrun|no commentary|gaming|gamer|esports)\b/i,
  /\b(ps[345]|playstation|xbox( series [xs])?|nintendo|switch 2|steam deck|rtx ?\d{4}|ray tracing|ultra (graphics|settings)|max settings|\d{3,4}p ?60 ?fps|4k ?60)\b/i,
  /\b(unreal engine ?\d?|ue[45]|unity engine|cryengine|frostbite)\b/i,
  /\b(forza( horizon| motorsport)?|gran turismo|need for speed|assetto corsa|beamng|iracing|gta ?(v|5|6|vi)?|grand theft auto|cyberpunk 2077|red dead|minecraft|fortnite|roblox|call of duty|warzone|valorant|counter-strike|cs2|league of legends|elden ring|zelda|mario kart|f1 ?2\d|the crew|euro truck simulator|flight simulator|msfs)\b/i,
  /\b(cgi|vfx|3d (render|animation|animated)|blender|cinema ?4d|octane render|animated (short|film|series)|pixar|cutscene|cinematic trailer|game trailer|official trailer)\b/i,
];
export const CGI_VISION_FACTOR = 0.35;

export const TEXT = {
  maxAncestorLevels: 8, // ancestors whose text counts as this post's caption
  maxAttrLevels: 20, // ancestors whose attributes (e.g. subreddit name) are read
  maxChars: 3000,
  // Ancestor attributes that describe the post (Reddit's <shreddit-post>, generic a11y attrs).
  attrs: ['subreddit-prefixed-name', 'subreddit-name', 'post-title', 'aria-label', 'title', 'alt', 'data-subreddit'],
};

// Text-cue score at or above which captions alone would flag media. Disabled (> 1): captions
// proved unreliable in both directions ("AI farm of phones" on real footage; AI videos with no
// caption cue), so only platform badges (PLATFORM_LABEL_PATTERNS) can decide without vision.
export const TEXT_DECISIVE = Infinity;
