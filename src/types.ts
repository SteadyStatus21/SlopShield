export type Mode = 'off' | 'flag' | 'block';

export interface Settings {
  mode: Mode;
  fastMode: boolean; // text/label signals only, no vision
  threshold: number;
  debug: boolean;
  useGpu: boolean; // WebGPU (default): ~10x faster on Intel iGPUs. Off = single-threaded WASM only
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Verdict {
  score: number; // 0..1, combined
  flagged: boolean; // score >= threshold at the time it was computed
  reasons: string[];
  label: boolean;
  text: number | null;
  vision: number | null;
  frames: number;
  at: number;
}

export interface BackendInfo {
  backend: 'engine' | 'webgpu' | 'wasm' | 'none'; // engine = desktop app (engine/), else in-browser
  device?: string; // engine only: e.g. 'GPU (Intel(R) Iris(R) Xe Graphics)'
  modelLoaded: boolean;
  error?: string;
}

// content → sw
export type ContentMsg =
  | { kind: 'CAPTURE_REQ'; rect: Rect; viewportWidth: number; viewportHeight: number }
  | { kind: 'CACHE_GET'; key: string }
  | { kind: 'CACHE_SET'; key: string; verdict: Verdict }
  | { kind: 'JANK' }; // the page had too much long-task time recently

export type CaptureResp =
  | { ok: true; score: number }
  | { ok: false; reason: 'busy' | 'blank' | 'offscreen' | 'model' | 'error'; detail?: string };

// sw → offscreen
export type OffscreenMsg =
  | { target: 'offscreen'; kind: 'INFER'; images: string[]; gpu: boolean } // data URLs, already resized
  | { target: 'offscreen'; kind: 'STATUS' };

export type InferResp =
  | { ok: true; scores: number[]; backend: BackendInfo['backend']; device?: string }
  | { ok: false; error: string };

// popup → sw
export type UiMsg = { kind: 'GET_STATUS' };

export type Msg = ContentMsg | OffscreenMsg | UiMsg;

export interface ModelMeta {
  size: number;
  // If set: resize shortest edge to this, then center-crop size×size. Otherwise squash to size×size.
  resizeShortest?: number;
  mean: number[];
  std: number[];
  aiIndex: number;
  inputName: string;
  logitBias?: number; // added to a single raw logit before the sigmoid (model's published calibration)
  name?: string;
  labels?: string[];
}
