import { DEFAULT_THRESHOLD } from './config';
import type { Settings } from './types';

export const DEFAULT_SETTINGS: Settings = {
  mode: 'flag',
  fastMode: false,
  threshold: DEFAULT_THRESHOLD,
  debug: false,
  useGpu: true, // WebGPU is ~10x faster on Intel iGPUs; falls back to WASM when unavailable
};

export async function getSettings(): Promise<Settings> {
  const s = await chrome.storage.local.get('settings');
  return { ...DEFAULT_SETTINGS, ...(s.settings ?? {}) };
}

export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = { ...(await getSettings()), ...patch };
  await chrome.storage.local.set({ settings: next });
  return next;
}

export function onSettingsChanged(cb: (s: Settings) => void) {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.settings) cb({ ...DEFAULT_SETTINGS, ...changes.settings.newValue });
  });
}
