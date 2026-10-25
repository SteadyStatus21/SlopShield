// Mode: Off | Flag | Block; fast mode; GPU opt-in; threshold slider; debug crop.
import { getSettings, saveSettings } from '../settings';
import type { BackendInfo, Mode } from '../types';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

interface Status {
  backend: BackendInfo;
  lastCrops: string[]; // the tiles the model saw for the most recent item
}

async function refresh() {
  const s = await getSettings();
  document.querySelectorAll<HTMLButtonElement>('#mode button').forEach((b) => {
    b.classList.toggle('on', b.dataset.mode === s.mode);
  });
  $<HTMLInputElement>('fast').checked = s.fastMode;
  $<HTMLInputElement>('gpu').checked = s.useGpu;
  $<HTMLInputElement>('threshold').value = String(s.threshold);
  $('thVal').textContent = s.threshold.toFixed(2);
  $<HTMLInputElement>('debug').checked = s.debug;

  const st: Status | null = await chrome.runtime.sendMessage({ kind: 'GET_STATUS' });
  if (!st) return;
  const be = $('backend');
  be.textContent = !st.backend.modelLoaded
    ? st.backend.error
      ? 'model: error'
      : 'model: idle'
    : st.backend.backend === 'engine'
      ? 'desktop engine'
      : `in-browser: ${st.backend.backend}`;
  be.className = `pill ${st.backend.modelLoaded ? 'ok' : st.backend.error ? 'bad' : ''}`;
  be.title = st.backend.error ?? st.backend.device ?? '';
  const crops = $('crops');
  crops.hidden = !(s.debug && st.lastCrops.length);
  if (crops.dataset.key !== st.lastCrops.join('|').slice(-64)) {
    crops.dataset.key = st.lastCrops.join('|').slice(-64);
    crops.replaceChildren(
      ...st.lastCrops.map((src) => Object.assign(document.createElement('img'), { src, alt: 'model input tile' })),
    );
  }
}

document.querySelectorAll<HTMLButtonElement>('#mode button').forEach((b) =>
  b.addEventListener('click', async () => {
    await saveSettings({ mode: b.dataset.mode as Mode });
    refresh();
  }),
);

const toggle = (id: string, key: 'fastMode' | 'useGpu' | 'debug') =>
  $<HTMLInputElement>(id).addEventListener('change', async (e) => {
    await saveSettings({ [key]: (e.target as HTMLInputElement).checked });
    refresh();
  });
toggle('fast', 'fastMode');
toggle('gpu', 'useGpu');
toggle('debug', 'debug');

$<HTMLInputElement>('threshold').addEventListener('input', (e) => {
  $('thVal').textContent = Number((e.target as HTMLInputElement).value).toFixed(2);
});
$<HTMLInputElement>('threshold').addEventListener('change', (e) =>
  saveSettings({ threshold: Number((e.target as HTMLInputElement).value) }),
);

refresh();
setInterval(refresh, 1000);
