// Mode: Off | Flag | Block; fast mode; GPU opt-in; threshold slider; debug crop.
import { ENGINE } from '../config';
import { getSettings, saveSettings } from '../settings';
import type { BackendInfo, ClearCacheResp, Mode } from '../types';

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
  be.className = `flag ${st.backend.modelLoaded ? 'ok' : st.backend.error ? 'bad' : ''}`;
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

$('flush').addEventListener('click', async () => {
  const r: ClearCacheResp | null = await chrome.runtime.sendMessage({ kind: 'CLEAR_CACHE' });
  $('flushMsg').textContent = r ? `Cleared ${r.cleared}. Re-checking…` : 'Could not clear the cache.';
});

// Desktop engine status, checked directly so it shows even before anything has been scanned.
function checkEngine() {
  const pill = $('engine');
  let ws: WebSocket;
  try {
    ws = new WebSocket(ENGINE.url);
  } catch {
    return;
  }
  const done = (text: string, cls: string, title = '') => {
    pill.textContent = text;
    pill.className = `flag ${cls}`;
    pill.title = title;
    ws.close();
  };
  ws.onopen = () => ws.send(JSON.stringify({ type: 'hello' }));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(String(ev.data));
    done('Desktop engine: connected', 'ok', `${msg.device} · ${msg.model}`);
  };
  ws.onerror = () => done('Desktop engine: not running (using in-browser model)', '', 'Start engine/start_engine.bat');
}

$('version').textContent = `v${chrome.runtime.getManifest().version}`;
$('setup').addEventListener('click', (e) => {
  e.preventDefault();
  chrome.tabs.create({ url: chrome.runtime.getURL('welcome/welcome.html') });
});
checkEngine();
refresh();
setInterval(refresh, 1000);
