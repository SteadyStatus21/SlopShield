// First-run setup: 1) desktop helper (engine) with a live connection check, 2) warn or block, 3) done.
import { ENGINE } from '../config';
import { saveSettings } from '../settings';
import type { Mode } from '../types';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const steps = [$('step1'), $('step2'), $('step3')];

function go(i: number) {
  steps.forEach((s, k) => {
    s.classList.toggle('current', k === i);
    s.classList.toggle('done', k < i);
  });
  steps[i].scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Step 1: poll the engine until it answers hello.
let engineOk = false;
function checkEngine() {
  if (engineOk) return;
  let ws: WebSocket;
  try {
    ws = new WebSocket(ENGINE.url);
  } catch {
    return;
  }
  ws.onopen = () => ws.send(JSON.stringify({ type: 'hello' }));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(String(ev.data));
    engineOk = true;
    $('engineStatus').className = 'flag ok';
    $('engineText').textContent = `Helper connected${msg.device ? ` (${String(msg.device).replace(/ \(.*\)$/, '')})` : ''}`;
    $<HTMLButtonElement>('next1').disabled = false;
    ws.close();
  };
  ws.onerror = () => ws.close();
}
checkEngine();
const poll = setInterval(() => (engineOk ? clearInterval(poll) : checkEngine()), 2000);

$('next1').addEventListener('click', () => go(1));
$('skip1').addEventListener('click', () => go(1));

// Step 2: warn or block.
document.querySelectorAll<HTMLButtonElement>('.choice').forEach((b) =>
  b.addEventListener('click', async () => {
    document.querySelectorAll('.choice').forEach((x) => x.classList.toggle('on', x === b));
    await saveSettings({ mode: b.dataset.mode as Mode });
    setTimeout(() => go(2), 250);
  }),
);

// Step 3: done.
$('finish').addEventListener('click', async () => {
  await chrome.storage.local.set({ onboarded: true });
  const tab = await chrome.tabs.getCurrent();
  if (tab?.id !== undefined) chrome.tabs.remove(tab.id);
  else window.close();
});

$('version').textContent = `v${chrome.runtime.getManifest().version}`;
