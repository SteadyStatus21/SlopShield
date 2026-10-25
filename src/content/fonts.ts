// IBM Plex for our overlays. Shadow-DOM styles can't declare @font-face, so the faces are added to the
// page's FontFaceSet under our own family names (no clash with a site's own Plex). They're built from
// bytes rather than a URL, so a strict page CSP (font-src) can't block them; on failure the overlays
// just fall back to system fonts.
const FACES: Array<[family: string, file: string, weight: string]> = [
  ['SlopShield Plex Sans', 'fonts/plex-sans.woff2', '100 700'],
  ['SlopShield Plex Mono', 'fonts/plex-mono-400.woff2', '400'],
  ['SlopShield Plex Mono', 'fonts/plex-mono-500.woff2', '500'],
];

let started = false;

export function loadFonts() {
  if (started) return;
  started = true;
  for (const [family, file, weight] of FACES) {
    fetch(chrome.runtime.getURL(file))
      .then((r) => r.arrayBuffer())
      .then((bytes) => {
        const face = new FontFace(family, bytes, { weight });
        document.fonts.add(face);
        return face.load();
      })
      .catch(() => {});
  }
}
