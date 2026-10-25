# SlopShield

A Chrome extension (Manifest V3) that spots likely AI-generated video on any webpage and flags or blurs it. **Everything runs on your device.** No video, frames or page text leave the machine.

Built for HackPSU Fall 2026. The main audience is older adults, who are targeted by AI scam videos on social media (fake celebrity investment pitches, fake disaster footage, "type AMEN" engagement bait).

> TODO before submission: add sourced stats for the problem statement (e.g. FTC Consumer Sentinel fraud losses for 60+, FBI IC3 Elder Fraud Report).

## How it works

```
content script ──(video rects, page text)──▶ service worker ──▶ offscreen document
   │                                             │                  (ONNX Runtime Web,
   │◀────────(verdict: score, reasons)───────────┘                   WebGPU → WASM fallback)
   └─ applies blur overlay / badge
```

For each video or image that is at least half on screen (including ones inside shadow DOM, as on Reddit), the cheapest checks run first and the pipeline stops as soon as it's confident:

1. **Platform labels.** Looks for "AI info", "Made with AI", "Altered or synthetic content" and similar in the post around the video. A label is decisive, so no inference runs.
2. **Text cues.** Scores hashtags (`#sora`, `#veo3`, `#aivideo` …) and caption keywords, combined with noisy-OR. In **fast mode** the pipeline stops here.
3. **Vision.** Takes up to 6 frames of a video, 0.5 s apart (1 for an image), while it is visible. The service worker calls `captureVisibleTab`, crops to the video, center-crops to the model's input (384 px), and the offscreen document runs the ONNX model, a 22M-param ViT-S (Community Forensics frontier model, Aug 2026) with a head SlopShield trained on modern AI video, flagging only when both agree (see `MODEL.md`: 57/60 modern AI videos caught; 2 false alarms across 140 held-out real videos, photos, game screenshots and labelled feed items). Frames are combined with a trimmed mean.
4. **Combine.** A weighted score (`WEIGHTS` in `src/config.ts`) is compared with your threshold. Every verdict comes with `reasons[]`, which the badge or overlay shows.

Verdicts are cached per video (`currentSrc`, or page URL + index for MediaSource players), so a video is never scanned twice.

## Low resource use

SlopShield is built to run on ordinary laptops without slowing them down:

- **Nothing runs until media is on screen.** Videos and images are found with `MutationObserver` + `IntersectionObserver`, swept at most once per second while the page is idle. Off-screen and tiny (<150 px) media (avatars, icons) are skipped.
- **Cheapest signal first.** Labels and text cost microseconds. The model only runs when those are inconclusive, and fast mode turns it off entirely.
- **Few frames.** At most 6 frames per video (fewer when the first 3 are clearly real), 1 per image. Sampling stops after 2 if they agree confidently, and runs once per video (cached).
- **One screenshot, one batch.** All media visible at the same moment share one `captureVisibleTab` screenshot, and their crops run through the model as one batch. Chrome allows ~2 screenshots/sec, so the engine tops out at ~2 fps.
- **Hard compute budget.** After each batch the engine rests in proportion to how long inference took, so the model uses at most ~40% of the GPU/CPU (`VISION.maxDutyCycle`). The queue is capped; extra requests get "busy" and retry, so work never piles up while scrolling.
- **GPU by default, CPU fallback.** Measured on the minimum-spec target, an Intel Iris Xe laptop (i7-12700H): **~150 ms/frame on WebGPU** (≈2 fps, capped by the screenshot quota) vs ~1.6 s/frame on single-threaded WASM. Multi-threaded WASM barely helped (1.4 s with 4 threads). Without WebGPU it falls back to WASM automatically, and the popup's "Use GPU" toggle forces CPU if the browser stutters.
- **Off the main thread.** Cropping happens in the service worker and inference in an offscreen document, so pages don't jank.
- **Small model.** A 22M-parameter ViT-S (87 MB ONNX). It replaced a 93M SigLIP-B that was slower and less accurate (see `MODEL.md`).
- **Unloads when idle.** The offscreen document closes itself after 90 s without inference, which frees the model's memory. It reloads on demand.
- **Blank frames skipped.** Black, loading or DRM frames (near-zero variance) are dropped before inference.

## Privacy

- Frames are captured, cropped and scored entirely in the extension. The model is packaged inside the extension, and inference makes no network requests (check the DevTools Network tab).
- Settings live in `chrome.storage.local`. Verdicts live in `chrome.storage.session` and are cleared when the browser closes.
- There is no backend. Everything runs in the browser extension.

## Setup

```bash
npm install
python scripts/fetch_model.py      # downloads and exports the detector into models/ (see MODEL.md)
npm run build                      # → dist/
```

Then open `chrome://extensions`, enable Developer mode, click **Load unpacked**, and pick `dist/`.

Without `models/model.onnx` the extension still works with labels and text only. The popup shows `model: error`.

### Optional: desktop engine (faster, uses the Intel GPU natively)

`engine/slopshield_engine.py` is a small local app that scores crops with OpenVINO on the GPU
(Iris Xe: ~47 ms of inference, ~120 ms per crop round trip, vs ~150 ms in-browser WebGPU). When it's running, the extension
sends crops to it; when it isn't, the extension scores in the browser as before. The popup shows
which one is active ("desktop engine" / "in-browser").

```bash
.venv/Scripts/pip install -r engine/requirements.txt
engine/start_engine.bat            # or: python engine/slopshield_engine.py
```

**Labeling:** while the engine runs, flagged/blocked items show "Is this right? [Yes, AI] [Not AI]"
(and in Debug mode every scanned item's tag has [AI] [Real]). Each click saves the item's recent
crops to `dataset/ai/` or `dataset/real/` on your machine for evaluation and retraining.

**Retrain from your labels:** double-click `train.bat` (or `python scripts/train_from_labels.py`). It holds
out ~1 in 5 labelled items, extracts features on the GPU (cached, so repeat runs take ~30 s), retrains the
head on the Hugging Face set + your labels, prints a held-out scorecard (old vs new), backs up the current
model to `models/_ref/backups/`, installs the new one and rebuilds `dist/`. Then restart the engine and
reload the extension. `train.bat --dry-run` reports without installing.

It listens only on `127.0.0.1:47321` and accepts only the SlopShield extension (the manifest pins
the extension ID `jpcjjoehiklpjalioikahehafipnbjjk`), so websites can't use it. Nothing leaves the machine.

### Demo feed

Put clips in `demo/videos/` named `ai_XX.mp4` / `real_XX.mp4` and list them in `demo/clips.js`. Then serve the folder (content scripts don't run on `file://` unless you allow it) and open http://127.0.0.1:8765/demo/:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

### Eval

```bash
python scripts/eval.py demo/videos --frames 5 --threshold 0.6
```

This prints per-clip scores, precision/recall/F1 and the confusion matrix.

| Set | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Modern AI video (Sora, Veo 3, Veo 3.1, Kling) vs. real YouTube/AudioSet clips | 120 | 0.98 | 0.95 | 0.97 |
| [AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark) random sample, still images (40 real / 40 AI, 14 generators) | 80 | 1.00 | 0.53 | 0.69 |
| Video-game screenshots ([game_screenshots_11k](https://huggingface.co/datasets/badigadiii/game_screenshots_11k)), all not-AI | 30 | — | — | 0/30 flagged |
| Held-out slice of our own labelled feed (Instagram/X/Reddit) | 24 | 0.86 | 0.43 | 0.57 |

Measured Oct 25, 2026 with `python scripts/bench.py` (the same tiling, trimming and agreement rule as the extension) at threshold 0.6. Overall: 84/114 AI caught, 2/140 real items flagged. The model is tuned to avoid false alarms, so it gives up some recall on older still-image generators. The previous model (SigLIP-B) flagged **16/30 game screenshots** as AI; `MODEL.md` has the full comparison. Samples are small.

**Video games and CGI.** The first model called rendered graphics "AI" on every frame, so more frames only made it more confident. Switching models fixed most of this (0/30 game screenshots flagged). As a second safeguard, SlopShield detects gameplay/CGI context (game titles, "gameplay", engine names, "CGI", "3D render"…) and skips the image model there, judging those videos on labels and text only.

> TODO: add demo-clip numbers.

## Using it

- **Popup:** mode (Off / Flag / Block), fast mode, GPU toggle, sensitivity threshold, and a debug view of the last crop.
- **Flag:** a red "Likely AI · 87%" badge on the video. Hover it for the reasons.
- **Block:** the video is blurred and paused, with an explanation and a **Show anyway** button.

## Limitations

- Tested on the local demo feed and YouTube. It runs on any site (Facebook, Instagram, X, TikTok web, Reddit), but those aren't tested.
- Without WebGPU, a frame takes ~1.6 s on the CPU, so a busy feed is scanned gradually. With WebGPU the ~2 screenshots/sec quota is the limit.

- Image-trained detectors have a domain gap on compressed video frames and on newer generators. Expect false positives and false negatives, and treat the score as a warning, not proof.
- Videos inside cross-origin iframes (e.g. embedded YouTube players) aren't scanned yet (`all_frames: false`).
- Overlays don't show in fullscreen.
- DRM video captures as black and is skipped. Labels and text still apply.
- Only the active tab can be captured, so background tabs are scanned when you switch to them.
- Platform label strings are English-only.

## Next steps

Parent mode (PIN-protected "Show anyway" and off switch), a per-site allowlist, and session stats in the popup. Video-native detection (temporal artifacts), a smaller distilled model, more languages for labels, a mobile build, and user feedback to correct verdicts.

## Project layout

Tunables are in `src/config.ts`. Model details are in `MODEL.md`.
