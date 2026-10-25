# SlopShield — game plan

HackPSU Fall 2026. **Submission due Sun Oct 25, 12:00 PM** (hacking ends 1:45 PM). Last updated Sat Oct 24, ~18:30.

## Where we are

- Extension works end to end in Helium: finds images/videos (incl. shadow DOM), tiles crops, scores on WebGPU (~150 ms/frame on Iris Xe, verified identical to CPU), flags or blocks, debug tags, ~2 fps budget with jank back-off.
- Detector: frontier ViT-S + our social-video head, "both must agree" rule (`MODEL.md`).
  Held-out: 53/60 modern AI videos caught; 1/60 real videos, 0/40 photos, 0/30 game screenshots flagged; user clips 3/7.
- Captions are a weak nudge only; just platform badges are decisive.
- Repo: github.com/SteadyStatus21/SlopShield (commit + push every change, author SteadyStatus21, Claude co-author).

## Priority 1 — false positives on real video (the thing hurting us now)

Example: "Talking To Brick Wall — Maronzio Vance (2010)" on X blocked at **0.99**. A real video wrongly blocked is worse than a miss for our users.

0. **Label while you browse (built, Oct 24 22:10).** With the desktop engine running, every flag/block
   overlay has "Is this right? [Yes, AI] [Not AI]", and in Debug mode every scanned item's tag has
   [AI] [Real]. A click saves that item's recent clean crops to `dataset/ai/<site>_<id>/` or
   `dataset/real/<site>_<id>/` (re-labelling moves it). Next: retrain the head on these labels with
   OpenVINO feature extraction (minutes, not 40 min of CPU), keeping a held-out split for honest numbers.
1. **Collect real negatives from the actual feeds.** Restructure `dataset/` into:
   - `dataset/ai/` — verified AI (move the current 7 clips here)
   - `dataset/real/` — real videos the model wrongly flagged, plus ordinary real feed videos (old uploads, vlogs, music videos, webcam/phone footage, screen recordings, cartoons/anime, game clips)
   Name doesn't matter; the folder is the label. Add a clip every time the model is wrong in either direction.
2. **Measure on it** (held-out): `python scripts/eval.py dataset/real --label real --frames 6` and the same for `dataset/ai --label ai`. Track false-positive rate as the headline number.
3. **Retrain the head with hard negatives.** Add a real-video source mix to `scripts/finetune_head.py` beyond AudioSet: low-quality/old web video, music videos, talking heads, animation, game footage, screen recordings (from HF, disjoint from test). Feature cache means reruns only process new clips. Never train on `dataset/` — it's the test set.
4. **Pick the operating point for low false positives.** Choose threshold / agreement levels (`AGREE_HEAD_P`, `AGREE_BASE_P`) so held-out real FP ≤ ~2%, and report the recall we get there.
5. **Safer UX for uncertain verdicts.** Vision-only verdicts below a high bar (e.g. < 0.9) get the Flag badge, not a full Block; Block reserved for very confident or platform-labelled media. Show the score/reasons in the block panel (currently hidden in the small layout).

## Priority 2 — catch more AI video (after FP is under control)

- 4/7 of the user's in-the-wild clips are still missed: add more modern generator data (Seedance, Wan, Hailuo, Veo 3.1, Sora 2) to the training mix, with heavy repost/compression augmentation.
- Stretch: temporal signal (frame-to-frame consistency, e.g. objects teleporting) — a per-frame image model can't see motion glitches.

## Priority 3 — submission (keep ≥3 h before noon for this)

- [ ] `demo/videos/` + `demo/clips.js`: 10+ labelled clips (half real, half AI) for the live demo; demo must work offline.
- [ ] README: problem statement with sources, final eval tables (from `MODEL.md`), limitations, setup.
- [ ] ≤5 min video: problem → live scroll on demo feed/X → block → Wi-Fi off → numbers → limitations → next steps.
- [ ] Devpost write-up.
- [ ] Rebuild `dist/` from a clean checkout to confirm setup instructions work (`npm ci`, `scripts/finetune_head.py --reexport-from` or fetch, `npm run build`).

## Next steps to pitch (not building now)

Parent mode (PIN), per-site allowlist, popup stats, remote-access/tech-support scam warnings, mobile, temporal video model.

## Commands

```
npm run build                                   # → dist/, load unpacked in Helium
.venv/Scripts/python scripts/eval.py dataset/ai --label ai --frames 6
.venv/Scripts/python scripts/eval.py dataset/real --label real --frames 6
.venv/Scripts/python scripts/finetune_head.py --exclude <test-video-dir> --exclude dataset --out models
```
