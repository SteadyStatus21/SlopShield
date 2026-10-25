## Inspiration

Our grandparents' feeds are full of videos that never happened: a "soldier's daughter" singing to troops, a celebrity explaining how to double your savings, a cute animal rescue that's physically impossible. Modern video generators (Sora, Veo 3, Kling, Seedance) made this content cheap, and scammers noticed. Americans 60 and older reported **$7.75 billion** in internet-crime losses to the FBI in 2025, about 60% more than in 2024, and IC3 logged over 3,100 senior complaints that explicitly involved AI ([AARP on the FBI IC3 2025 report](https://www.aarp.org/money/scams-fraud/fbi-ftc-report-2025-losses.html), [HousingWire](https://www.housingwire.com/articles/fbi-seniors-cybercrime-2025/)).

Platform "AI info" labels exist, but most AI posts don't carry one. We wanted something that sits between the feed and the person and says *"careful, this is probably AI"* — without sending anyone's browsing to a server.

## What it does

**SlopShield** is a Chrome/Chromium extension (plus an optional desktop engine) that scans the images and videos you scroll past on Instagram, X, Reddit, YouTube or any site, and:

- **Flags** likely-AI media with a badge, or **blocks** it (blur + pause) with a one-line explanation and a "Show anyway" button.
- Runs **100% on your device.** Frames are cropped and scored locally; nothing is uploaded.
- Lets you **correct it with one click** ("Yes, AI" / "Not AI"). Each click saves that item as a labelled example, and `train.bat` retrains the detector from your labels in about a minute — but only installs (and pushes) the new model if it beats the old one on a held-out benchmark.

## How we built it

**Extension (TypeScript, Manifest V3).** A content script finds media (including inside Reddit's shadow DOM), watches what's actually on screen, and requests frames. The service worker captures the visible tab, trims letterbox/padding bars, cuts each item into square tiles that cover the whole post, and batches them for inference. Overlays live in shadow DOM so they never break the page.

**Detector.** A ViT-S/16 trained on images from 4,800+ generators ([Community Forensics](https://huggingface.co/papers/2411.04125), CVPR 2025, frontier fine-tune from Aug 2026), plus a small neural-network decision head we trained ourselves on ~15k crops of modern AI video (Veo 3, Veo 3.1, Kling, Sora) vs. real video, with frames degraded to look like reposted social-media video, real photos, game screenshots, generated text/meme cards, and the user's own labels. It only flags when the head **and** the base model agree, which is what keeps false alarms down.

**Fast on a normal laptop.** Our minimum spec is an i7-12700H with Intel Iris Xe graphics:

| Runtime | Time per crop |
|---|---|
| WASM (CPU, in browser) | ~1.6 s |
| WebGPU (Iris Xe, in browser) | ~150 ms |
| **Desktop engine, OpenVINO on Iris Xe** | **~47 ms** |

The engine is a small Python app on `127.0.0.1` that only accepts our extension's origin; if it's not running, the extension falls back to WebGPU automatically.

**Measuring honestly.** `scripts/bench.py` scores every candidate model exactly the way the extension does, on sets it never trained on: 60 modern AI + 60 real YouTube videos, 80 still images, 30 video-game screenshots, and a held-out slice of our own labelled feed.

## Results (held-out, threshold 0.6)

| | Modern AI videos caught | Real videos flagged | Real photos flagged | Game screenshots flagged |
|---|---|---|---|---|
| Our first model (SigLIP) | — | — | 2/40 | **16/30** |
| Off-the-shelf CommunityForensics | 14/60 | 0/60 | 0/40 | 0/30 |
| Frontier base model alone | 57/60 | **12/60** | 0/40 | 0/30 |
| **SlopShield (MLP head + agreement rule)** | **57/60** | **1/60** | **0/40** | **0/30** |

On our own in-the-wild feed (held-out labels) it still misses about half of the AI clips — modern generators reposted through Instagram/X are the hardest case, and that's exactly what the labelling loop is for.

## Challenges we ran into

- **Video games and CGI** look "synthetic" to image detectors on every frame — our first model flagged 16 of 30 game screenshots. Fixed by switching models and requiring two models to agree.
- **Instagram Reels fought back.** Pausing a blocked reel made Instagram call `play()` again: ~4,500 play events per second per blocked reel, which made the browser unusable. We now pause once, mute, and hide instead of fighting.
- **Thumbnails vs. videos.** Instagram stacks a poster image under every reel; we were blocking the poster while the real video played on top.
- **Our own UI leaked into the model's input:** a debug tag inside one crop, and black carousel padding in others both caused false positives. Now tags hide during capture and flat borders are trimmed.
- **GPU backends lie.** DirectML on Iris Xe returned the same number for every image; we verified each backend (WebGPU, OpenVINO) against CPU outputs before trusting it.
- **Captions are unreliable** in both directions ("AI farm of phones" on real footage; AI videos with no caption at all), so only real platform badges can override the vision model.

## Accomplishments that we're proud of

- Catching **57 of 60** modern AI videos while flagging **1 of 60** real videos, **0 of 40** real photos and **0 of 30** game screenshots — on a laptop iGPU, with no cloud. Overall: **84/114 AI caught, 2/140 real items flagged** across every held-out set.
- A real human-in-the-loop: label in the feed → retrain in ~1 minute → auto-install only if the benchmark improves → auto-commit to GitHub.
- Every number above comes from a reproducible benchmark script, not a demo cherry-pick.

## What we learned

Per-frame image detectors can't see motion glitches (a spoon teleporting into a pan), they're easily fooled by graphics, text cards and phone beauty filters, and accuracy numbers mean nothing unless you test on the content your users actually see. Measuring on our own feed changed almost every design decision.

## What's next

- Temporal (multi-frame) detection to catch motion artifacts.
- A second opinion from a larger detector (we already run the Gemma-based Smaug detector on the Iris Xe via OpenVINO) for borderline calls.
- Parent mode (PIN-protected), per-site allowlists, and warnings for remote-access / tech-support scam pages.
- A mobile version.
