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

**Detector.** A ViT-S/16 trained on images from 4,800+ generators ([Community Forensics](https://huggingface.co/papers/2411.04125), CVPR 2025, frontier fine-tune from Aug 2026), plus a small neural-network decision head we trained ourselves on ~19k crops: modern AI video (Veo 3, Veo 3.1, Kling, Sora) and 470 phone-style AI shorts vs. real YouTube clips and modern stock footage, with frames degraded to look like reposted social-media video, plus real photos, game screenshots, generated text/meme cards, and the user's own labels. The base model keeps a veto when it's almost certain a frame is real, which keeps games and photos clean.

**Fast on a normal laptop.** Our minimum spec is an i7-12700H with Intel Iris Xe graphics:

| Runtime | Time per crop |
|---|---|
| WASM (CPU, in browser) | ~1.6 s |
| WebGPU (Iris Xe, in browser) | ~150 ms |
| **Desktop engine, OpenVINO on Iris Xe** | **~47 ms** |

The engine is a small Python app on `127.0.0.1` that only accepts our extension's origin; if it's not running, the extension falls back to WebGPU automatically.

**Measuring honestly.** `scripts/bench.py` scores every candidate model exactly the way the extension does, on sets it never trained on: 60 modern AI + 60 real YouTube videos, 241 phone-style AI shorts, 183 clips from the newest generators (Seedance 1.5 Pro, Kling 2.5), 198 real stock videos, 80 still images, 30 video-game screenshots, and a held-out slice of our own labelled feed.

## Results (held-out, threshold 0.6)

| | Modern AI videos caught | Phone-style AI shorts caught | Newest generators caught | Real videos flagged | Real stock videos flagged | Real photos flagged | Game screenshots flagged |
|---|---|---|---|---|---|---|---|
| Our first model (SigLIP) | — | — | — | — | — | 2/40 | **16/30** |
| Off-the-shelf CommunityForensics | 14/60 | — | — | 0/60 | — | 0/40 | 0/30 |
| Frontier base model alone | 57/60 | 81/241 | 163/183 | **12/60** | 19/198 | 0/40 | 0/30 |
| **SlopShield** | **54/60** | **207/241** | **162/183** | **1/60** | **13/198** | **0/40** | **0/30** |

On our own feed (115 posts we labelled on Instagram, X and Facebook, cross-validated) it catches about 64% of the AI posts and flags about 9% of the real ones. Ordinary phone-style AI is the hardest case, which is why we spent the last night adding 470 phone-style AI shorts and 200 modern real videos to training: held-out shorts went from 43% caught to 86%, while false alarms on modern real stock footage halved (26 of 198 to 13).

## Challenges we ran into

- **Video games and CGI** look "synthetic" to image detectors on every frame — our first model flagged 16 of 30 game screenshots. Fixed by switching models and requiring two models to agree.
- **Instagram Reels fought back.** Pausing a blocked reel made Instagram call `play()` again: ~4,500 play events per second per blocked reel, which made the browser unusable. We now pause once, mute, and hide instead of fighting.
- **Thumbnails vs. videos.** Instagram stacks a poster image under every reel; we were blocking the poster while the real video played on top.
- **Our own UI leaked into the model's input:** a debug tag inside one crop, and black carousel padding in others both caused false positives. Now tags hide during capture and flat borders are trimmed.
- **GPU backends lie.** DirectML on Iris Xe returned the same number for every image; we verified each backend (WebGPU, OpenVINO) against CPU outputs before trusting it.
- **Captions are unreliable** in both directions ("AI farm of phones" on real footage; AI videos with no caption at all), so only real platform badges can override the vision model.

## Accomplishments that we're proud of

- Catching **54 of 60** modern AI videos, **207 of 241** phone-style AI shorts and **162 of 183** clips from the newest generators, while flagging **1 of 60** real videos, **13 of 198** real stock videos, **0 of 40** real photos and **0 of 30** game screenshots — on a laptop iGPU, with no cloud. Overall: **450/538 AI caught, 16/338 real items flagged** across every held-out set.
- A real human-in-the-loop: label in the feed → retrain in ~1 minute → auto-install only if the benchmark improves → auto-commit to GitHub.
- Every number above comes from a reproducible benchmark script, not a demo cherry-pick.

## What we learned

Per-frame image detectors can't see motion glitches (a spoon teleporting into a pan), they're easily fooled by graphics, text cards and phone beauty filters, and accuracy numbers mean nothing unless you test on the content your users actually see. Measuring on our own feed changed almost every design decision. For example, we got a 20x larger Gemma-based detector (Smaug, 439M params) running on the Iris Xe through OpenVINO as a "second opinion". It looked great on public benchmarks, but on our own labelled feed it ranked AI vs. real worse than a coin flip (AUC 0.45 vs. 0.69 for ours), so it didn't ship. And the safety gate we'd added early on (only flag when the base model agrees) turned out, once the training data was better, to be mostly vetoing real catches: relaxing it took held-out phone-style AI from 62% caught to 86% with no extra false alarms on photos or games.

## What's next

- Temporal (multi-frame) detection to catch motion artifacts.
- More in-the-wild training data: ordinary phone-style AI clips (Sora 2 app reposts) are the hardest case, and the one-click labelling loop is how we collect them.
- Parent mode (PIN-protected), per-site allowlists, and warnings for remote-access / tech-support scam pages.
- A mobile version.
