# Model

| | |
|---|---|
| Model | [`buildborderless/CommunityForensics-DeepfakeDet-ViT`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) |
| Paper | [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://huggingface.co/papers/2411.04125) (CVPR 2025) |
| Architecture | ViT-Small/16 (22M params, 12 layers, 384 hidden), fine-tuned from `timm/vit_small_patch16_384.augreg_in21k_ft_in1k` |
| Training data | 2.7M images from 4,803 generators |
| License | MIT |
| Output | One logit; `sigmoid(logit)` = P(AI-generated) |
| Input | Resize shortest edge to 440, center-crop 384×384, CLIP mean/std |
| Size | 87 MB fp32 ONNX (4-bit loses recall, so it isn't used; see below) |
| Source | `python scripts/fetch_model.py` (`torch.onnx.export`, opset 17). The preprocessing goes into `models/model.json`, which the extension and `scripts/eval.py` read. |


## Shipped model (Oct 25, 2026, 04:35): frontier ViT-S + MLP head, base-model veto

`models/model.onnx` = `Thermostatic/community-forensics-frontier-detector-2026-08` (ViT-S/16 @384, frozen) plus a
64-unit MLP head on [384-d features, base logit], trained by `scripts/train_from_labels.py` (via `train.bat`) on:

- ~2,000 clips: AI = Veo 3, Veo 3.1, Kling, Sora, other Rapidata generators, and 472 phone-style AI shorts
  (`ymo2017/ai-shorts-videos`); real = AudioSet YouTube clips and 208 modern Pexels stock videos. 3 frames each,
  clean and social-media-degraded.
- ~1,600 stills: frontier synthetic AI images; real photos, game screenshots, generated text/UI cards.
- The user's labels: each crop of an "AI" item counts x4, of a "Not AI" item x8.

Output = `min(head, calibrated base + shift)`: the head decides (>= 0.6), and the base model keeps a veto only when it
is almost certain a frame is real (< 0.2%). Every candidate is scored by `scripts/bench.py`; `train.bat` installs
(and pushes) it only if the objective (AI recall - 4 x false-alarm rate, each averaged over the held-out sets) wins.

Held-out (never trained on), threshold 0.6, `python scripts/bench.py`:

| Set | Old CommunityForensics | Frontier only | MLP head only | **Shipped** |
|---|---|---|---|---|
| Modern AI video caught (60: Sora/Veo3/Veo3.1/Kling) | 14 | 57 | 54 | **54** |
| Real video flagged (60, AudioSet) | 0 | 12 | 1 | **1** |
| Phone-style AI shorts caught (241, held out by concept) | — | 81 | 232 | **207** |
| Current-gen AI video caught (183: Seedance 1.5 Pro, Kling 2.5 Turbo; held out by prompt) | — | 163 | 163 | **162** |
| Modern real stock video flagged (198, Pexels) | — | 19 | 13 | **13** |
| AI still images caught (40, AIGC benchmark) | 34 | 30 | 21 | 21 |
| Real photos flagged (40) | 0 | 0 | 1 | **0** |
| Game screenshots flagged (30) | 0 | 0 | 0 | **0** |
| User's in-the-wild AI caught (14, held-out labels) | — | 6 | 8 | 6 |
| User's in-the-wild real flagged (10, held-out labels) | — | 3 | 2 | 2 |
| **All sets** | — | 337/538 AI, 34/338 real | 478/538, 17/338 | **450/538, 16/338** |

The 24 held-out feed items are too few to steer by, so recipe decisions used 5-fold cross-validation over all 115
labelled feed items (3 seeds, every item scored out-of-fold). Shipped recipe: 43.3/68 AI caught, 4.0/47 real flagged.
Previous recipe (no shorts/Pexels, labels x4/x4, gate 0.05): 37.3/68 and 4.0/47.

Also tried on Oct 25 and rejected (same cross-validation): weighting all labels 8x or 16x; averaging head and base
logits instead of `min`; thresholds 0.4-0.8; ensembles of 2-6 MLP seeds; multi-layer backbone features (CLS tokens of
blocks 3/6/9, mean patch token); cropping the bottom 12-30% of each tile to drop player controls; the low-quality
continuation of the base model as backbone; Smaug as a second opinion; training on the current-gen clips and 4 more
Pexels shards (+0.7/68 AI caught on the feed for +1.0/47 real flagged, so they only feed the benchmark); the extra
Pexels alone (held-out Pexels false alarms 13.7 -> 8.7/198, but -4 shorts, -5 current-gen clips and -0.6/68 on the
feed). None beat the shipped setup on the user's feed.

How we got here (pooled over the original 4 sets, 114 AI / 140 real): first head 77/114 AI with 7/140 real flagged;
border-trimmed crops 75/6; expanded data + logistic head 79/2; MLP head 84/2. Then on all 7 sets (538 AI / 338 real):
that model 354/28; + phone-style AI shorts, Pexels real video and x8 "Not AI" labels 391/13; base veto lowered from
0.05 to 0.002 450/16.

## How it was chosen (measured Oct 24, 2026)

Test sets:
- **AIGC**: 80 random images from [TheKernel01/AIGC-Detection-Benchmark](https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark) (40 real photos, 40 AI from 14 generators).
- **Games**: 30 random video-game screenshots from [badigadiii/game_screenshots_11k](https://huggingface.co/datasets/badigadiii/game_screenshots_11k). These aren't AI-generated, so every flag is a false positive. This is the failure users hit (racing gameplay flagged at 1.00).

Speed is onnxruntime-web 1.20.1 WASM, 1 thread, in Node (similar to the extension's offscreen document). Threshold 0.6.

| Model | Params | Input | ms/frame | AIGC precision | AIGC recall | Real photos flagged | Games flagged |
|---|---|---|---|---|---|---|---|
| `Ateeqq/ai-vs-human-image-detector` (SigLIP-B/16), 4-bit | 93M | 224 | ~2080 | 0.91 | 0.50 | 2/40 | **16/30** |
| same, 160 px | 93M | 160 | ~980 | 0.82 | 0.55 | 5/40 | 21/30 |
| **CommunityForensics ViT-S, fp32** | **22M** | **384** | **~1610** | **1.00** | **0.85** | **0/40** | **0/30** |
| CommunityForensics ViT-S, 4-bit | 22M | 384 | ~1720 | 1.00 | 0.78 | 0/40 | — |
| CommunityForensics ViT-S @320 | 22M | 320 | ~1090 | 1.00 | 0.73 | 0/40 | 0/30 |
| CommunityForensics ViT-S @288 | 22M | 288 | ~910 | 1.00 | 0.60 | 0/40 | 0/30 |
| CommunityForensics ViT-S @256 | 22M | 256 | ~620 | 1.00 | 0.43 | 0/40 | 0/30 |

Lower resolutions interpolate the position embeddings (`scripts/fetch_model.py --size N`). If speed matters more than recall, 320 px is the best trade.

Caveats:
- n is small (80 + 30).
- Community Forensics was trained on thousands of generators, possibly including older ones in the AIGC benchmark (ADM, GLIDE), so recall may be optimistic.
- Video generators (Sora, Veo, Kling) and video-frame compression aren't covered by these test sets.

## Runtime speed on the minimum-spec laptop (Intel i7-12700H, Iris Xe), measured Oct 24, 2026

In-browser (Chromium), onnxruntime-web 1.20.1, median of 8 runs:

| Model | WebGPU (Iris Xe) | WASM ×1 thread | WASM ×4 threads |
|---|---|---|---|
| CommunityForensics ViT-S @384 (shipped) | **160 ms** (batch of 3: 432 ms) | 1614 ms | 1402 ms |
| CommunityForensics ViT-S @320 | 103 ms | — | — |
| CommunityForensics ViT-S @256 | 132 ms | — | — |

WebGPU is ~10× faster than the CPU path and hits the 2 fps target, which is also the ceiling set by `captureVisibleTab`'s ~2 screenshots/sec. Hugging Face (searched Oct 24, 2026) has no small pretrained AI-image detector (MobileNet/EfficientViT/MobileViT-class). Those exist only as generic ImageNet backbones, so a faster CPU path would mean distilling this model into one.

## Other candidates (checked Oct 24, 2026)

| Model | Params | License | Notes |
|---|---|---|---|
| `Bombek1/ai-image-detector-siglip-dinov2` | ~700M (SigLIP2-So400m + DINOv2-L, LoRA) | MIT | Reports 97% cross-dataset accuracy, 9% FP on casual photos. 392 px through two large encoders is ~20× our compute: WebGPU-only, not tested. Ships custom `model.py`. |
| `haywoodsloan/ai-image-detector-deploy` | 195M (SwinV2) | Apache-2.0 | Not tested. |
| `abacusai/Smaug-AI-Image-Detector` | 439M (EmbeddingGemma-2: Gemma 4 vision tower + 24-layer text stack, fully fine-tuned) | Apache-2.0 | **Tested** (1 centre 528 px crop, PyTorch fp32 CPU). AIGC 1.00 precision / 0.53 recall, 0/40 real and 0/30 games flagged. On *modern* generators only (SD 1.4/1.5/XL, Midjourney, DALL·E 2, Wukong) it caught 14/16 vs CommunityForensics' 12/16 (Midjourney 4/4 vs 2/4). Its card reports 0.973 AUC on OpenFake Reddit vs 0.744 for CommunityForensics. Cost: 10.2 s/image on 1 thread (7.6 s on 14), ~1.75 GB RAM fp32, 878 MB download; no ONNX export exists. Converted to OpenVINO (1.7 s/crop fp16 on Iris Xe) and benchmarked as a second opinion on Oct 25: on the user's held-out feed it ranks AI vs. real worse than chance (AUC 0.45 vs. 0.69 for the shipped model), so it is **not used**. |
| `Thermostatic/community-forensics-low-quality-detector-2026-08` | 22M (same ViT-S, continued training of our base with downscale/recompression augmentation) | MIT | **Tested Oct 25** as a drop-in backbone. On the 115 items labelled from the user's feed, its base score was *worse* (AUC 0.626 vs 0.658; 29/68 vs 35/68 AI caught), and it flagged 6/30 game screenshots (base model: 0/30). Not used. |
| `onnx-community/ai-image-detect-distilled-ONNX` | ~15M ViT | MIT | Card reports 72% real-world accuracy. |
| `Organika/sdxl-detector` | 87M Swin | CC-BY-NC-3.0 | Non-commercial license. |
| `dima806/ai_vs_real_image_detection` | 86M ViT | Apache-2.0 | Trained on CIFAKE (32 px images). |

## Quantization notes (from the earlier SigLIP model)

On SigLIP-B, dynamic int8 (weights + activations) drifted badly from fp32 (one verdict flipped by 0.98). 4-bit weight-only MatMulNBits stayed close and cut the size 343 → 57 MB, with no CPU speedup. For CommunityForensics, 4-bit cut recall from 0.85 to 0.78, so the extension ships fp32. FP4 isn't available in ONNX Runtime Web.
