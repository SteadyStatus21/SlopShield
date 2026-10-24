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


## Shipped model (Oct 24, 2026): frontier ViT-S + social-video head, agreement rule

`models/model.onnx` = `Thermostatic/community-forensics-frontier-detector-2026-08` (ViT-S/16 @384, frozen) plus a
logistic head trained by `scripts/finetune_head.py` on 238 AI clips (Veo 3, Veo 3.1, Kling, Sora, other generators)
and 240 real AudioSet clips, clean and social-media-degraded. Output = `min(head, calibrated base + 1.79)`: it flags
only when head >= 0.6 AND base >= 0.2. Rebuild: `python scripts/finetune_head.py --exclude <test dir> --exclude dataset`.

Held-out (never trained on), threshold 0.6:

| Set | Old CommunityForensics | Frontier only | Head only | **Shipped (agreement)** |
|---|---|---|---|---|
| Modern AI video caught (60: Sora/Veo3/Veo3.1/Kling) | 14 | 55 | 56 | **53** |
| Real video flagged (60, AudioSet) | 0 | 11 | 3 | **1** |
| Real photos flagged (40) | 0 | 0 | 10 | **0** |
| Game screenshots flagged (30) | 0 | 0 | 8 | **0** |
| AI still images caught (40, older generators) | 34 | 29 | 30 | 24 |
| User's in-the-wild AI clips caught (7) | 0 | 3 | 4 | 3 |

The extension's WebGPU path was checked against native CPU outputs (identical to 3 decimals).

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
| `abacusai/Smaug-AI-Image-Detector` | 439M (EmbeddingGemma-2: Gemma 4 vision tower + 24-layer text stack, fully fine-tuned) | Apache-2.0 | **Tested** (1 centre 528 px crop, PyTorch fp32 CPU). AIGC 1.00 precision / 0.53 recall, 0/40 real and 0/30 games flagged. On *modern* generators only (SD 1.4/1.5/XL, Midjourney, DALL·E 2, Wukong) it caught 14/16 vs CommunityForensics' 12/16 (Midjourney 4/4 vs 2/4). Its card reports 0.973 AUC on OpenFake Reddit vs 0.744 for CommunityForensics. Cost: 10.2 s/image on 1 thread (7.6 s on 14), ~1.75 GB RAM fp32, 878 MB download; no ONNX export exists. Candidate for an optional GPU "high accuracy" mode. |
| `onnx-community/ai-image-detect-distilled-ONNX` | ~15M ViT | MIT | Card reports 72% real-world accuracy. |
| `Organika/sdxl-detector` | 87M Swin | CC-BY-NC-3.0 | Non-commercial license. |
| `dima806/ai_vs_real_image_detection` | 86M ViT | Apache-2.0 | Trained on CIFAKE (32 px images). |

## Quantization notes (from the earlier SigLIP model)

On SigLIP-B, dynamic int8 (weights + activations) drifted badly from fp32 (one verdict flipped by 0.98). 4-bit weight-only MatMulNBits stayed close and cut the size 343 → 57 MB, with no CPU speedup. For CommunityForensics, 4-bit cut recall from 0.85 to 0.78, so the extension ships fp32. FP4 isn't available in ONNX Runtime Web.
