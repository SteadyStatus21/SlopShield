"""Download a Hugging Face image classifier, export to ONNX, quantize (4-bit weights by default).

Writes models/model.onnx and models/model.json (preprocessing + label index the
extension reads at runtime, so swapping models needs no code change).

    pip install torch transformers onnx onnx-ir onnxruntime
    python scripts/fetch_model.py [--model Ateeqq/ai-vs-human-image-detector] [--bits 4|8|32]

--bits 8   dynamic int8 (weights + activations), runs everywhere.
--bits 4   4-bit block-quantized MatMul weights (MatMulNBits); other ops stay fp32.
           Smaller download/RAM; measure accuracy with scripts/eval.py before switching.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

# ViT-S/16 @384 trained on 2.7M images from 4,803 generators (Community Forensics, CVPR 2025). MIT.
DEFAULT_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
AI_LABEL = re.compile(r"^(ai|fake|deepfake|artificial|generated|synthetic|sd|ai[-_ ]?generated)", re.I)


class Logits(torch.nn.Module):
    def __init__(self, m, interpolate: bool):
        super().__init__()
        self.m = m
        self.interpolate = interpolate

    def forward(self, pixel_values):
        if self.interpolate:  # run a ViT at a non-native size by resizing its position embeddings
            return self.m(pixel_values=pixel_values, interpolate_pos_encoding=True).logits
        return self.m(pixel_values=pixel_values).logits


def quantize_4bit(src: Path, dst: Path, block_size: int, accuracy_level: int | None):
    import onnx

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    q = MatMulNBitsQuantizer(onnx.load(str(src)), block_size=block_size, is_symmetric=True, accuracy_level=accuracy_level)
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="models")
    ap.add_argument("--bits", type=int, choices=[4, 8, 32], default=32)
    ap.add_argument("--block-size", type=int, default=32, help="4-bit quantization block size")
    ap.add_argument("--accuracy-level", type=int, choices=[1, 2, 3, 4], default=None,
                    help="MatMulNBits compute type (4 = int8 compute, faster on CPU)")
    ap.add_argument("--keep-fp32", action="store_true", help="also keep models/model_fp32.onnx for comparisons")
    ap.add_argument("--size", type=int, default=None,
                    help="input size in px (multiple of the patch size); smaller = fewer tokens = faster. Default: model's native size")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out / "_export"
    tmp.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).eval()

    # Two preprocessing styles: squash to size×size, or resize shortest edge then center-crop.
    center_crop = bool(getattr(proc, "do_center_crop", False) and getattr(proc, "crop_size", None))
    if center_crop:
        native = proc.crop_size["height"]
        resize_shortest = proc.size["shortest_edge"]
    else:
        native = proc.size.get("height") or proc.size.get("shortest_edge") or 224
        resize_shortest = None
    size = args.size or native
    if resize_shortest:
        resize_shortest = round(resize_shortest * size / native)  # keep the crop ratio

    if cfg.num_labels == 1:
        # Single sigmoid output = P(fake) for the detectors we use; check the model card if swapping.
        labels, ai_idx = ["fake (sigmoid)"], 0
    else:
        labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
        ai_idx = next((i for i, l in enumerate(labels) if AI_LABEL.match(l)), None)
        if ai_idx is None:
            raise SystemExit(f"Can't tell which label means AI: {labels}. Edit models/model.json by hand.")

    fp32 = tmp / "model_fp32.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        Logits(model, interpolate=size != native),
        (dummy,),
        str(fp32),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: plain opset-17 graph that ORT Web handles
    )

    final = out / "model.onnx"
    if args.bits == 32:
        shutil.copy(fp32, final)
    elif args.bits == 8:
        quantize_dynamic(str(fp32), str(final), weight_type=QuantType.QInt8)
    else:
        quantize_4bit(fp32, final, args.block_size, args.accuracy_level)
    if args.keep_fp32:
        shutil.copy(fp32, out / "model_fp32.onnx")

    meta = {
        "name": args.model,
        "size": size,
        **({"resizeShortest": resize_shortest} if resize_shortest else {}),
        "mean": list(proc.image_mean),
        "std": list(proc.image_std),
        "aiIndex": ai_idx,
        "labels": labels,
        "inputName": "pixel_values",
        "bits": args.bits,
    }
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {final} ({final.stat().st_size / 1e6:.1f} MB), AI label = {labels[ai_idx]!r}")


if __name__ == "__main__":
    main()
