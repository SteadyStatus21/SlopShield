"""Accuracy of models/model.onnx on labeled frames → precision/recall/F1 + confusion matrix.

Labels come from the file name prefix (ai_* / real_*), a parent folder named ai/ or real/, or --label.
Videos (.mp4/.webm/.mov) are sampled into frames first (needs opencv-python).

    pip install onnxruntime-directml pillow numpy opencv-python   # or onnxruntime (CPU only)
    python scripts/eval.py demo/videos --frames 5 --threshold 0.6
"""

import argparse
import math
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

IMG = {".jpg", ".jpeg", ".png", ".webp"}
VID = {".mp4", ".webm", ".mov", ".mkv"}


def label_of(p: Path):
    for part in [p.name.lower(), *[x.lower() for x in p.parts[::-1]]]:
        if part.startswith("ai"):
            return 1
        if part.startswith("real"):
            return 0
    return None


def video_frames(path: Path, n: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


MAX_TILES = 3  # keep in sync with VISION.maxTiles in src/config.ts


def tiles(img: Image.Image, meta):
    """Square tiles covering the image, exactly like the extension (background/capture.ts tiles())."""
    w, h = img.size
    short, long = min(w, h), max(w, h)
    side = round(short * meta["size"] / meta["resizeShortest"]) if meta.get("resizeShortest") else short
    n = min(MAX_TILES, max(1, math.ceil(long / side - 0.15)))
    out = []
    for i in range(n):
        t = 0.5 if n == 1 else i / (n - 1)
        along, across = round((long - side) * t), round((short - side) / 2)
        x, y = (along, across) if w >= h else (across, along)
        out.append(img.crop((x, y, x + side, y + side)))
    return out


def preprocess(tile: Image.Image, meta):
    s = meta["size"]
    x = np.asarray(tile.convert("RGB").resize((s, s), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def tile_logit(sess, meta, tile):
    out = sess.run(None, {sess.get_inputs()[0].name: preprocess(tile, meta)})[0][0]
    if out.shape[0] == 1:
        return float(out[0])
    e = np.exp(out - out.max())
    p = min(1 - 1e-6, max(1e-6, float(e[meta["aiIndex"]] / e.sum())))
    return math.log(p / (1 - p))


def p_ai(sess, meta, img):
    """Mean tile logit (+ calibration bias) → probability, as in background/sw.ts + offscreen/infer.ts."""
    z = np.mean([tile_logit(sess, meta, t) for t in tiles(img.convert("RGB"), meta)]) + meta.get("logitBias", 0.0)
    return float(1 / (1 + np.exp(-z)))


def trimmed_mean(xs, frac=0.25):
    xs = sorted(xs)
    k = int(len(xs) * frac)
    kept = xs[k : len(xs) - k] if len(xs) - 2 * k > 0 else xs
    return sum(kept) / len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default="models/model.onnx")
    ap.add_argument("--meta", default="models/model.json")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--frames", type=int, default=5, help="frames sampled per video")
    ap.add_argument("--gpu", action="store_true", help="use DirectML (check results against CPU first)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the model (default 4 keeps fans down)")
    ap.add_argument("--label", choices=["ai", "real"], default=None,
                    help="treat every file in the folder as this label (for unlabeled folders such as dataset/)")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    # CPU with a thread cap by default, so evals don't max out a laptop. DirectML (--gpu) is opt-in:
    # on an Intel Iris Xe (driver as of Oct 2026) it returned a constant for every ViT we tried
    # and was barely faster (127 vs 136 ms/frame); verify outputs against --cpu before trusting it.
    cv2.setNumThreads(2)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    gpu = args.gpu and "DmlExecutionProvider" in ort.get_available_providers()
    if gpu:
        so.enable_mem_pattern = False  # required by the DirectML provider
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, so, providers=providers)
    print(f"  running on {'GPU (DirectML)' if gpu else f'CPU ({args.threads} threads)'}", file=sys.stderr, flush=True)

    files = []
    for p in sorted(Path(args.folder).rglob("*")):
        if p.suffix.lower() not in IMG | VID:
            continue
        y = {"ai": 1, "real": 0}[args.label] if args.label else label_of(p.relative_to(args.folder))
        if y is not None:
            files.append((p, y))

    rows = []  # (name, label, score)
    for i, (p, y) in enumerate(files, 1):
        if p.suffix.lower() in VID:
            frames = video_frames(p, args.frames)
            if not frames:
                continue
            score = trimmed_mean([p_ai(sess, meta, f) for f in frames])  # same aggregation as the extension
        else:
            score = p_ai(sess, meta, Image.open(p))
        rows.append((p.name, y, score))
        if i % 10 == 0 or i == len(files):
            print(f"  scored {i}/{len(files)}", file=sys.stderr, flush=True)

    if not rows:
        raise SystemExit("No labeled files found (name them ai_* / real_* or put them in ai/ and real/).")

    tp = sum(1 for _, y, s in rows if y == 1 and s >= args.threshold)
    fp = sum(1 for _, y, s in rows if y == 0 and s >= args.threshold)
    fn = sum(1 for _, y, s in rows if y == 1 and s < args.threshold)
    tn = sum(1 for _, y, s in rows if y == 0 and s < args.threshold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    for name, y, s in rows:
        mark = "ok " if (s >= args.threshold) == (y == 1) else "ERR"
        print(f"{mark} {'AI  ' if y else 'REAL'} {s:.3f}  {name}")
    print(f"\nn={len(rows)}  threshold={args.threshold}")
    print(f"precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  accuracy={(tp + tn) / len(rows):.3f}")
    print("confusion matrix (rows = truth, cols = predicted):")
    print("            pred_real  pred_ai")
    print(f"  real      {tn:9d}  {fp:7d}")
    print(f"  ai        {fn:9d}  {tp:7d}")


if __name__ == "__main__":
    main()
