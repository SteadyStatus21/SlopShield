"""Adapt the detector to reposted social-media AI video by retraining its decision head.

Backbone: Thermostatic/community-forensics-frontier-detector-2026-08 (ViT-S/16 @384, frozen).
New head: logistic regression on [384-d pooled features, original logit], trained on frames from
modern AI video (Veo 3, Veo 3.1, Kling, Sora, other Rapidata generators) vs real video (AudioSet),
each frame used clean and "social-media degraded" (downscale, JPEG re-encode, vertical crop,
caption text). Exports models/model.onnx (fp32) + models/model.json for the extension.

    python scripts/finetune_head.py --exclude <dir-with-test-videos> [--per-source 60] [--real 300]

Test videos (and dataset/) must be excluded so the evaluation stays honest.
"""

import argparse
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

REPO = "Thermostatic/community-forensics-frontier-detector-2026-08"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE, SHORT = 384, 440
BIAS = -0.7403357915937764  # published calibration of the base model

AI_SOURCES = [  # dataset, path filter
    ("Rapidata/text-2-video-human-preferences-veo3", lambda p: True),  # veo3 + other generators, all AI
    ("Rapidata/text-2-video-human-preferences-veo3.1", lambda p: True),
    ("bitmind/klingai-videos", lambda p: True),
    ("Rapidata/sora-video-generation-style-likert-scoring", lambda p: p.endswith("_sora.mp4")),
]
REAL_SOURCES = [("chikuwa-AI/AudioSet_balanced_video", lambda p: True)]


def list_videos(ds, keep):
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ds}/tree/main?recursive=1"))
    return [f["path"] for f in tree if f.get("type") == "file" and f["path"].endswith(".mp4") and keep(f["path"])
            and (f.get("size") or 0) < 15_000_000]


def download(jobs, out):
    out.mkdir(parents=True, exist_ok=True)

    def one(job):
        url, dst = job
        if not dst.exists():
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:
                print("fail", url, e, flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, jobs))


def frames(path, n=3):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def degrade(img, rng):
    """Make a frame look like a reposted social-media video frame."""
    w, h = img.size
    if rng.random() < 0.5:  # crop to vertical 9:16 (reels/shorts reposts of landscape video)
        cw = min(w, int(h * 9 / 16))
        x = rng.randint(0, w - cw)
        img = img.crop((x, 0, x + cw, h))
    s = rng.uniform(0.35, 0.8)  # downscale then upscale (platform transcode + display scaling)
    small = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.BILINEAR)
    if rng.random() < 0.5:  # burned-in caption text
        d = ImageDraw.Draw(small)
        txt = rng.choice(["wait for it 😳", "POV: you see this", "no way this is real", "LIKE & FOLLOW", "Part 2"])
        y = int(small.height * rng.uniform(0.1, 0.8))
        d.text((10, y), txt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0),
               font=ImageFont.load_default(size=max(12, small.width // 14)))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=rng.randint(35, 75))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize(img.size, Image.BILINEAR)


def to_tensor(img):
    k = SHORT / min(img.size)
    img = img.resize((round(img.width * k), round(img.height * k)), Image.BICUBIC)
    l, t = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    x = (np.asarray(img.crop((l, t, l + SIZE, t + SIZE)), np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))


# Agreement rule. The new head alone overflags real photos and game captures (10/40 and 8/30 on
# held-out sets) while the base model alone overflags low-quality real video (11/60); flagging
# only when both agree had 1 false positive across 160 held-out real items. Encoded as one logit:
#   out = min(head_logit, base_logit + BIAS + AGREE_SHIFT)
# which reaches the default threshold (0.6) exactly when head ≥ 0.6 AND calibrated base ≥ 0.2.
AGREE_HEAD_P, AGREE_BASE_P = 0.6, 0.05  # gate lowered from 0.2 on Oct 24 (held-out sweep: +2/12 AI, +1/30 games)


def agree_shift(base_p: float = AGREE_BASE_P) -> float:
    """Logit shift so min(head, base + BIAS + shift) reaches AGREE_HEAD_P exactly when base == base_p."""
    return float(np.log(AGREE_HEAD_P / (1 - AGREE_HEAD_P)) - np.log(base_p / (1 - base_p)))


AGREE_SHIFT = agree_shift()


class Detector(torch.nn.Module):
    """Frozen ViT + new head, combined with the base model by the agreement rule (exported to ONNX)."""

    def __init__(self, vit, w, v, b, agree=True, base_p=AGREE_BASE_P):
        super().__init__()
        self.vit = vit
        self.agree = agree
        self.shift = agree_shift(base_p)
        self.w = torch.nn.Parameter(torch.tensor(np.asarray(w).reshape(-1), dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(float(v), dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.tensor(float(b), dtype=torch.float32))

    def forward(self, pixel_values):
        f = self.vit.forward_head(self.vit.forward_features(pixel_values), pre_logits=True)
        base = self.vit.head(f)
        head = f @ self.w[:, None] + base * self.v + self.b
        if not self.agree:
            return head
        return torch.minimum(head, base + (BIAS + self.shift))


def load_vit(weights):
    vit = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    weights = Path(weights)
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/main/model.safetensors", weights)
    vit.load_state_dict(load_file(str(weights)))
    return vit.eval()


def export(vit, w, v, b, out, trained_on, agree=True, base_p=AGREE_BASE_P):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "head.npz", w=np.asarray(w).reshape(-1), v=v, b=b, agree=agree, base_p=base_p)  # lets later runs re-export without retraining
    torch.onnx.export(Detector(vit, w, v, b, agree, base_p).eval(), (torch.randn(1, 3, SIZE, SIZE),), str(out / "model.onnx"),
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    meta = {"name": f"{REPO} + SlopShield social-video head (agreement rule)", "size": SIZE, "resizeShortest": SHORT,
            "mean": MEAN, "std": STD, "aiIndex": 0, "labels": ["ai (sigmoid)"], "inputName": "pixel_values",
            "rule": (f"min(head, base{BIAS:+.3f}{agree_shift(base_p):+.3f}): head>={AGREE_HEAD_P} AND base>={base_p}"
                     if agree else "head only"),
            "trainedOn": trained_on}
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'model.onnx'} and model.json", flush=True)


def head_from_onnx(path):
    """Recover (w, v, b) from a model exported by this script (for re-exporting without retraining)."""
    import onnx
    from onnx import numpy_helper

    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer}
    w = next(v for k, v in inits.items() if v.shape in ((384, 1), (384,)) and not k.startswith("vit."))
    return w.reshape(-1), float(inits["v"]), float(inits["b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", action="append", default=[], help="dirs whose video stems must not be used for training")
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--real", type=int, default=240)
    ap.add_argument("--data", default="models/_train")
    ap.add_argument("--out", default="models")
    ap.add_argument("--weights", default="models/_ref/frontier/model.safetensors", help="base model weights (downloaded if missing)")
    ap.add_argument("--reexport-from", default=None, help="skip training: take the head from this exported ONNX")
    args = ap.parse_args()
    rng = random.Random(0)

    if args.reexport_from:
        w, v, b = head_from_onnx(args.reexport_from)
        export(load_vit(args.weights), w, v, b, args.out, {"reexportedFrom": str(args.reexport_from)})
        return

    excluded = {p.stem for d in args.exclude for p in Path(d).rglob("*.mp4")}
    excluded |= {s.split("_", 2)[-1] for s in excluded}  # test files are saved as <label>_<src>_<stem>

    data = Path(args.data)
    jobs, label_of = [], {}
    for sources, label, n in ((AI_SOURCES, 1, args.per_source), (REAL_SOURCES, 0, args.real)):
        for ds, keep in sources:
            vids = [p for p in list_videos(ds, keep) if Path(p).stem not in excluded]
            for p in rng.sample(vids, min(n, len(vids))):
                dst = data / f"{label}_{ds.split('/')[1][:20]}_{Path(p).stem}.mp4"
                jobs.append((f"https://huggingface.co/datasets/{ds}/resolve/main/{p}", dst))
                label_of[dst] = label
    print(f"training videos: {sum(label_of.values())} AI, {len(label_of) - sum(label_of.values())} real", flush=True)
    download(jobs, data)

    vit = load_vit(args.weights)

    # Features are cached per video, so retraining after adding clips only processes new ones.
    cache_path = data / "features_cache.npz"
    cache = dict(np.load(cache_path, allow_pickle=True)["cache"].item()) if cache_path.exists() else {}
    feats, base, ys, groups = [], [], [], []
    with torch.no_grad():
        for i, (path, y) in enumerate(label_of.items()):
            if not path.exists():
                continue
            if path.name not in cache:
                rows = []
                for fr in frames(path):
                    for view in (fr, degrade(fr, rng)):
                        f = vit.forward_head(vit.forward_features(to_tensor(view)[None]), pre_logits=True)
                        rows.append((f[0].numpy(), float(vit.head(f)[0, 0])))
                cache[path.name] = rows
            for f, bl in cache[path.name]:
                feats.append(f)
                base.append(bl)
                ys.append(y)
                groups.append(i)
            if (i + 1) % 25 == 0:
                print(f"  features: {i + 1}/{len(label_of)} videos", flush=True)
                np.savez(cache_path, cache=np.array(cache, dtype=object))
    np.savez(cache_path, cache=np.array(cache, dtype=object))

    X = np.hstack([np.array(feats), np.array(base)[:, None]])
    y, g = np.array(ys), np.array(groups)
    clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")
    # Grouped CV (frames of one video never split across folds) to report honest training accuracy.
    pred = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
    base_pred = 1 / (1 + np.exp(-(np.array(base) + BIAS)))
    for name, p in (("base model", base_pred), ("new head (CV)", pred)):
        tp = ((p >= 0.5) & (y == 1)).sum(); fp = ((p >= 0.5) & (y == 0)).sum()
        print(f"{name:14s} recall={tp / (y == 1).sum():.3f}  false-positive rate={fp / (y == 0).sum():.3f}", flush=True)
    clf.fit(X, y)

    n_ai = int(sum(label_of.values()))
    export(vit, clf.coef_[0][:-1], clf.coef_[0][-1], clf.intercept_[0], args.out,
           {"aiVideos": n_ai, "realVideos": len(label_of) - n_ai})

if __name__ == "__main__":
    main()
