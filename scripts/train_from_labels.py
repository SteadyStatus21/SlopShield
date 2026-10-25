"""Retrain the detector head from your labels (dataset/ai, dataset/real) — label, run, repeat.

  1. Collect labelled items: dataset/<ai|real>/<item>/tile*.png (saved by the extension's
     "Yes, AI" / "Not AI" buttons) and any videos/images under dataset/<ai|real>/.
  2. Hold out ~1 in 5 items (fixed by item name) so every run reports honest before/after numbers.
  3. Extract features on the GPU with OpenVINO (cached: only new items cost time).
  4. Train a new head on the Hugging Face set (models/_train) + your training items (weighted up).
  5. Print a held-out scorecard (old vs new, both decision rules) and per-item mistakes.
  6. Back up the current model to models/_ref/backups/<time>/ and install the new one.

    python scripts/train_from_labels.py [--rule agree|head] [--user-weight 4] [--dry-run]

Then rebuild the extension (npm run build) and restart the engine. train.bat does all of it.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_head as fh  # noqa: E402  (frames, degrade, to_tensor, load_vit, export, constants)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
MODELS = ROOT / "models"
TRAIN_DIR = MODELS / "_train"
FEATURES_ONNX = MODELS / "_ref" / "features.onnx"
CACHE = MODELS / "_train" / "features_ov_v2.npz"  # v2: crops border-trimmed like the extension
IMG = {".png", ".jpg", ".jpeg", ".webp"}
VID = {".mp4", ".webm", ".mov", ".mkv"}
THRESHOLD = 0.6
MAX_TILES = 3


def say(msg=""):
    print(msg, flush=True)


# ---------- features (OpenVINO on the GPU) ----------

def build_feature_model(vit):
    """ONNX of the frozen backbone returning [384 pooled features, base logit] (built once)."""
    import torch

    class Feat(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, pixel_values):
            f = self.v.forward_head(self.v.forward_features(pixel_values), pre_logits=True)
            return torch.cat([f, self.v.head(f)], dim=1)

    FEATURES_ONNX.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Feat(vit).eval(), (torch.randn(1, 3, fh.SIZE, fh.SIZE),), str(FEATURES_ONNX),
                      input_names=["pixel_values"], output_names=["features"], opset_version=17, dynamo=False)


class Extractor:
    def __init__(self):
        import openvino as ov

        core = ov.Core()
        dev = "GPU" if "GPU" in core.available_devices else "CPU"
        # fp32 so features match the CPU-extracted ones the head is trained against.
        self.req = core.compile_model(str(FEATURES_ONNX), dev, {"INFERENCE_PRECISION_HINT": "f32"}).create_infer_request()
        self.device = dev
        self.mean = np.array(fh.MEAN, np.float32)
        self.std = np.array(fh.STD, np.float32)

    def __call__(self, tile: Image.Image):
        x = (np.asarray(tile.convert("RGB").resize((fh.SIZE, fh.SIZE), Image.BICUBIC), np.float32) / 255 - self.mean) / self.std
        out = self.req.infer({0: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        return out[:-1].astype(np.float32), float(out[-1])


def trim_borders(img: Image.Image, flat_std: float = 6.0) -> Image.Image:
    """Drop flat rows/columns at the edges (letterbox/pillarbox/padding). Mirrors capture.ts trimBorders()."""
    w, h = img.size
    n = 64
    cw, ch = max(8, round(n * w / max(w, h))), max(8, round(n * h / max(w, h)))
    a = np.asarray(img.convert("RGB").resize((cw, ch), Image.BILINEAR), np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    colf, rowf = y.std(0) < flat_std, y.std(1) < flat_std
    l, r, t, b = 0, cw - 1, 0, ch - 1
    while l < r and colf[l]: l += 1
    while r > l and colf[r]: r -= 1
    while t < b and rowf[t]: t += 1
    while b > t and rowf[b]: b -= 1
    kw, kh = (r - l + 1) / cw, (b - t + 1) / ch
    if kw * kh < 0.25 or (kw > 0.97 and kh > 0.97):
        return img
    return img.crop((round(l / cw * w), round(t / ch * h), round(l / cw * w) + round(kw * w), round(t / ch * h) + round(kh * h)))


def tiles(img: Image.Image, trim: bool = True):
    """Same square tiles the extension scores (background/capture.ts trimBorders() + tiles())."""
    if trim:
        img = trim_borders(img)
    w, h = img.size
    short, long = min(w, h), max(w, h)
    side = round(short * fh.SIZE / fh.SHORT)
    n = min(MAX_TILES, max(1, math.ceil(long / side - 0.15)))
    out = []
    for i in range(n):
        t = 0.5 if n == 1 else i / (n - 1)
        along, across = round((long - side) * t), round((short - side) / 2)
        x, y = (along, across) if w >= h else (across, along)
        out.append(img.crop((x, y, x + side, y + side)))
    return out


# ---------- data ----------

def saved_tile(path):
    """A tile saved by the extension; older ones may still contain padding bars, so trim + retile those."""
    img = Image.open(path).convert("RGB")
    trimmed = trim_borders(img)
    return [img] if trimmed is img else tiles(trimmed, trim=False)


def user_items():
    """{item_id: (label 0/1, [(cache_key, loader)])} from dataset/ai and dataset/real."""
    items = {}
    for lab_name, y in (("ai", 1), ("real", 0)):
        root = DATASET / lab_name
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and any(p.glob("tile*.png")):  # saved by the extension: tiles are model-ready
                items[f"{lab_name}/{p.name}"] = (y, [(f"u:{t.relative_to(DATASET)}", lambda t=t: saved_tile(t))
                                                     for t in sorted(p.glob("tile*.png"))])
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.suffix.lower() in VID:  # 6 frames, tiled like the extension
                        items[f"{lab_name}/{f.stem}"] = (y, [(f"u:{f.relative_to(DATASET)}",
                                                              lambda f=f: [t for fr in fh.frames(f, 6) for t in tiles(fr)])])
                    elif f.suffix.lower() in IMG:
                        items[f"{lab_name}/{f.stem}"] = (y, [(f"u:{f.relative_to(DATASET)}", lambda f=f: tiles(Image.open(f).convert("RGB")))])
            elif p.suffix.lower() in IMG | VID:
                items[f"{lab_name}/{p.stem}"] = (y, [(f"u:{p.relative_to(DATASET)}", lambda p=p: tiles(Image.open(p).convert("RGB")))])
    return items


TRAIN_IMG = MODELS / "_train_img"


def public_training(extract, cache, verbose=True):
    """Features for the public training data, extracting (and caching) anything new.

    Clips in models/_train (1_* = AI, 0_* = real): 3 frames, clean + social-media-degraded view, centre crop.
    Stills in models/_train_img/<ai|real>/: tiled like the extension, clean + degraded view.
    Returns F (n,384), B (n,), y (n,), source name per crop.
    """
    rng = random.Random(0)
    F, B, Y, SRC = [], [], [], []
    vids = sorted(TRAIN_DIR.glob("*.mp4"))
    stills = [(p, 1 if p.parent.name == "ai" else 0) for p in sorted(TRAIN_IMG.glob("*/*")) if p.suffix.lower() in IMG]
    todo = sum(f"hf:{v.name}" not in cache for v in vids) + sum(f"img:{p.parent.name}/{p.name}" not in cache for p, _ in stills)
    if verbose and todo:
        say(f"      extracting public training data: {todo} new items (cached afterwards)")
    done = 0
    for v in vids:
        key = f"hf:{v.name}"
        if key not in cache:
            rows = []
            for fr in fh.frames(v):
                fr = trim_borders(fr)
                for view in (fr, fh.degrade(fr, rng)):
                    k = fh.SHORT / min(view.size)
                    im = view.resize((round(view.width * k), round(view.height * k)), Image.BICUBIC)
                    l, t = (im.width - fh.SIZE) // 2, (im.height - fh.SIZE) // 2
                    rows.append(extract(im.crop((l, t, l + fh.SIZE, t + fh.SIZE))))
            cache[key] = rows
            done += 1
            if done % 50 == 0:
                save_cache(cache)
                if verbose:
                    say(f"      public data: {done}/{todo}")
        for f, b in cache[key]:
            F.append(f); B.append(b); Y.append(1 if v.name.startswith("1_") else 0); SRC.append("clip")
    for p, y in stills:
        key = f"img:{p.parent.name}/{p.name}"
        if key not in cache:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                cache[key] = []
                continue
            cache[key] = [extract(t) for view in (img, fh.degrade(img, rng)) for t in tiles(view)]
            done += 1
            if done % 50 == 0:
                save_cache(cache)
                if verbose:
                    say(f"      public data: {done}/{todo}")
        for f, b in cache[key]:
            F.append(f); B.append(b); Y.append(y); SRC.append("still:" + p.name.split("_")[0])
    save_cache(cache)
    return F, B, Y, SRC


def is_holdout(item_id: str) -> bool:
    return int(hashlib.sha1(item_id.encode()).hexdigest(), 16) % 5 == 0


def load_cache():
    return dict(np.load(CACHE, allow_pickle=True)["cache"].item()) if CACHE.exists() else {}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, cache=np.array(cache, dtype=object))


# ---------- scoring ----------

def logits(F, B, head, rule, base_p=fh.AGREE_BASE_P):
    w, v, b = head
    h = F @ w + B * v + b
    if rule == "agree":
        h = np.minimum(h, B + fh.BIAS + fh.agree_shift(base_p))
    return h


def item_probs(rows, head, rule, base_p=fh.AGREE_BASE_P):
    """Per item: mean tile logit → probability (as the extension does)."""
    out = {}
    for item, (y, F, B) in rows.items():
        out[item] = (y, float(1 / (1 + np.exp(-logits(F, B, head, rule, base_p).mean()))))
    return out


def scorecard(name, probs):
    ai = [p for y, p in probs.values() if y == 1]
    real = [p for y, p in probs.values() if y == 0]
    caught = sum(p >= THRESHOLD for p in ai)
    fp = sum(p >= THRESHOLD for p in real)
    return f"{name:34s} AI caught {caught:3d}/{len(ai):<3d}  real wrongly flagged {fp:3d}/{len(real):<3d}"


def main():
    ap = argparse.ArgumentParser(description="Retrain the head from your labels")
    ap.add_argument("--rule", choices=["agree", "head"], default="agree",
                    help="agree: flag only if new head AND base model agree (fewer false alarms); head: new head alone")
    ap.add_argument("--user-weight", type=float, default=4.0, help="how much more your labels count than the HF set")
    ap.add_argument("--dry-run", action="store_true", help="train and report, but don't install")
    ap.add_argument("--base-gate", type=float, default=fh.AGREE_BASE_P,
                    help="agree rule: base model must give at least this (lower = catches more, more false alarms)")
    ap.add_argument("--reexport", action="store_true", help="skip training: re-export the installed head (e.g. new --base-gate)")
    args = ap.parse_args()
    t_start = time.time()

    say("=" * 72)
    say("SlopShield: retrain from your labels")
    say("=" * 72)

    vit = fh.load_vit(str(MODELS / "_ref" / "frontier" / "model.safetensors"))
    if args.reexport:
        h = np.load(MODELS / "head.npz")
        backup = MODELS / "_ref" / "backups" / time.strftime("%Y%m%d-%H%M%S")
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("model.onnx", "model.json", "head.npz"):
            if (MODELS / name).exists():
                shutil.copy2(MODELS / name, backup / name)
        say(f"re-exporting installed head with rule={args.rule}, base gate={args.base_gate} (backup: {backup.relative_to(ROOT)})")
        fh.export(vit, h["w"], float(h["v"]), float(h["b"]), MODELS, {"reexported": time.strftime("%Y-%m-%d %H:%M")},
                  agree=args.rule == "agree", base_p=args.base_gate)
        say("Done. Rebuild (npm run build), restart the engine, reload the extension.")
        return
    if not FEATURES_ONNX.exists():
        say("[1/6] building GPU feature extractor (one-time) ...")
        build_feature_model(vit)
    extract = Extractor()
    say(f"[1/6] feature extractor ready on {extract.device}")

    # --- your labels ---
    items = user_items()
    hold = {k for k in items if is_holdout(k)}
    n_ai = sum(1 for y, _ in items.values() if y == 1)
    say(f"[2/6] your labels: {len(items)} items ({n_ai} AI, {len(items) - n_ai} real); "
        f"held out for testing: {len(hold)} ({sum(items[k][0] for k in hold)} AI, {sum(1 - items[k][0] for k in hold)} real)")
    if len(hold) < 6:
        say("      note: the held-out set is small, so its numbers are noisy. Keep labelling.")

    cache = load_cache()
    new = 0
    user_rows = {}
    for i, (item, (y, sources)) in enumerate(items.items(), 1):
        F, B = [], []
        for key, load in sources:
            if key not in cache:
                cache[key] = [extract(t) for t in load()]
                new += 1
            for f, b in cache[key]:
                F.append(f)
                B.append(b)
        if F:
            user_rows[item] = (y, np.array(F), np.array(B))
        if i % 25 == 0:
            say(f"      features: {i}/{len(items)} items")
    say(f"[3/6] features for your items: {sum(len(r[2]) for r in user_rows.values())} crops ({new} new, rest cached)")

    # --- public training set (models/_train clips + models/_train_img stills) ---
    hf_F, hf_B, hf_y, _src = public_training(extract, cache)
    hf_videos = sorted(TRAIN_DIR.glob("*.mp4"))
    say(f"      public training set: {len(hf_videos)} clips + {len(list(TRAIN_IMG.glob('*/*')))} stills, {len(hf_y)} crops ({sum(hf_y)} AI)")

    # --- train ---
    from sklearn.linear_model import LogisticRegression

    tr_items = [k for k in user_rows if k not in hold]
    X = [np.hstack([np.array(hf_F), np.array(hf_B)[:, None]])]
    Y = [np.array(hf_y)]
    W = [np.ones(len(hf_y))]
    for k in tr_items:
        y, F, B = user_rows[k]
        X.append(np.hstack([F, B[:, None]]))
        Y.append(np.full(len(B), y))
        W.append(np.full(len(B), args.user_weight / max(1, len(B)) * 6))  # each item counts like ~6 crops x weight
    X, Y, W = np.vstack(X), np.concatenate(Y), np.concatenate(W)
    say(f"[4/6] training head on {len(Y)} crops ({len(tr_items)} of your items, weight x{args.user_weight:g}) ...")
    clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(X, Y, sample_weight=W)
    new_head = (clf.coef_[0][:-1], float(clf.coef_[0][-1]), float(clf.intercept_[0]))

    old_path = MODELS / "head.npz" if (MODELS / "head.npz").exists() else MODELS / "_ref" / "agree" / "head.npz"
    o = np.load(old_path)
    old_head = (o["w"], float(o["v"]), float(o["b"]))
    old_rule = "agree" if ("agree" not in o or bool(o["agree"])) else "head"
    old_gate = float(o["base_p"]) if "base_p" in o else 0.2

    # --- scorecard ---
    hold_rows = {k: user_rows[k] for k in hold if k in user_rows}
    train_rows = {k: user_rows[k] for k in tr_items}
    say("[5/6] scorecard (threshold 0.6, per item)")
    say("  HELD-OUT items (never trained on) -- the honest numbers:")
    say("    " + scorecard(f"current model ({old_rule}, gate {old_gate:g})", item_probs(hold_rows, old_head, old_rule, old_gate)))
    for gate in (0.2, 0.1, 0.05, 0.02):
        mark = " <- installs" if args.rule == "agree" and abs(gate - args.base_gate) < 1e-9 else ""
        say("    " + scorecard(f"new head, agree, gate {gate:g}{mark}", item_probs(hold_rows, new_head, "agree", gate)))
    say("    " + scorecard("new head alone" + (" <- installs" if args.rule == "head" else ""), item_probs(hold_rows, new_head, "head")))
    say("    (head alone also flagged 9/40 real photos + 6/30 game screenshots in an earlier check; the gate prevents that)")
    say("  YOUR TRAINING items (in-sample, optimistic; shows it learned your labels):")
    say("    " + scorecard(f"current model ({old_rule}, gate {old_gate:g})", item_probs(train_rows, old_head, old_rule, old_gate)))
    say("    " + scorecard(f"new head, rule={args.rule}, gate {args.base_gate:g}", item_probs(train_rows, new_head, args.rule, args.base_gate)))
    wrong = [(k, y, p) for k, (y, p) in item_probs(hold_rows, new_head, args.rule, args.base_gate).items() if (p >= THRESHOLD) != (y == 1)]
    if wrong:
        say("  held-out mistakes of the new model:")
        for k, y, p in sorted(wrong):
            say(f"    {'AI  ' if y else 'REAL'} scored {p:.2f}  dataset/{k}")

    if args.dry_run:
        say("[6/6] dry run: nothing installed.")
        return
    # --- install ---
    backup = MODELS / "_ref" / "backups" / time.strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "model.json", "head.npz"):
        if (MODELS / name).exists():
            shutil.copy2(MODELS / name, backup / name)
    say(f"[6/6] backed up current model to {backup.relative_to(ROOT)}; exporting new model ...")
    fh.export(vit, *new_head, MODELS, {"hfClips": len(hf_videos), "yourItems": len(tr_items), "heldOut": len(hold_rows),
                                       "userWeight": args.user_weight, "trainedAt": time.strftime("%Y-%m-%d %H:%M")},
              agree=args.rule == "agree", base_p=args.base_gate)
    say(f"Done in {time.time() - t_start:.0f}s. Installed models/model.onnx ({args.rule} rule).")
    say("Undo: copy the files from the backup folder above back into models/.")


if __name__ == "__main__":
    main()
