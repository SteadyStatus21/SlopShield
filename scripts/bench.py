"""Benchmark a detector head on every held-out set, scored exactly like the extension.

Sets (none are ever trained on):
  user-holdout  ~1 in 5 of your labelled items in dataset/ (same split as train_from_labels.py)
  video         bench/vidtest: 60 modern AI videos (Veo 3, Veo 3.1, Kling, Sora) + 60 real YouTube clips
  images        bench/aigc: 40 AI + 40 real still images (AIGC-Detection-Benchmark sample)
  games         bench/games: 30 real video-game screenshots (all should be "not AI")
  shorts        bench/shorts/ai: phone-style AI shorts (ymo2017/ai-shorts-videos), 1/3 of concepts held out
  pexels        bench/pexels/real: modern real stock video (Pexels), 1/3 of clips held out
  newgen        bench/newgen/ai: current-generation AI video (Seedance 1.5 Pro, Kling 2.5 Turbo), 1/3 held out
Sets whose folder is missing are skipped (scripts/fetch_training_data.py creates shorts and pexels).

Features come from the frozen backbone (OpenVINO, cached in models/_train/features_ov.npz), so
evaluating a new head/rule takes seconds. Scoring: per frame, mean tile logit -> probability; per
item, trimmed mean of frame probabilities; flagged if >= 0.6.

    python scripts/bench.py                    # score the installed head (models/head.npz)
    python scripts/bench.py --head path.npz    # score another head
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_head as fh  # noqa: E402
import train_from_labels as tl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
THRESHOLD = 0.6
VIDEO_FRAMES = 6


def _frames_of(path: Path):
    """List of frames, each a list of tiles (as the extension captures them)."""
    if path.suffix.lower() in tl.VID:
        return [tl.tiles(fr) for fr in fh.frames(path, VIDEO_FRAMES)]
    return [tl.tiles(Image.open(path).convert("RGB"))]


def load_sets(extract=None, verbose=True):
    """{set_name: {item: (label, [frame -> (F[n,384], B[n])])}} using/filling the feature cache."""
    cache = tl.load_cache()
    dirty = 0

    def feats(key, load):
        nonlocal dirty, extract
        if key not in cache:
            if extract is None:
                extract = tl.Extractor()
            cache[key] = [[extract(t) for t in frame] for frame in load()]
            dirty += 1
            if dirty % 25 == 0:
                tl.save_cache(cache)
                if verbose:
                    print(f"    cached {dirty} new bench items ...", flush=True)
        frames = []
        for fr in cache[key]:
            if fr:
                F, B = zip(*fr)
                frames.append((np.array(F), np.array(B)))
        return frames

    sets = {}
    # user hold-out (tiles saved by the extension: treat each tile as one frame)
    hold = {}
    for item, (y, sources) in tl.user_items().items():
        if not tl.is_holdout(item):
            continue
        frames = []
        for key, load in sources:
            if key not in cache:
                if extract is None:
                    extract = tl.Extractor()
                cache[key] = [extract(t) for t in load()]
                dirty += 1
            frames += [(np.array([f]), np.array([b])) for f, b in cache[key]]
        if frames:
            hold[item] = (y, frames)
    sets["user-holdout"] = hold
    for name, sub in (("video", "vidtest"), ("images", "aigc"), ("games", "games"), ("shorts", "shorts"), ("pexels", "pexels"),
                      ("newgen", "newgen")):
        if not (BENCH / sub).exists():
            continue
        d = {}
        for p in sorted((BENCH / sub).rglob("*")):
            if p.suffix.lower() not in tl.IMG | tl.VID:
                continue
            y = 1 if p.parent.name == "ai" else 0
            d[str(p.relative_to(BENCH))] = (y, feats(f"b:{p.relative_to(BENCH)}", lambda p=p: _frames_of(p)))
        sets[name] = d
    if dirty:
        tl.save_cache(cache)
    return sets


def trimmed_mean(xs, frac=0.25):
    xs = sorted(xs)
    k = int(len(xs) * frac)
    kept = xs[k: len(xs) - k] if len(xs) - 2 * k > 0 else xs
    return float(np.mean(kept))


def frame_logit(F, B, head, rule, gate):
    """Mean tile logit for one frame under a head + rule. `head` = (w, v, b) or a callable F,B -> logits."""
    z = head(F, B) if callable(head) else fh.head_logits(head, F, B)
    if rule == "agree":
        z = np.minimum(z, B + fh.BIAS + fh.agree_shift(gate))
    elif rule == "base":
        z = B + fh.BIAS
    return float(np.mean(z))


def item_prob(frames, head, rule, gate, agg="trimmed"):
    ps = [1 / (1 + np.exp(-frame_logit(F, B, head, rule, gate))) for F, B in frames]
    if agg == "trimmed":
        return trimmed_mean(ps)
    if agg == "mean":
        return float(np.mean(ps))
    if agg == "median":
        return float(np.median(ps))
    raise ValueError(agg)


def evaluate(sets, head, rule="agree", gate=fh.AGREE_BASE_P, agg="trimmed", threshold=THRESHOLD):
    """{set: (ai_caught, n_ai, real_flagged, n_real)} plus 'ALL'."""
    out = {}
    tot = [0, 0, 0, 0]
    for name, items in sets.items():
        c = [0, 0, 0, 0]
        for y, frames in items.values():
            p = item_prob(frames, head, rule, gate, agg)
            if y:
                c[0] += p >= threshold
                c[1] += 1
            else:
                c[2] += p >= threshold
                c[3] += 1
        out[name] = tuple(c)
        tot = [a + b for a, b in zip(tot, c)]
    out["ALL"] = tuple(tot)
    return out


def fmt(res):
    cells = []
    for name, (ca, na, fr, nr) in res.items():
        cells.append(f"{name}: AI {ca}/{na}" + (f", real-flag {fr}/{nr}" if nr else ""))
    return " | ".join(cells)


def objective(res, fp_weight=4.0):
    """Higher is better: AI recall minus fp_weight x false-positive rate, each averaged over the sets (so a big set
    like shorts can't drown out the small held-out slice of your own feed)."""
    sets = [c for name, c in res.items() if name != "ALL"]
    recall = np.mean([ca / na for ca, na, _, _ in sets if na])
    fp = np.mean([fr / nr for _, _, fr, nr in sets if nr])
    return float(recall - fp_weight * fp)


def load_head(path):
    head, agree, gate = fh.load_head(path)
    return head, ("agree" if agree else "head"), gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ROOT / "models" / "head.npz"))
    args = ap.parse_args()
    sets = load_sets()
    head, rule, gate = load_head(args.head)
    print(f"sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))
    print(f"base model alone      : {fmt(evaluate(sets, head, 'base'))}")
    print(f"installed ({rule}, gate {gate:g}): {fmt(evaluate(sets, head, rule, gate))}")
    print(f"same head alone       : {fmt(evaluate(sets, head, 'head'))}")


if __name__ == "__main__":
    main()
