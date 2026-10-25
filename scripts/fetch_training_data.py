"""Grow the training set for scripts/train_from_labels.py (never touches bench/ or dataset/).

  videos  -> models/_train/<1|0>_<source>_<id>.mp4   (1 = AI, 0 = real)
  stills  -> models/_train_img/<ai|real>/<source>_<id>.<ext>

Sources
  AI video    Rapidata Veo 3 / Veo 3.1 / Sora sets, bitmind Kling (more clips than before)
  AI shorts   ymo2017/ai-shorts-videos: phone-style vertical AI shorts; 1/3 of concepts held out in bench/shorts
  AI new-gen  Seedance 1.5 Pro (330-prompt sample) + Kling 2.5 Turbo Pro (200): held-out bench/newgen only
  real video  AudioSet balanced + unbalanced (YouTube clips)
  real modern Pexels stock video (zengxianyu/open-sora-pexels-subset): 4 shards train + bench, 4 more bench only
  AI stills   Thermostatic/frontier-synthetic-images-2026 (FLUX.2, etc.)
  real stills AIGC-Detection-Benchmark real photos, video-game screenshots,
              + locally generated text cards / UI screenshots (meme cards were a false-positive source)

Bench items are excluded: videos by name, stills by content hash.

    python scripts/fetch_training_data.py [--ai-per-source 150] [--real-videos 500] [--stills 400]
"""

import argparse
import hashlib
import io
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_head as fh  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "models" / "_train"
TRAIN_IMG = ROOT / "models" / "_train_img"
BENCH = ROOT / "bench"


def say(m):
    print(m, flush=True)


def bench_hashes():
    return {hashlib.md5(p.read_bytes()).hexdigest() for p in BENCH.rglob("*") if p.is_file()}


def bench_video_stems():
    stems = {p.stem for p in (BENCH / "vidtest").rglob("*.mp4")}
    return stems | {s.split("_", 2)[-1] for s in stems}


def rows(ds, offset, length=10, config="default", split="train"):
    """datasets-server rows, with back-off on rate limits (HTTP 429) instead of hammering the API."""
    import time
    import urllib.error

    q = urllib.parse.urlencode({"dataset": ds, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = 15 * 2 ** attempt
            say(f"    rate-limited by Hugging Face; waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("still rate-limited after retries")


def fetch_stills(ds, n, out_dir, prefix, keep, bad_hashes, rng, total=None, img_col="image"):
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob(f"{prefix}_*")))
    if have >= n:
        return have
    total = total or rows(ds, 0, 1)["num_rows_total"]
    tries = 0
    while have < n and tries < n * 3:
        tries += 1
        try:
            batch = rows(ds, rng.randrange(0, max(1, total - 10)))["rows"]
        except Exception as e:
            say(f"    rows error {e}")
            continue
        for r in batch:
            row = r["row"]
            if not keep(row):
                continue
            try:
                data = urllib.request.urlopen(row[img_col]["src"], timeout=60).read()
            except Exception:
                continue
            h = hashlib.md5(data).hexdigest()
            if h in bad_hashes:
                continue
            (out_dir / f"{prefix}_{h[:12]}.jpg").write_bytes(data)
            have += 1
            if have >= n:
                break
        if have % 50 < 10:
            say(f"    {prefix}: {have}/{n}")
    return have


FONTS = ["arialbd.ttf", "impact.ttf", "arial.ttf", "segoeuib.ttf", "verdanab.ttf", "comicbd.ttf"]
WORDS = ("when you realize it's monday again pov your mom finds out wait for it nobody: me at 3am "
         "this is fine the way he looked at me no thoughts just vibes tell me why breaking news "
         "you won't believe what happened next tag a friend who does this rate this 1-10").split()


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{f}", size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def synth_cards(n, out_dir, rng):
    """Real (not AI) negatives: meme/text cards and app-UI-like screenshots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob("synth_*")))
    for i in range(have, n):
        w, h = rng.choice([(1080, 1080), (1080, 1350), (1080, 1920), (1280, 720)])
        kind = rng.random()
        bg = tuple(rng.randrange(256) for _ in range(3))
        img = Image.new("RGB", (w, h), bg)
        d = ImageDraw.Draw(img)
        if kind < 0.5:  # text card
            fg = (0, 0, 0) if sum(bg) > 380 else (255, 255, 255)
            y = rng.randrange(40, h // 3)
            for _ in range(rng.randrange(2, 7)):
                line = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(2, 6)))
                d.text((rng.randrange(20, 120), y), line, fill=fg, font=font(rng.randrange(48, 110)),
                       stroke_width=rng.choice([0, 0, 3]), stroke_fill=(0, 0, 0))
                y += rng.randrange(90, 160)
        else:  # app UI: header bar, rows, buttons, avatars
            d.rectangle((0, 0, w, 110), fill=tuple(rng.randrange(256) for _ in range(3)))
            d.text((30, 30), " ".join(rng.choice(WORDS) for _ in range(2)), fill="white", font=font(48))
            y = 140
            while y < h - 100:
                d.ellipse((30, y, 120, y + 90), fill=tuple(rng.randrange(256) for _ in range(3)))
                d.text((150, y + 10), " ".join(rng.choice(WORDS) for _ in range(rng.randrange(2, 5))), fill=(20, 20, 20), font=font(36))
                d.rounded_rectangle((w - 260, y + 15, w - 40, y + 75), 20, fill=(0, 120, 255))
                y += rng.randrange(120, 200)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randrange(60, 92))
        (out_dir / f"synth_{i:04d}.jpg").write_bytes(buf.getvalue())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-per-source", type=int, default=150)
    ap.add_argument("--real-videos", type=int, default=500)
    ap.add_argument("--stills", type=int, default=400, help="per still source")
    args = ap.parse_args()
    rng = random.Random(1)
    bad = bench_hashes()
    excluded = bench_video_stems()

    say("[1/4] videos (AI + real) ...")
    jobs = []
    for sources, label, n in ((fh.AI_SOURCES, 1, args.ai_per_source), (fh.REAL_SOURCES, 0, args.real_videos // 2)):
        for ds, keep in sources:
            vids = [p for p in fh.list_videos(ds, keep) if Path(p).stem not in excluded]
            for p in rng.sample(vids, min(n, len(vids))):
                tag = Path(p).parent.name.replace("/", "_")[:12]
                dst = TRAIN / f"{label}_{ds.split('/')[1][:20]}_{tag}_{Path(p).stem}.mp4"
                if not any(TRAIN.glob(f"{label}_{ds.split('/')[1][:20]}*_{Path(p).stem}.mp4")):
                    jobs.append((f"https://huggingface.co/datasets/{ds}/resolve/main/{p}", dst))
    # extra real variety: AudioSet unbalanced (via datasets-server video URLs)
    ub = "chikuwa-AI/AudioSet_unbalanced_video"
    want = args.real_videos - args.real_videos // 2
    have_ub = len(list(TRAIN.glob("0_AudioSet_unbal_*.mp4")))
    total = rows(ub, 0, 1)["num_rows_total"]
    while have_ub + sum(1 for _, d in jobs if "unbal" in d.name) < want:
        for r in rows(ub, rng.randrange(0, total - 10))["rows"]:
            src = r["row"]["video"]["src"] if isinstance(r["row"]["video"], dict) else None
            if src:
                jobs.append((src, TRAIN / f"0_AudioSet_unbal_{hashlib.md5(src.encode()).hexdigest()[:12]}.mp4"))
    say(f"    downloading {len(jobs)} new clips ...")
    done = [0]

    def get(job):
        url, dst = job
        if dst.exists():
            return
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            if len(data) < 15_000_000:
                dst.write_bytes(data)
        except Exception as e:
            say(f"    fail {dst.name}: {e}")
        done[0] += 1
        if done[0] % 50 == 0:
            say(f"    clips: {done[0]}/{len(jobs)}")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, jobs))
    fetch_shorts()
    fetch_pexels()
    fetch_newgen()
    say(f"    training clips now: {len(list(TRAIN.glob('1_*.mp4')))} AI, {len(list(TRAIN.glob('0_*.mp4')))} real")

    say("[2/4] AI stills (frontier synthetic 2026) ...")
    fetch_stills("Thermostatic/frontier-synthetic-images-2026", args.stills, TRAIN_IMG / "ai", "frontier",
                 lambda r: str(r.get("label")) == "1", bad, rng)
    say("[3/4] real stills (photos + game screenshots) ...")
    fetch_real_photos(args.stills, bad, rng)
    fetch_stills("badigadiii/game_screenshots_11k", args.stills, TRAIN_IMG / "real", "game", lambda r: True, bad, rng)
    say("[4/4] synthetic text cards / UI screenshots (real negatives) ...")
    synth_cards(args.stills, TRAIN_IMG / "real", rng)
    say(f"stills now: {len(list((TRAIN_IMG / 'ai').glob('*')))} AI, {len(list((TRAIN_IMG / 'real').glob('*')))} real")


SHORTS = "ymo2017/ai-shorts-videos"  # AI-generated vertical shorts: phone-style selfies, street scenes, pets


def shorts_concept(name):
    """Versions of one short (-v5, -final, -720x1280, -24fps) share a concept, so they stay on one side of the split."""
    return re.sub(r"(-v\d+.*|-final.*|-\d+x\d+.*|-\d+fps.*)$", "", name)


def fetch_shorts():
    """Phone-style AI shorts, split by concept: 2/3 -> models/_train/1_shorts_*.mp4, 1/3 -> bench/shorts/ai (held out)."""
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_tree(SHORTS, repo_type="dataset", path_in_repo="videos", recursive=True)
             if getattr(f, "size", None) and f.path.endswith(".mp4") and f.size < 40_000_000]
    held_dir = BENCH / "shorts" / "ai"
    held_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in files:
        name = Path(f.path).name
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        held = int(hashlib.sha1(shorts_concept(name[:-4]).encode()).hexdigest(), 16) % 3 == 0
        dst = held_dir / f"ai_shorts_{safe}" if held else TRAIN / f"1_shorts_{safe}"
        if not dst.exists():
            jobs.append((f"https://huggingface.co/datasets/{SHORTS}/resolve/main/{urllib.parse.quote(f.path)}", dst))
    say(f"    downloading {len(jobs)} shorts ...")

    def get(job):
        try:
            urllib.request.urlretrieve(*job)
        except Exception as e:
            say(f"    fail {job[1].name}: {e}")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get, jobs))
    say(f"    shorts: {len(list(TRAIN.glob('1_shorts_*.mp4')))} for training, {len(list(held_dir.glob('*.mp4')))} held out")


PEXELS = "zengxianyu/open-sora-pexels-subset"  # real stock video (Pexels, crawled early 2024), WebDataset shards
PEXELS_SHARDS = ["videos/5000_0_shard_0000.tar", "videos/5000_0_shard_0020.tar", "videos/5000_1_shard_0010.tar",
                 "videos/5000_1_shard_0040.tar"]
# More held-out real video for the benchmark. Their training share is parked in models/_ref/pexels_extra: the Oct 25
# cross-validation showed no clear gain from training on it.
PEXELS_BENCH_SHARDS = ["videos/5000_0_shard_0005.tar", "videos/5000_0_shard_0030.tar", "videos/5000_1_shard_0025.tar",
                       "videos/5000_1_shard_0050.tar"]


SEEDANCE = "sonnymushroom/seedance1.5pro_gen"  # Seedance 1.5 Pro generations (several seeds per prompt)
KLING25 = "yang29/GAD-Video-Data-200-KlingAI-2.5-Turbo-Pro"


def fetch_newgen(n_seedance=330):
    """Current-generation AI video, split by prompt (seeds of one Seedance prompt stay together): 1/3 -> bench/newgen/ai
    (held out); 2/3 parked in models/_ref/newgen_train (the Oct 25 cross-validation showed no clear gain from training
    on it: +0.7/68 AI on the user's feed for +1.0/47 real flagged)."""
    from huggingface_hub import HfApi

    held_dir = BENCH / "newgen" / "ai"
    park = ROOT / "models" / "_ref" / "newgen_train"
    for d in (held_dir, park):
        d.mkdir(parents=True, exist_ok=True)
    held = lambda concept: int(hashlib.sha1(concept.encode()).hexdigest(), 16) % 3 == 0
    jobs = []
    sd = [f for f in HfApi().list_repo_tree(SEEDANCE, repo_type="dataset", recursive=True)
          if getattr(f, "size", None) and f.path.endswith(".mp4") and f.path.startswith("train/") and f.size < 40_000_000]
    by_prompt = {}
    for f in sd:
        by_prompt.setdefault(f"{re.sub(r'_seed[0-9]+$', '', f.path.split('/')[1])}/{Path(f.path).stem}", []).append(f)
    rng = random.Random(5)
    for prompt in rng.sample(sorted(by_prompt), min(n_seedance, len(by_prompt))):
        f = rng.choice(by_prompt[prompt])
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f.path[6:-4])
        dst = held_dir / f"ai_newgen_seedance_{name}.mp4" if held("sd:" + prompt) else park / f"1_newgen_seedance_{name}.mp4"
        jobs.append((f"https://huggingface.co/datasets/{SEEDANCE}/resolve/main/{f.path}", dst))
    for f in HfApi().list_repo_tree(KLING25, repo_type="dataset", recursive=True):
        if getattr(f, "size", None) and f.path.endswith(".mp4"):
            row = f.path.split("_")[1]
            dst = held_dir / f"ai_newgen_kling25_{row}.mp4" if held("kl:" + row) else park / f"1_newgen_kling25_{row}.mp4"
            jobs.append((f"https://huggingface.co/datasets/{KLING25}/resolve/main/{urllib.parse.quote(f.path)}", dst))
    jobs = [j for j in jobs if not j[1].exists()]
    say(f"    downloading {len(jobs)} new-gen AI clips ...")

    def get(job):
        try:
            urllib.request.urlretrieve(*job)
        except Exception as e:
            say(f"    fail {job[1].name}: {e}")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get, jobs))
    say(f"    new-gen: {len(list(held_dir.glob('*.mp4')))} held out, {len(list(park.glob('*.mp4')))} parked")


def fetch_pexels():
    """Modern real video (stock footage, ~40% vertical), split by clip: 1/3 -> bench/pexels/real (held out), 2/3 ->
    models/_train/0_pexels_*.mp4 (PEXELS_SHARDS) or models/_ref/pexels_extra (PEXELS_BENCH_SHARDS). Only .mp4 members
    are read, by basename."""
    import tarfile

    held_dir = BENCH / "pexels" / "real"
    held_dir.mkdir(parents=True, exist_ok=True)
    extra = ROOT / "models" / "_ref" / "pexels_extra"
    extra.mkdir(parents=True, exist_ok=True)
    for shard in PEXELS_SHARDS + PEXELS_BENCH_SHARDS:
        train_dir = TRAIN if shard in PEXELS_SHARDS else extra
        tmp = TRAIN / Path(shard).name
        say(f"    pexels {shard} ...")
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{PEXELS}/resolve/main/{shard}", tmp)
        with tarfile.open(tmp) as tf:
            for m in tf:
                if not (m.isfile() and m.name.lower().endswith(".mp4")) or m.size > 40_000_000:
                    continue
                vid = re.sub(r"[^A-Za-z0-9_-]", "_", Path(m.name).stem)[:60]
                held = int(hashlib.sha1(vid.encode()).hexdigest(), 16) % 3 == 0
                dst = held_dir / f"real_pexels_{vid}.mp4" if held else train_dir / f"0_pexels_{vid}.mp4"
                if not dst.exists():
                    dst.write_bytes(tf.extractfile(m).read())
        tmp.unlink()
    say(f"    pexels: {len(list(TRAIN.glob('0_pexels_*.mp4')))} for training, {len(list(held_dir.glob('*.mp4')))} held out")


def fetch_real_photos(n, bad, rng):
    """Real photos from the AIGC benchmark (test split), skipping anything in bench/aigc."""
    out = TRAIN_IMG / "real"
    out.mkdir(parents=True, exist_ok=True)
    ds = "TheKernel01/AIGC-Detection-Benchmark"
    have = len(list(out.glob("photo_*")))
    total = 125026
    tries = 0
    while have < n and tries < n:
        tries += 1
        try:
            batch = rows(ds, rng.randrange(0, total - 10), 10, split="test")["rows"]
        except Exception as e:
            say(f"    rows error {e}")
            continue
        for r in batch:
            if r["row"]["label"] != 0:
                continue
            try:
                data = urllib.request.urlopen(r["row"]["image"]["src"], timeout=60).read()
            except Exception:
                continue
            h = hashlib.md5(data).hexdigest()
            if h in bad:
                continue
            (out / f"photo_{h[:12]}.jpg").write_bytes(data)
            have += 1
        if have % 50 < 6:
            say(f"    photo: {have}/{n}")


if __name__ == "__main__":
    main()
