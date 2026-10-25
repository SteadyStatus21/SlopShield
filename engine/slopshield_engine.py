"""SlopShield Engine — local desktop companion that scores media crops for the extension.

The extension still finds media and captures crops; when this engine is running it sends the crops
here instead of scoring them in the browser. Native OpenVINO on the Intel iGPU is ~3x faster than the
browser's WebGPU path (Iris Xe: 47 ms vs ~150 ms per crop for the shipped ViT-S), which leaves room
for more frames and bigger models. If the engine isn't running, the extension falls back to WebGPU.

Security: listens on 127.0.0.1 only and accepts WebSocket connections only from the SlopShield
extension's origin (pinned extension ID), so web pages can't use it. Nothing leaves the machine.

    pip install -r engine/requirements.txt
    python engine/slopshield_engine.py [--device GPU|CPU] [--port 47321]
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import websockets
from PIL import Image

VERSION = "0.1.0"
DEFAULT_PORT = 47321
ROOT = Path(__file__).resolve().parent.parent
# Must match manifest.json "key" (pinned so the ID is the same on every machine).
ALLOWED_ORIGINS = {"chrome-extension://jpcjjoehiklpjalioikahehafipnbjjk"}

log = logging.getLogger("slopshield-engine")


class Detector:
    """The extension's model (models/model.onnx + model.json), compiled with OpenVINO."""

    def __init__(self, model_dir: Path, device: str):
        self.meta = json.loads((model_dir / "model.json").read_text())
        core = ov.Core()
        if device == "AUTO":
            device = "GPU" if "GPU" in core.available_devices else "CPU"
        # fp16 on the GPU: verified against fp32 CPU outputs (differences ~0.03 in logit space).
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16" if device == "GPU" else "f32"}
        t0 = time.perf_counter()
        self.compiled = core.compile_model(str(model_dir / "model.onnx"), device, cfg)
        self.request = self.compiled.create_infer_request()
        self.device = f"{device} ({core.get_property(device, 'FULL_DEVICE_NAME')})"
        log.info("model %s compiled on %s in %.1fs", self.meta.get("name", "?"), self.device, time.perf_counter() - t0)
        self.size = int(self.meta["size"])
        self.mean = np.array(self.meta["mean"], np.float32)
        self.std = np.array(self.meta["std"], np.float32)

    def _tensor(self, data_url: str) -> np.ndarray:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(img, np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def score(self, images: list[str]) -> list[float]:
        """P(AI) per image, same math as the extension's offscreen/infer.ts pAI()."""
        out = []
        for url in images:  # one at a time: fixed input shape, no recompiles
            logits = self.request.infer({0: self._tensor(url)})[0].ravel()
            if logits.size == 1:
                z = float(logits[0]) + float(self.meta.get("logitBias", 0.0))
                out.append(float(1.0 / (1.0 + np.exp(-z))))
            else:
                e = np.exp(logits - logits.max())
                out.append(float(e[int(self.meta.get("aiIndex", 0))] / e.sum()))
        return out


def save_crops(folder: Path, images: list[str], scores: list[float], keep: int = 60):
    """Debug: write the exact crops the model scored (score in the file name), keeping the newest `keep`."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for i, (url, s) in enumerate(zip(images, scores)):
        (folder / f"{stamp}_{int(time.time() * 1000) % 1000:03d}_{i}_p{s:.2f}.png").write_bytes(
            base64.b64decode(url.split(",", 1)[1]))
    for old in sorted(folder.glob("*.png"))[:-keep]:
        old.unlink()


DATASET = ROOT / "dataset"


def save_label(label: str, key: str, page: str, images: list[str]) -> tuple[int, str]:
    """User feedback from the extension: store the item's crops under dataset/<label>/<site>_<id>/.

    Re-labelling the same item moves it (a correction must not leave a contradictory copy behind).
    """
    import hashlib
    import shutil
    from urllib.parse import urlparse

    if label not in ("ai", "real"):
        raise ValueError(f"bad label {label!r}")
    item = hashlib.sha1(key.encode()).hexdigest()[:12]
    site = (urlparse(page).hostname or "page").removeprefix("www.").split(".")[0]
    name = f"{site}_{item}"
    for other in ("ai", "real"):
        if other != label and (DATASET / other / name).exists():
            shutil.rmtree(DATASET / other / name)
    folder = DATASET / label / name
    folder.mkdir(parents=True, exist_ok=True)
    start = len(list(folder.glob("tile*.png")))
    for i, url in enumerate(images):
        (folder / f"tile{start + i:03d}.png").write_bytes(base64.b64decode(url.split(",", 1)[1]))
    (folder / "meta.json").write_text(json.dumps({"label": label, "key": key, "site": urlparse(page).hostname,
                                                  "labeledAt": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    return len(images), f"dataset/{label}/{name}"


async def serve(detector: Detector, port: int, crops_dir: Path | None = None):
    lock = asyncio.Lock()  # one inference at a time; the extension already budgets requests

    async def handler(ws):
        origin = ws.request.headers.get("Origin", "")
        if origin not in ALLOWED_ORIGINS:
            log.warning("rejected connection from origin %r", origin)
            await ws.close(code=4003, reason="origin not allowed")
            return
        log.info("extension connected")
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "hello":
                    await ws.send(json.dumps({"type": "hello", "version": VERSION, "device": detector.device,
                                              "model": detector.meta.get("name")}))
                elif msg.get("type") == "label":
                    try:
                        saved, folder = save_label(msg["label"], msg["key"], msg.get("page", ""), msg["images"])
                        log.info("labelled %s: %d crop(s) -> %s", msg["label"].upper(), saved, folder)
                        await ws.send(json.dumps({"type": "labeled", "id": msg.get("id"), "saved": saved, "folder": folder}))
                    except Exception as e:
                        log.exception("saving label failed")
                        await ws.send(json.dumps({"type": "error", "id": msg.get("id"), "error": str(e)}))
                elif msg.get("type") == "score":
                    t0 = time.perf_counter()
                    async with lock:
                        try:
                            scores = await asyncio.to_thread(detector.score, msg["images"])
                            reply = {"type": "scores", "id": msg.get("id"), "scores": scores,
                                     "ms": round((time.perf_counter() - t0) * 1000)}
                        except Exception as e:  # report, don't crash the connection
                            log.exception("scoring failed")
                            reply = {"type": "error", "id": msg.get("id"), "error": str(e)}
                    await ws.send(json.dumps(reply))
                    if reply["type"] == "scores":
                        log.info("%d crop(s) -> %s in %d ms", len(scores), [round(float(s), 2) for s in scores], reply["ms"])
                        if crops_dir:
                            save_crops(crops_dir, msg["images"], scores)
        except websockets.ConnectionClosed:
            pass
        log.info("extension disconnected")

    async with websockets.serve(handler, "127.0.0.1", port, max_size=32 * 1024 * 1024):
        log.info("SlopShield Engine %s listening on ws://127.0.0.1:%d (Ctrl+C to stop)", VERSION, port)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="AUTO", choices=["AUTO", "GPU", "CPU"])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--save-crops", nargs="?", const=str(ROOT / "engine" / "last_crops"), default=None,
                    help="debug: save the crops the model receives (newest 60) to this folder")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    # Fail fast (before the ~10 s model compile) if another engine already owns the port.
    import socket

    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            log.info("SlopShield Engine is already running on port %d. Nothing to do; close this window.", args.port)
            return
    detector = Detector(Path(args.models), args.device)
    try:
        asyncio.run(serve(detector, args.port, Path(args.save_crops) if args.save_crops else None))
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
