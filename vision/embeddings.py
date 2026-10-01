"""FlyCommander — card embeddings (the recognition signal that replaces OCR).

Given a rectified card image, produce a fixed-length descriptor such that
embeddings of the *same physical card* under different conditions (tilt,
sleeve, foil, glare, lighting, blur, partial occlusion, 180-degree flip) are
close, and embeddings of *different* cards are far apart. That is the whole
job. Everything else — Scryfall lookup, set/collector resolution, game state —
is downstream bookkeeping.

Three tiers, selected automatically so the system is useful on any machine:

1. ``DenseEmbedder``   — handcrafted multi-zone descriptor (no training,
   no torch). Works out of the box; ~0.6 ms/card on CPU.
2. ``DenseHeadEmbedder`` — the same descriptor pushed through a *learned*
   projection trained on synthetic scenes (``vision/weights/dense_head.npz``).
   Much better separation, still torch-free inference.
3. ``TorchEmbedder`` / ``OnnxEmbedder`` — a small CNN trained on the same
   synthetic pipeline (``make train-embedder``). Best accuracy; the ONNX
   variant runs without torch installed.

Design rules baked in here (they are what make the matcher robust):

* **Feature-space capture normalisation.** Every capture — training, library
  build, query — goes through the identical ``normalize_capture`` path:
  unsharp mask against blur, CLAHE against lighting, mask-aware inpainting of
  glare/occlusion, canonical resize, per-channel standardisation.
* **Zone features, not raw pixels.** Card layout is fixed and informative:
  title band, art box, type band, text box, collector line and set symbol are
  scored and embedded separately, so a card flooded by glare at the bottom is
  still matched from title + artwork.
* **No single-signal dependence.** The dense descriptor carries colour
  identity, low-frequency layout, high-pass structure, gradient orientations
  and per-zone statistics; gradient/colour blocks dominate so that brightness
  and white balance shifts do not move the embedding much.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from vision import rectify as R

try:
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DENSE_HEAD_WEIGHTS = WEIGHTS_DIR / "dense_head.npz"
CNN_TORCH_WEIGHTS = WEIGHTS_DIR / "embedder.pt"
CNN_ONNX_WEIGHTS = WEIGHTS_DIR / "embedder.onnx"
CHECKPOINT_DIR = Path("checkpoints")

# embedder input (width, height) — same 0.715 aspect as the canonical card
EMBED_SIZE = (128, 179)
# canonical size images are rectified to before embedding
CANON_SIZE = (R.CARD_W, R.CARD_H)

# handcrafted descriptor length (asserted at runtime — see dense_features)
FEATURE_DIM = 894
DENSE_HEAD_DIM = 256


# ---------------------------------------------------------------------------
# capture normalisation (shared by every consumer)
# ---------------------------------------------------------------------------
def normalize_capture(card: np.ndarray, mask: np.ndarray | None = None,
                      size: tuple[int, int] = EMBED_SIZE,
                      inpaint: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalise a rectified capture for feature extraction.

    Returns (bgr uint8 image, valid uint8 mask) both at ``size``. Invalid
    pixels (glare blobs, occlusion, out-of-frame area) are inpainted from
    their neighbourhood so texture features do not see them, and the mask is
    carried along so the matcher can down-weight the regions that were
    reconstructed.
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for capture normalisation")
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    if mask is None:
        mask = np.full(card.shape[:2], 255, np.uint8)
    elif mask.shape[:2] != card.shape[:2]:
        mask = cv2.resize(mask, (card.shape[1], card.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    small = cv2.resize(card, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    invalid = small_mask < 128
    if inpaint and bool(invalid.any()):
        frac = float(invalid.mean())
        if frac < 0.6:
            small = cv2.inpaint(small, invalid.astype(np.uint8) * 255, 3,
                                cv2.INPAINT_TELEA)
    return small, small_mask


def enhance(img: np.ndarray, sharpen: float = 0.55,
            clahe: float = 2.0) -> np.ndarray:
    """Blur- and lighting-robust enhancement used before feature extraction."""
    out = img.astype(np.float32)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = out * (1.0 + sharpen) - blur * sharpen
    out = np.clip(out, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# handcrafted descriptor
# ---------------------------------------------------------------------------
def _grid(img: np.ndarray, gw: int, gh: int) -> np.ndarray:
    return cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)


def _zscore(block: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Zero-mean/unit-scale a block, then apply its fusion weight.

    Per-block normalisation is what stops the descriptor from being dominated
    by the parts of a card that are the *same on every card* (uniform frame
    colour, white text box). Without it, cosine similarity between two
    different cards sits around 0.8 and retrieval collapses.
    """
    x = block.reshape(-1).astype(np.float32)
    x = x - float(x.mean())
    x = x / (float(np.std(x)) + 1e-4)
    return x * weight


def _high_pass(gray: np.ndarray, sigma: float = 2.2) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    hp = gray - blur
    return hp / (float(np.std(hp)) + 1e-4)


def dense_features(card: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """894-d pose/lighting-tolerant descriptor of a rectified card.

    Design: every block is captured at a resolution matched to what it can
    actually discriminate, z-scored independently and then weighted, so the
    final vector is dominated by *card-specific* structure (artwork, text,
    symbols, layout) rather than card-common structure (frame colour,
    white text box). Weights were tuned on the synthetic benchmark in
    ``scripts/eval_vision.py`` — see ``tests/test_embeddings.py`` for the
    regression guards.

    ===============  ====  =============================================
    block            dim   what it captures
    ===============  ====  =============================================
    colour           105   colour identity (5x7 grid, 3 channels)
    layout low       140   frame/title/text-box geometry (10x14 gray)
    layout high      140   borders, rules text, frame edges (high-pass)
    gradient orient   96   edge structure, illumination invariant (3x4x8)
    artwork colour   105   illustration palette (5x7x3)
    artwork struct   140   illustration edges (high-pass)
    text strip       144   title/type/collector bands as one strip (24x6)
    zone stats        24   per-zone mean/std/edge/glare energy
    ===============  ====  =============================================
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for dense features")
    card, mask = normalize_capture(card, mask, size=EMBED_SIZE)
    card = enhance(card)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    hp = _high_pass(gray_f)

    colour = _zscore(_grid(cv2.cvtColor(card, cv2.COLOR_BGR2RGB), 7, 5) / 255.0,
                     0.35)
    layout_low = _zscore(_grid(gray_f, 14, 10), 0.20)
    layout_hp = _zscore(_grid(hp, 14, 10), 0.55)

    # gradient orientation histograms (illumination invariant): 3x4 cells x 8
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % math.pi) / math.pi * 8.0
    bins = np.clip(ang.astype(np.int32), 0, 7)
    hog = np.zeros((4, 3, 8), np.float32)
    h, w = gray.shape
    for row in range(4):
        y0, y1 = int(h * row / 4), int(h * (row + 1) / 4)
        for col in range(3):
            x0, x1 = int(w * col / 3), int(w * (col + 1) / 3)
            hist = np.bincount(bins[y0:y1, x0:x1].reshape(-1),
                               weights=mag[y0:y1, x0:x1].reshape(-1),
                               minlength=8)[:8]
            hog[row, col] = hist / (float(hist.sum()) + 1e-5)
    hog = _zscore(hog, 0.50)

    art = R.crop_zone(card, "art")
    art_gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    art_colour = _zscore(_grid(cv2.cvtColor(art, cv2.COLOR_BGR2RGB), 7, 5) / 255.0,
                         0.60)
    art_struct = _zscore(_grid(_high_pass(art_gray, 1.8), 14, 10), 0.80)

    # the three text-ish bands stacked into one strip: title, type, collector
    strips = []
    for zone in ("title", "type", "collector"):
        z = R.crop_zone(card, zone)
        zg = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        strip = cv2.resize(zg, (24, max(1, zg.shape[0] * 24 // zg.shape[1])),
                           interpolation=cv2.INTER_AREA)
        strips.append(strip)
    text_strip = np.vstack(strips)
    text_strip = _zscore(cv2.resize(text_strip, (24, 6),
                                    interpolation=cv2.INTER_AREA), 0.45)

    zone_stats = []
    for zone in ("title", "art", "type", "text", "collector", "setsymbol"):
        zg = cv2.cvtColor(R.crop_zone(card, zone),
                          cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        zone_stats.extend([float(zg.mean()), float(zg.std()),
                           float(np.mean(np.abs(zg - cv2.GaussianBlur(zg, (0, 0), 1.5))))
                           * 8.0, float(np.mean(zg > 0.86))])
    zone_stats = _zscore(np.clip(np.array(zone_stats, np.float32), 0, 2), 0.25)

    feat = np.concatenate([colour, layout_low, layout_hp, hog, art_colour,
                           art_struct, text_strip, zone_stats])
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
    if feat.size != FEATURE_DIM:  # pragma: no cover - guard against drift
        raise AssertionError(f"dense feature dim {feat.size} != {FEATURE_DIM}")
    norm = float(np.linalg.norm(feat))
    return (feat / norm if norm > 1e-6 else feat).astype(np.float32)


def dense_features_batch(cards: Sequence[np.ndarray],
                         masks: Sequence[np.ndarray] | None = None
                         ) -> np.ndarray:
    masks = masks or [None] * len(cards)  # type: ignore[list-item]
    return np.stack([dense_features(c, m) for c, m in zip(cards, masks)])


def flip180(img: np.ndarray) -> np.ndarray:
    """A card is geometrically its own 180-degree rotation; recognition must
    consider both readings (the matcher scores both)."""
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate_card(img: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a rectified card by a multiple of 90 degrees (clockwise)."""
    rotation = int(rotation) % 360
    if rotation == 0:
        return img
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be a multiple of 90 degrees")


# the four readings of a card that a matcher must always consider: a card on a
# table can be upright, tapped (90), upside-down (180) or tapped-and-flipped
CARD_ROTATIONS = (0, 90, 180, 270)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# embedder interface
# ---------------------------------------------------------------------------
class Embedder:
    """Base class: image(s) → L2-normalised embedding(s)."""

    name: str = "base"
    dim: int = 0
    trained: bool = False

    def embed_batch(self, cards: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray] | None = None) -> np.ndarray:
        raise NotImplementedError

    def embed(self, card: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.embed_batch([card], None if mask is None else [mask])[0]

    def embed_with_flip(self, card: np.ndarray, mask: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
        """(embedding, flipped embedding) — the matcher takes the better one."""
        masks = None if mask is None else [mask, None]
        out = self.embed_batch([card, flip180(card)], masks)
        return out[0], out[1]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "trained": self.trained}


class DenseEmbedder(Embedder):
    """Zero-training descriptor (works everywhere, no torch)."""

    name = "dense"
    dim = FEATURE_DIM

    def embed_batch(self, cards: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray] | None = None) -> np.ndarray:
        if masks is None:
            masks = [None] * len(cards)  # type: ignore[list-item]
        out = []
        for card, mask in zip(cards, masks):
            out.append(dense_features(card, mask))
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)


class DenseHeadEmbedder(Embedder):
    """Learned linear projection of the dense descriptor (torch-free).

    Weights come from ``scripts/train_embedder.py`` (``--head dense``) and are
    a plain matrix + bias, so inference is a single matmul: microsecond cost
    on top of the descriptor.
    """

    name = "dense-head"
    trained = True

    def __init__(self, path: str | Path = DENSE_HEAD_WEIGHTS) -> None:
        self.path = Path(path)
        data = np.load(self.path)
        self.w = data["w"].astype(np.float32)          # (FEATURE_DIM, D)
        self.b = data["b"].astype(np.float32)
        self.mu = data["mu"].astype(np.float32)
        self.sigma = np.where(data["sigma"].astype(np.float32) < 1e-6, 1.0,
                              data["sigma"].astype(np.float32))
        self.dim = int(self.w.shape[1])
        if "meta" in data:
            import json
            self.meta = json.loads(str(data["meta"]))
        else:
            self.meta = {}

    def embed_batch(self, cards: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray] | None = None) -> np.ndarray:
        feats = dense_features_batch(cards, masks)
        x = (feats - self.mu) / self.sigma
        out = x @ self.w + self.b
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms < 1e-6, 1.0, norms)).astype(np.float32)


class _TorchEmbedderBase(Embedder):
    trained = True

    def _finish(self, out: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms < 1e-6, 1.0, norms)).astype(np.float32)

    def _prepare(self, cards: Sequence[np.ndarray],
                 masks: Sequence[np.ndarray] | None) -> np.ndarray:
        """NCHW float32 batch in [-1, 1] (the network's expected input)."""
        if masks is None:
            masks = [None] * len(cards)  # type: ignore[list-item]
        batch = []
        for card, mask in zip(cards, masks):
            img, _ = normalize_capture(card, mask, size=EMBED_SIZE)
            img = enhance(img)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - 0.5) / 0.25
            batch.append(rgb.transpose(2, 0, 1))
        return np.stack(batch).astype(np.float32) if batch else np.zeros(
            (0, 3, EMBED_SIZE[1], EMBED_SIZE[0]), np.float32)


class TorchEmbedder(_TorchEmbedderBase):
    """Small CNN embedder (PyTorch). Trained by scripts/train_embedder.py."""

    name = "cnn-torch"

    def __init__(self, path: str | Path = CNN_TORCH_WEIGHTS,
                 device: str | None = None) -> None:
        import torch  # local import: torch is optional everywhere else
        self.path = Path(path)
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.meta = ckpt.get("meta", {})
        self.width = int(cfg.get("width", 32))
        self.dim = int(cfg.get("dim", 256))
        self.size = tuple(cfg.get("size", EMBED_SIZE))
        self.model = CardEmbedNet(dim=self.dim, width=self.width)
        self.model.load_state_dict(ckpt["state_dict"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        self._torch = torch

    def embed_batch(self, cards: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray] | None = None) -> np.ndarray:
        if not len(cards):
            return np.zeros((0, self.dim), np.float32)
        x = self._prepare(cards, masks)
        torch = self._torch
        with torch.no_grad():
            out = self.model(torch.from_numpy(x).to(self.device))
            out = torch.nn.functional.normalize(out, dim=1)
        if self.device != "cpu":
            out = out.cpu()
        return out.numpy().astype(np.float32)


class OnnxEmbedder(_TorchEmbedderBase):
    """Same CNN, exported to ONNX — inference without PyTorch installed."""

    name = "cnn-onnx"

    def __init__(self, path: str | Path = CNN_ONNX_WEIGHTS,
                 providers: list[str] | None = None) -> None:
        import json
        import onnxruntime as ort  # type: ignore
        self.path = Path(path)
        meta_path = self.path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.dim = int(self.meta.get("dim", 256))
        self.size = tuple(self.meta.get("size", EMBED_SIZE))
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed_batch(self, cards: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray] | None = None) -> np.ndarray:
        if not len(cards):
            return np.zeros((0, self.dim), np.float32)
        x = self._prepare(cards, masks)
        out = self.session.run(None, {self.input_name: x})[0]
        return self._finish(np.asarray(out, np.float32))


# ---------------------------------------------------------------------------
# CNN definition (shared by training, torch inference and ONNX export)
# ---------------------------------------------------------------------------
def _build_net(dim: int = 256, width: int = 32):
    """Tiny CNN: ~0.6M params, ~2 ms/card on a modern CPU, <1 ms on a 3070 Ti."""
    import torch.nn as nn

    def block(cin, cout, stride=2):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class GeM(nn.Module):
        """Generalised-mean pooling — robust to occlusion and glare blobs."""

        def __init__(self, p: float = 3.0) -> None:
            super().__init__()
            self.p = nn.Parameter(__import__("torch").tensor(float(p)))

        def forward(self, x):
            import torch
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=1e-6).pow(p)
            x = x.mean(dim=(2, 3)).pow(1.0 / p)
            return x

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                block(3, width, 2),            # 64x90
                block(width, width * 2, 2),    # 32x45
                block(width * 2, width * 4, 2),  # 16x23
                block(width * 4, width * 4, 1),
                block(width * 4, width * 8, 2),  # 8x12
            )
            self.pool = GeM()
            self.head = nn.Sequential(
                nn.Linear(width * 8, dim, bias=False),
                nn.BatchNorm1d(dim),
            )

        def forward(self, x):
            import torch
            x = self.features(x)
            x = self.pool(x)
            x = self.head(x)
            return torch.nn.functional.normalize(x, dim=1)

    return Net()


def CardEmbedNet(dim: int = 256, width: int = 32):
    """Public factory for the embedder CNN (imported by the training script)."""
    return _build_net(dim=dim, width=width)


# ---------------------------------------------------------------------------
# auto-selection
# ---------------------------------------------------------------------------
def available_embedders() -> dict[str, str]:
    """Where each tier's weights live (for `--embedder auto` diagnostics)."""
    import os
    return {
        "onnx": str(CNN_ONNX_WEIGHTS),
        "torch": str(CNN_TORCH_WEIGHTS),
        "dense-head": str(DENSE_HEAD_WEIGHTS),
        "dense": "<built-in>",
        "env": os.environ.get("FLYCOMMANDER_EMBEDDER", ""),
    }


def load_embedder(kind: str = "auto", path: str | Path | None = None
                  ) -> Embedder:
    """Load the best available embedder.

    ``auto`` preference order: ONNX → torch CNN → dense head → raw dense.
    Any requested tier that is unavailable falls back with the reason carried
    in ``describe()["fallbackReason"]`` — never raises, so the pipeline always
    runs (this is the difference between "no model yet" and "no recognition").
    """
    import os
    env = os.environ.get("FLYCOMMANDER_EMBEDDER")
    if kind == "auto" and env:
        kind = env
    reason = ""

    def _try(fn, label):
        nonlocal reason
        try:
            emb = fn()
            if reason:
                setattr(emb, "fallback_reason", reason)
            return emb
        except Exception as exc:  # pragma: no cover - environment dependent
            reason = f"{label} unavailable ({type(exc).__name__}: {exc})"
            return None

    order = ["onnx", "torch", "dense-head", "dense"] if kind == "auto" else [kind]
    for tier in order:
        emb: Embedder | None = None
        if tier == "onnx":
            p = Path(path) if path and str(path).endswith(".onnx") else CNN_ONNX_WEIGHTS
            if p.exists():
                emb = _try(lambda: OnnxEmbedder(p), "ONNX embedder")
        elif tier == "torch":
            p = Path(path) if path and str(path).endswith((".pt", ".pth")) else CNN_TORCH_WEIGHTS
            if p.exists():
                emb = _try(lambda: TorchEmbedder(p), "PyTorch embedder")
        elif tier == "dense-head":
            p = Path(path) if path and str(path).endswith(".npz") else DENSE_HEAD_WEIGHTS
            if p.exists():
                emb = _try(lambda: DenseHeadEmbedder(p), "dense head")
        elif tier == "dense":
            emb = DenseEmbedder()
        if emb is not None:
            return emb
    return DenseEmbedder()
