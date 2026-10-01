"""FlyCommander — card detection (multi-hypothesis, angle-agnostic).

The detector's contract:

    detect(frame) -> [CardDetection(quad, confidence, ...)]

It NEVER filters by angle, aspect band or "flatness". A card seen at 60
degrees of keystone, rotated 130 degrees, sleeved, foiled, half under another
card, or hanging off the frame edge is still a card — the geometry stage
corrects it and the matcher decides whether the *content* is readable. The
only thing detection decides is *where the card polygons are*, plus a
confidence that expresses how card-like each candidate is.

Two complementary proposal sources are merged (both tilt-independent):

* edge quads   — Canny gradient map → contours → polygon approximation.
                 Strong when the card silhouette is visible.
* foreground   — flood-fill segmentation of the low-gradient background from
                 the frame border → connected components → quads. Strong when
                 the card boundary is soft but the card surface is busy.

Candidates are then scored by evidence that does not depend on angle:

* edge support      — fraction of the quad perimeter sitting on gradients
                      (measured by vision.rectify.refine_quad)
* interior structure— card surfaces carry layout edges; table/background does
                      not (a strong, cheap discriminator)
* border contrast   — inside/outside intensity step
* size + shape      — plausibility, with a *wide* aspect band and no rejection

Optional learned 'cardness' classifier (vision/weights/cardness.npz, trained
by scripts/train_embedder.py on synthetic scenes) is applied to the rectified
candidate when the weights are present; it sharpens the confidence without ever
being required.

A YOLO/RT-DETR ONNX detector can be dropped in (``model_path=...``); when
onnxruntime or the model file is missing the classical pipeline runs instead,
so the system keeps working on any machine at any stage of training.
"""
from __future__ import annotations

import math
import time
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
CARDNESS_WEIGHTS = WEIGHTS_DIR / "cardness.npz"

# Detection is deliberately permissive: a missed card is unrecoverable, a
# spurious candidate costs one cheap rectification + embedding lookup.
MIN_AREA_FRAC = 0.010          # smallest card we even consider (small table card)
MAX_AREA_FRAC = 0.95           # whole frame covered by one card
MIN_CONFIDENCE = 0.15          # below this a candidate is dropped as noise
# (permissive by design: a missed card is unrecoverable, a false positive
#  costs one embedding lookup and is discarded by the identity gate)
NMS_IOU = 0.45
# A much larger, less-structured blob that is mostly occupied by a real card is
# background/shadow around it. Enabled by default; disabling trades misses for
# false positives (measured in scripts/eval_vision.py).
DROP_BACKGROUND_BLOBS = False
BLOB_STRUCTURE_MARGIN = 0.12
REFINE_MAX_DIM = 1000          # detection works on a downscaled frame


@dataclass
class CardDetection:
    """One detected card polygon with its evidence."""

    quad: np.ndarray                       # 4x2 float32, tl,tr,br,bl (portrait)
    confidence: float
    area_frac: float
    tilt_deg: float = 0.0
    edge_support: float = 0.0
    interior_structure: float = 0.0
    border_contrast: float = 0.0
    cardness: float = 0.0
    source: str = "edge"
    notes: list[str] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        c = self.quad.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0, y0 = self.quad[:, 0].min(), self.quad[:, 1].min()
        x1, y1 = self.quad[:, 0].max(), self.quad[:, 1].max()
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quad": [[round(float(x), 2), round(float(y), 2)]
                     for x, y in self.quad],
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 3),
            "areaFrac": round(self.area_frac, 4),
            "tiltDeg": round(self.tilt_deg, 1),
            "edgeSupport": round(self.edge_support, 3),
            "interiorStructure": round(self.interior_structure, 3),
            "borderContrast": round(self.border_contrast, 3),
            "cardness": round(self.cardness, 3),
            "source": self.source,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# learned cardness classifier (optional, tiny logistic head)
# ---------------------------------------------------------------------------
class CardnessScorer:
    """Logistic-regression 'is this rectified crop a card?' scorer.

    Weights are produced by scripts/train_embedder.py alongside the embedder.
    The feature transform is shared with vision.embeddings.dense_features so
    training and inference cannot drift apart.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else CARDNESS_WEIGHTS
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.load()

    @property
    def available(self) -> bool:
        return self.w is not None

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = np.load(self.path)
            self.w = data["w"].astype(np.float32)
            self.b = float(data["b"])
            self.mu = data["mu"].astype(np.float32)
            self.sigma = data["sigma"].astype(np.float32)
            return True
        except Exception:  # pragma: no cover - corrupt weights
            self.w = None
            return False

    def score(self, card: np.ndarray) -> float:
        """0..1 probability that a rectified card image is really a card."""
        if self.w is None:
            return 0.0
        from vision.embeddings import dense_features

        x = dense_features(card).astype(np.float32)
        if x.size != self.w.size:
            return 0.0
        x = (x - self.mu) / np.where(self.sigma < 1e-6, 1.0, self.sigma)
        z = float(x @ self.w + self.b)
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


# ---------------------------------------------------------------------------
# proposal sources
# ---------------------------------------------------------------------------
def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Canny edges with CLAHE normalisation (robust to uneven lighting)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    median = float(np.median(blur))
    lo = int(max(20, 0.66 * median))
    hi = int(min(255, max(lo + 30, 1.33 * median)))
    edges = cv2.Canny(blur, lo, hi, L2gradient=True)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                             iterations=1)
    return edges


def _quad_from_contour(contour: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray | None:
    """Best 4-point approximation of a contour (polygon sweep + min-area rect)."""
    h, w = frame_shape
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    peri = float(cv2.arcLength(contour, True))
    best: np.ndarray | None = None
    best_score = -1.0
    for eps in (0.015, 0.025, 0.04, 0.06, 0.09):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(-1, 2).astype(np.float32)
            # prefer the approximation whose area tracks the contour
            score = 1.0 - abs(R.quad_area(quad) - area) / max(area, 1.0)
            if score > best_score:
                best, best_score = quad, score
    if best is None:
        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw < 4 or rh < 4:
            return None
        best = cv2.boxPoints(rect).astype(np.float32)
    return best


def _edge_quads(frame: np.ndarray, scale: float) -> list[tuple[np.ndarray, float]]:
    """Contour-derived quads (already rescaled to frame coordinates)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = _edge_map(gray)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    out: list[tuple[np.ndarray, float]] = []
    for _, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < MIN_AREA_FRAC * frame_area * 0.6:
            continue
        quad = _quad_from_contour(contour, (h, w))
        if quad is None:
            continue
        quad = R.ensure_portrait(R.order_corners(quad))
        fill = R.quad_area(quad) / area if area > 0 else 0.0
        out.append((quad / scale, float(min(1.2, fill))))
    return out


def _foreground_quads(frame: np.ndarray, scale: float
                      ) -> list[tuple[np.ndarray, float]]:
    """Background-segmentation quads: 'everything that is not the table'."""
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    smooth = cv2.GaussianBlur(grad, (0, 0), 6.0)
    thresh = max(8.0, float(np.percentile(smooth, 55)))
    busy = (smooth > thresh).astype(np.uint8)
    busy = cv2.morphologyEx(busy, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # flood fill the quiet area from the border → that is the background
    h, w = busy.shape
    ff = busy.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, max(1, w // 16)):
        if ff[0, x] == 0:
            cv2.floodFill(ff, mask, (x, 0), 2)
        if ff[h - 1, x] == 0:
            cv2.floodFill(ff, mask, (x, h - 1), 2)
    for y in range(0, h, max(1, h // 16)):
        if ff[y, 0] == 0:
            cv2.floodFill(ff, mask, (0, y), 2)
        if ff[y, w - 1] == 0:
            cv2.floodFill(ff, mask, (w - 1, y), 2)
    foreground = np.where(ff == 2, 0, 1).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    frame_area = float(h * w)
    out: list[tuple[np.ndarray, float]] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_FRAC * frame_area:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        quad = _quad_from_contour(contour, (h, w))
        if quad is None:
            continue
        quad = R.ensure_portrait(R.order_corners(quad))
        solidity = area / max(1.0, R.quad_area(quad))
        out.append((quad / scale, float(min(1.2, solidity))))
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _interior_structure(gray: np.ndarray, quad: np.ndarray) -> float:
    """Edge density *inside* the quad relative to the whole frame.

    Cards carry text, art, frames and borders — a lot of structure per unit
    area. Table cloth / wood / background carry little. Ratio-based, so it is
    invariant to lighting and camera.
    """
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.clip(quad.astype(np.int32), [-1, -1], [w, h])
    cv2.fillPoly(mask, [R.order_corners(pts.astype(np.float32)).astype(np.int32)], 1)
    area = int(mask.sum())
    if area < 64:
        return 0.0
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    inside = float(grad[mask == 1].mean())
    outside = float(grad.reshape(-1)[np.flatnonzero(mask.reshape(-1) == 0)].mean()) \
        if area < grad.size else 0.0
    if outside <= 1e-3:
        return 1.0
    return float(np.clip((inside / outside - 1.0) / 1.6, 0.0, 1.0))


def _border_contrast(gray: np.ndarray, quad: np.ndarray) -> float:
    """Normalised inside/outside intensity step across the quad boundary."""
    h, w = gray.shape[:2]
    inner = np.zeros((h, w), np.uint8)
    outer = np.zeros((h, w), np.uint8)
    centre = quad.mean(axis=0)
    inner_q = centre + (quad - centre) * 0.86
    outer_q = centre + (quad - centre) * 1.18
    cv2.fillPoly(inner, [R.order_corners(inner_q).astype(np.int32)], 1)
    cv2.fillPoly(outer, [R.order_corners(outer_q).astype(np.int32)], 1)
    ring = (outer == 1) & (inner == 0)
    if int(inner.sum()) < 64 or int(ring.sum()) < 64:
        return 0.0
    blob = cv2.GaussianBlur(gray, (0, 0), 3.0)
    inside_mean = float(blob[inner == 1].mean())
    ring_mean = float(blob[ring].mean())
    std = float(blob.std()) + 1e-3
    return float(np.clip(abs(inside_mean - ring_mean) / (0.6 * std), 0.0, 1.0))


#: how much plausibility a frame-clipped band keeps (see shape_plausibility)
FRAME_CLIP_STRIP_PENALTY = 0.55


def shape_plausibility(quad: np.ndarray,
                       frame_shape: tuple[int, int] | None = None) -> float:
    """0..1: how card-shaped is this quad, independent of tilt and content?

    Deliberately permissive (a card at 60 degrees of keystone or half covered
    by a hand must still pass), but it eliminates the systematic proposals of
    segmentation: long thin strips, near-square blobs, and anything that is
    basically the whole frame.
    """
    w, h = R.quad_side_lengths(quad)
    long_side, short_side = max(w, h), max(1e-3, min(w, h))
    ratio = short_side / long_side
    # MTG is 0.716; sleeves and keystone move this a lot, strips do not
    if ratio > 0.92:
        ratio_score = max(0.0, 1.0 - (ratio - 0.92) / 0.08)     # squares
    else:
        ratio_score = float(np.clip((ratio - 0.30) / 0.22, 0.0, 1.0))
    score = 0.30 + 0.70 * ratio_score
    if frame_shape is not None:
        fh, fw = frame_shape[:2]
        area_frac = R.quad_area(quad) / float(max(1.0, fh * fw))
        if area_frac > 0.90:
            score *= max(0.0, 1.0 - (area_frac - 0.90) / 0.10)
        span_x = (quad[:, 0].max() - quad[:, 0].min()) / max(1.0, fw)
        span_y = (quad[:, 1].max() - quad[:, 1].min()) / max(1.0, fh)
        if span_x > 0.85 and span_y > 0.85:
            score *= 0.35          # the whole viewport is not a card
        if max(span_x, span_y) > 0.90 and ratio < 0.62:
            # a band clipped by the frame edge: segmentation handed us the
            # table *around* the card (or a lighting seam), not the card
            score *= FRAME_CLIP_STRIP_PENALTY
    return float(np.clip(score, 0.0, 1.0))


def score_candidate(frame_gray: np.ndarray, quad: np.ndarray,
                    edge_support: float = 0.0) -> dict[str, float]:
    """Tilt-independent evidence for one quad candidate."""
    w, h = R.quad_side_lengths(quad)
    aspect = (w / h) if h > 0 else 1.0
    if aspect > 1.0:                      # landscape reading of a portrait card
        aspect = 1.0 / aspect
    # 0.716 portrait; wide tolerance (sleeves, keystone, occlusion)
    aspect_fit = float(np.clip(1.0 - abs(aspect - R.CARD_ASPECT) / 0.30, 0.0, 1.0))
    structure = _interior_structure(frame_gray, quad)
    contrast = _border_contrast(frame_gray, quad)
    return {
        "edge_support": float(np.clip(edge_support, 0.0, 1.0)),
        "aspect_fit": aspect_fit,
        "structure": structure,
        "contrast": contrast,
        "plausibility": shape_plausibility(quad, frame_gray.shape[:2]),
    }


def _confidence(evidence: dict[str, float], area_frac: float,
                cardness: float | None = None) -> float:
    """Combine evidence into 0..1 detection confidence.

    Aspect ratio is a *weak* term (0.10): the pipeline must not need a
    rectangular card to see one. Edge support and interior structure carry the
    decision; the learned cardness head (when available) refines it.
    """
    conf = (
        0.34 * evidence["edge_support"]
        + 0.24 * evidence["structure"]
        + 0.16 * evidence["contrast"]
        + 0.10 * evidence["aspect_fit"]
        + 0.16 * float(np.clip(math.sqrt(max(area_frac, 0.0)) / 0.5, 0, 1))
    )
    if cardness is not None:
        conf = 0.65 * conf + 0.35 * cardness
    # shape gate: never a hard reject (occlusion distorts shape badly), but a
    # strip or a frame-filling blob must fall well below real cards
    plausibility = float(evidence.get("plausibility", 1.0))
    conf *= 0.30 + 0.70 * plausibility
    return float(np.clip(conf, 0.0, 1.0))


def _drop_redundant(dets: list[CardDetection]) -> list[CardDetection]:
    """Remove candidates that are explained by better candidates.

    Two systematic failure modes of proposal generators, both solved here by
    scanning in descending confidence and dropping anything already explained:

    * **inner features** — the art box, rules-text box or P/T box of a card
      produces its own card-shaped contour; it is contained in a higher
      confidence candidate at <80% of its area, so it is dropped.
    * **container quads** — two adjacent cards merge into one wide contour
      (aspect-wise it can still look card-like). It contains *several* smaller
      candidates, so it is dropped in favour of the individual cards.
    """
    area = {id(d): R.quad_area(d.quad) for d in dets}
    kept: list[CardDetection] = []
    for det in sorted(dets, key=lambda d: -d.confidence):
        a = area[id(det)]
        if a <= 0:
            continue
        redundant = False
        for k in kept:
            ka = area[id(k)]
            ratio = a / ka if ka > 0 else 1.0
            if R.quad_iou(det.quad, k.quad) > 0.55:
                redundant = True
                det.notes.append(f"overlaps {k.source} candidate — dropped")
                break
            if ratio < 0.85 and _quad_contains(k.quad, det.quad):
                redundant = True
                det.notes.append("inside a better candidate — dropped")
                break
            # a much larger, *less structured* blob whose area is mostly
            # occupied by a real card is background/shadow around that card
            if (DROP_BACKGROUND_BLOBS and ratio > 1.6
                    and _quad_contains(det.quad, k.quad, margin=0.72)
                    and det.interior_structure
                    <= k.interior_structure + BLOB_STRUCTURE_MARGIN):
                redundant = True
                det.notes.append("background blob around a card — dropped")
                break
        if redundant:
            continue
        # container test: am I just the union of two or more kept cards?
        contained_area = sum(area[id(k)] for k in kept
                             if _quad_contains(det.quad, k.quad))
        if len(kept) >= 2 and contained_area >= 0.55 * a:
            det.notes.append("container of smaller cards — dropped")
            continue
        kept.append(det)
    return kept


def _quad_contains(container: np.ndarray, inner: np.ndarray,
                   margin: float = 0.85) -> bool:
    """True when `inner`'s centroid lies well inside `container`."""
    c = inner.mean(axis=0)
    centre = container.mean(axis=0)
    scaled = centre + (container - centre) * margin
    hull = R.order_corners(scaled)

    def _side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    signs = [_side(hull[i], hull[(i + 1) % 4], c) for i in range(4)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def _nms(dets: list[CardDetection], iou: float = NMS_IOU) -> list[CardDetection]:
    """Greedy non-maximum suppression over quad IoU."""
    order = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[CardDetection] = []
    for det in order:
        if all(R.quad_iou(det.quad, k.quad) < iou for k in kept):
            kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# ONNX detector (YOLO / RT-DETR) — optional accelerator
# ---------------------------------------------------------------------------
class OnnxCardDetector:
    """YOLO-style ONNX detector adapter (optional; classical is the floor).

    Expected export: input `images` NCHW float32 0..1 (letterboxed), output
    either `[N, 6]` (x0,y0,x1,y1,conf,cls) or `[1, N, 6]`. OBB exports with a
    rotation term are accepted as `[N, 7]` (…, angle) → the box is rotated back
    into a quad. Nothing here is mandatory: without onnxruntime or a trained
    model file the caller silently falls back to the classical detector.
    """

    def __init__(self, model_path: str | Path, conf: float = 0.25,
                 input_size: tuple[int, int] = (640, 640)) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.input_size = input_size
        self.session = None
        self._load()

    @property
    def available(self) -> bool:
        return self.session is not None

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:  # pragma: no cover - optional dep
            return
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(self.model_path),
                                            providers=providers)

    def detect(self, frame: np.ndarray) -> list[CardDetection]:
        if self.session is None:
            return []
        h, w = frame.shape[:2]
        iw, ih = self.input_size
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (iw, ih), swapRB=True,
                                     crop=False)
        name = self.session.get_inputs()[0].name
        out = self.session.run(None, {name: blob})[0]
        out = np.squeeze(out)
        if out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.T                      # [N, C]
        sx, sy = w / float(iw), h / float(ih)
        dets: list[CardDetection] = []
        for row in np.atleast_2d(out):
            if row.shape[0] < 6:
                continue
            score = float(row[4])
            if score < self.conf:
                continue
            if row.shape[0] >= 7:            # oriented box (cx, cy, w, h, a)
                cx, cy, bw, bh, ang = [float(v) for v in row[:5]]
                cx, cy, bw, bh = cx * sx, cy * sy, bw * sx, bh * sy
                rect = ((cx, cy), (bw, bh), math.degrees(ang))
                quad = R.ensure_portrait(R.order_corners(
                    cv2.boxPoints(rect).astype(np.float32)))
            else:
                x0, y0, x1, y1 = [float(v) for v in row[:4]]
                bbox = (x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
                quad = R.ensure_portrait(R.order_corners(
                    R.quad_from_bbox(bbox)))
            dets.append(CardDetection(
                quad=quad, confidence=score,
                area_frac=R.quad_area(quad) / float(w * h),
                tilt_deg=R.estimate_tilt_deg(quad), source="onnx"))
        return dets


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
class CardDetector:
    """Detect every card polygon in a frame, at any angle.

    Parameters
    ----------
    model_path : optional YOLO/RT-DETR ONNX model. When it loads, its boxes are
        merged with the classical proposals (the union is strictly better than
        either alone early on, and the learned detector dominates once trained).
    cardness   : use the learned cardness head when its weights exist.
    refine     : sub-pixel quad refinement (recommended; +accuracy, ~2 ms).
    """

    def __init__(self, model_path: str | Path | None = None,
                 conf: float = MIN_CONFIDENCE, cardness: bool = True,
                 refine: bool = True, max_dim: int = REFINE_MAX_DIM) -> None:
        self.conf = conf
        self.refine = refine
        self.max_dim = max_dim
        self.onnx = OnnxCardDetector(model_path, conf=conf) if model_path else None
        self.cardness_scorer = CardnessScorer() if cardness else None
        self.last_timings: dict[str, float] = {}

    @property
    def backend(self) -> str:
        if self.onnx is not None and self.onnx.available:
            return "onnx+classical" if self.onnx else "onnx"
        return "classical"

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, score_cardness: bool = True
               ) -> list[CardDetection]:
        """All card candidates in `frame`, best first."""
        if not CV_AVAILABLE or frame is None or frame.size == 0:
            return []
        t0 = time.time()
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_dim / float(max(h, w)))
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames_small = frame if scale == 1.0 else cv2.resize(
            frame, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)

        proposals: list[tuple[np.ndarray, float, str]] = []
        for quad, fill in _edge_quads(frames_small, scale):
            proposals.append((quad, fill, "edge"))
        for quad, solidity in _foreground_quads(frames_small, 1.0):
            proposals.append((quad * (1.0 / scale) if scale != 1.0 else quad,
                              solidity, "foreground"))
        if self.onnx is not None and self.onnx.available:
            for det in self.onnx.detect(frame):
                proposals.append((det.quad, 1.0, "onnx"))
        t_prop = time.time()

        # deduplicate proposals cheaply before the expensive scoring
        merged: list[tuple[np.ndarray, float, str]] = []
        for quad, quality, source in proposals:
            if R.quad_area(quad) < 0.5 * MIN_AREA_FRAC * frame.shape[0] * frame.shape[1]:
                continue
            dup = False
            for i, (mq, mq_quality, _s) in enumerate(merged):
                if R.quad_iou(quad, mq) > 0.6:
                    dup = True
                    if quality > mq_quality:
                        merged[i] = (quad, quality, source)
                    break
            if not dup:
                merged.append((quad, quality, source))

        dets: list[CardDetection] = []
        for quad, _quality, source in merged:
            support = 0.0
            if self.refine:
                refined, support = R.refine_quad(gray, quad * scale)
                refined = refined / scale
                if R.quad_iou(refined, quad) > 0.45:
                    quad = R.ensure_portrait(R.order_corners(refined))
            area_frac = R.quad_area(quad) / float(frame.shape[0] * frame.shape[1])
            if area_frac > MAX_AREA_FRAC:
                continue
            evidence = score_candidate(gray, quad * scale, support)
            cardness = None
            if (score_cardness and self.cardness_scorer is not None
                    and self.cardness_scorer.available and area_frac > 0.02):
                try:
                    rect = R.rectify(frame, quad, size=(192, 268), refine=False)
                    cardness = self.cardness_scorer.score(rect.image)
                except Exception:  # pragma: no cover - defensive
                    cardness = None
            conf = _confidence(evidence, area_frac, cardness)
            if conf < self.conf:
                continue
            det = CardDetection(
                quad=quad, confidence=conf, area_frac=area_frac,
                tilt_deg=R.long_axis_angle(quad),
                edge_support=evidence["edge_support"],
                interior_structure=evidence["structure"],
                border_contrast=evidence["contrast"],
                cardness=cardness if cardness is not None else 0.0,
                source=source,
            )
            det.notes.append(f"plausibility {evidence['plausibility']:.2f}")
            if area_frac < 0.02:
                det.notes.append("very small in frame")
            dets.append(det)
        t_score = time.time()

        dets = _drop_redundant(dets)
        dets = _nms(dets)
        self.last_timings = {
            "proposeMs": round((t_prop - t0) * 1000, 2),
            "scoreMs": round((t_score - t_prop) * 1000, 2),
            "totalMs": round((time.time() - t0) * 1000, 2),
            "proposals": float(len(merged)),
        }
        return dets

    # ------------------------------------------------------------------
    def detect_and_rectify(self, frame: np.ndarray, size: tuple[int, int] = (
            R.CARD_W, R.CARD_H)) -> list[R.RectifiedCard]:
        """Detection + perspective correction in one call (the common path)."""
        out: list[R.RectifiedCard] = []
        for det in self.detect(frame):
            try:
                rect = R.rectify(frame, det.quad, size=size, refine=False)
            except Exception:  # pragma: no cover - defensive
                continue
            rect.edge_support = det.edge_support
            out.append(rect)
        return out

    def detect_dicts(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.detect(frame)]


# module-level convenience (mirrors the legacy detect_card_region API)
_DEFAULT: CardDetector | None = None


def default_detector() -> CardDetector:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CardDetector()
    return _DEFAULT


def detect_cards(frame: np.ndarray, **kwargs: Any) -> list[CardDetection]:
    """Module-level helper: detect cards with the shared default detector."""
    return default_detector().detect(frame, **kwargs)
