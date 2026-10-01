"""FlyCommander — geometric card correction (perspective, rotation, glare).

This is the stage between "the detector says there is a card over there" and
"here is a normalized 492x688 image of it". It never rejects a card for being
tilted, rotated, sleeved or partly occluded: it produces the *best possible*
normalized image plus a confidence and a validity mask that downstream
recognition uses to ignore regions it cannot trust (glare, occlusion, out of
frame).

Why this module exists separately from detection: rectification is pure
geometry and must be exactly testable. Given a quad, the warp is
deterministic; given a warp, the error is measurable (we test it against
synthetic scenes with known ground-truth corners).

Public API
----------
order_corners(pts)                 -> tl, tr, br, bl ordering (rotation-safe)
refine_quad(gray, quad)            -> sub-pixel corners from image gradients
quad_from_bbox(...)                -> fallback quad when only a box exists
quad_visible_fraction(quad, shape) -> how much of the card is inside the frame
estimate_tilt_deg(quad)            -> in-plane rotation (0 = upright portrait)
rectify(frame, quad)               -> (card_bgr, mask) canonical + validity
rectify_batch(...)                 -> stacked cards for batched embedding
card_quality(card, mask)           -> glare/sharpness/coverage diagnostics
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

try:  # optional heavy dep — same guard pattern as the rest of the vision stack
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

# Canonical card size (MTG 63x88 mm). 492x688 is ~7.8 px/mm and matches the
# legacy physical/card_scan.py geometry so old OCR crops still line up.
CARD_W, CARD_H = 492, 688
CARD_ASPECT = CARD_W / CARD_H                    # 0.715 portrait

# Feature zones (fractions of the canonical card) — single source of truth
# for every consumer (embeddings, matcher, OCR fallback, debug UI).
ZONES: dict[str, tuple[float, float, float, float]] = {
    "border": (0.00, 0.00, 1.00, 1.00),
    "title": (0.05, 0.030, 0.95, 0.105),
    "mana": (0.60, 0.030, 0.95, 0.105),
    "art": (0.075, 0.105, 0.925, 0.530),
    "type": (0.05, 0.545, 0.95, 0.595),
    "text": (0.075, 0.600, 0.925, 0.900),
    "pt": (0.68, 0.900, 0.93, 0.965),
    "collector": (0.02, 0.940, 0.55, 0.988),
    "setsymbol": (0.86, 0.940, 0.99, 0.995),
}


# ---------------------------------------------------------------------------
# corner geometry (pure numpy — unit-testable without OpenCV)
# ---------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl for a *portrait* card.

    Rotation-invariant: works for any in-plane rotation, keystone, or the
    tapped (90 degree) orientation. The only ambiguity a polygon cannot
    resolve is 180 degrees — a card and its own half-turn have identical
    geometry — so recognition resolves that (the matcher scores both the
    crop and its 180 flip), while this function guarantees a deterministic
    tl/tr/br/bl ordering modulo 180 for everything else.

    Method: sort cyclically by centroid angle, then decide which *pair of
    opposite edges* is the long (vertical) one. For a portrait card
    tl→tr is the short top edge, so the winding starts at the corner where
    a short edge turns into a long edge with clockwise (image-space) sense.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {pts.shape[0]}")
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    quad = pts[np.argsort(ang)]

    # Keep a clockwise (image-space, y-down) winding: positive cross product.
    edge_a = quad[1] - quad[0]
    edge_b = quad[2] - quad[0]
    cross = float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    if cross < 0:
        quad = quad[[0, 3, 2, 1]]

    lengths = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
               for i in range(4)]

    # Two possible "vertical edge" pairings: (0,2) vs (1,3).
    def pairing_quality(first: int) -> tuple[float, float]:
        """(asymmetry, portrait_bonus) for the pairing starting at edge `first`."""
        a, b = lengths[first], lengths[(first + 2) % 4]
        c, d = lengths[(first + 1) % 4], lengths[(first + 3) % 4]
        long_avg = (a + b) / 2.0
        short_avg = (c + d) / 2.0
        if long_avg <= 1e-3 or short_avg <= 1e-3:
            return 1.0, 0.0
        asymmetry = abs(a - b) / max(a, b) + abs(c - d) / max(c, d)
        ratio = short_avg / long_avg
        # a portrait card has a short/long ratio well below 1 (0.716 nominal);
        # penalise pairings that would rectify the card sideways
        bonus = max(0.0, 1.0 - ratio / 0.90)
        return asymmetry, bonus

    best_start, best_key = 0, None
    for start in (0, 1):
        asymmetry, bonus = pairing_quality(start)
        # lower asymmetry is good; being portrait is worth up to ~1.0
        key = asymmetry - 0.9 * bonus
        if best_key is None or key < best_key:
            best_key, best_start = key, start
    quad = np.roll(quad, -best_start, axis=0)
    # After the roll, edge quad[0]→quad[1] is one of the *long* (vertical)
    # edges. tl/tr/br/bl wants the SHORT (top) edge there, so rotate once
    # more: quad[0]→quad[1] becomes the top edge and quad[3]→quad[0] the left.
    quad = np.roll(quad, -1, axis=0)
    return quad.astype(np.float32)


def quad_side_lengths(quad: np.ndarray) -> tuple[float, float]:
    """(top width, left height) of an ordered quad."""
    quad = np.asarray(quad, np.float32)
    w = float((np.linalg.norm(quad[1] - quad[0])
               + np.linalg.norm(quad[2] - quad[3])) / 2.0)
    h = float((np.linalg.norm(quad[3] - quad[0])
               + np.linalg.norm(quad[2] - quad[1])) / 2.0)
    return w, h


def is_portrait(quad: np.ndarray) -> bool:
    """True when the ordered quad reads as a portrait card (taller than wide)."""
    w, h = quad_side_lengths(quad)
    if w <= 0 or h <= 0:
        return True
    return h >= w


def ensure_portrait(quad: np.ndarray) -> np.ndarray:
    """Rotate the corner order so the card is portrait (rotation-safe)."""
    if is_portrait(quad):
        return np.asarray(quad, np.float32)
    q = np.asarray(quad, np.float32)
    return q[[1, 2, 3, 0]]


def estimate_tilt_deg(quad: np.ndarray) -> float:
    """In-plane rotation of a portrait quad: 0 = upright, 90 = tapped.

    Cards are overwhelmingly observed in [0, 180) — a card is
    indistinguishable from its own 180 rotation without reading content, so
    the angle is reported modulo 180 (the tap detector interprets the
    orientation timeline, not a single frame).
    """
    q = np.asarray(quad, np.float32)
    top = q[1] - q[0]
    ang = math.degrees(math.atan2(top[1], top[0]))
    return float(ang % 180.0)


def long_axis_angle(quad: np.ndarray) -> float:
    """Angle of the card's long axis from vertical, in [0, 180).

    0 means the card stands upright (untapped), ~90 means it lies sideways
    (tapped). This is the *physical* orientation of the card on the table and
    is independent of how `order_corners` chose to label the corners, which is
    why tap/orientation tracking uses this and not the ordered quad.
    """
    q = np.asarray(quad, np.float32)
    if q.shape != (4, 2):
        return 0.0
    lengths = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    if not lengths:
        return 0.0
    long_edge = np.asarray(q[(lengths.index(max(lengths)) + 1) % 4] - q[lengths.index(max(lengths))], np.float32)
    ang = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180.0
    # angle of the long axis from vertical (90° in the vector sense = upright)
    return float(abs(ang - 90.0) % 180.0)


def quad_from_bbox(bbox: Sequence[float]) -> np.ndarray:
    """(x, y, w, h) → four corners (legacy detector compatibility)."""
    x, y, w, h = [float(v) for v in bbox[:4]]
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    np.float32)


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of a quad (unsigned)."""
    q = np.asarray(quad, np.float32)
    x, y = q[:, 0], q[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def quad_visible_fraction(quad: np.ndarray, shape: Iterable[int]) -> float:
    """Fraction of the card quad that lies inside the frame."""
    h, w = int(shape[0]), int(shape[1])
    area = quad_area(quad)
    if area <= 0:
        return 0.0
    q = np.asarray(quad, np.float32)
    cx = np.clip(q[:, 0], 0, w - 1)
    cy = np.clip(q[:, 1], 0, h - 1)
    clipped = np.stack([cx, cy], axis=1)
    return float(min(1.0, quad_area(clipped) / area))


def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int] | None = None
             ) -> float:
    """Intersection-over-union of two quads (rasterised on a small canvas)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    x0 = min(a[:, 0].min(), b[:, 0].min())
    y0 = min(a[:, 1].min(), b[:, 1].min())
    x1 = max(a[:, 0].max(), b[:, 0].max())
    y1 = max(a[:, 1].max(), b[:, 1].max())
    span = max(x1 - x0, y1 - y0)
    if span <= 0:
        return 0.0
    scale = 256.0 / span
    w, h = int((x1 - x0) * scale) + 4, int((y1 - y0) * scale) + 4

    def _mask(q):
        canvas = np.zeros((h, w), np.uint8)
        pts = ((q - np.array([x0, y0], np.float32)) * scale + 2).astype(np.int32)
        if CV_AVAILABLE:
            cv2.fillPoly(canvas, [pts], 1)
        else:  # pragma: no cover - tiny fallback
            for px, py in pts:
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = 1
        return canvas

    ma, mb = _mask(a), _mask(b)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# sub-pixel refinement
# ---------------------------------------------------------------------------
def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Total-least-squares line fit; returns (point, unit direction)."""
    if points.shape[0] < 3:
        return None
    mean = points.mean(axis=0)
    centred = points - mean
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
    direction = vt[0]
    return mean.astype(np.float32), direction.astype(np.float32)


def _intersect(p1, d1, p2, d2) -> np.ndarray | None:
    """Intersection of two lines given point + direction."""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float32)
    det = float(np.linalg.det(a))
    if abs(det) < 1e-6:
        return None
    t = np.linalg.solve(a, p2 - p1)
    return (p1 + t[0] * d1).astype(np.float32)


def refine_quad(gray: np.ndarray, quad: np.ndarray, search_px: int = 4,
                samples: int = 48) -> tuple[np.ndarray, float]:
    """Snap quad edges to image gradients, then re-intersect → sub-pixel corners.

    For each of the four edges we sample points, search along the edge normal
    for the strongest gradient (the card border), fit a robust line and
    intersect neighbouring lines. Returns (refined_quad, edge_support) where
    edge_support is the fraction of perimeter samples that found a strong
    gradient — a tilt-independent card-likeness score.

    Never raises: on any failure the input quad is returned with support 0.
    """
    if not CV_AVAILABLE or gray is None or gray.size == 0:
        return np.asarray(quad, np.float32), 0.0
    gray = np.asarray(gray)
    if gray.ndim == 3:                     # accept a colour frame as well
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2:
        return np.asarray(quad, np.float32), 0.0
    quad = np.asarray(quad, np.float32).reshape(-1, 2)
    h, w = gray.shape[:2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    support_hits = 0
    support_total = 0
    lines: list[tuple[np.ndarray, np.ndarray] | None] = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 8:
            lines.append(None)
            continue
        direction = edge / length
        normal = np.array([-direction[1], direction[0]], np.float32)
        pts: list[np.ndarray] = []
        for t in np.linspace(0.08, 0.92, samples):
            base = p0 + edge * float(t)
            best_v, best_m = 0.0, 0.0
            for off in np.linspace(-search_px, search_px, 2 * search_px + 1):
                px = base + normal * float(off)
                xi, yi = int(round(px[0])), int(round(px[1]))
                if 0 <= xi < w and 0 <= yi < h:
                    value = float(mag[yi, xi])
                    if value > best_m:
                        best_m, best_v = value, float(off)
            support_total += 1
            if best_m > 25.0:
                support_hits += 1
                pts.append(base + normal * best_v)
        if len(pts) >= 6:
            # robust refit: drop the worst 20% residuals, fit again
            arr = np.stack(pts)
            fit = _fit_line(arr)
            if fit is not None:
                mean, direction2 = fit
                rel = arr - mean
                resid = np.abs(direction2[0] * rel[:, 1]
                               - direction2[1] * rel[:, 0])
                keep = resid <= np.quantile(resid, 0.8)
                if int(keep.sum()) >= 4:
                    fit = _fit_line(arr[keep])
            lines.append(fit)
        else:
            lines.append(None)

    support = support_hits / support_total if support_total else 0.0
    corners: list[np.ndarray] = []
    ok = True
    for i in range(4):
        prev_fit = lines[(i - 1) % 4]
        cur_fit = lines[i]
        if prev_fit is None or cur_fit is None:
            ok = False
            break
        pt = _intersect(prev_fit[0], prev_fit[1], cur_fit[0], cur_fit[1])
        if pt is None:
            ok = False
            break
        corners.append(pt)
    if not ok:
        return quad, support
    refined = np.stack(corners).astype(np.float32)
    # sanity: refined quad must stay within 12% of the detected size
    if abs(quad_area(refined) - quad_area(quad)) > 0.25 * quad_area(quad):
        return quad, support
    return refined, support


def refine_quad_cv(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """cv2.cornerSubPix refinement on top of the gradient fit (best effort)."""
    if not CV_AVAILABLE:
        return np.asarray(quad, np.float32)
    pts = np.asarray(quad, np.float32).reshape(-1, 1, 2).copy()
    h, w = gray.shape[:2]
    if pts[:, :, 0].max() < 2 or pts[:, :, 1].max() < 2:
        return np.asarray(quad, np.float32)
    pts[:, :, 0] = np.clip(pts[:, :, 0], 1, w - 2)
    pts[:, :, 1] = np.clip(pts[:, :, 1], 1, h - 2)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          20, 0.02))
    except Exception:  # pragma: no cover - OpenCV edge cases
        return np.asarray(quad, np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# rectification
# ---------------------------------------------------------------------------
@dataclass
class RectifiedCard:
    """Normalized card image + everything downstream needs to trust it."""

    image: np.ndarray                            # canonical BGR card
    mask: np.ndarray                             # uint8, 255 = usable pixels
    quad: np.ndarray                             # source quad in the frame
    tilt_deg: float                              # long-axis angle (0=upright)
    source_tilt_deg: float                       # ordered-quad rotation
    area_frac: float                             # card area / frame area
    edge_support: float                          # gradient support of the quad
    visible_frac: float                          # fraction inside the frame
    glare_frac: float = 0.0                      # fraction of blown-out pixels
    sharpness: float = 0.0                       # Laplacian variance of the crop
    quality: float = 0.0                         # combined 0..1
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of canonical pixels that carry real (unmasked) content."""
        if self.mask.size == 0:
            return 0.0
        return float(np.count_nonzero(self.mask) / self.mask.size)

    def zone(self, name: str) -> np.ndarray:
        """Crop a named zone (title, art, collector...) from the card."""
        return crop_zone(self.image, name)

    def to_dict(self, include_image: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "quad": [[round(float(x), 2), round(float(y), 2)]
                     for x, y in self.quad],
            "tiltDeg": round(self.tilt_deg, 2),
            "areaFrac": round(self.area_frac, 4),
            "edgeSupport": round(self.edge_support, 3),
            "visibleFrac": round(self.visible_frac, 3),
            "glareFrac": round(self.glare_frac, 3),
            "sharpness": round(self.sharpness, 1),
            "quality": round(self.quality, 3),
            "coverage": round(self.coverage, 3),
            "notes": list(self.notes),
        }
        return d


def crop_zone(card: np.ndarray, zone: str) -> np.ndarray:
    """Crop a normalized zone out of a canonical card image."""
    h, w = card.shape[:2]
    x0, y0, x1, y1 = ZONES[zone]
    return card[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def glare_mask(card: np.ndarray, threshold: int = 248,
               min_blob_px: int = 24) -> np.ndarray:
    """Boolean mask of blown-out (specular) pixels on a rectified card.

    The threshold sits at true saturation (248+): the white title/text boxes
    of a normal card are ~235-245 and must NOT be masked out, while lamp
    reflections and foil hotspots clip to 250-255.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
    mask = (gray >= threshold).astype(np.uint8)
    if min_blob_px > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
                keep[labels == i] = 1
        mask = keep
    return mask.astype(bool)


def rectify(frame: np.ndarray, quad: np.ndarray | Sequence[float] | dict,
            size: tuple[int, int] = (CARD_W, CARD_H),
            refine: bool = True, valid_mask: bool = True,
            border_pad: float = 0.0) -> RectifiedCard:
    """Warp a detected quad to the canonical card rectangle.

    Accepts a 4x2 quad, a legacy (x, y, w, h) bbox, or a dict carrying either
    (`corners` / `bboxFrame`). Produces the card image, an occlusion/glare
    validity mask, and the diagnostics recognition needs.

    A card is never rejected here: heavy perspective is corrected, heavy
    occlusion reduces coverage, heavy glare reduces usable pixels. The caller
    decides what to do with a low-quality rectification (and even a 40%-glare
    card can still be identified from its artwork).
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for rectification")
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    H, W = frame.shape[:2]

    raw = quad
    if isinstance(raw, dict):
        for key in ("corners", "quad", "bboxFrame"):
            if raw.get(key) is not None:
                raw = raw[key]
                break
        else:
            raise ValueError("rectify got a dict without corners/bboxFrame")
    raw = np.asarray(raw, np.float32)
    if raw.size == 4 and raw.shape != (4, 2):
        quad_pts = quad_from_bbox(raw.reshape(-1))
    else:
        quad_pts = raw.reshape(-1, 2)
    if quad_pts.shape[0] != 4:
        raise ValueError("rectify needs four corners or an (x, y, w, h) bbox")

    physical_angle = long_axis_angle(quad_pts)
    ordered = ensure_portrait(order_corners(quad_pts))
    source_tilt = estimate_tilt_deg(ordered)
    edge_support = 0.0
    gray_full = None
    if refine:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        snap_gray = gray_full
        scale = 1.0
        max_dim = max(H, W)
        if max_dim > 1400:      # refine on a downscaled image for speed
            scale = max_dim / 1400.0
            snap_gray = cv2.resize(gray_full, (int(W / scale), int(H / scale)),
                                   interpolation=cv2.INTER_AREA)
        candidate, edge_support = refine_quad(snap_gray, ordered / scale)
        if candidate is not ordered:
            candidate = candidate * scale
        candidate = ensure_portrait(order_corners(candidate))
        if quad_iou(candidate, ordered) > 0.5:
            ordered = candidate
            if scale == 1.0:
                ordered = ensure_portrait(order_corners(
                    refine_quad_cv(gray_full, ordered)))

    # pad by a hair so the card border is included after the warp
    if border_pad > 0:
        centre = ordered.mean(axis=0)
        ordered = centre + (ordered - centre) * (1.0 + border_pad)

    dst = np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1],
                    [0, size[1] - 1]], np.float32)
    M = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
    card = cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    mask = np.full(size[::-1], 255, np.uint8) if valid_mask else np.zeros(
        size[::-1], np.uint8)
    notes: list[str] = []
    visible = quad_visible_fraction(ordered, (H, W))
    if visible < 0.999:
        notes.append("card extends past the frame edge")
        # pixels warped from outside the frame are not trustworthy
        src_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(src_mask, [ordered.astype(np.int32)], 255)
        warped_valid = cv2.warpPerspective(src_mask, M, size,
                                           flags=cv2.INTER_NEAREST)
        mask = np.minimum(mask, warped_valid)

    glare = glare_mask(card) if valid_mask else np.zeros(size[::-1], bool)
    glare_frac = float(glare.mean()) if glare.size else 0.0
    if glare_frac > 0.02:
        notes.append(f"{glare_frac * 100:.0f}% glare")
        mask[glare] = 0
    gray_card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray_card, cv2.CV_64F).var())
    area_frac = quad_area(ordered) / float(W * H)

    if sharpness < 30:
        notes.append("soft focus")
    scale_w = math.sqrt(quad_area(ordered)) if quad_area(ordered) > 0 else 0.0
    if scale_w < 90:
        notes.append("small in frame")

    # combined 0..1 quality: sharpness, usable pixels, edge support, size
    q_sharp = float(np.clip(sharpness / 320.0, 0, 1))
    q_cover = float(np.clip(1.0 - glare_frac, 0, 1)) * visible
    q_edge = float(np.clip(edge_support / 0.85, 0, 1))
    q_size = float(np.clip(math.sqrt(max(area_frac, 0.0)) / 0.35, 0, 1))
    quality = 0.40 * q_sharp + 0.25 * q_cover + 0.20 * q_edge + 0.15 * q_size

    return RectifiedCard(image=card, mask=mask, quad=ordered,
                         tilt_deg=physical_angle, source_tilt_deg=source_tilt,
                         area_frac=area_frac, edge_support=edge_support,
                         visible_frac=visible, glare_frac=glare_frac,
                         sharpness=sharpness, quality=quality, notes=notes)


def rectify_batch(frame: np.ndarray, quads: Sequence[np.ndarray],
                  size: tuple[int, int] = (CARD_W, CARD_H)
                  ) -> list[RectifiedCard]:
    """Rectify several quads from one frame (no refinement; batched use)."""
    return [rectify(frame, q, size=size, refine=False) for q in quads]


def normalized_small(card: RectifiedCard | np.ndarray, size: tuple[int, int] = (128, 179)
                     ) -> np.ndarray:
    """Small BGR copy of a rectified card (embedder input / UI thumbnails)."""
    img = card.image if isinstance(card, RectifiedCard) else card
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def warp_debug_overlay(frame: np.ndarray, rect: RectifiedCard,
                       colour: tuple[int, int, int] = (80, 220, 80)
                       ) -> np.ndarray:
    """Draw the detected quad + corners on the frame (debug UI)."""
    out = frame.copy()
    pts = rect.quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, colour, 2)
    for i, (x, y) in enumerate(rect.quad.astype(int)):
        cv2.circle(out, (int(x), int(y)), 4, colour, -1)
        cv2.putText(out, "tl tr br bl".split()[i], (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out
