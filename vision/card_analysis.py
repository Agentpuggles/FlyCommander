"""FlyCommander — MTG-aware card scene analysis (vision layer).

Answers, in order, the questions a scanning table must get RIGHT before any
OCR runs:

    1. Is there a card-shaped object at all?          (presence)
    2. Is it big enough / flat enough to read?        (geometry)
    3. Is the image sharp and lit well enough?        (quality)
    4. Only then: can we read the text?               (OCR)

The single invariant of this module: **"OCR failure" is only ever reported
when a card was actually detected.** A bare desk never produces
"unreadable text" — it produces "No Magic card detected."

Failure states (mutually exclusive, in pipeline order):
    no_card        nothing card-shaped in view
    multiple_cards more than one plausible card candidate
    too_small      card candidate far from the camera
    bad_quality    frame too blurry / too dark / blown out / heavy glare
    ocr_failed     card present, quality fine, text unreadable
    card_detected  everything fine (ANY tilt/rotation — see below)

There is deliberately NO "bad_angle" rejection: this is a tabletop
observer, not a document scanner. A tilted/rotated card triggers
perspective correction (four-corner homography to the canonical 63x88
rectangle) and then reads exactly like a flat card. `analysis["perspective"]`
reports what was corrected; extreme tilts merely lower confidence.

MTG card geometry: 63 x 88 mm → aspect 0.716 (portrait). Sleeves, mild
rotation and perspective change this by a few percent; the classifier
tolerates that via ASPECT_TOLERANCE and the near/wide bands below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physical.card_scan import (
    CV_AVAILABLE,
    FrameQuality,
    score_frame_quality,
)

try:  # optional heavy dep (same guard pattern as physical/card_scan.py)
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

# --------------------------------------------------------------------------
# MTG geometry constants
# --------------------------------------------------------------------------
CARD_MM_W, CARD_MM_H = 63.0, 88.0
MTG_CARD_ASPECT = CARD_MM_W / CARD_MM_H          # ≈ 0.716 portrait
MTG_CARD_ASPECT_INV = 1.0 / MTG_CARD_ASPECT      # ≈ 1.397 landscape measure

# Fraction of the frame's area a card must occupy to be readable.
# A 1920x1080 overhead shot of a table: a playable card ≳ 4–6% of frame.
MIN_CARD_AREA_FRAC = 0.04          # below → "Move the card closer."
GOOD_CARD_AREA_FRAC = 0.08         # above → geometry contributes full confidence

# Floor at which a card-shaped blob is even *considered* a candidate (below
# this it is noise). Between the floor and MIN_CARD_AREA_FRAC the pipeline
# reports "too_small" rather than "no_card" — that distinction is the point.
CANDIDATE_AREA_FRAC = 0.012

# Aspect tolerance: sleeves add ~1–2mm per edge, perspective foreshortens.
ASPECT_TOLERANCE = 0.14            # |aspect/0.716 − 1| within ±14% → "matches"
ASPECT_NEAR_TOL = 0.28             # within ±28% → "possible card" (uncertain)

# Tilt handling: a tabletop observer, not a document scanner — tilt NEVER
# rejects a card. Up to this angle the four-corner homography still produces
# readable OCR; beyond it confidence degrades but the pipeline still tries.
MAX_USABLE_TILT_DEG = 55.0
MAX_TILT_FOR_FULL_CONFIDENCE = 10.0

# Glare: fraction of the card region that is blown-out white
MAX_GLARE_FRAC = 0.30

# Blur (Laplacian variance) below which OCR on the rectified card is hopeless
MIN_READABLE_SHARPNESS = 45.0


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------
@dataclass
class CardCandidate:
    """One card-shaped region with MTG-specific scores."""
    bbox_frame: tuple[float, float, float, float]   # x, y, w, h (full-res px)
    area_frac: float
    aspect: float                     # w/h of the oriented rect (portrait-normalized)
    tilt_deg: float                   # rotation of the rect from upright
    fill_ratio: float                 # contour area / rect area (1.0 = clean quad)
    glare_frac: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    corners: list[tuple[float, float]] = field(default_factory=list)
    # ^ four full-res corner points (tl, tr, br, bl) when available — the
    #   source for perspective correction (empty = axis-aligned bbox only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bboxFrame": [round(v, 1) for v in self.bbox_frame],
            "areaFrac": round(self.area_frac, 4),
            "aspect": round(self.aspect, 3),
            "tiltDeg": round(self.tilt_deg, 1),
            "fillRatio": round(self.fill_ratio, 3),
            "glareFrac": round(self.glare_frac, 3),
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "corners": [[round(x, 1), round(y, 1)] for x, y in self.corners],
        }


@dataclass
class CardAnalysis:
    """Full scene verdict. `state` is one of the module-level states."""
    state: str
    card_detected: bool
    confidence: float
    reason: str
    candidates: list[CardCandidate] = field(default_factory=list)
    quality: FrameQuality | None = None
    ocr: dict[str, Any] | None = None      # present ONLY when a card exists
    perspective: str = "unknown"           # "flat" | "corrected" | "unknown"

    @property
    def best_candidate(self) -> CardCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "card_detected": self.card_detected,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
            "quality": self.quality.to_dict() if self.quality else None,
            "ocr": self.ocr,
            "perspective": self.perspective,
            "multiple_cards": len(self.candidates) > 1,
            "best_candidate": (self.best_candidate.to_dict()
                               if self.best_candidate else None),
        }


# user-facing instruction strings (single source of truth for UI + tests)
MSG_NO_CARD = "No Magic card detected. Place one card in view."
MSG_TOO_SMALL = "Move the card closer."
MSG_BLUR = "Hold still / improve focus."
MSG_LIGHTING = "Improve lighting."
MSG_OCR_FAIL = "Card detected, but text cannot be read."
MSG_MULTIPLE = "Multiple cards detected. Scan one card at a time."
MSG_QUALITY = "Image quality low."
# Kept only for the server's defensive legacy branch — the pipeline itself
# never emits a bad_angle rejection anymore.
MSG_BAD_ANGLE = "Card detected at an angle."
MSG_PERSPECTIVE = "Card detected at an angle — perspective corrected."


# --------------------------------------------------------------------------
# pure scoring helpers (no cv2 — unit-testable anywhere)
# --------------------------------------------------------------------------
def aspect_verdict(aspect: float) -> tuple[str, float]:
    """Classify a portrait-normalized w/h ratio against MTG dimensions.

    Returns (verdict, aspect_confidence) where verdict is one of
    'match' | 'near' | 'off'.
    """
    if aspect <= 0:
        return "off", 0.0
    rel_dev = abs(aspect / MTG_CARD_ASPECT - 1.0)
    if rel_dev <= ASPECT_TOLERANCE:
        return "match", 1.0 - 0.4 * (rel_dev / ASPECT_TOLERANCE)
    if rel_dev <= ASPECT_NEAR_TOL:
        return "near", 0.45
    return "off", 0.0


def score_candidate_geometry(area_frac: float, aspect: float, tilt_deg: float,
                             fill_ratio: float,
                             glare_frac: float = 0.0) -> tuple[float, list[str]]:
    """Confidence (0..1) + reasons for one candidate's geometry.

    Tilt reduces confidence smoothly but NEVER gates the pipeline —
    perspective correction handles it (see module docstring).
    """
    reasons: list[str] = []
    conf = 0.0

    verdict, a_conf = aspect_verdict(aspect)
    if verdict == "match":
        conf += 0.45 * a_conf
        reasons.append("rectangle matches MTG dimensions (63x88mm)")
    elif verdict == "near":
        conf += 0.20
        reasons.append("aspect near MTG dimensions but distorted")
    else:
        reasons.append(f"aspect {aspect:.2f} does not match MTG 0.716")

    # area: closer to GOOD_CARD_AREA_FRAC+ → better
    if area_frac >= GOOD_CARD_AREA_FRAC:
        conf += 0.25
    elif area_frac >= MIN_CARD_AREA_FRAC:
        conf += 0.25 * (area_frac / GOOD_CARD_AREA_FRAC)
        reasons.append("card is small in frame")
    else:
        reasons.append("card is too far from the camera")

    # tilt: degrades linearly toward MAX_USABLE_TILT_DEG; never gates
    tilt = min(abs(tilt_deg), 180.0 - abs(tilt_deg)) if tilt_deg else 0.0
    if tilt <= MAX_TILT_FOR_FULL_CONFIDENCE:
        conf += 0.15
    elif tilt <= MAX_USABLE_TILT_DEG:
        conf += 0.15 * (1.0 - (tilt - MAX_TILT_FOR_FULL_CONFIDENCE)
                        / (MAX_USABLE_TILT_DEG - MAX_TILT_FOR_FULL_CONFIDENCE))
        reasons.append(f"tilted {tilt:.0f}° (perspective corrected)")
    else:
        conf += 0.02
        reasons.append(f"extreme tilt {tilt:.0f}° — attempting correction")

    # fill: a clean quadrilateral fills most of its min-area rect
    conf += 0.10 * max(0.0, min(1.0, fill_ratio))
    if fill_ratio < 0.6:
        reasons.append("edges are incomplete or occluded")

    if glare_frac > MAX_GLARE_FRAC:
        reasons.append("heavy glare on card surface")
        conf *= 0.7

    return min(1.0, conf), reasons


def _order_corners(pts: "np.ndarray") -> "np.ndarray":
    """Order four points as tl, tr, br, bl (pure numpy, no cv2)."""
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    rotated = pts[np.argsort(angles)]
    start = int(np.argmin(rotated.sum(axis=1)))
    return np.roll(rotated, -start, axis=0)


def is_ocr_failure(ocr: dict[str, Any] | None) -> bool:
    """True iff OCR *ran* on a detected card and produced nothing usable.

    Deliberately False when `ocr` is None/empty — i.e. when no card was
    detected, this must never claim an OCR failure.
    """
    if not ocr:
        return False
    name = (ocr.get("name") or {}).get("normalized", "")
    name_conf = float((ocr.get("name") or {}).get("confidence", 0.0))
    collector = (ocr.get("collector") or {})
    has_collector = bool(collector.get("set") and collector.get("number"))
    col_conf = float(collector.get("confidence", 0.0))
    has_text = bool(name) or has_collector
    usable = (len(name) >= 3 and name_conf >= 0.35) or \
             (has_collector and col_conf >= 0.35)
    return has_text and not usable


# --------------------------------------------------------------------------
# cv2-backed detection
# --------------------------------------------------------------------------
def detect_card_candidates(frame: "np.ndarray") -> list[CardCandidate]:
    """All plausible card-quadrilaterals in the frame, scored vs MTG geometry."""
    if not CV_AVAILABLE or frame is None:
        return []
    scale = 640.0 / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    H, W = small.shape[:2]
    frame_area = float(H * W)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # detection-only gain for near-dark scenes: without it, edge detection
    # finds nothing and an unlit-but-present card would be reported as
    # "no card". Quality scoring still uses the RAW frame, so the pipeline
    # correctly reports "improve lighting" afterwards.
    brightness = float(gray.mean())
    if brightness < 60.0:
        gain = min(4.0, 60.0 / max(1.0, brightness))
        gray = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    cands: list[CardCandidate] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * CANDIDATE_AREA_FRAC:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 1 or rh <= 1:
            continue
        # normalize: portrait → aspect = short/long < 1
        if rw < rh:
            aspect, tilt = rw / rh, angle
        else:
            aspect, tilt = rh / rw, angle + 90.0
        tilt = tilt % 180.0
        fill = area / (rw * rh) if rw * rh > 0 else 0.0
        area_frac = area / frame_area

        # glare: fraction of the rect interior that is blown out
        glare = _glare_fraction(small, rect)

        conf, reasons = score_candidate_geometry(area_frac, aspect, tilt,
                                                 fill, glare)
        if area_frac < CANDIDATE_AREA_FRAC:
            continue                    # noise, not even a candidate
        x0 = max(0, int((cx - rw / 2.0) / scale))
        y0 = max(0, int((cy - rh / 2.0) / scale))
        w0 = int(rw / scale)
        h0 = int(rh / scale)
        # four real corners (full-res, ordered tl,tr,br,bl) — the input for
        # perspective correction. boxPoints handles both rotation AND the
        # keystone component of tilt in one rect.
        corners_full = [[float(px) / scale, float(py) / scale]
                        for px, py in cv2.boxPoints(rect)]
        cands.append(CardCandidate(
            bbox_frame=(float(x0), float(y0), float(w0), float(h0)),
            area_frac=area_frac, aspect=aspect, tilt_deg=tilt,
            fill_ratio=fill, glare_frac=glare, confidence=conf,
            reasons=reasons, corners=corners_full))

    cands.sort(key=lambda k: k.confidence, reverse=True)
    # keep only genuinely card-plausible candidates for the multi-card verdict
    return [c for c in cands if c.confidence >= 0.35][:4]


def _glare_fraction(small: "np.ndarray", rect) -> float:
    mask = np.zeros(small.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.intp(cv2.boxPoints(rect))], 255)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    vals = gray[mask > 0]
    if vals.size == 0:
        return 0.0
    return float((vals > 245).sum()) / float(vals.size)


def detect_corners(frame: "np.ndarray",
                   bbox_frame: tuple[float, float, float, float] | None = None
                   ) -> "np.ndarray | None":
    """Four card corners (full-res, tl/tr/br/bl) inside a bbox region.

    Strategy: contour the bbox crop (background suppressed via Canny flood
    fill) and take the min-area quadrilateral. Returns None when no coherent
    card quad can be found. Never raises.
    """
    if not CV_AVAILABLE or frame is None:
        return None
    H, W = frame.shape[:2]
    if bbox_frame is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = [int(v) for v in bbox_frame]
    x, y = max(0, x), max(0, y)
    w, h = min(W - x, w), min(H - y, h)
    if w < 8 or h < 8:
        return None
    try:
        crop = frame[y:y + h, x:x + w]
        scale = 640.0 / max(crop.shape[:2])
        small = cv2.resize(crop, None, fx=scale, fy=scale) if scale < 1.0 else crop
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        crop_area = small.shape[0] * small.shape[1]
        if area < crop_area * 0.10:
            return None                 # no dominant card-shaped blob
            # (a normally-placed card ≈ 10–25% of a full-frame search —
            #  25% would reject perfectly playable placements)
        quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(quad) == 4:
            pts = quad.reshape(-1, 2).astype(np.float32)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        pts = _order_corners(pts)
        pts = _validate_quad(pts, cv2.contourArea(c))
        if pts is None:
            return None
        return pts / scale + np.array([x, y], dtype=np.float32)
    except Exception:
        return None


def _validate_quad(pts: "np.ndarray", contour_area: float
                   ) -> "np.ndarray | None":
    """Reject degenerate ordered quads (self-intersecting / tiny)."""
    if pts is None or len(pts) != 4:
        return None
    quad_area = float(abs(cv2.contourArea(pts.astype(np.float32))))
    if quad_area <= 0 or contour_area <= 0:
        return None
    if quad_area / contour_area < 0.75:   # ordering produced a bowtie
        return None
    return pts


def rectify(frame: "np.ndarray", corners) -> "np.ndarray":
    """Perspective-normalize a card to the canonical 492x688 image.

    Accepts four ordered corners (tl,tr,br,bl) — preferred, real homography —
    or a legacy (x, y, w, h) bbox, or a candidate dict with either. Rotation
    AND keystone perspective are corrected; downstream OCR behaves as if the
    card were flat.
    """
    from physical.card_scan import CARD_W, CARD_H
    H, W = frame.shape[:2]
    if isinstance(corners, dict):
        corners = corners.get("corners") or corners.get("bboxFrame")
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape == (4, 2):
        src = corners
        quad_area = abs(cv2.contourArea(src))
        bbox_area = float((src[:, 0].max() - src[:, 0].min())
                          * (src[:, 1].max() - src[:, 1].min()))
        if bbox_area > 0 and quad_area / bbox_area < 0.5:
            raise ValueError("degenerate corner quad")
        M = cv2.getPerspectiveTransform(src.astype(np.float32), np.array(
            [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1],
             [0, CARD_H - 1]], dtype=np.float32))
        return cv2.warpPerspective(frame, M, (CARD_W, CARD_H),
                                   flags=cv2.INTER_CUBIC)
    # legacy axis-aligned bbox path
    x, y, w, h = [int(v) for v in corners.reshape(-1)[:4]]
    x, y = max(0, x), max(0, y)
    w, h = min(W - x, w), min(H - y, h)
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError("empty rectification crop")
    return cv2.resize(crop, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def analyze_card_candidates(frame: "np.ndarray",
                            quality: FrameQuality | None = None,
                            ocr_fn=None) -> dict[str, Any]:
    """Full decision pipeline for one frame → dict (JSON-friendly).

    `ocr_fn(card_image) -> ocr_dict` is injected so callers decide when (and
    on what image) OCR runs; the pipeline only calls it AFTER a card exists.
    """
    q = quality or score_frame_quality(frame)
    candidates = detect_card_candidates(frame)

    # ---- 1. presence -----------------------------------------------------
    if not candidates:
        return CardAnalysis(
            state="no_card", card_detected=False, confidence=0.0,
            reason=MSG_NO_CARD, candidates=[], quality=q).to_dict()

    # ---- 2. multiple cards ----------------------------------------------
    strong = [c for c in candidates if c.confidence >= 0.55]
    if len(strong) > 1:
        return CardAnalysis(
            state="multiple_cards", card_detected=True,
            confidence=strong[0].confidence, reason=MSG_MULTIPLE,
            candidates=candidates, quality=q).to_dict()

    best = candidates[0]

    # ---- 3. geometry: size gate only — tilt NEVER gates -----------------
    if best.area_frac < MIN_CARD_AREA_FRAC:
        return CardAnalysis(
            state="too_small", card_detected=True, confidence=best.confidence,
            reason=MSG_TOO_SMALL, candidates=candidates, quality=q).to_dict()

    # ---- 4. image quality (incl. per-card glare) -------------------------
    # lighting BEFORE blur: darkness depresses Laplacian variance, so a dark
    # scene would otherwise get the misleading "hold still" advice.
    quality_hints: list[str] = []
    if q.brightness < 45 or q.brightness > 215:
        quality_hints.append(MSG_LIGHTING)
    if q.sharpness < MIN_READABLE_SHARPNESS:
        quality_hints.append(MSG_BLUR)
    if best.glare_frac > MAX_GLARE_FRAC:
        quality_hints.append("reduce glare on the card")
    if q.brightness < 45:
        quality_hints.append("move the card closer to the light")
    if quality_hints:
        return CardAnalysis(
            state="bad_quality", card_detected=True, confidence=best.confidence,
            reason=MSG_QUALITY + " " + " ".join(quality_hints),
            candidates=candidates, quality=q).to_dict()

    # ---- 5. perspective correction + OCR (a card exists at ANY angle) ----
    tilt = min(abs(best.tilt_deg), 180.0 - abs(best.tilt_deg))
    perspective = "corrected" if (tilt > 2.0 or best.corners) else "flat"
    card_detected = best.confidence >= 0.35
    if ocr_fn is not None:
        try:
            src = best.corners or best.bbox_frame
            card_img = rectify(frame, src)
            ocr = ocr_fn(card_img)
        except Exception:
            card_img, ocr = None, None
    else:
        card_img, ocr = None, None

    if is_ocr_failure(ocr):
        return CardAnalysis(
            state="ocr_failed", card_detected=True,
            confidence=best.confidence, reason=MSG_OCR_FAIL,
            candidates=candidates, quality=q, ocr=ocr,
            perspective=perspective).to_dict()

    # successful text reading is itself detection evidence — a card whose
    # OCR came back clean IS detected, whatever the geometry scored
    if ocr:
        name_ok = len((ocr.get("name") or {}).get("normalized", "")) >= 3
        col_ok = bool((ocr.get("collector") or {}).get("set"))
        if name_ok or col_ok:
            card_detected = True

    # report why we accepted an angled card instead of hiding the correction
    if tilt > MAX_TILT_FOR_FULL_CONFIDENCE and perspective == "corrected":
        reason = MSG_PERSPECTIVE
    else:
        reason = (best.reasons[0] if best.reasons
                  else "rectangle matches MTG dimensions")
    state = "card_detected" if card_detected else "correcting_perspective"
    return CardAnalysis(
        state=state, card_detected=card_detected,
        confidence=best.confidence, reason=reason,
        candidates=candidates, quality=q, ocr=ocr,
        perspective=perspective).to_dict()


def analyze_frame(frame: "np.ndarray", quality: FrameQuality | None = None) -> dict[str, Any]:
    """Watcher-loop entry point: analysis WITHOUT OCR (fast, every frame)."""
    return analyze_card_candidates(frame, quality=quality, ocr_fn=None)


# --------------------------------------------------------------------------
# debug overlay
# --------------------------------------------------------------------------
_DEBUG_GREEN = (80, 220, 80)     # BGR — valid card
_DEBUG_YELLOW = (60, 200, 240)   # BGR — uncertain
_DEBUG_RED = (60, 60, 230)       # BGR — no valid card


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "card_detected":
        return _DEBUG_GREEN
    if state in ("card_uncertain", "correcting_perspective",
                 "multiple_cards", "too_small", "bad_angle",
                 "bad_quality", "ocr_failed"):
        return _DEBUG_YELLOW
    return _DEBUG_RED


def annotate_frame(frame: "np.ndarray", analysis: dict[str, Any] | None,
                   camera: dict[str, Any] | None = None) -> "np.ndarray":
    """Debug overlay: candidate outlines (green=valid / yellow=uncertain /
    red=none), state banner, and camera settings burned into the frame."""
    if not CV_AVAILABLE or frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    lw = max(2, w // 640)
    fs = max(0.6, w / 1920.0)
    a = analysis or {"state": "no_analysis", "reason": "no analysis yet",
                     "candidates": []}

    for c in a.get("candidates", []):
        conf = float(c.get("confidence", 0.0))
        cc = (_DEBUG_GREEN if conf >= 0.55
              else _DEBUG_YELLOW if conf >= 0.35 else _DEBUG_RED)
        corners = c.get("corners") or []
        if len(corners) == 4:
            # detected card outline + corner points (the homography source)
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, cc, lw)
            for px, py in corners:
                cv2.circle(out, (int(px), int(py)), lw * 3, cc, -1)
        else:
            x, y, cw, ch = [int(v) for v in c.get("bboxFrame", (0, 0, 0, 0))]
            cv2.rectangle(out, (x, y), (x + cw, y + ch), cc, lw)
        cv2.putText(out, f"{conf:.2f}",
                    (int(corners[0][0]) if corners else 0,
                     max(14, int((corners[0][1] if corners else 0) - 8))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, cc, lw)

    color = _state_color(str(a.get("state", "")))
    banner_h = int(34 * (w / 1280.0) + 18)
    cv2.rectangle(out, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.putText(out, f"{a.get('state', '?')} — {a.get('reason', '')}"[:110],
                (10, banner_h - 12), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1,
                color, lw)
    if camera:
        info = (f"{camera.get('format', '?')} {camera.get('width', '?')}x"
                f"{camera.get('height', '?')} @{camera.get('fps', '?')}"
                f"  measured {camera.get('measuredFps') or 0:.1f} fps")
        cv2.rectangle(out, (0, h - banner_h), (w, h), (30, 30, 30), -1)
        cv2.putText(out, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (220, 220, 220), lw)
    return out


def analyze_for_scan(frame: "np.ndarray", ocr_fn) -> dict[str, Any]:
    """Scan entry point: analysis WITH OCR once geometry+quality pass.

    Returns the same dict as analyze_card_candidates; callers can branch on
    `state` and never need to guess whether OCR actually ran.
    """
    return analyze_card_candidates(frame, quality=None, ocr_fn=ocr_fn)
