"""FlyCommander physical-table mode — card scanning (CV layer).

Success-first scanning: the camera frame is rectified to a full normalized
card, and *two independent* OCR regions are attempted — the card name
(primary signal) and the collector/set line (secondary). Neither region's
failure fails registration; evidence flows to `physical/identifier.py`.

Everything here degrades gracefully: if OpenCV/Tesseract are missing the
module reports `available=False` and the UI falls back to manual entry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Normalized card dimensions (MTG 63×88 mm at ~7.8 px/mm)
CARD_W, CARD_H = 492, 688
CARD_ASPECT = CARD_W / CARD_H

try:  # optional heavy deps — the rest of the app works without them
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

try:  # pragma: no cover
    import pytesseract  # type: ignore

    TESS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore
    TESS_AVAILABLE = False

SCAN_AVAILABLE = CV_AVAILABLE and TESS_AVAILABLE


def tesseract_ready() -> tuple[bool, str]:
    """(is_usable, version_or_reason) for the OCR rescue path.

    `TESS_AVAILABLE` only means the Python wrapper imports — the tesseract
    *binary* is a separate install. Scanning cannot tell which is missing,
    so the table reports the real reason at startup instead of silently
    falling back to "not identified".
    """
    if not TESS_AVAILABLE:
        return False, "pytesseract not installed"
    try:
        return True, str(pytesseract.get_tesseract_version())
    except Exception as exc:                      # binary missing / not on PATH
        return False, f"tesseract binary unusable ({type(exc).__name__})"

# Tesseract char whitelists CAN contain spaces through pytesseract: it builds
# the command line with `shlex.split(config)`, so a *quoted* whitelist keeps
# its space (verified against pytesseract 0.3.x source). Name OCR therefore
# runs constrained — which is what research/computer_vision/ocr_and_detection.md
# recommends, because an unconstrained read of the stylized Beleren name line
# happily invents digits and punctuation out of background texture.
_NAME_WHITELIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789 '-,.!&/:*"
)
_COLLECTOR_WHITELIST = "0123456789/().ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _ocr_config(psm: int, whitelist: str | None) -> str:
    """Tesseract config string; the whitelist is quoted so spaces survive."""
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f' -c tessedit_char_whitelist="{whitelist}"'
    return cfg


# ---------------------------------------------------------------------------
# frame quality
# ---------------------------------------------------------------------------
@dataclass
class FrameQuality:
    sharpness: float          # Laplacian variance (higher = sharper)
    brightness: float         # mean luminance 0-255
    contrast: float           # luminance stddev
    score: float              # combined 0-1 usability score
    hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sharpness": round(self.sharpness, 1),
                "brightness": round(self.brightness, 1),
                "contrast": round(self.contrast, 1),
                "score": round(self.score, 3),
                "hints": list(self.hints)}


def score_frame_quality(frame: "np.ndarray") -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    hints: list[str] = []
    if sharpness < 60:
        hints.append("Image is blurry — hold the card steady")
    if brightness < 45:
        hints.append("More light needed")
    elif brightness > 215:
        hints.append("Too much light or glare")
    if contrast < 25:
        hints.append("Low contrast — reduce glare or add light")

    # combined 0-1: sharpness is the dominant factor
    s = min(1.0, sharpness / 400.0)
    b = 1.0 if 60 <= brightness <= 200 else max(0.0, 1.0 - abs(brightness - 130) / 130.0)
    c = min(1.0, contrast / 60.0)
    return FrameQuality(sharpness, brightness, contrast,
                        score=0.6 * s + 0.25 * b + 0.15 * c, hints=hints)


def pick_best_frame(frames: list["np.ndarray"]) -> tuple[int, "np.ndarray",
                                                          FrameQuality]:
    """Choose the sharpest, best-exposed frame from a burst."""
    scored = [(i, f, score_frame_quality(f)) for i, f in enumerate(frames)]
    scored.sort(key=lambda t: t[2].score, reverse=True)
    return scored[0]


# ---------------------------------------------------------------------------
# card detection + rectification
# ---------------------------------------------------------------------------
def detect_card_region(frame: "np.ndarray",
                       with_center: bool = False):
    """Find the largest card-like quadrilateral; return (rectified card,
    orientation_degrees) — or additionally the frame-space center when
    `with_center` — or None.

    Delegates candidate scoring to vision/card_analysis.py (MTG aspect
    0.716, confidence + reasons); kept as the legacy single-card API.
    """
    scale = 640.0 / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * 0.04:          # too small to be a held card
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if not 1.18 <= aspect <= 1.75:        # MTG ≈ 1.396 (perspective-tolerant)
            continue
        if area > best_area:
            best_area = area
            best = (c, rect)

    if best is None:
        return None
    c, ((cx, cy), (w, h), angle) = best

    # normalize orientation: card portrait = long side vertical
    if w > h:
        angle += 90.0
        w, h = h, w
    # box points → ordered corners in the small image, then rescale
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    src = _order_corners(box * (1.0 / scale))
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1],
                    [0, CARD_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    card = cv2.warpPerspective(frame, M, (CARD_W, CARD_H))
    orientation = (90.0 - angle) % 180.0       # 0 = upright portrait
    if with_center:
        cx_f, cy_f = cx / scale, cy / scale
        return card, float(orientation), (float(cx_f), float(cy_f))
    return card, float(orientation)


def _order_corners(pts: "np.ndarray") -> "np.ndarray":
    """tl, tr, br, bl ordering."""
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    rotated = pts[order]
    # rotate so the top-left-most point (smallest x+y) comes first
    sums = rotated.sum(axis=1)
    start = int(np.argmin(sums))
    return np.roll(rotated, -start, axis=0)


# ---------------------------------------------------------------------------
# OCR regions
# ---------------------------------------------------------------------------
def extract_name_crop(card: "np.ndarray") -> "np.ndarray":
    h, w = card.shape[:2]
    return card[int(h * 0.035):int(h * 0.145), int(w * 0.06):int(w * 0.94)]


def extract_collector_crop(card: "np.ndarray") -> "np.ndarray":
    h, w = card.shape[:2]
    return card[int(h * 0.945):int(h * 0.995), int(w * 0.03):int(w * 0.52)]


def _prep_variants(crop: "np.ndarray") -> list["np.ndarray"]:
    """A few preprocessing variants — normal, sharpened+thresholded."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    out = [gray]
    big = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (0, 0), 2.0)
    sharp = cv2.addWeighted(big, 1.7, blur, -0.7, 0)
    out.append(sharp)
    _, th = cv2.threshold(sharp, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th)
    return out


def _run_ocr(img: "np.ndarray", psm: int,
             whitelist: str | None = None) -> tuple[str, float]:
    cfg = _ocr_config(psm, whitelist)
    try:
        data = pytesseract.image_to_data(img, config=cfg,
                                         output_type=pytesseract.Output.DICT)
    except Exception:
        return "", 0.0
    words, confs = [], []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = text.strip()
        if text and conf >= 0:
            words.append(text)
            confs.append(float(conf))
    text = " ".join(words)
    mean_conf = float(np.mean(confs)) / 100.0 if confs else 0.0
    return text, mean_conf


_NAME_NOISE = re.compile(r"[^A-Za-z '\-]")


def normalize_name(raw: str) -> str:
    """OCR garbage → plausible card-name text."""
    s = _NAME_NOISE.sub(" ", raw)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ocr_card_regions(card: "np.ndarray") -> dict[str, Any]:
    """Independent OCR of name + collector regions.

    Returns raw + normalized text and confidence per region; a region may
    simply be empty — that is not an error.
    """
    name_raw, name_conf = "", 0.0
    for variant in _prep_variants(extract_name_crop(card)):
        for psm in (7, 6):
            text, conf = _run_ocr(variant, psm, _NAME_WHITELIST)
            norm = normalize_name(text)
            if len(norm) > len(normalize_name(name_raw)) and len(norm) >= 3:
                name_raw, name_conf = text, conf
    if len(normalize_name(name_raw)) < 3:
        # A whitelist costs a legitimate character on the odd glyph (accented
        # names, unusual promo frames). If it left us with nothing readable,
        # retry unconstrained rather than drop the name signal entirely.
        for variant in _prep_variants(extract_name_crop(card)):
            text, conf = _run_ocr(variant, 7)
            norm = normalize_name(text)
            if len(norm) > len(normalize_name(name_raw)) and len(norm) >= 3:
                name_raw, name_conf = text, conf

    col_raw, col_conf = "", 0.0
    for variant in _prep_variants(extract_collector_crop(card)):
        for psm in (7, 6):
            text, conf = _run_ocr(variant, psm, _COLLECTOR_WHITELIST)
            if len(text.strip()) > len(col_raw):
                col_raw, col_conf = text, conf

    from physical.registration import parse_collector_line
    parsed = parse_collector_line(col_raw)
    set_code, number = parsed if parsed else ("", "")

    return {
        "name": {"raw": name_raw, "normalized": normalize_name(name_raw),
                 "confidence": round(name_conf, 3)},
        "collector": {"raw": col_raw, "set": set_code,
                      "number": number, "confidence": round(col_conf, 3)},
    }


def encode_jpeg_b64(img: "np.ndarray | None", max_w: int = 360,
                    quality: int = 70) -> str | None:
    """Small JPEG data-URL for UI debug views (None-safe)."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
