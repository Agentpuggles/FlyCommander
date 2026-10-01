"""FlyCommander physical-table mode — card registration (mtgscan-style).

The strongest idea in mtgscan: identify cards by OCR-ing the small
*collector number / set code* line in the bottom-left of a modern card
(e.g. "216/281 FDN") and doing an exact Scryfall lookup — rather than
OCR-ing stylized card names. Language-agnostic, reprint-accurate.

Registration is a deliberate, one-time-per-card action (not per-frame):

    camera frame → card region → rectify → collector-info crop
      → Tesseract OCR → parse "num/num SET" → Scryfall cache lookup
      → PhysicalObject with persistent tracking ID

If OpenCV/Tesseract are unavailable, `register_from_crop` degrades to a
manual fallback (user types set + collector number) so the game can always
proceed — the hybrid philosophy applies to identification too.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from physical.scryfall_cache import CardInfo, ScryfallCache, normalize_set_code

# mtgscan prints e.g. "216/281 FDN" or "216 (281) FDN" bottom-left.
_COLLECTOR_RE = re.compile(
    r"([0-9]{1,3}[A-Za-z]?)\s*(?:[/•\.]\s*|\(\s*)\s*([0-9]{1,3}[A-Za-z]?)\s*\)?"
    r"\s*([A-Za-z]{2,6})?")
_SET_ONLY_RE = re.compile(r"\b([A-Za-z]{3})\b\s*$")
_NUM_ONLY_RE = re.compile(r"^\s*([0-9]{1,3}[A-Za-z]?)\b")


@dataclass
class RegistrationResult:
    success: bool
    tracking_id: str | None = None
    name: str | None = None
    set_code: str | None = None
    collector_number: str | None = None
    card_type: str | None = None
    base_power: float | None = None
    base_toughness: float | None = None
    confidence: float = 0.0
    source: str = "ocr"          # ocr | manual | cache
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success, "trackingId": self.tracking_id,
            "name": self.name, "set": self.set_code,
            "collectorNumber": self.collector_number,
            "cardType": self.card_type, "basePower": self.base_power,
            "baseToughness": self.base_toughness,
            "confidence": round(self.confidence, 3), "source": self.source,
            "error": self.error,
        }


class CardRegistrar:
    def __init__(self, cache: ScryfallCache) -> None:
        self.cache = cache
        self._tess_ok: bool | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def register_from_frame(self, frame, bbox: tuple[float, float, float, float],
                            use_network: bool = True) -> RegistrationResult:
        """Full pipeline on one camera frame (OpenCV + Tesseract). Degrades
        to a clear failure (use manual registration) if CV is unavailable."""
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError:
            return RegistrationResult(
                success=False,
                error="OpenCV not installed; use register_manual()")

        x, y, w, h = [int(v) for v in bbox]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return RegistrationResult(success=False, error="empty crop")
        return self.register_from_crop(crop, use_network=use_network)

    def register_from_crop(self, crop, use_network: bool = True
                           ) -> RegistrationResult:
        """OCR the collector-info region of a dewarped card crop."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return RegistrationResult(
                success=False,
                error="OpenCV not installed; use register_manual()")

        parsed = self._ocr_collector_info(crop, cv2, np)
        if parsed is None:
            return RegistrationResult(
                success=False,
                error="could not read collector info; use register_manual()")
        set_code, number, conf = parsed
        return self.register_manual(set_code, number,
                                    use_network=use_network, ocr_conf=conf)

    def register_manual(self, set_code: str, collector_number: str,
                        use_network: bool = True,
                        ocr_conf: float = 1.0) -> RegistrationResult:
        """Exact lookup by set + collector number (mtgscan's key step)."""
        set_code = normalize_set_code(set_code)
        number = str(collector_number).strip().upper()
        info = self.cache.get(set_code, number, use_network=use_network)
        if info is None:
            return RegistrationResult(
                success=False, set_code=set_code, collector_number=number,
                error=f"no Scryfall card for {set_code} #{number}"
                      + (" (offline?)" if not self.cache.allow_network else ""))
        return RegistrationResult(
            success=True, name=info.name, set_code=info.set_code,
            collector_number=info.collector_number,
            card_type=info.card_type, base_power=info.base_power,
            base_toughness=info.base_toughness,
            confidence=ocr_conf, source="scryfall")

    # ------------------------------------------------------------------
    # OCR internals (mtgscan-inspired)
    # ------------------------------------------------------------------
    def _ocr_collector_info(self, crop, cv2, np):
        """Crop the bottom-left strip, preprocess, Tesseract, parse."""
        if self._tess_ok is None:
            try:
                import pytesseract  # noqa: F401
                self._tess_ok = True
            except ImportError:
                self._tess_ok = False
        if not self._tess_ok:
            return None

        h, w = crop.shape[:2]
        # bottom-left ~55% x 12% strip where "216/281 FDN" lives
        strip = crop[int(h * 0.86):int(h * 0.98), 0:int(w * 0.55)]
        if strip.size == 0:
            return None
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0,
                          interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        import pytesseract
        text = pytesseract.image_to_string(
            th, config="--psm 7 -c tessedit_char_whitelist="
                       "0123456789/().ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        text = text.strip()
        for pattern in (_COLLECTOR_RE, _SET_ONLY_RE, _NUM_ONLY_RE):
            m = pattern.search(text)
            if m:
                groups = m.groups()
                number = groups[0]
                set_code = next((g for g in reversed(groups) if g and g.isalpha()), "")
                if set_code:
                    return set_code, number, 0.85
        return None


def parse_collector_line(text: str) -> tuple[str, str] | None:
    """Parse 'num/num SET' style text → (set, number); unit-testable.

    Handles OCR mush like '216/281FDN' where the set's first letter glues
    onto the denominator: the stray letter is moved back onto the set code."""
    m = _COLLECTOR_RE.search(text)
    if m:
        number = m.group(1)
        denom = m.group(2) or ""
        set_code = next((g for g in reversed(m.groups()) if g and g.isalpha()), "")
        # '281F' + 'DN' → number 281, set 'FDN'
        if set_code and len(set_code) < 3 and denom and denom[-1].isalpha() \
                and len(denom) > 1 and number != denom:
            set_code = denom[-1] + set_code
        if set_code and len(set_code) >= 2:
            return normalize_set_code(set_code), number.upper()
        if not set_code:
            return "", number.upper()
    m = _NUM_ONLY_RE.search(text)
    if m:
        return "", m.group(1).upper()
    return None
