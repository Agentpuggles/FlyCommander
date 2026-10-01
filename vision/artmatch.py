"""FlyCommander — artwork similarity matching (visual recognition signal).

OCR is the primary identification signal, but real tables produce cards whose
small text is unreadable while the ARTWORK is still recognizable. This module
provides the second, text-independent signal:

    rectified card → artwork crop → average-hash (aHash)
    Scryfall card art (cached on disk) → aHash
    similarity = 1 − hamming(aHash_a, aHash_b) / bits      (0..1)

Design rules:
  * art images are cached under data/art_cache/ — one download per printing
  * no network at all when allow_network is False (fully offline tables)
  * score() returns None (never raises) when art is unavailable — the
    identifier simply falls back to OCR-only confidence
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

try:  # same optional-dep guard as the rest of the vision stack
    import cv2  # type: ignore
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

ART_CACHE_DIR = Path("data/art_cache")
_HASH_BITS = 64          # 8x8 average hash
# artwork zone of a canonical (492x688) rectified card — the illustration box
ART_CROP = (0.10, 0.09, 0.90, 0.53)   # x0, y0, x1, y1 (fractions)


def average_hash(img: "np.ndarray", size: int = 8) -> "np.ndarray | None":
    """8x8 average hash of a grayscale image (bool array, size*size bits)."""
    if cv2 is None or img is None or img.size == 0:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return small > float(small.mean())


def hamming(h1: "np.ndarray", h2: "np.ndarray") -> int:
    return int(np.count_nonzero(h1 != h2))


def art_crop(card_img: "np.ndarray") -> "np.ndarray | None":
    """The illustration zone of a canonical rectified card."""
    h, w = card_img.shape[:2]
    x0, y0, x1, y1 = ART_CROP
    crop = card_img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    return crop if crop.size else None


def artwork_similarity(card_img: "np.ndarray", ref_art: "np.ndarray | None"
                       ) -> float | None:
    """0..1 similarity between a rectified card's art and a reference image.

    BOTH sides are cropped to the illustration zone before hashing — the
    reference is expected to be a FULL card image (Scryfall's normal URI),
    exactly like the rectified capture. None when either side is unusable.
    """
    if ref_art is None or card_img is None or card_img.size == 0:
        return None

    def _hash_of(img):
        crop = art_crop(img)
        return average_hash(crop if crop is not None and crop.size else img)

    a = _hash_of(card_img)
    b = _hash_of(ref_art)
    if a is None or b is None:
        return None
    return 1.0 - hamming(a, b) / float(_HASH_BITS)


class ArtMatcher:
    """Reference-art provider + similarity scorer with a disk cache."""

    def __init__(self, cache_dir: str | Path | None = None,
                 allow_network: bool = True) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else ART_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network

    # ------------------------------------------------------------------
    def _cache_path(self, set_code: str, collector_number: str) -> Path:
        safe = f"{set_code}_{collector_number}".replace("/", "_")
        return self.cache_dir / f"{safe}.jpg"

    def _download(self, url: str) -> "np.ndarray | None":
        if not (self.allow_network and url):
            return None
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "FlyCommander/1.0 (artwork matching)",
                "Accept": "image/*",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

    def reference_art(self, info: Any) -> "np.ndarray | None":
        """Cached-or-downloaded grayscale Scryfall art for one printing.

        `info` needs .set_code, .collector_number, .image_uris (URL string).
        """
        if cv2 is None or info is None:
            return None
        path = self._cache_path(info.set_code, info.collector_number)
        if path.exists():
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        url = getattr(info, "image_uris", "") or ""
        img = self._download(url)
        if img is not None:
            try:
                cv2.imwrite(str(path), img)
            except Exception:
                pass
        return img

    def score(self, card_img: "np.ndarray | None", info: Any) -> float | None:
        """Artwork similarity for a rectified card vs a Scryfall printing."""
        if card_img is None or card_img.size == 0:
            return None
        ref = self.reference_art(info)
        return artwork_similarity(card_img, ref)
