"""Artwork-similarity tests (vision/artmatch.py) — synthetic, offline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2", reason="OpenCV required for hash math")

from vision import artmatch as am  # noqa: E402


class _FakeInfo:
    """Duck-typed stand-in for CardInfo (set, number, art URL)."""

    def __init__(self, set_code="FDN", number="216", uri=""):
        self.set_code = set_code
        self.collector_number = number
        self.image_uris = uri


def _art_from_levels(levels, block_w=47, block_h=37):
    """300x380 art built from an 8x8 grid of gray levels — coarse structure
    that survives aHash downsampling (pure noise collapses to a flat mean)."""
    art = np.zeros((300, 380), np.uint8)
    for i in range(8):
        for j in range(8):
            art[i * block_h:(i + 1) * block_h,
                j * block_w:(j + 1) * block_w] = levels[i][j]
    return art


def _levels(seed=0, invert=False):
    rng = np.random.default_rng(seed)
    lv = rng.integers(30, 225, (8, 8), dtype=np.uint8)
    return (255 - lv) if invert else lv


def _card_image(levels=None):
    """Canonical-size card with a distinctive art block."""
    img = np.full((688, 492, 3), 230, np.uint8)
    art = _art_from_levels(levels if levels is not None else _levels(seed=3))
    img[70:370, 55:435, 0] = art
    img[70:370, 55:435, 1] = art
    img[70:370, 55:435, 2] = art
    return img


def test_average_hash_is_deterministic_and_64bit():
    h = am.average_hash(_card_image())
    assert h is not None and h.size == 64
    assert np.array_equal(h, am.average_hash(_card_image()))


def test_identical_art_scores_high():
    img = _card_image()
    s = am.artwork_similarity(img, img)
    assert s is not None and s > 0.90


def test_different_art_scores_lower():
    s = am.artwork_similarity(_card_image(_levels(seed=3)),
                              _card_image(_levels(seed=3, invert=True)))
    assert s is not None and s < 0.90


def test_art_crop_isolates_the_illustration_zone():
    img = np.zeros((688, 492, 3), np.uint8)
    img[100:300, 100:300] = 255                      # inside ART_CROP
    crop = am.art_crop(img)
    assert crop is not None and crop.size > 0
    assert crop.mean() > 50                          # the white block is in


def test_unavailable_reference_returns_none():
    s = am.artwork_similarity(_card_image(), None)
    assert s is None
    assert am.artwork_similarity(None, _card_image()) is None


def test_reference_art_reads_from_disk_cache(tmp_path, monkeypatch):
    """Pre-seed the cache with a real image; no network needed."""
    import cv2 as _cv2
    img = _card_image()
    ref = am.ArtMatcher(cache_dir=tmp_path, allow_network=False)
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    _cv2.imwrite(str(tmp_path / "FDN_216.jpg"), gray)
    out = ref.reference_art(_FakeInfo())
    assert out is not None
    s = am.artwork_similarity(img, out)
    assert s is not None and s > 0.80


def test_reference_art_offline_without_cache_is_none(tmp_path):
    ref = am.ArtMatcher(cache_dir=tmp_path, allow_network=False)
    assert ref.reference_art(_FakeInfo(uri="http://x/y.jpg")) is None


def test_score_is_none_safe(tmp_path):
    ref = am.ArtMatcher(cache_dir=tmp_path, allow_network=False)
    assert ref.score(None, _FakeInfo()) is None
    assert ref.score(_card_image(), None) is None
