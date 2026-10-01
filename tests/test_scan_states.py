"""Server-level scan-state mapping tests.

Pins the scan_frames() state contract — especially that bad_angle is a
card-detected, retryable coaching state (status "retry", card_detected True),
never a "no card" style failure. Uses a real PhysicalTableApp against tmp_path
with the analysis stage monkeypatched (no camera, no network, no OCR).
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F841  (server scan path needs OpenCV)

from physical.server import PhysicalTableApp          # noqa: E402
from vision.card_analysis import (                    # noqa: E402
    MSG_BAD_ANGLE, MSG_MULTIPLE, MSG_NO_CARD, MSG_TOO_SMALL,
)


@pytest.fixture()
def app(tmp_path):
    return PhysicalTableApp(data_dir=tmp_path, allow_network=False)


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _images() -> list[str]:
    ok, buf = cv2.imencode(".jpg", _frame())
    assert ok
    return [base64.b64encode(buf.tobytes()).decode()]


def _patch_analysis(monkeypatch, result: dict) -> None:
    """Replace analyze_for_scan as scan_frames imports it (source module)."""
    def _fake(frame, ocr_fn=None):
        return dict(result)
    monkeypatch.setattr("vision.card_analysis.analyze_for_scan", _fake)


def _candidate(conf: float = 0.582) -> dict:
    return {"bboxFrame": [10, 20, 100, 140], "confidence": conf,
            "tiltDeg": 40.0, "areaFrac": 0.10, "aspect": 0.95,
            "reasons": ["glare near threshold"]}


# ---------------------------------------------------------------------------
# bad_angle is LEGACY-ONLY: the pipeline perspective-corrects instead of
# rejecting. The defensive server branch must never tell the player to
# flatten a card (a tabletop observer, not a document scanner).
# ---------------------------------------------------------------------------

def test_legacy_bad_angle_never_asks_player_to_flatten(app, monkeypatch):
    _patch_analysis(monkeypatch, {
        "state": "bad_angle", "card_detected": True, "confidence": 0.582,
        "reason": MSG_BAD_ANGLE, "best_candidate": _candidate(),
        "candidates": [_candidate()], "ocr": None,
    })
    r = app.scan_frames(_images())
    assert r["state"] == "bad_angle"
    assert r["status"] == "retry"                    # NOT "nocard"
    assert r["card_detected"] is True                # NOT False
    assert "flatten" not in r["message"].lower()
    assert "correcting perspective" in r["message"].lower()
    assert r["confidence"] == pytest.approx(0.582)


# ---------------------------------------------------------------------------
# the red family keeps its honest failure semantics
# ---------------------------------------------------------------------------

def test_no_card_reports_no_card(app, monkeypatch):
    _patch_analysis(monkeypatch, {
        "state": "no_card", "card_detected": False, "confidence": 0.0,
        "reason": MSG_NO_CARD, "candidates": [], "ocr": None,
    })
    r = app.scan_frames(_images())
    assert r["state"] == "no_card"
    assert r["status"] == "nocard"
    assert r["card_detected"] is False


def test_too_small_is_detected_but_unscannable(app, monkeypatch):
    _patch_analysis(monkeypatch, {
        "state": "too_small", "card_detected": True, "confidence": 0.61,
        "reason": MSG_TOO_SMALL, "best_candidate": _candidate(),
        "candidates": [_candidate()], "ocr": None,
    })
    r = app.scan_frames(_images())
    assert r["state"] == "too_small"
    assert r["status"] == "nocard"
    assert r["card_detected"] is True
    assert r["analysis"]["state"] == "too_small"     # analysis passed through


def test_multiple_cards_is_detected_but_unscannable(app, monkeypatch):
    _patch_analysis(monkeypatch, {
        "state": "multiple_cards", "card_detected": True, "confidence": 0.7,
        "reason": MSG_MULTIPLE, "best_candidate": _candidate(0.7),
        "candidates": [_candidate(0.7), _candidate(0.6)], "ocr": None,
    })
    r = app.scan_frames(_images())
    assert r["state"] == "multiple_cards"
    assert r["status"] == "nocard"
    assert r["card_detected"] is True


# ---------------------------------------------------------------------------
# ocr_failed keeps the debug payload (a card exists, after all)
# ---------------------------------------------------------------------------

def test_ocr_failed_returns_crops_and_evidence(app, monkeypatch):
    _patch_analysis(monkeypatch, {
        "state": "ocr_failed", "card_detected": True, "confidence": 0.66,
        "reason": "Card detected, but text cannot be read.",
        "best_candidate": _candidate(0.66), "candidates": [_candidate(0.66)],
        "ocr": {"name": {"raw": "???", "normalized": "", "confidence": 0.1},
                "collector": {"raw": "", "set": "", "number": "",
                              "confidence": 0.0}},
    })
    r = app.scan_frames(_images())
    assert r["state"] == "ocr_failed"
    assert r["card_detected"] is True
    assert r["rectifiedCard"].startswith("data:image")   # debug payload kept
    assert r["evidence"]["nameRaw"] == "???"
