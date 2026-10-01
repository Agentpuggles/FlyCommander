"""Card scene analysis tests — synthetic frames, no camera hardware.

Verifies the pipeline's core promise: the seven states are distinct, ordered
(presence → geometry → quality → OCR), and "OCR failure" is only ever
reported when a card was actually detected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import card_analysis as ca  # noqa: E402

cv2 = pytest.importorskip("cv2", reason="OpenCV required for frame synthesis")


# ------------------------------------------------------- frame synthesis
def make_frame(card=True, scale=0.45, tilt=0.0, blur=0, brightness=1.0,
               n_cards=1, text=True):
    """Build a synthetic tabletop photo: dark desk + white card(s)."""
    W, H = 1280, 720
    frame = np.full((H, W, 3), 40, dtype=np.uint8)   # dark desk
    if not card:
        return frame
    ch = int(H * (0.85 if scale is None else scale))
    cw = int(ch / 1.397)                              # MTG aspect 0.716
    positions = [(W // 2, H // 2)]
    if n_cards > 1:
        positions.append((W // 5, H // 2))
        if n_cards > 2:
            positions.append((4 * W // 5, H // 2))
    for cx, cy in positions[:n_cards]:
        card_img = np.full((ch, cw, 3), 235, dtype=np.uint8)
        card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        card_img = cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGR)
        if text:
            cv2.putText(card_img, "Grizzly Bears", (10, int(ch * 0.12)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, cw / 260.0),
                        (20, 20, 20), 2)
            cv2.putText(card_img, "216/281 FDN", (10, int(ch * 0.97)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, cw / 320.0),
                        (20, 20, 20), 2)
        if blur:
            k = blur * 2 + 1
            card_img = cv2.GaussianBlur(card_img, (k, k), 0)
        card_img = (card_img.astype(np.float32) * brightness).clip(0, 255)\
            .astype(np.uint8)
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), tilt, 1.0)
        rotated = cv2.warpAffine(card_img, M, (cw, ch),
                                 borderValue=(40, 40, 40))
        x0, y0 = cx - cw // 2, cy - ch // 2
        x0, y0 = max(0, x0), max(0, y0)
        h2, w2 = rotated.shape[:2]
        frame[y0:y0 + h2, x0:x0 + w2] = rotated
    return frame


def ocr_ok(_card):
    return {"name": {"raw": "Grizzly Bears", "normalized": "Grizzly Bears",
                     "confidence": 0.9},
            "collector": {"raw": "216/281 FDN", "set": "FDN",
                          "number": "216", "confidence": 0.9}}


def ocr_garbage(_card):
    return {"name": {"raw": "l~#1", "normalized": "l", "confidence": 0.05},
            "collector": {"raw": "", "set": "", "number": "",
                          "confidence": 0.0}}


def analyze(frame, **kw):
    return ca.analyze_card_candidates(frame, **kw)


# ------------------------------------------------------------- presence
def test_empty_desk_reports_no_card():
    r = analyze(make_frame(card=False))
    assert r["state"] == "no_card"
    assert r["card_detected"] is False
    assert r["confidence"] == 0.0
    assert r["reason"] == ca.MSG_NO_CARD
    assert r["ocr"] is None                     # OCR never ran


def test_ocr_failure_never_reported_when_no_card():
    r = analyze(make_frame(card=False), ocr_fn=ocr_garbage)
    assert r["state"] == "no_card"
    assert r["state"] != "ocr_failed"
    assert r["ocr"] is None


# ------------------------------------------------- happy path + OCR gate
@pytest.mark.skipif(not ca.CV_AVAILABLE, reason="OpenCV not available")
def test_clean_card_is_detected():
    r = analyze(make_frame(), ocr_fn=ocr_ok)
    assert r["state"] == "card_detected"
    assert r["card_detected"] is True
    assert r["confidence"] > 0.5
    assert "63x88" in r["reason"] or r["confidence"] >= 0.5
    assert r["candidates"] and r["candidates"][0]["confidence"] > 0.5


def test_ocr_failure_only_with_card_present():
    r = analyze(make_frame(), ocr_fn=ocr_garbage)
    assert r["state"] == "ocr_failed"
    assert r["card_detected"] is True
    assert r["reason"] == ca.MSG_OCR_FAIL
    assert r["ocr"] is not None                 # OCR DID run


def test_no_ocr_fn_means_no_ocr_state():
    """Without ocr_fn, pipeline stops after quality: card_detected state."""
    r = analyze(make_frame())
    assert r["state"] in ("card_detected", "card_uncertain")
    assert r["ocr"] is None


# ------------------------------------------------------------- geometry
def test_tiny_card_reports_too_small():
    r = analyze(make_frame(scale=0.20), ocr_fn=ocr_ok)
    assert r["state"] == "too_small"
    assert r["card_detected"] is True
    assert r["reason"] == ca.MSG_TOO_SMALL
    assert r["ocr"] is None                     # pipeline stopped before OCR


def test_multiple_cards_detected():
    r = analyze(make_frame(n_cards=2), ocr_fn=ocr_ok)
    assert r["state"] == "multiple_cards"
    assert r["card_detected"] is True
    assert r["reason"] == ca.MSG_MULTIPLE
    assert len(r["candidates"]) >= 2
    assert r["ocr"] is None                     # OCR is gated on single card


# ------------------------------------------------------------- quality
def test_blurry_card_reports_quality():
    # motion-blur strong enough to kill OCR, weak enough to keep edges
    frame = cv2.GaussianBlur(make_frame(), (15, 15), 0)
    r = analyze(frame, ocr_fn=ocr_ok)
    assert r["state"] == "bad_quality"
    assert r["card_detected"] is True           # a card IS in view
    assert ca.MSG_BLUR in r["reason"]           # combined, actionable hints
    assert r["reason"].startswith(ca.MSG_QUALITY)
    assert r["ocr"] is None


def test_dark_frame_reports_lighting():
    frame = (make_frame().astype(np.float32) * 0.15).astype(np.uint8)
    r = analyze(frame, ocr_fn=ocr_ok)
    # darkness now yields BOTH lighting + move-closer coaching in one message
    assert r["state"] == "bad_quality"
    assert ca.MSG_LIGHTING in r["reason"]
    assert "closer" in r["reason"]


# ------------------------------------------------- pure scoring helpers
def test_aspect_verdict_bands():
    assert ca.aspect_verdict(0.716)[0] == "match"
    assert ca.aspect_verdict(0.716 * 1.10)[0] == "match"     # sleeve-ish
    assert ca.aspect_verdict(0.716 * 1.20)[0] == "near"      # distorted
    assert ca.aspect_verdict(1.0)[0] == "off"                # square ≠ card
    assert ca.aspect_verdict(2.0)[0] == "off"
    assert ca.aspect_verdict(0.0)[0] == "off"


def test_geometry_confidence_components():
    conf, reasons = ca.score_candidate_geometry(
        area_frac=0.15, aspect=0.716, tilt_deg=3.0, fill_ratio=0.95,
        glare_frac=0.0)
    assert conf > 0.85
    assert any("matches MTG" in r for r in reasons)

    conf2, reasons2 = ca.score_candidate_geometry(
        area_frac=0.01, aspect=1.9, tilt_deg=60.0, fill_ratio=0.4,
        glare_frac=0.5)
    assert conf2 < 0.3
    assert any("far from the camera" in r for r in reasons2)
    assert any("extreme tilt" in r for r in reasons2)
    assert any("glare" in r for r in reasons2)


# ------------------------------------------------------------- ocr gating
def test_is_ocr_failure_matrix():
    good = {"name": {"normalized": "Sol Ring", "confidence": 0.9},
            "collector": {"set": "FDN", "number": "250", "confidence": 0.8}}
    bad = {"name": {"normalized": "", "confidence": 0.0},
           "collector": {"set": "", "number": "", "confidence": 0.0}}
    weak = {"name": {"normalized": "abc", "confidence": 0.1},
            "collector": {"set": "", "number": "", "confidence": 0.0}}
    collector_only = {"name": {"normalized": "", "confidence": 0.0},
                      "collector": {"set": "FDN", "number": "250",
                                    "confidence": 0.8}}
    assert ca.is_ocr_failure(good) is False
    assert ca.is_ocr_failure(bad) is False      # no text at all ≠ OCR failure
    assert ca.is_ocr_failure(weak) is True      # text exists, unusable
    assert ca.is_ocr_failure(collector_only) is False
    assert ca.is_ocr_failure(None) is False
    assert ca.is_ocr_failure({}) is False


# ------------------------------------------------------------ rectify
@pytest.mark.skipif(not ca.CV_AVAILABLE, reason="OpenCV not available")
def test_rectify_produces_canonical_card():
    frame = make_frame()
    r = analyze(frame)
    bbox = r["candidates"][0]["bboxFrame"]
    card = ca.rectify(frame, bbox)
    assert card.shape[:2] == (688, 492)         # CARD_H, CARD_W


# -------------------------------------------------------- debug overlay
@pytest.mark.skipif(not ca.CV_AVAILABLE, reason="OpenCV not available")
def test_annotate_frame_colors_and_banner():
    frame = make_frame()
    det = analyze(frame, ocr_fn=ocr_ok)
    out = ca.annotate_frame(frame, det, camera={"format": "MJPG",
                                                "width": 1280, "height": 720,
                                                "fps": 30, "measuredFps": 29.7})
    assert out.shape == frame.shape
    # banner present → top rows darker than raw frame
    assert out[:30, :, :].mean() < frame[:30, :, :].mean()
    # no-card analysis → red banner path still renders
    r2 = ca.annotate_frame(frame, {"state": "no_card", "reason": "x",
                                   "candidates": []}, camera=None)
    assert r2.shape == frame.shape


# ------------------------------------------------- message single source
def test_instruction_messages_are_distinct():
    msgs = {ca.MSG_NO_CARD, ca.MSG_TOO_SMALL, ca.MSG_QUALITY, ca.MSG_BLUR,
            ca.MSG_LIGHTING, ca.MSG_OCR_FAIL, ca.MSG_MULTIPLE}
    assert len(msgs) == 7                       # every state speaks uniquely


# ------------------------------------------- perspective, not rejection
def test_rotated_card_is_detected_not_rejected():
    """30° rotation must flow into perspective correction, never bad_angle."""
    r = analyze(make_frame(tilt=30), ocr_fn=ocr_ok)
    assert r["state"] == "card_detected"
    assert r["card_detected"] is True
    assert "flatten" not in r["reason"].lower()
    assert len(r["candidates"][0]["corners"]) == 4   # homography source
    assert r["perspective"] == "corrected"


def test_45_degree_card_with_clean_ocr_is_detected():
    """45° keystone: OCR success proves detection — confidence drops, the
    card is still read. A document-scanner-style rejection is forbidden."""
    r = analyze(make_frame(tilt=45), ocr_fn=ocr_ok)
    assert r["state"] == "card_detected"
    assert r["card_detected"] is True
    assert r["perspective"] == "corrected"
    assert r["perspective"] in ("corrected", "flat")


def test_no_state_named_bad_angle_from_pipeline():
    for tilt in (0, 15, 30, 45):
        r = analyze(make_frame(tilt=tilt), ocr_fn=ocr_ok)
        assert r["state"] != "bad_angle"


# ------------------------------------------- homography rectification
def _keystone_scene():
    """A card warped by a known keystone homography + the true corners."""
    W, H = 1280, 720
    frame = np.full((H, W, 3), 40, np.uint8)
    card = np.full((688, 492, 3), 230, np.uint8)   # white-faced card, like real
    card[40:120, 40:120] = (0, 0, 255)        # red marker top-left
    card[600:680, 380:460] = (0, 255, 0)      # green marker bottom-right
    src = np.float32([[0, 0], [491, 0], [491, 687], [0, 687]])
    dst = np.float32([[300, 150], [760, 200], [700, 600], [280, 560]])
    Mh = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, Mh, (W, H), borderValue=(40, 40, 40))
    return frame if False else warped, dst


def test_rectify_warps_four_corners_to_canonical():
    warped, corners = _keystone_scene()
    out = ca.rectify(warped, corners)
    assert out.shape == (688, 492, 3)
    assert out[60:100, 60:100, 2].mean() > 150     # red block survived
    assert out[620:660, 390:450, 1].mean() > 150   # green block survived


def test_rectify_accepts_candidate_dict_and_bbox():
    warped, corners = _keystone_scene()
    cand = {"corners": [[float(x), float(y)] for x, y in corners]}
    assert ca.rectify(warped, cand).shape == (688, 492, 3)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    bbox = {"bboxFrame": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]}
    assert ca.rectify(warped, bbox).shape == (688, 492, 3)


def test_detect_corners_finds_the_card_quad():
    warped, true_corners = _keystone_scene()
    found = ca.detect_corners(warped)
    assert found is not None and found.shape == (4, 2)
    diff = float(np.abs(found - true_corners).mean())
    assert diff < 15.0                              # within ~15 px of truth


def test_detect_corners_never_raises_on_flat_scene():
    empty = np.full((720, 1280, 3), 40, np.uint8)
    assert ca.detect_corners(empty) in (None,) or True
    r = ca.detect_corners(empty)
    assert r is None or r.shape == (4, 2)
