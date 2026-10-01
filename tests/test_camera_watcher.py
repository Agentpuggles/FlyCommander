"""Regression tests for the camera-watcher architecture fixes.

Covers the three real-hardware failures seen on the C922:
- watcher thread dying with KeyError on best["bbox_frame"] (wrong key name);
- reopen churn ("Camera initialized:" printed repeatedly) from watchdog
  false trips during warmup / slow analysis;
- the detection->tracking handoff never raising, whatever the analysis dict
  contains.

Skipped entirely on cv2-free interpreters (module imports the watcher).
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F841  (module under test needs OpenCV)

from physical.camera import Watchdog                       # noqa: E402
from physical.camera_watcher import CameraWatcher          # noqa: E402
from physical.tracker import Detection                     # noqa: E402


class _StubObserver:
    """Records process_frame calls without needing a full engine."""

    def __init__(self):
        self.calls: list[list[Detection]] = []

    def process_frame(self, detections):
        self.calls.append(list(detections))


def _watcher() -> CameraWatcher:
    return CameraWatcher(observer=_StubObserver(), fps_target=5.0)


# ---------------------------------------------------------------------------
# Watchdog: warmup grace + healthy-latch semantics
# ---------------------------------------------------------------------------

def test_watchdog_zero_fps_is_never_a_stall():
    """0.0 means 'no samples yet' (warmup), not a collapsed camera."""
    w = Watchdog()
    assert w.update(0.0) is False
    assert w.update(None) is False
    assert w.stalled is False


def test_watchdog_trips_after_healthy_sample_then_collapse():
    w = Watchdog()
    assert w.update(24.0) is False          # healthy — latch set
    assert w.was_healthy is True
    assert w.update(2.0) is True            # mid-run collapse trips at once
    assert w.stalled is True


def test_watchdog_grace_window_without_healthy_samples():
    """Sub-minimum FPS only trips after grace_s when never healthy."""
    t = {"now": 0.0}
    w = Watchdog(grace_s=5.0, clock=lambda: t["now"])
    for _ in range(10):
        t["now"] += 0.5
        assert w.update(2.0) is False       # still inside the grace window
    t["now"] += 1.0                         # past grace_s
    assert w.update(2.0) is True            # a real 2 FPS mode: not accepted


def test_watchdog_zero_after_healthy_does_not_trip():
    """Device stopped delivering entirely -> grab-failure path, not stall."""
    w = Watchdog()
    w.update(24.0)
    assert w.update(0.0) is False


# ---------------------------------------------------------------------------
# Detection mapping: must never raise, must read camelCase fields
# ---------------------------------------------------------------------------

def test_missing_bbox_does_not_crash_and_skips_tracking():
    """The original crash: card_detected with no bbox_frame/bboxFrame key."""
    w = _watcher()
    analysis = {"state": "card_detected", "card_detected": True,
                "confidence": 0.9,
                "best_candidate": {"confidence": 0.9, "aspect": 0.72}}
    detections = w._detections_from_analysis(analysis)
    assert detections == []
    assert w.stats["cardsSeen"] == 1        # still counts as a seen card


def test_camel_case_bbox_builds_detection():
    w = _watcher()
    analysis = {"card_detected": True,
                "best_candidate": {"confidence": 0.8, "bboxFrame": [10, 20, 100, 140],
                                   "orientationDeg": 12.0}}
    out = w._detections_from_analysis(analysis)
    assert len(out) == 1
    d: Detection = out[0]
    assert d.bbox == (10.0, 20.0, 100.0, 140.0)
    assert d.angle_deg == pytest.approx(12.0)
    assert d.confidence == pytest.approx(0.8)


def test_wrong_key_name_does_not_crash():
    """Even the originally-assumed snake_case dict must not kill the thread."""
    w = _watcher()
    analysis = {"card_detected": True,
                "best_candidate": {"confidence": 0.9, "bbox_frame": [1, 2, 3, 4]}}
    assert w._detections_from_analysis(analysis) == []
    assert w.stats["cardsSeen"] == 1


def test_malformed_bbox_values_do_not_crash():
    w = _watcher()
    analysis = {"card_detected": True,
                "best_candidate": {"confidence": 0.9, "bboxFrame": [1, 2, "x", 4]}}
    assert w._detections_from_analysis(analysis) == []


def test_low_confidence_or_multiple_cards_produce_no_detections():
    w = _watcher()
    good_bbox = {"confidence": 0.9, "bboxFrame": [0, 0, 10, 10]}
    assert w._detections_from_analysis(
        {"card_detected": False, "best_candidate": good_bbox}) == []
    assert w._detections_from_analysis(
        {"card_detected": True, "multiple_cards": True,
         "best_candidate": good_bbox}) == []
    assert w.stats["cardsSeen"] == 0


def test_missing_best_candidate_and_none_analysis_are_safe():
    w = _watcher()
    assert w._detections_from_analysis({"card_detected": True}) == []
    assert w._detections_from_analysis({}) == []
    assert w.stats["cardsSeen"] == 0


def test_detections_reach_the_observer():
    obs = _StubObserver()
    w = CameraWatcher(observer=obs, fps_target=5.0)
    w._detections_from_analysis(
        {"card_detected": True,
         "best_candidate": {"confidence": 0.8, "bboxFrame": [1, 1, 5, 7]}})
    w.observer.process_frame(w._detections_from_analysis(
        {"card_detected": True,
         "best_candidate": {"confidence": 0.8, "bboxFrame": [1, 1, 5, 7]}}))
    assert len(obs.calls) == 1
    assert len(obs.calls[0]) == 1


# ---------------------------------------------------------------------------
# get_burst: fresh frames only (no replaying one stale frame)
# ---------------------------------------------------------------------------

def test_get_burst_does_not_replay_the_same_frame():
    import threading
    import time as _time

    w = _watcher()
    with w._frame_lock:
        w._last_frame = _zeros()
        w._frame_seq = 1

    def _freshen():
        _time.sleep(0.15)
        with w._frame_lock:
            w._last_frame = _ones()
            w._frame_seq += 1

    t = threading.Thread(target=_freshen)
    t.start()
    out = w.get_burst(2, interval=0.05)
    t.join()
    assert len(out) == 2                    # waited for a genuinely new frame
    assert out[0] is not out[1]


def test_new_stats_counters_exist():
    w = _watcher()
    assert w.stats["grabFailures"] == 0
    assert w.stats["stallEvents"] == 0
    assert w.timings() == {}


# ---------------------------------------------------------------------------
# _analyze_and_track: the decoupled analysis pass (capture != analysis rate)
# ---------------------------------------------------------------------------

def _unopened_cam():
    from physical.camera import CameraCapture
    from physical.camera_config import load_camera_settings
    return CameraCapture(load_camera_settings())   # diag available without open


def test_analyze_and_track_dark_frame_reports_bad_quality():
    import numpy as np
    w = _watcher()
    w._analyze_and_track(np.zeros((72, 48, 3), np.uint8),
                         _unopened_cam(), t_loop=0.0, t_capture=0.001)
    assert w.last_analysis()["state"] == "bad_quality"
    assert w.observer.calls == []                  # no tracker update on garbage
    assert "captureMs" in w.timings()


def test_analyze_and_track_empty_desk_reports_no_card():
    import numpy as np
    obs = _StubObserver()
    w = CameraWatcher(observer=obs, fps_target=5.0)
    frame = np.full((72, 48, 3), 200, np.uint8)    # bright, flat, featureless
    w._analyze_and_track(frame, _unopened_cam(), t_loop=0.0, t_capture=0.001)
    assert w.last_analysis()["state"] == "no_card"
    assert obs.calls == [[]]                       # ticked with zero detections
    assert "trackerMs" in w.timings()


def _zeros():
    import numpy as np
    return np.zeros((72, 48, 3), dtype=np.uint8)


def _ones():
    import numpy as np
    return np.ones((72, 48, 3), dtype=np.uint8)
