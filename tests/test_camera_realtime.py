"""Camera pipeline responsiveness: capture must never wait for analysis.

The bug this file exists for: capture, analysis and MJPEG encoding all ran
inline in one loop. An analysis pass costs far more than a frame period, so
the loop dropped to a few Hz while the camera kept producing 30 FPS; OpenCV's
V4L2 queue filled, and every frame the browser saw was the *oldest* queued
one — a choppy, half-a-second-behind preview on a perfectly healthy camera.

The fix is structural (capture / analysis / preview threads + a depth-1
queue), so the tests here assert *rates*: a deliberately slow analysis pass
must not slow the capture loop or the preview stream.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F841  (the watcher needs OpenCV)

from physical.camera_config import CameraSettings, load_camera_settings  # noqa: E402
from physical.camera_watcher import CameraWatcher  # noqa: E402


class _StubObserver:
    def __init__(self):
        self.calls = 0

    def process_frame(self, detections):
        self.calls += 1


class _FakeCam:
    """A well-behaved device: paced reads, never fails."""

    def __init__(self, fps: float = 50.0, shape=(48, 64, 3)):
        self._fps = fps
        self._shape = shape
        self.frames = 0
        self.closed = False

        self.diag = types.SimpleNamespace(
            device="/dev/fake",
            format="MJPG",
            width=shape[1],
            height=shape[0],
            fps=fps,
            buffer_size=1,
            requested_buffer_size=1,
            opened_ok=True,
            error="",
            to_dict=lambda: {"device": "/dev/fake", "format": "MJPG",
                             "width": shape[1], "height": shape[0],
                             "fps": fps, "bufferSize": 1},
            describe=lambda: "Camera initialized: fake device",
        )

    @property
    def measured_fps(self) -> float:
        return self._fps

    def read(self):
        time.sleep(1.0 / self._fps)          # the device paces the reader
        self.frames += 1
        return True, np.full(self._shape, 128, np.uint8)

    def close(self):
        self.closed = True


def _running_watcher(monkeypatch, *, device_fps=50.0, analysis_fps=20.0,
                     analysis_cost=0.12, preview_fps=15.0,
                     preview_width=0, shape=(48, 64, 3)):
    """Start a watcher on a fake camera with a deliberately slow analysis."""
    observer = _StubObserver()
    settings = CameraSettings(analysis_fps=analysis_fps, preview_fps=preview_fps,
                              preview_width=preview_width,
                              source="test")
    watcher = CameraWatcher(observer=observer, settings=settings)
    cam = _FakeCam(fps=device_fps, shape=shape)
    calls = {"n": 0}

    def _slow_analysis(self, frame, cam=None, t_loop=0.0, t_capture=0.0):
        calls["n"] += 1
        time.sleep(analysis_cost)            # expensive detection/matching
        observer.process_frame([])

    monkeypatch.setattr(watcher, "_analyze_and_track", _slow_analysis)
    monkeypatch.setattr(watcher, "_open_camera", lambda: _install(watcher, cam))
    watcher.start()
    return watcher, cam, calls


def _install(watcher, cam) -> bool:
    watcher._cam = cam
    return True


@pytest.fixture()
def stopped():
    watchers = []
    yield watchers
    for w in watchers:
        w.stop()


# ---------------------------------------------------------------------------
# the core regression: a slow analysis must not throttle capture or preview
# ---------------------------------------------------------------------------

def test_slow_analysis_does_not_slow_capture(monkeypatch, stopped):
    watcher, cam, calls = _running_watcher(monkeypatch, device_fps=50.0,
                                           analysis_fps=20.0, analysis_cost=0.12)
    stopped.append(watcher)
    deadline = time.time() + 1.5
    while time.time() < deadline and watcher.stats["frames"] < 40:
        time.sleep(0.02)

    captured = watcher.stats["frames"]
    # the device offers ~50 FPS; a 120 ms analysis would have capped an inline
    # loop at ~8 FPS (≈12 frames in 1.5 s). Capture must stay near the device.
    assert captured >= 30, f"capture throttled by analysis: {captured} frames"
    # ...and analysis is still bounded by its own cost, not by capture
    assert calls["n"] <= captured
    assert watcher.stats["captureFps"] > 20.0


def test_preview_keeps_flowing_while_analysis_is_slow(monkeypatch, stopped):
    watcher, cam, calls = _running_watcher(monkeypatch, device_fps=50.0,
                                           analysis_fps=20.0,
                                           analysis_cost=0.12,
                                           preview_fps=15.0)
    stopped.append(watcher)
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if watcher.stats["previewHz"] > 0 and watcher.stats["frames"] > 30:
            break
        time.sleep(0.02)

    assert watcher.stats["frames"] >= 30
    assert watcher.stats["previewHz"] >= 5.0, (
        f"preview starved by analysis: {watcher.stats['previewHz']} Hz")
    assert watcher.timings()["previewMs"] < 120.0   # encodes, not analysis


def test_preview_is_downscaled(monkeypatch):
    """A 1080p JPEG encode per frame is pure waste on a browser-sized <img>."""
    settings = CameraSettings(preview_width=32, source="test")
    watcher = CameraWatcher(observer=_StubObserver(), settings=settings)
    frame = np.full((480, 640, 3), 128, np.uint8)
    jpeg = watcher._encode_preview(frame)
    assert jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 32
    assert decoded.shape[0] == 24

    watcher.settings = CameraSettings(preview_width=0, source="test")
    watcher.preview_width = 0
    full = cv2.imdecode(
        np.frombuffer(watcher._encode_preview(frame), np.uint8),
        cv2.IMREAD_COLOR)
    assert full.shape[1] == 640


def test_capture_loop_stops_cleanly(monkeypatch):
    watcher, cam, calls = _running_watcher(monkeypatch, device_fps=50.0)
    time.sleep(0.2)
    watcher.stop()
    assert cam.closed is True
    assert watcher.stats["running"] is False
    assert watcher.stats["cameraReady"] is False


# ---------------------------------------------------------------------------
# config plumbing for the new knobs
# ---------------------------------------------------------------------------

def test_pipeline_defaults_are_preview_friendly():
    s = load_camera_settings()
    assert s.buffer_size == 1            # depth-1 queue: always the newest frame
    assert s.preview_width == 960
    assert s.preview_fps == 15.0
    assert s.analysis_fps == 5.0
    assert s.analysis_width == 0
    assert "preview_width" in s.to_dict()


def test_pipeline_knobs_load_from_json(tmp_path):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text('{"camera_width": 1280, "camera_height": 720,'
                   ' "camera_fps": 30, "preview_width": 640,'
                   ' "preview_fps": 10, "analysis_fps": 3,'
                   ' "camera_buffer_size": 2, "analysis_width": 960}',
                   encoding="utf-8")
    s = load_camera_settings(str(cfg))
    assert (s.width, s.height) == (1280, 720)
    assert s.preview_width == 640
    assert s.preview_fps == 10.0
    assert s.analysis_fps == 3.0
    assert s.buffer_size == 2
    assert s.analysis_width == 960
    assert s.preview_interval == pytest.approx(0.1)
    assert s.analysis_interval == pytest.approx(1.0 / 3.0)


def test_pipeline_knobs_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_PREVIEW_WIDTH", "480")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_ANALYSIS_FPS", "8")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_BUFFER_SIZE", "3")
    s = load_camera_settings()
    assert s.preview_width == 480
    assert s.analysis_fps == 8.0
    assert s.buffer_size == 3


def test_bad_pipeline_knob_is_a_config_error(tmp_path):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text('{"preview_fps": 0}', encoding="utf-8")
    with pytest.raises(Exception):
        load_camera_settings(str(cfg))
