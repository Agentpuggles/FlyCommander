"""FlyCommander physical-table mode — camera capture with explicit mode control.

The Logitech C922 Pro Stream exposes YUYV modes like 2304x1536 @ 2 FPS. Left
to its own devices, OpenCV negotiates them and card scanning becomes a
slideshow. This module opens the device *deliberately*:

    V4L2 backend → /dev/video0 → MJPG FOURCC → resolution → FPS
    → read-back verification of every negotiated property
    → fallback chain (1280x720@60 → 1280x720@30 → 960x720@30)
    → live measured-FPS watchdog (never silently accept a 2 FPS mode)

All negotiation happens once at open; the capture loop itself only reads
frames and monitors the achieved rate.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from physical.camera_config import (
    FALLBACK_MODES,
    MIN_ACCEPTABLE_FPS,
    CameraSettings,
    g_fmt,
)

# Injectable monotonic clock (tests), defaulting to the real one.
Monotonic = Callable[[], float]

try:  # optional heavy dep
    import cv2  # type: ignore
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False


class CameraOpenError(RuntimeError):
    """Raised when no acceptable mode could be negotiated."""


@dataclass
class CameraDiagnostics:
    """What the camera *actually* negotiated (vs what was requested)."""
    device: str
    backend: str = ""
    format: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: float = 0.0
    requested_format: str = ""
    attempt_index: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    measured_fps: float | None = None          # live loop measurement, if any
    opened_ok: bool = False
    error: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "backend": self.backend,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "requested": {
                "width": self.requested_width,
                "height": self.requested_height,
                "fps": self.requested_fps,
                "format": self.requested_format,
            },
            "attemptIndex": self.attempt_index,
            "attempts": self.attempts,
            "measuredFps": self.measured_fps,
            "openedOk": self.opened_ok,
            "error": self.error,
        }

    def describe(self) -> str:
        if not self.opened_ok:
            return (f"Camera FAILED to open: {self.device}\n"
                    f"  error: {self.error}\n"
                    f"  attempts: {self.attempts}")
        mismatch = []
        if self.format != self.requested_format:
            mismatch.append(
                f"format {self.requested_format}→{self.format}")
        if (self.width, self.height) != (self.requested_width,
                                         self.requested_height):
            mismatch.append(
                f"resolution {self.requested_width}x{self.requested_height}"
                f"→{self.width}x{self.height}")
        if abs(self.fps - self.requested_fps) > 0.51:
            mismatch.append(f"fps {g_fmt(self.requested_fps)}→{g_fmt(self.fps)}")
        lines = [
            f"Camera initialized:",
            f" Device: {self.device}",
            f" Format: {self.format}",
            f" Resolution: {self.width}x{self.height}",
            f" FPS: {g_fmt(self.fps)}",
            f" Backend: {self.backend}",
        ]
        if self.attempt_index > 0:
            lines.append(f" Note: preferred mode failed; using fallback "
                         f"#{self.attempt_index}")
        if mismatch:
            lines.append(f" MISMATCH (requested vs negotiated): " + "; ".join(mismatch))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _fourcc(fmt: str) -> int:
    return cv2.VideoWriter_fourcc(*fmt.upper())


def _read_back(cap) -> tuple[str, int, int, float]:
    fmt_num = int(cap.get(cv2.CAP_PROP_FOURCC))
    b = bytes([ (fmt_num >> (8 * i)) & 0xFF for i in range(4)])
    # V4L2 reports the fourcc little-endian; render as ASCII in read order
    fmt = b.decode("ascii", errors="replace").strip("\x00 ")
    return (fmt, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)))


def _acceptable(negotiated: tuple[str, int, int, float],
                requested_fmt: str, min_fps: float = MIN_ACCEPTABLE_FPS) -> tuple[bool, str]:
    """Gate for accepting a negotiated mode. Never silently accept junk."""
    fmt, w, h, fps = negotiated
    if w <= 0 or h <= 0:
        return False, "negotiated resolution is 0x0"
    if fps < min_fps:
        return False, (f"negotiated FPS {g_fmt(fps)} is below the minimum "
                       f"usable rate {g_fmt(min_fps)} (stalled YUYV-style mode)")
    if requested_fmt == "MJPG" and fmt and fmt != "MJPG":
        return False, f"requested MJPG but device negotiated {fmt}"
    return True, ""


class CameraCapture:
    """Explicit-mode camera handle.

    Usage:
        cam = CameraCapture(settings)
        cam.open()              # negotiates, verifies, falls back
        print(cam.diag.describe())
        ok, frame = cam.read()
        ...
        cam.close()
    """

    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings
        self.cap: Any = None
        self.diag = CameraDiagnostics(device=settings.device)
        self._fps_monitor: _FpsMonitor | None = None

    # ------------------------------------------------------------------
    # opening / negotiation
    # ------------------------------------------------------------------
    def open(self) -> "CameraCapture":
        if not CV_AVAILABLE:
            raise CameraOpenError("OpenCV is not available in this environment")
        s = self.settings
        s.validate()
        self.diag.requested_width = s.width
        self.diag.requested_height = s.height
        self.diag.requested_fps = s.fps
        self.diag.requested_format = s.format.upper()

        attempts: list[dict[str, Any]] = []
        # preferred mode first, then the fallback chain
        modes = [{"width": s.width, "height": s.height, "fps": s.fps}]
        modes += [dict(m) for m in FALLBACK_MODES]
        # de-duplicate (preferred may equal a fallback entry)
        seen = set()
        modes = [m for m in modes
                 if (m["width"], m["height"], m["fps"]) not in seen
                 and not seen.add((m["width"], m["height"], m["fps"]))]

        first_error = ""
        for idx, mode in enumerate(modes):
            entry: dict[str, Any] = {
                "width": mode["width"], "height": mode["height"],
                "fps": mode["fps"], "format": s.format.upper(),
            }
            cap = cv2.VideoCapture(s.device, cv2.CAP_V4L2)
            try:
                if not cap.isOpened():
                    entry["result"] = "open-failed"
                    attempts.append(entry)
                    if not first_error:
                        first_error = f"could not open {s.device}"
                    continue
                # explicit property order: FOURCC first so resolution changes
                # pick a compatible MJPG mode, then size, then rate
                cap.set(cv2.CAP_PROP_FOURCC, _fourcc(s.format))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode["height"])
                cap.set(cv2.CAP_PROP_FPS, mode["fps"])

                negotiated = _read_back(cap)
                entry["negotiated"] = {
                    "format": negotiated[0], "width": negotiated[1],
                    "height": negotiated[2], "fps": negotiated[3],
                }
                ok, why = _acceptable(negotiated, s.format.upper())
                if not ok:
                    entry["result"] = f"rejected: {why}"
                    attempts.append(entry)
                    cap.release()
                    continue
                # prove frames actually flow at the claimed rate
                got_frame = False
                for _ in range(3):
                    ok_read, _f = cap.read()
                    if ok_read:
                        got_frame = True
                        break
                if not got_frame:
                    entry["result"] = "rejected: no frames after 3 read attempts"
                    attempts.append(entry)
                    cap.release()
                    continue

                entry["result"] = "ok"
                attempts.append(entry)
                self.cap = cap
                self.diag.opened_ok = True
                self.diag.backend = "V4L2"
                self.diag.format, self.diag.width, self.diag.height, self.diag.fps = negotiated
                self.diag.attempt_index = idx
                self.diag.attempts = attempts
                self._fps_monitor = _FpsMonitor()
                return self
            except Exception as exc:  # pragma: no cover — defensive
                entry["result"] = f"exception: {exc}"
                attempts.append(entry)
                try:
                    cap.release()
                except Exception:
                    pass

        self.diag.attempts = attempts
        self.diag.opened_ok = False
        self.diag.error = first_error or "all mode attempts failed"
        raise CameraOpenError(self.diag.describe())

    # ------------------------------------------------------------------
    # reading / lifecycle
    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, Any]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok and self._fps_monitor is not None:
            self._fps_monitor.tick()
        return ok, frame

    @property
    def measured_fps(self) -> float | None:
        return self._fps_monitor.fps if self._fps_monitor else None

    def close(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
        self.diag.opened_ok = False

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Live status incl. measured FPS for the debug panel."""
        d = self.diag.to_dict()
        d["measuredFps"] = self.measured_fps
        return d


class _FpsMonitor:
    """Rolling measured-FPS estimate from actual frame reads."""

    def __init__(self, window_s: float = 4.0) -> None:
        self.window_s = window_s
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._stamps.append(now)
            cutoff = now - self.window_s
            while self._stamps and self._stamps[0] < cutoff:
                self._stamps.pop(0)

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            if span <= 0:
                return 0.0
            return (len(self._stamps) - 1) / span


class Watchdog:
    """Flags a stall when measured FPS collapses below MIN_ACCEPTABLE_FPS
    after the camera has demonstrably been healthy (device renegotiated
    behind our back, USB bandwidth drop, etc.).

    Two guards prevent false trips on a *working* camera:

    - a ``measured_fps`` of 0.0 (no samples yet) or None is never a stall —
      this covers warmup, where the rolling FPS estimate needs ≥2 frames;
    - a sub-minimum reading only trips after either one healthy sample has
      been seen (mid-run collapse) or ``grace_s`` elapsed without any
      healthy sample (the device genuinely opened in a 2 FPS stall mode —
      never silently accepted, just not panic-reopened mid-warmup).
    """

    def __init__(self, min_fps: float = MIN_ACCEPTABLE_FPS,
                 grace_s: float = 5.0, clock: Monotonic = time.monotonic) -> None:
        self.min_fps = min_fps
        self.grace_s = grace_s
        self._clock = clock
        self._opened_at = clock()
        self.was_healthy = False
        self.stalled = False

    def update(self, measured_fps: float | None) -> bool:
        if measured_fps is not None and measured_fps >= self.min_fps:
            self.was_healthy = True
            self.stalled = False
        elif (measured_fps is not None and measured_fps > 0
              and (self.was_healthy
                   or self._clock() - self._opened_at > self.grace_s)):
            self.stalled = True
        return self.stalled
