"""FlyCommander physical-table mode — camera configuration.

Loads camera settings from (in increasing precedence):

1. Built-in defaults tuned for the Logitech C922 Pro Stream:
       device /dev/video0, MJPG, 1920x1080 @ 30 FPS
   (The C922 ships 2304x1536@2fps YUYV modes that OpenCV happily negotiates
   on its own — unusable for live card scanning. We never accept those.)
2. A JSON config file: `physical/camera_config.json` first, then
   `camera_config.json` at the repo root (or an explicit path).
3. Environment variables FLYCOMMANDER_CAMERA_DEVICE / _WIDTH / _HEIGHT /
   _FPS / _FORMAT.

Config keys (JSON):

    {
      "camera_device": "/dev/video0",
      "camera_width": 1920,
      "camera_height": 1080,
      "camera_fps": 30,
      "camera_format": "MJPG",

      // pipeline knobs (all optional)
      "camera_buffer_size": 1,     // V4L2 queue depth: 1 = always the newest frame
      "preview_width": 960,        // MJPEG preview is downscaled before encoding
      "preview_fps": 15,           // preview encode cadence
      "preview_quality": 70,       // JPEG quality for the preview stream
      "analysis_fps": 5,           // detection/recognition cadence
      "analysis_width": 0          // 0 = analyse at native resolution
   }

The pipeline knobs exist because the capture loop, the analysis pass and the
preview encoder run on three separate threads: capture always runs at the
device rate, so a slow analysis can never make the *preview* choppy or
stale.

This module must stay importable without OpenCV (the plain-python test
environment has no cv2), so it does pure data handling. Hardware
negotiation lives in physical/camera.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0
DEFAULT_FORMAT = "MJPG"

# --- pipeline defaults -----------------------------------------------------
# A depth-1 V4L2 queue is the difference between "live" and "half a second
# behind": OpenCV's default queue keeps several frames buffered, so a reader
# that only consumes 5 FPS is handed frames the camera captured long ago.
DEFAULT_BUFFER_SIZE = 1
# The MJPEG preview is encoded from a downscaled copy: a 1920x1080 JPEG
# encode costs ~20 ms of the loop's time budget and buys nothing on a
# browser-sized <img>.
DEFAULT_PREVIEW_WIDTH = 960
DEFAULT_PREVIEW_FPS = 15.0
DEFAULT_PREVIEW_QUALITY = 70
# Detection + rectification + embedding is the expensive pass; 5 Hz is plenty
# for card recognition and keeps a 4-core laptop cool.
DEFAULT_ANALYSIS_FPS = 5.0
# 0 = analyse at the camera's native resolution (recommended: the detector
# downscales internally); set e.g. 1280 on a slow machine.
DEFAULT_ANALYSIS_WIDTH = 0

# Fallback chain when the preferred mode fails to negotiate (C922 modes that
# are known to work over UVC; order = preference).
FALLBACK_MODES: tuple[dict[str, Any], ...] = (
    {"width": 1280, "height": 720, "fps": 60.0},
    {"width": 1280, "height": 720, "fps": 30.0},
    {"width": 960, "height": 720, "fps": 30.0},
)

# A live camera below this FPS is considered a broken/stalled mode (e.g. the
# C922's 2304x1536 YUYV @ 2fps). Never silently accept it.
MIN_ACCEPTABLE_FPS = 5.0

CONFIG_FILENAMES = ("camera_config.json",)          # repo root
PACKAGE_CONFIG_FILENAME = "camera_config.json"       # physical/

# Common user-friendly spellings → canonical V4L2 FOURCC
FORMAT_ALIASES = {"MJPEG": "MJPG", "JPG": "MJPG", "MPEG": "MJPG"}


def canonical_format(fmt: str) -> str:
    """Normalize a user-supplied format to a 4-char FOURCC."""
    f = str(fmt).strip().upper()
    return FORMAT_ALIASES.get(f, f)


class CameraConfigError(ValueError):
    """Invalid camera configuration values."""


@dataclass
class CameraSettings:
    """Requested camera mode (what we *ask* V4L2 for)."""
    device: str = DEFAULT_DEVICE
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    format: str = DEFAULT_FORMAT
    # ---- pipeline knobs (capture/analysis/preview decoupling) ----------
    buffer_size: int = DEFAULT_BUFFER_SIZE
    preview_width: int = DEFAULT_PREVIEW_WIDTH
    preview_fps: float = DEFAULT_PREVIEW_FPS
    preview_quality: int = DEFAULT_PREVIEW_QUALITY
    analysis_fps: float = DEFAULT_ANALYSIS_FPS
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH
    source: str = "defaults"            # defaults | file | env | merged
    config_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if not self.device:
            raise CameraConfigError("camera_device must be a non-empty string")
        # NOTE: a /dev node that is absent right now is NOT a config error
        # (config may be authored elsewhere; hardware may appear later).
        # CameraCapture.open() fails loudly with diagnostics when the device
        # is truly unusable.
        if not (1 <= int(self.width) <= 8192):
            raise CameraConfigError(f"camera_width out of range: {self.width}")
        if not (1 <= int(self.height) <= 8192):
            raise CameraConfigError(f"camera_height out of range: {self.height}")
        if not (1.0 <= float(self.fps) <= 240.0):
            raise CameraConfigError(f"camera_fps out of range: {self.fps}")
        fmt = str(self.format).upper()
        if len(fmt) != 4 or not fmt.isalnum():
            raise CameraConfigError(
                f"camera_format must be a 4-char FOURCC like MJPG, got {self.format!r}")
        if not (0 <= int(self.buffer_size) <= 16):
            raise CameraConfigError(
                f"camera_buffer_size out of range: {self.buffer_size}")
        if not (0 <= int(self.preview_width) <= 8192):
            raise CameraConfigError(
                f"preview_width out of range: {self.preview_width}")
        if not (1.0 <= float(self.preview_fps) <= 60.0):
            raise CameraConfigError(f"preview_fps out of range: {self.preview_fps}")
        if not (1 <= int(self.preview_quality) <= 100):
            raise CameraConfigError(
                f"preview_quality out of range: {self.preview_quality}")
        if not (0.5 <= float(self.analysis_fps) <= 60.0):
            raise CameraConfigError(f"analysis_fps out of range: {self.analysis_fps}")
        if not (0 <= int(self.analysis_width) <= 8192):
            raise CameraConfigError(
                f"analysis_width out of range: {self.analysis_width}")

    @property
    def device_exists(self) -> bool:
        return Path(self.device).exists()

    # ------------------------------------------------------------------
    @property
    def preview_interval(self) -> float:
        """Seconds between preview encodes (0 fps-safe)."""
        return 1.0 / max(0.5, float(self.preview_fps))

    @property
    def analysis_interval(self) -> float:
        """Seconds between analysis passes."""
        return 1.0 / max(0.5, float(self.analysis_fps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_device": self.device,
            "camera_width": int(self.width),
            "camera_height": int(self.height),
            "camera_fps": float(self.fps),
            "camera_format": str(self.format).upper(),
            "camera_buffer_size": int(self.buffer_size),
            "preview_width": int(self.preview_width),
            "preview_fps": float(self.preview_fps),
            "preview_quality": int(self.preview_quality),
            "analysis_fps": float(self.analysis_fps),
            "analysis_width": int(self.analysis_width),
            "source": self.source,
            "config_path": self.config_path,
        }

    def describe(self) -> str:
        lines = [f"Device: {self.device}",
                 f"Format: {str(self.format).upper()}",
                 f"Resolution: {int(self.width)}x{int(self.height)}",
                 f"FPS: {g_fmt(self.fps)}",
                 f"Buffers: {int(self.buffer_size)}",
                 f"Preview: {int(self.preview_width) or 'native'}px @ "
                 f"{g_fmt(self.preview_fps)} fps (q{int(self.preview_quality)})",
                 f"Analysis: {g_fmt(self.analysis_fps)} fps"
                 + (f" @ {int(self.analysis_width)}px"
                    if self.analysis_width else " (native)")]
        if self.device.startswith("/dev/") and not self.device_exists:
            lines.append(f"WARNING: {self.device} does not exist right now")
        return "\n".join(lines)


def g_fmt(x: float) -> str:
    """Format an FPS number without trailing '.0' noise."""
    return f"{x:g}"


# ---------------------------------------------------------------------------
# loading / merging
# ---------------------------------------------------------------------------
def _coerce(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON keys/values into the canonical camera_* schema."""
    out: dict[str, Any] = {}
    # accept both camera_* and short forms (device/width/height/fps/format)
    alias = {
        "camera_device": "device", "device": "device",
        "camera_width": "width", "width": "width",
        "camera_height": "height", "height": "height",
        "camera_fps": "fps", "fps": "fps",
        "camera_format": "format", "format": "format",
        "camera_buffer_size": "buffer_size", "buffer_size": "buffer_size",
        "camera_preview_width": "preview_width", "preview_width": "preview_width",
        "camera_preview_fps": "preview_fps", "preview_fps": "preview_fps",
        "camera_preview_quality": "preview_quality",
        "preview_quality": "preview_quality",
        "camera_analysis_fps": "analysis_fps", "analysis_fps": "analysis_fps",
        "camera_analysis_width": "analysis_width",
        "analysis_width": "analysis_width",
    }
    ints = {"width", "height", "buffer_size", "preview_width",
            "preview_quality", "analysis_width"}
    floats = {"fps", "preview_fps", "analysis_fps"}
    for k, v in d.items():
        name = alias.get(k)
        if name is None:
            out[k] = v                      # unknown keys preserved in extras
            continue
        if name in ints:
            out[name] = int(v)
        elif name in floats:
            out[name] = float(v)
        elif name == "device":
            out[name] = str(v)
        elif name == "format":
            out[name] = canonical_format(v)
    return out


def _apply_env(d: dict[str, Any]) -> dict[str, Any]:
    env = os.environ
    if env.get("FLYCOMMANDER_CAMERA_DEVICE"):
        d["device"] = env["FLYCOMMANDER_CAMERA_DEVICE"]
    for name, var in (("width", "FLYCOMMANDER_CAMERA_WIDTH"),
                      ("height", "FLYCOMMANDER_CAMERA_HEIGHT")):
        if env.get(var):
            try:
                d[name] = int(env[var])
            except ValueError as exc:
                raise CameraConfigError(f"{var} must be an integer") from exc
    if env.get("FLYCOMMANDER_CAMERA_FPS"):
        try:
            d["fps"] = float(env["FLYCOMMANDER_CAMERA_FPS"])
        except ValueError as exc:
            raise CameraConfigError("FLYCOMMANDER_CAMERA_FPS must be a number") from exc
    if env.get("FLYCOMMANDER_CAMERA_FORMAT"):
        d["format"] = canonical_format(env["FLYCOMMANDER_CAMERA_FORMAT"])
    for name, var in (("buffer_size", "FLYCOMMANDER_CAMERA_BUFFER_SIZE"),
                      ("preview_width", "FLYCOMMANDER_CAMERA_PREVIEW_WIDTH"),
                      ("preview_quality", "FLYCOMMANDER_CAMERA_PREVIEW_QUALITY"),
                      ("analysis_width", "FLYCOMMANDER_CAMERA_ANALYSIS_WIDTH")):
        if env.get(var):
            try:
                d[name] = int(env[var])
            except ValueError as exc:
                raise CameraConfigError(f"{var} must be an integer") from exc
    for name, var in (("preview_fps", "FLYCOMMANDER_CAMERA_PREVIEW_FPS"),
                      ("analysis_fps", "FLYCOMMANDER_CAMERA_ANALYSIS_FPS")):
        if env.get(var):
            try:
                d[name] = float(env[var])
            except ValueError as exc:
                raise CameraConfigError(f"{var} must be a number") from exc
    return d


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    """Config file search order: explicit path → package dir → repo root."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    pkg = Path(__file__).resolve().parent / PACKAGE_CONFIG_FILENAME
    if pkg.exists():
        return pkg
    root = Path(__file__).resolve().parents[1]
    for name in CONFIG_FILENAMES:
        p = root / name
        if p.exists():
            return p
    return None


def load_camera_settings(config_path: str | Path | None = None) -> CameraSettings:
    """Build the effective CameraSettings from defaults + file + env."""
    merged: dict[str, Any] = {}
    source = "defaults"
    path_used: Path | None = None

    cfg_file = find_config_file(config_path)
    if cfg_file is not None:
        try:
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CameraConfigError(
                f"invalid JSON in {cfg_file}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CameraConfigError(f"{cfg_file} must contain a JSON object")
        merged = _coerce(raw)
        source, path_used = "file", cfg_file

    before_env = dict(merged)
    merged = _apply_env(merged)
    if merged != before_env:
        source = "env" if source in ("defaults", "env") else "merged"

    known = {"device", "width", "height", "fps", "format", "buffer_size",
             "preview_width", "preview_fps", "preview_quality",
             "analysis_fps", "analysis_width"}
    extras = {k: v for k, v in merged.items() if k not in known}

    settings = CameraSettings(
        device=str(merged.get("device", DEFAULT_DEVICE)),
        width=int(merged.get("width", DEFAULT_WIDTH)),
        height=int(merged.get("height", DEFAULT_HEIGHT)),
        fps=float(merged.get("fps", DEFAULT_FPS)),
        format=canonical_format(merged.get("format", DEFAULT_FORMAT)),
        buffer_size=int(merged.get("buffer_size", DEFAULT_BUFFER_SIZE)),
        preview_width=int(merged.get("preview_width", DEFAULT_PREVIEW_WIDTH)),
        preview_fps=float(merged.get("preview_fps", DEFAULT_PREVIEW_FPS)),
        preview_quality=int(merged.get("preview_quality", DEFAULT_PREVIEW_QUALITY)),
        analysis_fps=float(merged.get("analysis_fps", DEFAULT_ANALYSIS_FPS)),
        analysis_width=int(merged.get("analysis_width", DEFAULT_ANALYSIS_WIDTH)),
        source=source,
        config_path=str(path_used) if path_used else None,
        extras=extras,
    )
    settings.validate()
    return settings


def merged_settings(base: CameraSettings, override: dict[str, Any]) -> CameraSettings:
    """Programmatic override (used by tests and CLI flags)."""
    ov = _coerce(override)
    return replace(
        base,
        device=str(ov.get("device", base.device)),
        width=int(ov.get("width", base.width)),
        height=int(ov.get("height", base.height)),
        fps=float(ov.get("fps", base.fps)),
        format=str(ov.get("format", base.format)).upper(),
        buffer_size=int(ov.get("buffer_size", base.buffer_size)),
        preview_width=int(ov.get("preview_width", base.preview_width)),
        preview_fps=float(ov.get("preview_fps", base.preview_fps)),
        preview_quality=int(ov.get("preview_quality", base.preview_quality)),
        analysis_fps=float(ov.get("analysis_fps", base.analysis_fps)),
        analysis_width=int(ov.get("analysis_width", base.analysis_width)),
        source="merged",
    )
