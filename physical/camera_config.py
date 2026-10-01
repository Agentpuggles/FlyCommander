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
      "camera_format": "MJPG"
   }

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

    @property
    def device_exists(self) -> bool:
        return Path(self.device).exists()

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_device": self.device,
            "camera_width": int(self.width),
            "camera_height": int(self.height),
            "camera_fps": float(self.fps),
            "camera_format": str(self.format).upper(),
            "source": self.source,
            "config_path": self.config_path,
        }

    def describe(self) -> str:
        lines = [f"Device: {self.device}",
                 f"Format: {str(self.format).upper()}",
                 f"Resolution: {int(self.width)}x{int(self.height)}",
                 f"FPS: {g_fmt(self.fps)}"]
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
    }
    for k, v in d.items():
        name = alias.get(k)
        if name is None:
            out[k] = v                      # unknown keys preserved in extras
            continue
        if name in ("width", "height"):
            out[name] = int(v)
        elif name == "fps":
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

    known = {"device", "width", "height", "fps", "format"}
    extras = {k: v for k, v in merged.items() if k not in known}

    settings = CameraSettings(
        device=str(merged.get("device", DEFAULT_DEVICE)),
        width=int(merged.get("width", DEFAULT_WIDTH)),
        height=int(merged.get("height", DEFAULT_HEIGHT)),
        fps=float(merged.get("fps", DEFAULT_FPS)),
        format=canonical_format(merged.get("format", DEFAULT_FORMAT)),
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
        source="merged",
    )
