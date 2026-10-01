"""Camera configuration tests (no hardware, no cv2 required).

Covers: defaults, config-file loading, env overrides, precedence,
validation errors, fallback chain order, settings merge, describe output.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical import camera_config as cc  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FLYCOMMANDER_CAMERA_DEVICE", "FLYCOMMANDER_CAMERA_WIDTH",
                "FLYCOMMANDER_CAMERA_HEIGHT", "FLYCOMMANDER_CAMERA_FPS",
                "FLYCOMMANDER_CAMERA_FORMAT"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------- defaults
def test_defaults_match_c922_target():
    s = cc.load_camera_settings(config_path="/nonexistent/path.json")
    assert s.device == "/dev/video0"
    assert (s.width, s.height) == (1920, 1080)
    assert s.fps == 30.0
    assert s.format == "MJPG"
    assert s.source == "defaults"


def test_fallback_chain_order_and_content():
    modes = [m for m in cc.FALLBACK_MODES]
    assert modes[0] == {"width": 1280, "height": 720, "fps": 60.0}
    assert modes[1] == {"width": 1280, "height": 720, "fps": 30.0}
    assert modes[2] == {"width": 960, "height": 720, "fps": 30.0}


def test_min_acceptable_fps_rejects_2fps_modes():
    assert cc.MIN_ACCEPTABLE_FPS > 2.0


# ------------------------------------------------------------ config file
def test_config_file_loading(tmp_path, monkeypatch):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text(json.dumps({
        "camera_device": "/dev/video2",
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps": 60,
        "camera_format": "mjpg",       # case-insensitive
    }))
    s = cc.load_camera_settings(config_path=cfg)
    assert s.device == "/dev/video2"
    assert (s.width, s.height, s.fps) == (1280, 720, 60.0)
    assert s.format == "MJPG"
    assert s.source == "file"
    assert s.config_path == str(cfg)


def test_short_key_aliases_accepted(tmp_path):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text(json.dumps({
        "device": "/dev/video1", "width": 960, "height": 720,
        "fps": 30, "format": "YUYV",
    }))
    s = cc.load_camera_settings(config_path=cfg)
    assert s.device == "/dev/video1"
    assert s.format == "YUYV"


def test_invalid_json_raises(tmp_path):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text("{not json")
    with pytest.raises(cc.CameraConfigError, match="invalid JSON"):
        cc.load_camera_settings(config_path=cfg)


def test_non_object_json_raises(tmp_path):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text("[1, 2, 3]")
    with pytest.raises(cc.CameraConfigError, match="JSON object"):
        cc.load_camera_settings(config_path=cfg)


def test_package_level_config_discovery(tmp_path, monkeypatch):
    """Drop a config into the package dir → found without explicit path."""
    pkg_cfg = Path(cc.__file__).parent / "camera_config.json"
    assert not pkg_cfg.exists(), "repo must not ship a live camera config"
    # emulate: point find_config_file at an explicit missing path → None
    assert cc.find_config_file(explicit="/no/such/file.json") is None


# ----------------------------------------------------------------- env
def test_env_overrides(monkeypatch):
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_DEVICE", "/dev/video3")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_WIDTH", "640")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_HEIGHT", "480")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_FPS", "15")
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_FORMAT", "mjpeg")
    s = cc.load_camera_settings(config_path="/nonexistent/path.json")
    assert s.device == "/dev/video3"
    assert (s.width, s.height, s.fps) == (640, 480, 15.0)
    assert s.format == "MJPG"                # MJPEG canonicalized to FOURCC
    assert s.source == "env"


def test_env_invalid_number_raises(monkeypatch):
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_WIDTH", "wide")
    with pytest.raises(cc.CameraConfigError, match="integer"):
        cc.load_camera_settings(config_path="/nonexistent/path.json")


# ------------------------------------------------------------- validation
def test_validation_rejects_bad_values():
    with pytest.raises(cc.CameraConfigError):
        cc.CameraSettings(device="").validate()
    with pytest.raises(cc.CameraConfigError, match="out of range"):
        cc.CameraSettings(width=0).validate()
    with pytest.raises(cc.CameraConfigError, match="out of range"):
        cc.CameraSettings(fps=0.5).validate()
    with pytest.raises(cc.CameraConfigError, match="FOURCC"):
        cc.CameraSettings(format="TOOLONG").validate()


# ------------------------------------------------------------- precedence
def test_file_then_env_precedence(tmp_path, monkeypatch):
    cfg = tmp_path / "camera_config.json"
    cfg.write_text(json.dumps({"camera_width": 1280, "camera_height": 720}))
    monkeypatch.setenv("FLYCOMMANDER_CAMERA_WIDTH", "640")
    s = cc.load_camera_settings(config_path=cfg)
    assert s.width == 640                 # env beats file
    assert s.height == 720                # file beats defaults
    assert s.fps == 30.0                  # default untouched
    assert s.source == "merged"


# ------------------------------------------------------------------ misc
def test_missing_device_node_is_warning_not_error():
    """A config naming an absent /dev node loads fine; the device check
    lives at open() time, not config time."""
    s = cc.CameraSettings(device="/dev/video-does-not-exist")
    s.validate()                              # must NOT raise
    assert s.device_exists is False
    assert "WARNING" in s.describe()


def test_merged_settings_helper():
    base = cc.CameraSettings()
    s = cc.merged_settings(base, {"width": 800, "fps": 25})
    assert (s.width, s.fps) == (800, 25.0)
    assert (s.height, s.device) == (base.height, base.device)
    assert s.source == "merged"


def test_describe_output_format():
    s = cc.CameraSettings()
    text = s.describe()
    assert "Device: /dev/video0" in text
    assert "Format: MJPG" in text
    assert "Resolution: 1920x1080" in text
    assert "FPS: 30" in text


def test_to_dict_uses_camera_prefixed_keys():
    d = cc.CameraSettings().to_dict()
    assert d["camera_device"] == "/dev/video0"
    assert d["camera_width"] == 1920
    assert d["camera_height"] == 1080
    assert d["camera_fps"] == 30.0
    assert d["camera_format"] == "MJPG"
