"""Static guards for the browser UI + server request hygiene.

The browser page renders card names that come from Scryfall (or from OCR of
a card), so they are untrusted input: research/security/security_review.md
finding #3 ("card-name-derived HTML insertion"). These tests are greps, not
a browser — they exist so the escaping cannot quietly regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F841  (kept consistent with UI tests)

UI = Path(__file__).resolve().parents[1] / "physical" / "ui.html"
SERVER = Path(__file__).resolve().parents[1] / "physical" / "server.py"


@pytest.fixture(scope="module")
def ui() -> str:
    return UI.read_text(encoding="utf-8")


def test_card_names_are_escaped_everywhere(ui):
    """Every card-name interpolation must go through esc()."""
    bad = [
        line for line in ui.splitlines()
        if ("${c.name}" in line or "${card.name}" in line or "${top.name}" in line)
        and "esc(" not in line
    ]
    assert bad == [], f"unescaped card name in ui.html: {bad}"


def test_escape_helper_exists(ui):
    assert "function esc(" in ui


def test_post_bodies_are_bounded():
    """Unbounded Content-Length = trivial OOM on the table's tiny server."""
    src = SERVER.read_text(encoding="utf-8")
    assert "MAX_REQUEST_BYTES" in src
    assert "413" in src


def test_ocr_name_whitelist_survives_shlex():
    """pytesseract builds argv with shlex.split — a quoted whitelist keeps
    its spaces, so name OCR can (and should) be character-constrained."""
    import shlex

    from physical.card_scan import _NAME_WHITELIST, _ocr_config

    cfg = _ocr_config(7, _NAME_WHITELIST)
    args = shlex.split(cfg)
    assert args[:2] == ["--psm", "7"]
    assert "tessedit_char_whitelist=" + _NAME_WHITELIST in args
    assert "The Gitrog Monster"[1] in _NAME_WHITELIST      # 'h'
    assert " " in _NAME_WHITELIST                          # spaces preserved


def test_ocr_config_has_no_whitelist_when_not_asked():
    from physical.card_scan import _ocr_config

    assert _ocr_config(6, None) == "--psm 6"
