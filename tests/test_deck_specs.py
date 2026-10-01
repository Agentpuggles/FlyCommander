"""Tests for deck-spec handling (Python side of Forge-owned resolution)."""
from __future__ import annotations

from pathlib import Path

import pytest

from flycommander.deck_specs import (
    RANDOM,
    DeckSpec,
    classify_spec,
    expand_specs,
)


def test_random_spec_is_forwarded():
    spec = classify_spec("random")
    assert spec.kind == "random"
    assert spec.java_arg == "random"


def test_random_is_case_insensitive():
    assert classify_spec("RANDOM").kind == "random"
    assert classify_spec("Random").java_arg == RANDOM


def test_plain_name_is_forwarded_untouched():
    spec = classify_spec("Atraxa AI Deck")
    assert spec.kind == "name"
    assert spec.java_arg == "Atraxa AI Deck"
    assert spec.path is None


def test_existing_dck_path_becomes_file_spec(tmp_path):
    dck = tmp_path / "My Deck.dck"
    dck.write_text("[metadata]\nName=My Deck\n")
    spec = classify_spec(str(dck))
    assert spec.kind == "file"
    assert spec.path == dck
    assert spec.java_arg == str(dck)


def test_missing_dck_path_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        classify_spec(str(tmp_path / "Nope.dck"))


def test_relative_dck_resolved_against_deck_dir(tmp_path):
    dck = tmp_path / "Rel.dck"
    dck.write_text("[metadata]\nName=Rel\n")
    spec = classify_spec("Rel.dck", deck_dir=tmp_path)
    assert spec.kind == "file"
    assert spec.path == dck


def test_relative_missing_dck_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        classify_spec("Ghost.dck", deck_dir=tmp_path)


def test_empty_spec_raises():
    with pytest.raises(ValueError):
        classify_spec("   ")


def test_expand_specs_collects_errors(tmp_path):
    good = tmp_path / "G.dck"
    good.write_text("[metadata]\nName=G\n")
    with pytest.raises(ValueError, match="not found"):
        expand_specs(["random", "Some Deck", str(tmp_path / "X.dck")])


def test_expand_specs_mixed_kinds(tmp_path):
    good = tmp_path / "G.dck"
    good.write_text("[metadata]\nName=G\n")
    specs = expand_specs(["random", "Atraxa AI Deck", str(good)])
    assert [s.kind for s in specs] == ["random", "name", "file"]


def test_dck_suffix_without_file_but_deck_dir_treated_as_name_when_missing():
    """A 'Name.dck' that exists nowhere is still a plausible Forge pool name —
    classify raises only for path-like specs; a bare 'Name.dck' with a deck_dir
    that doesn't exist on disk is forwarded as a file spec candidate... but our
    rule: if deck_dir given and file missing, it's an error."""
    with pytest.raises(ValueError):
        classify_spec("Ghost.dck", deck_dir=Path("/nonexistent"))
