"""Identification honesty tests (the "the card is the gitrog monster" bug).

Symptom: a table whose only library is the 48-card *synthetic* demo set, with
an untrained embedder, showed

    name OCR: "Ravenous Voyager 25" 5%
    1. Ravenous Voyager 25 [tpp #378] 0%   Select
    2. Storm Voyager 15    [ygo #151] 0%   Select
    ...

Two real defects there:

1. **Junk was offered as a choice.** A 5% neighbour from a demo library is
   noise, not a candidate — and it was labelled "name OCR" although no OCR had
   run (it was the visual index's best guess).
2. **The one path that could name a real card never ran.** The OCR + Scryfall
   name lookup works with no image library at all, but the neural scan
   short-circuited before it.

Plus the escape hatch every physical table needs: when the camera cannot read
a card, the player types its name.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # noqa: F841  (the scan path needs OpenCV)

from physical.identifier import Candidate                       # noqa: E402
from physical.server import PhysicalTableApp                    # noqa: E402
from physical.vision_pipeline import RecognitionConfig          # noqa: E402
from vision.matcher import MatchCandidate, MatchResult          # noqa: E402


# ---------------------------------------------------------------------------
# fakes: the visual pipeline, stood in for (no camera, no real index)
# ---------------------------------------------------------------------------
class _FakeRectified:
    def __init__(self, image):
        self.image = image
        self.quality_score = 0.8


class _FakeCard:
    def __init__(self, match, image):
        self.track_id = "card1"
        self.match = match
        self.rectified = _FakeRectified(image)
        self.stable_key = ""
        self.stable_name = ""
        self.stable_confidence = 0.0
        self.stable_votes = 0
        self.frames_seen = 1
        self.detection = None

    @property
    def top(self):
        return self.match.best

    @property
    def confidence(self):
        return float(self.match.best.confidence) if self.match.best else 0.0

    def to_dict(self, include_image: bool = False):
        return {"trackId": self.track_id, "framesSeen": self.frames_seen,
                "match": self.match.to_dict(), "cardImage": None}


class _FakeScene:
    def __init__(self, card):
        self.cards = [card] if card is not None else []
        self.timings = {"totalMs": 1.0}

    def best(self):
        return self.cards[0] if self.cards else None

    def to_dict(self, include_images: bool = False):
        return {"cards": [c.to_dict() for c in self.cards],
                "timings": self.timings, "count": len(self.cards)}


class _FakeRecognizer:
    """A visual library that only knows demo cards (weak, unknown matches)."""

    def __init__(self, card, embedder_trained: bool = False):
        self._card = card
        self.ready = True
        self.last_scene = None
        self.calls = {"recognize": 0, "register_card": 0, "save_index": 0}
        self.embedder = type("E", (), {
            "describe": staticmethod(lambda: {"name": "dense", "dim": 32,
                                              "trained": embedder_trained})})()

    def recognize(self, frame, **_kw):
        self.calls["recognize"] += 1
        return _FakeScene(self._card)

    def register_card(self, card, set_code, number, name, image_path=""):
        self.calls["register_card"] += 1
        return {"indexSize": 1}

    def save_index(self):
        self.calls["save_index"] += 1

    def status(self):
        return {"ready": True, "indexSize": 1}


def _weak_match(conf: float = 0.054, unknown: bool = True) -> MatchResult:
    """What a demo library returns for a real card: a 5% neighbour."""
    cands = [
        MatchCandidate(set_code="tpp", collector_number="378",
                       name="Ravenous Voyager 25", confidence=conf,
                       scores={"embed": conf}, type_line="Creature"),
        MatchCandidate(set_code="ygo", collector_number="151",
                       name="Storm Voyager 15", confidence=conf * 0.9,
                       scores={"embed": conf}),
    ]
    return MatchResult(candidates=cands, unknown=unknown, top_score=conf)


@pytest.fixture()
def app(tmp_path):
    return PhysicalTableApp(data_dir=tmp_path, allow_network=False)


def _install(app, conf=0.054, unknown=True, embedder_trained=False):
    frame = np.full((240, 320, 3), 128, np.uint8)
    card = _FakeCard(_weak_match(conf, unknown), frame)
    app.recognizer = _FakeRecognizer(card, embedder_trained=embedder_trained)
    app.vision_config = RecognitionConfig()
    return card


# ---------------------------------------------------------------------------
# 1. junk must never be offered as a choice
# ---------------------------------------------------------------------------

def test_five_percent_library_noise_is_not_a_candidate(app, monkeypatch):
    _install(app, conf=0.054, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", False)  # no OCR either

    out = app.scan_frames_neural([_jpeg()])

    assert out["candidates"] == [], (
        "a 5% neighbour from a demo library must never become a Select button")
    assert out["status"] == "no_candidates"
    assert out["registered"] is None
    # ...and it is honest about what the number means
    assert "visual-index" == out["evidence"]["source"]
    assert "closest in library" in out["message"]
    assert "Ravenous Voyager" in out["message"]


def test_unknown_match_is_reported_as_not_identified(app, monkeypatch):
    _install(app, conf=0.9, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", False)
    out = app.scan_frames_neural([_jpeg()])
    assert out["status"] == "no_candidates"      # unknown beats a high score


# ---------------------------------------------------------------------------
# 2. the OCR + Scryfall rescue: how a real card gets named
# ---------------------------------------------------------------------------

def _jpeg() -> str:
    import base64
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 128, np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _gitrog() -> Candidate:
    return Candidate(name="The Gitrog Monster", set_code="soi",
                     collector_number="245", card_type="Legendary Creature",
                     base_power=6.0, base_toughness=6.0,
                     oracle_id="oracle-gitrog", name_match=0.97,
                     combined_confidence=0.86, source="name+fuzzy")


def test_ocr_rescue_identifies_the_real_card(app, monkeypatch):
    """The reported bug: the card is the Gitrog Monster, the library is demo."""
    _install(app, conf=0.054, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", True)
    monkeypatch.setattr(
        "physical.server.ocr_card_regions",
        lambda img, **kw: {"name": {"raw": "The Gitrog Monster",
                                    "normalized": "The Gitrog Monster",
                                    "confidence": 0.91},
                           "collector": {"raw": "245/297 SOI", "set": "soi",
                                         "number": "245", "confidence": 0.8}})
    monkeypatch.setattr(app.identifier, "identify", lambda ev: [_gitrog()])
    monkeypatch.setattr(app.identifier, "rescore", lambda cands, img: cands)

    out = app.scan_frames_neural([_jpeg()])

    assert out["pipeline"] == "neural+ocr"
    assert out["status"] == "ok"
    assert out["candidates"][0]["name"] == "The Gitrog Monster"
    assert out["evidence"]["source"] == "ocr-name"
    assert out["evidence"]["nameRaw"] == "The Gitrog Monster"
    # a confident rescue registers the card on the table, as before
    assert out["registered"] is not None
    assert out["registered"]["name"] == "The Gitrog Monster"
    assert any(o.name == "The Gitrog Monster"
               for o in app.state.battlefield("player"))


def test_ocr_rescue_end_to_end_through_the_identifier(app, monkeypatch):
    """OCR text -> Evidence -> Scryfall name lookup -> card on the table.

    Only the network call is mocked; the scoring that decides whether the
    answer is good enough to auto-register is the real thing.
    """
    from physical.scryfall_cache import CardInfo

    _install(app, conf=0.054, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", True)
    monkeypatch.setattr(
        "physical.server.ocr_card_regions",
        lambda img, **kw: {"name": {"raw": "The Gitrog Monster",
                                    "normalized": "The Gitrog Monster",
                                    "confidence": 0.88},
                           "collector": {"raw": "", "set": "", "number": "",
                                         "confidence": 0.0}})
    monkeypatch.setattr(
        app.identifier, "fuzzy_by_name",
        lambda name: CardInfo(name="The Gitrog Monster", set_code="soi",
                              collector_number="245", oracle_id="o-gitrog",
                              card_type="Legendary Creature", base_power=6.0,
                              base_toughness=6.0, image_uris={},
                              fetched_at=0.0))

    out = app.scan_frames_neural([_jpeg()])

    assert out["status"] == "ok"
    assert out["candidates"][0]["name"] == "The Gitrog Monster"
    assert out["registered"]["name"] == "The Gitrog Monster"
    assert app.state.battlefield("player")


def test_ocr_rescue_runs_only_when_the_visual_match_is_weak(app, monkeypatch):
    """A confident visual match must not pay for an OCR pass."""
    _install(app, conf=0.95, unknown=False)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", True)

    def _boom(img, **kw):
        raise AssertionError("OCR must not run on a confident visual match")

    monkeypatch.setattr("physical.server.ocr_card_regions", _boom)
    out = app.scan_frames_neural([_jpeg()])
    assert out["pipeline"] == "neural"
    assert out["status"] == "ok"


def test_ocr_rescue_is_skipped_when_ocr_is_unavailable(app, monkeypatch):
    _install(app, conf=0.054, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", False)
    out = app.scan_frames_neural([_jpeg()])
    assert out["pipeline"] == "neural"
    assert out["candidates"] == []


def test_remembering_a_card_needs_a_trained_embedder(app, monkeypatch):
    """Random features would only add confident-looking noise to the index."""
    _install(app, conf=0.054, unknown=True, embedder_trained=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", True)
    monkeypatch.setattr("physical.server.ocr_card_regions",
                        lambda img, **kw: {"name": {"raw": "The Gitrog Monster",
                                                    "normalized": "The Gitrog Monster",
                                                    "confidence": 0.9},
                                           "collector": {"raw": "", "set": "",
                                                         "number": "", "confidence": 0.0}})
    monkeypatch.setattr(app.identifier, "identify", lambda ev: [_gitrog()])
    monkeypatch.setattr(app.identifier, "rescore", lambda cands, img: cands)

    app.scan_frames_neural([_jpeg()])
    assert app.recognizer.calls["register_card"] == 1

    # untrained embedder → the card is registered on the table but NOT indexed
    app.state.reset() if hasattr(app.state, "reset") else None
    _install(app, conf=0.054, unknown=True, embedder_trained=False)
    monkeypatch.setattr(app.identifier, "identify", lambda ev: [_gitrog()])
    monkeypatch.setattr(app.identifier, "rescore", lambda cands, img: cands)
    app.scan_frames_neural([_jpeg()])
    assert app.recognizer.calls["register_card"] == 0


# ---------------------------------------------------------------------------
# 3. the escape hatch: typing the name
# ---------------------------------------------------------------------------

def test_register_by_name_uses_scryfall_fuzzy(app, monkeypatch):
    from physical.scryfall_cache import CardInfo

    app.identifier.allow_network = True      # the lookup is mocked; allow it
    info = CardInfo(name="The Gitrog Monster", set_code="soi",
                    collector_number="245", oracle_id="oracle-gitrog",
                    card_type="Legendary Creature — Frog Horror",
                    base_power=6.0, base_toughness=6.0, image_uris={},
                    fetched_at=0.0)
    monkeypatch.setattr(app.identifier, "fuzzy_by_name", lambda name: info)

    out = app.register_by_name("gitrog monster")

    assert out["status"] == "ok"
    assert out["name"] == "The Gitrog Monster"
    assert any(o.name == "The Gitrog Monster"
               for o in app.state.battlefield("player"))


def test_register_by_name_reports_unknown_names(app, monkeypatch):
    app.identifier.allow_network = True
    monkeypatch.setattr(app.identifier, "fuzzy_by_name", lambda name: None)
    out = app.register_by_name("asdfquux")
    assert out["status"] == "error"
    assert "no Scryfall card" in out["error"]


def test_register_by_name_rejects_stubs(app):
    assert app.register_by_name("a")["status"] == "error"


# ---------------------------------------------------------------------------
# 4. the chooser's index must match what the UI listed
# ---------------------------------------------------------------------------

def test_chooser_index_matches_the_listed_candidates(app, monkeypatch):
    """register_candidate(i) must register the i-th row the player saw."""
    _install(app, conf=0.054, unknown=True)
    monkeypatch.setattr("physical.server.SCAN_AVAILABLE", True)
    monkeypatch.setattr("physical.server.ocr_card_regions",
                        lambda img, **kw: {"name": {"raw": "The Gitrog Monster",
                                                    "normalized": "The Gitrog Monster",
                                                    "confidence": 0.9},
                                           "collector": {"raw": "", "set": "",
                                                         "number": "", "confidence": 0.0}})
    weak = Candidate(name="The Gitrog Monster", set_code="soi",
                     collector_number="245", combined_confidence=0.30,
                     source="name+fuzzy")
    monkeypatch.setattr(app.identifier, "identify", lambda ev: [weak])
    monkeypatch.setattr(app.identifier, "rescore", lambda cands, img: cands)

    out = app.scan_frames_neural([_jpeg()])
    assert [c["name"] for c in out["candidates"]] == ["The Gitrog Monster"]
    reg = app.register_candidate(0)
    assert reg["name"] == "The Gitrog Monster"


# ---------------------------------------------------------------------------
# 5. the UI is told what kind of library it has
# ---------------------------------------------------------------------------

def test_library_summary_separates_synthetic_from_real(tmp_path):
    from cards.database import CardDatabase

    db = CardDatabase(tmp_path / "cards", allow_network=False)
    try:
        assert db.library_summary()["synthetic"] == 0
        added = db.import_synthetic(4)
        assert added == 4
        summary = db.library_summary()
        assert summary["synthetic"] == 4
        assert summary["real"] == 0
    finally:
        db.close()


def test_vision_status_reports_ocr_availability(app):
    status = app.vision_status()
    assert "ocrAvailable" in status
    assert "library" in status


def test_ocr_availability_reports_the_real_reason(app):
    """TESS_AVAILABLE only means the wrapper imports — say what is missing."""
    from physical.card_scan import tesseract_ready

    ok, why = tesseract_ready()
    assert isinstance(ok, bool) and isinstance(why, str)
    status = app.vision_status()
    assert status["ocrAvailable"] is bool(status["ocrAvailable"])
    assert "ocrDetail" in status
    if not ok:
        assert status["ocrAvailable"] is False
        assert "tesseract" in status["ocrDetail"].lower()
