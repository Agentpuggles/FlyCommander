"""Tests for the physical vision layer: tracker, observer, registration.

No camera hardware is used here — detections are synthetic. The live camera
path (getUserMedia → /api/register/frame) is exercised manually via the UI.
"""
from __future__ import annotations

import pytest

from physical.engine import Engine
from physical.events import EventLog
from physical.observer import PhysicalObserver
from physical.registration import parse_collector_line
from physical.scryfall_cache import ScryfallCache, normalize_set_code
from physical.state import PhysicalGameState
from physical.tracker import CardTracker, Detection


def det(x, y, angle=0.0, conf=1.0):
    return Detection(bbox=(x, y, 63, 88), angle_deg=angle, confidence=conf)


# ------------------------------------------------------------- tracker
def test_stable_id_across_small_movements():
    tr = CardTracker()
    r1 = tr.update([det(100, 100)])
    r2 = tr.update([det(105, 103)])   # small shift: same track
    tid = next(iter(r1))
    assert tid in r2 and r2[tid]["status"] in ("ok", "settling")


def test_occlusion_grace_then_retirement():
    tr = CardTracker(grace_frames=2)
    r1 = tr.update([det(50, 50)])
    tid = next(iter(r1))
    r2 = tr.update([])                # missed 1
    assert r2[tid]["status"] == "occluded"
    r3 = tr.update([])                # missed 2 → still within grace
    assert r3[tid]["status"] == "occluded"
    r4 = tr.update([])                # missed 3 > grace → retired
    assert r4[tid]["status"] == "retired"


def test_disappear_reappear_nearby_is_same_track():
    tr = CardTracker(grace_frames=5)
    r1 = tr.update([det(200, 200)])
    tid = next(iter(r1))
    tr.update([])
    r3 = tr.update([det(210, 205)])
    assert tid in r3  # identity retained (not a new track)


def test_really_new_object_gets_new_track():
    tr = CardTracker(grace_frames=1)
    r1 = tr.update([det(10, 10)])
    old_tid = next(iter(r1))
    tr.update([])                        # missed 1 (grace boundary)
    r3 = tr.update([])                   # missed 2 → retired
    assert r3[old_tid]["status"] == "retired"
    r4 = tr.update([det(400, 400)])      # elsewhere → genuinely new track
    assert any(info["status"] == "new" for info in r4.values())


# ------------------------------------------------------------- tapping
def test_tap_transition_emitted_after_settling():
    tr = CardTracker(settle_frames=2, tap_tolerance_deg=30.0)
    r1 = tr.update([det(0, 0, angle=0.0)])
    tid = next(iter(r1))
    out = []
    for _ in range(3):
        r = tr.update([det(0, 0, angle=91.0)])
        out.append(r[tid])
    assert "tapEvents" in out[-1]
    assert out[-1]["tapEvents"] == ["card_tapped"]


def test_untap_transition_emitted():
    tr = CardTracker(settle_frames=1)
    r1 = tr.update([det(0, 0, angle=90.0)])
    tid = next(iter(r1))
    assert r1[tid]["tapped"] is True
    r2 = tr.update([det(0, 0, angle=2.0)])
    assert r2[tid].get("tapEvents") == ["card_untapped"]


def test_brief_rotation_flicker_does_not_emit():
    tr = CardTracker(settle_frames=3)
    r1 = tr.update([det(0, 0, angle=0.0)])
    tid = next(iter(r1))
    tr.update([det(0, 0, angle=88.0)])    # momentary — 1 settling frame
    r3 = tr.update([det(0, 0, angle=3.0)])  # back down before settling
    assert "tapEvents" not in r3[tid]
    assert r3[tid]["tapped"] is False


def test_imperfect_angles_count_as_tapped():
    tr = CardTracker(tap_tolerance_deg=30.0)
    r = tr.update([det(0, 0, angle=75.0)])
    tid = next(iter(r))
    assert r[tid]["tapped"] is True        # 75° ≈ tapped, tolerance ok


# ------------------------------------------------------- observer→engine
def _observer():
    engine = Engine(PhysicalGameState(), EventLog())
    obs = PhysicalObserver(engine)
    return engine, obs


def _register_bound(obs: PhysicalObserver, engine: Engine, name="Elvish Mystic"):
    reg = engine.apply_player("card_registered", name=name,
                              cardType="Creature — Elf", basePower=1,
                              baseToughness=1, zone="battlefield",
                              controller="player")
    tr = obs.tracker.update([det(100, 100)])
    track_id = next(iter(tr))
    obs.bind(track_id, reg["trackingId"])
    return track_id, reg["trackingId"]


def test_observer_conveys_tap_to_engine():
    engine, obs = _observer()
    track_id, state_id = _register_bound(obs, engine)
    # a physical rotation spans a few frames (smoothing + settle hysteresis);
    # by the third frame at ~90° the tap must be authoritative in state
    obs.process_frame([det(100, 100, angle=88.0)])
    obs.process_frame([det(100, 100, angle=92.0)])
    obs.process_frame([det(100, 100, angle=92.0)])
    obj = engine.state.get(state_id)
    assert obj.tapped is True
    types = [e.type for e in engine.log.recent(10)]
    assert "card_tapped" in types


def test_observer_ignores_unbound_tracks():
    engine, obs = _observer()
    obs.process_frame([det(10, 10)])
    obs.process_frame([det(10, 10, angle=90.0)])
    obs.process_frame([det(10, 10, angle=90.0)])
    assert engine.state.objects == {}  # nothing registered → no state changes


def test_bound_state_receives_occlusion():
    engine, obs = _observer()
    _, state_id = _register_bound(obs, engine)
    obs.process_frame([])   # occluded frame 1
    obj = engine.state.get(state_id)
    assert obj.occluded_frames >= 1 and obj.zone == "battlefield"


# ------------------------------------------------------- registration
def test_parse_collector_line_variants():
    assert parse_collector_line("216/281 FDN") == ("FDN", "216")
    assert parse_collector_line("042 (280) NEO") == ("NEO", "042")
    assert parse_collector_line("garbage") is None


def test_normalize_set_code():
    assert normalize_set_code("fdn") == "FDN"
    assert normalize_set_code("f.d.n!7") == "FDN"
    assert normalize_set_code("M3H") == "MH"


def test_scryfall_cache_roundtrip_offline(tmp_path):
    cache = ScryfallCache(tmp_path / "c.sqlite3", allow_network=False)
    from physical.scryfall_cache import CardInfo
    info = CardInfo(name="Sol Ring", set_code="FDN", collector_number="250",
                    oracle_id="abc", card_type="Artifact",
                    base_power=None, base_toughness=None,
                    image_uris={}, fetched_at=0.0)
    cache.put(info)
    got = cache.get("FDN", "250", use_network=False)
    assert got is not None and got.name == "Sol Ring"
    assert cache.get("FDN", "999", use_network=False) is None  # offline miss
    cache.close()


def test_manual_registration_rejects_unknown_card(tmp_path):
    cache = ScryfallCache(tmp_path / "c.sqlite3", allow_network=False)
    from physical.registration import CardRegistrar
    reg = CardRegistrar(cache)
    out = reg.register_manual("FDN", "999999", use_network=False)
    assert out.success is False and "no Scryfall card" in out.error
    cache.close()


# ------------------------------------------------------- engine+server glue
def test_engine_handles_unknown_object_gracefully():
    engine = Engine(PhysicalGameState(), EventLog())
    r = engine.apply_player("card_tapped", trackingId="card-99999")
    assert r["status"] == "unknown_object"


def test_spell_cast_and_resolve():
    engine = Engine(PhysicalGameState(), EventLog())
    engine.apply_player("spell_cast", name="Counterspell", controller="player")
    assert len(engine.state.stack) == 1
    engine.apply_player("spell_resolved")
    assert len(engine.state.stack) == 0
