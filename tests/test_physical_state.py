"""Tests for the physical-table state model and reconciliation engine."""
from __future__ import annotations

import numpy as np
import pytest

from physical.engine import Engine
from physical.events import Event, EventLog
from physical.state import (
    PhysicalGameState,
    angular_distance,
    is_tapped_orientation,
)


@pytest.fixture()
def engine() -> Engine:
    return Engine(PhysicalGameState(), EventLog())  # in-memory log


def reg(engine: Engine, name: str = "Llanowar Elves", zone: str = "battlefield",
         **kw) -> str:
    res = engine.apply_player(
        "card_registered", name=name, zone=zone, controller="player",
        cardType="Creature — Elf", basePower=1, baseToughness=1, **kw)
    return res["trackingId"]


# ------------------------------------------------------- lifecycle & zones
def test_card_registered_enters_battlefield_sick(engine):
    tid = reg(engine)
    obj = engine.state.get(tid)
    assert obj is not None and obj.zone == "battlefield"
    assert obj.summoning_sick is True
    assert obj.entered_battlefield_turn == 1


def test_entered_battlefield_resets_sickness_clock(engine):
    tid = reg(engine)
    engine.apply_player("zone_change", trackingId=tid, to="graveyard")
    engine.apply_player("zone_change", trackingId=tid, to="hand")
    engine.apply_player("entered_battlefield", trackingId=tid)
    obj = engine.state.get(tid)
    assert obj.zone == "battlefield"
    assert obj.summoning_sick is True  # fresh arrival


def test_zone_change_to_graveyard_updates_lists(engine):
    tid = reg(engine)
    engine.apply_player("zone_change", trackingId=tid, to="graveyard")
    obj = engine.state.get(tid)
    assert obj.zone == "graveyard"
    assert engine.state.players["player"].graveyard == [tid]
    assert not obj.tapped and not obj.attacking  # leaves combat on death


def test_tracking_id_persists_across_zones(engine):
    tid = reg(engine)
    for zone in ("graveyard", "exile", "hand", "battlefield"):
        engine.apply_player("zone_change", trackingId=tid, to=zone)
        assert engine.state.get(tid).tracking_id == tid


def test_occlusion_keeps_identity(engine):
    tid = reg(engine)
    engine.apply_vision(Event(type="disappeared", origin="vision",
                              payload={"trackingId": tid, "frames": 3}))
    obj = engine.state.get(tid)
    assert obj.zone == "battlefield" and obj.occluded_frames == 3
    # reappears — same object, not a new one
    engine.apply_vision(Event(type="appeared", origin="vision",
                              payload={"trackingId": tid}))
    assert obj.occluded_frames == 0
    assert obj.name == "Llanowar Elves"


def test_vision_zone_change_requires_confirmation(engine):
    tid = reg(engine)
    res = engine.apply_vision(Event(
        type="zone_change", origin="vision",
        payload={"trackingId": tid, "to": "graveyard"}))
    assert res["status"] == "pending_confirmation"
    assert engine.state.get(tid).zone == "battlefield"  # unchanged until confirm
    engine.confirm_pending(0)
    assert engine.state.get(tid).zone == "graveyard"


def test_reject_pending_leaves_state_alone(engine):
    tid = reg(engine)
    engine.apply_vision(Event(type="zone_change", origin="vision",
                              payload={"trackingId": tid, "to": "exile"}))
    engine.reject_pending(0)
    assert engine.state.get(tid).zone == "battlefield"
    assert engine.pending == []


# ------------------------------------------------------- tapping & rotation
def test_tap_and_untap_transitions(engine):
    tid = reg(engine)
    r = engine.apply_player("card_tapped", trackingId=tid)
    assert r["changed"] is True
    assert engine.state.get(tid).tapped is True
    r = engine.apply_player("card_untapped", trackingId=tid)
    assert r["changed"] is True
    assert engine.state.get(tid).tapped is False


def test_tap_event_no_change_flag_when_already_tapped(engine):
    tid = reg(engine)
    engine.apply_player("card_tapped", trackingId=tid)
    r = engine.apply_player("card_tapped", trackingId=tid)
    assert r["changed"] is False


def test_rotation_threshold_configurable():
    assert is_tapped_orientation(90.0) is True
    assert is_tapped_orientation(70.0) is True     # within default ±30°
    assert is_tapped_orientation(20.0) is False
    assert is_tapped_orientation(91.0, tolerance=5.0) is True
    assert is_tapped_orientation(80.0, tolerance=5.0) is False
    assert is_tapped_orientation(268.0) is True    # upside-down tap


def test_angular_distance_wraps():
    assert angular_distance(350, 10) == 20
    assert angular_distance(0, 0) == 0
    assert angular_distance(180, 0) == 180


def test_untap_step_untaps_only_active_player(engine):
    tid_p = reg(engine)
    engine.apply_player("card_tapped", trackingId=tid_p)
    # fly's turn: player's card stays tapped
    engine.apply_player("untap_step", controller="fly")
    assert engine.state.get(tid_p).tapped is True
    engine.apply_player("untap_step", controller="player")
    assert engine.state.get(tid_p).tapped is False


# ------------------------------------------------------- summoning sickness
def test_sickness_ends_after_controller_starts_next_turn(engine):
    tid = reg(engine)                      # turn 1: sick
    assert engine.state.get(tid).summoning_sick
    engine.apply_player("turn_advanced", activePlayer="fly")
    assert engine.state.get(tid).summoning_sick   # not controller's turn yet
    engine.apply_player("turn_advanced", activePlayer="player")
    assert not engine.state.get(tid).summoning_sick  # own turn started: free


def test_haste_cancels_sickness(engine):
    tid = reg(engine)
    engine.apply_player("ability_granted", trackingId=tid, ability="haste")
    assert engine.state.get(tid).summoning_sick is False


def test_sickness_only_affects_creatures(engine):
    tid = engine.apply_player(
        "card_registered", name="Plains", cardType="Land",
        zone="battlefield", controller="player")["trackingId"]
    assert engine.state.get(tid).summoning_sick is False


# ------------------------------------------------------- counters & buffs
def test_plus_one_counters(engine):
    tid = reg(engine)
    engine.apply_player("counter_added", trackingId=tid, counter="+1/+1")
    engine.apply_player("counter_added", trackingId=tid, counter="+1/+1")
    obj = engine.state.get(tid)
    assert obj.counters["+1/+1"] == 2
    assert obj.current_power() == 3 and obj.current_toughness() == 3


def test_minus_one_counters_can_canel_plus(engine):
    tid = reg(engine)
    engine.apply_player("counter_added", trackingId=tid, counter="+1/+1")
    engine.apply_player("counter_added", trackingId=tid, counter="-1/-1")
    obj = engine.state.get(tid)
    assert obj.current_power() == 1 and obj.current_toughness() == 1


def test_counter_removal_to_zero_deletes_key(engine):
    tid = reg(engine)
    engine.apply_player("counter_added", trackingId=tid, counter="shield")
    engine.apply_player("counter_removed", trackingId=tid, counter="shield")
    assert "shield" not in engine.state.get(tid).counters


def test_temporary_buff_and_expiry(engine):
    tid = reg(engine)
    engine.apply_player("temp_buff", trackingId=tid, power=2, toughness=2,
                        expires="end_of_turn", source="Giant Growth")
    obj = engine.state.get(tid)
    assert obj.current_power() == 3
    engine.apply_player("turn_advanced", activePlayer="player")
    assert engine.state.get(tid).current_power() == 1  # expired


def test_temporary_debuff(engine):
    tid = reg(engine)
    engine.apply_player("temp_debuff", trackingId=tid, power=2, toughness=2)
    obj = engine.state.get(tid)
    assert obj.current_power() == -1 and obj.current_toughness() == -1


def test_ability_and_status_updates(engine):
    tid = reg(engine)
    engine.apply_player("ability_granted", trackingId=tid, ability="flying")
    assert "flying" in engine.state.get(tid).granted_abilities
    engine.apply_player("ability_removed", trackingId=tid, ability="flying")
    assert "flying" not in engine.state.get(tid).granted_abilities
    engine.apply_player("status_set", trackingId=tid, flag="cant_block")
    assert "cant_block" in engine.state.get(tid).status_flags


def test_damage_marked_and_cleared(engine):
    tid = reg(engine)
    engine.apply_player("damage_marked", trackingId=tid, amount=2)
    assert engine.state.get(tid).damage_marked == 2
    engine.apply_player("damage_cleared", trackingId=tid)
    assert engine.state.get(tid).damage_marked == 0


def test_set_power_toughness(engine):
    tid = reg(engine)
    engine.apply_player("pt_set", trackingId=tid, power=5, toughness=5)
    obj = engine.state.get(tid)
    assert obj.current_power() == 5 and obj.current_toughness() == 5


# ------------------------------------------------------- tokens
def test_token_creation_and_removal(engine):
    r = engine.apply_player("token_created", tokenType="Soldier", count=3,
                            power=1, toughness=1, controller="player")
    tid = r["trackingId"]
    obj = engine.state.get(tid)
    assert obj.is_token and obj.token_count == 3
    assert obj.current_power() == 1
    r = engine.apply_player("token_removed", trackingId=tid)
    assert r["remaining"] == 2
    r = engine.apply_player("token_removed", trackingId=tid)
    engine.apply_player("token_removed", trackingId=tid)
    assert engine.state.get(tid) is None  # fully gone


def test_token_sick_on_arrival(engine):
    r = engine.apply_player("token_created", tokenType="Saproling", count=1,
                            power=1, toughness=1)
    assert engine.state.get(r["trackingId"]).summoning_sick is True


# ------------------------------------------------------- life
def test_life_changes(engine):
    engine.apply_player("life_changed", player="player", delta=-3)
    assert engine.state.players["player"].life == 37
    engine.apply_player("life_changed", player="fly", delta=5)
    assert engine.state.players["fly"].life == 45


def test_commander_damage(engine):
    engine.apply_player("commander_damage", player="player",
                        source="fly-attraxa", amount=7)
    engine.apply_player("commander_damage", player="player",
                        source="fly-attraxa", amount=3)
    p = engine.state.players["player"]
    assert p.commander_damage_taken["fly-attraxa"] == 10
    assert p.life == 30


# ------------------------------------------------------- combat
def test_attack_and_block_flow(engine):
    atk = reg(engine)
    blk = engine.apply_player(
        "card_registered", name="Solemn Simulacrum", cardType="Creature",
        basePower=2, baseToughness=2, zone="battlefield",
        controller="fly")["trackingId"]
    engine.apply_player("combat_begin")
    assert engine.state.combat.active
    engine.apply_player("attack_declared", trackingId=atk, target="fly")
    engine.apply_player("block_declared", trackingId=blk, attacker=atk)
    assert engine.state.combat.attackers == {atk: "fly"}
    assert engine.state.combat.blockers == {blk: atk}
    assert engine.state.get(atk).tapped is True
    engine.apply_player("combat_damage", target="fly", amount=0)  # blocked, 0 dmg
    engine.apply_player("combat_end")
    assert not engine.state.combat.active
    assert not engine.state.get(atk).attacking
    assert not engine.state.get(blk).blocking


def test_attack_retraction(engine):
    tid = reg(engine)
    engine.apply_player("attack_declared", trackingId=tid, target="fly")
    engine.apply_player("attack_retracted", trackingId=tid)
    obj = engine.state.get(tid)
    assert not obj.attacking and engine.state.combat.attackers == {}


# ------------------------------------------------------- events & log
def test_event_log_records_everything(tmp_path):
    log = EventLog(tmp_path / "ev.jsonl")
    eng = Engine(PhysicalGameState(), log)
    tid = eng.apply_player("card_registered", name="Sol Ring",
                           cardType="Artifact", zone="battlefield",
                           controller="player")["trackingId"]
    eng.apply_player("card_tapped", trackingId=tid)
    eng.apply_player("counter_added", trackingId=tid, counter="charge")
    lines = (tmp_path / "ev.jsonl").read_text().strip().splitlines()
    types = [__import__("json").loads(l)["type"] for l in lines]
    assert types == ["card_registered", "card_tapped", "counter_added"]
    # human-readable descriptions exist
    desc = log.describe_recent(3)
    assert "tapped" in desc[1]


def test_event_validation():
    with pytest.raises(ValueError):
        Event(type="not_a_real_event", origin="player").validate()
    with pytest.raises(ValueError):
        Event(type="life_changed", origin="martians").validate()


# ------------------------------------------------------- hidden information
def test_public_observation_hides_private_zones(engine):
    tid = reg(engine)
    engine.apply_player("zone_change", trackingId=tid, to="hand")
    obs = engine.state.public_observation(viewer="fly")
    # human hand identity absent
    assert all(o.get("trackingId") != tid for o in obs["battlefield"])
    assert obs["players"]["player"].get("handCount") == 0 or \
        obs["players"]["player"].get("handCount") is not None
    assert "hand" not in obs  # no hand bucket exposed at all
    assert "library" not in obs


def test_fly_observation_shape_matches_forge_encoder(engine):
    reg(engine)
    fly_board_tid = engine.apply_player(
        "card_registered", name="Ornithopter", cardType="Creature",
        basePower=0, baseToughness=2, zone="battlefield",
        controller="fly")["trackingId"]
    obs = engine.state.to_fly_observation()
    fly = obs["players"][0]
    assert fly["isFly"] is True
    assert fly["creatures"] == 1
    assert fly["hand"] == engine.state.fly_hand_count  # count only
    assert obs["players"][1]["hand"] == 0              # human: count only


def test_fly_observation_feeds_real_encoder_unchanged(engine):
    from flycommander.sensory_encoder import observation_to_state
    reg(engine)
    v = observation_to_state(engine.state.to_fly_observation())
    assert v.shape == (64,)
    assert np.isfinite(v).all()


# ------------------------------------------------------- brain adapter
def test_brain_adapter_decision_and_panel(engine):
    from brain.connectome import build_synthetic_connectome
    from brain.mushroom_body import MushroomBody
    from physical.brain_adapter import FlyBrainAdapter

    mb = MushroomBody(64, connectome=build_synthetic_connectome(n_pn=64))
    adapter = FlyBrainAdapter(mb)
    reg(engine)
    out = adapter.decide(engine.state)
    assert out["decision"]["action"] in ("play", "attack", "hold", "interact")
    scores = out["decision"]["actionScores"]
    assert set(scores) == {"play", "attack", "hold", "interact"}
    panel = adapter.brain_panel()
    assert panel["kcTotal"] == 4064
    assert panel["kcActive"] == adapter.last_kc_active
