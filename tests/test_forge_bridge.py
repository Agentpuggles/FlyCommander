"""Tests for the Forge ⇄ Fly integration layer (no Java required)."""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from brain.connectome import build_synthetic_connectome
from brain.dopamine_plasticity import DopamineSystem
from brain.mushroom_body import MushroomBody
from flycommander.forge_agent_client import FlyBrainServer
from flycommander.reward_shaping import RewardComputer
from flycommander.sensory_encoder import N_SENSORY, observation_to_state


def sample_observation() -> dict:
    return {
        "turn": 5,
        "phase": "MAIN1",
        "stackSize": 1,
        "attackers": 0,
        "blockers": 0,
        "players": [
            {"name": "FlyBrain", "isFly": True, "life": 32, "poison": 0,
             "hand": 6, "library": 40, "graveyard": 4, "exile": 1,
             "commandZone": 1, "creatures": 3, "lands": 5,
             "commanderDamage": 4,
             "board": [{"n": "Rhystic Study", "t": 0, "ctl": "FlyBrain"},
                       {"n": "Solemn Simulacrum", "t": 1, "ctl": "FlyBrain"}]},
            {"name": "Ai(2)", "isFly": False, "life": 40, "poison": 0,
             "hand": 7, "library": 35, "graveyard": 9, "exile": 0,
             "commandZone": 0, "creatures": 6, "lands": 8,
             "commanderDamage": 0, "board": []},
        ],
        "canPlay": {"land": True, "spell": True, "ability": False},
    }


# ------------------------------------------------------------- encoder
def test_observation_to_state_shape_and_ranges():
    v = observation_to_state(sample_observation())
    assert v.shape == (N_SENSORY,)
    assert v.min() >= 0.0 and v.max() <= 2.0


def test_fly_seats_first_and_low_life_encodes_high():
    obs = sample_observation()
    v = observation_to_state(obs)
    assert v[0] == pytest.approx(1.0 - 32 / 40.0)          # fly life channel
    assert v[1] == pytest.approx(1.0 - 40 / 40.0)          # opponent life


def test_board_identity_hash_channels_fire():
    v = observation_to_state(sample_observation())
    assert v[44:56].sum() == pytest.approx(0.5)  # two cards → 2 × 0.25


def test_empty_observation_is_safe():
    v = observation_to_state({})
    assert v.shape == (N_SENSORY,) and not v.any()


# ------------------------------------------------------------- rewards
def test_reward_step_shapes_life_and_board_changes():
    rc = RewardComputer()
    obs0 = sample_observation()
    r0 = rc.step(obs0)
    assert not r0.terminal and r0.reward == pytest.approx(-0.001)

    obs1 = sample_observation()
    obs1["players"][0]["life"] = 30                      # fly lost 2 life
    obs1["players"][0]["board"].pop()                    # fly lost a permanent
    r1 = rc.step(obs1)
    assert r1.reward < r0.reward
    assert "life_delta" in r1.components


def test_terminal_win_and_loss():
    rc = RewardComputer()
    rc.step(sample_observation())
    assert rc.terminal({"flyWon": True, "turns": 10}).reward > 0
    assert rc.terminal({"flyWon": False, "turns": 10}).reward < 0


def test_reward_is_clipped_per_step():
    rc = RewardComputer()
    rc.step(sample_observation())
    obs = sample_observation()
    obs["players"][0]["life"] = 0                        # massive swing
    r = rc.step(obs)
    assert -0.5 <= r.reward <= 0.5


# ------------------------------------------------- brain server /decide
@pytest.fixture()
def brain_server():
    mb = MushroomBody(64, connectome=build_synthetic_connectome(n_pn=64))
    ds = DopamineSystem(mb)
    server = FlyBrainServer(mb, ds, RewardComputer(), port=8799, learn=True)
    server.start()
    yield server, mb
    server.stop()


def _post_decide(port: int, obs: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/decide",
        data=json.dumps({"observation": obs}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def test_decide_endpoint_returns_action(brain_server):
    server, _ = brain_server
    resp = _post_decide(server.port, sample_observation())
    assert resp["action"] in (0, 1, 2, 3)
    assert server.decisions_served == 1


def test_legal_mask_respected(brain_server):
    server, _ = brain_server
    obs = sample_observation()
    obs["canPlay"] = {"land": False, "spell": False, "ability": False}
    for _ in range(10):
        resp = _post_decide(server.port, obs)
        assert resp["action"] in (1, 2)  # attack or hold only


def test_episode_finish_learns_and_logs(brain_server, tmp_path):
    server, mb = brain_server
    log = tmp_path / "ep.jsonl"
    server.log_path = str(log)
    _post_decide(server.port, sample_observation())
    summary = server.finish_episode({"flyWon": True, "turns": 8,
                                     "reason": "AllOpponentsLost"})
    assert summary["flyWon"] is True
    assert summary["steps"] == 1
    assert log.exists()
    assert server.episodes_done == 1


def test_encoder_feeds_brain(brain_server):
    server, mb = brain_server
    obs = sample_observation()
    state = observation_to_state(obs)
    kc = mb.encode_state(state)
    assert kc.shape == (mb.connectome.n_kc,)
    assert 0.05 <= kc.mean() <= 0.15
    resp = _post_decide(server.port, obs)
    assert resp["actionName"] in ("play", "attack", "hold", "interact")
