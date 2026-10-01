"""Tests for the spiking brain machinery."""
from __future__ import annotations

import numpy as np
import pytest

from brain.connectome import (
    DAN_COUNT,
    KC_COUNT,
    MBON_COUNT,
    build_synthetic_connectome,
)
from brain.dopamine_plasticity import DPRConfig, DopamineSystem
from brain.lif_engine import LIFPopulation, winner_take_all
from brain.mushroom_body import N_MACRO_ACTIONS, MushroomBody


# ---------------------------------------------------------------- LIF engine
def test_lif_spikes_when_driven_and_resets():
    pop = LIFPopulation(8)
    strong = np.full(8, 50.0)
    spikes = pop.run(strong, steps=5)
    assert spikes.any()
    # membrane reset after spike
    assert (pop.v[spikes] < pop.v_thresh).all()


def test_lif_silent_when_unforced():
    pop = LIFPopulation(64)
    spikes = pop.run(np.zeros(64), steps=200)
    assert not spikes.any()


def test_lif_refractory_prevents_immediate_respiking():
    pop = LIFPopulation(4)
    pop.v[:] = pop.v_thresh + 1.0  # would spike on next step
    first = pop.step(np.zeros(4))
    assert first.any()
    pop.v[:] = pop.v_thresh + 1.0
    second = pop.step(np.zeros(4))
    assert not second.any()  # within 2.2 ms refractory at dt=0.5


def test_winner_take_all_sparsity():
    activity = np.array([0.1, 5.0, 3.0, 0.0, 2.0, 7.0, 1.0, 0.2])
    code = winner_take_all(activity, keep_fraction=0.25)
    assert code.sum() == 2
    assert code[5] == 1.0 and code[1] == 1.0


def test_winner_take_all_zero_input_is_all_zero():
    code = winner_take_all(np.zeros(10), keep_fraction=0.5)
    assert code.sum() == 0


# ------------------------------------------------------------- connectome
def test_connectome_statistics_match_fly_mb():
    c = build_synthetic_connectome(n_pn=64)
    assert c.n_kc == KC_COUNT == 4064
    assert c.n_mbon == MBON_COUNT == 97
    assert c.n_dan == DAN_COUNT == 344
    # PN→KC convergence ~7 inputs per KC
    fan_in = c.pn_to_kc.sum(axis=0)
    assert (fan_in == 7).all()
    # DAN→MBON gates are signed: appetitive (+), aversive (−)
    assert (c.dan_to_mbon[: c.n_dan // 2] >= 0).all()
    assert (c.dan_to_mbon[c.n_dan // 2 :] <= 0).all()
    assert c.kc_to_mbon.shape == (KC_COUNT, MBON_COUNT)


def test_connectome_save_load_roundtrip(tmp_path):
    c = build_synthetic_connectome(n_pn=32, seed=7)
    path = str(tmp_path / "mb.npz")
    c.save(path)
    c2 = MushroomBodyConnectome.load(path)
    assert np.array_equal(c.pn_to_kc, c2.pn_to_kc)
    assert np.array_equal(c.kc_to_mbon, c2.kc_to_mbon)
    assert c2.meta["source"] == "synthetic"


from brain.connectome import MushroomBodyConnectome  # noqa: E402


# ------------------------------------------------------------ mushroom body
@pytest.fixture()
def mb():
    c = build_synthetic_connectome(n_pn=64, seed=3)
    return MushroomBody(64, connectome=c, seed=3)


def test_encoding_is_sparse(mb):
    rng = np.random.default_rng(0)
    kc = mb.encode_state(rng.normal(size=64))
    assert kc.shape == (KC_COUNT,)
    assert 0.05 <= kc.mean() <= 0.15  # ~10% sparse code
    assert set(np.unique(kc)) <= {0.0, 1.0}


def test_encoding_is_deterministic_per_state(mb):
    rng = np.random.default_rng(1)
    s = rng.normal(size=64)
    assert np.array_equal(mb.encode_state(s), mb.encode_state(s))


def test_similar_states_overlap_more_than_random(mb):
    rng = np.random.default_rng(2)
    s = rng.normal(size=64)
    near = s + rng.normal(size=64) * 0.05
    far = rng.normal(size=64)
    k0, k1, k2 = mb.encode_state(s), mb.encode_state(near), mb.encode_state(far)
    assert (k0 * k1).sum() >= (k0 * k2).sum()


def test_decide_respects_illegal_mask(mb):
    rng = np.random.default_rng(3)
    kc = mb.encode_state(rng.normal(size=64))
    mask = np.array([False, False, True, False])
    for _ in range(20):
        assert mb.decide(kc, mask) == 2


def test_decide_returns_valid_action(mb):
    rng = np.random.default_rng(4)
    kc = mb.encode_state(rng.normal(size=64))
    mask = np.array([True, True, False, True])
    for _ in range(50):
        assert mb.decide(kc, mask) in (0, 1, 3)


def test_value_estimate_bounded(mb):
    rng = np.random.default_rng(5)
    kc = mb.encode_state(rng.normal(size=64))
    v = mb.value_estimate(kc)
    assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


# -------------------------------------------------------- dopamine plasticity
def test_dpr_reward_predominantly_strengthens(mb):
    """Active rows get the DAN-gated update; inactive rows only get the small
    global STM relaxation, so reward moves active rows far more."""
    ds = DopamineSystem(mb)
    kc = mb.encode_state(np.random.default_rng(6).normal(size=64))
    before = mb.connectome.kc_to_mbon.astype(np.float64)
    ds.update(kc, reward=1.0)
    after = mb.connectome.kc_to_mbon.astype(np.float64)
    active_delta = np.abs(after[kc > 0] - before[kc > 0]).sum()
    inactive_delta = np.abs(after[kc == 0] - before[kc == 0]).sum()
    assert active_delta > inactive_delta
    assert np.abs(after[kc == 0] - before[kc == 0]).max() < 0.01  # only STM decay


def test_dpr_reward_increases_then_punishment_decreases(mb):
    ds = DopamineSystem(mb)
    kc = mb.encode_state(np.random.default_rng(7).normal(size=64))
    row = 17
    kc[:] = 0
    kc[row] = 1.0
    before = float(mb.connectome.kc_to_mbon[row].sum())
    ds.update(kc, reward=+1.0)
    mid = float(mb.connectome.kc_to_mbon[row].sum())
    assert mid >= before
    ds.update(kc, reward=-1.0)
    after = float(mb.connectome.kc_to_mbon[row].sum())
    assert after <= mid


def test_dpr_weights_stay_bounded(mb):
    ds = DopamineSystem(mb, DPRConfig(learning_rate=0.5))
    kc = mb.encode_state(np.random.default_rng(8).normal(size=64))
    for _ in range(50):
        ds.update(kc, reward=+1.0)
    assert mb.connectome.kc_to_mbon.max() <= 2.0


def test_rpe_uses_running_baseline(mb):
    ds = DopamineSystem(mb)
    kc = mb.encode_state(np.random.default_rng(9).normal(size=64))
    info = ds.update(kc, reward=0.4)
    assert info["rpe"] == pytest.approx(0.4, abs=1e-6)  # baseline starts at 0
    info2 = ds.update(kc, reward=0.4)
    assert info2["rpe"] < info["rpe"]  # baseline crept up → smaller RPE


def test_checkpoint_roundtrip(mb, tmp_path):
    ds = DopamineSystem(mb)
    kc = mb.encode_state(np.random.default_rng(10).normal(size=64))
    ds.update(kc, reward=0.3)
    path = str(tmp_path / "ckpt.npz")
    mb.save(path)
    mb2 = MushroomBody(64)
    mb2.load_weights(path)
    assert np.allclose(mb2.connectome.kc_to_mbon, mb.connectome.kc_to_mbon)
    assert mb2.temperature == mb.temperature
    assert N_MACRO_ACTIONS == 4
