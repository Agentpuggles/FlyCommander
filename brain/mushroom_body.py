"""FlyCommander — the mushroom-body decision circuit.

Biological pipeline: sensory channels (PNs) → sparse Kenyon-cell code → KC→MBON
synapses (the plastic memory) → MBON firing → action valence. Dopaminergic
neurons gate learning via `brain/dopamine_plasticity.py`.

Macro actions the MB chooses among (indices must match the Java bridge's
proposal stream):
    0 cast/play best spell/land per Forge AI   (approach)
    1 attack with recommended attackers          (approach)
    2 hold — do nothing this phase               (avoid)
    3 use forge AI's removal/interaction pick    (targeted approach)
"""
from __future__ import annotations

import numpy as np

from brain.connectome import MushroomBodyConnectome, build_synthetic_connectome
from brain.lif_engine import LIFPopulation, winner_take_all

N_MACRO_ACTIONS = 4
ACTION_NAMES = ("play", "attack", "hold", "interact")


class MushroomBody:
    """Spiking MB circuit producing action valences over Forge macro-actions."""

    def __init__(
        self,
        n_sensory_channels: int,
        connectome: MushroomBodyConnectome | None = None,
        seed: int = 2049,
    ) -> None:
        self.n_sensory = int(n_sensory_channels)
        self.connectome = connectome or build_synthetic_connectome(n_pn=self.n_sensory, seed=seed)
        if self.connectome.n_pn != self.n_sensory:
            raise ValueError("connectome PN count must match sensory channel count")

        self.kc = LIFPopulation(self.connectome.n_kc, rng=np.random.default_rng(seed))
        self.mbon = LIFPopulation(self.connectome.n_mbon, rng=np.random.default_rng(seed + 1))

        # Learned readout MBON → macro action, initialized near-uniform.
        rng = np.random.default_rng(seed + 2)
        self.w_mbon_action = rng.normal(0.0, 0.05, size=(self.connectome.n_mbon, N_MACRO_ACTIONS))
        self.action_eligibility: np.ndarray | None = None

        # Learning bookkeeping (used by dopamine_plasticity)
        self.value_baseline = 0.0
        self.temperature = 0.5

    # ------------------------------------------------------------------
    def encode_state(self, state_vec: np.ndarray) -> np.ndarray:
        """Dense game features → sparse KC spike pattern.

        PN fan-in (fixed random projection) then APL-like winner-take-all
        (~10% of KCs stay active — the fly's sparse odor code).
        """
        state_vec = np.asarray(state_vec, dtype=np.float64)
        if state_vec.shape != (self.n_sensory,):
            raise ValueError(f"state vector must have {self.n_sensory} channels")
        pn_activity = np.tanh(state_vec)          # PN firing-rate proxy in (-1, 1)
        kc_input = self.connectome.pn_to_kc.T @ pn_activity
        kc_input = np.clip(kc_input, 0.0, None)
        return winner_take_all(kc_input, keep_fraction=0.10)

    # ------------------------------------------------------------------
    def decide(self, kc_spikes: np.ndarray, legal_mask: np.ndarray) -> int:
        """KC spikes → MBON LIF response → masked softmax over macro actions."""
        legal_mask = np.asarray(legal_mask, dtype=bool)
        if legal_mask.shape != (N_MACRO_ACTIONS,) or not legal_mask.any():
            return 2  # hold is always safe

        mbon_current = kc_spikes @ self.connectome.kc_to_mbon
        mbon_activity = self.mbon.rates(mbon_current, steps=8)

        action_scores = mbon_activity @ self.w_mbon_action
        action_scores[~legal_mask] = -np.inf

        self.action_eligibility = kc_spikes.copy()  # for DPR updates
        return self._softmax_sample(action_scores, self.temperature)

    def action_valences(self, kc_spikes: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Exposed MBON valences per action (for logging/inspection)."""
        mbon_current = kc_spikes @ self.connectome.kc_to_mbon
        mbon_activity = self.mbon.rates(mbon_current, steps=8)
        scores = mbon_activity @ self.w_mbon_action
        scores[~np.asarray(legal_mask, dtype=bool)] = -np.inf
        return scores

    # ------------------------------------------------------------------
    def _softmax_sample(self, scores: np.ndarray, temperature: float) -> int:
        finite = np.isfinite(scores)
        if not finite.any():
            return 2
        z = scores[finite] / max(1e-3, float(temperature))
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(np.flatnonzero(finite)[self.kc.rng.choice(p.size, p=p)])

    # ------------------------------------------------------------------
    def value_estimate(self, kc_spikes: np.ndarray) -> float:
        """Critic-ish scalar read from MBON activity (used for RPE)."""
        mbon_current = kc_spikes @ self.connectome.kc_to_mbon
        mbon_activity = self.mbon.rates(mbon_current, steps=8)
        return float(mbon_activity.mean() * 2.0 - 1.0)

    def anneal(self, factor: float = 0.995, floor: float = 0.15) -> None:
        self.temperature = max(floor, self.temperature * factor)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            kc_to_mbon=self.connectome.kc_to_mbon,
            w_mbon_action=self.w_mbon_action,
            value_baseline=self.value_baseline,
            temperature=self.temperature,
        )

    def load_weights(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        if data["kc_to_mbon"].shape != self.connectome.kc_to_mbon.shape:
            raise ValueError("checkpoint connectome shape mismatch")
        self.connectome.kc_to_mbon = data["kc_to_mbon"]
        self.w_mbon_action = data["w_mbon_action"]
        self.value_baseline = float(data["value_baseline"])
        self.temperature = float(data["temperature"])
