"""FlyCommander — dopaminergic plasticity rule (DPR).

Reward arrives from the Python-side reward computer (reward_shaping.py). DANs
convert the RPE (reward minus value baseline) into per-MBON modulatory gates
(PAM-like appetitive / PPL1-like aversive), then plastic KC→MBON synapses
update only where KCs were active at decision time. STM→LTM consolidation
relaxes weights toward a decaying equilibrium, mimicking the fly's memory
phases (γ = labile STM, αβ = stable LTM).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain.mushroom_body import MushroomBody


@dataclass
class DPRConfig:
    learning_rate: float = 0.01
    rpe_gain: float = 1.0
    ltm_decay: float = 0.9995      # STM decay toward LTM equilibrium
    consolidation: float = 0.5     # weight on consolidated (decayed) component
    w_min: float = 0.0
    w_max: float = 2.0
    baseline_rate: float = 0.01    # value-baseline learning rate


class DopamineSystem:
    def __init__(self, mb: MushroomBody, config: DPRConfig | None = None) -> None:
        self.mb = mb
        self.cfg = config or DPRConfig()

    # ------------------------------------------------------------------
    def dan_activation(self, reward: float, rpe: float) -> np.ndarray:
        """Per-DAN activation: PAM (first half) for good news, PPL1 (second
        half) for bad news, scaled by |RPE|."""
        n_dan = self.mb.connectome.n_dan
        n_app = n_dan // 2
        act = np.zeros(n_dan, dtype=np.float64)
        signal = self.cfg.rpe_gain * rpe
        if signal > 0:
            act[:n_app] = min(1.0, signal)
        else:
            act[n_app:] = min(1.0, -signal)
        return act

    # ------------------------------------------------------------------
    def update(self, kc_spikes: np.ndarray, reward: float) -> dict:
        """One DPR update. Returns diagnostics (RPE, DAN drive, mean weight)."""
        mb = self.mb
        kc_spikes = np.asarray(kc_spikes, dtype=np.float64)
        rpe = float(reward) - mb.value_baseline

        # --- DAN activation from RPE
        dan_act = self.dan_activation(reward, rpe)

        # --- modulatory gate per MBON compartment (signed, sparse)
        dan_gate = self.mb.connectome.dan_to_mbon.T @ dan_act

        # --- plasticity only at KCs that were active at decision time
        active = kc_spikes > 0
        w = mb.connectome.kc_to_mbon.astype(np.float64)
        delta = self.cfg.learning_rate * np.outer(active * kc_spikes, dan_gate)
        w_new = np.clip(w + delta, self.cfg.w_min, self.cfg.w_max)

        # --- STM→LTM: blend updated weights toward decayed equilibrium
        w_ltm = w * self.cfg.ltm_decay
        mb.connectome.kc_to_mbon = (
            (1.0 - self.cfg.consolidation) * w_new + self.cfg.consolidation * w_ltm
        ).astype(np.float32)

        # --- critic baseline creep
        mb.value_baseline += self.cfg.baseline_rate * rpe

        return {
            "reward": float(reward),
            "rpe": rpe,
            "dan_drive": float(np.abs(dan_gate).sum()),
            "mean_weight": float(mb.connectome.kc_to_mbon.mean()),
            "baseline": float(mb.value_baseline),
        }
