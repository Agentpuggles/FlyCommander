"""FlyCommander — Leaky Integrate-and-Fire neural engine.

Defaults follow the biophysical parameters used by the FlyWire drone project
(Shiu et al. 2024): resting potential -52 mV, threshold -45 mV, membrane
resistance 10 kOhm*cm^2, membrane capacitance 2 uF/cm^2, synaptic decay 5 ms,
synaptic weight 0.275 mV, refractory period 2.2 ms.
"""
from __future__ import annotations

import numpy as np


class LIFPopulation:
    """A population of leaky integrate-and-fire neurons sharing parameters."""

    def __init__(
        self,
        size: int,
        v_rest: float = -52.0,
        v_thresh: float = -45.0,
        r_m: float = 10.0,
        c_m: float = 2.0,
        dt: float = 0.5,
        refractory_ms: float = 2.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.size = int(size)
        self.v_rest = float(v_rest)
        self.v_thresh = float(v_thresh)
        self.r_m = float(r_m)
        self.c_m = float(c_m)
        self.dt = float(dt)
        self.refractory_ms = float(refractory_ms)
        self.rng = rng or np.random.default_rng()

        self.v = np.full(self.size, self.v_rest, dtype=np.float64)
        self.last_spike_ms = np.full(self.size, -np.inf, dtype=np.float64)
        self.time_ms = 0.0

    # ------------------------------------------------------------------
    @property
    def tau_ms(self) -> float:
        """Membrane time constant tau = R * C."""
        return self.r_m * self.c_m

    def reset(self) -> None:
        self.v[:] = self.v_rest
        self.last_spike_ms[:] = -np.inf
        self.time_ms = 0.0

    # ------------------------------------------------------------------
    def step(self, input_current: np.ndarray) -> np.ndarray:
        """Advance one dt with the given input currents; return spike mask.

        dV/dt = (-(V - V_rest) + R*I) / tau
        Spikes reset V to rest and start the refractory clock.
        """
        input_current = np.asarray(input_current, dtype=np.float64)
        if input_current.shape != (self.size,):
            raise ValueError(f"input shape {input_current.shape} != ({self.size},)")

        in_refractory = (self.time_ms - self.last_spike_ms) < self.refractory_ms

        leak = -(self.v - self.v_rest)
        drive = self.r_m * input_current
        d_v = (leak + drive) / self.tau_ms

        self.v += d_v * self.dt
        self.v[in_refractory] = self.v_rest  # clamped during refractory period

        spiked = (self.v >= self.v_thresh) & ~in_refractory
        self.v[spiked] = self.v_rest
        self.last_spike_ms[spiked] = self.time_ms
        self.time_ms += self.dt
        return spiked

    def run(self, input_current: np.ndarray, steps: int) -> np.ndarray:
        """Run several timesteps; return the OR of spike masks across steps."""
        any_spikes = np.zeros(self.size, dtype=bool)
        for _ in range(int(steps)):
            any_spikes |= self.step(input_current)
        return any_spikes

    def rates(self, input_current: np.ndarray, steps: int = 20) -> np.ndarray:
        """Approximate firing rates (Hz-ish proxy): fraction of steps spiking."""
        count = np.zeros(self.size, dtype=np.float64)
        for _ in range(int(steps)):
            count += self.step(input_current)
        return count / float(steps)


def winner_take_all(activity: np.ndarray, keep_fraction: float = 0.10) -> np.ndarray:
    """Sparse code: exactly the top `keep_fraction` of units stay active."""
    activity = np.asarray(activity, dtype=np.float64)
    n_keep = max(1, int(round(keep_fraction * activity.size)))
    if not np.any(activity > 0):
        return np.zeros_like(activity)
    order = np.argsort(-activity, kind="stable")
    out = np.zeros_like(activity)
    out[order[:n_keep]] = 1.0
    return out
