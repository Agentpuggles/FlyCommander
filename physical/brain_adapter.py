"""FlyCommander physical-table mode — Fly brain adapter.

Feeds `PhysicalGameState` into the SAME brain used for Forge training —
no second brain, no second encoder. `to_fly_observation()` on the state
produces the exact dict shape the Forge agent emits; the sensory encoder
consumes it unchanged. Macro-action indices and names are shared.

The fly never sees hidden information: `to_fly_observation()` is built from
`public_observation` semantics — hand/library are counts only, and the
human's private zones never appear.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from brain.mushroom_body import ACTION_NAMES, MushroomBody
from flycommander.sensory_encoder import observation_to_state
from physical.state import PhysicalGameState


class FlyBrainAdapter:
    def __init__(self, mb: MushroomBody) -> None:
        self.mb = mb
        self.last_valences: list[float] | None = None
        self.last_kc_active: int = 0

    # ------------------------------------------------------------------
    def observe(self, state: PhysicalGameState) -> np.ndarray:
        """PhysicalGameState → sensory vector (existing encoder)."""
        obs = state.to_fly_observation()
        return observation_to_state(obs)

    def kc_code(self, state: PhysicalGameState) -> np.ndarray:
        """Sparse Kenyon-cell code for the current public state."""
        return self.mb.encode_state(self.observe(state))

    # ------------------------------------------------------------------
    def decide(self, state: PhysicalGameState,
               legal: tuple[bool, bool, bool, bool] | None = None
               ) -> dict[str, Any]:
        """Choose a macro action exactly like the Forge bridge does."""
        kc = self.kc_code(state)
        self.last_kc_active = int(kc.sum())
        mask = np.array(legal if legal is not None else (True, True, True, True))
        action = self.mb.decide(kc, mask)
        valences = self.mb.action_valences(kc, mask)
        self.last_valences = [None if not np.isfinite(v) else float(v)
                              for v in valences]
        return self.announce(action, state)

    # ------------------------------------------------------------------
    def announce(self, action: int, state: PhysicalGameState) -> dict[str, Any]:
        """Human-readable action announcement + decision telemetry."""
        fly_board = state.battlefield("fly")
        attackers = [o for o in fly_board
                     if o.is_creature and not o.summoning_sick]
        detail: dict[str, Any] = {"action": int(action),
                                  "actionName": ACTION_NAMES[action]}
        if action == 1:  # attack
            detail["attackers"] = [o.name for o in attackers[:3]]
            detail["target"] = "Flynn"
            detail["damage"] = sum(int(o.current_power() or 0)
                                   for o in attackers[:3])
        elif action == 0:  # play
            detail["note"] = "fly plays its best available card"
        elif action == 3:  # interact
            detail["note"] = "fly uses removal / interaction"
        scores = [round(v, 3) if v is not None else None
                  for v in (self.last_valences or [])]
        return {
            "decision": {
                "action": detail["actionName"],
                "detail": detail,
                "actionScores": {
                    name: scores[i] if i < len(scores) else None
                    for i, name in enumerate(ACTION_NAMES)
                },
                "kcActive": self.last_kc_active,
                "temperature": round(self.mb.temperature, 3),
                "valueBaseline": round(self.mb.value_baseline, 4),
            },
        }

    # ------------------------------------------------------------------
    def brain_panel(self) -> dict[str, Any]:
        """Telemetry for the UI brain panel (KC/MBON/DAN/RPE view)."""
        panel: dict[str, Any] = {
            "kcActive": self.last_kc_active,
            "kcTotal": self.mb.connectome.n_kc,
            "temperature": round(self.mb.temperature, 3),
            "valueBaseline": round(self.mb.value_baseline, 4),
        }
        if self.last_valences is not None:
            panel["actionScores"] = {
                name: (round(v, 3) if v is not None else None)
                for name, v in zip(ACTION_NAMES, self.last_valences)
            }
        weights = self.mb.connectome.kc_to_mbon
        panel["meanSynapticWeight"] = round(float(weights.mean()), 5)
        panel["strongSynapses"] = int((weights > 0.2).sum())
        return panel
