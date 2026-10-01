"""FlyCommander — reward computer.

The Java agent does not judge the fly; it only reports state. Rewards are
computed here, Python-side, from *consecutive* observations plus the terminal
result: board development, life swings, commander damage, card advantage and
opponent removal all map to small potential-shaped signals; winning or losing
the game is the dominant terminal reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RewardWeights:
    life_delta: float = 0.02          # per life point swing (fly vs best opp)
    board_delta: float = 0.05         # per net creature advantage change
    hand_delta: float = 0.01          # card advantage
    opp_creature_kill: float = 0.15   # opponents lost creatures since last step
    took_commander_damage: float = -0.10
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    time_penalty: float = -0.001      # per decision step (prefer efficient wins)


@dataclass
class RewardBreakdown:
    reward: float
    terminal: bool
    components: dict[str, float] = field(default_factory=dict)


class RewardComputer:
    """Stateful: call :meth:`step` each observation, then :meth:`terminal`."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.w = weights or RewardWeights()
        self._prev: dict[str, Any] | None = None
        self._steps = 0

    def reset(self) -> None:
        self._prev = None
        self._steps = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _seat_split(obs: dict[str, Any]) -> tuple[dict, list[dict]]:
        players = obs.get("players", [])
        fly = next((p for p in players if p.get("isFly")), players[0] if players else {})
        opps = [p for p in players if p is not fly]
        return fly, opps

    @staticmethod
    def _score(p: dict) -> float:
        """Cheap board-score: creatures + small weight for other permanents."""
        board = p.get("board", [])
        return float(len(board))

    def step(self, obs: dict[str, Any]) -> RewardBreakdown:
        fly, opps = self._seat_split(obs)
        comp: dict[str, float] = {}

        life = float(fly.get("life", 40))
        opp_best_life = max((float(o.get("life", 40)) for o in opps), default=40.0)
        board_adv = self._score(fly) - sum(self._score(o) for o in opps) / max(1, len(opps))
        hand = float(fly.get("hand", 0))
        opp_creatures = sum(float(o.get("creatures", 0)) for o in opps)
        cmd_dmg = float(fly.get("commanderDamage", 0))

        if self._prev is not None:
            pf = self._prev["fly"]
            comp["life_delta"] = self.w.life_delta * ((life - pf["life"]) - 0.5 * (opp_best_life - self._prev["opp_best_life"]))
            comp["board_delta"] = self.w.board_delta * (board_adv - self._prev["board_adv"])
            comp["hand_delta"] = self.w.hand_delta * (hand - pf["hand"])
            comp["opp_removal"] = self.w.opp_creature_kill * max(
                0.0, self._prev["opp_creatures"] - opp_creatures)
            comp["commander_damage"] = self.w.took_commander_damage * max(
                0.0, cmd_dmg - self._prev["cmd_dmg"])

        self._prev = {
            "fly": {"life": life, "hand": hand},
            "opp_best_life": opp_best_life,
            "board_adv": board_adv,
            "opp_creatures": opp_creatures,
            "cmd_dmg": cmd_dmg,
        }
        self._steps += 1
        total = float(sum(comp.values())) + self.w.time_penalty
        # keep per-step signals small next to the terminal ±1
        total = float(np.clip(total, -0.5, 0.5))
        return RewardBreakdown(reward=total, terminal=False, components=comp)

    def terminal(self, result: dict[str, Any]) -> RewardBreakdown:
        fly_won = bool(result.get("flyWon"))
        is_draw = "draw" in str(result.get("reason", "")).lower()
        if is_draw:
            r = self.w.draw
        else:
            r = self.w.win if fly_won else self.w.loss
        # longer wins are worth slightly less (efficiency pressure)
        turns = float(result.get("turns", 0))
        if fly_won and turns > 0:
            r *= max(0.5, 1.0 - 0.01 * turns)
        return RewardBreakdown(
            reward=r,
            terminal=True,
            components={"turns": turns, "win": 1.0 if fly_won else 0.0},
        )
