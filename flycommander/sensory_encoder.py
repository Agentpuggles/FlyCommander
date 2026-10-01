"""FlyCommander — sensory encoding: Forge observation → dense state vector.

The Java agent publishes a structured observation (life totals, zone counts,
board identities). This module compresses it into a fixed-length dense
sensory vector (the "projection neuron" activity) that the mushroom body
encodes into a sparse Kenyon-cell population code.

Feature layout (N_SENSORY = 64 channels, grouped):
    [0:4]     per-seat life, normalized: 1 - life/40 (0 = full life, 1 = dead)
    [4:8]     per-seat poison / 10
    [8:12]    per-seat hand size / 8
    [12:16]   per-seat creature count / 15
    [16:20]   per-seat land count / 20
    [20:24]   per-seat graveyard size / 40
    [24:28]   per-seat exile size / 20
    [28:32]   per-seat commander damage taken / 40
    [32:36]   per-seat command-zone count / 2
    [36:40]   per-seat library size / 60
    [40:44]   stack size / 10, attackers / 10, blockers / 10, turn / 30
    [44:56]   fly's battlefield card-identity hash channels (12 buckets)
    [56:64]   spare (currently zero)
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

N_SENSORY = 64
N_SEATS = 4


def _bucket(name: str, n_buckets: int) -> int:
    """Stable hash of a card name into [0, n_buckets)."""
    h = hashlib.md5(name.lower().encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def observation_to_state(obs: dict[str, Any]) -> np.ndarray:
    """Convert one Java-agent observation dict into the dense sensory vector."""
    v = np.zeros(N_SENSORY, dtype=np.float64)
    players = obs.get("players", [])
    if not players:
        return v

    # put the fly first
    fly = next((p for p in players if p.get("isFly")), players[0])
    seats = [fly] + [p for p in players if p is not fly][: N_SEATS - 1]

    for i, p in enumerate(seats):
        base = i
        v[0 * N_SEATS + base] = 1.0 - min(1.0, p.get("life", 40) / 40.0)
        v[1 * N_SEATS + base] = min(1.0, p.get("poison", 0) / 10.0)
        v[2 * N_SEATS + base] = min(1.0, p.get("hand", 0) / 8.0)
        v[3 * N_SEATS + base] = min(1.0, p.get("creatures", 0) / 15.0)
        v[4 * N_SEATS + base] = min(1.0, p.get("lands", 0) / 20.0)
        v[5 * N_SEATS + base] = min(1.0, p.get("graveyard", 0) / 40.0)
        v[6 * N_SEATS + base] = min(1.0, p.get("exile", 0) / 20.0)
        v[7 * N_SEATS + base] = min(1.0, p.get("commanderDamage", 0) / 40.0)
        v[8 * N_SEATS + base] = min(1.0, p.get("commandZone", 0) / 2.0)
        v[9 * N_SEATS + base] = min(1.0, p.get("library", 0) / 60.0)

    v[40] = min(1.0, obs.get("stackSize", 0) / 10.0)
    v[41] = min(1.0, obs.get("attackers", 0) / 10.0)
    v[42] = min(1.0, obs.get("blockers", 0) / 10.0)
    v[43] = min(1.0, obs.get("turn", 0) / 30.0)

    # fly's board → 12 identity-hash channels (soft counts)
    for card in fly.get("board", []):
        b = _bucket(card.get("n", ""), 12)
        v[44 + b] = min(1.5, v[44 + b] + 0.25)

    return np.clip(v, 0.0, 2.0)
