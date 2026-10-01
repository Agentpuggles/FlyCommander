"""Encoder separation probe (research/audit tooling — NOT part of the runtime).

Builds strategically OPPOSITE observation pairs, encodes each with the real
sensory encoder + mushroom body, and reports how much the sparse KC codes
overlap. Chance overlap for two independent ~10%-active random codes is
~0.05 (Jaccard) / ~0.10 (min-overlap); values far above that mean the encoder
leaves strategically distinct states sharing most of their KC ensemble.

Run from repo root:  PYTHONPATH=. .venv/bin/python research/sensory_encoding/encoder_probe.py

Results (2026-10-01, seed 2049, n_kc=4064):
    winning_vs_losing   jaccard=0.17 min_overlap=0.29 state_L2=2.38
    low_vs_high_life    jaccard=0.58 min_overlap=0.73 state_L2=0.92
    turn2_vs_turn40     jaccard=0.38 min_overlap=0.55 state_L2=1.60
    chance              jaccard=0.052 min_overlap=0.102
"""
from __future__ import annotations

import numpy as np

from flycommander.sensory_encoder import observation_to_state
from brain.mushroom_body import MushroomBody

SEED = 2049


def player(is_fly, life=40, hand=5, creatures=0, lands=0, gy=0, exile=0,
           cmdr=0, cmdzone=1, lib=45, board=()):
    return {
        "isFly": is_fly,
        "life": life,
        "poison": 0,
        "hand": hand,
        "creatures": creatures,
        "lands": lands,
        "graveyard": gy,
        "exile": exile,
        "commanderDamage": cmdr,
        "commandZone": cmdzone,
        "library": lib,
        "board": [{"n": n} for n in board],
    }


def opp(life=40, creatures=0, board=()):
    return player(False, life=life, creatures=creatures, board=board)


def obs_win():
    """Fly dominating: big board, healthy, opponents battered."""
    fly = player(True, life=38, hand=6, creatures=8, lands=9, gy=3, lib=38,
                 board=["Serra Angel", "Serra Angel", "Grizzly Bears",
                        "Grizzly Bears", "Hill Giant", "Hill Giant",
                        "Craw Wurm", "Craw Wurm", "Plains", "Plains"])
    o1 = opp(life=6, creatures=1, board=["Swamp"])
    o2 = opp(life=4, creatures=1, board=["Island"])
    o3 = opp(life=9, creatures=2, board=["Mountain", "Forest"])
    return {"players": [fly, o1, o2, o3], "stackSize": 0, "attackers": 0,
            "blockers": 0, "turn": 18}


def obs_lose():
    """Fly nearly dead: empty board, opponents healthy."""
    fly = player(True, life=2, hand=1, creatures=0, lands=2, gy=12, lib=20,
                 cmdr=32, board=["Plains", "Plains"])
    o1 = opp(life=39, creatures=7,
             board=["Grizzly Bears", "Grizzly Bears", "Hill Giant",
                    "Hill Giant", "Craw Wurm", "Swamp", "Swamp", "Swamp"])
    o2 = opp(life=40, creatures=6,
             board=["Serra Angel", "Serra Angel", "Island", "Island",
                    "Island", "Island", "Island"])
    o3 = opp(life=37, creatures=5,
             board=["Hill Giant", "Mountain", "Mountain", "Forest", "Forest"])
    return {"players": [fly, o1, o2, o3], "stackSize": 0, "attackers": 0,
            "blockers": 0, "turn": 19}


def obs_low_life():
    fly = player(True, life=2, hand=5, creatures=4, lands=6, lib=35,
                 board=["Serra Angel", "Grizzly Bears", "Hill Giant", "Plains"])
    o1 = opp(life=30, creatures=3, board=["Swamp", "Swamp", "Craw Wurm"])
    o2 = opp(life=28, creatures=3, board=["Island", "Island", "Hill Giant"])
    o3 = opp(life=33, creatures=2, board=["Mountain", "Forest"])
    return {"players": [fly, o1, o2, o3], "stackSize": 0, "attackers": 0,
            "blockers": 0, "turn": 12}


def obs_high_life():
    o = obs_low_life()
    o["players"][0]["life"] = 39
    return o


def obs_early():
    o = obs_low_life()
    fly = o["players"][0]
    fly.update({"life": 40, "hand": 7, "creatures": 0, "lands": 2,
                "library": 51, "commanderDamage": 0})
    fly["board"] = [{"n": "Plains"}, {"n": "Island"}]
    o["turn"] = 2
    for p in o["players"][1:]:
        p["life"] = 40
        p["creatures"] = 0
        p["board"] = p["board"][:2]
    return o


def obs_late():
    o = obs_low_life()
    o["turn"] = 40
    return o


def overlap(a: np.ndarray, b: np.ndarray) -> dict:
    A = set(np.flatnonzero(a).tolist())
    B = set(np.flatnonzero(b).tolist())
    inter = len(A & B)
    union = len(A | B)
    return {
        "jaccard": inter / union if union else 0.0,
        "min_overlap": inter / min(len(A), len(B)) if A and B else 0.0,
        "sizes": (len(A), len(B)),
    }


mb = MushroomBody(n_sensory_channels=64, seed=SEED)

PAIRS = {
    "winning_vs_losing": (obs_win(), obs_lose()),
    "low_vs_high_life": (obs_low_life(), obs_high_life()),
    "turn2_vs_turn40": (obs_early(), obs_late()),
}

print(f"seed={SEED}  n_kc={mb.connectome.n_kc}")
for name, (o1, o2) in PAIRS.items():
    s1, s2 = observation_to_state(o1), observation_to_state(o2)
    k1 = mb.encode_state(s1)
    k2 = mb.encode_state(s2)
    m = overlap(k1, k2)
    state_l2 = float(np.linalg.norm(s1 - s2))
    print(f"{name:22s} jaccard={m['jaccard']:.2f} min_overlap={m['min_overlap']:.2f} "
          f"sizes={m['sizes']} state_L2={state_l2:.2f}")

rng = np.random.default_rng(0)
n = mb.connectome.n_kc
c = [overlap(rng.random(n) < 0.1, rng.random(n) < 0.1) for _ in range(200)]
print(f"chance: jaccard={np.mean([x['jaccard'] for x in c]):.3f} "
      f"min_overlap={np.mean([x['min_overlap'] for x in c]):.3f}")
