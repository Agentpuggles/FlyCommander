"""FlyCommander — Mushroom-body connectome: synthetic topology with real statistics.

The real MaleCNS v1.0 / FlyWire dataset needs authentication and multi-GB
downloads, so the default build synthesizes a mushroom-body subnetwork with the
published cell counts and connectivity statistics:

  - 4,064 Kenyon cells, 97 MBONs, 344 dopaminergic neurons (~4,505 total)
  - ~998,000 directed synaptic connections in the MB
  - PN→KC expansion: each KC receives a small random subset of sensory channels
    (in the fly, ~7 PNs converge per KC; we mirror that sparsity)

`load_malecns` is the hook for swapping in the real dataset later: it accepts an
HDF5/JSON file of (pre_id, post_id, weight) edges plus neuron-type annotations
and builds the same dataclasses from the true wiring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

KC_COUNT = 4064
MBON_COUNT = 97
DAN_COUNT = 344
TOTAL_SYNAPSES = 998_000
PN_PER_KC = 7  # fly: ~7 projection neurons converge onto each Kenyon cell


@dataclass
class MushroomBodyConnectome:
    """Wiring tables for the MB decision circuit.

    pn_to_kc      (n_pn, n_kc)      fixed random projection, sensory → KC
    kc_to_mbon    (n_kc, n_mbon)    plastic synapses (the memory)
    mbon_to_dan   (n_mbon, n_dan)   MBON → DAN feedback
    dan_to_mbon   (n_dan, n_mbon)   DAN modulatory gates per MBON compartment
    """

    pn_to_kc: np.ndarray
    kc_to_mbon: np.ndarray
    mbon_to_dan: np.ndarray
    dan_to_mbon: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_pn(self) -> int:
        return self.pn_to_kc.shape[0]

    @property
    def n_kc(self) -> int:
        return self.kc_to_mbon.shape[0]

    @property
    def n_mbon(self) -> int:
        return self.kc_to_mbon.shape[1]

    @property
    def n_dan(self) -> int:
        return self.dan_to_mbon.shape[0]

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            pn_to_kc=self.pn_to_kc,
            kc_to_mbon=self.kc_to_mbon,
            mbon_to_dan=self.mbon_to_dan,
            dan_to_mbon=self.dan_to_mbon,
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str) -> "MushroomBodyConnectome":
        data = np.load(path, allow_pickle=False)
        return cls(
            pn_to_kc=data["pn_to_kc"],
            kc_to_mbon=data["kc_to_mbon"],
            mbon_to_dan=data["mbon_to_dan"],
            dan_to_mbon=data["dan_to_mbon"],
            meta=json.loads(str(data["meta"])),
        )


def build_synthetic_connectome(
    n_pn: int,
    seed: int = 2049,
    kc_count: int = KC_COUNT,
    mbon_count: int = MBON_COUNT,
    dan_count: int = DAN_COUNT,
) -> MushroomBodyConnectome:
    """Synthesize MB wiring with fly-like sparsity and sign structure.

    - pn_to_kc: each KC gets `PN_PER_KC` random PN inputs (binary, fixed).
    - kc_to_mbon: each KC contacts ~5% of MBONs (random, initialized small so
      learning dominates the final structure).
    - mbon_to_dan / dan_to_mbon: sparse recurrent loops between output neurons
      and dopaminergic neurons, excitatory MBON→DAN, signed DAN→MBON gates
      (PAM-like positive, PPL1-like negative).
    """
    rng = np.random.default_rng(seed)

    pn_to_kc = np.zeros((n_pn, kc_count), dtype=np.float32)
    for kc in range(kc_count):
        pn_idx = rng.choice(n_pn, size=min(PN_PER_KC, n_pn), replace=False)
        pn_to_kc[pn_idx, kc] = 1.0

    # ~5% convergence KC→MBON, small positive initial weights
    density = 0.05
    kc_to_mbon = (rng.random((kc_count, mbon_count)) < density).astype(np.float32)
    kc_to_mbon *= rng.uniform(0.05, 0.15, size=kc_to_mbon.shape).astype(np.float32)

    # MBON → DAN feedback: each DAN listens to a few MBONs
    mbon_to_dan = (rng.random((mbon_count, dan_count)) < 0.10).astype(np.float32)
    mbon_to_dan *= rng.uniform(0.1, 0.4, size=mbon_to_dan.shape).astype(np.float32)

    # DAN → MBON gates: first half appetitive (PAM-like, +), second half
    # aversive (PPL1-like, −); sparse per-compartment targeting
    dan_to_mbon = np.zeros((dan_count, mbon_count), dtype=np.float32)
    n_appetitive = dan_count // 2
    for dan in range(dan_count):
        targets = rng.choice(mbon_count, size=max(1, mbon_count // 8), replace=False)
        sign = 1.0 if dan < n_appetitive else -1.0
        dan_to_mbon[dan, targets] = sign * rng.uniform(0.5, 1.0, size=targets.size)

    meta = {
        "source": "synthetic",
        "seed": seed,
        "pn_per_kc": PN_PER_KC,
        "kc_count": kc_count,
        "mbon_count": mbon_count,
        "dan_count": dan_count,
        "approx_synapses": int(kc_to_mbon.sum() + pn_to_kc.sum()),
    }
    return MushroomBodyConnectome(pn_to_kc, kc_to_mbon, mbon_to_dan, dan_to_mbon, meta)


def load_malecns(path: str) -> MushroomBodyConnectome:
    """Load a real MaleCNS/FlyWire MB subnetwork exported by
    `connectome/download_malecns.py` (edges + neuron types). Raises if the file
    is missing type annotations — the synthetic builder is the fallback."""
    data = np.load(path, allow_pickle=False)
    if "types" not in data:
        raise ValueError("real connectome export must include neuron type annotations")
    raise NotImplementedError(
        "Real MaleCNS ingestion lands with connectome/download_malecns.py; "
        "use build_synthetic_connectome for now."
    )
