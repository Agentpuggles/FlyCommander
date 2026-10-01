"""FlyCommander — real MaleCNS v1.0 mushroom-body extraction (stub).

The synthetic connectome in ``brain/connectome.py`` already matches the
published cell counts and sparsity statistics. This module is the planned
pipeline for swapping in the *real* wiring once FlyWire access is available:

1. Obtain a FlyWire (Codex) auth token: https://codex.flywire.ai → set it in
   the ``FLYWIRE_TOKEN`` environment variable.
2. Query the MaleCNS v1.0 / FAFB segmentation for the mushroom-body core:
   Kenyon cells (~4,064), MBONs (~97), DANs (~344).
3. Pull the synapse table for those root IDs (``synapse_cleft`` / partner
   tables via the FlyWire API or neuprint-python).
4. Reduce to (pre_id, post_id, weight) adjacency, map types, and export an
   ``.npz`` that ``MushroomBodyConnectome.load`` can consume.

Run:
    python connectome/download_malecns.py --out data/mushroom_body_real.npz
"""
from __future__ import annotations

import argparse
import os


MB_QUERY_HINTS = {
    "KC":  {"type": ["Kenyon", "KC"]},
    "MBON": {"type": ["MBON"]},
    "DAN":  {"type": ["DAN", "PPL1", "PAM"]},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mushroom_body_real.npz")
    args = ap.parse_args()

    token = os.environ.get("FLYWIRE_TOKEN")
    if not token:
        print("Set FLYWIRE_TOKEN (https://codex.flywire.ai) and re-run.")
        return 2

    raise NotImplementedError(
        "FlyWire ingestion is not implemented in this build. Steps: "
        "query MB root IDs by type → pull synapse partners → build adjacency "
        "→ save npz with keys pn_to_kc, kc_to_mbon, mbon_to_dan, dan_to_mbon.")


if __name__ == "__main__":
    raise SystemExit(main())
