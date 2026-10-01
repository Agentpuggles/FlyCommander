"""FlyCommander — card identification from dewarped crops (v1 stub).

Planned: perceptual-hash + ORB feature matching against the Scryfall image
corpus (~30k unique arts), or CLIP/DINOv2 embedding similarity. Stubbed until
the detector produces crops to identify.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class CardIdentifier:
    """Match dewarped card crops against a Scryfall-derived reference set."""

    def __init__(self, reference_dir: str | None = None) -> None:
        self.reference_dir = reference_dir
        self._embeddings: dict[str, np.ndarray] = {}

    def build_index(self) -> int:
        raise NotImplementedError("requires the Scryfall image corpus")

    def identify(self, crop: np.ndarray) -> dict[str, Any]:
        _ = crop
        return {"name": None, "confidence": 0.0}
