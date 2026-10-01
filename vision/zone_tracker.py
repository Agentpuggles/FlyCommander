"""FlyCommander — spatial zone classification (v1 stub).

Planned: ArUco markers on playmat corners → homography → per-player zone
polygons (battlefield / graveyard / command zone / library), mapping detected
card centers into named zones.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ZoneTracker:
    def __init__(self, marker_config: str | None = None) -> None:
        self.marker_config = marker_config
        self._homography: np.ndarray | None = None
        self._zones: dict[str, np.ndarray] = {}

    def calibrate(self, frame: np.ndarray) -> bool:
        """Detect ArUco markers and compute the table homography (stub)."""
        _ = frame
        return False

    def classify(self, card_center_xy: tuple[float, float]) -> str:
        """Map a table-space point to a zone name (stub)."""
        _ = card_center_xy
        return "unknown"

    def snapshot(self) -> dict[str, Any]:
        """Observed zone layout as a structured dict (stub)."""
        return {"zones": {}, "calibrated": self._homography is not None}
