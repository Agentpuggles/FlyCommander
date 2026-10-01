"""FlyCommander — real-time tabletop integration (v1 stub).

Wires the planned perception stack into the brain:

    camera frame → CardDetector → CardIdentifier → ZoneTracker → observation
    observation → sensory_encoder → MushroomBody → suggestion overlay

The Forge side stays authoritative for rules; here the fly watches a physical
table and suggests actions. Requires the vision models to be trained first.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vision.card_detector import CardDetector
from vision.card_identifier import CardIdentifier
from vision.zone_tracker import ZoneTracker


class LivePlay:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.detector = CardDetector()
        self.identifier = CardIdentifier()
        self.zones = ZoneTracker()

    def observe_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """One camera frame → structured observation (stub pipeline)."""
        boxes = self.detector.detect(frame)
        del boxes  # identification and zone mapping land with real models
        return {"cards": [], "zones": self.zones.snapshot()}

    def suggest(self, observation: dict[str, Any]) -> int:
        """Brain's macro suggestion for the current physical state."""
        raise NotImplementedError("requires trained vision + calibrated zones")
