"""FlyCommander — tapped/untapped detection (v1 stub).

Planned: card bounding-box orientation analysis — tapped cards rotate ~90°,
changing the box aspect ratio and corner order from the detector.
"""
from __future__ import annotations

import numpy as np

from vision.card_detector import CardBox


class TapDetector:
    """Classify card orientation from bounding-box geometry."""

    TAP_ANGLE_THRESHOLD = 45.0

    @staticmethod
    def is_tapped(box: CardBox) -> bool:
        return abs(box.angle_deg) >= TapDetector.TAP_ANGLE_THRESHOLD

    @staticmethod
    def estimate_angle(box: CardBox, frame: np.ndarray) -> float:
        """Refine rotation from image content (stub — returns detector angle)."""
        _ = frame
        return box.angle_deg
