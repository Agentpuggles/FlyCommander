"""FlyCommander — card region detection (perception layer, v1 stub).

Planned pipeline (see README): YOLOv8 fine-tuned on ~500 annotated tabletop
MTG frames → per-card bounding boxes with rotation estimates. The interface
below is what live_play.py will consume; the implementation is stubbed until
a labeled dataset and training run exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CardBox:
    xyxy: tuple[float, float, float, float]
    confidence: float
    angle_deg: float = 0.0   # ~0 untapped, ~90 tapped


class CardDetector:
    """Detect MTG card rectangles in an overhead camera frame."""

    def __init__(self, model_path: str | None = None,
                 conf: float = 0.5) -> None:
        self.model_path = model_path
        self.conf = conf
        if model_path:
            raise NotImplementedError(
                "YOLOv8 MTG detector not trained yet — collect frames first")

    def detect(self, frame: np.ndarray) -> list[CardBox]:
        """Return card bounding boxes for one BGR frame (stub)."""
        _ = frame.shape  # a real impl uses this
        return []
