"""FlyCommander physical-table mode — stable card tracker.

Maintains tracking identities across frames so a small shift never looks like
leave+enter. Matching is greedy IoU with a centroid-distance fallback.
Disappeared tracks survive a configurable grace period (occlusion) before
being retired.

Orientation is tracked per frame and smoothed; tap transitions are emitted
only when the smoothed orientation crosses a configurable angle threshold
*and* settles for `settle_frames` consecutive frames (hysteresis — prevents
flicker while a player rotates a card).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physical.state import angular_distance, is_tapped_orientation, now


@dataclass
class Detection:
    """One raw detector output for a single frame (no identity yet)."""
    bbox: tuple[float, float, float, float]   # x, y, w, h (table px)
    angle_deg: float = 0.0
    confidence: float = 1.0

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Track:
    track_id: str
    bbox: tuple[float, float, float, float]
    angle_deg: float
    confidence: float
    last_seen: float
    missed_frames: int = 0
    age_frames: int = 1
    # orientation smoothing + tap hysteresis
    tapped: bool = False
    _pending_tapped: bool | None = None
    _pending_frames: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)  # (t, angle)

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class CardTracker:
    """Greedy IoU tracker with occlusion grace and tap hysteresis."""

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_centroid_dist: float = 120.0,
        grace_frames: int = 10,
        tap_threshold_deg: float = 90.0,
        tap_tolerance_deg: float = 30.0,
        settle_frames: int = 2,
        angle_smooth_alpha: float = 0.6,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.grace_frames = grace_frames
        self.tap_threshold_deg = tap_threshold_deg
        self.tap_tolerance_deg = tap_tolerance_deg
        self.settle_frames = settle_frames
        self.angle_smooth_alpha = angle_smooth_alpha
        self.tracks: dict[str, Track] = {}
        self._next_id = 1
        self._frame = 0

    # ------------------------------------------------------------------
    def update(self, detections: list[Detection]) -> dict[str, dict[str, Any]]:
        """Advance one frame. Returns per-track update report including tap
        transition events (`tapEvents`)."""
        self._frame += 1
        report: dict[str, dict[str, Any]] = {}
        unmatched_dets = list(range(len(detections)))

        # --- match existing tracks (greedy, best IoU first) ---------------
        pairs: list[tuple[float, str, int]] = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(detections):
                ov = iou(tr.bbox, det.bbox)
                if ov >= self.iou_threshold:
                    pairs.append((ov, tid, di))
                elif (angular_distance(tr.angle_deg, det.angle_deg) < 15.0
                      and _dist(tr.centroid, det.centroid) < self.max_centroid_dist):
                    pairs.append((0.05, tid, di))  # weak centroid match
        pairs.sort(reverse=True)

        used_tracks: set[str] = set()
        for score, tid, di in pairs:
            if tid in used_tracks or di not in unmatched_dets:
                continue
            used_tracks.add(tid)
            unmatched_dets.remove(di)
            tr = self.tracks[tid]
            det = detections[di]
            report[tid] = self._update_track(tr, det)

        # --- unmatched tracks: occlusion or retirement --------------------
        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks:
                tr.missed_frames += 1
                if tr.missed_frames > self.grace_frames:
                    report[tid] = {"status": "retired"}
                    del self.tracks[tid]
                else:
                    report[tid] = {"status": "occluded",
                                   "missedFrames": tr.missed_frames}

        # --- new tracks ----------------------------------------------------
        for di in unmatched_dets:
            det = detections[di]
            tid = f"trk-{self._next_id:04d}"
            self._next_id += 1
            tapped = is_tapped_orientation(
                det.angle_deg, self.tap_threshold_deg, self.tap_tolerance_deg)
            self.tracks[tid] = Track(
                track_id=tid, bbox=det.bbox, angle_deg=det.angle_deg,
                confidence=det.confidence, last_seen=now(), tapped=tapped)
            report[tid] = {"status": "new", "tapped": tapped,
                           "angle": det.angle_deg}
        return report

    # ------------------------------------------------------------------
    def _update_track(self, tr: Track, det: Detection) -> dict[str, Any]:
        tr.bbox = det.bbox
        tr.missed_frames = 0
        tr.age_frames += 1
        tr.last_seen = now()
        tr.confidence = det.confidence
        # circular-ish smoothing via shortest-path blend
        delta = ((det.angle_deg - tr.angle_deg + 180.0) % 360.0) - 180.0
        tr.angle_deg = (tr.angle_deg + self.angle_smooth_alpha * delta) % 360.0
        tr.history.append((tr.last_seen, tr.angle_deg))
        if len(tr.history) > 120:
            tr.history = tr.history[-60:]

        target_tapped = is_tapped_orientation(
            tr.angle_deg, self.tap_threshold_deg, self.tap_tolerance_deg)
        if target_tapped == tr.tapped:
            tr._pending_tapped = None
            tr._pending_frames = 0
            return {"status": "ok", "tapped": tr.tapped, "angle": tr.angle_deg}

        if tr._pending_tapped == target_tapped:
            tr._pending_frames += 1
        else:
            tr._pending_tapped = target_tapped
            tr._pending_frames = 1

        if tr._pending_frames >= self.settle_frames:
            tr.tapped = target_tapped
            tr._pending_tapped = None
            tr._pending_frames = 0
            return {"status": "ok", "tapped": tr.tapped,
                    "angle": tr.angle_deg,
                    "tapEvents": ["card_tapped" if tr.tapped else "card_untapped"]}
        return {"status": "settling", "tapped": tr.tapped,
                "angle": tr.angle_deg}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
