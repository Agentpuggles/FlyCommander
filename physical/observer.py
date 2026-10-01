"""FlyCommander physical-table mode — physical observer.

Sits between the raw CV tracker and the state engine. Converts tracker
reports into *candidate* events (tap/untap, appeared/occluded/retired,
moved) and feeds them to the engine, which decides what is authoritative.
Tracking IDs map 1:1 to state tracking IDs once a card is registered.
"""
from __future__ import annotations

from typing import Any

from physical.engine import Engine
from physical.events import Event
from physical.tracker import CardTracker, Detection


class PhysicalObserver:
    def __init__(self, engine: Engine, tracker: CardTracker | None = None,
                 track_to_state: dict[str, str] | None = None) -> None:
        self.engine = engine
        self.tracker = tracker or CardTracker()
        self.track_to_state: dict[str, str] = dict(track_to_state or {})

    # ------------------------------------------------------------------
    def bind(self, track_id: str, state_tracking_id: str) -> None:
        """Attach a registered card identity to a vision track."""
        self.track_to_state[track_id] = state_tracking_id

    def unbind(self, track_id: str) -> None:
        self.track_to_state.pop(track_id, None)

    # ------------------------------------------------------------------
    def process_frame(self, detections: list[Detection]) -> dict[str, Any]:
        """One frame: tracker update → candidate events → engine."""
        report = self.tracker.update(detections)
        applied: list[dict] = []
        for tid, info in report.items():
            state_id = self.track_to_state.get(tid)
            status = info.get("status")

            if status == "new":
                applied.append({"track": tid, "status": "new_track",
                                "bound": state_id})
                continue  # identity assignment is a player action

            if state_id is None:
                continue  # unregistered physical card: tracked but unbound

            if "tapEvents" in info:
                etype = info["tapEvents"][0]
                res = self.engine.apply_vision(Event(
                    type=etype, origin="vision",
                    payload={"trackingId": state_id,
                             "orientationDegrees": round(info.get("angle", 0.0), 1),
                             "orientationConfidence": 0.8},
                    confidence=0.8))
                applied.append({"track": tid, "event": etype,
                                "result": res})

            if status == "occluded":
                res = self.engine.apply_vision(Event(
                    type="disappeared", origin="vision",
                    payload={"trackingId": state_id,
                             "frames": info.get("missedFrames", 1)},
                    confidence=0.9))
                applied.append({"track": tid, "event": "disappeared"})
            elif status == "retired":
                applied.append({"track": tid, "status": "retired",
                                "bound": state_id})
                self.unbind(tid)
            elif status in ("ok", "settling"):
                res = self.engine.apply_vision(Event(
                    type="appeared", origin="vision",
                    payload={"trackingId": state_id},
                    confidence=info.get("confidence", 0.9)))
                applied.append({"track": tid, "status": status})
                # Orientation reconciliation: if a settled track's tapped
                # state disagrees with the authoritative object (e.g. the
                # camera joined mid-game), vision corrects state.
                if status == "ok" and "tapEvents" not in info:
                    obj = self.engine.state.get(state_id)
                    if obj is not None and obj.tapped != bool(info.get("tapped")):
                        etype = "card_tapped" if info.get("tapped") \
                            else "card_untapped"
                        self.engine.apply_vision(Event(
                            type=etype, origin="vision",
                            payload={"trackingId": state_id,
                                     "orientationDegrees": round(
                                         info.get("angle", 0.0), 1),
                                     "orientationConfidence": 0.8},
                            confidence=0.8))
                        applied.append({"track": tid,
                                        "event": etype + " (reconciled)"})
        return {"tracks": report, "applied": applied}

    # ------------------------------------------------------------------
    def tracker_tracks_snapshot(self) -> list[dict[str, Any]]:
        """Live track geometry for the debug/tuning view."""
        out = []
        for tid, tr in self.tracker.tracks.items():
            out.append({
                "trackId": tid,
                "stateId": self.track_to_state.get(tid),
                "centroid": [round(tr.centroid[0]), round(tr.centroid[1])],
                "angle": round(tr.angle_deg, 1),
                "tapped": tr.tapped,
                "missedFrames": tr.missed_frames,
                "ageFrames": tr.age_frames,
            })
        return out

    def registered_summary(self) -> list[dict[str, Any]]:
        out = []
        for tid, state_id in self.track_to_state.items():
            obj = self.engine.state.get(state_id)
            if obj is not None:
                d = obj.to_dict()
                d["visionTrack"] = tid
                out.append(d)
        return out
