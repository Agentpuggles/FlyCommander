"""FlyCommander — the neural vision pipeline (physical table integration).

This module is the bridge between the computer-vision stack
(``vision/detector`` → ``vision/rectify`` → ``vision/embeddings`` →
``vision/matcher``) and the physical table state machine
(``physical/tracker`` → ``physical/observer`` → ``physical/engine``).

It owns exactly three responsibilities:

1. **Recognition** — turn a camera frame into ``RecognizedCard`` objects:
   quad, rectified image, identity candidates, confidence, orientation, and
   the diagnostics the UI shows. Never raises for a "bad" frame; a frame with
   no cards is a normal result.

2. **Identity stabilisation** — a single frame's top-1 can flip between two
   printings of the same card, or between two visually similar cards, while
   the player is placing a card. Recognitions are associated across frames
   (quad IoU + centroid distance) and each track keeps a confidence-weighted
   vote over its history. The *stable* identity — not the per-frame one — is
   what the state layer is told, which removes identity flicker from the
   game state.

3. **Index lifecycle** — build/load the recognition index from
   ``cards.database.CardDatabase``, hot-add unknown cards, and expose status
   for the UI. The index is what makes "no manual registration" possible:
   it is built once from Scryfall, and every new card is a row, not a task.

Design rule: **no message ever tells the player to flatten a card.** The only
player-facing outcomes are "recognised as X", "not sure between X and Y",
"card not in the library" and "move closer / more light" when the image truly
cannot support recognition — and even then the pipeline keeps trying with the
full frame.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from vision import rectify as R
from vision.detector import CardDetection, CardDetector
from vision.embeddings import Embedder, load_embedder
from vision.matcher import CardIndex, CardMatcher, MatchResult

try:
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class RecognitionConfig:
    """Tunables for the live pipeline (sane defaults; overridable)."""

    index_dir: str = "data/cards"
    detector_model: str = ""            # optional YOLO/RT-DETR ONNX
    embedder: str = "auto"
    max_analysis_fps: float = 6.0       # detection+recognition cadence
    max_cards_per_frame: int = 12
    auto_accept_confidence: float = 0.66    # register without asking
    suggest_confidence: float = 0.35        # offer as a suggestion
    # Never offer a candidate weaker than this. Below ~20% a "match" is what
    # random feature vectors look like, and offering it as a pick is worse
    # than saying "I don't know" (a 48-card demo library scores ~5% on a real
    # card — that is not a suggestion, it is noise with a Select button).
    min_offer_confidence: float = 0.20
    # When the visual library cannot answer, read the card's own text and ask
    # Scryfall by name (works with no image library at all).
    ocr_rescue: bool = True
    ocr_auto_accept_confidence: float = 0.60
    stable_votes: int = 2                   # frames before an identity is stable
    vote_window: int = 12
    track_iou: float = 0.25
    track_max_distance: float = 140.0
    unknown_card_capture: bool = True       # keep a crop of unknown cards

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class RecognizedCard:
    """One card seen in one frame, with its identity evidence."""

    track_id: str
    rectified: R.RectifiedCard
    match: MatchResult
    detection: CardDetection | None = None
    first_seen: float = field(default_factory=time.time)
    frames_seen: int = 1
    votes: deque = field(default_factory=lambda: deque(maxlen=12))
    stable_key: str = ""
    stable_name: str = ""
    stable_confidence: float = 0.0
    stable_votes: int = 0
    last_seen: float = field(default_factory=time.time)

    # -- geometry shortcuts -------------------------------------------------
    @property
    def quad(self) -> np.ndarray:
        return self.rectified.quad

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        q = self.rectified.quad
        return (float(q[:, 0].min()), float(q[:, 1].min()),
                float(q[:, 0].max() - q[:, 0].min()),
                float(q[:, 1].max() - q[:, 1].min()))

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.rectified.quad.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def orientation_deg(self) -> float:
        """Long-axis angle of the card on the table (0 upright, ~90 tapped)."""
        return float(self.rectified.tilt_deg)

    @property
    def top(self):
        return self.match.best

    @property
    def confidence(self) -> float:
        return float(self.match.best.confidence) if self.match.best else 0.0

    def to_dict(self, include_image: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trackId": self.track_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "quad": [[round(float(x), 1), round(float(y), 1)]
                     for x, y in self.rectified.quad],
            "orientationDeg": round(self.orientation_deg, 1),
            "framesSeen": self.frames_seen,
            "stable": {"key": self.stable_key, "name": self.stable_name,
                       "confidence": round(self.stable_confidence, 3),
                       "votes": self.stable_votes},
            "match": self.match.to_dict(),
            "geometry": self.rectified.to_dict(),
        }
        if include_image and CV_AVAILABLE:
            from physical.card_scan import encode_jpeg_b64

            out["cardImage"] = encode_jpeg_b64(self.rectified.image, max_w=280)
        return out


@dataclass
class SceneAnalysis:
    """Everything the pipeline saw in one frame."""

    cards: list[RecognizedCard] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    index_size: int = 0
    pipeline: str = "neural"

    @property
    def identified(self) -> list[RecognizedCard]:
        return [c for c in self.cards if c.stable_key]

    def best(self) -> RecognizedCard | None:
        """The card the table is most likely asking about (largest, newest)."""
        if not self.cards:
            return None
        return max(self.cards, key=lambda c: (c.rectified.area_frac,
                                              c.confidence))

    def to_dict(self, include_images: bool = False) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "indexSize": self.index_size,
            "count": len(self.cards),
            "cards": [c.to_dict(include_image=include_images) for c in self.cards],
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# recognition engine
# ---------------------------------------------------------------------------
def _quad_contains_center(container: np.ndarray, inner: np.ndarray,
                          margin: float = 0.85) -> bool:
    """True when `inner`'s centroid sits well inside `container`."""
    centre = np.asarray(container, np.float32).mean(axis=0)
    hull = centre + (np.asarray(container, np.float32) - centre) * margin
    hull = R.order_corners(hull)
    c = np.asarray(inner, np.float32).mean(axis=0)

    def _side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    signs = [_side(hull[i], hull[(i + 1) % 4], c) for i in range(4)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


class CardRecognizer:
    """Detector + rectifier + matcher + identity stabilisation."""

    def __init__(self, config: RecognitionConfig | None = None,
                 embedder: Embedder | None = None,
                 detector: CardDetector | None = None,
                 database: Any | None = None) -> None:
        self.config = config or RecognitionConfig()
        self.embedder = embedder or load_embedder(self.config.embedder)
        self.detector = detector or CardDetector(
            model_path=self.config.detector_model or None)
        self.database = database
        self.matcher: CardMatcher | None = None
        self.tracks: dict[str, RecognizedCard] = {}
        self._track_seq = 0
        self._lock = threading.RLock()
        self.last_index_error = ""
        self.last_scene: SceneAnalysis | None = None
        self.recognition_log: deque = deque(maxlen=50)
        self.auto_registered: dict[str, str] = {}     # stable_key -> tracking id
        from vision.reference_scanner import ReferenceScanner
        self.reference_scanner = ReferenceScanner(Path(self.config.index_dir) / "references")
        self.load_index()

    # ------------------------------------------------------------------
    # index lifecycle
    # ------------------------------------------------------------------
    def load_index(self) -> bool:
        """Load the persisted index (returns False when there is none yet)."""
        try:
            index = CardIndex.load(self.config.index_dir)
        except Exception as exc:
            self.last_index_error = f"{type(exc).__name__}: {exc}"
            return False
        if index.size == 0:
            return False
        if index.embedder_name != self.embedder.name and index.dim != self.embedder.dim:
            # a different embedder built the index — rebuild instead of
            # silently comparing apples to oranges
            self.last_index_error = (f"index built with '{index.embedder_name}' "
                                     f"(dim {index.dim}) but embedder is "
                                     f"'{self.embedder.name}' (dim {self.embedder.dim})")
            return False
        self.matcher = CardMatcher(index, embedder=self.embedder)
        return True

    def build_index(self, limit: int | None = None,
                    embedder: Embedder | None = None,
                    progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """(Re)build the index from whatever the card database holds."""
        if self.database is None:
            from cards.database import CardDatabase

            self.database = CardDatabase(self.config.index_dir,
                                         allow_network=False)
        index = self.database.build_index(embedder or self.embedder,
                                          limit=limit, progress=progress)
        self.matcher = CardMatcher(index, embedder=self.embedder)
        return {"cards": index.size, "dim": index.dim,
                "embedder": index.embedder_name}

    @property
    def ready(self) -> bool:
        return self.reference_scanner.ready or (self.matcher is not None and self.matcher.ready)

    def status(self) -> dict[str, Any]:
        index = self.matcher.index if self.matcher else None
        return {
            "pipeline": "reference-art" if self.reference_scanner.ready else "neural",
            "referenceScanner": self.reference_scanner.status(),
            "ready": self.ready,
            "indexSize": index.size if index else 0,
            "indexDir": self.config.index_dir,
            "detector": self.detector.backend,
            "embedder": self.embedder.describe(),
            "embedderFallback": getattr(self.embedder, "fallback_reason", ""),
            "indexError": self.last_index_error,
            "cardnessModel": bool(getattr(self.detector, "cardness_scorer", None)
                                  and self.detector.cardness_scorer.available),
            "recognitionLogSize": len(self.recognition_log),
        }

    # ------------------------------------------------------------------
    # tracking / stabilisation
    # ------------------------------------------------------------------
    def _new_track_id(self) -> str:
        self._track_seq += 1
        return f"card{self._track_seq}"

    def _associate(self, rect: R.RectifiedCard
                   ) -> tuple[RecognizedCard | None, float, bool]:
        """Find the existing track this rectification belongs to.

        Returns (track, iou). Matching uses quad IoU, containment (inner
        features of a card — art box, text box — are the same card), and a
        centroid-distance fallback for a card that moved between frames.
        """
        best: RecognizedCard | None = None
        best_iou = 0.0
        contained = False
        centroid = rect.quad.mean(axis=0)
        for track in self.tracks.values():
            iou = R.quad_iou(track.rectified.quad, rect.quad)
            if iou >= self.config.track_iou and iou > best_iou:
                best, best_iou, contained = track, iou, False
                continue
            if iou < 0.05:
                if _quad_contains_center(track.rectified.quad, rect.quad):
                    # an inner feature of an existing card (art box, text box),
                    # not a second card
                    if best_iou < 0.2:
                        best, best_iou, contained = track, max(best_iou, 0.2), True
                    continue
                t_c = track.rectified.quad.mean(axis=0)
                if float(np.linalg.norm(t_c - centroid)) < self.config.track_max_distance:
                    if best is None:
                        best, contained = track, False
        return best, best_iou, contained

    def _update_track(self, track: RecognizedCard, rect: R.RectifiedCard,
                      match: MatchResult) -> None:
        track.rectified = rect
        track.match = match
        track.frames_seen += 1
        track.last_seen = time.time()
        if match.unknown:
            track.votes.clear()
        if match.best is not None and not match.unknown:
            track.votes.append((match.best.key, match.best.name,
                                match.best.confidence,
                                match.best.confidence >= 0.5))
            recent = list(track.votes)[-3:]
            # a card that now reads consistently as a *different* card resets
            # the vote window (the player replaced the card on the table)
            if (track.stable_name and len(recent) == 3
                    and all(v[1] != track.stable_name for v in recent)
                    and all(v[2] >= 0.4 for v in recent)):
                track.votes.clear()
                track.votes.extend(recent)
        self._recompute_stable(track)

    def _recompute_stable(self, track: RecognizedCard) -> None:
        """Confidence-weighted majority vote over the track's vote window.

        An identity is only published once it has been seen at least
        `stable_votes` times: a single frame may not name a card on the table,
        which is what removes identity flicker (and stops a stray detection
        from inventing a card) in the game state.
        """
        if not track.votes or len(track.votes) < max(1, self.config.stable_votes):
            track.stable_key = ""
            track.stable_name = ""
            track.stable_confidence = 0.0
            track.stable_votes = len(track.votes)
            return
        weights: dict[str, float] = {}
        names: dict[str, str] = {}
        counts: Counter = Counter()
        for key, name, conf, _shifted in track.votes:
            weights[key] = weights.get(key, 0.0) + max(0.05, conf)
            names[key] = name
            counts[key] += 1
        key, weight = max(weights.items(), key=lambda kv: kv[1])
        total = sum(weights.values()) or 1.0
        track.stable_key = key
        track.stable_name = names.get(key, "")
        track.stable_confidence = float(min(1.0, weight / total))
        track.stable_votes = int(counts[key])

    def _retire_stale(self, now: float, timeout: float = 4.0) -> None:
        for tid in [t for t, card in self.tracks.items()
                    if now - card.last_seen > timeout]:
            self.tracks.pop(tid, None)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def recognize(self, frame: np.ndarray, *, match_cards: bool = True,
                  max_cards: int | None = None,
                  stabilise: bool = True) -> SceneAnalysis:
        """Full pipeline for one frame.

        ``match_cards=False`` runs detection + rectification only (useful for
        a fast presence/tap pass when the index is empty or the CPU is busy).
        """
        t_start = time.time()
        scene = SceneAnalysis(frame_shape=frame.shape[:2] if frame is not None else (0, 0),
                              index_size=self.matcher.index.size if self.matcher else 0)
        if not CV_AVAILABLE or frame is None or frame.size == 0:
            scene.notes.append("no frame")
            return scene

        max_cards = max_cards or self.config.max_cards_per_frame
        if match_cards and self.reference_scanner.ready:
            return self._recognize_references(frame, max_cards, stabilise)
        detections = self.detector.detect(frame)
        if len(detections) > max_cards:
            detections = sorted(detections, key=lambda d: -d.confidence)[:max_cards]
        t_detect = time.time()

        rects: list[R.RectifiedCard] = []
        for det in detections:
            try:
                rect = R.rectify(frame, det.quad, refine=False)
            except Exception:  # pragma: no cover - defensive
                continue
            rect.edge_support = det.edge_support
            rects.append(rect)
        t_rect = time.time()

        with self._lock:
            # largest first so a card is tracked before its inner features are
            # considered (they then associate to it instead of spawning tracks)
            order = sorted(range(len(rects)),
                           key=lambda i: -R.quad_area(rects[i].quad))
            used_this_frame: set[str] = set()
            accepted_quads: list[np.ndarray] = []
            for i in order:
                rect, det = rects[i], detections[i]
                # inner features of an already-accepted card in this frame
                # (art box, text box, P/T box) are not separate cards
                if any(_quad_contains_center(q, rect.quad)
                       for q in accepted_quads):
                    continue
                track = None
                iou = 0.0
                contained = False
                if stabilise:
                    track, iou, contained = self._associate(rect)
                if track is not None and track.track_id in used_this_frame:
                    # the same card seen twice in one frame (an inner contour
                    # or a near-duplicate proposal): merge, never double-count
                    if iou > 0.35 or contained:
                        continue
                    track = None
                fresh = track is None
                if fresh:
                    track = RecognizedCard(track_id=self._new_track_id(),
                                           rectified=rect,
                                           match=MatchResult(),
                                           detection=det, frames_seen=0)
                    self.tracks[track.track_id] = track
                match = (self.matcher.match_rectified(rect)
                         if (match_cards and self.matcher and self.matcher.ready) else MatchResult(
                             notes=["no index — detection only"]))
                track.detection = det
                self._update_track(track, rect, match)
                used_this_frame.add(track.track_id)
                accepted_quads.append(rect.quad)
                scene.cards.append(track)
            if stabilise:
                self._retire_stale(time.time())

        t_match = time.time()
        scene.timings = {
            "detectMs": round((t_detect - t_start) * 1000, 2),
            "rectifyMs": round((t_rect - t_detect) * 1000, 2),
            "matchMs": round((t_match - t_rect) * 1000, 2),
            "totalMs": round((time.time() - t_start) * 1000, 2),
        }
        if not self.ready:
            scene.notes.append("recognition index is empty — run the index build")
        for card in scene.cards:
            if card.match.unknown and card.top is not None:
                scene.notes.append(f"{card.track_id}: not in library "
                                   f"(best guess {card.top.name} "
                                   f"{card.top.confidence:.0%})")
        self.last_scene = scene
        if scene.cards:
            self.recognition_log.append({
                "t": time.time(),
                "cards": [{"track": c.track_id, "name": c.stable_name,
                           "confidence": round(c.stable_confidence, 3),
                           "top": (c.top.to_dict() if c.top else None),
                           "bbox": [round(v, 1) for v in c.bbox]}
                          for c in scene.cards],
            })
        return scene

    def _recognize_references(self, frame, max_cards, stabilise):
        """Real reference matching owns this mode; never fall through to demo guesses."""
        t0 = time.time()
        hits = self.reference_scanner.recognize(frame, max_cards)
        scene = SceneAnalysis(frame_shape=frame.shape[:2], pipeline="reference-art",
                              index_size=len(self.reference_scanner.refs))
        with self._lock:
            used = set()
            for hit in hits:
                rect = R.rectify(frame, hit.quad, refine=False)
                track = self._associate(rect)[0] if stabilise else None
                if track is None or track.track_id in used:
                    track = RecognizedCard(self._new_track_id(), rect, hit.match, frames_seen=0)
                    self.tracks[track.track_id] = track
                self._update_track(track, rect, hit.match)
                used.add(track.track_id)
                scene.cards.append(track)
            self._retire_stale(time.time())
        scene.timings = {"totalMs": (time.time() - t0) * 1000}
        scene.notes = ["Local artwork matching; confirm before adding to battlefield."]
        if not hits:
            scene.notes.append("No artwork verified. Import this printing, move closer, or try Scan for OCR.")
        self.last_scene = scene
        return scene

    # ------------------------------------------------------------------
    def recognize_card_image(self, card: np.ndarray, top_k: int = 5) -> MatchResult:
        """Identify a single, already-rectified card image (manual/UI path)."""
        if not self.ready:
            return MatchResult(notes=["recognition index is empty"])
        if self.reference_scanner.ready:
            hits = self.reference_scanner.recognize(card, 1)
            return hits[0].match if hits else MatchResult(notes=["No reference artwork verified"])
        assert self.matcher is not None
        return self.matcher.match(card, k=top_k)

    def register_card(self, card: np.ndarray, set_code: str, collector_number: str,
                      name: str, image_path: str = "") -> dict[str, Any]:
        """Hot-add a card to the live index (unknown-card flow)."""
        if self.matcher is None:
            index = CardIndex.empty(self.embedder)
            self.matcher = CardMatcher(index, embedder=self.embedder)
        _, match = self.matcher.add_card(card, set_code, collector_number, name,
                                         image_path=image_path)
        return {"indexSize": self.matcher.index.size,
                "selfMatch": match.to_dict()}

    def save_index(self) -> Path | None:
        if self.matcher is None:
            return None
        return self.matcher.index.save(self.config.index_dir)

    # ------------------------------------------------------------------
    def detection_only(self, frame: np.ndarray) -> list[CardDetection]:
        return self.detector.detect(frame)

    def close(self) -> None:
        self.tracks.clear()
