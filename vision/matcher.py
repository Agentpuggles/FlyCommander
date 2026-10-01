"""FlyCommander — card matching: embeddings + artwork + layout + text signals.

Recognition answers one question: *which printing is this card?* OCR used to
answer it; it now only ever votes. The matcher fuses six independent signals,
any subset of which can be missing without breaking the pipeline:

===============  ==========================================================
signal           what it sees
===============  ==========================================================
embedding        learned/handcrafted descriptor cosine (principal signal)
artwork          normalised cross-correlation of the illustration zone
title            the name text band as image (font-tolerant "text" signal)
collector        collector number / set-line region correlation
layout           whole-card high-pass structure (frame + text box geometry)
colour           frame/border colour signature (colour identity prior)
===============  ==========================================================

The pipeline is a retrieval + rerank: a single matmul against the whole index
(30k printings ≈ 5 ms) produces the top-K candidates, then the cheap
image-space correlations are computed only for those K. This is what makes
"recognize any card without registering it" practical: the index is built from
Scryfall once, and new sets are just new rows.

Robustness rules encoded here:

* **180-degree ambiguity** — a card's geometry is identical under a half turn,
  so both readings are embedded and the better one wins.
* **Glare / occlusion** — masked pixels were inpainted upstream; the coverage
  fraction scales confidence down instead of rejecting the card.
* **Ambiguous printings** — different printings of the same card are
  visually near-identical and legitimately indistinguishable; they are
  reported as a ranked set with `ambiguous_printing=True` rather than
  pretending certainty.
* **Unknown cards** — a card that is not in the index returns `unknown=True`
  with the best score, so the UI can offer "add this card" instead of lying.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from cards.database import IndexEntry
from vision import rectify as R
from vision.embeddings import (CARD_ROTATIONS, Embedder, dense_features, flip180,
                               load_embedder, normalize_capture, rotate_card)

try:
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

# ---- recognition thresholds (calibrated defaults; overridable from index) --
UNKNOWN_THRESHOLD = 0.42        # below → "card not in the index"
GOOD_CONFIDENCE = 0.62          # above → auto-register without asking
AMBIGUOUS_MARGIN = 0.06         # top-2 closer than this → ambiguous printings

# ---- fusion weights (sum ≈ 1 before the calibration transform) -------------
W_EMBED = 0.52
W_ART = 0.20
W_LAYOUT = 0.10
W_TITLE = 0.10
W_COLLECTOR = 0.05
W_COLOUR = 0.03

# signature grid sizes (width, height) — small, fast, and enough to correlate
SIG_SHAPES = {
    "art": (10, 14),
    "title": (24, 5),
    "collector": (20, 5),
    "layout": (17, 12),
}
SIGNATURE_DIM = sum(w * h for w, h in SIG_SHAPES.values())     # 564


# ---------------------------------------------------------------------------
# image-space signatures
# ---------------------------------------------------------------------------
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors (0 when either is constant)."""
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _zone_sig(gray: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """High-pass, size-normalised crop → correlation-friendly vector."""
    small = cv2.resize(gray, shape, interpolation=cv2.INTER_AREA).astype(np.float32)
    blur = cv2.GaussianBlur(small, (0, 0), 1.1)
    hp = small - blur
    hp = hp / (float(np.std(hp)) + 1e-4)
    return hp.reshape(-1).astype(np.float32)


def zone_signature(card: np.ndarray) -> np.ndarray:
    """564-d multi-zone correlation signature of a canonical card image."""
    if card.ndim == 3:
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    else:
        gray = card
    sigs = [_zone_sig(R.crop_zone(gray, zone), shape)
            for zone, shape in (("art", SIG_SHAPES["art"]),
                                ("title", SIG_SHAPES["title"]),
                                ("collector", SIG_SHAPES["collector"]),
                                ("border", SIG_SHAPES["layout"]))]
    return np.concatenate(sigs).astype(np.float32)


def zone_signature_batch(cards: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([zone_signature(c) for c in cards]) if cards else np.zeros(
        (0, SIGNATURE_DIM), np.float32)


SIG_SLICES = {
    "art": slice(0, SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1]),
    "title": slice(SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1],
                   SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1]
                   + SIG_SHAPES["title"][0] * SIG_SHAPES["title"][1]),
    "collector": slice(
        SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1]
        + SIG_SHAPES["title"][0] * SIG_SHAPES["title"][1],
        SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1]
        + SIG_SHAPES["title"][0] * SIG_SHAPES["title"][1]
        + SIG_SHAPES["collector"][0] * SIG_SHAPES["collector"][1]),
    "layout": slice(
        SIG_SHAPES["art"][0] * SIG_SHAPES["art"][1]
        + SIG_SHAPES["title"][0] * SIG_SHAPES["title"][1]
        + SIG_SHAPES["collector"][0] * SIG_SHAPES["collector"][1],
        SIGNATURE_DIM),
}


def colour_signature(card: np.ndarray) -> np.ndarray:
    """Mean BGR of the card frame border (colour identity prior)."""
    h, w = card.shape[:2]
    b = max(2, int(min(h, w) * 0.022))
    strips = [card[b:h - b, 0:b], card[b:h - b, w - b:w],
              card[0:b, 0:w], card[h - b:h, 0:w]]
    return np.concatenate([s.reshape(-1, 3).mean(axis=0) for s in strips
                           if s.size]).astype(np.float32)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    """Maps fused scores to a probability-like confidence.

    Defaults are sane out-of-the-box values; `scripts/train_embedder.py`
    fits them on synthetic validation data (Platt scaling over the fused
    score + margin) and stores them in the index metadata.
    """

    slope: float = 9.0
    bias: float = -3.4
    margin_weight: float = 4.0
    coverage_weight: float = 0.9
    unknown_threshold: float = UNKNOWN_THRESHOLD
    good_confidence: float = GOOD_CONFIDENCE

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> "Calibration":
        meta = meta or {}
        cal = meta.get("calibration") or {}
        return cls(
            slope=float(cal.get("slope", 9.0)),
            bias=float(cal.get("bias", -3.4)),
            margin_weight=float(cal.get("marginWeight", 4.0)),
            coverage_weight=float(cal.get("coverageWeight", 0.9)),
            unknown_threshold=float(cal.get("unknownThreshold", UNKNOWN_THRESHOLD)),
            good_confidence=float(cal.get("goodConfidence", GOOD_CONFIDENCE)))

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "bias": self.bias,
                "marginWeight": self.margin_weight,
                "coverageWeight": self.coverage_weight,
                "unknownThreshold": self.unknown_threshold,
                "goodConfidence": self.good_confidence}

    def confidence(self, fused: float, margin: float, coverage: float) -> float:
        z = (self.slope * (fused - 0.5) + self.margin_weight * min(margin, 0.35)
             + self.coverage_weight * (coverage - 0.85))
        z = max(-30.0, min(30.0, z))
        return float(1.0 / (1.0 + math.exp(-z)))


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
@dataclass
class MatchCandidate:
    """One possible identity for a scanned card, with its evidence."""

    set_code: str
    collector_number: str
    name: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    image_path: str = ""
    oracle_id: str = ""
    type_line: str = ""
    color_identity: list[str] = field(default_factory=list)
    cmc: float = 0.0
    flipped: bool = False
    source: str = "index"

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.collector_number}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "set": self.set_code,
            "collectorNumber": self.collector_number,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(float(v), 3) for k, v in self.scores.items()},
            "imagePath": self.image_path, "oracleId": self.oracle_id,
            "cardType": self.type_line,
            "colorIdentity": list(self.color_identity), "cmc": self.cmc,
            "flipped": self.flipped, "source": self.source,
        }


@dataclass
class MatchResult:
    """Outcome of matching one rectified card."""

    candidates: list[MatchCandidate] = field(default_factory=list)
    unknown: bool = True
    top_score: float = 0.0
    margin: float = 0.0
    flipped: bool = False
    rotation: int = 0                # 0/90/180/270: reading that matched
    coverage: float = 1.0
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confident(self) -> bool:
        return bool(self.candidates) and not self.unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "unknown": self.unknown,
            "topScore": round(self.top_score, 3),
            "margin": round(self.margin, 3),
            "flipped": self.flipped,
            "rotation": self.rotation,
            "coverage": round(self.coverage, 3),
            "notes": list(self.notes),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }


class CardIndex:
    """Vector index of known printings (embeddings + zone signatures + meta)."""

    def __init__(self, vectors: np.ndarray, entries: list[IndexEntry],
                 embedder_name: str = "dense", dim: int = 0,
                 signatures: np.ndarray | None = None,
                 colours: np.ndarray | None = None,
                 dense_vectors: np.ndarray | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.vectors = np.asarray(vectors, np.float32)
        self.entries = entries
        self.embedder_name = embedder_name
        self.dim = int(dim or (self.vectors.shape[1] if self.vectors.size else 0))
        self.signatures = signatures if signatures is not None else np.zeros(
            (len(entries), SIGNATURE_DIM), np.float32)
        self.colours = colours if colours is not None else np.zeros(
            (len(entries), 3), np.float32)
        self.dense_vectors = dense_vectors
        self.meta = meta or {}
        self._sig_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.entries)

    @classmethod
    def empty(cls, embedder: Embedder | None = None) -> "CardIndex":
        dim = embedder.dim if embedder else 0
        return cls(np.zeros((0, dim), np.float32), [], 
                   embedder_name=embedder.name if embedder else "none", dim=dim)

    # ------------------------------------------------------------------
    def search(self, query_vectors: np.ndarray, k: int = 25
               ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine top-K over the whole index (one matmul per query)."""
        if self.size == 0 or query_vectors.size == 0:
            return np.zeros((len(query_vectors), 0), np.float32), np.zeros(
                (len(query_vectors), 0), np.int64)
        sims = query_vectors.astype(np.float32) @ self.vectors.T
        k = min(k, self.size)
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, order]
        return sims[rows, idx], idx

    def reference_signature(self, i: int) -> np.ndarray:
        return self.signatures[i].astype(np.float32)

    # ------------------------------------------------------------------
    def save(self, directory: str | Path = "data/cards") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "index.npz"
        payload = {
            "vectors": self.vectors.astype(np.float16),
            "signatures": self.signatures.astype(np.float16),
            "colours": self.colours.astype(np.float32),
        }
        if self.dense_vectors is not None:
            payload["dense_vectors"] = self.dense_vectors.astype(np.float16)
        np.savez_compressed(path, **payload)
        (directory / "index.json").write_text(json.dumps({
            "embedder": self.embedder_name, "dim": self.dim,
            "count": self.size, "meta": self.meta,
            "entries": [e.to_dict() for e in self.entries]}, indent=1))
        return path

    @classmethod
    def load(cls, directory: str | Path = "data/cards",
             mmap: bool = False) -> "CardIndex":
        directory = Path(directory)
        meta_path = directory / "index.json"
        npz_path = directory / "index.npz"
        if not meta_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"no index in {directory}")
        meta = json.loads(meta_path.read_text())
        entries = [IndexEntry.from_dict(d) for d in meta.get("entries", [])]
        data = np.load(npz_path, mmap_mode="r" if mmap else None)
        vectors = np.asarray(data["vectors"], np.float32)
        signatures = np.asarray(data["signatures"], np.float32) \
            if "signatures" in data else np.zeros((len(entries), SIGNATURE_DIM),
                                                  np.float32)
        colours = np.asarray(data["colours"], np.float32) \
            if "colours" in data else np.zeros((len(entries), 3), np.float32)
        dense = np.asarray(data["dense_vectors"], np.float32) \
            if "dense_vectors" in data else None
        return cls(vectors=vectors, entries=entries,
                   embedder_name=meta.get("embedder", "dense"),
                   dim=int(meta.get("dim", vectors.shape[1] if vectors.size else 0)),
                   signatures=signatures, colours=colours, dense_vectors=dense,
                   meta=meta.get("meta", {}))

    # ------------------------------------------------------------------
    @classmethod
    def from_specs(cls, specs: Sequence[Any], embedder: Embedder | None = None,
                   meta: dict[str, Any] | None = None) -> "CardIndex":
        """Build an index directly from synthetic `CardSpec`s (tests/demos)."""
        from vision.synthetic import render_card

        embedder = embedder or load_embedder("dense")
        cards = [render_card(spec) for spec in specs]
        vectors = embedder.embed_batch(cards)
        sigs = zone_signature_batch(cards)
        colours = np.stack([colour_signature(c) for c in cards])
        entries = [IndexEntry(set_code=s.set_code.lower(),
                              collector_number=str(s.collector_number),
                              name=s.name, oracle_id=f"oracle-{s.key}",
                              type_line=s.kind.title(),
                              color_identity=list(s.colors), source="synthetic")
                   for s in specs]
        return cls(vectors=vectors, entries=entries, embedder_name=embedder.name,
                   dim=embedder.dim, signatures=sigs, colours=colours,
                   meta=meta or {"source": "synthetic", "count": len(specs)})

    # ------------------------------------------------------------------
    @classmethod
    def from_image_dir(cls, folder: str | Path, embedder: Embedder | None = None,
                       meta: dict[str, Any] | None = None) -> "CardIndex":
        """Index every image in a folder: `<set>_<number>.jpg` or `<name>.png`."""
        folder = Path(folder).expanduser()
        embedder = embedder or load_embedder()
        entries: list[IndexEntry] = []
        cards: list[np.ndarray] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in (
                    ".jpg", ".jpeg", ".png", ".webp"):
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            stem = path.stem
            if "_" in stem:
                set_code, _, number = stem.rpartition("_")
            else:
                set_code, number = "lcl", str(len(entries) + 1)
            entries.append(IndexEntry(set_code=set_code.lower(),
                                      collector_number=number,
                                      name=stem.replace("_", " "),
                                      image_path=str(path), source="local"))
            cards.append(img)
        if not cards:
            return cls.empty(embedder)
        vectors = embedder.embed_batch(cards)
        return cls(vectors=vectors, entries=entries, embedder_name=embedder.name,
                   dim=embedder.dim, signatures=zone_signature_batch(cards),
                   colours=np.stack([colour_signature(c) for c in cards]),
                   meta=meta or {"source": "folder", "folder": str(folder)})


# ---------------------------------------------------------------------------
# matcher
# ---------------------------------------------------------------------------
class CardMatcher:
    """Retrieve + rerank card identity from a rectified capture."""

    def __init__(self, index: CardIndex, embedder: Embedder | None = None,
                 candidates: int = 25, calibration: Calibration | None = None,
                 use_dense_rerank: bool = True) -> None:
        self.index = index
        self.embedder = embedder or load_embedder("dense")
        self.candidates = candidates
        self.calibration = calibration or Calibration.from_meta(index.meta)
        self.use_dense_rerank = use_dense_rerank
        self.last_timings: dict[str, float] = {}
        self._dense_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.index.size > 0

    def describe(self) -> dict[str, Any]:
        return {"indexSize": self.index.size, "dim": self.index.dim,
                "embedder": self.embedder.describe(),
                "embedderName": self.index.embedder_name,
                "calibration": self.calibration.to_dict(),
                "indexMeta": {k: v for k, v in self.index.meta.items()
                              if k != "entries"}}

    # ------------------------------------------------------------------
    def match(self, card: np.ndarray, mask: np.ndarray | None = None,
              coverage: float | None = None, k: int = 5,
              rectified: R.RectifiedCard | None = None) -> MatchResult:
        """Identify one rectified card image.

        All four readings of the card (0/90/180/270) are embedded and searched
        together: a tapped card, a card lying upside-down and a card placed
        sideways are all normal situations on a real table, and the reading
        that matches best is reported in `MatchResult.rotation`.
        """
        t0 = time.time()
        result = MatchResult()
        if not self.ready or card is None or card.size == 0:
            result.notes.append("empty index or empty card")
            return result
        coverage = float(coverage if coverage is not None else
                         (rectified.coverage if rectified else 1.0))
        t_embed = time.time()
        rotations = list(CARD_ROTATIONS)
        rotated = [rotate_card(card, r) for r in rotations]
        masks = None if mask is None else [mask] * len(rotated)
        embeddings = self.embedder.embed_batch(rotated, masks)
        t_search = time.time()
        sims, idx = self.index.search(embeddings, k=self.candidates)
        t_rerank = time.time()

        # merge readings: best embedding score per reference and which
        # rotation produced it
        merged: dict[int, tuple[float, int]] = {}
        for row, rotation in enumerate(rotations):
            for j in range(sims.shape[1]):
                i = int(idx[row, j])
                s = float(sims[row, j])
                prev = merged.get(i)
                if prev is None or s > prev[0]:
                    merged[i] = (s, rotation)
        ordered = sorted(merged.items(), key=lambda kv: -kv[1][0])[: self.candidates]
        query_sig = zone_signature(card)
        sig_by_rotation = {r: zone_signature(img) for r, img in zip(rotations, rotated)}
        query_colour = colour_signature(card)
        query_dense = None
        if self.use_dense_rerank and self.index.dense_vectors is not None:
            query_dense = dense_features(card, mask)

        fused: list[tuple[float, int, dict[str, float], int]] = []
        for i, (embed_sim, embed_rotation) in ordered:
            if i < 0 or i >= self.index.size:
                continue
            sig = self.index.reference_signature(i)

            # Geometry cannot resolve which way up a card is, so the
            # image-space signals are evaluated in all four readings; the
            # best-scoring reading per candidate decides (the embedding's
            # rotation is used only as the tie-breaker/start point).
            def _orient_scores(qsig: np.ndarray) -> tuple[dict[str, float], float]:
                out = {
                    "art": max(0.0, _ncc(qsig[SIG_SLICES["art"]],
                                         sig[SIG_SLICES["art"]])),
                    "title": max(0.0, _ncc(qsig[SIG_SLICES["title"]],
                                           sig[SIG_SLICES["title"]])),
                    "collector": max(0.0, _ncc(qsig[SIG_SLICES["collector"]],
                                               sig[SIG_SLICES["collector"]])),
                    "layout": max(0.0, _ncc(qsig[SIG_SLICES["layout"]],
                                            sig[SIG_SLICES["layout"]])),
                }
                key = (W_ART * out["art"] + W_TITLE * out["title"]
                       + W_COLLECTOR * out["collector"] + W_LAYOUT * out["layout"])
                return out, key

            best_scores, best_key, best_rotation = None, -99.0, embed_rotation
            for rotation in rotations:
                candidate_scores, key = _orient_scores(sig_by_rotation[rotation])
                if rotation == embed_rotation:
                    key += 0.02      # slight preference for the embedding's pick
                if key > best_key:
                    best_scores, best_key, best_rotation = candidate_scores, key, rotation
            scores: dict[str, float] = {"embed": float(embed_sim)}
            scores.update(best_scores or {})
            use_flip = best_rotation in (180, 270)
            ref_colour = (self.index.colours[i]
                          if 0 <= i < self.index.colours.shape[0] else None)
            if ref_colour is not None and ref_colour.shape == query_colour.shape:
                delta = float(np.abs(query_colour - ref_colour).mean()) / 255.0
                scores["colour"] = float(max(0.0, 1.0 - delta * 2.2))
            else:
                scores["colour"] = 0.5
            dense_refs = self.index.dense_vectors
            if (query_dense is not None and dense_refs is not None
                    and i < dense_refs.shape[0]):
                scores["dense"] = float(np.dot(query_dense, dense_refs[i]))

            fused_score = (W_EMBED * scores["embed"] + W_ART * scores["art"]
                           + W_LAYOUT * scores["layout"] + W_TITLE * scores["title"]
                           + W_COLLECTOR * scores["collector"]
                           + W_COLOUR * scores["colour"])
            fused.append((fused_score, i, scores, best_rotation))

        fused.sort(key=lambda t: -t[0])
        result.timings = {
            "embedMs": round((t_search - t_embed) * 1000, 2),
            "searchMs": round((t_rerank - t_search) * 1000, 2),
            "rerankMs": round((time.time() - t_rerank) * 1000, 2),
        }
        if not fused:
            result.notes.append("no candidates from index")
            return result

        top_score, top_i, top_scores, top_rotation = fused[0]
        runner = fused[1][0] if len(fused) > 1 else 0.0
        margin = max(0.0, top_score - runner)
        result.top_score = top_score
        result.margin = margin
        result.rotation = top_rotation
        result.flipped = top_rotation in (180, 270)
        result.coverage = coverage

        for fused_score, i, scores, use_flip in fused[:k]:
            entry = self.index.entries[i]
            conf = self.calibration.confidence(
                fused_score, max(0.0, fused_score - runner), coverage)
            result.candidates.append(MatchCandidate(
                set_code=entry.set_code, collector_number=entry.collector_number,
                name=entry.name, confidence=conf, scores=scores,
                image_path=entry.image_path, oracle_id=entry.oracle_id,
                type_line=entry.type_line,
                color_identity=list(entry.color_identity), cmc=entry.cmc,
                flipped=use_flip in (180, 270), source=entry.source))

        # guessing between two printings of the SAME card is not ambiguity
        # about identity — flag it so the UI can offer the chooser instead
        same_name = (len(result.candidates) > 1
                     and result.candidates[0].name == result.candidates[1].name)
        if same_name:
            result.notes.append("ambiguous printing (same card name)")
        if result.candidates[0].confidence < self.calibration.unknown_threshold:
            result.unknown = True
            result.notes.append("below unknown-threshold — card not in index?")
        else:
            result.unknown = False
        if coverage < 0.6:
            result.notes.append("partially occluded or glaring")
        if top_rotation:
            result.notes.append(f"matched at rotation {top_rotation}°"
                                + (" (sideways/tapped)" if top_rotation in (90, 270)
                                   else " (upside-down)"))
        result.timings["totalMs"] = round((time.time() - t0) * 1000, 2)
        return result

    # ------------------------------------------------------------------
    def match_rectified(self, rect: R.RectifiedCard, k: int = 5) -> MatchResult:
        """Match a `RectifiedCard` (carries its own mask/coverage)."""
        return self.match(rect.image, mask=rect.mask, coverage=rect.coverage, k=k,
                          rectified=rect)

    def match_scene(self, rects: Sequence[R.RectifiedCard], k: int = 5
                    ) -> list[MatchResult]:
        return [self.match_rectified(r, k=k) for r in rects]

    # ------------------------------------------------------------------
    def add_card(self, card_image: np.ndarray, set_code: str, collector_number: str,
                 name: str, image_path: str = "", rebuild: bool = False
                 ) -> tuple[int, MatchResult]:
        """Add one card to the live index (the "register unknown card" flow).

        Embeds the image, appends it with its signatures, and returns the
        resulting self-match so the UI can confirm the card round-trips.
        """
        vec = self.embedder.embed_batch([card_image])[0]
        sig = zone_signature(card_image)
        colour = colour_signature(card_image)
        entry = IndexEntry(set_code=set_code.lower(),
                           collector_number=str(collector_number), name=name,
                           image_path=image_path, source="manual")
        self.index.entries.append(entry)
        self.index.vectors = np.vstack([self.index.vectors, vec[None, :]])
        self.index.signatures = np.vstack([self.index.signatures, sig[None, :]])
        self.index.colours = np.vstack([self.index.colours, colour[None, :]])
        if self.index.dense_vectors is not None:
            self.index.dense_vectors = np.vstack(
                [self.index.dense_vectors, dense_features(card_image)[None, :]])
        self.index.dim = self.index.vectors.shape[1]
        return self.index.size - 1, self.match(card_image)

    def rebuild_dense_vectors(self, images: Sequence[np.ndarray] | None = None
                              ) -> None:
        """Populate the optional dense rerank matrix (from images)."""
        if images is None:
            images = [cv2.imread(e.image_path, cv2.IMREAD_COLOR)
                      for e in self.index.entries]
        feats = [dense_features(img) for img in images if img is not None]
        if feats:
            self.index.dense_vectors = np.stack(feats).astype(np.float32)
