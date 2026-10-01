"""FlyCommander physical-table mode — evidence fusion & identification.

Turns OCR evidence (name ± collector info) into ranked card candidates.
Design rules:

  * name OCR is the PRIMARY signal; its failure must not fail registration
  * collector info is a strong SECONDARY signal; same rule
  * fuzzy matching is expected (OCR noise like "Doublinq Season")
  * Scryfall is queried only during registration, never per frame
  * every successful lookup lands in the local cache / name index
"""
from __future__ import annotations

import difflib
import json
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physical.scryfall_cache import HTTP_HEADERS, CardInfo, ScryfallCache
from vision.artmatch import ArtMatcher

NAMED_API = "https://api.scryfall.com/cards/named?fuzzy={name}"
SEARCH_API = "https://api.scryfall.com/cards/search?q={q}&unique=prints"
MAX_PRINT_CANDIDATES = 5


def name_similarity(a: str, b: str) -> float:
    """0-1 similarity between OCR text and a real card name."""
    if not a or not b:
        return 0.0
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass
class Evidence:
    """All available signals for one registration attempt."""
    name_raw: str = ""
    name_conf: float = 0.0          # OCR mean word confidence 0-1
    set_code: str = ""
    collector_number: str = ""
    collector_conf: float = 0.0
    visual_score: float = 0.0       # optional image-similarity hook
    frame_quality: float = 0.0

    def to_dict(self) -> dict:
        return {"nameRaw": self.name_raw, "nameConfidence": round(self.name_conf, 3),
                "set": self.set_code, "collectorNumber": self.collector_number,
                "collectorConfidence": round(self.collector_conf, 3),
                "visualScore": round(self.visual_score, 3),
                "frameQuality": round(self.frame_quality, 3)}


@dataclass
class Candidate:
    name: str
    set_code: str
    collector_number: str
    card_type: str = ""
    base_power: float | None = None
    base_toughness: float | None = None
    oracle_id: str = ""
    combined_confidence: float = 0.0
    name_match: float = 0.0
    collector_match: bool = False
    visual_score: float = 0.0       # per-candidate artwork similarity (0..1)
    image_uri: str = ""            # Scryfall art URL (debug panel preview)
    source: str = ""                # "name+fuzzy", "name+setnum", "setnum", "cache"

    def to_dict(self) -> dict:
        return {"name": self.name, "set": self.set_code,
                "collectorNumber": self.collector_number,
                "cardType": self.card_type,
                "basePower": self.base_power, "baseToughness": self.base_toughness,
                "oracleId": self.oracle_id,
                "combinedConfidence": round(self.combined_confidence, 3),
                "nameMatch": round(self.name_match, 3),
                "collectorMatch": self.collector_match,
                "visualScore": round(self.visual_score, 3),
                "imageUri": self.image_uri,
                "source": self.source}


class CardIdentifier:
    def __init__(self, cache: ScryfallCache, allow_network: bool = True,
                 art_matcher: ArtMatcher | None = None) -> None:
        self.cache = cache
        self.allow_network = allow_network
        self.art_matcher = art_matcher or ArtMatcher(
            allow_network=allow_network)

    # ------------------------------------------------------------------
    # local name index (grows with every cached card)
    # ------------------------------------------------------------------
    def _name_index_names(self) -> list[str]:
        with self.cache._lock:
            rows = self.cache._conn.execute(
                "SELECT DISTINCT name FROM cards").fetchall()
        return [r[0] for r in rows]

    def local_name_match(self, name: str) -> list[CardInfo]:
        """Best fuzzy matches from the local cache (offline path)."""
        names = self._name_index_names()
        if not names:
            return []
        scored = sorted(((name_similarity(name, n), n) for n in names),
                        reverse=True)[:3]
        out: list[CardInfo] = []
        for score, n in scored:
            if score < 0.55:
                continue
            with self.cache._lock:
                row = self.cache._conn.execute(
                    "SELECT set_code, collector_number FROM cards"
                    " WHERE name=? ORDER BY fetched_at LIMIT 1", (n,)).fetchone()
            if row:
                info = self.cache.get(row[0], row[1], use_network=False)
                if info:
                    out.append(info)
        return out

    # ------------------------------------------------------------------
    # network lookups (registration-time only)
    # ------------------------------------------------------------------
    def _get_json(self, url: str) -> dict | None:
        if not self.allow_network:
            return None
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        if data.get("object") == "error":
            return None
        return data

    def fuzzy_by_name(self, name: str) -> CardInfo | None:
        url = NAMED_API.format(name=urllib.parse.quote(name))
        data = self._get_json(url)
        if data is None:
            return None
        info = CardInfo.from_scryfall(data)
        self.cache.put(info)
        return info

    def prints_by_name(self, name: str) -> list[CardInfo]:
        """All printings of an exact name (for the WHICH CARD? chooser)."""
        q = urllib.parse.quote(f'!"{name}"')
        data = self._get_json(SEARCH_API.format(q=q))
        if data is None or not data.get("data"):
            return []
        out = []
        for j in data["data"][:MAX_PRINT_CANDIDATES]:
            info = CardInfo.from_scryfall(j)
            self.cache.put(info)
            out.append(info)
        return out

    # ------------------------------------------------------------------
    # main entry: evidence → candidates
    # ------------------------------------------------------------------
    def identify(self, ev: Evidence) -> list[Candidate]:
        candidates: list[Candidate] = []
        name = ev.name_raw.strip()
        has_name = len(name) >= 3
        has_collector = bool(ev.set_code and ev.collector_number)

        # --- CASE 2: name + set/number → confirm against exact printing
        if has_name and has_collector:
            info = self.cache.get(ev.set_code, ev.collector_number,
                                  use_network=self.allow_network)
            if info is None:
                info = None  # exact lookup failed; fall through to name path
            if info is not None:
                match = name_similarity(name, info.name)
                candidates.append(Candidate(
                    name=info.name, set_code=info.set_code,
                    collector_number=info.collector_number,
                    card_type=info.card_type, base_power=info.base_power,
                    base_toughness=info.base_toughness, oracle_id=info.oracle_id,
                    name_match=match, collector_match=True,
                    image_uri=info.image_uris,
                    source="name+setnum"))

        # --- CASE 1/3: name → fuzzy (cache first, then Scryfall named)
        if has_name:
            primary: CardInfo | None = None
            local = self.local_name_match(name)
            if local and name_similarity(name, local[0].name) >= 0.82:
                primary = local[0]
            else:
                primary = self.fuzzy_by_name(name)
            if primary is not None:
                match = name_similarity(name, primary.name)
                if not any(c.name == primary.name
                           and c.set_code == primary.set_code
                           for c in candidates):
                    candidates.append(Candidate(
                        name=primary.name, set_code=primary.set_code,
                        collector_number=primary.collector_number,
                        card_type=primary.card_type,
                        base_power=primary.base_power,
                        base_toughness=primary.base_toughness,
                        oracle_id=primary.oracle_id,
                        name_match=match, collector_match=False,
                        image_uri=primary.image_uris,
                        source="name+fuzzy"))
                # ambiguity: same name, multiple printings → WHICH CARD?
                if match >= 0.75:
                    seen = {(primary.set_code, primary.collector_number)}
                    # cached printings first (works offline)
                    for pr in self._cached_printings(primary.name):
                        if (pr.set_code, pr.collector_number) not in seen:
                            seen.add((pr.set_code, pr.collector_number))
                            candidates.append(self._printing_candidate(name, pr))
                    # then the full Scryfall print list (online)
                    for pr in self.prints_by_name(primary.name):
                        if (pr.set_code, pr.collector_number) not in seen:
                            seen.add((pr.set_code, pr.collector_number))
                            candidates.append(self._printing_candidate(name, pr))

        # --- CASE 4: collector-only fallback
        if not candidates and has_collector:
            info = self.cache.get(ev.set_code, ev.collector_number,
                                  use_network=self.allow_network)
            if info is not None:
                candidates.append(Candidate(
                    name=info.name, set_code=info.set_code,
                    collector_number=info.collector_number,
                    card_type=info.card_type, base_power=info.base_power,
                    base_toughness=info.base_toughness, oracle_id=info.oracle_id,
                    name_match=0.0, collector_match=True,
                    image_uri=info.image_uris, source="setnum"))

        self._score(candidates, ev)
        candidates.sort(key=lambda c: c.combined_confidence, reverse=True)
        return candidates

    def _cached_printings(self, name: str) -> list[CardInfo]:
        """All cached printings of an exact name (offline ambiguity)."""
        with self.cache._lock:
            rows = self.cache._conn.execute(
                "SELECT set_code, collector_number FROM cards WHERE name=?",
                (name,)).fetchall()
        out = []
        for st, num in rows:
            info = self.cache.get(st, num, use_network=False)
            if info:
                out.append(info)
        return out

    def _printing_candidate(self, ocr_name: str, info: CardInfo) -> Candidate:
        return Candidate(
            name=info.name, set_code=info.set_code,
            collector_number=info.collector_number,
            card_type=info.card_type, base_power=info.base_power,
            base_toughness=info.base_toughness, oracle_id=info.oracle_id,
            name_match=name_similarity(ocr_name, info.name),
            collector_match=False, image_uri=info.image_uris,
            source="name+prints")

    # ------------------------------------------------------------------
    def _score(self, candidates: list[Candidate], ev: Evidence) -> None:
        """Combined confidence — never inflated by a single weak signal."""
        for c in candidates:
            name_score = ev.name_conf * c.name_match
            if c.collector_match:
                col_score = ev.collector_conf
            else:
                col_score = 0.0
            if name_score > 0 and col_score > 0:
                # independent agreeing signals reinforce each other
                combined = 0.6 * name_score + 0.4 * col_score + 0.05
            elif name_score > 0:
                combined = 0.92 * name_score          # name only
            elif col_score > 0:
                combined = 0.88 * col_score           # exact set+number only
            else:
                combined = 0.0
            combined *= (0.85 + 0.15 * min(1.0, ev.frame_quality))
            if ev.visual_score:
                combined = min(1.0, combined + 0.05 * ev.visual_score)
            c.combined_confidence = min(1.0, combined)

    def rescore(self, candidates: list[Candidate], card_img: "np.ndarray | None",
                ) -> list[Candidate]:
        """Artwork-similarity pass over ranked candidates (in-place).

        This is the text-independent signal: when name/collector OCR is weak,
        matching artwork can still identify the card. Similarity is fused
        with the existing combined confidence — a LOW art score pulls a
        candidate down, an unavailable score changes nothing. Re-sorted by
        the new combined confidence.
        """
        if card_img is None or card_img.size == 0:
            return candidates
        for c in candidates:
            art = self.art_matcher.score(card_img, _ArtRef(c))
            if art is None:
                continue                      # no art available: no-op
            c.visual_score = art
            bonus = max(0.0, art - 0.55) * 0.20   # only similarity > 0.55 reinforces
            penalty = max(0.0, 0.45 - art) * 0.25  # dissimilar art disagrees
            c.combined_confidence = min(1.0, max(0.0,
                c.combined_confidence + bonus - penalty))
        candidates.sort(key=lambda c: c.combined_confidence, reverse=True)
        return candidates


class _ArtRef:
    """Duck-typed view of a Candidate for ArtMatcher.reference_art()."""

    def __init__(self, c: Candidate) -> None:
        self.set_code = c.set_code
        self.collector_number = c.collector_number
        self.image_uris = c.image_uri
