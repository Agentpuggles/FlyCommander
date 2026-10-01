"""FlyCommander — local card store (metadata + image paths + embeddings).

One SQLite database holds everything the recognition index needs:

* ``cards``       metadata for every known printing (Scryfall fields)
* ``images``      on-disk image path + a cheap content fingerprint
* ``embeddings``  optional cached vectors (blob) so a restart is instant

SQLite is deliberate: zero dependencies, transactional, and 30k rows are
nothing. The heavy vector search is done in numpy by ``vision.matcher`` from
a memory-mapped array built here, so the DB stays a bookkeeper and never sits
in the hot path.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from cards.scryfall import ScryfallCard, clean_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    set_code         TEXT NOT NULL,
    collector_number TEXT NOT NULL,
    name             TEXT NOT NULL,
    name_fold        TEXT NOT NULL,
    set_name         TEXT DEFAULT '',
    oracle_id        TEXT DEFAULT '',
    scryfall_id      TEXT DEFAULT '',
    layout           TEXT DEFAULT 'normal',
    type_line        TEXT DEFAULT '',
    mana_cost        TEXT DEFAULT '',
    cmc              REAL DEFAULT 0,
    colors           TEXT DEFAULT '[]',
    color_identity   TEXT DEFAULT '[]',
    rarity           TEXT DEFAULT 'common',
    power            TEXT,
    toughness        TEXT,
    oracle_text      TEXT DEFAULT '',
    released_at      TEXT DEFAULT '',
    image_small      TEXT DEFAULT '',
    image_normal     TEXT DEFAULT '',
    image_large      TEXT DEFAULT '',
    image_art        TEXT DEFAULT '',
    updated_at       REAL DEFAULT 0,
    PRIMARY KEY (set_code, collector_number)
);
CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name_fold);
CREATE INDEX IF NOT EXISTS idx_cards_oracle ON cards(oracle_id);

CREATE TABLE IF NOT EXISTS images (
    set_code         TEXT NOT NULL,
    collector_number TEXT NOT NULL,
    path             TEXT NOT NULL,
    bytes            INTEGER DEFAULT 0,
    source           TEXT DEFAULT '',
    fingerprint      TEXT DEFAULT '',
    updated_at       REAL DEFAULT 0,
    PRIMARY KEY (set_code, collector_number)
);

CREATE TABLE IF NOT EXISTS embeddings (
    model            TEXT NOT NULL,
    set_code         TEXT NOT NULL,
    collector_number TEXT NOT NULL,
    dim              INTEGER NOT NULL,
    vector           BLOB NOT NULL,
    updated_at       REAL DEFAULT 0,
    PRIMARY KEY (model, set_code, collector_number)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def fold_name(name: str) -> str:
    """Lowercase, punctuation-stripped name key for fuzzy lookups."""
    return "".join(ch for ch in clean_name(name).lower() if ch.isalnum() or ch == " ")


@dataclass
class ImageRecord:
    set_code: str
    collector_number: str
    path: str
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.collector_number}"


class CardStore:
    """Thread-safe SQLite store for card metadata, images and embeddings."""

    def __init__(self, path: str | Path = "data/cards/cards.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "CardStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- metadata ------------------------------------------------------
    def upsert_cards(self, cards: Iterable[ScryfallCard], commit_every: int = 2000
                     ) -> int:
        """Insert/replace card metadata in batches (bulk import friendly)."""
        n = 0
        rows = []
        now = time.time()
        with self._lock:
            for card in cards:
                rows.append((
                    card.set_code, str(card.collector_number), card.name,
                    fold_name(card.name), card.set_name, card.oracle_id,
                    card.scryfall_id, card.layout, card.type_line, card.mana_cost,
                    card.cmc, json.dumps(card.colors), json.dumps(card.color_identity),
                    card.rarity, card.power, card.toughness, card.oracle_text,
                    card.released_at, card.image_uris.small, card.image_uris.normal,
                    card.image_uris.large, card.image_uris.art_crop, now))
                if len(rows) >= commit_every:
                    n += self._commit_rows(rows)
                    rows = []
            if rows:
                n += self._commit_rows(rows)
        return n

    def _commit_rows(self, rows: list[tuple]) -> int:
        self._conn.executemany(
            """INSERT INTO cards (set_code, collector_number, name, name_fold,
                    set_name, oracle_id, scryfall_id, layout, type_line, mana_cost,
                    cmc, colors, color_identity, rarity, power, toughness,
                    oracle_text, released_at, image_small, image_normal,
                    image_large, image_art, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(set_code, collector_number) DO UPDATE SET
                    name=excluded.name, name_fold=excluded.name_fold,
                    set_name=excluded.set_name, oracle_id=excluded.oracle_id,
                    scryfall_id=excluded.scryfall_id, layout=excluded.layout,
                    type_line=excluded.type_line, mana_cost=excluded.mana_cost,
                    cmc=excluded.cmc, colors=excluded.colors,
                    color_identity=excluded.color_identity, rarity=excluded.rarity,
                    power=excluded.power, toughness=excluded.toughness,
                    oracle_text=excluded.oracle_text,
                    released_at=excluded.released_at,
                    image_small=excluded.image_small,
                    image_normal=excluded.image_normal,
                    image_large=excluded.image_large,
                    image_art=excluded.image_art,
                    updated_at=excluded.updated_at""", rows)
        self._conn.commit()
        return len(rows)

    def get_card(self, set_code: str, collector_number: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cards WHERE set_code=? AND collector_number=?",
                (set_code.lower(), str(collector_number))).fetchone()
        if row is None:
            return None
        return self._card_row_to_dict(row)

    def find_by_name(self, name: str, limit: int = 10,
                     exact: bool = False) -> list[dict[str, Any]]:
        key = fold_name(name)
        with self._lock:
            if exact:
                rows = self._conn.execute(
                    "SELECT * FROM cards WHERE name_fold=? ORDER BY released_at DESC"
                    " LIMIT ?", (key, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM cards WHERE name_fold LIKE ? ORDER BY"
                    " length(name_fold) LIMIT ?", (f"%{key}%", limit)).fetchall()
        return [self._card_row_to_dict(r) for r in rows]

    def name_index(self, limit: int | None = None) -> list[tuple[str, str, str]]:
        """(name_fold, set_code, collector_number) for every stored printing."""
        sql = ("SELECT name_fold, set_code, collector_number FROM cards"
               " ORDER BY name_fold")
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [(r[0], r[1], r[2]) for r in self._conn.execute(sql).fetchall()]

    def distinct_names(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT name FROM cards ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def count_cards(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0])

    # -- images --------------------------------------------------------
    def record_image(self, set_code: str, collector_number: str, path: str | Path,
                     source: str = "", fingerprint: str = "") -> None:
        p = Path(path)
        size = p.stat().st_size if p.exists() else 0
        with self._lock:
            self._conn.execute(
                """INSERT INTO images (set_code, collector_number, path, bytes,
                        source, fingerprint, updated_at) VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(set_code, collector_number) DO UPDATE SET
                        path=excluded.path, bytes=excluded.bytes,
                        source=excluded.source, fingerprint=excluded.fingerprint,
                        updated_at=excluded.updated_at""",
                (set_code.lower(), str(collector_number), str(p), size, source,
                 fingerprint, time.time()))
            self._conn.commit()

    def image_records(self) -> list[ImageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT set_code, collector_number, path, source FROM images"
                " WHERE bytes > 1024 ORDER BY set_code, collector_number").fetchall()
        return [ImageRecord(r[0], r[1], r[2], r[3]) for r in rows]

    def image_count(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM images WHERE bytes > 1024").fetchone()[0])

    # -- embeddings ----------------------------------------------------
    def put_embeddings(self, model: str, keys: Sequence[tuple[str, str]],
                       vectors: np.ndarray) -> int:
        assert len(keys) == len(vectors), "keys/vectors length mismatch"
        now = time.time()
        rows = [(model, s.lower(), str(n), int(vectors.shape[1]),
                 v.astype(np.float32).tobytes(), now)
                for (s, n), v in zip(keys, vectors)]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO embeddings (model, set_code, collector_number, dim,
                        vector, updated_at) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(model, set_code, collector_number) DO UPDATE SET
                        dim=excluded.dim, vector=excluded.vector,
                        updated_at=excluded.updated_at""", rows)
            self._conn.commit()
        return len(rows)

    def get_embeddings(self, model: str) -> tuple[list[tuple[str, str]], np.ndarray]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT set_code, collector_number, dim, vector FROM embeddings"
                " WHERE model=?", (model,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), np.float32)
        dim = rows[0][2]
        keys = [(r[0], r[1]) for r in rows]
        mat = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32)
        return keys, mat.reshape(-1, dim).copy()

    # -- meta ----------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key)"
                " DO UPDATE SET value=excluded.value",
                (key, value if isinstance(value, str) else json.dumps(value)))
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?",
                                     (key,)).fetchone()
        if row is None:
            return default
        value = row[0]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    # ------------------------------------------------------------------
    @staticmethod
    def _card_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for field_name in ("colors", "color_identity"):
            try:
                d[field_name] = json.loads(d.get(field_name) or "[]")
            except json.JSONDecodeError:  # pragma: no cover
                d[field_name] = []
        return d
