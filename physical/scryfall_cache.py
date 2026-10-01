"""FlyCommander physical-table mode — local Scryfall cache.

One HTTP fetch per unique (set, collector_number) ever seen; everything else
serves from the local SQLite database. Mirrors mtgscan's use of Scryfall as
the authoritative identification source while keeping the game fully
playable offline after initial registration.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

SCRYFALL_API = "https://api.scryfall.com/cards/{set}/{number}"
HTTP_HEADERS = {
    "User-Agent": "FlyCommander/1.0 (physical-table registration)",
    "Accept": "application/json",   # Scryfall rejects requests without it
}
HTTP_TIMEOUT = 10.0


@dataclass
class CardInfo:
    name: str
    set_code: str
    collector_number: str
    oracle_id: str
    card_type: str
    base_power: float | None
    base_toughness: float | None
    image_uris: dict[str, str]
    fetched_at: float

    def to_dict(self) -> dict:
        return {
            "name": self.name, "set": self.set_code,
            "collectorNumber": self.collector_number,
            "oracleId": self.oracle_id, "cardType": self.card_type,
            "basePower": self.base_power, "baseToughness": self.base_toughness,
            "imageUris": self.image_uris, "fetchedAt": self.fetched_at,
        }

    @classmethod
    def from_scryfall(cls, j: dict) -> "CardInfo":
        def _num(key: str) -> float | None:
            v = j.get(key)
            try:
                return float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None
        return cls(
            name=j.get("name", "Unknown"),
            set_code=j.get("set", "").upper(),
            collector_number=str(j.get("collector_number", "")),
            oracle_id=j.get("oracle_id", ""),
            card_type=j.get("type_line", ""),
            base_power=_num("power"),
            base_toughness=_num("toughness"),
            image_uris=(j.get("image_uris") or {}).get("normal", ""),
            fetched_at=time.time(),
        )


class ScryfallCache:
    """SQLite-backed local cache with graceful offline degradation."""

    def __init__(self, db_path: str | Path = "data/scryfall_cache.sqlite3",
                 allow_network: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network
        # shared across HTTP threads → allow cross-thread use, serialize access
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cards (
                   set_code TEXT NOT NULL,
                   collector_number TEXT NOT NULL,
                   name TEXT, oracle_id TEXT, card_type TEXT,
                   base_power REAL, base_toughness REAL,
                   image_uris TEXT, fetched_at REAL,
                   PRIMARY KEY (set_code, collector_number))""")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    def get(self, set_code: str, collector_number: str,
            use_network: bool = True) -> CardInfo | None:
        set_code = set_code.strip().upper()
        number = str(collector_number).strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT name, set_code, collector_number, oracle_id, card_type,"
                " base_power, base_toughness, image_uris, fetched_at"
                " FROM cards WHERE set_code=? AND collector_number=?",
                (set_code, number)).fetchone()
        if row is not None:
            return CardInfo(
                name=row[0], set_code=row[1], collector_number=row[2],
                oracle_id=row[3], card_type=row[4], base_power=row[5],
                base_toughness=row[6],
                image_uris=json.loads(row[7] or "{}"),
                fetched_at=row[8])
        if use_network and self.allow_network:
            info = self._fetch(set_code, number)
            if info is not None:
                self.put(info)
                return info
        return None

    def put(self, info: CardInfo) -> None:
        # Canonicalise the primary key here, not just in get(): Scryfall's
        # JSON uses a lowercase set code, so a CardInfo built from raw API
        # data (or by hand) used to be stored as "soi" and then looked up as
        # "SOI" — an invisible miss that silently emptied the offline cache.
        info = replace(info,
                       set_code=str(info.set_code).strip().upper(),
                       collector_number=str(info.collector_number).strip())
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?)",
                (info.set_code, info.collector_number, info.name,
                 info.oracle_id, info.card_type, info.base_power,
                 info.base_toughness, json.dumps(info.image_uris),
                 info.fetched_at))
            self._conn.commit()

    # ------------------------------------------------------------------
    def _fetch(self, set_code: str, number: str) -> CardInfo | None:
        url = SCRYFALL_API.format(set=set_code.lower(), number=number)
        req = urllib.request.Request(url, headers=HTTP_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        if data.get("object") == "error":
            return None
        return CardInfo.from_scryfall(data)


def normalize_set_code(raw: str) -> str:
    """OCR output → canonical set code: strip junk, uppercase, 3-5 letters."""
    s = "".join(ch for ch in raw.upper() if ch.isalpha())
    return s[:5]
