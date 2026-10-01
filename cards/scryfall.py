"""FlyCommander — Scryfall access (card oracle data + card images).

The recognition index is built from Scryfall's public API. Two access patterns:

* **Bulk data** (`/bulk-data` → ``default_cards``) for the *metadata* of every
  printing (name, set, collector number, colors, layout, image URLs). One
  download, then everything is local and searchable offline.
* **Per-card images** (`image_uris.small|normal|large`) for the *visual* index.
  Downloads are throttled (Scryfall asks for ~10 req/s max) and cached to disk
  keyed by ``set_collector`` so a re-run costs nothing.

Forge integration: if a local Forge installation is present, its card-image
cache is used **before** hitting the network — Forge already downloads the
same Scryfall images, and on a machine that runs Forge that is thousands of
files saved. Detection is best-effort and silent.

Everything here degrades gracefully offline: `ScryfallClient.available` is
False, calls return empty results, and callers fall back to whatever is
already cached (the project rule: never make the network a hard dependency of
a physical table).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

API_ROOT = "https://api.scryfall.com"
USER_AGENT = "FlyCommander/1.0 (+https://github.com/Agentpuggles/FlyCommander)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
IMAGE_HEADERS = {"User-Agent": USER_AGENT, "Accept": "image/*"}
DEFAULT_RATE_LIMIT_S = 0.12          # ~8 req/s, inside Scryfall's guidance
BULK_TYPE = "default_cards"


class ScryfallError(RuntimeError):
    """Raised for API-level failures (404, rate limit, malformed payload)."""


@dataclass
class ScryfallImage:
    """Image URLs for one card face."""

    small: str = ""
    normal: str = ""
    large: str = ""
    art_crop: str = ""

    def best(self, size: str = "small") -> str:
        return {"small": self.small, "normal": self.normal,
                "large": self.large, "art": self.art_crop}.get(size, self.small) \
            or self.normal or self.large or self.small or self.art_crop

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "ScryfallImage":
        data = data or {}
        return cls(small=data.get("small", ""), normal=data.get("normal", ""),
                   large=data.get("large", ""), art_crop=data.get("art_crop", ""))


@dataclass
class ScryfallCard:
    """The subset of a Scryfall card object the project needs."""

    scryfall_id: str
    oracle_id: str
    name: str
    set_code: str
    set_name: str
    collector_number: str
    layout: str = "normal"
    type_line: str = ""
    mana_cost: str = ""
    cmc: float = 0.0
    colors: list[str] = field(default_factory=list)
    color_identity: list[str] = field(default_factory=list)
    rarity: str = "common"
    power: str | None = None
    toughness: str | None = None
    oracle_text: str = ""
    image_uris: ScryfallImage = field(default_factory=ScryfallImage)
    released_at: str = ""
    digital: bool = False
    promo_types: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.collector_number}"

    @property
    def is_commander_legal(self) -> bool:
        return self.layout not in ("token", "emblem", "art_series", "planar",
                                   "scheme", "vanguard") and not self.digital

    def to_dict(self) -> dict[str, Any]:
        return {
            "scryfallId": self.scryfall_id, "oracleId": self.oracle_id,
            "name": self.name, "set": self.set_code, "setName": self.set_name,
            "collectorNumber": self.collector_number, "layout": self.layout,
            "typeLine": self.type_line, "manaCost": self.mana_cost,
            "cmc": self.cmc, "colors": list(self.colors),
            "colorIdentity": list(self.color_identity), "rarity": self.rarity,
            "power": self.power, "toughness": self.toughness,
            "oracleText": self.oracle_text, "releasedAt": self.released_at,
            "imageUris": {"small": self.image_uris.small,
                          "normal": self.image_uris.normal,
                          "large": self.image_uris.large,
                          "artCrop": self.image_uris.art_crop},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ScryfallCard":
        faces = data.get("card_faces") or []
        face = faces[0] if faces else {}
        images = ScryfallImage.from_json(data.get("image_uris")
                                         or face.get("image_uris"))
        return cls(
            scryfall_id=data.get("id", ""),
            oracle_id=data.get("oracle_id", ""),
            name=data.get("name", "") or face.get("name", ""),
            set_code=(data.get("set") or "").lower(),
            set_name=data.get("set_name", ""),
            collector_number=str(data.get("collector_number", "")),
            layout=data.get("layout", "normal"),
            type_line=data.get("type_line", "") or face.get("type_line", ""),
            mana_cost=data.get("mana_cost", "") or face.get("mana_cost", ""),
            cmc=float(data.get("cmc") or 0.0),
            colors=list(data.get("colors") or []),
            color_identity=list(data.get("color_identity") or []),
            rarity=data.get("rarity", "common"),
            power=data.get("power"), toughness=data.get("toughness"),
            oracle_text=data.get("oracle_text", "") or face.get("oracle_text", ""),
            image_uris=images,
            released_at=data.get("released_at", ""),
            digital=bool(data.get("digital", False)),
            promo_types=list(data.get("promo_types") or []),
        )


# ---------------------------------------------------------------------------
# Forge image cache discovery (free thousands of images when Forge is present)
# ---------------------------------------------------------------------------
_FORGE_HINTS = (
    "~/Library/Application Support/Forge/cache",
    "~/.forge/cache",
    "~/.cache/forge",
    "~/Downloads/mtg forge/forge-gui-desktop/cache",
    "~/Downloads/mtg forge/cache",
)


def forge_image_roots(extra: Sequence[str | Path] = ()) -> list[Path]:
    """Candidate Forge image-cache directories that exist on this machine."""
    roots: list[Path] = []
    env = os.environ.get("FORGE_CACHE")
    if env:
        roots.append(Path(env).expanduser())
    for hint in _FORGE_HINTS:
        roots.append(Path(hint).expanduser())
    for extra_path in extra:
        roots.append(Path(extra_path).expanduser())
    return [r for r in roots if r.is_dir()]


def find_forge_image(set_code: str, collector_number: str,
                     roots: Iterable[Path] | None = None) -> Path | None:
    """Locate a Forge-cached card image (jpg/png, any of Forge's layouts)."""
    set_code = set_code.lower()
    number = str(collector_number)
    roots = list(roots) if roots is not None else forge_image_roots()
    candidates = [
        f"{number}.jpg", f"{number}.png", f"{number}.full.jpg",
        f"{set_code}_{number}.jpg", f"{set_code}_{number}.png",
        f"{set_code}_{number}.full.jpg",
    ]
    for root in roots:
        for sub in (root / "pics" / set_code, root / set_code, root / "cards" / set_code,
                    root):
            for name in candidates:
                path = sub / name
                if path.exists() and path.is_file() and path.stat().st_size > 1024:
                    return path
    return None


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class ScryfallClient:
    """Throttled Scryfall client with on-disk caching.

    Parameters
    ----------
    allow_network : False → every method degrades to a cache-only no-op.
    cache_dir     : where bulk data and images are stored.
    rate_limit_s  : minimum spacing between API requests.
    """

    def __init__(self, cache_dir: str | Path = "data/scryfall",
                 allow_network: bool = True,
                 rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
                 timeout_s: float = 30.0,
                 image_size: str = "small") -> None:
        self.cache_dir = Path(cache_dir)
        self.image_dir = self.cache_dir / "images"
        self.bulk_dir = self.cache_dir / "bulk"
        self.allow_network = allow_network
        self.rate_limit_s = rate_limit_s
        self.timeout_s = timeout_s
        self.image_size = image_size
        self.last_request = 0.0
        self.request_count = 0
        self.error_count = 0
        self.last_error = ""
        for path in (self.cache_dir, self.image_dir, self.bulk_dir):
            path.mkdir(parents=True, exist_ok=True)
        self._forge_roots = forge_image_roots()

    # -- plumbing ------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)
        self.last_request = time.time()

    def _request(self, url: str, headers: dict[str, str] | None = None,
                 raw: bool = False) -> Any:
        if not self.allow_network:
            raise ScryfallError("network disabled (allow_network=False)")
        self._throttle()
        req = urllib.request.Request(url, headers=headers or DEFAULT_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read()
            self.request_count += 1
            return payload if raw else json.loads(payload.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise ScryfallError(self.last_error) from exc

    def get_json(self, path_or_url: str, params: dict[str, Any] | None = None
                 ) -> dict[str, Any] | None:
        """GET a Scryfall JSON endpoint; None on any failure (never raises)."""
        url = path_or_url if path_or_url.startswith("http") else \
            f"{API_ROOT}{path_or_url}"
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        try:
            data = self._request(url)
        except ScryfallError:
            return None
        if isinstance(data, dict) and data.get("object") == "error":
            self.last_error = str(data.get("details", "scryfall error"))
            return None
        return data

    # -- card lookups --------------------------------------------------
    def named(self, name: str, fuzzy: bool = True,
              set_code: str | None = None) -> ScryfallCard | None:
        params: dict[str, Any] = {"fuzzy": str(bool(fuzzy)).lower()}
        if set_code:
            params["set"] = set_code
        data = self.get_json(f"/cards/named?name={urllib.parse.quote(name)}", params)
        return ScryfallCard.from_json(data) if data else None

    def card_by_set_number(self, set_code: str, number: str) -> ScryfallCard | None:
        data = self.get_json(f"/cards/{set_code.lower()}/{urllib.parse.quote(str(number))}")
        return ScryfallCard.from_json(data) if data else None

    def search(self, query: str, unique: str = "prints", page_limit: int = 3
               ) -> list[ScryfallCard]:
        """Paged search (`q=` syntax). Stops on any failure — partial is fine."""
        out: list[ScryfallCard] = []
        url = f"{API_ROOT}/cards/search?q={urllib.parse.quote(query)}&unique={unique}"
        pages = 0
        while url and pages < page_limit:
            data = self.get_json(url)
            if not data:
                break
            for item in data.get("data", []):
                out.append(ScryfallCard.from_json(item))
            url = data.get("next_page") if data.get("has_more") else ""
            pages += 1
        return out

    # -- images --------------------------------------------------------
    def image_path(self, card: ScryfallCard) -> Path:
        safe = f"{card.set_code}_{card.collector_number}".replace("/", "_")
        return self.image_dir / f"{safe}.jpg"

    def ensure_image(self, card: ScryfallCard, size: str | None = None,
                     use_forge: bool = True) -> Path | None:
        """Local path to this card's picture (cache → Forge → download)."""
        path = self.image_path(card)
        if path.exists() and path.stat().st_size > 1024:
            return path
        if use_forge and self._forge_roots:
            found = find_forge_image(card.set_code, card.collector_number,
                                     self._forge_roots)
            if found is not None:
                try:  # cache Forge's copy under our own sane filename
                    shutil.copyfile(found, path)
                    return path
                except OSError:
                    return found
        url = card.image_uris.best(size or self.image_size)
        if not url or not self.allow_network:
            return None
        try:
            payload = self._request(url, headers=IMAGE_HEADERS, raw=True)
        except ScryfallError:
            return None
        if len(payload) < 1024:
            return None
        path.write_bytes(payload)
        return path

    # -- bulk data -----------------------------------------------------
    def bulk_metadata(self, bulk_type: str = BULK_TYPE) -> dict[str, Any] | None:
        data = self.get_json("/bulk-data")
        if not data:
            return None
        for entry in data.get("data", []):
            if entry.get("type") == bulk_type:
                return entry
        return None

    def download_bulk(self, bulk_type: str = BULK_TYPE,
                      force: bool = False) -> Path | None:
        """Download (or reuse) the bulk oracle-card file; returns its path."""
        target = self.bulk_dir / f"{bulk_type}.json"
        meta_path = self.bulk_dir / f"{bulk_type}.meta.json"
        if target.exists() and not force and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            entry = self.bulk_metadata(bulk_type)
            if entry and entry.get("updated_at") == meta.get("updated_at"):
                return target
        entry = self.bulk_metadata(bulk_type)
        if not entry or not entry.get("download_uri"):
            return target if target.exists() else None
        try:
            self._throttle()
            req = urllib.request.Request(entry["download_uri"],
                                         headers=DEFAULT_HEADERS)
            tmp = target.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 256)
            tmp.replace(target)
            meta_path.write_text(json.dumps(
                {"updated_at": entry.get("updated_at"),
                 "size": entry.get("size"), "type": bulk_type}, indent=2))
            return target
        except (ScryfallError, OSError):
            return target if target.exists() else None

    def iter_bulk_cards(self, bulk_type: str = BULK_TYPE,
                        file_path: Path | None = None) -> Iterator[ScryfallCard]:
        """Stream cards from the bulk file (memory-friendly JSON streaming)."""
        path = file_path or (self.bulk_dir / f"{bulk_type}.json")
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            decoder = json.JSONDecoder()
            buffer = ""
            started = False
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                if not started:
                    idx = buffer.find("[")
                    if idx < 0:
                        continue
                    buffer = buffer[idx + 1:]
                    started = True
                while True:
                    buffer = buffer.lstrip().lstrip(",")
                    if not buffer:
                        break
                    try:
                        obj, end = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    buffer = buffer[end:]
                    if obj is None:
                        continue
                    try:
                        yield ScryfallCard.from_json(obj)
                    except Exception:  # pragma: no cover - malformed entry
                        continue
                if not buffer.strip():     # pragma: no cover
                    buffer = ""


def clean_name(name: str) -> str:
    """Normalize a card name for matching (split cards, unicode dashes)."""
    name = re.sub(r"\s*//\s*", " // ", name.strip())
    name = name.replace("’", "'").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", name)
