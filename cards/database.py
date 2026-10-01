"""FlyCommander — the card database and recognition-index builder.

``CardDatabase`` is the single source of truth for "what cards exist and what
do they look like". It can be populated from:

* **Scryfall bulk data** — `sync_scryfall()` downloads ``default_cards`` once
  and imports every printing (metadata only, a few minutes → hours of network
  depending on the connection). `download_images()` then fills the visual
  index, reusing Forge's local image cache when available.
* **A local image folder** — `import_folder()` for decks, cubes or proxies the
  user already has on disk (``naming: <set>_<number>.jpg`` or ``<name>.png``).
* **Synthetic specs** — `import_synthetic()` renders the procedural card set
  (tests, demos and the embedder's training library, no copyright issue).

Whatever the source, `build_index(embedder)` produces a `vision.matcher
.CardIndex`: an immutable numpy matrix of card vectors + metadata, saved to
``index.npz`` so startup is a load, not a rebuild.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from cards.cache import CardStore, fold_name
from cards.scryfall import ScryfallCard, ScryfallClient, clean_name

try:
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

DATA_ROOT = Path("data/cards")
INDEX_NAME = "index.npz"
INDEX_META = "index.json"

# layouts that never appear as a physical, recognizable single card face
SKIP_LAYOUTS = {"token", "emblem", "art_series", "double_faced_token",
                "planar", "scheme", "vanguard", "augment", "host"}


@dataclass
class IndexEntry:
    """One indexed printing: identity + vectors + provenance."""

    set_code: str
    collector_number: str
    name: str
    image_path: str = ""
    oracle_id: str = ""
    type_line: str = ""
    color_identity: list[str] = field(default_factory=list)
    cmc: float = 0.0
    source: str = "scryfall"

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.collector_number}"

    def to_dict(self) -> dict[str, Any]:
        return {"set": self.set_code, "collectorNumber": self.collector_number,
                "name": self.name, "imagePath": self.image_path,
                "oracleId": self.oracle_id, "typeLine": self.type_line,
                "colorIdentity": list(self.color_identity), "cmc": self.cmc,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IndexEntry":
        return cls(set_code=d["set"], collector_number=d["collectorNumber"],
                   name=d["name"], image_path=d.get("imagePath", ""),
                   oracle_id=d.get("oracleId", ""),
                   type_line=d.get("typeLine", ""),
                   color_identity=list(d.get("colorIdentity") or []),
                   cmc=float(d.get("cmc") or 0.0),
                   source=d.get("source", "scryfall"))


class CardDatabase:
    """Metadata store + image library + recognition index builder."""

    def __init__(self, root: str | Path = DATA_ROOT,
                 allow_network: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = CardStore(self.root / "cards.sqlite3")
        self.client = ScryfallClient(cache_dir=self.root / "scryfall",
                                     allow_network=allow_network)
        self.allow_network = allow_network
        self.image_dir = self.root / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / INDEX_NAME
        self.index_meta_path = self.root / INDEX_META

    # ------------------------------------------------------------------
    # metadata ingestion
    # ------------------------------------------------------------------
    def sync_scryfall(self, bulk_type: str = "default_cards",
                      force: bool = False,
                      progress: Callable[[int], None] | None = None,
                      limit: int | None = None) -> dict[str, Any]:
        """Import every printing from Scryfall bulk data (offline → no-op)."""
        if not self.allow_network:
            return {"status": "offline", "imported": 0,
                    "note": "network disabled — using existing cache"}
        path = self.client.download_bulk(bulk_type, force=force)
        if path is None or not path.exists():
            return {"status": "no_bulk", "imported": 0,
                    "error": self.client.last_error}
        count = 0

        def _filtered():
            nonlocal count
            for card in self.client.iter_bulk_cards(bulk_type, path):
                if card.layout in SKIP_LAYOUTS:
                    continue
                if limit is not None and count >= limit:
                    return
                count += 1
                if progress and count % 5000 == 0:
                    progress(count)
                yield card

        imported = self.store.upsert_cards(_filtered())
        self.store.set_meta("scryfall_synced_at", time.time())
        return {"status": "ok", "imported": imported, "bulk_file": str(path),
                "total_cards": self.store.count_cards()}

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    def download_images(self, limit: int | None = None,
                        sets: Sequence[str] | None = None,
                        size: str = "small",
                        progress: Callable[[int, int], None] | None = None,
                        only_missing: bool = True) -> dict[str, Any]:
        """Download card pictures for indexed printings (Forge cache first).

        This is the expensive step of a full-library build: 30k printings at
        ~8 req/s is roughly an hour. It is incremental — re-running continues
        where it stopped — and `sets=`/`limit=` make it practical to build a
        small index first and grow it.
        """
        existing = {r.key for r in self.store.image_records()} if only_missing else set()
        set_filter = {s.lower() for s in sets} if sets else None
        rows: list[ScryfallCard] = []
        with self.store._lock:  # metadata read is cheap and rare
            query = "SELECT * FROM cards"
            params: tuple = ()
            if set_filter:
                placeholders = ",".join("?" for _ in set_filter)
                query += f" WHERE set_code IN ({placeholders})"
                params = tuple(set_filter)
            query += " ORDER BY released_at DESC, name"
            cur = self.store._conn.execute(query, params)
            for row in cur:
                d = self.store._card_row_to_dict(row)
                rows.append(ScryfallCard(
                    scryfall_id=d.get("scryfall_id", ""),
                    oracle_id=d.get("oracle_id", ""), name=d["name"],
                    set_code=d["set_code"], set_name=d.get("set_name", ""),
                    collector_number=d["collector_number"],
                    layout=d.get("layout", "normal"),
                    type_line=d.get("type_line", ""), cmc=float(d.get("cmc") or 0),
                    colors=d.get("colors") or [],
                    color_identity=d.get("color_identity") or [],
                    rarity=d.get("rarity", "common")))
                # keep URLs straight from the row (no extra parsing cost)
                rows[-1].image_uris.small = d.get("image_small", "")
                rows[-1].image_uris.normal = d.get("image_normal", "")
                rows[-1].image_uris.large = d.get("image_large", "")
                rows[-1].image_uris.art_crop = d.get("image_art", "")
        downloaded = skipped = failed = 0
        for i, card in enumerate(rows):
            if limit is not None and downloaded >= limit:
                break
            if card.key in existing:
                skipped += 1
                continue
            path = self.client.ensure_image(card, size=size)
            if path is None:
                failed += 1
            else:
                self.store.record_image(card.set_code, card.collector_number,
                                        path, source="forge" if "forge" in str(path)
                                        else "scryfall")
                downloaded += 1
            if progress and downloaded % 100 == 0:
                progress(downloaded, failed)
        return {"status": "ok", "downloaded": downloaded, "skipped": skipped,
                "failed": failed, "images_total": self.store.image_count()}

    def import_folder(self, folder: str | Path, source: str = "local",
                      recursive: bool = True) -> int:
        """Index a folder of card images the user already has.

        Filenames are parsed as ``<set>_<number>.<ext>`` (preferred, exact
        identity) or ``<card name>.<ext>`` (resolved against known metadata).
        """
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            return 0
        pattern = "**/*" if recursive else "*"
        count = 0
        for path in sorted(folder.glob(pattern)):
            if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg",
                                                                 ".png", ".webp"):
                continue
            stem = path.stem
            match = re.match(r"^([A-Za-z0-9]{2,6})[_\-\s](\d+[a-z]?)$", stem)
            if match:
                set_code, number = match.group(1).lower(), match.group(2)
                if self.store.get_card(set_code, number) is None:
                    self.store.upsert_cards([ScryfallCard(
                        scryfall_id="", oracle_id="", name=stem,
                        set_code=set_code, set_name="", collector_number=number)])
                self.store.record_image(set_code, number, path, source=source)
                count += 1
                continue
            name = clean_name(stem.replace("_", " "))
            hits = self.store.find_by_name(name, limit=1, exact=True)
            if hits:
                self.store.record_image(hits[0]["set_code"],
                                        hits[0]["collector_number"], path,
                                        source=source)
            else:
                # unknown card: keep it addressable by its own name
                self.store.upsert_cards([ScryfallCard(
                    scryfall_id="", oracle_id=f"local:{fold_name(name)}",
                    name=name, set_code="lcl", set_name="Local",
                    collector_number=str(count + 1))])
                self.store.record_image("lcl", str(count + 1), path, source=source)
            count += 1
        return count

    def import_synthetic(self, count: int = 24, seed: int = 7, size: int = 300
                         ) -> int:
        """Render and index procedural cards (no network, no copyright).

        Used by the test suite, the demo mode and the embedder training runs.
        """
        from vision.synthetic import default_library, render_card

        specs = default_library(count, seed=seed)
        for spec in specs:
            img = render_card(spec)
            if size and size != img.shape[0]:
                scale = size / float(img.shape[0])
                img = cv2.resize(img, (int(img.shape[1] * scale), size),
                                 interpolation=cv2.INTER_AREA)
            path = self.image_dir / f"syn_{spec.set_code.lower()}_{spec.collector_number}.jpg"
            cv2.imwrite(str(path), img)
            self.store.upsert_cards([ScryfallCard(
                scryfall_id=f"syn-{spec.key}", oracle_id=f"oracle-{spec.key}",
                name=spec.name, set_code=spec.set_code.lower(),
                set_name="Synthetic", collector_number=spec.collector_number,
                type_line=spec.kind.title(), colors=list(spec.colors))])
            self.store.record_image(spec.set_code.lower(),
                                    spec.collector_number, path, source="synthetic")
        return len(specs)

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    def build_index(self, embedder=None, limit: int | None = None,
                    progress: Callable[[int, int], None] | None = None):
        """Build (and persist) the recognition index from all stored images.

        Returns a `vision.matcher.CardIndex`. Embeddings are cached in SQLite
        per model name, so rebuilding after adding cards only embeds the new
        ones.
        """
        from vision.embeddings import load_embedder
        from vision.matcher import (SIGNATURE_DIM, CardIndex, colour_signature,
                                    zone_signature)

        embedder = embedder or load_embedder()
        records = self.store.image_records()
        if limit:
            records = records[:limit]
        if not records:
            return CardIndex.empty(embedder)

        cached_keys, cached_vecs = self.store.get_embeddings(embedder.name)
        cache = {k: i for i, k in enumerate(cached_keys)}
        entries: list[IndexEntry] = []
        vectors: list[np.ndarray] = []
        signatures: list[np.ndarray] = []
        colours: list[np.ndarray] = []
        fresh_keys: list[tuple[str, str]] = []
        fresh_vecs: list[np.ndarray] = []
        todo: list[tuple[int, Any, np.ndarray]] = []

        for rec in records:
            meta = self.store.get_card(rec.set_code, rec.collector_number) or {}
            entry = IndexEntry(set_code=rec.set_code,
                               collector_number=rec.collector_number,
                               name=meta.get("name", f"{rec.set_code} {rec.collector_number}"),
                               image_path=rec.path, oracle_id=meta.get("oracle_id", ""),
                               type_line=meta.get("type_line", ""),
                               color_identity=list(meta.get("color_identity") or []),
                               cmc=float(meta.get("cmc") or 0), source=rec.source)
            idx = len(entries)
            entries.append(entry)
            key = (rec.set_code.lower(), str(rec.collector_number))
            img = cv2.imread(rec.path, cv2.IMREAD_COLOR) if CV_AVAILABLE else None
            if img is None:
                vectors.append(np.zeros(embedder.dim, np.float32))
                signatures.append(np.zeros(SIGNATURE_DIM, np.float32))
                colours.append(np.zeros(3, np.float32))
            else:
                # image-space signals (artwork/title/collector/layout + frame
                # colour) are cheap to build and are what make recognition
                # robust when the embedding is uncertain
                signatures.append(zone_signature(img))
                colours.append(colour_signature(img))
                if key in cache:
                    vectors.append(cached_vecs[cache[key]])
                else:
                    todo.append((idx, entry, img))
                    vectors.append(np.zeros(embedder.dim, np.float32))

        # embed the new images in batches (this is the only slow part)
        batch_imgs: list[np.ndarray] = []
        batch_idx: list[int] = []
        done = 0
        for idx, entry, img in todo:
            batch_imgs.append(img)
            batch_idx.append(idx)
            if len(batch_imgs) >= 32:
                vecs = embedder.embed_batch(batch_imgs)
                for i, v in zip(batch_idx, vecs):
                    vectors[i] = v
                    fresh_keys.append((entries[i].set_code.lower(),
                                       str(entries[i].collector_number)))
                    fresh_vecs.append(v)
                done += len(batch_imgs)
                if progress:
                    progress(done, len(todo))
                batch_imgs, batch_idx = [], []
        if batch_imgs:
            vecs = embedder.embed_batch(batch_imgs)
            for i, v in zip(batch_idx, vecs):
                vectors[i] = v
                fresh_keys.append((entries[i].set_code.lower(),
                                   str(entries[i].collector_number)))
                fresh_vecs.append(v)

        mat = np.stack(vectors).astype(np.float32) if vectors else np.zeros(
            (0, embedder.dim), np.float32)
        sig_mat = (np.stack(signatures).astype(np.float32) if signatures else
                   np.zeros((0, SIGNATURE_DIM), np.float32))
        colour_mat = (np.stack(colours).astype(np.float32) if colours else
                      np.zeros((0, 3), np.float32))
        meta: dict[str, Any] = {"built_at": time.time(),
                                "embedder": embedder.describe()}
        # calibration fitted by scripts/eval_vision.py --calibrate (optional)
        try:
            calib_path = Path(__file__).resolve().parents[1] / "vision" / "weights" / "calibration.json"
            if calib_path.exists():
                meta["calibration"] = json.loads(calib_path.read_text())
        except Exception:  # pragma: no cover - never fail a build on this
            pass
        index = CardIndex(vectors=mat, entries=entries, embedder_name=embedder.name,
                          dim=embedder.dim, signatures=sig_mat, colours=colour_mat,
                          meta=meta)
        index.save(self.root)
        self.store.put_embeddings(embedder.name, fresh_keys,
                                  np.stack(fresh_vecs) if fresh_vecs else
                                  np.zeros((0, embedder.dim), np.float32))
        return index

    def load_index(self, embedder=None):
        """Load the persisted index (None when it has not been built yet)."""
        from vision.matcher import CardIndex

        if not self.index_path.exists():
            return None
        return CardIndex.load(self.root)

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "cards": self.store.count_cards(),
            "images": self.store.image_count(),
            "index": self.index_path.exists(),
            "indexPath": str(self.index_path),
            "lastSync": self.store.get_meta("scryfall_synced_at"),
            "network": self.allow_network,
            "scryfallRequests": self.client.request_count,
        }

    def library_summary(self) -> dict[str, Any]:
        """`stats()` plus a real-vs-synthetic breakdown.

        This distinction matters to the player: a library of synthetic demo
        cards can never recognise a real card, and the UI has to say so
        instead of reporting a 5% "best guess" as if it were a match.
        """
        out = self.stats()
        try:
            with self.store._lock:
                row = self.store._conn.execute(
                    "SELECT COUNT(*) FROM cards WHERE set_name='Synthetic'"
                ).fetchone()
            synthetic = int(row[0]) if row else 0
        except Exception:      # pragma: no cover - defensive (schema drift)
            synthetic = 0
        out["synthetic"] = synthetic
        out["real"] = max(0, int(out.get("cards", 0)) - synthetic)
        return out

    def close(self) -> None:
        self.store.close()
