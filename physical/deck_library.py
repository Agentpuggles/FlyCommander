"""Explicit, bounded deck import for the artwork scanner (no demo seed).

One name per line, optionally `1 Card Name (SET) 123` (Arena export).
Name-only imports fetch up to four distinct illustrations. Specifying a printing
is preferable: different artwork cannot be matched from an unseen reference.
Network happens only in an explicit import job; scans are local.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from urllib.parse import quote

from cards.database import IndexEntry
from cards.scryfall import ScryfallCard
from physical.scryfall_cache import CardInfo


def parse_deck(text: str) -> list[tuple[str, str, str]]:
    if not isinstance(text, str) or len(text) > 40000:
        raise ValueError("Deck text must be at most 40,000 characters")
    rows = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.casefold().rstrip(':') in {
            'deck', 'commander', 'sideboard', 'maybeboard', 'companion'}:
            continue
        line = re.sub(r'^\d+x?\s+', '', line, flags=re.I)
        m = re.fullmatch(r'(.+?)\s+\(([A-Za-z0-9]{2,8})\)\s+([A-Za-z0-9-]+)(?:\s+\*F\*)?', line)
        row = (m[1].strip(), m[2].lower(), m[3]) if m else (line, '', '')
        if len(row[0]) > 200:
            raise ValueError("Card name too long; use one card per line")
        key = tuple(x.casefold() for x in row)
        if key not in seen:
            rows.append(row)
            seen.add(key)
    if not rows or len(rows) > 150:
        raise ValueError("Paste 1–150 unique cards, one per line")
    return rows


class DeckLibrary:
    def __init__(self, scanner, client, cache):
        self.scanner, self.client, self.cache = scanner, client, cache
        self._lock = threading.Lock()
        self._job = {"status": "idle", "done": 0, "total": 0, "errors": []}

    def status(self):
        with self._lock:
            return {**self._job, "errors": list(self._job['errors']),
                    "scanner": self.scanner.status()}

    def start(self, text):
        rows = parse_deck(text)
        with self._lock:
            if self._job['status'] == 'running':
                return {"status": "busy", "error": "A deck import is already running"}
            if not self.client.allow_network:
                return {"status": "error", "error": "Offline: existing library is available; import needs Scryfall"}
            self._job = {"status": "running", "done": 0, "total": len(rows), "errors": []}
        threading.Thread(target=self._run, args=(rows,), daemon=True,
                         name='deck-library').start()
        return {"status": "running", "total": len(rows)}

    def _run(self, rows):
        root = self.scanner.root
        manifest = root / 'references.json'
        try:
            existing = json.loads(manifest.read_text()) if manifest.exists() else []
            refs = {r['file']: r for r in existing}
            for name, set_code, number in rows:
                try:
                    if set_code:
                        data = self.client.get_json(f'/cards/{quote(set_code)}/{quote(number)}')
                        # A mistyped set/number must not silently import another name.
                        if data and name.casefold() not in {
                            data.get('name', '').casefold(),
                            *[f.get('name', '').casefold() for f in data.get('card_faces', [])]}:
                            raise ValueError('printing does not match card name')
                        cards = [data] if data else []
                    else:
                        # Exact name query, no fuzzy guess and no full bulk download.
                        query = '!"' + name.replace('"', '').replace('\\', '') + '" game:paper'
                        data = self.client.get_json('/cards/search?unique=art&order=released&q=' + quote(query))
                        cards = (data or {}).get('data', [])[:4]
                    if not cards:
                        raise ValueError(self.client.last_error or 'card not found')
                    for data in cards:
                        if any(kind in data.get('type_line', '') for kind in ('Saga', 'Class', 'Case', 'Room')):
                            raise ValueError('unusual artwork region is not supported yet; use Add by name')
                        if data.get('layout', 'normal') not in {
                            'normal', 'transform', 'modal_dfc', 'meld', 'adventure', 'token'}:
                            raise ValueError('artwork layout ' + data['layout'] +
                                             ' is not supported yet; use Add by name')
                        faces = data.get('card_faces') or []
                        variants = [data] if data.get('image_uris') else [
                            {**data, **face, 'oracle_id': data.get('oracle_id', '')}
                            for face in faces if face.get('image_uris')]
                        if not variants:
                            raise ValueError('no reference image for this printing')
                        for face in variants:
                            if any(kind in face.get('type_line', '') for kind in ('Saga', 'Class', 'Case', 'Room')):
                                raise ValueError('unusual artwork region is not supported yet; use Add by name')
                            card = ScryfallCard.from_json(face)
                            # Face-aware cache key; normal-size pictures retain SIFT detail.
                            token = hashlib.sha256((card.key + card.name).encode()).hexdigest()[:24]
                            file = token + '.jpg'
                            path = root / file
                            if file not in refs and len(refs) >= 600:
                                raise ValueError('library limit: 600 faces; use a separate data directory for another deck')
                            if not path.exists():
                                payload = self.client._request(card.image_uris.best('normal'),
                                    headers={'User-Agent': 'FlyCommander/1.0', 'Accept': 'image/*'}, raw=True)
                                import cv2
                                import numpy as np
                                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                                if image is None:
                                    raise ValueError('unreadable reference image')
                                if not cv2.imwrite(str(path), image):
                                    raise ValueError('could not save reference image')
                            entry = IndexEntry(card.set_code, card.collector_number, card.name,
                                image_path=str(path), oracle_id=card.oracle_id, type_line=card.type_line)
                            refs[file] = {'file': file, 'entry': entry.to_dict(),
                                          'cardInfo': CardInfo.from_scryfall(face).to_dict()}
                            self.cache.put(CardInfo.from_scryfall(face))
                except Exception as exc:
                    with self._lock:
                        self._job['errors'].append(f'{name}: {exc}')
                finally:
                    with self._lock:
                        self._job['done'] += 1
            # Atomic persistence; a failed row does not discard previous cards.
            tmp = manifest.with_suffix('.tmp')
            tmp.write_text(json.dumps(list(refs.values())))
            tmp.replace(manifest)
            self.scanner.reload()
            with self._lock:
                self._job['status'] = 'partial' if self._job['errors'] else 'complete'
                if self.scanner.last_error:
                    self._job['errors'].append(self.scanner.last_error)
                    self._job['status'] = 'error'
        except Exception as exc:
            with self._lock:
                self._job['status'] = 'error'
                self._job['errors'].append(str(exc))
