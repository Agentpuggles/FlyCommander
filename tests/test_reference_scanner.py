"""Offline pixel-level tests, NOT a real-camera accuracy benchmark.

Procedural textured art gives repeatable known correspondences, with shared
frames/rules text to catch false identities based on card layout alone.
"""
import base64
import json

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from cards.database import IndexEntry
from vision.reference_scanner import ReferenceScanner
from physical.deck_library import DeckLibrary, parse_deck
from physical.server import PhysicalTableApp


def card_image(seed):
    rng = np.random.default_rng(seed)
    image = np.full((560, 400, 3), 55, np.uint8)
    cv2.rectangle(image, (15, 15), (385, 545), (160, 170, 180), 8)
    cv2.putText(image, 'SAME TITLE AND FRAME', (20, 62), 0, .65, (235, 235, 235), 2)
    # Detailed, unique illustration on otherwise identical cards.
    for _ in range(270):
        x, y = int(rng.integers(53, 346)), int(rng.integers(100, 280))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(image, (x, y), int(rng.integers(2, 12)), color, -1)
    for y in range(330, 500, 20):
        cv2.putText(image, 'Shared rules text not identity', (20, y), 0, .5, (240, 240, 240), 1)
    return image


def add_refs(root, seeds=(1, 2, 3)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        file = f'{seed}.jpg'
        cv2.imwrite(str(root / file), card_image(seed))
        entry = IndexEntry('tst', str(seed), f'Card {seed}', type_line='Creature')
        rows.append({'file': file, 'entry': entry.to_dict()})
    (root / 'references.json').write_text(json.dumps(rows))
    return ReferenceScanner(root)


def place(frame, card, quad):
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [399, 0], [399, 559], [0, 559]]), np.float32(quad))
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(card, H, size)
    mask = cv2.warpPerspective(np.full(card.shape[:2], 255, np.uint8), H, size)
    frame[mask > 0] = warped[mask > 0]
    return frame


def scene(seed=1):
    return place(np.full((800, 1100, 3), 28, np.uint8), card_image(seed),
                 [[250, 110], [610, 150], [650, 650], [180, 620]])


@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
def test_real_pixels_perspective_lighting_rotation(tmp_path, rotation):
    scanner = add_refs(tmp_path)
    frame = np.rot90(scene(), rotation).copy()
    frame = cv2.convertScaleAbs(frame, alpha=.8, beta=15)
    frame = cv2.GaussianBlur(frame, (3, 3), .6)
    hits = scanner.recognize(frame)
    assert len(hits) == 1
    assert hits[0].match.best.name == 'Card 1'
    assert hits[0].match.best.scores['artInliers'] >= 12
    assert not hits[0].match.unknown


@pytest.mark.parametrize('seed', [7, 11, 19])
def test_shared_frame_and_text_cannot_name_unseen_art(tmp_path, seed):
    scanner = add_refs(tmp_path)
    assert scanner.recognize(scene(seed)) == []


def test_blank_and_random_background_do_not_create_cards(tmp_path):
    scanner = add_refs(tmp_path)
    assert scanner.recognize(np.full((800, 1100, 3), 80, np.uint8)) == []
    rng = np.random.default_rng(900)
    assert scanner.recognize(rng.integers(0, 255, (800, 1100, 3), dtype=np.uint8)) == []
    assert scanner.recognize(None) == []


def test_multiple_cards_and_duplicate_copies(tmp_path):
    scanner = add_refs(tmp_path)
    frame = np.full((750, 1200, 3), 28, np.uint8)
    for x, seed in [(15, 1), (410, 2), (805, 1)]:
        place(frame, card_image(seed), [[x, 100], [x + 330, 90], [x + 340, 610], [x, 630]])
    hits = scanner.recognize(frame)
    assert sorted(h.match.best.name for h in hits) == ['Card 1', 'Card 1', 'Card 2']


def test_competing_names_with_same_art_require_choice(tmp_path):
    scanner = add_refs(tmp_path, seeds=(1,))
    rows = json.loads((tmp_path / 'references.json').read_text())
    cv2.imwrite(str(tmp_path / 'copy.jpg'), card_image(1))
    rows.append({'file': 'copy.jpg', 'entry': IndexEntry('tst', '99', 'Another name').to_dict()})
    (tmp_path / 'references.json').write_text(json.dumps(rows))
    scanner.reload()
    result = scanner.recognize(scene())[0].match
    assert result.unknown
    assert {c.name for c in result.candidates} == {'Card 1', 'Another name'}


def test_scanner_persists_and_loads_without_network(tmp_path):
    add_refs(tmp_path)
    scanner = ReferenceScanner(tmp_path)
    assert scanner.ready
    assert scanner.recognize(scene())[0].match.best.name == 'Card 1'


def jpeg(frame):
    return base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()


def test_scan_to_confirmed_battlefield_no_ocr_no_trained_model(tmp_path, monkeypatch):
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    app.recognizer.reference_scanner = add_refs(tmp_path / 'cards' / 'references')
    monkeypatch.setattr('physical.server.SCAN_AVAILABLE', False)
    out = app.scan_frames([jpeg(scene())])
    assert out['pipeline'] == 'reference-art'
    assert out['candidates'][0]['name'] == 'Card 1'
    assert out['registered'] is None
    assert not app.state.battlefield('player')
    assert app.register_candidate(-1)['status'] == 'error'
    assert app.register_candidate(0)['status'] == 'ok'
    assert app.state.battlefield('player')[0].name == 'Card 1'
    assert app.register_candidate(0)['status'] == 'error'  # consumed choice
    # Explicitly identifying the same name at another location can add a copy.
    second = app.vision_register({'name': 'Card 1', 'set': 'tst', 'collectorNumber': '1',
                                 'trackId': 'second-copy'})
    assert second['status'] == 'ok'
    assert len(app.state.battlefield('player')) == 2
    app.vision_register({'name': 'Card 1', 'set': 'tst', 'collectorNumber': '1',
                         'trackId': 'second-copy'})
    assert len(app.state.battlefield('player')) == 2


def test_no_image_build_does_not_seed_demo_cards(tmp_path):
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    out = app.vision_build_index()
    assert out['status'] == 'error'
    assert app.card_db.store.count_cards() == 0


def test_unknown_votes_never_become_stable(tmp_path):
    from physical.vision_pipeline import CardRecognizer, RecognitionConfig, RecognizedCard
    from vision.matcher import MatchResult, MatchCandidate
    from vision.rectify import rectify
    rec = CardRecognizer(RecognitionConfig(index_dir=str(tmp_path)))
    rect = rectify(card_image(1), np.float32([[0, 0], [399, 0], [399, 559], [0, 559]]))
    match = MatchResult(candidates=[MatchCandidate('x', '1', 'Wrong', .9)], unknown=True)
    track = RecognizedCard('1', rect, match)
    for _ in range(20):
        rec._update_track(track, rect, match)
    assert not track.stable_key
    assert track.stable_votes == 0


def test_deck_parser_preserves_faces_and_deduplicates():
    assert parse_deck('Commander\n1 The Gitrog Monster (SOI) 245\nDeck\n10 Forest\n2x Forest\n1 Fire // Ice\n') == [
        ('The Gitrog Monster', 'soi', '245'), ('Forest', '', ''), ('Fire // Ice', '', '')]


@pytest.mark.parametrize('text', ['', 'Deck\n', 'a' * 40001, '\n'.join(f'Card {i}' for i in range(151))])
def test_bad_decks_rejected(text):
    with pytest.raises(ValueError):
        parse_deck(text)


def test_offline_import_is_explicit(tmp_path):
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    assert app.deck_library.start('1 Sol Ring')['status'] == 'error'
    assert not app.card_db.client.allow_network
    assert not app.identifier.allow_network


def test_import_real_metadata_both_faces_and_partial_errors(tmp_path, monkeypatch):
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    client = app.card_db.client
    card = {'set': 'tst', 'collector_number': '1', 'name': 'Front // Back',
            'oracle_id': 'o', 'type_line': 'Creature', 'card_faces': [
                {'name': 'Front', 'image_uris': {'normal': 'https://example.invalid/front'}, 'power': '2', 'toughness': '3'},
                {'name': 'Back', 'image_uris': {'normal': 'https://example.invalid/back'}, 'power': '3', 'toughness': '4'}]}
    monkeypatch.setattr(client, 'get_json', lambda url: card if url.endswith('/1') else None)
    monkeypatch.setattr(client, '_request', lambda *a, **kw: cv2.imencode('.jpg', card_image(1))[1].tobytes())
    # Run synchronously for deterministic tests; HTTP starts this in a thread.
    app.deck_library._job = {'status': 'running', 'done': 0, 'total': 2, 'errors': []}
    app.deck_library._run([('Front // Back', 'tst', '1'), ('Missing', 'tst', '2')])
    status = app.deck_library.status()
    assert status['status'] == 'partial'
    assert status['done'] == 2
    assert len(status['errors']) == 1
    rows = json.loads((app.recognizer.reference_scanner.root / 'references.json').read_text())
    assert {r['entry']['name'] for r in rows} == {'Front', 'Back'}
    assert app.recognizer.reference_scanner.ready


def test_browser_scanner_controls_and_no_implicit_demo_build():
    from pathlib import Path
    ui = (Path(__file__).parents[1] / 'physical/ui.html').read_text()
    assert 'getUserMedia' in ui and 'createImageBitmap' in ui
    assert 'JSON.stringify({images})' in ui
    assert 'async function buildIndex(synthetic = 48)' not in ui
    assert 'Live identification (confirm to add)' in ui


def test_overlapping_copies_are_not_merged(tmp_path):
    scanner = add_refs(tmp_path, seeds=(1,))
    frame = np.full((750, 1100, 3), 28, np.uint8)
    for x in (30, 180):
        place(frame, card_image(1), [[x, 60], [x + 399, 60], [x + 399, 619], [x, 619]])
    assert len(scanner.recognize(frame)) == 2


def test_large_matching_background_is_rejected(tmp_path):
    scanner = add_refs(tmp_path, seeds=(1,))
    frame = cv2.resize(card_image(1)[95:280, 48:350], (1100, 750))
    # Matching illustration on a playmat is not a plausible full card in view.
    assert scanner.recognize(frame) == []


def test_double_face_confirmation_uses_face_specific_stats(tmp_path, monkeypatch):
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    app.recognizer.reference_scanner = add_refs(tmp_path / 'refs', seeds=(1,))
    scanner = app.recognizer.reference_scanner
    scanner.refs[0]['cardInfo'] = {'cardType': 'Creature', 'basePower': 2.,
                                   'baseToughness': 3., 'oracleId': 'o-front'}
    from physical.scryfall_cache import CardInfo
    app.cache.put(CardInfo('Back', 'tst', '1', 'o-back', 'Creature', 9., 9., {}, 0.))
    out = app.vision_register({'name': 'Card 1', 'set': 'tst', 'collectorNumber': '1'})
    obj = app.state.get(out['trackingId'])
    assert (obj.base_power, obj.base_toughness) == (2., 3.)
    assert obj.oracle_id == 'o-front'


def test_malformed_manifest_is_reported(tmp_path):
    (tmp_path / 'references.json').write_text('not json')
    scanner = ReferenceScanner(tmp_path)
    assert not scanner.ready
    assert scanner.status()['error']


def test_scanner_ui_javascript_smoke():
    """Exercise rendering and escaped inline callbacks without browser downloads."""
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional; needed for UI runtime smoke test')
    root = Path(__file__).parents[1]
    subprocess.run([node, 'tests/ui_scanner_smoke.cjs'], cwd=root, check=True,
                   capture_output=True, text=True, timeout=15)
