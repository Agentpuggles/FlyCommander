import pytest

from physical.decklists import forge_deck_text, parse_commander_decklist


def test_parse_and_serialize_common_printing_suffixes():
    source = """# exported by a deck builder
Commander:
1 The Gitrog Monster (SOI) 245

Mainboard
1 Sol Ring (CMM) 396 *F*
1x Arcane Signet|MOC|298
97 Forest
Sideboard
1 Counterspell
"""
    commanders, main = parse_commander_decklist(source)
    assert [(row.count, row.name) for row in commanders] == [(1, "The Gitrog Monster")]
    assert [(row.count, row.name) for row in main] == [
        (1, "Sol Ring"), (1, "Arcane Signet"), (97, "Forest")]
    serialized = forge_deck_text(source, "My deck")
    assert serialized.startswith("[metadata]\nName=My deck\n[Commander]\n")
    assert "1 The Gitrog Monster\n[Main]\n" in serialized
    assert "1 Arcane Signet\n97 Forest\n" in serialized
    assert "Counterspell" not in serialized


def test_partner_commanders_and_bracketed_forge_sections():
    source = """[metadata]
Name=Example
[Commander]
1 Tymna the Weaver
1 Thrasios, Triton Hero
[Main]
98 Island
"""
    commanders, main = parse_commander_decklist(source)
    assert sum(row.count for row in commanders) == 2
    assert sum(row.count for row in main) == 98


@pytest.mark.parametrize("source, message", [
    ("Mainboard\n100 Forest", "Commander section"),
    ("Commander\n2 Sol Ring\nMainboard\n98 Forest", "Each commander"),
    ("Commander\n1 Sol Ring\nMainboard\n98 Forest", "100 cards total"),
    ("Commander\n1 Sol Ring\nMainboard\n98 Forest\nSideboard\n5 Island", "100 cards total"),
    ("Commander\n1 Sol Ring\nMainboard\n", "Mainboard section"),
])
def test_rejects_incomplete_or_invalid_deck_structure(source, message):
    with pytest.raises(ValueError, match=message):
        parse_commander_decklist(source)


def test_unknown_bracketed_sections_are_not_parsed_as_cards():
    source = """[Commander]
1 Sol Ring
[Main]
99 Forest
[Wishboard]
1 Black Lotus
"""
    _, main = parse_commander_decklist(source)
    assert sum(row.count for row in main) == 99


def test_bounds_deck_text_and_row_counts():
    with pytest.raises(ValueError, match="no longer than"):
        parse_commander_decklist("x" * 40_001)
    many_rows = "Commander\n1 Sol Ring\nMainboard\n" + "1 Forest\n" * 251
    with pytest.raises(ValueError, match="too many rows"):
        parse_commander_decklist(many_rows)
