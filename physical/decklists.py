"""Parse Commander deck lists into Forge's small, portable ``.dck`` format.

This parser is deliberately structural: Forge remains the authority for card
existence, deck legality, and gameplay. It accepts common plain-text exports
(one card per line with optional quantities and Scryfall set/collector suffixes)
and requires explicit Commander and main-deck sections so a commander is never
guessed from an arbitrary first card.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

MAX_DECK_TEXT = 40_000
MAX_DECK_CARDS = 100
MAX_DECK_ROWS = 250

_COMMANDER_HEADERS = {"commander", "commanders", "commander zone"}
_MAIN_HEADERS = {"main", "mainboard", "maindeck", "deck", "library"}
_IGNORED_HEADERS = {
    "sideboard", "maybeboard", "considering", "tokens", "companion",
    "wishlist", "description", "metadata",
}

# Common Moxfield/Archidekt/Scryfall suffix, e.g. (MOC) 123 or (MOC) 123 *F*.
_PRINTING_SUFFIX = re.compile(
    r"\s+\([A-Za-z0-9]{2,8}\)\s+[A-Za-z0-9-]+(?:\s+\*F\*)?$",
    re.IGNORECASE,
)
_QUANTITY_PREFIX = re.compile(r"^(?:(\d+)\s*x?\s+)?(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DeckRow:
    count: int
    name: str


def _header(line: str) -> str | None:
    """Return a recognized section name, with optional brackets/colon removed."""
    text = line.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    text = text.rstrip(":").strip().casefold()
    if text in _COMMANDER_HEADERS:
        return "commander"
    if text in _MAIN_HEADERS:
        return "main"
    if text in _IGNORED_HEADERS:
        return "ignored"
    return None


def _parse_row(line: str) -> DeckRow:
    match = _QUANTITY_PREFIX.fullmatch(line.strip())
    if not match:
        raise ValueError(f"Could not read deck-list row: {line!r}")
    count = int(match.group(1) or "1")
    name = match.group(2).strip()

    # Forge .dck exports may encode a printing as Name|SET|number. The card
    # database resolves the canonical name for the game; the printing hint is
    # intentionally left out of the rules deck rather than mis-parsed as part
    # of the card name.
    if "|" in name:
        name = name.split("|", 1)[0].strip()
    name = _PRINTING_SUFFIX.sub("", name).strip()
    name = re.sub(r"\s+#\d+\s*$", "", name).strip()
    name = re.sub(r"\s+\*F\*\s*$", "", name, flags=re.IGNORECASE).strip()

    if count < 1 or count > MAX_DECK_CARDS:
        raise ValueError(f"Card quantity must be between 1 and {MAX_DECK_CARDS}: {line!r}")
    if not name or len(name) > 200 or any(ch in name for ch in "\r\n\x00"):
        raise ValueError(f"Invalid card name in row: {line!r}")
    return DeckRow(count=count, name=name)


def parse_commander_decklist(text: str) -> tuple[list[DeckRow], list[DeckRow]]:
    """Return ``(commanders, mainboard)`` after basic Commander-size checks.

    Accepted section forms include ``Commander`` / ``Mainboard``, bracketed
    Forge sections such as ``[Commander]`` / ``[Main]``, and common labels such
    as ``Deck``. Sideboards and maybeboard sections are ignored. Commander text
    must identify one or two commanders and the total deck must contain exactly
    100 cards. Forge performs all card-database and format legality checks.
    """
    if not isinstance(text, str) or len(text) > MAX_DECK_TEXT:
        raise ValueError(f"Deck list must be text no longer than {MAX_DECK_TEXT:,} characters")

    sections: dict[str, list[DeckRow]] = {"commander": [], "main": []}
    active = ""
    rows_seen = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//", ";")):
            continue

        # Existing Forge files include metadata before the deck sections.
        if line.casefold().startswith("name="):
            continue
        section = _header(line)
        if section is not None:
            active = section
            continue
        if line.startswith("[") and line.endswith("]"):
            # Unknown .dck section (e.g. [Sideboard]) should not become a card.
            active = "ignored"
            continue
        if not active or active == "ignored":
            continue

        rows_seen += 1
        if rows_seen > MAX_DECK_ROWS:
            raise ValueError(f"Deck list has too many rows (maximum {MAX_DECK_ROWS})")
        row = _parse_row(line)
        sections[active].append(row)

    commanders = sections["commander"]
    mainboard = sections["main"]
    if not commanders:
        raise ValueError("Add a Commander section with one or two commander cards")
    commander_count = sum(row.count for row in commanders)
    if commander_count not in (1, 2):
        raise ValueError("Commander section must contain one commander, or two partner commanders")
    if any(row.count != 1 for row in commanders):
        raise ValueError("Each commander must appear exactly once")
    if not mainboard:
        raise ValueError("Add a Mainboard section containing the rest of the deck")
    total = commander_count + sum(row.count for row in mainboard)
    if total != MAX_DECK_CARDS:
        raise ValueError(f"Commander deck must contain 100 cards total; found {total}")
    return commanders, mainboard


def forge_deck_text(text: str, deck_name: str) -> str:
    """Validate a list and serialize it as Forge's ``.dck`` text format."""
    name = re.sub(r"[\r\n\x00]", " ", str(deck_name)).strip()
    name = name[:120] or "FlyCommander deck"
    commanders, mainboard = parse_commander_decklist(text)
    lines = ["[metadata]", f"Name={name}", "[Commander]"]
    lines.extend(f"{row.count} {row.name}" for row in commanders)
    lines.append("[Main]")
    lines.extend(f"{row.count} {row.name}" for row in mainboard)
    return "\n".join(lines) + "\n"
