"""FlyCommander — deck-spec handling for the Python orchestration layer.

Deck *discovery and randomization live on the Java side* (Forge's own
Commander pool, via ``DeckProxy.getAllCommanderDecks()`` / ``MyRandom``).
Python only validates and forwards spec strings to the JVM:

    "random"       Forge picks from its normal Commander Decks pool
    "<deck name>"  a name in Forge's Commander deck storage (or pool)
    "<path>.dck"   an explicit deck file (existing behaviour)

Python deliberately does not scan Forge's directories or duplicate Forge's
deck-discovery logic; it keeps only the minimal file-path validation so that
typo'd local paths fail fast with a clear message.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM = "random"


@dataclass(frozen=True)
class DeckSpec:
    """A deck reference forwarded verbatim to the Java resolver.

    kind: "random" | "name" | "file"
    """

    raw: str
    kind: str
    path: Path | None = None

    @property
    def java_arg(self) -> str:
        """Value passed to the JVM (paths absolute, others verbatim)."""
        if self.kind == "file" and self.path is not None:
            return str(self.path)
        return self.raw


def classify_spec(spec: str, deck_dir: Path | None = None) -> DeckSpec:
    """Classify a user-supplied deck spec without doing any Forge discovery.

    Raises ValueError for path-like specs that don't exist; name and random
    specs are forwarded as-is (Forge owns whether they resolve).
    """
    s = spec.strip()
    if not s:
        raise ValueError("empty deck spec")
    if s.lower() == RANDOM:
        return DeckSpec(raw=RANDOM, kind="random")

    # Path-like: only trust it if the file actually exists.
    candidate = Path(s)
    if candidate.suffix == ".dck" or candidate.is_absolute() or candidate.exists():
        resolved = candidate if candidate.is_absolute() else (deck_dir / candidate) \
            if deck_dir else candidate
        if not resolved.exists():
            raise ValueError(f"deck file not found: {resolved}")
        return DeckSpec(raw=s, kind="file", path=resolved)

    # Plain name → Forge storage resolves it (case-insensitively if needed).
    return DeckSpec(raw=s, kind="name")


def expand_specs(specs: list[str], deck_dir: Path | None = None) -> list[DeckSpec]:
    """Classify several specs; all errors are raised together."""
    errors: list[str] = []
    out: list[DeckSpec] = []
    for s in specs:
        try:
            out.append(classify_spec(s, deck_dir))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return out
