"""FlyCommander — procedural MTG-like card & scene synthesis.

This module is the *data engine* the recognition stack is built on. It has two
jobs and they share the same code:

1. **Training data** for the detector and the fine-tuned embedder. A physical
   card sitting on a table is a rectangle whose appearance is perturbed by
   perspective, rotation, sleeves, foil, glare, shadows, blur, noise and
   occlusion. All of those are cheap to simulate and the labels (corners,
   identity) are known exactly — no human annotation, which is the project
   rule "do not make the user manually register cards" applied to model
   training too.

2. **Deterministic fixtures** for the test-suite. `render_card()` is a pure
   function of a `CardSpec`, so tests can build a reference library, then ask
   the pipeline to identify warped/glared/occluded captures of those same
   cards and measure real accuracy — no camera, no network, no copyrighted
   imagery.

The renders are *MTG-shaped*, not MTG-copyrighted: same layout geometry
(title band, art box, type line, text box, P/T, collector line), procedurally
generated abstract artwork, ASCII names. Every visual signal the real
pipeline relies on (border, art, mana symbols, name text, collector number,
set symbol) exists at the correct position and scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

try:  # optional heavy dep, same guard pattern as the rest of the vision stack
    import cv2  # type: ignore

    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    CV_AVAILABLE = False

# canonical card geometry (identical to vision.rectify / physical.card_scan)
CARD_W, CARD_H = 492, 688

# Normalized layout of a modern MTG frame (fractions of card width/height).
LAYOUT = {
    "border": 0.028,
    "title_band": (0.035, 0.100),      # y0, y1
    "art_box": (0.075, 0.105, 0.925, 0.528),   # x0, y0, x1, y1
    "type_band": (0.545, 0.592),
    "text_box": (0.075, 0.600, 0.925, 0.905),
    "pt_box": (0.700, 0.905, 0.925, 0.960),
    "collector": (0.900, 0.945, 0.975, 0.985),
}

# MTG color-identity → frame palette (BGR). Muted, close to real frame tones.
FRAME_COLORS: dict[str, tuple[int, int, int]] = {
    "W": (222, 228, 236),
    "U": (198, 178, 132),
    "B": (86, 76, 74),
    "R": (78, 92, 196),
    "G": (110, 150, 122),
    "gold": (120, 176, 208),
    "artifact": (178, 186, 190),
    "land": (150, 168, 176),
    "colorless": (168, 176, 180),
}
MANA_COLORS: dict[str, tuple[int, int, int]] = {
    "W": (245, 245, 235),
    "U": (232, 190, 130),
    "B": (150, 140, 135),
    "R": (90, 105, 220),
    "G": (130, 175, 140),
    "C": (200, 205, 210),
}
_KIND_BY_TYPE = {
    "creature": "Creature — Beast",
    "instant": "Instant",
    "sorcery": "Sorcery",
    "artifact": "Artifact",
    "enchantment": "Enchantment",
    "land": "Land",
}

_WORDS_A = ("Grizzled", "Verdant", "Hollow", "Storm", "Ember", "Pale", "Iron",
            "Whispering", "Thorned", "Gilded", "Feral", "Silent", "Cinder",
            "Tidal", "Umbral", "Sunlit", "Ravenous", "Frostbound")
_WORDS_B = ("Bears", "Sentinel", "Oracle", "Warden", "Marauder", "Hydra",
            "Archivist", "Serpent", "Wanderer", "Colossus", "Heretic",
            "Falcon", "Behemoth", "Voyager", "Reaver", "Golem", "Dryad",
            "Spider")


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CardSpec:
    """Everything needed to render a card deterministically."""

    name: str
    set_code: str = "TST"
    collector_number: str = "1"
    colors: tuple[str, ...] = ("G",)
    kind: str = "creature"
    rarity: str = "common"
    power: str | None = "3"
    toughness: str | None = "3"
    seed: int = 0

    @property
    def key(self) -> str:
        return f"{self.set_code}:{self.collector_number}"

    @property
    def identity(self) -> str:
        if self.kind == "land":
            return "land"
        if self.kind == "artifact" and not self.colors:
            return "artifact"
        if len(self.colors) > 1:
            return "gold"
        if not self.colors:
            return "colorless"
        return self.colors[0]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "set": self.set_code,
                "collectorNumber": self.collector_number,
                "colors": list(self.colors), "kind": self.kind,
                "rarity": self.rarity, "power": self.power,
                "toughness": self.toughness, "seed": self.seed,
                "oracleId": f"oracle-{self.key}"}


def random_spec(rng: np.random.Generator, index: int = 0) -> CardSpec:
    """One random-but-valid card spec (for synthetic datasets)."""
    kind = str(rng.choice(["creature", "creature", "creature", "instant",
                           "sorcery", "artifact", "enchantment", "land"]))
    n_colors = int(rng.integers(1, 3))
    colors = tuple(sorted(rng.choice(list("WUBRG"), size=n_colors,
                                     replace=False).tolist()))
    name = f"{rng.choice(_WORDS_A)} {rng.choice(_WORDS_B)}"
    if index:
        # make names unique so the identity task stays well-posed
        name = f"{name} {index}"
    set_code = "".join(rng.choice(list("ABCDEFGHJKLMNOPQRSTUVWXYZ"),
                                  size=3).tolist())
    power = toughness = None
    if kind == "creature":
        power = str(int(rng.integers(1, 9)))
        toughness = str(int(rng.integers(1, 9)))
    return CardSpec(name=name, set_code=set_code,
                    collector_number=str(int(rng.integers(1, 400))),
                    colors=colors, kind=kind,
                    rarity=str(rng.choice(["common", "uncommon", "rare",
                                           "mythic"])),
                    power=power, toughness=toughness, seed=int(index) or int(
                        rng.integers(1, 2**31 - 1)))


def default_library(n: int = 24, seed: int = 7) -> list[CardSpec]:
    """A reproducible library of `n` distinct cards (test fixtures)."""
    rng = np.random.default_rng(seed)
    return [random_spec(rng, index=i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# procedural artwork
# ---------------------------------------------------------------------------
# palettes: (dark, mid, light, accent) as BGR fractions — keeps every card
# inside a coherent colour story while compositions differ wildly
_PALETTES = (
    ((0.10, 0.16, 0.42), (0.35, 0.45, 0.85), (0.92, 0.85, 0.70), (0.20, 0.75, 0.85)),
    ((0.28, 0.10, 0.10), (0.55, 0.22, 0.18), (0.95, 0.72, 0.35), (0.15, 0.15, 0.65)),
    ((0.10, 0.22, 0.12), (0.22, 0.48, 0.26), (0.62, 0.88, 0.55), (0.30, 0.60, 0.95)),
    ((0.12, 0.12, 0.30), (0.30, 0.25, 0.55), (0.75, 0.55, 0.90), (0.45, 0.85, 0.95)),
    ((0.30, 0.28, 0.12), (0.60, 0.55, 0.25), (0.95, 0.90, 0.60), (0.20, 0.40, 0.80)),
    ((0.08, 0.08, 0.10), (0.30, 0.30, 0.35), (0.85, 0.88, 0.92), (0.40, 0.30, 0.75)),
    ((0.20, 0.12, 0.08), (0.48, 0.30, 0.15), (0.90, 0.65, 0.35), (0.25, 0.55, 0.70)),
    ((0.14, 0.18, 0.20), (0.35, 0.48, 0.52), (0.80, 0.90, 0.88), (0.55, 0.35, 0.20)),
)

_ART_STYLES = ("landscape", "portrait", "abstract", "cosmic", "storm", "forest")


def _gradient(rgb: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Full-size vertical gradient from `top` colour to `bottom` colour."""
    h, w = rgb.shape[:2]
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(-1, 1, 1)
    column = (top.reshape(1, 1, 3) * (1.0 - ramp)
              + bottom.reshape(1, 1, 3) * ramp)      # (h, 1, 3)
    return np.broadcast_to(column, (h, w, 3)).copy()


def _blob_field(rng, w, h, cell=24, gain=1.0) -> np.ndarray:
    cell = int(max(6, cell * float(rng.uniform(0.5, 2.2))))
    small = rng.random((max(2, h // cell), max(2, w // cell), 3)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (field - 0.5) * 255 * gain


def _poly(pts, colour):
    return [np.array(pts, np.int32).reshape(-1, 1, 2), colour]


def _art_patch(spec: CardSpec, w: int, h: int) -> np.ndarray:
    """Seeded procedural illustration — one of six composition styles.

    Diversity is the point: a recognition benchmark is only meaningful if the
    library's artworks are genuinely different from each other (see the
    ``art NCC`` measurement in tests/test_synthetic.py). Each style draws a
    different kind of scene, with per-card palette, layout, subject and
    texture, so no two seeds produce near-identical art.
    """
    rng = np.random.default_rng(spec.seed * 2654435761 % (2**32))
    palette = _PALETTES[int(rng.integers(0, len(_PALETTES)))]
    dark, mid, light, accent = (np.array(c, np.float32) * 255.0 for c in palette)
    # shuffle the palette roles per card so the same palette still looks new
    if rng.random() < 0.5:
        dark, light = light * 0.5, dark + 60.0

    style = int(rng.integers(0, len(_ART_STYLES)))
    # per-card hue rotation keeps palettes recognisable yet distinct
    if rng.random() < 0.6:
        rot = float(rng.uniform(-0.18, 0.18))
        idx = np.arange(3)
        dark = np.clip(dark * (1.0 + rot * np.cos(idx * 2.1)), 0, 255)
        mid = np.clip(mid * (1.0 + rot * np.sin(idx * 1.7)), 0, 255)
        accent = np.roll(accent, int(rng.integers(0, 3)))
    base = _gradient(np.zeros((h, w, 3), np.float32), light, dark)

    if style == 0:      # landscape: ridges + sun
        base[:] = _gradient(base, light, mid)
        for ridge in range(int(rng.integers(2, 5))):
            depth = 0.45 + 0.5 * ridge / 4.0
            colour = (dark * (1 - ridge / 5.0) + mid * (ridge / 5.0)).tolist()
            pts = [[0, h - 1]]
            for x in range(0, w + 1, max(1, w // 12)):
                y = int(h * depth + math.sin(x / max(8.0, w / 6.0) + ridge * 1.7)
                        * h * 0.06 + rng.normal(0, h * 0.015))
                pts.append([x, min(h - 1, max(0, y))])
            pts.append([w - 1, h - 1])
            cv2.fillPoly(base, [np.array(pts, np.int32)], colour)
        cx, cy = int(rng.uniform(0.15, 0.85) * w), int(rng.uniform(0.1, 0.4) * h)
        r = int(rng.uniform(0.05, 0.14) * w)
        cv2.circle(base, (cx, cy), r, light.tolist(), -1, cv2.LINE_AA)
        cv2.circle(base, (cx, cy), int(r * 1.5), accent.tolist(),
                   max(1, int(r * 0.12)), cv2.LINE_AA)

    elif style == 1:    # portrait: big centred subject on a vignette
        base[:] = _gradient(base, mid, dark)
        base += _blob_field(rng, w, h, cell=16, gain=0.35)
        cx, cy = w // 2 + int(rng.uniform(-0.1, 0.1) * w), int(h * 0.55)
        body = [(cx + int(rng.normal(0, w * 0.10)), cy + int(rng.normal(0, h * 0.10)))
                for _ in range(int(rng.integers(6, 11)))]
        hull = cv2.convexHull(np.array(body, np.int32))
        cv2.fillPoly(base, [hull], (mid * 0.45).tolist())
        head_r = int(rng.uniform(0.09, 0.16) * w)
        cv2.circle(base, (cx, cy - int(h * 0.10)), head_r, (mid * 0.35).tolist(), -1,
                   cv2.LINE_AA)
        for eye in (-1, 1):
            cv2.circle(base, (cx + eye * head_r // 2, cy - int(h * 0.11)),
                       max(1, head_r // 5), accent.tolist(), -1, cv2.LINE_AA)

    elif style == 2:    # abstract: arcs and ribbons
        base[:] = _gradient(base, dark, mid)
        for _ in range(int(rng.integers(4, 9))):
            centre = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
            axes = (int(rng.uniform(0.15, 0.75) * w), int(rng.uniform(0.1, 0.6) * h))
            colour = (light if rng.random() < 0.5 else accent).tolist()
            cv2.ellipse(base, centre, axes, float(rng.uniform(0, 180)),
                        float(rng.uniform(0, 360)), float(rng.uniform(60, 360)),
                        colour, max(1, int(rng.uniform(0.01, 0.05) * h)), cv2.LINE_AA)

    elif style == 3:    # cosmic: starfield + nebula + planet
        base[:] = _gradient(base, (dark * 0.4), dark)
        base += _blob_field(rng, w, h, cell=30, gain=0.6)
        stars = rng.random((h, w)) < 0.006
        base[stars] = 255.0
        pl_r = int(rng.uniform(0.12, 0.28) * w)
        px, py = int(rng.uniform(0.15, 0.85) * w), int(rng.uniform(0.4, 0.9) * h)
        cv2.circle(base, (px, py), pl_r, mid.tolist(), -1, cv2.LINE_AA)
        cv2.ellipse(base, (px, py), (int(pl_r * 1.7), int(pl_r * 0.5)),
                    float(rng.uniform(-30, 30)), 0, 360, accent.tolist(),
                    max(1, pl_r // 8), cv2.LINE_AA)

    elif style == 4:    # storm: diagonal rain + bolt
        base[:] = _gradient(base, dark, (dark * 0.5))
        for _ in range(int(rng.integers(30, 60))):
            x0 = int(rng.uniform(0, w)); y0 = int(rng.uniform(0, h))
            dx = int(rng.uniform(-0.1, 0.1) * w); dy = int(rng.uniform(0.15, 0.35) * h)
            cv2.line(base, (x0, y0), (x0 + dx, min(h - 1, y0 + dy)),
                     (light * 0.7).tolist(), 1, cv2.LINE_AA)
        bolt = [(int(rng.uniform(0.35, 0.65) * w), 0)]
        for k in range(1, 8):
            bolt.append((bolt[-1][0] + int(rng.normal(0, w * 0.07)),
                         int(h * k / 8.0)))
        cv2.polylines(base, [np.array(bolt, np.int32)], False, accent.tolist(),
                      max(2, w // 40), cv2.LINE_AA)

    else:               # forest: trunks + canopy
        base[:] = _gradient(base, (mid * 0.8), dark)
        for _ in range(int(rng.integers(6, 14))):
            x = int(rng.uniform(0, w))
            thickness = max(2, int(rng.uniform(0.01, 0.035) * w))
            cv2.line(base, (x, h), (x + int(rng.normal(0, w * 0.05)), int(h * 0.25)),
                     (dark * 0.55).tolist(), thickness, cv2.LINE_AA)
        for _ in range(int(rng.integers(10, 20))):
            centre = (int(rng.uniform(0, w)), int(rng.uniform(0, int(h * 0.45))))
            cv2.circle(base, centre, int(rng.uniform(0.05, 0.16) * w),
                       (light * rng.uniform(0.5, 0.9)).tolist(), -1, cv2.LINE_AA)

    # every style gets one dominant subject silhouette: the strongest
    # low-frequency discriminator between two cards of the same style
    subj = rng.random()
    if subj < 0.34:
        cx = int(rng.uniform(0.2, 0.8) * w); cy = int(rng.uniform(0.3, 0.7) * h)
        pts = np.array([[cx + int(rng.normal(0, w * 0.16)),
                         cy + int(rng.normal(0, h * 0.16))]
                        for _ in range(int(rng.integers(3, 7)))], np.int32)
        cv2.fillPoly(base, [cv2.convexHull(pts)],
                     (mid * rng.uniform(0.3, 0.8)).tolist())
    elif subj < 0.67:
        cv2.rectangle(base, (int(rng.uniform(0, 0.5) * w), int(rng.uniform(0, 0.5) * h)),
                      (int(rng.uniform(0.5, 1.0) * w), int(rng.uniform(0.5, 1.0) * h)),
                      (accent * rng.uniform(0.4, 1.0)).tolist(),
                      max(2, int(rng.uniform(0.03, 0.12) * w)))
    else:
        bands = int(rng.integers(3, 8))
        for k in range(bands):
            y0 = int(h * k / bands)
            cv2.rectangle(base, (0, y0), (w, y0 + max(2, h // bands - 1)),
                          (mid * (0.2 + 0.8 * rng.random())).tolist(), -1)

    # shared finishing: grain, vignette, a few strokes (structure noise)
    base += rng.normal(0, 6.0, base.shape).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.72
    base *= (1.0 - 0.30 * np.clip(r, 0, 1))[:, :, None]
    for _ in range(int(rng.integers(6, 18))):
        x0, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        x1, y1 = x0 + int(rng.normal(0, w * 0.08)), y0 + int(rng.normal(0, h * 0.05))
        cv2.line(base, (x0, y0), (x1, y1),
                 (accent * rng.uniform(0.4, 1.0)).tolist(),
                 max(1, int(rng.uniform(1, 3))), cv2.LINE_AA)
    return np.clip(base, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# card rendering
# ---------------------------------------------------------------------------
def _text(img, s, org, scale, colour, thickness=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def _rounded(img, p0, p1, colour, radius=8, thickness=-1):
    cv2.rectangle(img, p0, p1, colour, thickness)
    if radius > 0 and thickness == -1:
        r = radius
        for cx, cy in ((p0[0], p0[1]), (p1[0], p0[1]), (p0[0], p1[1]),
                       (p1[0], p1[1])):
            cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)


def render_card(spec: CardSpec, size: tuple[int, int] = (CARD_W, CARD_H),
                sleeve: bool = False) -> np.ndarray:
    """Render one card as a BGR uint8 image (pure function of `spec`)."""
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for card synthesis")
    W, H = size
    frame_colour = FRAME_COLORS[spec.identity]
    card = np.full((H, W, 3), frame_colour, np.uint8)
    bw = max(2, int(W * LAYOUT["border"]))
    cv2.rectangle(card, (0, 0), (W - 1, H - 1), (18, 18, 20), bw)
    cv2.rectangle(card, (bw, bw), (W - bw, H - bw), frame_colour, -1)

    # --- title band ------------------------------------------------------
    y0, y1 = (int(H * f) for f in LAYOUT["title_band"])
    _rounded(card, (bw, y0), (W - bw, y1), (238, 234, 224), radius=int(H * 0.012))
    name_scale = max(0.45, W / 492.0 * 0.62)
    _text(card, spec.name.upper()[:26], (int(W * 0.055), int((y0 + y1) / 2 + H * 0.012)),
          name_scale, (28, 26, 24), 2)

    # mana cost: pips right-aligned in the title band
    pips = list(spec.colors) or (["C"] if spec.kind != "land" else [])
    if spec.kind != "land":
        pip_r = int(H * 0.016)
        cx = W - bw - int(W * 0.03) - pip_r
        cy = (y0 + y1) // 2
        for colour in reversed(pips or ["C"]):
            cv2.circle(card, (cx, cy), pip_r, MANA_COLORS.get(colour, (210, 210, 210)), -1, cv2.LINE_AA)
            cv2.circle(card, (cx, cy), pip_r, (40, 40, 40), 1, cv2.LINE_AA)
            cx -= int(pip_r * 2.3) + 2

    # --- art -------------------------------------------------------------
    ax0, ay0, ax1, ay1 = LAYOUT["art_box"]
    ax0, ay0 = int(W * ax0), int(H * ay0)
    ax1, ay1 = int(W * ax1), int(H * ay1)
    art = _art_patch(spec, ax1 - ax0, ay1 - ay0)
    card[ay0:ay1, ax0:ax1] = art
    cv2.rectangle(card, (ax0 - 2, ay0 - 2), (ax1 + 1, ay1 + 1), (30, 30, 30), 2)

    # --- type line -------------------------------------------------------
    ty0, ty1 = (int(H * f) for f in LAYOUT["type_band"])
    _rounded(card, (bw, ty0), (W - bw, ty1), (232, 228, 218), radius=int(H * 0.008))
    _text(card, _KIND_BY_TYPE.get(spec.kind, "Creature")[:30],
          (int(W * 0.06), ty1 - int(H * 0.008)),
          max(0.35, W / 492.0 * 0.42), (35, 33, 30), 1)

    # --- rules text box --------------------------------------------------
    tx0, tx1 = int(W * LAYOUT["text_box"][0]), int(W * LAYOUT["text_box"][2])
    ty0b, ty1b = int(H * LAYOUT["text_box"][1]), int(H * LAYOUT["text_box"][3])
    _rounded(card, (tx0, ty0b), (tx1, ty1b), (241, 238, 230),
             radius=int(H * 0.01))
    rng = np.random.default_rng(spec.seed + 991)
    line_y = ty0b + int(H * 0.035)
    for line in range(int(rng.integers(2, 4))):
        w_line = int((tx1 - tx0) * rng.uniform(0.45, 0.9))
        cv2.line(card, (tx0 + 8, line_y), (tx0 + 8 + w_line, line_y),
                 (60, 58, 55), max(1, int(H * 0.004)), cv2.LINE_AA)
        line_y += int(H * 0.028)
    _text(card, "Flying, vigilance", (tx0 + 8, line_y + int(H * 0.02)),
          max(0.32, W / 492.0 * 0.36), (70, 68, 64), 1)

    # --- power/toughness -------------------------------------------------
    if spec.kind == "creature" and spec.power:
        px0, py0, px1, py1 = LAYOUT["pt_box"]
        _rounded(card, (int(W * px0), int(H * py0)), (int(W * px1), int(H * py1)),
                 (240, 236, 226), radius=int(H * 0.008))
        _text(card, f"{spec.power}/{spec.toughness}",
              (int(W * (px0 + 0.02)), int(H * (py1 - 0.012))),
              max(0.4, W / 492.0 * 0.45), (25, 25, 25), 2)

    # --- collector line + set symbol ------------------------------------
    cx0, cy0, cx1, cy1 = LAYOUT["collector"]
    collector = f"{spec.set_code.upper()} • EN • {spec.collector_number} {spec.rarity[0].upper()}"
    _text(card, collector, (int(W * (cx0 - 0.885)), int(H * cy1)),
          max(0.22, W / 492.0 * 0.245), (48, 46, 44), 1)
    # set symbol: small distinct glyph (triangle in a circle, seeded rotation)
    sx, sy = int(W * 0.945), int(H * 0.963)
    cv2.circle(card, (sx, sy), int(H * 0.014), (60, 58, 56), -1, cv2.LINE_AA)
    ang = (spec.seed % 12) * 30.0
    pts = []
    for k in range(3):
        a = math.radians(ang + k * 120.0)
        pts.append([int(sx + math.cos(a) * H * 0.010),
                    int(sy + math.sin(a) * H * 0.010)])
    cv2.fillPoly(card, [np.array(pts, np.int32)], (245, 243, 238), cv2.LINE_AA)

    if sleeve:
        card = add_sleeve(card)
    return card


def add_sleeve(card: np.ndarray, border_frac: float = 0.035,
               tint: tuple[int, int, int] = (30, 32, 38)) -> np.ndarray:
    """Wrap a card in a glossy sleeve: dark border, sheen, inner shadow."""
    h, w = card.shape[:2]
    b = max(3, int(max(h, w) * border_frac))
    out = np.full((h + 2 * b, w + 2 * b, 3), tint, np.uint8)
    noise = np.random.default_rng(h * 7919 + w).normal(0, 5, out.shape[:2])
    out = np.clip(out.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    out[b:b + h, b:b + w] = card
    # inner bevel highlight top-left, shadow bottom-right
    cv2.line(out, (b, b), (b + w - 1, b), (90, 95, 110), 1, cv2.LINE_AA)
    cv2.line(out, (b, b), (b, b + h - 1), (90, 95, 110), 1, cv2.LINE_AA)
    cv2.line(out, (b, b + h - 1), (b + w - 1, b + h - 1), (5, 5, 7), 2)
    cv2.rectangle(out, (b - 1, b - 1), (b + w, b + h), (12, 12, 15), 1)
    return out


# ---------------------------------------------------------------------------
# degradations
# ---------------------------------------------------------------------------
def add_glare(card: np.ndarray, rng: np.random.Generator,
              strength: float = 0.8, blobs: int = 2) -> np.ndarray:
    """Specular highlights (foil / glossy sleeve / overhead lamp)."""
    h, w = card.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(blobs)):
        cx = int(rng.uniform(0, w))
        cy = int(rng.uniform(0, h))
        r = int(rng.uniform(0.08, 0.35) * max(h, w))
        value = float(rng.uniform(0.55, 1.0)) * strength
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (cx, cy), r, value, -1)
        mask = np.maximum(mask, cv2.GaussianBlur(m, (0, 0), r * 0.55))
    mask = np.clip(mask, 0, 1)[:, :, None]
    white = np.full_like(card, 255, np.float32)
    return np.clip(card.astype(np.float32) * (1 - mask) + white * mask,
                   0, 255).astype(np.uint8)


def add_foil(card: np.ndarray, rng: np.random.Generator,
             strength: float = 0.5) -> np.ndarray:
    """Rainbow sheen + sparkle, as seen on foiled cards under a lamp."""
    h, w = card.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (xx / max(1.0, w) * 3.0 + yy / max(1.0, h) * 2.0) * math.pi
    hue = (np.sin(phase + float(rng.uniform(0, 6.28))) + 1.0) / 2.0
    rainbow = np.stack([hue, np.roll(hue, 3, axis=1), np.roll(hue, 7, axis=0)],
                       axis=-1) * 255.0
    out = card.astype(np.float32) * (1 - strength * 0.35) + rainbow * (strength * 0.35)
    sparkle = rng.random((h, w)) < 0.004
    out[sparkle] = np.clip(out[sparkle] + 120.0, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_shadow(card: np.ndarray, rng: np.random.Generator,
               strength: float = 0.35) -> np.ndarray:
    """Soft directional shadow across the card (hand / lamp / card above)."""
    h, w = card.shape[:2]
    ang = float(rng.uniform(0, math.pi * 2))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    ramp = np.clip((proj - proj.min()) / max(1e-3, np.ptp(proj)), 0, 1)
    return np.clip(card.astype(np.float32) * (1 - strength * ramp[:, :, None]),
                   0, 255).astype(np.uint8)


def degrade(img: np.ndarray, rng: np.random.Generator, *, blur: float = 0.0,
            noise: float = 0.0, brightness: float = 1.0, contrast: float = 1.0,
            colour_shift: float = 0.0) -> np.ndarray:
    """Camera-side degradations: optics, sensor, exposure, white balance."""
    out = img.astype(np.float32)
    if blur > 0:
        k = int(max(1, round(blur))) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), blur)
    out *= brightness
    out = (out - 128.0) * contrast + 128.0
    if colour_shift:
        out *= (1.0 + rng.normal(0, colour_shift, 3)).astype(np.float32)[None, None, :]
    if noise > 0:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# backgrounds + scenes
# ---------------------------------------------------------------------------
def make_background(size: tuple[int, int], rng: np.random.Generator,
                    kind: str | None = None) -> np.ndarray:
    """Table-like background: cloth, wood, or playmat with a soft vignette."""
    W, H = size
    kind = kind or str(rng.choice(["cloth", "wood", "mat", "cloth"]))
    if kind == "wood":
        base = np.zeros((H, W, 3), np.float32)
        base[:] = np.array(rng.uniform(0.18, 0.42, 3), np.float32) * 255
        for i in range(0, W, max(3, W // 90)):
            shade = 1.0 + 0.12 * math.sin(i / max(6.0, W / 26.0))
            base[:, i:i + max(2, W // 160)] *= shade
    elif kind == "mat":
        base = np.zeros((H, W, 3), np.float32)
        base[:] = np.array(rng.uniform(0.08, 0.3, 3), np.float32) * 255
        cv2.rectangle(base, (int(W * 0.06), int(H * 0.06)),
                      (int(W * 0.94), int(H * 0.94)),
                      tuple(float(v) for v in np.array(rng.uniform(0.15, 0.45, 3)) * 255), -1)
    else:  # cloth
        low = rng.uniform(0.12, 0.4, (max(2, H // 40), max(2, W // 40), 3)).astype(np.float32)
        base = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC) * 255.0
        fibres = rng.normal(0, 9, (H, W, 1)).astype(np.float32)
        base = np.clip(base + fibres, 0, 255)
    base = np.clip(base, 0, 255)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2)) / 0.707
    base *= (1.0 - 0.28 * np.clip(r, 0, 1))[:, :, None]
    base = degrade(base.astype(np.uint8), rng, noise=float(rng.uniform(1.5, 6.0)),
                   brightness=float(rng.uniform(0.75, 1.15)))
    return base


@dataclass
class SceneConditions:
    """Physical conditions under which a scene is captured."""

    n_cards: int = 1
    scale: float = 0.55          # card height as a fraction of frame height
    yaw: float = 22.0            # perspective strength (degrees of keystone)
    pitch: float = 16.0
    rotation: float = 0.0        # in-plane rotation, degrees
    sleeve: bool = False
    foil: bool = False
    glare: float = 0.0           # 0..1 probability/strength
    blur: float = 0.0
    noise: float = 2.0
    brightness: float = 1.0
    occlusion: float = 0.0       # 0..1 fraction of another card covering this one
    overlap: bool = False        # cards deliberately overlapping
    background: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SceneCard:
    """One card placed in a scene, with exact ground-truth geometry."""

    spec: CardSpec | None
    quad: np.ndarray                  # 4x2 float32, tl,tr,br,bl in frame px
    occluded_quad: np.ndarray | None = None   # visible corners if occluded
    visible_frac: float = 1.0         # fraction NOT occluded by other cards
    in_frame_frac: float = 1.0        # fraction inside the captured frame

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.quad[:, 0], self.quad[:, 1]
        return (float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min()))

    def to_dict(self) -> dict[str, Any]:
        return {"spec": self.spec.to_dict() if self.spec else None,
                "quad": [[round(float(x), 2), round(float(y), 2)]
                         for x, y in self.quad],
                "visibleFrac": round(self.visible_frac, 3),
                "inFrameFrac": round(self.in_frame_frac, 3),
                "bbox": [round(v, 2) for v in self.bbox]}


@dataclass
class Scene:
    """A rendered frame plus exact labels."""

    image: np.ndarray
    cards: list[SceneCard] = field(default_factory=list)
    conditions: SceneConditions | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"width": int(self.image.shape[1]),
                "height": int(self.image.shape[0]),
                "cards": [c.to_dict() for c in self.cards],
                "conditions": self.conditions.to_dict() if self.conditions else {}}


def _card_quad(cx: float, cy: float, card_w: float, card_h: float,
               rotation: float, yaw: float, pitch: float) -> np.ndarray:
    """Corners of a card at (cx, cy) with in-plane rotation + keystone."""
    base = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                    np.float32)
    pts = base * np.array([card_w, card_h], np.float32)
    # keystone: scale x by depth (top edge "further" than bottom, or vice versa)
    y_top = pts[0:2, 1]
    depth = np.tanh(np.radians(pitch)) if pitch else 0.0
    factor = 1.0 + depth * np.where(y_top < 0, -1.0, 1.0) * 0.5
    pts[0:2, 0] *= factor
    pts[2:4, 0] *= factor
    # yaw: independent horizontal squeeze of the left/right edges
    yaw_f = np.cos(np.radians(min(abs(yaw), 70.0))) ** 0.75
    pts[:, 0] *= 1.0 if yaw == 0 else max(0.35, yaw_f)
    if rotation:
        a = math.radians(rotation)
        rot = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]], np.float32)
        pts = pts @ rot.T
    return (pts + np.array([cx, cy], np.float32)).astype(np.float32)


def render_card_rgba(spec: CardSpec, conditions: SceneConditions,
                     rng: np.random.Generator, scale_px: float = 1.0
                     ) -> np.ndarray:
    """Render a card with the degradations implied by `conditions`."""
    h = int(CARD_H * scale_px)
    w = int(CARD_W * scale_px)
    img = render_card(spec, (w, h), sleeve=conditions.sleeve)
    if conditions.foil:
        img = add_foil(img, rng, strength=float(rng.uniform(0.25, 0.6)))
    if conditions.glare > 0:
        img = add_glare(img, rng, strength=conditions.glare * float(rng.uniform(0.7, 1.2)),
                        blobs=int(rng.integers(1, 4)))
    if rng.random() < 0.5:
        img = add_shadow(img, rng, strength=float(rng.uniform(0.1, 0.35)))
    return degrade(img, rng, blur=conditions.blur, noise=conditions.noise,
                   brightness=conditions.brightness,
                   contrast=float(rng.uniform(0.85, 1.2)),
                   colour_shift=0.02)


def compose_scene(specs: Sequence[CardSpec], conditions: SceneConditions,
                  size: tuple[int, int] = (1280, 720),
                  seed: int = 0) -> Scene:
    """Paste warped cards onto a background, returning exact labels.

    Cards are composited back-to-front, so with `conditions.overlap` the
    ground-truth for the lower card records which part stays visible. Nothing
    is ever rejected for being tilted — the label is simply the true quad.
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for scene synthesis")
    W, H = size
    rng = np.random.default_rng(seed)
    canvas = make_background((W, H), rng, conditions.background)
    scene = Scene(image=canvas, conditions=conditions)

    n = max(1, min(int(conditions.n_cards), len(specs)))
    chosen = list(specs[:n])
    card_h = H * conditions.scale
    card_w = card_h * CARD_W / CARD_H

    # layout slots: grid-free jitter around the frame centre
    if n == 1:
        centres = [(W * 0.5, H * 0.55)]
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        centres = []
        for i in range(n):
            r, c = divmod(i, cols)
            centres.append((W * (c + 0.5) / cols, H * (r + 0.5) / rows))
    spread = 0.42 if conditions.overlap else 0.0

    placements: list[tuple[CardSpec, np.ndarray, np.ndarray, float]] = []
    for spec, (cx, cy) in zip(chosen, centres):
        cx += float(rng.uniform(-0.06, 0.06)) * W
        cy += float(rng.uniform(-0.05, 0.05)) * H
        if conditions.overlap and n > 1:
            cx += float(rng.uniform(-1, 1)) * spread * W / 6.0
            cy += float(rng.uniform(-1, 1)) * spread * H / 6.0
        rot = conditions.rotation + float(rng.uniform(-6, 6))
        quad = _card_quad(cx, cy, card_w, card_h, rot,
                          conditions.yaw + float(rng.uniform(-4, 4)),
                          conditions.pitch + float(rng.uniform(-3, 3)))
        placements.append((spec, quad, np.array([cx, cy], np.float32), rot))

    # paint far cards first: sort by ascending y so the bottom card occludes
    order = sorted(range(len(placements)),
                   key=lambda i: (placements[i][2][1] + rng.normal(0, 20.0)))
    visible = {i: 1.0 for i in range(len(placements))}
    in_frame = {i: 1.0 for i in range(len(placements))}
    # occlusion mask accumulator: pixels already painted
    painted = np.zeros((H, W), np.uint8)
    quads: dict[int, np.ndarray] = {}
    for i in order:
        spec, quad, centre, rot = placements[i]
        quads[i] = quad
        scale_px = max(0.35, min(1.6, card_h / CARD_H)) * float(rng.uniform(0.98, 1.02))
        img = render_card_rgba(spec, conditions, rng, scale_px)
        ch, cw = img.shape[:2]
        src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]],
                       np.float32)
        M = cv2.getPerspectiveTransform(src, quad)
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        alpha = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (W, H),
                                    flags=cv2.INTER_NEAREST)
        keep = (alpha > 0) & (painted == 0)
        canvas[keep] = warped[keep]
        card_area = int(np.count_nonzero(alpha > 0))
        if card_area:
            visible[i] = float(np.count_nonzero(keep)) / card_area
        # how much of the card is even inside the frame (warpPerspective
        # clips to the canvas, so alpha already excludes out-of-frame area)
        quad_area = max(1.0, float(abs(cv2.contourArea(quad))))
        in_frame[i] = min(1.0, card_area / (quad_area + 1e-6))
        painted = np.maximum(painted, alpha)

    for i, (spec, quad, _centre, _rot) in enumerate(placements):
        scene.cards.append(SceneCard(spec=spec, quad=quad,
                                     visible_frac=visible.get(i, 1.0),
                                     in_frame_frac=in_frame.get(i, 1.0)))
    return scene


# ---------------------------------------------------------------------------
# dataset helpers (training + evaluation)
# ---------------------------------------------------------------------------
def random_conditions(rng: np.random.Generator, hard: bool = False
                      ) -> SceneConditions:
    """A plausible random capture condition (curriculum-friendly).

    `hard=True` samples the long tail the OCR pipeline used to fail on:
    steep perspective, rotation, foil, glare, occlusion, bad exposure.
    """
    if hard:
        return SceneConditions(
            n_cards=int(rng.integers(1, 4)),
            scale=float(rng.uniform(0.22, 0.8)),
            yaw=float(rng.uniform(0, 55)),
            pitch=float(rng.uniform(0, 50)),
            rotation=float(rng.uniform(-180, 180)),
            sleeve=bool(rng.random() < 0.75),
            foil=bool(rng.random() < 0.4),
            glare=float(rng.uniform(0.0, 1.0)) if rng.random() < 0.6 else 0.0,
            blur=float(rng.uniform(0, 3.0)) if rng.random() < 0.5 else 0.0,
            noise=float(rng.uniform(1.0, 9.0)),
            brightness=float(rng.uniform(0.5, 1.5)),
            occlusion=float(rng.uniform(0, 0.5)),
            overlap=bool(rng.random() < 0.5),
        )
    return SceneConditions(
        n_cards=int(rng.integers(1, 3)),
        scale=float(rng.uniform(0.35, 0.75)),
        yaw=float(rng.uniform(0, 30)),
        pitch=float(rng.uniform(0, 25)),
        rotation=float(rng.uniform(-180, 180)),
        sleeve=bool(rng.random() < 0.5),
        foil=bool(rng.random() < 0.2),
        glare=float(rng.uniform(0.0, 0.6)) if rng.random() < 0.35 else 0.0,
        blur=float(rng.uniform(0, 1.5)) if rng.random() < 0.3 else 0.0,
        noise=float(rng.uniform(1.0, 7.0)),
        brightness=float(rng.uniform(0.7, 1.3)),
        overlap=bool(rng.random() < 0.35),
    )


def scene_stream(specs: Iterable[CardSpec], n_scenes: int, *,
                 hard_frac: float = 0.5, size: tuple[int, int] = (1280, 720),
                 seed: int = 0):
    """Yield `(scene_index, Scene)` samples for detector/embedder training."""
    specs = list(specs)
    rng = np.random.default_rng(seed)
    for i in range(n_scenes):
        hard = rng.random() < hard_frac
        cond = random_conditions(rng, hard=hard)
        n = min(cond.n_cards, len(specs))
        pick = rng.choice(len(specs), size=n, replace=False)
        chosen = [specs[int(j)] for j in pick]
        yield i, compose_scene(chosen, cond, size=size,
                               seed=int(rng.integers(0, 2**31 - 1)))


# ---------------------------------------------------------------------------
# fast single-card capture augmentation (training/validation queries)
# ---------------------------------------------------------------------------
def augmented_capture(spec: CardSpec, rng: np.random.Generator, *,
                      hard: bool = False,
                      size: tuple[int, int] = (492, 688),
                      render_scale: float = 0.65) -> np.ndarray:
    """One realistic *rectified* capture of `spec`, as recognition sees it.

    The recognition pipeline hands the matcher a perspective-corrected card
    image. What remains between that image and a clean library scan is:
    residual rotation (a card cannot be told from its own half turn, and a
    tapped card is a quarter turn), optical blur, sensor noise, exposure and
    white-balance drift, sleeve gloss, foil sheen, glare, and partial
    occlusion/background bleed at the edges.

    This function simulates exactly those, at ~6 ms per sample, which is what
    makes on-the-fly training data practical. It is deliberately *the same
    transform* used at inference time downstream (normalize_capture), so
    training and production cannot drift apart.
    """
    W, H = size
    rw = max(64, int(W * render_scale))
    rh = max(90, int(H * render_scale))
    card = render_card(spec, (rw, rh), sleeve=bool(rng.random() < (0.6 if hard else 0.25)))

    if rng.random() < (0.5 if hard else 0.15):
        card = add_foil(card, rng, strength=float(rng.uniform(0.2, 0.7)))
    if rng.random() < (0.7 if hard else 0.25):
        card = add_glare(card, rng,
                         strength=float(rng.uniform(0.4, 1.1)) if hard else float(rng.uniform(0.2, 0.6)),
                         blobs=int(rng.integers(1, 4)))

    # residual perspective: the rectifier removes most keystone, not all of it
    jitter = (rng.uniform(-0.05, 0.05, (4, 2)).astype(np.float32)
              * np.array([W, H], np.float32)) if hard else \
        (rng.normal(0, 0.02, (4, 2)).astype(np.float32) * np.array([W, H], np.float32))
    src = np.array([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]], np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32) + jitter
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_AREA if rw > W else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # rotation: quarter turns (tapped/upside-down) plus a small residual angle
    quarter = int(rng.integers(0, 4))
    if quarter:
        out = cv2.rotate(out, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                               cv2.ROTATE_90_COUNTERCLOCKWISE][quarter - 1])
        # Rectification always emits the canonical *portrait* frame: a tapped
        # or upside-down card arrives as sideways content inside a portrait
        # image, not as a landscape image. Squeezing back to (W, H) reproduces
        # exactly what the matcher receives (normalize_capture resizes to a
        # fixed size anyway).
        if out.shape[:2] != (H, W):
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    residual = float(rng.uniform(-12, 12) if hard else rng.uniform(-3, 3))
    if abs(residual) > 0.4:
        m2 = cv2.getRotationMatrix2D((W / 2, H / 2), residual, 1.0)
        out = cv2.warpAffine(out, m2, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # occlusion / background bleed: a neighbouring card or a hand covers a corner
    if rng.random() < (0.5 if hard else 0.15):
        frac = float(rng.uniform(0.10, 0.45) if hard else rng.uniform(0.05, 0.2))
        cw = int(W * rng.uniform(0.2, 0.6))
        ch = int(H * frac)
        x0 = int(rng.uniform(0, W - cw))
        y0 = int(rng.choice([0, H - ch]))
        colour = tuple(float(v) for v in rng.uniform(0, 90, 3))
        cv2.rectangle(out, (x0, y0), (x0 + cw, y0 + ch), colour, -1)
        if rng.random() < 0.5:
            cv2.rectangle(out, (x0, y0), (x0 + cw, y0 + ch),
                          tuple(float(v) for v in rng.uniform(120, 220, 3)), 3)

    return degrade(out, rng,
                   blur=float(rng.uniform(0, 3.2)) if hard else float(rng.uniform(0, 1.0)),
                   noise=float(rng.uniform(1.5, 9.0)) if hard else float(rng.uniform(1.0, 4.0)),
                   brightness=float(rng.uniform(0.55, 1.45)) if hard else float(rng.uniform(0.85, 1.15)),
                   contrast=float(rng.uniform(0.8, 1.25)),
                   colour_shift=float(rng.uniform(0.0, 0.04)))


def background_crop(rng: np.random.Generator, size: tuple[int, int] = (492, 688)) -> np.ndarray:
    """A table-only crop — the negative class for the cardness classifier."""
    W, H = size
    bg = make_background((W, H), rng)
    if rng.random() < 0.5:      # sometimes a hand/other object intrudes
        x0, y0 = int(rng.uniform(0, W * 0.5)), int(rng.uniform(0, H * 0.5))
        cv2.rectangle(bg, (x0, y0), (x0 + int(rng.uniform(0.2, 0.6) * W),
                                     y0 + int(rng.uniform(0.2, 0.6) * H)),
                      tuple(float(v) for v in rng.uniform(40, 210, 3)), -1)
    return degrade(bg, rng, blur=float(rng.uniform(0, 2.0)),
                   noise=float(rng.uniform(1.0, 7.0)),
                   brightness=float(rng.uniform(0.6, 1.4)))


def place_card_image(card_image: np.ndarray, conditions: SceneConditions,
                     size: tuple[int, int] = (1280, 720), seed: int = 0
                     ) -> Scene:
    """Composite one *arbitrary* card image into a scene (known ground truth).

    Same geometry, background and degradation path as `compose_scene`, but
    the card content is supplied by the caller — a Scryfall scan, a stored
    library image, or a real photograph crop. Used to test the pipeline
    against the exact images the recognition index actually holds.
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV is required for scene synthesis")
    W, H = size
    rng = np.random.default_rng(seed)
    canvas = make_background((W, H), rng, conditions.background)
    card_h = H * conditions.scale
    card_w = card_h * CARD_W / CARD_H
    cx = W * 0.5 + float(rng.uniform(-0.05, 0.05)) * W
    cy = H * 0.55 + float(rng.uniform(-0.05, 0.05)) * H
    quad = _card_quad(cx, cy, card_w, card_h,
                      conditions.rotation + float(rng.uniform(-5, 5)),
                      conditions.yaw + float(rng.uniform(-3, 3)),
                      conditions.pitch + float(rng.uniform(-2, 2)))
    img = degrade(card_image, rng,
                  blur=conditions.blur + float(rng.uniform(0, 0.8)),
                  noise=max(1.0, conditions.noise),
                  brightness=conditions.brightness,
                  contrast=float(rng.uniform(0.9, 1.15)),
                  colour_shift=0.02)
    if conditions.glare > 0:
        img = add_glare(img, rng, strength=conditions.glare * float(rng.uniform(0.6, 1.1)))
    ch, cw = img.shape[:2]
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (W, H),
                                flags=cv2.INTER_NEAREST)
    keep = alpha > 0
    canvas[keep] = warped[keep]
    scene = Scene(image=canvas, conditions=conditions)
    scene.cards.append(SceneCard(spec=None, quad=quad, visible_frac=1.0))
    return scene
