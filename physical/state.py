"""FlyCommander physical-table mode — authoritative state model.

The camera proposes; this module disposes. Everything here is the
authoritative, persistent, interpreted public game state. Raw vision output
never writes directly into these structures — it arrives as *candidate
events* reconciled by `physical/engine.py`.

Zones (public unless noted):
    battlefield, graveyard, exile, command, stack  — public
    hand, library                                   — private (counts only
                                                      for other players)
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
ZONES = ("battlefield", "graveyard", "exile", "command", "stack", "hand",
         "library", "limbo")

PUBLIC_ZONES = frozenset({"battlefield", "graveyard", "exile", "command",
                          "stack"})
PRIVATE_ZONES = frozenset({"hand", "library"})

# Creatures that can tap/attack; used to gate tap/sickness logic
CREATURE_TYPES = ("creature",)
PLANESWALKER_TYPES = ("planeswalker",)

_TAP_MIN_DEG = 45.0   # anything >= 45° from upright counts as tapped
_TAP_MAX_DEG = 135.0  # wraps: 135..315 are also "rotated" via abs delta
_orientation_counter = itertools.count(1)
_card_id_counter = itertools.count(1)


def now() -> float:
    return time.time()


def new_tracking_id() -> str:
    return f"card-{next(_card_id_counter):05d}"


def angular_distance(a: float, b: float) -> float:
    """Shortest angular distance between two orientations in degrees."""
    d = (float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def is_tapped_orientation(orientation_deg: float,
                          threshold: float = 90.0,
                          tolerance: float = 30.0) -> bool:
    """Configurable tapped test: within tolerance of 90° (or 270°)."""
    d1 = angular_distance(orientation_deg, threshold)
    d2 = angular_distance(orientation_deg, threshold + 180.0)
    return min(d1, d2) <= tolerance


@dataclass
class Modifier:
    """A temporary power/toughness modification."""
    power: float = 0.0
    toughness: float = 0.0
    expires: str = "end_of_turn"   # end_of_turn | end_of_combat | manual
    source: str = ""               # free-text, e.g. "Giant Growth"
    created_turn: int = 0

    def to_dict(self) -> dict:
        return {"power": self.power, "toughness": self.toughness,
                "expires": self.expires, "source": self.source,
                "createdTurn": self.created_turn}

    @classmethod
    def from_dict(cls, d: dict) -> "Modifier":
        return cls(power=d.get("power", 0.0), toughness=d.get("toughness", 0.0),
                   expires=d.get("expires", "end_of_turn"),
                   source=d.get("source", ""),
                   created_turn=d.get("createdTurn", 0))


@dataclass
class PhysicalObject:
    """A tracked physical game object: a card or a token.

    Identity (name/set/collector number) is fixed at registration; everything
    else is live game state maintained by the engine.
    """
    tracking_id: str = field(default_factory=new_tracking_id)
    name: str = "Unknown"
    is_token: bool = False
    token_type: str = ""            # "Soldier", "Treasure", … for tokens
    set_code: str = ""
    collector_number: str = ""
    oracle_id: str = ""
    card_type: str = ""             # creature/land/artifact/enchantment/…
    base_power: float | None = None
    base_toughness: float | None = None

    zone: str = "battlefield"
    controller: str = "player"      # "player" (human) or "fly"
    owner: str = "player"

    # --- vision-tracked geometry ------------------------------------------
    position: tuple[float, float] = (0.0, 0.0)   # table-space (px)
    orientation_deg: float = 0.0
    orientation_confidence: float = 0.0
    confidence: float = 1.0
    last_seen: float = field(default_factory=now)
    occluded_frames: int = 0
    moved_since_last: bool = False

    # --- status ------------------------------------------------------------
    tapped: bool = False
    entered_battlefield_turn: int | None = None
    entered_battlefield_timestamp: float | None = None
    controlled_since_turn: int | None = None

    counters: dict[str, int] = field(default_factory=dict)
    temporary_modifiers: list[Modifier] = field(default_factory=list)
    granted_abilities: set[str] = field(default_factory=set)
    status_flags: set[str] = field(default_factory=set)  # e.g. "cant_attack"
    damage_marked: float = 0.0

    # --- combat -------------------------------------------------------------
    attacking: bool = False
    attack_target: str | None = None
    blocking: bool = False
    blocked_by: str | None = None

    # token stacks: e.g. 3x Soldier represented by one tracked object
    token_count: int = 1

    # ------------------------------------------------------------------
    @property
    def is_creature(self) -> bool:
        if self.card_type:
            return "creature" in self.card_type.lower()
        return self.base_power is not None and self.base_toughness is not None

    @property
    def summoning_sick(self) -> bool:
        """Rules-derived: a creature is sick until its controller has begun a
        turn after the one during which it arrived/changed control. Haste
        (granted ability) exempts. Uses state.turns_started, maintained by
        the engine on turn advancement — never inferred from vision."""
        if not self.is_creature or self.zone != "battlefield":
            return False
        if "haste" in self.granted_abilities:
            return False
        arrival = max(t for t in (self.entered_battlefield_turn,
                                  self.controlled_since_turn)
                      if t is not None) if (
            self.entered_battlefield_turn is not None
            or self.controlled_since_turn is not None) else None
        if arrival is None:
            return False
        last_start = self._turns_started_ref().get(
            self.controller, arrival)
        return arrival >= last_start

    def _turns_started_ref(self) -> dict[str, int]:
        # state back-reference is injected by PhysicalGameState at creation
        return getattr(self, "_state_ref", None).turns_started \
            if getattr(self, "_state_ref", None) is not None else {}

    @property
    def can_tap(self) -> bool:
        return (self.is_creature
                or "artifact" in self.card_type.lower()
                or "land" in self.card_type.lower())

    def current_power(self) -> float | None:
        if self.base_power is None:
            return None
        p = self.base_power
        p += self.counters.get("+1/+1", 0) - self.counters.get("-1/-1", 0)
        for m in self.temporary_modifiers:
            p += m.power
        return p

    def current_toughness(self) -> float | None:
        if self.base_toughness is None:
            return None
        t = self.base_toughness
        t += self.counters.get("+1/+1", 0) - self.counters.get("-1/-1", 0)
        for m in self.temporary_modifiers:
            t += m.toughness
        return t

    # ------------------------------------------------------------------
    def to_dict(self, include_private: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trackingId": self.tracking_id,
            "name": self.name,
            "isToken": self.is_token,
            "tokenType": self.token_type,
            "tokenCount": self.token_count,
            "type": self.card_type,
            "zone": self.zone,
            "controller": self.controller,
            "owner": self.owner,
            "position": list(self.position),
            "orientationDegrees": round(self.orientation_deg, 1),
            "orientationConfidence": round(self.orientation_confidence, 3),
            "confidence": round(self.confidence, 3),
            "tapped": self.tapped,
            "summoningSick": self.summoning_sick,
            "counters": dict(self.counters),
            "temporaryModifiers": [m.to_dict() for m in self.temporary_modifiers],
            "abilities": sorted(self.granted_abilities),
            "statusFlags": sorted(self.status_flags),
            "damageMarked": self.damage_marked,
            "attacking": self.attacking,
            "attackTarget": self.attack_target,
            "blocking": self.blocking,
            "blockedBy": self.blocked_by,
            "moved": self.moved_since_last,
            "occludedFrames": self.occluded_frames,
        }
        if self.current_power() is not None:
            d["power"] = self.current_power()
            d["toughness"] = self.current_toughness()
            d["basePower"] = self.base_power
            d["baseToughness"] = self.base_toughness
        if self.entered_battlefield_turn is not None:
            d["enteredBattlefieldTurn"] = self.entered_battlefield_turn
        # identity details stay out of public views when requested
        if include_private:
            d["set"] = self.set_code
            d["collectorNumber"] = self.collector_number
            d["oracleId"] = self.oracle_id
        return d


@dataclass
class PhysicalPlayer:
    name: str
    life: int = 40
    commander_damage_taken: dict[str, int] = field(default_factory=dict)
    # private zones are only counts (unless this player is the viewer)
    hand_count: int = 0
    library_count: int = 0
    graveyard: list[str] = field(default_factory=list)   # tracking ids
    command_zone: list[str] = field(default_factory=list)
    poison: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "life": self.life,
                "poison": self.poison, "handCount": self.hand_count,
                "libraryCount": self.library_count,
                "graveyard": list(self.graveyard),
                "commandZone": list(self.command_zone),
                "commanderDamage": dict(self.commander_damage_taken)}


@dataclass
class CombatState:
    """Explicit combat declarations (hybrid: vision candidates + manual)."""
    active: bool = False
    attackers: dict[str, str] = field(default_factory=dict)   # id -> target
    blockers: dict[str, str] = field(default_factory=dict)    # id -> attacker id

    def to_dict(self) -> dict:
        return {"active": self.active,
                "attackers": dict(self.attackers),
                "blockers": dict(self.blockers)}


@dataclass
class StackEntry:
    tracking_id: str
    name: str
    controller: str
    timestamp: float = field(default_factory=now)

    def to_dict(self) -> dict:
        return {"trackingId": self.tracking_id, "name": self.name,
                "controller": self.controller}


@dataclass
class PhysicalGameState:
    """Authoritative physical game state (public info only, by construction).

    The human is 'player'; the Fly is 'fly'. The Fly's battlefield is digital
    (never seen by the camera); the human's battlefield is physical.
    """
    turn: int = 1
    phase: str = "main1"   # main1, combat, blockers, main2, end
    active_player: str = "player"
    priority_player: str = "player"

    players: dict[str, PhysicalPlayer] = field(default_factory=lambda: {
        "player": PhysicalPlayer("Flynn"),
        "fly": PhysicalPlayer("FlyBrain"),
    })
    objects: dict[str, PhysicalObject] = field(default_factory=dict)
    combat: CombatState = field(default_factory=CombatState)
    stack: list[StackEntry] = field(default_factory=list)
    known_public_effects: list[str] = field(default_factory=list)
    # global turn counter per controller's most recent turn start
    turns_started: dict[str, int] = field(
        default_factory=lambda: {"player": 1, "fly": 0})

    # hidden counts (never identities) for private zones
    fly_hand_count: int = 7

    # ------------------------------------------------------------------
    def objects_in(self, zone: str, controller: str | None = None
                   ) -> list[PhysicalObject]:
        out = [o for o in self.objects.values() if o.zone == zone]
        if controller is not None:
            out = [o for o in out if o.controller == controller]
        return out

    def battlefield(self, controller: str | None = None
                    ) -> list[PhysicalObject]:
        return self.objects_in("battlefield", controller)

    def get(self, tracking_id: str) -> PhysicalObject | None:
        return self.objects.get(tracking_id)

    # ------------------------------------------------------------------
    def public_observation(self, viewer: str) -> dict[str, Any]:
        """Strictly public view. Private zones appear as counts only, and
        identities are stripped for any card the viewer can't legally see.

        This is the ONLY way state reaches the Fly brain or the UI.
        """
        me = self.players.get(viewer)
        obs: dict[str, Any] = {
            "turn": self.turn,
            "phase": self.phase,
            "activePlayer": self.active_player,
            "priorityPlayer": self.priority_player,
            "combat": self.combat.to_dict(),
            "stack": [s.to_dict() for s in self.stack],
            "knownPublicEffects": list(self.known_public_effects),
            "players": {},
            "battlefield": [],
            "graveyards": {},
            "exile": [],
            "commandZones": {},
        }
        for pid, p in self.players.items():
            entry: dict[str, Any] = {
                "name": p.name, "life": p.life, "poison": p.poison,
                "commanderDamage": dict(p.commander_damage_taken),
            }
            if pid == viewer:
                entry["handCount"] = p.hand_count
                entry["libraryCount"] = p.library_count
            else:
                # other players' private zones: counts only, identities hidden
                if pid == "fly":
                    entry["handCount"] = self.fly_hand_count
                else:
                    entry["handCount"] = p.hand_count
                entry["libraryCount"] = p.library_count
            obs["players"][pid] = entry

        for o in self.objects.values():
            # emit by zone buckets (private zones never emitted)
            if o.zone == "battlefield":
                obs["battlefield"].append(o.to_dict(include_private=True))
            elif o.zone == "graveyard":
                obs["graveyards"].setdefault(o.controller, []).append(
                    o.to_dict(include_private=True))
            elif o.zone == "exile":
                obs["exile"].append(o.to_dict(include_private=True))
            elif o.zone == "command":
                obs["commandZones"].setdefault(o.controller, []).append(
                    o.to_dict(include_private=True))
        obs.pop("_scratch", None)
        # face-down / hidden objects never leak: objects in hand/library are
        # simply absent from the observation entirely.
        return obs

    # ------------------------------------------------------------------
    def to_fly_observation(self) -> dict[str, Any]:
        """Shape the public state into the SAME dict format the Forge agent
        emits, so the existing sensory encoder consumes it unchanged.

        The fly is the viewer: its own hand stays hidden (count only), the
        human's hand/library are counts only.
        """
        fly = self.players["fly"]
        player_entries = []
        for pid in ("fly", "player"):
            p = self.players[pid]
            board = self.battlefield(pid)
            player_entries.append({
                "name": p.name,
                "isFly": pid == "fly",
                "life": p.life,
                "poison": p.poison,
                "hand": self.fly_hand_count if pid == "fly" else p.hand_count,
                "library": p.library_count,
                "graveyard": len(self.objects_in("graveyard", pid)),
                "exile": len(self.objects_in("exile", pid)),
                "commandZone": len(self.objects_in("command", pid)),
                "creatures": sum(1 for o in board if o.is_creature),
                "lands": sum(1 for o in board if "land" in o.card_type.lower()),
                "commanderDamage": sum(p.commander_damage_taken.values()),
                "board": [
                    {"n": o.name, "t": 1 if o.tapped else 0, "ctl": o.controller}
                    for o in board[:24]
                ],
            })
        n_attackers = len(self.combat.attackers)
        n_blockers = len(self.combat.blockers)
        return {
            "turn": self.turn,
            "phase": self.phase,
            "players": player_entries,
            "stackSize": len(self.stack),
            "attackers": n_attackers,
            "blockers": n_blockers,
            "canPlay": {"land": False, "spell": False, "ability": False},
        }
