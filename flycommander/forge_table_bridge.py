"""FlyCommander — physical table → Forge bridge.

The physical table never rules-judges on its own: it *observes* (camera →
geometry → recognition → tracking) and *proposes* events. Forge owns the
rules, the stack, priority, combat and the real game state. This module is the
wire between the two:

    physical Event  →  table action  →  POST /table/events  →  Forge applies
    Forge observation  ←  GET /observation  ←  fly/UI reads the real state

Contract (implemented by the Java agent in [forge_patch/src/fly/agent]):

    POST /table/events
      {"tableId": "physical", "seat": "player", "events": [ <action>, ... ]}
      → asynchronous engine: {"status": "queued", "accepted": n,
                              "duplicates": n, "pending": n}
        (actions are applied on the Forge game thread at the next priority)
      → synchronous engine: {"status": "ok"|"partial", "applied": n,
         "rejected": n, "results": [{"actionId": "...",
         "status": "applied"|"rejected", "reason": "..."}]}

    GET  /table/queue
      → {"pending": n, "applied": n, "rejected": n, "dropped": n,
         "duplicates": n, "recent": ["applied:set_tapped", ...]}

    POST /table/state
      {"tableId": "physical", "seat": "player", "battlefield": [ ... ]}
      → same envelope (idempotent full resync after a drift / reconnect)

    GET  /health, /observation, /result   (existing agent surface)

Design rules:

* **Never raise.** Forge may not be running (there is no Java in a plain
  Python checkout); the bridge degrades to `status="offline"`, spools the
  actions in order and flushes them on the next successful contact.
* **Ordered and bounded.** The spool is FIFO with a hard cap; on overflow the
  oldest entries are dropped and counted (`dropped`), never silently lost.
* **Idempotent.** Actions carry `actionId = tableId:seq`; Forge ignores an
  actionId it has already applied, so a retried batch cannot double-apply.
* **Truthful.** Every action mirrors one observed physical fact (a card was
  put down, a card was turned sideways). The bridge never invents game
  actions and never guesses a zone that vision cannot see.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Iterable

__all__ = [
    "ForgeTableBridge",
    "EVENT_TO_ACTION",
    "TRACKING_ONLY_EVENTS",
    "FORGE_AGENT_DEFAULT",
]

FORGE_AGENT_DEFAULT = "http://127.0.0.1:8791"

#: physical event type → Forge table action that mirrors it.
#:
#: The action vocabulary is deliberately small and physical: Forge decides what
#: a card *means* (a Forest is a land, a Vampire has a triggered ability), the
#: bridge only says what the human *did* on the table.
EVENT_TO_ACTION: dict[str, str] = {
    # objects appearing / moving between physical zones
    "card_registered": "put_onto_battlefield",
    "entered_battlefield": "put_onto_battlefield",
    "left_battlefield": "move_zone",
    "zone_change": "move_zone",
    "token_created": "put_onto_battlefield",
    "token_removed": "remove_permanent",
    # tapping
    "card_tapped": "set_tapped",
    "card_untapped": "set_tapped",
    # counters / stats / abilities
    "counter_added": "set_counters",
    "counter_removed": "set_counters",
    "damage_marked": "mark_damage",
    "damage_cleared": "clear_damage",
    "pt_set": "set_pt",
    "ability_granted": "grant_ability",
    "ability_removed": "revoke_ability",
    "status_set": "set_status",
    # players
    "life_changed": "set_life",
    "poison_changed": "set_poison",
    "commander_damage": "set_commander_damage",
    # combat declarations made physically (attackers turned sideways)
    "attack_declared": "declare_attacker",
    "attack_retracted": "retract_attacker",
    "block_declared": "declare_blocker",
    "block_retracted": "retract_blocker",
    "combat_damage": "deal_combat_damage",
    # stack — a physical card cast from hand
    "spell_cast": "cast_spell",
    "spell_resolved": "resolve_spell",
    # turn flow (the human's turn is driven by the physical table)
    "turn_advanced": "advance_turn",
    "phase_changed": "set_phase",
    "untap_step": "untap_all",
    # correction: the player fixed a wrong identity in the UI
    "identity_corrected": "correct_identity",
}

#: Events that only exist inside the physical tracker/mirror and must never be
#: forwarded: they describe vision bookkeeping, not game facts.
TRACKING_ONLY_EVENTS = frozenset({
    "moved", "appeared", "disappeared", "reappeared",
    "temp_buff", "temp_debuff",            # Forge recomputes from real effects
    "combat_begin", "combat_end",          # Forge owns combat structure
    "registration_started", "registration_failed",
    "candidate_rejected",
})


def _camel_to_snake(key: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in key)


def _field(obj: Any, *keys: str, default: Any = None) -> Any:
    """Read the first present of ``keys`` from a dict or an object.

    Dicts use camelCase (the UI/JSON shape), objects use snake_case (the
    dataclasses in ``physical/state.py``), and every key is also tried with the
    other convention plus the project's ``*_code`` variants, so callers can say
    ``_field(obj, "set")`` and get ``set_code`` off a PhysicalObject.
    """
    if obj is None:
        return default
    for key in keys:
        names = [key, _camel_to_snake(key)]
        if key == "set":
            names += ["set_code", "setCode", "set_code_upper"]
        elif key in ("power", "toughness"):
            names += ["base_" + key, "base" + key.capitalize()]
        elif key == "collectorNumber":
            names += ["collector_number"]
        elif key == "isToken":
            names += ["is_token"]
        elif key == "trackingId":
            names += ["tracking_id"]
        for name in names:
            if isinstance(obj, dict):
                if name in obj and obj[name] is not None:
                    return obj[name]
                continue
            if hasattr(obj, name):
                value = getattr(obj, name)
                if callable(value) and not isinstance(value, type):
                    value = value()
                if value is not None:
                    return value
    return default


def _iter_objects(state: Any) -> list[Any]:
    """Objects on a PhysicalGameState (dict form or object form)."""
    objects = state.get("objects") if isinstance(state, dict) \
        else getattr(state, "objects", None)
    if isinstance(objects, dict):
        return list(objects.values())
    return list(objects or [])


def _normalise(event: Any) -> dict[str, Any]:
    """Accept a physical Event, a dict, or a to_dict() result."""
    if hasattr(event, "to_dict"):
        data = event.to_dict()
    elif hasattr(event, "type"):            # Event dataclass
        data = {"type": event.type, "origin": event.origin,
                "payload": getattr(event, "payload", {}) or {},
                "confidence": getattr(event, "confidence", 1.0),
                "turn": getattr(event, "turn", 0)}
    else:
        data = dict(event)
    data.setdefault("payload", {})
    return data


class ForgeTableBridge:
    """Mirrors physical-table events into the running Forge game.

    Parameters
    ----------
    base_url:
        Forge agent base URL (default ``http://127.0.0.1:8791``).
    seat:
        Which Forge seat the physical table drives (default ``"player"``).
    table_id:
        Stable id of this physical table; part of every ``actionId`` so a
        replayed batch is ignored by Forge.
    timeout:
        Per-request timeout in seconds. Keep it short: the table must never
        stall because Forge is busy thinking.
    spool_limit:
        Maximum number of unsent actions kept while Forge is unreachable.
    """

    def __init__(self, base_url: str = FORGE_AGENT_DEFAULT,
                 seat: str = "player",
                 table_id: str = "physical",
                 timeout: float = 2.0,
                 spool_limit: int = 256,
                 enabled: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.seat = seat
        self.table_id = table_id
        self.timeout = timeout
        self.spool_limit = max(1, int(spool_limit))
        self.enabled = enabled

        self._seq = 0
        self._spool: deque[dict[str, Any]] = deque()
        self.applied = 0
        self.rejected = 0
        self.dropped = 0
        self.sent_batches = 0
        self.last_error: str | None = None
        self.last_contact: float | None = None
        self.last_http_status: int | None = None
        self.last_action: dict[str, Any] | None = None
        # card identity seen by the table, used to resolve "this permanent"
        self.tracks: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # status / introspection
    # ------------------------------------------------------------------
    def status(self, probe: bool = False) -> dict[str, Any]:
        """Bridge state for the UI. ``probe=True`` pings the agent first."""
        if probe:
            self.health()
        return {
            "enabled": self.enabled,
            "url": self.base_url,
            "seat": self.seat,
            "tableId": self.table_id,
            "connected": self.last_error is None and self.last_contact is not None,
            "pending": len(self._spool),
            "applied": self.applied,
            "rejected": self.rejected,
            "dropped": self.dropped,
            "batches": self.sent_batches,
            "lastError": self.last_error,
            "lastContact": self.last_contact,
            "lastAction": self.last_action,
        }

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _request(self, path: str, payload: dict[str, Any] | None = None,
                 method: str = "GET") -> dict[str, Any] | None:
        if not self.enabled:
            self.last_error = "bridge disabled"
            return None
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                self.last_contact = time.time()
                self.last_error = None
                self.last_http_status = resp.status
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as exc:       # reachable but unhappy
            self.last_contact = time.time()
            self.last_http_status = exc.code
            self.last_error = f"http {exc.code} {path}"
            body = None
            try:
                raw = exc.read().decode("utf-8")
                body = json.loads(raw) if raw.strip() else None
            except Exception:                        # noqa: BLE001 - best effort
                body = None
            if exc.code >= 500 or exc.code == 409:
                # Forge is not ready / has no seat yet: the action is NOT
                # accepted, keep it spooled for a later retry.
                return None
            return body or {}                       # 4xx: honest refusal
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def health(self) -> dict[str, Any] | None:
        """Forge agent health, or ``None`` when unreachable."""
        return self._request("/health")

    def observation(self) -> dict[str, Any] | None:
        """The fly's structured view of the real Forge game state."""
        return self._request("/observation")

    def remote_result(self) -> dict[str, Any] | None:
        return self._request("/result")

    # ------------------------------------------------------------------
    # event translation
    # ------------------------------------------------------------------
    def translate(self, event: Any) -> dict[str, Any] | None:
        """Convert one physical event into a Forge table action.

        Returns ``None`` for tracking-only events (and for events the bridge
        cannot express physically — those are counted as skipped).
        """
        data = _normalise(event)
        etype = str(data.get("type", ""))
        action = EVENT_TO_ACTION.get(etype)
        if action is None:
            return None
        payload = dict(data.get("payload") or {})
        tracking_id = str(payload.get("trackingId")
                          or payload.get("tracking_id") or "")
        known = self.tracks.get(tracking_id, {}) if tracking_id else {}

        self._seq += 1
        out: dict[str, Any] = {
            "actionId": f"{self.table_id}:{self._seq}",
            "action": action,
            "physicalEvent": etype,
            "origin": data.get("origin", "vision"),
            "confidence": float(data.get("confidence", 1.0) or 0.0),
            "turn": int(data.get("turn", 0) or 0),
            "trackingId": tracking_id,
            "at": float(data.get("timestamp", time.time())),
        }

        # identity: set+collector when the recogniser knows it, else the name
        set_code = payload.get("set") or known.get("set")
        collector = payload.get("collectorNumber") or known.get("collectorNumber")
        name = payload.get("name") or known.get("name")
        if set_code and collector:
            out["cardKey"] = f"{set_code}:{collector}"
        if name:
            out["name"] = name
        if payload.get("oracleId"):
            out["oracleId"] = payload["oracleId"]

        if action == "put_onto_battlefield":
            out["zone"] = payload.get("to") or "battlefield"
            out["token"] = bool(payload.get("isToken")
                                or etype == "token_created")
            if out["token"]:
                out["name"] = payload.get("tokenType") or payload.get("name", "Token")
                out["tokenType"] = payload.get("tokenType", out["name"])
                out["count"] = int(payload.get("count", 1) or 1)
                out.setdefault("power", payload.get("power"))
                out.setdefault("toughness", payload.get("toughness"))
            out["tapped"] = bool(payload.get("tapped", False))
            if payload.get("summoningSick") is not None:
                out["summoningSick"] = bool(payload["summoningSick"])
            out["controller"] = payload.get("controller", self.seat)
            if payload.get("cardType"):
                out["cardType"] = payload["cardType"]
            if payload.get("counters"):
                out["counters"] = dict(payload["counters"])
        elif action == "set_tapped":
            out["tapped"] = etype == "card_tapped"
        elif action == "move_zone":
            out["to"] = payload.get("to") or payload.get("zone") or "graveyard"
            out["from"] = payload.get("from", "battlefield")
        elif action == "remove_permanent":
            out["zone"] = payload.get("from", "battlefield")
        elif action == "set_counters":
            out["counter"] = payload.get("counter") or payload.get("counterType") \
                or "+1/+1"
            amount = int(payload.get("amount", 1) or 1)
            out["amount"] = amount if etype == "counter_added" else -amount
        elif action == "mark_damage":
            out["amount"] = int(payload.get("amount", payload.get("damage", 1)) or 1)
            out["source"] = payload.get("source")
            out["deathtouch"] = bool(payload.get("deathtouch", False))
        elif action == "set_pt":
            out["power"] = payload.get("power")
            out["toughness"] = payload.get("toughness")
            out["untilEndOfTurn"] = bool(payload.get("temporary", False))
        elif action in ("grant_ability", "revoke_ability"):
            out["ability"] = payload.get("ability")
            out["duration"] = payload.get("duration", "permanent")
        elif action == "set_status":
            out["flags"] = sorted(payload.get("flags") or
                                  payload.get("statusFlags") or [])
        elif action == "set_life":
            out["life"] = payload.get("life")
            out["delta"] = payload.get("delta")
            out["player"] = payload.get("player", self.seat)
        elif action == "set_poison":
            out["amount"] = int(payload.get("poison", payload.get("amount", 0)) or 0)
        elif action == "set_commander_damage":
            out["amount"] = int(payload.get("damage",
                                            payload.get("amount", 0)) or 0)
            out["source"] = payload.get("source") or payload.get("commander")
        elif action in ("declare_attacker", "retract_attacker"):
            out["target"] = payload.get("target", "fly")
        elif action in ("declare_blocker", "retract_blocker"):
            out["blocking"] = payload.get("blocking") or payload.get("target")
        elif action == "deal_combat_damage":
            out["amount"] = int(payload.get("amount", 0) or 0)
            out["target"] = payload.get("target")
        elif action == "cast_spell":
            out["targets"] = payload.get("targets") or []
            out["from"] = payload.get("from", "hand")
        elif action == "advance_turn":
            out["turn"] = payload.get("turn")
            out["activePlayer"] = payload.get("activePlayer", self.seat)
        elif action == "set_phase":
            out["phase"] = payload.get("phase")
        return out

    def remember_track(self, tracking_id: str, *, name: str | None = None,
                       set_code: str | None = None,
                       collector_number: str | None = None,
                       oracle_id: str | None = None) -> None:
        """Remember the identity behind a tracking id (from the recogniser)."""
        if not tracking_id:
            return
        entry = self.tracks.setdefault(tracking_id, {})
        if name:
            entry["name"] = name
        if set_code:
            entry["set"] = set_code
        if collector_number:
            entry["collectorNumber"] = collector_number
        if oracle_id:
            entry["oracleId"] = oracle_id

    def forget_track(self, tracking_id: str) -> None:
        self.tracks.pop(tracking_id, None)

    # ------------------------------------------------------------------
    # sending
    # ------------------------------------------------------------------
    def send(self, events: Iterable[Any]) -> dict[str, Any]:
        """Translate + send a batch of physical events (spool on failure)."""
        actions: list[dict[str, Any]] = []
        for event in events:
            try:
                action = self.translate(event)
            except Exception as exc:                 # noqa: BLE001 - never raise
                self.last_error = f"translate: {exc}"
                continue
            if action is not None:
                actions.append(action)
        if not actions:
            return {"status": "noop", "queued": 0, "applied": 0,
                    "spooled": len(self._spool)}
        self._spool.extend(actions)
        self._trim_spool()
        return self.flush()

    def queue_stats(self) -> dict[str, Any] | None:
        """Forge-side counters for queued table actions (None if offline)."""
        return self._request("/table/queue")

    def flush(self) -> dict[str, Any]:
        """Try to deliver everything spooled, oldest first."""
        if not self._spool:
            return {"status": "noop", "queued": 0, "applied": 0, "rejected": 0,
                    "spooled": 0}
        batch = list(self._spool)
        response = self._request("/table/events",
                                 {"tableId": self.table_id, "seat": self.seat,
                                  "events": batch},
                                 method="POST")
        queued = len(batch)
        if response is None:
            return {"status": "offline", "queued": queued, "applied": 0,
                    "spooled": len(self._spool), "error": self.last_error}
        if self.last_http_status is not None and 400 <= self.last_http_status < 500:
            # the agent refused the batch itself (bad payload / no seat): it
            # will never be accepted, so drop it and say so instead of
            # retrying it forever
            drops = len(self._spool)
            self._spool.clear()
            self.rejected += drops
            self.sent_batches += 1
            return {"status": "rejected", "queued": queued, "applied": 0,
                    "rejected": drops, "spooled": 0,
                    "error": self.last_error}
        if str(response.get("status", "")).lower() == "queued"                 or "accepted" in response:
            # Asynchronous Forge agent: the batch is now owned by the game
            # thread. Take the Forge-side verdict from /table/queue, which is
            # the authoritative applied/rejected accounting.
            for _ in range(min(queued, len(self._spool))):
                self._spool.popleft()
            self.sent_batches += 1
            if batch:
                self.last_action = batch[-1]
            stats = self.queue_stats() or {}
            remote_applied = int(stats.get("applied", 0) or 0)
            remote_rejected = int(stats.get("rejected", 0) or 0)
            if remote_applied:
                self.applied = remote_applied
            if remote_rejected:
                self.rejected = remote_rejected
            return {"status": "queued", "queued": queued,
                    "accepted": int(response.get("accepted", queued) or 0),
                    "duplicates": int(response.get("duplicates", 0) or 0),
                    "applied": remote_applied, "rejected": remote_rejected,
                    "pending": int(stats.get("pending", 0) or 0),
                    "spooled": len(self._spool),
                    "recent": (stats.get("recent") or [])[:20]}
        results = response.get("results") or []
        rejected = [r for r in results
                    if str(r.get("status", "")).lower() in ("rejected", "error")]
        applied = int(response.get("applied", len(results) - len(rejected)))
        # drop the delivered prefix: the agent answers per action, in order
        delivered = applied + len(rejected)
        for _ in range(min(delivered, len(self._spool))):
            self._spool.popleft()
        self.applied += applied
        self.rejected += len(rejected)
        self.sent_batches += 1
        if batch:
            self.last_action = batch[-1]
        status = "ok" if not rejected else ("partial" if applied else "rejected")
        return {"status": status, "queued": queued, "applied": applied,
                "rejected": len(rejected), "spooled": len(self._spool),
                "results": results[:20]}

    def sync_state(self, state: Any) -> dict[str, Any]:
        """Idempotent full resync of the player's physical battlefield.

        Used after a reconnect or when the player presses *Sync to Forge*: the
        mirror is re-stated wholesale, so Forge can drop anything it holds for
        that seat which the table no longer sees.
        """
        battlefield: list[dict[str, Any]] = []
        try:
            for obj in _iter_objects(state):
                if _field(obj, "zone") != "battlefield":
                    continue
                if _field(obj, "controller") != self.seat:
                    continue
                entry: dict[str, Any] = {
                    "trackingId": _field(obj, "trackingId"),
                    "name": _field(obj, "name"),
                    "set": _field(obj, "set"),
                    "collectorNumber": _field(obj, "collectorNumber"),
                    "tapped": bool(_field(obj, "tapped", default=False)),
                    "counters": dict(_field(obj, "counters", default={}) or {}),
                    "isToken": bool(_field(obj, "isToken", default=False)),
                    "power": _field(obj, "power"),
                    "toughness": _field(obj, "toughness"),
                }
                if entry["set"] and entry["collectorNumber"]:
                    entry["cardKey"] = f"{entry['set']}:{entry['collectorNumber']}"
                battlefield.append({k: v for k, v in entry.items()
                                    if v is not None or k == "tapped"})
            players = _field(state, "players", default={})
            if isinstance(players, dict):
                life = _field(players.get(self.seat), "life")
            else:
                life = None
        except Exception as exc:                     # noqa: BLE001 - never raise
            self.last_error = f"sync_state: {exc}"
            return {"status": "error", "error": self.last_error}

        payload = {"tableId": self.table_id, "seat": self.seat,
                   "battlefield": battlefield}
        if life is not None:
            payload["life"] = life
        response = self._request("/table/state", payload, method="POST")
        if response is None:
            return {"status": "offline", "battlefield": len(battlefield),
                    "error": self.last_error}
        return {"status": "ok", "battlefield": len(battlefield),
                "life": life, "response": response}

    # ------------------------------------------------------------------
    def _trim_spool(self) -> None:
        while len(self._spool) > self.spool_limit:
            self._spool.popleft()
            self.dropped += 1
