"""FlyCommander physical-table mode — state reconciliation engine.

Owns the authoritative `PhysicalGameState`. Vision produces *candidate*
events; this engine validates them against rules + current state and applies
what survives. Player-origin events are authoritative and always applied.

Reconciliation principles:
  - state is never rebuilt from raw frames; it is patched by events
  - identity travels with tracking IDs across zone changes and occlusion
  - low-confidence vision events that would destroy information (zone
    changes, combat declarations) require player confirmation
  - summoning sickness and temporary-modifier expiry are RULES-derived, not
    vision-derived, and are recomputed on turn advancement
"""
from __future__ import annotations

from typing import Any

from physical.events import Event, EventLog, VISION_CONFIRM_REQUIRED
from physical.state import (
    Modifier,
    PhysicalObject,
    PhysicalPlayer,
    PhysicalGameState,
    StackEntry,
    new_tracking_id,
    now,
)


class Engine:
    def __init__(self, state: PhysicalGameState | None = None,
                 log: EventLog | None = None) -> None:
        self.state = state or PhysicalGameState()
        self.log = log or EventLog()
        # pending vision events awaiting player confirmation
        self.pending: list[Event] = []

    # ------------------------------------------------------------------
    # event entry points
    # ------------------------------------------------------------------
    def apply_vision(self, event: Event) -> dict[str, Any]:
        """Apply a vision candidate. Some types require confirmation first."""
        event.validate()
        if event.type in VISION_CONFIRM_REQUIRED:
            self.pending.append(event)
            return {"status": "pending_confirmation", "event": event.type}
        return self._apply(event)

    def confirm_pending(self, index: int = 0) -> dict[str, Any]:
        """Player confirms the oldest pending vision event."""
        if not self.pending:
            return {"status": "no_pending"}
        ev = self.pending.pop(index)
        ev.origin = "player"  # confirmed → authoritative
        return self._apply(ev)

    def reject_pending(self, index: int = 0) -> dict[str, Any]:
        if not self.pending:
            return {"status": "no_pending"}
        ev = self.pending.pop(index)
        self.log.append(Event(type="candidate_rejected", origin="player",
                              payload={"rejectedType": ev.type},
                              note=f"rejected {ev.type}"))
        return {"status": "rejected"}

    def apply_player(self, etype: str, **payload: Any) -> dict[str, Any]:
        """Player correction/annotation — always authoritative."""
        ev = Event(type=etype, origin="player", payload=payload,
                   turn=self.state.turn)
        return self._apply(ev)

    # ------------------------------------------------------------------
    # core dispatcher
    # ------------------------------------------------------------------
    def _apply(self, ev: Event) -> dict[str, Any]:
        handler = getattr(self, f"_on_{ev.type}", None)
        result: dict[str, Any] = {"status": "applied", "event": ev.type}
        if handler is not None:
            outcome = handler(ev)
            if isinstance(outcome, dict):
                result.update(outcome)
        ev.turn = self.state.turn
        self.log.append(ev)
        return result

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _on_card_registered(self, ev: Event) -> dict:
        p = dict(ev.payload)
        tid = p.get("trackingId") or new_tracking_id()
        obj = PhysicalObject(
            tracking_id=tid,
            name=p.get("name", "Unknown"),
            set_code=p.get("set", ""),
            collector_number=p.get("collectorNumber", ""),
            oracle_id=p.get("oracleId", ""),
            card_type=p.get("cardType", ""),
            base_power=p.get("basePower"),
            base_toughness=p.get("baseToughness"),
            zone=p.get("zone", "battlefield"),
            controller=p.get("controller", "player"),
            owner=p.get("controller", "player"),
            confidence=p.get("confidence", 1.0),
        )
        if obj.zone == "battlefield":
            obj.entered_battlefield_turn = self.state.turn
            obj.entered_battlefield_timestamp = now()
            obj.controlled_since_turn = self.state.turn
        self.state.objects[tid] = obj
        obj._state_ref = self.state  # for rules queries (turns_started)
        if obj.zone == "graveyard":
            self.state.players[obj.controller].graveyard.append(tid)
        return {"trackingId": tid, "summoningSick": obj.summoning_sick}

    def _on_entered_battlefield(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        if obj.zone != "battlefield":
            self._remove_from_zone_lists(obj)
            obj.zone = "battlefield"
        obj.entered_battlefield_turn = self.state.turn
        obj.entered_battlefield_timestamp = now()
        obj.controlled_since_turn = self.state.turn
        obj.tapped = False  # ETB untapped by default; correct manually otherwise
        obj.attacking = obj.blocking = False
        ev.payload.setdefault("summoningSick", obj.summoning_sick)
        return {"trackingId": obj.tracking_id}

    def _on_left_battlefield(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        target = ev.payload.get("to", "graveyard")
        self.move_object(obj, target, ev)
        return {"trackingId": obj.tracking_id, "to": target}

    def _on_zone_change(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            # a zone change for an unknown object can mean it came from hand
            if ev.payload.get("to") == "battlefield" and ev.payload.get("name"):
                ev2 = Event(type="card_registered", origin=ev.origin,
                            payload=dict(ev.payload))
                self.log.append(ev2)
                self._on_card_registered(ev2)
                obj = self.state.objects.get(ev2.payload.get("trackingId", ""))
                if obj is not None:
                    return {"status": "registered_from_hand",
                            "trackingId": obj.tracking_id}
            return {"status": "unknown_object"}
        target = ev.payload.get("to", "graveyard")
        self.move_object(obj, target, ev)
        return {"trackingId": obj.tracking_id, "to": target}

    def _on_moved(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.position = tuple(ev.payload.get("position", obj.position))  # type: ignore[assignment]
        obj.moved_since_last = True
        return {"trackingId": obj.tracking_id}

    def _on_appeared(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.occluded_frames = 0
        obj.confidence = max(obj.confidence, ev.confidence)
        return {"trackingId": obj.tracking_id}

    def _on_disappeared(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.occluded_frames += int(ev.payload.get("frames", 1))
        # identity retained — occlusion is not a zone change
        return {"trackingId": obj.tracking_id,
                "occluded": obj.occluded_frames}

    # ------------------------------------------------------------------
    # tapping
    # ------------------------------------------------------------------
    def _on_card_tapped(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.orientation_deg = float(ev.payload.get("orientationDegrees",
                                                   obj.orientation_deg))
        obj.orientation_confidence = float(
            ev.payload.get("orientationConfidence", ev.confidence))
        changed = not obj.tapped
        obj.tapped = True
        return {"trackingId": obj.tracking_id, "changed": changed}

    def _on_card_untapped(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.orientation_deg = float(ev.payload.get("orientationDegrees",
                                                   obj.orientation_deg))
        obj.orientation_confidence = float(
            ev.payload.get("orientationConfidence", ev.confidence))
        changed = obj.tapped
        obj.tapped = False
        return {"trackingId": obj.tracking_id, "changed": changed}

    # ------------------------------------------------------------------
    # counters / modifiers / abilities / damage / P/T
    # ------------------------------------------------------------------
    def _on_counter_added(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        name = ev.payload.get("counter", "+1/+1")
        amount = int(ev.payload.get("amount", 1))
        obj.counters[name] = obj.counters.get(name, 0) + amount
        if obj.counters[name] <= 0:
            obj.counters.pop(name, None)
        return {"trackingId": obj.tracking_id, "counters": dict(obj.counters)}

    def _on_counter_removed(self, ev: Event) -> dict:
        ev.payload["amount"] = -abs(int(ev.payload.get("amount", 1)))
        return self._on_counter_added(ev)

    def _on_temp_buff(self, ev: Event) -> dict:
        return self._add_modifier(ev, positive=True)

    def _on_temp_debuff(self, ev: Event) -> dict:
        return self._add_modifier(ev, positive=False)

    def _add_modifier(self, ev: Event, positive: bool) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        power = abs(float(ev.payload.get("power", 0)))
        tough = abs(float(ev.payload.get("toughness", 0)))
        if not positive:
            power, tough = -power, -tough
        obj.temporary_modifiers.append(Modifier(
            power=power, toughness=tough,
            expires=ev.payload.get("expires", "end_of_turn"),
            source=ev.payload.get("source", ""),
            created_turn=self.state.turn))
        return {"trackingId": obj.tracking_id,
                "power": obj.current_power(), "toughness": obj.current_toughness()}

    def _on_ability_granted(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.granted_abilities.add(str(ev.payload.get("ability", "")).lower())
        return {"trackingId": obj.tracking_id,
                "abilities": sorted(obj.granted_abilities)}

    def _on_ability_removed(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.granted_abilities.discard(str(ev.payload.get("ability", "")).lower())
        return {"trackingId": obj.tracking_id,
                "abilities": sorted(obj.granted_abilities)}

    def _on_status_set(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        flag = str(ev.payload.get("flag", ""))
        value = bool(ev.payload.get("value", True))
        if value:
            obj.status_flags.add(flag)
        else:
            obj.status_flags.discard(flag)
        return {"trackingId": obj.tracking_id,
                "statusFlags": sorted(obj.status_flags)}

    def _on_damage_marked(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.damage_marked += float(ev.payload.get("amount", 0))
        return {"trackingId": obj.tracking_id,
                "damageMarked": obj.damage_marked}

    def _on_damage_cleared(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.damage_marked = 0.0
        return {"trackingId": obj.tracking_id, "damageMarked": 0.0}

    def _on_pt_set(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.base_power = ev.payload.get("power", obj.base_power)
        obj.base_toughness = ev.payload.get("toughness", obj.base_toughness)
        return {"trackingId": obj.tracking_id,
                "power": obj.current_power(),
                "toughness": obj.current_toughness()}

    # ------------------------------------------------------------------
    # tokens
    # ------------------------------------------------------------------
    def _on_token_created(self, ev: Event) -> dict:
        token_type = ev.payload.get("tokenType", "Token")
        power = ev.payload.get("power")
        tough = ev.payload.get("toughness")
        obj = PhysicalObject(
            name=f"{ev.payload.get('count', 1)}× {token_type}",
            is_token=True,
            token_type=token_type,
            card_type=ev.payload.get("cardType", "creature" if power else "token"),
            base_power=power, base_toughness=tough,
            zone=ev.payload.get("zone", "battlefield"),
            controller=ev.payload.get("controller", "player"),
            owner=ev.payload.get("controller", "player"),
            token_count=int(ev.payload.get("count", 1)),
        )
        if obj.zone == "battlefield":
            obj.entered_battlefield_turn = self.state.turn
            obj.entered_battlefield_timestamp = now()
            obj.controlled_since_turn = self.state.turn
        self.state.objects[obj.tracking_id] = obj
        obj._state_ref = self.state
        ev.payload["trackingId"] = obj.tracking_id
        return {"trackingId": obj.tracking_id}

    def _on_token_removed(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None or not obj.is_token:
            return {"status": "unknown_object"}
        if obj.token_count > 1:
            obj.token_count -= 1
            obj.name = f"{obj.token_count}× {obj.token_type}"
            return {"trackingId": obj.tracking_id,
                    "remaining": obj.token_count}
        self._remove_from_zone_lists(obj)
        del self.state.objects[obj.tracking_id]
        return {"status": "removed"}

    # ------------------------------------------------------------------
    # combat
    # ------------------------------------------------------------------
    def _on_combat_begin(self, ev: Event) -> dict:
        self.state.combat.active = True
        self.state.phase = "combat"
        return {}

    def _on_combat_end(self, ev: Event) -> dict:
        self.state.combat = type(self.state.combat)()  # fresh CombatState
        self.state.phase = "main2"
        # combat damage clears at end of combat; attack/block flags reset
        for o in self.state.objects.values():
            o.damage_marked = 0.0
            o.attacking = False
            o.blocking = False
            o.attack_target = None
            o.blocked_by = None
        return {}

    def _on_attack_declared(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.attacking = True
        obj.attack_target = ev.payload.get("target", "fly")
        obj.tapped = True  # attacking taps the creature (Vigilance: correct later)
        self.state.combat.active = True
        self.state.combat.attackers[obj.tracking_id] = str(obj.attack_target)
        return {"trackingId": obj.tracking_id}

    def _on_attack_retracted(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.attacking = False
        obj.attack_target = None
        self.state.combat.attackers.pop(obj.tracking_id, None)
        return {"trackingId": obj.tracking_id}

    def _on_block_declared(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        attacker_id = ev.payload.get("attacker", "")
        obj.blocking = True
        obj.blocked_by = attacker_id
        self.state.combat.blockers[obj.tracking_id] = attacker_id
        return {"trackingId": obj.tracking_id}

    def _on_block_retracted(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        obj.blocking = False
        obj.blocked_by = None
        self.state.combat.blockers.pop(obj.tracking_id, None)
        return {"trackingId": obj.tracking_id}

    def _on_combat_damage(self, ev: Event) -> dict:
        target = ev.payload.get("target", "")
        amount = float(ev.payload.get("amount", 0))
        if target in self.state.players:
            self.state.players[target].life -= int(amount)
            return {"target": target,
                    "life": self.state.players[target].life}
        return {"target": target, "life": None}

    # ------------------------------------------------------------------
    # turns / phases / players / stack
    # ------------------------------------------------------------------
    def _on_turn_advanced(self, ev: Event) -> dict:
        self.state.turn += 1
        self.state.active_player = ev.payload.get("activePlayer",
                                                  self.state.active_player)
        self.state.priority_player = self.state.active_player
        self.state.phase = "main1"
        self.state.turns_started[self.state.active_player] = self.state.turn
        self._expire_temporaries("end_of_turn")
        ev.payload["turn"] = self.state.turn
        return {"turn": self.state.turn}

    def _on_phase_changed(self, ev: Event) -> dict:
        self.state.phase = ev.payload.get("phase", self.state.phase)
        return {"phase": self.state.phase}

    def _on_untap_step(self, ev: Event) -> dict:
        n = 0
        for o in self.state.objects.values():
            if o.controller == ev.payload.get("controller",
                                              self.state.active_player):
                if o.tapped:
                    o.tapped = False
                    n += 1
        self._expire_temporaries("end_of_turn")
        return {"untapped": n}

    def _on_life_changed(self, ev: Event) -> dict:
        pid = ev.payload.get("player", "player")
        player = self.state.players.get(pid)
        if player is None:
            return {"status": "unknown_player"}
        delta = int(ev.payload.get("delta", 0))
        player.life += delta
        ev.payload["life"] = player.life
        return {"player": pid, "life": player.life}

    def _on_poison_changed(self, ev: Event) -> dict:
        pid = ev.payload.get("player", "player")
        player = self.state.players.get(pid)
        if player is None:
            return {"status": "unknown_player"}
        player.poison += int(ev.payload.get("delta", 1))
        return {"player": pid, "poison": player.poison}

    def _on_commander_damage(self, ev: Event) -> dict:
        pid = ev.payload.get("player", "player")
        player = self.state.players.get(pid)
        if player is None:
            return {"status": "unknown_player"}
        source = ev.payload.get("source", "fly-commander")
        amount = int(ev.payload.get("amount", 0))
        player.commander_damage_taken[source] = (
            player.commander_damage_taken.get(source, 0) + amount)
        player.life -= amount
        return {"player": pid,
                "commanderDamage": dict(player.commander_damage_taken)}

    def _on_spell_cast(self, ev: Event) -> dict:
        self.state.stack.append(StackEntry(
            tracking_id=ev.payload.get("trackingId", ""),
            name=ev.payload.get("name", "?"),
            controller=ev.payload.get("controller", "player")))
        return {"stackSize": len(self.state.stack)}

    def _on_spell_resolved(self, ev: Event) -> dict:
        if self.state.stack:
            self.state.stack.pop()
        return {"stackSize": len(self.state.stack)}

    def _on_identity_corrected(self, ev: Event) -> dict:
        obj = self._obj(ev)
        if obj is None:
            return {"status": "unknown_object"}
        for key, attr in (("name", "name"), ("set", "set_code"),
                          ("collectorNumber", "collector_number"),
                          ("oracleId", "oracle_id"),
                          ("cardType", "card_type"),
                          ("basePower", "base_power"),
                          ("baseToughness", "base_toughness")):
            if key in ev.payload:
                setattr(obj, attr, ev.payload[key])
        return {"trackingId": obj.tracking_id, "name": obj.name}

    # registration_started/failed are log-only; candidate_rejected too
    def _on_registration_started(self, ev: Event) -> dict:
        return {}
    def _on_registration_failed(self, ev: Event) -> dict:
        return {}
    def _on_candidate_rejected(self, ev: Event) -> dict:
        return {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _obj(self, ev: Event) -> PhysicalObject | None:
        tid = ev.payload.get("trackingId")
        return self.state.get(tid) if tid else None

    def _remove_from_zone_lists(self, obj: PhysicalObject) -> None:
        player = self.state.players.get(obj.controller)
        if player is None:
            return
        for lst in (player.graveyard,):
            if obj.tracking_id in lst:
                lst.remove(obj.tracking_id)

    def move_object(self, obj: PhysicalObject, to_zone: str, ev: Event) -> None:
        from physical.state import PUBLIC_ZONES
        self._remove_from_zone_lists(obj)
        if obj.zone == "battlefield" and to_zone != "battlefield":
            obj.attacking = False
            obj.blocking = False
            self.state.combat.attackers.pop(obj.tracking_id, None)
            self.state.combat.blockers.pop(obj.tracking_id, None)
        obj.zone = to_zone
        obj.tapped = False
        if to_zone == "graveyard":
            self.state.players[obj.controller].graveyard.append(obj.tracking_id)
        obj.moved_since_last = True

    def _expire_temporaries(self, scope: str) -> None:
        for o in self.state.objects.values():
            o.temporary_modifiers = [
                m for m in o.temporary_modifiers if m.expires != scope
            ]

    # convenience for UI -------------------------------------------------
    def snapshot(self, viewer: str = "fly") -> dict[str, Any]:
        return self.state.public_observation(viewer)
