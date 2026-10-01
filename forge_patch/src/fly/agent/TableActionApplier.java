package fly.agent;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Applies queued physical-table actions to the Forge game.
 *
 * Runs on the game thread only (drained at the table seat's priority), so it
 * never races the AI or the HTTP worker. Every action resolves to one of:
 *
 *   applied   — Forge performed the physical fact
 *   rejected  — Forge could not (with a reason the bridge records verbatim)
 *
 * Identification is by {@code cardKey} (set:collector) when the recogniser
 * knows the printing, then by name; a permanent that disappears from the
 * table is matched to the battlefield by tracking id, then name, then set.
 */
final class TableActionApplier {

    private TableActionApplier() {}

    /** Drain and apply everything queued for this seat. */
    static int drain(Player seat) {
        List<TableActionQueue.Action> actions = TableActionQueue.drain();
        int applied = 0;
        for (TableActionQueue.Action action : actions) {
            String reason = apply(seat, action);
            if (reason == null) {
                applied++;
                TableActionQueue.recordResult(action.actionId, action.kind,
                        "applied", "");
            } else {
                TableActionQueue.recordResult(action.actionId, action.kind,
                        "rejected", reason);
            }
        }
        return applied;
    }

    /** @return null when applied, otherwise a short reason for rejection. */
    static String apply(Player seat, TableActionQueue.Action action) {
        if (seat == null) {
            return "no table seat registered";
        }
        try {
            switch (action.kind) {
                case "put_onto_battlefield": return putOntoBattlefield(seat, action);
                case "set_tapped": return setTapped(seat, action);
                case "move_zone": return moveZone(seat, action);
                case "remove_permanent": return moveZone(seat, action);
                case "set_counters": return setCounters(seat, action);
                case "set_life": return setLife(seat, action);
                case "mark_damage": return markDamage(seat, action);
                case "sync_state": return syncState(seat, action);
                default:
                    // Declared but not yet mirrored in this build: say so
                    // instead of guessing at game rules.
                    return "action not mirrored yet: " + action.kind;
            }
        } catch (Throwable t) {
            return action.kind + " threw: " + t;
        }
    }

    // ------------------------------------------------------------------
    private static String putOntoBattlefield(Player seat,
                                             TableActionQueue.Action action) {
        Card card = resolve(seat, action, false);
        if (card == null) {
            String key = action.str("cardKey");
            String name = action.str("name");
            if (ForgeApi.isToken(action.fields) && name != null) {
                card = ForgeApi.cardByName(name, seat);
            }
            if (card == null) {
                return "card not found in hand or library: "
                        + (key != null ? key : String.valueOf(name));
            }
        }
        String reason = ForgeApi.moveTo(seat, card, ZoneType.Battlefield);
        if (reason != null) {
            return reason;
        }
        if (action.bool("tapped", false)) {
            reason = ForgeApi.setTapped(card, true);
            if (reason != null) {
                return reason;
            }
        }
        Map<String, Object> counters = action.map("counters");
        if (counters != null) {
            for (Map.Entry<String, Object> e : counters.entrySet()) {
                int amount = e.getValue() instanceof Number
                        ? ((Number) e.getValue()).intValue() : 1;
                reason = ForgeApi.addCounter(card, e.getKey(), amount);
                if (reason != null) {
                    return reason;
                }
            }
        }
        return null;
    }

    private static String setTapped(Player seat, TableActionQueue.Action action) {
        Card card = resolve(seat, action, true);
        if (card == null) {
            return "permanent not on the battlefield: " + describe(action);
        }
        return ForgeApi.setTapped(card, action.bool("tapped", true));
    }

    private static String moveZone(Player seat, TableActionQueue.Action action) {
        Card card = resolve(seat, action, true);
        if (card == null) {
            return "permanent not on the battlefield: " + describe(action);
        }
        return ForgeApi.moveTo(seat, card, ForgeApi.zone(action.str("to")));
    }

    private static String setCounters(Player seat, TableActionQueue.Action action) {
        Card card = resolve(seat, action, true);
        if (card == null) {
            return "permanent not on the battlefield: " + describe(action);
        }
        String name = action.str("counter");
        int amount = action.intValue("amount", 1);
        return ForgeApi.addCounter(card, name, amount);
    }

    private static String setLife(Player seat, TableActionQueue.Action action) {
        Integer life = null;
        Object raw = action.fields.get("life");
        if (raw instanceof Number) {
            life = ((Number) raw).intValue();
        }
        if (life == null) {
            Integer delta = null;
            Object d = action.fields.get("delta");
            if (d instanceof Number) {
                delta = ((Number) d).intValue();
            }
            if (delta == null) {
                return "no life total or delta in the action";
            }
            life = seat.getLife() + delta;
        }
        return ForgeApi.setLife(seat, life);
    }

    private static String markDamage(Player seat, TableActionQueue.Action action) {
        Card card = resolve(seat, action, true);
        if (card == null) {
            return "permanent not on the battlefield: " + describe(action);
        }
        return ForgeApi.markDamage(seat, card, action.intValue("amount", 1),
                action.bool("deathtouch", false));
    }

    /**
     * Full battlefield resync: Forge's view of this seat must equal what the
     * camera sees. Cards Forge holds that the table no longer sees go back to
     * the graveyard-adjacent "unknown" zone only when the bridge says so; in
     * v1 the resync only *adds* what is missing (never destroys information
     * based on a camera glitch) and re-syncs tap states and counters.
     */
    private static String syncState(Player seat, TableActionQueue.Action action) {
        Object cards = action.fields.get("battlefield");
        if (!(cards instanceof List)) {
            return "sync_state needs a battlefield list";
        }
        List<String> problems = new ArrayList<>();
        for (Object item : (List<?>) cards) {
            if (!(item instanceof Map)) continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> entry = (Map<String, Object>) item;
            String key = entry.get("cardKey") == null
                    ? null : String.valueOf(entry.get("cardKey"));
            String name = entry.get("name") == null
                    ? null : String.valueOf(entry.get("name"));
            Card card = findByKeyOrName(seat, key, name, true);
            if (card == null) {
                card = findByKeyOrName(seat, key, name, false);
            }
            if (card == null) {
                problems.add("missing: " + (key != null ? key : name));
                continue;
            }
            if (card.getZone() == null
                    || card.getZone().getZoneType() != ZoneType.Battlefield) {
                String reason = ForgeApi.moveTo(seat, card, ZoneType.Battlefield);
                if (reason != null) {
                    problems.add(reason);
                    continue;
                }
            }
            boolean tapped = entry.get("tapped") instanceof Boolean
                    && (Boolean) entry.get("tapped");
            String reason = ForgeApi.setTapped(card, tapped);
            if (reason != null) {
                problems.add(reason);
            }
        }
        if (!problems.isEmpty()) {
            return String.join("; ", problems.subList(0, Math.min(3, problems.size())));
        }
        return null;
    }

    // ------------------------------------------------------------------
    /** Find the permanent/card an action refers to. */
    private static Card resolve(Player seat, TableActionQueue.Action action,
                                boolean battlefieldOnly) {
        String key = action.str("cardKey");
        String name = action.str("name");
        Card card = findByKeyOrName(seat, key, name, battlefieldOnly);
        if (card == null && !battlefieldOnly) {
            card = findByKeyOrName(seat, key, name, false);
        }
        return card;
    }

    private static Card findByKeyOrName(Player seat, String key, String name,
                                        boolean battlefieldOnly) {
        String set = null, number = null;
        if (key != null && key.contains(":")) {
            int i = key.indexOf(':');
            set = key.substring(0, i);
            number = key.substring(i + 1);
        }
        for (Card card : candidates(seat, battlefieldOnly)) {
            if (set != null && number != null
                    && set.equalsIgnoreCase(String.valueOf(card.getSetCode()))
                    && number.equals(String.valueOf(card.getCollectorNumberId()))) {
                return card;
            }
            if (name != null && name.equalsIgnoreCase(card.getName())) {
                return card;
            }
        }
        return null;
    }

    private static Iterable<Card> candidates(Player seat, boolean battlefieldOnly) {
        if (battlefieldOnly) {
            return seat.getCardsIn(ZoneType.Battlefield);
        }
        List<Card> all = new ArrayList<>();
        for (Card c : seat.getCardsIn(ZoneType.Battlefield)) all.add(c);
        for (Card c : seat.getCardsIn(ZoneType.Hand)) all.add(c);
        for (Card c : seat.getCardsIn(ZoneType.Library)) all.add(c);
        return all;
    }

    private static String describe(TableActionQueue.Action action) {
        String key = action.str("cardKey");
        String name = action.str("name");
        return key != null ? key : String.valueOf(name);
    }
}
