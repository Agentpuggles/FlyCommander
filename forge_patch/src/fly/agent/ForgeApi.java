package fly.agent;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Map;

/**
 * Every Forge mutation the physical table performs, in one place.
 *
 * The physical table is an *observer with a mirror*, never a rules engine:
 * it reports "a card was put down", "a card was turned sideways", "this
 * permanent went to the graveyard". Forge decides what those facts mean.
 *
 * Patch authors: all Forge API assumptions live here and nowhere else. They
 * are pinned to Forge 2.0.15 (see research/forge/forge_internals.md); if
 * {@code make compile} fails after a Forge upgrade, the fix is one of the
 * four calls below, not a change in the engine or the vision pipeline.
 * Each method returns null on success or a short reason string on failure —
 * callers translate that into an honest per-action {@code rejected} result.
 */
final class ForgeApi {

    private ForgeApi() {}

    /** Move a card between zones; returns null on success. */
    static String moveTo(Player seat, Card card, ZoneType zone) {
        try {
            Game game = seat.getGame();
            Card moved = game.getAction().moveTo(zone, card, null, null);
            if (moved == null) {
                return "Forge refused the move to " + zone;
            }
            return null;
        } catch (Throwable t) {
            return "moveTo failed: " + t;
        }
    }

    /** Tap / untap a permanent in place. */
    static String setTapped(Card card, boolean tapped) {
        try {
            if (card.isTapped() == tapped) {
                return null;
            }
            card.setTapped(tapped);
            return null;
        } catch (Throwable t) {
            return "setTapped failed: " + t;
        }
    }

    /** Add (or remove, with a negative amount) counters. */
    static String addCounter(Card card, String counterName, int amount) {
        try {
            CounterType type = counterType(counterName);
            if (type == null) {
                return "unknown counter type: " + counterName;
            }
            card.addCounter(type, amount, true);
            return null;
        } catch (Throwable t) {
            return "addCounter failed: " + t;
        }
    }

    /** Set a player's life total (Forge clamps and fires the usual events). */
    static String setLife(Player seat, int life) {
        try {
            seat.setLife(life, null);
            return null;
        } catch (Throwable t) {
            return "setLife failed: " + t;
        }
    }

    /** Mark damage on a permanent (deathtouch / lethal handling stays Forge's). */
    static String markDamage(Player seat, Card card, int amount, boolean deathtouch) {
        try {
            card.addDamage(amount, deathtouch ? null : card.getController(), true);
            Game game = seat.getGame();
            game.getAction().checkStateEffects(true);
            return null;
        } catch (Throwable t) {
            return "markDamage failed: " + t;
        }
    }

    /** Look a card up by name in the card database (tokens, unknown printings). */
    static Card cardByName(String name, Player owner) {
        try {
            return forge.game.card.CardFactory.getCard(name, owner,
                    owner.getGame());
        } catch (Throwable t) {
            return null;
        }
    }

    private static CounterType counterType(String name) {
        if (name == null || name.isEmpty()) {
            return CounterType.P1P1;
        }
        String key = name.trim();
        if ("+1/+1".equals(key) || "1/1".equals(key) || "+1/+1 counter".equals(key)) {
            return CounterType.P1P1;
        }
        if ("-1/-1".equals(key) || "-1/-1 counter".equals(key)) {
            return CounterType.M1M1;
        }
        try {
            return CounterType.getType(key);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Zone name → Forge zone, for the table's physical vocabulary. */
    static ZoneType zone(String name) {
        if (name == null) {
            return ZoneType.Battlefield;
        }
        switch (name.toLowerCase()) {
            case "graveyard": return ZoneType.Graveyard;
            case "exile": return ZoneType.Exile;
            case "hand": return ZoneType.Hand;
            case "library": return ZoneType.Library;
            case "command":
            case "command_zone": return ZoneType.Command;
            case "stack": return ZoneType.Stack;
            default: return ZoneType.Battlefield;
        }
    }

    /** True when the action payload is a token rather than a paper card. */
    static boolean isToken(Map<String, Object> fields) {
        Object v = fields.get("token");
        return v instanceof Boolean && (Boolean) v;
    }
}
