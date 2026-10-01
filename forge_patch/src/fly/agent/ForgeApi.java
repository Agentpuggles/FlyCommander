package fly.agent;

import forge.StaticData;
import forge.game.Game;
import forge.game.GameEntityCounterTable;
import forge.game.card.CardDamageTable;
import forge.game.card.CounterEnumType;
import forge.game.spellability.SpellAbility;
import forge.item.IPaperCard;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Map;

/**
 * Legacy state-edit helpers pinned to Forge 2.0.15, commit
 * 4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676.
 * These are NOT a casting/legality API. /table/events and /table/state remain
 * disabled; game actions must be driven by Forge controllers/effects.
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

    /** Legacy counter delta. Never feed a negative count to addCounter.
     * A real resolving effect must supply its cause; null denotes a state edit.
     * Source player is explicit rather than inferred from target control.
     */
    static String addCounter(Player source, Card card, String counterName, int amount, SpellAbility cause) {
        try {
            CounterType type = counterType(counterName);
            if (type == null) {
                return "unknown counter type: " + counterName;
            }
            if (amount == Integer.MIN_VALUE) return "counter delta out of range";
            if (amount > 0) {
                GameEntityCounterTable table = new GameEntityCounterTable();
                card.addCounter(type, amount, source, table);
                // addCounter only queues the addition. This applies replacements
                // and fires the aggregate counter triggers, as CountersPutEffect does.
                table.replaceCounterEffect(card.getGame(), cause);
            } else if (amount < 0) {
                if (!card.canRemoveCounters(type)) return "Forge forbids removing these counters";
                card.subtractCounter(type, -amount, source);
            }
            return null;
        } catch (Throwable t) {
            return "addCounter failed: " + t;
        }
    }

    /** Legacy life-total edit; not a life-payment or spell-legality check. */
    static String setLife(Player seat, int life) {
        try {
            seat.setLife(life, null);
            return null;
        } catch (Throwable t) {
            return "setLife failed: " + t;
        }
    }

    /**
     * Delegate an entire Forge-prepared damage batch, preserving simultaneity.
     * Source keys must be the source LKI cards (Game.getChangeZoneLKIInfo), as
     * DamageDealEffect does. Deathtouch/lifelink come from those cards, not flags.
     * Never call this with camera observations or replay an already-resolved batch.
     * Forge's resolution/priority loop owns the subsequent state-based actions;
     * checking them here would be premature during a multi-part spell resolution.
     */
    static void dealDamage(Game game, boolean combat, CardDamageTable damage,
                           CardDamageTable prevention, GameEntityCounterTable counters,
                           SpellAbility cause) {
        game.getAction().dealDamage(combat, damage, prevention, counters, cause);
    }

    /** Resolve a real printed card, NOT a token recipe, before constructing it. */
    static Card cardByName(String name, Player owner) {
        if (name == null || name.isBlank() || owner == null) return null;
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) paper = StaticData.instance().getVariantCards().getCard(name);
        return paper == null ? null : forge.game.card.CardFactory.getCard(paper, owner, owner.getGame());
    }

    private static CounterType counterType(String name) {
        if (name == null || name.isEmpty()) {
            return CounterEnumType.P1P1;
        }
        String key = name.trim();
        if ("+1/+1".equals(key) || "1/1".equals(key) || "+1/+1 counter".equals(key)) {
            return CounterEnumType.P1P1;
        }
        if ("-1/-1".equals(key) || "-1/-1 counter".equals(key)) {
            return CounterEnumType.M1M1;
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
