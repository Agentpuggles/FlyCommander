package fly.agent;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * Builds a structured JSON observation of the current game state from the
 * perspective of one player (the fly). Kept deliberately terse: the Python
 * side computes rewards from consecutive observations.
 */
public final class GameObserver {
    private GameObserver() {}

    /** Observation from the fly player's seat. */
    public static String snapshotJson(Game game, Player fly) {
        StringBuilder sb = new StringBuilder(8192);
        sb.append('{');
        putInt(sb, "turn", game.getPhaseHandler().getTurn());
        putStr(sb, "phase", game.getPhaseHandler().getPhase().name());
        putBool(sb, "gameOver", game.isGameOver());

        // --- players -------------------------------------------------
        sb.append("\"players\":[");
        boolean first = true;
        for (Player p : game.getPlayers()) {
            if (!first) sb.append(',');
            first = false;
            boolean isFly = (p == fly);
            sb.append('{');
            putStr(sb, "name", p.getName());
            putBool(sb, "isFly", isFly);
            putInt(sb, "life", p.getLife());
            putInt(sb, "poison", p.getPoisonCounters());
            putInt(sb, "hand", countZone(p, ZoneType.Hand));
            putInt(sb, "library", countZone(p, ZoneType.Library));
            putInt(sb, "graveyard", countZone(p, ZoneType.Graveyard));
            putInt(sb, "exile", countZone(p, ZoneType.Exile));
            putInt(sb, "commandZone", countZone(p, ZoneType.Command));
            putInt(sb, "creatures", countCreatures(p));
            putInt(sb, "lands", countLands(p));

            // commander damage taken (sum across commanders)
            int cmdDmg = 0;
            for (Player opp : p.getOpponents()) {
                for (Card c : opp.getCardsIn(ZoneType.Command)) {
                    cmdDmg += p.getCommanderDamage(c);
                }
            }
            putInt(sb, "commanderDamage", cmdDmg);

            // top battlefield cards (bounded, for identity features)
            sb.append("\"board\":[");
            List<Card> board = boardCards(p);
            int n = Math.min(24, board.size());
            for (int i = 0; i < n; i++) {
                if (i > 0) sb.append(',');
                Card c = board.get(i);
                sb.append("{\"n\":\"").append(esc(c.getName()))
                  .append("\",\"t\":").append(c.isTapped() ? 1 : 0)
                  .append(",\"ctl\":\"").append(esc(c.getController().getName()))
                  .append("\"}");
            }
            sb.append(']');
            sb.append('}');
        }
        sb.append(']');

        // --- stack + combat ------------------------------------------
        sb.append("\"stackSize\":").append(game.getStack().size()).append(',');
        Combat combat = game.getCombat();
        int attackers = 0, blockers = 0;
        if (combat != null) {
            attackers = combat.getAttackers().size();
            blockers = combat.getAllBlockers().size();
        }
        putInt(sb, "attackers", attackers);
        putInt(sb, "blockers", blockers);

        // --- fly's playable abilities right now ----------------------
        // (enumerate over hand + battlefield; bounded to keep snapshots cheap)
        List<SpellAbility> playable = new ArrayList<>();
        List<Card> scan = new ArrayList<>();
        for (Card c : fly.getCardsIn(ZoneType.Hand)) scan.add(c);
        for (Card c : fly.getCardsIn(ZoneType.Battlefield)) scan.add(c);
        int scanned = 0;
        for (Card c : scan) {
            if (scanned++ >= 40) break;
            try {
                playable.addAll(c.getAllPossibleAbilities(fly, false));
            } catch (Throwable t) {
                // leave this card's abilities out of the snapshot
            }
        }
        List<String> cls = new ArrayList<>();
        boolean hasLand = false, hasSpell = false, hasAbility = false;
        for (SpellAbility sa : playable) {
            if (sa.isLandAbility()) {
                hasLand = true;
            } else if (sa.isSpell()) {
                hasSpell = true;
            } else {
                hasAbility = true;
            }
        }
        putBool(sb, "land", hasLand);
        putBool(sb, "spell", hasSpell);
        putBool(sb, "ability", hasAbility);
        sb.setLength(sb.length() - 1); // trailing comma from putBool
        sb.append(']');

        sb.append('}');
        return sb.toString();
    }

    // ------------------------------------------------------------------
    private static int countZone(Player p, ZoneType zone) {
        return p.getCardsIn(zone).size();
    }

    private static int countCreatures(Player p) {
        int n = 0;
        for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
            if (c.isCreature()) n++;
        }
        return n;
    }

    private static int countLands(Player p) {
        int n = 0;
        for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
            if (c.isLand()) n++;
        }
        return n;
    }

    private static List<Card> boardCards(Player p) {
        List<Card> out = new ArrayList<>();
        for (Card c : p.getCardsIn(ZoneType.Battlefield)) {
            out.add(c);
        }
        return out;
    }

    // ------------------------------------------------------------------
    private static void putInt(StringBuilder sb, String k, int v) {
        sb.append('"').append(k).append("\":").append(v).append(',');
    }

    private static void putBool(StringBuilder sb, String k, boolean v) {
        sb.append('"').append(k).append("\":").append(v).append(',');
    }

    private static void putStr(StringBuilder sb, String k, String v) {
        sb.append('"').append(k).append("\":\"").append(esc(v)).append("\",");
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "'")
                .replace("\n", " ").replace("\r", " ");
    }
}
