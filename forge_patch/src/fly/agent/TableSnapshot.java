package fly.agent;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import java.util.*;

/** Built only on the game thread. Opponent private zones are counts, not identities. */
final class TableSnapshot {
    private TableSnapshot() { }

    static Map<String, Object> capture(Game game, Player human) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("mode", "forge-digital-human");
        state.put("limitations", "Forge controls all game rules. The browser human controller is under development; physical card identity, zones, library order and camera-observed actions are not synchronized.");
        state.put("paper", "This pod plays the Forge digital deck generated from your list. It does not track physical copies or a physical library. Scanning only identifies cards; it never plays or moves them.");
        state.put("turn", game.getPhaseHandler().getTurn());
        state.put("phase", String.valueOf(game.getPhaseHandler().getPhase()));
        Player active = game.getPhaseHandler().getPlayerTurn();
        state.put("activePlayer", active == null ? "" : active.getName());
        state.put("gameOver", game.isGameOver());
        state.put("humanDeck", AgentServer.resolvedHumanDeckName());
        state.put("opponentDecks", AgentServer.resolvedAiDeckNames());

        List<Object> seats = new ArrayList<>();
        for (Player p : game.getPlayers()) {
            Map<String, Object> seat = new LinkedHashMap<>();
            seat.put("name", p.getName());
            seat.put("human", p == human);
            seat.put("life", p.getLife());
            seat.put("poison", p.getPoisonCounters());
            seat.put("handCount", p.getCardsIn(ZoneType.Hand).size());
            seat.put("libraryCount", p.getCardsIn(ZoneType.Library).size());
            seat.put("graveyardCount", p.getCardsIn(ZoneType.Graveyard).size());
            seat.put("Battlefield", cards(p.getCardsIn(ZoneType.Battlefield), false));
            seat.put("Command", cards(p.getCardsIn(ZoneType.Command), false));
            seat.put("Graveyard", cards(p.getCardsIn(ZoneType.Graveyard), false));
            seat.put("Exile", cards(p.getCardsIn(ZoneType.Exile), false));
            if (p == human) {
                seat.put("Hand", cards(p.getCardsIn(ZoneType.Hand), true));
            }
            Map<String, Integer> commanderDamage = new LinkedHashMap<>();
            for (Player opponent : p.getOpponents()) {
                for (Card commander : opponent.getCardsIn(ZoneType.Command)) {
                    int damage = p.getCommanderDamage(commander);
                    if (damage > 0) commanderDamage.put(commander.getName(), damage);
                }
            }
            seat.put("commanderDamage", commanderDamage);
            seats.add(seat);
        }
        state.put("seats", seats);

        List<String> stack = new ArrayList<>();
        for (var item : game.getStack()) stack.add(item.getStackDescription());
        state.put("stack", stack);
        return state;
    }

    private static List<Object> cards(Iterable<Card> cards, boolean privateHand) {
        List<Object> result = new ArrayList<>();
        for (Card c : cards) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", c.getId());
            item.put("name", c.isFaceDown() && !privateHand ? "Face-down card" : c.getName());
            item.put("tapped", c.isTapped());
            item.put("creature", c.isCreature());
            if (c.isCreature()) {
                item.put("power", c.getNetPower());
                item.put("toughness", c.getNetToughness());
            }
            result.add(item);
        }
        return result;
    }
}
