package fly.agent;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import java.util.*;

/** Built only on the game thread. Opponent private zones are counts, not identities. */
final class TableSnapshot {
    static Map<String, Object> capture(Game game, Player human) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("mode", "experimental-assisted");
        state.put("limitations", "AI proposes plays and chooses targets, mana, modes, X, combat, discards, replacement and other effect choices. You approve plays, keep/mulligan, and pass priority. Not full manual control. Runtime verification pending.");
        state.put("paper", "Forge owns shuffle and draws. Retrieve the displayed hand from your matching paper deck; do not shuffle or draw independently. Scanner observations never play cards.");
        state.put("turn", game.getPhaseHandler().getTurn());
        state.put("phase", String.valueOf(game.getPhaseHandler().getPhase()));
        state.put("activePlayer", String.valueOf(game.getPhaseHandler().getPlayerTurn()));
        List<Object> seats = new ArrayList<>();
        for (Player p : game.getPlayers()) {
            Map<String, Object> seat = new LinkedHashMap<>();
            seat.put("name", p.getName());
            seat.put("human", p == human);
            seat.put("life", p.getLife());
            seat.put("handCount", p.getCardsIn(ZoneType.Hand).size());
            seat.put("libraryCount", p.getCardsIn(ZoneType.Library).size());
            for (ZoneType z : List.of(ZoneType.Battlefield, ZoneType.Command, ZoneType.Graveyard, ZoneType.Exile))
                seat.put(z.name(), cards(p.getCardsIn(z), false));
            if (p == human) seat.put("Hand", cards(p.getCardsIn(ZoneType.Hand), true));
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
            result.add(Map.of("id", c.getId(), "name", c.isFaceDown() && !privateHand ? "Face-down card" : c.getName(),
                    "tapped", c.isTapped()));
        }
        return result;
    }
}
