package fly.agent;

import forge.game.GameAction;
import forge.game.GameEntity;
import forge.game.GameEntityCounterTable;
import forge.game.card.Card;
import forge.game.card.CardDamageTable;
import forge.game.card.CardFactory;
import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.item.IPaperCard;

/** Signature smoke test against the actual built Forge jar, not stub classes.
 * Does not verify replacement/prevention effects or gameplay.
 */
public final class ForgeApiContractTest {
    public static void main(String[] args) throws Exception {
        GameEntity.class.getMethod("addCounter", CounterType.class, int.class,
                Player.class, GameEntityCounterTable.class);
        Card.class.getMethod("subtractCounter", CounterType.class, int.class, Player.class);
        GameEntityCounterTable.class.getMethod("replaceCounterEffect", forge.game.Game.class, SpellAbility.class);
        GameAction.class.getMethod("dealDamage", boolean.class, CardDamageTable.class,
                CardDamageTable.class, GameEntityCounterTable.class, SpellAbility.class);
        CardFactory.class.getMethod("getCard", IPaperCard.class, Player.class, forge.game.Game.class);
        Card.class.getMethod("getPaperCard");
        IPaperCard.class.getMethod("getEdition");
        IPaperCard.class.getMethod("getCollectorNumber");
        var parser = ForgeApi.class.getDeclaredMethod("counterType", String.class);
        parser.setAccessible(true);
        if (parser.invoke(null, "+1/+1") != CounterEnumType.P1P1
                || parser.invoke(null, "-1/-1") != CounterEnumType.M1M1)
            throw new AssertionError("counter mapping");
        System.out.println("Forge API signatures and counter mappings passed; gameplay NOT tested");
    }
}
