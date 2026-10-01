package fly.agent;

import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import java.util.*;

/**
 * Browser-driven paper-assist seat. The player chooses any legal card/ability;
 * Forge's AI controller handles the detailed resolution choices and combat.
 * The virtual Forge deck is authoritative; this is not physical-library sync.
 */
public class PhysicalTableController extends PlayerControllerAi {
    private final Player me;

    public PhysicalTableController(Game game, Player player, forge.LobbyPlayer lobbyPlayer) {
        super(game, player, lobbyPlayer);
        me = player;
        getAi().setUseSimulation(null);
    }

    private String ask(String kind, String text, List<String> choices) {
        String choice = HumanDecisionChannel.ask(TableSnapshot.capture(getGame(), me), kind, text, choices);
        System.out.println("[Human " + kind + "] " + choice);
        return choice;
    }

    private List<String> askOptions(String kind, String text,
                                    List<DecisionBroker.Option> options, int min, int max) {
        List<String> selected = HumanDecisionChannel.askOptions(
                TableSnapshot.capture(getGame(), me), kind, text, options, min, max);
        System.out.println("[Human " + kind + "] selected " + selected.size() + " option(s)");
        return selected;
    }

    @Override
    public Player chooseStartingPlayer(boolean isFirstGame) {
        List<Player> players = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (Player player : getGame().getPlayersInTurnOrder()) {
            String id = "seat-" + players.size();
            players.add(player);
            options.add(new DecisionBroker.Option(id, player.getName()));
        }
        List<String> selected = askOptions("starting_player",
                "Choose which player starts the four-player Commander game.", options, 1, 1);
        return players.get(Integer.parseInt(selected.get(0).substring("seat-".length())));
    }

    @Override
    public boolean mulliganKeepHand(Player firstPlayer, int cardsToReturn) {
        return ask("mulligan",
                "Keep this Forge-generated opening hand? If you keep, choose " + cardsToReturn
                        + " card(s) to put on the bottom of your library.",
                List.of("Keep hand", "Mulligan")).equals("Keep hand");
    }

    @Override
    public CardCollectionView tuckCardsViaMulligan(CardCollectionView hand, int cardsToReturn) {
        if (cardsToReturn <= 0) return CardCollection.EMPTY;
        List<Card> cards = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (Card card : hand) {
            String id = "card-" + cards.size();
            cards.add(card);
            options.add(new DecisionBroker.Option(id, card.getName()));
        }
        List<String> selected = askOptions("mulligan_bottom",
                "Select the card(s) to put on the bottom of your library.",
                options, cardsToReturn, cardsToReturn);
        CardCollection result = new CardCollection();
        for (String id : selected) {
            int index = Integer.parseInt(id.substring("card-".length()));
            result.add(cards.get(index));
        }
        return result;
    }

    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        CardCollection cards = new CardCollection();
        cards.addAll(me.getCardsIn(ZoneType.Hand));
        cards.addAll(me.getCardsIn(ZoneType.Battlefield));
        cards.addAll(me.getCardsActivatableInExternalZones(true));

        List<SpellAbility> abilities = new ArrayList<>();
        Set<SpellAbility> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Card card : cards) {
            try {
                for (SpellAbility ability : card.getAllPossibleAbilities(me, true)) {
                    if (ability != null && seen.add(ability)) abilities.add(ability);
                }
            } catch (RuntimeException ignored) {
                // One malformed/unsupported card must not hide the rest of the hand.
            }
        }

        List<DecisionBroker.Option> options = new ArrayList<>();
        options.add(new DecisionBroker.Option("pass", "Pass priority"));
        for (int i = 0; i < abilities.size(); i++) {
            SpellAbility ability = abilities.get(i);
            String kind = ability.isLandAbility() ? "Play land"
                    : ability.isSpell() ? "Cast spell" : "Activate ability";
            String details = String.valueOf(ability).replaceAll("\\s+", " ").trim();
            if (details.length() > 150) details = details.substring(0, 147) + "…";
            String label = kind + ": " + ability.getHostCard().getName();
            if (!details.isEmpty()) label += " — " + details;
            options.add(new DecisionBroker.Option("ability-" + i, label));
        }

        List<String> selected = askOptions("priority",
                abilities.isEmpty()
                        ? "No legal card plays or activated abilities are available. Pass priority."
                        : "Choose one legal card/ability or pass. Forge AI will handle detailed targets, modes and payment.",
                options, 1, 1);
        String id = selected.get(0);
        if ("pass".equals(id)) return null; // Forge 2.0.15 priority-pass sentinel.
        int index = Integer.parseInt(id.substring("ability-".length()));
        return List.of(abilities.get(index));
    }

    @Override
    public void declareAttackers(Player attacker, Combat combat) {
        ask("combat", "Forge AI will choose a legal attack plan for your creatures. Combat is AI-assisted in this version.",
                List.of("Let Forge AI choose attackers"));
        super.declareAttackers(attacker, combat);
        publishHumanTable();
    }

    @Override
    public void declareBlockers(Player defender, Combat combat) {
        ask("combat", "Forge AI will choose a legal block plan for your creatures. Combat is AI-assisted in this version.",
                List.of("Let Forge AI choose blockers"));
        super.declareBlockers(defender, combat);
        publishHumanTable();
    }

    @Override
    public boolean playChosenSpellAbility(SpellAbility sa) {
        boolean result = super.playChosenSpellAbility(sa);
        publishHumanTable();
        return result;
    }

    private void publishHumanTable() {
        HumanDecisionChannel.publish(TableSnapshot.capture(getGame(), me));
    }

    public Player tablePlayer() { return me; }
}
