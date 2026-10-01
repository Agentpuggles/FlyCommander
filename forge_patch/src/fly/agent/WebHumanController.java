package fly.agent;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.Player;
import forge.player.PlayerControllerHuman;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import java.util.*;

/** Real human controller. Explicit digital-hand handshake mode, NOT paper play. */
public final class WebHumanController extends PlayerControllerHuman {
    private final DecisionBroker decisions = new DecisionBroker();
    private final Player human;
    public WebHumanController(Game game, Player player, LobbyPlayer lobby) {
        super(game, player, lobby);
        human = player;
        setGui(new WebHumanGui(decisions).create());
        System.out.println("[RuntimeTest] WebHumanController instantiated: " + player.getName());
    }
    public DecisionBroker decisions() { return decisions; }
    private void requireTest() {
        if (!Boolean.getBoolean("fly.agent.controllerTest"))
            throw new IllegalStateException("Physical input synchronization is not installed");
    }
    private List<String> ask(String kind, String message, List<DecisionBroker.Option> options, int min, int max) {
        requireTest();
        try {
            System.out.println("[RuntimeTest] Forge requests human " + kind);
            List<String> result = decisions.ask(kind, "CONTROLLER TEST — DIGITAL HAND, NOT PAPER PLAY\n" + message,
                    options, min, max, 0);
            System.out.println("[RuntimeTest] Human response returned to Forge: " + kind + " " + result);
            return result;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Human decision interrupted", e);
        } catch (java.util.concurrent.TimeoutException e) {
            throw new IllegalStateException(e);
        }
    }
    @Override public Player chooseStartingPlayer(boolean isFirstGame) {
        List<Player> players = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (Player p : getGame().getPlayersInTurnOrder()) {
            options.add(new DecisionBroker.Option(Integer.toString(players.size()), p.getName()));
            players.add(p);
        }
        return players.get(Integer.parseInt(ask("starting_player", "Forge asks you to choose who starts.", options, 1, 1).get(0)));
    }
    @Override public boolean mulliganKeepHand(Player first, int cardsToReturn) {
        StringBuilder hand = new StringBuilder("Forge-generated TEST opening hand:\n");
        for (var c : human.getCardsIn(ZoneType.Hand)) hand.append(c.getName()).append(" #").append(c.getId()).append('\n');
        hand.append("Cards to return if keeping: ").append(cardsToReturn);
        return ask("mulligan", hand.toString(), List.of(new DecisionBroker.Option("keep", "Keep test hand"),
                new DecisionBroker.Option("mulligan", "Mulligan")), 1, 1).get(0).equals("keep");
    }
    @Override public CardCollectionView tuckCardsViaMulligan(CardCollectionView hand, int count) {
        List<forge.game.card.Card> cards = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (var card : hand) {
            options.add(new DecisionBroker.Option(Integer.toString(cards.size()), card.getName() + " #" + card.getId()));
            cards.add(card);
        }
        CardCollection selected = new CardCollection();
        for (String id : ask("mulligan_bottom", "Select test hand cards to return to library.", options, count, count))
            selected.add(cards.get(Integer.parseInt(id)));
        return selected;
    }
    @Override public List<SpellAbility> chooseSpellAbilityToPlay() {
        ask("priority", "YOUR PRIORITY — Forge turn " + getGame().getPhaseHandler().getTurn()
                + " / " + getGame().getPhaseHandler().getPhase()
                + " / stack size " + getGame().getStack().size()
                + "\nHandshake scope: explicit pass only. Casting/activation controls are not implemented here.",
                List.of(new DecisionBroker.Option("pass", "Pass priority")), 1, 1);
        // Forge 2.0.15 PhaseHandler.mainLoopStep: null breaks the action loop
        // as "I pass". InputPassPriority.getChosenSa() likewise remains null
        // after a real human pass. An empty list would re-prompt this same seat.
        return null;
    }
    // These are waiting indicators, not decisions; Forge calls them for players
    // who are waiting while another player selects the starting player.
    @Override public void awaitNextInput() { System.out.println("[RuntimeTest] Flynn waiting for another player"); }
    @Override public void cancelAwaitNextInput() { }
}
