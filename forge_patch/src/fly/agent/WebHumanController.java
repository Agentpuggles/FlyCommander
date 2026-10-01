package fly.agent;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameEntityView;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.combat.Combat;
import forge.game.player.DelayedReveal;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.util.collect.FCollectionView;

import java.util.*;

/** Forge's human controller backed by browser decisions; no AI makes decisions for this seat. */
public final class WebHumanController extends forge.player.PlayerControllerHuman {
    private final DecisionBroker decisions = new DecisionBroker();
    private final Player human;
    private final WebHumanGui webGui;

    public WebHumanController(Game game, Player player, LobbyPlayer lobby) {
        super(game, player, lobby);
        human = player;
        webGui = new WebHumanGui(decisions, this);
        setGui(webGui.create());
        WebHumanSession.register(decisions);
        publishGame();
        System.out.println("[ForgeHuman] WebHumanController instantiated: " + player.getName());
    }

    public DecisionBroker decisions() { return decisions; }

    private List<String> ask(String kind, String message, List<DecisionBroker.Option> options,
                             int min, int max) {
        try {
            return decisions.ask(kind, message, options, min, max, 0);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Human decision interrupted", e);
        } catch (java.util.concurrent.TimeoutException e) {
            throw new IllegalStateException("Human decision timed out; no action was submitted", e);
        }
    }

    private void publishGame() {
        WebHumanSession.updateGame(getGame(), human);
    }

    @Override public Player chooseStartingPlayer(boolean isFirstGame) {
        publishGame();
        List<Player> players = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (Player p : getGame().getPlayersInTurnOrder()) {
            String id = "seat-" + players.size();
            options.add(new DecisionBroker.Option(id, p.getName()));
            players.add(p);
        }
        return players.get(Integer.parseInt(ask("starting_player", "Choose who will take the first turn.",
                options, 1, 1).get(0).substring("seat-".length())));
    }

    @Override public boolean mulliganKeepHand(Player first, int cardsToReturn) {
        publishGame();
        StringBuilder hand = new StringBuilder("Forge-generated opening hand:\n");
        for (Card card : human.getCardsIn(ZoneType.Hand)) hand.append(card.getName()).append('\n');
        hand.append("If you keep, Forge will ask you to choose ").append(cardsToReturn)
                .append(" card(s) to put on the bottom.");
        return "keep".equals(ask("mulligan", hand.toString(), List.of(
                new DecisionBroker.Option("keep", "Keep hand"),
                new DecisionBroker.Option("mulligan", "Mulligan")), 1, 1).get(0));
    }

    @Override public CardCollectionView tuckCardsViaMulligan(CardCollectionView hand, int count) {
        if (count <= 0) return CardCollection.EMPTY;
        List<Card> cards = new ArrayList<>();
        List<DecisionBroker.Option> options = new ArrayList<>();
        Map<String, Integer> copies = new HashMap<>();
        for (Card card : hand) {
            int copy = copies.merge(card.getName(), 1, Integer::sum);
            String id = "hand-card-" + cards.size();
            cards.add(card);
            String label = card.getName() + (copy > 1 ? " (copy " + copy + ")" : "");
            options.add(new DecisionBroker.Option(id, label));
        }
        List<String> selected = ask("mulligan_bottom",
                "Choose the card(s) to put on the bottom of your library.", options, count, count);
        CardCollection result = new CardCollection();
        for (String id : selected) result.add(cards.get(Integer.parseInt(id.substring("hand-card-".length()))));
        return result;
    }

    /** Forge supplies the current legal action set; null is Forge's pass-priority sentinel. */
    @Override public List<SpellAbility> chooseSpellAbilityToPlay() {
        publishGame();
        List<Card> cards = new ArrayList<>();
        cards.addAll(human.getCardsIn(ZoneType.Hand));
        cards.addAll(human.getCardsIn(ZoneType.Battlefield));
        cards.addAll(human.getCardsActivatableInExternalZones(true));

        List<SpellAbility> abilities = new ArrayList<>();
        Set<SpellAbility> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Card card : cards) {
            try {
                for (SpellAbility ability : card.getAllPossibleAbilities(human, true)) {
                    if (ability != null && seen.add(ability)) abilities.add(ability);
                }
            } catch (RuntimeException failure) {
                System.err.println("[ForgeHuman] Could not enumerate abilities for " + card.getName() + ": " + failure);
            }
        }

        List<DecisionBroker.Option> options = new ArrayList<>();
        options.add(new DecisionBroker.Option("pass", "Pass priority"));
        for (int index = 0; index < abilities.size(); index++) {
            SpellAbility ability = abilities.get(index);
            String action = ability.isLandAbility() ? "Play land"
                    : ability.isSpell() ? "Cast spell" : "Activate ability";
            String details = String.valueOf(ability).replaceAll("\\s+", " ").trim();
            if (details.length() > 140) details = details.substring(0, 137) + "…";
            String label = action + ": " + ability.getHostCard().getName();
            if (!details.isEmpty()) label += " — " + details;
            options.add(new DecisionBroker.Option("ability-" + index, label));
        }

        String phase = String.valueOf(getGame().getPhaseHandler().getPhase());
        String prompt = abilities.isEmpty()
                ? "No legal card plays or activated abilities are available. Pass priority."
                : "Choose one legal land play, spell or activated ability, or pass. Forge will request any targets, costs and other required choices separately.";
        String selected = ask("priority", "Turn " + getGame().getPhaseHandler().getTurn()
                + " · " + phase + " · stack " + getGame().getStack().size() + "\n" + prompt,
                options, 1, 1).get(0);
        if ("pass".equals(selected)) return null;
        int index = Integer.parseInt(selected.substring("ability-".length()));
        return List.of(abilities.get(index));
    }

    @Override public boolean playChosenSpellAbility(SpellAbility ability) {
        try {
            return super.playChosenSpellAbility(ability);
        } finally {
            publishGame();
        }
    }

    @Override public boolean chooseTargetsFor(SpellAbility ability) {
        webGui.setTargetContext(ability);
        try {
            return super.chooseTargetsFor(ability);
        } finally {
            webGui.clearTargetContext();
        }
    }

    @Override public void declareAttackers(Player attackingPlayer, Combat combat) {
        webGui.setCombatContext(attackingPlayer, combat, true);
        try {
            super.declareAttackers(attackingPlayer, combat);
        } finally {
            webGui.clearCombatContext();
            publishGame();
        }
    }

    @Override public void declareBlockers(Player defender, Combat combat) {
        webGui.setCombatContext(defender, combat, false);
        try {
            super.declareBlockers(defender, combat);
        } finally {
            webGui.clearCombatContext();
            publishGame();
        }
    }

    /** Keep Forge's entity candidates and return exactly the selected Forge objects. */
    @Override public <T extends GameEntity> T chooseSingleEntityForEffect(
            FCollectionView<T> optionList, DelayedReveal delayedReveal, SpellAbility ability,
            String title, boolean isOptional, Player targetedPlayer, Map<String, Object> params) {
        if (optionList.isEmpty()) return null;
        Map<GameEntityView, T> entityByView = new HashMap<>();
        List<GameEntityView> views = new ArrayList<>();
        for (T entity : optionList) {
            GameEntityView view = GameEntityView.get(entity);
            views.add(view);
            entityByView.put(view, entity);
        }
        GameEntityView selected = getGui().chooseSingleEntityForEffect(title, views, delayedReveal, isOptional);
        return entityByView.get(selected);
    }

    @Override public <T extends GameEntity> List<T> chooseEntitiesForEffect(
            FCollectionView<T> optionList, int min, int max, DelayedReveal delayedReveal,
            SpellAbility ability, String title, Player targetedPlayer, Map<String, Object> params) {
        if (optionList.isEmpty()) return new ArrayList<>();
        Map<GameEntityView, T> entityByView = new HashMap<>();
        List<GameEntityView> views = new ArrayList<>();
        for (T entity : optionList) {
            GameEntityView view = GameEntityView.get(entity);
            views.add(view);
            entityByView.put(view, entity);
        }
        List<GameEntityView> selected = getGui().chooseEntitiesForEffect(title, views,
                Math.min(min, views.size()), Math.min(max, views.size()), delayedReveal);
        List<T> result = new ArrayList<>();
        if (selected != null) for (GameEntityView view : selected) {
            T entity = entityByView.get(view);
            if (entity != null) result.add(entity);
        }
        return result;
    }

    // Waiting indicators are not decisions; actual Forge prompts are brokered above.
    @Override public void awaitNextInput() { }
    @Override public void cancelAwaitNextInput() { }
}
