package fly.agent;

import forge.LobbyPlayer;
import forge.deck.CardPool;
import forge.game.GameEntity;
import forge.game.GameEntityView;
import forge.game.card.Card;
import forge.game.card.CardUtil;
import forge.game.card.CardView;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.player.DelayedReveal;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityView;
import forge.gui.FThreads;
import forge.gui.control.PlaybackSpeed;
import forge.gui.interfaces.IGuiGame;
import forge.gamemodes.match.input.Input;
import forge.gamemodes.match.input.InputAttack;
import forge.gamemodes.match.input.InputBlock;
import forge.gamemodes.match.input.InputPayMana;
import forge.gamemodes.match.input.InputSelectManyBase;
import forge.gamemodes.match.input.InputSelectTargets;
import forge.item.PaperCard;
import forge.util.FSerializableFunction;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;

/**
 * Headless Forge GUI bridge. Forge continues to own and validate every action;
 * browser answers are converted back to Forge's own human-input callbacks.
 */
public final class WebHumanGui implements InvocationHandler {
    private record RoutedOption(String id, String label, Runnable apply) { }
    private record TargetOption(String id, String label, CardView card, PlayerView player) { }
    private record CombatContext(Player subject, Combat combat, boolean attackers) { }
    private record TargetContext(SpellAbility ability, List<TargetOption> options,
                                 int existingTargets) { }

    private final DecisionBroker broker;
    private final WebHumanController controller;
    private volatile List<CardView> selectableCards = List.of();
    private volatile List<CardView> weaklySelectableCards = List.of();
    private volatile int selectableMin;
    private volatile int selectableMax;
    private volatile String promptMessage = "Forge is waiting for a decision.";
    private volatile CardView promptCard;
    private volatile String button1 = "OK";
    private volatile String button2 = "Cancel";
    private volatile boolean button1Enabled;
    private volatile boolean button2Enabled;
    private volatile CombatContext combatContext;
    private volatile TargetContext targetContext;
    private volatile boolean driverScheduled;
    private volatile boolean driving;
    private volatile boolean rerunRequested;
    private volatile String statusMessage = "";

    public WebHumanGui(DecisionBroker broker, WebHumanController controller) {
        this.broker = Objects.requireNonNull(broker);
        this.controller = Objects.requireNonNull(controller);
    }

    public IGuiGame create() {
        return (IGuiGame) Proxy.newProxyInstance(IGuiGame.class.getClassLoader(),
                new Class<?>[]{IGuiGame.class}, this);
    }

    public void setCombatContext(Player subject, Combat combat, boolean attackers) {
        combatContext = new CombatContext(subject, combat, attackers);
    }

    public void clearCombatContext() { combatContext = null; }

    /** The live Forge input still performs all target legality checks. This is only its browser-facing option list. */
    public void setTargetContext(SpellAbility ability) {
        if (ability == null || ability.getTargetRestrictions() == null) {
            targetContext = null;
            return;
        }
        List<TargetOption> options = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int index = 0;
        try {
            for (Card card : CardUtil.getValidCardsToTarget(ability)) {
                CardView view = CardView.get(card);
                String key = "card:" + view.getId();
                if (seen.add(key)) {
                    options.add(new TargetOption("target-" + index++, safeLabel(view, index), view, null));
                }
            }
            for (GameEntity entity : ability.getTargetRestrictions().getAllCandidates(ability, true)) {
                if (entity instanceof Player player) {
                    PlayerView view = player.getView();
                    String key = "player:" + view.getId();
                    if (seen.add(key)) {
                        options.add(new TargetOption("target-" + index++, view.getName(), null, view));
                    }
                }
            }
        } catch (RuntimeException failure) {
            System.err.println("[ForgeHuman] Could not prepare target display list: " + failure);
        }
        targetContext = new TargetContext(ability, List.copyOf(options), ability.getTargets().size());
    }

    public void clearTargetContext() { targetContext = null; }

    @Override public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (method.getDeclaringClass() == Object.class) {
            return switch (method.getName()) {
                case "toString" -> "ForgeWebHumanGui";
                case "hashCode" -> System.identityHashCode(proxy);
                case "equals" -> proxy == args[0];
                default -> throw new UnsupportedOperationException(method.getName());
            };
        }
        if (method.isDefault()) return InvocationHandler.invokeDefault(proxy, method, args);

        return switch (method.getName()) {
            case "getGameView", "getGamestate" -> null;
            case "isNetGame", "isGamePaused", "isSelecting" -> method.getName().equals("isSelecting")
                    ? !selectableCards.isEmpty() : false;
            case "isUiSetToSkipPhase" -> false;
            case "getDayTime" -> "";
            case "getGameSpeed" -> PlaybackSpeed.NORMAL;
            case "confirm" -> confirmFromGui(args);
            case "showConfirmDialog" -> showConfirmDialog(args);
            case "showOptionDialog" -> showOptionDialog(args);
            case "showInputDialog" -> showInputDialog(args);
            case "getAbilityToPlay" -> getAbilityToPlay(args);
            case "getChoices" -> getChoices(args);
            case "getInteger" -> getInteger(args);
            case "one" -> one(args);
            case "oneOrNone" -> oneOrNone(args);
            case "many" -> many(args);
            case "order" -> order(args);
            case "insertInList" -> insertInList(args);
            case "chooseSingleEntityForEffect" -> chooseSingleEntity(args);
            case "chooseEntitiesForEffect" -> chooseEntities(args);
            case "assignCombatDamage" -> assignCombatDamage(args);
            case "assignGenericAmount" -> assignGenericAmount(args);
            case "manipulateCardList" -> manipulateCardList(args);
            case "sideboard" -> sideboard(args);
            case "setSelectables" -> { setSelectables((Iterable<CardView>) args[0], (Integer) args[1], (Integer) args[2]); yield null; }
            case "clearSelectables" -> { selectableCards = List.of(); selectableMin = 0; selectableMax = 0; yield null; }
            case "setWeaklySelectable" -> { weaklySelectableCards = cardViews((Iterable<?>) args[0]); yield null; }
            case "clearWeaklySelectable" -> { weaklySelectableCards = List.of(); yield null; }
            case "showPromptMessage" -> { showPromptMessage(args); yield null; }
            case "updateButtons" -> { updateButtons(args); yield null; }
            case "setHighlighted", "setPanelSelection", "setCard", "setPlayerAvatar",
                 "setGameView", "setOriginalGameController", "setGameController", "setSpectator",
                 "openView", "afterGameEnd", "showCombat", "flashIncorrectAction", "alertUser",
                 "finishGame", "handleGameEvent", "updateRevealedCards", "updateCard", "updateCards",
                 "refreshCardDetails", "refreshField", "setGamePause", "setGameSpeed", "updateDayTime",
                 "awaitNextInput", "cancelAwaitNextInput", "showWaitingTimer", "updateAutoPassPrompt",
                 "setCurrentPlayer", "applyYieldUpdate", "setNetGame", "updateDrawOffer", "updatePhase",
                 "updateTurn", "updatePlayerControl", "enableOverlay", "disableOverlay", "showManaPool",
                 "hideManaPool", "updateStack", "notifyStackAddition", "notifyStackRemoval", "handleLandPlayed",
                 "updateZones", "updateManaPool", "updateLives", "updateShards", "updateDependencies",
                 "showRevealedCards", "hideRevealedCards", "refreshYieldUi", "applyDelta" -> null;
            case "message" -> { rememberMessage(args); yield null; }
            case "showErrorDialog" -> { rememberMessage(args); yield null; }
            case "reveal" -> null;
            default -> throw new UnsupportedOperationException("Forge GUI method not implemented: " + method);
        };
    }

    private void setSelectables(Iterable<CardView> source, int min, int max) {
        selectableCards = cardViews(source);
        selectableMin = Math.max(0, min);
        selectableMax = Math.max(selectableMin, max);
        scheduleInputDriver();
    }

    private List<CardView> cardViews(Iterable<?> source) {
        if (source == null) return List.of();
        List<CardView> result = new ArrayList<>();
        Set<Integer> ids = new HashSet<>();
        for (Object item : source) {
            if (item instanceof CardView view && ids.add(view.getId())) result.add(view);
        }
        return List.copyOf(result);
    }

    private void showPromptMessage(Object[] args) {
        promptMessage = args.length > 1 && args[1] instanceof String text ? text : "Forge is waiting for a decision.";
        promptCard = args.length > 2 && args[2] instanceof CardView view ? view : null;
        scheduleInputDriver();
    }

    private void updateButtons(Object[] args) {
        if (args.length >= 6) {
            button1 = String.valueOf(args[1]);
            button2 = String.valueOf(args[2]);
            button1Enabled = Boolean.TRUE.equals(args[3]);
            button2Enabled = Boolean.TRUE.equals(args[4]);
        } else if (args.length >= 5) {
            button1 = String.valueOf(args[1]);
            button2 = String.valueOf(args[2]);
            button1Enabled = Boolean.TRUE.equals(args[3]);
            button2Enabled = Boolean.TRUE.equals(args[4]);
        }
        scheduleInputDriver();
    }

    private void rememberMessage(Object[] args) {
        String message = args.length > 0 ? String.valueOf(args[0]) : "";
        if (args.length > 1 && args[1] instanceof String title && !title.isBlank()) {
            statusMessage = title + ": " + message;
        } else {
            statusMessage = message;
        }
        WebHumanSession.updateMessage(statusMessage);
    }

    private boolean confirmFromGui(Object[] args) throws Exception {
        String question = args.length > 1 ? String.valueOf(args[1]) : "Confirm?";
        boolean defaultYes = args.length > 2 && Boolean.TRUE.equals(args[2]);
        @SuppressWarnings("unchecked") List<String> supplied = args.length > 3 && args[3] instanceof List<?> list
                ? (List<String>) list : List.of("Yes", "No");
        List<String> labels = supplied == null || supplied.size() < 2 ? List.of("Yes", "No") : supplied;
        List<DecisionBroker.Option> options = List.of(
                new DecisionBroker.Option("yes", labels.get(0)),
                new DecisionBroker.Option("no", labels.get(1)));
        String answer = broker.ask("confirm", question, options, 1, 1, 0).get(0);
        return "yes".equals(answer);
    }

    private boolean showConfirmDialog(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        String yes = args.length > 2 ? String.valueOf(args[2]) : "Yes";
        String no = args.length > 3 ? String.valueOf(args[3]) : "No";
        List<DecisionBroker.Option> options = List.of(
                new DecisionBroker.Option("yes", yes), new DecisionBroker.Option("no", no));
        return "yes".equals(broker.ask("confirm", message, options, 1, 1, 0).get(0));
    }

    private int showOptionDialog(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        @SuppressWarnings("unchecked") List<String> choices = args.length > 3 && args[3] instanceof List<?> list
                ? (List<String>) list : List.of();
        if (choices.isEmpty()) return -1;
        List<Object> selected = chooseObjects("choose_one", message, choices, 1, 1, null);
        return choices.indexOf(selected.get(0));
    }

    private String showInputDialog(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        String initial = args.length > 3 && args[3] instanceof String text ? text : "";
        @SuppressWarnings("unchecked") List<String> choices = args.length > 4 && args[4] instanceof List<?> list
                ? (List<String>) list : List.of();
        if (!choices.isEmpty()) {
            List<Object> selected = chooseObjects("choose_one", message, choices, 1, 1, null);
            return String.valueOf(selected.get(0));
        }
        try {
            return broker.askText("text_input", message + (initial.isBlank() ? "" : "\nCurrent value: " + initial),
                    1, 256, 0);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Human text input interrupted", e);
        } catch (TimeoutException e) {
            throw new IllegalStateException("Human text input timed out", e);
        }
    }

    private Object getAbilityToPlay(Object[] args) throws Exception {
        CardView host = args.length > 0 && args[0] instanceof CardView view ? view : null;
        List<?> abilities = args.length > 1 && args[1] instanceof List<?> list ? list : List.of();
        if (abilities.isEmpty()) return null;
        if (abilities.size() == 1) return abilities.get(0);
        String title = (host == null ? "Choose an ability" : safeLabel(host, 0))
                + "\nChoose the Forge ability to use.";
        List<Object> selected = chooseObjects("choose_ability", title, abilities, 1, 1, null);
        return selected.isEmpty() ? null : selected.get(0);
    }

    private Object getChoices(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        int min = (Integer) args[1];
        int max = (Integer) args[2];
        List<?> choices = args[3] instanceof List<?> list ? list : List.of();
        Object display = args.length > 5 ? args[5] : null;
        return chooseObjects("choose_cards", message, choices, min, max, display);
    }

    private Integer getInteger(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        int min = (Integer) args[1];
        int max = (Integer) args[2];
        if (min >= max) return min;
        try {
            return broker.askInteger("choose_number", message, min, max, 0);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Human numeric decision interrupted", e);
        } catch (TimeoutException e) {
            throw new IllegalStateException("Human numeric decision timed out", e);
        }
    }

    private Object one(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        List<?> choices = args[1] instanceof List<?> list ? list : List.of();
        Object display = args.length > 2 ? args[2] : null;
        List<Object> selected = chooseObjects("choose_one", message, choices, 1, 1, display);
        return selected.isEmpty() ? null : selected.get(0);
    }

    private Object oneOrNone(Object[] args) throws Exception {
        String message = String.valueOf(args[0]);
        List<?> choices = args[1] instanceof List<?> list ? list : List.of();
        List<Object> selected = chooseObjects("choose_one", message, choices, 0, 1, null);
        return selected.isEmpty() ? null : selected.get(0);
    }

    private Object many(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        String top = String.valueOf(args[1]);
        int min = (Integer) args[2];
        int max = (Integer) args[3];
        List<?> source = args[4] instanceof List<?> list ? list : List.of();
        List<?> destination = args.length > 5 && args[5] instanceof List<?> list ? list : List.of();
        List<Object> selected = chooseObjects("choose_cards", title + "\n" + top, source, min, max, null);
        if (destination.isEmpty()) return selected;
        List<Object> result = new ArrayList<>(destination);
        result.addAll(selected);
        return result;
    }

    private Object order(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        String top = String.valueOf(args[1]);
        List<?> source = args.length > 4 && args[4] instanceof List<?> list ? list : List.of();
        List<?> destination = args.length > 5 && args[5] instanceof List<?> list ? list : List.of();
        @SuppressWarnings("unchecked") Class<?> methodClass = IGuiGame.OrderResult.class;
        List<Object> ordered = chooseObjects("choose_order", title + "\n" + top,
                source, source.size(), source.size(), null);
        if (!destination.isEmpty()) {
            List<Object> complete = new ArrayList<>(destination);
            complete.addAll(ordered);
            ordered = complete;
        }
        return new IGuiGame.OrderResult<>(ordered, false);
    }

    private Object insertInList(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        Object newItem = args[1];
        List<?> oldItems = args.length > 2 && args[2] instanceof List<?> list ? list : List.of();
        List<DecisionBroker.Option> positions = new ArrayList<>();
        positions.add(new DecisionBroker.Option("position-0", "Put first"));
        for (int index = 0; index < oldItems.size(); index++) {
            positions.add(new DecisionBroker.Option("position-" + (index + 1),
                    "After " + safeLabel(oldItems.get(index), index)));
        }
        int position = Integer.parseInt(broker.ask("choose_order", title, positions, 1, 1, 0)
                .get(0).substring("position-".length()));
        List<Object> result = new ArrayList<>(oldItems);
        result.add(Math.min(position, result.size()), newItem);
        return result;
    }

    private Object chooseSingleEntity(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        List<?> choices = args.length > 1 && args[1] instanceof List<?> list ? list : List.of();
        boolean optional = args.length > 3 && Boolean.TRUE.equals(args[3]);
        List<Object> selected = chooseObjects("choose_one", title, choices, optional ? 0 : 1, 1, null);
        return selected.isEmpty() ? null : selected.get(0);
    }

    private Object chooseEntities(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        List<?> choices = args.length > 1 && args[1] instanceof List<?> list ? list : List.of();
        int min = args.length > 2 ? (Integer) args[2] : 0;
        int max = args.length > 3 ? (Integer) args[3] : choices.size();
        return chooseObjects("choose_cards", title, choices, min, max, null);
    }

    private Object assignCombatDamage(Object[] args) throws Exception {
        CardView attacker = (CardView) args[0];
        @SuppressWarnings("unchecked") List<CardView> blockers = (List<CardView>) args[1];
        int damage = (Integer) args[2];
        GameEntityView defender = (GameEntityView) args[3];
        boolean maySkip = Boolean.TRUE.equals(args[5]);
        Map<CardView, Integer> assigned = new LinkedHashMap<>();
        int remaining = damage;
        List<CardView> targets = new ArrayList<>(blockers);
        boolean hasDefender = defender != null;
        if (hasDefender) targets.add(null);
        if (targets.isEmpty()) {
            assigned.put(null, damage);
            return assigned;
        }
        for (int index = 0; index < targets.size(); index++) {
            CardView target = targets.get(index);
            boolean last = index == targets.size() - 1;
            int min = last ? remaining : 0;
            int max = remaining;
            if (target == null) {
                String label = safeLabel(defender, index);
                int value = askInteger("assign_combat_damage", "Assign combat damage from "
                        + safeLabel(attacker, 0) + " to " + label + " (" + min + "–" + max + ").", min, max);
                if (value > 0) assigned.put(null, value);
                remaining -= value;
                continue;
            }
            int value = askInteger("assign_combat_damage", "Assign combat damage from "
                    + safeLabel(attacker, 0) + " to " + safeLabel(target, index)
                    + " (" + min + "–" + max + ").", min, max);
            if (value > 0) assigned.put(target, value);
            remaining -= value;
        }
        if (remaining != 0) {
            // Do not invent damage assignment; let Forge re-request after an invalid result.
            throw new IllegalStateException("Combat damage assignment did not account for all damage");
        }
        if (maySkip && assigned.isEmpty()) return null;
        return assigned;
    }

    private Object assignGenericAmount(Object[] args) throws Exception {
        CardView source = (CardView) args[0];
        @SuppressWarnings("unchecked") Map<Object, Integer> available = (Map<Object, Integer>) args[1];
        int amount = (Integer) args[2];
        boolean atLeastOne = Boolean.TRUE.equals(args[3]);
        String label = String.valueOf(args[4]);
        Map<Object, Integer> result = new LinkedHashMap<>();
        int remaining = amount;
        List<Object> targets = new ArrayList<>(available.keySet());
        for (int index = 0; index < targets.size(); index++) {
            Object target = targets.get(index);
            int maximum = Math.min(remaining, Math.max(0, available.getOrDefault(target, remaining)));
            int minimum = atLeastOne && remaining > 0 ? 1 : 0;
            if (index == targets.size() - 1) minimum = remaining;
            minimum = Math.min(minimum, maximum);
            String question = "Assign " + label + " from " + safeLabel(source, 0) + " to "
                    + safeLabel(target, index) + " (" + minimum + "–" + maximum + ").";
            int value = askInteger("assign_amount", question, minimum, maximum);
            if (value > 0) result.put(target, value);
            remaining -= value;
        }
        if (remaining != 0) throw new IllegalStateException("Forge amount allocation is incomplete");
        return result;
    }

    private Object manipulateCardList(Object[] args) throws Exception {
        String title = String.valueOf(args[0]);
        List<CardView> cards = iterableCardViews(args[1]);
        return chooseObjects("choose_order", title, cards, cards.size(), cards.size(), null);
    }

    private Object sideboard(Object[] args) {
        CardPool main = (CardPool) args[1];
        return main == null ? List.of() : main.toFlatList();
    }

    private List<Object> chooseObjects(String kind, String message, List<?> choices,
                                       int min, int max, Object display) throws Exception {
        if (choices == null || choices.isEmpty()) return new ArrayList<>();
        int lower = Math.max(0, Math.min(min, choices.size()));
        int upper = max < 0 ? choices.size() : Math.max(lower, Math.min(max, choices.size()));
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (int index = 0; index < choices.size(); index++) {
            options.add(new DecisionBroker.Option("choice-" + index,
                    displayLabel(display, choices.get(index), index)));
        }
        List<String> selected = broker.ask(kind, message, options, lower, upper, 0);
        List<Object> result = new ArrayList<>();
        for (String id : selected) result.add(choices.get(Integer.parseInt(id.substring("choice-".length()))));
        return result;
    }

    private String displayLabel(Object display, Object value, int index) {
        if (display instanceof Function<?, ?> function) {
            try {
                @SuppressWarnings("unchecked") Function<Object, Object> format = (Function<Object, Object>) function;
                Object text = format.apply(value);
                if (text != null && !String.valueOf(text).isBlank()) return safeText(String.valueOf(text));
            } catch (RuntimeException ignored) { }
        }
        return safeLabel(value, index);
    }

    private String safeLabel(Object value, int index) {
        if (value == null) return "None";
        if (value instanceof Card card) return safeLabel(CardView.get(card), index);
        if (value instanceof CardView card) {
            if (!card.canBeShownTo(controller.getPlayer().getView())) return "Hidden card";
            if (card.isFaceDown() && !card.canFaceDownBeShownTo(controller.getPlayer().getView())) return "Face-down card";
            String name = card.getName();
            return safeText(name == null || name.isBlank() ? "Card" : name);
        }
        if (value instanceof Player player) return safeText(player.getName());
        if (value instanceof PlayerView player) return safeText(player.getName());
        if (value instanceof SpellAbility ability) {
            String host = ability.getHostCard() == null ? "Ability" : safeLabel(CardView.get(ability.getHostCard()), index);
            String details = String.valueOf(ability).replaceAll("\\s+", " ").trim();
            return safeText(details.isBlank() ? host : host + " — " + details);
        }
        if (value instanceof SpellAbilityView ability) {
            CardView host = ability.getHostCard();
            if (host != null && !host.canBeShownTo(controller.getPlayer().getView())) return "Ability on hidden card";
            return safeText(ability.getDescription());
        }
        if (value instanceof GameEntityView entity) return safeText(entity.getName());
        if (value instanceof PaperCard card) return safeText(card.getName());
        if (value instanceof String || value instanceof Number || value instanceof Enum<?>) return safeText(String.valueOf(value));
        String simpleName = value.getClass().getSimpleName();
        return safeText((simpleName.isBlank() ? "Choice" : simpleName) + " " + (index + 1));
    }

    private String safeText(String value) {
        String text = value == null ? "" : value.replaceAll("\\s+", " ").trim();
        if (text.length() > 240) return text.substring(0, 237) + "…";
        return text;
    }

    private List<CardView> iterableCardViews(Object source) {
        if (!(source instanceof Iterable<?> iterable)) return List.of();
        List<CardView> result = new ArrayList<>();
        for (Object value : iterable) if (value instanceof CardView card) result.add(card);
        return result;
    }

    private void scheduleInputDriver() {
        if (driving) {
            rerunRequested = true;
            return;
        }
        if (driverScheduled) return;
        driverScheduled = true;
        FThreads.invokeInEdtLater(() -> {
            driverScheduled = false;
            driveCurrentInput();
        });
    }

    private void driveCurrentInput() {
        if (driving) {
            rerunRequested = true;
            return;
        }
        Input current = currentInput();
        if (current == null) return;
        driving = true;
        rerunRequested = false;
        try {
            int actions = 0;
            while (current != null && actions++ < 1000) {
                List<RoutedOption> options = buildInputOptions(current);
                if (options.isEmpty()) {
                    WebHumanSession.fault("Forge requested an input that the browser controller cannot present yet: "
                            + current.getClass().getSimpleName() + ". No game action was selected.");
                    return;
                }
                List<DecisionBroker.Option> brokerOptions = new ArrayList<>();
                for (RoutedOption option : options) brokerOptions.add(new DecisionBroker.Option(option.id(), option.label()));
                String kind = inputKind(current);
                String message = currentMessage();

                if (current instanceof InputSelectTargets || current instanceof InputSelectManyBase<?>) {
                    List<DecisionBroker.Option> controls = new ArrayList<>();
                    controls.add(new DecisionBroker.Option("confirm", "Confirm selection"));
                    if (button2Enabled) controls.add(new DecisionBroker.Option("cancel", button2));
                    int[] bounds = selectionBounds(current, options.size());
                    DecisionBroker.Answer answer = broker.askWithActions(kind, message, brokerOptions,
                            bounds[0], bounds[1], controls, 0);
                    if ("cancel".equals(answer.action())) {
                        if (currentInput() == current) controller.getInputProxy().selectButtonCancel();
                        return;
                    }
                    for (String id : answer.selected()) {
                        if (currentInput() != current) break;
                        RoutedOption option = routed(options, id);
                        if (option != null) option.apply().run();
                    }
                    if (currentInput() == current && "confirm".equals(answer.action())) {
                        controller.getInputProxy().selectButtonOK();
                    }
                    if (currentInput() != current) return;
                    continue;
                }

                List<String> selected = broker.ask(kind, message, brokerOptions, 1, 1, 0);
                RoutedOption selectedOption = routed(options, selected.get(0));
                if (selectedOption != null) selectedOption.apply().run();
                if (currentInput() != current) return;

                if (current instanceof InputPayMana) {
                    // Mana activation can switch inputs or refresh the cost on the game thread.
                    // Wait for Forge's next GUI callback rather than issuing a stale second tap.
                    rerunRequested = true;
                    return;
                }
                if (current instanceof InputAttack || current instanceof InputBlock) continue;
                return;
            }
            if (actions >= 1000) WebHumanSession.fault("Forge input exceeded the browser decision safety limit.");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            WebHumanSession.fault("Browser decision was interrupted.");
        } catch (TimeoutException e) {
            WebHumanSession.fault("Browser decision timed out; no action was selected.");
        } catch (RuntimeException e) {
            WebHumanSession.fault("Browser input failed safely: " + e.getMessage());
        } finally {
            driving = false;
            if (rerunRequested) {
                rerunRequested = false;
                scheduleInputDriver();
            }
        }
    }

    private Input currentInput() {
        return controller.getInputProxy().getInput();
    }

    private String currentMessage() {
        String message = promptMessage == null ? "" : promptMessage;
        if (promptCard != null) message = safeLabel(promptCard, 0) + "\n" + message;
        if (statusMessage != null && !statusMessage.isBlank()) message += "\n" + statusMessage;
        return message;
    }

    private String inputKind(Input input) {
        if (input instanceof InputAttack) return "declare_attackers";
        if (input instanceof InputBlock) return "declare_blockers";
        if (input instanceof InputPayMana) return "pay_mana";
        if (input instanceof InputSelectTargets) return "choose_targets";
        if (input instanceof InputSelectManyBase<?>) return "choose_cards";
        return "forge_input";
    }

    private int[] selectionBounds(Input input, int optionCount) {
        int min = Math.min(selectableMin, optionCount);
        int max = Math.min(Math.max(min, selectableMax), optionCount);
        if (input instanceof InputSelectTargets && targetContext != null) {
            SpellAbility ability = targetContext.ability();
            int initial = targetContext.existingTargets();
            min = Math.max(0, ability.getMinTargets() - initial);
            max = Math.max(min, ability.getMaxTargets() - initial);
            min = Math.min(min, optionCount);
            max = Math.min(max, optionCount);
        }
        if (input instanceof InputSelectManyBase<?> && button2Enabled) min = 0;
        return new int[]{min, max};
    }

    private List<RoutedOption> buildInputOptions(Input input) {
        if (input instanceof InputAttack) return attackerOptions(input);
        if (input instanceof InputBlock) return blockerOptions(input);
        if (input instanceof InputPayMana) return manaOptions(input);
        if (input instanceof InputSelectTargets) return targetOptions();
        if (input instanceof InputSelectManyBase<?>) return selectableOptions(input, selectableCards);
        if (!selectableCards.isEmpty()) return selectableOptions(input, selectableCards);

        List<RoutedOption> options = new ArrayList<>();
        if (button1Enabled) options.add(new RoutedOption("button-1", button1,
                () -> controller.getInputProxy().selectButtonOK()));
        if (button2Enabled) options.add(new RoutedOption("button-2", button2,
                () -> controller.getInputProxy().selectButtonCancel()));
        return options;
    }

    private List<RoutedOption> targetOptions() {
        List<RoutedOption> options = new ArrayList<>();
        TargetContext context = targetContext;
        List<TargetOption> targets = context == null ? new ArrayList<>() : context.options();
        if (targets.isEmpty()) {
            int index = 0;
            for (CardView card : selectableCards) {
                String id = "target-card-" + index;
                options.add(new RoutedOption(id, safeLabel(card, index++), () -> selectCard(card)));
            }
            return options;
        }
        for (TargetOption target : targets) {
            if (target.card() != null) {
                options.add(new RoutedOption(target.id(), target.label(), () -> selectCard(target.card())));
            } else if (target.player() != null) {
                options.add(new RoutedOption(target.id(), target.label(), () -> selectPlayer(target.player())));
            }
        }
        return options;
    }

    private List<RoutedOption> selectableOptions(Input input, List<CardView> cards) {
        List<RoutedOption> options = new ArrayList<>();
        int index = 0;
        for (CardView card : cards) {
            String id = "card-" + index;
            options.add(new RoutedOption(id, safeLabel(card, index++), () -> selectCard(card)));
        }
        return options;
    }

    private List<RoutedOption> manaOptions(Input input) {
        List<RoutedOption> options = new ArrayList<>();
        int index = 0;
        for (CardView view : weaklySelectableCards) {
            Card card = controller.getCard(view);
            if (card == null || input.getActivateAction(card) == null) continue;
            String id = "mana-source-" + index;
            options.add(new RoutedOption(id, "Pay with " + safeLabel(view, index++), () -> selectCard(view)));
        }
        if (button1Enabled) options.add(new RoutedOption("mana-button-1", button1,
                () -> controller.getInputProxy().selectButtonOK()));
        if (button2Enabled) options.add(new RoutedOption("mana-button-2", button2,
                () -> controller.getInputProxy().selectButtonCancel()));
        return options;
    }

    private List<RoutedOption> attackerOptions(Input input) {
        CombatContext context = combatContext;
        if (context == null || context.combat() == null) return List.of();
        Combat combat = context.combat();
        Player attackingPlayer = context.subject();
        List<RoutedOption> options = new ArrayList<>();
        int index = 0;
        for (GameEntity defender : combat.getDefenders()) {
            for (Card attacker : attackingPlayer.getCreaturesInPlay()) {
                if (combat.isAttacking(attacker) || !CombatUtil.canAttack(attacker, defender)) continue;
                String id = "attack-" + index++;
                String label = "Attack with " + safeLabel(CardView.get(attacker), index)
                        + " at " + safeLabel(defender, index);
                options.add(new RoutedOption(id, label, () -> {
                    selectDefender(defender);
                    if (currentInput() == input) selectCard(CardView.get(attacker));
                }));
            }
        }
        options.add(new RoutedOption("finish-attackers", "Finish attacker declarations",
                () -> controller.getInputProxy().selectButtonOK()));
        return options;
    }

    private List<RoutedOption> blockerOptions(Input input) {
        CombatContext context = combatContext;
        if (context == null || context.combat() == null) return List.of();
        Combat combat = context.combat();
        Player defender = context.subject();
        List<RoutedOption> options = new ArrayList<>();
        int index = 0;
        for (Card attacker : combat.getAttackers()) {
            for (Card blocker : defender.getCreaturesInPlay()) {
                boolean assigned = combat.isBlocking(blocker, attacker);
                if (!assigned && !CombatUtil.canBlock(attacker, blocker, combat)) continue;
                String id = "block-" + index++;
                String label = (assigned ? "Remove " : "Block ") + safeLabel(CardView.get(attacker), index)
                        + (assigned ? " from " : " with ") + safeLabel(CardView.get(blocker), index);
                options.add(new RoutedOption(id, label, () -> {
                    selectCard(CardView.get(attacker));
                    if (currentInput() == input) selectCard(CardView.get(blocker));
                }));
            }
        }
        options.add(new RoutedOption("finish-blockers", "Finish blocker declarations",
                () -> controller.getInputProxy().selectButtonOK()));
        return options;
    }

    private void selectCard(CardView card) {
        if (card != null) controller.getInputProxy().selectCard(card, null, null);
    }

    private void selectPlayer(PlayerView player) {
        if (player != null) controller.getInputProxy().selectPlayer(player, null);
    }

    private void selectDefender(GameEntity defender) {
        if (defender instanceof Player player) selectPlayer(player.getView());
        else if (defender instanceof Card card) selectCard(CardView.get(card));
    }

    private RoutedOption routed(List<RoutedOption> options, String id) {
        for (RoutedOption option : options) if (option.id().equals(id)) return option;
        return null;
    }

    private Integer askInteger(String kind, String message, int min, int max) throws Exception {
        if (min >= max) return min;
        return broker.askInteger(kind, message, min, max, 0);
    }
}
