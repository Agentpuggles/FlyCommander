package fly.agent;

import forge.GuiDesktop;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameStage;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.zone.ZoneType;
import forge.gui.GuiBase;
import forge.item.PaperCard;
import forge.model.FModel;
import forge.player.GamePlayerUtil;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Real Forge integration for the browser-controlled human. The test uses a
 * small controlled Forge state (the same style as Forge's own AITest), but all
 * requested actions are submitted through WebHumanController's real GUI/input
 * path and all zone changes, spell resolution, combat and damage are executed
 * by Forge. No Forge game-state classes are mocked.
 *
 * Usage: WebHumanGameIntegrationTest <real-Forge-deck-file>
 */
public final class WebHumanGameIntegrationTest {
    private static final long STEP_TIMEOUT_SECONDS = 20;

    private WebHumanGameIntegrationTest() { }

    public static void main(String[] args) {
        try {
            if (args.length != 1) {
                throw new IllegalArgumentException("Usage: WebHumanGameIntegrationTest <deck-file>");
            }
            run(args[0]);
            System.out.println("PASS: WebHumanController played a land, cast and resolved a spell, and attacked for real Forge damage");
            System.exit(0);
        } catch (Throwable failure) {
            failure.printStackTrace();
            System.exit(1);
        }
    }

    private static void run(String deckPath) throws Exception {
        System.setProperty("fly.agent.controllerTest", "true");
        GuiBase.setInterface(new GuiDesktop());
        FModel.initialize(null, null);

        Deck deck = DeckResolver.resolve(deckPath);
        List<RegisteredPlayer> seats = new ArrayList<>();
        RegisteredPlayer humanSeat = RegisteredPlayer.forCommander(deck);
        WebHumanLobbyPlayer humanLobby = new WebHumanLobbyPlayer("Flynn");
        humanSeat.setPlayer(humanLobby);
        seats.add(humanSeat);
        for (int n = 1; n <= 3; n++) {
            RegisteredPlayer seat = RegisteredPlayer.forCommander(deck);
            seat.setPlayer(GamePlayerUtil.createAiPlayer("Integration AI " + n, n + 1, ""));
            seats.add(seat);
        }

        GameRules rules = new GameRules(GameType.Commander);
        rules.setAppliedVariants(EnumSet.of(GameType.Commander));
        Match match = new Match(rules, seats, "Web human Forge integration");
        Game game = match.createGame();
        Player human = game.getPlayers().get(0);
        check(human.getController() instanceof WebHumanController, "Seat 0 is not a WebHumanController");
        check(!human.getController().isAI(), "Seat 0 must be a real human controller");
        WebHumanController controller = (WebHumanController) human.getController();

        // Install real Forge card objects into a controlled starting state. The
        // rule-engine actions under test are not performed by this setup code.
        game.setAge(GameStage.Play);
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, human);
        game.getPhaseHandler().onStackResolved();
        Card plains = card("Plains", human, game);
        Card memnite = card("Memnite", human, game); // zero-cost artifact creature
        human.getZone(ZoneType.Hand).add(plains);
        human.getZone(ZoneType.Hand).add(memnite);
        game.copyLastState();

        ExecutorService forgeThread = Executors.newSingleThreadExecutor(r -> {
            Thread thread = new Thread(r, "forge-web-human-integration");
            thread.setDaemon(true);
            return thread;
        });
        ScriptedDecisions script = new ScriptedDecisions(controller, game, memnite);
        try {
            // Forge asks for priority. The script chooses the real Plains land
            // ability, then the real Memnite spell ability, then passes. The
            // first game-thread call stops with Memnite on the actual stack.
            step(forgeThread, () -> game.getPhaseHandler().mainLoopStep());
            script.assertHealthy();
            check(script.landChosen, "Script did not choose Forge's Plains land ability");
            check(script.spellChosen, "Script did not choose Forge's Memnite spell ability");
            check(script.sawSpellOnStack, "Memnite was not observed on Forge's stack before resolution");
            check(contains(human, ZoneType.Battlefield, plains), "Forge did not move Plains to the battlefield");
            check(contains(game, ZoneType.Stack, memnite), "Forge did not put Memnite on the stack");
            check(human.getLandsPlayedThisTurn() == 1, "Forge did not record the land play");

            // The other three real Forge AI controllers pass. Their final pass
            // resolves the spell through Forge's stack, moving Memnite to play.
            passOpponentSeats(game, forgeThread);
            script.assertHealthy();
            check(game.getStack().isEmpty(), "Forge stack was not empty after resolution");
            check(contains(human, ZoneType.Battlefield, memnite), "Resolved Memnite did not enter the battlefield");
            check(!contains(human, ZoneType.Hand, memnite), "Resolved Memnite remained in hand");

            // A real, haste-bearing printed creature gives the combat path a
            // legal attacker without bypassing summoning sickness. As in Forge's
            // own controlled-state tests, only the initial battlefield fixture
            // is installed directly; attacker declaration and combat damage are
            // still driven by the normal phase handler and human input callbacks.
            Card goblin = card("Raging Goblin", human, game);
            goblin.setController(human, game.getNextTimestamp());
            human.getZone(ZoneType.Battlefield).add(goblin);
            game.copyLastState();
            Player defendingAi = game.getPlayers().get(1);
            int lifeBeforeCombat = defendingAi.getLife();

            // Advance through Forge's real COMBAT_BEGIN and DECLARE_ATTACKERS
            // turn-based actions. InputAttack is displayed by Forge and the
            // script selects the attacker through WebHumanGui/InputProxy.
            step(forgeThread, () -> game.getPhaseHandler().devAdvanceToPhase(PhaseType.COMBAT_DECLARE_ATTACKERS));
            script.assertHealthy();
            check(game.getPhaseHandler().getPhase() == PhaseType.COMBAT_DECLARE_ATTACKERS,
                    "Forge did not reach attacker declaration");
            check(game.getCombat() != null && game.getCombat().isAttacking(goblin, defendingAi),
                    "Forge did not declare Raging Goblin attacking Integration AI 1");

            passPriorityCycle(game, forgeThread);
            check(game.getPhaseHandler().getPhase() == PhaseType.COMBAT_DECLARE_BLOCKERS,
                    "Forge did not advance to blocker declaration after the attacker window");
            passPriorityCycle(game, forgeThread);
            check(game.getPhaseHandler().getPhase() == PhaseType.COMBAT_FIRST_STRIKE_DAMAGE,
                    "Forge did not advance to the first-strike damage step");

            // No first-strike creatures are involved. Forge's own no-priority
            // phase loop advances to regular combat damage and applies it.
            for (int n = 0; n < 4; n++) {
                step(forgeThread, () -> game.getPhaseHandler().mainLoopStep());
            }
            script.assertHealthy();
            check(game.getPhaseHandler().getPhase() == PhaseType.COMBAT_DAMAGE,
                    "Forge did not reach regular combat damage");
            check(defendingAi.getLife() == lifeBeforeCombat - goblin.getNetPower(),
                    "Forge did not apply Raging Goblin's actual unblocked combat damage");
            check(game.getCombat().isAttacking(goblin, defendingAi), "Combat assignment disappeared before damage");
        } finally {
            script.close();
            var pending = controller.decisions().snapshot();
            if (pending != null) controller.decisions().cancel(pending.id(), "integration test cleanup");
            forgeThread.shutdownNow();
        }
        script.assertHealthy();
    }

    private static Card card(String name, Player owner, Game game) {
        PaperCard paper = FModel.getMagicDb().getCommonCards().getCard(name);
        check(paper != null, "Real Forge card database is missing " + name);
        Card result = CardFactory.getCard(paper, owner, game);
        result.setGameTimestamp(game.getNextTimestamp());
        return result;
    }

    private static void passOpponentSeats(Game game, ExecutorService thread) throws Exception {
        for (int n = 1; n <= 3; n++) {
            check(game.getPhaseHandler().getPriorityPlayer() == game.getPlayers().get(n),
                    "Wrong Forge AI received priority at seat " + n);
            step(thread, () -> game.getPhaseHandler().mainLoopStep());
        }
    }

    private static void passPriorityCycle(Game game, ExecutorService thread) throws Exception {
        for (int n = 0; n < 4; n++) {
            check(game.getPhaseHandler().getPriorityPlayer() == game.getPlayers().get(n),
                    "Wrong player received priority at seat " + n);
            step(thread, () -> game.getPhaseHandler().mainLoopStep());
        }
    }

    private static void step(ExecutorService executor, CheckedRunnable action) throws Exception {
        Future<?> future = executor.submit(() -> {
            try {
                action.run();
            } catch (Exception failure) {
                throw new RuntimeException(failure);
            }
        });
        try {
            future.get(STEP_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (TimeoutException timeout) {
            future.cancel(true);
            throw new AssertionError("Forge game action did not complete within " + STEP_TIMEOUT_SECONDS + " seconds", timeout);
        }
    }

    @FunctionalInterface
    private interface CheckedRunnable { void run() throws Exception; }

    private static boolean contains(Player player, ZoneType zone, Card expected) {
        for (Card card : player.getCardsIn(zone)) {
            if (card == expected || card.getId() == expected.getId()) return true;
        }
        return false;
    }

    private static boolean contains(Game game, ZoneType zone, Card expected) {
        for (Card card : game.getCardsIn(zone)) {
            if (card == expected || card.getId() == expected.getId()) return true;
        }
        return false;
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /**
     * Responds only to Forge requests. It selects actual option IDs published
     * by WebHumanController/WebHumanGui; the test never calls PlaySpellAbility,
     * GameAction, Combat.addAttacker or a life-change API itself.
     */
    private static final class ScriptedDecisions implements AutoCloseable {
        private final WebHumanController controller;
        private final Game game;
        private final Card spell;
        private final AtomicBoolean running = new AtomicBoolean(true);
        private final Set<String> seenRequestIds = new HashSet<>();
        private final Thread thread;
        private volatile Throwable failure;
        private volatile boolean landChosen;
        private volatile boolean spellChosen;
        private volatile boolean sawSpellOnStack;
        private volatile boolean attackerChosen;
        private int priorityStage;

        private ScriptedDecisions(WebHumanController controller, Game game, Card spell) {
            this.controller = controller;
            this.game = game;
            this.spell = spell;
            thread = new Thread(this::run, "scripted-forge-human-decisions");
            thread.setDaemon(true);
            thread.start();
        }

        private void run() {
            while (running.get()) {
                try {
                    DecisionBroker.Request request = controller.decisions().snapshot();
                    if (request == null || !seenRequestIds.add(request.id())) {
                        Thread.sleep(4);
                        continue;
                    }
                    respond(request);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable problem) {
                    failure = problem;
                    DecisionBroker.Request pending = controller.decisions().snapshot();
                    if (pending != null) controller.decisions().cancel(pending.id(), "scripted test response failed: " + problem);
                    return;
                }
            }
        }

        private void respond(DecisionBroker.Request request) {
            String selected;
            if ("priority".equals(request.kind())) {
                if (priorityStage == 0) {
                    selected = find(request, "Play land: Plains");
                    landChosen = true;
                    priorityStage = 1;
                } else if (priorityStage == 1) {
                    selected = find(request, "Cast spell: Memnite");
                    spellChosen = true;
                    priorityStage = 2;
                } else if (priorityStage == 2) {
                    sawSpellOnStack = game.getStack().size() == 1 && contains(game, ZoneType.Stack, spell);
                    priorityStage = 3;
                    selected = "pass";
                } else {
                    selected = "pass";
                }
            } else if ("declare_attackers".equals(request.kind())) {
                if (!attackerChosen) {
                    selected = find(request, "Attack with Raging Goblin at Integration AI 1");
                    attackerChosen = true;
                } else {
                    selected = find(request, "Finish attacker declarations");
                }
            } else if ("confirm".equals(request.kind())) {
                selected = request.options().stream()
                        .filter(option -> "yes".equals(option.id()))
                        .map(DecisionBroker.Option::id)
                        .findFirst()
                        .orElseThrow(() -> new AssertionError("Forge confirm prompt omitted its yes option"));
            } else {
                throw new AssertionError("Unexpected Forge decision kind: " + request.kind() + " / " + request.message());
            }
            if (!controller.decisions().submit(request.id(), List.of(selected))) {
                throw new AssertionError("Scripted response rejected for " + request.kind() + " request " + request.id());
            }
        }

        private String find(DecisionBroker.Request request, String labelPrefix) {
            return request.options().stream()
                    .filter(option -> option.label().startsWith(labelPrefix))
                    .map(DecisionBroker.Option::id)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Forge did not offer '" + labelPrefix
                            + "'; received options: " + request.options()));
        }

        private void assertHealthy() {
            if (failure != null) throw new AssertionError("Scripted Forge human decision failed", failure);
        }

        @Override public void close() throws InterruptedException {
            running.set(false);
            thread.interrupt();
            thread.join(2_000);
        }
    }
}
