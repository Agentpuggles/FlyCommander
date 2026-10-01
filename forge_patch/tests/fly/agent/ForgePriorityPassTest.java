package fly.agent;

import forge.GuiDesktop;
import forge.deck.Deck;
import forge.game.*;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.RegisteredPlayer;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import forge.gui.GuiBase;
import forge.model.FModel;

import java.util.*;
import java.util.concurrent.*;

/** Real Forge PhaseHandler regression, run with the built jar + res/.
 * Controlled state setup follows upstream forge.ai.AITest; NOT a full match or
 * casting/payment test. No copied priority algorithm and no mocked Forge classes.
 */
public final class ForgePriorityPassTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }

    /** Counts actual controller calls; delegates all decisions to the real Fly. */
    private static final class CountingFlyLobby extends FlyLobbyPlayer {
        int calls;
        CountingFlyLobby(String name) { super(name); }
        @Override public Player createIngamePlayer(Game game, int seat) {
            Player player = new Player(getName(), game, seat);
            player.setFirstController(new FlyPlayerController(game, player, this) {
                @Override public List<SpellAbility> chooseSpellAbilityToPlay() {
                    if (++calls > 1) throw new AssertionError("Fly re-entered action loop instead of passing");
                    return super.chooseSpellAbilityToPlay();
                }
            });
            return player;
        }
    }

    public static void main(String[] args) {
        // Explicit exit also terminates Forge's desktop/background threads.
        try { run(); System.exit(0); }
        catch (Throwable failure) { failure.printStackTrace(); System.exit(1); }
    }

    private static void run() throws Exception {
        System.setProperty("fly.agent.controllerTest", "true");
        GuiBase.setInterface(new GuiDesktop());
        FModel.initialize(null, null);
        Deck deck = DeckResolver.resolve(args[0]);
        List<RegisteredPlayer> seats = new ArrayList<>();
        RegisteredPlayer humanSeat = RegisteredPlayer.forCommander(deck);
        humanSeat.setPlayer(new WebHumanLobbyPlayer("Flynn"));
        seats.add(humanSeat);
        List<CountingFlyLobby> flies = new ArrayList<>();
        for (int n = 1; n <= 3; n++) {
            CountingFlyLobby lobby = new CountingFlyLobby("Fly #" + n);
            flies.add(lobby);
            RegisteredPlayer seat = RegisteredPlayer.forCommander(deck);
            seat.setPlayer(lobby);
            seats.add(seat);
        }
        GameRules rules = new GameRules(GameType.Commander);
        rules.setAppliedVariants(EnumSet.of(GameType.Commander));
        Game game = new Match(rules, seats, "Priority regression").createGame();
        Player human = game.getPlayers().get(0);
        check(human.getController() instanceof WebHumanController, "Wrong human controller");
        check(!human.getController().isAI(), "Human must not be an AI");
        WebHumanController controller = (WebHumanController) human.getController();

        // Controlled initialized state as in upstream AITest, rather than starting
        // the full mulligan/game loop. The human is active so Fly passes are off-main
        // and do not need an external brain service or synthetic brain responses.
        game.setAge(GameStage.Play);
        game.getPhaseHandler().devModeSet(PhaseType.MAIN1, human);
        game.getPhaseHandler().onStackResolved();

        // A real printed card's ability already on the stack. We test priority and
        // resolution, NOT legal activation/cost payment; setup bypasses those costs.
        var paper = FModel.getMagicDb().getCommonCards().getCard("Fountain of Youth");
        check(paper != null, "Fountain of Youth missing from real card database");
        Card fountain = CardFactory.getCard(paper, human, game);
        fountain.setGameTimestamp(game.getNextTimestamp());
        human.getZone(ZoneType.Battlefield).add(fountain);
        SpellAbility gainLife = null;
        for (SpellAbility sa : fountain.getSpellAbilities())
            if (sa.getApi() == ApiType.GainLife) { gainLife = sa; break; }
        check(gainLife != null, "Real card GainLife ability missing");
        gainLife.setActivatingPlayer(human);
        game.getStack().add(gainLife);
        game.copyLastState();
        game.getPhaseHandler().setPriority(human);
        int life = human.getLife();
        check(game.getStack().size() == 1, "Ability not on Forge stack");

        ExecutorService gameThread = Executors.newSingleThreadExecutor();
        try {
            String first = humanPass(game, controller, gameThread, null);
            check(game.getStack().size() == 1 && human.getLife() == life,
                    "Stack resolved before opponents passed");
            passFlies(game, gameThread, flies);
            check(game.getStack().isEmpty() && human.getLife() == life + 1,
                    "All passes did not resolve the real ability exactly once");
            check(game.getPhaseHandler().getPhase() == PhaseType.MAIN1,
                    "Resolving a stack item must not also end the phase");
            check(game.getPhaseHandler().getPriorityPlayer() == human,
                    "Forge did not return priority to active player after resolution");

            String second = humanPass(game, controller, gameThread, first);
            passFlies(game, gameThread, flies);
            check(game.getPhaseHandler().getPhase() != PhaseType.MAIN1,
                    "Four passes on empty stack did not advance phase");
            check(game.getPhaseHandler().getPriorityPlayer() == human,
                    "Priority not returned to active player in new phase");
            humanPass(game, controller, gameThread, second);
            check(!game.isGameOver(), "Game unexpectedly ended");
            System.out.println("PASS: human response -> next player; three Fly passes -> stack resolution; "
                    + "fresh window -> phase advance; human priority legitimately returns");
        } finally {
            var pending = controller.decisions().snapshot();
            if (pending != null) controller.decisions().cancel(pending.id(), "test cleanup");
            gameThread.shutdownNow();
        }
    }

    private static String humanPass(Game game, WebHumanController controller,
                                     ExecutorService thread, String previousId) throws Exception {
        check(game.getPhaseHandler().getPriorityPlayer() == controller.getPlayer(), "Human not priority player");
        Future<?> step = thread.submit(() -> game.getPhaseHandler().mainLoopStep());
        DecisionBroker.Request request = null;
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (request == null && !step.isDone() && System.nanoTime() < deadline) {
            request = controller.decisions().snapshot();
            if (request == null) Thread.sleep(5);
        }
        if (step.isDone()) step.get(); // propagate actual Forge failures
        check(request != null && "priority".equals(request.kind()), "No real human priority request");
        check(!step.isDone(), "Human did not block before the response");
        check(!request.id().equals(previousId), "Request ID was reused");
        if (previousId != null)
            check(!WebHumanSession.submit(Map.of("id", previousId, "selected", List.of("pass"))), "Stale pass accepted");
        check(WebHumanSession.submit(Map.of("id", request.id(), "selected", List.of("pass"))), "Pass rejected");
        check(!WebHumanSession.submit(Map.of("id", request.id(), "selected", List.of("pass"))), "Duplicate pass accepted");
        try { step.get(10, TimeUnit.SECONDS); }
        catch (TimeoutException e) {
            throw new AssertionError("One accepted pass did not return from Forge mainLoopStep; likely [] instead of null", e);
        }
        check(controller.decisions().snapshot() == null, "Human was re-prompted in same action loop");
        check(game.getPhaseHandler().getPriorityPlayer() == game.getNextPlayerAfter(controller.getPlayer()),
                "Human pass did not move priority to next player");
        return request.id();
    }

    private static void passFlies(Game game, ExecutorService thread, List<CountingFlyLobby> flies) throws Exception {
        for (int n = 0; n < 3; n++) {
            Player expected = game.getPlayers().get(n + 1);
            check(game.getPhaseHandler().getPriorityPlayer() == expected, "Fly priority order wrong");
            CountingFlyLobby fly = flies.get(n);
            fly.calls = 0;
            int stackBefore = game.getStack().size();
            PhaseType phaseBefore = game.getPhaseHandler().getPhase();
            thread.submit(() -> game.getPhaseHandler().mainLoopStep()).get(10, TimeUnit.SECONDS);
            check(fly.calls == 1, "Fly failed to pass in one controller call");
            if (n < 2) {
                check(game.getStack().size() == stackBefore, "Stack changed before all players passed");
                check(game.getPhaseHandler().getPhase() == phaseBefore, "Phase ended before all players passed");
            }
        }
    }
}
