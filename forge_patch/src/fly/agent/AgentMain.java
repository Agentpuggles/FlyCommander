package fly.agent;

import forge.GuiDesktop;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.GameEndReason;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.player.RegisteredPlayer;
import forge.gui.GuiBase;
import forge.model.FModel;
import forge.player.GamePlayerUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * FlyCommander entry point. Boots Forge headless (no GUI), starts the HTTP
 * agent server, resolves deck specs through Forge's own Commander deck pool
 * (see {@link DeckResolver}), then runs N full Commander games: seat 1 is the
 * fly brain (via FlyLobbyPlayer), remaining seats are stock Forge AI. In
 * physical pod mode seat 0 is a real browser-controlled Forge human and seats
 * 1–3 are stock Forge AI by default; Fly opponents are optional.
 *
 * Usage:
 *   java -cp patch:forge.jar fly.agent.AgentMain <games> <humanOrFlySpec> <aiSpec>...
 *
 * where each deck spec is "random" (Forge's Commander Decks pool), a deck
 * name from that pool, or an explicit path to a .dck file.
 */
public final class AgentMain {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: AgentMain <games> <flySpec> <aiSpec> [...]");
            System.err.println("  deck spec: \"random\", a Forge Commander deck name, or a .dck path");
            System.exit(2);
        }
        boolean controllerTest = Boolean.getBoolean("fly.agent.controllerTest");
        boolean physical = Boolean.getBoolean("fly.agent.physical");
        boolean assistedHuman = Boolean.getBoolean("fly.agent.assistedHuman");
        boolean table = "1".equals(System.getProperty("fly.agent.table"));
        boolean flyOpponents = physical && "fly".equalsIgnoreCase(System.getProperty("fly.agent.opponents", "forge"));
        if (assistedHuman && !physical) throw new IllegalArgumentException("assistedHuman requires physical pod mode");
        if (assistedHuman && controllerTest) throw new IllegalArgumentException("Choose assistedHuman or controllerTest, not both");
        if (table && !Boolean.getBoolean("fly.agent.experimentalAssisted"))
            throw new IllegalArgumentException("Table mode requires -Dfly.agent.experimentalAssisted=true; not a full human controller");
        if (table && args.length != 5) throw new IllegalArgumentException("Table mode requires exactly three opponents");
        if (physical && args.length != 5) throw new IllegalArgumentException("Human Commander pod needs exactly three opponent decks");
        if (controllerTest && !physical) throw new IllegalArgumentException("controllerTest requires physical seat mode");
        if (physical && table) throw new IllegalArgumentException("Choose physical or legacy table mode, not both");
        int games = Integer.parseInt(args[0]);
        String flySpec = args[1];
        List<String> aiSpecs = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            aiSpecs.add(args[i]);
        }

        // --- Forge headless bootstrap (exact sequence of `sim` CLI mode) ---
        GuiBase.setInterface(new GuiDesktop());
        System.out.println("[RuntimeTest] Initializing Forge 2.0.15");
        FModel.initialize(null, null);
        System.out.println("[RuntimeTest] Forge initialization returned");

        String brainUrl = System.getProperty("fly.agent.brainUrl");
        if (brainUrl != null) {
            BrainClient.setBrainUrl(brainUrl);
        }

        int agentPort = Integer.getInteger("fly.agent.port", 8791);
        AgentServer.start(agentPort);

        // Optional deterministic seeding for reproducible random pools
        String seedProp = System.getProperty("fly.agent.seed");
        if (seedProp != null) {
            try {
                forge.util.MyRandom.setRandom(new java.util.Random(Long.parseLong(seedProp)));
                System.out.println("[FlyAgent] random pool seeded with " + seedProp);
            } catch (NumberFormatException nfe) {
                System.err.println("[FlyAgent] ignoring bad seed '" + seedProp + "'");
            }
        }

        // --- decks: resolved through Forge's own pool/APIs -------------------
        System.out.println("[FlyAgent] commander pool size: " + DeckResolver.poolSize());
        Deck primaryDeck = DeckResolver.resolve(flySpec);
        System.out.println("[FlyAgent] " + (physical ? "human" : "fly") + " deck: " + primaryDeck.getName());

        List<Deck> aiDecks = new ArrayList<>();
        int aiIndex = 2;
        for (String spec : aiSpecs) {
            Deck d = DeckResolver.resolve(spec);
            aiDecks.add(d);
            System.out.println("[FlyAgent] opponent deck " + (aiIndex - 1) + ": " + d.getName());
            aiIndex++;
        }

        // --- Commander rules ------------------------------------------------
        GameRules rules = new GameRules(GameType.Commander);
        rules.setAppliedVariants(java.util.EnumSet.of(GameType.Commander));
        rules.setGamesPerMatch(1);

        forge.LobbyPlayer primaryLobby = physical
                ? (assistedHuman ? new TableLobbyPlayer("Paper human (legacy AI-assisted)") : new WebHumanLobbyPlayer("Flynn"))
                : table ? new TableLobbyPlayer("Paper human (legacy AI-assisted)") : new FlyLobbyPlayer("FlyBrain");
        RegisteredPlayer primarySeat = RegisteredPlayer.forCommander(primaryDeck);
        primarySeat.setPlayer(primaryLobby);

        List<RegisteredPlayer> seats = new ArrayList<>();
        seats.add(primarySeat);
        aiIndex = 2;
        for (Deck aiDeck : aiDecks) {
            RegisteredPlayer seat = RegisteredPlayer.forCommander(aiDeck);
            int opponentNumber = aiIndex - 1;
            if (physical && flyOpponents) {
                int port = Integer.getInteger("fly.agent.brainBasePort", 8792) + opponentNumber - 1;
                seat.setPlayer(new FlyLobbyPlayer("Fly #" + opponentNumber,
                        "http://127.0.0.1:" + port + "/decide"));
            } else if (physical) {
                seat.setPlayer(GamePlayerUtil.createAiPlayer("Forge AI #" + opponentNumber, aiIndex, ""));
            } else {
                seat.setPlayer(GamePlayerUtil.createAiPlayer("Ai(" + aiIndex + ")-" + aiDeck.getName(), aiIndex, ""));
            }
            seats.add(seat);
            aiIndex++;
        }

        Match match = new Match(rules, seats, "FlyCommander");
        AgentServer.setDecks(primaryDeck.getName(),
                aiDecks.stream().map(Deck::getName).toList());
        AgentServer.markReady();

        int flyWins = 0;
        for (int g = 1; g <= games; g++) {
            System.out.println("[FlyAgent] === game " + g + "/" + games + " ===");
            AgentServer.beginGame(g);
            Game game = match.createGame();
            // Register any optional Fly seat for the existing observer path.
            registerFlyPlayer(game);
            Player humanSeat = null;
            if (physical && !assistedHuman && !table) {
                for (var p : game.getPlayers()) {
                    if (p.getLobbyPlayer() instanceof WebHumanLobbyPlayer) {
                        humanSeat = p;
                        WebHumanSession.updateGame(game, humanSeat);
                        break;
                    }
                }
            }
            if (table || assistedHuman) {
                for (var p : game.getPlayers()) if (p.getLobbyPlayer() instanceof TableLobbyPlayer) {
                    AgentGameState.registerTable(p);
                }
                Player tableSeat = AgentGameState.tableSeat();
                if (tableSeat != null) HumanDecisionChannel.publish(TableSnapshot.capture(game, tableSeat));
            }
            for (var p : game.getPlayers())
                System.out.println("[RuntimeTest] Seat " + p.getId() + ": " + p.getName() + " / " + p.getController().getClass().getName());
            if (physical && controllerTest && flyOpponents) {
                System.out.println("[RuntimeTest] CONTROLLER TEST ONLY: Forge-generated hands, NOT physical library gameplay");
                for (int n = 1; n <= 3; n++) {
                    String url = "http://127.0.0.1:" + (Integer.getInteger("fly.agent.brainBasePort", 8792) + n - 1) + "/stats";
                    var response = java.net.http.HttpClient.newHttpClient().send(
                            java.net.http.HttpRequest.newBuilder(java.net.URI.create(url)).timeout(java.time.Duration.ofSeconds(5)).GET().build(),
                            java.net.http.HttpResponse.BodyHandlers.ofString());
                    if (response.statusCode() != 200) throw new IllegalStateException("Fly #" + n + " health failed");
                    System.out.println("[RuntimeTest] Fly #" + n + " connected: " + url + " " + response.body());
                }
            }
            System.out.println("[RuntimeTest] Calling Match.startGame");
            try {
                match.startGame(game);
            } catch (Throwable failure) {
                failure.printStackTrace();
                String fault = "Forge runtime stopped: " + failure;
                if (assistedHuman) HumanDecisionChannel.fault(fault);
                else WebHumanSession.fault(fault);
                // Retain HTTP error evidence for inspection, without continuing the game.
                new java.util.concurrent.CountDownLatch(1).await();
                return;
            }

            if (table || assistedHuman) HumanDecisionChannel.finish(TableSnapshot.capture(game, AgentGameState.tableSeat()));
            GameEndReason reason = game.getOutcome().getWinCondition();
            boolean primaryWon = game.getOutcome().isWinner(primaryLobby);
            boolean humanPod = physical && !assistedHuman && !table;
            boolean humanWon = physical && primaryWon;
            boolean flyWon = !physical && !table && primaryWon;
            if (humanPod) WebHumanSession.finish(game, humanSeat, humanWon, reason.name());
            if (flyWon) flyWins++;
            String resultJson = (assistedHuman || humanPod)
                    ? assistedResultJson(g, humanWon || (assistedHuman && primaryWon), reason, game)
                    : resultJson(g, flyWon, reason, game);
            AgentServer.recordResult(resultJson);
            System.out.println(humanPod
                    ? "[FlyAgent] digital human result: " + resultJson
                    : assistedHuman
                            ? "[FlyAgent] legacy paper-assist human result: " + resultJson
                            : "[FlyAgent] game " + g + " result: " + resultJson);

            try {
                Thread.sleep(1000); // give the Python side a beat to poll
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }

        if (physical && !assistedHuman && !table) System.out.println("[FlyAgent] digital human Commander match finished.");
        else if (assistedHuman) System.out.println("[FlyAgent] legacy paper-assist match finished.");
        else System.out.println("[FlyAgent] FLY WIN RATE: " + flyWins + "/" + games);
        Thread.sleep(15000); // let the Python side fetch /result before exit
        System.exit(0);
    }

    private static void registerFlyPlayer(Game game) {
        // The fly's RegisteredPlayer is seat 0; its in-game Player is created
        // by Game's construction from the RegisteredPlayer list.
        for (forge.game.player.Player p : game.getPlayers()) {
            if (p.getLobbyPlayer() instanceof FlyLobbyPlayer) {
                AgentGameState.register(game, p);
                return;
            }
        }
    }

    private static String resultJson(int game, boolean flyWon,
                                     GameEndReason reason, Game g) {
        return "{\"game\":" + game
                + ",\"flyWon\":" + flyWon
                + ",\"reason\":\"" + reason + '"'
                + ",\"turns\":" + g.getOutcome().getLastTurnNumber()
                + '}';
    }

    private static String assistedResultJson(int game, boolean humanWon,
                                             GameEndReason reason, Game g) {
        return "{\"game\":" + game
                + ",\"humanWon\":" + humanWon
                + ",\"reason\":\"" + reason + '"'
                + ",\"turns\":" + g.getOutcome().getLastTurnNumber()
                + '}';
    }
}
