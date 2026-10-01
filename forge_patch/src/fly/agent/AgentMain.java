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
 * fly brain (via FlyLobbyPlayer), remaining seats are stock Forge AI.
 *
 * Usage:
 *   java -cp patch:forge.jar fly.agent.AgentMain <games> <flySpec> <aiSpec>...
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
        boolean physical = Boolean.getBoolean("fly.agent.physical");
        boolean table = "1".equals(System.getProperty("fly.agent.table"));
        if (table && !Boolean.getBoolean("fly.agent.experimentalAssisted"))
            throw new IllegalArgumentException("Table mode requires -Dfly.agent.experimentalAssisted=true; not a full human controller");
        if (table && args.length != 5) throw new IllegalArgumentException("Table mode requires exactly three opponents");
        if (physical && args.length != 5) throw new IllegalArgumentException("Physical pod needs three Fly decks");
        int games = Integer.parseInt(args[0]);
        String flySpec = args[1];
        List<String> aiSpecs = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            aiSpecs.add(args[i]);
        }

        // --- Forge headless bootstrap (exact sequence of `sim` CLI mode) ---
        GuiBase.setInterface(new GuiDesktop());
        FModel.initialize(null, null);

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
        Deck flyDeck = DeckResolver.resolve(flySpec);
        System.out.println("[FlyAgent] fly deck: " + flyDeck.getName());

        List<Deck> aiDecks = new ArrayList<>();
        int aiIndex = 2;
        for (String spec : aiSpecs) {
            Deck d = DeckResolver.resolve(spec);
            aiDecks.add(d);
            System.out.println("[FlyAgent] ai deck " + aiIndex + ": " + d.getName());
            aiIndex++;
        }

        // --- Commander rules ------------------------------------------------
        GameRules rules = new GameRules(GameType.Commander);
        rules.setAppliedVariants(java.util.EnumSet.of(GameType.Commander));
        rules.setGamesPerMatch(1);

        forge.LobbyPlayer flyLobby = physical ? new WebHumanLobbyPlayer("Flynn") : table ? new TableLobbyPlayer("Paper human (AI-assisted)") : new FlyLobbyPlayer("FlyBrain");
        RegisteredPlayer flySeat = RegisteredPlayer.forCommander(flyDeck);
        flySeat.setPlayer(flyLobby);

        List<RegisteredPlayer> seats = new ArrayList<>();
        seats.add(flySeat);
        aiIndex = 2;
        for (Deck aiDeck : aiDecks) {
            RegisteredPlayer seat = RegisteredPlayer.forCommander(aiDeck);
            int flyNumber = aiIndex - 1;
            seat.setPlayer(physical ? new FlyLobbyPlayer("Fly #" + flyNumber,
                    "http://127.0.0.1:" + (Integer.getInteger("fly.agent.brainBasePort", 8792) + flyNumber - 1) + "/decide")
                    : GamePlayerUtil.createAiPlayer("Ai(" + aiIndex + ")-" + aiDeck.getName(), aiIndex, ""));
            seats.add(seat);
            aiIndex++;
        }

        Match match = new Match(rules, seats, "FlyCommander");
        AgentServer.setDecks(flyDeck.getName(),
                aiDecks.stream().map(Deck::getName).toList());
        AgentServer.markReady();

        int flyWins = 0;
        for (int g = 1; g <= games; g++) {
            System.out.println("[FlyAgent] === game " + g + "/" + games + " ===");
            AgentServer.beginGame(g);
            Game game = match.createGame();
            // register the in-game fly Player once created (seat 0)
            registerFlyPlayer(game);
            if (table) {
                for (var p : game.getPlayers()) if (p.getLobbyPlayer() instanceof TableLobbyPlayer) {
                    AgentGameState.register(game, p);
                    AgentGameState.registerTable(p);
                }
            }
            if (physical) {
                // Never let Forge deal an invented paper hand. Until the source
                // draw/library hook is installed this mode is a wiring diagnostic.
                WebHumanSession.fault("Physical library/draw source hook is not installed. Four seats created; game has NOT started. No paper hand was invented.");
                new java.util.concurrent.CountDownLatch(1).await();
            }
            match.startGame(game);

            if (table) HumanDecisionChannel.finish(TableSnapshot.capture(game, AgentGameState.tableSeat()));
            GameEndReason reason = game.getOutcome().getWinCondition();
            boolean flyWon = game.getOutcome().isWinner(flyLobby);
            if (flyWon) flyWins++;
            String resultJson = resultJson(g, flyWon, reason, game);
            AgentServer.recordResult(resultJson);
            System.out.println("[FlyAgent] game " + g + " result: " + resultJson);

            try {
                Thread.sleep(1000); // give the Python side a beat to poll
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println("[FlyAgent] FLY WIN RATE: " + flyWins + "/" + games);
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
}
