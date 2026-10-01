package fly.agent;

import forge.game.Game;
import forge.game.player.Player;

/** Static registry so HTTP handlers can reach the live game safely. */
final class AgentGameState {
    private static volatile Game game;
    private static volatile Player flyPlayer;
    private static volatile Player tableSeat;

    private AgentGameState() {}

    static void register(Game g, Player fly) {
        game = g;
        flyPlayer = fly;
    }

    /** Register the seat whose board the physical table mirrors. */
    static void registerTable(Player seat) {
        tableSeat = seat;
    }

    static boolean hasTableSeat() {
        return tableSeat != null;
    }

    static Game currentGame() {
        Game g = game;
        if (g == null) {
            throw new IllegalStateException("no game running yet");
        }
        return g;
    }

    static Player flyPlayer() {
        Player p = flyPlayer;
        if (p == null) {
            throw new IllegalStateException("fly player not registered yet");
        }
        return p;
    }

    /** May be null: a match can run without a physical table. */
    static Player tableSeat() {
        return tableSeat;
    }
}
