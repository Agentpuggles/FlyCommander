package fly.agent;

import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * Lobby-level seat for the human whose board is the physical table.
 * Install {@link PhysicalTableController} as the in-game controller.
 */
public class TableLobbyPlayer extends LobbyPlayerAi {

    public TableLobbyPlayer(String name) {
        super(name, null);
    }

    @Override
    public Player createIngamePlayer(Game game, int seat) {
        Player player = new Player(getName(), game, seat);
        PlayerController controller = new PhysicalTableController(game, player, this);
        player.setFirstController(controller);
        return player;
    }
}
