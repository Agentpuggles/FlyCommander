package fly.agent;

import forge.game.Game;
import forge.game.player.Player;
import forge.player.LobbyPlayerHuman;

/** A real human lobby/player factory, not an AI impersonating Flynn. */
public final class WebHumanLobbyPlayer extends LobbyPlayerHuman {
    public WebHumanLobbyPlayer(String name) { super(name); }
    @Override public Player createIngamePlayer(Game game, int id) {
        Player player = new Player(getName(), game, id);
        WebHumanController controller = new WebHumanController(game, player, this);
        player.setFirstController(controller);
        WebHumanSession.register(controller.decisions());
        return player;
    }
}
