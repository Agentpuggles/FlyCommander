package fly.agent;

import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.player.PlayerController;

/**
 * Lobby-level fly player. Subclasses LobbyPlayerAi so deck/profile plumbing
 * works unchanged, but installs {@link FlyPlayerController} as the in-game
 * controller so the fly brain owns macro decisions.
 */
public class FlyLobbyPlayer extends LobbyPlayerAi {

    public FlyLobbyPlayer(String name) {
        super(name, null);
    }

    @Override
    public Player createIngamePlayer(Game game, int seat) {
        Player player = new Player(getName(), game, seat);
        PlayerController flyController =
                new FlyPlayerController(game, player, this);
        player.setFirstController(flyController);
        return player;
    }
}
