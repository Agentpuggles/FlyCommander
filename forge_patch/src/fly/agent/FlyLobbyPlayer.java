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

    private final String brainUrl;

    public FlyLobbyPlayer(String name) { this(name, null); }

    public FlyLobbyPlayer(String name, String brainUrl) {
        super(name, null);
        this.brainUrl = brainUrl;
    }

    @Override
    public Player createIngamePlayer(Game game, int seat) {
        Player player = new Player(getName(), game, seat);
        PlayerController flyController =
                new FlyPlayerController(game, player, this, brainUrl);
        player.setFirstController(flyController);
        return player;
    }
}
