package fly.agent;

import forge.LobbyPlayer;
import forge.game.Game;
import forge.game.player.Player;
import forge.player.PlayerControllerHuman;

/** Real human controller path; deliberately not wired into the assisted launcher. */
public final class WebHumanController extends PlayerControllerHuman {
    private final DecisionBroker decisions = new DecisionBroker();
    public WebHumanController(Game game, Player player, LobbyPlayer lobby) {
        super(game, player, lobby);
        setGui(new WebHumanGui(decisions).create());
    }
    public DecisionBroker decisions() { return decisions; }
}
