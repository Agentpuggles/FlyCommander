package fly.agent;

import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * The human's seat, driven by the physical table instead of by Forge's AI.
 *
 * The player's real board is observed by FlyCommander's vision pipeline and
 * posted to {@link AgentServer} as table actions. This controller is the point
 * where those actions enter the game: at every priority it drains the queue
 * (game thread, so no HTTP thread ever touches game state) and then **passes**
 * — it never plays a card on its own. Everything the seat does was done by the
 * human's hands on the table.
 *
 * Forge stays the rules engine: it validates each move (an illegal one is
 * rejected with a reason the physical table records), owns the stack, and
 * resolves combat. The AI seats opposite play their own game normally.
 */
public class PhysicalTableController extends PlayerControllerAi {

    private final Player me;

    public PhysicalTableController(Game game, Player player,
                                   forge.LobbyPlayer lobbyPlayer) {
        super(game, player, lobbyPlayer);
        this.me = player;
        try {
            getAi().setUseSimulation(null);
        } catch (Throwable ignored) {
            // keep default behavior if the option API shifts
        }
    }

    /** Apply queued physical actions, then pass priority. */
    @Override
    public List<SpellAbility> chooseSpellAbilityToPlay() {
        TableActionApplier.drain(me);
        return List.of();
    }

    /** Attackers were declared physically (cards turned sideways). */
    @Override
    public void declareAttackers(Player attackingPlayer,
                                 forge.game.combat.Combat combat) {
        TableActionApplier.drain(me);
    }

    public Player tablePlayer() {
        return me;
    }
}
