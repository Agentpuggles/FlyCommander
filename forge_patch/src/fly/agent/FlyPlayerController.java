package fly.agent;

import forge.ai.AiController;
import forge.ai.AiPlayDecision;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseHandler;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * The fly's seat: a {@link PlayerControllerAi} whose macro decisions (which
 * main-phase play to make, whether to attack) are delegated to the external
 * fly brain via the agent server. All micro decisions (targets, blockers,
 * mana, damage assignment) are inherited from Forge's AI so the fly plays
 * legally while the brain owns strategy.
 */
public class FlyPlayerController extends PlayerControllerAi {

    /** Macro-action indices; must match the Python side. */
    public static final int ACT_PLAY = 0;      // AI's best main-phase play
    public static final int ACT_ATTACK = 1;    // attack with AI-recommended set
    public static final int ACT_HOLD = 2;      // do nothing
    public static final int ACT_INTERACT = 3;  // AI's removal/interaction pick

    private final Player me;
    private final String brainUrl;
    private String lastDecisionKey = "";
    private int decisionsThisPhase = 0;
    private static final int MAX_DECISIONS_PER_PHASE = 8;

    public FlyPlayerController(Game game, Player player, forge.LobbyPlayer lobbyPlayer) {
        this(game, player, lobbyPlayer, null);
    }

    public FlyPlayerController(Game game, Player player, forge.LobbyPlayer lobbyPlayer, String brainUrl) {
        super(game, player, lobbyPlayer);
        this.me = player;
        this.brainUrl = brainUrl;
        // Heuristics only — full/hybrid simulation is far too slow at every
        // priority pass. The fly plays fast, if imperfect.
        try {
            getAi().setUseSimulation(null);
        } catch (Throwable ignored) {
            // keep default behavior if the option API shifts
        }
    }

    /**
     * Called by the game loop every time the fly may act during main phases.
     * We consult the brain and translate its choice into concrete actions.
     */
    @Override
    public java.util.List<SpellAbility> chooseSpellAbilityToPlay() {
        // Gate: the brain only decides on the fly's own main phases. Everything
        // else (instants on others' turns, triggers, combat priority) passes
        // through instantly — a documented v1 limitation.
        PhaseHandler ph = getGame().getPhaseHandler();
        boolean myMain = ph.getPhase().isMain() && ph.getPlayerTurn() == me;
        String key = ph.getTurn() + ":" + ph.getPhase().name();
        if (!key.equals(lastDecisionKey)) {
            lastDecisionKey = key;
            decisionsThisPhase = 0;
        }
        if (!myMain || decisionsThisPhase >= MAX_DECISIONS_PER_PHASE
                || getGame().isGameOver()) {
            return java.util.List.of();
        }
        decisionsThisPhase++;

        // Snapshot for the brain (cheap, best-effort)
        AgentServer.pushObservation(GameObserver.snapshotJson(getGame(), me));

        String observation = GameObserver.snapshotJson(getGame(), me);
        String context = System.getProperty("fly.agent.decisionContext", "");
        int choice = brainUrl == null ? BrainClient.queryAction(observation, context)
                : BrainClient.queryAction(brainUrl, observation, context);
        AgentServer.recordDecision(choice);

        switch (choice) {
            case ACT_PLAY: {
                SpellAbility best = pickBestSa(false);
                if (best != null) {
                    AgentServer.recordAction("play:" + best.getHostCard().getName());
                    return java.util.List.of(best);
                }
                AgentServer.recordAction("play:none");
                return java.util.List.of(); // nothing worth playing → hold
            }
            case ACT_INTERACT: {
                SpellAbility sa = pickBestSa(true);
                if (sa != null) {
                    AgentServer.recordAction("interact:" + sa.getHostCard().getName());
                    return java.util.List.of(sa);
                }
                AgentServer.recordAction("interact:none");
                return java.util.List.of();
            }
            case ACT_ATTACK:
                // handled in declareAttackers(); pass priority for now
                AgentServer.recordAction("attack:deferred");
                return java.util.List.of();
            case ACT_HOLD:
            default:
                AgentServer.recordAction("hold");
                return java.util.List.of();
        }
    }

    /**
     * Attacks only when the brain said so; otherwise empty attack. Uses the
     * inherited AiController heuristics to choose *which* attackers.
     */
    @Override
    public void declareAttackers(Player attackingPlayer, forge.game.combat.Combat combat) {
        int last = AgentServer.lastDecision();
        if (last == ACT_ATTACK) {
            AgentServer.recordAction("attack:executed");
            super.declareAttackers(attackingPlayer, combat);
        } else {
            AgentServer.recordAction("attack:skipped");
            // leave combat empty
        }
    }

    // ------------------------------------------------------------------
    private SpellAbility pickBestSa(boolean interactionOnly) {
        AiController ai = getAi();
        List<SpellAbility> options = ai.chooseSpellAbilityToPlay();
        if (options == null || options.isEmpty()) {
            return null;
        }
        SpellAbility best = null;
        int bestScore = Integer.MIN_VALUE;
        for (SpellAbility sa : options) {
            int score = scoreAbility(ai, sa);
            if (interactionOnly && !isInteraction(sa)) {
                continue;
            }
            if (score > bestScore) {
                bestScore = score;
                best = sa;
            }
        }
        return best;
    }

    private int scoreAbility(AiController ai, SpellAbility sa) {
        AiPlayDecision d;
        try {
            d = ai.canPlaySa(sa);
        } catch (Throwable t) {
            return Integer.MIN_VALUE;
        }
        if (d == AiPlayDecision.WillPlay || d == AiPlayDecision.MandatoryPlay) return 10;
        if (d == AiPlayDecision.AddBoardPresence) return 8;
        if (d == AiPlayDecision.ImpactCombat) return 7;
        if (d == AiPlayDecision.Removal) return 9;
        if (d == AiPlayDecision.Tempo || d == AiPlayDecision.CardAdvantage) return 6;
        if (d == AiPlayDecision.WaitForCombat) return 2;
        return 0;
    }

    private static boolean isInteraction(SpellAbility sa) {
        if (sa.isSpell() && sa.getHostCard() != null && sa.getHostCard().isInstant()) {
            return true;
        }
        var api = sa.getApi();
        return api == forge.game.ability.ApiType.Destroy
                || api == forge.game.ability.ApiType.Counter
                || api == forge.game.ability.ApiType.DamageAll
                || api == forge.game.ability.ApiType.Sacrifice;
    }
}
