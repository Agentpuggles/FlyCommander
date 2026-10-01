package fly.agent;

import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.combat.Combat;
import java.util.*;

/** Experimental assisted seat, NOT a full human controller.
 * Only stock-AI prepared plays are offered. All micro choices remain stock AI.
 * Approval precedes playChosenSpellAbility; no scanner mutation is consumed.
 */
public class PhysicalTableController extends PlayerControllerAi {
    private final Player me;
    public PhysicalTableController(Game game, Player player, forge.LobbyPlayer lobbyPlayer) {
        super(game, player, lobbyPlayer);
        me = player;
        getAi().setUseSimulation(null);
    }
    private String ask(String kind, String text, List<String> choices) {
        String choice = HumanDecisionChannel.ask(TableSnapshot.capture(getGame(), me), kind, text, choices);
        System.out.println("[Human " + kind + "] " + choice);
        return choice;
    }
    @Override public boolean mulliganKeepHand(Player firstPlayer, int cardsToReturn) {
        return ask("mulligan", "Keep this Forge hand? AI chooses any cards returned to the library (" + cardsToReturn + ").",
                List.of("Keep", "Mulligan")).equals("Keep");
    }
    @Override public List<SpellAbility> chooseSpellAbilityToPlay() {
        List<SpellAbility> proposed = super.chooseSpellAbilityToPlay();
        if (proposed == null) proposed = List.of();
        StringBuilder preview = new StringBuilder();
        for (SpellAbility sa : proposed) {
            preview.append(sa.isLandAbility() ? "Play land: " : sa.isSpell() ? "Cast: " : "Activate: ");
            preview.append(sa.getHostCard().getName()).append(" — ").append(sa.toString());
            for (SpellAbility sub = sa; sub != null; sub = sub.getSubAbility())
                if (sub.usesTargeting()) preview.append(" Targets: ").append(sub.getTargets());
            preview.append("\n");
        }
        List<String> choices = proposed.isEmpty() ? List.of("Pass priority") : List.of("Approve AI proposal", "Pass priority");
        String choice = ask("priority", proposed.isEmpty() ? "No AI-proposed play. This does not mean no legal plays exist." : preview.toString(), choices);
        return choice.equals("Approve AI proposal") ? proposed : List.of();
    }
    @Override public void declareAttackers(Player attacker, Combat combat) {
        ask("combat", "Forge AI will choose your attackers and defenders, including mandatory attacks. Manual combat is not implemented.", List.of("Delegate attacks to AI"));
        super.declareAttackers(attacker, combat);
    }
    @Override public void declareBlockers(Player defender, Combat combat) {
        ask("combat", "Forge AI will choose your blockers. Manual combat is not implemented.", List.of("Delegate blocks to AI"));
        super.declareBlockers(defender, combat);
    }
    public Player tablePlayer() { return me; }
}
