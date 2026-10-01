# Action Space Audit

Status: RESEARCH DOCUMENT (audit only; no code changed).
Last verified: 2026-10-01. Evidence: [brain/mushroom_body.py](../../brain/mushroom_body.py),
[forge_patch/src/fly/agent/FlyPlayerController.java](../../forge_patch/src/fly/agent/FlyPlayerController.java),
[research/forge/forge_internals.md](../forge/forge_internals.md).

---

## 1. The four macro actions

| Index | Name | Claimed semantics | Actual mechanism |
|---|---|---|---|
| 0 | `play` | cast/play best spell or land | Fly's `chooseSpellAbilityToPlay` override accepts/uses Forge's top-scored candidate in the fly's own main phases (≤8 activations/phase) |
| 1 | `attack` | attack with recommended attackers | Sets `last decision = ACT_ATTACK`; the *next* `declareAttackers` call is allowed to return Forge's recommended attacker set; otherwise delegates |
| 2 | `hold` | do nothing this phase/window | Pass priority / decline |
| 3 | `interact` | use Forge's removal/interaction pick | Accept Forge-proposed interaction candidate; if none proposed, the server holds silently |

Key mechanics:

- Actions are sampled from a masked softmax over MBON-derived scores; empty
  legal mask → deterministic hold (`decide` returns 2).
- `action_eligibility` (KC spikes at decision time) is captured for DPR.
- Actions are **per decision window** (each priority window the controller is
  polled), not per turn; "play then attack" emerges across windows, and the
  ≤8/phase cap bounds how many windows produce plays.

## 2. Who actually decides (boundary summary)

Fly decides: *whether* to accept Forge's suggested play, whether to bless the
attack, whether to pass, whether to accept Forge's removal pick.

Forge AI decides everything else, including:
- target selection for the accepted spell
- **which opponents get attacked** (defender assignment inside the attacker set)
- blocking (both directions), damage assignment order
- mana spending/ordering, land-drop timing beyond the proposal
- instant-speed responses (fly has no window where it can respond at all —
  `chooseSpellAbilityToPlay` is restricted to own main phases)
- mulligans, trigger ordering, mode choices, X-values, alternative costs

**Multiplayer-specific hole:** choosing *whom to attack* is arguably the most
political decision in Commander; it is entirely inside Forge's proposal. The
fly cannot express a threat-assessment or coalition strategy even in principle
with this action space.

## 3. Markov sufficiency problems

- Legality of `play` depends on mana, which the state does not encode → the
  same encoded state can make `play` legal in one game and illegal in
  another. Masked-softmax learning treats legality as exogenous noise.
- `interact`'s meaning depends on Forge's current proposal (which card? what
  target?), invisible in the observation. The same action index can be
  "kill their commander" or "kill a 1/1" — the policy cannot distinguish.
- `attack`'s outcome depends on who blocks — decided later by Forge AI.
- The encoder has no channel for "what did I just do" (no action-history
  features), so credit for sequences rests entirely on KC-overlap between
  successive states.

## 4. Masking and learning bias

- Masked actions get −inf scores → zero probability → never sampled → never
  reinforced. No bias *against* legality per se, but state-conditional
  legality (mana, phase) is unobservable to the policy, so value estimates
  conflate "good action" with "happened to be legal."
- Silent-hold fallback on empty `interact` proposals (claims ACT-003) means
  "fly chose hold" and "no interaction existed" are indistinguishable in the
  decision log — corrupts action-preference statistics unless the proposal
  emptiness is logged (currently it is not).

## 5. Is four actions enough?

No — and the honest framing is *which* decisions to move into the fly first,
cheapest-to-highest value:

1. **Target choice** for the accepted spell (candidates exist in Forge's
   targeting pipeline; exposing k targets per spell is a modest patch).
2. **Attack target** (which opponent) — pure political value, low engine risk.
3. **Block/no-block** on Forge's behalf when the fly defends (currently pure
   Forge AI; also a confound source for the fly's own reward).
4. Instant-speed windows (largest change; requires controller restructuring
   to poll the fly on others' turns).

Each widening changes the candidate stream contract with the Java side —
coordinate with [research/forge/forge_internals.md](../forge/forge_internals.md).

## 6. Experiment hooks

E-ACT-01: log proposal-stream composition (how often are 2+ candidates
actually available per window? if ~1, the fly's "choice" is mostly accept/
decline and widening actions matters less than widening proposals).
E-ACT-02: attack-target ablation (fly chooses defender vs Forge default).
See [experiments/roadmap.md](../experiments/roadmap.md).
