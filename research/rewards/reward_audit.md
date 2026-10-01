# Reward Function Audit

Status: RESEARCH DOCUMENT (audit only; no code changed).
Last verified: 2026-10-01. Evidence: full read of
[flycommander/reward_shaping.py](../../flycommander/reward_shaping.py) and
[brain/dopamine_plasticity.py](../../brain/dopamine_plasticity.py).

---

## 1. Exact mechanics (verified line-by-line)

Per-step (computed vs a `_prev` snapshot of the previous observation):

| Component | Formula | Weight |
|---|---|---|
| `life_delta` | `(Δfly_life − 0.5 · Δopp_best_life)` | 0.02 / life point |
| `board_delta` | `Δ(board_adv)`, `board_adv = len(fly.board) − mean(len(opp boards))` | 0.05 / permanent |
| `hand_delta` | `Δ(fly hand size)` | 0.01 / card |
| `opp_removal` | `max(0, prev_opp_creatures − opp_creatures)` summed over opponents | 0.15 / creature |
| `commander_damage` | `max(0, Δfly commander damage taken)` | −0.10 / point |
| `time_penalty` | every step | −0.001 |

- `opp_best_life = max(opponent life totals)`.
- Total = sum of components + time_penalty, **clipped to [−0.5, +0.5]**.
- Terminal: draw iff `"draw" in reason.lower()`; else `+1` win / `−1` loss;
  winning reward decayed by `max(0.5, 1 − 0.01 · turns)`.
- RPE for plasticity: `r − value_baseline`; scalar baseline creeps by
  `0.01 · RPE` per update. **No discount factor γ, no bootstrapping, no
  return-based credit anywhere.**

## 2. Exploitable shortcuts / reward-hacking vectors

These are hypotheses to be *tested* (see E-RL-01..04), not asserted:

1. **Free removal credit.** `opp_removal` fires when opponent creature counts
   drop *for any reason*: they sacrificed, clashed with each other, died to
   SBA (legend rule), blocked each other, or died blocking the *fly's*
   attacks. A purely passive fly gets +0.15 per opponent creature that dies
   of natural causes. In a 4-player game opponents kill each other constantly
   — the fly can farm this by *staying alive and doing nothing*.
2. **Board-advantage by attrition.** Same structure: opponents' boards
   shrinking (any cause) raises `board_adv` → +0.05/permanent without action.
3. **Hand-size farming.** Draw engines yield +0.01/card; time penalty is only
   −0.001, so a slow draw loop nets ≈ +0.009/step indefinitely (clipped, but
   the clip only caps magnitude, not sign). Combined with (1) and (2),
   "turtle and draw" may be the highest-return policy short of winning.
4. **`max()` opponent-life blind spot.** `opp_best_life = max(...)` means
   damage to the *lower-life* opponents produces zero reward while the
   highest-life opponent is untouched. Damage must be focused on the
   healthiest opponent to score — the opposite of sound multiplayer strategy
   (usually you pressure the weakest/threat).
5. **Commander-damage win path is barely incentivized.** Taking commander
   damage is punished (−0.10/pt) but *dealing* it is not a component; a
   21-commander-damage kill scores only through `life_delta`'s 0.5-discounted
   channel. Meanwhile the encoder tracks commander damage per seat — the
   state knows about a dimension the reward nearly ignores.
6. **Clip distortion.** Big swings (25+ life swing alone saturates ±0.5) lose
   relative magnitude information; a board wipe (−0.05 × 15 permanents = −0.75
   → clipped −0.5) reads the same as −0.5 of life loss.
7. **`board` = ALL permanents.** Docstring says "creatures + small weight for
   other permanents"; code is `len(board)` — every land drop is +0.05.
   Ramp/play-density is over-rewarded relative to combat relevance.
   (Comment/code mismatch — flagged in claims RL-001 notes.)
8. **Draw detection by substring.** `is_draw` = `"draw" in reason.lower()` —
   fragile if Forge reason strings ever contain "draw" for non-draw endings.
   Cross-check terminal payload against Forge's actual game-end reasons.

## 3. The terminal-mapping anomaly (highest priority)

Logs show one episode summary with `flyWon: false, reason: "AllOpponentsLost"`.
By name, "AllOpponentsLost" should mean the fly won. If the mapping between
Forge's game-end state and `flyWon` is inverted or miswired in
`AgentServer`/`BrainClient`, then **every terminal reward is flipped** — the
single most damaging possible bug in the learning system (a systematic −1 for
wins trains the fly to lose). Status: observed once, not yet reproduced;
MED confidence. First experiment in the roadmap (E-RL-01) forces wins/losses
and inspects the terminal payload.

## 4. Structural gaps vs conventional RL

- **No γ, no value function, no bootstrapping.** Terminal ±1 lands only on
  KCs active at the *final* decision. Earlier decisions get credit only via
  step-shaped rewards. Multi-step tactics (ramp → sweep → attack) are
  invisible to the plasticity rule except through the step components.
- **Scalar baseline** cannot model state-dependent value (`value_estimate()`
  exists but is *unused* in plasticity — dead code path).
- **Confounded credit:** rewards include consequences of Forge-AI decisions
  (blockers, targets, damage assignment) that the fly doesn't control —
  variance, and possibly spurious state-reward correlations (e.g., Forge AI
  blocks less when fly's board is big → fly's big-board states get
  systematically more removal credit).
- **Nonstationary opponents** (Forge AI + other AI decks) and a single
  opponent pool → overfitting risk to pool idiosyncrasies.

## 5. What the reward *cannot* see at all

Mana efficiency, tempo (turn-level initiative), board quality (a 10/10 vs a
0/1 both count 1), card *quality* (hand content), political standing, threat
assessment, opponent elimination events (only via terminal), poison progress
as a reward signal (encoder has it; reward doesn't).

## 6. Experiment hooks

E-RL-01 terminal-mapping reproduction; E-RL-02 passive-fly reward farming
probe (hold-only agent, measure cumulative reward over N games); E-RL-03
reward-component ablation (remove opp_removal / board_delta and compare
learning curves vs full reward); E-RL-04 weight sensitivity sweep. Details in
[experiments/roadmap.md](../experiments/roadmap.md).
