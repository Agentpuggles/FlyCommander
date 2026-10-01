# Experiment Roadmap

Status: PLANNING DOCUMENT. Created 2026-10-01.
Conventions: each experiment states Hypothesis / Independent Variable /
Dependent Variable / Controls / Runs & Seeds / Expected / Interpretation.
Grouped by purpose. IDs referenced from claims DB and gap docs.

---

# 1. Correctness (gate: do these FIRST)

## E-RL-01 Terminal-mapping reproduction
- **H:** The observed `flyWon=false, reason=AllOpponentsLost` is a mapping bug
  (or the label is correct and reason is misleading).
- **IV:** forced game endings (fly wins by damage, fly loses, draw).
- **DV:** terminal JSON payload fields (`flyWon`, `reason`) vs ground truth.
- **Control:** read the win-detection code path; run 3 scripted endings ×10.
- **Expected:** if fields disagree with ground truth → bug confirmed.
- **Interpretation:** if mapping inverted → all past training suspect; fix
  before any other experiment. If correct → record reason-string taxonomy.

## E-TRAIN-01 Checkpoint roundtrip integrity
- **H:** Loading a checkpoint under a different connectome seed corrupts
  policy silently.
- **IV:** load fly_demo.npz with seed 2049 vs seed 777.
- **DV:** action distribution on a fixed probe state; KC→MBON shape checks.
- **Expected:** same seed → identical distributions; different seed →
  garbage (proves the trap exists).
- **Output:** justification for adding full connectome + manifest to saves.

## E-ACT-01 Proposal-stream census
- **H:** Most decision windows offer ≤1 candidate, i.e. the fly's real
  choice is mostly accept/decline.
- **IV:** none (observational).
- **DV:** distribution of candidate counts per window over ≥100 games.
- **Expected:** majority single-candidate; if so, widening *actions* matters
  less than widening *proposals* — informs E-ACT-02.

# 2. Scientific validation (neuroscience honesty)

## E-NEU-01 KC scale sweep
- **H:** KC count (500–8000) doesn't change competence once fan-in and WTA
  fraction are fixed — i.e., 4064 is not load-bearing.
- **IV:** n_kc ∈ {500, 1000, 2000, 4064, 8000}.
- **DV:** win rate vs fixed opponent; encoder-probe separation.
- **Control:** same seed family, same reward; ≥5 seeds each.
- **Expected:** flat response; if 2000 ≈ biology ≈ 4064, relabel counts or
  rescale (claims NEU-001).

## E-NEU-02 Structured vs random DAN→MBON
- **H:** A compartment-structured dan_to_mbon (per-compartment valence,
  hemibrain-derived) learns faster/robustly vs random matrix.
- **IV:** dan_to_mbon ∈ {random signed split, structured, hemibrain-real}.
- **DV:** learning curves; forgetting under opponent switch.
- **Expected:** structured ≥ random; if no difference, compartmental story
  is irrelevant at 4 actions — document as EA not BS.

## E-SNN-01 LIF vs static activation
- **H:** LIF rates(8) ≡ static activation for this pipeline.
- **IV:** readout ∈ {LIF rates(8), ReLU(state-projected), tanh}.
- **DV:** win rate; KC-code overlap metrics; wall time.
- **Expected:** equivalent; informs whether "spiking" is load-bearing.

# 3. Learning validation

## E-RL-02 Passive-fly reward farming probe
- **H:** A hold-only agent accumulates positive reward per game via
  opp_removal/board_delta in 4-player games.
- **IV:** agent ∈ {hold-only, random, trained fly}.
- **DV:** mean cumulative step reward per game (exclude terminal).
- **Expected:** hold-only > 0 confirms farming gradient exists → fix
  reward (E-RL-03) or accept as known exploit.

## E-RL-03 Reward-component ablation
- **H:** Removing farmable components improves learned policy quality.
- **IV:** reward ∈ {full, −opp_removal, −board_delta, −hand_delta,
  terminal-only, max→sum opp-life}.
- **DV:** win rate vs held-out decks; entropy; farming metric from E-RL-02.
- **Control:** identical seeds/arm; ≥5 seeds × ≥300 games.
- **Expected:** terminal-only ≯ full on 10³ games (sparse); −opp_removal
  best balance; `max→sum` changes targeting behavior measurably.

## E-RL-04 Weight sensitivity sweep
- **H:** Results are robust to ±50% weight changes except opp_removal.
- **IV:** each weight × {0.5, 1, 2}.
- **DV:** win rate; reward decomposition drift.
- **Expected:** insensitive except removal/board weights.

## E-RL-05 State-dependent baseline
- **H:** A linear state-value baseline (on 64-dim state) reduces variance
  and speeds learning vs scalar baseline.
- **IV:** baseline ∈ {scalar creep, linear critic, value_estimate() wired}.
- **DV:** learning-curve AUC; variance of RPE sequence.
- **Expected:** linear critic > scalar; minimal code change.

## E-RL-06 Conventional-RL yardstick
- **H:** The MB agent is within O(1) of tabular Q-learning on identical
  features/actions (else the architecture is actively harmful).
- **IV:** learner ∈ {MB+DPR, tabular Q (ε-greedy) on same state}.
- **DV:** win rate; sample efficiency to 50% vs fixed opponent.
- **Expected:** comparable; large gap → investigate representation, not
  learning rule.

## E-SNN-03 Eligibility traces
- **H:** Trace decay ~0.9/step on KC activity materially improves
  multi-step credit (ramp→attack sequences).
- **IV:** trace ∈ {none, 0.8, 0.9, 0.95}.
- **DV:** win rate; diagnostic: reward assigned to decisions 5+ steps before
  terminal.
- **Expected:** 0.9 best; none worst.

## E-RL-07 Forgetting / reversal
- **H:** After training vs pool A, switching to pool B degrades A-performance
  (no replay); LTM blend slows degradation.
- **IV:** consolidation ∈ {0.5 (current), 0.0, 0.9}.
- **DV:** win rate on A after B-training.
- **Expected:** 0.9 retains more; trade-off vs adapting to B.

# 4. Performance

## E-PERF-01 Bottleneck profile
- **H:** Decision round-trip (HTTP+JSON) dominates per-game wall time, not
  brain compute.
- **IV:** none (measurement).
- **DV:** ms per decision split {Java obs build, HTTP, JSON parse, encode,
  MB forward, plasticity}; games/hour end-to-end.
- **Expected:** HTTP+JSON > 50% of round-trip; brain < 5 ms.
- **Output:** informs whether batch-observations or persistent socket is
  the right optimization (do NOT optimize before measuring).

## E-TRAIN-02 Seed-pinned reproducibility
- **H:** With per-game MyRandom seeding (small patch), same seed → identical
  game trajectory (same terminal state hash).
- **IV:** seed pinned vs not.
- **DV:** trajectory hash equality across 2 runs.
- **Expected:** pinned → equal; unpinned → different. Prerequisite for
  honest A/B experiments.

# 5. Physical-table robustness

## E-PHYS-01 OCR error benchmark
- **H:** Collector-number OCR ≥95% top-1 on clean flat scans degrades <70%
  on foils/sleeves/angles.
- **IV:** image condition {flat clean, sleeve, foil, 15° tilt, shadow}.
- **DV:** top-1 accuracy; Scryfall match rate; per-condition confusion
  examples.
- **Corpus:** ≥200 real photos × conditions; ground truth by hand.
- **Expected:** flat ≈95%+, foil/sleeve worst. Informs fallback design
  (candidate-set + human confirm).

## E-PHYS-02 Tap-detection benchmark
- **H:** IoU tracking + orientation threshold detects taps with ≥90%
  precision/recall on labeled video.
- **IV:** threshold parameter.
- **DV:** precision/recall on ≥20 annotated clips.
- **Expected:** precision high, recall drops on partial occlusion; document
  failure modes.

## E-PHYS-03 Sim/physical observation equivalence
- **H:** Same game state produces identical 64-dim states in Forge and
  physical mode (channel semantics match).
- **IV:** none (equivalence test).
- **DV:** per-channel diffs over ≥50 constructed scenarios.
- **Expected:** mismatches found in commandZone/commanderDamage semantics →
  fix physical obs builder (bug discovery, not tuning).

## E-PHYS-04 Impossible-transition fuzz
- **H:** PhysicalGameState rejects impossible transitions (life decreases by
  0 or negative, card appears in two zones).
- **IV:** fuzzed event streams (property-based).
- **DV:** fraction of invalid streams accepted.
- **Expected:** current code accepts some invalid sequences → validation
  gap documented.

# 6. Generalization

## E-EVAL-01 Held-out deck generalization
- **H:** Win rate on held-out decks is 5–15 points below training decks.
- **IV:** deck ∈ {training pool, held-out pool} at eval time only.
- **DV:** win rate (≥300 games/arm/seed, ≥5 seeds).
- **Expected:** gap exists; size quantifies overfitting to pool.

## E-EVAL-02 Full ablation suite (frozen protocol)
- Combines E-RL-03/05/06, E-SNN-01/03, E-NEU-01/02 under one frozen eval
  harness with the controls from evaluation/evaluation_protocol.md §2 —
  this is the "paper-grade" run; do not run before E-RL-01 passes.

---

## Execution order (dependency-aware)

1. E-RL-01, E-TRAIN-01, E-ACT-01 (correctness; cheap; Python-only or
   observational)
2. E-TRAIN-02, E-PERF-01 (infrastructure for everything else)
3. E-RL-02 → E-RL-03 (reward fixes with evidence)
4. E-SNN-03, E-RL-05, E-RL-06 (learning upgrades)
5. E-NEU-01/02, E-SNN-01 (science questions)
6. E-PHYS-01..04 (any time; independent)
7. E-EVAL-01/02 (only after step 3 fixes)

## Seeds and power (global conventions)

- Seed families: Python `2049 + 1000·k` for arm k; record per-run.
- ≥5 seeds for comparisons; ≥300 games/arm/seed for win-rate claims;
  report mean ± 95% CI (normal approx OK at n≥100).
- Pre-register direction + stop rule per experiment; publish negative
  results in this file with a "result:" addendum rather than deleting.
