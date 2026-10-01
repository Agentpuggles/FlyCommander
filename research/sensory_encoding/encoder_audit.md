# Sensory Encoder Audit

Status: RESEARCH DOCUMENT (audit only; no code changed).
Last verified: 2026-10-01. Evidence: full read of
[flycommander/sensory_encoder.py](../../flycommander/sensory_encoder.py) +
empirical probe ([encoder_probe.py](encoder_probe.py), run 2026-10-01).

---

## 1. What the encoder represents

`observation_to_state()` maps a Java observation dict → 64 dense channels:

| Channels | Content | Normalizer |
|---|---|---|
| 0–3 | per-seat life | `1 − life/40` |
| 4–7 | per-seat poison | `/10` |
| 8–11 | per-seat hand size | `/8` |
| 12–15 | per-seat creature count | `/15` |
| 16–19 | per-seat land count | `/20` |
| 20–23 | per-seat graveyard size | `/40` |
| 24–27 | per-seat exile size | `/20` |
| 28–31 | per-seat commander damage | `/40` |
| 32–35 | per-seat command-zone count | `/2` |
| 36–39 | per-seat library size | `/60` |
| 40–43 | stack size /10, attackers /10, blockers /10, turn /30 | |
| 44–55 | fly's own board → 12 MD5-hash buckets, soft counts (+0.25, cap 1.5) | |
| 56–63 | **spare, always zero** | |

Seat 0 = fly (fly is moved to the front of the seat list). Values clipped to
[0, 2]. Downstream: `tanh` → PN rates → fixed random PN→KC projection →
rectify → WTA keeps ~10% (≈406 of 4064 KCs).

## 2. What the encoder discards (before the encoder even runs)

Information lost at the **bridge** level (Java `AgentGameState`), i.e.
unrecoverable by any encoder redesign:

- Stack **contents** (only a count) — no spells/abilities on the stack, no
  targets, no modes chosen.
- Mana pools, mana available, lands tapped/untapped → fly cannot reason about
  whether it *can* afford a play; it only sees Forge-proposed candidates.
- Opponent board identities (only counts; own board has 12 hash buckets).
- Card types beyond creature/land counts (no artifacts/enchantments/
  planeswalkers/battles as distinct counts).
- P/T, counters, attachments (auras/equipment), tapped state, summoning
  sickness, ongoing continuous effects, active triggers.
- Exact life *history* (only current totals each step).
- Which cards are in own hand (only count) — the fly cannot plan specific
  plays, only "I have N cards."
- Revealed/known-top-card information, library order (fine to hide), but also
  own library composition (loses deck-archetype self-knowledge).

## 3. Empirical separation probe (reproducible)

Method: construct strategically opposite observation pairs, encode both with
the production encoder + `MushroomBody(seed=2049)`, compare active-KC sets.
Chance baseline from 200 random 10% masks: **Jaccard 0.052, min-overlap 0.102**.

| Pair | Jaccard | Min-overlap | State L2 |
|---|---|---|---|
| winning vs losing (grossly opposite boards) | 0.17 | 0.29 | 2.38 |
| low (2) vs high (39) life, otherwise identical | **0.58** | 0.73 | 0.92 |
| turn 2 vs turn 40 | 0.38 | 0.55 | 1.60 |

Interpretation:

- Every pair shares far more KCs than chance (3–11×) → the 4064-KC space is
  far from collision-free for game states.
- Worst case is the **single-channel strategic difference**: two states that
  differ *only* by 37 life points (dead vs healthy) still share 58% Jaccard /
  73% min-overlap of their KC ensembles. Life is a continuous, redundant-
  projected channel; the fixed random PN→KC projection + WTA spreads one
  channel's change across a partial re-shuffle rather than a distinct code.
- Grossly different boards *do* separate better (0.17) but still share ~118
  KCs of ~406.
- A 2026-10-01 earlier ad-hoc probe (different constructed pairs, unrecorded
  script) reported 0.61–0.68 "KC overlap" for similar contrasts. **The two
  probes disagree numerically**; the committed probe above is the reproducible
  one. The qualitative conclusion is stable in both: excess overlap vs chance,
  weakest discrimination for small-but-strategically-huge differences.

## 4. Structural issues

1. **Hash-bucket collisions.** `_bucket()` = MD5(name) % 12 over the full
   card-name space → expected ~thousands of distinct names per bucket across
   a set; within one deck collisions are rarer but guaranteed over time. Two
   different boards can produce identical channels 44–55. Count-vs-identity
   tradeoff is uncontrolled.
2. **Order/symmetry insensitivity** is *intentional* (permutation invariance
   over board entries) but also erases formation/order information that
   matters for combat math.
3. **Spare channels 56–63 always zero** — the encoder throws away 12.5% of
   its representational budget.
4. **Normalizer staleness:** life/40, turn/30, library/60 are calibrated for
   baseline Commander; long games (turn >30) saturate the turn channel;
   library sizes <60 (draw-heavy games) saturate library.
5. **Poison normalization** assumes the RC 10-poison rule (correct for this
   project; would break under Duel Commander variants).
6. **No temporal input:** each encode is memoryless; the MB sees no velocity
   (life *trend*, board *trend*). Reward shaping uses deltas; the policy
   network does not.
7. **Legality is external:** legal_mask comes from the Java proposal stream,
   not from the encoded state. The state alone cannot predict which actions
   are legal → hard for the value machinery to learn legality–state coupling.

## 5. Hidden-information audit (encoder view)

- Fly sees only **counts** of opponent hands/libraries/graveyards — no
  leakage of hidden card identities (good).
- Fly's own hand: only a count — no leakage, but also no self-knowledge to
  plan with (severe capability loss, not a fairness problem).
- The 12 identity buckets encode own-board card **names** — public
  information, no leak.
- Physical mode: `physical/brain_adapter.py` builds observations for the same
  encoder; equivalence of channel semantics (especially `commandZone`,
  `commanderDamage`, board entries) between Forge obs and physical obs is
  **assumed, not tested** (see claims SEN-004 and
  [physical_table/state_reconciliation.md](../physical_table/state_reconciliation.md)).

## 6. Experiment hooks

See [experiments/roadmap.md](../experiments/roadmap.md), section E-ENC-*:
separation sweep across seeds and pair families; ablation with/without hash
channels; linear-probe test (can life be decoded from KC code?); overlap vs
game-phase; replacing 12 hash buckets with opponent-aware channels.
