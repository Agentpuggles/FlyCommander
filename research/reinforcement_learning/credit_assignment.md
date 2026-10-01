# Reinforcement Learning Analysis

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Scope: learning architecture *independent* of the neuroscience analogy.
Evidence: reward_shaping.py, dopamine_plasticity.py, mushroom_body.py (all
read in full), plus literature.

---

## 1. What the learning system is, plainly

- On-policy, single-visit updates (no replay, no batch).
- Policy: masked softmax over 4 action scores derived from MBON rates ×
  `w_mbon_action` (the only *policy* weights; learned via the same DPR).
- "Critic": scalar `value_baseline`, updated `baseline += 0.01·RPE` — a
  moving average of rewards, not a value function.
- RPE = immediate shaped reward − baseline. No γ, no bootstrapping, no
  n-step returns, no terminal-to-history credit propagation.
- Exploration: temperature-annealed softmax (τ 0.5 → floor 0.15, ×0.995 per
  anneal call).

## 2. Is a DPR-style rule appropriate here?

Arguments **for** keeping the current family (with modifications):
- The MB story *is* reward-gated synaptic plasticity; the rule matches the
  analogy the project is explicitly exploring. Scientifically, the question
  "can a dopamine-gated Hebbian MB learn Commander?" is the *point* of the
  experiment.
- 4 actions, sparse decisions, reward mostly terminal — a tabular-grade
  problem where even crude credit assignment can show signal.
- One-shot-like learning in the biological MB (see mb_facts.md) is consistent
  with strong credit from few examples at this action granularity.

Arguments **against** (and they are serious):
- Without value bootstrapping or traces, terminal ±1 reaches only the final
  KC code; earlier decisions learn only from shaped step rewards, and the
  shaped rewards have documented farming exploits (rewards/reward_audit.md §2).
- Confounded rewards (Forge AI co-decides outcomes) inflate variance; with
  no advantage normalization, learning signal-to-noise is poor.

**Verdict:** keep the MB/Hebbian family as the *primary experimental subject*,
but treat the following as the experimentally-grounded minimal fix set, in
priority order: (1) verify terminal sign (E-RL-01), (2) add eligibility
traces (E-SNN-03), (3) state-dependent baseline (E-RL-05), (4) fix reward
farming (E-RL-03). Do not swap in PPO/DQN merely for convention; but DO run
one conventional baseline (tabular Q on the same features) as a yardstick —
E-RL-06 — so "how far from standard RL are we?" has a number.

## 3. Reward-hacking surface (summary; details in reward audit)

Passive-fly farming of `opp_removal`/`board_delta` in 4-player games is the
clearest exploitable gradient; "turtle and draw" (hand_delta + slow time
penalty) is the second. Both are measurable with a hold-only agent.

## 4. Stability, forgetting, nonstationarity

- **Catastrophic forgetting:** single weight matrix, no replay → a run of
  losses under a new opponent archetype will overwrite prior valence.
  Mitigation candidates: LTM blend (already present, untested), dual-rate
  synapses, or rehearsal — all experimentable (E-RL-07).
- **Nonstationary opponents:** Forge AI is deterministic-ish; the pool is
  fixed decks. Overfitting to pool is likely; held-out decks are the control
  (evaluation/evaluation_protocol.md).
- **Policy collapse:** temperature floor 0.15 keeps ε-equivalent exploration;
  entropy of action distribution should be logged per episode (currently it
  is not — training/logging gap).

## 5. Alternatives compared (not recommended wholesale — compared)

| Method | Would solve | Would break/complicate | Verdict |
|---|---|---|---|
| Eligibility traces added to DPR | multi-step credit | minimal (one vector) | **adopt-testable now** |
| Learned state-value critic (linear on state) | better RPE | small; keeps MB story | **adopt-testable** |
| n-step returns / TD(λ)-flavored credit | credit propagation | moderate rewrite of update loop | after traces |
| Tabular Q on 64-dim state | yardstick baseline | none | **as control only** |
| PPO / DQN / MuZero-style | performance ceiling | destroys MB framing; heavy deps; sample-hungry | out of scope for the science question |
| Offline RL from logged games | reuse logs | no logs exist yet | blocked on logging |

## 6. Sample-efficiency expectations (set BEFORE training runs)

Pre-register: with 4 actions and shaped rewards, meaningful signal (if any)
should appear within O(10³–10⁴) decisions. If 5×10⁴ decisions show no
above-chance action preference vs shuffled-reward control, the honest
conclusion is "no learning under current signal" — not "train longer."

## 7. Literature anchors

- Sutton & Barto 2018, *RL: An Introduction*, 2nd ed. — baselines, TD,
  credit assignment fundamentals.
- Ng, Harada & Russell 1999 — potential-based reward shaping (the step
  deltas are potential-*like* but the potential is not exactly a state
  function → policy-invariance guarantee does NOT hold; worth an experiment
  note).
- Izhikevich 2007 / Florian 2007 / Gerstner et al. 2018 — see snn doc.
- Legg & Hutter 2007 — general reward-hacking framing (light-touch).
- Amodei et al. 2016, "Concrete Problems in AI Safety" — reward hacking
  taxonomy; maps 1:1 onto §3.
