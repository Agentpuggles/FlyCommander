# Evaluation Protocol Research

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.

---

## 1. Metrics that matter

| Metric | What it tells you | Currently measurable? |
|---|---|---|
| Win rate vs held-out decks (+95% CI) | headline competence | needs eval harness; today only ad-hoc episode summaries |
| Legal-action rate | engine sanity | partially — silent holds mask it (claims ACT-003) |
| Action entropy per episode | exploration health / collapse | **not logged today** |
| Learning curve (reward/episode, smoothed) | learning existence | partial (JSONL ad hoc) |
| Survival time (turns) | defensive competence | derivable from terminal payload |
| Commander damage dealt/taken | aggro/defense balance | not logged |
| Card advantage trajectory | strategic quality | not logged |
| Reward-component breakdown | reward debugging | computed but not persisted |
| Win rate vs **shuffled-reward control** | is learning real? | needs harness |
| Win rate vs **random-action agent** | floor baseline | needs harness |
| Win rate of **pure Forge AI** (fly replaced) | ceiling baseline | easy — run unpatched controller |
| KC code diversity (mean pairwise Jaccard across states) | representation health | via encoder_probe pattern |

## 2. Controls required before any improvement claim

1. **Random-action agent** (same masking) — floor.
2. **Pure Forge AI** (no fly decisions) — ceiling reference and the
   anti-null-result control: if fly ≈ Forge AI, the fly adds nothing.
3. **Shuffled-reward training** — proves learning comes from reward, not
   from drift in weights (critical given the DPR family).
4. **No-learning checkpoint** (frozen weights) — separates exploration
   benefits from learning.
5. **Multiple seeds** (≥5) for any comparison; report mean ± CI, never
   single-run numbers.
6. **Held-out decks and opponents** — trained vs held-out win-rate gap is
   the generalization number.

## 3. Ablation suite (each = one experiment in roadmap)

- No learning (lr=0): isolates exploration/architecture value.
- No dopamine (dan_gate ≡ 0): isolates plasticity necessity.
- No sparse coding (keep_fraction=1.0): isolates WTA value; expect collapse
  of pattern separation (ties to encoder probe).
- Random connectome seed sweep: sensitivity to initialization.
- Reward variants: full vs no-opp_removal vs no-board_delta vs terminal-only.
- Encoder variants: no hash channels; spare channels populated; opponent-
  identity channels added.
- Action space: 4-action vs +target-choice vs +attack-target (requires Java
  patch extension).

## 4. Statistical standards

- ≥100 games per arm per seed for terminal-metric comparisons (win rate is
  binomial; 100 games → ±10% at 95% CI — wide; prefer 300+ for headline
  claims).
- Report CIs; pre-register expected direction before running ablations
  (mirrors RL doc §6).
- Never compare checkpoints trained on different Forge versions or deck
  pools without saying so.

## 5. Leakage controls

- Freeze evaluation decks at project start; never train on them.
- Hold out one opponent seat archetype entirely.
- If any deck filtering uses reward outcomes (e.g., "decks where Forge AI
  wins fast"), that information must not leak into training pool curation.
- Terminal-mapping bug (E-RL-01) is itself a *validity* leak: fix and re-run
  everything after any terminal-payload change.

## 6. Deliverable format (proposed, not implemented)

`eval/results/<date>_<name>/` containing: config.json (full), games.jsonl
(per-game terminal payloads + per-decision digests), summary.md with the
metric table above + CIs, and the checkpoint hash. This makes every claim in
FINAL_REPORT reproducible.
