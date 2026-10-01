# Training Protocol Research

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Evidence: [scripts/train.py](../../scripts/train.py),
[scripts/run_match.py](../../scripts/run_match.py), Makefile, checkpoints/,
logs/ inventory.

---

## 1. Current state (verified)

- **Checkpoints:** `checkpoints/fly_demo.npz`, `checkpoints/fly_ep4.npz`
  (~115 KB each). Contents (from `MushroomBody.save`): `kc_to_mbon`,
  `w_mbon_action`, `value_baseline`, `temperature`. **Not saved:**
  `pn_to_kc`, `dan_to_mbon`, seeds, encoder version, config snapshot,
  Forge version, deck list. Claims TRN-001.
- **Logs:** `logs/java.log` only. No per-decision JSONL currently on disk;
  the episode-summary JSONL line seen during audit
  (`flyWon: false, reason: "AllOpponentsLost"`) came from a training run and
  is the only terminal-evidence artifact. Claims TRN-002.
- **JVM amortization:** many games per JVM process (boot cost ~tens of
  seconds dominated by `FModel.initialize`) — correct design; see
  [../forge/forge_internals.md](../forge/forge_internals.md).
- **Seeds:** Python RNGs seeded (2049 family); Java-side `MyRandom` NOT
  pinned per game (claims FRG-004) → **full game reproducibility is
  currently impossible** even with same Python seed.

## 2. What a defensible curriculum looks like

Phase 0 — **Smoke/validity** (before any "training"):
- Terminal-mapping reproduction (E-RL-01). Nothing else matters if win/loss
  labels are flipped.
- Fixed-deck 1v1 vs single Forge AI deck; verify episodes terminate and
  rewards accumulate as expected; log every decision.

Phase 1 — **1v1 fixed decks**: learn basics against one deterministic-ish
opponent; measure learning curves vs shuffled-reward control.

Phase 2 — **1v1 deck diversity**: N≥5 held-out-random decks; watch for
collapse to pool idiosyncrasies.

Phase 3 — **Free-for-all multiplayer** (current default): adds politics +
farming exploit surface; keep only after reward fixes, else the fly learns
to turtle.

Phase 4 — **Held-out evaluation**: frozen checkpoints, decks never seen in
training, report mean ± CI over ≥100 games per checkpoint (see
[../evaluation/evaluation_protocol.md](../evaluation/evaluation_protocol.md)).

## 3. Seed and config discipline (to implement later, NOT in this audit)

Checkpoint should carry: git commit, encoder version hash, connectome seed,
training config JSON, Forge version + jar hash, deck list, episode count,
and a manifest of RNG states (Python at minimum). Saving `pn_to_kc`/
`dan_to_mbon` explicitly removes the seed-reconstruction trap (claims TRN-001
validation_needed).

## 4. Numbers to plan around

- Decisions per game (4-player): order 10²–10³ per game depending on pace.
- If pre-registered learning horizon is 10³–10⁴ meaningful decisions
  (RL doc §6), a signal check needs ~10–100 games; a learning-curve paper
  grade result needs ~10³ games; an ablation comparison needs ≥5 seeds ×
  2 arms × 10³ games. JVM amortization makes this feasible; HTTP round-trip
  cost per decision is the per-game floor (performance doc).
- Time penalty −0.001/step: 300-step game costs −0.3 total — comparable to
  several board_delta steps; fine, but record it in any reward ablation.

## 5. Open protocol questions

- Random decks (Scryfall-sampled legal commanders) vs curated pool: random
  decks maximize diversity but introduce rules-coverage variance from Forge
  card implementations; curated pool first, random later.
- Self-play (fly vs flies) is currently impossible with the single-fly
  AgentServer architecture (one BrainClient per server) — would need
  multi-client support; defer.
- Curriculum gating criteria should be pre-registered: e.g., "advance to
  phase 2 when win rate vs phase-1 opponent exceeds 50% with 95% CI
  excluding 40%."
