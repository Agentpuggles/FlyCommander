# FlyCommander — Master Research & Technical Audit (FINAL_REPORT)

Audit date: 2026-10-01 · Auditor: autonomous research pass (Buffy/Codebuff)
Scope: full repository + external dependencies + scientific literature.
**No production code was changed.** Only `research/**` was created.

Companion docs: [index.md](index.md) lists all 24 knowledge-base files;
[claims/claims.yaml](claims/claims.yaml) is the evidence spine (40 claims).

---

## 1. Executive summary — what FlyCommander currently is

FlyCommander is a **research instrument**, not a competent MTG player yet.
Concretely: a Forge 2.0.15 headless match server (patched via 8 Java classes)
that publishes observations over HTTP to a Python spiking "mushroom body"
(4064 KCs / 97 MBONs / 344 DANs, sparse PN→KC at 7 claws, WTA 10%, LIF rate
readout), which chooses among **4 macro actions** (play / attack / hold /
interact); Forge's heuristic AI decides everything else (targets, blockers,
damage, mana, instants, mulligans). Learning is a dopamine-gated Hebbian
update on KC→MBON plus a learned MBON→action readout, driven by shaped step
rewards and a ±1 terminal. A physical-table mode (OpenCV + Tesseract OCR +
Scryfall) feeds the same brain from a camera.

**The audit's three headline results:**

1. **A correctness gate exists before any training claim is trustworthy:**
   the terminal result mapping shows one anomalous record
   (`flyWon=false, reason=AllOpponentsLost`) — if the win/loss label is
   actually inverted, every terminal reward in every past run was wrong
   (E-RL-01 must run first).
2. **The representation, not the brain, is the bottleneck:** the bridge
   discards stack contents, mana, and opponent board identity *before* the
   encoder runs; the empirical probe shows strategically opposite states
   share far more KC code than chance (Jaccard 0.17–0.58 vs 0.052 baseline;
   worst when only life differs: 0.58). A policy cannot out-learn its state.
3. **The neuroscience framing needs honesty edits, not architecture edits:**
   counts (4064/97/344) are whole-CNS MaleCNS tallies presented as MB scale
   (adult MB ≈ 2000/34/~130, Aso 2014); the "DPR" name overstates fidelity
   to Gkanias 2022 (whose rule is not RPE and has different structure); the
   DAN→MBON compartment structure is currently decorative (random matrix).

None of these kill the project. All three are testable, and the fix path is
prioritized in §14.

## 2. Architecture (what is actually implemented)

Full detail: [architecture/flycommander_current.md](architecture/flycommander_current.md).
Verified highlights:

- **Control boundary (bytecode-verified):** `FlyPlayerController` overrides
  ONLY `chooseSpellAbilityToPlay` (own main phases, ≤8/phase) and
  `declareAttackers` (gated on prior ACT_ATTACK decision). Everything else
  = Forge AI. The fly picks among Forge-*proposed* candidates — a documented
  confound in every reward number the system produces.
- **State-loss chain (5 irrecovery points):** Forge GameState → obs dict
  (counts+names) → 64 dense channels → tanh+fixed projection → WTA 10% →
  4-action softmax. Losses compound; nothing downstream can recover them.
- **Learning:** DPR update `Δw = lr·outer(active KC, dan_gate)` (lr 0.01,
  clip [0,2], LTM blend 0.5 toward ×0.9995 decay), RPE = r − scalar baseline
  (creep 0.01), τ-annealed softmax (0.5→0.15 floor, ×0.995).
- **Checkpoints:** save only `kc_to_mbon`, `w_mbon_action`, baseline, τ —
  **not** `pn_to_kc`/`dan_to_mbon`/seed/config/Forge version → checkpoints
  are not self-describing (silent-corruption trap, claims TRN-001).
- **Logging:** `logs/java.log` only; no per-decision records persisted →
  credit-assignment debugging and entropy monitoring are impossible today.
- **Quirks (flagged, untouched):** decorative DANs; unused
  `value_estimate()`; silent hold on empty `interact` proposals; physical
  mode `canPlay` hardcoded false; `Thread.sleep(15000)` startup sync.

## 3. Research findings (by domain — one line each, links in index.md)

- **Magic rules** ([magic/](magic/comprehensive_rules_findings.md)): Comp
  Rules eff. 2025-06-06 verified for layers 613.1a–g, SBA 704.5a–z, 903.10a
  (21 commander damage), 40 life, poison 10. The state model cannot
  represent ~all of the rules that make Commander hard (layers, timestamps,
  replacement ordering, stack interactions) — table in the doc maps each
  missing rule ↔ missing state.
- **Forge** ([forge/](forge/forge_internals.md)): boot path, heuristic-AI
  structure, MyRandom (unpinned → no reproducible games), Forge ≠ rules
  divergence classes, upgrade-fragility of the patch.
- **Neuroscience** ([neuroscience/](neuroscience/mb_facts.md)): PN_PER_KC=7
  ✓ matches Aso 2014; sparsity direction ✓; counts ✗ (scope error); DPR ✗
  (naming); compartments decorative; memory-phase blend = pure engineering.
- **Connectome** ([connectome/](connectome/connectome_datasets.md)):
  FlyWire/hemibrain/MaleCNS landscape, licensing caution, swap feasibility —
  the DAN→MBON real-compartment matrix is the highest-value swap.
- **SNN** ([snn/](snn/lif_and_plasticity.md)): 8-step rate readout is a
  static activation in disguise; missing eligibility traces is the cheap,
  literature-backed upgrade (Izhikevich 2007, Gerstner 2018); STDP would be
  decorative here — do not add for fashion.
- **RL** ([reinforcement_learning/](reinforcement_learning/credit_assignment.md)):
  no γ/bootstrapping/traces; scalar baseline; confounded rewards; reward
  farming gradients (below); keep the MB family as the experiment's subject,
  add traces + linear critic + one tabular-Q yardstick.
- **Sensory encoding** ([sensory_encoding/](sensory_encoding/encoder_audit.md)):
  full discard list; 12 hash-bucket collisions; 12 spare channels unused;
  probe results; sim/physical equivalence unproven.
- **Action space** ([action_space/](action_space/action_space_audit.md)):
  legality unobservable in state; `interact` semantics opaque (Forge's
  choice); attack-target choice (the political core of Commander) entirely
  Forge's.
- **Rewards** ([rewards/](rewards/reward_audit.md)): exact mechanics
  verified; five farming vectors; `max()` opp-life perversity; board-score
  comment/code mismatch; draw-by-substring fragility.
- **Training/Evaluation** ([training/](training/training_protocol.md),
  [evaluation/](evaluation/evaluation_protocol.md)): curriculum phases,
  required controls (random-agent, pure-Forge-AI, shuffled-reward, frozen),
  statistical standards (≥5 seeds, ≥300 games/arm, CIs), leakage rules.
- **Vision/OCR** ([computer_vision/](computer_vision/ocr_and_detection.md),
  [physical_table/](physical_table/state_reconciliation.md)): Tesseract
  failure modes (foils, sleeves, overlap, alt-frames; Boldt 2019 evidence),
  confidence-gated + human-confirm design is correct, identify-on-pickup
  recommended; event-sourcing shape is right, invariant validation untested.
- **Security** ([security/](security/security_review.md)): realistic
  findings = schema-less JSON on all four HTTP legs + unbounded payloads;
  localhost posture acceptable; no secrets in repo; escape card names in
  browser UI.
- **Performance** ([performance/](performance/performance_notes.md)):
  hypotheses only (measure first): HTTP/JSON per-decision overhead and Forge
  simulation speed likely dominate; brain is µs–ms scale; nothing to
  optimize yet.
- **Related work** ([related_projects/](related_projects/related_projects.md)):
  Forge/XMage/Magarena/Argentum/phase.rs engines; commander-ai-lab as
  closest relative (verify before use); **no published spiking/neuro-inspired
  Commander agent found — the niche appears open.**

## 4. Correct assumptions (evidence-backed — keep)

1. PN→KC fan-in 7 matches biology (Aso 2014).
2. Sparse-code + expansion + WTA is the right MB-inspired shape, and 10%
   keep-fraction is in the biologically plausible ballpark.
3. Building on Forge (not writing an engine) is correct; the rules engine is
   the moat, and every serious project treats it that way.
4. Reward-gated KC→MBON plasticity as the sole memory locus matches the
   biological picture of where fly memory lives.
5. RPE-as-DAN-drive is defensible engineering (fly RPE neurons exist).
6. Human-in-the-loop, confidence-carrying physical reconciliation is the
   right pattern; hybrid vision/manual with authoritative engine validation
   matches prior art.
7. JVM amortization (many games per process) is the right training-loop
   design given FModel.initialize cost.
8. Localhost-only, no-auth posture is appropriate for the threat model.

## 5. Questionable assumptions (validate before relying)

1. **Terminal labels are correct** (anomaly observed — E-RL-01). Everything
   downstream inherits this doubt.
2. **Checkpoints reconstruct from seed 2049** — untested roundtrip; seed is
   undocumented in the checkpoint (E-TRAIN-01).
3. **"DPR" implements Gkanias-style dopamine plasticity** — it does not;
   it is a simpler dopamine-gated Hebbian rule (fine, but relabel; claims
   NEU-002).
4. **"Biological scale" 4064/97/344** — whole-CNS counts, not MB (claims
   NEU-001).
5. **8-step LIF "spiking" adds anything over a static activation** — likely
   not (E-SNN-01); the honest framing is "MB-shaped MLP" until shown
   otherwise.
6. **Rewards as designed teach good play** — farming gradients suggest
   otherwise (E-RL-02/03); `max()` opp-life rewards damaging the healthiest
   opponent.
7. **Sim and physical observations are equivalent** — assumed, untested
   (E-PHYS-03).
8. **`make test` green implies vision works** — vision tests skip under
   system python3 (no cv2); green ≠ exercised for vision.
9. **Forge AI proposals are a sensible action menu** — unmeasured census
   (E-ACT-01); if menus are mostly single-candidate, the fly's "agency" is
   mostly accept/decline.
10. **LIF numerics are stable at all regimes** — only the current regime is
    exercised (E-SNN-02).

## 6. Known gaps (implementation)

Highest-impact first (full tables in
[architecture/research_gaps.md](architecture/research_gaps.md)):
1. Terminal-result verification + reason-string table (A1).
2. Per-decision logging (decisions, candidate menus, entropy, τ, reward
   components) — currently nothing persists (A4, C6).
3. Self-describing checkpoints (connectome matrices + manifest) (A3).
4. Seeded per-game Java RNG (A2).
5. Encoder: populate spare channels; add opponent-identity and mana/tapped/
   stack-depth channels; replace or bound hash collisions (B1–B3).
6. Action space: target choice, attack target, block choice; instant-speed
   windows last (B4).
7. Eligibility traces + state-dependent baseline in plasticity (C1, C2).
8. Reward fixes: farmable components, `max()`→`sum`, deal-commander-damage
   signal (C3, C4).
9. Eval harness with the control suite (F1).
10. Physical-mode invariant validation + OCR benchmark corpus (E1, E2).

## 7. Critical dependencies

| Dependency | Role | Risk |
|---|---|---|
| Forge 2.0.15 jar (exact version) | rules engine + opponents + proposals | upgrade breaks patch + doc assumptions; pin or verify |
| Patched classes dir | control boundary | must recompile/verify per jar change |
| `~/.forge/decks/commander/` | deck pool | pool defines training distribution; never edit at runtime |
| Scryfall API + bulk data | card identity | Accept header, ~10 req/s, daily bulk; network dependency |
| OpenCV 5.0.0 + Tesseract 5.5.3 (venv only) | physical mode | system python3 lacks cv2 — environment split |
| Python 3.14.7 / NumPy 2.5.3 | brain | NumPy major-version sensitivity |
| FlyWire/MaleCNS/hemibrain (future) | real connectome | licensing + versioning; tokens must not enter repo |
| Comprehensive Rules (2025-06-06 extract) | rules authority | superseded by future WotC releases — re-check |

## 8. Scientific limitations (where the analogy diverges)

1. No spike timing → no STDP, no temporal coding; "LIF" is a rate function.
2. DAN compartments decorative; no per-compartment valence in function.
3. No recurrent APL inhibition; WTA is a sort.
4. One memory matrix: no multiple traces, no extinction, no reversal support.
5. No innate MBON valences; readout starts as noise, unlike approach/avoid
   baseline biology.
6. Memory-phase "consolidation" is a decay blend, not biology.
7. Scale and scope claims need the BS/EA/SP honesty labels now applied in
   the docs (and, ideally, in code docstrings at a future pass).

## 9. MTG limitations (what the current system cannot reason about)

Cannot represent (state-level): layers/timestamps; stack contents/targets;
mana availability; attachments; counters; card types beyond creature/land
counts; tapped state; own hand content; opponent board identity; poison as
reward (encoder only); commander tax count; revealed-zone contents.
Cannot act: target selection; defender choice; blocking; instant-speed;
mulligans; mode choices; activation counts beyond 8/phase.
Forge's implementation stands in for the rules — correct to within Forge's
card-implementation bugs, which are unmeasured for this deck pool (open
question 10).

## 10. Engineering limitations

No per-decision logs; non-self-describing checkpoints; unseeded Java RNG;
substring draw detection; silent holds; `Thread.sleep(15000)`; unused
`value_estimate()`; dual-environment test split; no eval harness; no CI.

## 11. Physical-table limitations

No labeled benchmarks (accuracy claims unsupported until E-PHYS-01/02);
OCR fragility to foils/sleeves/overlap/alt-frames; ~110 px/card at 1080p
full-table → identify-on-pickup recommended; observation equivalence
unproven; invariant validation unfuzzed.

## 12. Security concerns

Schema-less JSON on all four HTTP legs (robustness first); unbounded request
size; card names into browser UI need HTML escaping; Scryfall cache trusts
response shape; localhost/no-auth is a deliberate, acceptable posture; no
secrets in repo (keep FlyWire tokens out).

## 13. Performance concerns

Unmeasured by design (measure-first rule). Hypotheses: per-decision HTTP+JSON
and Forge simulation dominate; brain compute is negligible at this scale;
vision identification dominates physical latency. E-PERF-01 produces the
real table; the "Measured" section of performance_notes.md is intentionally
empty until then.

## 14. Research-backed roadmap (priority order)

1. **Gate experiments** — E-RL-01 (terminal mapping), E-TRAIN-01
   (checkpoint roundtrip), E-ACT-01 (proposal census). Cheap, Python-side.
2. **Instrumentation** — per-decision logging, entropy/τ curves, reward
   component persistence (enables everything else).
3. **Reward repair with evidence** — E-RL-02 (farming probe) → E-RL-03
   (component ablation).
4. **Learning upgrades** — eligibility traces (E-SNN-03), linear critic
   (E-RL-05), tabular-Q yardstick (E-RL-06).
5. **Science questions** — KC scale sweep (E-NEU-01), structured vs random
   DAN→MBON (E-NEU-02, with hemibrain matrix), LIF vs static (E-SNN-01).
6. **Representation** — encoder extensions + probe re-run; then action-space
   widening (targets, attack target) via new patch work.
7. **Physical mode** — E-PHYS-03 equivalence first, then OCR/tap corpora.
8. **Paper-grade run** — frozen eval harness, held-out decks, ≥5 seeds
   (E-EVAL-01/02) — only after 1–3.

All experiments specified with H/IV/DV/controls/seeds in
[experiments/roadmap.md](experiments/roadmap.md).

## 15. Open questions

Top items (full list with HOW-TO in
[open_questions/open_questions.md](open_questions/open_questions.md)):
terminal-label correctness across all game-end paths; Forge reason-string
enumeration; worst-case information survival through encoder; Forge-AI
proposal-quality correlation (confound strength); learner-vs-drift
(shuffled-reward control); temperature-floor reachability; Forge known-buggy
cards for this deck pool; multiplayer edge semantics in Forge vs rules text;
sim/physical equivalence; existence of any prior neuro-inspired Commander
agent (searched: none found, low-confidence negative).

## 16. Source index

See [sources/sources_index.md](sources/sources_index.md) (full table with
versions/access dates) and [academic/academic_index.md](academic/academic_index.md)
(DOIs). Anchor sources: MTG Comp Rules eff. 2025-06-06 (local extract);
Scryfall docs (accessed 2026-10-01); Aso et al. 2014 Cell 157;
Gkanias et al. 2022 eLife 11:e75611; FlyWire/hemibrain/MaleCNS datasets;
Forge 2.0.15 jar bytecode + docs/AI.md; Izhikevich 2007 / Gerstner et al.
2018 / Sutton & Barto 2018; Smith 2007 (Tesseract) / Boldt et al. 2019
(MTG OCR). Recorded disagreements are listed in sources_index.md §Disagreements.

---

## 17. Audit statistics (deliverable §36 of the brief)

- **Files inspected:** 52 source files (42 Python incl. 6 test files, 8
  Java) + Makefile/README/requirements; 102 repo files total outside
  .venv/.git/research; plus jar-level bytecode inspection (AgentMain,
  FlyPlayerController, MyRandom) and the local Comp Rules extract.
- **Tests run:** venv suite 106 passed / 1 skip; system python3 104 passed /
  3 skipped (vision, no cv2); Java patch compiles against Forge 2.0.15;
  plus audit-only scripts: encoder separation probe (reproducible, saved),
  claims-DB structural validation.
- **Domains covered:** MTG rules · card data (Scryfall) · Forge internals ·
  control boundary · neuroscience (MB) · connectomics · SNN methods · RL ·
  sensory encoding · action space · rewards · training · evaluation ·
  vision/OCR · physical state estimation · security · performance · related
  projects · academic literature.
- **Knowledge base produced:** 24 documents + 40-claim machine-readable DB
  + 1 reproducible probe script under `research/`.
