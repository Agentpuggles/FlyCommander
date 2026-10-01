# Open Questions & Unknown Unknowns

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Format: UNKNOWN / WHY IT MATTERS / HOW TO INVESTIGATE.
The mandatory "what are we missing that we haven't considered" list.

---

## Correctness unknowns

1. **UNKNOWN:** Is the terminal result mapping (`flyWon`, `reason`) correct in
   all game-end paths (concede, SBA losses, draws, timeouts)?
   **WHY:** if wrong, every terminal reward is wrong → all training invalid.
   **HOW:** E-RL-01 forced endings + code-path read of AgentServer/BrainClient
   win detection; build a reason-string test table.

2. **UNKNOWN:** Do episodes ever hang or leak (Forge triggers never resolving,
   mandatory-loop protection)?
   **WHY:** a hung game stalls the training loop silently; mandatory loops
   are a rules-edge case Forge handles but our wrapper may not surface.
   **HOW:** stress-run 1000 random-deck games; alarm on wall-time outliers;
   grep Forge changelog for loop-handling fixes after 2.0.15.

3. **UNKNOWN:** What do Forge "reason" strings actually enumerate?
   **WHY:** `terminal()` draw detection is substring-based ("draw") — a reason
   like "lost due to Draw effect" would misclassify.
   **HOW:** capture all terminal payloads across 100+ games; enumerate
   distinct reasons; replace substring check with exact table (future fix).

## Information unknowns

4. **UNKNOWN:** How much strategic information survives the bridge+encoder
   collapse in the *worst* case (not the constructed-pair case)?
   **WHY:** policy quality is bounded by the state code.
   **HOW:** linear-probe decoding of (life, creature advantage, turn) from
   KC codes across thousands of real game states; decode-accuracy profile =
   information audit (E-ENC-01 in roadmap extensions).

5. **UNKNOWN:** Does the fly accidentally receive information it shouldn't
   (hidden-info leak) in any obs field?
   **WHY:** fairness/validity of sim results.
   **HOW:** diff AgentGameState obs against Oracle-visible-information rules
   per zone; write a leak checklist test (currently: none found — own hand/
   library are counts only; opponent boards counts only — but verify
   `board` name lists never include face-down/revealed-hidden cards).

6. **UNKNOWN:** Are Forge-proposed candidate *qualities* systematically
   correlated with state features (creating spurious reward correlations)?
   **WHY:** the fly may learn state→action preferences that are actually
   Forge-AI quirks.
   **HOW:** E-ARCH-01: same states, Forge AI vs random controller; compare
   reward trajectories; variance decomposition.

## Learning unknowns

7. **UNKNOWN:** Can the DPR-family learner produce above-chance action
   preference at all under the current reward — or is all observed "learning"
   drift?
   **WHY:** pre-registration requires defining what failure looks like
   *before* long training runs.
   **HOW:** shuffled-reward control (evaluation doc §2.3) + pre-registered
   5×10⁴-decision horizon (RL doc §6).

8. **UNKNOWN:** Whether temperature annealing schedule (×0.995 per anneal
   call) ever actually reaches the floor in practice — anneal call frequency
   is not pinned in my audit notes.
   **WHY:** exploration collapse vs eternal noise changes everything.
   **HOW:** log τ per episode; plot; if τ hits floor in <10 episodes,
   exploration dies before any learning.

9. **UNKNOWN:** w_mbon_action and KC→MBON weights co-adapt under DPR —
   could they oscillate or collapse to a single-action policy?
   **WHY:** stability of the only two learned matrices.
   **HOW:** weight-norm trajectories per episode; entropy of action
   distribution; collapse detector in logging (E-TRAIN-03).

## Rules unknowns

10. **UNKNOWN:** Which Commander-relevant cards Forge 2.0.15 implements
    *incorrectly* (known-buggy list is release-dependent)?
    **WHY:** training on false dynamics.
    **HOW:** pick the project's deck pool; for each commander + top-20
    staples, check Forge changelog/issue tracker; maintain a
    known-bugs list in research/forge/.

11. **UNKNOWN:** Multiplayer edge semantics in Forge (APNAP ordering,
    range-of-influence in free-for-all, team effects) — verified against
    Comp Rules only on paper, not against Forge behavior.
    **HOW:** construct probe games for each edge (e.g., simultaneous SBA
    deaths with different controller orders); document Forge vs rules text.

## Physical-mode unknowns

12. **UNKNOWN:** Sim/physical observation equivalence (channel semantics).
    **WHY:** same brain, untested distribution shift.
    **HOW:** E-PHYS-03 channel-diff test.

13. **UNKNOWN:** Real-world OCR/tap accuracy (no labeled data exists).
    **WHY:** physical mode claims are currently unsupported.
    **HOW:** E-PHYS-01/02 corpora (needs user-side card photography).

## Meta unknowns

14. **UNKNOWN:** Whether any *other* FlyWire/MaleCNS-derived game-playing
    agent exists (negative search result, low confidence).
    **WHY:** novelty claim honesty.
    **HOW:** targeted scholar search at next research pass.

15. **UNKNOWN:** Forge 2.0.15 → current Forge: did AI proposal structure or
    controller API change (breaking our patch)?
    **WHY:** upgrade risk assessment.
    **HOW:** diff controller interface between 2.0.15 and latest release
    before any upgrade; re-run javap verification (forge_internals §7).

## Unknowns deliberately not chased (with reason)

- Enterprise security hardening (threat model is local research).
- Performance micro-optimization (measure first — E-PERF-01).
- Real-connectome integration feasibility *details* (blocked on download;
  roadmap exists in connectome doc §3).
