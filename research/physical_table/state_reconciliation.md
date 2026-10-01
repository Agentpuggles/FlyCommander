# Physical-State Reconciliation Research

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Evidence: [physical/state.py](../../physical/state.py),
[physical/events.py](../../physical/events.py),
[physical/engine.py](../../physical/engine.py),
[physical/observer.py](../../physical/observer.py) code reads.

---

## 1. The architecture pattern (and why it's right)

```text
camera frames → vision (uncertain observation, confidence-carrying)
             → events
human input  → confirmations/corrections (authoritative when given)
rules engine → authoritative validation (engine.py)
             → PhysicalGameState snapshot
             → same brain adapter → same encoder → same mushroom body
```

This is an event-sourced state machine with human-in-the-loop correction —
the established pattern for vision-driven game tracking (e.g., scoreboard
systems, chess-board trackers). FlyCommander already has the shape; the
research question is what's missing from the standard playbook.

## 2. Standard playbook items — present vs missing

| Mechanism | Status in repo | Notes |
|---|---|---|
| Append-only event log | present (events.py) | good foundation |
| Snapshot + replay | partial | verify replay determinism |
| Confidence propagation | partial — vision confidence exists but state doesn't track per-fact confidence | gap |
| Impossible-transition rejection | untested | E-PHYS-04 fuzz |
| Undo/correction | present (human correction path) | verify ordering guarantees |
| Temporal consistency (monotonic life, zone-conservation) | untested | zone conservation is the strongest invariant to fuzz |
| Ambiguous-event queue | implicit | make explicit: vision events below confidence threshold should queue for human confirm, not apply silently |
| Stable entity IDs across occlusion | IoU tracker; occlusion gaps unbenchmarked | E-PHYS-02 |

## 3. Invariants worth enforcing (proposed for future work, not implemented)

- **Zone conservation:** every physical card is in exactly one zone; total
  card count per deck constant (minus tokens).
- **Life monotonicity:** life only changes via damage/gain events; no
  spontaneous jumps > plausible max.
- **Tap state coherence:** tapped permanent must be on battlefield; tap
  angle must be within [0°, ±90°+tolerance] of recorded orientation.
- **Commander traceability:** commander must be traceable through
  command-zone → battlefield → library/graveyard/exile → command-zone loops
  (the hardest invariant; commander tax counting depends on it).

## 4. Comparison to prior art (patterns to borrow)

- **Chess DGT boards / ArUco chess:** deterministic piece placement with
  move-legalization — FlyCommander's equivalent is legalizing vision events
  against the rules engine before applying; engine.py does this, keep it
  authoritative.
- **Scoreboard-style state estimators:** maintain belief + ask human only
  when belief crosses ambiguity threshold — better than current
  confirm-everything or trust-everything binaries.
- **Event sourcing discipline (Kleppmann, *Designing Data-Intensive
  Applications*):** events immutable, derived state rebuildable — matches
  the repo's design; add snapshot/replay tests to prove it.

## 5. Sim/physical equivalence (the cross-cutting risk)

Same brain, two observation builders. Until E-PHYS-03 proves channel
equivalence, the physical mode is *distributionally untested* — the fly
could behave arbitrarily there. Highest-priority physical-mode experiment.

## 6. Camera/geometry research notes

- Fixed overhead camera, cards on matte playmat → classical detection works.
- Perspective correction via playmat fiducials (or quad detection of known
  mat markings) stabilizes tap-angle measurement.
- 1080p ≈ 2.1 MP: a 4-player table fits ~40 card-widths across the long
  edge at best — per-card resolution ~110 px wide → collector-number OCR
  marginal; expect to need either 4K or per-card zoom-and-rescan flow for
  identification (design implication: identify on pickup, not from table
  overview).
