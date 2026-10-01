# Research Gaps: Missing, Uncertain, Potentially Incorrect

Status: RESEARCH DOCUMENT (audit only; no code changed). Last verified: 2026-10-01.
Companion to [flycommander_current.md](flycommander_current.md) (what IS built).
Each gap lists evidence and the experiment/claim IDs that address it.

---

## A. Correctness-critical (fix before trusting any training result)

| # | Gap | Evidence | Addressed by |
|---|---|---|---|
| A1 | Terminal result mapping suspect: `flyWon=false` + `AllOpponentsLost` observed once; if mapping is inverted, all terminal rewards are flipped | logs/ episode JSONL; claims RL-005 | E-RL-01 |
| A2 | Java-side game RNG not seeded per game → no reproducible episodes | claims FRG-004 | E-TRAIN-02 |
| A3 | Checkpoints omit `pn_to_kc`/`dan_to_mbon` + seed/config/Forge-version manifest → silent corruption on seed mismatch | claims TRN-001 | E-TRAIN-01 |
| A4 | Silent hold when `interact` has no candidate — decision logs can't distinguish abstention from nonexistence | claims ACT-003 | E-ACT-01 |

## B. Representation gaps (encoder + action space)

| # | Gap | Evidence | Addressed by |
|---|---|---|---|
| B1 | No stack contents, mana, opponent board identities, card types beyond creature/land counts | encoder audit §2 | E-ENC-03 |
| B2 | Own-board 12 hash buckets collide across card space; opponents are anonymous counts | claims SEN-003 | E-ENC-02 |
| B3 | 12 spare channels (56–63) unused | encoder audit §1 | E-ENC-02 |
| B4 | Fly cannot choose targets, defenders, blocks, or act at instant speed | claims ACT-002 | E-ACT-02/03/04 |
| B5 | Legality is unobservable in state (mana not encoded); mask is exogenous | action_space audit §3 | E-ENC-03, E-ACT-01 |
| B6 | No temporal features (trends/velocities) despite delta-based rewards | encoder audit §4.6 | E-ENC-04 |
| B7 | Sim-vs-physical observation equivalence unproven (same brain, untested channel semantics) | claims SEN-004 | E-PHYS-03 |

## C. Learning-system gaps

| # | Gap | Evidence | Addressed by |
|---|---|---|---|
| C1 | No γ / bootstrapping / traces: terminal ±1 credits only final decision's KCs | RL doc §1 | E-SNN-03 |
| C2 | Scalar baseline can't represent state values; `value_estimate()` unused | claims RL-002 | E-RL-05 |
| C3 | Reward farming: opp_removal/board_delta fire on any opponent-board shrink; hand_delta farming | reward audit §2 | E-RL-02/03 |
| C4 | `max()` opp-life focusing rewards damaging the healthiest opponent | reward audit §2.4 | E-RL-03 |
| C5 | Confounded credit: Forge AI co-decides outcomes the fly is rewarded for | claims RL-004 | E-ARCH-01 |
| C6 | Action entropy / KC diversity not logged → collapse invisible | evaluation doc §1 | E-TRAIN-03 |
| C7 | Single opponent pool → overfit; no held-out discipline yet | evaluation doc §5 | E-EVAL-02 |

## D. Neuroscience-fidelity gaps (honesty, not performance)

| # | Gap | Evidence |
|---|---|---|
| D1 | 4064/97/344 are whole-CNS MaleCNS counts presented as MB scale | claims NEU-001; connectome doc §4 |
| D2 | "DPR" name overstates fidelity to Gkanias 2022 (not an RPE rule; different structure) | claims NEU-002; mb_facts §2 |
| D3 | dan_to_mbon random → compartmental valence structure is decorative | claims NEU-005 |
| D4 | No spike timing, no eligibility, no recurrent APL | SNN doc §1, §3 |

## E. Physical-table gaps

| # | Gap | Evidence | Addressed by |
|---|---|---|---|
| E1 | No labeled OCR/tap-detection benchmark; Tesseract failure modes unmeasured on MTG foils/sleeves | claims VIS-001/002 | E-PHYS-01/02 |
| E2 | Impossible-transition validation of event stream untested (fuzz) | claims VIS-003 | E-PHYS-04 |
| E3 | Vision/server security surface untested (malformed JSON, oversized frames) | claims SEC-001 | E-SEC-01 |

## F. Process gaps

- No eval harness / results directory convention (evaluation doc §6).
- No pre-registered success criteria (RL doc §6 pre-registration note).
- Dual Python environments (venv vs system python3) make `make test` and
  venv results diverge silently (claims TRN-003).
- Thread.sleep(15000) startup sync is brittle (claims PRF-002) — flagged,
  not changed, per audit rules.

---

Priority reading order for a new contributor: this file →
[flycommander_current.md](flycommander_current.md) →
[../claims/claims.yaml](../claims/claims.yaml) →
[../experiments/roadmap.md](../experiments/roadmap.md).
