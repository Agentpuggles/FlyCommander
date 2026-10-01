# Research Knowledge Base — Index

Created by the 2026-10-01 master audit. Entry points:
**[FINAL_REPORT.md](FINAL_REPORT.md)** (read first) →
[architecture/flycommander_current.md](architecture/flycommander_current.md) →
[experiments/roadmap.md](experiments/roadmap.md).

| Path | Contents |
|---|---|
| [README.md](README.md) | conventions: BS/EA/SP labels, confidence, version discipline |
| [FINAL_REPORT.md](FINAL_REPORT.md) | the master audit report |
| [architecture/flycommander_current.md](architecture/flycommander_current.md) | what is actually built (component map, control boundary, state-loss chain) |
| [architecture/research_gaps.md](architecture/research_gaps.md) | missing/uncertain/incorrect, by component (A–F sections) |
| [claims/claims.yaml](claims/claims.yaml) | 40 machine-readable claims with sources, versions, confidence, validation needs |
| [magic/comprehensive_rules_findings.md](magic/comprehensive_rules_findings.md) | rules audit: missing-state ↔ rules gaps, verified facts, Duel/poison divergences |
| [forge/forge_internals.md](forge/forge_internals.md) | Forge 2.0.15 boot path, AI architecture, RNG, patch surface, divergences |
| [neuroscience/mb_facts.md](neuroscience/mb_facts.md) | primary-literature MB facts vs project assumptions (BS/EA/SP) |
| [connectome/connectome_datasets.md](connectome/connectome_datasets.md) | FlyWire/MaleCNS/hemibrain: licensing, access, swap feasibility |
| [snn/lif_and_plasticity.md](snn/lif_and_plasticity.md) | LIF numerics, R-STDP/eligibility literature, method comparison |
| [reinforcement_learning/credit_assignment.md](reinforcement_learning/credit_assignment.md) | DPR suitability, credit assignment, alternatives, pre-registration |
| [sensory_encoding/encoder_audit.md](sensory_encoding/encoder_audit.md) | 64-channel audit, discards, empirical separation probe |
| [sensory_encoding/encoder_probe.py](sensory_encoding/encoder_probe.py) | reproducible probe (results in docstring) |
| [action_space/action_space_audit.md](action_space/action_space_audit.md) | 4-macro-action semantics, boundary, Markov sufficiency |
| [rewards/reward_audit.md](rewards/reward_audit.md) | exact reward mechanics, hacking vectors, terminal anomaly |
| [training/training_protocol.md](training/training_protocol.md) | checkpoint/log state, curriculum phases, seed discipline |
| [evaluation/evaluation_protocol.md](evaluation/evaluation_protocol.md) | metrics, controls, ablations, statistics, leakage |
| [computer_vision/ocr_and_detection.md](computer_vision/ocr_and_detection.md) | pipeline assessment, Tesseract failure modes, alternatives |
| [physical_table/state_reconciliation.md](physical_table/state_reconciliation.md) | event sourcing, invariants, camera geometry |
| [security/security_review.md](security/security_review.md) | attack surface, realistic findings, do-not-do list |
| [performance/performance_notes.md](performance/performance_notes.md) | cost inventory, bottleneck hypotheses, do-not-optimize list |
| [related_projects/related_projects.md](related_projects/related_projects.md) | MTG AI, engines, card-recognition, SNN agents survey |
| [academic/academic_index.md](academic/academic_index.md) | primary papers by domain with DOIs |
| [sources/sources_index.md](sources/sources_index.md) | master source index + recorded disagreements |
| [open_questions/open_questions.md](open_questions/open_questions.md) | UNKNOWN/WHY/HOW-TO list incl. unknown unknowns |
| [experiments/roadmap.md](experiments/roadmap.md) | 20+ experiments with H/IV/DV/controls/seeds, execution order |

Suggested reading order for a new contributor:
1. FINAL_REPORT.md
2. architecture/ (both files)
3. claims/claims.yaml (skim; it is the evidence spine)
4. experiments/roadmap.md §1 (correctness gates)
5. Domain docs as needed.
