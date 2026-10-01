# FlyCommander Research Knowledge Base

This directory is the project's **traceable research foundation**. Every
scientific or technical claim that shapes FlyCommander design should live
here with a source, a confidence level, and an implementation status.

## Layout

| Directory | Contents |
| --- | --- |
| [architecture/](architecture/) | What the code *actually* does (`flycommander_current.md`), research gaps (`research_gaps.md`) |
| [magic/](magic/) | Comprehensive Rules findings relevant to the AI state model |
| [forge/](forge/) | Forge 2.0.15 internals, control boundary, determinism |
| [neuroscience/](neuroscience/) | Mushroom-body anatomy/physiology vs. our implementation |
| [connectome/](connectome/) | FlyWire/MaleCNS datasets, licensing, swap feasibility |
| [snn/](snn/) | LIF numerics, plasticity rules, SNN-RL literature |
| [reinforcement_learning/](reinforcement_learning/) | Credit assignment, reward hacking, evaluation methodology |
| [sensory_encoding/](sensory_encoding/) | Encoder audit + measured collision results |
| [action_space/](action_space/) | Macro-action analysis |
| [rewards/](rewards/) | Reward function audit + hack inventory |
| [training/](training/) | Curriculum & protocol design |
| [evaluation/](evaluation/) | Metrics, ablations, reproducibility |
| [computer_vision/](computer_vision/) | Detection/OCR/tracking literature & practice |
| [physical_table/](physical_table/) | State reconciliation, hidden info at the table |
| [security/](security/) | Local attack surface audit |
| [performance/](performance/) | Known/likely bottlenecks |
| [related_projects/](related_projects/) | Comparable systems and what to learn from them |
| [academic/](academic/) | Primary-literature index |
| [claims/](claims/) | Machine-readable claims database (`claims.yaml`) |
| [open_questions/](open_questions/) | UNKNOWN / WHY IT MATTERS / HOW TO INVESTIGATE |
| [experiments/](experiments/) | `roadmap.md` with hypothesis-driven experiments |
| [sources/](sources/) | Per-domain source index with URLs, versions, access dates |

## Conventions

- Every claim file marks each claim:
  - **BIOLOGICALLY SUPPORTED** / **ENGINEERING APPROXIMATION** / **SPECULATIVE**
  - **VERIFIED IN REPO** / **ASSUMED** (implementation status)
  - Confidence: HIGH / MEDIUM / LOW
- Version-sensitive facts carry the version and access date
  (e.g., "Comprehensive Rules effective 2025-06-06", "Forge 2.0.15",
  "FlyWire FAFB v783", "MaleCNS released 2025").
- Disagreements between sources are **kept and documented**, never silently
  merged.

## Entry points

1. [FINAL_REPORT.md](FINAL_REPORT.md) — the full audit
2. [architecture/flycommander_current.md](architecture/flycommander_current.md) — code-verified architecture
3. [experiments/roadmap.md](experiments/roadmap.md) — what to run next
4. [claims/claims.yaml](claims/claims.yaml) — machine-readable claims DB
