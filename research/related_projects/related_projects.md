# Related Projects Survey

Status: RESEARCH DOCUMENT. Sources verified via web search 2026-10-01
([sources/web_sources.md](../sources/web_sources.md) for URLs/access dates).
Licenses must be re-verified in each repo before copying any code.

---

## 1. MTG rules engines

| Project | What it is | Lessons for FlyCommander | License (verify) |
|---|---|---|---|
| **Forge** (card-forge.github.io) | the engine we build on | heuristic AI architecture (docs/AI.md); card scripts; MyRandom seeding | open source (per-repo) |
| **XMage** | client-server MTG with online play | server/client split; protocol design; also heuristic AI | MIT-ish (verify) |
| **Magarena** | open-source MTG AI player (older) | documented the cost of hand-written card AI; card-grammar approach | GPL-family (verify) |
| **Argentum** (wingedsheep.com, 2026 writeup) | hobbyist rules engine + client | shows the current practical floor for from-scratch engines; WebSocket session design | personal project |
| **phase.rs** | web MTG rules engine (Rust) | alternative implementation strategies | verify |
| **CLIPS-based card compiler** (HN 2023) | English card text → rules DSL | the "parse Oracle text into effect grammar" approach — relevant if fly ever needs card semantics | research code |

Takeaway: every serious engine treats rules fidelity as the moat; AI layers
are thin heuristic/selector layers on top. FlyCommander's choice to *reuse*
Forge rather than build an engine is correct and should be stated as a
strength in docs.

## 2. MTG AI / ML projects

| Project | What it does | Lessons |
|---|---|---|
| **commander-ai-lab** (GitHub, KoalaTrapLord) | "rules-complete Forge engine + Monte Carlo simulator + LLM deck builder + ML training pipeline" for Commander | closest known relative; validates Forge-as-simulator for Commander; compare their reward/eval conventions; **verify existence/license before drawing on it** |
| **Neo Forge** (DokkodoLabs, itch.io) | Commander vs AI client on Forge | UX packaging of exactly our simulation target; not research |
| **Magarena AI literature talks** ("Lessons from Developing an AI to Play Magic") | heuristic AI postmortems | proposal/candidate-scoring architecture parallels ours |
| Academic MTG agents (various) | mostly limited-format or micro-decision studies | none address Commander multiplayer politics — FlyCommander's niche is genuinely underexplored |

Takeaway: no published work does *spiking/neuro-inspired Commander play* —
the project's scientific niche appears open (as of 2026-10-01 search).

## 3. Card recognition systems

- Community OCR projects (e.g., the r/computervision "Reviving MTG Card
  Identification — OCR → LLM cleanup" thread, 2025): report the same
  Tesseract failure profile our design anticipates (preprocessing sensitivity,
  foil/sleeve trouble); LLM cleanup helps but breaks determinism.
- Boldt et al. 2019 (EasyChair preprint): MTG OCR benchmark study; errors
  persist even with clean images → supports confidence-gated + human-confirm
  design.
- Commercial-grade approaches (TCGPlayer scanning apps) use proprietary
  embeddings + cloud; not reproducible.

## 4. Spiking / neuromorphic game agents & fly-inspired models

- **Gkanias et al. 2022 (eLife)** — incentive circuit in 12 neurons; closest
  philosophical relative; demonstrates fly-mechanism → autonomous-agent
  translation; also demonstrates how much simplification is accepted in
  publication-grade work.
- Reward-modulated STDP literature (Izhikevich 2007; Florian 2007; Frémaux
  & Gerstner 2016) — the methods family our DPR approximates.
- NeuroBench / snnTorch / Norse ecosystems — tooling if the project ever
  needs *real* spiking simulation; current scale doesn't justify them.
- Connectome-derived robotics (FlyWire/hemibrain-based navigation models) —
  show the "real wiring into small agent" pattern our connectome doc
  describes; all in navigation/olfaction, none in strategic games.

## 5. What FlyCommander can learn / should avoid

**Learn:**
- Forge-as-oracle + thin-learner layer is the community-standard shape.
- Held-out deck evaluation and pool-overfitting awareness (commander-ai-lab
  discusses; our eval doc operationalizes).
- Confidence-gated OCR with human fallback (vision literature + practice).

**Avoid:**
- LLM-in-the-loop decision cleanup (nondeterminism, cost, kills the
  scientific point of a small-brain agent).
- Copying code from GPL-family engines into our patch (license entanglement)
  — read for ideas, never paste.
- Chasing YOLO/VLM pipelines before the labeled-data bottleneck is solved.

## 6. Honest gaps in this survey

- commander-ai-lab was located but not deeply inspected (time; low evidence
  it is maintained). Before relying on it: verify activity, license, and
  whether its "Monte Carlo simulator" runs Commander-legal multiplayer.
- No search was made for *non-English* MTG-AI work; possible blind spot.
- Neuromorphic+games literature beyond R-STDP surveys was sampled, not
  exhausted.
