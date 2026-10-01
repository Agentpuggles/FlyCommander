# Source Index (Master)

Status: maintained index. Access dates recorded per source-discipline rules.
Authority levels: OFFICIAL / PRIMARY / CODE / DATASET / SECONDARY.

---

## Magic: The Gathering

| Source | Type | Version/Access | Used for |
|---|---|---|---|
| MTG Comprehensive Rules (Wizards of the Coast) | OFFICIAL | eff. 2025-06-06; local extract `/tmp/mtgrules.txt` (12,549 lines), accessed 2026-10-01 | layers 613, SBA 704, combat 508–510, Commander 903 (40 life, 21 dmg, tax), poison 10 |
| Scryfall API docs (https://scryfall.com/docs/api) | OFFICIAL | accessed 2026-10-01 | Accept-header requirement (400 w/o), ~10 req/s rate limit, bulk data daily JSONL.gz |
| Scryfall bulk data (https://scryfall.com/docs/api/bulk-data) | DATASET | daily; Oracle ~23.4 MB JSONL.gz observed | card identity/legality pipeline |
| Forge docs/AI.md + 2.0.15 source tree | OFFICIAL (project) | 2.0.15 | AI architecture (SpellAbilityChooser, CardEvaluator, combat classes) |
| Duel Commander rules (duelcommander.com) | OFFICIAL (variant) | accessed 2026-10-01 | divergence notes: 20-life start, 21-commander-damage rule removed — NOT used by this project |

## Neuroscience

| Source | Type | Version/Access | Used for |
|---|---|---|---|
| Aso et al. 2014, Cell 157(3) (doi:10.1016/j.cell.2014.02.045) | PRIMARY | re-read 2026-09 session | MB counts, compartments, claws |
| Gkanias et al. 2022, eLife 11:e75611 (PMC8975552) | PRIMARY | re-read this audit | DPR formula; not-RPE finding |
| Claridge-Chang 2009; Burke 2012; Waddell 2010 | PRIMARY | standard field refs (see academic index caveats) | valence/DAN clusters |
| FlyWire (flywire.ai) | DATASET | accessed 2026-10-01 (licensing page) | CC-BY-type terms; versioning via Codex/neuprint |
| Hemibrain (Scheffer et al. 2020, eLife 9:e57443) | DATASET/PRIMARY | accessed 2026-10-01 | MB-subset wiring recommendation |
| MaleCNS releases | DATASET | via connectome/download_malecns.py target | whole-CNS counts source |

## SNN / RL

| Source | Type | Used for |
|---|---|---|
| Izhikevich 2007 (doi:10.1093/cercor/bhl149); Florian 2007; Frémaux & Gerstner 2016; Gerstner et al. 2018 | PRIMARY | eligibility traces / R-STDP family |
| Sutton & Barto 2018 | PRIMARY (book) | TD/baseline/credit fundamentals |
| Ng et al. 1999 (ICML) | PRIMARY | shaping policy-invariance caveat |
| Amodei et al. 2016 (arXiv:1606.06565) | PRIMARY | reward-hacking taxonomy |

## Vision / OCR

| Source | Type | Used for |
|---|---|---|
| Smith 2007, ICDAR (doi:10.1109/ICDAR.2007.437) | PRIMARY | Tesseract engine internals |
| Boldt et al. 2019 (EasyChair QT9R) | PRIMARY | MTG-OCR error evidence |
| Tesseract docs (OEM/PSM), OpenCV docs (cv2 5.0.0 API) | OFFICIAL | config guidance |
| r/computervision MTG-OCR thread (2025) | SECONDARY | community failure-mode corroboration only |

## Forge / software

| Source | Type | Used for |
|---|---|---|
| forge-gui-desktop-2.0.15-jar-with-dependencies.jar | CODE | bytecode verification (javap), boot path, MyRandom |
| Forge GitHub repository (forge-gui) | CODE/OFFICIAL | docs/AI.md, changelogs |
| FlyCommander repo itself | CODE | all implementation claims (files cited per doc) |

## Web sources (searched 2026-10-01, URLs in related_projects doc)

- card-forge.github.io/forge/ (Forge homepage)
- github.com/KoalaTrapLord/commander-ai-lab (closest related project)
- dokkodolabs.itch.io/neo-forge (Commander client)
- wingedsheep.com Argentum writeup (2026)
- news.ycombinator.com/item?id=38651346 (CLIPS card compiler)
- reddit r/computervision MTG OCR thread (secondary)

## Disagreements encountered (kept explicit, not resolved by fiat)

1. Encoder-probe overlap numbers: earlier ad-hoc probe (0.61–0.68) vs
   committed reproducible probe (0.17–0.58 Jaccard) — both recorded;
   committed probe is canonical (encoder_audit.md §3).
2. Biology-scale counts: Aso adult-MB (~2000/34/~130) vs MaleCNS whole-CNS
   (4064/97/344) — both true of *different scopes*; project mislabels scope
   (claims NEU-001).
3. "DPR" naming: project docstring vs Gkanias 2022 actual rule — conflict
   recorded, project rule renamed in docs as "dopamine-gated Hebbian" (claims
   NEU-002).
4. Comprehensive Rules 2025-06-06 is the governing version here; any newer
   WotC release supersedes silently — re-check before rule-citing after set
   releases.
