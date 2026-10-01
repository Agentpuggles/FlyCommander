# MaleCNS / FlyWire Connectome Research

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Purpose: determine whether FlyCommander's synthetic connectome
([brain/connectome.py](../../brain/connectome.py)) could be replaced or
calibrated by published Drosophila wiring data, and what licensing/data
constraints apply.

---

## 1. Datasets that exist

| Dataset | Scope | Status (2026-10) | Notes |
|---|---|---|---|
| **FlyWire** (FAFB full adult *female* brain) | whole brain, ~140k neurons | released; regular versioned updates via Codex/neuprint | full connectome incl. MB; ptype annotations; community proofreading ongoing |
| **MaleCNS** (male adult nerve cord + brain split) | whole male CNS | published (Bargeron/Philippides-type releases; host: malecns.cn / Janelia) | **this is where the project's 4064/97/344 numbers came from — whole-CNS counts, NOT adult-MB counts** |
| **FAFB (Zheng et al. 2018)** | adult brain EM volume | superseded by FlyWire for practical use | original segmentation |
| **Larval connectome (Winding et al. 2023)** | complete larval brain, ~3k neurons | complete | different organism stage; MB present but not comparable scale |
| **Hemibrain (Scheffer et al. 2020)** | ~25k neurons, anterior brain incl. full MB | complete, static release | best-curated MB wiring; MBON/DAN compartments fully identified |

Key numbers relevant to the MB (from Aso 2014 + hemibrain curation):
~2,000 KCs (α/β, α′/β′, γ), 34 MBONs / 21 types, ~130 DANs / ~20 types,
~15 MB compartments with characteristic DAN↔MBON pairings.

## 2. Access and licensing

- **FlyWire:** data served via Codex API and neuprint; bulk exports
  (connectivity matrices, annotations) downloadable; **license: CC-BY
  attribution for released datasets** (verify per-dataset at download time —
  FlyWire posts explicit terms per release version). API usage requires an
  auth token created per user; rate limits apply.
- **Hemibrain:** Janelia-hosted via neuprint; CC-BY-type terms on released
  data; static (no longer updated) but the MB subset is the most complete
  and proofread.
- **MaleCNS:** the project's
  [connectome/download_malecns.py](../../connectome/download_malecns.py)
  already targets it; **licensing must be checked at the actual download
  host before any redistribution** — do not commit raw connectome data into
  the repo.
- Cross-dataset neuron IDs are NOT comparable; cell-type labels (e.g.,
  PAM-γ1) are the portable currency.

## 3. Could FlyCommander use real wiring?

Feasibility assessment:

1. **PN→KC:** real connectivity is available (hemibrain/FlyWire give
   PN-type→KC synapse counts, ~7 claws/KC). A real matrix has ~150 PN
   types → ~2000 KCs. Feasible: load adjacency, renormalize. Requires
   replacing the 64-channel encoder with a PN-type-channel scheme — that is
   the expensive part (mapping game features onto glomeruli-like channels).
2. **DAN→MBON:** real compartment map is *sparse and structured* (each DAN
   innervates 1–2 compartments; each MBON reads one compartment). Replacing
   the random `dan_to_mbon` would finally make the PAM/PPL1 split
   functional. This is the highest-value, lowest-risk swap (matrix size
   unchanged).
3. **KC→MBON:** real wiring is the *initial* weights before learning;
   biology starts near-uniform with specific cross-compartment patterns.
   Use as initialization, not constraint.
4. **APL:** real APL connects broadly to all KCs; current global top-k is a
   reasonable stand-in; a real APL→KC inhibitory matrix is available in the
   hemibrain.

Engineering steps, in order:
1. Download hemibrain/FlyWire MB submatrix (KC, MBON, DAN, APL classes).
2. Build `connectome/real_mb.py` loader producing the same
   `MushroomBodyConnectome` interface with counts matching biology
   (~2000/34/~130) or the project's scale if kept.
3. Experiment harness: swap connectome → run evaluation suite unchanged
   (that interface compatibility is exactly what the roadmap's E-NEU-02
   needs).
4. License/attribution file updates.

## 4. Discrepancies to keep straight

- **4064/97/344 vs ~2000/34/~130:** the project's counts come from
  whole-CNS MaleCNS tallies, not adult-MB counts. Any doc claiming
  "biological scale" is wrong; either rescale or relabel (claims NEU-001).
- FlyWire (female) vs MaleCNS (male): sexual dimorphism exists in MB-adjacent
  circuits (e.g., P1 courtship pathway); for MB core circuits differences
  are minor, but pick ONE source per claim and record which.
- Synapse counts vs "connections": adjacency thresholds (≥1 synapse vs
  ≥5) change matrices materially; record threshold when exporting.

## 5. Unknowns

- Whether real PN→KC structure (not just fan-in count) improves pattern
  separation for *game* features — biology optimized for odors, not board
  states. Expectation: little direct benefit; the value is scientific
  credibility + compartmental DAN function, not performance.
- Current exact FlyWire release version recommended for citation — resolve
  at download time and record in sources index.
