# Mushroom Body Neuroscience: Facts vs FlyCommander Assumptions

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Sources: primary literature (Aso 2014, Gkanias 2022 and others below);
repo code ([brain/connectome.py](../../brain/connectome.py),
[brain/mushroom_body.py](../../brain/mushroom_body.py),
[brain/dopamine_plasticity.py](../../brain/dopamine_plasticity.py)).
Labels: BIOLOGICALLY SUPPORTED (BS) / ENGINEERING APPROXIMATION (EA) /
SPECULATIVE (SP).

---

## 1. Reference numbers (adult *Drosophila* MB)

| Quantity | Biology (adult MB) | FlyCommander | Verdict |
|---|---|---|---|
| Kenyon cells | ~2,000 (Aso 2014) | 4,064 | EA — 2× biology; **the project's own number is a whole-CNS MaleCNS count, not an MB count** (see [connectome](../connectome/connectome_datasets.md)) |
| MBONs | 34 cells / 21 cell types (Aso 2014) | 97 | EA — again whole-CNS MaleCNS count, not MB |
| DANs | ~130–140 / ~20 cell types (Aso 2014) | 344 | EA — same provenance issue |
| PN channels | ~150 glomeruli / PN types | 64 | EA — fewer than biology, in the right ballpark |
| PN→KC fan-in | ~7 claws per KC (Aso 2014) | PN_PER_KC = 7 | **BS — matches** |
| KC sparsity | ~1–5% of KCs per odor, highly selective | 10% kept by WTA | EA — right order, ~2–10× denser than typical odor responses |
| MB compartments | ~15 compartments (Aso 2014), each = specific DAN+MBON pairing | none functional (signed half-split DAN matrix, decorative) | partial mismatch — see NEU-005 |
| Memory phases | γ lobe labile STM; αβ stable LTM; consolidation over hours | weight decay equilibrium blend (ltm_decay 0.9995, consolidation 0.5) | EA — plausible as engineering; no biological mechanism claimed in literature in this exact form |
| APL | one GABAergic neuron, broad MB feedback inhibition, WTA-like | global WTA over KC input, keep top 10% | BS in spirit (APL enforces sparsity); EA in mechanism (global top-k, not recurrent inhibitory dynamics) |

## 2. Learning and dopamine

- **BS:** KC→MBON synapses are the memory substrate; DAN input to a
  compartment gates potentiation/depression of that compartment's KC→MBON
  synapses; PAM cluster ≈ appetitive, PPL1 ≈ aversive signals; valence is
  compartment- and MBON-specific (Aso 2014; Claridge-Chang et al. 2009,
  Nature 462; Burke et al. 2012, J Neurosci 32).
- **BS:** odor→valence learning is surprisingly one-shot capable in flies —
  supports the idea that a small agent could learn significant valence from
  few games.
- **CAUTION (documented disagreement):** the project cites Gkanias et al.
  2022 (eLife 11:e75611, "incentive circuit"). Re-reading that paper: its DPR
  is ΔW = δ_j(t)·[k_i(t)+W−w_rest] with w_rest = 1, four synaptic effects
  (depression, potentiation, recovery, saturation), 12 neurons = 3 DAN types
  (discharging/charging/forgetting) × 3 MBON types (susceptible/restrained/
  LTM) per motivation, 2 motivations. **It is not an RPE rule** and its
  purpose is maximizing separation of reinforced inputs, not value learning.
  FlyCommander's update (`delta = lr · outer(active·KC, dan_gate)`, lr 0.01)
  is a *dopamine-gated Hebbian* rule. Naming it "DPR" overstates fidelity —
  claims NEU-002 records this as `implementation_status: conflicts`.
- **BS (weakly):** reward-prediction-error-like dopaminergic signaling does
  exist in flies (PAM-γ1 RPE neurons, e.g. Wakabayashi & Ichinose 2021-type
  findings; prior work by DasGupta/Bernard reviews), so using RPE as the DAN
  drive is a defensible *engineering* choice — but it is a hybrid, not a
  direct copy of either Aso-style compartmental valence or Gkanias DPR.

## 3. Where the analogy is strongest

1. Sparse random projection (PN→KC) + WTA sparsification → pattern
   separation. The fly's KC code genuinely expands dimensionality and
   decorrelates inputs (Campbell et al. 2018-type findings on KC expansion
   layer); the project's architecture implements exactly this shape.
2. A small reward-gated plastic locus (KC→MBON) as the only learned weights
   — matches the biological picture that MB memory lives at that synapse.
3. Compressed state → behavior via a modest readout (MBON→action).

## 4. Where the analogy is weakest (be honest in docs)

1. **No temporal dimension.** Real MB dynamics unfold over ~100 ms–seconds
   with oscillations, timing windows, and phase-locking; FlyCommander's
   "LIF" populations are evaluated for 8 steps and read as rates — there is
   no real spike timing, no STDP, no eligibility trace. Claims SNN-002.
2. **DAN→MBON structure is decorative.** `dan_to_mbon` is a fixed random
   matrix; the signed half-split (PAM/PPL1) has no compartment specificity,
   so the documented biological per-compartment valence map has no
   functional counterpart. The system behaves like a scalar gate × active-KC
   outer product.
3. **No recurrent inhibition.** APL is modeled as one-shot top-k, not a
   recurrent inhibitory loop; biological WTA emerges from dynamics, ours is
   an explicit sort.
4. **No multiple memories / no extinction.** Single weight matrix; no
   decay-based forgetting except the LTM blend; no reversal learning support.
5. **Innate valence absent.** Real MBONs have baseline valences (approach/
   avoid readouts exist without training); `w_mbon_action` starts as small
   random noise instead.

## 5. Numbers FlyCommander should know (citation-ready)

- Aso et al. 2014, *Cell* 157(3): 586–596, doi:10.1016/j.cell.2014.02.045 —
  the compartment map; source of ~2000 KC / 34 MBON / ~130 DAN counts.
- Gkanias et al. 2022, *eLife* 11:e75611, doi:10.7554/eLife.75611 (PMC8975552)
  — incentive circuit; DPR formula; NOT RPE.
- Claridge-Chang et al. 2009, *Nature* 462: 930–934 — aversive memory at
  PPL1-γ1pedc; first "dopamine = teaching signal" optogenetic proof in fly.
- Burke et al. 2012, *J Neurosci* 32(40): 13962–13975 — appetitive PAM
  reinforcement.
- Waddell 2010, *Curr Biol* 20: R752 — review of MB valence circuitry.
- ModA/Mushroom-body connectomics: see
  [connectome/connectome_datasets.md](../connectome/connectome_datasets.md)
  for FlyWire/MaleCNS numbers and licensing.

## 6. What remains unknown

- Whether 8-step "rates" from the LIF engine produce any useful dynamic
  structure at all, or are equivalent to a static activation function
  (testable: E-SNN-01 — compare LIF rates vs plain ReLU readout).
- Whether the 2×-biological KC count changes anything once PN→KC is fixed
  random with 7 claws (testable: E-NEU-01 — scale sweep KC 500–8000).
- Whether DAN compartmentalization would add function here or is redundant
  given 4 actions (testable: E-NEU-02 — structured dan_to_mbon vs random).
