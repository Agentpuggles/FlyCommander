# Spiking Neural Network Mechanics: Implementation vs Established Methods

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Evidence: [brain/lif_engine.py](../../brain/lif_engine.py),
[brain/mushroom_body.py](../../brain/mushroom_body.py),
[brain/dopamine_plasticity.py](../../brain/dopamine_plasticity.py) + literature below.

---

## 1. What the implementation actually does

- **LIF populations** (`LIFPopulation`): discrete-time leaky integrators;
  `rates(v, steps=8)` runs 8 micro-steps and returns mean firing activity —
  used as a *rate readout*, not as event-driven spiking.
- **WTA:** `winner_take_all(kc_input, keep_fraction=0.10)` — a sort-based
  top-k; no lateral inhibition dynamics, no refractory period.
- **Plasticity:** single instantaneous update per decision,
  `delta = lr · outer(active_kc, dan_gate)`, clipped to [0, 2], blended
  toward a decayed equilibrium (ltm_decay 0.9995 × 0.5 consolidation).
- **No eligibility traces, no STDP windows, no spike-timing dependence** of
  any kind. Claims SNN-001/SNN-002.

## 2. Numerical observations

- Euler integration at the current dt/τ regime is stable for the values in
  use (τ on the order of the step size; no divergence observed in tests).
  Stress-testing dt→τ ratios and near-threshold inputs is untested
  (roadmap E-SNN-02).
- Clipping [0,2] bounds runaway weights; combined with the 0.9995 decay the
  fixed point of an always-active KC/DAN pair is < w_max — no saturation
  lock observed, but not proven for adversarial input sequences.
- Dense matrices (4064×97 ≈ 394k weights) are trivially small for NumPy;
  sparsity of the *code* (406 active) is not exploited in the `outer()`
  update — could use sparse structure if profiling ever demands it.

## 3. Comparison with established methods

| Established approach | Used here? | Comment |
|---|---|---|
| Discrete LIF with refractory period | partial | no refractory modeled |
| Event-driven simulation | no | 8-step rate readout instead; fine at this scale |
| Eligibility traces (Izhikevich 2007; R-STDP) | **no** | biggest gap for multi-step credit |
| Reward-modulated STDP (Florian 2007; Izhikevich 2007) | no | timing-less Hebbian substitute |
| Neuromodulated third-factor rules (Gerstner et al. 2018, Neuron review) | conceptually | "neuromodulated" ≈ dan_gate; but no *dynamics* of the modulator |
| Sparse random projections + WTA (Kaneka/Olshausen-style) | yes | core of the design; biologically supported for MB |
| Dopamine as RPE (Schultz; and fly PAM-γ1 RPE neurons) | yes (scalar) | defensible engineering choice |
| Recurrent inhibitory WTA (APL loop) | no | static top-k instead |

Assessment: for a 4-action decision problem with one decision per window,
rate-readout LIF + gated Hebbian is *adequate*; the literature-backed
additions that would matter most are (a) eligibility traces for multi-step
credit and (b) a learned value function rather than a scalar baseline. Neither
requires real spiking. **Do not adopt STDP for fashion's sake** — with no
meaningful spike timing in the pipeline, STDP would be decorative; eligibility
traces are the cheap, testable upgrade (E-SNN-03).

## 4. Literature anchors

- Izhikevich 2007, *Cerebral Cortex* 17:2443 — dopaminergic STDP with
  eligibility traces; the canonical third-factor design.
- Florian 2007, *Neural Computation* 19:1468 — R-STDP convergence analysis.
- Gerstner et al. 2018, *Neuron* 99:274 — eligibility traces & neuromodulation
  review; maps exactly onto "dan_gate × eligibility" structure the code could
  adopt.
- Maass 1997 — LIF computational power (background).
- Frémaux & Gerstner 2016, *Front Neural Circuits* 9:85 — reward-modulated
  STDP survey.

## 5. Testable hypotheses (→ roadmap)

- E-SNN-01: LIF rates(8) vs static activation (ReLU/tanh) — is the "spiking"
  layer adding anything? Expect: equivalent; result informs whether to keep
  the LIF engine.
- E-SNN-02: stability sweep over dt/τ, near-threshold regimes.
- E-SNN-03: add eligibility trace (decay ~0.9/step) to DPR update — expect
  faster learning of multi-step sequences; measure via win-rate delta and
  credit-assignment diagnostics on synthetic tasks first.
