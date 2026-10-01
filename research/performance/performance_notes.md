# Performance Notes

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Basis: architecture analysis + matrix sizes from
[brain/connectome.py](../../brain/connectome.py); no profiling run yet
(E-PERF-01 defines it; audit rule: measure before optimizing).

---

## 1. Cost inventory (analytical, to be verified by E-PERF-01)

| Stage | Expected magnitude | Basis |
|---|---|---|
| JVM boot incl. `FModel.initialize` | tens of seconds, **once per process** | bytecode-verified boot path; known Forge behavior |
| Per-decision: Java obs build | sub-ms to few ms | simple field reads |
| Per-decision: HTTP + JSON round-trip | ~0.1–2 ms localhost, but library overhead can dominate; **suspected dominant per-decision cost** | small payloads, but per-call framework overhead |
| Python: `observation_to_state` | µs | 64 floats |
| Python: encode (tanh + 64→4064 matmul + WTA sort) | ~0.1–1 ms | 4064-vector ops |
| Python: MB forward (4064→97, 344→97, 97→4) | <0.5 ms | ~400k multiply-adds |
| Python: DPR update (outer product 406×97 + clip + blend) | <1 ms | sparse code, dense storage |
| Checkpoint save (npz compressed ~115 KB) | ms | tiny |
| Vision: detection+tracking per frame | 10–50 ms @1080p classical CV | typical OpenCV contour pipeline |
| OCR per identified card | 50–500 ms (Tesseract) | engine speed, depends on PSM/scale |
| Scryfall exact-match lookup | 0 cached / ~100–300 ms network | rate limit ~10 req/s |

## 2. Bottleneck hypotheses (test, then fix)

1. **H1 (likely):** per-decision HTTP+JSON overhead dominates training
   wall-clock once JVM boot is amortized → mitigation: persistent socket /
   batching observations is *not* needed unless games/hour blocks
   experiments; measure first.
2. **H2 (likely):** throughput is actually gated by Forge game simulation
   speed with heuristic AI (its combat simulations), not by the fly at all.
   If so, the fly is never the bottleneck and Python perf work is pointless.
3. **H3:** vision identification (not detection) dominates physical-mode
   latency → mitigate by identifying on pickup rather than continuous scan.

## 3. Structural notes

- Dense 4064×97 storage for kc_to_mbon is fine (~3 MB fp32); sparsity
  would add complexity for no measurable gain at this size.
- The update is O(active KCs × n_mbon) per step — already optimal shape.
- `Thread.sleep(15000)` startup sync (claims PRF-002): wasted up to 15 s
  per JVM boot; replace-with-readiness-poll is a *bug-adjacent* perf item,
  flagged only.
- Memory: the pair of processes is modest (<1 GB Python, JVM heap dominated
  by card DB); bulk-data parse is the only spike risk.

## 4. Do-not-optimize list

- The LIF engine's 8-step loop (µs-scale).
- JSON format (human-debuggable beats binary until H1 proves otherwise).
- Encoder hashing (µs).
- Anything in vision before E-PHYS benchmarks exist (no baseline to beat).

## 5. Experiment

E-PERF-01 (roadmap §4): instrument decision round-trip with per-stage
timers for one 100-game run; report median/p95 per stage and games/hour.
Deliverable: one table in this file's "Measured" section (to be added after
the run — intentionally absent now; no invented numbers).
