# Computer Vision / OCR Research (Physical Table)

Status: RESEARCH DOCUMENT. Last verified: 2026-10-01.
Evidence: [vision/](../../vision/) and [physical/](../../physical/) code reads;
environment probe (venv: OpenCV 5.0.0, pytesseract/Tesseract 5.5.3; system
python3 lacks cv2 — `make test` therefore skips vision tests there).
External sources listed in [../sources/vision_ocr_sources.md](../sources/vision_ocr_sources.md).

---

## 1. Current pipeline (as implemented)

1. **Detection:** OpenCV-based card region detection in the camera frame
   (contour/quad heuristics).
2. **Tracking:** IoU-based assignment across frames → stable per-card track
   IDs ([vision/zone_tracker.py](../../vision/zone_tracker.py)).
3. **Tap detection:** orientation change over time
   ([vision/tap_detector.py](../../vision/tap_detector.py)).
4. **Identification:** crop → preprocess → Tesseract OCR of the collector
   number (+ set code) → exact match against Scryfall
   ([physical/identifier.py](../../physical/identifier.py),
   [physical/scryfall_cache.py](../../physical/scryfall_cache.py)).
5. **Reconciliation:** vision events + human confirmations merged by the
   physical engine into PhysicalGameState (see
   [../physical_table/state_reconciliation.md](../physical_table/state_reconciliation.md)).

## 2. How collector-number OCR fails (literature + mechanism analysis)

Tesseract on MTG collector numbers is a *hard* instance of OCR: tiny text
region, high-contrast but stylized font, and physically hard imaging. Known
failure modes:

1. **Foil cards** — specular reflection breaks binarization; the collector
   number region has metallic background on many sets. Expected worst case.
2. **Sleeves** — change contrast, add glare and halation; shift color balance
   that adaptive thresholds depend on.
3. **Tilt/perspective** — small in-plane rotation (<10°) is handled by
   Tesseract reasonably; perspective skew (card not flat to camera) is not.
4. **Overlap/occlusion** — collector number is at the bottom-left edge of
   modern frames — exactly where a sleeve corner or neighboring card can
   cover it.
5. **Alternate treatments** — full-art frames, extended art, promo/box
   toppers, invert frames (showcase) move or restyle the collector number;
   some (e.g., certain Japanese alt-art) use non-Latin fonts nearby.
6. **Ambiguity after OCR:** even 98% per-character accuracy on "195" can
   yield "1985", "195", "795" — but exact-match against Scryfall saves many
   of these (candidate sets are sparse). Real danger is *plausible wrong*
   matches within the same set (e.g., "13" vs "18").
7. **Boldt et al. (2019)** (MTG card OCR study) report errors even with
   aligned, clear images — i.e., a nonzero irreducible error rate for
   Tesseract-only approaches; supports a *never fully trust OCR* design.

Mitigations ranked by effort/benefit:
- Constrain OCR to expected number region (already done via crop).
- Character allowlist `0123456789a-zA-Z` + PSM 7 (single line); OEM 1.
- Multi-scale OCR + confidence voting; reject low-confidence and fall back
  to candidate-set + human confirmation (the project's hybrid philosophy —
  correct approach; make the fallback *explicit* per-card).
- Fallback identifiers: card frame color histogram + mana-cost region as
  secondary signals before human query.
- ArUco/card-size fiducials solve *registration and tap geometry* but cannot
  identify pre-existing cards; only useful for new infrastructure.

## 3. Alternatives compared

| Approach | Pros | Cons | Verdict for FlyCommander |
|---|---|---|---|
| Classical OpenCV (current) | deterministic, no GPU, debuggable | brittle to lighting | keep as first stage |
| Tesseract collector-number OCR | free, offline, exact-ID when clean | failure modes above | keep + confidence gating |
| Template matching vs Scryfall art crops | robust to font issues | needs good alignment; DB of crops | good fallback #1 |
| YOLO-based card detector | robust detection in clutter | needs labeled MTG data + training infra | overkill now; revisit if detection is the bottleneck |
| Vision-language models | flexible, handles odd frames | network dependency, nondeterminism, cost | against project's offline/deterministic ethos; last resort |
| ArUco fiducials | rock-solid geometry/taps | requires marking cards | user choice; document as option |

## 4. What should stay deterministic (design principle)

Registration, tap/zone geometry, ID resolution (exact Scryfall match), and
state reconciliation should be deterministic functions of inputs. Only
detection/OCR may be heuristic, and their *uncertainty must be carried
forward* as confidence values into the state layer, not silently dropped.

## 5. Missing evaluation assets (blockers for honesty)

- No labeled photo corpus; no annotated tap videos. Until E-PHYS-01/02 run,
  any accuracy statement about the physical pipeline is unsupported.
