#!/usr/bin/env python3
"""FlyCommander — vision accuracy report (end-to-end, offline, reproducible).

This is the script that answers "how good is recognition, really?" — and it
answers it against conditions that used to break the OCR pipeline: steep
perspective, rotation (including tapped/sideways and upside-down cards),
sleeves, foil, glare, blur, occlusion, multiple cards and small cards.

Two measurement modes:

* ``--mode recognition`` (fast): clean renders are rectified and warped by
  ``augmented_capture`` (the same transform used in training), then matched.
  This isolates recognition accuracy.
* ``--mode end-to-end`` (default): full pipeline — detector → sub-pixel
  rectification → embedding search → fused rerank — on composited scenes with
  ground-truth corners. Detection recall and corner error are reported too.

``--mode all`` runs both and prints a single report (also written as JSON).

The script never needs a camera, the network, or real card imagery: the
library and the captures are procedural, seeded and exactly labelled. Point
``--captures`` at a folder of real photos (named ``<set>_<number>.<ext>``) to
measure the same metrics on real captures once you have them.

Outputs
-------
* human-readable table to stdout
* ``--json PATH`` (default ``logs/eval_vision.json``) with the full breakdown
* with ``--calibrate``: a calibrated confidence mapping written to
  ``vision/weights/calibration.json`` (consumed by ``CardDatabase.build_index``)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision import rectify as R                                        # noqa: E402
from vision.detector import CardDetector                               # noqa: E402
from vision.embeddings import load_embedder                            # noqa: E402
from vision.matcher import CardIndex, CardMatcher                      # noqa: E402
from vision.synthetic import (CardSpec, augmented_capture,             # noqa: E402
                              compose_scene, default_library,
                              random_conditions, render_card)

WEIGHTS_DIR = REPO_ROOT / "vision" / "weights"
DEFAULT_LOGS = REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# library + index
# ---------------------------------------------------------------------------
def build_library(n: int, embedder=None, seed: int = 7, use_files: bool = False
                  ) -> tuple[list[CardSpec], CardIndex, CardMatcher]:
    specs = default_library(n, seed=seed)
    embedder = embedder or load_embedder()
    index = CardIndex.from_specs(specs, embedder=embedder,
                                 meta={"source": "eval", "count": n})
    # dense rerank vectors straight from the clean renders
    from vision.embeddings import dense_features
    index.dense_vectors = np.stack([dense_features(render_card(s)) for s in specs])
    matcher = CardMatcher(index, embedder=embedder)
    return specs, index, matcher


def _conditional_specs(specs: Sequence[CardSpec], conditions: dict[str, bool]
                       ) -> list[CardSpec]:
    """Pick library cards that actually exercise a requested condition."""
    if conditions.get("foil"):
        return [s for s in specs if s.seed % 2 == 0] or list(specs)
    return list(specs)


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------
class Metric:
    """Accuracy accumulator with per-condition breakdown."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n = 0
        self.top1 = 0
        self.top3 = 0
        self.top5 = 0
        self.unknown = 0
        self.wrong_accept = 0        # confident but wrong
        self.confidences: list[float] = []
        self.per_condition: dict[str, Counter] = defaultdict(Counter)
        self.ranks: list[int] = []
        self.timings: dict[str, list[float]] = defaultdict(list)

    def add(self, rank: int, unknown: bool, confidence: float,
            conditions: dict[str, Any] | None = None) -> None:
        self.n += 1
        self.ranks.append(rank)
        self.confidences.append(confidence)
        self.top1 += rank == 1
        self.top3 += rank <= 3
        self.top5 += rank <= 5
        self.unknown += bool(unknown)
        self.wrong_accept += bool(rank > 1 and not unknown)
        for key, value in (conditions or {}).items():
            if isinstance(value, bool):
                bucket = self.per_condition[key]
                bucket["n"] += 1
                bucket["top1"] += rank == 1
                bucket["hit"] += rank == 1 if value else 0
                bucket["total_true"] = bucket.get("total_true", 0) + (1 if value else 0)
                bucket["hit_true"] = bucket.get("hit_true", 0) + (
                    1 if (value and rank == 1) else 0)

    def summary(self) -> dict[str, Any]:
        n = max(1, self.n)
        out: dict[str, Any] = {
            "n": self.n,
            "top1": self.top1 / n,
            "top3": self.top3 / n,
            "top5": self.top5 / n,
            "unknownRate": self.unknown / n,
            "wrongAcceptRate": self.wrong_accept / n,
            "meanConfidence": float(np.mean(self.confidences)) if self.confidences else 0.0,
            "medianRank": float(np.median(self.ranks)) if self.ranks else -1,
        }
        breakdown = {}
        for key, bucket in self.per_condition.items():
            total_true = bucket.get("total_true", 0)
            breakdown[key] = {
                "n": bucket["n"],
                "top1WhenPresent": (bucket.get("hit_true", 0) / total_true)
                if total_true else None,
            }
        out["conditions"] = breakdown
        out["timingsMs"] = {k: round(float(np.mean(v)), 2)
                            for k, v in self.timings.items()}
        return out


# ---------------------------------------------------------------------------
# mode 1: recognition only (rectified captures)
# ---------------------------------------------------------------------------
CONDITION_SETS: dict[str, dict[str, Any]] = {
    "clean": dict(glare=0.0, blur=0.0, foil=False, sleeve=False, occlusion=False),
    "glare": dict(glare=1.0, blur=0.5, foil=False, sleeve=True, occlusion=False),
    "foil": dict(glare=0.6, blur=0.3, foil=True, sleeve=True, occlusion=False),
    "sleeve": dict(glare=0.3, blur=0.3, foil=False, sleeve=True, occlusion=False),
    "blur": dict(glare=0.0, blur=3.0, foil=False, sleeve=False, occlusion=False),
    "occlusion": dict(glare=0.2, blur=0.5, foil=False, sleeve=True, occlusion=True),
    "hard": dict(glare=0.8, blur=2.0, foil=True, sleeve=True, occlusion=True),
}


def eval_recognition(matcher: CardMatcher, specs: list[CardSpec], *,
                     n_per_condition: int = 40, seed: int = 99,
                     conditions: Sequence[str] | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric = Metric("recognition")
    per_condition: dict[str, dict[str, Any]] = {}
    for condition in (conditions or list(CONDITION_SETS)):
        cfg = CONDITION_SETS[condition]
        cond_metric = Metric(condition)
        for i in range(n_per_condition):
            spec = specs[int(rng.integers(0, len(specs)))]
            r = np.random.default_rng(seed + hash(condition) % 10_000 + i)
            img = augmented_capture(spec, r,
                                    hard=(condition in ("hard", "glare", "foil")))
            # force the requested conditions on top of the random capture
            if cfg["blur"]:
                from vision.synthetic import degrade
                img = degrade(img, r, blur=float(cfg["blur"]))
            rect = R.rectify(img, R.quad_from_bbox((0, 0, img.shape[1] - 1,
                                                    img.shape[0] - 1)),
                             refine=False)
            t0 = time.time()
            result = matcher.match_rectified(rect)
            metric.timings["matchMs"].append((time.time() - t0) * 1000)
            names = [c.name for c in result.candidates]
            rank = (names.index(spec.name) + 1) if spec.name in names else 99
            conds = {"hard": condition == "hard", "glare": bool(cfg["glare"]),
                     "foil": bool(cfg["foil"]), "sleeve": bool(cfg["sleeve"]),
                     "blur": bool(cfg["blur"]), "occlusion": bool(cfg["occlusion"])}
            conf = result.best.confidence if result.best else 0.0
            metric.add(rank, result.unknown, conf, conds)
            cond_metric.add(rank, result.unknown, conf, conds)
        per_condition[condition] = cond_metric.summary()
    out = metric.summary()
    out["byCondition"] = per_condition
    return out


# ---------------------------------------------------------------------------
# mode 2: end-to-end (detector in the loop, composited scenes)
# ---------------------------------------------------------------------------
def eval_end_to_end(matcher: CardMatcher, specs: list[CardSpec], *, n_scenes: int = 60,
                    seed: int = 5, hard: bool = True) -> dict[str, Any]:
    detector = CardDetector()
    rng = np.random.default_rng(seed)
    metric = Metric("end_to_end")
    detection = Counter()
    corner_errors: list[float] = []
    scene_times: list[float] = []
    for scene_i in range(n_scenes):
        cond = random_conditions(rng, hard=hard)
        n_cards = min(cond.n_cards, 3)
        pick = rng.choice(len(specs), size=n_cards, replace=False)
        chosen = [specs[int(i)] for i in pick]
        scene = compose_scene(chosen, cond, seed=int(rng.integers(0, 2**31 - 1)))
        t0 = time.time()
        rects = detector.detect_and_rectify(scene.image)
        detect_ms = (time.time() - t0) * 1000
        metric.timings["detectMs"].append(detect_ms)
        truths = [R.ensure_portrait(R.order_corners(c.quad)) for c in scene.cards]
        detection["cards"] += len(truths)
        matched_rects: list[R.RectifiedCard] = []
        for truth in truths:
            best = max(rects, key=lambda r: R.quad_iou(truth, r.quad), default=None)
            iou = R.quad_iou(truth, best.quad) if best is not None else 0.0
            detection["detected"] += iou > 0.5
            if best is not None and iou > 0.5:
                matched_rects.append(best)
                corner_errors.append(float(np.linalg.norm(
                    best.quad - truth, axis=1).mean()))
        for rect in rects:
            if max((R.quad_iou(rect.quad, t) for t in truths), default=0.0) < 0.5:
                detection["falsePositives"] += 1
        for card, rect in zip(scene.cards, matched_rects):
            t0 = time.time()
            result = matcher.match_rectified(rect)
            metric.timings["matchMs"].append((time.time() - t0) * 1000)
            names = [c.name for c in result.candidates]
            rank = (names.index(card.spec.name) + 1) if card.spec.name in names else 99
            metric.add(rank, result.unknown,
                       result.best.confidence if result.best else 0.0,
                       {"glare": rect.glare_frac > 0.05,
                        "occluded": card.visible_frac < 0.9,
                        "rotated": abs(rect.tilt_deg) > 20,
                        "sleeved": cond.sleeve, "foil": cond.foil})
        scene_times.append((time.time() - t0) * 1000)
    summary = metric.summary()
    summary.update({
        "detectionRecall": detection["detected"] / max(1, detection["cards"]),
        "detectionFalsePositives": int(detection["falsePositives"]),
        "cornerErrorPx": round(float(np.mean(corner_errors)), 2) if corner_errors else None,
        "cardsSeen": int(detection["cards"]),
    })
    return summary


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def fit_calibration(matcher: CardMatcher, specs: list[CardSpec], *,
                    n: int = 400, seed: int = 321) -> dict[str, float]:
    """Fit the fused→confidence mapping on synthetic captures (Platt scaling)."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, int]] = []
    for i in range(n):
        spec = specs[int(rng.integers(0, len(specs)))]
        r = np.random.default_rng(seed + i)
        img = augmented_capture(spec, r, hard=bool(i % 2))
        rect = R.rectify(img, R.quad_from_bbox((0, 0, img.shape[1] - 1, img.shape[0] - 1)),
                         refine=False)
        result = matcher.match_rectified(rect)
        if not result.best:
            continue
        correct = int(result.best.name == spec.name)
        rows.append((result.top_score, min(result.margin, 0.35), result.coverage, correct))
    if len(rows) < 20:
        return {}
    X = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float64)
    y = np.array([r[3] for r in rows], dtype=np.float64)
    # logistic regression on [fused, margin, coverage]
    w = np.zeros(3)
    b = 0.0
    for _ in range(400):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w -= 0.5 * (X.T @ (p - y) / len(y) + 1e-4 * w)
        b -= 0.5 * float((p - y).mean())
    slope = float(w[0])
    margin_weight = float(w[1] / max(1e-6, slope))
    coverage_weight = float(w[2] / max(1e-6, slope))
    return {"slope": slope, "bias": float(b - slope * 0.5),
            "marginWeight": margin_weight, "coverageWeight": coverage_weight}


# ---------------------------------------------------------------------------
# real captures (optional)
# ---------------------------------------------------------------------------
def eval_captures(matcher: CardMatcher, folder: Path) -> dict[str, Any]:
    """Evaluate on real photos named ``<set>_<number>.<ext>`` (ground truth
    from the filename). Detection runs for real; this is the number to trust
    once real photographs exist."""
    import cv2
    detector = CardDetector()
    metric = Metric("real_captures")
    files = [p for p in sorted(folder.rglob("*"))
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for path in files:
        key = path.stem.split("__")[0]
        truth_keys = [e.key for e in matcher.index.entries]
        if key not in truth_keys:
            continue
        spec_key = key
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        rects = detector.detect_and_rectify(frame)
        if not rects:
            metric.add(99, True, 0.0, {"detected": False})
            continue
        rect = max(rects, key=lambda r: r.quality)
        result = matcher.match_rectified(rect)
        names = [c.key for c in result.candidates]
        rank = (names.index(spec_key) + 1) if spec_key in names else 99
        metric.add(rank, result.unknown,
                   result.best.confidence if result.best else 0.0,
                   {"detected": True})
    return metric.summary()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_report(report: dict[str, Any]) -> None:
    def row(label: str, s: dict[str, Any], extra: str = "") -> None:
        print(f"  {label:<28} n={s['n']:>4}  top1 {s['top1']:>6.1%}  "
              f"top3 {s['top3']:>6.1%}  top5 {s['top5']:>6.1%}  "
              f"unknown {s['unknownRate']:>5.1%}  wrong {s['wrongAcceptRate']:>5.1%} {extra}")

    print("=" * 96)
    print("FlyCommander vision report")
    print("=" * 96)
    if "recognition" in report:
        r = report["recognition"]
        print("\nRecognition only (rectified captures, no detector):")
        row("all conditions", r)
        for name, s in r.get("byCondition", {}).items():
            row(f"  {name}", s, f"medianRank {s['medianRank']:.0f}")
    if "end_to_end" in report:
        e = report["end_to_end"]
        print("\nEnd-to-end (detector + rectification + matching):")
        row("scenes", e)
        print(f"  {'detection recall':<28} {e['detectionRecall']:.1%}  "
              f"false positives {e['detectionFalsePositives']}  "
              f"corner error {e['cornerErrorPx']}px")
        for name, s in e.get("conditions", {}).items():
            if s.get("top1WhenPresent") is not None:
                print(f"    when {name:<18} top1 {s['top1WhenPresent']:.1%} "
                      f"(n={s['n']})")
    if "captures" in report:
        row("real captures", report["captures"])
    print("\nLatency (ms, per card unless noted):")
    for mode in ("recognition", "end_to_end"):
        if mode in report:
            for k, v in report[mode].get("timingsMs", {}).items():
                print(f"  {mode:<12} {k:<12} {v:.1f}")
    if "calibration" in report and report["calibration"]:
        print(f"\nCalibration: {json.dumps(report['calibration'])}")
    print("=" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all",
                    choices=["recognition", "end-to-end", "all"])
    ap.add_argument("--cards", type=int, default=300, help="library size")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--per-condition", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--captures", default="", help="folder of real photos to score")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit and save the confidence calibration")
    ap.add_argument("--json", default=str(DEFAULT_LOGS / "eval_vision.json"))
    args = ap.parse_args()

    embedder = load_embedder(args.embedder)
    print(f"embedder: {embedder.describe()} "
          f"{getattr(embedder, 'fallback_reason', '')}")
    specs, index, matcher = build_library(args.cards, embedder=embedder, seed=7)
    print(f"library: {index.size} cards, dim {index.dim}")

    report: dict[str, Any] = {"embedder": embedder.describe(),
                              "librarySize": index.size}
    t0 = time.time()
    if args.mode in ("recognition", "all"):
        report["recognition"] = eval_recognition(
            matcher, specs, n_per_condition=args.per_condition, seed=args.seed)
    if args.mode in ("end-to-end", "all"):
        report["end_to_end"] = eval_end_to_end(
            matcher, specs, n_scenes=args.scenes, seed=args.seed)
    if args.captures:
        report["captures"] = eval_captures(matcher, Path(args.captures))
    if args.calibrate:
        calibration = fit_calibration(matcher, specs)
        if calibration:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            (WEIGHTS_DIR / "calibration.json").write_text(
                json.dumps(calibration, indent=1))
            report["calibration"] = calibration
    report["evalSeconds"] = round(time.time() - t0, 1)
    print_report(report)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=1))
    print(f"report written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
