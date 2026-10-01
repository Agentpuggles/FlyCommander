"""Deck-sized, training-free visual recognition using real reference artwork.

Unlike the experimental dense index, this path never names a nearest neighbour
without geometric evidence. Only illustration features vote (not the shared MTG
frame/text). SIFT + RANSAC locates cards directly in the camera frame, so a missed
contour or unreadable title does not prevent recognition. Scores are heuristic
match strengths, NOT calibrated probabilities or a claim of printing identity.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # optional outside physical mode
    cv2 = None

from cards.database import IndexEntry
from vision.matcher import MatchCandidate, MatchResult
from vision import rectify as R


@dataclass
class ReferenceHit:
    quad: np.ndarray
    match: MatchResult


class ReferenceScanner:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.refs: list[dict] = []
        self._flann = None
        self._owners = np.empty(0, np.int32)
        self.last_error = ""
        self.reload()

    @property
    def ready(self):
        return self._flann is not None

    def status(self):
        with self._lock:
            return {"ready": self.ready, "references": len(self.refs),
                    "backend": "SIFT artwork + RANSAC", "error": self.last_error}

    def card_info(self, set_code, number, name):
        """Face-specific metadata; a double-faced printing has one set/number."""
        with self._lock:
            return next((dict(r.get("cardInfo", {})) for r in self.refs
                         if r["entry"]["set"].casefold() == set_code.casefold()
                         and r["entry"]["collectorNumber"] == number
                         and r["entry"]["name"] == name), {})

    @staticmethod
    def _features(image, reference=False, art_bottom=.51):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = None
        if reference:
            # Conservative shared illustration interior: old, modern and full-art.
            # Exclude title, mana symbols, frame and rules even if that costs recall.
            h, w = gray.shape
            mask = np.zeros_like(gray)
            mask[int(h * .17):int(h * art_bottom), int(w * .12):int(w * .88)] = 255
        return cv2.SIFT_create(nfeatures=1400 if reference else 5000,
                              contrastThreshold=.025).detectAndCompute(gray, mask)

    def reload(self):
        """Build a new in-memory snapshot, then atomically swap it for readers."""
        if cv2 is None:
            self.last_error = "Install requirements-physical.txt for visual recognition"
            return
        manifest = self.root / "references.json"
        if not manifest.exists():
            return
        refs = []
        unusable = []
        try:
            for row in json.loads(manifest.read_text()):
                # Manifest contains only locally generated basenames.
                path = self.root / Path(row["file"]).name
                image = cv2.imread(str(path))
                if image is None:
                    unusable.append(row["entry"].get("name", path.name))
                    continue
                image = cv2.resize(image, (400, 560))
                bottom = .40 if "Planeswalker" in row["entry"].get("typeLine", "") else .51
                kp, desc = self._features(image, reference=True, art_bottom=bottom)
                if desc is None or len(kp) < 12:
                    unusable.append(row["entry"].get("name", path.name))
                    continue
                refs.append({**row, "points": np.float32([k.pt for k in kp]),
                             "desc": desc, "image": image, "artBottom": bottom})
            flann = None
            owners = np.empty(0, np.int32)
            if refs:
                descriptors = np.concatenate([r["desc"] for r in refs])
                owners = np.concatenate([np.full(len(r["desc"]), i, np.int32)
                                         for i, r in enumerate(refs)])
                flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
                flann.add([descriptors])
                flann.train()
            with self._lock:
                self.refs, self._flann, self._owners = refs, flann, owners
                self.last_error = ("Unusable artwork (missing image or too few features): " + ", ".join(unusable)
                                   if unusable else "")
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            self.last_error = str(exc)

    def recognize(self, frame, max_cards=12):
        if cv2 is None or frame is None or not frame.size:
            return []
        with self._lock:
            if not self.ready:
                return []
            scale = min(1.0, 1600 / max(frame.shape[:2]))
            image = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            kp, desc = self._features(image)
            if desc is None or len(kp) < 12:
                return []
            points = np.float32([k.pt for k in kp])
            # Multiple neighbours keep duplicate-art printings from starving votes.
            neighbours = self._flann.knnMatch(desc, k=min(8, len(self._owners)))
            votes = np.zeros(len(self.refs), np.float32)
            for group in neighbours:
                if not group:
                    continue
                for owner in {int(self._owners[m.trainIdx]) for m in group
                              if m.distance <= min(240, group[0].distance * 1.15)}:
                    votes[owner] += 1
            votes /= np.sqrt([len(r["desc"]) for r in self.refs])
            shortlist = np.argsort(-votes)[:min(len(self.refs), max(16, max_cards * 4))]
            hits = []
            for i in shortlist:
                if votes[i] <= 0:
                    continue
                ref = self.refs[i]
                # Scene -> reference: two copies on the table must not become
                # each other's second neighbour and fail the Lowe ratio test.
                pairs = cv2.BFMatcher().knnMatch(desc, ref["desc"], k=2)
                good = [a for a, b in pairs if a.distance < .72 * b.distance]
                for _ in range(max_cards):
                    if len(good) < 12:
                        break
                    src = np.float32([ref["points"][m.trainIdx] for m in good])
                    dst = np.float32([points[m.queryIdx] for m in good])
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=4000)
                    if H is None or mask is None or not np.isfinite(H).all():
                        break
                    inliers = mask.ravel().astype(bool)
                    # Several scene features mapping to one artwork point are
                    # not independent evidence (e.g. repeated symbols/texture).
                    count = len({m.trainIdx for m, keep in zip(good, inliers) if keep})
                    if count < 12:
                        break
                    support = cv2.contourArea(cv2.convexHull(src[inliers])) / (
                        400 * 560 * .76 * (ref["artBottom"] - .17))
                    if support < .16:
                        break
                    quad = cv2.perspectiveTransform(np.float32(
                        [[[0, 0], [399, 0], [399, 559], [0, 559]]]), H)[0]
                    if not self._sane_quad(quad, image.shape):
                        break
                    inside = np.array([cv2.pointPolygonTest(quad, tuple(map(float, p)), False) >= 0
                                       for p in dst])
                    # Outliers on another copy aren't evidence against this one.
                    ratio = float(inliers.sum()) / max(1, int(inside.sum()))
                    if ratio < .45:
                        break
                    # Keypoints alone can agree on repeated template fragments.
                    # Verify that the aligned illustration pixels agree too.
                    art_correlation = self._art_correlation(ref, image, H)
                    if art_correlation < .20:
                        break
                    entry = IndexEntry.from_dict(ref["entry"])
                    strength = min(.98, .55 + min(count, 70) * .004 + min(support, .8) * .15)
                    candidate = MatchCandidate(
                        entry.set_code, entry.collector_number, entry.name, strength,
                        scores={"artInliers": count, "inlierRatio": min(1., ratio), "artCoverage": support,
                                "artCorrelation": art_correlation},
                        oracle_id=entry.oracle_id, type_line=entry.type_line,
                        image_path=str(self.root / ref["file"]), source="reference-art")
                    hits.append((quad / scale, candidate, count))
                    # Remove this instance, then fit another copy of the card.
                    good = [m for m, used in zip(good, inliers) if not used]
            # Group competing identities at the same location; never hide a tie.
            groups = []
            for quad, candidate, count in sorted(hits, key=lambda h: -h[2]):
                group = next((g for g in groups if self._same_location(g[0], quad)), None)
                if group is None:
                    groups.append([quad, [(candidate, count)]])
                else:
                    group[1].append((candidate, count))
            results = []
            for quad, candidates in groups[:max_cards]:
                best, n = candidates[0]
                rival = next((c for c in candidates[1:] if c[0].name != best.name), None)
                ambiguous = rival is not None and (n < rival[1] * 1.5 or n < rival[1] + 6)
                notes = ["Artwork verified; confirm card name. Reference printing may differ."]
                if ambiguous:
                    notes.append("Ambiguous artwork — choose the card; no automatic registration.")
                results.append(ReferenceHit(quad, MatchResult(
                    candidates=[c for c, _ in candidates[:5]], unknown=ambiguous,
                    top_score=best.confidence, notes=notes)))
            return results

    @staticmethod
    def _same_location(a, b):
        # IoU alone merges genuinely distinct copies in a fan. Equivalent
        # homographies should project nearly the same four card corners.
        a, b = R.order_corners(a), R.order_corners(b)
        diagonal = max(1., float(np.linalg.norm(a[0] - a[2])))
        return float(np.linalg.norm(a - b, axis=1).mean()) < .06 * diagonal

    @staticmethod
    def _art_correlation(ref, scene, H):
        try:
            aligned = cv2.warpPerspective(scene, np.linalg.inv(H), (400, 560))
        except (np.linalg.LinAlgError, cv2.error):
            return 0.
        def detail(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.GaussianBlur(gray, (0, 0), 1.)
            gray -= cv2.GaussianBlur(gray, (0, 0), 4.)
            return gray[95:int(560 * ref["artBottom"]), 48:352].reshape(-1)
        a, b = detail(ref["image"]), detail(aligned)
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _sane_quad(q, shape):
        h, w = shape[:2]
        if not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.float32)):
            return False
        if (q[:, 0].min() < -.15 * w or q[:, 0].max() > 1.15 * w
                or q[:, 1].min() < -.15 * h or q[:, 1].max() > 1.15 * h):
            return False
        # Reference corners are clockwise in image coordinates. A reflected
        # mapping is not a physical face-up card and must not be accepted.
        area = cv2.contourArea(q, oriented=True)
        edges = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        # Perspective-tolerant, but no collapsed or wildly extrapolated card.
        return (area > 2000 and area < h * w * 1.15 and edges.min() > 35
                and edges.max() / edges.min() < 5
                and .35 < (edges[0] + edges[2]) / (edges[1] + edges[3]) < 1.2)
