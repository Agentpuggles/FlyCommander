"""FlyCommander physical-table mode — background camera watcher.

The single owner of /dev/video0. Every frame flows through this one capture
loop and is fanned out from it:

    camera_watcher (sole V4L2 owner)
        +--> tracker/state updates (observer.process_frame)
        +--> MJPEG preview stream (snapshot_jpeg / debug_snapshot)

No other component may open the device. OCR/identification never happens
here — only on explicit Scan requests (server.scan_frames), which reuse the
watcher's latest frames via get_burst().

Camera control: the loop does NOT open the device itself; it uses
physical/camera.py's CameraCapture, which enforces V4L2 + MJPG + explicit
resolution/FPS with verification and a fallback chain, and refuses broken
modes (e.g. 2304x1536 @ 2 FPS YUYV) instead of silently accepting them.

Reopen policy (latched): after a successful open, the loop never
spontaneously closes the camera again. Grab failures and watchdog stalls
increment stats["grabFailures"]/["stallEvents"] and keep serving the last
good frame; only an OS-level device loss (open fails) or a stop() triggers a
fresh negotiation and a new "Camera initialized:" log block.

Quality gating: if the scene's burst quality collapses (lens covered, lights
out), the loop reports hints instead of spamming bad events.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from physical.camera import CameraCapture, Watchdog
from physical.card_scan import CV_AVAILABLE, score_frame_quality
from physical.observer import PhysicalObserver
from physical.tracker import Detection

_REOPEN_BACKOFF_S = 5.0
_PREVIEW_INTERVAL_S = 0.10          # MJPEG preview cadence (~10 FPS encode)


class CameraWatcher:
    """Owns the camera; samples frames; feeds the tracker/observer.

    Public API used elsewhere (server, launcher):
        start() / stop() / get_burst(n) / snapshot_jpeg() / stats
        last_analysis()  — latest per-frame detection analysis (debug UI)
        timings()        — per-stage ms (capture/analysis/tracker/preview)
    """

    def __init__(self, observer: PhysicalObserver,
                 settings: Any | None = None,
                 config_path: str | None = None,
                 fps_target: float = 5.0,
                 recognizer: Any | None = None) -> None:
        from physical.camera_config import load_camera_settings  # local: avoids import cycles
        self.observer = observer
        # neural recognition pipeline (physical/vision_pipeline.CardRecognizer).
        # When present and its index is ready, every analysis pass runs
        # detection -> perspective correction -> identification; otherwise the
        # legacy analysis path is used so the table always works.
        self.recognizer = recognizer
        self.settings = settings or load_camera_settings(config_path)
        self.config_path = config_path
        self.device_index = self.settings.device          # back-compat alias
        self.width = self.settings.width
        self.height = self.settings.height
        self.interval = 1.0 / fps_target                  # analysis cadence, not capture rate
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._open_attempts = 0        # consecutive failed opens (backoff)
        self._last_open_error = ""     # de-duplicates the open-failure log
        self._last_open_log = 0.0
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._frame_seq = 0                               # freshness counter for get_burst
        self._cam: CameraCapture | None = None
        self._last_analysis: dict[str, Any] | None = None
        self._last_jpeg: bytes | None = None              # cached MJPEG preview payload
        self._preview_seq = 0                             # increments per NEW preview frame
        self._last_preview_t = 0.0                        # last preview encode time
        self._timings: dict[str, float] = {}
        self.stats: dict[str, Any] = {
            "running": False, "frames": 0, "cardsSeen": 0,
            "lastError": "", "quality": None, "cameraReady": False,
            "diagnostics": None, "fpsTarget": fps_target,
            "grabFailures": 0, "stallEvents": 0,
        }

    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not CV_AVAILABLE:
            self.stats["lastError"] = "OpenCV not available"
            return False
        self._thread = threading.Thread(target=self._loop,
                                        name="camera-watcher", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------
    def get_burst(self, n: int = 6, interval: float = 0.12) -> list:
        """Sample the most recent frame from the live loop, spaced apart.

        The loop keeps only the latest frame (single capture owner); a burst
        therefore waits for n fresh frames instead of replaying one.
        """
        out: list = []
        last_seq = -1
        deadline = time.time() + n * interval + 2.0
        while len(out) < n and time.time() < deadline:
            with self._frame_lock:
                frame = self._last_frame
                seq = self._frame_seq
            if frame is not None and seq != last_seq:
                out.append(frame.copy())
                last_seq = seq
            time.sleep(interval)
        return out

    def preview_frame(self) -> tuple[bytes | None, int]:
        """Latest encoded preview frame + preview sequence number.

        The sequence advances only when a NEW preview frame was encoded, so
        consumers (the MJPEG stream) can deduplicate by identity, not time.
        """
        with self._frame_lock:
            return self._last_jpeg, self._preview_seq

    def snapshot_jpeg(self) -> bytes | None:
        """Latest raw frame as JPEG bytes (for the UI stream)."""
        with self._frame_lock:
            jpeg = self._last_jpeg
            if self._last_frame is None:
                return None
        if jpeg is not None:
            return jpeg
        import cv2
        ok, buf = cv2.imencode(".jpg", self._last_frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def last_analysis(self) -> dict[str, Any] | None:
        """Latest per-frame analysis (candidates, confidence, reasons)."""
        with self._frame_lock:
            return dict(self._last_analysis) if self._last_analysis else None

    def timings(self) -> dict[str, float]:
        """Latest per-stage timings in ms (capture/analysis/tracker/preview)."""
        with self._frame_lock:
            return dict(self._timings)

    def camera_settings(self) -> dict[str, Any]:
        return self.settings.to_dict()

    def debug_snapshot(self) -> bytes | None:
        """Latest frame with detection overlay + camera banner as JPEG."""
        from vision.card_analysis import annotate_frame
        with self._frame_lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
            analysis = dict(self._last_analysis) if self._last_analysis else None
        if frame is None:
            return None
        cam = self._cam
        camera = self._camera_summary(cam) if cam else None
        overlay = annotate_frame(frame, analysis, camera=camera)
        import cv2
        ok, buf = cv2.imencode(".jpg", overlay,
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        stop = self._stop
        while not stop.is_set():
            opened = self._open_camera()
            if not opened:
                # Nothing usable: surface the error, wait, retry. Never exit
                # silently — the UI shows lastError. Back off exponentially
                # (5 s → 60 s) so a machine with no camera logs one line, not
                # a flood, and recovers quickly when the camera appears.
                self._open_attempts += 1
                delay = min(60.0, _REOPEN_BACKOFF_S * (2 ** (self._open_attempts - 1)))
                self.stats["reopenIn"] = round(delay, 1)
                stop.wait(delay)
                continue
            self._open_attempts = 0
            self.stats["reopenIn"] = 0.0
            cam = self._cam
            assert cam is not None
            self.stats["running"] = True
            self.stats["cameraReady"] = True
            self.stats["lastError"] = ""
            self.stats["diagnostics"] = cam.diag.to_dict()
            self.stats["grabFailures"] = 0
            self.stats["stallEvents"] = 0
            print("[camera] ===== single capture owner: camera-watcher =====",
                  flush=True)
            print("[camera] " + cam.diag.describe().replace("\n", "\n[camera] "),
                  flush=True)
            watchdog = Watchdog()

            # CAPTURE at the device's native rate (a read costs ~5 ms); run
            # the (heavier) analysis only every self.interval seconds. This
            # keeps measured_fps a TRUE device rate — the watchdog compares
            # it against MIN_ACCEPTABLE_FPS with real meaning — and keeps
            # the freshest possible frame for bursts and the preview.
            last_analysis_t = 0.0
            try:
                while not stop.is_set():
                    t_loop = time.time()
                    ok, frame = cam.read()
                    t_capture = time.time() - t_loop
                    if not ok:
                        self.stats["grabFailures"] += 1
                        self.stats["lastError"] = (
                            f"frame grab failed ({self.stats['grabFailures']} consecutive)")
                        time.sleep(0.5)
                        # do NOT renegotiate mid-session: keep the single-owner
                        # device open, ride out USB hiccups, serve the last frame
                        continue

                    self.stats["grabFailures"] = 0
                    self.stats["frames"] += 1
                    with self._frame_lock:
                        self._last_frame = frame
                        self._frame_seq += 1

                    # live FPS watchdog: true capture rate; count the event,
                    # never renegotiate mid-session
                    measured = cam.measured_fps
                    if watchdog.update(measured):
                        self.stats["stallEvents"] += 1
                        self.stats["lastError"] = (
                            f"camera FPS collapsed to {measured:.1f} "
                            f"(stall #{self.stats['stallEvents']})")

                    # ---- analysis cadence (fps_target, ~5 Hz): quality,
                    # detection, tracker update ----
                    if t_loop - last_analysis_t >= self.interval:
                        last_analysis_t = t_loop
                        self._analyze_and_track(frame, cam, t_loop, t_capture)

                    # ---- MJPEG preview: throttle heavy JPEG encode into
                    # its own cadence (~4 FPS) ----
                    t2 = time.time()
                    if t2 - self._last_preview_t >= _PREVIEW_INTERVAL_S:
                        try:
                            import cv2
                            ok_j, buf = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ok_j:
                                with self._frame_lock:
                                    self._last_jpeg = buf.tobytes()
                                    self._preview_seq += 1
                        except Exception:
                            pass  # preview encoding must never kill capture
                        self._last_preview_t = t2

                    time.sleep(0.005)   # poll cadence; read() blocks on the camera anyway
            finally:
                cam.close()
                self._cam = None
                self.stats["running"] = False
                self.stats["cameraReady"] = False
            if not stop.is_set() and self.stats.get("lastError"):
                stop.wait(_REOPEN_BACKOFF_S)

    # ------------------------------------------------------------------
    def _open_camera(self) -> bool:
        """Open once with the explicit settings; return success."""
        from physical.camera import CameraOpenError
        cam: CameraCapture | None = None
        try:
            cam = CameraCapture(self.settings)
            cam.open()
        except CameraOpenError as exc:
            summary = str(exc).splitlines()[0]
            self.stats["lastError"] = summary
            if cam is not None:
                self.stats["diagnostics"] = cam.diag.to_dict()
            # Log the headline once, then only when the failure changes (or
            # every tenth attempt as a heartbeat) — a missing device must not
            # fill the log with the same stack of attempts.
            now = time.time()
            changed = summary != self._last_open_error
            if changed or now - self._last_open_log > 300.0:
                attempts = "; ".join(
                    a.get("result", "?") for a in self.stats.get(
                        "diagnostics", {}).get("attempts", []))
                print(f"[camera] open failed: {summary}"
                      + (f" (attempts: {attempts})" if attempts else "")
                      + ("" if changed else
                         f"  [will keep retrying; {self._open_attempts} attempts so far]"),
                      flush=True)
                self._last_open_log = now
            self._last_open_error = summary
            return False
        self._cam = cam
        return True

    def _detections_from_analysis(self, analysis: dict[str, Any]) -> list[Detection]:
        """Confident card analysis -> tracker Detections.

        Every field is optional: a detection without usable geometry is still
        a detection — it contributes no tracking update, bumps cardsSeen, and
        must never raise (never trust dict keys; a watcher thread that dies
        silently kills the whole table).

        The neural pipeline reports *all* cards in view (multi-card scenes are
        normal on a table); the legacy path reports just the best candidate.
        """
        detections: list[Detection] = []
        cards = analysis.get("cards")
        if analysis.get("pipeline") == "neural" and isinstance(cards, list):
            for card in cards:
                try:
                    bbox = card.get("bbox")
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        continue
                    x, y, w, h = (float(v) for v in bbox)
                    detections.append(Detection(
                        bbox=(x, y, w, h),
                        angle_deg=float(card.get("orientationDeg") or 0.0),
                        confidence=float(card.get("match", {}).get("topScore") or 0.5)))
                except (TypeError, ValueError, AttributeError):
                    continue
            return detections
        best = analysis.get("best_candidate") or {}
        if (analysis.get("card_detected")
                and not analysis.get("multiple_cards")
                and float(best.get("confidence") or 0.0) >= 0.55):
            bbox = best.get("bboxFrame")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    x, y, w, h = (float(v) for v in bbox)
                    detections.append(Detection(
                        bbox=(x, y, w, h),
                        angle_deg=float(best.get("orientationDeg") or 0.0),
                        confidence=float(best["confidence"])))
                except (TypeError, ValueError):
                    print("[camera] malformed bboxFrame — skipping tracking "
                          "update this frame")
            else:
                # card_detected without geometry: keep the state, skip the
                # crop/tracking update, log it — and never KeyError.
                print("[camera] detection lacks bboxFrame — skipping "
                      "tracking update this frame", flush=True)
            self.stats["cardsSeen"] += 1
        return detections

    def _analyze_and_track(self, frame, cam: CameraCapture,
                           t_loop: float, t_capture: float) -> None:
        """One analysis pass: quality gate -> detection -> tracker update.

        Runs at the analysis cadence (~5 Hz), never at capture rate. Timings
        land in stats/debug-info per stage (capture | analysis | tracker).
        """
        t0 = time.time()
        q = score_frame_quality(frame)
        self.stats["quality"] = q.to_dict()
        if q.score < 0.18:
            # actionable, not just "unusable": name the dominant problem
            if q.brightness < 45:
                reason = "Image quality low. Improve lighting."
            elif q.brightness > 215:
                reason = "Image quality low. Reduce glare / exposure."
            else:
                reason = "Image quality low. Hold still / improve focus."
            analysis = {
                "state": "bad_quality",
                "card_detected": False,
                "confidence": 0.0,
                "reason": reason,
                "candidates": [],
                "camera": self._camera_summary(cam),
            }
            self._store_analysis(analysis)
            self._timings = {
                "captureMs": round(t_capture * 1000, 1),
                "analysisMs": round((time.time() - t0) * 1000, 1),
                "trackerMs": 0.0,
            }
            return

        if self.recognizer is not None and getattr(self.recognizer, "ready", False):
            try:
                analysis = self._neural_analysis(frame, cam, q)
            except Exception as exc:      # never kill the capture thread
                print(f"[vision] neural analysis failed: {type(exc).__name__}: {exc}",
                      flush=True)
                analysis = None
            if analysis is None:
                analysis = self._legacy_analysis(frame, q)
        else:
            analysis = self._legacy_analysis(frame, q)
        analysis["camera"] = self._camera_summary(cam)
        self._store_analysis(analysis)
        t_analysis = time.time() - t0

        # feed the tracker only confident single-card geometry so tap/presence
        # tracking stays unambiguous; can never crash the thread on
        # unexpected analysis shapes
        t1 = time.time()
        detections = self._detections_from_analysis(analysis)
        self.observer.process_frame(detections)
        t_tracker = time.time() - t1

        self._timings = {
            "captureMs": round(t_capture * 1000, 1),
            "analysisMs": round(t_analysis * 1000, 1),
            "trackerMs": round(t_tracker * 1000, 1),
        }

    def _legacy_analysis(self, frame, quality) -> dict[str, Any]:
        """OCR-era analysis (kept as the fallback when no index exists yet)."""
        from vision.card_analysis import analyze_frame

        return analyze_frame(frame, quality=quality)

    def _neural_analysis(self, frame, cam: CameraCapture, quality) -> dict[str, Any]:
        """Detection + perspective correction + recognition for one frame.

        The result keeps the legacy keys the UI and observer understand
        (state/card_detected/reason/candidates/best_candidate) and adds the
        full neural payload under `cards` / `pipeline` / `timings`.
        """
        scene = self.recognizer.recognize(frame)
        cards = scene.cards
        best = scene.best()
        top_conf = best.confidence if best else 0.0
        if not cards:
            reason = ("No card detected — place a card in view."
                      if quality.score >= 0.18 else
                      "Image quality low. Improve lighting.")
        elif best is not None and best.stable_key:
            reason = (f"{best.stable_name} ({best.stable_confidence:.0%}"
                      f", {len(cards)} card{'s' if len(cards) > 1 else ''} in view)")
        elif best is not None and best.top is not None:
            reason = (f"{len(cards)} card(s) detected · best guess "
                      f"{best.top.name} ({best.top.confidence:.0%})")
        else:
            reason = f"{len(cards)} card(s) detected"
        analysis: dict[str, Any] = {
            "state": "card_detected" if cards else "no_card",
            "card_detected": bool(cards),
            "confidence": top_conf,
            "reason": reason,
            "candidates": [c.to_dict() for c in cards],
            "best_candidate": cards[0].to_dict() if cards else None,
            "cards": [c.to_dict() for c in cards],
            "multiple_cards": len(cards) > 1,
            "pipeline": "neural",
            "indexSize": scene.index_size,
            "notes": list(scene.notes),
            "timings": scene.timings,
            "quality": quality.to_dict(),
        }
        self.stats["cardsSeen"] = self.stats.get("cardsSeen", 0) + len(cards)
        return analysis

    def _camera_summary(self, cam: CameraCapture) -> dict[str, Any]:
        d = cam.diag
        return {
            "device": d.device, "format": d.format,
            "width": d.width, "height": d.height, "fps": d.fps,
            "measuredFps": cam.measured_fps,
        }

    def _store_analysis(self, analysis: dict[str, Any]) -> None:
        with self._frame_lock:
            self._last_analysis = analysis
