"""FlyCommander physical-table mode — background camera watcher.

The single owner of /dev/video0. Every frame flows through this one capture
loop and is fanned out from it:

    camera_watcher (sole V4L2 owner)
        +--> capture thread  ......... reads at the device rate, nothing else
        +--> analysis thread ......... detection/recognition (analysis_fps)
        +--> preview thread  ......... downscaled MJPEG encode (preview_fps)

Decoupling is deliberate and is the fix for a choppy/laggy preview: the
analysis pass (detection + rectification + embedding) costs far more than one
frame period, and the JPEG encode is not free either. When both ran inline in
the capture loop they throttled it to a few Hz, OpenCV's V4L2 queue filled,
and every frame handed to the browser was the *oldest* queued one — smooth
30 FPS in, laggy 4 FPS out. Now capture never waits for anything, and the
queue depth is 1, so the preview is always the newest frame.

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


class _RateMeter:
    """Rolling events-per-second estimate (windowed on the monotonic clock)."""

    def __init__(self, window_s: float = 4.0) -> None:
        self.window_s = window_s
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._stamps.append(now)
            cutoff = now - self.window_s
            while self._stamps and self._stamps[0] < cutoff:
                self._stamps.pop(0)

    @property
    def rate(self) -> float:
        with self._lock:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            if span <= 0:
                return 0.0
            return (len(self._stamps) - 1) / span


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
                 fps_target: float | None = None,
                 recognizer: Any | None = None,
                 preview_fps: float | None = None,
                 preview_width: int | None = None) -> None:
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
        # cadences: capture runs at the device rate, analysis and preview at
        # their own (configurable) rates — never coupled to each other.
        self.analysis_fps = float(fps_target or self.settings.analysis_fps)
        self.interval = 1.0 / self.analysis_fps
        self.preview_fps = float(preview_fps or self.settings.preview_fps)
        self.preview_interval = 1.0 / max(0.5, self.preview_fps)
        self.preview_width = int(preview_width if preview_width is not None
                                 else self.settings.preview_width)
        self.preview_quality = int(self.settings.preview_quality)
        self.analysis_width = int(self.settings.analysis_width)
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
        self._last_capture_s = 0.0                        # last read() cost (seconds)
        self._timings: dict[str, float] = {}
        self._capture_rate = _RateMeter()
        self._analysis_rate = _RateMeter()
        self._preview_rate = _RateMeter()
        self.stats: dict[str, Any] = {
            "running": False, "frames": 0, "cardsSeen": 0,
            "lastError": "", "quality": None, "cameraReady": False,
            "diagnostics": None, "fpsTarget": self.analysis_fps,
            "grabFailures": 0, "stallEvents": 0,
            # per-thread health (visible in the debug panel)
            "captureFps": 0.0, "analysisHz": 0.0, "previewHz": 0.0,
            "previewMs": 0.0, "analysisMs": 0.0, "captureMs": 0.0,
            "analysisSkips": 0, "previewWidth": self.preview_width,
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

    def snapshot_jpeg(self, quality: int | None = None) -> bytes | None:
        """Latest frame as JPEG bytes, at full camera resolution.

        The *stream* is downscaled on purpose (see `_encode_preview`); a
        one-off snapshot is the one place where the full frame is wanted.
        """
        with self._frame_lock:
            if self._last_frame is None:
                return None
            frame = self._last_frame
        try:
            import cv2
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(quality or 85)])
        except Exception:
            return None
        return buf.tobytes() if ok else None

    def last_analysis(self) -> dict[str, Any] | None:
        """Latest per-frame analysis (candidates, confidence, reasons)."""
        with self._frame_lock:
            return dict(self._last_analysis) if self._last_analysis else None

    def timings(self) -> dict[str, float]:
        """Latest per-stage timings in ms + per-thread rates.

        captureMs/analysisMs/trackerMs come from the analysis pass;
        previewMs and the *Hz numbers are measured per thread, which is what
        makes a "the preview is choppy" report diagnosable at a glance.
        """
        with self._frame_lock:
            out = dict(self._timings)
        live = {
            "previewMs": self.stats.get("previewMs", 0.0),
            "captureFps": self.stats.get("captureFps", 0.0),
            "analysisHz": self.stats.get("analysisHz", 0.0),
            "previewHz": self.stats.get("previewHz", 0.0),
            "analysisSkips": self.stats.get("analysisSkips", 0),
        }
        if not out and not any(live.values()):
            return {}          # nothing has run yet — "no analysis" stays empty
        out.update(live)
        return out

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
            print(f"[camera] threads: capture @ device rate · analysis "
                  f"@ {self.analysis_fps:g} Hz · preview @ {self.preview_fps:g} Hz"
                  + (f" ({self.preview_width}px)" if self.preview_width else ""),
                  flush=True)

            # Three threads, one camera: capture never blocks on analysis or
            # encoding, so the preview stays live even when recognition is
            # expensive (see the module docstring).
            session = threading.Event()
            workers = [
                threading.Thread(target=self._analysis_loop, args=(session,),
                                 name="camera-analysis", daemon=True),
                threading.Thread(target=self._preview_loop, args=(session,),
                                 name="camera-preview", daemon=True),
            ]
            for t in workers:
                t.start()
            try:
                self._capture_loop(cam, session)
            finally:
                session.set()
                for t in workers:
                    t.join(timeout=5.0)
                cam.close()
                self._cam = None
                self.stats["running"] = False
                self.stats["cameraReady"] = False
                self.stats["captureFps"] = 0.0
                self.stats["analysisHz"] = 0.0
                self.stats["previewHz"] = 0.0
            if not stop.is_set() and self.stats.get("lastError"):
                stop.wait(_REOPEN_BACKOFF_S)

    # ------------------------------------------------------------------
    # the three loops
    # ------------------------------------------------------------------
    def _capture_loop(self, cam: CameraCapture, session: threading.Event) -> None:
        """Sole reader of the device: grab, publish, repeat. Nothing else.

        Deliberately dumb and fast — no analysis, no encoding — so the device
        is drained at its native rate and the published frame is always the
        newest one (queue depth is 1, see camera.CameraCapture.open).
        """
        stop = self._stop
        watchdog = Watchdog()
        while not (stop.is_set() or session.is_set()):
            t_loop = time.time()
            ok, frame = cam.read()
            t_capture = time.time() - t_loop
            if not ok:
                self.stats["grabFailures"] += 1
                self.stats["lastError"] = (
                    f"frame grab failed ({self.stats['grabFailures']} consecutive)")
                # do NOT renegotiate mid-session: keep the single-owner
                # device open, ride out USB hiccups, serve the last frame
                stop.wait(0.5)
                continue

            self.stats["grabFailures"] = 0
            self.stats["frames"] += 1
            with self._frame_lock:
                self._last_frame = frame
                self._frame_seq += 1
            self._last_capture_s = t_capture
            self._capture_rate.tick()
            self.stats["captureFps"] = round(self._capture_rate.rate, 1)
            self.stats["captureMs"] = round(t_capture * 1000, 1)

            # live FPS watchdog: true capture rate; count the event,
            # never renegotiate mid-session
            measured = cam.measured_fps
            if watchdog.update(measured):
                self.stats["stallEvents"] += 1
                self.stats["lastError"] = (
                    f"camera FPS collapsed to {measured:.1f} "
                    f"(stall #{self.stats['stallEvents']})")

            stop.wait(0.001)   # yield; read() blocks on the camera anyway

    def _analysis_loop(self, session: threading.Event) -> None:
        """Detection/recognition at analysis_fps (never at capture rate).

        A slow pass simply means the next one starts late — the capture loop
        keeps running at full speed either way.
        """
        stop = self._stop
        next_t = 0.0
        last_seq = -1
        while not (stop.is_set() or session.is_set()):
            now = time.monotonic()
            if now < next_t:
                stop.wait(min(0.05, next_t - now))
                continue
            frame, seq = self._latest_frame()
            if frame is None or seq == last_seq:
                # no fresh frame yet (camera slower than the cadence)
                if frame is not None:
                    self.stats["analysisSkips"] += 1
                stop.wait(0.02)
                continue
            last_seq = seq
            t0 = time.time()
            try:
                self._analyze_and_track(frame, self._cam, t_loop=t0,
                                        t_capture=self._last_capture_s)
            except Exception as exc:      # never kill the analysis thread
                print(f"[camera] analysis failed: {type(exc).__name__}: {exc}",
                      flush=True)
            self._analysis_rate.tick()
            self.stats["analysisHz"] = round(self._analysis_rate.rate, 1)
            self.stats["analysisMs"] = round((time.time() - t0) * 1000, 1)
            next_t = max(time.monotonic(), next_t + self.interval)

    def _preview_loop(self, session: threading.Event) -> None:
        """MJPEG preview at preview_fps, encoded from a downscaled copy."""
        stop = self._stop
        next_t = 0.0
        last_seq = -1
        while not (stop.is_set() or session.is_set()):
            now = time.monotonic()
            if now < next_t:
                stop.wait(min(0.02, next_t - now))
                continue
            frame, seq = self._latest_frame()
            if frame is None or seq == last_seq:
                stop.wait(0.01)
                continue
            last_seq = seq
            t0 = time.time()
            try:
                jpeg = self._encode_preview(frame)
                if jpeg is not None:
                    with self._frame_lock:
                        self._last_jpeg = jpeg
                        self._preview_seq += 1
                    self._preview_rate.tick()
            except Exception:
                pass      # preview encoding must never kill the stream
            self._last_preview_t = time.time()
            self.stats["previewMs"] = round((time.time() - t0) * 1000, 1)
            self.stats["previewHz"] = round(self._preview_rate.rate, 1)
            next_t = max(time.monotonic(), now + self.preview_interval)

    def _latest_frame(self) -> tuple[Any, int]:
        with self._frame_lock:
            return self._last_frame, self._frame_seq

    def _encode_preview(self, frame) -> bytes | None:
        """Downscale (INTER_AREA) then encode — cheaper and just as legible."""
        import cv2

        img = frame
        width = self.preview_width
        if width and frame is not None:
            h, w = frame.shape[:2]
            if w > width:
                h2 = max(1, int(round(h * (width / float(w)))))
                img = cv2.resize(frame, (width, h2),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, self.preview_quality])
        return buf.tobytes() if ok else None

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

    def _analyze_and_track(self, frame, cam: CameraCapture | None = None,
                           t_loop: float = 0.0, t_capture: float = 0.0) -> None:
        """One analysis pass: quality gate -> detection -> tracker update.

        Runs on the analysis thread at the analysis cadence (~5 Hz by
        default), never at capture rate. Timings land in stats/debug-info per
        stage (capture | analysis | tracker).
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

    def _neural_analysis(self, frame, cam: CameraCapture | None, quality) -> dict[str, Any]:
        """Detection + perspective correction + recognition for one frame.

        The result keeps the legacy keys the UI and observer understand
        (state/card_detected/reason/candidates/best_candidate) and adds the
        full neural payload under `cards` / `pipeline` / `timings`.

        When ``analysis_width`` is configured (> 0) the frame is downscaled
        before detection — the single biggest lever on a slow machine — and
        the returned geometry (bbox/quad) is scaled back to camera pixels so
        tracking and the debug overlay stay in the same coordinate space.
        """
        scale = 1.0
        vision_frame = frame
        if self.analysis_width and frame is not None and frame.shape[1] > self.analysis_width:
            import cv2
            scale = self.analysis_width / float(frame.shape[1])
            vision_frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_AREA)
        scene = self.recognizer.recognize(vision_frame)
        cards = scene.cards
        # NOTE: the recognizer's tracks own their quads (they are matched
        # across frames), so a downscaled analysis frame is corrected on the
        # way OUT (in the dicts below) and never by mutating the tracks.
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
        card_dicts = [c.to_dict() for c in cards]
        if scale != 1.0:
            inv = 1.0 / scale
            for d in card_dicts:
                bbox = d.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    d["bbox"] = [round(float(v) * inv, 1) for v in bbox]
                quad = d.get("quad")
                if isinstance(quad, list):
                    d["quad"] = [[round(float(x) * inv, 1),
                                  round(float(y) * inv, 1)] for x, y in quad]
        analysis: dict[str, Any] = {
            "state": "card_detected" if cards else "no_card",
            "card_detected": bool(cards),
            "confidence": top_conf,
            "reason": reason,
            "candidates": card_dicts,
            "best_candidate": card_dicts[0] if card_dicts else None,
            "cards": card_dicts,
            "multiple_cards": len(cards) > 1,
            "pipeline": "neural",
            "indexSize": scene.index_size,
            "notes": list(scene.notes),
            "timings": scene.timings,
            "quality": quality.to_dict(),
        }
        if scale != 1.0:
            analysis["analysisScale"] = round(scale, 3)
        self.stats["cardsSeen"] = self.stats.get("cardsSeen", 0) + len(cards)
        return analysis

    def _camera_summary(self, cam: CameraCapture | None) -> dict[str, Any]:
        if cam is None:
            return {}
        d = cam.diag
        return {
            "device": d.device, "format": d.format,
            "width": d.width, "height": d.height, "fps": d.fps,
            "measuredFps": cam.measured_fps,
        }

    def _store_analysis(self, analysis: dict[str, Any]) -> None:
        with self._frame_lock:
            self._last_analysis = analysis
