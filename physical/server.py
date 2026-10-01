"""FlyCommander physical-table mode — local server + UI.

Zero new dependencies: stdlib http.server + a single-page HTML/JS UI.
The browser owns the camera (mtgscan's getUserMedia approach); the server
owns state, events, registration, and the fly brain.

Endpoints:
  GET  /                     UI page
  GET  /api/state            full UI snapshot (state + brain + events + pending)
  POST /api/event            apply a player event  {"type": "...", ...payload}
  GET  /api/pending          vision events awaiting confirmation
  POST /api/pending/confirm  confirm pending[i]
  POST /api/pending/reject   reject pending[i]
  POST /api/register         manual registration {"set","collectorNumber"}
  POST /api/brain/decide     ask the fly for a decision now
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from brain.connectome import build_synthetic_connectome
from brain.dopamine_plasticity import DopamineSystem
from brain.mushroom_body import MushroomBody
from physical.brain_adapter import FlyBrainAdapter
from physical.camera_watcher import CameraWatcher
from physical.card_scan import (
    CV_AVAILABLE,
    SCAN_AVAILABLE,
    encode_jpeg_b64,
    extract_collector_crop,
    extract_name_crop,
    ocr_card_regions,
    pick_best_frame,
)


def cv2_decode(buf):
    """Decode image bytes; returns None if cv2 missing or undecodable."""
    if not CV_AVAILABLE:
        return None
    import cv2
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
from physical.engine import Engine
from physical.events import Event, EventLog
from physical.identifier import CardIdentifier, Evidence
from physical.registration import CardRegistrar
from physical.scryfall_cache import ScryfallCache
from physical.state import PhysicalGameState

UI_PATH = Path(__file__).resolve().parent / "ui.html"


class PhysicalTableApp:
    """Wires state, engine, registrar, brain adapter, and decision loop."""

    def __init__(self, data_dir: str | Path = "data/physical",
                 log_path: str | Path | None = None,
                 checkpoint: str | None = None,
                 allow_network: bool = True,
                 camera_config: str | Path | None = None) -> None:
        self.state = PhysicalGameState()
        self.log = EventLog(log_path or Path(data_dir) / "events.jsonl")
        self.engine = Engine(self.state, self.log)
        self.cache = ScryfallCache(Path(data_dir) / "scryfall.sqlite3")
        self.cache.allow_network = allow_network
        self.registrar = CardRegistrar(self.cache)
        self.identifier = CardIdentifier(self.cache,
                                         allow_network=allow_network)
        self.last_scan: dict | None = None   # debug payload for the UI
        self.last_scan_timings: dict[str, float] = {}  # ocrMs / identifyMs
        # continuous tap/presence tracking (no OCR in this loop)
        from physical.observer import PhysicalObserver
        from physical.tracker import CardTracker
        self.observer = PhysicalObserver(self.engine,
                                         tracker=CardTracker(
                                             tap_tolerance_deg=30.0,
                                             settle_frames=2))
        self.watcher = CameraWatcher(self.observer,
                                     config_path=camera_config)

        connectome = build_synthetic_connectome(n_pn=64)
        self.mb = MushroomBody(64, connectome=connectome)
        if checkpoint:
            self.mb.load_weights(checkpoint)
        self.dopamine = DopamineSystem(self.mb)
        self.adapter = FlyBrainAdapter(self.mb)
        self.last_decision: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def ui_snapshot(self) -> dict[str, Any]:
        """Everything the UI needs in one payload."""
        return {
            "state": self.state.public_observation(viewer="player"),
            "brain": self.adapter.brain_panel(),
            "decision": self.last_decision.get("decision"),
            "events": self.log.describe_recent(40),
            "pending": [
                {"index": i, "type": e.type, "payload": e.payload,
                 "confidence": round(e.confidence, 2)}
                for i, e in enumerate(self.engine.pending)
            ],
            "camera": self.watcher.stats,
        }

    # ------------------------------------------------------------------
    def apply_player_event(self, etype: str, payload: dict) -> dict:
        return self.engine.apply_player(etype, **payload)

    def register_manual(self, set_code: str, collector_number: str,
                        zone: str = "battlefield") -> dict:
        result = self.registrar.register_manual(set_code, collector_number,
                                                use_network=True)
        if not result.success:
            return {"status": "error", "error": result.error}
        reg = self.engine.apply_player(
            "card_registered", name=result.name, set=result.set_code,
            collectorNumber=result.collector_number,
            cardType=result.card_type, basePower=result.base_power,
            baseToughness=result.base_toughness, zone=zone,
            controller="player", confidence=result.confidence)
        return {"status": "ok", **reg, "name": result.name,
                "set": result.set_code,
                "collectorNumber": result.collector_number}

    # ------------------------------------------------------------------
    # multi-signal registration (success-first)
    # ------------------------------------------------------------------
    def scan_frames(self, images_b64: list[str] | None = None) -> dict:
        """Full pipeline over a camera burst: presence → geometry → quality →
        OCR → identify. Each stage has its OWN failure state; OCR is only ever
        run on frames where a card was actually detected.

        States (result["state"]): error | no_card | multiple_cards |
        too_small | bad_angle | bad_quality | ocr_failed | no_candidates | ok
        Legacy keys (status/nocard/quality/message/candidates) are preserved
        for the existing UI.
        """
        import base64
        if not SCAN_AVAILABLE:
            return {"status": "error", "state": "error",
                    "error": "OpenCV/Tesseract missing on server; use manual entry",
                    "candidates": []}
        frames: list = []
        for b64 in (images_b64 or [])[:12]:
            try:
                payload = base64.b64decode(b64.split(",", 1)[-1])
                arr = np.frombuffer(payload, dtype=np.uint8)
                img = cv2_decode(arr)
                if img is not None:
                    frames.append(img)
            except Exception:
                continue
        if not frames:
            frames = self.watcher.get_burst(6)
        if not frames:
            return {"status": "error", "state": "error",
                    "error": "no readable frames",
                    "candidates": []}

        from vision.card_analysis import analyze_for_scan, rectify as rectify_card
        best_i, frame, quality = pick_best_frame(frames)

        def _ocr(card_img):
            t = time.time()
            out_ocr = ocr_card_regions(card_img)
            self.last_scan_timings["ocrMs"] = round((time.time() - t) * 1000, 1)
            return out_ocr

        analysis = analyze_for_scan(frame, ocr_fn=_ocr)
        state = analysis["state"]

        # --- pipeline failure states (each with its own honest message) ---
        if state == "no_card":
            out = {"status": "nocard", "state": state,
                   "card_detected": False,
                   "confidence": 0.0,
                   "quality": quality.to_dict(),
                   "message": analysis["reason"],
                   "candidates": []}
            self.last_scan = {"analysis": analysis,
                              "frameIndex": best_i,
                              "framesReceived": len(frames),
                              "frameQuality": quality.to_dict()}
            return out
        # bad_angle: the card IS detected — perspective is just too steep.
        # This is actionable coaching (flatten the card), not a failure, and
        # must never be reported like "there is no card".
        if state in ("multiple_cards", "too_small", "bad_quality"):
            out = {"status": "quality" if state == "bad_quality" else "nocard",
                   "state": state,
                   "card_detected": analysis["card_detected"],
                   "confidence": analysis["confidence"],
                   "quality": quality.to_dict(),
                   "message": analysis["reason"],
                   "candidates": [],
                   "analysis": analysis}
            self.last_scan = {"analysis": analysis,
                              "frameIndex": best_i,
                              "framesReceived": len(frames),
                              "frameQuality": quality.to_dict()}
            return out
        if state == "bad_angle":
            # LEGACY: the pipeline no longer rejects angled cards (they get
            # perspective-corrected instead); this branch only defends against
            # older analysis dicts. Never ask the player to flatten anything.
            out = {"status": "retry",
                   "state": state,
                   "card_detected": True,
                   "confidence": analysis["confidence"],
                   "quality": quality.to_dict(),
                   "message": "Card detected at an angle — correcting perspective. Hold still.",
                   "candidates": [],
                   "analysis": analysis}
            self.last_scan = {"analysis": analysis,
                              "frameIndex": best_i,
                              "framesReceived": len(frames),
                              "frameQuality": quality.to_dict()}
            return out

        # --- a single, readable-sized card exists; OCR already ran ---------
        ocr = analysis.get("ocr") or {
            "name": {"raw": "", "normalized": "", "confidence": 0.0},
            "collector": {"raw": "", "set": "", "number": "",
                          "confidence": 0.0}}
        if state == "ocr_failed":
            bc = analysis.get("best_candidate") or {}
            src = bc.get("corners") or bc.get("bboxFrame")
            try:
                card_img = rectify_card(frame, src) if src is not None else None
            except Exception:
                card_img = None
            self.last_scan = {
                "analysis": analysis, "frameIndex": best_i,
                "framesReceived": len(frames),
                "frameQuality": quality.to_dict(),
                "cardOrientation": bc.get("tiltDeg", 0.0),
                "perspective": analysis.get("perspective", "unknown"),
                "rectifiedCard": encode_jpeg_b64(card_img),
                "nameCrop": encode_jpeg_b64(extract_name_crop(card_img)),
                "collectorCrop": encode_jpeg_b64(extract_collector_crop(card_img)),
                "ocr": ocr,
            }
            return {"status": "no_candidates", "state": "ocr_failed",
                    "card_detected": True,
                    "confidence": analysis["confidence"],
                    "quality": quality.to_dict(),
                    "message": analysis["reason"],
                    "candidates": [],
                    "evidence": {"nameRaw": ocr["name"]["raw"],
                                 "nameConfidence": ocr["name"]["confidence"],
                                 "set": ocr["collector"]["set"],
                                 "collectorNumber": ocr["collector"]["number"]},
                    "rectifiedCard": self.last_scan["rectifiedCard"],
                    "nameCrop": self.last_scan["nameCrop"],
                    "collectorCrop": self.last_scan["collectorCrop"]}

        ev = Evidence(
            name_raw=ocr["name"]["normalized"],
            name_conf=ocr["name"]["confidence"],
            set_code=ocr["collector"]["set"],
            collector_number=ocr["collector"]["number"],
            collector_conf=ocr["collector"]["confidence"],
            frame_quality=quality.score)

        # perspective-corrected card image — feeds BOTH artwork matching and
        # the debug panel (corners → homography; bbox resize as fallback)
        bc = analysis.get("best_candidate") or {}
        src = bc.get("corners") or bc.get("bboxFrame")
        t_rect = time.time()
        try:
            card_img = rectify_card(frame, src) if src is not None else None
        except Exception:
            card_img = None
        self.last_scan_timings["rectifyMs"] = round(
            (time.time() - t_rect) * 1000, 1)

        t_ident = time.time()
        candidates = self.identifier.identify(ev)
        # artwork-similarity pass: the text-independent signal — rescues scans
        # whose OCR was weak, penalizes mismatches, no-op when art unavailable
        self.identifier.rescore(candidates, card_img)
        self.last_scan_timings["identifyMs"] = round(
            (time.time() - t_ident) * 1000, 1)

        self.last_scan = {
            "analysis": analysis,
            "frameQuality": quality.to_dict(),
            "frameIndex": best_i,
            "framesReceived": len(frames),
            "cardOrientation": bc.get("tiltDeg", 0.0),
            "perspective": analysis.get("perspective", "unknown"),
            "rectifiedCard": encode_jpeg_b64(card_img),
            "nameCrop": encode_jpeg_b64(extract_name_crop(card_img)),
            "collectorCrop": encode_jpeg_b64(extract_collector_crop(card_img)),
            "ocr": ocr,
            "evidence": ev.to_dict(),
        }
        self._last_candidates = candidates  # for the WHICH CARD? chooser

        result: dict = {
            "status": "ok" if candidates else "no_candidates",
            "state": state,
            "card_detected": True,
            "confidence": analysis["confidence"],
            "candidates": [c.to_dict() for c in candidates[:5]],
            "evidence": ev.to_dict(),
            "message": ("" if candidates else
                        "Card seen but not identified — try Retry Scan or manual entry"),
        }
        if not candidates:
            result.update({k: v for k, v in self.last_scan.items()
                           if k in ("rectifiedCard", "nameCrop", "collectorCrop")})
        # auto-register the confident top candidate — but never duplicate a
        # card already on the battlefield (Commander is singleton)
        if candidates and candidates[0].combined_confidence >= 0.60:
            top = candidates[0]
            existing = next((o for o in self.state.battlefield("player")
                             if o.name.lower() == top.name.lower()), None)
            if existing is not None:
                result["registered"] = {"trackingId": existing.tracking_id,
                                        "name": existing.name,
                                        "duplicate": True}
                state_id = existing.tracking_id
            else:
                reg = self.engine.apply_player(
                    "card_registered", name=top.name, set=top.set_code,
                    collectorNumber=top.collector_number,
                    cardType=top.card_type, basePower=top.base_power,
                    baseToughness=top.base_toughness, zone="battlefield",
                    controller="player", confidence=top.combined_confidence)
                result["registered"] = {**reg, "name": top.name}
                state_id = reg.get("trackingId")
            # bind to the freshest live vision track so tap events flow
            if state_id:
                tracks = self.observer.tracker_tracks_snapshot()
                if tracks:
                    newest = min(tracks, key=lambda t: t.get("ageFrames", 0))
                    self.observer.bind(newest["trackId"], state_id)
        return result

    def register_candidate(self, index: int) -> dict:
        """Player picked one of the offered candidates."""
        cand_list = getattr(self, "_last_candidates", [])
        if not cand_list or index >= len(cand_list):
            return {"status": "error", "error": "no such candidate"}
        c = cand_list[index]
        reg = self.engine.apply_player(
            "card_registered", name=c.name, set=c.set_code,
            collectorNumber=c.collector_number, cardType=c.card_type,
            basePower=c.base_power, baseToughness=c.base_toughness,
            zone="battlefield", controller="player",
            confidence=c.combined_confidence)
        return {"status": "ok", **reg, "name": c.name}

    # ------------------------------------------------------------------
    # debug endpoints
    # ------------------------------------------------------------------
    def debug_info(self) -> dict:
        """Everything the debug panel needs in one payload: camera settings,
        diagnostics, latest per-frame analysis (with confidence/reasons),
        and the last scan's OCR detail."""
        cam = (self.watcher.camera_settings()
               if hasattr(self.watcher, "camera_settings") else {})
        return {   # timings: lastScan = ocrMs/identifyMs, watcherLoop = per-stage ms
            "camera": cam,
            "diagnostics": self.watcher.stats.get("diagnostics"),
            "cameraStats": {k: v for k, v in self.watcher.stats.items()
                            if k != "diagnostics"},
            "timings": {"watcherLoop": self.watcher.timings(),
                        "lastScan": dict(self.last_scan_timings)},
            "analysis": self.watcher.last_analysis(),
            "lastScan": self.last_scan,
            "lastCandidates": [c.to_dict() for c in
                               (getattr(self, "_last_candidates", None) or [])[:5]],
        }

    def debug_frame_jpeg(self) -> bytes | None:
        """Raw frame with detection overlay (outlines + state banner +
        camera settings), for the debug panel."""
        return self.watcher.debug_snapshot()

    def decide(self) -> dict:
        legal = self._legal_actions()
        self.last_decision = self.adapter.decide(self.state, legal)
        return self.last_decision

    def _legal_actions(self) -> tuple[bool, bool, bool, bool]:
        """Physical-mode legality: the fly can only attack if it has healthy
        creatures; play/interact legality is coarse (kept honest)."""
        creatures = [o for o in self.state.battlefield("fly") if o.is_creature]
        ready = [o for o in creatures
                 if not o.tapped and not o.summoning_sick
                 and "cant_attack" not in o.status_flags]
        return (True, bool(ready), True, True)


class PhysicalTableServer:
    def __init__(self, app: PhysicalTableApp, port: int = 8795) -> None:
        self.app = app
        self.port = port
        self._http: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        app = self.app

        def ui_html() -> str:
            # re-read per request so UI edits show up without a server restart
            if UI_PATH.exists():
                return UI_PATH.read_text(encoding="utf-8")
            return "<html><body><h1>ui.html missing</h1></body></html>"

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"   # keep-alive + working MJPEG stream

            def log_message(self, *args):  # silence
                pass

            def _send(self, code: int, body, ctype: str):
                data = body.encode("utf-8") if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_json(self, code: int, obj: dict):
                self._send(code, json.dumps(obj), "application/json")

            def _send_mjpeg(self):
                """Multipart MJPEG stream of the watcher's latest preview.

                Frame-deduplicated: only pushes when a NEW encoded frame is
                available (no stale replays), polling at ~30 Hz so effective
                preview rate = the watcher's encode cadence (~10 FPS).
                Runs until the client disconnects (write error)."""
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_seen = -1
                try:
                    while True:
                        jpeg, seq = app.watcher.preview_frame()
                        if jpeg is not None and seq != last_seen:
                            last_seen = seq
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(jpeg)).encode()
                                + b"\r\n\r\n" + jpeg + b"\r\n")
                        threading.Event().wait(0.033)
                except (ConnectionAbortedError, BrokenPipeError, OSError):
                    pass  # client left — normal end of stream

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if not length:
                    return {}
                try:
                    return json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    return {}

            def do_GET(self):
                # strip query string so cache-busters like ?ts= don't break routing
                path = self.path.split("?", 1)[0]
                if path == "/" or path.startswith("/index"):
                    self._send(200, ui_html(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._send_json(200, app.ui_snapshot())
                elif path == "/api/pending":
                    self._send_json(200, {"pending": app.ui_snapshot()["pending"]})
                elif path == "/api/scan/debug":
                    self._send_json(200, {"scan": app.last_scan})
                elif path == "/api/debug/frame":
                    self._send(200, app.debug_frame_jpeg(), "image/jpeg")
                elif path == "/api/debug/info":
                    self._send_json(200, app.debug_info())
                elif path == "/api/camera/stream":
                    self._send_mjpeg()
                elif path == "/api/camera/snapshot":
                    jpeg = app.watcher.snapshot_jpeg()
                    if jpeg is None:
                        self._send_json(503, {"error": "camera warming up"})
                    else:
                        self._send(200, jpeg, "image/jpeg")
                elif path == "/api/tracks":
                    self._send_json(200, {
                        "tracks": app.observer.tracker_tracks_snapshot(),
                        "bindings": dict(app.observer.track_to_state),
                        "camera": app.watcher.stats})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self):
                body = self._read_json()
                try:
                    if self.path.split("?",1)[0] == "/api/event":
                        out = app.apply_player_event(
                            str(body.get("type", "")),
                            body.get("payload", {}))
                        self._send_json(200, out)
                    elif self.path.split("?",1)[0] == "/api/register":
                        out = app.register_manual(
                            str(body.get("set", "")),
                            str(body.get("collectorNumber", "")),
                            zone=str(body.get("zone", "battlefield")))
                        self._send_json(200, out)
                    elif self.path.split("?",1)[0] == "/api/register/frame":
                        self._send_json(200, app.scan_frames(
                            list(body.get("images", []))))
                    elif self.path.split("?",1)[0] == "/api/register/candidate":
                        self._send_json(200, app.register_candidate(
                            int(body.get("index", 0))))
                    elif self.path.split("?",1)[0] == "/api/scan/debug":
                        self._send_json(200, {"scan": app.last_scan})
                    elif self.path.split("?",1)[0] == "/api/pending/confirm":
                        idx = int(body.get("index", 0))
                        self._send_json(200, app.engine.confirm_pending(idx))
                    elif self.path.split("?",1)[0] == "/api/pending/reject":
                        self._send_json(200, app.engine.reject_pending(
                            int(body.get("index", 0))))
                    elif self.path.split("?",1)[0] == "/api/brain/decide":
                        self._send_json(200, app.decide())
                    else:
                        self._send_json(404, {"error": "not found"})
                except (ValueError, KeyError) as exc:
                    self._send_json(400, {"error": str(exc)})

        self._http = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._http.serve_forever,
                         name="physical-table", daemon=True).start()

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http = None
