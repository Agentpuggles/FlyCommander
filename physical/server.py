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
  GET  /api/vision/status    recognition index / detector / embedder state
  POST /api/vision/identify  identify every card in the current frame
  POST /api/vision/register  confirm a card (optionally add to the index)
  GET  /api/forge/status     physical-table ⇄ Forge bridge state
  POST /api/forge/sync       full idempotent battlefield resync to Forge
"""
from __future__ import annotations

import json
import os
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
from cards.database import CardDatabase
from physical.engine import Engine
from physical.events import Event, EventLog
from physical.identifier import CardIdentifier, Evidence
from physical.registration import CardRegistrar
from physical.scryfall_cache import ScryfallCache
from physical.state import PhysicalGameState
from physical.vision_pipeline import CardRecognizer, RecognitionConfig
from flycommander.forge_table_bridge import ForgeTableBridge

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
                                         on_event=self.mirror_to_forge,
                                         tracker=CardTracker(
                                             tap_tolerance_deg=30.0,
                                             settle_frames=2))
        # ---- neural recognition pipeline (the primary vision path) ---------
        self.vision_config = RecognitionConfig(
            index_dir=str(Path(data_dir) / "cards"))
        self.card_db = CardDatabase(self.vision_config.index_dir,
                                    allow_network=allow_network)
        self.recognizer = CardRecognizer(self.vision_config,
                                         database=self.card_db)
        self.last_recognition: dict | None = None
        # ---- Forge bridge: the physical mirror feeds the real rules engine --
        self.forge = ForgeTableBridge(
            base_url=os.environ.get("FLYCOMMANDER_FORGE_AGENT",
                                    "http://127.0.0.1:8791"),
            table_id=os.environ.get("FLYCOMMANDER_TABLE_ID", "physical"),
            enabled=os.environ.get("FLYCOMMANDER_FORGE_SYNC", "1") != "0")
        self.watcher = CameraWatcher(self.observer,
                                     config_path=camera_config,
                                     recognizer=self.recognizer)

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
            "vision": self.vision_status(),
            "forge": self.forge.status(),
        }

    # ------------------------------------------------------------------
    # neural vision API (used by the UI and by physical-table play)
    # ------------------------------------------------------------------
    def vision_status(self) -> dict:
        """Index/embedder/detector state for the UI vision panel."""
        status = self.recognizer.status()
        status["cards"] = self.card_db.stats()
        status["recognition"] = self.last_recognition
        status["lastScene"] = (self.recognizer.last_scene.to_dict(include_images=False)
                               if self.recognizer.last_scene else None)
        return status

    def vision_identify(self, images_b64: list[str] | None = None,
                        include_images: bool = True) -> dict:
        """Identify every card in the current frame (or supplied frames).

        This is the *fast* path: no OCR, no registration. It returns each
        card with its ranked candidates so the UI can show "recognised as X"
        or offer a chooser when two printings are genuinely ambiguous.
        """
        frame = None
        if images_b64:
            import base64
            for b64 in images_b64[:4]:
                try:
                    payload = base64.b64decode(str(b64).split(",", 1)[-1])
                    img = cv2_decode(np.frombuffer(payload, dtype=np.uint8))
                    if img is not None:
                        frame = img
                        break
                except Exception:
                    continue
        if frame is None:
            burst = self.watcher.get_burst(3)
            frame = burst[-1] if burst else None
        if frame is None:
            return {"status": "error", "error": "no frame available"}
        scene = self.recognizer.recognize(frame)
        out = scene.to_dict(include_images=include_images)
        self.last_recognition = {"t": time.time(),
                                 "cards": [c.to_dict(include_image=False)
                                           for c in scene.cards],
                                 "timings": scene.timings}
        out["status"] = "ok"
        return out

    def vision_build_index(self, synthetic: int = 0, limit: int | None = None,
                           sync_scryfall: bool = False,
                           download_images: int = 0) -> dict:
        """Build/refresh the recognition index from local, synthetic or
        Scryfall sources (network use is explicit and opt-in)."""
        notes: list[str] = []
        if sync_scryfall:
            notes.append(str(self.card_db.sync_scryfall(limit=limit)))
        if download_images:
            notes.append(str(self.card_db.download_images(limit=download_images)))
        if synthetic:
            notes.append(f"synthetic cards added: {self.card_db.import_synthetic(synthetic)}")
        if self.card_db.store.image_count() == 0:
            # a table with no images is useless: seed the demo library so the
            # pipeline is immediately usable and verifiable
            added = self.card_db.import_synthetic(synthetic or 48)
            notes.append(f"seeded {added} synthetic cards (no images found)")
        info = self.recognizer.build_index(limit=limit)
        info["notes"] = notes
        info["cards"] = self.card_db.stats()
        return {"status": "ok", **info}

    def vision_register(self, payload: dict) -> dict:
        """Confirm one recognised card: register it on the table and, when it
        was not in the index, remember it so the same card is instant next
        time (this is the *only* manual step the design allows)."""
        name = str(payload.get("name", "")).strip()
        set_code = str(payload.get("set", "")).strip().lower()
        number = str(payload.get("collectorNumber", "")).strip()
        zone = str(payload.get("zone", "battlefield"))
        if not name:
            return {"status": "error", "error": "name required"}
        if not (set_code and number):
            info = self.card_db.client.named(name)
            if info is not None:
                set_code, number = info.set_code, info.collector_number
        card_image_b64 = payload.get("cardImage")
        added_to_index = None
        if payload.get("addToIndex") and card_image_b64:
            import base64
            try:
                data = base64.b64decode(str(card_image_b64).split(",", 1)[-1])
                img = cv2_decode(np.frombuffer(data, dtype=np.uint8))
                if img is not None:
                    added_to_index = self.recognizer.register_card(
                        img, set_code or "lcl", number or str(
                            self.card_db.store.image_count() + 1), name)
                    self.recognizer.save_index()
            except Exception as exc:      # pragma: no cover - defensive
                added_to_index = {"error": f"{type(exc).__name__}: {exc}"}
        reg = self.engine.apply_player(
            "card_registered", name=name, set=set_code, collectorNumber=number,
            cardType=payload.get("cardType", ""), zone=zone,
            controller="player",
            confidence=float(payload.get("confidence") or 0.9))
        return {"status": "ok", **{k: v for k, v in reg.items() if k != "status"},
                "name": name, "index": added_to_index}

    # ------------------------------------------------------------------
    def apply_player_event(self, etype: str, payload: dict) -> dict:
        result = self.engine.apply_player(etype, **payload)
        self.mirror_to_forge({"type": etype, "origin": "player",
                              "payload": payload})
        return result

    # ------------------------------------------------------------------
    # Forge mirror
    # ------------------------------------------------------------------
    def mirror_to_forge(self, event: Any) -> dict | None:
        """Forward one physical fact to Forge (spooled when Forge is down).

        Vision is the table's eyes, not its judge: every event that survives
        the physical reconciliation above is handed to Forge, which owns the
        rules, the stack and the real state. Identity (set + collector, token
        type, counters) is attached from the physical mirror so Forge resolves
        the *exact* printing the camera saw.
        """
        if not self.forge.enabled:
            return None
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        payload = data.setdefault("payload", {})
        tid = payload.get("trackingId") or payload.get("tracking_id")
        obj = self.state.get(tid) if tid else None
        if obj is not None:
            payload.setdefault("name", obj.name)
            if obj.set_code:
                payload.setdefault("set", obj.set_code)
            if obj.collector_number:
                payload.setdefault("collectorNumber", str(obj.collector_number))
            if obj.oracle_id:
                payload.setdefault("oracleId", obj.oracle_id)
            if obj.is_token:
                payload.setdefault("isToken", True)
                payload.setdefault("tokenType", obj.token_type or obj.name)
            if obj.counters and "counter" not in payload:
                payload.setdefault("counters", dict(obj.counters))
            if data.get("type") == "zone_change":
                payload.setdefault("from", obj.zone)
        try:
            return self.forge.send([data])
        except Exception as exc:                              # noqa: BLE001
            return {"status": "error", "error": str(exc)}

    def forge_status(self, probe: bool = True) -> dict:
        status = self.forge.status(probe=probe)
        if probe and status["connected"]:
            stats = self.forge.queue_stats()
            if stats:
                status["forge"] = stats
        return status

    def forge_sync(self) -> dict:
        """Push the whole physical battlefield to Forge (idempotent)."""
        return self.forge.sync_state(self.state)

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
        if not SCAN_AVAILABLE and not self.recognizer.ready:
            return {"status": "error", "state": "error",
                    "error": "OpenCV/Tesseract missing on server; use manual entry",
                    "candidates": []}
        if self.recognizer.ready:
            neural = self.scan_frames_neural(images_b64)
            if neural is not None:
                return neural
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
                self.forge.remember_track(
                    best.track_id or "",
                    name=top.name, set_code=top.set_code,
                    collector_number=str(top.collector_number),
                    oracle_id=getattr(top, "oracle_id", None))
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

    def scan_frames_neural(self, images_b64: list[str] | None = None) -> dict | None:
        """Neural scan path: identify the best card in the frame and offer it.

        Returns None when the neural path cannot run (no index / no cv2), so
        the caller can fall back to the OCR pipeline. The result keeps the
        legacy scan contract (status/state/candidates/evidence) *and* carries
        the full recognition payload, so the existing UI keeps working while
        the new panel gets richer data.
        """
        if not CV_AVAILABLE or not self.recognizer.ready:
            return None
        import base64
        frames: list = []
        for b64 in (images_b64 or [])[:12]:
            try:
                payload = base64.b64decode(str(b64).split(",", 1)[-1])
                img = cv2_decode(np.frombuffer(payload, dtype=np.uint8))
                if img is not None:
                    frames.append(img)
            except Exception:
                continue
        if not frames:
            frames = self.watcher.get_burst(3)
        if not frames:
            return {"status": "error", "state": "error",
                    "error": "no readable frames", "candidates": [],
                    "pipeline": "neural"}
        frame = frames[-1]
        t0 = time.time()
        scene = self.recognizer.recognize(frame)
        best = scene.best()
        self.last_recognition = {"t": time.time(),
                                 "cards": [c.to_dict(include_image=False)
                                           for c in scene.cards],
                                 "timings": scene.timings}
        if best is None:
            self.last_scan = {"pipeline": "neural", "scene": scene.to_dict()}
            return {"status": "nocard", "state": "no_card", "card_detected": False,
                    "confidence": 0.0, "candidates": [], "pipeline": "neural",
                    "message": "No card detected — place a card in view.",
                    "analysis": scene.to_dict()}
        # the *identity* candidates for the card being scanned (what the UI
        # and the WHICH CARD? chooser need), plus the other cards in view
        candidates = [c.to_dict() for c in best.match.candidates[:5]]
        for cand in candidates:
            cand["trackId"] = best.track_id
            cand["stable"] = bool(best.stable_key == cand.get("set", "").lower()
                                  + ":" + str(cand.get("collectorNumber", "")))
        other_cards = [c.to_dict(include_image=False) for c in scene.cards
                       if c is not best]
        top = best.top
        # a stable, confident identity auto-registers (singleton rule applies)
        registered = None
        # an explicit scan (frames supplied by the UI) is an intentional
        # "identify this" action, so a single confident frame is enough; the
        # live watcher path still requires the identity to be stable
        stable_enough = (best.stable_votes >= self.vision_config.stable_votes
                         or bool(images_b64))
        if (top is not None and not best.match.unknown
                and top.confidence >= self.vision_config.auto_accept_confidence
                and stable_enough):
            existing = next((o for o in self.state.battlefield("player")
                             if o.name.lower() == top.name.lower()), None)
            if existing is not None:
                registered = {"trackingId": existing.tracking_id,
                              "name": existing.name, "duplicate": True}
            else:
                self.forge.remember_track(
                    best.track_id or "",
                    name=top.name, set_code=top.set_code,
                    collector_number=str(top.collector_number),
                    oracle_id=getattr(top, "oracle_id", None))
                reg = self.engine.apply_player(
                    "card_registered", name=top.name, set=top.set_code,
                    collectorNumber=top.collector_number,
                    cardType=getattr(top, "type_line", "") or "",
                    zone="battlefield",
                    controller="player", confidence=top.confidence)
                registered = {**reg, "name": top.name}
            if registered and registered.get("trackingId"):
                self.observer.bind(best.track_id, registered["trackingId"])
        self.last_scan = {
            "pipeline": "neural",
            "scene": scene.to_dict(),
            "timings": scene.timings,
            "recognised": best.to_dict(include_image=True),
            "totalMs": round((time.time() - t0) * 1000, 1),
        }
        status = "ok" if (top is not None and not best.match.unknown) else "no_candidates"
        message = ""
        if top is None:
            message = "Card detected but not identified yet — hold still a moment."
        elif best.match.unknown:
            message = (f"Card not in the library (closest: {top.name} "
                       f"{top.confidence:.0%}). Add it to recognise it instantly.")
        return {
            "status": status,
            "state": "card_detected",
            "card_detected": True,
            "pipeline": "neural",
            "confidence": top.confidence if top else 0.0,
            "candidates": candidates,
            "evidence": {"visualScore": top.scores.get("embed", 0.0) if top else 0.0,
                         "nameRaw": top.name if top else "",
                         "nameConfidence": top.confidence if top else 0.0,
                         "set": top.set_code if top else "",
                         "collectorNumber": top.collector_number if top else ""},
            "message": message,
            "rectifiedCard": (self.last_scan["recognised"] or {}).get("cardImage"),
            "registered": registered,
            "recognition": best.to_dict(include_image=False),
            "otherCards": other_cards,
            "scene": scene.to_dict(include_images=False),
        }

    def register_candidate(self, index: int) -> dict:
        """Player picked one of the offered candidates (OCR or neural path)."""
        cand_list = getattr(self, "_last_candidates", [])
        if not cand_list:
            scan = self.last_scan or {}
            scene_cards = ((scan.get("scene") or {}).get("cards") or [])
            if scan.get("pipeline") == "neural" and scene_cards:
                cand_list = [c["match"]["candidates"] for c in scene_cards
                             if c.get("match", {}).get("candidates")]
                cand_list = [c for group in cand_list for c in group]
            if not cand_list:
                rec = (scan.get("recognised") or {}).get("match", {})
                cand_list = list(rec.get("candidates") or [])
            if not cand_list:
                return {"status": "error", "error": "no scan to register from"}
        if index >= len(cand_list):
            return {"status": "error", "error": "no such candidate"}
        raw = cand_list[index]
        if isinstance(raw, dict) and "name" in raw and "combinedConfidence" not in raw:
            # neural candidate dict
            reg = self.engine.apply_player(
                "card_registered", name=raw.get("name", ""),
                set=raw.get("set", ""),
                collectorNumber=str(raw.get("collectorNumber", "")),
                cardType=raw.get("cardType", ""), zone="battlefield",
                controller="player",
                confidence=float(raw.get("confidence") or 0.7))
            return {"status": "ok", **reg, "name": raw.get("name", "")}
        c = raw
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
    """Local HTTP server for the physical table (UI + state + vision API).

    Binds 0.0.0.0 by default: the table is often viewed from another device
    (tablet on the table, phone as a second screen) or through a proxy when
    running inside a dev sandbox. Pass host="127.0.0.1" to restrict it to the
    local machine.
    """

    def __init__(self, app: PhysicalTableApp, port: int = 8795,
                 host: str = "0.0.0.0") -> None:
        self.app = app
        self.port = port
        self.host = host
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
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError):
                    # the client went away (closed tab, timed-out curl): not an
                    # error worth a traceback in the table's log
                    self.close_connection = True

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
                # Read the request body EXACTLY once (do_POST does it at the
                # top). Reading again on a keep-alive connection blocks until
                # the client sends more bytes — a request that never answers.
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
                elif path == "/api/vision/status":
                    self._send_json(200, app.vision_status())
                elif path == "/api/forge/status":
                    self._send_json(200, app.forge_status(probe=True))
                elif path == "/api/forge/queue":
                    self._send_json(200, app.forge.queue_stats() or {})
                elif path == "/api/vision/identify":
                    self._send_json(200, app.vision_identify())
                elif path == "/api/vision/log":
                    self._send_json(200, {"log": list(app.recognizer.recognition_log)[-20:]})
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
                    elif self.path.split("?",1)[0] == "/api/vision/identify":
                        self._send_json(200, app.vision_identify(
                            list(body.get("images", [])),
                            include_images=bool(body.get("includeImages", True))))
                    elif self.path.split("?",1)[0] == "/api/vision/index":
                        self._send_json(200, app.vision_build_index(
                            synthetic=int(body.get("synthetic", 0)),
                            limit=body.get("limit"),
                            sync_scryfall=bool(body.get("syncScryfall", False)),
                            download_images=int(body.get("downloadImages", 0))))
                    elif self.path.split("?",1)[0] == "/api/vision/register":
                        self._send_json(200, app.vision_register(body))
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
                    elif self.path.split("?",1)[0] == "/api/forge/sync":
                        self._send_json(200, app.forge_sync())
                    elif self.path.split("?",1)[0] == "/api/brain/decide":
                        self._send_json(200, app.decide())
                    else:
                        self._send_json(404, {"error": "not found"})
                except (ValueError, KeyError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError):
                    self.close_connection = True

        self._http = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self._http.serve_forever,
                         name="physical-table", daemon=True).start()

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http = None
