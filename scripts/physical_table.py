#!/usr/bin/env python3
"""FlyCommander — physical-table mode launcher.

Starts the local server (state + events + registration + fly brain) and
serves the spectator UI. The server owns the camera and streams MJPEG to the
browser; the browser never opens /dev/video itself.

    python scripts/physical_table.py [--port 8795] [--checkpoint ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical.card_scan import SCAN_AVAILABLE, tesseract_ready     # noqa: E402
from physical.server import PhysicalTableApp, PhysicalTableServer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Magic Fly physical-table mode")
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--data-dir", default="data/physical")
    ap.add_argument("--checkpoint", default=None,
                    help="fly brain checkpoint from Forge training")
    ap.add_argument("--offline", action="store_true",
                    help="never contact Scryfall (cache/manual only)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default 0.0.0.0 so a tablet/phone on "
                         "the network can open the table)")
    ap.add_argument("--build-index", type=int, default=0, metavar="N",
                    help="build the recognition index from N synthetic cards "
                         "(use with --sync-scryfall for real cards)")
    ap.add_argument("--sync-scryfall", action="store_true",
                    help="download Scryfall bulk card data and build the "
                         "recognition index from real card images")
    ap.add_argument("--images", type=int, default=0, metavar="N",
                    help="download at most N card images while building")
    ap.add_argument("--no-camera", action="store_true",
                    help="browser-camera/photo mode: do not open a server camera")
    ap.add_argument("--camera-config", default=None,
                    help="path to camera_config.json "
                         "(default: physical/camera_config.json or repo root)")
    ap.add_argument("--show-camera-config", action="store_true",
                    help="print the effective camera settings and exit")
    args = ap.parse_args()

    if args.show_camera_config:
        from physical.camera_config import load_camera_settings
        print(load_camera_settings(args.camera_config).describe())
        return 0

    app = PhysicalTableApp(data_dir=args.data_dir,
                           checkpoint=args.checkpoint,
                           camera_config=args.camera_config,
                           allow_network=not args.offline)

    if args.build_index or args.sync_scryfall:
        info = app.vision_build_index(synthetic=args.build_index,
                                      sync_scryfall=args.sync_scryfall,
                                      download_images=args.images)
        if info.get("status") == "error":
            print(f"[vision] {info['error']}")
            return 1
        print(f"[vision] index ready: {info['cards']} entries, embedder "
              f"{info['embedder']} (dim {info['dim']})")
        for note in info.get("notes", []):
            print(f"[vision] {note}")

    server = PhysicalTableServer(app, port=args.port, host=args.host)
    server.start()
    if args.no_camera:
        print("[physical] browser-camera/photo mode (server camera disabled)")
    elif app.watcher.start():
        print("[physical] camera watcher running")
    else:
        print("[physical] camera watcher unavailable:", app.watcher.stats["lastError"])
    status = app.recognizer.status()
    if status.get("referenceScanner", {}).get("ready"):
        print(f"[vision] artwork scanner active — {status['referenceScanner']['references']} reference faces; confirm to add")
    elif status["ready"]:
        print(f"[vision] neural recognition active — {status['indexSize']} cards "
              f"indexed, detector={status['detector']}, "
              f"embedder={status['embedder']['name']}"
              f"{' (trained)' if status['embedder'].get('trained') else ' (UNTRAINED)'}")
    else:
        print("[vision] import your deck in the table UI to enable real artwork matching. "
              "No model training required. OCR/manual entry remain available.")

    # Which of the three identification paths can actually answer right now?
    # Say it out loud at startup: "it didn't recognise my card" is otherwise
    # undebuggable from the UI alone.
    library = app.card_db.library_summary()
    ocr_ok, ocr_why = tesseract_ready()
    ocr_state = f"available (tesseract {ocr_why})" if ocr_ok else \
        f"UNAVAILABLE — {ocr_why}"
    print(f"[vision] library: {library['cards']} cards "
          f"({library['real']} real / {library['synthetic']} synthetic demo), "
          f"{library['images']} images")
    print(f"[vision] OCR + Scryfall rescue: {ocr_state}")
    if library["real"] == 0 and library["synthetic"] > 0:
        print("[vision]   → the demo library cannot name real cards from pixels; "
              "Scan reads the card's text instead")
    if not status["embedder"].get("trained") and not status.get("referenceScanner", {}).get("ready"):
        print("[vision]   → experimental neural embedder is untrained. "
              "Use Import real card images for the training-free artwork scanner.")
    if not (SCAN_AVAILABLE and ocr_ok):
        print("[vision]   → OCR rescue is unavailable; imported artwork or Add by name still work")
    print(f"[physical] Magic Fly table running:  http://127.0.0.1:{args.port}")
    print("[physical] choose server camera, this device’s browser camera, or upload a photo")
    print("[physical] press 📷 Scan card to identify a card, or type its name "
          "under the video")
    print("[physical] the fly brain panel updates with every decision.")
    try:
        while True:
            import time
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[physical] shutting down")
        app.watcher.stop()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
