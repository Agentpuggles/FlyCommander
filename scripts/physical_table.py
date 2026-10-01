#!/usr/bin/env python3
"""FlyCommander — physical-table mode launcher.

Starts the local server (state + events + registration + fly brain) and
serves the spectator UI with browser camera capture.

    python scripts/physical_table.py [--port 8795] [--checkpoint ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
                           camera_config=args.camera_config)
    if args.offline:
        app.cache.allow_network = False

    if args.build_index or args.sync_scryfall:
        info = app.vision_build_index(synthetic=args.build_index,
                                      sync_scryfall=args.sync_scryfall,
                                      download_images=args.images)
        print(f"[vision] index ready: {info['cards']} entries, embedder "
              f"{info['embedder']} (dim {info['dim']})")
        for note in info.get("notes", []):
            print(f"[vision] {note}")

    server = PhysicalTableServer(app, port=args.port, host=args.host)
    server.start()
    if app.watcher.start():
        print("[physical] camera watcher running")
    else:
        print("[physical] camera watcher unavailable:", app.watcher.stats["lastError"])
    status = app.recognizer.status()
    if status["ready"]:
        print(f"[vision] neural recognition active — {status['indexSize']} cards "
              f"indexed, detector={status['detector']}, "
              f"embedder={status['embedder']['name']}")
    else:
        print("[vision] no recognition index yet — run with --build-index 64 "
              "(demo library) or --sync-scryfall (real cards); the OCR path "
              "stays available in the meantime")
    print(f"[physical] Magic Fly table running:  http://127.0.0.1:{args.port}")
    print("[physical] allow the browser camera when prompted (mtgscan-style")
    print("[physical] getUserMedia). Cards are recognised automatically;")
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
