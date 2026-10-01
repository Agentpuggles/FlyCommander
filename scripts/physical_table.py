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

    server = PhysicalTableServer(app, port=args.port)
    server.start()
    if app.watcher.start():
        print("[physical] camera watcher running (tap/presence tracking, no OCR)")
    else:
        print("[physical] camera watcher unavailable:", app.watcher.stats["lastError"])
    print(f"[physical] Magic Fly table running:  http://127.0.0.1:{args.port}")
    print("[physical] allow the browser camera when prompted (mtgscan-style")
    print("[physical] getUserMedia). Register cards via OCR or set+number;")
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
