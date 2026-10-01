#!/usr/bin/env python3
"""Four-seat diagnostic; --controller-test permits DIGITAL hands, never paper play."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
import shlex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flycommander.fly_pod import FlyPod
from flycommander.deck_specs import classify_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--controller-test', action='store_true', help='Allow Forge-generated DIGITAL test hands, NOT paper gameplay')
    ap.add_argument('--runtime-dir', type=Path, default=ROOT / 'data/forge-built-runtime')
    ap.add_argument('--human-deck', required=True)
    ap.add_argument('--fly-deck', action='append', required=True)
    args = ap.parse_args()
    if len(args.fly_deck) != 3:
        ap.error('Exactly three --fly-deck arguments required')
    try:
        specs = [classify_spec(s) for s in [args.human_deck, *args.fly_deck]]
    except ValueError as exc:
        ap.error(str(exc))
    if specs[0].kind != 'file':
        ap.error('Human must provide a matching physical deck .dck file')
    decks = [str(s.path.resolve()) if s.path else s.java_arg for s in specs]
    runtime = args.runtime_dir.expanduser().resolve()
    jar = runtime / 'forge-gui-desktop-2.0.15-jar-with-dependencies.jar'
    classes = runtime / 'forge-agent-patch/classes'
    if not jar.is_file() or not (classes/'fly/agent/WebHumanLobbyPlayer.class').is_file():
        ap.error('Build first: python scripts/build_forge_source.py')
    if not (runtime / 'res').is_dir():
        ap.error('Runtime res/ directory is missing')
    pod = FlyPod(checkpoint_dir=None if args.controller_test else ROOT/'data/fly-pod-checkpoints')
    try:
        pod.start()
        cmd = ['java', '-Xmx3g', '-Dfly.agent.physical=true',
            *(['-Dfly.agent.controllerTest=true'] if args.controller_test else []),
            '-cp', os.pathsep.join([str(classes), str(jar)]),
            'fly.agent.AgentMain', '1', *decks]
        print('[launcher] ' + shlex.join(cmd), flush=True)
        print('[launcher] cwd=' + str(runtime), flush=True)
        return subprocess.call(cmd, cwd=runtime)
    finally:
        pod.stop()


if __name__ == '__main__':
    raise SystemExit(main())
