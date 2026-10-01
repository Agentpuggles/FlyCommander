#!/usr/bin/env python3
"""Launch a four-seat Forge Commander pod with a browser-controlled human.

Forge AI opponents are the default and need no Fly services. Use
--opponents fly only to opt into three external Fly brains. --assisted-human
selects the older AI-assisted table mode; it is not physical-library sync.
--controller-test remains a separate digital-hand diagnostic mode.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import os
import shlex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flycommander.deck_specs import classify_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assisted-human', action='store_true',
                    help='Start the browser-driven human seat with Forge AI handling detailed card choices')
    ap.add_argument('--controller-test', action='store_true', help='Allow Forge-generated DIGITAL test hands, NOT paper gameplay')
    ap.add_argument('--runtime-dir', type=Path, default=ROOT / 'data/forge-built-runtime')
    ap.add_argument('--human-deck', required=True)
    ap.add_argument('--opponents', choices=('forge', 'fly'), default='forge',
                    help='Opponent controller for seats 1–3 (default: Forge AI)')
    ap.add_argument('--opponent-deck', '--fly-deck', dest='opponent_deck', action='append', required=True,
                    help='Commander deck for one of the three opponents (repeat exactly three times)')
    args = ap.parse_args()
    if len(args.opponent_deck) != 3:
        ap.error('Exactly three --opponent-deck arguments required')
    if args.assisted_human and args.controller_test:
        ap.error('--assisted-human and --controller-test are different modes; choose one')
    try:
        specs = [classify_spec(s) for s in [args.human_deck, *args.opponent_deck]]
    except ValueError as exc:
        ap.error(str(exc))
    if specs[0].kind != 'file':
        ap.error('Human must provide a matching physical deck .dck file')
    decks = [str(s.path.resolve()) if s.path else s.java_arg for s in specs]
    runtime = args.runtime_dir.expanduser().resolve()
    jar = runtime / 'forge-gui-desktop-2.0.15-jar-with-dependencies.jar'
    classes = runtime / 'forge-agent-patch/classes'
    controller_class = ('PhysicalTableController.class' if args.assisted_human
                        else 'WebHumanLobbyPlayer.class')
    if not jar.is_file() or not (classes/'fly/agent'/controller_class).is_file():
        ap.error('Build first: python scripts/build_forge_source.py')
    if not (runtime / 'res').is_dir():
        ap.error('Runtime res/ directory is missing')
    pod = None
    if args.opponents == 'fly':
        # Keep the optional learned-opponent dependency out of the default Forge-AI path.
        from flycommander.fly_pod import FlyPod
        pod = FlyPod(checkpoint_dir=None if args.controller_test else ROOT/'data/fly-pod-checkpoints')
    try:
        if pod is not None:
            pod.start()
        cmd = ['java', '-Xmx3g', '-Dfly.agent.physical=true',
            f'-Dfly.agent.opponents={args.opponents}',
            *(['-Dfly.agent.assistedHuman=true'] if args.assisted_human else []),
            *(['-Dfly.agent.controllerTest=true'] if args.controller_test else []),
            '-cp', os.pathsep.join([str(classes), str(jar)]),
            'fly.agent.AgentMain', '1', *decks]
        print('[launcher] ' + shlex.join(cmd), flush=True)
        print('[launcher] cwd=' + str(runtime), flush=True)
        return subprocess.call(cmd, cwd=runtime)
    finally:
        if pod is not None:
            pod.stop()


if __name__ == '__main__':
    raise SystemExit(main())
