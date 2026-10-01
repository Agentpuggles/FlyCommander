#!/usr/bin/env python3
"""Compile and launch the explicitly experimental, AI-assisted human seat."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    home = Path(args.forge_dir).expanduser().resolve()
    jar = home / 'forge-gui-desktop-2.0.15-jar-with-dependencies.jar'
    if not jar.is_file() or not (home / 'res').is_dir():
        raise ValueError('Expected an extracted Forge 2.0.15 distribution with its jar and res/ directory')
    if len(args.ai_deck) != 3:
        raise ValueError('Exactly three --ai-deck arguments are required')
    decks = []
    for spec in [args.human_deck, *args.ai_deck]:
        path = Path(spec).expanduser().resolve()
        if path.suffix.lower() != '.dck' or not path.is_file():
            raise ValueError(f'Provide an existing Forge .dck file: {spec}')
        decks.append(str(path))
    java, javac = shutil.which('java'), shutil.which('javac')
    if not java or not javac:
        raise ValueError('A full JDK is required: put java and javac on PATH (use the Java version required by your Forge build)')
    classes = ROOT / 'data' / 'forge-patch-classes'
    classes.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / 'forge_patch/src').rglob('*.java'))
    compile_cmd = [javac, '-cp', str(jar), '-d', str(classes), *map(str, sources)]
    launch_cmd = [java, '-Xmx3g', '-Dfly.agent.table=1', '-Dfly.agent.experimentalAssisted=true',
                  f'-Dfly.agent.port={args.agent_port}', '-cp', os.pathsep.join([str(classes), str(jar)]),
                  'fly.agent.AgentMain', '1', *decks]
    return home, compile_cmd, launch_cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forge-dir', required=True)
    ap.add_argument('--human-deck', required=True)
    ap.add_argument('--ai-deck', action='append', default=[])
    ap.add_argument('--agent-port', type=int, default=8791)
    ap.add_argument('--experimental-assisted', action='store_true', required=True,
                    help='acknowledge that micro decisions are AI-controlled and runtime verification is pending')
    args = ap.parse_args()
    try:
        home, compile_cmd, launch_cmd = prepare(args)
        subprocess.run(compile_cmd, check=True)
        print('Experimental assisted game. Open /play on your physical-table server.', flush=True)
        print('Scanner observations never change this game. Forge owns the shuffled hand.', flush=True)
        return subprocess.call(launch_cmd, cwd=home)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Cannot start game: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
