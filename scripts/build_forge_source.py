#!/usr/bin/env python3
"""Verify pinned Forge source, build desktop reactor, then compile the agent.

Requires git, Maven and a full JDK 17+ on PATH. Never edits upstream source.
Large source/build/resource artifacts stay in ignored data/.
"""
import argparse
from pathlib import Path
import shutil
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TAG = 'forge-2.0.15'
TAG_SHA = 'b18e110a32462958aeda4b7fbac48c06399a2c23'
COMMIT = '4ec5f1a2c32fa90ecb983a72b9eb47aa5c5d7676'


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def verify(source):
    for ref, expected in ((TAG, TAG_SHA), (TAG + '^{}', COMMIT), ('HEAD', COMMIT)):
        actual = git(source, 'rev-parse', ref)
        if actual != expected:
            raise ValueError(f'{ref}: expected {expected}, got {actual}')
    if git(source, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Upstream tracked source is modified; refusing an unpinned build')
    if not (source / 'forge-gui/res').is_dir():
        raise ValueError('Forge external res/ directory is missing')


def check_tools():
    missing = [tool for tool in ('java', 'javac', 'mvn') if not shutil.which(tool)]
    if missing:
        raise ValueError('Missing: ' + ', '.join(missing) +
                         '. Ubuntu/Debian: sudo apt-get install openjdk-17-jdk maven; set JAVA_HOME and PATH.')
    versions = {}
    for tool in ('java', 'javac', 'mvn'):
        result = subprocess.run([tool, '-version' if tool != 'mvn' else '--version'],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        versions[tool] = result.stdout.strip()
    for tool in ('java', 'javac'):
        match = re.search(r'(?:version\s+"?|javac\s+)(\d+)', versions[tool])
        if not match or int(match.group(1)) < 17:
            raise ValueError(f'{tool} must be version 17 or newer: {versions[tool]}')
    if 'Apache Maven' not in versions['mvn']:
        raise ValueError('mvn is not Apache Maven: ' + versions['mvn'])
    sources = list((ROOT / 'forge_patch/src/fly/agent').glob('*.java'))
    if not sources or not (ROOT / 'forge_patch/src/fly/agent/AgentMain.java').is_file():
        raise ValueError('FlyCommander Java patch missing under forge_patch/src/fly/agent')
    return versions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'data/forge-source-checkout')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    source = args.source.resolve()
    try:
        verify(source)
        print(f'Verified {TAG}: tag {TAG_SHA}, commit {COMMIT}', flush=True)
        if args.verify_only:
            return 0
        versions = check_tools()
        for tool, version in versions.items():
            print(f'{tool}: {version}', flush=True)
        # install makes reactor dependencies available to the standalone assembly goal.
        # Skip native launchers, not Java compilation. Upstream tests are a separate gate.
        subprocess.run(['mvn', '-B', '-pl', 'forge-gui-desktop', '-am',
                        '-DskipTests', '-Dlaunch4j.skip=true', 'install'], cwd=source, check=True)
        jar = source / 'forge-gui-desktop/target/forge-gui-desktop-2.0.15-jar-with-dependencies.jar'
        if not jar.is_file():
            raise ValueError(f'Build did not produce expected runtime: {jar}')
        runtime = ROOT / 'data/forge-built-runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jar, runtime / jar.name)
        resources = runtime / 'res'
        if not resources.exists():
            resources.symlink_to(source / 'forge-gui/res', target_is_directory=True)
        elif resources.resolve() != (source / 'forge-gui/res').resolve():
            raise ValueError('Runtime res/ points at a different resource tree')
        classes = runtime / 'forge-agent-patch/classes'
        classes.mkdir(parents=True, exist_ok=True)
        subprocess.run(['javac', '--release', '17', '-cp', str(jar), '-d', str(classes),
                        *map(str, sorted((ROOT / 'forge_patch/src').rglob('*.java')))], check=True)
        print(f'Compiled agent and staged runtime at {runtime}. Gameplay has NOT been tested.')
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Forge source build stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
