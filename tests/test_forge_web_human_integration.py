"""Run a real Forge game through the browser-backed human action path.

This is intentionally an external-runtime integration test: it skips rather
than substituting Forge stubs, a Python rules model, or source-text assertions.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_web_human_land_spell_resolution_and_combat(tmp_path):
    root = Path(__file__).resolve().parents[1]
    jar = Path(os.environ.get(
        'FORGE_TEST_JAR',
        str(root / 'data/forge-built-runtime/forge-gui-desktop-2.0.15-jar-with-dependencies.jar'),
    )).resolve()
    runtime = Path(os.environ.get('FORGE_TEST_RUNTIME', str(jar.parent))).resolve()
    if (not shutil.which('javac') or not shutil.which('java') or not jar.is_file()
            or not (runtime / 'res').is_dir()):
        pytest.skip('JDK + real Forge jar + external res/ required; human integration NOT executed')

    sources = sorted((root / 'forge_patch/src').rglob('*.java'))
    sources.append(root / 'forge_patch/tests/fly/agent/WebHumanGameIntegrationTest.java')
    subprocess.run(
        ['javac', '--release', '17', '-cp', str(jar), '-d', str(tmp_path), *map(str, sources)],
        check=True,
        timeout=120,
    )
    subprocess.run(
        ['java', '-Xmx3g', '-cp', os.pathsep.join([str(tmp_path), str(jar)]),
         'fly.agent.WebHumanGameIntegrationTest',
         str(root / 'tests/fixtures/controller-test.dck')],
        cwd=runtime,
        check=True,
        timeout=420,
    )
