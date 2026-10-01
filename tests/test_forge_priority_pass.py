"""Run the real Forge phase/stack regression when local runtime prerequisites exist.

No Python priority model, source-text surrogate or mocked Forge result.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_real_forge_human_pass_advances_priority_and_stack(tmp_path):
    root = Path(__file__).resolve().parents[1]
    jar = Path(os.environ.get('FORGE_TEST_JAR', str(root / 'data/forge-built-runtime/forge-gui-desktop-2.0.15-jar-with-dependencies.jar'))).resolve()
    runtime = Path(os.environ.get('FORGE_TEST_RUNTIME', str(jar.parent))).resolve()
    if not shutil.which('javac') or not shutil.which('java') or not jar.is_file() or not (runtime / 'res').is_dir():
        pytest.skip('JDK + real Forge jar + external res/ required; priority regression NOT executed')
    sources = sorted((root / 'forge_patch/src').rglob('*.java'))
    sources.append(root / 'forge_patch/tests/fly/agent/ForgePriorityPassTest.java')
    subprocess.run(['javac', '--release', '17', '-cp', str(jar), '-d', str(tmp_path),
                    *map(str, sources)], check=True, timeout=120)
    # Card DB boot can be slow. Individual loop steps have tight Java deadlines
    # so the former infinite human pass loop fails with a useful diagnostic.
    subprocess.run(['java', '-Xmx3g', '-cp', os.pathsep.join([str(tmp_path), str(jar)]),
                    'fly.agent.ForgePriorityPassTest', str(root / 'tests/fixtures/controller-test.dck')],
                   cwd=runtime, check=True, timeout=420)
