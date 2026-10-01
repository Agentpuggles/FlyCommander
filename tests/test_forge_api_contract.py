"""Optional real-JAR compilation/signature test. Never substitutes Forge stubs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_patch_against_real_forge_jar(tmp_path):
    root = Path(__file__).resolve().parents[1]
    jar = Path(os.environ.get('FORGE_TEST_JAR', str(root / 'data/forge-built-runtime/forge-gui-desktop-2.0.15-jar-with-dependencies.jar')))
    if not shutil.which('javac') or not shutil.which('java') or not jar.is_file():
        pytest.skip('Real Forge jar and JDK required; API compilation NOT verified')
    sources = sorted((root / 'forge_patch/src').rglob('*.java'))
    sources.append(root / 'forge_patch/tests/fly/agent/ForgeApiContractTest.java')
    subprocess.run(['javac', '--release', '17', '-cp', str(jar), '-d', str(tmp_path),
                    *map(str, sources)], check=True, timeout=120)
    subprocess.run(['java', '-cp', os.pathsep.join([str(tmp_path), str(jar)]),
                    'fly.agent.ForgeApiContractTest'], check=True, timeout=30)
