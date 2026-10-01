"""Run Java lifecycle tests when a JDK is available, otherwise explicitly skip."""
import shutil
import subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize("name", ["DecisionBroker", "PhysicalIdentityLedger"])
def test_java_decision_lifecycle(tmp_path, name):
    if not shutil.which('javac') or not shutil.which('java'):
        pytest.skip('Full JDK unavailable: Java concurrency test NOT executed')
    root = Path(__file__).resolve().parents[1]
    sources = [root/f'forge_patch/src/fly/agent/{name}.java',
               root/f'forge_patch/tests/fly/agent/{name}Test.java']
    subprocess.run(['javac', '--release', '17', '-d', str(tmp_path), *map(str, sources)], check=True)
    subprocess.run(['java', '-cp', str(tmp_path), f'fly.agent.{name}Test'], check=True, timeout=15)
