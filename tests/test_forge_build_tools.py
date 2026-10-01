import subprocess
import pytest
from scripts import build_forge_source as build


def test_missing_tools_actionable(monkeypatch):
    monkeypatch.setattr(build.shutil, 'which', lambda _: None)
    with pytest.raises(ValueError, match='openjdk-17-jdk maven'):
        build.check_tools()


@pytest.mark.parametrize('version', ['javac 11.0.2', 'not a compiler'])
def test_old_or_invalid_compiler(monkeypatch, version):
    monkeypatch.setattr(build.shutil, 'which', lambda name: '/bin/'+name)
    values = {'java': 'openjdk version "17.0.1"', 'javac': version, 'mvn': 'Apache Maven 3.9.9'}
    monkeypatch.setattr(build.subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, values[cmd[0]]))
    with pytest.raises(ValueError, match='javac must be version 17'):
        build.check_tools()


def test_tool_versions(monkeypatch):
    monkeypatch.setattr(build.shutil, 'which', lambda name: '/bin/'+name)
    values = {'java': 'openjdk version "17.0.1"', 'javac': 'javac 17.0.1', 'mvn': 'Apache Maven 3.9.9'}
    monkeypatch.setattr(build.subprocess, 'run', lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, values[cmd[0]]))
    assert build.check_tools() == values
