"""Own the browser-launched four-seat Forge pod process.

The server never shells a user string: deck lists are parsed and serialized to
private ``.dck`` files, while opponent modes are restricted to a Forge-random
prebuilt or one of those generated files. Forge/JVM work runs in its own process
group so it can be stopped with the Python table server.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid

from physical.decklists import forge_deck_text

ROOT = Path(__file__).resolve().parents[1]
FORGE_JAR_NAME = "forge-gui-desktop-2.0.15-jar-with-dependencies.jar"


class PodSessionManager:
    def __init__(self, data_dir: str | Path = "data/physical",
                 runtime_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        configured = runtime_dir or os.environ.get("FLYCOMMANDER_FORGE_RUNTIME_DIR")
        self.runtime_dir = Path(configured or ROOT / "data/forge-built-runtime").expanduser().resolve()
        self.session_dir = self.data_dir / "pod"
        self.deck_dir = self.session_dir / "decks"
        self.log_path = self.session_dir / "session.log"
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._session_id: str | None = None
        self._stopping = False
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._last_decks: list[str] = []
        self._deck_paths: list[Path] = []

    def _runtime_error(self) -> str | None:
        jar = self.runtime_dir / FORGE_JAR_NAME
        classes = self.runtime_dir / "forge-agent-patch" / "classes"
        required = classes / "fly" / "agent" / "WebHumanLobbyPlayer.class"
        if not jar.is_file() or not required.is_file():
            return ("Forge runtime not built. Run `python3 scripts/build_forge_source.py` "
                    "with the pinned Forge source, Java 17+ and Maven; expected "
                    f"{jar} and WebHumanLobbyPlayer classes.")
        if not (self.runtime_dir / "res").is_dir():
            return f"Forge runtime is missing its res/ directory: {self.runtime_dir / 'res'}"
        java = shutil.which("java")
        if java is None:
            return "Java 17+ runtime not found on PATH; install/configure a JDK and restart."
        try:
            version = subprocess.run([java, "-version"], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     timeout=5, check=False).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Could not verify the Java runtime: {exc}"
        match = re.search(r'(?:version\s+"?|openjdk\s+)([0-9]+)', version, re.IGNORECASE)
        if not match or int(match.group(1)) < 17:
            return f"Java 17+ is required to run Forge; detected: {version.strip() or 'unknown'}"
        return None

    def _write_deck(self, serialized: str, label: str, session_id: str) -> Path:
        self.deck_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c.lower() if c.isalnum() else "-" for c in label).strip("-")[:24]
        path = self.deck_dir / f"{session_id}-{safe or 'deck'}.dck"
        self._deck_paths.append(path)
        path.write_text(serialized, encoding="utf-8")
        return path

    def _cleanup_decks(self) -> None:
        for path in self._deck_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._deck_paths.clear()

    def start(self, payload: dict) -> dict:
        """Validate setup, create generated Forge decks and start the pod."""
        if not isinstance(payload, dict):
            return {"status": "error", "error": "Invalid setup payload"}
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return {**self.status(), "status": "busy",
                        "error": "A Commander pod is already running"}
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            self._process = None
            self._stopping = False
            self._last_error = None
            self._session_id = None
            self._started_at = None
            self._last_decks = []
            self._cleanup_decks()

            human_list = payload.get("humanDeck")
            if not isinstance(human_list, str):
                return {"status": "error", "error": "Paste your 100-card Commander list"}
            opponents = payload.get("opponents")
            if not isinstance(opponents, list) or len(opponents) != 3:
                return {"status": "error", "error": "Choose a deck for each of the three AI seats"}

            try:
                human_contents = forge_deck_text(human_list, "Your Forge deck")
                ai_specs: list[str] = []
                ai_labels: list[str] = []
                custom_decks: list[tuple[str, str]] = []
                for number, entry in enumerate(opponents, start=1):
                    if not isinstance(entry, dict):
                        raise ValueError(f"AI seat {number}: invalid deck choice")
                    mode = entry.get("mode")
                    if mode == "random":
                        ai_specs.append("random")
                        ai_labels.append("Random Forge Commander deck")
                    elif mode == "custom":
                        deck_text = entry.get("deck")
                        if not isinstance(deck_text, str):
                            raise ValueError(f"AI seat {number}: paste a Commander deck list")
                        label = f"AI {number} custom deck"
                        try:
                            serialized = forge_deck_text(deck_text, label)
                        except ValueError as exc:
                            raise ValueError(f"AI seat {number}: {exc}") from exc
                        custom_decks.append((label, serialized))
                        ai_labels.append(f"AI {number}: custom deck list")
                        # Reserve a placeholder; replaced by the generated path below.
                        ai_specs.append("")
                    else:
                        raise ValueError(f"AI seat {number}: choose Random prebuilt or Custom list")
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}

            runtime_error = self._runtime_error()
            if runtime_error:
                self._last_error = runtime_error
                return {"status": "error", "error": runtime_error}

            session_id = uuid.uuid4().hex[:12]
            try:
                human_path = self._write_deck(human_contents, "Your digital Forge deck", session_id)
                custom_iter = iter(custom_decks)
                for index, spec in enumerate(ai_specs):
                    if spec == "":
                        label, serialized = next(custom_iter)
                        ai_specs[index] = str(self._write_deck(serialized, label, session_id))

                self.session_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-u", str(ROOT / "scripts" / "run_physical_pod.py"),
                    "--opponents", "forge",
                    "--runtime-dir", str(self.runtime_dir),
                    "--human-deck", str(human_path),
                ]
                for spec in ai_specs:
                    command.extend(("--fly-deck", spec))

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                self._stopping = False
                self._last_error = None
                self._session_id = session_id
                self._started_at = time.time()
                self._last_decks = ["Your Forge digital deck", *ai_labels]
                self.log_path.write_text("", encoding="utf-8")
                self._log_handle = self.log_path.open("ab", buffering=0)
                try:
                    self._process = subprocess.Popen(
                        command, cwd=ROOT, env=env,
                        stdout=self._log_handle, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as exc:
                    self._log_handle.close()
                    self._log_handle = None
                    self._process = None
                    self._cleanup_decks()
                    self._last_error = f"Could not start the pod launcher: {exc}"
                    return {"status": "error", "error": self._last_error}
                return {
                    "status": "starting", "sessionId": session_id,
                    "decks": list(self._last_decks),
                    "message": "Starting Forge and loading the four Commander seats…",
                }
            except (ValueError, OSError, StopIteration) as exc:
                self._cleanup_decks()
                return {"status": "error", "error": str(exc)}

    def _tail(self, lines: int = 80) -> list[str]:
        try:
            if not self.log_path.is_file():
                return []
            # Bound reads even if a long-running game has a large log.
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 256_000))
                data = handle.read().decode("utf-8", errors="replace")
            return data.splitlines()[-lines:]
        except OSError:
            return []

    @staticmethod
    def _resolved_decks(lines: list[str]) -> list[str]:
        resolved: list[tuple[int, str]] = []
        for line in lines:
            marker = "[FlyAgent] opponent deck "
            if marker not in line:
                marker = "[FlyAgent] ai deck "
            if marker not in line or ": " not in line:
                continue
            try:
                rest = line.split(marker, 1)[1]
                index_text, name = rest.split(": ", 1)
                resolved.append((int(index_text), name.strip()))
            except (ValueError, IndexError):
                continue
        return [name for _, name in sorted(resolved)]

    def status(self) -> dict:
        with self._lock:
            lines = self._tail()
            process = self._process
            if process is None:
                current = "error" if self._last_error else "idle"
                exit_code = None
            else:
                exit_code = process.poll()
                if exit_code is None:
                    current = "running" if any("Calling Match.startGame" in line for line in lines) else "starting"
                elif self._stopping:
                    current = "stopped"
                elif exit_code == 0:
                    current = "finished"
                else:
                    current = "failed"
                    if not self._last_error:
                        self._last_error = f"Pod launcher exited with code {exit_code}"
                if exit_code is not None and self._log_handle is not None:
                    self._log_handle.close()
                    self._log_handle = None
                if exit_code is not None:
                    self._cleanup_decks()
            return {
                "status": current,
                "sessionId": self._session_id,
                "startedAt": self._started_at,
                "exitCode": exit_code,
                "error": self._last_error,
                "decks": list(self._last_decks),
                "resolvedAiDecks": self._resolved_decks(lines),
                "log": lines[-40:],
            }

    def stop(self) -> dict:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            self._stopping = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.wait(timeout=3)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            return self.status()

    def close(self) -> None:
        self.stop()
