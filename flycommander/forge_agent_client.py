"""FlyCommander — Forge ⇄ Fly bridge.

Two halves:

* :class:`AgentClient` — polls the Java agent (port 8791) for health,
  observations and game results.
* :class:`FlyBrainServer` — an HTTP server (port 8792) the Java patch calls at
  every fly decision point (``POST /decide``). Each call encodes the
  observation through the sensory pathway, picks a macro action with the
  mushroom body, and journals the step. When a game result arrives, the
  dopaminergic system replays the episode and updates KC→MBON weights online.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

from brain.dopamine_plasticity import DopamineSystem
from brain.mushroom_body import ACTION_NAMES, MushroomBody
from flycommander.reward_shaping import RewardComputer
from flycommander.sensory_encoder import N_SENSORY, observation_to_state


class AgentClient:
    """Client for the Java-side agent HTTP endpoints (default port 8791)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8791") -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=5) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def health(self) -> dict[str, Any] | None:
        return self._get("/health")

    def observation(self) -> dict[str, Any] | None:
        return self._get("/observation")

    def result(self) -> dict[str, Any] | None:
        return self._get("/result")

    def wait_until_ready(self, timeout_s: float = 300.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            h = self.health()
            if h and h.get("status") == "ready":
                return True
            time.sleep(1.0)
        return False


class FlyBrainServer:
    """Serves /decide for the Java patch and runs online learning."""

    def __init__(
        self,
        mb: MushroomBody,
        dopamine: DopamineSystem,
        reward: RewardComputer,
        port: int = 8792,
        log_path: str | None = None,
        on_episode_end: Callable[[dict], None] | None = None,
        learn: bool = True,
    ) -> None:
        self.mb = mb
        self.dopamine = dopamine
        self.reward = reward
        self.port = port
        self.log_path = log_path
        self.on_episode_end = on_episode_end
        self.learn = learn

        self._lock = threading.Lock()
        self._episode: list[dict] = []
        self._pending_obs: dict | None = None
        self.episodes_done = 0
        self.decisions_served = 0
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence request logging
                pass

            def do_POST(self):
                if self.path != "/decide":
                    self._send(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    payload = {}
                try:
                    action = server._handle_decide(payload.get("observation") or {})
                    self._send(200, {"action": action,
                                     "actionName": ACTION_NAMES[action]})
                except Exception as exc:  # never crash the game loop
                    self._send(200, {"action": 2, "error": str(exc)})

            def do_GET(self):
                if self.path == "/stats":
                    self._send(200, {
                        "episodesDone": server.episodes_done,
                        "decisionsServed": server.decisions_served,
                        "temperature": server.mb.temperature,
                    })
                else:
                    self._send(404, {"error": "not found"})

            def _send(self, code: int, body: dict):
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._http = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(
            target=self._http.serve_forever, name="fly-brain-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http = None

    # ------------------------------------------------------------------
    def _legal_mask(self, obs: dict) -> np.ndarray:
        can = obs.get("canPlay", {}) if isinstance(obs.get("canPlay"), dict) else {}
        mask = np.array([
            bool(can.get("land") or can.get("spell")),  # 0 play
            True,                                        # 1 attack
            True,                                        # 2 hold
            bool(can.get("ability")),                    # 3 interact
        ])
        return mask

    def _handle_decide(self, obs: dict) -> int:
        with self._lock:
            state = observation_to_state(obs)
            kc = self.mb.encode_state(state)

            # reward for the previous action, now that we see the outcome
            if self._pending_obs is not None and self._episode:
                rb = self.reward.step(obs)
                self._episode[-1]["reward"] = rb.reward
                self._episode[-1]["rewardComponents"] = rb.components

            mask = self._legal_mask(obs)
            action = self.mb.decide(kc, mask)
            valences = self.mb.action_valences(kc, mask)

            self._episode.append({
                "obs": obs,
                "kcIdx": np.flatnonzero(kc).tolist(),
                "action": int(action),
                "actionName": ACTION_NAMES[action],
                "valences": [None if not np.isfinite(v) else round(float(v), 4)
                             for v in valences],
                "reward": 0.0,
            })
            self._pending_obs = obs
            self.decisions_served += 1
            return int(action)

    # ------------------------------------------------------------------
    def finish_episode(self, result: dict) -> dict:
        """Apply terminal reward, replay DPR updates, log and reset."""
        with self._lock:
            rb = self.reward.terminal(result)
            if self._episode:
                self._episode[-1]["reward"] += rb.reward
                discounted = rb.reward
                # temporal discount backwards through the episode
                gamma = 0.97
                for step in reversed(self._episode[:-1]):
                    discounted = step["reward"] + gamma * discounted
                    step["trainReward"] = discounted
                self._episode[-1]["trainReward"] = self._episode[-1]["reward"]

            if self.learn:
                kc_bank = np.zeros(self.mb.connectome.n_kc, dtype=np.float64)
                for step in self._episode:
                    kc_bank[:] = 0.0
                    kc_bank[step["kcIdx"]] = 1.0
                    self.dopamine.update(kc_bank, step.get("trainReward", 0.0))
                self.mb.anneal()

            summary = {
                "flyWon": bool(result.get("flyWon")),
                "turns": result.get("turns"),
                "reason": result.get("reason"),
                "steps": len(self._episode),
                "actions": [s["actionName"] for s in self._episode],
                # raw rewards (trainReward is the discounted return, used for
                # learning only — summing it would double-count)
                "totalReward": float(sum(s.get("reward", 0.0)
                                         for s in self._episode)),
                "temperature": round(self.mb.temperature, 4),
            }
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "summary": summary,
                        "steps": [
                            {k: s[k] for k in ("action", "actionName",
                                               "reward", "trainReward",
                                               "kcIdx") if k in s}
                            for s in self._episode
                        ],
                    }) + "\n")

            self._episode = []
            self._pending_obs = None
            self.reward.reset()
            self.episodes_done += 1
            if self.on_episode_end:
                self.on_episode_end(summary)
            return summary
