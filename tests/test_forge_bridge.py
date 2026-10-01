"""Physical table → Forge bridge tests.

The real counterpart is the Java agent in ``forge_patch/src/fly/agent``
(``/table/events``, ``/table/state``, ``/table/queue``). These tests run the
Python half against a *fake* agent that implements the exact wire contract,
so the contract itself is pinned here: if the Java side changes shape, this
file has to change with it.

What is verified:

* every physical event type is either mirrored by an action or documented as
  tracking-only (no event can silently fall on the floor),
* translation carries the recognised identity (set:collector) and the
  physical facts (tapped, counters, zone, token),
* an unreachable Forge spools in order, never raises, and flushes exactly once
  when the agent comes back,
* action ids make a retried batch idempotent,
* the full-state resync mirrors only the player's battlefield,
* the physical server mirrors player events and exposes the status endpoint.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flycommander.forge_table_bridge import (
    EVENT_TO_ACTION,
    TRACKING_ONLY_EVENTS,
    ForgeTableBridge,
)
from physical.events import EVENT_TYPES, Event


# ----------------------------------------------------------------------
# fake Forge agent
# ----------------------------------------------------------------------
class FakeForgeAgent:
    """Minimal stand-in for fly.agent.AgentServer's /table/* surface."""

    def __init__(self, mode: str = "queue", fail_first: int = 0,
                 reject_kinds: set[str] | None = None):
        self.mode = mode                     # "queue" | "sync"
        self.fail_first = fail_first         # simulate Forge not running yet
        self.reject_kinds = set(reject_kinds or ())
        self.batches: list[list[dict]] = []
        self.syncs: list[dict] = []
        self.applied = 0
        self.rejected = 0
        self.recent: list[str] = []
        self._seen: set[str] = set()
        self.get_paths: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self.port = 0
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------
    def start(self) -> "FakeForgeAgent":
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):        # silence
                pass

            def _json(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):                # noqa: N802
                agent.get_paths.append(self.path)
                if self.path == "/health":
                    self._json(200, {"status": "ready", "tableSeat": True})
                elif self.path == "/table/queue":
                    self._json(200, {"pending": max(0, len(agent.batches) - 1),
                                     "applied": agent.applied,
                                     "rejected": agent.rejected,
                                     "recent": agent.recent})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):               # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if agent.fail_first > 0:
                    agent.fail_first -= 1
                    self._json(503, {"error": "forge booting"})
                    return
                if self.path == "/table/events":
                    events = body.get("events") or []
                    agent.batches.append(events)
                    applied, rejected, dupes = 0, 0, 0
                    for event in events:
                        aid = event.get("actionId")
                        if aid in agent._seen:      # engine de-duplicates too
                            dupes += 1
                            continue
                        agent._seen.add(aid)
                        if event.get("action") in agent.reject_kinds:
                            rejected += 1
                            agent.recent.append(
                                f"rejected:{event.get('action')}")
                        else:
                            applied += 1
                            agent.recent.append(f"applied:{event.get('action')}")
                    agent.applied += applied
                    agent.rejected += rejected
                    if agent.mode == "queue":
                        self._json(200, {"status": "queued",
                                         "accepted": applied + rejected,
                                         "duplicates": dupes, "pending": 0})
                    else:
                        self._json(200, {
                            "status": "ok" if not rejected else "partial",
                            "applied": applied, "rejected": rejected,
                            "results": [
                                {"actionId": e.get("actionId"),
                                 "status": "rejected"
                                 if e.get("action") in agent.reject_kinds
                                 else "applied"}
                                for e in events]})
                elif self.path == "/table/state":
                    agent.syncs.append(body)
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture()
def agent():
    a = FakeForgeAgent().start()
    yield a
    a.stop()


# ----------------------------------------------------------------------
# contract completeness
# ----------------------------------------------------------------------
def test_every_physical_event_is_mirrored_or_tracking_only():
    covered = set(EVENT_TO_ACTION) | set(TRACKING_ONLY_EVENTS)
    assert set(EVENT_TYPES) - covered == set(), \
        "an event type would be silently dropped by the bridge"
    assert not (set(EVENT_TO_ACTION) & set(TRACKING_ONLY_EVENTS))


def test_translation_carries_identity_and_physical_fact():
    bridge = ForgeTableBridge(enabled=False)
    action = bridge.translate(Event(
        type="card_tapped", origin="vision", confidence=0.8,
        payload={"trackingId": "card3", "name": "Atraxa, Praetors' Voice",
                 "set": "2xm", "collectorNumber": "197",
                 "orientationDegrees": 91.0}))
    assert action["action"] == "set_tapped"
    assert action["tapped"] is True
    assert action["cardKey"] == "2xm:197"
    assert action["name"] == "Atraxa, Praetors' Voice"
    assert action["actionId"].startswith("physical:")

    untap = bridge.translate(Event(type="card_untapped", origin="vision",
                                   payload={"trackingId": "card3"}))
    assert untap["action"] == "set_tapped" and untap["tapped"] is False


def test_translation_uses_remembered_track_identity():
    bridge = ForgeTableBridge(enabled=False)
    bridge.remember_track("card7", name="Sol Ring", set_code="c21",
                          collector_number="263", oracle_id="abc")
    action = bridge.translate(Event(type="zone_change", origin="vision",
                                    payload={"trackingId": "card7",
                                             "to": "graveyard"}))
    assert action["cardKey"] == "c21:263"
    assert action["to"] == "graveyard"
    bridge.forget_track("card7")
    assert "cardKey" not in bridge.translate(
        Event(type="zone_change", origin="vision", payload={"trackingId": "card7"}))


def test_tokens_and_counters_are_expressed_physically():
    bridge = ForgeTableBridge(enabled=False)
    token = bridge.translate(Event(type="token_created", origin="player",
                                   payload={"tokenType": "Soldier", "count": 3,
                                            "power": 1, "toughness": 1}))
    assert token["action"] == "put_onto_battlefield" and token["token"] is True
    assert token["name"] == "Soldier" and token["count"] == 3

    plus = bridge.translate(Event(type="counter_added", origin="vision",
                                  payload={"trackingId": "card1",
                                           "counter": "+1/+1", "amount": 2}))
    assert (plus["action"], plus["counter"], plus["amount"]) == (
        "set_counters", "+1/+1", 2)
    minus = bridge.translate(Event(type="counter_removed", origin="vision",
                                   payload={"trackingId": "card1",
                                            "counter": "-1/-1"}))
    assert minus["amount"] == -1


def test_tracking_only_events_are_not_sent(agent):
    bridge = ForgeTableBridge(base_url=agent.url)
    result = bridge.send([Event(type="moved", origin="vision", payload={}),
                          Event(type="candidate_rejected", origin="vision")])
    assert result["status"] == "noop"
    assert agent.batches == []


# ----------------------------------------------------------------------
# spooling / ordering / idempotency
# ----------------------------------------------------------------------
def test_status_probe_uses_health_route_and_reports_running(agent):
    bridge = ForgeTableBridge(base_url=agent.url)
    status = bridge.status(probe=True)
    assert status["connected"] is True
    assert status["lastError"] is None
    assert agent.get_paths == ["/health"]


def test_physical_ui_refreshes_forge_health_indicator():
    from pathlib import Path

    ui = (Path(__file__).resolve().parents[1] / "physical" / "ui.html").read_text()
    assert "void refreshForge();" in ui
    assert 'fetch(API + "/api/forge/status"' in ui
    assert '"🟢 running"' in ui
    assert '"🔴 not running"' in ui


def test_unreachable_forge_spools_without_raising():
    bridge = ForgeTableBridge(base_url="http://127.0.0.1:9", timeout=0.2)
    out = bridge.send([Event(type="card_tapped", origin="vision",
                             payload={"trackingId": "card1"})])
    assert out["status"] == "offline" and out["spooled"] == 1
    status = bridge.status(probe=True)
    assert status["connected"] is False
    assert status["lastError"]


def test_spool_flushes_in_order_once_forge_returns():
    agent = FakeForgeAgent(fail_first=1).start()
    try:
        bridge = ForgeTableBridge(base_url=agent.url)
        events = [Event(type="card_tapped", origin="vision",
                        payload={"trackingId": "card1", "name": "Forest"}),
                  Event(type="card_untapped", origin="vision",
                        payload={"trackingId": "card1", "name": "Forest"}),
                  Event(type="life_changed", origin="player",
                        payload={"player": "player", "life": 38})]
        first = bridge.send(events)
        assert first["status"] == "offline" and first["spooled"] == 3
        second = bridge.flush()
        assert second["status"] in ("queued", "ok")
        assert second["spooled"] == 0
        assert len(agent.batches) == 1
        kinds = [e["action"] for e in agent.batches[0]]
        assert kinds == ["set_tapped", "set_tapped", "set_life"]
        assert agent.batches[0][0]["tapped"] is True
        assert agent.batches[0][1]["tapped"] is False
        assert agent.batches[0][2]["life"] == 38
    finally:
        agent.stop()


def test_retried_batch_is_idempotent_on_the_forge_side():
    """A restarted bridge replaying its spool must not double-apply.

    Action ids are ``<tableId>:<seq>``, so a replay of the same table id is
    recognised by Forge and ignored — the same physical fact is never applied
    twice, while a genuinely new fact (new seq) still applies.
    """
    agent = FakeForgeAgent().start()
    try:
        bridge = ForgeTableBridge(base_url=agent.url, table_id="table-a")
        bridge.send([Event(type="card_tapped", origin="vision",
                           payload={"trackingId": "card1"})])
        bridge.send([Event(type="card_tapped", origin="vision",
                           payload={"trackingId": "card1"})])
        assert agent.applied == 2

        # restart: same table id, sequence counter back to 0 → replays
        replay = ForgeTableBridge(base_url=agent.url, table_id="table-a")
        out = replay.send([Event(type="card_tapped", origin="vision",
                                 payload={"trackingId": "card1"})])
        assert agent.applied == 2, "replayed action must be de-duplicated"
        assert out["duplicates"] >= 1

        # a different table is a different source of truth and does apply
        other = ForgeTableBridge(base_url=agent.url, table_id="table-b")
        other.send([Event(type="card_tapped", origin="vision",
                          payload={"trackingId": "card1"})])
        assert agent.applied == 3
    finally:
        agent.stop()


def test_rejections_are_reported_not_hidden():
    # Forge refuses this one (e.g. the creature is not a legal attacker); the
    # bridge must surface the verdict instead of pretending it applied
    agent = FakeForgeAgent(mode="sync",
                           reject_kinds={"declare_attacker"}).start()
    try:
        bridge = ForgeTableBridge(base_url=agent.url)
        out = bridge.send([Event(type="attack_declared", origin="vision",
                                 payload={"trackingId": "card1",
                                          "target": "fly"})])
        assert out["status"] == "rejected" and out["rejected"] == 1
        assert bridge.status()["rejected"] == 1
    finally:
        agent.stop()


def test_spool_is_bounded_and_counts_drops():
    bridge = ForgeTableBridge(base_url="http://127.0.0.1:9", timeout=0.2,
                              spool_limit=3)
    for _ in range(5):
        bridge.send([Event(type="card_tapped", origin="vision",
                           payload={"trackingId": "card1"})])
    status = bridge.status()
    assert status["pending"] == 3 and status["dropped"] == 2


# ----------------------------------------------------------------------
# full resync
# ----------------------------------------------------------------------
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_full_resync_mirrors_only_the_player_battlefield(agent):
    bridge = ForgeTableBridge(base_url=agent.url)
    state = _Obj(objects={
        "card1": _Obj(tracking_id="card1", name="Forest", set_code="fdn",
                      collector_number="280", zone="battlefield",
                      controller="player", tapped=True, counters={},
                      is_token=False, base_power=None, base_toughness=None),
        "card2": _Obj(tracking_id="card2", name="Sol Ring", set_code="c21",
                      collector_number="263", zone="battlefield",
                      controller="fly", tapped=False, counters={},
                      is_token=False, base_power=None, base_toughness=None),
        "card3": _Obj(tracking_id="card3", name="Lightning Bolt", set_code="2x2",
                      collector_number="117", zone="graveyard",
                      controller="player", tapped=False, counters={},
                      is_token=False, base_power=None, base_toughness=None),
    }, players={"player": _Obj(life=37), "fly": _Obj(life=40)})
    out = bridge.sync_state(state)
    assert out["status"] == "ok" and out["battlefield"] == 1
    sent = agent.syncs[0]
    assert [c["name"] for c in sent["battlefield"]] == ["Forest"]
    assert sent["battlefield"][0]["cardKey"] == "fdn:280"
    assert sent["battlefield"][0]["tapped"] is True
    assert sent["life"] == 37
    assert sent["seat"] == "player" and sent["tableId"] == "physical"


def test_resync_accepts_plain_dict_state(agent):
    bridge = ForgeTableBridge(base_url=agent.url)
    out = bridge.sync_state({"objects": [
        {"trackingId": "c9", "name": "Arcane Signet", "set": "c19",
         "collectorNumber": "211", "zone": "battlefield",
         "controller": "player", "tapped": False}],
        "players": {"player": {"life": 40}}})
    assert out["status"] == "ok"
    assert agent.syncs[0]["battlefield"][0]["cardKey"] == "c19:211"


def test_async_verdict_comes_from_the_queue_endpoint():
    agent = FakeForgeAgent(mode="queue").start()
    try:
        bridge = ForgeTableBridge(base_url=agent.url)
        out = bridge.send([Event(type="set_tapped", origin="vision", payload={}),
                           Event(type="card_tapped", origin="vision",
                                 payload={"trackingId": "card1"})])
        assert out["status"] in ("queued", "noop")
        assert bridge.status()["applied"] == agent.applied
        assert bridge.status()["pending"] == 0
    finally:
        agent.stop()


# ----------------------------------------------------------------------
# physical server integration
# ----------------------------------------------------------------------
def test_server_mirrors_player_events_and_reports_status(tmp_path):
    from physical.server import PhysicalTableApp

    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    counted: list[dict] = []
    stub = _StubBridge()
    stub.connected = True                            # type: ignore[attr-defined]

    def _send(events):
        counted.extend(events)
        return {"status": "queued", "queued": len(events)}

    stub.send = _send
    stub.queue_stats = lambda: {"applied": len(counted), "rejected": 0,
                                "pending": 0}
    app.forge = stub
    app.apply_player_event("life_changed",
                           {"player": "player", "life": 35, "delta": -5})
    assert counted and counted[0]["type"] == "life_changed"
    assert counted[0]["payload"]["life"] == 35
    status = app.forge_status()
    assert status["connected"] and status["forge"]["applied"] == 1
    # the UI snapshot carries the bridge state
    assert app.ui_snapshot()["forge"]["connected"] is True


def test_no_post_route_hangs(tmp_path):
    """Every POST route must answer promptly (keep-alive body-read regression).

    A handler that reads the request body twice appears to hang forever on
    HTTP/1.1 keep-alive, because the second read blocks waiting for bytes the
    client will never send. This walks the whole POST surface with a short
    timeout so that bug class cannot come back silently.
    """
    import urllib.request

    from physical.server import PhysicalTableApp, PhysicalTableServer

    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    app.forge = _StubBridge()
    server = PhysicalTableServer(app, port=0, host="127.0.0.1")
    server.start()
    try:
        port = server._http.server_address[1]
        posts = {
            "/api/event": {"type": "life_changed",
                           "payload": {"player": "player", "life": 39}},
            "/api/forge/sync": {},
            "/api/vision/identify": {},
            "/api/pending/confirm": {"index": 0},
            "/api/pending/reject": {"index": 0},
            "/api/scan/debug": {},
            "/api/brain/decide": {},
            "/api/register/frame": {},
        }
        for route, body in posts.items():
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{route}",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    assert resp.status == 200, (route, resp.status)
            except Exception as exc:                  # noqa: BLE001 - reported
                pytest.fail(f"POST {route} did not answer: {exc!r}")
        for route in ("/api/state", "/api/vision/status", "/api/forge/status",
                      "/api/forge/queue", "/api/pending", "/api/tracks"):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}{route}", timeout=5) as resp:
                    assert resp.status == 200, (route, resp.status)
            except Exception as exc:                  # noqa: BLE001 - reported
                pytest.fail(f"GET {route} did not answer: {exc!r}")
    finally:
        server.stop()


class _StubBridge:
    """Offline bridge stub: keeps route tests fast and deterministic."""

    enabled = True

    def send(self, events):
        return {"status": "queued", "queued": len(events)}

    def status(self, probe=False):
        return {"connected": bool(getattr(self, "connected", False)),
                "pending": 0, "enabled": True, "url": "http://stub",
                "seat": "player", "tableId": "physical", "applied": 0,
                "rejected": 0, "dropped": 0, "batches": 0, "lastError": None,
                "lastContact": None, "lastAction": None}

    def queue_stats(self):
        return {}

    def sync_state(self, state):
        return {"status": "ok", "battlefield": 0}

    def health(self):
        return None

    def remember_track(self, *args, **kwargs):
        pass

    def forget_track(self, *args, **kwargs):
        pass


def test_disabled_bridge_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYCOMMANDER_FORGE_SYNC", "0")
    from physical.server import PhysicalTableApp

    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    assert app.forge.enabled is False
    assert app.mirror_to_forge(Event(type="card_tapped", origin="vision",
                                     payload={})) is None
