import json
import urllib.error
import urllib.request

from physical.pod_session import PodSessionManager


VALID_LIST = "Commander\n1 The Gitrog Monster\nMainboard\n99 Forest\n"


def test_runtime_preflight_is_clear(tmp_path):
    manager = PodSessionManager(data_dir=tmp_path, runtime_dir=tmp_path / "missing")
    result = manager.start({"humanDeck": VALID_LIST,
                            "opponents": [{"mode": "random"}] * 3})
    assert result["status"] == "error"
    assert "Forge runtime not built" in result["error"]
    assert manager.status()["status"] == "error"


def test_start_validates_three_custom_or_random_ai_decks_before_spawn(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    (runtime / "res").mkdir(parents=True)
    (runtime / "forge-agent-patch/classes/fly/agent").mkdir(parents=True)
    (runtime / "forge-agent-patch/classes/fly/agent/WebHumanLobbyPlayer.class").touch()
    (runtime / "forge-gui-desktop-2.0.15-jar-with-dependencies.jar").touch()
    manager = PodSessionManager(data_dir=tmp_path / "data", runtime_dir=runtime)

    def unexpected_spawn(*args, **kwargs):
        raise AssertionError("invalid setup must not launch a process")
    monkeypatch.setattr("physical.pod_session.subprocess.Popen", unexpected_spawn)

    invalid_count = manager.start({"humanDeck": VALID_LIST,
                                   "opponents": [{"mode": "random"}] * 2})
    assert invalid_count["status"] == "error"
    assert "three AI seats" in invalid_count["error"]

    invalid_list = manager.start({"humanDeck": "Mainboard\n100 Forest",
                                  "opponents": [{"mode": "random"}] * 3})
    assert invalid_list["status"] == "error"
    assert "Commander section" in invalid_list["error"]

    invalid_ai = manager.start({"humanDeck": VALID_LIST,
                                "opponents": [{"mode": "random"},
                                              {"mode": "custom", "deck": "bad"},
                                              {"mode": "random"}]})
    assert invalid_ai["status"] == "error"
    assert "AI seat 2" in invalid_ai["error"]


def test_pod_status_api_and_runtime_error(tmp_path):
    from flycommander.human_game import HumanGameClient
    from physical.server import PhysicalTableApp, PhysicalTableServer

    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    app.human_game = HumanGameClient("http://127.0.0.1:1")
    app.pod_session = PodSessionManager(data_dir=tmp_path,
                                        runtime_dir=tmp_path / "missing-runtime")
    server = PhysicalTableServer(app, port=0, host="127.0.0.1")
    server.start()
    try:
        base = f"http://127.0.0.1:{server._http.server_port}"
        with urllib.request.urlopen(base + "/api/pod/status") as response:
            assert json.load(response)["status"] == "idle"
        request = urllib.request.Request(
            base + "/api/pod/start",
            data=json.dumps({"humanDeck": VALID_LIST,
                             "opponents": [{"mode": "random"}] * 3}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            result = json.loads(error.read())
        else:
            raise AssertionError("missing Forge runtime should fail preflight")
        assert "Forge runtime not built" in result["error"]
    finally:
        server.stop()
