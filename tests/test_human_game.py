"""Gateway/launcher tests; these do not verify a live Forge game."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flycommander.human_game import HumanGameClient
import pytest


def test_disconnected_never_queues():
    client = HumanGameClient('http://127.0.0.1:1')
    assert client.state()['status'] == 'disconnected'
    assert client.decide({'id': 'a', 'choice': 'Pass priority'})['status'] == 'disconnected'


def test_bad_decision_is_rejected_without_network():
    client = HumanGameClient('http://127.0.0.1:1')
    for body in ({}, {'id': 'x', 'choice': 3}, [], {'id': 'x'*257, 'choice': 'y'}):
        assert client.decide(body)['status'] == 'rejected'


def test_gateway_preserves_prompt_and_rejection():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"status":"awaiting_decision","prompt":{"id":"fresh"}}')
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(409); self.end_headers()
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HumanGameClient(f'http://127.0.0.1:{server.server_port}')
        assert client.state()['prompt']['id'] == 'fresh'
        assert client.decide({'id': 'old', 'choice': 'Pass priority', 'extra': 1})['status'] == 'rejected'
        assert seen == [{'id': 'old', 'choice': 'Pass priority'}]
    finally:
        server.shutdown(); server.server_close()


def test_launcher_requires_real_distribution(tmp_path):
    from scripts.run_paper_game import prepare
    with pytest.raises(ValueError, match='extracted Forge'):
        prepare(argparse.Namespace(forge_dir=str(tmp_path)))


def test_game_page_and_disconnected_route(tmp_path):
    import urllib.request
    from physical.server import PhysicalTableApp, PhysicalTableServer
    app = PhysicalTableApp(data_dir=tmp_path, allow_network=False)
    app.human_game = HumanGameClient('http://127.0.0.1:1')
    server = PhysicalTableServer(app, port=0, host='127.0.0.1')
    server.start()
    try:
        base = f'http://127.0.0.1:{server._http.server_port}'
        with urllib.request.urlopen(base+'/play') as r:
            page = r.read()
            assert b'Digital Forge Commander' in page
            assert b'physical sync not installed' in page
            assert b'Start 4-player Commander' in page
            assert b'Enable webcam' in page
            assert b"input.id='typedDecisionInput'" in page
            assert b'input.focus()' in page
            assert b'restoreInputFocus' in page
            assert b'/api/pod/start' in page
        with urllib.request.urlopen(base+'/api/game') as r:
            assert json.load(r)['status'] == 'disconnected'
        with urllib.request.urlopen(base+'/api/pod/status') as r:
            assert json.load(r)['status'] == 'idle'
    finally:
        server.stop()


def test_broker_transport_preserves_actions_and_empty_cancel(monkeypatch):
    client = HumanGameClient()
    calls = []
    monkeypatch.setattr(client, 'request', lambda path, body=None: calls.append((path, body)) or {'status':'accepted'})
    assert client.decide({'id':'prompt', 'selected':['b','a'], 'action':'confirm'})['status'] == 'accepted'
    assert client.decide({'id':'optional', 'selected':[], 'action':'cancel'})['status'] == 'accepted'
    assert calls == [
        ('/human/decision', {'id':'prompt', 'selected':['b','a'], 'action':'confirm'}),
        ('/human/decision', {'id':'optional', 'selected':[], 'action':'cancel'}),
    ]
    for body in (
        {'id':'prompt', 'selected':['a','a']},
        {'id':'prompt', 'selected':[1]},
        {'id':'prompt', 'selected':'a'},
        {'id':'prompt', 'selected':['a'], 'action':[]},
    ):
        assert client.decide(body)['status'] == 'rejected'
    assert len(calls) == 2
