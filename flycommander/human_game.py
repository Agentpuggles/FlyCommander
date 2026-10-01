"""Same-origin gateway for experimental Forge human prompts. No offline move queue."""
import json
import os
import urllib.error
import urllib.request


class HumanGameClient:
    def __init__(self, url=None):
        self.url = (url or os.environ.get('FLYCOMMANDER_AGENT_URL', 'http://127.0.0.1:8791')).rstrip('/')

    def request(self, path, body=None):
        req = urllib.request.Request(self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=2) as response:
                result = json.loads(response.read(2_000_000))
            if not isinstance(result, dict):
                raise ValueError('Invalid agent response')
            return result
        except urllib.error.HTTPError as exc:
            return {'status': 'rejected', 'error': f'Agent rejected request (HTTP {exc.code}). Refresh before trying again.'}
        except (OSError, ValueError):
            return {'status': 'disconnected', 'error': 'No human-game agent connected. Scanner only; no game action was queued.'}

    def state(self):
        return self.request('/human/state')

    def decide(self, body):
        if isinstance(body, dict) and 'selected' in body:
            selected = body['selected']
            if (not isinstance(body.get('id'), str) or not 0 < len(body['id']) <= 256
                    or not isinstance(selected, list) or len(selected) > 600
                    or not all(isinstance(v, str) and 0 < len(v) <= 256 for v in selected)
                    or len(set(selected)) != len(selected)):
                return {'status': 'rejected', 'error': 'Invalid decision selection.'}
            return self.request('/human/decision', {'id': body['id'], 'selected': selected})
        if not isinstance(body, dict) or not all(isinstance(body.get(k), str) and 0 < len(body[k]) <= 256 for k in ('id', 'choice')):
            return {'status': 'rejected', 'error': 'A prompt ID and choice are required.'}
        return self.request('/human/decision', {'id': body['id'], 'choice': body['choice']})
