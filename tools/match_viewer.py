"""Read-only loopback APIs and assets for live Snake match observations."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / 'viewer'
LIVE_DIR = ROOT / 'games' / 'live'
INITIAL_SNAKE_LENGTH = 3
ALLOWED_BROWSER_ORIGINS = frozenset(
    {
        'https://codechallenge.net.ar',
        'http://codechallenge.net.ar',
        'http://127.0.0.1:8765',
        'http://localhost:8765',
    }
)


def read_snapshot(path):
    """Read one live JSON snapshot and tolerate files being replaced concurrently."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def food_eaten(snapshot, side):
    """Use recorded food counts and avoid inferring full growth from partial cube views."""
    stored = snapshot.get('food_eaten') or {}
    if side in stored:
        return stored[side]
    if int((snapshot.get('turn_data') or {}).get('levels', 1)) > 1:
        return None
    return flat_food_estimate(snapshot, side)


def flat_food_estimate(snapshot, side):
    """Estimate historical growth only when a complete flat board is available."""
    board = str((snapshot.get('turn_data') or {}).get('board', ''))
    return max(0, sum(char in (side, side.lower()) for char in board) - INITIAL_SNAKE_LENGTH)


def game_summary(snapshot):
    """Build the compact match-list schema from a public live snapshot."""
    data = snapshot.get('turn_data') or {}
    return {
        'game_id': snapshot.get('game_id', ''),
        'status': snapshot.get('status', 'playing'),
        'started_at': snapshot.get('started_at', snapshot.get('updated_at', 0)),
        'updated_at': snapshot.get('updated_at', 0),
        'turn_number': snapshot.get('turn_number', 0),
        'bot_version': snapshot.get('bot_version', ''),
        'player_1': data.get('player_1', 'Jugador A'),
        'player_2': data.get('player_2', 'Jugador B'),
        'score_1': data.get('score_1', 0),
        'score_2': data.get('score_2', 0),
        'remaining_moves': data.get('remaining_moves', 0),
        'side': data.get('side', ''),
        'food_eaten_1': food_eaten(snapshot, 'A'),
        'food_eaten_2': food_eaten(snapshot, 'B'),
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve local viewer assets and read-only match APIs without exposing bot credentials."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def end_headers(self):
        """Disable caching and allow only the configured browser origins."""
        self.send_header('Cache-Control', 'no-store')
        origin = self.headers.get('Origin', '')
        if origin in ALLOWED_BROWSER_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
            self.send_header('Vary', 'Origin')
        super().end_headers()

    def do_OPTIONS(self):
        """Answer read-only browser preflight requests."""
        self.send_response(204)
        self.end_headers()

    def send_json(self, payload, status=200):
        """Send a UTF-8 JSON response with explicit status and content length."""
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Route match APIs and delegate static assets to the standard file handler."""
        path = urlparse(self.path).path
        if path == '/api/games':
            return self.send_json({'games': recent_games()})
        if path.startswith('/api/game/'):
            return self.send_game(unquote(path.removeprefix('/api/game/')))
        if path == '/api/health':
            return self.send_json({'ok': True})
        return super().do_GET()

    def send_game(self, game_id):
        """Validate a match identifier and return its public snapshot or a JSON error."""
        if not valid_game_id(game_id):
            return self.send_json({'error': 'invalid game id'}, 400)
        snapshot = read_snapshot(LIVE_DIR / f'game_{game_id}.json')
        if snapshot is None:
            return self.send_json({'error': 'game not found'}, 404)
        return self.send_json(snapshot)

    def log_message(self, fmt, *args):
        """Suppress routine successful requests while retaining other server diagnostics."""
        if args and str(args[1]) == '200':
            return
        super().log_message(fmt, *args)


def valid_game_id(game_id):
    """Restrict snapshot identifiers to safe filename characters."""
    return bool(game_id) and all(char.isalnum() or char in '-_' for char in game_id)


def recent_games():
    """Read valid snapshots and order the bounded match list by activity and status."""
    if not LIVE_DIR.exists():
        return []
    summaries = []
    for path in LIVE_DIR.glob('game_*.json'):
        snapshot = read_snapshot(path)
        if snapshot:
            summaries.append(game_summary(snapshot))
    return sorted(summaries, key=game_sort_key)[:100]


def game_sort_key(game):
    """Keep live games first and use the appropriate start or finish timestamp."""
    playing = game['status'] == 'playing'
    timestamp = game['started_at'] if playing else game['updated_at']
    return game['status'] == 'finished', timestamp


def main():
    """Start the loopback viewer server and release its socket when interrupted."""
    parser = argparse.ArgumentParser(description='Live viewer for CodeChallenge Snake matches')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f'Snake live viewer: http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
