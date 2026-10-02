import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / 'viewer'
LIVE_DIR = ROOT / 'games' / 'live'


def read_snapshot(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def game_summary(snapshot):
    data = snapshot.get('turn_data') or {}
    return {
        'game_id': snapshot.get('game_id', ''),
        'status': snapshot.get('status', 'playing'),
        'updated_at': snapshot.get('updated_at', 0),
        'turn_number': snapshot.get('turn_number', 0),
        'bot_version': snapshot.get('bot_version', ''),
        'player_1': data.get('player_1', 'Jugador A'),
        'player_2': data.get('player_2', 'Jugador B'),
        'score_1': data.get('score_1', 0),
        'score_2': data.get('score_2', 0),
        'remaining_moves': data.get('remaining_moves', 0),
        'side': data.get('side', ''),
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/games':
            snapshots = []
            for path in LIVE_DIR.glob('game_*.json') if LIVE_DIR.exists() else ():
                snapshot = read_snapshot(path)
                if snapshot:
                    snapshots.append(game_summary(snapshot))
            snapshots.sort(key=lambda game: game['updated_at'], reverse=True)
            self.send_json({'games': snapshots[:100]})
            return
        if parsed.path.startswith('/api/game/'):
            game_id = unquote(parsed.path.removeprefix('/api/game/'))
            safe_id = ''.join(ch for ch in game_id if ch.isalnum() or ch in '-_')
            if not safe_id or safe_id != game_id:
                self.send_json({'error': 'invalid game id'}, 400)
                return
            snapshot = read_snapshot(LIVE_DIR / f'game_{safe_id}.json')
            if snapshot is None:
                self.send_json({'error': 'game not found'}, 404)
                return
            self.send_json(snapshot)
            return
        if parsed.path == '/api/health':
            self.send_json({'ok': True})
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        if args and str(args[1]) == '200':
            return
        super().log_message(fmt, *args)


def main():
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
