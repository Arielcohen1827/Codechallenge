import asyncio
import json
import os
from pathlib import Path
import sys
import time

try:
    import websockets
except ModuleNotFoundError:
    websockets = None

from snake_bot.bot_version import BOT_VERSION, BOT_VERSION_NOTES
from snake_bot.bot_weights import load_active_weights
from snake_bot.food_planner import choose_food_plan
from snake_bot.snake_brain import SnakeBrain, final_safe_direction
from snake_bot.snake_state import apply_move, legal_moves, parse_state, step
from v7_logging import safe_message, safe_data, verification_record


# A running text log of events received / actions sent per game, written to
# games/game_<game_id>.log when the match ends.
LOG_DIR = Path('games')
LIVE_DIR = LOG_DIR / 'live'
HISTORY = {}
LOGGED_META = set()
LIVE_STARTED_AT = {}
LIVE_FOOD_COUNTS = {}
BOT = SnakeBrain()
INITIAL_SNAKE_LENGTH = 3
ACCEPT_INCOMING_CHALLENGES = os.getenv(
    'ACCEPT_INCOMING_CHALLENGES',
    '1',
).strip().lower() in {'1', 'true', 'yes', 'on'}
ALLOWED_CHALLENGE_OPPONENTS = frozenset(
    name.strip().lower()
    for name in os.getenv(
        'ALLOWED_CHALLENGE_OPPONENTS',
        'arielcohen,Charmander',
    ).split(',')
    if name.strip()
)
OUTGOING_CHALLENGE_TARGET = os.getenv('OUTGOING_CHALLENGE_TARGET', '').strip()
OUTGOING_CHALLENGE_GAME = os.getenv('OUTGOING_CHALLENGE_GAME', 'snake').strip()
INCOMING_CHALLENGE_EXPECTED_OPPONENT = os.getenv(
    'INCOMING_CHALLENGE_EXPECTED_OPPONENT',
    '',
).strip().lower()
try:
    DUPLICATE_ACCEPT_COUNT = max(
        1,
        min(3, int(os.getenv('DUPLICATE_ACCEPT_COUNT', '1'))),
    )
except ValueError:
    DUPLICATE_ACCEPT_COUNT = 1
DUPLICATE_ACCEPT_OPPONENT = os.getenv(
    'DUPLICATE_ACCEPT_OPPONENT',
    '',
).strip().lower()
DUPLICATE_ACCEPT_ONCE = os.getenv(
    'DUPLICATE_ACCEPT_ONCE',
    '0',
).strip().lower() in {'1', 'true', 'yes', 'on'}
DUPLICATE_ACCEPT_USED = False


def log_metadata(game_id):
    if game_id in LOGGED_META:
        return
    HISTORY.setdefault(game_id, []).append(
        '= '
        + json.dumps(
            {
                'event': 'bot_version',
                'version': BOT_VERSION,
                'notes': BOT_VERSION_NOTES,
            }
        )
    )
    LOGGED_META.add(game_id)


def log_event(game_id, message):
    log_metadata(game_id)
    HISTORY.setdefault(game_id, []).append('< ' + json.dumps(safe_message(message)))


def log_action(game_id, message):
    log_metadata(game_id)
    HISTORY.setdefault(game_id, []).append('> ' + json.dumps(safe_message(message)))


def log_debug(game_id, message):
    log_metadata(game_id)
    HISTORY.setdefault(game_id, []).append('? ' + json.dumps(safe_message(message)))


def diagnostic_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception:
        return None


def log_verification(game_id, data, direction=None, finished=False, debug=None):
    try:
        previous = None
        for line in reversed(HISTORY.get(game_id, ())):
            if line.startswith('? ') or line.startswith('< '):
                previous = json.loads(line[2:]).get('verification')
                if previous:
                    break
        record = verification_record(data, BOT.observed_states.get(str(game_id)),
                                     previous, direction, finished)
        if finished:
            # Attach to the existing event to preserve the protocol log order.
            last_line = HISTORY[game_id][-1]
            event = json.loads(last_line[2:])
            if not last_line.startswith('< ') or event.get('event') != 'game_over':
                return
            event['verification'] = record
            HISTORY[game_id][-1] = '< ' + json.dumps(event)
        else:
            log_debug(game_id, {**(debug or {'event': 'decision_debug'}), 'verification': record})
    except Exception:
        # Incomplete diagnostics must never affect sending or brain cleanup.
        pass


def write_game_log(game_id):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"game_{game_id}.log"
        with path.open("w", encoding='utf-8') as f:
            f.write("\n".join(HISTORY.get(game_id, [])) + "\n")
        print_diagnostic(f"saved {path}")
    except OSError as e:
        print_diagnostic(f"could not write game log: {type(e).__name__}")


def write_live_snapshot(game_id, data, status='playing', decision=None, direction=None):
    """Publish the latest server board for the local live viewer."""
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = ''.join(ch for ch in str(game_id) if ch.isalnum() or ch in '-_')
        if not safe_id:
            return
        now = time.time()
        started_at = LIVE_STARTED_AT.setdefault(game_id, now)
        board = str(data.get('board', ''))
        current_food_counts = {
            'A': max(0, sum(char in ('A', 'a') for char in board) - INITIAL_SNAKE_LENGTH),
            'B': max(0, sum(char in ('B', 'b') for char in board) - INITIAL_SNAKE_LENGTH),
        }
        previous_food_counts = LIVE_FOOD_COUNTS.get(game_id, {})
        food_counts = {
            side: max(current_food_counts[side], previous_food_counts.get(side, 0))
            for side in ('A', 'B')
        }
        LIVE_FOOD_COUNTS[game_id] = food_counts
        turn_number = sum(
            1
            for line in HISTORY.get(game_id, ())
            if line.startswith('< ') and '"event": "your_turn"' in line
        )
        payload = {
            'game_id': str(game_id),
            'status': status,
            'started_at': started_at,
            'updated_at': now,
            'turn_number': turn_number,
            'food_eaten': food_counts,
            'bot_version': BOT_VERSION,
            'turn_data': safe_data(data),
            'decision': safe_message(decision) if decision else None,
            'direction': direction,
        }
        path = LIVE_DIR / f'game_{safe_id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload), encoding='utf-8')
        temporary.replace(path)
    except Exception as e:
        print_diagnostic(f"could not write live snapshot: {type(e).__name__}")


async def send(websocket, action, data):
    message = json.dumps(
        {
            'action': action,
            'data': data,
        }
    )
    try:
        print_diagnostic(json.dumps(safe_message({'action': action, 'data': data})))
    except Exception:
        pass
    await websocket.send(message)


async def send_configured_challenge(websocket):
    if not OUTGOING_CHALLENGE_TARGET:
        return False
    await send(
        websocket,
        'challenge',
        {
            'opponent': OUTGOING_CHALLENGE_TARGET,
            'game': OUTGOING_CHALLENGE_GAME,
        },
    )
    print_diagnostic(
        f"one-shot websocket challenge sent: "
        f"{OUTGOING_CHALLENGE_TARGET} ({OUTGOING_CHALLENGE_GAME})"
    )
    return True


async def start(auth_token):
    if websockets is None:
        raise RuntimeError("websockets is required to connect to the server")
    uri = "wss://server.codechallenge.net.ar/ws?token={}".format(auth_token)
    # uri = "ws://localhost:5000/ws?token={}".format(auth_token)
    outgoing_challenge_sent = False
    while True:
        try:
            print_diagnostic('connection to wss://server.codechallenge.net.ar/ws')
            async with websockets.connect(uri) as websocket:
                print_diagnostic('connection READY!')
                if not outgoing_challenge_sent:
                    outgoing_challenge_sent = await send_configured_challenge(websocket)
                await play(websocket)
        except KeyboardInterrupt:
            print_diagnostic('Exiting...')
            break
        except Exception:
            print_diagnostic('connection error!')
            time.sleep(3)


async def on_game_over(websocket, request_data):
    game_id = request_data['data'].get('game_id')
    if game_id:
        try:
            diagnostic_call(log_event, game_id, request_data)
            log_verification(game_id, request_data['data'], finished=True)
            decision = BOT.decision_debug(game_id)
            diagnostic_call(write_live_snapshot,
                game_id,
                request_data['data'],
                status='finished',
                decision=decision,
                direction=(decision or {}).get('direction'),
            )
        finally:
            LIVE_STARTED_AT.pop(game_id, None)
            LIVE_FOOD_COUNTS.pop(game_id, None)
            BOT.forget(game_id)
        diagnostic_call(write_game_log, game_id)
        LOGGED_META.discard(game_id)


def is_tournament_challenge(data):
    if any(
        data.get(field)
        for field in (
            'tournament',
            'tournament_id',
            'championship',
            'championship_id',
        )
    ):
        return True
    return any(
        marker in str(data.get(field, '')).strip().lower()
        for field in ('source', 'mode', 'type', 'kind', 'challenge_type')
        for marker in ('tournament', 'championship', 'torneo', 'campeonato')
    )


async def on_challenge(websocket, request_data):
    global DUPLICATE_ACCEPT_USED

    data = request_data['data']
    challenge_id = data['challenge_id']
    opponent = str(data.get('opponent', 'desconocido'))
    normalized_opponent = opponent.strip().lower()
    if not ACCEPT_INCOMING_CHALLENGES:
        print_diagnostic(
            f"all incoming challenges disabled: {challenge_id} "
            f"(opponent: {opponent})"
        )
        return
    if (
        INCOMING_CHALLENGE_EXPECTED_OPPONENT
        and normalized_opponent != INCOMING_CHALLENGE_EXPECTED_OPPONENT
    ):
        print_diagnostic(
            f"unexpected incoming challenge ignored: {challenge_id} "
            f"(opponent: {opponent})"
        )
        return
    tournament = is_tournament_challenge(data)
    if not tournament and normalized_opponent not in ALLOWED_CHALLENGE_OPPONENTS:
        print_diagnostic(
            f"untrusted incoming challenge ignored: {challenge_id} "
            f"(opponent: {opponent})"
        )
        return
    reason = 'tournament' if tournament else 'allowed opponent'
    print_diagnostic(
        f"incoming challenge accepted by policy: {challenge_id} "
        f"(opponent: {opponent}; reason: {reason})"
    )
    duplicate_matches = (
        not DUPLICATE_ACCEPT_OPPONENT
        or normalized_opponent == DUPLICATE_ACCEPT_OPPONENT
    )
    duplicate_available = not DUPLICATE_ACCEPT_ONCE or not DUPLICATE_ACCEPT_USED
    accept_count = (
        DUPLICATE_ACCEPT_COUNT
        if duplicate_matches and duplicate_available
        else 1
    )
    if accept_count > 1 and DUPLICATE_ACCEPT_ONCE:
        DUPLICATE_ACCEPT_USED = True
    for attempt in range(1, accept_count + 1):
        await send(
            websocket,
            'accept_challenge',
            {
                'challenge_id': challenge_id,
            },
        )
        print_diagnostic(
            f"challenge acceptance {attempt}/{accept_count}: "
            f"{challenge_id}"
        )


async def on_your_turn(websocket, request_data):
    diagnostic_call(log_event, request_data['data']['game_id'], request_data)
    await process_your_turn(websocket, request_data)


HANDLERS = {
    'game_over': on_game_over,
    'challenge': on_challenge,
    'your_turn': on_your_turn,
}


def print_diagnostic(message):
    try:
        text = str(message)
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        print(text.encode(encoding, errors='backslashreplace').decode(encoding))
    except Exception:
        pass


async def play(websocket):
    while True:
        try:
            request = await websocket.recv()
            request_data = json.loads(request)
            diagnostic_call(lambda: print_diagnostic('< ' + json.dumps(safe_message(request_data))))
            handler = HANDLERS.get(request_data['event'])
            if handler:
                await handler(websocket, request_data)
        except KeyboardInterrupt:
            print_diagnostic('Exiting...')
            break
        except Exception as e:
            diagnostic_call(print_diagnostic, 'error {}'.format(type(e).__name__))
            break  # force login again


async def process_your_turn(websocket, request_data):
    await process_move(websocket, request_data)


async def process_move(websocket, request_data):
    data = request_data['data']
    diagnostic_call(print_diagnostic, data.get('board', ''))
    direction = BOT.choose_move(data)
    if direction is None:
        return
    direction = BOT.safe_direction(data, direction)
    debug = BOT.decision_debug(data['game_id'])
    move = {
        'game_id': data['game_id'],
        'turn_token': data['turn_token'],
        'direction': direction,
    }
    diagnostic_call(write_live_snapshot,
        move['game_id'],
        data,
        decision=debug,
        direction=direction,
    )
    await send(websocket, 'move', move)
    log_verification(move['game_id'], data, direction, debug=debug)
    diagnostic_call(log_action, move['game_id'], {'action': 'move', 'data': move})
    BOT.commit_move(data, direction)


async def process_wall(websocket, request_data):
    await send(
        websocket,
        'wall',
        {
            'game_id': request_data['data']['game_id'],
            'turn_token': request_data['data']['turn_token'],
            'row': 0,
            'col': 0,
            'orientation': 'h',
        },
    )


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        load_active_weights()
        auth_token = sys.argv[1]
        asyncio.get_event_loop().run_until_complete(start(auth_token))
    else:
        print_diagnostic('please provide your auth_token')
