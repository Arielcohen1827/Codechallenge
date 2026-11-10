"""WebSocket entry point, redacted logging and per-match decision dispatch."""

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
from snake_bot import snake_brain as _brain
from snake_bot import snake_state as _state
from snake_bot.snake_brain import SnakeBrain
from snake_bot.snake_state import parse_state
from v7_logging import safe_message, safe_data, verification_record

# Compatibility exports for clients that import the launcher.
final_safe_direction = _brain.final_safe_direction
apply_move = _state.apply_move
legal_moves = _state.legal_moves

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
INCOMING_CHALLENGE_EXPECTED_OPPONENT = (
    os.getenv(
        'INCOMING_CHALLENGE_EXPECTED_OPPONENT',
        '',
    )
    .strip()
    .lower()
)
try:
    DUPLICATE_ACCEPT_COUNT = max(
        1,
        min(3, int(os.getenv('DUPLICATE_ACCEPT_COUNT', '1'))),
    )
except ValueError:
    DUPLICATE_ACCEPT_COUNT = 1
DUPLICATE_ACCEPT_OPPONENT = (
    os.getenv(
        'DUPLICATE_ACCEPT_OPPONENT',
        '',
    )
    .strip()
    .lower()
)
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


def log_verification(
    game_id, data, direction=None, finished=False, debug=None, action_type='move'
):
    """Attach observational reconciliation and an optional sent-action prediction."""
    try:
        record = verification_record(
            data,
            BOT.observed_states.get(str(game_id)),
            previous_verification(game_id),
            direction,
            finished,
            action_type=action_type,
        )
        if finished:
            attach_finished_verification(game_id, record)
        else:
            log_debug(
                game_id,
                {
                    **(debug or {'event': 'decision_debug'}),
                    'verification': record,
                },
            )
    except Exception:
        # Incomplete diagnostics must never affect sending or brain cleanup.
        pass


def previous_verification(game_id):
    for line in reversed(HISTORY.get(game_id, ())):
        if line.startswith(('? ', '< ')):
            record = json.loads(line[2:]).get('verification')
            if record:
                return record
    return None


def attach_finished_verification(game_id, record):
    last_line = HISTORY[game_id][-1]
    event = json.loads(last_line[2:])
    if last_line.startswith('< ') and event.get('event') == 'game_over':
        event['verification'] = record
        HISTORY[game_id][-1] = '< ' + json.dumps(event)


def write_game_log(game_id):
    """Persist the completed match's redacted event history."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"game_{game_id}.log"
        with path.open("w", encoding='utf-8') as f:
            f.write("\n".join(HISTORY.get(game_id, [])) + "\n")
        print_diagnostic(f"saved {path}")
    except OSError as e:
        print_diagnostic(f"could not write game log: {type(e).__name__}")


def write_live_snapshot(
    game_id,
    data,
    status='playing',
    decision=None,
    direction=None,
    action_type='move',
):
    """Publish the latest server board for the local live viewer."""
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = snapshot_id(game_id)
        if not safe_id:
            return
        payload = snapshot_payload(game_id, data, status, decision)
        payload.update(direction=direction, action_type=action_type)
        path = LIVE_DIR / f'game_{safe_id}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload), encoding='utf-8')
        temporary.replace(path)
    except Exception as error:
        print_diagnostic(f"could not write live snapshot: {type(error).__name__}")


def snapshot_id(game_id):
    return ''.join(char for char in str(game_id) if char.isalnum() or char in '-_')


def snapshot_payload(game_id, data, status, decision):
    now = time.time()
    started_at = LIVE_STARTED_AT.setdefault(game_id, now)
    return {
        'game_id': str(game_id),
        'status': status,
        'started_at': started_at,
        'updated_at': now,
        'turn_number': snapshot_turn_number(game_id),
        'food_eaten': visible_food_counts(game_id, data),
        'bot_version': BOT_VERSION,
        'turn_data': safe_data(data),
        'decision': safe_message(decision) if decision else None,
    }


def visible_food_counts(game_id, data):
    """Partial body visibility cannot establish a snake's total growth."""
    if data.get('levels', 1) > 1:
        return {'A': None, 'B': None}
    return snapshot_food_counts(game_id, str(data.get('board', '')))


def snapshot_food_counts(game_id, board):
    previous = LIVE_FOOD_COUNTS.get(game_id, {})
    counts = {
        side: max(previous.get(side, 0), snapshot_growth(board, side)) for side in ('A', 'B')
    }
    LIVE_FOOD_COUNTS[game_id] = counts
    return counts


def snapshot_growth(board, side):
    return max(0, board.count(side) + board.count(side.lower()) - INITIAL_SNAKE_LENGTH)


def snapshot_turn_number(game_id):
    return sum(
        1
        for line in HISTORY.get(game_id, ())
        if line.startswith('< ') and '"event": "your_turn"' in line
    )


async def send(websocket, action, data):
    """Send one protocol action while redacting its diagnostic copy."""
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
    """Authenticate the client and reconnect after recoverable connection failures."""
    if websockets is None:
        raise RuntimeError("websockets is required to connect to the server")
    uri = "wss://server.codechallenge.net.ar/ws?token={}".format(auth_token)
    outgoing_challenge_sent = False
    while True:
        try:
            print_diagnostic('connection to wss://server.codechallenge.net.ar/ws')
            async with websockets.connect(uri) as websocket:
                print_diagnostic('connection READY!')
                outgoing_challenge_sent = await challenge_once(websocket, outgoing_challenge_sent)
                await play(websocket)
        except KeyboardInterrupt:
            print_diagnostic('Exiting...')
            break
        except Exception:
            print_diagnostic('connection error!')
            time.sleep(3)


async def challenge_once(websocket, already_sent):
    if already_sent:
        return True
    return await send_configured_challenge(websocket)


async def on_game_over(websocket, request_data):
    """Store the final observation and release this match's runtime memory."""
    game_id = request_data['data'].get('game_id')
    if game_id:
        try:
            diagnostic_call(log_event, game_id, request_data)
            log_verification(game_id, request_data['data'], finished=True)
            decision = BOT.decision_debug(game_id)
            diagnostic_call(
                write_live_snapshot,
                game_id,
                request_data['data'],
                status='finished',
                decision=decision,
                direction=(decision or {}).get('direction'),
                action_type=(decision or {}).get('action_type', 'move'),
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
    """Apply the configured acceptance policy to an incoming challenge."""
    data = request_data['data']
    challenge_id = data['challenge_id']
    opponent = str(data.get('opponent', 'desconocido'))
    reason = 'tournament' if is_tournament_challenge(data) else 'any opponent'
    print_diagnostic(
        f"incoming challenge accepted by policy: {challenge_id} "
        f"(opponent: {opponent}; reason: {reason})"
    )
    accept_count = challenge_accept_count(opponent.strip().lower())
    for attempt in range(1, accept_count + 1):
        await send(websocket, 'accept_challenge', {'challenge_id': challenge_id})
        print_diagnostic(f"challenge acceptance {attempt}/{accept_count}: {challenge_id}")


def challenge_accept_count(opponent):
    if DUPLICATE_ACCEPT_OPPONENT and opponent != DUPLICATE_ACCEPT_OPPONENT:
        return 1
    if DUPLICATE_ACCEPT_ONCE and DUPLICATE_ACCEPT_USED:
        return 1
    return reserve_duplicate_acceptance()


def reserve_duplicate_acceptance():
    global DUPLICATE_ACCEPT_USED
    if DUPLICATE_ACCEPT_COUNT > 1 and DUPLICATE_ACCEPT_ONCE:
        DUPLICATE_ACCEPT_USED = True
    return DUPLICATE_ACCEPT_COUNT


async def on_your_turn(websocket, request_data):
    """Dispatch a server turn through the production decision path."""
    diagnostic_call(log_event, request_data['data']['game_id'], request_data)
    await process_your_turn(websocket, request_data)


async def on_play_data(websocket, request_data):
    """Receive observations without requesting a new move."""
    raw = request_data.get('data', {})
    game_id = raw.get('game_id')
    if game_id is None:
        return
    diagnostic_call(log_event, game_id, request_data)
    data = merge_action_fields(raw, request_data)
    if not complete_play_data(data):
        return
    state = BOT.observe_state(data)
    log_verification(game_id, data)
    if state is not None:
        diagnostic_call(
            write_live_snapshot,
            game_id,
            data,
            action_type=data.get('action', data.get('action_type')),
        )


def merge_action_fields(raw, request):
    """Complete missing echoed action fields using validated envelope values."""
    data = dict(raw)
    options = {
        'action': ('move', 'shoot'),
        'action_type': ('move', 'shoot'),
        'direction': ('up', 'down', 'left', 'right', 'ascend', 'descend'),
    }
    for key, allowed in options.items():
        value = request.get(key)
        if key not in data and value in allowed:
            data[key] = value
    return data


def complete_play_data(data):
    """Require the fields needed to reconcile a server observation."""
    return (
        isinstance(data.get('board'), str)
        and data.get('side') is not None
        and isinstance(data.get('remaining_moves'), int)
        and complete_play_scores(data)
    )


def complete_play_scores(data):
    return all(
        any(isinstance(data.get(key), (int, float)) for key in keys)
        for keys in (('score_1', 'score_A'), ('score_2', 'score_B'))
    )


HANDLERS = {
    'game_over': on_game_over,
    'challenge': on_challenge,
    'your_turn': on_your_turn,
    'play_data': on_play_data,
}


def print_diagnostic(message):
    try:
        text = str(message)
        encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        print(text.encode(encoding, errors='backslashreplace').decode(encoding))
    except Exception:
        pass


async def play(websocket):
    """Dispatch incoming WebSocket events until the connection ends."""
    while True:
        try:
            request = await websocket.recv()
            request_data = json.loads(request)
            diagnostic_call(
                lambda: print_diagnostic('< ' + json.dumps(safe_message(request_data)))
            )
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
    """Route a turn event to the shared move-or-shoot implementation."""
    await process_move(websocket, request_data)


async def process_move(websocket, request_data):
    """Choose, validate and send an action before recording its unconfirmed prediction."""
    data = request_data['data']
    diagnostic_call(print_diagnostic, data.get('board', ''))
    action = BOT.choose_action(data)
    if action is None:
        return
    action = BOT.safe_action(data, action)
    if action is None:
        return
    action = affordable_action(data, action)
    direction = action['direction']
    action_type = action['action_type']
    debug = BOT.decision_debug(data['game_id'])
    if debug is not None:
        debug = {**debug, 'action_type': action_type, 'direction': direction}
    move = {
        'game_id': data['game_id'],
        'turn_token': data['turn_token'],
        'direction': direction,
    }
    diagnostic_call(
        write_live_snapshot,
        move['game_id'],
        data,
        decision=debug,
        direction=direction,
        action_type=action_type,
    )
    await send(websocket, action_type, move)
    log_verification(move['game_id'], data, direction, debug=debug, action_type=action_type)
    diagnostic_call(
        log_action,
        move['game_id'],
        {'action': action_type, 'action_type': action_type, 'data': move},
    )
    BOT.commit_action(data, action)


def affordable_action(data, action):
    """Protect the network boundary when a custom policy proposes a shot."""
    if action['action_type'] != 'shoot':
        return action
    state = parse_state(data)
    if state.scores[state.side] >= 5000:
        return action
    return {
        'action_type': 'move',
        'direction': BOT.safe_direction(data, action['direction']),
    }


async def process_wall(websocket, request_data):
    """Forward the legacy wall event through the normal action handler."""
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
