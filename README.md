# codechallenge-test-client

A minimal **bot client** for [The Code Challenge](https://codechallenge.net.ar).
It connects to the match server over a websocket using your bot's token,
auto-accepts challenges, and plays. Use it as a starting point (and a smoke
test) for writing your own bot.

## How it works

Your bot authenticates with its **token** (from **My Bots** on the web) and
opens a websocket to the server:

```
wss://server.codechallenge.net.ar/ws?token=<YOUR_BOT_TOKEN>   # production
ws://localhost:5000/ws?token=<YOUR_BOT_TOKEN>                          # local
```

The server then sends events and the bot replies with actions (JSON):

| Event          | The bot does…                                                        |
| -------------- | -------------------------------------------------------------------- |
| `list_users`   | nothing (just who's online)                                          |
| `challenge`    | replies `accept_challenge` with the `challenge_id`                   |
| `your_turn`    | plays a move — replies `move` with the move data + the `turn_token`  |
| `game_over`    | nothing (the match ended)                                            |

> The example move logic in `run.py` plays **Connect 4** (it picks a random
> column). That `process_your_turn` / `process_move` part is exactly where you
> put your own strategy — and where you adapt it to another game's action shape.

## Requirements

- Python 3.9+
- `websockets` (see `requirements.txt`)

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py <YOUR_BOT_TOKEN>
```

Get `<YOUR_BOT_TOKEN>` from **My Bots** in the web app. By default `run.py`
connects to the production server; switch the `uri` in `run.py` to the
`localhost` line to play against a local server.

> `start.sh` / `start_dev.sh` are convenience runners kept out of git because
> they may embed your personal token.

## Tests

`test_run.py` covers the event handling, the move replies and the game log,
using a fake websocket — nothing connects to the network.

```bash
python -m unittest discover -v
```

They also run on GitHub Actions for every push and pull request
(`.github/workflows/tests.yml`), on Python 3.9 and 3.12.

## Game logs

When a match ends, the client writes a **`game_<game_id>.log`** in the working
directory with everything that happened: each event received (`<`) and action
sent (`>`), as JSON, ending with the `game_over` event. Useful for replaying or
debugging a match. These files are git-ignored.

Each new log starts with a local metadata line (`=`) containing the bot version
from `bot_version.py`. Bump `BOT_VERSION` before a new test batch, for example
`1.0`, `1.1`, `1.2`, so match results can be compared by version.

```
= {"event": "bot_version", "version": "1.0", "notes": "..."}
< {"event": "your_turn", "data": {"board": "...", "game_id": "g_9f", "turn_token": "t_01", ...}}
? {"event": "decision_debug", "direction": "right", "reason": "food_plan", ...}
> {"action": "move", "data": {"game_id": "g_9f", "turn_token": "t_01", "col": 3}}
...
< {"event": "game_over", "data": {"board": "...", "game_id": "g_9f", ...}}
```

The `?` lines are local-only decision diagnostics. They include legal moves, the
chosen food plan, candidate move penalties, rival pressure, and why the move was
selected. They are not sent to the server.

## Offline bot lab

`bot_lab.py` runs repeatable local checks for the Snake bot without connecting
to the Code Challenge server.

Run deterministic self-play simulations:

```bash
python bot_lab.py simulate --games 10 --seed 1 --turns 120
```

Reevaluate real match positions using the current bot version:

```bash
python bot_lab.py logs game_*.log
```

Useful metrics:

- `food_per_100_turns`: how often the bot is eating.
- `edge_moves_per_100_turns`: how often it goes to border/corner positions.
- `avg_center_score`: whether it is tending toward center control.
- `safety_SAFE` / `safety_ACCEPTABLE_RISK`: immediate safety classification of
  the current version's decisions on logged positions.

## Write your own bot

You don't need this client — any websocket client works. The contract is:

1. Connect to `ws(s)://<server>/ws?token=<your bot token>`.
2. On `challenge`, send `{"action": "accept_challenge", "data": {"challenge_id": "..."}}`.
3. On `your_turn`, read `data` (board / game state, `game_id`, `turn_token`) and
   send your move: `{"action": "move", "data": { ... , "turn_token": "..." }}`.
