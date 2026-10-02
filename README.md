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

`run.py` uses the competitive Snake strategy in this repository. It supports
variable 12-20 row/column boards, numbered food, permanent multiplier pickups,
the shrinking `#` wall, and the 3-5 copies of every numbered food introduced
in game version 6.

The decision layer also includes a compact pure-Python search engine. It uses
bitboards, iterative deepening, alpha-beta pruning, and a transposition table to
look several alternating turns ahead under a strict per-move time budget. Its
score can refine choices that passed the exact safety checks, but it cannot
promote a move already classified as suicidal or a forced loss.

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

When a match ends, the client writes a **`games/game_<game_id>.log`** file with
everything that happened: each event received (`<`) and action
sent (`>`), as JSON, ending with the `game_over` event. Useful for replaying or
debugging a match. The `games/` directory is git-ignored.

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

## Live match viewer

While `run.py` is playing, it writes the latest state of every match to
`games/live/`. Start the local viewer in another terminal:

```bash
python match_viewer.py
```

Then open `http://127.0.0.1:8765`. The page refreshes every 750 ms and shows all
active/recent matches, scores, multipliers, the current target digit, and the
bot's latest decision. The viewer is local and never receives the bot token.

## Offline bot lab

`bot_lab.py` runs repeatable local checks for the Snake bot without connecting
to the Code Challenge server.

Run deterministic v6 self-play simulations. Board dimensions are randomized
between 12 and 20 for every seed unless `--rows` and `--cols` are supplied:

```bash
python bot_lab.py simulate --games 10 --seed 1 --turns 120
```

Reevaluate real match positions using the current bot version:

```bash
python bot_lab.py logs game_*.log
```

For a faster sampled pass:

```bash
python bot_lab.py logs game_*.log --max-positions 300
```

Useful metrics:

- `food_per_100_turns`: how often the bot eats the correct numbered food.
- `avg_score_A` / `avg_score_B`: real game score, using `+1` for an ordinary
  move; special cells replace that point with their exact reward or penalty
  (`digit * 100 * multiplier`, `+50`, or `-500`).
- `pickups_A` / `pickups_B`: multiplier pickups collected.
- `wrong_digits_A` / `wrong_digits_B`: penalized digits eaten.
- `wall_hits_A` / `wall_hits_B`: penalized attempts to move into `#`.
- `edge_moves_per_100_turns`: how often it goes to border/corner positions.
- `avg_center_score`: whether it is tending toward center control.
- `safety_SAFE` / `safety_ACCEPTABLE_RISK`: immediate safety classification of
  the current version's decisions on logged positions.

## Weight Optimizer

`optimize_weights.py` runs a reproducible random search over the bot's scoring
weights. It refines `weights/active_weights.json` when that file exists (or the
defaults otherwise). Each candidate plays against that baseline and alternates
sides, so the optimizer can compare real game score and score difference.

Quick search:

```bash
python optimize_weights.py --trials 12 --games 3 --turns 80
```

Use saved real games as extra pressure:

```bash
python optimize_weights.py --trials 20 --games 4 --turns 100 --logs game_*.log
```

By default the optimizer samples up to 300 real log positions per candidate. Use
`--max-log-positions 0` to evaluate every logged position.

Save and activate the best weights locally:

```bash
python optimize_weights.py --trials 20 --games 4 --turns 100 --logs game_*.log --activate
```

Refine only the score-lead control strategy without disturbing the other
learned weights:

```bash
python optimize_weights.py --tune-advantage-only --trials 20 --games 4 --turns 150
```

The bot automatically loads `weights/active_weights.json` when present. Delete
that file to return to the default weights in `bot_weights.py`.

In optimizer output:

- `score` is the optimizer's combined training score.
- `game_score` is the candidate's real average game score.
- `diff` is candidate score minus opponent score.
- `w` is candidate wins over candidate games.
- `objective_improvement` is the optimizer score gained over the loaded
  baseline; validated mode requires a positive minimum before activation.

By default, each seed is played twice:

- candidate as A vs default as B
- default as A vs candidate as B

That mirror test makes the comparison less dependent on starting side. Use
`--no-mirror-seeds` only for quick rough experiments.

Validated search mode keeps running batches until it finds a candidate that
beats the thresholds, and only activates weights when the candidate passes:

```bash
python optimize_weights.py --until-improvement --trials 30 --max-batches 6 --games 6 --turns 150 --logs game_*.log --max-log-positions 300 --min-diff 80 --max-edge100 50 --activate
```

If no candidate passes, it still writes `weights/best_weights.json`, but does
not write `weights/active_weights.json`.

## Write your own bot

You don't need this client — any websocket client works. The contract is:

1. Connect to `ws(s)://<server>/ws?token=<your bot token>`.
2. On `challenge`, send `{"action": "accept_challenge", "data": {"challenge_id": "..."}}`.
3. On `your_turn`, read `data` (board / game state, `game_id`, `turn_token`) and
   send your move: `{"action": "move", "data": { ... , "turn_token": "..." }}`.
