import argparse
from collections import Counter
from dataclasses import dataclass
import json
import random
from pathlib import Path

from analyze_matches import parse_log
from bot_version import BOT_VERSION
from food_planner import (
    center_control_score,
    classify_safety,
    edge_distance,
    position_control_penalty,
)
from snake_brain import SnakeBrain
from snake_state import GameState, apply_move, legal_moves, parse_state, step


DIRECTIONS = {'up', 'down', 'left', 'right'}


@dataclass
class SimResult:
    seed: int
    winner: str | None
    turns: int
    score_a: int
    score_b: int
    foods_a: int
    foods_b: int
    no_legal_side: str | None
    edge_moves: int
    center_score_total: int


def render_board(state):
    board = [[' ' for _ in range(state.cols)] for _ in range(state.rows)]
    for r, c in state.food:
        board[r][c] = '*'
    for side in ('A', 'B'):
        body = state.body(side)
        for index, (r, c) in enumerate(reversed(body)):
            board[r][c] = side if index == len(body) - 1 else side.lower()
    return '\n'.join('|' + ''.join(row) + '|' for row in board)


def make_turn_data(state, game_id, turn_index):
    return {
        'game_id': game_id,
        'turn_token': f't_{turn_index}',
        'side': state.side,
        'board': render_board(state),
        'rows': state.rows,
        'cols': state.cols,
        'score_1': state.scores.get('A', 0),
        'score_2': state.scores.get('B', 0),
        'remaining_moves': state.remaining_moves,
    }


def spawn_food(state, rng, food_count):
    occupied = state.occupied()
    food = set(state.food)
    empties = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.cols)
        if (r, c) not in occupied and (r, c) not in food
    ]
    rng.shuffle(empties)
    while len(food) < food_count and empties:
        food.add(empties.pop())
    return GameState(
        rows=state.rows,
        cols=state.cols,
        board=render_board(state),
        side=state.side,
        enemy=state.enemy,
        snakes=state.snakes,
        food=frozenset(food),
        scores=state.scores,
        remaining_moves=state.remaining_moves,
        reliable_tails=state.reliable_tails,
    )


def initial_state(seed, rows=15, cols=15, food_count=3, max_turns=300):
    rng = random.Random(seed)
    middle = rows // 2
    snakes = {
        'A': ((middle, 2), (middle, 1), (middle, 0)),
        'B': ((middle, cols - 3), (middle, cols - 2), (middle, cols - 1)),
    }
    state = GameState(
        rows=rows,
        cols=cols,
        board=(),
        side='A',
        enemy='B',
        snakes=snakes,
        food=frozenset(),
        scores={'A': 0, 'B': 0},
        remaining_moves=max_turns,
        reliable_tails=frozenset({'A', 'B'}),
    )
    return spawn_food(state, rng, food_count)


def winner_for(state, dead_side=None):
    if dead_side == 'A':
        return 'B'
    if dead_side == 'B':
        return 'A'
    score_a = state.scores.get('A', 0)
    score_b = state.scores.get('B', 0)
    if score_a == score_b:
        return None
    return 'A' if score_a > score_b else 'B'


def simulate_game(seed, max_turns=300, rows=15, cols=15, food_count=3):
    rng = random.Random(seed)
    state = initial_state(seed, rows, cols, food_count, max_turns)
    brain = SnakeBrain(enable_debug=False)
    metrics = Counter()
    game_id = f'sim_{seed}'

    for turn_index in range(max_turns):
        side = state.side
        legal = legal_moves(state, side)
        if not legal:
            return SimResult(
                seed=seed,
                winner=winner_for(state, side),
                turns=turn_index,
                score_a=state.scores.get('A', 0),
                score_b=state.scores.get('B', 0),
                foods_a=metrics['foods_A'],
                foods_b=metrics['foods_B'],
                no_legal_side=side,
                edge_moves=metrics['edge_moves'],
                center_score_total=metrics['center_score_total'],
            )

        data = make_turn_data(state, game_id, turn_index)
        direction = brain.choose_move(data)
        direction = brain.safe_direction(data, direction)
        if direction not in legal:
            direction = legal[0]
        if position_control_penalty(state, side, direction) > 0:
            metrics['edge_moves'] += 1

        target = step(state.head(side), direction)
        ate = target in state.food
        state = apply_move(state, direction, side)
        if ate:
            metrics[f'foods_{side}'] += 1
        head = state.head(side)
        metrics['center_score_total'] += center_control_score(state, head)
        brain.commit_move(data, direction)
        state = spawn_food(state, rng, food_count)

    return SimResult(
        seed=seed,
        winner=winner_for(state),
        turns=max_turns,
        score_a=state.scores.get('A', 0),
        score_b=state.scores.get('B', 0),
        foods_a=metrics['foods_A'],
        foods_b=metrics['foods_B'],
        no_legal_side=None,
        edge_moves=metrics['edge_moves'],
        center_score_total=metrics['center_score_total'],
    )


def summarize_simulations(results):
    total_turns = sum(result.turns for result in results)
    games = len(results)
    wins = Counter(result.winner or 'draw' for result in results)
    deaths = Counter(result.no_legal_side for result in results if result.no_legal_side)
    foods_a = sum(result.foods_a for result in results)
    foods_b = sum(result.foods_b for result in results)
    edge_moves = sum(result.edge_moves for result in results)
    center_total = sum(result.center_score_total for result in results)
    return {
        'bot_version': BOT_VERSION,
        'games': games,
        'avg_turns': round(total_turns / games, 2) if games else 0,
        'wins_A': wins['A'],
        'wins_B': wins['B'],
        'draws': wins['draw'],
        'deaths_A': deaths['A'],
        'deaths_B': deaths['B'],
        'foods_A': foods_a,
        'foods_B': foods_b,
        'food_per_100_turns': round((foods_a + foods_b) * 100 / total_turns, 2) if total_turns else 0,
        'edge_moves_per_100_turns': round(edge_moves * 100 / total_turns, 2) if total_turns else 0,
        'avg_center_score': round(center_total / total_turns, 2) if total_turns else 0,
    }


def evaluate_log_decisions(paths):
    brain = SnakeBrain(enable_debug=False)
    metrics = Counter()
    by_reason = Counter()
    for path in paths:
        turns, _, _, _, _ = parse_log(Path(path))
        for index, data in enumerate(turns):
            state = parse_state(data)
            side = state.side
            legal = legal_moves(state, side)
            metrics['positions'] += 1
            if not legal:
                metrics['no_legal_positions'] += 1
                continue

            direction = brain.choose_move(data)
            direction = brain.safe_direction(data, direction)
            debug = brain.decision_debug(data.get('game_id', 'default')) or {}
            by_reason[debug.get('reason', 'unknown')] += 1
            if direction not in DIRECTIONS:
                metrics['invalid_direction'] += 1
                continue
            if direction not in legal:
                metrics['illegal_after_repair'] += 1
                continue

            after = apply_move(state, direction, side)
            safety, _, _, _ = classify_safety(after, side)
            metrics[f'safety_{safety}'] += 1
            if step(state.head(side), direction) in state.food:
                metrics['food_taken_now'] += 1
            if position_control_penalty(state, side, direction) > 0:
                metrics['edge_or_corner_move'] += 1
            brain.commit_move(data, direction)

    metrics['bot_version'] = BOT_VERSION
    metrics['reasons'] = dict(by_reason)
    return metrics


def expand_log_paths(raw_paths):
    paths = []
    for raw_path in raw_paths:
        matches = sorted(Path('.').glob(raw_path)) if any(ch in raw_path for ch in '*?[') else []
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(raw_path))
    return paths


def print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description='Offline lab for the Snake bot.')
    sub = parser.add_subparsers(dest='command', required=True)

    sim = sub.add_parser('simulate', help='run deterministic self-play simulations')
    sim.add_argument('--games', type=int, default=50)
    sim.add_argument('--seed', type=int, default=1)
    sim.add_argument('--turns', type=int, default=300)
    sim.add_argument('--rows', type=int, default=15)
    sim.add_argument('--cols', type=int, default=15)
    sim.add_argument('--food', type=int, default=3)
    sim.add_argument('--json', action='store_true')

    logs = sub.add_parser('logs', help='reevaluate historical log positions with the current bot')
    logs.add_argument('paths', nargs='+')
    logs.add_argument('--json', action='store_true')

    args = parser.parse_args()
    if args.command == 'simulate':
        results = [
            simulate_game(args.seed + offset, args.turns, args.rows, args.cols, args.food)
            for offset in range(args.games)
        ]
        summary = summarize_simulations(results)
        if args.json:
            print_json(summary)
        else:
            print(f"bot_version: {summary['bot_version']}")
            print(f"games: {summary['games']}")
            print(f"avg_turns: {summary['avg_turns']}")
            print(f"wins_A: {summary['wins_A']}  wins_B: {summary['wins_B']}  draws: {summary['draws']}")
            print(f"deaths_A: {summary['deaths_A']}  deaths_B: {summary['deaths_B']}")
            print(f"foods_A: {summary['foods_A']}  foods_B: {summary['foods_B']}")
            print(f"food_per_100_turns: {summary['food_per_100_turns']}")
            print(f"edge_moves_per_100_turns: {summary['edge_moves_per_100_turns']}")
            print(f"avg_center_score: {summary['avg_center_score']}")
    elif args.command == 'logs':
        summary = evaluate_log_decisions(expand_log_paths(args.paths))
        if args.json:
            print_json(summary)
        else:
            for key, value in summary.items():
                print(f'{key}: {value}')


if __name__ == '__main__':
    main()
