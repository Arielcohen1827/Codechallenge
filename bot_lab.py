import argparse
from collections import Counter
from dataclasses import dataclass
import json
import random
from pathlib import Path

from analyze_matches import parse_log
from bot_version import BOT_VERSION
from bot_weights import load_active_weights, set_active_weights
from food_planner import (
    center_control_score,
    classify_safety,
    edge_distance,
    position_control_penalty,
)
from snake_brain import SnakeBrain
from snake_state import (
    CRASH_PENALTY,
    RIVAL_CRASH_REWARD,
    GameState,
    apply_move,
    cyclic_digit,
    legal_moves,
    parse_state,
    step,
)


DIRECTIONS = {'up', 'down', 'left', 'right'}


@dataclass
class SimResult:
    seed: int
    candidate_side: str | None
    winner: str | None
    turns: int
    score_a: int
    score_b: int
    foods_a: int
    foods_b: int
    pickups_a: int
    pickups_b: int
    wrong_digits_a: int
    wrong_digits_b: int
    wall_hits_a: int
    wall_hits_b: int
    no_legal_side: str | None
    edge_moves: int
    center_score_total: int


def render_board(state):
    board = [[' ' for _ in range(state.cols)] for _ in range(state.rows)]
    for r, c in state.walls:
        board[r][c] = '#'
    if state.food_values:
        for (r, c), digit in state.food_values.items():
            board[r][c] = str(digit)
    else:
        for r, c in state.food:
            board[r][c] = '*'
    for r, c in state.pickups:
        board[r][c] = 'X'
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
        'board_size': f'{state.rows}x{state.cols}',
        'score_1': state.scores.get('A', 0),
        'score_2': state.scores.get('B', 0),
        'remaining_moves': state.remaining_moves,
        'multiplier_1': state.multipliers.get('A', 1),
        'multiplier_2': state.multipliers.get('B', 1),
    }


def spawn_food(state, rng, food_count, numbered=True, pickup_count=2):
    occupied = state.occupied()
    food = set(state.food)
    food_values = dict(state.food_values)
    pickups = set(state.pickups)
    walls = set(state.walls)
    empties = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.cols)
        if (r, c) not in occupied and (r, c) not in food_values and (r, c) not in food and (r, c) not in pickups
    ]
    rng.shuffle(empties)
    next_digit = state.next_food_digit or 1
    if numbered:
        desired = [cyclic_digit(next_digit, offset) for offset in range(food_count)]
        existing = set(food_values.values())
        for digit in desired:
            if digit not in existing and empties:
                food_values[empties.pop()] = digit
                existing.add(digit)
        food = {pos for pos, digit in food_values.items() if digit == next_digit}
    else:
        while len(food) < food_count and empties:
            food.add(empties.pop())
    while len(pickups) < pickup_count and empties:
        pickups.add(empties.pop())
    if not walls:
        walls = set(spawn_wall(empties, state.rows, state.cols, rng))
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
        food_values=food_values,
        next_food_digit=next_digit if numbered else None,
        pickups=frozenset(pickups),
        multipliers=state.multipliers,
        walls=frozenset(walls),
    )


def spawn_wall(empties, rows, cols, rng):
    available = set(empties)
    candidates = []
    max_length = min(11, max(rows, cols))
    lengths = [length for length in range(3, max_length + 1, 2)]
    for length in lengths:
        if length <= cols:
            for row in range(rows):
                for start in range(cols - length + 1):
                    cells = tuple((row, start + offset) for offset in range(length))
                    if all(cell in available for cell in cells):
                        candidates.append(cells)
        if length <= rows:
            for col in range(cols):
                for start in range(rows - length + 1):
                    cells = tuple((start + offset, col) for offset in range(length))
                    if all(cell in available for cell in cells):
                        candidates.append(cells)
    return frozenset(rng.choice(candidates)) if candidates else frozenset()


def initial_state(seed, rows=15, cols=15, food_count=5, max_turns=300, numbered=True):
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
        next_food_digit=1 if numbered else None,
        multipliers={'A': 1, 'B': 1},
    )
    return spawn_food(state, rng, food_count, numbered=numbered)


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


def simulate_game(seed, max_turns=300, rows=None, cols=None, food_count=5, weights_by_side=None, candidate_side=None):
    rng = random.Random(seed)
    rows = rows or rng.randint(12, 20)
    cols = cols or rng.randint(12, 20)
    state = initial_state(seed, rows, cols, food_count, max_turns)
    brains = {'A': SnakeBrain(enable_debug=False), 'B': SnakeBrain(enable_debug=False)}
    metrics = Counter()
    game_id = f'sim_{seed}'

    for turn_index in range(max_turns):
        side = state.side
        if weights_by_side and side in weights_by_side:
            set_active_weights(weights_by_side[side])
        legal = legal_moves(state, side)
        if not legal:
            enemy = state.enemy
            final_scores = dict(state.scores)
            final_scores[side] = final_scores.get(side, 0) + CRASH_PENALTY
            final_scores[enemy] = final_scores.get(enemy, 0) + RIVAL_CRASH_REWARD
            if final_scores.get('A', 0) == final_scores.get('B', 0):
                winner = None
            else:
                winner = 'A' if final_scores.get('A', 0) > final_scores.get('B', 0) else 'B'
            return SimResult(
                seed=seed,
                candidate_side=candidate_side,
                winner=winner,
                turns=turn_index,
                score_a=final_scores.get('A', 0),
                score_b=final_scores.get('B', 0),
                foods_a=metrics['foods_A'],
                foods_b=metrics['foods_B'],
                pickups_a=metrics['pickups_A'],
                pickups_b=metrics['pickups_B'],
                wrong_digits_a=metrics['wrong_digits_A'],
                wrong_digits_b=metrics['wrong_digits_B'],
                wall_hits_a=metrics['wall_hits_A'],
                wall_hits_b=metrics['wall_hits_B'],
                no_legal_side=side,
                edge_moves=metrics['edge_moves'],
                center_score_total=metrics['center_score_total'],
            )

        data = make_turn_data(state, game_id, turn_index)
        brain = brains[side]
        direction = brain.choose_move(data)
        direction = brain.safe_direction(data, direction)
        if direction not in legal:
            direction = legal[0]
        if position_control_penalty(state, side, direction) > 0:
            metrics['edge_moves'] += 1

        target = step(state.head(side), direction)
        ate = target in state.food
        took_pickup = target in state.pickups
        ate_wrong_digit = target in state.wrong_food()
        hit_wall = target in state.walls
        state = apply_move(state, direction, side)
        if ate:
            metrics[f'foods_{side}'] += 1
        if took_pickup:
            metrics[f'pickups_{side}'] += 1
        if ate_wrong_digit:
            metrics[f'wrong_digits_{side}'] += 1
        if hit_wall:
            metrics[f'wall_hits_{side}'] += 1
        head = state.head(side)
        metrics['center_score_total'] += center_control_score(state, head)
        brain.commit_move(data, direction)
        state = spawn_food(state, rng, food_count)

    return SimResult(
        seed=seed,
        candidate_side=candidate_side,
        winner=winner_for(state),
        turns=max_turns,
        score_a=state.scores.get('A', 0),
        score_b=state.scores.get('B', 0),
        foods_a=metrics['foods_A'],
        foods_b=metrics['foods_B'],
        pickups_a=metrics['pickups_A'],
        pickups_b=metrics['pickups_B'],
        wrong_digits_a=metrics['wrong_digits_A'],
        wrong_digits_b=metrics['wrong_digits_B'],
        wall_hits_a=metrics['wall_hits_A'],
        wall_hits_b=metrics['wall_hits_B'],
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
    pickups_a = sum(result.pickups_a for result in results)
    pickups_b = sum(result.pickups_b for result in results)
    wrong_digits_a = sum(result.wrong_digits_a for result in results)
    wrong_digits_b = sum(result.wrong_digits_b for result in results)
    wall_hits_a = sum(result.wall_hits_a for result in results)
    wall_hits_b = sum(result.wall_hits_b for result in results)
    score_a = sum(result.score_a for result in results)
    score_b = sum(result.score_b for result in results)
    edge_moves = sum(result.edge_moves for result in results)
    center_total = sum(result.center_score_total for result in results)
    candidate_results = [result for result in results if result.candidate_side in ('A', 'B')]
    candidate_score = sum(result.score_a if result.candidate_side == 'A' else result.score_b for result in candidate_results)
    opponent_score = sum(result.score_b if result.candidate_side == 'A' else result.score_a for result in candidate_results)
    candidate_foods = sum(result.foods_a if result.candidate_side == 'A' else result.foods_b for result in candidate_results)
    opponent_foods = sum(result.foods_b if result.candidate_side == 'A' else result.foods_a for result in candidate_results)
    candidate_wall_hits = sum(
        result.wall_hits_a if result.candidate_side == 'A' else result.wall_hits_b
        for result in candidate_results
    )
    opponent_wall_hits = sum(
        result.wall_hits_b if result.candidate_side == 'A' else result.wall_hits_a
        for result in candidate_results
    )
    candidate_wins = sum(1 for result in candidate_results if result.winner == result.candidate_side)
    candidate_losses = sum(
        1
        for result in candidate_results
        if result.winner in ('A', 'B') and result.winner != result.candidate_side
    )
    candidate_deaths = sum(1 for result in candidate_results if result.no_legal_side == result.candidate_side)
    opponent_deaths = sum(
        1
        for result in candidate_results
        if result.no_legal_side in ('A', 'B') and result.no_legal_side != result.candidate_side
    )
    return {
        'bot_version': BOT_VERSION,
        'games': games,
        'avg_turns': round(total_turns / games, 2) if games else 0,
        'avg_score_A': round(score_a / games, 2) if games else 0,
        'avg_score_B': round(score_b / games, 2) if games else 0,
        'avg_total_game_score': round((score_a + score_b) / games, 2) if games else 0,
        'avg_score_diff_A_minus_B': round((score_a - score_b) / games, 2) if games else 0,
        'wins_A': wins['A'],
        'wins_B': wins['B'],
        'draws': wins['draw'],
        'deaths_A': deaths['A'],
        'deaths_B': deaths['B'],
        'foods_A': foods_a,
        'foods_B': foods_b,
        'pickups_A': pickups_a,
        'pickups_B': pickups_b,
        'wrong_digits_A': wrong_digits_a,
        'wrong_digits_B': wrong_digits_b,
        'wall_hits_A': wall_hits_a,
        'wall_hits_B': wall_hits_b,
        'food_per_100_turns': round((foods_a + foods_b) * 100 / total_turns, 2) if total_turns else 0,
        'edge_moves_per_100_turns': round(edge_moves * 100 / total_turns, 2) if total_turns else 0,
        'avg_center_score': round(center_total / total_turns, 2) if total_turns else 0,
        'candidate_games': len(candidate_results),
        'candidate_wins': candidate_wins,
        'candidate_losses': candidate_losses,
        'candidate_deaths': candidate_deaths,
        'opponent_deaths': opponent_deaths,
        'avg_candidate_score': round(candidate_score / len(candidate_results), 2) if candidate_results else 0,
        'avg_opponent_score': round(opponent_score / len(candidate_results), 2) if candidate_results else 0,
        'avg_candidate_score_diff': round((candidate_score - opponent_score) / len(candidate_results), 2)
        if candidate_results
        else 0,
        'candidate_foods': candidate_foods,
        'opponent_foods': opponent_foods,
        'candidate_wall_hits': candidate_wall_hits,
        'opponent_wall_hits': opponent_wall_hits,
    }


def iter_log_entries(paths):
    for path in paths:
        turns, _, _, _, _ = parse_log(Path(path))
        for index, data in enumerate(turns):
            yield path, index, data


def select_log_entries(paths, max_positions=None):
    entries = list(iter_log_entries(paths))
    if not max_positions or max_positions >= len(entries):
        return entries, False
    stride = max(1, len(entries) // max_positions)
    selected = entries[::stride][:max_positions]
    return selected, True


def evaluate_log_decisions(paths, max_positions=None):
    entries, sampled = select_log_entries(paths, max_positions)
    sequential_brain = SnakeBrain(enable_debug=False)
    metrics = Counter()
    by_reason = Counter()
    for _, _, data in entries:
        brain = SnakeBrain(enable_debug=False) if sampled else sequential_brain
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
        if step(state.head(side), direction) in state.walls:
            metrics['wall_hit'] += 1
        if position_control_penalty(state, side, direction) > 0:
            metrics['edge_or_corner_move'] += 1
        if not sampled:
            brain.commit_move(data, direction)

    metrics['bot_version'] = BOT_VERSION
    metrics['sampled_positions'] = sampled
    metrics['reasons'] = dict(by_reason)
    return metrics


def expand_log_paths(raw_paths):
    paths = []
    for raw_path in raw_paths:
        matches = sorted(Path('.').glob(raw_path)) if any(ch in raw_path for ch in '*?[') else []
        if not matches and any(ch in raw_path for ch in '*?[') and Path(raw_path).parent == Path('.'):
            matches = sorted(Path('games').glob(raw_path))
        if matches:
            paths.extend(matches)
        elif Path(raw_path).exists():
            paths.append(Path(raw_path))
        elif Path('games', raw_path).exists():
            paths.append(Path('games', raw_path))
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
    sim.add_argument('--rows', type=int, default=0, help='0 chooses a random size from 12 to 20')
    sim.add_argument('--cols', type=int, default=0, help='0 chooses a random size from 12 to 20')
    sim.add_argument('--food', type=int, default=5)
    sim.add_argument('--json', action='store_true')

    logs = sub.add_parser('logs', help='reevaluate historical log positions with the current bot')
    logs.add_argument('paths', nargs='+')
    logs.add_argument('--max-positions', type=int, default=0)
    logs.add_argument('--json', action='store_true')

    args = parser.parse_args()
    load_active_weights()
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
            print(f"avg_score_A: {summary['avg_score_A']}  avg_score_B: {summary['avg_score_B']}")
            print(f"wins_A: {summary['wins_A']}  wins_B: {summary['wins_B']}  draws: {summary['draws']}")
            print(f"deaths_A: {summary['deaths_A']}  deaths_B: {summary['deaths_B']}")
            print(f"foods_A: {summary['foods_A']}  foods_B: {summary['foods_B']}")
            print(f"pickups_A: {summary['pickups_A']}  pickups_B: {summary['pickups_B']}")
            print(f"wrong_digits_A: {summary['wrong_digits_A']}  wrong_digits_B: {summary['wrong_digits_B']}")
            print(f"wall_hits_A: {summary['wall_hits_A']}  wall_hits_B: {summary['wall_hits_B']}")
            print(f"food_per_100_turns: {summary['food_per_100_turns']}")
            print(f"edge_moves_per_100_turns: {summary['edge_moves_per_100_turns']}")
            print(f"avg_center_score: {summary['avg_center_score']}")
    elif args.command == 'logs':
        summary = evaluate_log_decisions(expand_log_paths(args.paths), args.max_positions or None)
        if args.json:
            print_json(summary)
        else:
            for key, value in summary.items():
                print(f'{key}: {value}')


if __name__ == '__main__':
    main()
