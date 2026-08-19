import argparse
import json
import random
import time
from pathlib import Path

from bot_lab import evaluate_log_decisions, expand_log_paths, simulate_game, summarize_simulations
from bot_weights import DEFAULT_WEIGHTS, reset_active_weights, save_weights, set_active_weights


TUNABLE_RANGES = {
    'food_distance_cost': (350, 900),
    'race_margin_value': (300, 1200),
    'hunger_value': (80, 420),
    'space_after_value': (6, 35),
    'exits_after_value': (80, 700),
    'tail_reachable_bonus': (0, 1400),
    'center_unit_value': (20, 180),
    'edge_depth_value': (80, 1000),
    'center_edge0_penalty': (800, 8000),
    'center_edge1_penalty': (0, 3500),
    'next_food_value': (40, 450),
    'current_target_bonus': (0, 3200),
    'safe_bonus': (700, 4000),
    'acceptable_risk_bonus': (-500, 1800),
    'dangerous_penalty': (2500, 16_000),
    'lost_race_penalty': (6000, 40_000),
    'position_edge0_penalty': (1500, 14_000),
    'position_edge1_penalty': (0, 5000),
    'position_edge2_penalty': (0, 2200),
    'position_corner_penalty': (0, 9000),
    'zero_escape_penalty': (5000, 20_000),
    'delayed_zero_penalty': (9000, 40_000),
    'one_escape_penalty': (800, 7000),
    'small_region_penalty': (600, 3500),
    'one_exit_penalty': (1000, 9000),
    'territory_gap_penalty': (80, 600),
    'hot_lost_food_base': (500, 6000),
    'hot_lost_food_enemy_bonus': (300, 3000),
    'fallback_progress_value': (300, 2200),
    'fallback_hunger_progress_value': (20, 180),
    'fallback_food_distance_cost': (20, 220),
    'fallback_cycle_penalty': (100, 1500),
    'fallback_danger_penalty': (3000, 20_000),
}


def random_candidate(rng):
    weights = dict(DEFAULT_WEIGHTS)
    for key, (low, high) in TUNABLE_RANGES.items():
        weights[key] = rng.randint(low, high)
    return weights


def mutate_candidate(base, rng, strength):
    weights = dict(base)
    for key, (low, high) in TUNABLE_RANGES.items():
        if rng.random() > 0.65:
            continue
        value = weights[key]
        factor = rng.uniform(1 - strength, 1 + strength)
        weights[key] = max(low, min(high, int(value * factor)))
    return weights


def simulation_score(summary):
    candidate_games = max(1, summary.get('candidate_games', 0))
    win_rate = summary.get('candidate_wins', 0) / candidate_games
    loss_rate = summary.get('candidate_losses', 0) / candidate_games
    return (
        summary['avg_candidate_score'] * 3.0
        + summary['avg_candidate_score_diff'] * 5.5
        + win_rate * 1400
        - loss_rate * 900
        + summary['avg_turns'] * 12
        + summary['food_per_100_turns'] * 80
        + summary['avg_center_score'] * 0.12
        - summary['edge_moves_per_100_turns'] * 45
        - summary.get('candidate_deaths', 0) * 2500
        + summary.get('opponent_deaths', 0) * 700
    )


def log_score(summary):
    positions = max(1, summary.get('positions', 0))
    unsafe = summary.get('safety_DANGEROUS', 0) + summary.get('safety_SUICIDAL', 0)
    edge_rate = summary.get('edge_or_corner_move', 0) * 100 / positions
    food_rate = summary.get('food_taken_now', 0) * 100 / positions
    return food_rate * 90 - edge_rate * 18 - unsafe * 1200


def evaluate_candidate(weights, args, log_paths):
    results = []
    for offset in range(args.games):
        candidate_side = 'A' if offset % 2 == 0 else 'B'
        weights_by_side = {
            candidate_side: weights,
            'B' if candidate_side == 'A' else 'A': DEFAULT_WEIGHTS,
        }
        results.append(
            simulate_game(
                args.seed + offset,
                args.turns,
                args.rows,
                args.cols,
                args.food,
                weights_by_side=weights_by_side,
                candidate_side=candidate_side,
            )
        )
    sim_summary = summarize_simulations(results)
    score = simulation_score(sim_summary)
    log_summary = None
    if log_paths:
        set_active_weights(weights)
        log_summary = evaluate_log_decisions(log_paths, args.max_log_positions or None)
        score += log_score(log_summary)
    return score, sim_summary, log_summary


def print_candidate(rank, score, sim_summary, log_summary=None):
    print(
        f"#{rank} score={round(score, 2)} "
        f"game_score={sim_summary['avg_candidate_score']} "
        f"diff={sim_summary['avg_candidate_score_diff']} "
        f"w={sim_summary['candidate_wins']}/{sim_summary['candidate_games']} "
        f"turns={sim_summary['avg_turns']} "
        f"food100={sim_summary['food_per_100_turns']} "
        f"edge100={sim_summary['edge_moves_per_100_turns']} "
        f"center={sim_summary['avg_center_score']} "
        f"deaths={sim_summary['deaths_A'] + sim_summary['deaths_B']}"
    )
    if log_summary:
        print(
            f"   logs positions={log_summary.get('positions', 0)} "
            f"food_now={log_summary.get('food_taken_now', 0)} "
            f"edge={log_summary.get('edge_or_corner_move', 0)} "
            f"danger={log_summary.get('safety_DANGEROUS', 0)} "
            f"suicidal={log_summary.get('safety_SUICIDAL', 0)}"
        )


def main():
    parser = argparse.ArgumentParser(description='Random-search optimizer for Snake bot weights.')
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--turns', type=int, default=80)
    parser.add_argument('--rows', type=int, default=15)
    parser.add_argument('--cols', type=int, default=15)
    parser.add_argument('--food', type=int, default=3)
    parser.add_argument('--logs', nargs='*', default=())
    parser.add_argument('--max-log-positions', type=int, default=300)
    parser.add_argument('--output', default='weights/best_weights.json')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    log_paths = expand_log_paths(args.logs) if args.logs else []
    best = None
    candidates = [dict(DEFAULT_WEIGHTS)]
    while len(candidates) < args.trials:
        if best and rng.random() < 0.45:
            candidates.append(mutate_candidate(best['weights'], rng, strength=0.35))
        else:
            candidates.append(random_candidate(rng))

    started = time.time()
    for index, weights in enumerate(candidates, start=1):
        score, sim_summary, log_summary = evaluate_candidate(weights, args, log_paths)
        record = {
            'score': score,
            'weights': dict(weights),
            'simulation': sim_summary,
            'logs': log_summary,
        }
        if best is None or score > best['score']:
            best = record
        if not args.json:
            print_candidate(index, score, sim_summary, log_summary)

    save_weights(args.output, best['weights'])
    if args.activate:
        save_weights('weights/active_weights.json', best['weights'])
    reset_active_weights()

    result = {
        'best_score': round(best['score'], 2),
        'best_weights_path': args.output,
        'active_weights_path': 'weights/active_weights.json' if args.activate else None,
        'elapsed_seconds': round(time.time() - started, 2),
        'simulation': best['simulation'],
        'logs': best['logs'],
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print('best:')
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
