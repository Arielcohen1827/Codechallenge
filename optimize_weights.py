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
    'chain_food_value': (80, 650),
    'chain_race_value': (40, 450),
    'chain_close_food_bonus': (0, 2200),
    'hunger_close_food_value': (20, 220),
    'denial_food_bonus': (300, 3600),
    'denial_chain_bonus': (100, 1800),
    'rival_food_threat_value': (100, 1000),
    'rival_chain_threat_value': (60, 700),
    'rival_chain_ignore_penalty': (400, 4000),
    'current_target_bonus': (0, 3200),
    'safe_bonus': (700, 4000),
    'acceptable_risk_bonus': (-500, 1800),
    'dangerous_penalty': (2500, 16_000),
    'lost_race_penalty': (6000, 40_000),
    'non_immediate_edge_food_penalty': (1500, 14_000),
    'low_space_food_penalty': (6000, 28_000),
    'long_low_space_food_penalty': (3000, 24_000),
    'one_exit_food_penalty': (3000, 18_000),
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
    'fallback_hunger_food_progress': (40, 320),
    'fallback_hunger_food_distance': (10, 160),
    'fallback_hunger_food_race': (20, 240),
    'fallback_denial_progress': (30, 450),
    'fallback_denial_reach_bonus': (100, 1800),
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
        sides = ('A', 'B') if args.mirror_seeds else ('A' if offset % 2 == 0 else 'B',)
        for candidate_side in sides:
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


def is_valid_improvement(record, args):
    sim = record['simulation']
    logs = record.get('logs') or {}
    unsafe_logs = logs.get('safety_DANGEROUS', 0) + logs.get('safety_SUICIDAL', 0)
    if sim['avg_candidate_score_diff'] < args.min_diff:
        return False
    if args.require_winning_record and sim['candidate_wins'] <= sim['candidate_losses']:
        return False
    if sim['candidate_deaths'] > args.max_candidate_deaths:
        return False
    if sim['edge_moves_per_100_turns'] > args.max_edge100:
        return False
    if sim['food_per_100_turns'] < args.min_food100:
        return False
    if unsafe_logs > args.max_log_unsafe:
        return False
    return True


def print_candidate(rank, score, sim_summary, log_summary=None):
    losses = sim_summary.get('candidate_losses', 0)
    print(
        f"#{rank} score={round(score, 2)} "
        f"game_score={sim_summary['avg_candidate_score']} "
        f"diff={sim_summary['avg_candidate_score_diff']} "
        f"w={sim_summary['candidate_wins']}-{losses}/{sim_summary['candidate_games']} "
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


def make_candidate(index, best, rng):
    if index == 1:
        return dict(DEFAULT_WEIGHTS)
    if best and rng.random() < 0.55:
        return mutate_candidate(best['weights'], rng, strength=0.30)
    return random_candidate(rng)


def run_batch(args, log_paths, rng, batch_index=1, previous_best=None):
    best = previous_best
    batch_best = None
    accepted = None
    for index in range(1, args.trials + 1):
        weights = make_candidate(index, best, rng)
        score, sim_summary, log_summary = evaluate_candidate(weights, args, log_paths)
        record = {
            'score': score,
            'weights': dict(weights),
            'simulation': sim_summary,
            'logs': log_summary,
            'batch': batch_index,
            'trial': index,
            'valid_improvement': False,
        }
        record['valid_improvement'] = is_valid_improvement(record, args)
        if best is None or score > best['score']:
            best = record
        if batch_best is None or score > batch_best['score']:
            batch_best = record
        if accepted is None and record['valid_improvement']:
            accepted = record
        if not args.json:
            prefix = f"batch={batch_index} " if args.until_improvement else ""
            print(prefix, end="")
            print_candidate(index, score, sim_summary, log_summary)
            if record['valid_improvement']:
                print(
                    f"   accepted diff={sim_summary['avg_candidate_score_diff']} "
                    f"edge100={sim_summary['edge_moves_per_100_turns']} "
                    f"food100={sim_summary['food_per_100_turns']}"
                )
    return best, batch_best, accepted


def main():
    parser = argparse.ArgumentParser(description='Random-search optimizer for Snake bot weights.')
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--turns', type=int, default=80)
    parser.add_argument('--rows', type=int, default=0, help='0 randomizes the dimension from 12 to 20')
    parser.add_argument('--cols', type=int, default=0, help='0 randomizes the dimension from 12 to 20')
    parser.add_argument('--food', type=int, default=5)
    parser.add_argument('--logs', nargs='*', default=())
    parser.add_argument('--max-log-positions', type=int, default=300)
    parser.add_argument('--output', default='weights/best_weights.json')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--no-mirror-seeds', action='store_false', dest='mirror_seeds')
    parser.set_defaults(mirror_seeds=True)
    parser.add_argument('--until-improvement', action='store_true')
    parser.add_argument('--max-batches', type=int, default=8)
    parser.add_argument('--min-diff', type=float, default=80)
    parser.add_argument('--max-edge100', type=float, default=50)
    parser.add_argument('--min-food100', type=float, default=13)
    parser.add_argument('--max-candidate-deaths', type=int, default=0)
    parser.add_argument('--max-log-unsafe', type=int, default=0)
    parser.add_argument('--allow-losing-record', action='store_false', dest='require_winning_record')
    parser.set_defaults(require_winning_record=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    log_paths = expand_log_paths(args.logs) if args.logs else []
    best = None
    accepted = None

    started = time.time()
    batches = args.max_batches if args.until_improvement else 1
    for batch_index in range(1, batches + 1):
        if args.until_improvement and not args.json:
            print(f"batch {batch_index}/{batches}")
        best, _, accepted = run_batch(args, log_paths, rng, batch_index, best)
        if accepted:
            best = accepted
            break

    save_weights(args.output, best['weights'])
    active_path = None
    if args.activate and (not args.until_improvement or accepted):
        save_weights('weights/active_weights.json', best['weights'])
        active_path = 'weights/active_weights.json'
    reset_active_weights()

    result = {
        'best_score': round(best['score'], 2),
        'best_weights_path': args.output,
        'active_weights_path': active_path,
        'accepted_improvement': bool(accepted),
        'mirror_seeds': args.mirror_seeds,
        'elapsed_seconds': round(time.time() - started, 2),
        'simulation': best['simulation'],
        'logs': best['logs'],
    }
    if args.until_improvement:
        result['acceptance_thresholds'] = {
            'min_diff': args.min_diff,
            'max_edge100': args.max_edge100,
            'min_food100': args.min_food100,
            'max_candidate_deaths': args.max_candidate_deaths,
            'max_log_unsafe': args.max_log_unsafe,
            'require_winning_record': args.require_winning_record,
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print('best:')
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
