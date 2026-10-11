"""Progressive mirrored weight search; outputs are never activated."""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from snake_bot import bot_weights
from tools.benchmark_bot import load_configuration, run_benchmark


SAFE_TUNABLE_KEYS = (
    'food_distance_cost', 'race_margin_value', 'hunger_value', 'space_after_value',
    'exits_after_value', 'tail_reachable_bonus', 'center_unit_value', 'edge_depth_value',
    'next_food_value', 'chain_food_value', 'chain_race_value', 'chain_close_food_bonus',
    'hunger_close_food_value', 'current_target_bonus', 'fallback_denial_progress',
    'fallback_cycle_penalty',
)
MIN_INFORMATIVE_SEEDS = 8


def random_candidate(base_weights, rng, tunable_ranges, key=None, direction=None):
    weights = dict(base_weights)
    key = key or rng.choice(list(tunable_ranges))
    low, high = tunable_ranges[key]
    center = base_weights[key]
    direction = direction or rng.choice((-1, 1))
    weights[key] = rng.randint(low, center - 1) if direction < 0 else rng.randint(center + 1, high)
    return weights


def competitive_rank(summary):
    total = max(1, summary['total'])
    return (summary['wins'] / total, -summary['losses'] / total,
            summary['draws'] / total, summary['median_score_diff'] or 0,
            summary['mean_score_diff'] or 0)


def improvement_evidence(summary, finalists):
    grouped = {}
    seen = set()
    repeat_stats = {}
    for pair in summary['pairs']['records']:
        identity = (pair['seed'], pair['repetition'])
        if (identity in seen or set(pair['candidate_sides']) != {'A', 'B'}
                or len(pair['candidate_sides']) != 2
                or any(type(pair[key]) is not int or pair[key] < 0
                       for key in ('wins', 'losses', 'draws'))
                or sum(pair[key] for key in ('wins', 'losses', 'draws')) != 2):
            raise ValueError('Evidence requires complete unique mirrored seed/repetition pairs.')
        seen.add(identity)
        grouped.setdefault(pair['seed'], []).append(pair)
        stats = repeat_stats.setdefault(str(pair['repetition']), {'wins': 0, 'losses': 0, 'draws': 0})
        for key in stats:
            stats[key] += pair[key]
    repeats = {pair['repetition'] for pairs in grouped.values() for pair in pairs}
    if (any({p['repetition'] for p in pairs} != repeats for pairs in grouped.values())
            or 2 * len(seen) != summary['total']
            or any(sum(stats[key] for stats in repeat_stats.values()) != summary[key]
                   for key in ('wins', 'losses', 'draws'))):
        raise ValueError('Evidence counts or repetitions disagree with the competitive summary.')
    for stats in repeat_stats.values():
        stats['net_wins'] = stats['wins'] - stats['losses']
    records = []
    for seed, pairs in sorted(grouped.items()):
        records.append({'seed': seed, 'repetitions': len(pairs),
                        'net_wins': sum(p['wins'] - p['losses'] for p in pairs),
                        'draw_games': sum(p['draws'] for p in pairs),
                        'score_diff_sum': sum(p['score_diff_sum'] for p in pairs)})
    positive = sum(record['net_wins'] > 0 for record in records)
    negative = sum(record['net_wins'] < 0 for record in records)
    informative = positive + negative
    alpha = 0.05 / max(1, finalists)
    z = statistics.NormalDist().inv_cdf(1 - alpha)
    lower = None
    if informative:
        proportion = positive / informative
        lower = ((proportion + z * z / (2 * informative)
                  - z * math.sqrt(proportion * (1 - proportion) / informative
                                  + z * z / (4 * informative * informative)))
                 / (1 + z * z / informative))
    sides = summary['by_candidate_side']
    reasons = []
    if sides['A']['total'] != sides['B']['total'] or not sides['A']['total']:
        reasons.append('unbalanced_candidate_sides')
    if summary['wins'] <= summary['losses']:
        reasons.append('nonpositive_competitive_record')
    if informative < MIN_INFORMATIVE_SEEDS:
        reasons.append('insufficient_independent_seed_evidence')
    if lower is None or lower <= 0.5:
        reasons.append('paired_support_inconclusive')
    if any(stats['net_wins'] < 0 for stats in repeat_stats.values()):
        reasons.append('repeat_regression')
    return {'valid_improvement': not reasons, 'reasons': reasons,
            'seed_groups': len(records), 'informative_seed_groups': informative,
            'positive_seed_groups': positive, 'negative_seed_groups': negative,
            'tied_seed_groups': len(records) - informative,
            'draw_games': summary['draws'], 'wilson_lower': lower,
            'adjusted_one_sided_alpha': alpha, 'finalists_compared': finalists,
            'minimum_informative_seed_groups': MIN_INFORMATIVE_SEEDS,
            'seed_records': records,
            'by_repetition': repeat_stats,
            'limitations': 'Self-play evidence, not a guarantee against other opponents.'}


def build_plan(trials=12, top_b=4, top_c=2, a_seeds=2, b_seeds=6,
               final_seeds=24, repetitions=1, seed=1, seconds_per_game=None,
               budget_minutes=None, smoke=False):
    counts = (trials, top_b, top_c, a_seeds, b_seeds, final_seeds, repetitions)
    if any(value <= 0 for value in counts) or repetitions not in (1, 2):
        raise ValueError('Counts must be positive; final repetitions must be 1 or 2.')
    if not smoke and (top_c not in (2, 3) or trials < top_c or top_b < top_c or final_seeds < 8):
        raise ValueError('Use 2 or 3 finalists, enough A/B candidates and >=8 final seeds, or --smoke.')
    if (seconds_per_game is None) != (budget_minutes is None):
        raise ValueError('Provide both --seconds-per-game and --budget-minutes.')
    if seconds_per_game is not None and (
            not math.isfinite(seconds_per_game) or not math.isfinite(budget_minutes)
            or seconds_per_game <= 0 or budget_minutes <= 0):
        raise ValueError('Timing estimates and budget must be finite positive numbers.')
    if smoke:
        trials = top_b = top_c = a_seeds = b_seeds = final_seeds = repetitions = 1
    top_b = min(top_b, trials)
    top_c = min(top_c, top_b)

    def games():
        return 2 * (trials * a_seeds + top_b * b_seeds + top_c * final_seeds * repetitions)

    if seconds_per_game is not None:
        while games() * seconds_per_game > budget_minutes * 60 * 0.8:
            options = []
            if trials > max(top_b, min(counts[0], 2 * top_c)):
                options.append((trials * a_seeds, 'trials'))
            if top_b > top_c:
                options.append((top_b * b_seeds, 'top_b'))
            if b_seeds > 2 and not smoke:
                options.append((top_b * b_seeds, 'b_seeds'))
            if a_seeds > 1:
                options.append((trials * a_seeds, 'a_seeds'))
            if final_seeds > 8 and not smoke:
                options.append((top_c * final_seeds * repetitions, 'final_seeds'))
            if not options:
                raise ValueError('Budget cannot fit the minimum plan; use --dry-run or explicit --smoke.')
            dimension = max(options)[1]
            if dimension == 'trials':
                trials -= 1
            elif dimension == 'top_b':
                top_b -= 1
            elif dimension == 'b_seeds':
                b_seeds -= 1
            elif dimension == 'a_seeds':
                a_seeds -= 1
            else:
                final_seeds -= 1
    stages = []
    for name, candidates, seeds, repeats in (
            ('A', trials, a_seeds, 1), ('B', top_b, b_seeds, 1),
            ('C', top_c, final_seeds, repetitions)):
        stages.append({'name': name, 'candidates': candidates, 'seed': seed,
                       'seeds': seeds, 'repetitions': repeats,
                       'games': 2 * candidates * seeds * repeats})
        seed += seeds
    return {'turns': 400, 'mirror_seeds': True, 'stages': stages,
            'estimated_games': games(), 'seconds_per_game': seconds_per_game,
            'estimated_seconds': games() * seconds_per_game if seconds_per_game else None,
            'budget_minutes': budget_minutes, 'smoke': smoke, 'overhead_reserve_fraction': 0.2,
            'budget_note': 'Plan is approximate; unknown game duration prevents a hard deadline.',
            'evidence_note': 'A/B provisional; C uses fresh seeds, repeats grouped by seed.'}


def validate_output(output, inputs):
    output = Path(output).resolve()
    parts = {part.lower() for part in output.parts}
    if (output.exists() or output in inputs or any(output in path.parents for path in inputs)
            or parts & {'weights', 'logs', 'active_weights.json', 'best_weights.json'}
            or output.suffix):
        raise ValueError('Output must be a new directory outside weights, logs and input paths.')
    return output


def load_bounds(path, base):
    raw = json.loads(Path(path).read_text(encoding='utf-8')) if path else {
        key: (base[key] - max(1, abs(base[key]) // 10),
              base[key] + max(1, abs(base[key]) // 10)) for key in SAFE_TUNABLE_KEYS}
    return validate_bounds(raw, base)


def validate_bounds(raw, base):
    if not isinstance(raw, dict) or not raw:
        raise ValueError('Bounds must be a nonempty JSON object.')
    bounds = {}
    for key, limits in raw.items():
        if (key not in SAFE_TUNABLE_KEYS or key not in base
                or not isinstance(limits, (list, tuple)) or len(limits) != 2
                or any(type(value) is not int for value in limits)):
            raise ValueError('Bounds must name supported coefficients, never strategy/depth/time controls.')
        low, high = limits
        center = base[key]
        radius = max(1, abs(center) // 10)
        if not center - radius <= low < center < high <= center + radius:
            raise ValueError('Bounds must straddle the current base within 10 percent (minimum one unit).')
        bounds[key] = (low, high)
    return bounds


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run_search(base_path, output, bounds, plan, seed, rows=0, cols=0, food=5):
    previous = dict(bot_weights.ACTIVE_WEIGHTS)
    index = []
    started = time.monotonic()
    try:
        baseline = load_configuration(base_path)
        bounds = validate_bounds(bounds, baseline['effective_weights'])
        output = validate_output(output, {Path(base_path).resolve()})
        trials = plan['stages'][0]['candidates']
        capacity = sum(high - low for low, high in bounds.values())
        if trials > capacity:
            raise ValueError(f'Requested {trials} candidates but only {capacity} distinct local mutations are available.')
        output.mkdir(parents=True, exist_ok=False)
        reference = output / 'reference.json'
        write_json(reference, baseline['effective_weights'])
        rng = random.Random(seed)
        keys = list(bounds)
        rng.shuffle(keys)
        directions = {key: rng.choice((-1, 1)) for key in keys}
        strata = [(key, directions[key]) for key in keys] + [(key, -directions[key]) for key in keys]
        remaining = {
            (key, direction): list(range(low, baseline['effective_weights'][key])) if direction < 0
            else list(range(baseline['effective_weights'][key] + 1, high + 1))
            for key, (low, high) in bounds.items() for direction in (-1, 1)
        }
        candidates = []
        for number in range(1, trials + 1):
            for offset in range(len(strata)):
                key, direction = strata[(number - 1 + offset) % len(strata)]
                if remaining[key, direction]:
                    break
            values = remaining[key, direction]
            weights = dict(baseline['effective_weights'])
            weights[key] = values.pop(rng.randrange(len(values)))
            candidate = {'id': number, 'weights_path': str(output / f'candidate-{number:03d}.json'),
                         'changed_key': key, 'base_value': baseline['effective_weights'][key],
                         'candidate_value': weights[key]}
            write_json(candidate['weights_path'], weights)
            candidates.append(candidate)
        write_json(output / 'manifest.json', {
            'version': 7, 'baseline': baseline, 'plan': plan,
            'bounds': bounds, 'candidate_seed': seed, 'candidates': candidates,
            'sampled_keys': sorted({c['changed_key'] for c in candidates}),
            'unsampled_keys': sorted(set(bounds) - {c['changed_key'] for c in candidates}),
            'simulator_parameters': {'turns': 400, 'rows': rows, 'cols': cols, 'food': food},
            'selection': 'Competitive record, then median/mean score difference; diagnostics excluded.',
        })
        survivors = candidates
        for stage in plan['stages']:
            ranked = []
            for candidate in survivors[:stage['candidates']]:
                report_path = output / f"benchmark-{stage['name']}-{candidate['id']:03d}.json"
                if report_path.exists():
                    raise ValueError('Refusing to overwrite a benchmark report.')
                report = run_benchmark(
                    reference=reference, candidate=Path(candidate['weights_path']),
                    seeds=stage['seeds'], seed=stage['seed'], repetitions=stage['repetitions'],
                    output=report_path, turns=400, rows=rows, cols=cols, food=food)
                summary = report['summary']
                expected_pairs = {(game_seed, repeat)
                                  for game_seed in range(stage['seed'], stage['seed'] + stage['seeds'])
                                  for repeat in range(1, stage['repetitions'] + 1)}
                pairs = summary['pairs']['records']
                if (len(pairs) != len(expected_pairs)
                        or {(pair['seed'], pair['repetition']) for pair in pairs} != expected_pairs):
                    raise ValueError('Benchmark seed/repetition grid disagrees with the fresh stage plan.')
                record = {**candidate, 'stage': stage['name'], 'report_path': str(report_path),
                          'rank': competitive_rank(summary), 'summary': summary,
                          'valid_improvement': False, 'status': 'provisional'}
                if stage['name'] == 'C':
                    evidence = improvement_evidence(summary, len(survivors[:stage['candidates']]))
                    record['evidence'] = evidence
                    record['valid_improvement'] = evidence['valid_improvement'] and not plan['smoke']
                    record['status'] = 'supported' if record['valid_improvement'] else 'inconclusive'
                index.append(record)
                ranked.append(record)
            ranked.sort(key=lambda item: item['rank'], reverse=True)
            survivors = ranked
            write_json(output / f"index-{stage['name']}.json", ranked)
        accepted = next((item for item in survivors if item['valid_improvement']), None)
        result = {'version': 7, 'status': 'completed', 'plan': plan, 'index': index,
                  'best_candidate': survivors[0], 'accepted_candidate': accepted,
                  'accepted_improvement': accepted is not None,
                  'elapsed_seconds': time.monotonic() - started,
                  'active_weights_written': False}
        write_json(output / 'summary.json', result)
        return result
    except Exception as error:
        if index or ('candidates' in locals() and output.is_dir()):
            write_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                               'index': index, 'accepted_improvement': False})
        raise
    finally:
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output-dir', '--output', dest='output_dir', type=Path, required=True)
    parser.add_argument('--bounds', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    for name, default in (('trials', 12), ('top-b', 4), ('top-c', 2),
                          ('stage-a-seeds', 2), ('stage-b-seeds', 6),
                          ('final-seeds', 24), ('final-repetitions', 1),
                          ('seed', 1), ('rows', 0), ('cols', 0), ('food', 5)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seconds-per-game', type=float)
    parser.add_argument('--budget-minutes', type=float)
    args = parser.parse_args(argv)
    try:
        base = load_configuration(args.base)
        bounds = load_bounds(args.bounds, base['effective_weights'])
        inputs = {args.base.resolve()}
        if args.bounds:
            inputs.add(args.bounds.resolve())
        output = validate_output(args.output_dir, inputs)
        if args.rows < 0 or (args.cols != 0 and args.cols < 6) or not 1 <= args.food <= 9:
            raise ValueError('Rows must be 0 or >=1, cols 0 or >=6, food 1..9.')
        plan = build_plan(args.trials, args.top_b, args.top_c, args.stage_a_seeds,
                          args.stage_b_seeds, args.final_seeds, args.final_repetitions,
                          args.seed, args.seconds_per_game, args.budget_minutes, args.smoke)
        if args.run and args.seconds_per_game is None:
            raise ValueError('--run requires measured --seconds-per-game and --budget-minutes.')
        if args.run or args.smoke:
            result = run_search(args.base, output, bounds, plan, args.seed,
                                args.rows, args.cols, args.food)
        else:
            result = {'status': 'planned', 'plan': plan, 'bounds': bounds,
                      'baseline': base, 'output_dir': str(output)}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return result
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
