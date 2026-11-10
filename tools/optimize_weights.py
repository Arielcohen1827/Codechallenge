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
    'food_distance_cost',
    'race_margin_value',
    'hunger_value',
    'space_after_value',
    'exits_after_value',
    'tail_reachable_bonus',
    'center_unit_value',
    'edge_depth_value',
    'next_food_value',
    'chain_food_value',
    'chain_race_value',
    'chain_close_food_bonus',
    'hunger_close_food_value',
    'current_target_bonus',
    'fallback_denial_progress',
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
    return (
        summary['wins'] / total,
        -summary['losses'] / total,
        summary['draws'] / total,
        summary['median_score_diff'] or 0,
        summary['mean_score_diff'] or 0,
    )


def improvement_evidence(summary, finalists):
    """Validate mirrored seed groups and calculate conservative improvement evidence."""
    grouped, seen, repeat_stats = group_evidence_pairs(summary['pairs']['records'])
    validate_evidence_totals(summary, grouped, seen, repeat_stats)
    for stats in repeat_stats.values():
        stats['net_wins'] = stats['wins'] - stats['losses']
    records = seed_evidence_records(grouped)
    positive = sum(record['net_wins'] > 0 for record in records)
    negative = sum(record['net_wins'] < 0 for record in records)
    informative = positive + negative
    alpha = 0.05 / max(1, finalists)
    lower = paired_wilson_lower(positive, informative, alpha)
    reasons = improvement_reasons(summary, repeat_stats, informative, lower)
    return evidence_result(
        summary, finalists, records, repeat_stats, positive, negative, alpha, lower, reasons
    )


def group_evidence_pairs(pairs):
    grouped, seen, repeat_stats = {}, set(), {}
    for pair in pairs:
        identity = (pair['seed'], pair['repetition'])
        validate_evidence_pair(pair, identity, seen)
        seen.add(identity)
        grouped.setdefault(pair['seed'], []).append(pair)
        stats = repeat_stats.setdefault(
            str(pair['repetition']), {'wins': 0, 'losses': 0, 'draws': 0}
        )
        add_pair_counts(stats, pair)
    return grouped, seen, repeat_stats


def validate_evidence_pair(pair, identity, seen):
    if identity in seen or not mirrored_pair(pair) or not pair_outcomes_valid(pair):
        raise ValueError('Evidence requires complete unique mirrored seed/repetition pairs.')


def mirrored_pair(pair):
    return set(pair['candidate_sides']) == {'A', 'B'} and len(pair['candidate_sides']) == 2


def pair_outcomes_valid(pair):
    return (
        all(valid_outcome_count(pair[key]) for key in ('wins', 'losses', 'draws'))
        and sum(pair[key] for key in ('wins', 'losses', 'draws')) == 2
    )


def valid_outcome_count(value):
    return type(value) is int and value >= 0


def add_pair_counts(stats, pair):
    for key in stats:
        stats[key] += pair[key]


def validate_evidence_totals(summary, grouped, seen, repeat_stats):
    if (
        not matching_repetitions(grouped)
        or 2 * len(seen) != summary['total']
        or not matching_outcome_totals(summary, repeat_stats)
    ):
        raise ValueError('Evidence counts or repetitions disagree with the competitive summary.')


def matching_repetitions(grouped):
    repeats = {pair['repetition'] for pairs in grouped.values() for pair in pairs}
    return all(pair_repetitions(pairs) == repeats for pairs in grouped.values())


def pair_repetitions(pairs):
    return {pair['repetition'] for pair in pairs}


def matching_outcome_totals(summary, repeat_stats):
    return all(
        sum(stats[key] for stats in repeat_stats.values()) == summary[key]
        for key in ('wins', 'losses', 'draws')
    )


def seed_evidence_records(grouped):
    return [
        {
            'seed': seed,
            'repetitions': len(pairs),
            'net_wins': sum(pair['wins'] - pair['losses'] for pair in pairs),
            'draw_games': sum(pair['draws'] for pair in pairs),
            'score_diff_sum': sum(pair['score_diff_sum'] for pair in pairs),
        }
        for seed, pairs in sorted(grouped.items())
    ]


def paired_wilson_lower(positive, informative, alpha):
    z = statistics.NormalDist().inv_cdf(1 - alpha)
    if not informative:
        return None
    proportion = positive / informative
    return (
        proportion
        + z * z / (2 * informative)
        - z
        * math.sqrt(
            proportion * (1 - proportion) / informative + z * z / (4 * informative * informative)
        )
    ) / (1 + z * z / informative)


def unbalanced_sides(summary):
    sides = summary['by_candidate_side']
    return sides['A']['total'] != sides['B']['total'] or not sides['A']['total']


def improvement_reasons(summary, repeat_stats, informative, lower):
    conditions = (
        ('unbalanced_candidate_sides', unbalanced_sides(summary)),
        ('nonpositive_competitive_record', summary['wins'] <= summary['losses']),
        ('insufficient_independent_seed_evidence', informative < MIN_INFORMATIVE_SEEDS),
        ('paired_support_inconclusive', lower is None or lower <= 0.5),
        ('repeat_regression', any(stats['net_wins'] < 0 for stats in repeat_stats.values())),
    )
    return [reason for reason, present in conditions if present]


def evidence_result(
    summary, finalists, records, repeat_stats, positive, negative, alpha, lower, reasons
):
    informative = positive + negative
    return {
        'valid_improvement': not reasons,
        'reasons': reasons,
        'seed_groups': len(records),
        'informative_seed_groups': informative,
        'positive_seed_groups': positive,
        'negative_seed_groups': negative,
        'tied_seed_groups': len(records) - informative,
        'draw_games': summary['draws'],
        'wilson_lower': lower,
        'adjusted_one_sided_alpha': alpha,
        'finalists_compared': finalists,
        'minimum_informative_seed_groups': MIN_INFORMATIVE_SEEDS,
        'seed_records': records,
        'by_repetition': repeat_stats,
        'limitations': 'Self-play evidence, not a guarantee against other opponents.',
    }


def build_plan(
    trials=12,
    top_b=4,
    top_c=2,
    a_seeds=2,
    b_seeds=6,
    final_seeds=24,
    repetitions=1,
    seed=1,
    seconds_per_game=None,
    budget_minutes=None,
    smoke=False,
):
    """Build a bounded mirrored comparison plan without activating any weights."""
    counts = (trials, top_b, top_c, a_seeds, b_seeds, final_seeds, repetitions)
    validate_plan_counts(counts, repetitions)
    validate_finalists(trials, top_b, top_c, final_seeds, smoke)
    validate_plan_timing(seconds_per_game, budget_minutes)
    sizes = dict(
        zip(
            ('trials', 'top_b', 'top_c', 'a_seeds', 'b_seeds', 'final_seeds', 'repetitions'),
            counts,
        )
    )
    if smoke:
        sizes = dict.fromkeys(sizes, 1)
    sizes['top_b'] = min(sizes['top_b'], sizes['trials'])
    sizes['top_c'] = min(sizes['top_c'], sizes['top_b'])
    fit_plan_budget(sizes, counts[0], smoke, seconds_per_game, budget_minutes)
    games = plan_game_count(sizes)
    return {
        'turns': 400,
        'mirror_seeds': True,
        'stages': plan_stages(sizes, seed),
        'estimated_games': games,
        'seconds_per_game': seconds_per_game,
        'estimated_seconds': games * seconds_per_game if seconds_per_game else None,
        'budget_minutes': budget_minutes,
        'smoke': smoke,
        'overhead_reserve_fraction': 0.2,
        'budget_note': 'Plan is approximate; unknown game duration prevents a hard deadline.',
        'evidence_note': 'A/B provisional; C uses fresh seeds, repeats grouped by seed.',
    }


def validate_plan_counts(counts, repetitions):
    if any(value <= 0 for value in counts) or repetitions not in (1, 2):
        raise ValueError('Counts must be positive; final repetitions must be 1 or 2.')


def validate_finalists(trials, top_b, top_c, final_seeds, smoke):
    if smoke:
        return
    if not valid_finalist_counts(trials, top_b, top_c, final_seeds):
        raise ValueError(
            'Use 2 or 3 finalists, enough A/B candidates and >=8 final seeds, or --smoke.'
        )


def valid_finalist_counts(trials, top_b, top_c, final_seeds):
    return top_c in (2, 3) and trials >= top_c and top_b >= top_c and final_seeds >= 8


def validate_plan_timing(seconds_per_game, budget_minutes):
    if (seconds_per_game is None) != (budget_minutes is None):
        raise ValueError('Provide both --seconds-per-game and --budget-minutes.')
    if seconds_per_game is not None and not valid_timing_values(seconds_per_game, budget_minutes):
        raise ValueError('Timing estimates and budget must be finite positive numbers.')


def valid_timing_values(*values):
    return all(math.isfinite(value) and value > 0 for value in values)


def plan_game_count(sizes):
    return 2 * (
        sizes['trials'] * sizes['a_seeds']
        + sizes['top_b'] * sizes['b_seeds']
        + sizes['top_c'] * sizes['final_seeds'] * sizes['repetitions']
    )


def fit_plan_budget(sizes, original_trials, smoke, seconds_per_game, budget_minutes):
    if seconds_per_game is None:
        return
    while plan_game_count(sizes) * seconds_per_game > budget_minutes * 60 * 0.8:
        options = plan_reductions(sizes, original_trials, smoke)
        if not options:
            raise ValueError(
                'Budget cannot fit the minimum plan; use --dry-run or explicit --smoke.'
            )
        dimension = max(options)[1]
        sizes[dimension] -= 1


def plan_reductions(sizes, original_trials, smoke):
    trials, top_b, top_c = sizes['trials'], sizes['top_b'], sizes['top_c']
    a_seeds, b_seeds = sizes['a_seeds'], sizes['b_seeds']
    final_seeds, repetitions = sizes['final_seeds'], sizes['repetitions']
    dimensions = (
        (trials * a_seeds, 'trials', trials > max(top_b, min(original_trials, 2 * top_c))),
        (top_b * b_seeds, 'top_b', top_b > top_c),
        (top_b * b_seeds, 'b_seeds', b_seeds > 2 and not smoke),
        (trials * a_seeds, 'a_seeds', a_seeds > 1),
        (top_c * final_seeds * repetitions, 'final_seeds', final_seeds > 8 and not smoke),
    )
    return [(cost, dimension) for cost, dimension, allowed in dimensions if allowed]


def plan_stages(sizes, seed):
    stages = []
    for name, candidates, seeds, repeats in (
        ('A', sizes['trials'], sizes['a_seeds'], 1),
        ('B', sizes['top_b'], sizes['b_seeds'], 1),
        ('C', sizes['top_c'], sizes['final_seeds'], sizes['repetitions']),
    ):
        stages.append(
            {
                'name': name,
                'candidates': candidates,
                'seed': seed,
                'seeds': seeds,
                'repetitions': repeats,
                'games': 2 * candidates * seeds * repeats,
            }
        )
        seed += seeds
    return stages


def validate_output(output, inputs):
    """Reject existing, protected and input-containing output locations before any writes."""
    output = Path(output).resolve()
    parts = {part.lower() for part in output.parts}
    if protected_output_path(output, inputs) or reserved_output_name(output, parts):
        raise ValueError('Output must be a new directory outside weights, logs and input paths.')
    return output


def protected_output_path(output, inputs):
    return output.exists() or output in inputs or any(output in path.parents for path in inputs)


def reserved_output_name(output, parts):
    return bool(
        parts & {'weights', 'logs', 'active_weights.json', 'best_weights.json'} or output.suffix
    )


def load_bounds(path, base):
    raw = (
        json.loads(Path(path).read_text(encoding='utf-8'))
        if path
        else {
            key: (
                base[key] - max(1, abs(base[key]) // 10),
                base[key] + max(1, abs(base[key]) // 10),
            )
            for key in SAFE_TUNABLE_KEYS
        }
    )
    return validate_bounds(raw, base)


def validate_bounds(raw, base):
    """Validate supported local mutations and return normalized integer intervals."""
    if not nonempty_bounds_object(raw):
        raise ValueError('Bounds must be a nonempty JSON object.')
    bounds = {}
    for key, limits in raw.items():
        if not valid_bound_definition(key, limits, base):
            raise ValueError(
                ('Bounds must name supported coefficients, never ' 'strategy/depth/time controls.')
            )
        low, high = limits
        center = base[key]
        radius = max(1, abs(center) // 10)
        if not center - radius <= low < center < high <= center + radius:
            raise ValueError(
                ('Bounds must straddle the current base within 10 percent ' '(minimum one unit).')
            )
        bounds[key] = (low, high)
    return bounds


def nonempty_bounds_object(raw):
    return isinstance(raw, dict) and bool(raw)


def valid_bound_definition(key, limits, base):
    return key in SAFE_TUNABLE_KEYS and key in base and valid_bound_pair(limits)


def valid_bound_pair(limits):
    return (
        isinstance(limits, (list, tuple))
        and len(limits) == 2
        and all(type(value) is int for value in limits)
    )


def write_json(path, payload):
    """Create a new JSON file exclusively so existing evidence cannot be overwritten."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run_search(base_path, output, bounds, plan, seed, rows=0, cols=0, food=5):
    """Execute an explicitly requested plan, restoring active weights even on failure."""
    previous = dict(bot_weights.ACTIVE_WEIGHTS)
    session = WeightSearchSession(base_path, output, bounds, plan, seed, rows, cols, food)
    try:
        return session.run()
    except Exception as error:
        session.record_failure(error)
        raise
    finally:
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)


class MutationSampler:
    """Consume distinct local mutations in the original stratified random order."""

    def __init__(self, bounds, baseline, seed):
        self.rng = random.Random(seed)
        self.baseline = baseline
        keys = list(bounds)
        self.rng.shuffle(keys)
        directions = {key: self.rng.choice((-1, 1)) for key in keys}
        self.strata = [(key, directions[key]) for key in keys] + [
            (key, -directions[key]) for key in keys
        ]
        self.remaining = mutation_ranges(bounds, baseline)

    def next_weights(self, number):
        """Consume one unused mutation and return its coefficient and complete configuration."""
        key, direction = self.available_stratum(number)
        values = self.remaining[key, direction]
        weights = dict(self.baseline)
        weights[key] = values.pop(self.rng.randrange(len(values)))
        return key, weights

    def available_stratum(self, number):
        for offset in range(len(self.strata)):
            key, direction = self.strata[(number - 1 + offset) % len(self.strata)]
            if self.remaining[key, direction]:
                return key, direction
        raise ValueError('No distinct local mutations remain.')


def mutation_ranges(bounds, baseline):
    return {
        (key, direction): (
            list(range(low, baseline[key]))
            if direction < 0
            else list(range(baseline[key] + 1, high + 1))
        )
        for key, (low, high) in bounds.items()
        for direction in (-1, 1)
    }


class WeightSearchSession:
    """Own one output transaction and its provisional comparison records."""

    def __init__(self, base_path, output, bounds, plan, seed, rows, cols, food):
        self.base_path, self.output = base_path, output
        self.bounds, self.plan, self.seed = bounds, plan, seed
        self.rows, self.cols, self.food = rows, cols, food
        self.index = []
        self.candidates = None
        self.started = time.monotonic()

    def run(self):
        """Generate candidates, evaluate the staged plan and persist its final evidence."""
        self.prepare()
        self.generate_candidates()
        self.write_manifest()
        survivors = self.evaluate_stages()
        result = self.final_result(survivors)
        write_json(self.output / 'summary.json', result)
        return result

    def prepare(self):
        self.baseline = load_configuration(self.base_path)
        self.bounds = validate_bounds(self.bounds, self.baseline['effective_weights'])
        self.output = validate_output(self.output, {Path(self.base_path).resolve()})
        trials = self.plan['stages'][0]['candidates']
        capacity = sum(high - low for low, high in self.bounds.values())
        if trials > capacity:
            raise ValueError(
                f'Requested {trials} candidates but only {capacity} '
                'distinct local mutations are available.'
            )
        self.output.mkdir(parents=True, exist_ok=False)
        self.reference = self.output / 'reference.json'
        write_json(self.reference, self.baseline['effective_weights'])

    def generate_candidates(self):
        sampler = MutationSampler(self.bounds, self.baseline['effective_weights'], self.seed)
        self.candidates = []
        for number in range(1, self.plan['stages'][0]['candidates'] + 1):
            key, weights = sampler.next_weights(number)
            candidate = {
                'id': number,
                'weights_path': str(self.output / f'candidate-{number:03d}.json'),
                'changed_key': key,
                'base_value': self.baseline['effective_weights'][key],
                'candidate_value': weights[key],
            }
            write_json(candidate['weights_path'], weights)
            self.candidates.append(candidate)

    def write_manifest(self):
        sampled = {candidate['changed_key'] for candidate in self.candidates}
        write_json(
            self.output / 'manifest.json',
            {
                'version': 7,
                'baseline': self.baseline,
                'plan': self.plan,
                'bounds': self.bounds,
                'candidate_seed': self.seed,
                'candidates': self.candidates,
                'sampled_keys': sorted(sampled),
                'unsampled_keys': sorted(set(self.bounds) - sampled),
                'simulator_parameters': {
                    'turns': 400,
                    'rows': self.rows,
                    'cols': self.cols,
                    'food': self.food,
                },
                'selection': (
                    'Competitive record, then median/mean score difference; '
                    'diagnostics excluded.'
                ),
            },
        )

    def evaluate_stages(self):
        survivors = self.candidates
        for stage in self.plan['stages']:
            finalists = survivors[: stage['candidates']]
            ranked = [
                self.evaluate_candidate(stage, candidate, len(finalists))
                for candidate in finalists
            ]
            ranked.sort(key=lambda item: item['rank'], reverse=True)
            survivors = ranked
            write_json(self.output / f"index-{stage['name']}.json", ranked)
        return survivors

    def evaluate_candidate(self, stage, candidate, finalists):
        report_path = self.output / f"benchmark-{stage['name']}-{candidate['id']:03d}.json"
        if report_path.exists():
            raise ValueError('Refusing to overwrite a benchmark report.')
        report = run_benchmark(
            reference=self.reference,
            candidate=Path(candidate['weights_path']),
            seeds=stage['seeds'],
            seed=stage['seed'],
            repetitions=stage['repetitions'],
            output=report_path,
            turns=400,
            rows=self.rows,
            cols=self.cols,
            food=self.food,
        )
        summary = report['summary']
        validate_stage_pairs(summary, stage)
        record = {
            **candidate,
            'stage': stage['name'],
            'report_path': str(report_path),
            'rank': competitive_rank(summary),
            'summary': summary,
            'valid_improvement': False,
            'status': 'provisional',
        }
        if stage['name'] == 'C':
            record.update(finalist_evidence(summary, finalists, self.plan['smoke']))
        self.index.append(record)
        return record

    def final_result(self, survivors):
        accepted = next((item for item in survivors if item['valid_improvement']), None)
        return {
            'version': 7,
            'status': 'completed',
            'plan': self.plan,
            'index': self.index,
            'best_candidate': survivors[0],
            'accepted_candidate': accepted,
            'accepted_improvement': accepted is not None,
            'elapsed_seconds': time.monotonic() - self.started,
            'active_weights_written': False,
        }

    def record_failure(self, error):
        """Preserve partial comparison records after a failed output transaction."""
        if self.index or (self.candidates is not None and self.output.is_dir()):
            write_json(
                self.output / 'failure.json',
                {
                    'status': 'failed',
                    'error_type': type(error).__name__,
                    'index': self.index,
                    'accepted_improvement': False,
                },
            )


def validate_stage_pairs(summary, stage):
    expected = {
        (game_seed, repeat)
        for game_seed in range(stage['seed'], stage['seed'] + stage['seeds'])
        for repeat in range(1, stage['repetitions'] + 1)
    }
    pairs = summary['pairs']['records']
    if len(pairs) != len(expected) or actual_pair_grid(pairs) != expected:
        raise ValueError('Benchmark seed/repetition grid disagrees with the fresh stage plan.')


def actual_pair_grid(pairs):
    return {(pair['seed'], pair['repetition']) for pair in pairs}


def finalist_evidence(summary, finalists, smoke):
    evidence = improvement_evidence(summary, finalists)
    valid = evidence['valid_improvement'] and not smoke
    return {
        'evidence': evidence,
        'valid_improvement': valid,
        'status': 'supported' if valid else 'inconclusive',
    }


def parse_search_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output-dir', '--output', dest='output_dir', type=Path, required=True)
    parser.add_argument('--bounds', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    for name, default in (
        ('trials', 12),
        ('top-b', 4),
        ('top-c', 2),
        ('stage-a-seeds', 2),
        ('stage-b-seeds', 6),
        ('final-seeds', 24),
        ('final-repetitions', 1),
        ('seed', 1),
        ('rows', 0),
        ('cols', 0),
        ('food', 5),
    ):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seconds-per-game', type=float)
    parser.add_argument('--budget-minutes', type=float)
    args = parser.parse_args(argv)
    return parser, args


def main(argv=None):
    """Validate CLI options and execute a search only when its run mode is explicitly requested."""
    parser, args = parse_search_arguments(argv)
    try:
        base = load_configuration(args.base)
        bounds = load_bounds(args.bounds, base['effective_weights'])
        inputs = {args.base.resolve()}
        if args.bounds:
            inputs.add(args.bounds.resolve())
        output = validate_output(args.output_dir, inputs)
        validate_search_dimensions(args.rows, args.cols, args.food)
        plan = build_plan(
            args.trials,
            args.top_b,
            args.top_c,
            args.stage_a_seeds,
            args.stage_b_seeds,
            args.final_seeds,
            args.final_repetitions,
            args.seed,
            args.seconds_per_game,
            args.budget_minutes,
            args.smoke,
        )
        result = requested_search(args, output, bounds, plan, base)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return result
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


def requested_search(args, output, bounds, plan, base):
    if args.run and args.seconds_per_game is None:
        raise ValueError('--run requires measured --seconds-per-game and --budget-minutes.')
    if args.run or args.smoke:
        result = run_search(
            args.base, output, bounds, plan, args.seed, args.rows, args.cols, args.food
        )
    else:
        result = {
            'status': 'planned',
            'plan': plan,
            'bounds': bounds,
            'baseline': base,
            'output_dir': str(output),
        }
    return result


def validate_search_dimensions(rows, cols, food):
    if rows < 0 or (cols != 0 and cols < 6) or not 1 <= food <= 9:
        raise ValueError('Rows must be 0 or >=1, cols 0 or >=6, food 1..9.')


if __name__ == '__main__':
    main()
