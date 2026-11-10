"""Bounded snapshot measurements, not an optimizer. Run as a module."""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from snake_bot import bot_weights
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_brain import SnakeBrain
from snake_bot.snake_state import legal_moves, manhattan, parse_state
from tools.analyze_matches import parse_log
from tools.bot_lab import (
    initial_state,
    make_turn_data,
    observe_simulation_state,
)
from v7_logging import safe_debug

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TYPES = {
    'foodrace',
    'X',
    '#',
    'wrongfood',
    'smallspace',
    'rivalnear',
    'lead',
    'deficit',
    'v7tails',
}
EXCLUDED_CONTROLS = {
    'iterative_search_time_ms',
    'iterative_search_max_depth',
    'deep_extra_depth_margin',
    'advantage_control_enabled',
}
VOLATILE = {
    'search_nodes',
    'search_transposition_hits',
    'search_elapsed_ms',
    'search_depth',
    'elapsed_ms',
    'nodes',
    'completed_depth',
}


def perturbations(value, percent):
    radius = abs(value) * percent // 100
    return [value - radius, value + radius] if radius else []


def active_inventory(path):
    """Read the audited decision-active coefficient list and validate its identity."""
    names = []
    for line in path.read_text(encoding='utf-8').splitlines():
        name = active_inventory_row(line)
        if name is not None:
            names.append(name)
    if not valid_active_inventory(names):
        raise ValueError(
            ('Audit must identify exactly 135 unique known decision-active ' 'weights')
        )
    return names


def active_inventory_row(line):
    cells = [cell.strip() for cell in line.split('|')]
    if len(cells) > 6 and cells[6] == 'decision-active':
        return cells[1]
    return None


def valid_active_inventory(names):
    return (
        len(names) == 135
        and len(set(names)) == 135
        and set(names) <= bot_weights.DEFAULT_WEIGHTS.keys()
    )


def canonical(value):
    """Remove volatile diagnostics before comparing two recorded decision results."""
    value = safe_debug(value)
    if isinstance(value, dict):
        return canonical_mapping(value)
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def canonical_mapping(value):
    return {key: canonical(item) for key, item in value.items() if key not in VOLATILE}


@contextmanager
def tracked_weights(weights):
    previous = bot_weights.ACTIVE_WEIGHTS
    reads = set()

    class TrackingDict(dict):
        def __getitem__(self, key):
            reads.add(key)
            return super().__getitem__(key)

    effective = dict(bot_weights.DEFAULT_WEIGHTS)
    effective.update(weights)
    bot_weights.ACTIVE_WEIGHTS = TrackingDict(effective)
    try:
        yield reads
    finally:
        bot_weights.ACTIVE_WEIGHTS = previous


class SearchReplay:
    """Optional conditional diagnostic helper; the RAW measurement never uses it."""

    def __enter__(self):
        self.results, self.hits = {}, 0
        original = IterativeSearchEngine.search

        def search(engine, state, side, legal, target=None):
            key = (
                state,
                side,
                tuple(legal),
                target,
                engine.time_budget_ms,
                engine.max_depth,
            )
            if key in self.results:
                self.hits += 1
            else:
                self.results[key] = copy.deepcopy(
                    original(engine, state, side, legal, target=target)
                )
            return copy.deepcopy(self.results[key])

        self.patcher = patch.object(IterativeSearchEngine, 'search', search)
        self.patcher.start()
        return self

    def __exit__(self, *args):
        self.patcher.stop()


def classify(changed, internal, read, states):
    """Classify evidence from confirmed changes, internal differences and observed reads."""
    if not states:
        return 'inconclusive'
    if not read:
        return 'inactive_in_corpus'
    if changed / states >= 0.25:
        return 'high'
    return 'moderate' if changed else 'low'


def select_representative(candidates, covered, used, used_paths):
    available = [item for item in candidates if (item[0], item[1]) not in used]
    return max(available, key=lambda item: (len(item[5] - covered), item[0] not in used_paths))


def state_types(state):
    """Tag an observed position for corpus coverage using objects, scores, races and space."""
    tags = object_state_types(state)
    tags.update(score_state_types(state))
    tags.update(race_state_types(state))
    tags.update(space_state_types(state))
    return tags


def object_state_types(state):
    available = {
        'X': state.pickups,
        '#': state.walls,
        'wrongfood': state.wrong_food(),
        'v7tails': state.tail_food_a | state.tail_food_b,
    }
    return {tag for tag, cells in available.items() if cells}


def score_state_types(state):
    gap = state.scores[state.side] - state.scores[state.enemy]
    if not gap:
        return set()
    return {'lead' if gap > 0 else 'deficit'}


def race_state_types(state):
    tags = set()
    head, enemy = state.head(state.side), state.head(state.enemy)
    if not head or not enemy:
        return tags
    if manhattan(head, enemy) <= 4:
        tags.add('rivalnear')
    if is_food_race(head, enemy, state.food):
        tags.add('foodrace')
    return tags


def is_food_race(head, enemy, food):
    return any(abs(manhattan(head, cell) - manhattan(enemy, cell)) <= 2 for cell in food)


def space_state_types(state):
    from snake_bot.food_planner import classify_safety

    _, region, _, _ = classify_safety(state, state.side)
    if region <= max(12, len(state.body(state.side)) * 2 + 4):
        return {'smallspace'}
    return set()


def log_context(turns, actions, diagnostics, index):
    """Replay observations/actions, using logged planner context rather than new decisions."""
    brain = SnakeBrain(enable_debug=False)
    for i, data in enumerate(turns[:index]):
        replay_logged_turn(brain, data, actions, diagnostics, i)
    data = turns[index]
    debug = logged_debug(diagnostics, index)
    restore_entry_context(brain, data, debug)
    return brain, debug


def logged_debug(diagnostics, index):
    return diagnostics[index] if index < len(diagnostics) else {}


def replay_logged_turn(brain, data, actions, diagnostics, index):
    game = str(data.get('game_id', 'default'))
    state = brain.observe_state(data)
    brain._update_food_clock(game, state.side, state)
    debug = logged_debug(diagnostics, index)
    next_debug = logged_debug(diagnostics, index + 1)
    restore_logged_target(brain, game, state.side, next_debug)
    restore_logged_progress(brain, game, state.side, debug)
    brain._remember(game, state)
    action = actions[index] if index < len(actions) else debug
    direction = action.get('direction')
    if direction:
        brain.commit_move(data, direction)


def restore_logged_target(brain, game, side, debug):
    target = debug.get('current_target')
    brain.targets[(game, side)] = tuple(target) if target else None


def restore_logged_progress(brain, game, side, debug):
    plan = debug.get('plan') or {}
    if plan.get('food'):
        brain.last_progress[(game, side)] = (tuple(plan['food']), plan.get('our_distance'))


def restore_entry_context(brain, data, debug):
    game, side = str(data.get('game_id', 'default')), data['side']
    restore_logged_target(brain, game, side, debug)
    if 'hunger' in debug:
        brain.turns_since_food[(game, side)] = max(0, debug['hunger'] - 1)
    if 'crash_history' in debug:
        brain.crash_history[game] = dict(debug['crash_history'])
    for extra in range(max(0, debug.get('active_games', 1) - len(brain.previous_snakes))):
        brain.previous_snakes[f'sensitivity-inert-{extra}'] = {
            'A': (),
            'B': (),
        }


def build_corpus():
    """Select four historical snapshots and four bounded synthetic controls."""
    candidates = real_snapshot_candidates()
    if not candidates:
        raise ValueError('No usable real v7 log snapshots')
    corpus = select_real_samples(candidates)
    corpus.extend(lab_corpus_sample(number) for number in range(4))
    if not REQUIRED_TYPES <= set().union(*(sample['types'] for sample in corpus)):
        raise ValueError('Required corpus types missing')
    return corpus


def real_snapshot_candidates():
    return [
        candidate
        for path in sorted((ROOT / 'games').glob('*.log'))
        for candidate in candidates_from_log(path)
    ]


def candidates_from_log(path):
    turns, actions, _, metadata, diagnostics = parse_log(path)
    if str(metadata.get('version')) != '2.16':
        return []
    candidates = []
    for index in (8, 40, 90, 140):
        candidate = log_snapshot_candidate(path, index, turns, actions, diagnostics)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def log_snapshot_candidate(path, index, turns, actions, diagnostics):
    if index >= len(turns) or turns[index].get('remaining_moves', 0) <= 0:
        return None
    state = parse_state(turns[index])
    if not legal_moves(state, state.side):
        return None
    return path, index, turns, actions, diagnostics, state_types(state)


def select_real_samples(candidates):
    corpus, covered, used, used_paths = [], set(), set(), set()
    for number in range(4):
        item = select_representative(candidates, covered, used, used_paths)
        path, index, turns, actions, diagnostics, tags = item
        used.add((path, index))
        used_paths.add(path)
        brain, debug = log_context(turns, actions, diagnostics, index)
        data = dict(turns[index])
        data.pop('turn_token', None)
        corpus.append(
            {
                'id': f'real-{number + 1}',
                'source': 'real_v7',
                'types': tags,
                'data': data,
                'brain': brain,
                'path': str(path.relative_to(ROOT)),
                'index': index,
                'logged_context': {
                    key: debug.get(key)
                    for key in ('current_target', 'hunger', 'repeat_count', 'active_games')
                },
            }
        )
        covered.update(tags)
    return corpus


def lab_corpus_sample(number):
    state = initial_state(number + 1, rows=7, cols=7, food_count=0)
    state = replace(state, **lab_sample_changes(number))
    data = make_turn_data(state, f'sensitivity-lab-{number}', number)
    data.pop('turn_token', None)
    brain = SnakeBrain(enable_debug=False)
    observe_simulation_state(brain, state, data)
    if not legal_moves(state, state.side):
        raise ValueError('Lab representative has no legal moves')
    return {
        'id': f'lab-{number + 1}',
        'source': 'lab',
        'types': state_types(state),
        'data': data,
        'brain': brain,
        'seed': number + 1,
        'context_scope': 'single authoritative lab snapshot; no invented prior targets',
    }


def lab_sample_changes(number):
    changes = {
        'food': frozenset({(2, 3)}),
        'food_values': {(2, 3): 1, (1, 3): 3},
        'pickups': frozenset({(4, 2)}),
        'walls': frozenset({(1, 1)}),
        'scores': {'A': 2200 if number % 2 == 0 else 0, 'B': 0 if number % 2 == 0 else 2200},
        'remaining_moves': 180,
    }
    if number == 2:
        changes.update(
            tail_food_a=frozenset({(2, 2)}),
            tail_food_b=frozenset({(4, 3)}),
            crashed_a=True,
            crashed_b=True,
        )
    if number == 3:
        changes.update(
            walls=lab_corridor_walls(),
            pickups=frozenset({(2, 1)}),
            food_values={(2, 3): 1, (2, 5): 3},
        )
    return changes


def lab_corridor_walls():
    return frozenset((r, c) for r in range(7) for c in range(7) if r not in (2, 3))


def evaluate(sample, weights):
    """Measure one isolated decision using the supplied complete parameter configuration."""
    brain = copy.deepcopy(sample['brain'])
    data = copy.deepcopy(sample['data'])
    started = time.perf_counter()
    with tracked_weights(weights) as reads:
        move = brain.choose_move(data)
        move = brain.safe_direction(data, move)
    debug = safe_debug(brain.decision_debug(data['game_id']) or {})
    return {
        'move': move,
        'diagnostics': canonical(debug),
        'reads': sorted(reads),
        'search': [
            {key: item.get(key) for key in VOLATILE if key in item}
            for item in debug.get('deep_moves', [])
        ],
        'seconds': time.perf_counter() - started,
    }


def protected_hashes(corpus):
    paths = set((ROOT / 'snake_bot').glob('*.py'))
    paths.update([ROOT / 'weights/active_weights.json', ROOT / 'docs/weight_tuning_audit.md'])
    paths.update(ROOT / sample['path'] for sample in corpus if 'path' in sample)
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.exists()
    }


def pilot(corpus, weights):
    """Run paired baseline controls and estimate the bounded measurement workload."""
    controls = {sample['id']: [evaluate(sample, weights) for _ in range(2)] for sample in corpus}
    return controls, pilot_summary(corpus, weights, controls)


def pilot_summary(corpus, weights, controls):
    total = sum(result['seconds'] for results in controls.values() for result in results)
    upper = maximum_variant_calls(weights, len(corpus))
    summary = {
        'calls': len(corpus) * 2,
        'seconds': total,
        'max_variant_calls': upper,
        'projected_seconds': total / (2 * len(corpus)) * upper,
        'move_noise_states': control_noise_count(controls, 'move'),
        'diagnostic_noise_states': control_noise_count(controls, 'diagnostics'),
    }
    return summary


def maximum_variant_calls(weights, states):
    eligible = [
        key
        for key in active_inventory(ROOT / 'docs/weight_tuning_audit.md')
        if key not in EXCLUDED_CONTROLS
    ]
    return (
        sum(
            len(set(perturbations(weights[key], 10) + perturbations(weights[key], 20)))
            for key in eligible
        )
        * states
    )


def control_noise_count(controls, field):
    return sum(results[0][field] != results[1][field] for results in controls.values())


def measure(corpus, weights, controls, progress=print):
    """Measure each declared control with paired baselines and bounded perturbations."""
    names = active_inventory(ROOT / 'docs/weight_tuning_audit.md')
    rows = []
    for number, key in enumerate(names, 1):
        rows.append(measure_weight(corpus, weights, controls, key))
        report_measurement_progress(number, len(names), progress)
    return rows


def report_measurement_progress(number, total, progress):
    """Emit the same periodic progress records without changing measurements."""
    if number % 15 == 0 or number == total:
        progress(json.dumps({'weights_done': number, 'weights_total': total}, sort_keys=True))


def noisy_control_states(corpus, controls):
    return {
        sample['id']
        for sample in corpus
        if controls[sample['id']][0]['move'] != controls[sample['id']][1]['move']
    }


def measure_weight(corpus, weights, controls, key):
    trials, tested = [], set()
    unusable = noisy_control_states(corpus, controls)
    if key not in EXCLUDED_CONTROLS:
        probe_weight(corpus, weights, controls, key, trials, tested, unusable)
    mark_usable_trials(trials, unusable)
    return measurement_row(corpus, weights, controls, key, trials, tested, unusable)


def probe_weight(corpus, weights, controls, key, trials, tested, unusable):
    for percent in (10, 20):
        values = untested_perturbations(weights[key], percent, tested)
        probe_candidates(corpus, weights, controls, key, percent, values, trials, tested, unusable)
        if any(trial['confirmed_change'] for trial in trials):
            break


def untested_perturbations(value, percent, tested):
    return [candidate for candidate in perturbations(value, percent) if candidate not in tested]


def probe_candidates(corpus, weights, controls, key, percent, values, trials, tested, unusable):
    for candidate in values:
        tested.add(candidate)
        variant = dict(weights, **{key: candidate})
        for sample in corpus:
            trials.append(
                measure_trial(
                    sample, weights, controls, key, percent, candidate, variant, unusable
                )
            )


def confirm_changed_move(sample, weights, variant, before, repeat, after, unusable):
    if after['move'] == before['move'] or before['move'] != repeat['move']:
        return None, None, False
    paired = evaluate(sample, weights)
    confirmation = evaluate(sample, variant)
    if paired['move'] != before['move']:
        unusable.add(sample['id'])
    confirmed = (paired['move'], confirmation['move']) == (before['move'], after['move'])
    return paired, confirmation, confirmed


def ranked_directions(diagnostics):
    return [item.get('direction') for item in diagnostics.get('deep_moves', [])]


def target_or_rank_changed(before, after):
    plan_before = before['diagnostics'].get('plan') or {}
    plan_after = after['diagnostics'].get('plan') or {}
    return ranked_directions(before['diagnostics']) != ranked_directions(
        after['diagnostics']
    ) or plan_before.get('food') != plan_after.get('food')


def measure_trial(sample, weights, controls, key, percent, candidate, variant, unusable):
    before, repeat = controls[sample['id']]
    after = evaluate(sample, variant)
    paired, confirmation, confirmed = confirm_changed_move(
        sample, weights, variant, before, repeat, after, unusable
    )
    comparable = before['diagnostics'] == repeat['diagnostics']
    internal = after['diagnostics'] != before['diagnostics']
    return {
        'state_id': sample['id'],
        'types': sorted(sample['types']),
        'value': candidate,
        'requested_percent': percent,
        'actual_percent': 100 * abs(candidate - weights[key]) / abs(weights[key]),
        'changed_move': after['move'] != before['move'],
        'confirmed_change': confirmed,
        'observed_diagnostic_difference': internal,
        'diagnostic_comparable': comparable,
        'internal_change': internal and comparable,
        'rank_or_target_change': target_or_rank_changed(before, after),
        'baseline_move_noise': before['move'] != repeat['move'],
        'result': after,
        'confirmation': confirmation,
        'paired_baseline': paired,
    }


def mark_usable_trials(trials, unusable):
    for trial in trials:
        trial['usable'] = trial['state_id'] not in unusable
        if not trial['usable']:
            trial['confirmed_change'] = False


def trial_states(trials, field):
    return {trial['state_id'] for trial in trials if trial[field]}


def states_read(corpus, controls, key, trials):
    reads = control_read_states(corpus, controls, key)
    reads.update(trial['state_id'] for trial in trials if key in trial['result']['reads'])
    return sorted(reads)


def control_read_states(corpus, controls, key):
    return {sample['id'] for sample in corpus if control_reads_weight(controls[sample['id']], key)}


def control_reads_weight(results, key):
    return any(key in result['reads'] for result in results)


def measurement_flags(key, tested):
    if key in EXCLUDED_CONTROLS:
        return ['excluded_control_budget_depth_or_categorical']
    return [] if tested else ['discrete_unprobeable_within_20%']


def measurement_classification(key, flags, changed, internal, reads, usable_states):
    if flags:
        return 'excluded_control' if key in EXCLUDED_CONTROLS else 'discrete_unprobeable'
    return classify(len(changed), len(internal), bool(reads), usable_states)


def confirmed_percentage(usable):
    return (
        100 * sum(trial['confirmed_change'] for trial in usable) / len(usable) if usable else None
    )


def rank_change_states(trials, changed, rank_changed):
    return sorted(
        {
            trial['state_id']
            for trial in trials
            if trial['internal_change'] and trial['rank_or_target_change'] == rank_changed
        }
        - changed
    )


def unconfirmed_diagnostic_states(trials):
    return sorted(
        {
            trial['state_id']
            for trial in trials
            if trial['observed_diagnostic_difference'] and not trial['diagnostic_comparable']
        }
    )


def minimum_confirmed_delta(trials):
    return min(
        (trial['actual_percent'] for trial in trials if trial['confirmed_change']), default=None
    )


def measurement_row(corpus, weights, controls, key, trials, tested, unusable):
    usable = [trial for trial in trials if trial['usable']]
    usable_states = len({trial['state_id'] for trial in usable})
    changed = trial_states(trials, 'confirmed_change')
    internal = trial_states(trials, 'internal_change')
    reads = states_read(corpus, controls, key, trials)
    flags = measurement_flags(key, tested)
    return {
        **measurement_identity(corpus, weights, key, tested, flags),
        'classification': measurement_classification(
            key, flags, changed, internal, reads, usable_states
        ),
        'states_read': reads,
        'usable_states': usable_states,
        'unusable_states': sorted(unusable),
        'usable_evaluations': len(usable),
        'states_tested': len({trial['state_id'] for trial in trials}),
        'evaluations': len(trials),
        'changed_states': sorted(changed),
        'changed_moves_percent': confirmed_percentage(usable),
        'internal_only_states': sorted(internal - changed),
        'rank_or_target_only_states': rank_change_states(trials, changed, True),
        'value_only_states': rank_change_states(trials, changed, False),
        'diagnostic_unconfirmed_states': unconfirmed_diagnostic_states(trials),
        'min_confirmed_delta_percent': minimum_confirmed_delta(trials),
        'trials': trials,
    }


def measurement_identity(corpus, weights, key, tested, flags):
    value = weights[key]
    return {
        'weight': key,
        'baseline': value,
        'flags': flags,
        'bounds_10': perturbations(value, 10) or [value, value],
        'bounds_20': perturbations(value, 20) or [value, value],
        'states_in_corpus': len(corpus),
    }


def main(argv=None):
    """Run the requested bounded measurement and save its protected-file audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run',
        action='store_true',
        help='Measure all 135 records after the eight-state pilot',
    )
    parser.add_argument('--max-projected-seconds', type=float, default=360)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    weights = bot_weights.load_weights(ROOT / 'weights/active_weights.json')
    corpus = build_corpus()
    before = protected_hashes(corpus)
    controls, summary = pilot(corpus, weights)
    print(json.dumps({'pilot': summary}, sort_keys=True), flush=True)
    rows = requested_measurement(args, summary, corpus, weights, controls)
    output = sensitivity_output_path(args.output)
    after = protected_hashes(corpus)
    payload = {
        'method': ('RAW unchanged choose_move + safe_direction; independent cloned ' 'contexts'),
        'scope': (
            'conditional corpus of eight snapshots, not global influence or '
            'competitive performance'
        ),
        'classification_rule': (
            'high >=25% corpus states confirmed; moderate any confirmed; low '
            'read/no confirmed; inactive_in_corpus unread'
        ),
        'excluded_controls': sorted(EXCLUDED_CONTROLS),
        'pilot': summary,
        'effective_baseline': weights,
        'baseline_controls': controls,
        'corpus': [serializable_sample(sample) for sample in corpus],
        'rows': rows,
        'elapsed_seconds': time.perf_counter() - started,
        'manifest_before': before,
        'manifest_after': after,
        'protected_files_unchanged': before == after,
    }
    write_sensitivity_report(output, payload)


def requested_measurement(args, summary, corpus, weights, controls):
    if args.run and summary['projected_seconds'] <= args.max_projected_seconds:
        return measure(
            corpus,
            weights,
            controls,
            progress=lambda line: print(line, flush=True),
        )
    elif args.run:
        print(
            ('Projected runtime exceeds the requested ceiling; measurement ' 'not launched.'),
            flush=True,
        )
    return []


def sensitivity_output_path(requested):
    output = requested or ROOT / 'games/sensitivity' / time.strftime('%Y%m%d-%H%M%S')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'games/sensitivity') or output.exists():
        raise ValueError('Output must be a new directory under ignored games/sensitivity/')
    return output


def serializable_sample(sample):
    return {
        key: sorted(value) if key == 'types' else value
        for key, value in sample.items()
        if key != 'brain'
    }


def write_sensitivity_report(output, payload):
    """Write the unchanged report schema after all protected-file checks."""
    rows, summary = payload['rows'], payload['pilot']
    before, after = payload['manifest_before'], payload['manifest_after']
    output.mkdir(parents=True)
    (output / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    report = [
        '# RAW snapshot sensitivity',
        '',
        payload['scope'],
        '',
        (
            f"Elapsed: {payload['elapsed_seconds']:.2f}s. Protected files "
            f"unchanged: {before == after}."
        ),
        f"Pilot: {json.dumps(summary, sort_keys=True)}",
        '',
        (
            'Two raw baseline repetitions; changed variants repeated with a '
            'paired fresh baseline.'
        ),
        (
            'Clock/depth/nodes excluded from internal-value comparison; '
            'retained separately in JSON.'
        ),
        (
            'Real history reconstructed from logged observations, actions and '
            'planner context; not a live-process clone.'
        ),
        ('Lab history starts at an authoritative snapshot. No game rollout ' 'or optimization.'),
        '',
        (
            '| Weight | Class | Read states | Tested states | Evaluations | '
            'Confirmed states | Move change % | Internal-only states | Min '
            'delta % |'
        ),
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        report.append(
            (
                f"| {row['weight']} | {row['classification']} | "
                f"{len(row['states_read'])} | {row['states_tested']} | "
                f"{row['evaluations']} | {len(row['changed_states'])} | "
                f"{row['changed_moves_percent']} | "
                f"{len(row['internal_only_states'])} | "
                f"{row['min_confirmed_delta_percent']} |"
            )
        )
    (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(
        json.dumps(
            {
                'output': str(output),
                'rows': len(rows),
                'elapsed_seconds': payload['elapsed_seconds'],
                'classes': dict(Counter(row['classification'] for row in rows)),
                'protected_files_unchanged': before == after,
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
