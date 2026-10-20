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
from tools.bot_lab import initial_state, make_turn_data, observe_simulation_state
from v7_logging import safe_debug

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TYPES = {'foodrace', 'X', '#', 'wrongfood', 'smallspace',
                  'rivalnear', 'lead', 'deficit', 'v7tails'}
EXCLUDED_CONTROLS = {'iterative_search_time_ms', 'iterative_search_max_depth',
                     'deep_extra_depth_margin', 'advantage_control_enabled'}
VOLATILE = {'search_nodes', 'search_transposition_hits', 'search_elapsed_ms',
            'search_depth', 'elapsed_ms', 'nodes', 'completed_depth'}


def perturbations(value, percent):
    radius = abs(value) * percent // 100
    return [value - radius, value + radius] if radius else []


def active_inventory(path):
    names = []
    for line in path.read_text(encoding='utf-8').splitlines():
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) > 6 and cells[6] == 'decision-active':
            names.append(cells[1])
    if len(names) != 135 or len(set(names)) != 135 or not set(names) <= bot_weights.DEFAULT_WEIGHTS.keys():
        raise ValueError('Audit must identify exactly 135 unique known decision-active weights')
    return names


def canonical(value):
    value = safe_debug(value)
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


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
            key = (state, side, tuple(legal), target, engine.time_budget_ms, engine.max_depth)
            if key in self.results:
                self.hits += 1
            else:
                self.results[key] = copy.deepcopy(original(engine, state, side, legal, target=target))
            return copy.deepcopy(self.results[key])

        self.patcher = patch.object(IterativeSearchEngine, 'search', search)
        self.patcher.start()
        return self

    def __exit__(self, *args):
        self.patcher.stop()


def classify(changed, internal, read, states):
    if not states:
        return 'inconclusive'
    if not read:
        return 'inactive_in_corpus'
    if changed and changed / states >= .25:
        return 'high'
    return 'moderate' if changed else 'low'


def select_representative(candidates, covered, used, used_paths):
    available = [item for item in candidates if (item[0], item[1]) not in used]
    return max(available, key=lambda item: (len(item[5] - covered), item[0] not in used_paths))


def state_types(state):
    tags = set()
    head, enemy = state.head(state.side), state.head(state.enemy)
    if state.pickups:
        tags.add('X')
    if state.walls:
        tags.add('#')
    if state.wrong_food():
        tags.add('wrongfood')
    if state.tail_food_a or state.tail_food_b:
        tags.add('v7tails')
    gap = state.scores[state.side] - state.scores[state.enemy]
    if gap:
        tags.add('lead' if gap > 0 else 'deficit')
    if head and enemy:
        if manhattan(head, enemy) <= 4:
            tags.add('rivalnear')
        if any(abs(manhattan(head, food) - manhattan(enemy, food)) <= 2 for food in state.food):
            tags.add('foodrace')
    from snake_bot.food_planner import classify_safety
    _, region, _, _ = classify_safety(state, state.side)
    if region <= max(12, len(state.body(state.side)) * 2 + 4):
        tags.add('smallspace')
    return tags


def log_context(turns, actions, diagnostics, index):
    """Replay observations/actions, using logged planner context rather than new decisions."""
    brain = SnakeBrain(enable_debug=False)
    for i, data in enumerate(turns[:index]):
        game = str(data.get('game_id', 'default'))
        state = brain.observe_state(data)
        brain._update_food_clock(game, state.side, state)
        debug = diagnostics[i] if i < len(diagnostics) else {}
        next_debug = diagnostics[i + 1] if i + 1 < len(diagnostics) else {}
        target = next_debug.get('current_target')
        brain.targets[(game, state.side)] = tuple(target) if target else None
        plan = debug.get('plan') or {}
        if plan.get('food'):
            brain.last_progress[(game, state.side)] = (tuple(plan['food']), plan.get('our_distance'))
        brain._remember(game, state)
        direction = actions[i].get('direction') if i < len(actions) else debug.get('direction')
        if direction:
            brain.commit_move(data, direction)
    data = turns[index]
    debug = diagnostics[index] if index < len(diagnostics) else {}
    game, side = str(data.get('game_id', 'default')), data['side']
    target = debug.get('current_target')
    brain.targets[(game, side)] = tuple(target) if target else None
    if 'hunger' in debug:
        brain.turns_since_food[(game, side)] = max(0, debug['hunger'] - 1)
    if 'crash_history' in debug:
        brain.crash_history[game] = dict(debug['crash_history'])
    for extra in range(max(0, debug.get('active_games', 1) - len(brain.previous_snakes))):
        brain.previous_snakes[f'sensitivity-inert-{extra}'] = {'A': (), 'B': ()}
    return brain, debug


def build_corpus():
    candidates = []
    for path in sorted((ROOT / 'games').glob('*.log')):
        turns, actions, _, metadata, diagnostics = parse_log(path)
        if str(metadata.get('version')) != '2.16':
            continue
        for index in (8, 40, 90, 140):
            if index < len(turns) and turns[index].get('remaining_moves', 0) > 0:
                state = parse_state(turns[index])
                if legal_moves(state, state.side):
                    candidates.append((path, index, turns, actions, diagnostics, state_types(state)))
    if not candidates:
        raise ValueError('No usable real v7 log snapshots')
    corpus, covered, used, used_paths = [], set(), set(), set()
    for number in range(4):
        item = select_representative(candidates, covered, used, used_paths)
        path, index, turns, actions, diagnostics, tags = item
        used.add((path, index))
        used_paths.add(path)
        brain, debug = log_context(turns, actions, diagnostics, index)
        data = dict(turns[index])
        data.pop('turn_token', None)
        corpus.append({'id': f'real-{number + 1}', 'source': 'real_v7', 'types': tags,
                       'data': data, 'brain': brain, 'path': str(path.relative_to(ROOT)),
                       'index': index, 'logged_context': {key: debug.get(key) for key in
                            ('current_target', 'hunger', 'repeat_count', 'active_games')}})
        covered.update(tags)
    for number in range(4):
        state = initial_state(number + 1, rows=7, cols=7, food_count=0)
        changes = {'food': frozenset({(2, 3)}), 'food_values': {(2, 3): 1, (1, 3): 3},
                   'pickups': frozenset({(4, 2)}), 'walls': frozenset({(1, 1)}),
                   'scores': {'A': 2200 if number % 2 == 0 else 0,
                              'B': 0 if number % 2 == 0 else 2200}, 'remaining_moves': 180}
        if number == 2:
            changes.update(tail_food_a=frozenset({(2, 2)}), tail_food_b=frozenset({(4, 3)}),
                           crashed_a=True, crashed_b=True)
        if number == 3:
            changes['walls'] = frozenset((r, c) for r in range(7) for c in range(7)
                                         if r not in (2, 3))
            changes['pickups'] = frozenset({(2, 1)})
            changes['food_values'] = {(2, 3): 1, (2, 5): 3}
        state = replace(state, **changes)
        data = make_turn_data(state, f'sensitivity-lab-{number}', number)
        data.pop('turn_token', None)
        brain = SnakeBrain(enable_debug=False)
        observe_simulation_state(brain, state, data)
        if not legal_moves(state, state.side):
            raise ValueError('Lab representative has no legal moves')
        corpus.append({'id': f'lab-{number + 1}', 'source': 'lab', 'types': state_types(state),
                       'data': data, 'brain': brain, 'seed': number + 1,
                       'context_scope': 'single authoritative lab snapshot; no invented prior targets'})
    if not REQUIRED_TYPES <= set().union(*(sample['types'] for sample in corpus)):
        raise ValueError('Required corpus types missing')
    return corpus


def evaluate(sample, weights):
    brain = copy.deepcopy(sample['brain'])
    data = copy.deepcopy(sample['data'])
    started = time.perf_counter()
    with tracked_weights(weights) as reads:
        move = brain.choose_move(data)
        move = brain.safe_direction(data, move)
    debug = safe_debug(brain.decision_debug(data['game_id']) or {})
    return {'move': move, 'diagnostics': canonical(debug), 'reads': sorted(reads),
            'search': [{key: item.get(key) for key in VOLATILE if key in item}
                       for item in debug.get('deep_moves', [])],
            'seconds': time.perf_counter() - started}


def protected_hashes(corpus):
    paths = set((ROOT / 'snake_bot').glob('*.py'))
    paths.update([ROOT / 'weights/active_weights.json', ROOT / 'docs/weight_tuning_audit.md'])
    paths.update(ROOT / sample['path'] for sample in corpus if 'path' in sample)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.exists()}


def pilot(corpus, weights):
    controls = {sample['id']: [evaluate(sample, weights) for _ in range(2)] for sample in corpus}
    total = sum(result['seconds'] for results in controls.values() for result in results)
    eligible = [key for key in active_inventory(ROOT / 'docs/weight_tuning_audit.md')
                if key not in EXCLUDED_CONTROLS]
    upper = sum(len(set(perturbations(weights[key], 10) + perturbations(weights[key], 20)))
                for key in eligible) * len(corpus)
    summary = {'calls': len(corpus) * 2, 'seconds': total, 'max_variant_calls': upper,
               'projected_seconds': total / (2 * len(corpus)) * upper,
               'move_noise_states': sum(results[0]['move'] != results[1]['move']
                                        for results in controls.values()),
               'diagnostic_noise_states': sum(results[0]['diagnostics'] != results[1]['diagnostics']
                                              for results in controls.values())}
    return controls, summary


def measure(corpus, weights, controls, progress=print):
    rows = []
    names = active_inventory(ROOT / 'docs/weight_tuning_audit.md')
    for number, key in enumerate(names, 1):
        value, trials, tested = weights[key], [], set()
        unusable = {sample['id'] for sample in corpus
                    if controls[sample['id']][0]['move'] != controls[sample['id']][1]['move']}
        if key not in EXCLUDED_CONTROLS:
            for percent in (10, 20):
                values = [candidate for candidate in perturbations(value, percent) if candidate not in tested]
                for candidate in values:
                    tested.add(candidate)
                    variant = dict(weights, **{key: candidate})
                    for sample in corpus:
                        before, repeat = controls[sample['id']]
                        after = evaluate(sample, variant)
                        changed = after['move'] != before['move']
                        confirmation, paired = None, None
                        confirmed = False
                        if changed and before['move'] == repeat['move']:
                            paired = evaluate(sample, weights)
                            confirmation = evaluate(sample, variant)
                            confirmed = (paired['move'] == before['move']
                                         and confirmation['move'] == after['move'])
                            if paired['move'] != before['move']:
                                unusable.add(sample['id'])
                        comparable = before['diagnostics'] == repeat['diagnostics']
                        internal = after['diagnostics'] != before['diagnostics']
                        ranks_before = [item.get('direction') for item in before['diagnostics'].get('deep_moves', [])]
                        ranks_after = [item.get('direction') for item in after['diagnostics'].get('deep_moves', [])]
                        plan_before = before['diagnostics'].get('plan') or {}
                        plan_after = after['diagnostics'].get('plan') or {}
                        rank_change = (ranks_before != ranks_after
                                       or plan_before.get('food') != plan_after.get('food'))
                        trials.append({'state_id': sample['id'], 'types': sorted(sample['types']),
                                       'value': candidate, 'requested_percent': percent,
                                       'actual_percent': 100 * abs(candidate - value) / abs(value),
                                       'changed_move': changed, 'confirmed_change': confirmed,
                                       'observed_diagnostic_difference': internal,
                                       'diagnostic_comparable': comparable,
                                       'internal_change': internal and comparable,
                                       'rank_or_target_change': rank_change,
                                       'baseline_move_noise': before['move'] != repeat['move'],
                                       'result': after, 'confirmation': confirmation,
                                       'paired_baseline': paired})
                if any(trial['confirmed_change'] for trial in trials):
                    break
        for trial in trials:
            trial['usable'] = trial['state_id'] not in unusable
            if not trial['usable']:
                trial['confirmed_change'] = False
        usable = [trial for trial in trials if trial['usable']]
        usable_states = len({trial['state_id'] for trial in usable})
        changed_states = {trial['state_id'] for trial in trials if trial['confirmed_change']}
        internal_states = {trial['state_id'] for trial in trials if trial['internal_change']}
        reads = {sample['id'] for sample in corpus if any(key in result['reads'] for result in controls[sample['id']])}
        reads.update(trial['state_id'] for trial in trials if key in trial['result']['reads'])
        flags = []
        if key in EXCLUDED_CONTROLS:
            flags.append('excluded_control_budget_depth_or_categorical')
        elif not tested:
            flags.append('discrete_unprobeable_within_20%')
        status = classify(len(changed_states), len(internal_states), bool(reads), usable_states)
        if flags:
            status = 'excluded_control' if key in EXCLUDED_CONTROLS else 'discrete_unprobeable'
        rows.append({'weight': key, 'baseline': value, 'classification': status, 'flags': flags,
                     'bounds_10': perturbations(value, 10) or [value, value],
                     'bounds_20': perturbations(value, 20) or [value, value],
                     'states_in_corpus': len(corpus), 'states_read': sorted(reads),
                     'usable_states': usable_states, 'unusable_states': sorted(unusable),
                     'usable_evaluations': len(usable),
                     'states_tested': len({trial['state_id'] for trial in trials}),
                     'evaluations': len(trials), 'changed_states': sorted(changed_states),
                     'changed_moves_percent': 100 * sum(trial['confirmed_change'] for trial in usable) / len(usable) if usable else None,
                     'internal_only_states': sorted(internal_states - changed_states),
                     'rank_or_target_only_states': sorted({trial['state_id'] for trial in trials if trial['internal_change'] and trial['rank_or_target_change']} - changed_states),
                     'value_only_states': sorted({trial['state_id'] for trial in trials if trial['internal_change'] and not trial['rank_or_target_change']} - changed_states),
                     'diagnostic_unconfirmed_states': sorted({trial['state_id'] for trial in trials if trial['observed_diagnostic_difference'] and not trial['diagnostic_comparable']}),
                     'min_confirmed_delta_percent': min((trial['actual_percent'] for trial in trials if trial['confirmed_change']), default=None),
                     'trials': trials})
        if number % 15 == 0 or number == len(names):
            progress(json.dumps({'weights_done': number, 'weights_total': len(names)}, sort_keys=True))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Measure all 135 records after the eight-state pilot')
    parser.add_argument('--max-projected-seconds', type=float, default=360)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    weights = bot_weights.load_weights(ROOT / 'weights/active_weights.json')
    corpus = build_corpus()
    before = protected_hashes(corpus)
    controls, summary = pilot(corpus, weights)
    print(json.dumps({'pilot': summary}, sort_keys=True), flush=True)
    rows = []
    if args.run and summary['projected_seconds'] <= args.max_projected_seconds:
        rows = measure(corpus, weights, controls, progress=lambda line: print(line, flush=True))
    elif args.run:
        print('Projected runtime exceeds the requested ceiling; measurement not launched.', flush=True)
    output = args.output or ROOT / 'games/sensitivity' / time.strftime('%Y%m%d-%H%M%S')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'games/sensitivity') or output.exists():
        raise ValueError('Output must be a new directory under ignored games/sensitivity/')
    after = protected_hashes(corpus)
    payload = {'method': 'RAW unchanged choose_move + safe_direction; independent cloned contexts',
               'scope': 'conditional corpus of eight snapshots, not global influence or competitive performance',
               'classification_rule': 'high >=25% corpus states confirmed; moderate any confirmed; low read/no confirmed; inactive_in_corpus unread',
               'excluded_controls': sorted(EXCLUDED_CONTROLS), 'pilot': summary,
               'effective_baseline': weights, 'baseline_controls': controls,
               'corpus': [{key: sorted(value) if key == 'types' else value for key, value in sample.items()
                           if key != 'brain'} for sample in corpus],
               'rows': rows, 'elapsed_seconds': time.perf_counter() - started,
               'manifest_before': before, 'manifest_after': after, 'protected_files_unchanged': before == after}
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    report = ['# RAW snapshot sensitivity', '', payload['scope'], '',
              f"Elapsed: {payload['elapsed_seconds']:.2f}s. Protected files unchanged: {before == after}.",
              f"Pilot: {json.dumps(summary, sort_keys=True)}", '',
              'Two raw baseline repetitions; changed variants repeated with a paired fresh baseline.',
              'Clock/depth/nodes excluded from internal-value comparison; retained separately in JSON.',
              'Real history reconstructed from logged observations, actions and planner context; not a live-process clone.',
              'Lab history starts at an authoritative snapshot. No game rollout or optimization.', '',
              '| Weight | Class | Read states | Tested states | Evaluations | Confirmed states | Move change % | Internal-only states | Min delta % |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {row['weight']} | {row['classification']} | {len(row['states_read'])} | {row['states_tested']} | {row['evaluations']} | {len(row['changed_states'])} | {row['changed_moves_percent']} | {len(row['internal_only_states'])} | {row['min_confirmed_delta_percent']} |")
    (output / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'rows': len(rows), 'elapsed_seconds': payload['elapsed_seconds'],
                      'classes': dict(Counter(row['classification'] for row in rows)),
                      'protected_files_unchanged': before == after}), flush=True)


if __name__ == '__main__':
    main()
