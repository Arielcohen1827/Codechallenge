"""Isolated planner revisions; production code and the v7 simulator are reused."""

from __future__ import annotations

import copy
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import replace
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from datetime import datetime, timezone
from types import FunctionType, MappingProxyType, SimpleNamespace
import uuid

from snake_bot import food_planner, move_search, snake_brain
from snake_bot.bot_weights import load_weights
from snake_bot import bot_weights
from snake_bot.snake_state import legal_moves, parse_state
from snake_bot.search_engine import IterativeSearchEngine, SearchResult
from tools import bot_lab

REFERENCE_SHA256 = 'd55ef912d547044c3c5b7b8f00a864313c923d2d37d8f327998c04ecca834dc4'
CANDIDATE_SHA256 = '64dd15b53bc755af443cee49948c368ce41bac3097a3eab32c5c95a6875e68e3'
WEIGHTS_SHA256 = 'fe93c636286b307218006ea6398cc16e9ab4cf609ef40a827567d79d228092a8'


def sha256(path):
    """Compute the digest used to pin an explicit historical source or artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    """Detach diagnostic structures through deterministic JSON serialization."""
    return json.loads(json.dumps(value, sort_keys=True))


def load_planner(path, expected_hash, weights):
    """Load only a source matching its expected digest with isolated immutable weights."""
    path = Path(path).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError(f'SHA256 mismatch: {path}')
    name = '_food_planner_compare_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Dataclasses need a registered module. Compile bytes to avoid writing pyc files.
    exec(compile(source, str(path), 'exec'), module.__dict__)
    frozen_weights = MappingProxyType(dict(weights))
    module.get_weights = lambda: frozen_weights
    return module


def clone_function(function, namespace):
    """Bind a function's code and defaults to a private dependency namespace."""
    cloned = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


def private_namespace(module, planner, weights):
    """Clone module logic with private planner and simulator dependencies."""
    namespace = dict(vars(module))
    replace_planner_functions(module, namespace, planner)
    namespace['get_weights'] = lambda: weights
    clone_owned_functions(module, namespace)
    if module is bot_lab:
        namespace['SimulationSession'] = private_class(bot_lab.SimulationSession, namespace)
    return namespace


def replace_planner_functions(module, namespace, planner):
    """Redirect imported planner functions to the selected audited revision."""
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == food_planner.__name__:
            namespace[name] = getattr(planner, value.__name__)


def clone_owned_functions(module, namespace):
    """Keep each module-owned function bound to the same isolated namespace."""
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            namespace[name] = clone_function(value, namespace)


def private_class(original, namespace):
    """Clone executable methods while retaining the original class contracts."""
    methods = {}
    for name, value in vars(original).items():
        if isinstance(value, FunctionType):
            methods[name] = clone_function(value, namespace)
        elif isinstance(value, staticmethod):
            methods[name] = staticmethod(clone_function(value.__func__, namespace))
    return type('Isolated' + original.__name__, (original,), methods)


def make_brain(planner, weights, tape=None):
    """Build an isolated bot whose planner, cache and engine use one audited revision."""
    weights = MappingProxyType(dict(weights))
    ranking = private_namespace(move_search, planner, weights)
    cache_class = private_class(move_search.TurnSearchCache, ranking)
    ranking['TurnSearchCache'] = cache_class
    ranking['IterativeSearchEngine'] = tape if tape is not None else IterativeSearchEngine
    namespace = private_namespace(snake_brain, planner, weights)
    namespace['TurnSearchCache'] = cache_class
    namespace['rank_deep_moves'] = ranking['rank_deep_moves']
    return private_class(snake_brain.SnakeBrain, namespace)(enable_debug=False)


class SearchTape:
    """Record real searches, or replay ONLY for controlled same-target validation."""

    def __init__(self, entries=None):
        self.replaying = entries is not None
        self.entries = copy.deepcopy(entries) if entries is not None else []
        self.index = 0
        self.target_match = True

    def fresh(self):
        """Create an independent tape reader from captured immutable search results."""
        return SearchTape(self.entries)

    def assert_consumed(self):
        """Reject a comparison that did not consume exactly its recorded search requests."""
        if self.index != len(self.entries):
            raise ValueError('unconsumed recorded searches')

    def from_game_state(self, state, time_budget_ms=90, max_depth=6):
        """Wrap a real engine to record or verify equivalent historical search inputs."""
        engine, compact = IterativeSearchEngine.from_game_state(state, time_budget_ms, max_depth)

        def search(compact_state, side, legal, target=None):
            """Record an actual search or replay only a matching captured request."""
            request = normalized(
                {
                    'state': asdict(compact_state),
                    'side': side,
                    'legal': list(legal),
                    'budget': time_budget_ms,
                    'max_depth': max_depth,
                    'target': target,
                }
            )
            if not self.replaying:
                result = engine.search(compact_state, side, legal, target=target)
                self.entries.append({'request': request, 'result': asdict(result)})
                self.index += 1
                return result
            return self.consume_entry(request)

        return SimpleNamespace(search=search), compact

    def consume_entry(self, request):
        """Advance exactly one captured request and reject mismatched replay inputs."""
        if self.index >= len(self.entries):
            raise ValueError('no recorded search for this call')
        entry = self.entries[self.index]
        self.index += 1
        self.target_match &= request['target'] == entry['request']['target']
        if not self.target_match:
            raise ValueError('search target input mismatch; replay is same-target only')
        if not matching_tape_inputs(request, entry['request']):
            raise ValueError('search input mismatch')
        return SearchResult(**copy.deepcopy(entry['result']))


def matching_tape_inputs(actual, recorded):
    return all(actual[key] == recorded[key] for key in actual if key != 'target')


def describe(plan):
    """Serialize the complete selected plan, including its computed entry direction."""
    if plan is None:
        return None
    return normalized(dict(asdict(plan), first_move=plan.first_move))


def classify(
    reference,
    candidate,
    reference_move,
    candidate_move,
    remaining,
    side,
    actor,
):
    """Classify objective, economic and horizon differences in saved decisions."""

    def objective(plan):
        """Identify a plan by its cell and objective kind for comparison."""
        return None if plan is None else (plan['food'], plan['objective_kind'])

    changed = reference != candidate
    unreachable = unreachable_multiplier(reference, remaining, side, actor)
    direction_changed = reference_move != candidate_move
    horizon = changed and unreachable
    reachable_x = changed_multiplier_economy(reference, candidate, unreachable)
    return {
        'any_plan_changed': changed,
        'objective_changed': objective(reference) != objective(candidate),
        'direction_changed': direction_changed,
        'reference_unreachable_x': unreachable,
        'horizon_x_plan_changed': horizon,
        'horizon_x_direction_changed': horizon and direction_changed,
        'horizon_related_direction_changed': horizon and direction_changed,
        'reachable_x_economic_changed': reachable_x,
    }


def unreachable_multiplier(reference, remaining, side, actor):
    """Use the existing alternating-turn X horizon classification."""
    return bool(
        reference
        and reference['objective_kind'] == 'multiplier'
        and 2 * reference['our_distance'] - (side == actor) > remaining
    )


def changed_multiplier_economy(reference, candidate, unreachable):
    """Compare economic terms only for a shared reachable multiplier objective."""
    if not same_multiplier_objective(reference, candidate) or unreachable:
        return False
    fields = (
        'value',
        'projected_income',
        'rival_projected_income',
        'economic_swing',
        'multiplier_marginal',
    )
    return any(reference.get(key) != candidate.get(key) for key in fields)


def same_multiplier_objective(reference, candidate):
    """Require both plans to target the same multiplier cell."""
    if not reference or not candidate:
        return False
    return (
        reference['objective_kind'] == candidate['objective_kind'] == 'multiplier'
        and reference['food'] == candidate['food']
    )


def changed_definitions(reference_source, candidate_source):
    """Require the candidate AST changes to remain inside the approved planner definitions."""

    def definitions(source):
        """Index top-level AST definitions for an exact source-change audit."""
        return {
            getattr(node, 'name', f'top_{index}'): ast.dump(node, include_attributes=False)
            for index, node in enumerate(ast.parse(source).body)
        }

    before, after = definitions(reference_source), definitions(candidate_source)
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )
    if set(changed) - {'build_food_plan', 'choose_food_plan'}:
        raise ValueError(f'planner changes outside allowed definitions: {changed}')
    return changed


def selected_plan(brain, planner, data):
    """Evaluate a copied brain's planner inputs without advancing actual match history."""
    probe = copy.copy(brain)
    probe.__dict__ = copy.deepcopy(brain.__dict__)
    game_id = str(data['game_id'])
    state = probe.observe_state(data)
    probe._update_food_clock(game_id, state.side, state)
    if not legal_moves(state, state.side) or planner.forced_kill_move(
        state, state.side, legal_moves(state, state.side)
    ):
        return None, probe
    target = probe.targets.get((game_id, state.side))
    hunger = probe.turns_since_food[(game_id, state.side)]
    plans = planner.build_food_plans(state, state.side, target, hunger)
    return (
        planner.choose_food_plan(state, state.side, target, hunger, plans=plans),
        probe,
    )


@contextmanager
def shared_weights(weights):
    """Temporarily apply explicit benchmark weights and restore them on every exit."""
    original = dict(bot_weights.get_weights())
    bot_weights.set_active_weights(weights)
    try:
        yield
    finally:
        bot_weights.set_active_weights(original)


def compare_game(seed, candidate_side, reference, candidate, weights, emit, max_turns=400):
    """Reuse simulate_game's code object and all v7 transitions with private globals."""
    planners = {'reference': reference, 'candidate': candidate}
    sides = iter(('A', 'B'))
    counts, timing = Counter(), Counter()
    first_plan, first_move = [], []

    def factory(enable_debug=False):
        """Construct the correctly assigned bot revision for each simulated participant."""
        side = next(sides)
        label = 'candidate' if side == candidate_side else 'reference'
        other_label = 'reference' if label == 'candidate' else 'candidate'
        planner, alternate = planners[label], planners[other_label]
        base = make_brain(planner, weights).__class__

        class ComparedBrain(base):
            """Capture independently timed planner decisions from equal history."""

            def choose_move(self, data):
                """Measure differences while advancing actual participant history."""
                started = perf_counter()
                before = copy.deepcopy(self.__dict__)
                actual_plan, probe = selected_plan(self, planner, data)
                other_plan, _ = selected_plan(self, alternate, data)
                preflight_ms = (perf_counter() - started) * 1000
                actual_selected, other_selected = describe(actual_plan), describe(other_plan)
                started = perf_counter()
                direction = super().choose_move(data)
                actual_ms = (perf_counter() - started) * 1000
                debug = copy.deepcopy(self.decision_debug(data['game_id']) or {})
                assert_selected_plan(debug, self, actual_plan, 'actual')
                shadow_direction, shadow_ms, shadow_debug, mode = measure_shadow_plan(
                    alternate,
                    weights,
                    before,
                    data,
                    direction,
                    actual_selected,
                    other_selected,
                    other_plan,
                )
                state = probe.current_states[str(data['game_id'])]
                (ref_plan, cand_plan), (ref_move, cand_move) = relative_plan_comparison(
                    label, actual_selected, other_selected, direction, shadow_direction
                )
                classification = classify(
                    ref_plan,
                    cand_plan,
                    ref_move,
                    cand_move,
                    state.remaining_moves,
                    side,
                    state.side,
                )
                key = (str(data['game_id']), side)
                row = {
                    'turn': int(data['turn_token'].split('_')[-1]),
                    'actor': side,
                    'actual_revision': label,
                    'snapshot': copy.deepcopy(data),
                    'history': {
                        'target': probe.targets.get(key),
                        'hunger': probe.turns_since_food[key],
                        'repeat_count': probe._repeat_count(str(data['game_id']), state),
                        'crash_history': probe.crash_history[str(data['game_id'])],
                        'tracked_bodies': state.snakes,
                        'reliable_tails': sorted(state.reliable_tails),
                    },
                    'reference_plan': ref_plan,
                    'candidate_plan': cand_plan,
                    'reference_direction': ref_move,
                    'candidate_direction': cand_move,
                    'actual_debug': debug,
                    'shadow_debug': shadow_debug,
                    'shadow_mode': mode,
                    'shadow_history_reconciled': True,
                    'classification': classification,
                    'timing_ms': {
                        'actual': actual_ms,
                        'shadow': shadow_ms,
                        'preflight': preflight_ms,
                    },
                }
                row = normalized(row)
                record_plan_comparison(
                    counts, timing, first_plan, first_move, row, classification, mode
                )
                emit(row)
                return direction

        return ComparedBrain(enable_debug=False)

    namespace = private_namespace(bot_lab, candidate, MappingProxyType(dict(weights)))
    namespace['SnakeBrain'] = factory
    started = perf_counter()
    with shared_weights(weights):
        result = namespace['simulate_game'](
            seed, max_turns=max_turns, candidate_side=candidate_side
        )
    timing['wall_ms'] = (perf_counter() - started) * 1000
    return {
        'match': asdict(result),
        'counts': dict(counts),
        'timing': dict(timing),
        'first_plan_divergence': first_plan[0] if first_plan else None,
        'first_move_divergence': first_move[0] if first_move else None,
    }


def assert_selected_plan(debug, brain, plan, scope):
    if normalized(debug.get('plan')) != normalized(brain._plan_debug(plan)):
        raise ValueError(scope + ' debug plan differs from full-plan preflight')


def measure_shadow_plan(
    alternate, weights, before, data, direction, actual_selected, other_selected, other_plan
):
    if actual_selected == other_selected:
        return direction, 0.0, None, 'equal_full_plan_no_search'
    shadow = make_brain(alternate, weights)
    shadow.__dict__ = before
    started = perf_counter()
    shadow_direction = shadow.choose_move(copy.deepcopy(data))
    elapsed = (perf_counter() - started) * 1000
    debug = copy.deepcopy(shadow.decision_debug(data['game_id']) or {})
    assert_selected_plan(debug, shadow, other_plan, 'shadow')
    return shadow_direction, elapsed, debug, 'independent_timed_estimate'


def relative_plan_comparison(label, actual_plan, other_plan, actual_move, other_move):
    if label == 'reference':
        return (actual_plan, other_plan), (actual_move, other_move)
    return (other_plan, actual_plan), (other_move, actual_move)


def record_plan_comparison(counts, timing, first_plan, first_move, row, classification, mode):
    counts.update({name: int(value) for name, value in classification.items()})
    counts[mode] += 1
    timing.update(row['timing_ms'])
    record_first_divergence(classification['any_plan_changed'], first_plan, row)
    record_first_divergence(classification['direction_changed'], first_move, row)


def record_first_divergence(changed, rows, row):
    if changed and not rows:
        rows.append(row)


def comparison_metrics(selected):
    """Summarize mirrored games with candidate-relative scores and win rates."""
    diffs = [candidate_score_difference(game) for game in selected]
    wins = sum(game['match']['winner'] == game['match']['candidate_side'] for game in selected)
    draws = sum(game['match']['winner'] is None for game in selected)
    return {
        'games': len(selected),
        'wins': wins,
        'draws': draws,
        'losses': len(selected) - wins - draws,
        'win_rate': optional_ratio(wins, len(selected)),
        'win_rate_excluding_draws': optional_ratio(wins, len(selected) - draws),
        **score_difference_statistics(diffs),
    }


def candidate_score_difference(game):
    """Orient score difference to the candidate independently of its assigned side."""
    match = game['match']
    diff = match['score_a'] - match['score_b']
    return diff if match['candidate_side'] == 'A' else -diff


def optional_ratio(numerator, denominator):
    """Leave rates undefined when no eligible observations exist."""
    return numerator / denominator if denominator else None


def score_difference_statistics(diffs):
    """Keep median and mean undefined for an empty set of games."""
    return {
        'median_score_diff': median(diffs) if diffs else None,
        'mean_score_diff': optional_ratio(sum(diffs), len(diffs)),
    }


def summarize(games, expected_games=24):
    """Report mirrored outcomes and the established exposure threshold with its limitations."""
    counts, timing = (Counter(), Counter())
    for game in games:
        counts.update(game['counts'])
        timing.update(game['timing'])
    exposure = comparison_exposure(games)
    pairs = comparison_pairs(games)
    complete = len(games) == expected_games
    return {
        'completed_games': len(games),
        'expected_games': expected_games,
        'candidate': comparison_metrics(games),
        'candidate_by_side': comparison_sides(games),
        'mirror_pairs': pairs,
        'counts': dict(counts),
        'timing_ms': dict(timing),
        'exposure': exposure,
        'conclusion': exposure_conclusion(complete, exposure['threshold_met']),
        'limitations': [
            'Independent timed shadow decisions can differ because of timing noise.',
            (
                'Equal full plans infer no planner-dependent change; no alternate timed movement '
                'was measured.'
            ),
            '24 unreplicated timed games do not prove universal competitive superiority.',
        ],
    }


def comparison_exposure(games):
    """Report association exposure without claiming exact causal effects."""
    exposed = [game for game in games if game['counts'].get('horizon_x_direction_changed', 0)]
    seeds = sorted({game['match']['seed'] for game in exposed})
    threshold_met = len(exposed) >= 6 and len(seeds) >= 3
    return {
        'games_with_horizon_related_direction_difference': len(exposed),
        'distinct_seeds': seeds,
        'minimum_games': 6,
        'minimum_distinct_seeds': 3,
        'threshold_met': threshold_met,
        'measurement': 'independently timed association, not exact causal effect',
    }


def comparison_sides(games):
    """Keep the candidate-side breakdown independent of aggregate totals."""
    return {
        side: comparison_metrics([g for g in games if g['match']['candidate_side'] == side])
        for side in ('A', 'B')
    }


def exposure_conclusion(complete, threshold_met):
    """Require a complete batch and sufficient exposure for its existing label."""
    return 'EXPOSICION_SUFICIENTE_ESTIMADA' if complete and threshold_met else 'INCONCLUSO'


def comparison_pairs(games):
    """Group complete A/B mirrored seeds while preserving sorted seed order."""
    pairs = []
    for seed in sorted({game['match']['seed'] for game in games}):
        pair = [game for game in games if game['match']['seed'] == seed]
        append_complete_pair(pairs, seed, pair)
    return pairs


def append_complete_pair(pairs, seed, games):
    """Add a seed only when both candidate assignments are represented."""
    if {game['match']['candidate_side'] for game in games} == {'A', 'B'}:
        metrics = comparison_metrics(games)
        pairs.append(
            {
                'seed': seed,
                'mean_candidate_score_diff': metrics['mean_score_diff'],
                **metrics,
                'matches': [game['match'] for game in games],
            }
        )


def atomic_json(path, payload):
    """Replace a report atomically after writing its complete temporary representation."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_inputs(config):
    """Reject execution if any recorded source or weight digest has changed."""
    hashes = {path: sha256(path) for path in config.get('input_hashes', {})}
    if hashes != config.get('input_hashes', {}):
        raise ValueError('input source/weights hashes changed during comparison')
    return {
        'verified': True,
        'hashes': hashes,
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }


def run_comparison(config, reference, candidate, weights, output):
    """Execute only explicitly requested mirrored comparisons and preserve per-turn evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    verify_inputs(config)
    atomic_json(output / 'manifest.json', config)
    completed = []
    total = 2 * len(config['seeds'])
    for seed in config['seeds']:
        for side in ('A', 'B'):
            stem = f'seed_{seed}_candidate_{side}'
            print(
                (f'[{len(completed) + 1}/{total}] seed={seed} ' f'candidate={side} starting'),
                flush=True,
            )
            try:
                with (output / (stem + '.jsonl')).open('x', encoding='utf-8') as stream:

                    def emit(row):
                        """Append a turn record and preserve report provenance."""
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
                        stream.flush()
                        if (row['turn'] + 1) % 50 == 0:
                            print(
                                (
                                    f'  turn={row["turn"] + 1} '
                                    f'remaining={row["snapshot"]["remaining_moves"]}'
                                ),
                                flush=True,
                            )

                    game = compare_game(
                        seed,
                        side,
                        reference,
                        candidate,
                        weights,
                        emit,
                        config['turns'],
                    )
                atomic_json(output / (stem + '.json'), game)
                completed.append(game)
                atomic_json(output / 'post_run_hashes.json', verify_inputs(config))
                atomic_json(output / 'summary.json', summarize(completed, total))
                print(
                    (
                        f'  complete turns={game["match"]["turns"]} '
                        f'winner={game["match"]["winner"]}'
                    ),
                    flush=True,
                )
            except BaseException as error:
                atomic_json(
                    output / 'error.json',
                    {
                        'seed': seed,
                        'candidate_side': side,
                        'type': type(error).__name__,
                        'message': str(error),
                    },
                )
                atomic_json(output / 'summary.json', summarize(completed, total))
                raise
    return summarize(completed, total)


def validate_seeds(values):
    """Require the fixed twelve-seed historical comparison contract."""
    if not isinstance(values, list) or len(values) != 12 or not valid_unique_seeds(values):
        raise ValueError('seed-list must contain exactly 12 distinct nonnegative integers')
    return values


def valid_unique_seeds(values):
    """Require distinct nonnegative integers without accepting booleans as seeds."""
    return not any(type(seed) is not int or seed < 0 for seed in values) and len(set(values)) == 12


def prove_snapshot(cases, reference, candidate, weights):
    """Certify the archived unreachable-X planner case against recorded diagnostic inputs."""
    case = special_snapshot(cases)
    diagnostic = case['diagnostic']
    state = tracked_snapshot_state(case, diagnostic)
    records = {}
    target = tuple(diagnostic['current_target']) if diagnostic.get('current_target') else None
    for label, planner in (('reference', reference), ('candidate', candidate)):
        started = perf_counter()
        selected = planner.choose_food_plan(
            state,
            state.side,
            target,
            diagnostic['hunger'],
            plans=planner.build_food_plans(state, state.side, target, diagnostic['hunger']),
        )
        plan_ms = (perf_counter() - started) * 1000
        records[label] = {'plan': describe(selected), 'plan_ms': plan_ms}
    old, new = (records['reference']['plan'], records['candidate']['plan'])
    historic = diagnostic['plan']
    verified = verified_snapshot_plans(old, new, historic, state.remaining_moves)
    if not verified:
        raise ValueError('5580bbd6 planner/isolation proof failed')
    return {
        'verified': verified,
        'filename': case['filename'],
        'debug_line': case['debug_line'],
        'remaining_moves': state.remaining_moves,
        'historical_direction': diagnostic['direction'],
        'historical_plan': historic,
        'results': records,
        'recorded_inputs': {
            'target': target,
            'hunger': diagnostic['hunger'],
            'repeat_count': diagnostic['repeat_count'],
            'crash_history': diagnostic['crash_history'],
        },
        'timed_search_executed': False,
        'brain_choose_move_executed': False,
        'scope': (
            'planner selection with recorded diagnostic inputs; '
            'not historical full-brain movement'
        ),
        'estimate': {
            'external_prior_24_games_minutes': [35, 42],
            'note': (
                'External estimate; actual per-turn wall/decision/preflight/shadow times are '
                'recorded during play.'
            ),
        },
    }


def special_snapshot(cases):
    """Select the unique archived horizon regression fixture."""
    matches = [case for case in cases if '5580bbd6' in case['filename'] and case.get('special')]
    if len(matches) != 1:
        raise ValueError('expected exactly one special 5580bbd6 snapshot')
    return matches[0]


def tracked_snapshot_state(case, diagnostic):
    """Restore explicitly recorded body order for the archived planner proof."""
    state = parse_state(case['observed'], crash_history=diagnostic['crash_history'])
    tracked = diagnostic['verification']['after']['tracked_body']
    state = replace(
        state,
        snakes={side: tuple(map(tuple, tracked[side]['body'])) for side in ('A', 'B')},
        reliable_tails=frozenset((side for side in ('A', 'B') if tracked[side]['order_reliable'])),
    )
    return state


def same_historic_plan(old, historic):
    """Require the archived objective cell and entry direction to match exactly."""
    return (
        old is not None
        and old['food'] == historic['food']
        and old['first_move'] == historic['first_move']
    )


def verified_snapshot_plans(old, new, historic, remaining):
    """Verify the archived unreachable X while preserving a changed candidate plan."""
    return (
        same_historic_plan(old, historic)
        and old['objective_kind'] == 'multiplier'
        and 2 * old['our_distance'] - 1 > remaining
        and new != old
    )


def main(argv=None):
    """Run the historical planner comparison after its provenance checks."""
    PlannerComparison(argv).run()


class PlannerComparison:
    """Own resolved inputs for the historical planner comparison CLI."""

    def __init__(self, argv=None):
        """Parse explicit source paths and the requested historical command."""
        self.parser = argparse.ArgumentParser(description=__doc__)
        for name in ('reference', 'candidate', 'weights', 'seed-list', 'frozen-cases', 'output'):
            self.parser.add_argument('--' + name, type=Path, required=True)
        self.parser.add_argument('--mode', choices=('proof', 'run'), default='proof')
        self.parser.add_argument('--proof-artifact', type=Path)
        self.args = self.parser.parse_args(argv)

    def run(self):
        """Validate all inputs before writing reports or invoking the runner."""
        self.load_inputs()
        self.verify_independent_proof()
        self.build_manifest()
        self.capture_hashes()
        self.execute()

    def load_inputs(self):
        """Load frozen revisions and certify the named horizon regression."""
        if sha256(self.args.weights) != WEIGHTS_SHA256:
            raise ValueError('weights SHA256 mismatch')
        self.weights = load_weights(self.args.weights)
        self.reference = load_planner(self.args.reference, REFERENCE_SHA256, self.weights)
        self.candidate = load_planner(self.args.candidate, CANDIDATE_SHA256, self.weights)
        self.changed = changed_definitions(
            self.args.reference.read_text(encoding='utf-8'),
            self.args.candidate.read_text(encoding='utf-8'),
        )
        self.seeds = validate_seeds(json.loads(self.args.seed_list.read_text(encoding='utf-8')))
        self.proof = prove_snapshot(
            json.loads(self.args.frozen_cases.read_text(encoding='utf-8')),
            self.reference,
            self.candidate,
            self.weights,
        )

    def verify_independent_proof(self):
        """Validate an optional independent proof artifact before execution."""
        if self.args.proof_artifact is not None:
            self.independent = json.loads(self.args.proof_artifact.read_text(encoding='utf-8'))
            if not independent_proof_valid(self.independent, self.args.frozen_cases):
                raise ValueError('independent proof artifact inputs mismatch')
            self.proof['independent_artifact'] = {
                'path': str(self.args.proof_artifact.resolve()),
                'sha256': sha256(self.args.proof_artifact),
                'proof': self.independent,
            }

    def build_manifest(self):
        """Capture shared rule sources and immutable comparison configuration."""
        self.sources = {
            str(path.resolve().relative_to(Path.cwd())): sha256(path)
            for path in sorted(Path('snake_bot').glob('*.py'))
            if path.name != 'food_planner.py'
        }
        self.config = {
            'seeds': self.seeds,
            'turns': 400,
            'games': 24,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(),
            'search_budgets': {
                key: self.weights[key]
                for key in (
                    'iterative_search_time_ms',
                    'iterative_search_max_depth',
                    'iterative_search_min_apply_depth',
                )
            },
            'reference': {'path': str(self.args.reference.resolve()), 'sha256': REFERENCE_SHA256},
            'candidate': {'path': str(self.args.candidate.resolve()), 'sha256': CANDIDATE_SHA256},
            'weights': {
                'path': str(self.args.weights.resolve()),
                'sha256': WEIGHTS_SHA256,
                'effective_sha256': hashlib.sha256(
                    json.dumps(self.weights, sort_keys=True).encode()
                ).hexdigest(),
            },
            'shared_sources': self.sources,
            'simulator_sha256': sha256(bot_lab.__file__),
            'tool_sha256': sha256(__file__),
            'frozen_cases_sha256': sha256(self.args.frozen_cases),
            'seed_list_sha256': sha256(self.args.seed_list),
            'changed_definitions': self.changed,
            'exposure_threshold': {
                'minimum_games': 6,
                'minimum_distinct_seeds': 3,
                'metric': 'horizon-related independently timed direction differences',
            },
            'method': (
                'independent full searches only when full selected plans differ; identical '
                'predecision history; no global swap'
            ),
            'proof': self.proof,
            'python': sys.version,
            'resume_supported': False,
        }

    def capture_hashes(self):
        """Capture every experiment input for later modification checks."""
        self.inputs = [
            self.args.reference,
            self.args.candidate,
            self.args.weights,
            self.args.seed_list,
            self.args.frozen_cases,
            Path(__file__),
            Path(bot_lab.__file__),
            *[Path(path) for path in self.sources],
        ]
        if self.args.proof_artifact is not None:
            self.inputs.append(self.args.proof_artifact)
        self.config['input_hashes'] = {str(path.resolve()): sha256(path) for path in self.inputs}

    def execute(self):
        """Prepare the proof artifact or execute the explicitly requested comparison."""
        if self.args.mode == 'proof':
            self.args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(self.args.output / 'manifest.json', self.config)
            print(json.dumps(self.proof, indent=2), flush=True)
        else:
            print(
                json.dumps(
                    run_comparison(
                        self.config, self.reference, self.candidate, self.weights, self.args.output
                    ),
                    indent=2,
                ),
                flush=True,
            )


def independent_proof_valid(artifact, frozen_cases):
    """Require a verified artifact before reading its source-hash evidence."""
    return artifact.get('verified') is True and independent_hashes_match(artifact, frozen_cases)


def independent_hashes_match(artifact, frozen_cases):
    """Compare archived proof inputs against the approved source digests."""
    return (
        artifact.get('weights_sha256') == WEIGHTS_SHA256
        and artifact.get('frozen_cases_sha256') == sha256(frozen_cases)
        and all(
            artifact['results'][role]['sha256'] == digest
            for role, digest in (('reference', REFERENCE_SHA256), ('candidate', CANDIDATE_SHA256))
        )
    )


if __name__ == '__main__':
    main()
