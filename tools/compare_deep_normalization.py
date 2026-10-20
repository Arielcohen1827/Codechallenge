"""Prepare or explicitly run isolated deep-normalization matches with shared-root diagnostics."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import MappingProxyType, SimpleNamespace

from snake_bot import food_planner, snake_brain
from snake_bot.search_engine import IterativeSearchEngine
from tools import bot_lab, compare_food_planners as shared, deep_clipping_replay as replay
from tools.compare_food_planners import (atomic_json, load_planner, normalized,
                                        private_class, private_namespace, sha256, shared_weights)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('C:/Users/ariel/.codex/visualizations/2026/10/07/01a11686-3a99-78d1-90d4-b5245bb38d12')
BASELINE = EXTERNAL / 'deep_clipping_phase1/move_search.baseline.py'
REFERENCE_SHA = replay.REFERENCE_SEARCH
PLANNER_SHA = replay.REFERENCE_X
ENGINE_SHA = 'a1bcc019422b4f470146ba64bcf1904d2badfaedf6f5e9812f852b4c5a733e81'
WEIGHTS_SHA = shared.WEIGHTS_SHA256


class RawSearchTape(shared.SearchTape):
    """Use the existing exact-request tape; the recorder may also use a frozen engine."""
    def __init__(self, entries=None, engine=IterativeSearchEngine):
        super().__init__(entries)
        self.engine = engine

    def from_game_state(self, state, time_budget_ms=90, max_depth=6):
        if self.replaying:
            return super().from_game_state(state, time_budget_ms, max_depth)
        engine, compact = self.engine.from_game_state(state, time_budget_ms, max_depth)

        def search(compact_state, side, legal, target=None):
            result = engine.search(compact_state, side, legal, target=target)
            request = normalized({'state': asdict(compact_state), 'side': side,
                                  'legal': list(legal), 'budget': time_budget_ms,
                                  'max_depth': max_depth, 'target': target})
            self.entries.append({'request': request, 'result': asdict(result)})
            self.index += 1
            return result

        return SimpleNamespace(search=search), compact


def make_brain(source, planner, weights, tape=None):
    weights = MappingProxyType(dict(weights))
    ranking = private_namespace(source, planner, weights)
    cache = private_class(source.TurnSearchCache, ranking)
    ranking['TurnSearchCache'] = cache
    ranking['IterativeSearchEngine'] = tape if tape is not None else IterativeSearchEngine
    namespace = private_namespace(snake_brain, planner, weights)
    namespace.update(TurnSearchCache=cache, rank_deep_moves=ranking['rank_deep_moves'])
    return private_class(snake_brain.SnakeBrain, namespace)(enable_debug=True)


def validate_candidate(before, after):
    old, new = ast.parse(before), ast.parse(after)
    dump = lambda node: ast.dump(node, include_attributes=False)
    helpers = [node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == '_search_adjustments']
    if len(helpers) != 1 or any(getattr(node, 'name', None) == '_search_adjustments' for node in old.body):
        raise ValueError('candidate must add only the _search_adjustments normalization helper')
    new.body.remove(helpers[0])
    old_rank = next(node for node in old.body if getattr(node, 'name', None) == 'rank_deep_moves')
    new_rank = next(node for node in new.body if getattr(node, 'name', None) == 'rank_deep_moves')
    expected = ast.parse('''
if search.completed_depth >= weights['iterative_search_min_apply_depth'] and search.scores:
    adjustments = _search_adjustments(state, analyses, search.completed_depth, weights)
    for item in analyses:
        adjustment = adjustments.get(item['direction'])
        if adjustment is None:
            continue
        item['search_adjustment'] = adjustment
        item['score'] += adjustment
    analyses.sort(key=deep_sort_key, reverse=True)
''').body[0]
    if dump(new_rank.body[-3]) != dump(expected):
        raise ValueError('candidate normalization call/body differs from approved minimal replacement')
    new_rank.body[-3] = copy.deepcopy(old_rank.body[-3])
    if dump(old) != dump(new):
        raise ValueError('candidate changes outside normalization helper/call')
    return ['_search_adjustments', 'rank_deep_moves']


def classify(reference, candidate, ref_move, cand_move, weights):
    rows = reference.get('deep_moves', [])
    raw = [row['search_score'] for row in rows if row.get('search_score') is not None and not row.get('forced_loss')]
    depth = max((row.get('search_depth', 0) for row in rows), default=0)
    remaining = reference.get('remaining_moves')
    root_terminal = remaining is not None and remaining <= 0 or any(score < -2500 for score in reference.get('scores', {}).values())
    horizon = remaining is not None and remaining <= depth
    extreme = any(abs(score) >= 300000000 for score in raw)
    mode = ('no_search' if not rows else 'depth_gate' if depth < weights['iterative_search_min_apply_depth']
            else 'no_applicable' if not raw else 'legacy_root_terminal' if root_terminal
            else 'legacy_horizon' if horizon else 'legacy_extreme' if extreme else 'raw_center')
    adjustments = lambda debug: {r['direction']: r.get('search_adjustment', 0) for r in debug.get('deep_moves', [])}
    order = lambda debug: [r['direction'] for r in debug.get('deep_moves', [])]
    return {'adjustment_changed': adjustments(reference) != adjustments(candidate),
            'order_changed': order(reference) != order(candidate), 'final_movement_changed': ref_move != cand_move,
            'horizon': horizon, 'root_terminal': root_terminal, 'applicable_extreme': extreme,
            'forced_loss': any(row.get('forced_loss') for row in rows),
            'terminal_class': replay.terminal_class(raw, remaining, depth), 'guard_mode': mode}


def compare_decision(actual, alternate, planner, weights, data, label, tape=None, fixture_repeat=False, choose=None):
    tape = tape if tape is not None else RawSearchTape()
    # Save the complete frame before the original choose_move updates its clock.
    frame = copy.deepcopy(actual.__dict__)
    choose = choose or actual.choose_move
    ranking = choose.__func__.__globals__['rank_deep_moves'].__globals__
    ranking['IterativeSearchEngine'] = tape
    started = perf_counter()
    direction = choose(copy.deepcopy(data))
    direction = actual.safe_direction(data, direction)
    actual_ms = (perf_counter() - started) * 1000
    tape.assert_consumed()
    shadow_tape = tape.fresh()
    shadow = make_brain(alternate, planner, weights, shadow_tape)
    if fixture_repeat:
        shadow.__class__._repeat_count = replay.hashed_repeat_count
    shadow.__dict__ = frame
    started = perf_counter()
    other = shadow.safe_direction(data, shadow.choose_move(copy.deepcopy(data)))
    shadow_ms = (perf_counter() - started) * 1000
    shadow_tape.assert_consumed()
    actual_debug = normalized(actual.decision_debug(data['game_id']) or {})
    other_debug = normalized(shadow.decision_debug(data['game_id']) or {})
    for key in ('plan', 'legal', 'hunger', 'repeat_count', 'current_target', 'search_time_budget_ms'):
        if actual_debug.get(key) != other_debug.get(key):
            raise ValueError('same-state diagnostic input mismatch: ' + key)
    reference, candidate = (actual_debug, other_debug) if label == 'reference' else (other_debug, actual_debug)
    ref_move, cand_move = (direction, other) if label == 'reference' else (other, direction)
    return {'actual_revision': label, 'snapshot': normalized(data), 'reference_debug': reference,
            'candidate_debug': candidate, 'reference_direction': ref_move, 'candidate_direction': cand_move,
            'actual_direction': direction, 'shadow_mode': 'shared_actual_root_result',
            'search_consumed': True, 'search_tape': normalized(tape.entries),
            'classification': classify(reference, candidate, ref_move, cand_move, weights),
            'timing_ms': {'actual': actual_ms, 'shadow_rerank': shadow_ms}}


def prove_case(case, reference, candidate, planner, weights):
    tape = RawSearchTape(engine=replay.FrozenEngine(case))
    brain = make_brain(reference, planner, weights, tape)
    brain.__class__._repeat_count = replay.hashed_repeat_count
    _, verification = replay.hydrate_brain(case, brain)
    row = compare_decision(brain, candidate, planner, weights, case['observed'], 'reference', tape, True)
    if len(tape.entries) != 1:
        raise ValueError('known fixture must consume exactly one frozen root result')
    raw = tape.entries[0]['result']['scores']
    for debug in (row['reference_debug'], row['candidate_debug']):
        if {r['direction']: r['search_score'] for r in debug['deep_moves'] if r['search_score'] is not None} != raw:
            raise ValueError('frozen raw results changed')
    return dict(row, verified=True, provenance=case['provenance'], verification=verification,
                raw_scores=raw, timed_search_executed=False,
                historical_request_certified=False,
                scope='current X64 baseline/candidate shared frozen raw; historical internal caches not certified')


def verify_inputs(config):
    hashes = {path: sha256(path) if Path(path).exists() else None for path in config['input_hashes']}
    if hashes != config['input_hashes']:
        raise ValueError('input source/weights/active file hashes changed')
    return {'verified': True, 'hashes': hashes, 'checked_at': datetime.now(timezone.utc).isoformat()}


def compare_game(seed, candidate_side, reference, candidate, planner, weights, emit, max_turns=400):
    sides, counts, timing, first = iter(('A', 'B')), Counter(), Counter(), {}

    def factory(enable_debug=False):
        side = next(sides)
        label = 'candidate' if side == candidate_side else 'reference'
        source, alternate = (candidate, reference) if label == 'candidate' else (reference, candidate)
        base = make_brain(source, planner, weights).__class__

        class ComparedBrain(base):
            def choose_move(self, data):
                row = compare_decision(self, alternate, planner, weights, data, label,
                                       choose=base.choose_move.__get__(self, type(self)))
                row.update(actor=side, turn=int(data['turn_token'].split('_')[-1]))
                flags = row['classification']
                counts.update({key: int(value) for key, value in flags.items() if isinstance(value, bool)})
                counts['guard/' + flags['guard_mode']] += 1
                counts['category/' + flags['terminal_class']] += 1
                timing.update(row['timing_ms'])
                for key in ('adjustment_changed', 'order_changed', 'final_movement_changed'):
                    if flags[key] and key not in first:
                        first[key] = row
                emit(row)
                return row['actual_direction']

        return ComparedBrain(enable_debug=True)

    namespace = private_namespace(bot_lab, planner, MappingProxyType(dict(weights)))
    namespace['SnakeBrain'] = factory
    started = perf_counter()
    with shared_weights(weights):
        result = namespace['simulate_game'](seed, max_turns=max_turns, candidate_side=candidate_side)
    return {'match': asdict(result), 'counts': dict(counts),
            'timing': dict(timing, wall_ms=(perf_counter() - started) * 1000), 'first_divergences': first}


def summarize(games, expected_games):
    result = shared.summarize(games, expected_games)
    return {'competition': result['candidate'], 'candidate_by_side': result['candidate_by_side'],
            'mirror_pairs': result['mirror_pairs'], 'completed_games': len(games), 'expected_games': expected_games,
            'complete': len(games) == expected_games, 'counts': result['counts'], 'timing_ms': result['timing_ms'],
            'limitations': ['Timed real matches are unreplicated; diagnostics share actual root results per decision.',
                            'Same-state counterfactuals do not predict alternate match trajectories.']}


def run_comparison(config, reference, candidate, planner, weights, output):
    verify_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'manifest.json', config)
    games = []
    for seed in config['seeds']:
        for side in ('A', 'B'):
            stem = f'seed_{seed}_candidate_{side}'
            try:
                verify_inputs(config)
                with (output / (stem + '.jsonl')).open('x', encoding='utf-8') as stream:
                    def emit(row):
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
                        stream.flush()
                    game = compare_game(seed, side, reference, candidate, planner, weights, emit, config['turns'])
                verified = verify_inputs(config)
                atomic_json(output / (stem + '.json'), game)
                games.append(game)
                atomic_json(output / 'post_run_hashes.json', verified)
                atomic_json(output / 'summary.json', summarize(games, config['games']))
                print(f'{len(games)}/{config["games"]} seed={seed} candidate={side} winner={game["match"]["winner"]}', flush=True)
            except BaseException as error:
                atomic_json(output / 'error.json', {'seed': seed, 'candidate_side': side,
                            'type': type(error).__name__, 'message': str(error)})
                atomic_json(output / 'summary.json', summarize(games, config['games']))
                raise
    return summarize(games, config['games'])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=('prepare', 'run'), default='prepare')
    result.add_argument('--reference', type=Path, default=BASELINE)
    result.add_argument('--candidate', type=Path, default=ROOT / 'snake_bot/move_search.py')
    result.add_argument('--candidate-sha', required=True)
    result.add_argument('--planner', type=Path, default=ROOT / 'snake_bot/food_planner.py')
    result.add_argument('--weights', type=Path, required=True)
    result.add_argument('--fixtures', type=Path, default=ROOT / 'tests/fixtures/deep_clipping_real_v7.json')
    result.add_argument('--shared-manifest', type=Path, default=ROOT / 'games/benchmarks/planner_x_12seeds_20261012/manifest.json')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--seeds', type=int, nargs='+', default=list(range(20261036, 20261048)))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    seeds = shared.validate_seeds(args.seeds)
    candidate_sha = args.candidate_sha.lower()
    if len(candidate_sha) != 64 or any(c not in '0123456789abcdef' for c in candidate_sha):
        raise ValueError('candidate SHA256 must be explicit hexadecimal digest')
    if args.candidate.resolve() != (ROOT / 'snake_bot/move_search.py').resolve():
        raise ValueError('candidate must be actual snake_bot/move_search.py')
    paths = [args.reference, args.candidate, args.weights, args.fixtures, args.planner, args.shared_manifest,
             Path(__file__), Path(shared.__file__), Path(replay.__file__), Path(bot_lab.__file__),
             *sorted((ROOT / 'snake_bot').glob('*.py')), ROOT / 'weights/active_weights.json']
    hashes = {str(path.resolve()): sha256(path) if path.exists() else None for path in paths}
    if sha256(args.weights) != WEIGHTS_SHA or sha256(ROOT / 'snake_bot/search_engine.py') != ENGINE_SHA:
        raise ValueError('weights/engine SHA256 mismatch')
    accepted = json.loads(args.shared_manifest.read_text())
    shared_sources = {str(path.relative_to(ROOT)): sha256(path)
                      for path in sorted((ROOT / 'snake_bot').glob('*.py'))
                      if path.name not in ('move_search.py', 'food_planner.py')}
    expected_sources = {name: digest for name, digest in accepted['shared_sources'].items()
                        if Path(name).name not in ('move_search.py', 'food_planner.py')}
    if shared_sources != expected_sources or sha256(bot_lab.__file__) != accepted['simulator_sha256']:
        raise ValueError('shared source/simulator differs from accepted X64 manifest')
    changed = validate_candidate(args.reference.read_text(), args.candidate.read_text())
    weights = shared.load_weights(args.weights)
    if weights['iterative_search_time_ms'] != 45 or weights['iterative_search_max_depth'] != 6:
        raise ValueError('search budget/depth changed')
    reference = load_planner(args.reference, REFERENCE_SHA, weights)
    candidate = load_planner(args.candidate, candidate_sha, weights)
    planner = load_planner(args.planner, PLANNER_SHA, weights)
    fixtures = json.loads(args.fixtures.read_text())['cases']
    case = next(c for c in fixtures if c['provenance']['game_id'].startswith('8d25279e') and c['provenance']['debug_line'] == 498)
    verify_inputs({'input_hashes': hashes})
    proof = prove_case(case, reference, candidate, planner, weights)
    config = {'seeds': seeds, 'games': 24, 'turns': 400, 'mode': args.mode,
              'created_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(), 'python': sys.version,
              'reference_sha256': REFERENCE_SHA, 'candidate_sha256': candidate_sha, 'planner_sha256': PLANNER_SHA,
              'weights_sha256': WEIGHTS_SHA, 'engine_sha256': ENGINE_SHA, 'changed_definitions': changed,
              'effective_weights_sha256': hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest(),
              'search_budget_ms': 45, 'max_depth': 6, 'input_hashes': hashes, 'proof': proof,
              'method': 'each real brain own timed engine; opposite rerank shares actual root result and cloned predecision history',
              'resume_supported': False}
    verify_inputs(config)
    if args.mode == 'prepare':
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / 'manifest.json', config)
        print(json.dumps({'prepared': True, 'candidate_sha256': candidate_sha,
                          'proof': proof['classification'], 'games_executed': 0}, indent=2))
    else:
        print(json.dumps(run_comparison(config, reference, candidate, planner, weights, args.output), indent=2))


if __name__ == '__main__':
    main()
