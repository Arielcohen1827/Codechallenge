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

from snake_bot import snake_brain
from snake_bot.search_engine import IterativeSearchEngine
from tools import (
    bot_lab,
    compare_food_planners as shared,
    deep_clipping_replay as replay,
)
from tools.compare_food_planners import (
    atomic_json,
    load_planner,
    normalized,
    private_class,
    private_namespace,
    sha256,
    shared_weights,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = replay.historical_support() / 'external'
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
        """Wrap the engine with a raw-search recording or replay tape."""
        if self.replaying:
            return super().from_game_state(state, time_budget_ms, max_depth)
        engine, compact = self.engine.from_game_state(state, time_budget_ms, max_depth)

        def search(compact_state, side, legal, target=None):
            """Capture a raw root result or replay one previously verified request."""
            result = engine.search(compact_state, side, legal, target=target)
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
            self.entries.append({'request': request, 'result': asdict(result)})
            self.index += 1
            return result

        return SimpleNamespace(search=search), compact


def make_brain(source, planner, weights, tape=None):
    """Build isolated historical ranking revisions around the same planner and weights."""
    weights = MappingProxyType(dict(weights))
    ranking = private_namespace(source, planner, weights)
    cache = private_class(source.TurnSearchCache, ranking)
    ranking['TurnSearchCache'] = cache
    ranking['IterativeSearchEngine'] = tape if tape is not None else IterativeSearchEngine
    namespace = private_namespace(snake_brain, planner, weights)
    namespace.update(TurnSearchCache=cache, rank_deep_moves=ranking['rank_deep_moves'])
    return private_class(snake_brain.SnakeBrain, namespace)(enable_debug=True)


def validate_candidate(before, after):
    """Require exactly the approved normalization helper and call-site AST change."""
    old, new = ast.parse(before), ast.parse(after)

    def dump(node):
        """Normalize an AST node for a formatting-independent source comparison."""
        return ast.dump(node, include_attributes=False)

    new.body.remove(normalization_helper(old, new))
    old_rank = named_definition(old, 'rank_deep_moves')
    new_rank = named_definition(new, 'rank_deep_moves')
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
        raise ValueError(
            ('candidate normalization call/body differs from approved ' 'minimal replacement')
        )
    new_rank.body[-3] = copy.deepcopy(old_rank.body[-3])
    if dump(old) != dump(new):
        raise ValueError('candidate changes outside normalization helper/call')
    return ['_search_adjustments', 'rank_deep_moves']


def named_definition(module, name):
    """Find a required top-level definition in an audited historical source."""
    return next(node for node in module.body if getattr(node, 'name', None) == name)


def normalization_helper(old, new):
    """Require exactly the approved added helper before comparing remaining AST."""
    helpers = [
        node
        for node in new.body
        if isinstance(node, ast.FunctionDef) and node.name == '_search_adjustments'
    ]
    validate_added_helper(helpers, old)
    return helpers[0]


def validate_added_helper(helpers, old):
    """Reject duplicate helpers and a baseline that already contains the addition."""
    existing = any(getattr(node, 'name', None) == '_search_adjustments' for node in old.body)
    if len(helpers) != 1 or existing:
        raise ValueError('candidate must add only the _search_adjustments normalization helper')


def classify(reference, candidate, ref_move, cand_move, weights):
    """Describe captured ranking changes and the historical normalization guard applied."""
    rows = reference.get('deep_moves', [])
    raw = applicable_scores(rows)
    depth = max((row.get('search_depth', 0) for row in rows), default=0)
    remaining = reference.get('remaining_moves')
    root_terminal = terminal_debug_state(reference, remaining)
    horizon = remaining is not None and remaining <= depth
    extreme = any(abs(score) >= 300000000 for score in raw)
    mode = normalization_mode(rows, depth, raw, weights, root_terminal, horizon, extreme)

    def adjustments(debug):
        """Extract per-direction normalization adjustments from captured diagnostics."""
        return {r['direction']: r.get('search_adjustment', 0) for r in debug.get('deep_moves', [])}

    def order(debug):
        """Extract the ranking order without changing any recorded scores."""
        return [r['direction'] for r in debug.get('deep_moves', [])]

    return {
        'adjustment_changed': adjustments(reference) != adjustments(candidate),
        'order_changed': order(reference) != order(candidate),
        'final_movement_changed': ref_move != cand_move,
        'horizon': horizon,
        'root_terminal': root_terminal,
        'applicable_extreme': extreme,
        'forced_loss': any(row.get('forced_loss') for row in rows),
        'terminal_class': replay.terminal_class(raw, remaining, depth),
        'guard_mode': mode,
    }


def applicable_scores(rows):
    """Select captured scores eligible for the historical normalization comparison."""
    return [
        row['search_score']
        for row in rows
        if row.get('search_score') is not None and not row.get('forced_loss')
    ]


def terminal_debug_state(debug, remaining):
    """Recognize an already-finished root from its recorded public counters."""
    return (remaining is not None and remaining <= 0) or any(
        score < -2500 for score in debug.get('scores', {}).values()
    )


def normalization_mode(rows, depth, raw, weights, terminal, horizon, extreme):
    """Explain which historical normalization gate applies to captured results."""
    if not rows:
        return 'no_search'
    if depth < weights['iterative_search_min_apply_depth']:
        return 'depth_gate'
    if not raw:
        return 'no_applicable'
    return legacy_normalization_mode(terminal, horizon, extreme)


def legacy_normalization_mode(terminal, horizon, extreme):
    """Preserve terminal, horizon and extreme-score gate priority."""
    if terminal:
        return 'legacy_root_terminal'
    if horizon:
        return 'legacy_horizon'
    return 'legacy_extreme' if extreme else 'raw_center'


def compare_decision(
    actual,
    alternate,
    planner,
    weights,
    data,
    label,
    tape=None,
    fixture_repeat=False,
    choose=None,
):
    """Compare reranking from an identical predecision frame and shared raw root results."""
    tape = tape if tape is not None else RawSearchTape()
    # Save the complete frame before the original choose_move updates its
    # clock.
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
    actual_debug = normalized_debug(actual, data['game_id'])
    other_debug = normalized_debug(shadow, data['game_id'])
    verify_same_state_debug(actual_debug, other_debug)
    ordered = ((actual_debug, direction), (other_debug, other))
    if label != 'reference':
        ordered = ordered[::-1]
    (reference, ref_move), (candidate, cand_move) = ordered
    return {
        'actual_revision': label,
        'snapshot': normalized(data),
        'reference_debug': reference,
        'candidate_debug': candidate,
        'reference_direction': ref_move,
        'candidate_direction': cand_move,
        'actual_direction': direction,
        'shadow_mode': 'shared_actual_root_result',
        'search_consumed': True,
        'search_tape': normalized(tape.entries),
        'classification': classify(reference, candidate, ref_move, cand_move, weights),
        'timing_ms': {'actual': actual_ms, 'shadow_rerank': shadow_ms},
    }


def normalized_debug(brain, game_id):
    """Snapshot diagnostic data without retaining live mutable bot references."""
    return normalized(brain.decision_debug(game_id) or {})


def verify_same_state_debug(actual_debug, other_debug):
    """Require equal non-normalization inputs for a meaningful reranking comparison."""
    for key in (
        'plan',
        'legal',
        'hunger',
        'repeat_count',
        'current_target',
        'search_time_budget_ms',
    ):
        if actual_debug.get(key) != other_debug.get(key):
            raise ValueError('same-state diagnostic input mismatch: ' + key)


def prove_case(case, reference, candidate, planner, weights):
    """Certify an archived case using its frozen root scores without executing a timed search."""
    tape = RawSearchTape(engine=replay.FrozenEngine(case))
    brain = make_brain(reference, planner, weights, tape)
    brain.__class__._repeat_count = replay.hashed_repeat_count
    _, verification = replay.hydrate_brain(case, brain)
    row = compare_decision(
        brain,
        candidate,
        planner,
        weights,
        case['observed'],
        'reference',
        tape,
        True,
    )
    if len(tape.entries) != 1:
        raise ValueError('known fixture must consume exactly one frozen root result')
    raw = tape.entries[0]['result']['scores']
    for debug in (row['reference_debug'], row['candidate_debug']):
        if frozen_scores(debug) != raw:
            raise ValueError('frozen raw results changed')
    return dict(
        row,
        verified=True,
        provenance=case['provenance'],
        verification=verification,
        raw_scores=raw,
        timed_search_executed=False,
        historical_request_certified=False,
        scope=(
            'current X64 baseline/candidate shared frozen raw; historical '
            'internal caches not certified'
        ),
    )


def frozen_scores(debug):
    """Extract recorded raw scores without applying a normalization transform."""
    return {
        r['direction']: r['search_score']
        for r in debug['deep_moves']
        if r['search_score'] is not None
    }


def verify_inputs(config):
    """Reject a run if any source or weight hash differs from its captured manifest."""
    hashes = {
        path: sha256(path) if Path(path).exists() else None for path in config['input_hashes']
    }
    if hashes != config['input_hashes']:
        raise ValueError('input source/weights/active file hashes changed')
    return {
        'verified': True,
        'hashes': hashes,
        'checked_at': datetime.now(timezone.utc).isoformat(),
    }


def compare_game(
    seed,
    candidate_side,
    reference,
    candidate,
    planner,
    weights,
    emit,
    max_turns=400,
):
    """Run explicitly requested isolated revisions with per-turn shared-root evidence."""
    sides, counts, timing, first = iter(('A', 'B')), Counter(), Counter(), {}

    def factory(enable_debug=False):
        """Assign an isolated historical ranking revision to the requested candidate side."""
        side = next(sides)
        label = 'candidate' if side == candidate_side else 'reference'
        source, alternate = (
            (candidate, reference) if label == 'candidate' else (reference, candidate)
        )
        base = make_brain(source, planner, weights).__class__

        class ComparedBrain(base):
            """Collect reranking evidence while committing actual match decisions."""

            def choose_move(self, data):
                """Compare the actual decision against an isolated reranking from equal history."""
                row = compare_decision(
                    self,
                    alternate,
                    planner,
                    weights,
                    data,
                    label,
                    choose=base.choose_move.__get__(self, type(self)),
                )
                row.update(actor=side, turn=int(data['turn_token'].split('_')[-1]))
                flags = row['classification']
                update_boolean_counts(counts, flags)
                counts['guard/' + flags['guard_mode']] += 1
                counts['category/' + flags['terminal_class']] += 1
                timing.update(row['timing_ms'])
                for key in (
                    'adjustment_changed',
                    'order_changed',
                    'final_movement_changed',
                ):
                    if flags[key] and key not in first:
                        first[key] = row
                emit(row)
                return row['actual_direction']

        return ComparedBrain(enable_debug=True)

    namespace = private_namespace(bot_lab, planner, MappingProxyType(dict(weights)))
    namespace['SnakeBrain'] = factory
    started = perf_counter()
    with shared_weights(weights):
        result = namespace['simulate_game'](
            seed, max_turns=max_turns, candidate_side=candidate_side
        )
    return {
        'match': asdict(result),
        'counts': dict(counts),
        'timing': dict(timing, wall_ms=(perf_counter() - started) * 1000),
        'first_divergences': first,
    }


def update_boolean_counts(counts, flags):
    """Count classification booleans while leaving named categories separate."""
    counts.update({key: int(value) for key, value in flags.items() if isinstance(value, bool)})


def summarize(games, expected_games):
    """Report historical comparison counts and timings without changing production strategy."""
    result = shared.summarize(games, expected_games)
    return {
        'competition': result['candidate'],
        'candidate_by_side': result['candidate_by_side'],
        'mirror_pairs': result['mirror_pairs'],
        'completed_games': len(games),
        'expected_games': expected_games,
        'complete': len(games) == expected_games,
        'counts': result['counts'],
        'timing_ms': result['timing_ms'],
        'limitations': [
            (
                'Timed real matches are unreplicated; diagnostics share '
                'actual root results per decision.'
            ),
            ('Same-state counterfactuals do not predict alternate match ' 'trajectories.'),
        ],
    }


def run_comparison(config, reference, candidate, planner, weights, output):
    """Execute a requested comparison while retaining incremental reports and input checks."""
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
                        """Append decisions and preserve incremental evidence."""
                        stream.write(json.dumps(row, sort_keys=True) + '\n')
                        stream.flush()

                    game = compare_game(
                        seed,
                        side,
                        reference,
                        candidate,
                        planner,
                        weights,
                        emit,
                        config['turns'],
                    )
                verified = verify_inputs(config)
                atomic_json(output / (stem + '.json'), game)
                games.append(game)
                atomic_json(output / 'post_run_hashes.json', verified)
                atomic_json(output / 'summary.json', summarize(games, config['games']))
                print(
                    (
                        f'{len(games)}/{config["games"]} seed={seed} '
                        f'candidate={side} winner={game["match"]["winner"]}'
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
                atomic_json(output / 'summary.json', summarize(games, config['games']))
                raise
    return summarize(games, config['games'])


def parser():
    """Define explicit source, digest, seed and output requirements for the historical CLI."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=('prepare', 'run'), default='prepare')
    result.add_argument('--reference', type=Path, default=BASELINE)
    result.add_argument('--candidate', type=Path, default=ROOT / 'snake_bot/move_search.py')
    result.add_argument('--candidate-sha', required=True)
    result.add_argument('--planner', type=Path, default=ROOT / 'snake_bot/food_planner.py')
    result.add_argument('--weights', type=Path, required=True)
    result.add_argument(
        '--fixtures',
        type=Path,
        default=ROOT / 'tests/fixtures/deep_clipping_real_v7.json',
    )
    result.add_argument(
        '--shared-manifest',
        type=Path,
        default=ROOT / 'games/benchmarks/planner_x_12seeds_20261012/manifest.json',
    )
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--seeds', type=int, nargs='+', default=list(range(20261036, 20261048)))
    return result


def main(argv=None):
    """Run the historical comparison CLI only after all provenance checks."""
    NormalizationComparison(argv).run()


class NormalizationComparison:
    """Keep historical comparison preparation separate from its explicit execution."""

    def __init__(self, argv=None):
        """Own the command arguments and resolved audited inputs."""
        self.args = parser().parse_args(argv)

    def run(self):
        """Validate all prerequisites in their original order before execution."""
        self.validate_request()
        self.capture_inputs()
        self.verify_shared_sources()
        self.load_revisions()
        self.select_fixture()
        self.prove_fixture()
        self.build_manifest()
        self.execute()

    def validate_request(self):
        """Validate the explicit seed list and audited candidate identity."""
        self.seeds = shared.validate_seeds(self.args.seeds)
        self.candidate_sha = self.args.candidate_sha.lower()
        if len(self.candidate_sha) != 64 or any(
            (c not in '0123456789abcdef' for c in self.candidate_sha)
        ):
            raise ValueError('candidate SHA256 must be explicit hexadecimal digest')
        if self.args.candidate.resolve() != (ROOT / 'snake_bot/move_search.py').resolve():
            raise ValueError('candidate must be actual snake_bot/move_search.py')

    def capture_inputs(self):
        """Capture input hashes and enforce the frozen engine and weights."""
        self.paths = [
            self.args.reference,
            self.args.candidate,
            self.args.weights,
            self.args.fixtures,
            self.args.planner,
            self.args.shared_manifest,
            Path(__file__),
            Path(shared.__file__),
            Path(replay.__file__),
            Path(bot_lab.__file__),
            *sorted((ROOT / 'snake_bot').glob('*.py')),
            ROOT / 'weights/active_weights.json',
        ]
        self.hashes = {
            str(path.resolve()): sha256(path) if path.exists() else None for path in self.paths
        }
        if (
            sha256(self.args.weights) != WEIGHTS_SHA
            or sha256(ROOT / 'snake_bot/search_engine.py') != ENGINE_SHA
        ):
            raise ValueError('weights/engine SHA256 mismatch')

    def verify_shared_sources(self):
        """Require the accepted shared simulator and rule-source manifest."""
        self.accepted = json.loads(self.args.shared_manifest.read_text())
        self.shared_sources = current_shared_sources()
        self.expected_sources = accepted_shared_sources(self.accepted)
        if (
            self.shared_sources != self.expected_sources
            or sha256(bot_lab.__file__) != self.accepted['simulator_sha256']
        ):
            raise ValueError('shared source/simulator differs from accepted X64 manifest')

    def load_revisions(self):
        """Load only the approved planner and ranking source revisions."""
        self.changed = validate_candidate(
            self.args.reference.read_text(), self.args.candidate.read_text()
        )
        self.weights = shared.load_weights(self.args.weights)
        if (
            self.weights['iterative_search_time_ms'] != 45
            or self.weights['iterative_search_max_depth'] != 6
        ):
            raise ValueError('search budget/depth changed')
        self.reference = load_planner(self.args.reference, REFERENCE_SHA, self.weights)
        self.candidate = load_planner(self.args.candidate, self.candidate_sha, self.weights)
        self.planner = load_planner(self.args.planner, PLANNER_SHA, self.weights)

    def select_fixture(self):
        """Select the named historical fixture required for this comparison."""
        self.fixtures = json.loads(self.args.fixtures.read_text())['cases']
        self.case = next(
            (
                c
                for c in self.fixtures
                if c['provenance']['game_id'].startswith('8d25279e')
                and c['provenance']['debug_line'] == 498
            )
        )

    def prove_fixture(self):
        """Certify the frozen fixture before any optional comparison run."""
        verify_inputs({'input_hashes': self.hashes})
        self.proof = prove_case(
            self.case, self.reference, self.candidate, self.planner, self.weights
        )

    def build_manifest(self):
        """Record immutable comparison provenance and verify all inputs."""
        self.config = {
            'seeds': self.seeds,
            'games': 24,
            'turns': 400,
            'mode': self.args.mode,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(),
            'python': sys.version,
            'reference_sha256': REFERENCE_SHA,
            'candidate_sha256': self.candidate_sha,
            'planner_sha256': PLANNER_SHA,
            'weights_sha256': WEIGHTS_SHA,
            'engine_sha256': ENGINE_SHA,
            'changed_definitions': self.changed,
            'effective_weights_sha256': hashlib.sha256(
                json.dumps(self.weights, sort_keys=True).encode()
            ).hexdigest(),
            'search_budget_ms': 45,
            'max_depth': 6,
            'input_hashes': self.hashes,
            'proof': self.proof,
            'method': (
                'each real brain own timed engine; opposite rerank shares actual root result and '
                'cloned predecision history'
            ),
            'resume_supported': False,
        }
        verify_inputs(self.config)

    def execute(self):
        """Prepare the manifest or invoke the explicitly requested runner."""
        if self.args.mode == 'prepare':
            self.args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(self.args.output / 'manifest.json', self.config)
            print(
                json.dumps(
                    {
                        'prepared': True,
                        'candidate_sha256': self.candidate_sha,
                        'proof': self.proof['classification'],
                        'games_executed': 0,
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    run_comparison(
                        self.config,
                        self.reference,
                        self.candidate,
                        self.planner,
                        self.weights,
                        self.args.output,
                    ),
                    indent=2,
                )
            )


def current_shared_sources():
    """Hash rule sources shared by both historical ranking revisions."""
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted((ROOT / 'snake_bot').glob('*.py'))
        if path.name not in ('move_search.py', 'food_planner.py')
    }


def accepted_shared_sources(accepted):
    """Normalize accepted manifest separators before comparing source hashes."""
    return {
        name.replace('\\', '/'): digest
        for name, digest in accepted['shared_sources'].items()
        if Path(name.replace('\\', '/')).name not in ('move_search.py', 'food_planner.py')
    }


if __name__ == '__main__':
    main()
