import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snake_bot import bot_weights
from tools import optimize_weights as optimizer


def competitive(seeds=20, repetitions=1):
    total = seeds * repetitions * 2
    return {
        'total': total,
        'wins': total,
        'losses': 0,
        'draws': 0,
        'win_rate': 1,
        'win_rate_without_draws': 1,
        'mean_score_diff': 10,
        'median_score_diff': 10,
        'by_candidate_side': {
            side: {
                'total': total // 2,
                'wins': total // 2,
                'losses': 0,
                'draws': 0,
                'mean_score_diff': 10,
            }
            for side in ('A', 'B')
        },
        'pairs': {
            'records': [
                {
                    'seed': seed,
                    'repetition': rep,
                    'candidate_sides': ['A', 'B'],
                    'wins': 2,
                    'losses': 0,
                    'draws': 0,
                    'score_diff_sum': 20,
                }
                for seed in range(seeds)
                for rep in range(1, repetitions + 1)
            ]
        },
        'events': {'candidate': {'wall_hits': 100}},
    }


class OptimizerTests(unittest.TestCase):
    def test_ranking_ignores_diagnostics_and_prioritizes_competition(self):
        winner = competitive()
        loser = dict(competitive(), wins=0, losses=40, mean_score_diff=1000000,
                     survival=1000000, logs={'food': 999})
        self.assertGreater(optimizer.competitive_rank(winner), optimizer.competitive_rank(loser))
        copied = dict(winner, events={}, survival=-999999, logs={})
        self.assertEqual(optimizer.competitive_rank(winner), optimizer.competitive_rank(copied))

    def test_candidate_is_local_seeded_and_never_mutates_base(self):
        base = {'center_unit_value': 80, 'iterative_search_time_ms': 45}
        bounds = {'center_unit_value': (72, 88)}
        first = optimizer.random_candidate(base, random.Random(7), bounds)
        self.assertEqual(first, optimizer.random_candidate(base, random.Random(7), bounds))
        self.assertTrue(72 <= first['center_unit_value'] <= 88)
        self.assertEqual(first['iterative_search_time_ms'], 45)
        self.assertEqual(base['center_unit_value'], 80)

    def test_evidence_uses_unique_seeds_not_repetitions(self):
        evidence = optimizer.improvement_evidence(competitive(seeds=2, repetitions=10), 2)
        self.assertFalse(evidence['valid_improvement'])
        self.assertEqual(evidence['seed_groups'], 2)
        self.assertEqual(evidence['informative_seed_groups'], 2)

    def test_evidence_reports_side_bias_without_requiring_both_sides_to_win(self):
        data = competitive()
        data['by_candidate_side']['B'].update(wins=0, draws=20)
        data.update(wins=20, draws=20)
        for pair in data['pairs']['records']:
            pair.update(wins=1, draws=1)
        self.assertTrue(optimizer.improvement_evidence(data, 3)['valid_improvement'])
        data['by_candidate_side']['B']['total'] = 19
        self.assertFalse(optimizer.improvement_evidence(data, 3)['valid_improvement'])

    def test_plan_counts_every_mirrored_game_at_400_turns(self):
        plan = optimizer.build_plan(trials=6, top_b=3, top_c=2, a_seeds=2,
                                    b_seeds=4, final_seeds=20, repetitions=2,
                                    seed=7, seconds_per_game=1, budget_minutes=5)
        self.assertEqual(plan['estimated_games'], 208)
        self.assertEqual(plan['estimated_seconds'], 208)
        self.assertEqual(plan['turns'], 400)
        stages = plan['stages']
        self.assertLess(stages[0]['seed'] + stages[0]['seeds'] - 1, stages[1]['seed'])
        self.assertLess(stages[1]['seed'] + stages[1]['seeds'] - 1, stages[2]['seed'])

    def test_plan_rejects_insufficient_budget_and_explains_estimate(self):
        with self.assertRaises(ValueError):
            optimizer.build_plan(seconds_per_game=60, budget_minutes=1)
        plan = optimizer.build_plan(seconds_per_game=1, budget_minutes=2)
        self.assertLessEqual(plan['estimated_seconds'], 96)
        self.assertIn('approximate', plan['budget_note'])

    def test_cli_rejects_activation_missing_base_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"center_unit_value":80}')
            bounds = root / 'bounds.json'
            bounds.write_text('{"center_unit_value":[72,88]}')
            common = [
                '--base',
                str(base),
                '--bounds',
                str(bounds),
                '--output-dir',
                str(root / 'out'),
            ]
            for extra in (
                ['--activate'],
                ['--base', str(root / 'missing.json')],
                ['--output-dir', str(root)],
                ['--output-dir', str(base)],
                [
                    '--output-dir',
                    str(root / 'weights' / 'active_weights.json'),
                ],
            ):
                with (
                    self.subTest(extra=extra),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        optimizer.main(common + extra)

    def test_cli_dry_run_does_not_write_or_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"center_unit_value":80}')
            bounds = root / 'bounds.json'
            bounds.write_text('{"center_unit_value":[72,88]}')
            with (
                patch.object(
                    optimizer, 'run_benchmark', create=True
                ) as benchmark,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                optimizer.main(
                    [
                        '--base',
                        str(base),
                        '--bounds',
                        str(bounds),
                        '--output-dir',
                        str(root / 'out'),
                        '--dry-run',
                    ]
                )
            benchmark.assert_not_called()
            self.assertFalse((root / 'out').exists())
            self.assertEqual(
                json.loads(output.getvalue())['plan']['turns'], 400
            )

    def test_progressive_run_saves_all_candidates_and_stage_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"center_unit_value":80}')
            output = root / 'out'
            plan = optimizer.build_plan(
                trials=6,
                top_b=3,
                top_c=2,
                a_seeds=1,
                b_seeds=2,
                final_seeds=20,
                seed=10,
            )
            calls = []

            def benchmark(**kwargs):
                calls.append(kwargs)
                summary = competitive(kwargs['seeds'], kwargs['repetitions'])
                for pair in summary['pairs']['records']:
                    pair['seed'] += kwargs['seed']
                report = {
                    'summary': summary,
                    'manifest': {
                        'seeds': list(
                            range(
                                kwargs['seed'],
                                kwargs['seed'] + kwargs['seeds'],
                            )
                        )
                    },
                }
                Path(kwargs['output']).write_text(json.dumps(report))
                bot_weights.ACTIVE_WEIGHTS['center_unit_value'] = -123
                return report

            previous = dict(bot_weights.ACTIVE_WEIGHTS)
            with patch.object(
                optimizer, 'run_benchmark', side_effect=benchmark
            ):
                result = optimizer.run_search(
                    base, output, {'center_unit_value': (72, 88)}, plan, 7
                )
            self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)
            self.assertEqual(len(calls), 11)
            self.assertTrue(all(call['turns'] == 400 for call in calls))
            self.assertEqual(
                [call['seed'] for call in calls],
                [10] * 6 + [11] * 3 + [13] * 2,
            )
            self.assertEqual(len(list(output.glob('candidate-*.json'))), 6)
            self.assertEqual(len(list(output.glob('benchmark-*.json'))), 11)
            self.assertTrue((output / 'manifest.json').is_file())
            self.assertTrue((output / 'summary.json').is_file())
            self.assertTrue(result['accepted_improvement'])
            self.assertTrue(
                all(
                    not item['valid_improvement']
                    for item in result['index']
                    if item['stage'] in ('A', 'B')
                )
            )
            self.assertEqual(
                json.loads(base.read_text()), {'center_unit_value': 80}
            )
            for candidate_id in {item['id'] for item in result['index']}:
                records = [
                    item
                    for item in result['index']
                    if item['id'] == candidate_id
                ]
                self.assertEqual(
                    len({item['weights_path'] for item in records}), 1
                )
                self.assertEqual(
                    len({item['candidate_value'] for item in records}), 1
                )

    def test_failure_restores_exact_memory_and_retains_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"center_unit_value":80}')
            previous = dict(bot_weights.ACTIVE_WEIGHTS)

            def broken(**kwargs):
                bot_weights.ACTIVE_WEIGHTS.clear()
                raise RuntimeError('benchmark failed')

            with patch.object(optimizer, 'run_benchmark', side_effect=broken):
                with self.assertRaisesRegex(RuntimeError, 'benchmark failed'):
                    optimizer.run_search(
                        base,
                        root / 'out',
                        {'center_unit_value': (72, 88)},
                        optimizer.build_plan(smoke=True),
                        7,
                    )
            self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)
            self.assertTrue((root / 'out' / 'candidate-001.json').exists())
            self.assertEqual(
                json.loads((root / 'out' / 'failure.json').read_text())[
                    'status'
                ],
                'failed',
            )

    def test_audit_defaults_are_current_centered_not_historical_intersections(self):
        base = dict(bot_weights.DEFAULT_WEIGHTS, food_distance_cost=899, race_margin_value=332,
                    fallback_cycle_penalty=376)
        bounds = optimizer.load_bounds(None, base)
        self.assertEqual(len(bounds), 16)
        self.assertEqual(bounds['food_distance_cost'], (810, 988))
        self.assertEqual(bounds['race_margin_value'], (299, 365))
        self.assertEqual(bounds['fallback_cycle_penalty'], (339, 413))
        self.assertNotIn('fallback_denial_reach_bonus', bounds)
        self.assertFalse(any(key.startswith('iterative_search_') for key in bounds))

    def test_only_one_audited_coordinate_changes_per_candidate(self):
        base = dict(bot_weights.DEFAULT_WEIGHTS)
        bounds = optimizer.load_bounds(None, base)
        for seed in range(20):
            candidate = optimizer.random_candidate(
                base, random.Random(seed), bounds
            )
            changed = [key for key in base if base[key] != candidate[key]]
            self.assertEqual(len(changed), 1)
            self.assertIn(changed[0], bounds)
            self.assertTrue(
                bounds[changed[0]][0]
                <= candidate[changed[0]]
                <= bounds[changed[0]][1]
            )

    def test_explicit_bounds_cannot_unfreeze_controls_or_exceed_local_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bounds.json'
            for payload in ({'race_margin_value': [630, 770]},
                            {'iterative_search_weight_percent': [34, 36]},
                            {'deep_extra_depth_margin': [2100, 2300]},
                            {'dangerous_penalty': [6400, 6600]},
                            {'food_distance_cost': [100, 1000]}):
                path.write_text(json.dumps(payload))
                if 'race_margin_value' in payload:
                    self.assertEqual(optimizer.load_bounds(path, bot_weights.DEFAULT_WEIGHTS),
                                     {'race_margin_value': (630, 770)})
                else:
                    with self.assertRaises(ValueError):
                        optimizer.load_bounds(path, bot_weights.DEFAULT_WEIGHTS)

    def test_budget_shrinks_final_seeds_but_keeps_at_least_eight_and_reserve(self):
        plan = optimizer.build_plan(seconds_per_game=1, budget_minutes=1)
        self.assertLessEqual(plan['estimated_games'], 48)
        self.assertGreaterEqual(plan['stages'][2]['seeds'], 8)
        self.assertGreaterEqual(plan['stages'][0]['candidates'], 4)
        self.assertEqual(plan['overhead_reserve_fraction'], 0.2)

    def test_repeat_regression_is_flagged_despite_positive_combined_record(self):
        data = competitive(seeds=20, repetitions=2)
        for pair in data['pairs']['records']:
            if pair['repetition'] == 2:
                pair.update(wins=0, losses=1, draws=1)
        data.update(wins=40, losses=20, draws=20)
        for side in data['by_candidate_side'].values():
            side.update(wins=20, losses=10, draws=10)
        result = optimizer.improvement_evidence(data, 2)
        self.assertFalse(result['valid_improvement'])
        self.assertIn('repeat_regression', result['reasons'])
        self.assertEqual(result['by_repetition']['2']['net_wins'], -20)

    def test_duplicate_or_incomplete_pairs_cannot_create_support(self):
        for change in ('duplicate', 'missing_side', 'missing_repeat'):
            data = competitive(repetitions=2)
            if change == 'duplicate':
                data['pairs']['records'].append(dict(data['pairs']['records'][0]))
            elif change == 'missing_side':
                data['pairs']['records'][0]['candidate_sides'] = ['A']
            else:
                data['pairs']['records'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                optimizer.improvement_evidence(data, 2)

    def test_tied_pairs_and_draws_are_explicitly_inconclusive(self):
        for tied_outcome in ('draws', 'split'):
            data = competitive(seeds=8)
            wins, losses, draws = (0, 0, 2) if tied_outcome == 'draws' else (1, 1, 0)
            for pair in data['pairs']['records']:
                pair.update(wins=wins, losses=losses, draws=draws)
            data.update(wins=wins * 8, losses=losses * 8, draws=draws * 8)
            for side in data['by_candidate_side'].values():
                side.update(wins=wins * 4, losses=losses * 4, draws=draws * 4)
            result = optimizer.improvement_evidence(data, 2)
            self.assertFalse(result['valid_improvement'])
            self.assertEqual(result['tied_seed_groups'], 8)
            self.assertIsNone(result['wilson_lower'])

    def test_stratified_candidates_cover_each_key_then_opposite_direction(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{}')
            bounds = optimizer.load_bounds(None, bot_weights.DEFAULT_WEIGHTS)
            plan = optimizer.build_plan(trials=32)
            with patch.object(
                optimizer, 'run_benchmark', side_effect=RuntimeError('stop')
            ):
                with self.assertRaises(RuntimeError):
                    optimizer.run_search(base, root / 'out', bounds, plan, 7)
            values = [
                json.loads(path.read_text())
                for path in sorted((root / 'out').glob('candidate-*.json'))
            ]
            changes = [
                {
                    key: value
                    for key, value in candidate.items()
                    if value != bot_weights.DEFAULT_WEIGHTS[key]
                }
                for candidate in values
            ]
            self.assertEqual(len(changes), 32)
            self.assertEqual(
                {next(iter(change)) for change in changes[:16]}, set(bounds)
            )
            for first, second in zip(changes[:16], changes[16:]):
                key = next(iter(first))
                center = bot_weights.DEFAULT_WEIGHTS[key]
                self.assertEqual(set(first), set(second))
                self.assertLess(
                    (first[key] - center) * (second[key] - center), 0
                )

    def test_benchmark_seed_grid_must_match_fresh_stage_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{}')
            with patch.object(
                optimizer,
                'run_benchmark',
                return_value={'summary': competitive(seeds=1)},
            ):
                with self.assertRaisesRegex(ValueError, 'seed'):
                    optimizer.run_search(
                        base,
                        root / 'out',
                        optimizer.load_bounds(
                            None, bot_weights.DEFAULT_WEIGHTS
                        ),
                        optimizer.build_plan(smoke=True, seed=1000),
                        7,
                    )

    def test_cli_smoke_has_six_mirrored_games_and_never_claims_improvement(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{}')
            calls = []

            def benchmark(**kwargs):
                calls.append(kwargs)
                summary = competitive(seeds=1)
                summary['pairs']['records'][0]['seed'] = kwargs['seed']
                report = {'summary': summary}
                Path(kwargs['output']).write_text(json.dumps(report))
                return report

            with (
                patch.object(
                    optimizer, 'run_benchmark', side_effect=benchmark
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = optimizer.main(
                    [
                        '--base',
                        str(base),
                        '--output-dir',
                        str(root / 'out'),
                        '--smoke',
                    ]
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(result['plan']['estimated_games'], 6)
            self.assertFalse(result['accepted_improvement'])
            self.assertFalse(result['best_candidate']['valid_improvement'])

    def test_cli_requires_timing_for_run_and_rejects_invalid_geometry_and_bounds(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{}')
            bounds = root / 'bad.json'
            bounds.write_text('[]')
            common = ['--base', str(base), '--output-dir', str(root / 'out')]
            for extra in (
                ['--run'],
                ['--food', '0'],
                ['--cols', '5'],
                ['--rows', '-1'],
                ['--bounds', str(bounds)],
                ['--seconds-per-game', 'nan', '--budget-minutes', '1'],
                ['--final-repetitions', '3'],
                ['--final-seeds', '7'],
                ['--trials', '0'],
                ['--seconds-per-game', '1'],
            ):
                with (
                    self.subTest(extra=extra),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        optimizer.main(common + extra)
            self.assertFalse((root / 'out').exists())

    def test_cli_run_uses_budget_plan_without_actual_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{}')
            with (
                patch.object(
                    optimizer,
                    'run_search',
                    return_value={'status': 'completed'},
                ) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                optimizer.main(
                    [
                        '--base',
                        str(base),
                        '--output-dir',
                        str(root / 'out'),
                        '--run',
                        '--seconds-per-game',
                        '1',
                        '--budget-minutes',
                        '2',
                    ]
                )
            self.assertLessEqual(
                run.call_args.args[3]['estimated_seconds'], 96
            )

    def test_exclusive_json_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'important.json'
            path.write_text('{"keep":true}')
            with self.assertRaises(FileExistsError):
                optimizer.write_json(path, {'keep': False})
            self.assertEqual(json.loads(path.read_text()), {'keep': True})

    def test_measured_throughput_rejects_hour_and_fits_eighty_game_three_hour_plan(self):
        with self.assertRaises(ValueError):
            optimizer.build_plan(seconds_per_game=104.69, budget_minutes=60)
        plan = optimizer.build_plan(trials=12, top_b=4, top_c=2, a_seeds=1,
                                    b_seeds=3, final_seeds=8, seconds_per_game=104.69,
                                    budget_minutes=180)
        self.assertEqual(plan['estimated_games'], 80)
        self.assertEqual([stage['games'] for stage in plan['stages']], [24, 24, 32])
        self.assertAlmostEqual(plan['estimated_seconds'], 8375.2)

    def test_exhausted_sparse_strata_use_remaining_mutations_deterministically(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"space_after_value":12,"center_unit_value":80}')
            original = base.read_bytes()
            previous = dict(bot_weights.ACTIVE_WEIGHTS)
            bounds = {
                'space_after_value': (11, 13),
                'center_unit_value': (72, 88),
            }
            plan = optimizer.build_plan(
                trials=18,
                top_b=2,
                top_c=2,
                a_seeds=1,
                b_seeds=2,
                final_seeds=8,
            )

            def benchmark(**kwargs):
                summary = competitive(kwargs['seeds'], kwargs['repetitions'])
                for pair in summary['pairs']['records']:
                    pair['seed'] += kwargs['seed']
                report = {'summary': summary}
                Path(kwargs['output']).write_text(json.dumps(report))
                return report

            runs = []
            for name in ('first', 'second'):
                with patch.object(
                    optimizer, 'run_benchmark', side_effect=benchmark
                ):
                    result = optimizer.run_search(
                        base, root / name, bounds, plan, 7
                    )
                candidates = [
                    json.loads(path.read_text())
                    for path in sorted((root / name).glob('candidate-*.json'))
                ]
                effective = dict(
                    bot_weights.DEFAULT_WEIGHTS,
                    space_after_value=12,
                    center_unit_value=80,
                )
                changes = [
                    tuple(
                        (key, value)
                        for key, value in candidate.items()
                        if effective[key] != value
                    )
                    for candidate in candidates
                ]
                self.assertEqual(len(changes), 18)
                self.assertEqual(len(set(changes)), 18)
                self.assertTrue(all(len(change) == 1 for change in changes))
                self.assertEqual(
                    {
                        value
                        for change in changes
                        for key, value in change
                        if key == 'space_after_value'
                    },
                    {11, 13},
                )
                self.assertEqual(
                    {
                        value
                        for change in changes
                        for key, value in change
                        if key == 'center_unit_value'
                    },
                    set(range(72, 89)) - {80},
                )
                self.assertEqual(result['status'], 'completed')
                runs.append(changes)
            self.assertEqual(runs[0], runs[1])
            self.assertEqual(base.read_bytes(), original)
            self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)

    def test_insufficient_total_mutations_rejects_before_any_exports_or_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"space_after_value":12}')
            original = base.read_bytes()
            previous = dict(bot_weights.ACTIVE_WEIGHTS)
            output = root / 'out'
            with patch.object(optimizer, 'run_benchmark') as benchmark:
                with self.assertRaisesRegex(ValueError, '3.*2'):
                    optimizer.run_search(base, output, {'space_after_value': (11, 13)},
                                         optimizer.build_plan(trials=3, top_b=2), 7)
            benchmark.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(base.read_bytes(), original)
            self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)


if __name__ == '__main__':
    unittest.main()
