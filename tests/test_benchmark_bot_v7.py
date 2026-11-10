"""Focused v7 reporting contracts without real self-play/search budgets."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snake_bot.game_rules import MAX_MOVES
from tools import benchmark_bot as benchmark
from tools.bot_lab import MECHANICS_VALIDATION_STATUS, SimResult, V7_METRICS
from tests.test_benchmark_bot import result as legacy_result


def sim_result(**changes):
    values = dict(seed=7, candidate_side=None, winner='B', turns=12,
                  score_a=-2501, score_b=120, foods_a=1, foods_b=2,
                  pickups_a=3, pickups_b=4, wrong_digits_a=5, wrong_digits_b=6,
                  wall_hits_a=7, wall_hits_b=8, control_moves_a=9, control_moves_b=10,
                  no_legal_side=None, edge_moves=11, center_score_total=12,
                  termination_reason='score_elimination', remaining_moves=388,
                  diagnostics={key: index + 1 for index, key in enumerate(V7_METRICS)})
    values.update(changes)
    return SimResult(**values)


class BenchmarkV7Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'reference.json'
        self.candidate = self.root / 'candidate.json'
        self.output = self.root / 'report.json'
        self.reference.write_text('{}')
        self.candidate.write_text('{}')

    def run_report(self, **overrides):
        params = dict(reference=self.reference, candidate=self.candidate,
                      seeds=1, seed=7, repetitions=1, output=self.output)
        params.update(overrides)
        return benchmark.run_benchmark(**params)

    def test_default_budget_and_measured_game_wall_time(self):
        with patch.object(benchmark, 'simulate_game', return_value=sim_result()) as simulate, \
                patch.object(benchmark, 'perf_counter', side_effect=[10, 12, 20, 26]):
            report = self.run_report()
        self.assertEqual([c.kwargs['max_turns'] for c in simulate.call_args_list], [400, 400])
        self.assertEqual([g['game_wall_time_seconds'] for g in report['games']], [2, 6])
        summary = report['summary']
        self.assertEqual(summary['total_game_wall_time_seconds'], 8)
        self.assertEqual(summary['seconds_per_game'], 4)
        self.assertEqual(summary['games_per_second'], 0.25)
        self.assertEqual(summary['median_game_wall_time_seconds'], 4)
        self.assertEqual(summary['by_candidate_side']['A']['seconds_per_game'], 2)
        self.assertEqual(json.loads(self.output.read_text()), report)

    def test_reject_budget_above_official_limit_before_simulation(self):
        with patch.object(benchmark, 'simulate_game') as simulate:
            with self.assertRaisesRegex(ValueError, '400'):
                self.run_report(turns=MAX_MOVES + 1)
            simulate.assert_not_called()

    def test_every_v7_metric_is_attributed_to_candidate_b_and_reference_a(self):
        result = sim_result()
        game = benchmark.normalize_game(result, 7, 1, 'B')
        for key in V7_METRICS:
            metric, side = key.rsplit('_', 1)
            role = 'candidate' if side == 'B' else 'reference'
            self.assertEqual(game['events'][role][metric], result.diagnostics[key], key)
        self.assertEqual(game['events']['candidate']['correct_foods'], 2)
        self.assertEqual(game['events']['candidate']['control_moves'], 10)
        self.assertEqual(game['termination_reason'], 'score_elimination')
        self.assertEqual(game['move_budget'], 400)
        self.assertEqual(game['remaining_moves'], 388)
        self.assertEqual(game['winner_participant'], 'candidate')
        self.assertIsNone(game['no_legal_participant'])

    def test_manifest_distinguishes_short_simulation_from_official_budget(self):
        with patch.object(benchmark, 'simulate_game', return_value=sim_result(move_budget=20)):
            report = self.run_report(turns=20)
        manifest = report['manifest']
        self.assertEqual(manifest['simulator_parameters']['max_turns'], 20)
        self.assertEqual(manifest['official_max_moves'], 400)
        self.assertEqual(manifest['mechanics_validation_status'], MECHANICS_VALIDATION_STATUS)
        limitations = ' '.join(manifest['limitations'])
        self.assertIn('pending live validation', limitations)
        self.assertIn('search time', limitations)
        self.assertNotIn('repairs illegal', limitations)
        self.assertNotIn('not exported by SimResult', limitations)

    def test_pair_competitive_and_terminal_summaries(self):
        rows = [
            benchmark.normalize_game(
                sim_result(winner='B', score_a=8, score_b=2), 7, 1, 'A'
            ),
            benchmark.normalize_game(
                sim_result(
                    winner=None,
                    score_a=4,
                    score_b=4,
                    termination_reason='movement_limit',
                ),
                7,
                1,
                'B',
            ),
        ]
        summary = benchmark.summarize(rows)
        self.assertEqual(
            (summary['wins'], summary['losses'], summary['draws']), (0, 1, 1)
        )
        self.assertEqual(
            summary['termination_reasons'],
            {'score_elimination': 1, 'movement_limit': 1},
        )
        pair = summary['pairs']['records'][0]
        self.assertEqual(
            (pair['wins'], pair['losses'], pair['draws']), (0, 1, 1)
        )
        self.assertEqual(pair['median_score_diff'], 3)
        self.assertEqual(pair['by_candidate_side']['A']['mean_score_diff'], 6)
        self.assertEqual(
            pair['by_candidate_side']['B']['median_score_diff'], 0
        )
        metric = 'tail_points'
        self.assertEqual(
            summary['mean_events']['candidate'][metric],
            (
                rows[0]['events']['candidate'][metric]
                + rows[1]['events']['candidate'][metric]
            )
            / 2,
        )

    def test_configuration_directory_and_hardlink_alias_are_protected(self):
        outputs = [self.root / 'config' / 'report.json']
        alias = self.root / 'alias.json'
        os.link(self.reference, alias)
        outputs.append(alias)
        for output in outputs:
            with self.subTest(output=output), patch.object(benchmark, 'simulate_game') as simulate:
                with self.assertRaises(ValueError):
                    self.run_report(output=output)
                simulate.assert_not_called()
        self.assertEqual(self.reference.read_text(), '{}')

    def test_cli_default_and_explicit_short_budget_keep_mirrored_pairs(self):
        args = [
            '--reference',
            str(self.reference),
            '--candidate',
            str(self.candidate),
            '--seeds',
            '1',
            '--seed',
            '7',
            '--repetitions',
            '1',
            '--output',
            str(self.output),
        ]
        for options, budget in (([], 400), (['--turns', '1'], 1)):
            with (
                self.subTest(budget=budget),
                patch.object(
                    benchmark,
                    'simulate_game',
                    return_value=sim_result(move_budget=budget),
                ) as simulate,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                benchmark.main(args + options)
            self.assertEqual(
                [
                    call.kwargs['candidate_side']
                    for call in simulate.call_args_list
                ],
                ['A', 'B'],
            )
            self.assertEqual(
                [call.kwargs['max_turns'] for call in simulate.call_args_list],
                [budget, budget],
            )
            report = json.loads(self.output.read_text())
            self.assertEqual(
                report['manifest']['simulator_parameters']['max_turns'], budget
            )
            self.assertEqual(
                report['manifest']['official_max_moves'], MAX_MOVES
            )

    def test_invalid_noninteger_and_lower_budgets_never_simulate(self):
        for budget in (0, -1, 1.5, True, '400'):
            with self.subTest(budget=budget), patch.object(benchmark, 'simulate_game') as simulate:
                with self.assertRaisesRegex(ValueError, '400'):
                    self.run_report(turns=budget)
                simulate.assert_not_called()

    def test_attribution_a_with_sparse_diagnostics_keeps_tail_and_food_distinct(
        self,
    ):
        result = sim_result(
            diagnostics={
                'tail_collected_A': 2,
                'tail_points_A': 600,
                'tail_generated_for_B': 4,
                'tail_cleaned_B': 3,
                'threshold_eliminations_B': 1,
                'unknown_secret': 'PRIVATE',
            }
        )
        game = benchmark.normalize_game(result, 7, 1, 'A')
        self.assertEqual(game['events']['candidate']['tail_collected'], 2)
        self.assertEqual(game['events']['candidate']['tail_points'], 600)
        self.assertEqual(game['events']['candidate']['correct_foods'], 1)
        self.assertEqual(game['events']['candidate']['wrong_digits'], 5)
        self.assertEqual(game['events']['candidate']['total_crashes'], 0)
        self.assertEqual(game['events']['reference']['tail_generated_for'], 4)
        self.assertEqual(game['events']['reference']['tail_cleaned'], 3)
        self.assertEqual(
            game['events']['reference']['threshold_eliminations'], 1
        )
        self.assertEqual(
            game['global_diagnostics'],
            {'edge_moves': 11, 'center_score_total': 12},
        )
        self.assertNotIn('PRIVATE', json.dumps(game))

    def test_legacy_and_empty_fixtures_have_zero_metrics_and_unknown_timing(
        self,
    ):
        game = benchmark.normalize_game(legacy_result(), 7, 1, 'A')
        for key in (
            'termination_reason',
            'move_budget',
            'remaining_moves',
            'game_wall_time_seconds',
        ):
            self.assertIsNone(game[key])
        # Also accept previously saved normalized rows without additive v7
        # fields.
        game['events'] = {
            role: {
                name: value
                for name, value in values.items()
                if name
                in ('correct_foods', 'wrong_digits', 'pickups', 'wall_hits')
            }
            for role, values in game['events'].items()
        }
        for games in ([], [game]):
            summary = benchmark.aggregate(games)
            self.assertIsNone(summary['seconds_per_game'])
            self.assertIsNone(summary['games_per_second'])
            self.assertIsNone(summary['total_game_wall_time_seconds'])
            self.assertEqual(
                summary['mean_events']['candidate']['tail_collected'],
                0 if games else None,
            )
        self.assertEqual(
            benchmark.aggregate([game])['termination_reasons'], {'unknown': 1}
        )
        current = benchmark.normalize_game(
            sim_result(diagnostics={}), 7, 1, 'B'
        )
        self.assertEqual(current['events']['candidate']['first_crashes'], 0)
        self.assertIsNone(current['game_wall_time_seconds'])

    def test_terminal_totals_and_zero_wall_time_throughput(self):
        rows = [
            benchmark.normalize_game(
                sim_result(
                    diagnostics={
                        'threshold_eliminations_A': 1,
                        'tail_points_B': 300,
                    }
                ),
                7,
                1,
                side,
            )
            for side in 'AB'
        ]
        for game in rows:
            game['game_wall_time_seconds'] = 0
        summary = benchmark.summarize(rows)
        self.assertEqual(summary['seconds_per_game'], 0)
        self.assertIsNone(summary['games_per_second'])
        self.assertEqual(
            summary['total_events']['candidate']['threshold_eliminations'], 1
        )
        self.assertEqual(
            summary['total_events']['reference']['threshold_eliminations'], 1
        )
        self.assertEqual(
            summary['total_events']['candidate']['tail_points'], 300
        )
        self.assertEqual(summary['pairs']['median_score_diff_sum'], 0)
        self.assertEqual(summary['pairs']['median_score_diff_mean'], 0)
        self.assertEqual(summary['pairs']['records'][0]['seconds_per_game'], 0)

    def test_hardlink_to_active_weights_is_protected_even_when_not_an_input(self):
        weights = self.root / 'weights'
        weights.mkdir()
        active = weights / 'active_weights.json'
        active.write_text('{"food_base_value": 10}')
        alias = self.root / 'out.json'
        os.link(active, alias)
        with patch.object(benchmark, '__file__', str(self.root / 'tools' / 'benchmark_bot.py')), \
                patch.object(benchmark, 'simulate_game') as simulate:
            with self.assertRaises(ValueError):
                self.run_report(output=alias)
            simulate.assert_not_called()
        self.assertEqual(active.read_text(), '{"food_base_value": 10}')

    def test_v7_simulation_error_restores_exact_weights_and_existing_report(
        self,
    ):
        from snake_bot import bot_weights

        identity = bot_weights.ACTIVE_WEIGHTS
        before = dict(identity)
        self.output.write_text('{"previous": true}')

        def fail(**kwargs):
            identity.clear()
            identity['changed'] = 1
            raise ValueError('Invalid direction')

        with (
            patch.object(benchmark, 'simulate_game', side_effect=fail),
            self.assertRaisesRegex(ValueError, 'Invalid direction'),
        ):
            self.run_report()
        self.assertIs(bot_weights.ACTIVE_WEIGHTS, identity)
        self.assertEqual(identity, before)
        self.assertEqual(self.output.read_text(), '{"previous": true}')


if __name__ == '__main__':
    unittest.main()
