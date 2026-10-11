import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from snake_bot import bot_weights
from tests.test_bot_lab_coverage import simulation_result
from tools import bot_lab, optimize_weights as optimizer


def summary(**changes):
    values = bot_lab.summarize_simulations([simulation_result(
        no_legal_side=None, wall_hits_a=0, wall_hits_b=0)])
    values.update(changes)
    return values


def options(**changes):
    values = dict(games=2, mirror_seeds=True, seed=7, turns=4, rows=5, cols=6,
                  food=2, max_log_positions=3, min_objective_improvement=50,
                  min_diff=80, require_winning_record=True, max_candidate_deaths=0,
                  max_candidate_wall_hits=0, max_edge100=50, min_food100=13,
                  max_log_unsafe=0, trials=3, json=True, until_improvement=False)
    values.update(changes)
    return SimpleNamespace(**values)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        self.addCleanup(self.restore_weights, previous)

    @staticmethod
    def restore_weights(previous):
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)

    def test_random_candidate_only_changes_requested_ranges_and_preserves_input(self):
        base = {'x': 10, 'y': 99}
        for seed in (1, 2):
            candidate = optimizer.random_candidate(base, random.Random(seed), {'x': (20, 30)})
            self.assertTrue(20 <= candidate['x'] <= 30)
            self.assertEqual(candidate['y'], 99)
        self.assertEqual(base, {'x': 10, 'y': 99})

    def test_mutation_can_skip_and_clamps_to_both_bounds(self):
        for chance, factor, expected in [(0.9, 1, 50), (0.1, 3, 60), (0.1, 0.1, 40)]:
            rng = Mock()
            rng.random.return_value = chance
            rng.uniform.return_value = factor
            base = {'x': 50, 'y': 7}
            candidate = optimizer.mutate_candidate(base, rng, 0.3, {'x': (40, 60)})
            self.assertEqual(candidate, {'x': expected, 'y': 7})
            self.assertEqual(base['x'], 50)
            if chance > 0.65:
                rng.uniform.assert_not_called()

    def test_simulation_score_accounts_for_wins_losses_deaths_and_wall_hits(self):
        data = summary(candidate_games=2, candidate_wins=1, candidate_losses=1,
                       candidate_deaths=1, opponent_deaths=1,
                       candidate_wall_hits=1, opponent_wall_hits=1)
        expected = (300 * 3 + 200 * 5.5 + 700 - 450 + 10 * 12 + 30 * 80
                    + 100 * 0.12 - 20 * 45 - 2500 + 700 - 900 + 250)
        self.assertEqual(optimizer.simulation_score(data), expected)
        self.assertEqual(optimizer.simulation_score(summary(candidate_games=0,
                                                            candidate_wins=0, candidate_losses=0)),
                         optimizer.simulation_score(summary()) - 1400)

    def test_log_score_rewards_food_and_penalizes_edges_and_unsafe_positions(self):
        self.assertEqual(optimizer.log_score({}), 0)
        self.assertEqual(optimizer.log_score({'positions': 10, 'food_taken_now': 2,
                                              'edge_or_corner_move': 1,
                                              'safety_DANGEROUS': 1, 'safety_SUICIDAL': 2}),
                         20 * 90 - 10 * 18 - 3 * 1200)

    def test_evaluation_pairs_identical_seeds_and_weights_on_both_sides(self):
        candidate, baseline = {'center_unit_value': 10}, {'center_unit_value': 20}
        for mirrored, expected in [(True, [(7, 'A'), (7, 'B'), (8, 'A'), (8, 'B')]),
                                    (False, [(7, 'A'), (8, 'B')])]:
            args = options(mirror_seeds=mirrored)
            with patch.object(optimizer, 'simulate_game', return_value=simulation_result()) as simulate:
                score, sim, logs = optimizer.evaluate_candidate(candidate, baseline, args, [])
            self.assertEqual([(c.args[0], c.kwargs['candidate_side']) for c in simulate.call_args_list], expected)
            for call in simulate.call_args_list:
                side = call.kwargs['candidate_side']
                self.assertEqual(call.kwargs['weights_by_side'][side], candidate)
                self.assertEqual(call.kwargs['weights_by_side']['B' if side == 'A' else 'A'], baseline)
                self.assertEqual(call.args[1:], (4, 5, 6, 2))
            self.assertEqual(score, optimizer.simulation_score(sim))
            self.assertIsNone(logs)

    def test_evaluation_includes_log_score_and_forwards_sampling_limit(self):
        logs = {'positions': 1, 'food_taken_now': 1}
        for limit, expected_limit in [(3, 3), (0, None)]:
            with patch.object(optimizer, 'simulate_game', return_value=simulation_result()), \
                    patch.object(optimizer, 'evaluate_log_decisions', return_value=logs) as evaluate:
                score, sim, returned_logs = optimizer.evaluate_candidate(
                    {'center_unit_value': 10}, {}, options(max_log_positions=limit), ['fixture.log'])
            evaluate.assert_called_once_with(['fixture.log'], expected_limit)
            self.assertEqual(returned_logs, logs)
            self.assertEqual(score, optimizer.simulation_score(sim) + optimizer.log_score(logs))

    def test_acceptance_rejects_each_independent_regression(self):
        args = options()
        record = {'score': 150, 'simulation': summary(), 'logs': None}
        self.assertTrue(optimizer.is_valid_improvement(record, args, 100))
        for field, value in [('avg_candidate_score_diff', 79), ('candidate_wins', 0),
                             ('candidate_deaths', 1), ('candidate_wall_hits', 1),
                             ('edge_moves_per_100_turns', 51), ('food_per_100_turns', 12)]:
            changed = dict(record, simulation=summary(**{field: value}))
            with self.subTest(field=field):
                self.assertFalse(optimizer.is_valid_improvement(changed, args, 100))
        self.assertFalse(optimizer.is_valid_improvement(dict(record, score=149), args, 100))
        for key in ('safety_DANGEROUS', 'safety_SUICIDAL'):
            self.assertFalse(optimizer.is_valid_improvement(dict(record, logs={key: 1}), args, 100))
        self.assertTrue(optimizer.is_valid_improvement(
            dict(record, simulation=summary(candidate_wins=0)), options(require_winning_record=False), 100))

    def test_candidate_generation_baseline_mutation_and_random_selection(self):
        base = {'x': 5}
        rng = Mock()
        first = optimizer.make_candidate(1, None, base, {'x': (1, 10)}, rng)
        self.assertEqual(first, base)
        self.assertIsNot(first, base)
        rng.random.return_value = 0.1
        rng.uniform.return_value = 1.2
        self.assertEqual(optimizer.make_candidate(2, {'weights': {'x': 8}}, base,
                                                 {'x': (1, 10)}, rng), {'x': 9})
        rng.random.return_value = 0.9
        rng.randint.return_value = 3
        self.assertEqual(optimizer.make_candidate(3, {'weights': {'x': 8}}, base,
                                                 {'x': (1, 10)}, rng), {'x': 3})

    def test_batch_keeps_best_but_accepts_first_valid_improvement(self):
        trials = [(100, summary(), None), (170, summary(), None), (200, summary(), None)]
        with patch.object(optimizer, 'evaluate_candidate', side_effect=trials), \
                patch.object(optimizer, 'make_candidate', side_effect=[{'x': 1}, {'x': 2}, {'x': 3}]):
            best, batch_best, accepted, baseline = optimizer.run_batch(
                options(), [], {}, {}, random.Random(1), batch_index=2)
        self.assertEqual(best, batch_best)
        self.assertEqual(best['score'], 200)
        self.assertEqual(accepted['score'], 170)
        self.assertEqual(accepted['objective_improvement'], 70)
        self.assertEqual(accepted['weights'], {'x': 2})
        self.assertEqual(accepted['batch'], 2)
        self.assertEqual(baseline, 100)

    def test_batch_preserves_previous_best_when_trials_regress(self):
        previous = {'score': 300, 'weights': {}}
        with patch.object(optimizer, 'evaluate_candidate', return_value=(90, summary(), {})), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            best, batch_best, accepted, baseline = optimizer.run_batch(
                options(trials=1, json=False, until_improvement=True), [], {}, {},
                random.Random(1), previous_best=previous, baseline_score=100)
        self.assertIs(best, previous)
        self.assertEqual(batch_best['score'], 90)
        self.assertIsNone(accepted)
        self.assertEqual(baseline, 100)
        self.assertIn('batch=1 #1 score=90', out.getvalue())

    def test_batch_text_prints_accepted_threshold_metrics_and_log_details(self):
        with patch.object(optimizer, 'evaluate_candidate', return_value=(200, summary(), {'positions': 2})), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            _, _, accepted, _ = optimizer.run_batch(options(trials=1, json=False), [], {}, {},
                                                    random.Random(1), baseline_score=100)
        self.assertTrue(accepted['valid_improvement'])
        self.assertIn('logs positions=2', out.getvalue())
        self.assertIn('accepted diff=200', out.getvalue())

    def test_cli_saves_candidate_json_without_activating_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.json'
            base.write_text('{"center_unit_value": 77}', encoding='utf-8')
            output = root / 'candidate.json'
            args = ['optimizer', '--trials', '1', '--base', str(base), '--output', str(output),
                    '--tune-advantage-only', '--logs', 'fixture.log', '--json']
            with patch('sys.argv', args), \
                    patch.object(optimizer, 'evaluate_candidate', return_value=(100, summary(), None)) as evaluate, \
                    patch.object(optimizer, 'time') as clock, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                clock.time.side_effect = [10, 12]
                optimizer.main()
            payload = json.loads(out.getvalue())
            self.assertEqual(payload['baseline_path'], str(base))
            self.assertEqual(payload['tuning_scope'], 'advantage_control')
            self.assertEqual(payload['elapsed_seconds'], 2)
            self.assertIsNone(payload['active_weights_path'])
            self.assertEqual(json.loads(output.read_text())['center_unit_value'], 77)
            self.assertEqual(evaluate.call_args.args[3], [Path('fixture.log')])

    def test_cli_stops_on_accepted_improvement_and_reports_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'candidate.json'
            with patch('sys.argv', ['optimizer', '--trials', '2', '--base', '', '--output', str(output),
                                    '--until-improvement', '--max-batches', '3', '--no-mirror-seeds', '--json']), \
                    patch.object(optimizer, 'evaluate_candidate', side_effect=[
                        (100, summary(), None), (200, summary(), None)]) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                optimizer.main()
            payload = json.loads(out.getvalue())
            self.assertTrue(payload['accepted_improvement'])
            self.assertEqual(payload['best_score'], 200)
            self.assertEqual(payload['objective_improvement'], 100)
            self.assertEqual(payload['acceptance_thresholds']['min_diff'], 80)
            self.assertFalse(payload['mirror_seeds'])
            self.assertIsNone(payload['active_weights_path'])
            self.assertEqual(evaluate.call_count, 2)
            self.assertTrue(output.exists())

    def test_cli_exhausts_batches_without_acceptance_and_prints_best(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'candidate.json'
            with patch('sys.argv', ['optimizer', '--trials', '1', '--base', '', '--output', str(output),
                                    '--until-improvement', '--max-batches', '2']), \
                    patch.object(optimizer, 'evaluate_candidate', return_value=(100, summary(), None)) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                optimizer.main()
            self.assertEqual(evaluate.call_count, 2)
            self.assertIn('batch 2/2', out.getvalue())
            self.assertIn('best:', out.getvalue())
            payload = json.loads(out.getvalue().split('best:\n')[1])
            self.assertFalse(payload['accepted_improvement'])
            self.assertIsNone(payload['baseline_path'])
            self.assertEqual(payload['tuning_scope'], 'all')
            self.assertEqual(json.loads(output.read_text()), bot_weights.DEFAULT_WEIGHTS)
