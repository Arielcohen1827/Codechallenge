import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import benchmark_bot as benchmark
from snake_bot import bot_weights


def result(winner='A', score_a=10, score_b=3):
    return SimpleNamespace(winner=winner, score_a=score_a, score_b=score_b,
                           turns=12, foods_a=1, foods_b=2, wrong_digits_a=3,
                           wrong_digits_b=4, pickups_a=5, pickups_b=6,
                           wall_hits_a=7, wall_hits_b=8, no_legal_side='B')


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'reference.json'
        self.candidate = self.root / 'candidate.json'
        self.reference.write_text('{}', encoding='utf-8')
        self.candidate.write_text('{"food_base_value": 12.9, "ignored": "PRIVATE"}',
                                  encoding='utf-8')
        self.output = self.root / 'reports' / 'report.json'

    def run_benchmark(self, **kwargs):
        return benchmark.run_benchmark(self.reference, self.candidate, 2, 17, 2,
                                       self.output, **kwargs)

    def test_side_attribution_includes_each_event_and_scores(self):
        for side, candidate_score, candidate_foods in [('A', 10, 1), ('B', 3, 2)]:
            with self.subTest(side=side):
                row = benchmark.normalize_game(result(), 17, 2, side)
                self.assertEqual(row['candidate_score'], candidate_score)
                self.assertEqual(row['events']['candidate']['correct_foods'], candidate_foods)
                offset = 0 if side == 'A' else 1
                self.assertEqual(row['events']['candidate']['wrong_digits'], 3 + offset)
                self.assertEqual(row['events']['candidate']['pickups'], 5 + offset)
                self.assertEqual(row['events']['candidate']['wall_hits'], 7 + offset)
                self.assertEqual(row['events']['reference']['correct_foods'], 3-candidate_foods)
                self.assertEqual(row['winner_participant'], 'candidate' if side == 'A' else 'reference')
                self.assertEqual(row['scores'], {'A': 10, 'B': 3})
                self.assertEqual(row['turns'], 12)
                self.assertEqual(row['participants'][side], 'candidate')

    def test_aggregate_wins_losses_draws_and_moments(self):
        rows = [benchmark.normalize_game(result(w, a, b), i, 1, side)
                for i, (w, a, b, side) in enumerate([
                    ('A', 10, 3, 'A'), ('A', 10, 3, 'B'), (None, 4, 4, 'A')])]
        summary = benchmark.aggregate(rows)
        self.assertEqual((summary['wins'], summary['losses'], summary['draws']), (1, 1, 1))
        self.assertEqual(summary['win_rate'], 1/3)
        self.assertEqual(summary['win_rate_without_draws'], 0.5)
        self.assertEqual(summary['mean_score_diff'], 0)
        self.assertEqual(summary['median_score_diff'], 0)
        self.assertEqual(summary['mean_scores']['candidate'], 17/3)
        self.assertEqual(summary['mean_turns'], 12)
        self.assertEqual(summary['mean_events']['candidate']['correct_foods'], 4/3)

    def test_all_draws_have_no_decisive_win_rate(self):
        rows = [benchmark.normalize_game(result(None, 4, 4), 1, 1, side) for side in 'AB']
        summary = benchmark.aggregate(rows)
        self.assertEqual(summary['win_rate'], 0)
        self.assertIsNone(summary['win_rate_without_draws'])
        self.assertEqual(summary['draws'], 2)

    def test_pairs_are_grouped_by_seed_and_repetition(self):
        rows = [benchmark.normalize_game(result('A', rep*10, 0), seed, rep, side)
                for rep in (1, 2) for seed in (7, 8) for side in 'AB']
        summary = benchmark.summarize(rows)
        self.assertEqual(len(summary['pairs']['records']), 4)
        self.assertEqual(summary['pairs']['total'], 4)
        self.assertEqual(summary['pairs']['mean_score_diff_sum'], 0)
        for pair in summary['pairs']['records']:
            self.assertEqual(pair['candidate_sides'], ['A', 'B'])
            self.assertEqual((pair['wins'], pair['losses'], pair['draws']), (1, 1, 0))
            self.assertEqual(pair['score_diff_mean'], 0)
        self.assertEqual(summary['by_candidate_side']['A']['mean_score_diff'], 15)
        self.assertEqual(summary['by_repetition']['2']['total'], 4)

    def test_unbalanced_pair_is_rejected(self):
        row = benchmark.normalize_game(result(), 7, 1, 'A')
        for rows in ([row], [row, row]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows)

    def test_runs_fixed_pairs_repeated_seeds_and_restores_exact_global(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        bot_weights.ACTIVE_WEIGHTS['extra_state'] = 42
        expected = dict(bot_weights.ACTIVE_WEIGHTS)
        self.addCleanup(lambda: (bot_weights.ACTIVE_WEIGHTS.clear(), bot_weights.ACTIVE_WEIGHTS.update(previous)))
        identity = bot_weights.ACTIVE_WEIGHTS
        def simulate(**kwargs):
            bot_weights.set_active_weights(kwargs['weights_by_side']['A'])
            return result()
        with patch.object(benchmark, 'simulate_game', side_effect=simulate) as mock:
            report = self.run_benchmark(turns=20)
        self.assertIs(bot_weights.ACTIVE_WEIGHTS, identity)
        self.assertEqual(identity, expected)
        self.assertEqual([(c.kwargs['seed'], c.kwargs['candidate_side']) for c in mock.call_args_list],
                         [(17, 'A'), (17, 'B'), (18, 'A'), (18, 'B')]*2)
        for call in mock.call_args_list:
            side = call.kwargs['candidate_side']
            self.assertEqual(call.kwargs['weights_by_side'][side]['food_base_value'], 12)
            self.assertEqual(call.kwargs['weights_by_side']['B' if side == 'A' else 'A'], bot_weights.DEFAULT_WEIGHTS)
            self.assertEqual(call.kwargs['max_turns'], 20)
        self.assertEqual(report['summary']['total'], 8)

    def test_failure_restores_global_and_does_not_write_report_or_weights(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        before = (self.reference.read_bytes(), self.candidate.read_bytes())
        def fail(**kwargs):
            bot_weights.ACTIVE_WEIGHTS.clear()
            raise RuntimeError('simulation failed')
        with patch.object(benchmark, 'simulate_game', side_effect=fail):
            with self.assertRaises(RuntimeError):
                self.run_benchmark()
        self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)
        self.assertFalse(self.output.exists())
        self.assertEqual(before, (self.reference.read_bytes(), self.candidate.read_bytes()))

    def test_manifest_and_json_roundtrip_no_unknown_values(self):
        before = (self.reference.read_bytes(), self.candidate.read_bytes())
        with patch.object(benchmark, 'simulate_game', return_value=result()), \
                patch.object(benchmark, 'git_snapshot', return_value={'commit': None, 'dirty': None}):
            report = self.run_benchmark()
        self.assertEqual(json.loads(self.output.read_text(encoding='utf-8')), report)
        self.assertEqual(set(report), {'manifest', 'summary', 'games'})
        manifest = report['manifest']
        self.assertIsNotNone(datetime.fromisoformat(manifest['created_at']).tzinfo)
        self.assertTrue(manifest['python_version'])
        self.assertTrue(manifest['platform'])
        self.assertEqual(manifest['seeds'], [17, 18])
        config = manifest['configurations']['candidate']
        self.assertEqual(config['ignored_key_count'], 1)
        self.assertEqual(set(config['effective_weights']), set(bot_weights.DEFAULT_WEIGHTS))
        self.assertEqual(config['effective_sha256'], benchmark.canonical_hash(config['effective_weights']))
        self.assertEqual(len(config['source_sha256']), 64)
        self.assertIn('tools/benchmark_bot.py', manifest['source_sha256'])
        self.assertEqual(len(manifest['source_sha256']['tools/bot_lab.py']), 64)
        self.assertNotIn('PRIVATE', json.dumps(report))
        self.assertEqual(before, (self.reference.read_bytes(), self.candidate.read_bytes()))

    def test_configuration_hash_depends_on_effective_integer_values(self):
        one = benchmark.load_configuration(self.candidate)
        self.candidate.write_text('{"food_base_value": "12"}', encoding='utf-8')
        two = benchmark.load_configuration(self.candidate)
        self.assertEqual(one['effective_sha256'], two['effective_sha256'])
        self.assertNotEqual(one['source_sha256'], two['source_sha256'])

    def test_invalid_configurations_do_not_echo_raw_values(self):
        for content in ('[]', '{"food_base_value":"PRIVATE"}', '{broken PRIVATE', '{"food_base_value":null}'):
            self.candidate.write_text(content, encoding='utf-8')
            with self.assertRaises(ValueError) as caught:
                benchmark.load_configuration(self.candidate)
            self.assertNotIn('PRIVATE', str(caught.exception))
        with self.assertRaises(ValueError):
            benchmark.load_configuration(self.root / 'missing.json')

    def test_invalid_counts_dimensions_and_food_prevent_simulation(self):
        cases = [{'seeds': 0}, {'repetitions': -1}, {'turns': 0}, {'rows': -1},
                 {'cols': 5}, {'cols': -1}, {'food': 0}, {'food': 10}]
        for override in cases:
            params = dict(reference=self.reference, candidate=self.candidate, seeds=2,
                          seed=17, repetitions=1, output=self.output)
            params.update(override)
            with self.subTest(override=override), patch.object(benchmark, 'simulate_game') as mock:
                with self.assertRaises(ValueError):
                    benchmark.run_benchmark(**params)
                mock.assert_not_called()

    def test_valid_boundaries_are_forwarded(self):
        for rows, cols, food in [(0, 0, 1), (1, 6, 9)]:
            with patch.object(benchmark, 'simulate_game', return_value=result()) as mock:
                self.run_benchmark(rows=rows, cols=cols, food=food)
            self.assertEqual(mock.call_args.kwargs['rows'], rows or None)
            self.assertEqual(mock.call_args.kwargs['cols'], cols or None)
            self.assertEqual(mock.call_args.kwargs['food_count'], food)

    def test_output_protection(self):
        for output in [self.reference, self.candidate, self.root/'out.log', self.root/'out.py',
                       self.root/'weights'/'report.json',
                       self.root/'active_weights.json', self.root/'best_weights.json']:
            with self.subTest(output=output), patch.object(benchmark, 'simulate_game') as mock:
                with self.assertRaises(ValueError):
                    benchmark.run_benchmark(self.reference, self.candidate, 1, 1, 1, output)
                mock.assert_not_called()

    def test_new_json_reports_can_be_written_under_games(self):
        for relative in ('games/report.json', 'games/benchmarks/smoke.json'):
            output = self.root / relative
            with patch.object(benchmark, 'simulate_game', return_value=result()):
                report = benchmark.run_benchmark(self.reference, self.candidate, 1, 1, 1, output)
            self.assertEqual(json.loads(output.read_text()), report)

    def test_existing_game_json_is_preserved(self):
        output = self.root / 'games' / 'live.json'
        output.parent.mkdir()
        output.write_text('{"live":true}')
        with patch.object(benchmark, 'simulate_game') as mock:
            with self.assertRaises(ValueError):
                benchmark.run_benchmark(self.reference, self.candidate, 1, 1, 1, output)
            mock.assert_not_called()
        self.assertEqual(output.read_text(), '{"live":true}')

    def test_repeated_benchmark_output_contains_latest_report_and_preserves_configs(self):
        output = self.root / 'games' / 'benchmarks' / 'comparison.json'
        before = (self.reference.read_bytes(), self.candidate.read_bytes())
        with patch.object(benchmark, 'simulate_game', return_value=result('A', 10, 3)):
            first = benchmark.run_benchmark(self.reference, self.candidate, 1, 1000, 1, output)
        with patch.object(benchmark, 'simulate_game', return_value=result('B', 2, 11)):
            latest = benchmark.run_benchmark(self.reference, self.candidate, 1, 1001, 1, output)
        saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(saved, latest)
        self.assertNotEqual(saved, first)
        self.assertEqual(saved['manifest']['seeds'], [1001])
        self.assertEqual(saved['games'][0]['scores'], {'A': 2, 'B': 11})
        self.assertEqual(before, (self.reference.read_bytes(), self.candidate.read_bytes()))

    def test_nonzero_pair_sums_and_even_median(self):
        rows = [benchmark.normalize_game(result('A', a, b), 7, rep, side)
                for rep, side, a, b in [(1, 'A', 12, 2), (1, 'B', 3, 7),
                                        (2, 'A', 1, 5), (2, 'B', 8, 2)]]
        summary = benchmark.summarize(rows)
        self.assertEqual(summary['mean_score_diff'], 1)
        self.assertEqual(summary['median_score_diff'], 0)
        pairs = summary['pairs']
        self.assertEqual([p['score_diff_sum'] for p in pairs['records']], [14, -10])
        self.assertEqual([p['score_diff_mean'] for p in pairs['records']], [7, -5])
        self.assertEqual(pairs['mean_score_diff_sum'], 2)
        self.assertEqual(pairs['mean_score_diff_mean'], 1)
        self.assertEqual(pairs['positive_score_diff_pairs'], 1)
        self.assertEqual(pairs['negative_score_diff_pairs'], 1)

    def test_equal_effective_configurations_are_recorded(self):
        self.candidate.write_text('{}')
        with patch.object(benchmark, 'simulate_game', return_value=result()):
            report = self.run_benchmark()
        self.assertTrue(report['manifest']['effective_hashes_equal'])

    def test_output_error_is_sanitized_and_restores_weights(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        with patch.object(benchmark, 'simulate_game', return_value=result()), \
                patch.object(Path, 'write_text', side_effect=OSError('PRIVATE')):
            with self.assertRaises(ValueError) as caught:
                self.run_benchmark()
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)

    def test_git_unavailable_and_dirty_status(self):
        with patch.object(benchmark.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertEqual(benchmark.git_snapshot(), {'commit': None, 'dirty': None})
        for status, dirty in [('', False), (' M snake_brain.py\n', True)]:
            with patch.object(benchmark.subprocess, 'run', side_effect=[
                    SimpleNamespace(stdout='abc123\n'), SimpleNamespace(stdout=status)]):
                self.assertEqual(benchmark.git_snapshot(), {'commit': 'abc123', 'dirty': dirty})

    def test_cli_requires_explicit_inputs_counts_and_output(self):
        args = ['--reference', str(self.reference), '--candidate', str(self.candidate),
                '--seeds', '2', '--seed', '17', '--repetitions', '1', '--output', str(self.output)]
        for index in range(0, len(args), 2):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.main(args[:index] + args[index+2:])
        with patch.object(benchmark, 'simulate_game', return_value=result()), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            benchmark.main(args)
        self.assertLessEqual(len(stdout.getvalue().splitlines()), 2)
        self.assertEqual(json.loads(self.output.read_text())['summary']['total'], 4)


if __name__ == '__main__':
    unittest.main()
