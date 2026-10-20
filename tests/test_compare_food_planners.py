"""Contracts for planner-only isolation and bounded counterfactual movements."""
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from snake_bot import bot_weights, food_planner, move_search, snake_brain
from snake_bot.search_engine import SearchResult
from tools import bot_lab
from tools import compare_food_planners as compare


ROOT = Path(__file__).resolve().parents[1]


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.weights = dict(bot_weights.DEFAULT_WEIGHTS)
        self.path = ROOT / 'snake_bot/food_planner.py'
        self.planner = compare.load_planner(self.path, compare.sha256(self.path), self.weights)

    def test_exact_hash_is_checked_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            compare.load_planner(self.path, '0' * 64, self.weights)
        self.assertEqual(compare.sha256(self.path), hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_private_bindings_reach_brain_cache_and_search_without_global_swap(self):
        before = dict(bot_weights.get_weights())
        brain = compare.make_brain(self.planner, self.weights, compare.SearchTape())
        namespace = brain.choose_move.__func__.__globals__
        self.assertIs(namespace['build_food_plans'], self.planner.build_food_plans)
        self.assertIs(namespace['choose_food_plan'], self.planner.choose_food_plan)
        ranking = namespace['rank_deep_moves'].__globals__
        self.assertIs(ranking['apply_move'], self.planner.apply_move)
        cache = namespace['TurnSearchCache']()
        self.assertIs(cache.apply.__func__.__globals__['apply_move'], self.planner.apply_move)
        self.assertIs(snake_brain.choose_food_plan, food_planner.choose_food_plan)
        self.assertIs(move_search.apply_move, food_planner.apply_move)
        self.assertEqual(bot_weights.get_weights(), before)
        self.assertIsNot(self.planner.FoodPlan, food_planner.FoodPlan)

    def test_same_version_frozen_control_matches_original_decision_and_history(self):
        for remaining in (18, 400):
            state = bot_lab.initial_state(901, rows=12, cols=12, max_turns=remaining)
            data = bot_lab.make_turn_data(state, 'control', 0)
            tape = compare.SearchTape()
            actual = compare.make_brain(self.planner, self.weights, tape)
            bot_lab.observe_simulation_state(actual, state, data)
            original = snake_brain.SnakeBrain(enable_debug=False)
            original.__dict__ = copy.deepcopy(actual.__dict__)
            with patch.object(compare.IterativeSearchEngine, 'search', return_value=SearchResult(
                    {'up': 123, 'right': 42, 'down': -10}, 4, 71, 2, 45.0)):
                direction = actual.choose_move(data)
            replay = compare.SearchTape(tape.entries)
            frozen = compare.make_brain(self.planner, self.weights, replay)
            frozen.__dict__ = copy.deepcopy(original.__dict__)
            frozen_direction = frozen.choose_move(data)
            replay.assert_consumed()
            with patch.object(move_search, 'IterativeSearchEngine', replay.fresh()):
                original_direction = original.choose_move(data)
            self.assertEqual(direction, frozen_direction)
            self.assertEqual(direction, original_direction)
            self.assertEqual(actual.__dict__, frozen.__dict__)
            self.assertEqual(actual.decision_debug('control'), original.decision_debug('control'))

    def test_weights_are_private_and_immutable(self):
        changed = dict(self.weights, food_distance_cost=999)
        other = compare.load_planner(self.path, compare.sha256(self.path), changed)
        self.assertNotEqual(other.get_weights()['food_distance_cost'], self.planner.get_weights()['food_distance_cost'])
        with self.assertRaises(TypeError):
            other.get_weights()['food_distance_cost'] = 1


class TelemetryTests(unittest.TestCase):
    def test_raw_search_replay_rejects_changed_state_and_changed_target(self):
        state = bot_lab.initial_state(901, rows=12, cols=12)
        tape = compare.SearchTape()
        engine, compact = tape.from_game_state(state, time_budget_ms=45, max_depth=6)
        result = SearchResult({'right': 7}, 4, 100, 0, 45.0)
        with patch.object(compare.IterativeSearchEngine, 'search', return_value=result):
            self.assertIs(engine.search(compact, 'A', ['right'], target=(1, 1)), result)
        replay = tape.fresh()
        engine, compact = replay.from_game_state(state, time_budget_ms=45, max_depth=6)
        with self.assertRaisesRegex(ValueError, 'target'):
            engine.search(compact, 'A', ['right'], target=(2, 2))
        replay.assert_consumed()
        bad = tape.fresh()
        engine, compact = bad.from_game_state(state, time_budget_ms=30, max_depth=6)
        with self.assertRaisesRegex(ValueError, 'search input'):
            engine.search(compact, 'A', ['right'], target=(1, 1))
        with self.assertRaisesRegex(ValueError, 'unconsumed'):
            tape.fresh().assert_consumed()
        with self.assertRaisesRegex(ValueError, 'no recorded'):
            engine.search(compact, 'A', ['right'])

    def test_horizon_classification_requires_changed_plan_and_unreachable_reference(self):
        old = {'food': [1, 8], 'objective_kind': 'multiplier', 'our_distance': 10,
               'first_move': 'right', 'value': 123, 'projected_income': 40}
        new = {'food': [2, 8], 'objective_kind': 'food', 'our_distance': 2,
               'first_move': 'down', 'value': 222, 'projected_income': 50}
        row = compare.classify(old, new, 'right', 'down', 18, 'A', 'A')
        self.assertTrue(row['horizon_x_direction_changed'])
        self.assertTrue(row['horizon_related_direction_changed'])
        self.assertNotIn('causal_horizon_x_direction_changed', row)
        for remaining, actor in ((20, 'A'), (21, 'B')):
            row = compare.classify(old, new, 'right', 'down', remaining, 'A', actor)
            self.assertFalse(row['reference_unreachable_x'])
        reachable = dict(old, our_distance=3, value=999, projected_income=77)
        row = compare.classify(old, reachable, 'right', 'right', 400, 'A', 'A')
        self.assertFalse(row['objective_changed'])
        self.assertTrue(row['reachable_x_economic_changed'])
        self.assertFalse(row['horizon_x_direction_changed'])
        self.assertFalse(compare.classify(None, None, 'up', 'up', 0, 'A', 'A')['any_plan_changed'])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.weights = dict(bot_weights.DEFAULT_WEIGHTS)
        path = ROOT / 'snake_bot/food_planner.py'
        self.planner = compare.load_planner(path, compare.sha256(path), self.weights)

    def test_equal_full_plans_skip_shadow_and_reuse_v7_loop(self):
        rows = []
        rng_before = random.getstate()
        with patch.object(compare, 'make_brain', wraps=compare.make_brain) as factory, \
                patch.object(compare.IterativeSearchEngine, 'search', return_value=SearchResult(
                    {}, 0, 0, 0, 45)):
            result = compare.compare_game(901, 'A', self.planner, self.planner,
                                          self.weights, rows.append, max_turns=2)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(result['match']['turns'], 2)
        self.assertEqual(result['counts']['direction_changed'], 0)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['shadow_mode'] == 'equal_full_plan_no_search' for row in rows))
        self.assertEqual([row['actor'] for row in rows], ['A', 'B'])
        self.assertEqual(random.getstate(), rng_before)
        self.assertTrue(all(row['history']['crash_history'] == {'A': False, 'B': False} for row in rows))

    def test_changed_plan_uses_independent_search_and_keeps_actual_history(self):
        other = compare.load_planner(ROOT / 'snake_bot/food_planner.py',
                                     compare.sha256(ROOT / 'snake_bot/food_planner.py'), self.weights)
        choose = other.choose_food_plan

        def altered(*args, **kwargs):
            from dataclasses import replace
            selected = choose(*args, **kwargs) or next(iter(kwargs['plans']), None)
            return replace(selected, value=selected.value + 1) if selected else None

        other.choose_food_plan = altered
        rows = []
        before = dict(bot_weights.get_weights())
        with patch.object(compare.IterativeSearchEngine, 'search', return_value=SearchResult(
                    {}, 0, 0, 0, 45)) as search, \
                patch.object(bot_lab, 'initial_state', return_value=bot_lab.initial_state(901)):
            result = compare.compare_game(901, 'B', self.planner, other,
                                          self.weights, rows.append, max_turns=2)
        self.assertEqual(search.call_count, 4)
        self.assertTrue(all(row['shadow_mode'] == 'independent_timed_estimate' for row in rows))
        self.assertTrue(all(row['classification']['any_plan_changed'] for row in rows))
        self.assertEqual(result['match']['candidate_side'], 'B')
        self.assertEqual(bot_weights.get_weights(), before)
        self.assertEqual([row['actual_revision'] for row in rows], ['reference', 'candidate'])
        self.assertTrue(all(row['shadow_history_reconciled'] for row in rows))

    def test_atomic_output_refuses_overwrite_and_failure_retains_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            config = {'seeds': [1], 'turns': 400}
            def game(seed, side, *args, **kwargs):
                if side == 'B':
                    raise RuntimeError('smoke error')
                return {'match': {'seed': seed, 'candidate_side': side, 'score_a': 8,
                                  'score_b': 3, 'winner': 'A', 'turns': 2},
                        'counts': {}, 'timing': {}, 'first_divergence': None}
            with patch.object(compare, 'compare_game', side_effect=game), \
                    self.assertRaisesRegex(RuntimeError, 'smoke error'):
                compare.run_comparison(config, self.planner, self.planner, self.weights, output)
            self.assertTrue((output / 'seed_1_candidate_A.json').exists())
            self.assertEqual(json.loads((output / 'summary.json').read_text())['completed_games'], 1)
            self.assertEqual(json.loads((output / 'error.json').read_text())['seed'], 1)
            with self.assertRaises(FileExistsError):
                compare.run_comparison(config, self.planner, self.planner, self.weights, output)

    def test_summary_uses_mirror_pairs_and_predeclared_exposure_threshold(self):
        def game(seed, side, score, horizon):
            return {'match': {'seed': seed, 'candidate_side': side,
                             'score_a': score if side == 'A' else 0,
                             'score_b': score if side == 'B' else 0,
                             'winner': side if score > 0 else None, 'turns': 400},
                    'counts': {'horizon_x_direction_changed': horizon}, 'timing': {}}
        games = [game(seed, side, 10 if side == 'A' else 20, 1)
                 for seed in range(3) for side in ('A', 'B')]
        summary = compare.summarize(games, expected_games=24)
        self.assertEqual(summary['candidate_by_side']['A']['mean_score_diff'], 10)
        self.assertEqual(summary['candidate_by_side']['B']['mean_score_diff'], 20)
        self.assertEqual(summary['mirror_pairs'][0]['mean_candidate_score_diff'], 15)
        self.assertEqual(summary['candidate']['median_score_diff'], 15)
        self.assertEqual(summary['candidate']['win_rate'], 1)
        self.assertEqual(summary['candidate']['win_rate_excluding_draws'], 1)
        self.assertEqual(summary['mirror_pairs'][0]['wins'], 2)
        self.assertTrue(summary['exposure']['threshold_met'])
        self.assertEqual(summary['conclusion'], 'INCONCLUSO')
        self.assertFalse(compare.summarize(games[:4], 24)['exposure']['threshold_met'])

    def test_shared_weights_restored_even_after_simulator_failure(self):
        original = dict(bot_weights.get_weights())
        changed = dict(self.weights, food_distance_cost=987)
        def fail(*args, **kwargs):
            self.assertEqual(bot_weights.get_weights(), changed)
            raise RuntimeError('simulation failed')
        with patch.object(bot_lab, 'initial_state', side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, 'simulation failed'):
            compare.compare_game(901, 'A', self.planner, self.planner, changed, lambda row: None, 2)
        self.assertEqual(bot_weights.get_weights(), original)

    def test_input_hashes_are_checked_after_play_and_before_results_are_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_text('fixed')
            config = {'input_hashes': {str(path): compare.sha256(path)}}
            self.assertTrue(compare.verify_inputs(config)['verified'])
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                compare.verify_inputs(config)

    def test_actual_plan_gate_fails_if_preflight_uses_different_input(self):
        select = compare.selected_plan
        def wrong(*args, **kwargs):
            from dataclasses import replace
            plan, probe = select(*args, **kwargs)
            return replace(plan, value=plan.value + 1) if plan else None, probe
        with patch.object(bot_lab, 'initial_state', return_value=bot_lab.initial_state(901)), \
                patch.object(compare, 'selected_plan', side_effect=wrong), \
                patch.object(compare.IterativeSearchEngine, 'search', return_value=SearchResult({}, 0, 0, 0, 45)), \
                self.assertRaisesRegex(ValueError, 'actual debug plan'):
            compare.compare_game(901, 'A', self.planner, self.planner, self.weights, lambda row: None, 2)

    def test_completed_run_saves_progress_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            row = {'turn': 49, 'snapshot': {'remaining_moves': 351}}
            def game(seed, side, reference, candidate, weights, emit, turns):
                emit(row)
                return {'match': {'seed': seed, 'candidate_side': side, 'score_a': 0,
                                  'score_b': 0, 'winner': None, 'turns': 0}, 'counts': {}, 'timing': {}}
            with patch.object(compare, 'compare_game', side_effect=game), \
                    patch('sys.stdout', new_callable=io.StringIO) as output_text:
                result = compare.run_comparison({'seeds': [1], 'turns': 400}, self.planner,
                                                self.planner, self.weights, output)
            self.assertEqual(result['completed_games'], 2)
            self.assertIn('turn=50', output_text.getvalue())
            self.assertTrue((output / 'post_run_hashes.json').exists())
            self.assertEqual(json.loads((output / 'seed_1_candidate_A.jsonl').read_text()), row)

    def test_seed_list_rejects_duplicates_and_wrong_count(self):
        self.assertEqual(compare.validate_seeds(list(range(12))), list(range(12)))
        for values in ([1] * 12, [1], list(range(13)), list(range(11)) + [True]):
            with self.assertRaises(ValueError):
                compare.validate_seeds(values)

    def test_source_guard_rejects_unrelated_changes(self):
        baseline = 'def build_food_plan():\n    return 1\ndef helper():\n    return 1\n'
        candidate = baseline.replace('return 1', 'return 2', 1)
        self.assertEqual(compare.changed_definitions(baseline, candidate), ['build_food_plan'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            compare.changed_definitions(baseline, candidate.replace('def helper()', 'def other()'))


class ProofTests(unittest.TestCase):
    def test_missing_known_snapshot_fails_before_play(self):
        with self.assertRaisesRegex(ValueError, '5580bbd6'):
            compare.prove_snapshot([], None, None, {})

    def test_cli_invalid_inputs_do_not_play_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights.json'
            weights.write_text('{}')
            args = ['--reference', str(ROOT / 'snake_bot/food_planner.py'),
                    '--candidate', str(ROOT / 'snake_bot/food_planner.py'),
                    '--weights', str(weights), '--seed-list', str(weights),
                    '--frozen-cases', str(weights), '--output', str(root / 'out'), '--mode', 'proof']
            with self.assertRaisesRegex(ValueError, 'SHA256'), \
                    patch.object(compare, 'compare_game') as game:
                compare.main(args)
            game.assert_not_called()
            self.assertFalse((root / 'out').exists())

    def test_proof_uses_recorded_inputs_and_never_brain_search(self):
        from dataclasses import replace
        state = bot_lab.initial_state(901, rows=12, cols=12, max_turns=18)
        data = bot_lab.make_turn_data(state, '5580bbd6', 0)
        planner = compare.load_planner(ROOT / 'snake_bot/food_planner.py',
                                       compare.sha256(ROOT / 'snake_bot/food_planner.py'), bot_weights.DEFAULT_WEIGHTS)
        old = planner.FoodPlan((7, 15), ((7, 2), (8, 2)), 13, None, 1, 100, 3, True, None,
                               'SAFE', 100, objective_kind='multiplier')
        new = replace(old, food=(9, 4), objective_kind='sequence_setup')
        diagnostic = {'crash_history': {'A': False, 'B': False}, 'current_target': [7, 15],
                      'hunger': 7, 'repeat_count': 0, 'direction': 'down', 'plan': compare.describe(old),
                      'verification': {'after': {'tracked_body': {
                          side: {'body': state.snakes[side], 'order_reliable': True} for side in ('A', 'B')}}}}
        case = {'filename': 'game_5580bbd6.log', 'special': True, 'debug_line': 576,
                'observed': data, 'diagnostic': diagnostic, 'observed_history': [{'observed': data}]}
        with patch.object(planner, 'build_food_plans', return_value=[old]), \
                patch.object(planner, 'choose_food_plan', side_effect=[old, new]) as select, \
                patch.object(compare.IterativeSearchEngine, 'search') as search:
            proof = compare.prove_snapshot([case], planner, planner, bot_weights.DEFAULT_WEIGHTS)
        self.assertTrue(proof['verified'])
        self.assertFalse(proof['timed_search_executed'])
        self.assertEqual(select.call_args.args[3], 7)
        search.assert_not_called()
        with patch.object(planner, 'build_food_plans', return_value=[]), \
                patch.object(planner, 'choose_food_plan', return_value=None), \
                self.assertRaisesRegex(ValueError, 'proof failed'):
            compare.prove_snapshot([case], planner, planner, bot_weights.DEFAULT_WEIGHTS)

    def test_valid_cli_proof_and_run_keep_hash_manifest_and_explicit_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / 'seeds.json'
            seeds.write_text(json.dumps(list(range(20261024, 20261036))))
            cases = root / 'cases.json'
            cases.write_text('[]')
            planner = compare.load_planner(ROOT / 'snake_bot/food_planner.py',
                                           compare.sha256(ROOT / 'snake_bot/food_planner.py'), bot_weights.DEFAULT_WEIGHTS)
            for mode in ('proof', 'run'):
                output = root / mode
                args = ['--reference', str(ROOT / 'snake_bot/food_planner.py'),
                        '--candidate', str(ROOT / 'snake_bot/food_planner.py'),
                        '--weights', str(seeds), '--seed-list', str(seeds),
                        '--frozen-cases', str(cases), '--output', str(output), '--mode', mode]
                with patch.object(compare, 'sha256', return_value=compare.WEIGHTS_SHA256), \
                        patch.object(compare, 'load_weights', return_value=bot_weights.DEFAULT_WEIGHTS), \
                        patch.object(compare, 'load_planner', return_value=planner), \
                        patch.object(compare, 'prove_snapshot', return_value={'verified': True}), \
                        patch.object(compare, 'run_comparison', return_value={}) as run, \
                        patch('sys.stdout', new_callable=io.StringIO):
                    compare.main(args)
                if mode == 'proof':
                    manifest = json.loads((output / 'manifest.json').read_text())
                    self.assertEqual(manifest['seeds'], list(range(20261024, 20261036)))
                    self.assertEqual(manifest['turns'], 400)
                    self.assertEqual(manifest['exposure_threshold']['minimum_games'], 6)
                    run.assert_not_called()
                else:
                    self.assertEqual(run.call_args.args[0]['games'], 24)


if __name__ == '__main__':
    unittest.main()
