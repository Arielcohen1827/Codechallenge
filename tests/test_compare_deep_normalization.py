import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from snake_bot import bot_weights, food_planner, move_search, snake_brain
from snake_bot.search_engine import IterativeSearchEngine
from tools import compare_deep_normalization as compare
from tools import deep_clipping_replay as replay


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = compare.BASELINE.parent / 'move_search.candidate.py'
CANDIDATE_SHA = 'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07'


class ComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = bot_weights.load_weights(compare.EXTERNAL / 'baseline_active_weights.json')
        cls.planner = compare.load_planner(ROOT / 'snake_bot/food_planner.py', compare.PLANNER_SHA, cls.weights)
        cls.reference = compare.load_planner(compare.BASELINE, compare.REFERENCE_SHA, cls.weights)
        cls.cases = json.loads((ROOT / 'tests/fixtures/deep_clipping_real_v7.json').read_text())['cases']
        cls.case = next(c for c in cls.cases if c['provenance']['game_id'].startswith('8d25279e')
                        and c['provenance']['debug_line'] == 498)

    def setUp(self):
        if 'cli_' not in self._testMethodName:
            return
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cli_root = Path(directory.name)
        shutil.copytree(ROOT / 'snake_bot', self.cli_root / 'snake_bot',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(CANDIDATE, self.cli_root / 'snake_bot/move_search.py')
        (self.cli_root / 'weights').mkdir()
        shutil.copyfile(ROOT / 'weights/active_weights.json',
                        self.cli_root / 'weights/active_weights.json')
        root_patch = patch.object(compare, 'ROOT', self.cli_root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_private_brains_share_planner_weights_but_not_production_globals(self):
        before = dict(bot_weights.get_weights())
        brain = compare.make_brain(self.reference, self.planner, self.weights)
        namespace = brain.choose_move.__func__.__globals__
        ranking = namespace['rank_deep_moves'].__globals__
        self.assertIs(namespace['choose_food_plan'], self.planner.choose_food_plan)
        self.assertIs(ranking['classify_safety'], self.planner.classify_safety)
        self.assertEqual(namespace['get_weights'](), self.weights)
        self.assertIsNot(namespace, vars(snake_brain))
        self.assertIsNot(ranking, vars(move_search))
        self.assertEqual(bot_weights.get_weights(), before)

    def test_real_frozen_proof_has_same_request_result_and_clock(self):
        with patch.object(IterativeSearchEngine, 'search', side_effect=AssertionError('timed search forbidden')):
            result = compare.prove_case(self.case, self.reference, self.reference, self.planner, self.weights)
        self.assertTrue(result['verified'])
        self.assertEqual(result['reference_debug']['plan'], result['candidate_debug']['plan'])
        self.assertEqual(result['reference_debug']['hunger'], self.case['diagnostic']['hunger'])
        self.assertEqual(result['reference_debug']['repeat_count'], self.case['diagnostic']['repeat_count'])
        self.assertEqual(result['raw_scores'], {'up': -431010, 'down': -322732, 'left': -429420})
        self.assertEqual(result['reference_direction'], result['candidate_direction'])

    def test_tape_rejects_every_request_difference_and_unconsumed_results(self):
        state, _ = replay.state_for(self.case)
        frozen = replay.FrozenEngine(self.case)
        tape = compare.RawSearchTape(engine=frozen)
        target = tuple(self.case['diagnostic']['plan']['food'])
        legal = self.case['diagnostic']['legal']
        budget = self.case['diagnostic']['search_time_budget_ms']
        engine, compact = tape.from_game_state(state, budget, 6)
        recorded = engine.search(compact, state.side, legal, target)
        for changes in ({'target': (0, 0)}, {'legal': list(reversed(legal))},
                        {'side': state.enemy}, {'budget': budget + 1}, {'depth': 5}):
            with self.subTest(changes=changes):
                shadow = tape.fresh()
                engine, compact = shadow.from_game_state(state, changes.get('budget', budget), changes.get('depth', 6))
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    engine.search(compact, changes.get('side', state.side), changes.get('legal', legal), changes.get('target', target))
        shadow = tape.fresh()
        with self.assertRaisesRegex(ValueError, 'unconsumed'):
            shadow.assert_consumed()
        engine, compact = shadow.from_game_state(state, budget, 6)
        result = engine.search(compact, state.side, legal, target)
        self.assertEqual(result, recorded)
        result.scores['up'] = 0
        self.assertEqual(tape.entries[0]['result']['scores']['up'], -431010)
        shadow.assert_consumed()

    def test_counterfactual_deepclones_frame_and_uses_no_timed_shadow(self):
        tape = compare.RawSearchTape(engine=replay.FrozenEngine(self.case))
        actual = compare.make_brain(self.reference, self.planner, self.weights, tape)
        actual.__class__._repeat_count = replay.hashed_repeat_count
        replay.hydrate_brain(self.case, actual)
        frame = copy.deepcopy(actual.__dict__)
        with patch.object(IterativeSearchEngine, 'search', side_effect=AssertionError('timed search forbidden')):
            row = compare.compare_decision(actual, self.reference, self.planner, self.weights,
                                           self.case['observed'], 'reference', tape=tape,
                                           fixture_repeat=True)
        self.assertEqual(row['shadow_mode'], 'shared_actual_root_result')
        self.assertEqual(row['reference_debug'], row['candidate_debug'])
        self.assertTrue(row['search_consumed'])
        key = (self.case['provenance']['game_id'], self.case['diagnostic']['side'])
        self.assertEqual(frame['turns_since_food'][key] + 1, actual.turns_since_food[key])

    def test_source_guard_rejects_whole_rank_and_other_helpers(self):
        source = compare.BASELINE.read_text()
        with self.assertRaisesRegex(ValueError, 'candidate|normaliz'):
            compare.validate_candidate(source, source)
        altered = source.replace("score += min(exits, 4)", "score += min(exits, 3)")
        with self.assertRaises(ValueError):
            compare.validate_candidate(source, altered)

    def test_input_hash_validation_detects_mutation_and_missing_active_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_text('original')
            missing = Path(directory) / 'active.json'
            config = {'input_hashes': {str(path): compare.sha256(path), str(missing): None}}
            self.assertTrue(compare.verify_inputs(config)['verified'])
            missing.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                compare.verify_inputs(config)
            missing.unlink()
            path.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'changed'):
                compare.verify_inputs(config)

    def test_summary_counts_wins_for_each_weighted_side_before_diagnostics(self):
        games = [{'match': {'seed': 1, 'candidate_side': side, 'winner': side,
                            'score_a': 20 if side == 'A' else 10,
                            'score_b': 10 if side == 'A' else 30},
                  'counts': {'adjustment_changed': 2}, 'timing': {}} for side in ('A', 'B')]
        summary = compare.summarize(games, 24)
        self.assertEqual(summary['competition']['wins'], 2)
        self.assertEqual(summary['competition']['mean_score_diff'], 15)
        self.assertEqual(summary['candidate_by_side']['B']['wins'], 1)
        self.assertEqual(summary['counts']['adjustment_changed'], 4)
        self.assertFalse(summary['complete'])

    def test_cli_requires_explicit_candidate_sha_and_defaults_to_prepare(self):
        with patch.object(compare, 'run_comparison') as run, self.assertRaises(SystemExit):
            compare.main(['--weights', str(compare.EXTERNAL / 'baseline_active_weights.json'),
                          '--output', 'never_created'])
        run.assert_not_called()
        args = compare.parser().parse_args(['--candidate-sha', '0' * 64, '--weights', 'weights.json', '--output', 'out'])
        self.assertEqual(args.mode, 'prepare')
        self.assertEqual(args.seeds, list(range(20261036, 20261048)))

    def candidate(self):
        return compare.load_planner(CANDIDATE, CANDIDATE_SHA, self.weights)

    def test_actual_candidate_real_loss_diverges_only_after_shared_root_search(self):
        with patch.object(IterativeSearchEngine, 'search', side_effect=AssertionError('no timed search')):
            result = compare.prove_case(self.case, self.reference, self.candidate(), self.planner, self.weights)
        self.assertTrue(result['classification']['adjustment_changed'])
        self.assertTrue(result['classification']['order_changed'])
        self.assertTrue(result['classification']['final_movement_changed'])
        self.assertEqual(result['reference_direction'], 'up')
        self.assertEqual(result['candidate_direction'], 'down')
        self.assertEqual(result['reference_debug']['plan'], result['candidate_debug']['plan'])
        self.assertEqual(result['search_tape'][0]['request']['target'], result['reference_debug']['plan']['food'])
        self.assertEqual(result['classification']['guard_mode'], 'raw_center')
        self.assertFalse(result['timed_search_executed'])

    def test_ast_guard_accepts_only_exact_call_and_rejects_unrelated_rank_changes(self):
        old = compare.BASELINE.read_text()
        new = CANDIDATE.read_text()
        self.assertEqual(compare.validate_candidate(old, new), ['_search_adjustments', 'rank_deep_moves'])
        for text in (new.replace("analyses.sort(key=deep_sort_key", "analyses.sort(key=other_key", 1),
                     new.replace('adjustments.get(item', 'other.get(item'),
                     new + '\ndef unrelated():\n    return 2\n'):
            with self.subTest(text=text[-80:]), self.assertRaises(ValueError):
                compare.validate_candidate(old, text)

    def test_tape_rejects_state_mutation_and_extra_calls(self):
        state, _ = replay.state_for(self.case)
        tape = compare.RawSearchTape(engine=replay.FrozenEngine(self.case))
        budget = self.case['diagnostic']['search_time_budget_ms']
        engine, compact = tape.from_game_state(state, budget, 6)
        legal = self.case['diagnostic']['legal']
        target = tuple(self.case['diagnostic']['plan']['food'])
        engine.search(compact, state.side, legal, target)
        shadow = tape.fresh()
        engine, compact = shadow.from_game_state(replace(state, remaining_moves=state.remaining_moves - 1), budget, 6)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            engine.search(compact, state.side, legal, target)
        shadow = tape.fresh()
        engine, compact = shadow.from_game_state(state, budget, 6)
        engine.search(compact, state.side, legal, target)
        with self.assertRaisesRegex(ValueError, 'no recorded'):
            engine.search(compact, state.side, legal, target)

    def test_classification_uses_applicable_input_scores_for_whole_group_guard(self):
        def debug(raw, depth=4, remaining=80, **extra):
            return dict(deep_moves=[dict(direction='up', search_score=raw, search_depth=depth)],
                        remaining_moves=remaining, scores={'A': 0, 'B': 0}, **extra)
        for d, mode in (({}, 'no_search'), (debug(10, depth=3), 'depth_gate'),
                        (debug(None), 'no_applicable'), (debug(10, remaining=0), 'legacy_root_terminal'),
                        (debug(10, remaining=4), 'legacy_horizon'), (debug(300000000), 'legacy_extreme'),
                        (debug(-431010), 'raw_center')):
            with self.subTest(mode=mode):
                self.assertEqual(compare.classify(d, d, 'up', 'up', self.weights)['guard_mode'], mode)
        d = debug(10)
        d['deep_moves'].append(dict(direction='down', search_score=500000000, search_depth=4, forced_loss=True))
        result = compare.classify(d, d, 'up', 'down', self.weights)
        self.assertEqual(result['guard_mode'], 'raw_center')
        self.assertTrue(result['forced_loss'])
        self.assertTrue(result['final_movement_changed'])

    def test_guarded_real_subset_preserves_ranking_and_safety(self):
        candidate = self.candidate()
        for category in ('terminal-horizon', 'terminal-compatible/extreme'):
            case = next(c for c in self.cases if c['analysis']['terminal_class'] == category
                        and c['diagnostic'].get('deep_moves'))
            state, _ = replay.state_for(case)
            rows = copy.deepcopy(case['diagnostic']['deep_moves'])
            adjustments = candidate._search_adjustments(state, rows, case['analysis']['depth'], self.weights)
            legacy = replay.legacy_adjustments(rows, case['analysis']['depth'])
            self.assertEqual({r['direction']: adjustments.get(r['direction'], 0) for r in rows}, legacy)
            forced = [r['direction'] for r in rows if r.get('forced_loss')]
            self.assertTrue(all(direction not in adjustments for direction in forced))

    def cli_args(self, output):
        return ['--candidate-sha', CANDIDATE_SHA,
                '--fixtures', str(ROOT / 'tests/fixtures/deep_clipping_real_v7.json'),
                '--shared-manifest', str(ROOT / 'games/benchmarks/planner_x_12seeds_20261012/manifest.json'),
                '--weights', str(compare.EXTERNAL / 'baseline_active_weights.json'), '--output', str(output)]

    def test_actual_cli_prepare_serializes_proof_manifest_and_no_games(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'prepared'
            with patch.object(compare, 'run_comparison') as run, \
                    patch.object(IterativeSearchEngine, 'search', side_effect=AssertionError('timed search')), \
                    patch('sys.stdout', new_callable=io.StringIO):
                compare.main(self.cli_args(output))
            run.assert_not_called()
            config = json.loads((output / 'manifest.json').read_text())
            self.assertTrue(compare.verify_inputs(config)['verified'])
            self.assertEqual(config['seeds'], list(range(20261036, 20261048)))
            self.assertEqual(config['games'], 24)
            self.assertEqual(config['search_budget_ms'], 45)
            self.assertEqual(config['max_depth'], 6)
            self.assertTrue(config['proof']['classification']['final_movement_changed'])
            self.assertIn(str((self.cli_root / 'weights/active_weights.json').resolve()), config['input_hashes'])
            self.assertIn('platform', config)

    def test_cli_invalid_hash_paths_or_budget_cannot_write_or_play(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'never'
            args = self.cli_args(output)
            variants = [args[:1] + ['bad'] + args[2:], args + ['--candidate', str(compare.BASELINE)],
                        args + ['--weights', str(ROOT / 'snake_bot/move_search.py')],
                        args[:1] + ['0' * 64] + args[2:]]
            for argv in variants:
                with self.subTest(argv=argv), patch.object(compare, 'run_comparison') as run, self.assertRaises(ValueError):
                    compare.main(argv)
                run.assert_not_called()
                self.assertFalse(output.exists())
            weights = dict(self.weights, iterative_search_time_ms=90)
            with patch.object(compare.shared, 'load_weights', return_value=weights), self.assertRaisesRegex(ValueError, 'budget'):
                compare.main(args)

    def test_cli_run_dispatches_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch('sys.stdout', new_callable=io.StringIO):
            output = Path(directory) / 'run'
            with patch.object(compare, 'run_comparison', return_value={'mocked': True}) as run:
                compare.main(self.cli_args(output) + ['--mode', 'run'])
            self.assertEqual(run.call_args.args[0]['mode'], 'run')
            self.assertFalse(output.exists())

    def test_cli_rejects_hash_change_during_proof_before_any_output_or_game(self):
        original_sha = compare.sha256
        for mode, changed_path in (('prepare', self.cli_root / 'snake_bot/move_search.py'),
                                   ('run', self.cli_root / 'snake_bot/snake_brain.py')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'never'
                args = self.cli_args(output) + ['--mode', mode]
                proof_completed = []

                def changing_sha(path):
                    if proof_completed and Path(path).resolve() == changed_path.resolve():
                        return '0' * 64
                    return original_sha(path)

                def proof(*args):
                    proof_completed.append(True)
                    return {'classification': {}}

                with patch.object(compare, 'sha256', side_effect=changing_sha), \
                        patch.object(compare, 'prove_case', side_effect=proof), \
                        patch.object(compare, 'run_comparison', return_value={}) as run, \
                        patch('sys.stdout', new_callable=io.StringIO), \
                        self.assertRaisesRegex(ValueError, 'hashes changed'):
                    compare.main(args)
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_refuses_changes_in_shared_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'accepted.json'
            accepted = json.loads((ROOT / 'games/benchmarks/planner_x_12seeds_20261012/manifest.json').read_text())
            accepted['shared_sources']['snake_bot\\snake_brain.py'] = '0' * 64
            manifest.write_text(json.dumps(accepted))
            with self.assertRaisesRegex(ValueError, 'shared source'), patch.object(compare, 'run_comparison') as run:
                compare.main(self.cli_args(Path(directory) / 'never') + ['--shared-manifest', str(manifest)])
            run.assert_not_called()

    def test_mocked_checkpoints_jsonl_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            config = {'input_hashes': {}, 'seeds': [20261036], 'games': 2, 'turns': 400}
            def mocked(seed, side, reference, candidate, planner, weights, emit, turns):
                emit({'mocked_result': True})
                return {'match': {'seed': seed, 'candidate_side': side, 'winner': side,
                                 'score_a': 2, 'score_b': 1}, 'counts': {}, 'timing': {}}
            with patch.object(compare, 'compare_game', side_effect=mocked), patch('sys.stdout', new_callable=io.StringIO):
                result = compare.run_comparison(config, self.reference, self.candidate(), self.planner, self.weights, output)
            self.assertTrue(result['complete'])
            self.assertEqual(json.loads((output / 'seed_20261036_candidate_A.jsonl').read_text()), {'mocked_result': True})
            self.assertTrue((output / 'post_run_hashes.json').exists())
            with self.assertRaises(FileExistsError):
                compare.run_comparison(config, None, None, None, None, output)
            count = []
            def fails(*args):
                if count:
                    raise ValueError('request mismatch')
                count.append(True)
                return mocked(*args)
            output = Path(directory) / 'partial'
            with patch.object(compare, 'compare_game', side_effect=fails), patch('sys.stdout', new_callable=io.StringIO), \
                    self.assertRaisesRegex(ValueError, 'request mismatch'):
                compare.run_comparison(config, None, None, None, None, output)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['completed_games'], 1)
            self.assertEqual(json.loads((output / 'error.json').read_text())['candidate_side'], 'B')

    def test_simulator_adapter_uses_existing_code_object_and_restores_weights(self):
        original = dict(bot_weights.get_weights())
        real_private = compare.private_namespace
        candidate = self.candidate()
        emitted = []
        def namespace(module, planner, weights):
            result = real_private(module, planner, weights)
            if module is not compare.bot_lab:
                return result
            self.assertIs(result['simulate_game'].__code__, compare.bot_lab.simulate_game.__code__)
            def fake_simulator(seed, max_turns, candidate_side):
                # Exercise both factories and real decisions on a frozen frame, no transitions/game.
                for _ in range(2):
                    brain = result['SnakeBrain']()
                    brain.__class__._repeat_count = replay.hashed_repeat_count
                    replay.hydrate_brain(self.case, brain)
                    original_compare = compare.compare_decision
                    def frozen(actual, alternate, planner, weights, data, label, **kwargs):
                        return original_compare(actual, alternate, planner, weights, data, label,
                                                compare.RawSearchTape(engine=replay.FrozenEngine(self.case)), True, **kwargs)
                    data = dict(self.case['observed'], turn_token='frozen_0')
                    brain.observation_tokens[self.case['provenance']['game_id']] = ('frozen_0', self.case['diagnostic']['remaining_moves'])
                    with patch.object(compare, 'compare_decision', side_effect=frozen):
                        brain.choose_move(data)
                from dataclasses import make_dataclass
                return make_dataclass('Result', [('winner', str), ('candidate_side', str)])('A', candidate_side)
            result['simulate_game'] = fake_simulator
            return result
        with patch.object(compare, 'private_namespace', side_effect=namespace), \
                patch.object(IterativeSearchEngine, 'search', side_effect=AssertionError('no timed search')):
            game = compare.compare_game(1, 'B', self.reference, candidate, self.planner, self.weights, emitted.append)
        self.assertEqual([r['actual_revision'] for r in emitted], ['reference', 'candidate'])
        self.assertEqual(game['counts']['final_movement_changed'], 2)
        self.assertIn('final_movement_changed', game['first_divergences'])
        self.assertEqual(bot_weights.get_weights(), original)


if __name__ == '__main__':
    unittest.main()
