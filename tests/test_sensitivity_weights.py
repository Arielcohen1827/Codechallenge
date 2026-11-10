import copy
import unittest
from unittest.mock import patch
from pathlib import Path

from snake_bot import bot_weights
from snake_bot.search_engine import IterativeSearchEngine, SearchResult
from tools import sensitivity_weights as sensitivity


class SensitivityTests(unittest.TestCase):
    def test_integer_bounds_stay_inside_requested_percent(self):
        self.assertEqual(sensitivity.perturbations(899, 10), [810, 988])
        self.assertEqual(sensitivity.perturbations(-62, 20), [-74, -50])
        for value in (0, 1, 3, 4, -4):
            self.assertEqual(sensitivity.perturbations(value, 20), [])

    def test_audit_inventory_is_exact_and_excludes_dormant_weights(self):
        names = sensitivity.active_inventory(sensitivity.ROOT / 'docs/weight_tuning_audit.md')
        self.assertEqual(len(names), 135)
        self.assertEqual(len(set(names)), 135)
        self.assertIn('iterative_search_time_ms', names)
        self.assertNotIn('adjacent_plan_tactical_override_margin', names)

    def test_canonical_diagnostics_exclude_clock_not_scores(self):
        first = {
            'turn_token': 'secret',
            'deep_moves': [
                {'score': 12, 'search_nodes': 8, 'search_elapsed_ms': 45}
            ],
        }
        second = copy.deepcopy(first)
        second['deep_moves'][0]['search_nodes'] = 90
        self.assertEqual(
            sensitivity.canonical(first), sensitivity.canonical(second)
        )
        second['deep_moves'][0]['score'] = 13
        self.assertNotEqual(
            sensitivity.canonical(first), sensitivity.canonical(second)
        )
        self.assertNotIn('secret', str(sensitivity.canonical(first)))

    def test_search_replay_requires_same_target_budget_depth_and_state(self):
        result = SearchResult({'right': 3}, 2, 5, 0, 1)
        engine = IterativeSearchEngine(4, 4, time_budget_ms=45, max_depth=3)
        with patch.object(IterativeSearchEngine, 'search', return_value=result) as raw:
            with sensitivity.SearchReplay() as replay:
                for target in (None, None, (1, 2)):
                    engine.search(('state',), 'A', ['right'], target=target)
                engine.time_budget_ms = 50
                engine.search(('state',), 'A', ['right'], target=None)
                engine.max_depth = 4
                engine.search(('state',), 'A', ['right'], target=None)
                engine.search(('other',), 'A', ['right'], target=None)
                self.assertEqual(raw.call_count, 5)
                self.assertEqual(replay.hits, 1)

    def test_weight_tracking_restores_global_dictionary_on_error(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with sensitivity.tracked_weights({'food_base_value': 123}) as reads:
                self.assertEqual(bot_weights.get_weights()['food_base_value'], 123)
                self.assertIn('food_base_value', reads)
                raise RuntimeError('probe')
        self.assertEqual(bot_weights.ACTIVE_WEIGHTS, previous)

    def test_classification_does_not_call_unread_globally_inactive(self):
        self.assertEqual(sensitivity.classify(0, 0, False, 8), 'inactive_in_corpus')
        self.assertEqual(sensitivity.classify(0, 1, True, 8), 'low')
        self.assertEqual(sensitivity.classify(1, 1, True, 8), 'moderate')
        self.assertEqual(sensitivity.classify(3, 3, True, 8), 'high')

    def fake_measure(self, baseline, repeat, evaluator):
        sample = {'id': 's', 'types': {'foodrace'}}
        with patch.object(sensitivity, 'active_inventory', return_value=['food_distance_cost']), \
                patch.object(sensitivity, 'evaluate', side_effect=evaluator):
            return sensitivity.measure([sample], {'food_distance_cost': 100},
                                       {'s': [baseline, repeat]}, progress=lambda line: None)[0]

    def result(self, move='up', score=1):
        return {'move': move, 'diagnostics': {'score': score},
                'reads': ['food_distance_cost'], 'seconds': .01}

    def test_noisy_moves_are_inconclusive_and_not_in_rate_denominator(self):
        row = self.fake_measure(self.result(), self.result('down'),
                                lambda sample, weights: self.result('left'))
        self.assertEqual(row['classification'], 'inconclusive')
        self.assertEqual(row['usable_states'], 0)
        self.assertIsNone(row['changed_moves_percent'])
        self.assertEqual(row['usable_evaluations'], 0)

    def test_unconfirmed_change_does_not_suppress_twenty_percent(self):
        row = self.fake_measure(
            self.result(),
            self.result(),
            lambda sample, weights: self.result(
                'left' if weights['food_distance_cost'] == 100 else 'down'
            ),
        )
        self.assertEqual(
            {trial['requested_percent'] for trial in row['trials']}, {10, 20}
        )
        self.assertFalse(
            any(trial['confirmed_change'] for trial in row['trials'])
        )

    def test_diagnostic_noise_is_not_causal_internal_only_signal(self):
        row = self.fake_measure(
            self.result(score=1),
            self.result(score=2),
            lambda sample, weights: self.result(score=3),
        )
        self.assertEqual(row['internal_only_states'], [])
        self.assertEqual(row['diagnostic_unconfirmed_states'], ['s'])
        self.assertFalse(row['trials'][0]['diagnostic_comparable'])

    def test_diversity_tiebreak_prefers_unused_path(self):
        candidates = [('a', 8, None, None, None, {'X'}),
                      ('b', 8, None, None, None, {'X'})]
        chosen = sensitivity.select_representative(candidates, {'X'}, set(), {'a'})
        self.assertEqual(chosen[0], 'b')

    def test_corpus_has_real_and_lab_contexts_and_required_types(self):
        state = sensitivity.initial_state(1, rows=7, cols=7)
        turns = [
            sensitivity.make_turn_data(state, 'fixture', i) for i in range(9)
        ]
        for i, data in enumerate(turns):
            data['remaining_moves'] = 400 - 2 * i
        diagnostics = [
            {'hunger': i, 'current_target': None, 'active_games': 1}
            for i in range(9)
        ]
        actions = [{'direction': 'up'} for _ in turns]
        with (
            patch.object(
                Path,
                'glob',
                return_value=[
                    sensitivity.ROOT / f'fixture-{i}.log' for i in range(4)
                ],
            ),
            patch.object(
                sensitivity,
                'parse_log',
                return_value=(
                    turns,
                    actions,
                    None,
                    {'version': '2.16'},
                    diagnostics,
                ),
            ),
        ):
            corpus = sensitivity.build_corpus()
        self.assertEqual(len(corpus), 8)
        self.assertEqual(
            {sample['source'] for sample in corpus}, {'real_v7', 'lab'}
        )
        types = set().union(*(sample['types'] for sample in corpus))
        self.assertTrue(sensitivity.REQUIRED_TYPES <= types)
        for sample in corpus:
            self.assertNotIn('turn_token', sample['data'])
            self.assertIn(
                str(sample['data']['game_id']), sample['brain'].previous_snakes
            )

    def test_evaluate_clones_all_brain_context_and_restores_weights(self):
        brain = sensitivity.SnakeBrain(enable_debug=False)
        for name, value in vars(brain).items():
            if isinstance(value, dict):
                value['fixture'] = {'nested': [name]}
        original = copy.deepcopy(vars(brain))
        observed = []

        def choose(clone, data):
            observed.append(copy.deepcopy(vars(clone)))
            for name, value in vars(clone).items():
                if isinstance(value, dict):
                    value['fixture']['nested'].append('changed')
            return 'up'

        sample = {'brain': brain, 'data': {'game_id': 'not-fixture'}}
        with (
            patch.object(sensitivity.SnakeBrain, 'choose_move', choose),
            patch.object(
                sensitivity.SnakeBrain,
                'safe_direction',
                side_effect=lambda data, move: move,
            ),
        ):
            for _ in range(2):
                self.assertEqual(
                    sensitivity.evaluate(sample, {})['move'], 'up'
                )
        self.assertEqual(observed, [original, original])
        self.assertEqual(vars(brain), original)


if __name__ == '__main__':
    unittest.main()
