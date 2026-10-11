import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from snake_bot import bot_weights
from snake_bot.snake_state import GameState, parse_state
from tools import bot_lab as lab


def state(**changes):
    base = GameState(rows=5, cols=6, board=(), side='A', enemy='B',
                     snakes={'A': ((2, 2), (2, 1), (2, 0)), 'B': ((4, 5),)},
                     food=frozenset(), scores={'A': 0, 'B': 0},
                     remaining_moves=10, reliable_tails=frozenset({'A', 'B'}))
    return replace(base, **changes)


def simulation_result(**changes):
    values = dict(seed=1, candidate_side='A', winner='A', turns=10,
                  score_a=300, score_b=100, foods_a=2, foods_b=1,
                  pickups_a=1, pickups_b=2, wrong_digits_a=3, wrong_digits_b=4,
                  wall_hits_a=5, wall_hits_b=6, control_moves_a=7, control_moves_b=8,
                  no_legal_side='B', edge_moves=2, center_score_total=1000)
    values.update(changes)
    return lab.SimResult(**values)


class LabStateTests(unittest.TestCase):
    def test_render_and_turn_data_roundtrip_numbered_items_and_snakes(self):
        original = state(food=frozenset({(0, 0)}), food_values={(0, 0): 1, (0, 1): 2},
                         next_food_digit=1, pickups=frozenset({(1, 0)}),
                         walls=frozenset({(3, 0), (3, 1), (3, 2)}),
                         scores={'A': 123, 'B': 456}, multipliers={'A': 2, 'B': 3})
        data = lab.make_turn_data(original, 'fixture', 7)
        parsed = parse_state(data)
        self.assertEqual(data['turn_token'], 't_7')
        self.assertEqual(data['board_size'], '5x6')
        for attribute in ('snakes', 'food', 'food_values', 'pickups', 'walls',
                          'scores', 'multipliers', 'remaining_moves'):
            self.assertEqual(getattr(parsed, attribute), getattr(original, attribute))

    def test_render_legacy_food_without_numbered_values(self):
        self.assertIn('|*     |', lab.render_board(state(food=frozenset({(0, 0)}))))
        self.assertNotIn('*', lab.render_board(state()))

    def test_spawn_numbered_food_preserves_copy_targets_and_occupied_cells(self):
        original = state(food_values={(0, 0): 9}, food=frozenset({(0, 0)}),
                         next_food_digit=9, food_copy_targets={9: 3, 1: 4})
        spawned = lab.spawn_food(original, random.Random(4), 2)
        self.assertEqual(spawned.food_copy_targets, {9: 3, 1: 4})
        self.assertEqual(list(spawned.food_values.values()).count(9), 3)
        self.assertEqual(list(spawned.food_values.values()).count(1), 4)
        self.assertEqual(spawned.food, frozenset(p for p, d in spawned.food_values.items() if d == 9))
        items = set(spawned.food_values) | set(spawned.pickups) | set(spawned.walls)
        self.assertTrue(items.isdisjoint(original.occupied()))
        self.assertEqual(len(spawned.pickups), 2)
        self.assertEqual(spawned, lab.spawn_food(original, random.Random(4), 2))

    def test_spawn_legacy_food_and_full_board_stop_at_available_space(self):
        original = state(walls=frozenset({(0, 0)}))
        spawned = lab.spawn_food(original, random.Random(8), 4, numbered=False, pickup_count=1)
        self.assertEqual(len(spawned.food), 4)
        self.assertEqual(len(spawned.pickups), 1)
        self.assertEqual(spawned.walls, original.walls)
        self.assertIsNone(spawned.next_food_digit)
        full = state(rows=1, cols=1, snakes={'A': ((0, 0),), 'B': ()})
        spawned = lab.spawn_food(full, random.Random(8), 5)
        self.assertFalse(spawned.food)
        self.assertFalse(spawned.pickups)
        self.assertFalse(spawned.walls)

    def test_spawn_wall_requires_contiguous_odd_length_available_cells(self):
        for cells, rows, cols in [([(0, i) for i in range(3)], 1, 3),
                                  ([(i, 0) for i in range(3)], 3, 1)]:
            with self.subTest(cells=cells):
                self.assertEqual(lab.spawn_wall(cells, rows, cols, random.Random(1)), frozenset(cells))
        self.assertEqual(lab.spawn_wall([(0, 0), (0, 2)], 1, 3, random.Random(1)), frozenset())

    def test_initial_state_dimensions_bodies_and_reproducibility(self):
        for numbered in (True, False):
            with self.subTest(numbered=numbered):
                initial = lab.initial_state(17, rows=7, cols=9, food_count=2,
                                            max_turns=20, numbered=numbered)
                self.assertEqual(initial, lab.initial_state(17, 7, 9, 2, 20, numbered))
                self.assertEqual(initial.snakes['A'], ((3, 2), (3, 1), (3, 0)))
                self.assertEqual(initial.snakes['B'], ((3, 6), (3, 7), (3, 8)))
                self.assertEqual(initial.remaining_moves, 20)
                self.assertEqual(initial.next_food_digit, 1 if numbered else None)

    def test_winner_uses_scores_ignoring_legacy_dead_side(self):
        for scores, dead, expected in [({'A': 1, 'B': 0}, None, 'A'),
                                      ({'A': 0, 'B': 1}, None, 'B'),
                                      ({'A': 1, 'B': 1}, None, None),
                                      ({'A': 999, 'B': 0}, 'A', 'A'),
                                      ({'A': 0, 'B': 999}, 'B', 'B')]:
            self.assertEqual(lab.winner_for(state(scores=scores), dead), expected)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        previous = dict(bot_weights.ACTIVE_WEIGHTS)
        self.addCleanup(self.restore_weights, previous)

    @staticmethod
    def restore_weights(previous):
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)

    def test_simulation_counts_food_pickup_wrong_digit_and_wall_events(self):
        cases = [({'food': frozenset({(2, 3)})}, 'foods_a', 100),
                 ({'pickups': frozenset({(2, 3)})}, 'pickups_a', 50),
                 ({'food_values': {(2, 3): 2, (0, 0): 1}, 'next_food_digit': 1,
                   'food': frozenset({(0, 0)})}, 'wrong_digits_a', -500),
                 ({'walls': frozenset({(2, 3)})}, 'wall_hits_a', -500)]
        for changes, counter, score in cases:
            with self.subTest(counter=counter):
                brain = Mock()
                brain.crash_history = {}
                brain.previous_snakes = {}
                brain.observed_states = {}
                brain.current_states = {}
                brain.observe_state.return_value = state(**changes)
                brain.choose_move.return_value = 'right'
                brain.safe_direction.return_value = 'right'
                brain.decision_debug.return_value = {'reason': 'advantage_control'}
                with patch.object(lab, 'initial_state', return_value=state(**changes)), \
                        patch.object(lab, 'SnakeBrain', return_value=brain), \
                        patch.object(lab, 'spawn_food', side_effect=lambda current, *a: current):
                    result = lab.simulate_game(17, 1, 5, 6, weights_by_side={'A': {}}, candidate_side='A')
                self.assertEqual(getattr(result, counter), 1)
                self.assertEqual(result.score_a, score)
                self.assertEqual(result.control_moves_a, 1)
                self.assertEqual(result.turns, 1)
                brain.commit_move.assert_called_once()

    def test_invalid_proposal_is_rejected_without_repair(self):
        brain = Mock()
        brain.choose_move.return_value = 'bad'
        brain.crash_history = {}
        brain.previous_snakes = {}
        brain.observed_states = {}
        brain.current_states = {}
        brain.observe_state.return_value = state()
        brain.decision_debug.return_value = {}
        with patch.object(lab, 'initial_state', return_value=state()), \
                patch.object(lab, 'SnakeBrain', return_value=brain), \
                patch.object(lab, 'legal_moves', return_value=['up']), \
                patch.object(lab, 'spawn_food', side_effect=lambda current, *a: current):
            with self.assertRaisesRegex(ValueError, 'direction'):
                lab.simulate_game(1, 1, 5, 6)
        brain.safe_direction.assert_not_called()
        brain.commit_move.assert_not_called()

    def test_trapped_side_crash_consumes_turn_without_rewarding_rival(self):
        for side, scores, expected_scores, winner in [
                ('A', {'A': 0, 'B': 0}, (-500, 0), 'B'),
                ('B', {'A': 0, 'B': 0}, (0, -500), 'A'),
                ('A', {'A': 1500, 'B': 0}, (0, 0), None)]:
            with self.subTest(side=side, scores=scores), \
                    patch.object(lab, 'initial_state', return_value=state(
                        rows=2, cols=2, snakes={side: ((0, 0),),
                            ('B' if side == 'A' else 'A'): ((0, 1), (1, 1), (1, 0))},
                        side=side, enemy='B' if side == 'A' else 'A', scores=scores,
                        crashed_a=False, crashed_b=False)):
                result = lab.simulate_game(1, 1, 2, 2)
                self.assertEqual((result.score_a, result.score_b), expected_scores)
                self.assertEqual(result.winner, winner)
                self.assertIsNone(result.no_legal_side)
                self.assertEqual(result.turns, 1)

    def test_seeded_random_dimensions_and_zero_turn_result(self):
        result = lab.simulate_game(3, 0)
        self.assertEqual(result.turns, 0)
        self.assertIsNone(result.winner)
        self.assertIsNone(result.no_legal_side)

    def test_summary_attributes_candidates_by_side_and_totals(self):
        results = [simulation_result(), simulation_result(candidate_side='B', winner='A')]
        summary = lab.summarize_simulations(results)
        expected = {'games': 2, 'candidate_games': 2, 'candidate_wins': 1,
                    'candidate_losses': 1, 'candidate_deaths': 1, 'opponent_deaths': 1,
                    'avg_candidate_score': 200, 'avg_opponent_score': 200,
                    'avg_candidate_score_diff': 0, 'candidate_foods': 3, 'opponent_foods': 3,
                    'candidate_wall_hits': 11, 'opponent_wall_hits': 11,
                    'candidate_control_moves': 15, 'opponent_control_moves': 15,
                    'food_per_100_turns': 30, 'edge_moves_per_100_turns': 20,
                    'avg_center_score': 100, 'avg_turns': 10,
                    'wins_A': 2, 'wins_B': 0, 'deaths_B': 2, 'pickups_A': 2,
                    'wrong_digits_B': 8, 'avg_total_game_score': 400}
        for key, value in expected.items():
            self.assertEqual(summary[key], value, key)

    def test_empty_and_unattributed_summary_has_zero_rates(self):
        for results in ([], [simulation_result(candidate_side=None, winner=None, turns=0)]):
            summary = lab.summarize_simulations(results)
            self.assertEqual(summary['candidate_games'], 0)
            self.assertEqual(summary['avg_candidate_score'], 0)
            self.assertEqual(summary['food_per_100_turns'], 0)
            self.assertEqual(summary['avg_center_score'], 0)
            self.assertEqual(summary['draws'], len(results))


class LabLogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = self.root / 'fixture.log'
        self.data = lab.make_turn_data(state(food=frozenset({(2, 3)})), 'fixture', 0)
        self.log.write_text('\n'.join('< ' + json.dumps({'event': 'your_turn', 'data': dict(
            self.data, turn_token='t_' + str(i))}) for i in range(6)), encoding='utf-8')

    def test_log_iteration_and_sampling_preserve_indices(self):
        entries = list(lab.iter_log_entries([self.log]))
        self.assertEqual([entry[1] for entry in entries], list(range(6)))
        selected, sampled = lab.select_log_entries([self.log], 2)
        self.assertTrue(sampled)
        self.assertEqual([entry[1] for entry in selected], [0, 3])
        selected, sampled = lab.select_log_entries([self.log], 10)
        self.assertFalse(sampled)
        self.assertEqual(selected, entries)

    def test_log_decisions_sequential_and_sampled_commit_behavior(self):
        for limit, count, commits, brains in [(None, 6, 6, 1), (2, 2, 0, 3)]:
            brain = Mock()
            brain.safe_direction.return_value = 'right'
            brain.decision_debug.return_value = {'reason': 'food_plan'}
            with self.subTest(limit=limit), patch.object(lab, 'SnakeBrain', return_value=brain) as factory:
                result = lab.evaluate_log_decisions([self.log], limit)
            self.assertEqual(result['positions'], count)
            self.assertEqual(result['food_taken_now'], count)
            self.assertEqual(result['reasons'], {'food_plan': count})
            self.assertEqual(brain.commit_move.call_count, commits)
            self.assertEqual(factory.call_count, brains)
            self.assertEqual(sum(v for k, v in result.items() if k.startswith('safety_')), count)

    def test_invalid_illegal_and_trapped_log_moves_are_counted(self):
        brain = Mock()
        brain.safe_direction.side_effect = ['invalid', 'left'] * 3
        brain.decision_debug.return_value = {}
        with patch.object(lab, 'SnakeBrain', return_value=brain):
            result = lab.evaluate_log_decisions([self.log])
        self.assertEqual(result['invalid_direction'], 3)
        self.assertEqual(result['illegal_after_repair'], 3)
        brain.commit_move.assert_not_called()
        with patch.object(lab, 'legal_moves', return_value=[]):
            result = lab.evaluate_log_decisions([self.log])
        self.assertEqual(result['no_legal_positions'], 6)
        self.assertEqual(result['reasons'], {})

    def test_log_wall_hits_and_edge_moves_are_counted(self):
        data = lab.make_turn_data(state(walls=frozenset({(2, 3)})), 'wall', 0)
        brain = Mock()
        brain.safe_direction.return_value = 'right'
        brain.decision_debug.return_value = {}
        with patch.object(lab, 'select_log_entries', return_value=([(self.log, 0, data)], False)), \
                patch.object(lab, 'SnakeBrain', return_value=brain), \
                patch.object(lab, 'position_control_penalty', return_value=1):
            result = lab.evaluate_log_decisions([self.log])
        self.assertEqual(result['wall_hit'], 1)
        self.assertEqual(result['edge_or_corner_move'], 1)

    def test_expand_paths_direct_fallback_glob_and_missing(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        Path('games').mkdir()
        Path('games/only.log').touch()
        Path('direct.log').touch()
        cases = [(['direct.log'], [Path('direct.log')]),
                 (['only.log'], [Path('games/only.log')]),
                 (['only*.log'], [Path('games/only.log')]),
                 (['direct*.log'], [Path('direct.log')]),
                 (['missing.log'], [Path('missing.log')])]
        for raw, expected in cases:
            self.assertEqual(lab.expand_log_paths(raw), expected)


class LabCliTests(unittest.TestCase):
    def test_simulate_cli_json_and_text_use_same_seed_sequence(self):
        for option in ([], ['--json']):
            with patch('sys.argv', ['bot_lab', 'simulate', '--games', '2', '--seed', '9',
                                    '--turns', '4', '--rows', '5', '--cols', '6'] + option), \
                    patch.object(lab, 'load_active_weights') as load, \
                    patch.object(lab, 'simulate_game', return_value=simulation_result()) as simulate, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                lab.main()
            self.assertEqual([c.args for c in simulate.call_args_list], [(9, 4, 5, 6, 5), (10, 4, 5, 6, 5)])
            load.assert_called_once_with()
            if option:
                self.assertEqual(json.loads(out.getvalue())['games'], 2)
            else:
                self.assertIn('games: 2', out.getvalue())
                self.assertIn('wall_hits_A: 10', out.getvalue())

    def test_logs_cli_forwards_limit_and_prints_summary(self):
        for option in ([], ['--json']):
            with patch('sys.argv', ['bot_lab', 'logs', 'fixture.log', '--max-positions', '2'] + option), \
                    patch.object(lab, 'load_active_weights'), \
                    patch.object(lab, 'evaluate_log_decisions', return_value={'positions': 2}) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                lab.main()
            evaluate.assert_called_once_with([Path('fixture.log')], 2)
            self.assertEqual(json.loads(out.getvalue()) if option else out.getvalue().strip(),
                             {'positions': 2} if option else 'positions: 2')
