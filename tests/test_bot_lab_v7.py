"""Bounded simulator contracts; no self-play or server dependency."""
import contextlib
import io
import json
import random
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from snake_bot.game_rules import MAX_MOVES
from snake_bot.snake_state import apply_move, parse_state
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_brain import SnakeBrain
from tests.test_bot_lab_coverage import state, simulation_result
from tools import bot_lab as lab


class BudgetSpawnTests(unittest.TestCase):
    def test_initial_official_budget_and_known_history(self):
        initial = lab.initial_state(1)
        self.assertEqual(initial.remaining_moves, MAX_MOVES)
        self.assertEqual((initial.crashed_a, initial.crashed_b), (False, False))
        for budget in (0, 20, 400):
            short = lab.initial_state(1, max_turns=budget)
            self.assertEqual(short.remaining_moves, budget)
            self.assertEqual((short.crashed_a, short.crashed_b), (False, False))

    def test_reject_budgets_outside_official_limit(self):
        for budget in (-1, 401):
            for function in (lab.initial_state, lab.simulate_game):
                with self.subTest(budget=budget, function=function.__name__):
                    with patch.object(lab, 'SnakeBrain', return_value=Mock()), \
                            self.assertRaisesRegex(ValueError, '400'):
                        function(1, max_turns=budget)

    def test_cli_defaults_to_official_budget(self):
        with patch('sys.argv', ['bot_lab', 'simulate', '--games', '1', '--json']), \
                patch.object(lab, 'load_active_weights'), \
                patch.object(lab, 'simulate_game', return_value=simulation_result()) as simulate, \
                contextlib.redirect_stdout(io.StringIO()):
            lab.main()
        self.assertEqual(simulate.call_args.args[1], 400)

    def test_spawns_exclude_both_tail_types_and_preserve_unknown_history(self):
        original = state(tail_food_a=frozenset({(0, 0), (0, 1)}),
                         tail_food_b=frozenset({(0, 2), (0, 3)}),
                         crashed_a=True, crashed_b=None, walls=frozenset({(1, 0)}))
        for numbered in (True, False):
            spawned = lab.spawn_food(original, random.Random(1), 100, numbered, 100)
            for name in ('tail_food_a', 'tail_food_b', 'crashed_a', 'crashed_b'):
                self.assertEqual(getattr(spawned, name), getattr(original, name))
            items = set(spawned.food) | set(spawned.food_values) | set(spawned.pickups) | set(spawned.walls)
            self.assertTrue(items.isdisjoint(original.tail_food_a | original.tail_food_b))

    def test_spawn_wall_does_not_overlap_new_food_or_pickups(self):
        for numbered in (True, False):
            spawned = lab.spawn_food(state(), random.Random(4), 1, numbered, 1)
            self.assertTrue(spawned.walls.isdisjoint(set(spawned.food_values) | spawned.food | spawned.pickups))

    def test_dead_side_argument_does_not_override_official_score_winner(self):
        for side in ('A', 'B'):
            self.assertEqual(lab.winner_for(state(scores={side: 900, 'B' if side == 'A' else 'A': 0}), side), side)
        self.assertEqual(lab.winner_for(state(scores={'A': -2501, 'B': -9999})), 'B')


class TransitionTests(unittest.TestCase):
    def transition(self, original, direction):
        with patch.object(lab, 'apply_move', wraps=apply_move) as delegate, \
                patch.object(lab, 'spawn_food') as spawn:
            after, events = lab.lab_transition(original, direction)
        delegate.assert_called_once_with(original, direction, original.side)
        spawn.assert_not_called()
        self.assertEqual(after, apply_move(original, direction))
        engine, compact = IterativeSearchEngine.from_game_state(original)
        _, expected = IterativeSearchEngine.from_game_state(after)
        self.assertEqual(engine.apply_move(compact, direction), expected)
        return after, events

    def test_first_subsequent_and_unknown_crashes_all_causes_both_sides(self):
        for side in ('A', 'B'):
            enemy = 'B' if side == 'A' else 'A'
            body = ((0, 2), (0, 1), (0, 0), (1, 0), (1, 1))
            for cause, direction, changes in (
                    ('border', 'up', {}), ('self', 'left', {}),
                    ('wall', 'right', {'walls': frozenset({(0, 3)})}),
                    ('rival', 'right', {'snakes': {side: body, enemy: ((0, 3), (1, 3))}})):
                for history, score, expected in ((False, 900, 0), (True, 900, 400), (None, 0, -500)):
                    with self.subTest(side=side, cause=cause, history=history):
                        original = state(side=side, enemy=enemy,
                                         snakes={side: body, enemy: ((4, 5),)},
                                         scores={side: score, enemy: 10},
                                         **{'crashed_a' if side == 'A' else 'crashed_b': history})
                        original = replace(original, **changes)
                        after, events = self.transition(original, direction)
                        self.assertEqual(after.scores, {side: expected, enemy: 10})
                        kind = 'first' if history is False else 'subsequent' if history is True else 'unknown'
                        self.assertEqual(events[f'{kind}_crashes_{side}'], 1)
                        self.assertEqual(events[f'total_crashes_{side}'], 1)
                        self.assertEqual(events[f'crashes_{cause}_{side}'], 1)
                        self.assertEqual(events[f'lost_length_{side}'], 2)
                        self.assertEqual(events[f'tail_generated_for_{enemy}'], 2)
                        self.assertEqual(after.body(side), body[:3])
                        self.assertEqual(events[f'foods_{side}'], 0)

    def test_tail_collection_and_cleaning_separate_from_food(self):
        for side in ('A', 'B'):
            enemy = 'B' if side == 'A' else 'A'
            for beneficiary, multiplier in ((side, 3), (enemy, 4)):
                with self.subTest(side=side, beneficiary=beneficiary):
                    original = state(side=side, enemy=enemy,
                                     snakes={side: ((2, 2), (2, 1), (2, 0)), enemy: ((4, 5),)},
                                     multipliers={side: multiplier, enemy: 1},
                                     **{'tail_food_a' if beneficiary == 'A' else 'tail_food_b': frozenset({(2, 3), (0, 0)})})
                    after, events = self.transition(original, 'right')
                    self.assertEqual(events[f'tail_collected_{side}'], int(beneficiary == side))
                    self.assertEqual(events[f'tail_cleaned_{side}'], int(beneficiary == enemy))
                    self.assertEqual(events[f'tail_points_{side}'], 300 if beneficiary == side else 0)
                    self.assertEqual(after.scores[side], 300 if beneficiary == side else 1)
                    self.assertEqual(len(after.body(side)), 4 if beneficiary == side else 3)
                    self.assertEqual(events[f'foods_{side}'], 0)

    def test_numbered_correct_wrong_and_classic_growth_not_tail(self):
        for changes, expected_score, counter in (
                ({'food': frozenset({(2, 3)})}, 100, 'foods_A'),
                ({'food': frozenset({(2, 3)}), 'food_values': {(2, 3): 1, (0, 0): 1, (0, 1): 2},
                  'next_food_digit': 1, 'food_copy_targets': {1: 3, 2: 4}}, 100, 'foods_A'),
                ({'food': frozenset({(0, 0)}), 'food_values': {(0, 0): 1, (2, 3): 2, (0, 1): 2},
                  'next_food_digit': 1, 'food_copy_targets': {1: 3, 2: 4}}, -500, 'wrong_digits_A')):
            original = state(**changes)
            after, events = self.transition(original, 'right')
            self.assertEqual(after.scores['A'], expected_score)
            self.assertEqual(len(after.body('A')), 4)
            self.assertEqual(events[counter], 1)
            self.assertEqual(events['tail_collected_A'], 0)
            self.assertEqual(events['tail_points_A'], 0)
            if counter == 'wrong_digits_A':
                self.assertEqual(after.food_copy_targets, original.food_copy_targets)
                self.assertEqual(after.food_values, {(0, 0): 1, (0, 1): 2})
                self.assertEqual(after.next_food_digit, 1)

    def test_elimination_threshold_and_last_move(self):
        for score, eliminated in ((-2000, False), (-2001, True)):
            after, events = self.transition(state(scores={'A': score, 'B': 0},
                                                 crashed_a=True, walls=frozenset({(2, 3)})), 'right')
            self.assertEqual(after.is_terminal(), eliminated)
            self.assertEqual(events['threshold_eliminations_A'], int(eliminated))
        after, events = self.transition(state(remaining_moves=1), 'right')
        self.assertTrue(after.is_terminal())
        self.assertEqual(events['threshold_eliminations_A'], 0)

    def test_invalid_directions_are_errors_and_not_crashes(self):
        for direction in ('bad', None):
            with self.assertRaisesRegex(ValueError, 'direction'), patch.object(lab, 'apply_move') as delegate:
                lab.lab_transition(state(), direction)
            delegate.assert_not_called()

    def test_empty_body_core_noop_has_no_crash_diagnostics(self):
        for side in ('A', 'B'):
            original = state(side=side, enemy='B' if side == 'A' else 'A',
                             snakes={'A': (), 'B': ()}, crashed_a=True, crashed_b=True)
            after, events = lab.lab_transition(original, 'up')
            self.assertEqual(after, original)
            self.assertEqual(dict(events), {})

    def test_terminal_transition_raises_without_events(self):
        for original in (state(remaining_moves=0), state(scores={'A': -2501, 'B': 0})):
            with self.assertRaisesRegex(ValueError, 'ended'):
                lab.lab_transition(original, 'right')


class ControlledBrain(SnakeBrain):
    def __init__(self, directions):
        super().__init__(enable_debug=False)
        self.directions = iter(directions)
        self.seen = []
        self.seen_bodies = []
        self.bodies_before_observation = []
        self.committed = []

    def observe_state(self, data):
        self.bodies_before_observation.append(self.previous_snakes.get(str(data['game_id'])))
        return super().observe_state(data)

    def choose_move(self, data):
        observed = self.observe_state(data)
        self.seen.append((observed.crashed_a, observed.crashed_b))
        self.seen_bodies.append(observed.snakes)
        return next(self.directions)

    def safe_direction(self, data, direction):
        raise AssertionError('Simulator must not repair chosen directions')

    def decision_debug(self, game_id):
        return {}

    def commit_move(self, data, direction):
        self.committed.append(direction)
        super().commit_move(data, direction)


class SimulationV7Tests(unittest.TestCase):
    def simulate(self, original, directions_a, directions_b=(), budget=1, respawn=False):
        brains = [ControlledBrain(directions_a), ControlledBrain(directions_b)]
        with patch.object(lab, 'initial_state', return_value=original), \
                patch.object(lab, 'SnakeBrain', side_effect=brains):
            if respawn:
                result = lab.simulate_game(1, budget, original.rows, original.cols, food_count=0)
            else:
                with patch.object(lab, 'spawn_food', side_effect=lambda current, *a: current):
                    result = lab.simulate_game(1, budget, original.rows, original.cols)
        return result, brains

    def test_honors_valid_crash_without_safe_repair_or_rival_bonus(self):
        for direction, cause, changes in (
                ('left', 'self', {}), ('up', 'border', {'snakes': {'A': ((0, 2), (0, 1), (0, 0)), 'B': ((4, 5),)}})):
            result, brains = self.simulate(state(crashed_a=False, crashed_b=False,
                                                 scores={'A': 900, 'B': 50}, **changes), [direction])
            self.assertEqual((result.score_a, result.score_b), (0, 50))
            self.assertEqual(brains[0].committed, [direction])
            self.assertEqual(result.diagnostics[f'crashes_{cause}_A'], 1)
            self.assertIsNone(result.no_legal_side)


    def test_summary_totals_are_by_side_and_do_not_mix_food_or_legacy_deaths(self):
        results = [simulation_result(diagnostics={'first_crashes_A': 2, 'subsequent_crashes_B': 3,
                                                  'unknown_crashes_B': 1, 'crashes_wall_A': 2,
                                                  'lost_length_A': 5, 'tail_generated_for_B': 5,
                                                  'tail_collected_B': 2, 'tail_cleaned_A': 1,
                                                  'tail_points_B': 600, 'threshold_eliminations_A': 1},
                                     no_legal_side=None, termination_reason='score_elimination', move_budget=400),
                   simulation_result(diagnostics={'first_crashes_A': 1, 'tail_points_B': 100},
                                     no_legal_side=None, termination_reason='simulation_budget', move_budget=20)]
        summary = lab.summarize_simulations(results)
        for key, expected in (('first_crashes_A', 3), ('subsequent_crashes_B', 3), ('unknown_crashes_B', 1),
                              ('tail_generated_for_B', 5), ('tail_collected_B', 2), ('tail_cleaned_A', 1),
                              ('tail_points_B', 700), ('threshold_eliminations_A', 1),
                              ('foods_A', 4), ('deaths_A', 0), ('deaths_B', 0)):
            self.assertEqual(summary[key], expected, key)
        self.assertEqual(summary['termination_reasons'], {'score_elimination': 1, 'simulation_budget': 1})
        self.assertEqual(summary['move_budgets'], {400: 1, 20: 1})

    def test_empty_and_legacy_results_have_zero_v7_metrics_and_pending_status(self):
        for results in ([], [simulation_result()]):
            summary = lab.summarize_simulations(results)
            self.assertEqual(summary['first_crashes_A'], 0)
            self.assertEqual(summary['tail_points_B'], 0)
            statuses = summary['mechanics_validation_status']
            self.assertEqual(statuses['crash_and_tail'],
                             'implemented according to official contract; pending live validation')
            self.assertEqual(statuses['wrong_number_growth'], 'observed in real v7 snapshots')
            self.assertEqual(json.loads(json.dumps(summary))['mechanics_validation_status'], statuses)

    def test_console_and_json_expose_v7_metrics_and_termination(self):
        for option in ([], ['--json']):
            result = simulation_result(diagnostics={'first_crashes_A': 1, 'tail_generated_for_B': 2},
                                       termination_reason='simulation_budget', move_budget=20)
            with patch('sys.argv', ['bot_lab', 'simulate', '--games', '1'] + option), \
                    patch.object(lab, 'load_active_weights'), \
                    patch.object(lab, 'simulate_game', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                lab.main()
            if option:
                summary = json.loads(output.getvalue())
                self.assertEqual(summary['tail_generated_for_B'], 2)
                self.assertEqual(summary['termination_reasons'], {'simulation_budget': 1})
            else:
                self.assertIn('first_crashes_A: 1', output.getvalue())
                self.assertIn('tail_generated_for_B: 2', output.getvalue())
                self.assertIn('simulation_budget', output.getvalue())
                self.assertIn('pending live validation', output.getvalue())

    def test_history_is_authoritative_for_both_brains_after_repeated_crashes_and_spawns(self):
        original = state(crashed_a=False, crashed_b=False,
                         snakes={'A': ((0, 2), (0, 1), (0, 0), (1, 0)),
                                 'B': ((4, 3), (4, 4), (4, 5), (3, 5))})
        result, brains = self.simulate(original, ['up', 'up'], ['down', 'down'], budget=4, respawn=True)
        self.assertEqual(brains[0].seen, [(False, False), (True, True)])
        self.assertEqual(brains[1].seen, [(True, False), (True, True)])
        self.assertEqual((result.score_a, result.score_b), (-1000, -1000))
        self.assertEqual(result.diagnostics['first_crashes_A'], 1)
        self.assertEqual(result.diagnostics['subsequent_crashes_A'], 1)
        self.assertEqual(result.diagnostics['total_crashes_A'], 2)
        self.assertEqual(result.diagnostics['total_crashes_B'], 2)
        self.assertEqual(result.diagnostics['tail_generated_for_B'], 1)
        self.assertEqual(result.diagnostics['tail_generated_for_A'], 1)

    def test_short_initial_history_and_reconstructed_unknown_history(self):
        for history in (False, True, None):
            original = state(crashed_a=history, crashed_b=None, remaining_moves=20)
            result, brains = self.simulate(original, ['right'])
            self.assertEqual(brains[0].seen, [(history, None)])
            self.assertEqual(result.score_a, 1)
        data = lab.make_turn_data(original, 'fixture', 0)
        self.assertNotIn('crash_history', data)
        self.assertNotIn('crashed_a', data)

    def test_reconstructed_winding_body_order_and_history_are_authoritative(self):
        for history in (False, None):
            original = state(rows=7, cols=7, remaining_moves=20,
                             snakes={'A': ((2, 2), (2, 3), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1)),
                                     'B': ((5, 4), (5, 5), (5, 6))},
                             crashed_a=history, crashed_b=True)
            result, brains = self.simulate(original, ['down'])
            self.assertEqual(brains[0].seen_bodies, [original.snakes])
            self.assertEqual(brains[0].bodies_before_observation[0], original.snakes)
            self.assertEqual(brains[0].seen, [(history, True)])
            self.assertEqual(result.score_a, 1)

    def test_reconstructed_unknown_positive_crash_remains_error(self):
        with self.assertRaisesRegex(ValueError, 'known crash history'):
            self.simulate(state(scores={'A': 900, 'B': 0}), ['left'])

    def test_invalid_direction_is_not_repaired_or_committed(self):
        for direction in ('bad', None):
            with self.assertRaisesRegex(ValueError, 'direction'):
                self.simulate(state(), [direction])

    def test_terminal_initial_states_do_not_choose_or_transition(self):
        for original, reason in ((state(remaining_moves=0), 'movement_limit'),
                                 (state(scores={'A': -2501, 'B': 0}), 'score_elimination')):
            result, brains = self.simulate(original, [], budget=400)
            self.assertEqual(result.turns, 0)
            self.assertEqual(brains[0].seen, [])
            self.assertEqual(result.termination_reason, reason)

    def test_budget_termination_distinguishes_official_limit_from_short_budget(self):
        for budget in (0, 1, 20, 400):
            original = state(remaining_moves=budget, crashed_a=False, crashed_b=False)
            # Repeated border crashes would eliminate, so cycle open cells instead.
            a = ['right', 'down', 'left', 'up'] * 50
            b = ['up', 'left', 'down', 'right'] * 50
            original = replace(original, rows=12, cols=12,
                               snakes={'A': ((2, 2),), 'B': ((8, 8),)})
            result, _ = self.simulate(original, a, b, budget=budget)
            self.assertEqual(result.turns, budget)
            self.assertEqual(result.move_budget, budget)
            self.assertEqual(result.termination_reason, 'movement_limit' if budget == 400 else 'simulation_budget')
        result, _ = self.simulate(state(remaining_moves=100), ['right'], budget=1)
        self.assertEqual(result.remaining_moves, 99)
        self.assertEqual(result.termination_reason, 'simulation_budget')

    def test_score_elimination_not_legacy_no_legal_death(self):
        for side in ('A', 'B'):
            enemy = 'B' if side == 'A' else 'A'
            original = state(side=side, enemy=enemy, snakes={side: ((2, 2), (2, 1)), enemy: ((4, 5),)},
                             scores={side: -2001, enemy: 0}, crashed_a=True, crashed_b=True)
            result, _ = self.simulate(original, ['left'] if side == 'A' else [],
                                      ['left'] if side == 'B' else [], budget=20)
            self.assertEqual(result.termination_reason, 'score_elimination')
            self.assertEqual(result.winner, enemy)
            self.assertEqual(result.diagnostics[f'threshold_eliminations_{side}'], 1)
            self.assertIsNone(result.no_legal_side)
