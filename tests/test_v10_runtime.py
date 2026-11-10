"""Runtime observations and lab RNG share the official v10 state semantics."""

from dataclasses import replace
import random
import unittest

from snake_bot.snake_brain import SnakeBrain
from snake_bot.snake_state import GameState, apply_action, parse_state
from tools import bot_lab
from tools.analyze_matches import state_signature
from v7_logging import safe_message, sent_prediction, verification_record


def position(**changes):
    state = GameState(
        rows=12,
        cols=12,
        board=(),
        side='A',
        enemy='B',
        snakes={
            'A': ((4, 4), (4, 3), (4, 2), (4, 1)),
            'B': ((4, 7), (5, 7), (6, 7), (7, 7)),
        },
        food=frozenset(),
        scores={'A': 6000, 'B': 700},
        remaining_moves=400,
        reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False,
        crashed_b=False,
    )
    return replace(state, **changes)


def payload(state, **extra):
    # Independent wire fixture: don't let lab-rendering bugs hide parser bugs.
    board = [[' ' for _ in range(state.cols)] for _ in range(state.rows)]
    for collection, symbol in ((state.walls, '#'), (state.stars, '\u2b50'),
                               (state.shot_food, '*')):
        for row, col in collection:
            board[row][col] = symbol
    for side in ('A', 'B'):
        head = side.lower() if state.invincibility(side) else side
        body = side if state.invincibility(side) else side.lower()
        for index, (row, col) in enumerate(state.body(side)):
            board[row][col] = head if index == 0 else body
    data = {'game_id': 'v10-runtime', 'turn_token': 'private-fixture',
            'side': state.side, 'rows': state.rows, 'cols': state.cols,
            'board': '\n'.join('|' + ''.join(row) + '|' for row in board),
            'score_1': state.scores['A'], 'score_2': state.scores['B'],
            'multiplier_1': state.multipliers['A'], 'multiplier_2': state.multipliers['B'],
            'remaining_moves': state.remaining_moves,
            'invincible_1': state.invincible_a, 'invincible_2': state.invincible_b}
    return {**data, **extra}


class RuntimeV10Tests(unittest.TestCase):
    def test_lab_render_roundtrips_both_inverted_snakes_and_counters(self):
        for a, b in ((0, 0), (50, 0), (0, 9), (1, 50)):
            with self.subTest(a=a, b=b):
                state = position(invincible_a=a, invincible_b=b,
                                 stars=frozenset({(2, 2)}))
                data = bot_lab.make_turn_data(state, 'roundtrip', 0)
                parsed = parse_state(data, state.snakes)
                self.assertEqual((data['invincible_1'], data['invincible_2']), (a, b))
                self.assertEqual(parsed.snakes, state.snakes)
                self.assertEqual(parsed.stars, state.stars)
                self.assertEqual(data['board'], payload(state)['board'])

    def test_logs_retain_authoritative_counters_and_correct_inverted_heads(self):
        for a, b in ((20, 0), (0, 50), (30, 7)):
            with self.subTest(a=a, b=b):
                state = position(invincible_a=a, invincible_b=b)
                data = payload(state)
                message = safe_message({'event': 'your_turn', 'data': data})
                self.assertEqual(message['data']['invincible_1'], a)
                self.assertEqual(message['data']['invincible_2'], b)
                self.assertNotIn('turn_token', message['data'])
                record = verification_record(data)
                self.assertEqual(record['after']['heads'], {'A': [4, 4], 'B': [4, 7]})
                self.assertEqual(record['after']['invincibility'], {'A': a, 'B': b})

    def test_duplicate_counter_correction_uses_server_and_does_not_swap_heads(self):
        brain = SnakeBrain()
        before = position(invincible_a=20, invincible_b=11)
        brain.observe_state(payload(before))
        corrected = brain.observe_state(payload(replace(before, invincible_a=18, invincible_b=9)))
        self.assertEqual((corrected.invincible_a, corrected.invincible_b), (18, 9))
        self.assertEqual(corrected.snakes, before.snakes)
        self.assertEqual(brain.crash_history['v10-runtime'], {'A': False, 'B': False})

    def test_protected_stationary_turn_is_not_a_first_crash(self):
        for score in (0, 6000):
            with self.subTest(score=score):
                brain = SnakeBrain()
                before = position(invincible_a=2, walls=frozenset({(3, 4)}),
                                  scores={'A': score, 'B': 700})
                first = payload(before)
                brain.observe_state(first)
                brain.commit_action(first, {'action_type': 'move', 'direction': 'up'})
                after = apply_action(before, 'move', 'up')
                after = apply_action(after, 'move', 'right')
                observed = brain.observe_state(payload(after))
                self.assertIs(observed.crashed_a, False)
                self.assertEqual(observed.body('A'), before.body('A'))

    def test_ram_during_unobserved_gap_does_not_invent_victim_crash(self):
        brain = SnakeBrain()
        before = position(invincible_b=3)
        brain.observe_state(payload(before))
        after = replace(before, remaining_moves=396, invincible_b=1,
                        snakes={**before.snakes, 'A': before.body('A')[:1]})
        observed = brain.observe_state(payload(after))
        self.assertIsNone(observed.crashed_a)
        self.assertIsNot(observed.crashed_a, True)

    def test_single_ram_play_data_preserves_histories_and_counter(self):
        brain = SnakeBrain()
        before = position(invincible_a=2,
                          snakes={'A': ((4, 6), (4, 5), (4, 4)),
                                  'B': ((4, 7), (5, 7), (6, 7))})
        data = payload(before)
        brain.observe_state(data)
        brain.commit_action(data, {'action_type': 'move', 'direction': 'right'})
        after = apply_action(before, 'move', 'right')
        observed = brain.observe_state(payload(after, action='move', direction='right'))
        self.assertEqual(brain.crash_history['v10-runtime'], {'A': False, 'B': False})
        self.assertEqual(observed.invincible_a, 1)
        self.assertEqual(observed.body('B'), after.body('B'))
        self.assertNotIn('v10-runtime', brain.pending_moves)

    def test_prediction_reports_star_and_ram_without_claiming_confirmation(self):
        star_state = position(stars=frozenset({(3, 4)}))
        star = sent_prediction(payload(star_state), star_state, 'up')
        self.assertTrue(star['star_consumed'])
        self.assertEqual((star['invincibility_before'], star['invincibility_after']), (0, 50))
        self.assertEqual(star['expected_invincibility'], {'A': 50, 'B': 0})
        ram_state = position(invincible_a=4,
                             snakes={'A': ((5, 6), (5, 5)),
                                     'B': ((4, 7), (5, 7), (6, 7), (7, 7))})
        ram = sent_prediction(payload(ram_state), ram_state, 'right')
        self.assertTrue(ram['ram_cut'])
        self.assertEqual(ram['ram_hit_coordinate'], (5, 7))
        self.assertEqual(ram['cut_length'], 3)
        self.assertEqual(ram['source'], 'prediction_not_server_confirmation')
        self.assertIsNone(ram['collision_cause'])

    def test_shoot_prediction_and_wire_guard_under_invincibility(self):
        before = position(invincible_a=50, invincible_b=50)
        prediction = sent_prediction(payload(before), before, 'right', 'shoot')
        self.assertEqual(prediction['shoot_cost'], 5000)
        self.assertTrue(prediction['shot_blocked_by_invincibility'])
        self.assertEqual(prediction['cut_length'], 0)
        brain = SnakeBrain()
        data = payload(before)
        brain.observe_state(data)
        action = brain.safe_action(data, {'action_type': 'shoot', 'direction': 'right'})
        self.assertEqual(action['action_type'], 'move')
        self.assertEqual(brain.observed_states['v10-runtime'].invincible_a, 50)

    def test_reconciliation_checks_authoritative_count_not_only_body_and_score(self):
        before = position(invincible_a=3)
        first = verification_record(payload(before), parse_state(payload(before)), direction='up')
        after = apply_action(before, 'move', 'up')
        valid = verification_record(payload(after), previous=first)
        self.assertEqual(valid['prediction_reconciliation'], 'consistent_body_and_score_only')
        inconsistent = verification_record(payload(replace(after, invincible_a=1)), previous=first)
        self.assertEqual(inconsistent['prediction_reconciliation'], 'differs_invincibility')

    def test_debug_exposes_v10_state_and_filters_sensitive_fields(self):
        brain = SnakeBrain()
        state = position(invincible_a=7, invincible_b=2,
                         stars=frozenset({(2, 2)}), remaining_moves=100)
        debug = brain._base_debug(payload(state), state, ['up'])
        safe = safe_message(debug)
        self.assertEqual((safe['invincible_self'], safe['invincible_rival']), (7, 2))
        self.assertEqual(safe['stars'], [[2, 2]])
        self.assertTrue(safe['star_frenzy'])

    def test_signature_distinguishes_protection_star_and_frenzy_state(self):
        before = position()
        for changed in (replace(before, invincible_a=1), replace(before, invincible_b=1),
                        replace(before, stars=frozenset({(0, 0)})),
                        replace(before, remaining_moves=100)):
            self.assertNotEqual(state_signature(before), state_signature(changed))

    def test_observer_preserves_exact_twenty_round_deadline_after_a_expiry(self):
        brain = SnakeBrain()
        before = position(remaining_moves=360, invincible_a=1,
                          star_spawn_due_round=None, walls=frozenset({(3, 4)}))
        brain.observe_state(payload(before))
        expired = apply_action(before, 'move', 'up')
        observed = brain.observe_state(payload(expired, action='move', direction='up'))
        self.assertEqual(observed.star_spawn_due_round, 40.5)
        next_state = apply_action(expired, 'move', 'right')
        observed = brain.observe_state(payload(next_state, action='move', direction='right'))
        self.assertEqual(observed.star_spawn_due_round, 40.5)
        self.assertEqual(observed.pending_star_spawns, 0)

    def test_ram_correction_can_reconcile_against_original_pending_move(self):
        brain = SnakeBrain()
        before = position(invincible_a=3,
                          snakes={'A': ((4, 6), (4, 5)), 'B': ((4, 7), (5, 7), (6, 7))})
        initial = payload(before)
        brain.observe_state(initial)
        brain.commit_action(initial, {'action_type': 'move', 'direction': 'right'})
        after = apply_action(before, 'move', 'right')
        brain.observe_state(payload(after, action='move', direction='left'))
        self.assertIn('v10-runtime', brain.pending_moves)
        observed = brain.observe_state(payload(after, action='move', direction='right'))
        self.assertNotIn('v10-runtime', brain.pending_moves)
        self.assertIs(observed.crashed_a, False)

    def test_lab_timeout_roundtrip_retains_terminal_defeat(self):
        state = position(invincible_a=12, timeouts_a=10)
        data = bot_lab.make_turn_data(state, 'timeout-loss', 0)
        parsed = parse_state(data)
        self.assertEqual(parsed.timeouts_a, 10)
        self.assertEqual(parsed.terminal_winner(), 'B')
        self.assertTrue(parsed.is_terminal())


class LabV10Tests(unittest.TestCase):
    def test_spawn_food_reserves_star_cells_and_frenzy_refills_ten(self):
        state = position(
            remaining_moves=100,
            pending_star_spawns=9,
            stars=frozenset({(0, 0)}),
            invincible_a=7,
            invincible_b=8,
        )
        result = bot_lab.spawn_food(
            state, random.Random(33), 2, pickup_count=1
        )
        self.assertEqual(len(result.stars), 10)
        self.assertEqual(result.pending_star_spawns, 0)
        self.assertFalse(
            result.stars
            & (set(result.food_values) | result.pickups | result.occupied())
        )
        self.assertEqual((result.invincible_a, result.invincible_b), (7, 8))

    def test_normal_star_rng_only_selects_nearest_equidistant_candidate(self):
        state = position(remaining_moves=360, pending_star_spawns=1,
                         snakes={'A': ((4, 2), (4, 1)), 'B': ((4, 8), (4, 9))})
        result = bot_lab.spawn_food(state, random.Random(19), 0, pickup_count=0)
        self.assertEqual(result.stars, frozenset({(4, 5)}))
        self.assertEqual(result.pending_star_spawns, 0)
        blocked = replace(state, invincible_a=1, star_spawn_due_round=None)
        result = bot_lab.spawn_food(blocked, random.Random(19), 0, pickup_count=0)
        self.assertFalse(result.stars)

    def test_invincible_wall_and_ram_are_not_lab_crashes_even_with_prior_crash(
        self,
    ):
        for kind in ('wall', 'ram'):
            with self.subTest(kind=kind):
                state = position(
                    invincible_a=3,
                    crashed_a=True,
                    walls=(
                        frozenset({(3, 4)}) if kind == 'wall' else frozenset()
                    ),
                    snakes=(
                        {'A': ((4, 4), (4, 3)), 'B': ((3, 4), (3, 5), (3, 6))}
                        if kind == 'ram'
                        else position().snakes
                    ),
                )
                after, events = bot_lab.lab_transition(state, 'up')
                self.assertEqual(events['total_crashes_A'], 0)
                self.assertEqual(events['wall_hits_A'], 0)
                self.assertEqual(after.scores['A'], state.scores['A'])
                if kind == 'ram':
                    self.assertEqual(events['ram_cuts_A'], 1)


if __name__ == '__main__':
    unittest.main()
