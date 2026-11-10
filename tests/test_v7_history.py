"""Real-turn history must not invent an unobserved first crash."""

from dataclasses import replace
import asyncio
import io
import unittest
from unittest.mock import AsyncMock, patch

import run

from snake_bot.snake_brain import SnakeBrain, final_safe_direction


def snapshot(remaining=400, side='A', game_id='one', a=0, b=0,
             body_a=((2, 2), (2, 1), (2, 0)),
             body_b=((4, 4), (4, 5), (4, 6))):
    board = [list(' ' * 7) for _ in range(7)]
    for player, body in (('A', body_a), ('B', body_b)):
        for index, (row, col) in enumerate(body):
            board[row][col] = player if index == 0 else player.lower()
    return {'board': '\n'.join(''.join(row) for row in board),
            'rows': 7, 'cols': 7, 'side': side, 'remaining_moves': remaining,
            'score_1': a, 'score_2': b, 'game_id': game_id,
            'turn_token': str(remaining)}


class V7HistoryTests(unittest.TestCase):
    def test_known_start_and_first_b_turn(self):
        for side, remaining, expected in (
            ('A', 400, (False, False)),
            ('B', 399, (None, False)),
            ('A', 350, (None, None)),
        ):
            state = SnakeBrain().observe_state(snapshot(remaining, side))
            self.assertEqual((state.crashed_a, state.crashed_b), expected)

    def test_own_crash_is_pending_until_observed(self):
        brain = SnakeBrain()
        data = snapshot(a=100)
        state = brain.observe_state(data)
        brain.current_states['one'] = replace(state, walls=frozenset({(1, 2)}))
        brain.commit_move(data, 'up')
        self.assertIs(brain.crash_history['one']['A'], False)
        observed = brain.observe_state(
            snapshot(398, a=0, b=1, body_b=((3, 4), (4, 4), (4, 5)))
        )
        self.assertIs(observed.crashed_a, True)
        self.assertIs(observed.crashed_b, False)
        self.assertEqual(observed.scores['A'], 0)

    def test_unknown_positive_crash_commit_does_not_guess_score(self):
        brain = SnakeBrain()
        data = snapshot(350, a=800, body_a=((0, 2), (0, 1), (0, 0), (1, 0)))
        brain.observe_state(data)
        brain.commit_move(data, 'up')
        self.assertIsNone(brain.crash_history['one']['A'])
        self.assertEqual(brain.observed_states['one'].scores['A'], 800)
        after = brain.observe_state(
            snapshot(
                348,
                a=300,
                b=1,
                body_a=((0, 2), (0, 1), (0, 0)),
                body_b=((3, 4), (4, 4), (4, 5)),
            )
        )
        self.assertIs(after.crashed_a, True)
        self.assertEqual(after.scores['A'], 300)

    def test_enemy_crash_without_tail_is_observed(self):
        brain = SnakeBrain()
        data = snapshot()
        brain.observe_state(data)
        brain.commit_move(data, 'up')
        after = brain.observe_state(
            snapshot(398, a=1, b=-500, body_a=((1, 2), (2, 2), (2, 1)))
        )
        self.assertIs(after.crashed_b, True)
        self.assertIs(after.crashed_a, False)

    def test_gaps_invalidate_false_but_preserve_true(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot())
        brain.crash_history['one']['B'] = True
        after = brain.observe_state(snapshot(390, a=50, b=20))
        self.assertIsNone(after.crashed_a)
        self.assertIs(after.crashed_b, True)

    def test_unexplained_movement_cannot_preserve_false(self):
        for after in (((0, 2), (0, 1), (0, 0)), ((1, 2), (2, 2), (2, 1))):
            brain = SnakeBrain()
            brain.observe_state(
                snapshot(body_a=((2, 2), (2, 1), (2, 0), (3, 0)))
            )
            state = brain.observe_state(
                snapshot(
                    398, a=1, body_a=after, body_b=((3, 4), (4, 4), (4, 5))
                )
            )
            self.assertIsNone(state.crashed_a)

    def test_shrink_across_gap_proves_a_crash(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot(body_a=((2, 2), (2, 1), (2, 0), (3, 0))))
        state = brain.observe_state(
            snapshot(390, a=5, body_a=((1, 2), (1, 1), (1, 0)))
        )
        self.assertIs(state.crashed_a, True)

    def test_server_contradicts_prediction_without_becoming_confirmed_false(
        self,
    ):
        brain = SnakeBrain()
        data = snapshot()
        brain.observe_state(data)
        brain.commit_move(data, 'up')
        state = brain.observe_state(
            snapshot(
                398,
                a=1,
                b=1,
                body_a=((3, 2), (2, 2), (2, 1)),
                body_b=((3, 4), (4, 4), (4, 5)),
            )
        )
        self.assertEqual(state.head('A'), (3, 2))
        self.assertIsNone(state.crashed_a)

    def test_duplicate_snapshot_does_not_consume_pending_action(self):
        brain = SnakeBrain()
        data = snapshot()
        first = brain.observe_state(data)
        brain.commit_move(data, 'up')
        self.assertEqual(brain.observe_state(data), first)
        self.assertIn('one', brain.pending_moves)
        corrected = dict(data, score_1=37)
        self.assertEqual(brain.observe_state(corrected).scores['A'], 37)

    def test_corrected_same_turn_invalidates_false_and_replaces_geometry(self):
        brain = SnakeBrain()
        data = snapshot()
        brain.observe_state(data)
        brain.commit_move(data, 'up')
        corrected = snapshot(a=-500, body_a=((3, 2), (2, 2), (2, 1)))
        state = brain.observe_state(corrected)
        self.assertIsNone(state.crashed_a)
        self.assertIs(state.crashed_b, False)
        self.assertEqual(brain.previous_snakes['one'], state.snakes)
        self.assertIn('one', brain.pending_moves)

    def test_stale_snapshot_does_not_replace_observation_or_pending(self):
        brain = SnakeBrain()
        recent = snapshot(398)
        state = brain.observe_state(recent)
        brain.commit_move(recent, 'up')
        predicted = dict(brain.previous_snakes['one'])
        pending = brain.pending_moves['one']
        self.assertIsNone(brain.observe_state(snapshot(400)))
        self.assertIs(brain.observed_states['one'], state)
        self.assertEqual(brain.previous_snakes['one'], predicted)
        self.assertEqual(brain.pending_moves['one'], pending)
        valid = brain.observe_state(
            snapshot(
                396,
                a=1,
                b=1,
                body_a=((1, 2), (2, 2), (2, 1)),
                body_b=((3, 4), (4, 4), (4, 5)),
            )
        )
        self.assertEqual(valid.remaining_moves, 396)
        self.assertNotIn('one', brain.pending_moves)

    def test_stale_turn_is_not_sent_or_logged(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot(398))
        with patch.object(run, 'BOT', brain), \
                patch.object(run, 'log_action') as action, \
                patch.object(run, 'write_live_snapshot') as live, \
                patch.object(run, 'print_diagnostic'), \
                patch.object(run, 'send', new=AsyncMock()) as send:
            asyncio.run(run.process_move(None, {'data': snapshot(400)}))
            send.assert_not_awaited()
            action.assert_not_called()
            live.assert_not_called()

    def test_history_isolated_and_forgotten(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot())
        brain.commit_move(snapshot(), 'up')
        self.assertIsNone(brain.observe_state(snapshot(300, game_id='two')).crashed_a)
        brain.forget('one')
        for mapping in (brain.crash_history, brain.observed_states,
                        brain.pending_moves, brain.previous_snakes):
            self.assertNotIn('one', mapping)
        self.assertIn('two', brain.observed_states)

    def test_safety_does_not_preserve_v6_winning_suicide(self):
        data = snapshot(a=10000)
        brain = SnakeBrain()
        brain.observe_state(data)
        self.assertNotEqual(brain.safe_direction(data, 'left'), 'left')
        self.assertNotEqual(final_safe_direction(data, 'left'), 'left')

    def test_forced_crash_fallback_does_not_reverse_into_neck(self):
        data = snapshot(body_a=((0, 0), (1, 0), (2, 0)))
        brain = SnakeBrain()
        brain.observe_state(data)
        with patch('snake_bot.snake_brain.legal_moves', return_value=[]):
            self.assertNotEqual(brain.safe_direction(data, 'down'), 'down')
            self.assertNotEqual(final_safe_direction(data, 'down'), 'down')
        # Put the neck above the head, where the old hard-coded up fallback failed.
        data = snapshot(body_a=((1, 0), (0, 0), (0, 1)))
        brain = SnakeBrain()
        brain.observe_state(data)
        with patch('snake_bot.snake_brain.legal_moves', return_value=[]):
            self.assertNotEqual(brain.safe_direction(data, 'up'), 'up')
            self.assertNotEqual(final_safe_direction(data, 'up'), 'up')

    def test_game_over_cleanup_survives_diagnostic_failure(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot())
        with patch.object(run, 'BOT', brain), \
                patch.object(run, 'log_event'), \
                patch.object(run, 'write_live_snapshot', side_effect=OSError), \
                patch.object(run, 'write_game_log') as write_log:
            result = asyncio.run(run.on_game_over(None, {'data': {'game_id': 'one'}}))
            self.assertIsNone(result)
            write_log.assert_called_once_with('one')
        self.assertNotIn('one', brain.crash_history)
        self.assertNotIn('one', brain.observed_states)
        self.assertNotIn('one', brain.current_states)
        self.assertNotIn('one', brain.previous_snakes)
        self.assertNotIn('one', brain.observation_tokens)

    def test_diagnostics_and_repetition_distinguish_v7_state(self):
        brain = SnakeBrain()
        state = brain.observe_state(snapshot())
        brain._remember('one', state)
        for changed in (replace(state, crashed_a=True),
                        replace(state, scores={'A': 1, 'B': 0}),
                        replace(state, tail_food_a=frozenset({(1, 1)}))):
            self.assertEqual(brain._repeat_count('one', changed), 0)
        debug = brain._base_debug(snapshot(), state, ['up'])
        self.assertEqual(debug['crash_history'], {'A': False, 'B': False})
        self.assertEqual(debug['tail_food_a'], [])

    def test_tail_food_proves_originators_previous_crash(self):
        for symbol, expected in (('\u24b6', (None, True)),
                                 ('\u24b7', (True, None))):
            data = snapshot(300)
            data['board'] = symbol + data['board'][1:]
            state = SnakeBrain().observe_state(data)
            self.assertEqual((state.crashed_a, state.crashed_b), expected)

    def test_missing_horizon_is_not_proof_of_initial_game(self):
        data = snapshot()
        data.pop('remaining_moves')
        state = SnakeBrain().observe_state(data)
        self.assertIsNone(state.crashed_a)
        self.assertIsNone(state.crashed_b)

    def test_failed_send_does_not_commit_a_predicted_move(self):
        brain = SnakeBrain()
        data = snapshot()
        with patch.object(run, 'BOT', brain), \
                patch.object(brain, 'choose_move', return_value='up'), \
                patch.object(brain, 'safe_direction', return_value='up'), \
                patch.object(brain, 'commit_move') as commit, \
                patch.object(run, 'log_action') as action, \
                patch.object(run, 'write_live_snapshot'), \
                patch.object(run, 'send', new=AsyncMock(side_effect=ConnectionError)):
            with self.assertRaises(ConnectionError):
                asyncio.run(run.process_move(None, {'data': data}))
            commit.assert_not_called()
            action.assert_not_called()

    def test_unicode_diagnostic_does_not_block_turn(self):
        brain = SnakeBrain()
        data = snapshot()
        data['board'] = '\u24b6' + data['board'][1:]
        buffer = io.BytesIO()
        console = io.TextIOWrapper(buffer, encoding='cp1252')
        with patch.object(run, 'BOT', brain), \
                patch.object(brain, 'choose_move', return_value='up'), \
                patch.object(brain, 'safe_direction', return_value='up'), \
                patch.object(brain, 'commit_move') as commit, \
                patch.object(run, 'log_action'), \
                patch.object(run, 'write_live_snapshot'), \
                patch.object(run, 'send', new=AsyncMock()) as send, \
                patch.object(run.sys, 'stdout', console):
            asyncio.run(run.process_move(None, {'data': data}))
            send.assert_awaited_once()
            commit.assert_called_once_with(data, 'up')
        console.flush()
        self.assertIn(b'\\u24b6', buffer.getvalue())

    def test_game_over_cleans_history(self):
        brain = SnakeBrain()
        brain.observe_state(snapshot())
        with patch.object(run, 'BOT', brain), \
                patch.object(run, 'log_event'), \
                patch.object(run, 'write_live_snapshot'), \
                patch.object(run, 'write_game_log'):
            asyncio.run(run.on_game_over(None, {'data': {'game_id': 'one'}}))
        self.assertNotIn('one', brain.crash_history)
        self.assertNotIn('one', brain.observed_states)


if __name__ == '__main__':
    unittest.main()
