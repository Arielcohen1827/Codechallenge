"""Wire actions and observational history for cumulative v8/v9 rules."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import run
from snake_bot.snake_brain import SnakeBrain
from snake_bot.snake_state import parse_state
from tests.test_v7_history import snapshot
from v7_logging import sent_prediction, verification_record


class ProtocolTests(unittest.TestCase):
    def test_shoot_wire_preserves_required_fields_and_commits_action(self):
        data = snapshot(a=6000, body_b=((2, 5), (3, 5), (4, 5)))
        brain = SnakeBrain()
        brain.observe_state(data)
        with (
            patch.object(run, 'BOT', brain),
            patch.object(
                brain,
                'choose_action',
                return_value={'action_type': 'shoot', 'direction': 'right'},
            ),
            patch.object(run, 'send', new=AsyncMock()) as send,
            patch.object(run, 'write_live_snapshot'),
            patch.object(run, 'log_verification'),
            patch.object(run, 'log_action'),
            patch.object(run, 'print_diagnostic'),
        ):
            asyncio.run(run.process_move(None, {'data': data}))
        send.assert_awaited_once_with(
            None,
            'shoot',
            {'game_id': 'one', 'turn_token': '400', 'direction': 'right'},
        )
        self.assertEqual(brain.pending_moves['one'][2], 'shoot')
        self.assertEqual(
            brain.previous_snakes['one']['A'],
            brain.observed_states['one'].body('A'),
        )

    def test_low_score_shoot_is_repaired_before_wire(self):
        for score in (0, 4999):
            with self.subTest(score=score):
                data = snapshot(a=score)
                brain = SnakeBrain()
                with (
                    patch.object(run, 'BOT', brain),
                    patch.object(
                        brain,
                        'choose_action',
                        return_value={
                            'action_type': 'shoot',
                            'direction': 'up',
                        },
                    ),
                    patch.object(run, 'send', new=AsyncMock()) as send,
                    patch.object(run, 'write_live_snapshot'),
                    patch.object(run, 'log_verification'),
                    patch.object(run, 'log_action'),
                    patch.object(run, 'print_diagnostic'),
                ):
                    asyncio.run(run.process_move(None, {'data': data}))
                self.assertEqual(send.await_args.args[1], 'move')

    def test_prediction_records_shoot_hit_and_miss_without_confirmation(self):
        for direction, hit in (('right', True), ('up', False)):
            data = snapshot(a=6000, body_b=((2, 5), (3, 5), (4, 5)))
            brain = SnakeBrain()
            state = brain.observe_state(data)
            record = verification_record(data, state, direction=direction, action_type='shoot')
            predicted = record['sent_action_prediction']
            self.assertEqual(predicted['source'], 'prediction_not_server_confirmation')
            self.assertEqual(predicted['action_type'], 'shoot')
            self.assertEqual(predicted['shoot_cost'], 5000)
            self.assertEqual(predicted['hit'], hit)
            self.assertEqual(predicted['score_before'], 6000)
            self.assertEqual(predicted['score_after'], 1000)
            self.assertEqual(predicted['hit_coordinate'], (2, 5) if hit else None)
            self.assertEqual(predicted['cut_length'], 2 if hit else 0)
            self.assertEqual(len(predicted['shot_food_generated']), 2 if hit else 0)
            json.dumps(record)

    def test_shoot_does_not_confirm_crash_from_body_shrink_across_gap(self):
        brain = SnakeBrain()
        brain.observe_state(
            snapshot(b=7000, body_a=((2, 2), (2, 1), (2, 0), (3, 0)))
        )
        after = brain.observe_state(
            snapshot(390, b=2000, body_a=((1, 2), (1, 1), (1, 0)))
        )
        self.assertIsNone(after.crashed_a)

    def test_confirmed_shoot_geometry_preserves_known_no_crash(self):
        for score in (5000, 6000):
            brain = SnakeBrain()
            data = snapshot(a=score)
            brain.observe_state(data)
            brain.commit_action(
                data, {'action_type': 'shoot', 'direction': 'up'}
            )
            state = brain.observe_state(
                snapshot(
                    398, a=score - 5000, b=1, body_b=((3, 4), (4, 4), (4, 5))
                )
            )
            self.assertIs(state.crashed_a, False)
            self.assertIs(state.crashed_b, False)

    def test_wire_boundary_refuses_unsafe_custom_shoot_policy(self):
        data = snapshot(a=4999)
        brain = SnakeBrain()
        attempted = {'action_type': 'shoot', 'direction': 'up'}
        with patch.object(run, 'BOT', brain), \
                patch.object(brain, 'choose_action', return_value=attempted), \
                patch.object(brain, 'safe_action', return_value=attempted), \
                patch.object(run, 'send', new=AsyncMock()) as send, \
                patch.object(run, 'write_live_snapshot'), patch.object(run, 'log_verification'), \
                patch.object(run, 'log_action'), patch.object(run, 'print_diagnostic'):
            asyncio.run(run.process_move(None, {'data': data}))
        self.assertEqual(send.await_args.args[1], 'move')

    def test_live_and_history_record_shoot_without_optional_decision_debug(
        self,
    ):
        data = snapshot(a=6000, body_b=((2, 5), (3, 5), (4, 5)))
        brain = SnakeBrain(enable_debug=False)
        brain.observe_state(data)
        attempted = {'action_type': 'shoot', 'direction': 'right'}
        with (
            patch.object(run, 'BOT', brain),
            patch.object(run, 'HISTORY', {}) as history,
            patch.object(run, 'LOGGED_META', set()),
            patch.object(brain, 'choose_action', return_value=attempted),
            patch.object(run, 'send', new=AsyncMock()),
            patch.object(run, 'write_live_snapshot') as live,
            patch.object(run, 'print_diagnostic'),
        ):
            asyncio.run(run.process_move(None, {'data': data}))
            records = [json.loads(line[2:]) for line in history['one']]
        self.assertEqual(live.call_args.kwargs['action_type'], 'shoot')
        self.assertEqual(records[-1]['action_type'], 'shoot')
        prediction = next(
            item['verification']['sent_action_prediction']
            for item in records
            if 'verification' in item
        )
        self.assertEqual(prediction['shoot_cost'], 5000)
        self.assertEqual(prediction['score_after'], 1000)

    def test_own_shoot_and_rival_cut_do_not_invent_first_crash(self):
        brain = SnakeBrain()
        data = snapshot(
            a=5000, b=7000, body_a=((2, 2), (2, 1), (2, 0), (3, 0))
        )
        brain.observe_state(data)
        brain.commit_action(data, {'action_type': 'shoot', 'direction': 'up'})
        state = brain.observe_state(snapshot(398, a=0, b=2000))
        self.assertIs(state.crashed_a, False)

    def test_unobserved_enemy_shot_at_exact_cost_is_not_confirmed_crash(self):
        brain = SnakeBrain()
        data = snapshot(b=5000)
        brain.observe_state(data)
        brain.commit_move(data, 'up')
        state = brain.observe_state(
            snapshot(398, a=1, b=0, body_a=((1, 2), (2, 2), (2, 1)))
        )
        self.assertIsNone(state.crashed_b)

    def test_bomb_prediction_keeps_unknown_respawn_separate_from_server_snapshot(self):
        data = snapshot(a=20)
        lines = [list(row) for row in data['board'].splitlines()]
        lines[1][4] = '\U0001f4a3'
        lines[6][0] = '*'
        data['board'] = '\n'.join(''.join(row) for row in lines)
        state = parse_state(data)
        prediction = sent_prediction(data, state, 'up')
        self.assertTrue(prediction['bomb_triggered'])
        self.assertEqual(prediction['bomb_type'], 'large')
        self.assertEqual(prediction['score_before'], 20)
        self.assertEqual(prediction['score_after'], 0)
        self.assertEqual(prediction['pending_bomb_respawns'], ('large',))
        self.assertIn('bomb_respawn_position_unknown', prediction['uncertainty'])
        observed = verification_record(data, state)['after']
        self.assertEqual(observed['source'], 'server_snapshot')
        self.assertEqual(observed['v8_v9_cells']['bombs_large'], [[1, 4]])
        self.assertEqual(observed['v8_v9_cells']['shot_food'], [[6, 0]])


if __name__ == '__main__':
    unittest.main()
