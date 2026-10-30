"""Shooting observations remain separate from predictions and turn permissions."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import run
from snake_bot.snake_brain import SnakeBrain
from snake_bot.snake_state import GameState, apply_action
from tools.bot_lab import make_turn_data
from v7_logging import safe_message, verification_record


def source(score=6000):
    return GameState(rows=20, cols=30, board=(), side='A', enemy='B',
        snakes={'A': ((5, 5), (5, 4), (5, 3), (5, 2)),
                'B': ((5, 10), (6, 10), (7, 10), (8, 10))},
        food=frozenset(), scores={'A': score, 'B': 100}, remaining_moves=400,
        reliable_tails=frozenset({'A', 'B'}), crashed_a=False, crashed_b=False)


def observed(state, *, action=None, direction=None, token=True):
    data = make_turn_data(state, 'v9-observed', 0)
    if action is not None:
        data.update(action=action, direction=direction)
    if not token:
        data.pop('turn_token', None)
    return data


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.brain = SnakeBrain()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(run, 'BOT', self.brain))
        self.history = self.stack.enter_context(patch.object(run, 'HISTORY', {}))
        self.stack.enter_context(patch.object(run, 'LOGGED_META', set()))
        self.live = self.stack.enter_context(patch.object(run, 'write_live_snapshot'))
        self.stack.enter_context(patch.object(run, 'print_diagnostic'))

    async def report(self, data, **envelope):
        await run.HANDLERS['play_data'](None, {'event': 'play_data', 'data': data, **envelope})

    def test_server_action_is_retained_without_promoting_prediction_metadata(self):
        message = safe_message({'event': 'play_data', 'data': {
            'game_id': 'v9-observed', 'action': 'shoot', 'direction': 'right',
            'turn_token': 'synthetic-private', 'unknown_hit_claim': True}})
        self.assertEqual(message['data']['action'], 'shoot')
        self.assertEqual(message['data']['direction'], 'right')
        self.assertNotIn('turn_token', message['data'])
        self.assertNotIn('unknown_hit_claim', message['data'])

    async def test_partial_action_only_event_is_logged_without_a_turn_or_state_update(self):
        self.brain.observe_state(observed(source()))
        previous = self.brain.observed_states['v9-observed']
        with patch.object(self.brain, 'choose_action') as choose, patch.object(run, 'send') as send:
            await self.report({'game_id': 'v9-observed', 'action': 'shoot', 'direction': 'right'})
        self.assertIs(self.brain.observed_states['v9-observed'], previous)
        choose.assert_not_called()
        send.assert_not_called()
        self.live.assert_not_called()
        event = next(json.loads(line[2:]) for line in self.history['v9-observed'] if line.startswith('< '))
        self.assertEqual(event['data']['action'], 'shoot')
        self.assertFalse(any('verification' in json.loads(line[2:])
                             for line in self.history['v9-observed']),
                         'partial metadata must not replace the last complete audit snapshot')

    async def test_shoot_play_data_then_enemy_move_and_new_turn_preserve_history(self):
        for reported_enemy_action in (False, True):
            with self.subTest(reported_enemy_action=reported_enemy_action):
                self.brain.forget('v9-observed')
                initial = source(5000)
                data = observed(initial)
                self.brain.observe_state(data)
                self.brain.commit_action(data, {'action_type': 'shoot', 'direction': 'right'})
                after_shot = apply_action(initial, 'shoot', 'right')
                shot_data = observed(after_shot, token=False)
                if reported_enemy_action:
                    await self.report(shot_data, action='shoot', direction='right')
                else:
                    await self.report({**shot_data, 'action': 'shoot', 'direction': 'right'})
                self.assertEqual(self.brain.crash_history['v9-observed'], {'A': False, 'B': False})
                self.assertNotIn('v9-observed', self.brain.pending_moves)
                after_enemy = apply_action(after_shot, 'move', 'down')
                enemy_data = observed(after_enemy, action='move' if reported_enemy_action else None,
                                      direction='down', token=False)
                await self.report(enemy_data)
                if not reported_enemy_action:
                    self.assertIsNone(self.live.call_args.kwargs['action_type'])
                self.assertEqual(self.brain.crash_history['v9-observed'], {'A': False, 'B': False})
                turn = observed(after_enemy)
                turn['turn_token'] = 'synthetic-new-permission'
                def move_action(payload):
                    self.brain.observe_state(payload)
                    return {'action_type': 'move', 'direction': 'up'}
                with patch.object(self.brain, '_choose_action', side_effect=move_action) as choose, \
                        patch.object(run, 'send', new_callable=unittest.mock.AsyncMock) as send:
                    await run.on_your_turn(None, {'event': 'your_turn', 'data': turn})
                choose.assert_called_once()
                send.assert_awaited_once()
                self.assertEqual(self.brain.crash_history['v9-observed'], {'A': False, 'B': False})
                self.assertEqual(len(self.brain.pending_moves['v9-observed']), 2)
                self.assertEqual(self.brain.observed_states['v9-observed'].body('B'), after_enemy.body('B'))

    async def test_contradiction_correction_duplicate_and_stale_never_confirm_a_crash(self):
        initial = source(5000)
        data = observed(initial)
        self.brain.observe_state(data)
        self.brain.commit_action(data, {'action_type': 'shoot', 'direction': 'right'})
        shot = apply_action(initial, 'shoot', 'right')
        ambiguous = observed(replace(shot, side='A', enemy='B'), action='shoot', direction='right')
        await self.report(ambiguous)
        self.assertIn('v9-observed', self.brain.pending_moves,
                      'a stationary head/cost cannot establish an ambiguous action actor')
        bad = observed(shot, action='shoot', direction='left')
        await self.report(bad)
        self.assertIsNone(self.brain.crash_history['v9-observed']['A'])
        self.assertIn('v9-observed', self.brain.pending_moves)
        good = observed(shot, action='shoot', direction='right')
        await self.report(good)
        self.assertEqual(self.brain.crash_history['v9-observed'], {'A': False, 'B': False})
        self.assertNotIn('v9-observed', self.brain.pending_moves)
        confirmed = self.brain.observed_states['v9-observed']
        await self.report({**good, 'turn_token': 'synthetic-refresh'})
        self.assertIs(self.brain.observed_states['v9-observed'], confirmed)
        await self.report(data)
        self.assertIs(self.brain.observed_states['v9-observed'], confirmed)
        self.assertEqual(self.brain.crash_history['v9-observed'], {'A': False, 'B': False})

    def test_shoot_prediction_reconciles_victim_and_food_without_server_confirmation(self):
        initial = source()
        brain = SnakeBrain()
        data = observed(initial)
        state = brain.observe_state(data)
        prediction = verification_record(data, state, direction='right', action_type='shoot')
        after = apply_action(initial, 'shoot', 'right')
        good = verification_record(observed(after, action='shoot', direction='right'), previous=prediction)
        self.assertEqual(good['prediction_reconciliation'], 'consistent_predicted_shoot_effects')
        self.assertEqual(good['after']['server_reported_action'], {'action_type': 'shoot', 'direction': 'right'})
        self.assertEqual(prediction['sent_action_prediction']['source'], 'prediction_not_server_confirmation')
        wrong = replace(after, snakes={**after.snakes, 'B': initial.body('B')})
        record = verification_record(observed(wrong, action='shoot', direction='right'), previous=prediction)
        self.assertEqual(record['prediction_reconciliation'], 'differs_predicted_shoot_effects')
        wrong_food = replace(after, shot_food=frozenset())
        record = verification_record(observed(wrong_food, action='shoot', direction='right'), previous=prediction)
        self.assertEqual(record['prediction_reconciliation'], 'differs_predicted_shoot_effects')


class RealWebsocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_loopback_serializes_move_shoot_exact_cost_and_low_score_guard(self):
        import websockets
        messages = []
        async def receive(socket):
            messages.append(json.loads(await socket.recv()))
            await socket.send('received')
        async with websockets.serve(receive, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            for index, (score, requested, expected) in enumerate((
                    (6000, 'shoot', 'shoot'), (5000, 'shoot', 'shoot'),
                    (6000, 'move', 'move'), (4999, 'shoot', 'move'))):
                data = observed(source(score))
                data['turn_token'] = f'synthetic-loopback-{index}'
                brain = SnakeBrain()
                def chosen(payload):
                    brain.observe_state(payload)
                    return {'action_type': requested, 'direction': 'right'}
                with patch.object(run, 'BOT', brain), patch.object(brain, 'choose_action', side_effect=chosen), \
                        patch.object(run, 'HISTORY', {}), patch.object(run, 'LOGGED_META', set()), \
                        patch.object(run, 'write_live_snapshot'), patch.object(run, 'print_diagnostic'):
                    async with websockets.connect(f'ws://127.0.0.1:{port}') as client:
                        await run.process_move(client, {'data': data})
                        self.assertEqual(await asyncio.wait_for(client.recv(), 3), 'received')
                self.assertEqual(messages[-1], {'action': expected, 'data': {
                    'game_id': data['game_id'], 'turn_token': data['turn_token'], 'direction': 'right'}})


if __name__ == '__main__':
    unittest.main()
