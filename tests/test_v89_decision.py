"""Production action arbitration reuses one comparable root search."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import run
from snake_bot.snake_brain import SnakeBrain
from snake_bot.search_engine import SearchResult
from snake_bot.snake_state import apply_action, apply_move, step
from tests.test_v7_history import snapshot


def shooting_turn(score=6000, remaining=220, wall=False):
    data = snapshot(remaining=remaining, a=score,
                    body_b=((2, 5), (3, 5), (4, 5)))
    if wall:
        board = [list(row) for row in data['board'].splitlines()]
        board[2][4] = '#'
        data['board'] = '\n'.join(''.join(row) for row in board)
    return data


class ActionDecisionTests(unittest.TestCase):
    def decide(self, data, shoot_score, move_score=10, depth=4):
        brain = SnakeBrain()
        raw = {'up': move_score, 'down': move_score, 'right': move_score,
               ('shoot', 'right'): shoot_score}
        result = SearchResult(raw, depth, 123, 4, 1.5)
        with patch('snake_bot.move_search.IterativeSearchEngine.search', return_value=result) as search:
            action = brain.choose_action(data)
        self.assertEqual(search.call_count, 1, 'action choice must reuse the movement tree')
        return brain, action

    def test_profitable_head_shot_wins_strict_comparable_root_search(self):
        data = shooting_turn()
        brain, action = self.decide(data, 20)
        self.assertEqual(action, {'action_type': 'shoot', 'direction': 'right'})
        debug = brain.decision_debug('one')
        self.assertEqual(debug['action_type'], 'shoot')
        self.assertEqual(debug['action_search']['shoot_cost'], 5000)
        self.assertEqual(debug['action_search']['completed_depth'], 4)
        self.assertEqual(debug['action_search']['nodes'], 123)

    def test_cost_or_tie_or_incomplete_search_keep_a_move(self):
        for shoot_score, depth in ((-5000, 4), (10, 4), (100, 0), (100, 3)):
            with self.subTest(score=shoot_score, depth=depth):
                _, action = self.decide(shooting_turn(), shoot_score, depth=depth)
                self.assertEqual(action['action_type'], 'move')

    def test_no_affordable_or_blocked_or_zero_cut_shot_is_selected(self):
        cases = [shooting_turn(4999), shooting_turn(wall=True),
                 dict(shooting_turn(), board=snapshot(
                     remaining=220, a=6000, body_b=((2, 5),))['board'])]
        for data in cases:
            with self.subTest(score=data['score_1'], board=data['board']):
                _, action = self.decide(data, 10**8)
                self.assertEqual(action['action_type'], 'move')

    def test_real_strategy_shoot_preserves_wire_shape_and_tokens(self):
        data = shooting_turn()
        brain = SnakeBrain()
        result = SearchResult({'up': 0, 'down': 0, 'right': 0, ('shoot', 'right'): 100},
                              4, 12, 0, 1)
        with patch.object(run, 'BOT', brain), \
                patch('snake_bot.move_search.IterativeSearchEngine.search', return_value=result), \
                patch.object(run, 'send', new=AsyncMock()) as send, \
                patch.object(run, 'write_live_snapshot'), patch.object(run, 'log_verification'), \
                patch.object(run, 'log_action'), patch.object(run, 'print_diagnostic'):
            asyncio.run(run.process_move(None, {'data': data}))
        send.assert_awaited_once_with(None, 'shoot', {
            'game_id': 'one', 'turn_token': '220', 'direction': 'right'})
        self.assertEqual(brain.pending_moves['one'][2], 'shoot')

    def test_actual_search_funds_profitable_nearby_shot_food_without_mocked_scores(self):
        ours = ((3, 1), (3, 0), (4, 0))
        rival = ((0, 2), (1, 2), (2, 2), (3, 2), (3, 3), (3, 4),
                 (4, 4), (4, 3), (4, 2), (4, 1), (5, 1), (5, 2))
        data = snapshot(remaining=220, a=6000, body_a=ours, body_b=rival)
        data['multiplier_1'] = 60
        brain = SnakeBrain()
        brain.previous_snakes['one'] = {'A': ours, 'B': rival}
        with patch.object(brain, '_search_time_budget', return_value=250):
            action = brain.choose_action(data)
        self.assertEqual(action['action_type'], 'shoot')
        state = brain.current_states['one']
        after_shot = apply_action(state, 'shoot', action['direction'])
        self.assertEqual(after_shot.scores['A'], 1000)
        next_cell = step(state.head('A'), action['direction'])
        self.assertIn(next_cell, after_shot.shot_food)
        # The distant rival cannot recover this adjacent food in one turn.
        after_reply = apply_move(after_shot, 'right', 'B')
        after_food = apply_move(after_reply, action['direction'], 'A')
        self.assertEqual(after_food.scores['A'], 7000)
        self.assertGreater(brain.decision_debug('one')['action_search']['selected_score'],
                           brain.decision_debug('one')['action_search']['best_move_score'])

    def test_actual_horizon_keeps_winning_lead_instead_of_spending_5000(self):
        data = shooting_turn(remaining=1)
        data['score_2'] = 4000
        brain = SnakeBrain()
        action = brain.choose_action(data)
        self.assertEqual(action['action_type'], 'move')
        after = apply_action(brain.current_states['one'], action['action_type'], action['direction'])
        self.assertEqual(after.terminal_winner(), 'A')

    def test_stale_cached_tree_cannot_turn_early_move_into_shoot(self):
        brain = SnakeBrain()
        data = shooting_turn()
        state = brain.observe_state(data)
        brain.action_searches['one'] = (state, SearchResult({('shoot', 'right'): 10**9}, 6, 1, 0, 1))
        with patch.object(brain, 'choose_move', return_value='up'):
            self.assertEqual(brain.choose_action(data), {'action_type': 'move', 'direction': 'up'})
        self.assertNotIn('one', brain.action_searches)

    def test_proven_move_economics_can_repair_bomb_or_enemy_shoot_exposure(self):
        bomb = snapshot(remaining=220, a=10000,
                        body_b=((0, 6), (1, 6), (2, 6)))
        board = [list(row) for row in bomb['board'].splitlines()]
        board[5][2] = '\U0001f4a3'
        bomb['board'] = '\n'.join(''.join(row) for row in board)
        exposed = snapshot(remaining=220, b=6000,
                           body_b=((3, 5), (4, 5), (5, 5)))
        ordinary = snapshot(remaining=220)
        for data, expected, reason in ((bomb, 'up', 'bomb_economics'),
                                       (exposed, 'up', 'opponent_shoot_reply'),
                                       (ordinary, 'down', None)):
            with self.subTest(context=reason):
                brain = SnakeBrain()
                def choose(actual):
                    state = brain.observe_state(actual)
                    brain.last_decisions['one'] = {'event': 'decision_debug'}
                    brain.action_searches['one'] = (state, SearchResult(
                        {'up': 100, 'down': 0}, 3, 30, 0, 1))
                    return 'down'
                with patch.object(brain, 'choose_move', side_effect=choose):
                    action = brain.choose_action(data)
                self.assertEqual(action, {'action_type': 'move', 'direction': expected})
                if reason:
                    self.assertEqual(brain.decision_debug('one')['reason'], 'shared_search_' + reason)

    def test_unknown_positive_crash_history_is_not_invented_by_exposure_diagnostic(self):
        data = shooting_turn(score=6000, wall=True)
        # Move right reaches this wall; its score depends on unknown v7 history.
        rows = [list(row) for row in data['board'].splitlines()]
        rows[2][3] = '#'
        data['board'] = '\n'.join(''.join(row) for row in rows)
        brain = SnakeBrain()
        def choose(actual):
            state = brain.observe_state(actual)
            brain.action_searches['one'] = (state, SearchResult({'right': 0, 'up': 100}, 4, 1, 0, 1))
            return 'right'
        with patch.object(brain, 'choose_move', side_effect=choose):
            action = brain.choose_action(data)
        self.assertEqual(action, {'action_type': 'move', 'direction': 'right'})
        self.assertIsNone(brain.observed_states['one'].crashed_a)
        self.assertIsNone(brain.crash_history['one']['A'])
        self.assertEqual(brain.observed_states['one'].scores['A'], 6000)

    def test_unselected_exposed_move_does_not_replace_safe_legacy_plan(self):
        data = snapshot(remaining=220, b=6000, body_b=((3, 5), (4, 5), (5, 5)))
        brain = SnakeBrain()
        def choose(actual):
            state = brain.observe_state(actual)
            brain.action_searches['one'] = (state, SearchResult({'up': 0, 'down': 100}, 4, 1, 0, 1))
            return 'up'
        with patch.object(brain, 'choose_move', side_effect=choose):
            action = brain.choose_action(data)
        self.assertEqual(action, {'action_type': 'move', 'direction': 'up'})

    def test_bomb_is_not_hard_banned_and_uncompleted_tree_is_not_used(self):
        data = snapshot(remaining=220, a=10000, body_b=((0, 6), (1, 6), (2, 6)))
        rows = [list(row) for row in data['board'].splitlines()]
        rows[5][2] = '\U0001f4a3'
        data['board'] = '\n'.join(''.join(row) for row in rows)
        for depth, values in ((3, {'up': 0, 'down': 100}), (0, {'up': 100, 'down': 0})):
            with self.subTest(depth=depth):
                brain = SnakeBrain()
                def choose(actual):
                    state = brain.observe_state(actual)
                    brain.action_searches['one'] = (state, SearchResult(values, depth, 1, 0, 1))
                    return 'down'
                with patch.object(brain, 'choose_move', side_effect=choose):
                    action = brain.choose_action(data)
                self.assertEqual(action, {'action_type': 'move', 'direction': 'down'})

    def test_trapped_head_can_use_one_root_tree_to_shoot_instead_of_wiping_score(self):
        data = snapshot(remaining=220, a=40000,
            body_a=((0, 0), (1, 0), (2, 0)),
            body_b=((0, 5), (0, 4), (0, 3), (0, 2), (0, 1)))
        brain = SnakeBrain()
        brain.previous_snakes['one'] = {'A': ((0, 0), (1, 0), (2, 0)),
                                       'B': ((0, 5), (0, 4), (0, 3), (0, 2), (0, 1))}
        with patch.object(brain, '_search_time_budget', return_value=250):
            action = brain.choose_action(data)
        self.assertEqual(action, {'action_type': 'shoot', 'direction': 'right'})
        after = apply_action(brain.current_states['one'], action['action_type'], action['direction'])
        self.assertEqual(after.scores['A'], 35000)
        self.assertFalse(after.last_action['bomb_triggered'])


if __name__ == '__main__':
    unittest.main()
