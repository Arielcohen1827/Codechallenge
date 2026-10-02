import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import bot_weights
import run
import snake_brain
from food_planner import (
    FoodPlan,
    can_reach_tail,
    distance_map,
    food_chain_score,
    nearest_next_food_distance,
    simulate_path_to_food,
)
from snake_brain import SnakeBrain, final_safe_direction
from snake_state import (
    GameState,
    normalize_side,
    parse_board,
    parse_state,
    reconstruct_snake,
    release_times,
    shrink_wall,
    temporal_shortest_path,
    track_snake,
)
from test_run import FakeWebSocket, turn


def empty_state(rows=3, cols=3, snakes=None, **changes):
    values = {
        'rows': rows,
        'cols': cols,
        'board': tuple(' ' * cols for _ in range(rows)),
        'side': 'A',
        'enemy': 'B',
        'snakes': snakes or {'A': ((1, 1),), 'B': ((2, 2),)},
        'food': frozenset(),
        'scores': {'A': 0, 'B': 0},
        'remaining_moves': 20,
    }
    values.update(changes)
    return GameState(**values)


class TestWeightPersistence(unittest.TestCase):
    def setUp(self):
        self.original = dict(bot_weights.get_weights())
        self.addCleanup(bot_weights.set_active_weights, self.original)

    def test_set_reset_save_and_load_weights(self):
        key = 'center_unit_value'
        bot_weights.set_active_weights({key: '123', 'not_a_weight': 999})
        self.assertEqual(bot_weights.get_weights()[key], 123)
        self.assertNotIn('not_a_weight', bot_weights.get_weights())

        bot_weights.reset_active_weights()
        self.assertEqual(bot_weights.get_weights()[key], bot_weights.DEFAULT_WEIGHTS[key])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'nested', 'weights.json')
            bot_weights.save_weights(path, bot_weights.get_weights())
            loaded = bot_weights.load_weights(path)
            self.assertEqual(loaded, bot_weights.DEFAULT_WEIGHTS)
            self.assertTrue(bot_weights.load_active_weights(path))
            self.assertFalse(bot_weights.load_active_weights(Path(directory, 'missing.json')))


class TestStateEdgeCases(unittest.TestCase):
    def test_board_parsing_infers_dimensions_and_pads_rows(self):
        board, rows, cols = parse_board('|A |\n| B|')
        self.assertEqual((rows, cols), (2, 2))
        self.assertEqual(board, ('A ', ' B'))

        padded, rows, cols = parse_board('A', rows=3, cols=2)
        self.assertEqual((rows, cols), (3, 2))
        self.assertEqual(padded, ('A ', '  ', '  '))

    def test_invalid_board_size_and_side_fallbacks(self):
        state = parse_state({'board': '| A|', 'board_size': 'invalid', 'side': '?'})
        self.assertEqual(state.side, 'A')
        self.assertEqual(normalize_side('?', (' B',)), 'B')
        self.assertEqual(normalize_side('?', ('  ',)), 'A')
        self.assertEqual(reconstruct_snake(('   ',), 'A'), ())

    def test_previous_snake_tracking_variants(self):
        self.assertEqual(track_snake(((0, 0),), ('   ',), 'A'), ())
        self.assertIsNone(track_snake((), ('A  ',), 'A'))
        self.assertIsNone(track_snake(((0, 0),), ('  A',), 'A'))
        self.assertEqual(
            track_snake(((1, 1), (1, 0)), ('   ', ' aA'), 'A'),
            ((1, 2), (1, 1)),
        )
        self.assertEqual(
            track_snake(((1, 1), (1, 0)), ('   ', 'aaA'), 'A'),
            ((1, 2), (1, 1), (1, 0)),
        )

        state = parse_state(
            turn('|Aa |\n|  B|'),
            previous_snakes={'A': ((0, 0), (0, 1)), 'B': ((1, 2),)},
        )
        self.assertEqual(state.reliable_tails, frozenset({'A', 'B'}))

    def test_wall_shapes_release_times_and_temporal_limits(self):
        self.assertEqual(shrink_wall({(0, 0), (1, 0), (2, 0)}), frozenset({(1, 0)}))
        diagonal = frozenset({(0, 0), (1, 1)})
        self.assertEqual(shrink_wall(diagonal), diagonal)

        state = empty_state(
            snakes={'A': ((1, 1),), 'B': ()},
            reliable_tails=frozenset({'A'}),
            walls=frozenset({(0, 0), (1, 0), (2, 0)}),
        )
        releases = release_times(state)
        self.assertEqual(releases[(0, 0)], 2)
        self.assertEqual(temporal_shortest_path(state, None, (0, 0), 'A'), None)
        self.assertEqual(temporal_shortest_path(state, (1, 1), (1, 1), 'A'), ((1, 1),))
        self.assertIsNone(temporal_shortest_path(state, (1, 1), (2, 2), 'A', max_time=1))


class TestPlannerEdgeCases(unittest.TestCase):
    def test_short_plan_and_invalid_simulated_paths(self):
        plan = FoodPlan(
            food=(0, 0),
            path=((0, 0),),
            our_distance=0,
            enemy_distance=None,
            race_margin=0,
            space_after=1,
            exits_after=0,
            tail_reachable_after=False,
            next_food_distance=None,
            safety='SAFE',
            value=0,
        )
        self.assertIsNone(plan.first_move)

        wall_state = empty_state(walls=frozenset({(1, 2)}))
        self.assertIsNone(simulate_path_to_food(wall_state, 'A', ((1, 1), (1, 2))))
        self.assertIsNone(simulate_path_to_food(wall_state, 'A', ((1, 1), (0, 0))))

    def test_empty_planner_helpers(self):
        state = empty_state(snakes={'A': (), 'B': ((2, 2),)})
        self.assertFalse(can_reach_tail(state, 'A'))
        self.assertEqual(distance_map(state, None, 'A'), {})
        self.assertIsNone(nearest_next_food_distance(state, 'A', (0, 0)))
        self.assertEqual(food_chain_score(state, 'A', (0, 0), 0), 0)

    def test_numbered_sequence_without_a_next_target(self):
        state = empty_state(
            food_values={(0, 0): 1},
            next_food_digit=None,
        )
        self.assertIsNone(nearest_next_food_distance(state, 'A', (2, 2)))


class TestBrainEdgeCases(unittest.TestCase):
    def test_no_legal_move_and_forced_kill_shortcuts(self):
        brain = SnakeBrain()
        trapped = turn('|A|', game_id='trapped')
        self.assertEqual(brain.choose_move(trapped), 'up')
        self.assertEqual(brain.decision_debug('trapped')['reason'], 'no_legal_moves')

        data = turn('|A  |\n|   |\n|  B|', game_id='kill')
        with patch.object(snake_brain, 'forced_kill_move', return_value='right'):
            self.assertEqual(brain.choose_move(data), 'right')
        self.assertEqual(brain.decision_debug('kill')['reason'], 'forced_kill')

    def test_safe_direction_and_commit_without_cached_state(self):
        brain = SnakeBrain()
        data = turn('|A  |\n|   |\n|  B|', game_id='repair')
        repaired = brain.safe_direction(data, 'left')
        self.assertIn(repaired, {'right', 'down'})
        self.assertEqual(brain.decision_debug('repair')['reason'], 'safe_direction_repair')

        brain.current_states.clear()
        brain.commit_move(data, 'right')
        self.assertIn('repair', brain.previous_snakes)

        no_move = turn('|A|', game_id='none')
        self.assertEqual(brain.safe_direction(no_move, 'left'), 'up')
        self.assertEqual(final_safe_direction(no_move, 'left'), 'up')
        self.assertEqual(final_safe_direction(data, 'right'), 'right')

    def test_advantage_and_sequence_hold_shortcuts(self):
        data = turn('|A  |\n| * |\n|  B|', game_id='branches')
        plan = FoodPlan(
            food=(1, 1),
            path=((0, 0), (1, 0), (1, 1)),
            our_distance=2,
            enemy_distance=2,
            race_margin=1,
            space_after=8,
            exits_after=2,
            tail_reachable_after=True,
            next_food_distance=None,
            safety='SAFE',
            value=100,
        )
        deep = [
            {'direction': 'right', 'score': 10, 'safety': 'SAFE', 'our_replies': 2},
            {'direction': 'down', 'score': 5, 'safety': 'SAFE', 'our_replies': 2},
        ]

        brain = SnakeBrain(enable_debug=False)
        with (
            patch.object(snake_brain, 'choose_food_plan', return_value=None),
            patch.object(snake_brain, 'rank_deep_moves', return_value=deep),
            patch.object(
                snake_brain,
                'advantage_control_move',
                return_value={'active': True, 'override': True, 'direction': 'right'},
            ),
        ):
            self.assertEqual(brain.choose_move(data), 'right')
            self.assertEqual(brain.decision_debug('branches')['reason'], 'advantage_control')

        brain = SnakeBrain(enable_debug=False)
        with (
            patch.object(snake_brain, 'choose_food_plan', return_value=plan),
            patch.object(snake_brain, 'rank_deep_moves', return_value=deep),
            patch.object(snake_brain, 'should_commit_to_food', return_value=True),
            patch.object(
                snake_brain,
                'advantage_control_move',
                return_value={'active': False, 'override': False, 'direction': None},
            ),
            patch.object(snake_brain, 'sequence_gate_hold_move', return_value='right'),
        ):
            self.assertEqual(brain.choose_move(data), 'right')
            self.assertEqual(brain.decision_debug('branches')['reason'], 'sequence_gate_hold')

    def test_empty_deep_search_uses_positioning_fallback(self):
        brain = SnakeBrain(enable_debug=False)
        data = turn('|A  |\n|   |\n|  B|', game_id='fallback')
        with (
            patch.object(snake_brain, 'choose_food_plan', return_value=None),
            patch.object(snake_brain, 'rank_deep_moves', return_value=[]),
            patch.object(snake_brain, 'fallback_food_positioning', return_value='right'),
        ):
            self.assertEqual(brain.choose_move(data), 'right')
        self.assertEqual(brain.decision_debug('fallback')['reason'], 'fallback_positioning')


class TestProtocolEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def test_start_requires_websockets(self):
        with patch.object(run, 'websockets', None):
            with self.assertRaisesRegex(RuntimeError, 'websockets'):
                await run.start('token')

    async def test_start_connects_and_handles_keyboard_interrupt(self):
        websocket = FakeWebSocket()

        class Connection:
            async def __aenter__(self):
                return websocket

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class Websockets:
            @staticmethod
            def connect(uri):
                return Connection()

        with (
            patch.object(run, 'websockets', Websockets),
            patch.object(run, 'play', AsyncMock(side_effect=KeyboardInterrupt)),
        ):
            await run.start('secret')

    async def test_play_keyboard_interrupt_and_wall_action(self):
        websocket = FakeWebSocket()
        websocket.recv = AsyncMock(side_effect=KeyboardInterrupt)
        await run.play(websocket)

        websocket = FakeWebSocket()
        await run.process_wall(
            websocket,
            {'data': {'game_id': 'g-wall', 'turn_token': 'turn-wall'}},
        )
        self.assertEqual(websocket.sent[0]['action'], 'wall')

    def test_write_log_handles_os_error(self):
        with patch('pathlib.Path.open', side_effect=OSError('blocked')):
            run.write_game_log('cannot-write')
        with patch('pathlib.Path.mkdir', side_effect=OSError('blocked')):
            run.write_live_snapshot('cannot-write', {})
