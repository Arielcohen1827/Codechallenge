import json
import os
import tempfile
import unittest
from dataclasses import replace

import run
from food_planner import (
    FoodPlan,
    advantage_control_context,
    advantage_control_move,
    build_food_plan,
    choose_food_plan,
    should_commit_to_food,
    tempo_race_margin,
)
from move_search import rank_deep_moves
from search_engine import IterativeSearchEngine
from snake_state import GameState, legal_moves, shrink_wall, temporal_shortest_path, track_snake


class FakeWebSocket:
    def __init__(self, incoming=()):
        self.incoming = [m if isinstance(m, str) else json.dumps(m) for m in incoming]
        self.sent = []

    async def recv(self):
        if not self.incoming:
            raise ConnectionResetError('no more messages')
        return self.incoming.pop(0)

    async def send(self, message):
        self.sent.append(json.loads(message))


def turn(board, side='A', score_1=0, score_2=0, remaining=250, game_id='g_1', **extra):
    rows = len(board.splitlines())
    cols = max(len(line.strip('|')) for line in board.splitlines())
    data = {
        'game_id': game_id,
        'turn_token': 't_1',
        'side': side,
        'board': board,
        'rows': rows,
        'cols': cols,
        'score_1': score_1,
        'score_2': score_2,
        'remaining_moves': remaining,
    }
    data.update(extra)
    return data


class HistoryTestCase(unittest.TestCase):
    def setUp(self):
        run.HISTORY.clear()
        run.BOT = run.SnakeBrain()
        self.addCleanup(run.HISTORY.clear)


class InTempDirTestCase(HistoryTestCase):
    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        previous_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        self.addCleanup(os.chdir, previous_cwd)

    def read_log(self, game_id):
        with open(os.path.join(self.tmpdir.name, 'games', f"game_{game_id}.log")) as f:
            return f.read()


class TestProtocol(InTempDirTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_challenge_is_accepted(self):
        websocket = FakeWebSocket([
            {'event': 'challenge', 'data': {'challenge_id': 'c_1'}},
        ])

        await run.play(websocket)

        self.assertEqual(websocket.sent, [{'action': 'accept_challenge', 'data': {'challenge_id': 'c_1'}}])

    async def test_your_turn_sends_direction_move(self):
        websocket = FakeWebSocket()

        await run.process_move(websocket, {'event': 'your_turn', 'data': turn('|A* |\n|   |\n|  B|')})

        self.assertEqual(websocket.sent[0]['action'], 'move')
        self.assertEqual(websocket.sent[0]['data']['direction'], 'right')
        self.assertIn('turn_token', websocket.sent[0]['data'])
        live_path = os.path.join(self.tmpdir.name, 'games', 'live', 'game_g_1.json')
        with open(live_path, encoding='utf-8') as file:
            snapshot = json.load(file)
        self.assertEqual(snapshot['direction'], 'right')
        self.assertEqual(snapshot['turn_data']['board'], '|A* |\n|   |\n|  B|')

    async def test_full_match_is_logged_in_order(self):
        websocket = FakeWebSocket([
            {'event': 'challenge', 'data': {'challenge_id': 'c_1'}},
            {'event': 'your_turn', 'data': turn('|A* |\n|   |\n|  B|')},
            {'event': 'game_over', 'data': {'game_id': 'g_1'}},
        ])

        await run.play(websocket)

        self.assertEqual([item['action'] for item in websocket.sent], ['accept_challenge', 'move'])
        self.assertEqual([line[0] for line in self.read_log('g_1').splitlines()], ['=', '<', '?', '>', '<'])

    async def test_game_log_includes_bot_version_metadata(self):
        websocket = FakeWebSocket([
            {'event': 'your_turn', 'data': turn('|A* |\n|   |\n|  B|')},
            {'event': 'game_over', 'data': {'game_id': 'g_1'}},
        ])

        await run.play(websocket)

        first_line = self.read_log('g_1').splitlines()[0]
        metadata = json.loads(first_line[2:])
        self.assertEqual(first_line[:2], '= ')
        self.assertEqual(metadata['event'], 'bot_version')
        self.assertEqual(metadata['version'], run.BOT_VERSION)

    async def test_game_log_includes_decision_debug(self):
        websocket = FakeWebSocket([
            {'event': 'your_turn', 'data': turn('|A* |\n|   |\n|  B|')},
            {'event': 'game_over', 'data': {'game_id': 'g_1'}},
        ])

        await run.play(websocket)

        debug_line = next(line for line in self.read_log('g_1').splitlines() if line.startswith('? '))
        debug = json.loads(debug_line[2:])
        self.assertEqual(debug['event'], 'decision_debug')
        self.assertEqual(debug['direction'], 'right')
        self.assertEqual(debug['plan']['food'], [0, 1])
        self.assertIn('candidate_moves', debug)

    async def test_malformed_message_stops_without_reply(self):
        websocket = FakeWebSocket(['not json'])

        await run.play(websocket)

        self.assertEqual(websocket.sent, [])


class TestFoodFirstBrain(HistoryTestCase):
    def choose(self, board, side='A', score_1=0, score_2=0, remaining=250, game_id='g_1'):
        return run.BOT.choose_move(turn(board, side, score_1, score_2, remaining, game_id))

    def test_adjacent_open_food_is_eaten(self):
        self.assertEqual(self.choose('|A*   |\n|     |\n|    B|'), 'right')

    def test_numbered_food_detects_cyclic_next_digit(self):
        board = '|A 8 9|\n| 1 2 |\n|3   B|'
        state = run.parse_state(turn(board))

        self.assertEqual(state.next_food_digit, 8)
        self.assertEqual(state.food, frozenset({(0, 2)}))
        self.assertEqual(set(state.food_values.values()), {1, 2, 3, 8, 9})
        self.assertEqual([digit for digit, _ in state.numbered_sequence()], [8, 9, 1, 2, 3])

    def test_correct_number_grows_and_uses_multiplier(self):
        board = '|A1 23|\n|  45B|'
        state = run.parse_state(turn(board, multiplier_1=2, multiplier_2=1))
        before_length = len(state.body('A'))

        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(len(after.body('A')), before_length + 1)
        self.assertEqual(after.scores['A'], 200)
        self.assertEqual(after.next_food_digit, 2)

    def test_correct_copy_removes_every_copy_and_advances_once(self):
        board = '|A1 12|\n|1 23B|\n| 345 |'
        state = run.parse_state(turn(board))
        before_length = len(state.body('A'))

        self.assertEqual(state.food, frozenset({(0, 1), (0, 3), (1, 0)}))
        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(len(after.body('A')), before_length + 1)
        self.assertEqual(after.scores['A'], 100)
        self.assertNotIn(1, after.food_values.values())
        self.assertEqual(after.next_food_digit, 2)
        self.assertEqual(
            after.food,
            frozenset(pos for pos, digit in after.food_values.items() if digit == 2),
        )

    def test_wrong_copy_only_removes_the_touched_cell(self):
        board = '|A2 12|\n|2 34B|\n| 5 2 |'
        state = run.parse_state(turn(board))
        before_copies = sum(value == 2 for value in state.food_values.values())

        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(after.scores['A'], -500)
        self.assertEqual(after.next_food_digit, 1)
        self.assertNotIn((0, 1), after.food_values)
        self.assertEqual(
            sum(value == 2 for value in after.food_values.values()),
            before_copies - 1,
        )

    def test_numbered_groups_preserve_all_copies(self):
        board = '|A1 12|\n|1 23B|\n| 345 |'
        state = run.parse_state(turn(board))

        groups = state.numbered_groups(limit=2)

        self.assertEqual(groups[0], (1, ((0, 1), (0, 3), (1, 0))))
        self.assertEqual(groups[1], (2, ((0, 4), (1, 2))))

    def test_wrong_number_penalizes_without_growth(self):
        board = '|A2 34|\n|1  5B|'
        state = run.parse_state(turn(board))
        before_length = len(state.body('A'))

        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(len(after.body('A')), before_length)
        self.assertEqual(after.scores['A'], -500)
        self.assertEqual(after.next_food_digit, 1)

    def test_bot_avoids_wrong_digit_when_another_move_is_available(self):
        board = '|A2 34|\n|1  5B|\n|     |'

        self.assertNotEqual(self.choose(board), 'right')

    def test_multiplier_pickup_scores_and_stacks_without_growth(self):
        board = '|AX 12|\n|345 B|'
        state = run.parse_state(turn(board, multiplier_1=2, multiplier_2=1))
        before_length = len(state.body('A'))

        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(len(after.body('A')), before_length)
        self.assertEqual(after.scores['A'], 50)
        self.assertEqual(after.multipliers['A'], 3)

    def test_wall_hit_penalizes_without_moving_or_ending_game(self):
        board = '|A#   B|\n|       |'
        state = run.parse_state(turn(board, score_1=1000))
        before = state.body('A')

        after = run.apply_move(state, 'right', 'A')

        self.assertEqual(state.walls, frozenset({(0, 1)}))
        self.assertIn('right', run.legal_moves(state, 'A'))
        self.assertEqual(after.body('A'), before)
        self.assertEqual(after.scores['A'], 500)
        self.assertEqual(after.side, 'B')

    def test_wall_shrinks_at_b_round_end(self):
        board = '|A      |\n| ##### |\n|    B  |'
        state = run.parse_state(turn(board, side='B'))

        after = run.apply_move(state, 'left', 'B')

        self.assertEqual(after.walls, frozenset({(1, 2), (1, 3), (1, 4)}))
        self.assertEqual(shrink_wall(after.walls), frozenset({(1, 3)}))
        self.assertEqual(shrink_wall({(1, 3)}), frozenset())

    def test_temporal_path_uses_wall_endpoint_after_it_shrinks(self):
        state = GameState(
            rows=3,
            cols=6,
            board=tuple(' ' * 6 for _ in range(3)),
            side='A',
            enemy='B',
            snakes={'A': ((1, 0),), 'B': ((2, 5),)},
            food=frozenset(),
            scores={'A': 0, 'B': 0},
            remaining_moves=20,
            reliable_tails=frozenset({'A', 'B'}),
            walls=frozenset({(1, 2), (1, 3), (1, 4)}),
        )

        self.assertEqual(
            temporal_shortest_path(state, (1, 0), (1, 2), 'A'),
            ((1, 0), (1, 1), (1, 2)),
        )

    def test_bot_avoids_wall_hit_when_open_move_exists(self):
        board = '|A# 1 |\n|     |\n|    B|'

        self.assertNotEqual(self.choose(board), 'right')

    def test_wall_bump_remains_available_when_it_is_only_action(self):
        board = '|aaaB|\n|aA# |\n|aaa |'
        state = run.parse_state(turn(board))

        self.assertEqual(run.legal_moves(state, 'A'), ['right'])
        self.assertEqual(self.choose(board), 'right')

    def test_wrong_digit_is_compared_against_wall_when_both_are_penalties(self):
        board = (
            '| # 34|\n'
            '|aA2 5|\n'
            '|aa1 B|'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_advantage_control_activates_only_with_a_defensible_lead(self):
        board = (
            '|###1###|\n'
            '|aaA    |\n'
            '|a  Bbbb|\n'
            '|a      |\n'
            '|a      |\n'
            '|a      |\n'
            '|a      |'
        )
        leading = run.parse_state(turn(board, score_1=2200, score_2=0, remaining=100))
        close = replace(leading, scores={'A': 300, 'B': 0})

        self.assertTrue(advantage_control_context(leading, 'A')['active'])
        self.assertFalse(advantage_control_context(close, 'A')['active'])

    def test_advantage_control_occupies_food_chokepoint(self):
        board = (
            '|###1###|\n'
            '|aaA    |\n'
            '|a  Bbbb|\n'
            '|a      |\n'
            '|a      |\n'
            '|a      |\n'
            '|a      |'
        )
        state = run.parse_state(turn(board, score_1=2200, score_2=0, remaining=100))
        analyses = [
            {
                'direction': direction,
                'score': 0,
                'forced_loss': False,
                'safety': 'SAFE',
                'our_replies': 3,
                'exits': 3,
                'enemy_food': False,
                'voronoi_ours': 10,
                'voronoi_enemy': 10,
            }
            for direction in ('down', 'right')
        ]

        control = advantage_control_move(
            state,
            'A',
            ['down', 'right'],
            analyses,
        )

        self.assertTrue(control['active'])
        self.assertEqual(control['direction'], 'right')
        self.assertGreater(control['moves'][0]['food_delay'], 0)

    def test_planner_targets_multiplier_or_correct_digit_never_wrong_digit(self):
        board = '|AX    |\n|      |\n|1 2345|\n|     B|'
        state = run.parse_state(turn(board, remaining=240))

        plan = choose_food_plan(state, 'A')

        self.assertIsNotNone(plan)
        self.assertIn(plan.food, state.objective_cells())
        self.assertNotIn(plan.food, state.wrong_food())

    def test_lost_current_race_positions_for_next_number(self):
        board = (
            '|A 2   |\n'
            '|      |\n'
            '|3 4 5 |\n'
            '|    1B|'
        )
        state = run.parse_state(turn(board))

        plan = choose_food_plan(state, 'A', hunger=12)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.objective_kind, 'sequence_setup')
        self.assertEqual(plan.sequence_rank, 1)
        self.assertEqual(abs(plan.food[0] - 0) + abs(plan.food[1] - 2), 1)

    def test_multiplier_is_more_valuable_when_we_are_behind_in_multiplier(self):
        board = '|A X 1|\n| 2345|\n|    B|'
        base = run.parse_state(turn(board))
        pickup = next(iter(base.pickups))
        behind = replace(base, multipliers={'A': 1, 'B': 5})
        ahead = replace(base, multipliers={'A': 5, 'B': 1})

        behind_plan = build_food_plan(behind, 'A', pickup, None, 0)
        ahead_plan = build_food_plan(ahead, 'A', pickup, None, 0)

        self.assertGreater(behind_plan.value, ahead_plan.value)

    def test_adjacent_multiplier_is_rejected_when_rival_can_close_every_exit(self):
        board = (
            '|              |\n'
            '|              |\n'
            '|              |\n'
            '|            3 |\n'
            '|         X    |\n'
            '|   aa   b 9   |\n'
            '|    a   b     |\n'
            '|    a   bb   2|\n'
            '|   aa  bbb    |\n'
            '|8  aaa b      |\n'
            '|     a bb1    |\n'
            '|     A XB     |'
        )
        data = turn(
            board,
            side='B',
            score_1=47254,
            score_2=13806,
            remaining=55,
            multiplier_1=12,
            multiplier_2=9,
        )

        move = run.BOT.choose_move(data)

        self.assertNotEqual(move, 'left')

    def test_board_size_is_used_when_rows_and_cols_are_missing(self):
        data = turn('|A  |\n|  B|')
        data.pop('rows')
        data.pop('cols')
        data['board_size'] = '2x3'

        state = run.parse_state(data)

        self.assertEqual((state.rows, state.cols), (2, 3))

    def test_open_food_at_three_cells_advances(self):
        self.assertEqual(self.choose('|A  * |\n|     |\n|    B|'), 'right')

    def test_long_food_uses_bfs_and_advances_consistently(self):
        first = '|A      * |\n|         |\n|        B|'
        second = '| A     * |\n|         |\n|        B|'

        self.assertEqual(self.choose(first), 'down')
        self.assertEqual(self.choose(second), 'down')

    def test_food_beats_slightly_larger_empty_space(self):
        board = (
            '|     |\n'
            '| A*  |\n'
            '|     |\n'
            '|     |\n'
            '|    B|'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_food_with_two_exits_and_large_region_is_eaten(self):
        board = (
            '|       |\n'
            '|   B   |\n'
            '|       |\n'
            '|  A*   |\n'
            '|       |\n'
            '|       |'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_food_inside_dead_bag_is_rejected(self):
        board = (
            '|aaaaa|\n'
            '|aA*aa|\n'
            '|a aaa|\n'
            '|B    |'
        )

        self.assertNotEqual(self.choose(board), 'right')

    def test_food_path_accounts_for_own_tail_moving(self):
        board = (
            '|Aaa* |\n'
            '|    B|'
        )
        state = run.parse_state(turn(board))

        self.assertIn('down', run.legal_moves(state, 'A'))
        self.assertNotEqual(self.choose(board), 'right')

    def test_target_persists_across_turns(self):
        first = '|A   * |\n|      |\n|     B|'
        second = '| A  * |\n|      |\n|     B|'

        self.assertEqual(self.choose(first), 'down')
        self.assertEqual(self.choose(second), 'right')

    def test_adjacent_food_overrides_old_far_target(self):
        first = '|A     * |\n|        |\n|       B|'
        second = '| A*   * |\n|        |\n|       B|'

        self.assertEqual(self.choose(first), 'down')
        self.assertEqual(self.choose(second), 'right')

    def test_repeated_cycle_breaks_toward_reachable_food(self):
        board = (
            '|     * |\n'
            '|       |\n'
            '|  A    |\n'
            '|       |\n'
            '|     B |'
        )

        for _ in range(4):
            self.choose(board)

        self.assertIn(self.choose(board), {'up', 'right'})

    def test_lost_race_switches_to_other_food(self):
        board = (
            '|A    *|\n'
            '|     B|\n'
            '|*     |'
        )

        self.assertEqual(self.choose(board), 'down')

    def test_late_game_abandons_food_rival_wins_first(self):
        board = (
            '|            *  |\n'
            '| aaA B         |\n'
            '| a   b         |\n'
            '| a   bbb       |\n'
            '| a     b       |\n'
            '| a     b       |\n'
            '| a     b   bb  |\n'
            '| a     b  bb   |\n'
            '| a     bbbb   *|\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '| aa     *      |'
        )

        self.assertEqual(self.choose(board), 'down')

    def test_late_game_avoids_closing_top_corridor(self):
        board = (
            '|   aA       *  |\n'
            '| aaa bbB       |\n'
            '| a   b         |\n'
            '| a   bbb       |\n'
            '| a     b       |\n'
            '| a     b       |\n'
            '| a     b       |\n'
            '| a     b  bb   |\n'
            '| a     bbbb   *|\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '| a             |\n'
            '|        *      |'
        )

        self.assertEqual(self.choose(board), 'down')

    def test_does_not_step_toward_food_enemy_takes_next(self):
        board = (
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|bbb            |\n'
            '|b              |\n'
            '|b              |\n'
            '|B              |\n'
            '|     a         |\n'
            '|*Aaaaa         |\n'
            '|          *    |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|      *        |'
        )

        self.assertEqual(self.choose(board, side='B'), 'right')

    def test_avoids_chasing_enemy_owned_food_cluster(self):
        board = (
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|    bbbbbB     |\n'
            '|  aaaaaaaA*    |\n'
            '|          *    |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|      *        |'
        )

        self.assertEqual(self.choose(board, side='B'), 'up')

    def test_repositions_when_rival_has_food_tempo(self):
        board = (
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|   *   aaaaab  |\n'
            '|       A    b  |\n'
            '|       * Bbbb  |\n'
            '|               |\n'
            '|       *       |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |'
        )

        self.assertEqual(self.choose(board, side='B'), 'down')

    def test_prefers_wider_territory_over_one_exit_food_route(self):
        board = (
            '|               |\n'
            '|         *     |\n'
            '|               |\n'
            '|               |\n'
            '|       b       |\n'
            '|  *    b       |\n'
            '|       bbb     |\n'
            '|         b     |\n'
            '|        Bb     |\n'
            '| *             |\n'
            '|        a      |\n'
            '|        a      |\n'
            '|        aAa    |\n'
            '|        aaa    |\n'
            '|               |'
        )

        self.assertEqual(self.choose(board, side='B'), 'left')

    def test_far_lost_food_does_not_push_away_from_future_food(self):
        board = (
            '|    aA *       |\n'
            '|  bBa          |\n'
            '|  b a          |\n'
            '|  b a         *|\n'
            '|  b a       *  |\n'
            '|  b a          |\n'
            '|  b a          |\n'
            '|  b            |\n'
            '|  b            |\n'
            '|  b            |\n'
            '|  b            |\n'
            '|  b            |\n'
            '|               |\n'
            '|               |\n'
            '|               |'
        )

        self.assertEqual(self.choose(board, side='B'), 'down')

    def test_closer_safe_food_beats_far_exclusive_food(self):
        board = (
            '|A *       *|\n'
            '|           |\n'
            '|          B|'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_food_chain_prefers_first_food_that_leaves_next_food_close(self):
        board = (
            '|A  * *|\n'
            '|      |\n'
            '|      |\n'
            '|*    B|'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_hunger_switches_from_old_target_to_closer_safe_food(self):
        board = (
            '|A    *|\n'
            '|      |\n'
            '|      |\n'
            '|*    B|'
        )
        state = run.parse_state(turn(board))

        plan = choose_food_plan(state, 'A', current_target=(0, 5), hunger=18)

        self.assertEqual(plan.food, (3, 0))
        self.assertEqual(plan.first_move, 'down')

    def test_equal_food_race_is_won_when_we_move_first(self):
        board = (
            '|A * B|\n'
            '|     |'
        )
        state = run.parse_state(turn(board, side='A'))

        self.assertEqual(tempo_race_margin(state, 'A', 2, 2), 1)

    def test_equal_food_race_is_lost_when_rival_moves_first(self):
        board = (
            '|A * B|\n'
            '|     |'
        )
        state = run.parse_state(turn(board, side='A'))
        enemy_turn = replace(state, side='B', enemy='A')

        self.assertEqual(tempo_race_margin(enemy_turn, 'A', 2, 2), -1)

    def test_temporal_path_uses_body_cell_after_it_frees(self):
        state = GameState(
            rows=3,
            cols=4,
            board=('    ', '    ', '    '),
            side='A',
            enemy='B',
            snakes={
                'A': ((2, 2), (2, 1), (1, 1), (1, 0)),
                'B': ((0, 3), (0, 2), (0, 1)),
            },
            food=frozenset({(0, 0)}),
            scores={'A': 0, 'B': 0},
            remaining_moves=40,
            reliable_tails=frozenset({'A'}),
        )

        self.assertEqual(
            temporal_shortest_path(state, (2, 2), (0, 0), 'A'),
            ((2, 2), (1, 2), (1, 1), (1, 0), (0, 0)),
        )

    def test_endgame_can_commit_to_risky_food(self):
        plan = FoodPlan(
            food=(0, 1),
            path=((0, 0), (0, 1)),
            our_distance=1,
            enemy_distance=4,
            race_margin=3,
            space_after=1,
            exits_after=0,
            tail_reachable_after=False,
            next_food_distance=None,
            safety='SUICIDAL',
            value=100,
            endgame_acceptable=True,
        )

        self.assertTrue(should_commit_to_food(plan, hunger=0, repeat_count=0))

    def test_lost_contested_food_is_not_chased(self):
        board = (
            '| A *     * |\n'
            '|   B       |\n'
            '|           |'
        )

        self.assertEqual(self.choose(board), 'down')

    def test_winning_race_goes_to_food(self):
        board = (
            '|A *   |\n'
            '|      |\n'
            '|     B|'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_food_beats_useless_pressure(self):
        board = (
            '|      |\n'
            '|  B   |\n'
            '|      |\n'
            '| A*   |\n'
            '|      |'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_forced_kill_can_override_food(self):
        board = (
            '|B  a|\n'
            '|aA a|\n'
            '| *aa|'
        )

        self.assertEqual(self.choose(board), 'up')

    def test_forced_kill_is_not_taken_when_terminal_score_still_loses(self):
        board = (
            '|B  a|\n'
            '|aA a|\n'
            '| *aa|'
        )
        data = turn(board, score_1=0, score_2=5000, game_id='losing_kill')

        run.BOT.choose_move(data)

        self.assertNotEqual(run.BOT.decision_debug('losing_kill')['reason'], 'forced_kill')

    def test_definitive_trap_food_is_not_eaten(self):
        board = (
            '| aa   |\n'
            '| A*aaB|\n'
            '| aa   |'
        )

        self.assertNotEqual(self.choose(board), 'right')

    def test_unchanged_committed_body_keeps_exact_tail_order(self):
        previous = ((1, 7), (1, 8), (2, 8), (2, 7), (2, 6), (3, 6), (3, 7))
        board = (
            '            ',
            '       Bb   ',
            '      bbb   ',
            '      bb    ',
        )

        self.assertEqual(track_snake(previous, board, 'B'), previous)

    def test_deep_search_treats_rival_zero_reply_as_forced_loss(self):
        state = GameState(
            rows=12,
            cols=12,
            board=tuple(' ' * 12 for _ in range(12)),
            side='B',
            enemy='A',
            snakes={
                'A': ((0, 5), (1, 5), (1, 4), (2, 4)),
                'B': ((1, 7), (1, 8), (2, 8), (2, 7), (2, 6), (3, 6), (3, 7)),
            },
            food=frozenset(),
            scores={'A': 530, 'B': 2579},
            remaining_moves=61,
            reliable_tails=frozenset({'A', 'B'}),
        )

        analyses = rank_deep_moves(state, 'B', legal_moves(state, 'B'))
        by_direction = {item['direction']: item for item in analyses}

        self.assertTrue(by_direction['left']['forced_loss'])
        self.assertEqual(analyses[0]['direction'], 'up')

    def test_ambiguous_body_shape_does_not_treat_body_as_free_tail(self):
        board = (
            '|         *     |\n'
            '|   *           |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|     *         |\n'
            '|          bb   |\n'
            '|          bb   |\n'
            '|          bB   |\n'
            '| A             |\n'
            '| a             |\n'
            '| aa            |\n'
            '| aa            |\n'
            '|               |'
        )
        state = run.parse_state(turn(board, side='B'))

        self.assertNotIn('up', run.legal_moves(state, 'B'))

    def test_rejects_suicidal_chokepoint_even_under_rival_pressure(self):
        board = (
            '|        bbbbbb |\n'
            '|   aABbbbaaaab |\n'
            '|   a     a   b |\n'
            '|   aaaaaaa   b |\n'
            '| *           b |\n'
            '|             b |\n'
            '|    *          |\n'
            '|               |\n'
            '|    *          |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |'
        )

        self.assertEqual(self.choose(board, side='B'), 'up')

    def test_prefers_move_with_fewer_rival_chokepoint_replies(self):
        board = (
            '|               |\n'
            '|               |\n'
            '|   *   bb      |\n'
            '|       bb      |\n'
            '|       bb      |\n'
            '|    * bbb      |\n'
            '|      bbbbbbb  |\n'
            '|            b  |\n'
            '|          aab  |\n'
            '|        aaaab  |\n'
            '|        a aab  |\n'
            '|        a a b  |\n'
            '|     aaaa A b  |\n'
            '|*    aa   A b  |\n'
            '|            B  |'
        )

        self.assertEqual(self.choose(board, side='B'), 'right')

    def test_safe_edge_food_is_eaten(self):
        board = (
            '|A*   |\n'
            '|     |\n'
            '|B    |'
        )

        self.assertEqual(self.choose(board), 'right')

    def test_non_immediate_low_space_food_plan_is_not_committed(self):
        plan = FoodPlan(
            food=(0, 4),
            path=((2, 4), (1, 4), (0, 4)),
            our_distance=2,
            enemy_distance=6,
            race_margin=4,
            space_after=34,
            exits_after=1,
            tail_reachable_after=False,
            next_food_distance=None,
            safety='ACCEPTABLE_RISK',
            value=20_000,
        )

        self.assertFalse(should_commit_to_food(plan, hunger=30, repeat_count=3))

    def test_adjacent_low_space_food_can_still_be_taken(self):
        plan = FoodPlan(
            food=(0, 1),
            path=((0, 0), (0, 1)),
            our_distance=1,
            enemy_distance=4,
            race_margin=3,
            space_after=30,
            exits_after=1,
            tail_reachable_after=False,
            next_food_distance=None,
            safety='ACCEPTABLE_RISK',
            value=100,
        )

        self.assertTrue(should_commit_to_food(plan, hunger=0, repeat_count=0))

    def test_adjacent_food_is_taken_despite_soft_rival_pressure(self):
        board = (
            '|               |\n'
            '|               |\n'
            '|  a            |\n'
            '|  a            |\n'
            '|  A            |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|       *       |\n'
            '|       *       |\n'
            '|               |\n'
            '|               |\n'
            '|            b  |\n'
            '|            bB*|'
        )

        self.assertEqual(self.choose(board, side='B'), 'right')

    def test_prefers_center_over_non_immediate_edge_food_corridor(self):
        board = (
            '|         *     |\n'
            '|               |\n'
            '|               |\n'
            '|               |\n'
            '|       aaa     |\n'
            '|       a a     |\n'
            '|       a a     |\n'
            '|       a       |\n'
            '|       a       |\n'
            '|       A       |\n'
            '|               |\n'
            '|               |\n'
            '|  b          * |\n'
            '| bb            |\n'
            '| bbbbB  *      |'
        )

        self.assertEqual(self.choose(board, side='B'), 'up')

    def test_final_safety_guard_replaces_illegal_direction(self):
        data = turn('|A   |\n|    |\n|   B|')

        self.assertIn(run.final_safe_direction(data, 'left'), {'right', 'down'})


class TestIterativeSearchEngine(unittest.TestCase):
    def assert_transition_matches(self, state, direction):
        engine, compact = IterativeSearchEngine.from_game_state(state)
        expected = run.apply_move(state, direction, state.side)
        actual = engine.apply_move(compact, direction)

        self.assertEqual(
            set(engine.legal_moves(compact)),
            set(run.legal_moves(state, state.side)),
        )
        self.assertEqual(actual.side, 0 if expected.side == 'A' else 1)
        self.assertEqual(actual.scores, (expected.scores['A'], expected.scores['B']))
        self.assertEqual(
            actual.multipliers,
            (expected.multipliers['A'], expected.multipliers['B']),
        )
        self.assertEqual(actual.remaining_moves, expected.remaining_moves)
        self.assertEqual(
            tuple(engine.cell(cell) for cell in actual.bodies[0]),
            expected.body('A'),
        )
        self.assertEqual(
            tuple(engine.cell(cell) for cell in actual.bodies[1]),
            expected.body('B'),
        )
        self.assertEqual(
            {engine.cell(cell) for cell in actual.walls},
            set(expected.walls),
        )
        self.assertEqual(
            {engine.cell(cell) for cell in actual.food},
            set(expected.food),
        )
        self.assertEqual(
            {engine.cell(cell) for cell in actual.pickups},
            set(expected.pickups),
        )
        self.assertEqual(actual.next_food_digit or None, expected.next_food_digit)

    def test_compact_transition_matches_numbered_food_rules(self):
        state = run.parse_state(turn(
            '|A1 12|\n'
            '|1X23B|\n'
            '| 345 |',
            multiplier_1=2,
            multiplier_2=1,
        ))

        self.assert_transition_matches(state, 'right')

    def test_compact_transition_matches_multiplier_rules(self):
        state = run.parse_state(turn(
            '|AX 12|\n'
            '|345 B|',
            multiplier_1=2,
            multiplier_2=1,
        ))

        self.assert_transition_matches(state, 'right')

    def test_compact_transition_matches_wall_hit_and_shrink(self):
        state = run.parse_state(turn(
            '|A     |\n'
            '|  B###|',
            side='B',
            score_1=100,
            score_2=1000,
        ))

        self.assert_transition_matches(state, 'right')

    def test_iterative_search_completes_useful_depth_within_budget(self):
        state = run.parse_state(turn(
            '|       1|\n'
            '|        |\n'
            '|  A     |\n'
            '|        |\n'
            '|    X   |\n'
            '|        |\n'
            '|     B  |\n'
            '|2       |',
            remaining=180,
        ))
        engine, compact = IterativeSearchEngine.from_game_state(
            state,
            time_budget_ms=1500,
            max_depth=6,
        )
        legal = run.legal_moves(state, state.side)

        result = engine.search(compact, state.side, legal, target=next(iter(state.food)))

        self.assertGreaterEqual(result.completed_depth, 4)
        self.assertEqual(set(result.scores), set(legal))
        self.assertLess(result.elapsed_ms, 2500)

    def test_search_ends_by_score_before_next_turn_death(self):
        state = GameState(
            rows=3,
            cols=3,
            board=('Bb ', 'aa*', ' aA'),
            side='A',
            enemy='B',
            snakes={
                'A': ((2, 2), (2, 1), (1, 1), (1, 0)),
                'B': ((0, 0), (0, 1)),
            },
            food=frozenset({(1, 2)}),
            scores={'A': 0, 'B': 800},
            remaining_moves=1,
            reliable_tails=frozenset({'A', 'B'}),
        )
        engine, compact = IterativeSearchEngine.from_game_state(state)

        result = engine.search(compact, 'A', ['up'], target=(1, 2))

        self.assertLess(result.scores['up'], 0)


if __name__ == '__main__':
    unittest.main()
