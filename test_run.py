import json
import os
import tempfile
import unittest

import run


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


def turn(board, side='A', score_1=0, score_2=0, remaining=250, game_id='g_1'):
    rows = len(board.splitlines())
    cols = max(len(line.strip('|')) for line in board.splitlines())
    return {
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

    def test_open_food_at_three_cells_advances(self):
        self.assertEqual(self.choose('|A  * |\n|     |\n|    B|'), 'right')

    def test_long_food_uses_bfs_and_advances_consistently(self):
        first = '|A      * |\n|         |\n|        B|'
        second = '| A     * |\n|         |\n|        B|'

        self.assertEqual(self.choose(first), 'right')
        self.assertEqual(self.choose(second), 'right')

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

        self.assertEqual(self.choose(first), 'right')
        self.assertEqual(self.choose(second), 'right')

    def test_adjacent_food_overrides_old_far_target(self):
        first = '|A     * |\n|        |\n|       B|'
        second = '| A*   * |\n|        |\n|       B|'

        self.assertEqual(self.choose(first), 'right')
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

    def test_definitive_trap_food_is_not_eaten(self):
        board = (
            '|aaaaaa|\n'
            '|aA*aaB|\n'
            '|aaaaaa|'
        )

        self.assertNotEqual(self.choose(board), 'right')

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


if __name__ == '__main__':
    unittest.main()
