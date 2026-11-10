"""Accumulative v8/v9 rule contracts, independent of strategy weights."""

from dataclasses import replace
import unittest

from snake_bot.snake_state import GameState, apply_move, parse_state
from snake_bot import snake_state


def state(**changes):
    source = GameState(
        rows=20, cols=20, board=(), side='A', enemy='B',
        snakes={'A': ((5, 5), (5, 4), (5, 3)),
                'B': ((15, 15), (15, 14), (15, 13))},
        food=frozenset(), scores={'A': 100, 'B': 100},
        remaining_moves=250, reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False, crashed_b=False)
    return replace(source, **changes)


class BombRulesTests(unittest.TestCase):
    def test_unicode_bombs_and_shot_food_parse_as_single_cells_on_all_board_sizes(self):
        for rows, cols in ((20, 20), (20, 30), (30, 20), (30, 30)):
            with self.subTest(rows=rows, cols=cols):
                board = [[' '] * cols for _ in range(rows)]
                for cell, symbol in (((0, 0), 'A'), ((rows-1, cols-1), 'B'),
                                     ((10, 10), '💣'), ((12, 12), '🧨'),
                                     ((0, 3), '*'), ((0, 5), '1')):
                    board[cell[0]][cell[1]] = symbol
                parsed = parse_state({'board': '\n'.join(''.join(row) for row in board),
                                      'board_size': f'{rows}x{cols}'})
                self.assertEqual((parsed.rows, parsed.cols), (rows, cols))
                self.assertEqual(parsed.bombs_large, frozenset({(10, 10)}))
                self.assertEqual(parsed.bombs_small, frozenset({(12, 12)}))
                self.assertEqual(parsed.shot_food, frozenset({(0, 3)}))
                self.assertEqual(parsed.food, frozenset({(0, 5)}))
                self.assertEqual(parsed.head('B'), (rows-1, cols-1))

    def test_chebyshev_diagonals_boundaries_and_positive_zero_negative_scores(self):
        for kind, bomb, triggered in (
                ('large', (7, 8), True), ('large', (8, 9), False),
                ('small', (6, 7), True), ('small', (7, 8), False)):
            for score in (100, 0, -100):
                with self.subTest(kind=kind, bomb=bomb, score=score):
                    source = state(scores={'A': score, 'B': 100},
                                   **{f'bombs_{kind}': frozenset({bomb})})
                    after = apply_move(source, 'right')
                    self.assertEqual(after.body('A'), ((5, 6), (5, 5), (5, 4)))
                    expected = (0 if score > 0 else score - 500) if triggered else score + 1
                    self.assertEqual(after.scores['A'], expected)
                    self.assertEqual((after.crashed_a, after.crashed_b), (False, False))
                    self.assertEqual(after.last_action['bomb_triggered'], triggered)
                    self.assertEqual(getattr(after, f'bombs_{kind}'),
                                     frozenset() if triggered else frozenset({bomb}))
                    self.assertEqual(after.pending_bomb_respawns, (kind,) if triggered else ())

    def test_simultaneous_detonations_only_penalize_once(self):
        source = state(scores={'A': -100, 'B': 100},
                       bombs_large=frozenset({(7, 8)}), bombs_small=frozenset({(6, 7)}))
        after = apply_move(source, 'right')
        self.assertEqual(after.scores['A'], -600)
        self.assertEqual(after.bombs_large | after.bombs_small, frozenset())
        self.assertEqual(sorted(after.pending_bomb_respawns), ['large', 'small'])

    def test_crash_tail_food_is_filtered_by_blast_zone(self):
        body = tuple((0, col) for col in range(8))
        source = state(snakes={'A': body, 'B': ((15, 15),)},
                       bombs_small=frozenset({(1, 4)}))
        after = apply_move(source, 'up')
        self.assertEqual(after.body('A'), body[:3])
        self.assertEqual(after.tail_food_b, frozenset({(0, 6), (0, 7)}))
        self.assertTrue(after.crashed_a)
        self.assertFalse(after.last_action['bomb_triggered'])

    def test_bombs_expire_at_remaining_move_boundary_in_parse_and_transition(self):
        for remaining in (100, 99):
            parsed = parse_state({'board': 'A 💣 🧨 B', 'remaining_moves': remaining})
            self.assertFalse(parsed.bombs_large | parsed.bombs_small)
        source = state(remaining_moves=101, bombs_large=frozenset({(10, 10)}),
                       pending_bomb_respawns=('small',))
        after = apply_move(source, 'right')
        self.assertEqual(after.remaining_moves, 100)
        self.assertFalse(after.bombs_large | after.bombs_small)
        self.assertEqual(after.pending_bomb_respawns, ())

    def test_last_bomb_turn_detonates_before_all_bombs_disappear(self):
        source = state(remaining_moves=101, scores={'A': 100, 'B': 100},
                       bombs_large=frozenset({(7, 8), (12, 12)}))
        after = apply_move(source, 'right')
        self.assertEqual(after.scores['A'], 0)
        self.assertTrue(after.last_action['bomb_triggered'])
        self.assertEqual(after.last_action['bomb_positions'], [(7, 8)])
        self.assertFalse(after.bombs_large | after.bombs_small)
        self.assertEqual(after.pending_bomb_respawns, ())


class ShootRulesTests(unittest.TestCase):
    def shoot(self, source, direction='right', side=None):
        return snake_state.apply_action(source, 'shoot', direction, side)

    def test_snapshot_without_digits_keeps_shot_food_separate_and_consumes_without_respawn(self):
        source = parse_state({'board': 'A*  B', 'remaining_moves': 200})
        self.assertEqual(source.food, frozenset())
        self.assertEqual(source.shot_food, frozenset({(0, 1)}))
        after = apply_move(source, 'right')
        self.assertEqual(after.scores['A'], 100)
        self.assertEqual(len(after.body('A')), 2)
        self.assertFalse(after.food | after.shot_food)
        self.assertIsNone(after.next_food_digit)

    def test_action_space_only_offers_shoot_when_actor_can_pay(self):
        for side, score in (('A', 4999), ('A', 5000), ('B', 6000)):
            with self.subTest(side=side, score=score):
                source = state(scores={side: score, 'B' if side == 'A' else 'A': 100})
                actions = snake_state.available_actions(source, side)
                shots = [direction for kind, direction in actions if kind == 'shoot']
                self.assertEqual(set(shots), set(snake_state.DIRS) if score >= 5000 else set())
                self.assertTrue(any(kind == 'move' for kind, _ in actions))
                for direction in shots:
                    after = self.shoot(source, direction, side)
                    self.assertEqual(after.scores[side], score - 5000)
                    self.assertEqual(after.body(side), source.body(side))

    def test_cost_and_stationary_turn_for_hit_miss_and_invalid_score(self):
        for score, hit in ((5000, True), (6000, False), (4999, True)):
            with self.subTest(score=score, hit=hit):
                body = (
                    ((5, 9), (5, 10), (5, 11)) if hit else ((10, 9), (10, 10))
                )
                source = state(
                    snakes={'A': ((5, 5), (5, 4)), 'B': body},
                    scores={'A': score, 'B': 123},
                    crashed_b=False,
                )
                after = self.shoot(source)
                self.assertEqual(after.body('A'), source.body('A'))
                self.assertEqual(
                    after.scores,
                    {'A': score - (5000 if score >= 5000 else 500), 'B': 123},
                )
                self.assertEqual(
                    after.remaining_moves, source.remaining_moves - 1
                )
                self.assertEqual((after.side, after.enemy), ('B', 'A'))
                self.assertEqual(
                    (after.crashed_a, after.crashed_b), (False, False)
                )
                self.assertEqual(
                    after.last_action['shoot_cost'],
                    5000 if score >= 5000 else 0,
                )
                self.assertEqual(
                    after.last_action['hit'], hit and score >= 5000
                )
                if score < 5000:
                    self.assertEqual(after.body('B'), body)
                    self.assertFalse(after.shot_food)
                    self.assertFalse(after.last_action['valid'])

    def test_ray_ignores_own_body_food_multiplier_and_bombs_but_stops_at_first_enemy(self):
        # A looping own body crosses the ray; the first enemy cell is its body.
        rival = ((4, 13), (5, 13), (5, 12), (5, 11), (6, 11), (7, 11))
        own = ((5, 5), (6, 5), (6, 6), (5, 6))
        source = state(snakes={'A': own, 'B': rival}, scores={'A': 6000, 'B': 333},
                       food=frozenset({(5, 7)}), food_values={(5, 7): 1},
                       pickups=frozenset({(5, 8)}), bombs_small=frozenset({(5, 10)}))
        after = self.shoot(source)
        self.assertEqual(after.body('A'), own)
        self.assertEqual(after.body('B'), rival[:3])
        self.assertEqual(after.shot_food, frozenset({(7, 11)}))
        self.assertEqual(after.last_action['hit_coordinate'], (5, 11))
        self.assertEqual(after.last_action['cut_length'], 3)
        self.assertEqual(after.last_action['shot_food_generated'], [(7, 11)])
        self.assertEqual(after.scores, {'A': 1000, 'B': 333})
        self.assertEqual(after.bombs_small, source.bombs_small)
        self.assertFalse(after.pending_bomb_respawns)
        self.assertEqual(after.food_values, source.food_values)
        self.assertEqual(after.pickups, source.pickups)

    def test_headshot_keeps_head_and_turns_only_detached_body_into_food(self):
        for direction, rival in (('right', ((5, 9), (5, 10), (5, 11))),
                                 ('up', ((1, 5), (1, 6), (1, 7)))):
            with self.subTest(direction=direction):
                source = state(snakes={'A': ((5, 5),), 'B': rival},
                               scores={'A': 5000, 'B': 100}, crashed_b=False)
                after = self.shoot(source, direction)
                self.assertEqual(after.body('B'), rival[:1])
                self.assertEqual(after.shot_food, frozenset(rival[1:]))
                self.assertNotIn(rival[0], after.shot_food)
                self.assertEqual(after.scores['B'], 100)
                self.assertFalse(after.crashed_b)
                self.assertEqual(after.last_action['cut_length'], len(rival) - 1)

    def test_wall_and_board_edge_terminate_ray_without_hit(self):
        for direction, wall in (('right', frozenset({(5, 7)})),
                                ('up', frozenset()), ('left', frozenset()),
                                ('down', frozenset())):
            with self.subTest(direction=direction):
                source = state(snakes={'A': ((5, 5),), 'B': ((5, 10), (5, 11))},
                               scores={'A': 5000, 'B': 100}, walls=wall)
                after = self.shoot(source, direction)
                self.assertEqual(after.body('B'), source.body('B'))
                self.assertFalse(after.shot_food)
                self.assertFalse(after.last_action['hit'])
                self.assertEqual(after.scores['A'], 0)

    def test_shot_food_is_edible_by_both_players_without_advancing_numeric_sequence(
        self,
    ):
        for side in ('A', 'B'):
            with self.subTest(side=side):
                source = state(
                    side=side,
                    enemy='B' if side == 'A' else 'A',
                    snakes={'A': ((5, 5), (5, 4)), 'B': ((10, 5), (10, 4))},
                    multipliers={'A': 2, 'B': 3},
                    shot_food=frozenset({(5, 6), (10, 6)}),
                    food=frozenset({(0, 0)}),
                    food_values={(0, 0): 1},
                    next_food_digit=1,
                )
                after = apply_move(source, 'right', side)
                self.assertEqual(
                    len(after.body(side)), len(source.body(side)) + 1
                )
                self.assertEqual(
                    after.scores[side], 100 + 100 * source.multipliers[side]
                )
                self.assertNotIn(after.head(side), after.shot_food)
                self.assertEqual(after.next_food_digit, 1)
                self.assertEqual(after.food_values, source.food_values)
                self.assertEqual(after.food, source.food)

    def test_shooting_as_b_advances_round_wall_and_bomb_expiry_without_moving(self):
        source = state(side='B', enemy='A', remaining_moves=101,
                       scores={'A': 100, 'B': 5000},
                       walls=frozenset({(0, 0), (0, 1), (0, 2)}),
                       bombs_large=frozenset({(10, 10)}))
        after = self.shoot(source, 'down')
        self.assertEqual(after.body('B'), source.body('B'))
        self.assertEqual(after.scores['B'], 0)
        self.assertEqual(after.walls, frozenset({(0, 1)}))
        self.assertEqual(after.remaining_moves, 100)
        self.assertFalse(after.bombs_large)
        self.assertFalse(after.crashed_b)

    def test_a_shot_suffix_is_reliably_tracked_from_the_next_snapshot(self):
        previous = {'B': ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5))}
        for length in (1, 2, 4):
            with self.subTest(length=length):
                board = [[' '] * 8 for _ in range(3)]
                board[0][0] = 'A'
                for index, (row, col) in enumerate(previous['B'][:length]):
                    board[row][col] = 'B' if index == 0 else 'b'
                parsed = parse_state({'board': '\n'.join(''.join(row) for row in board)}, previous)
                self.assertEqual(parsed.body('B'), previous['B'][:length])
                self.assertIn('B', parsed.reliable_tails)

    def test_30x30_smoke_detonate_shoot_and_consume_generated_food(self):
        board = [[' '] * 30 for _ in range(30)]
        for cell, symbol in (((5, 5), 'A'), ((5, 4), 'a'), ((5, 3), 'a'),
                             ((5, 2), 'a'), ((5, 12), 'B'), ((5, 13), 'b'),
                             ((7, 8), '💣')):
            board[cell[0]][cell[1]] = symbol
        source = parse_state({'board': '\n'.join(''.join(row) for row in board),
                              'score_1': 6000, 'score_2': 6000},
                             crash_history={'A': False, 'B': False})
        bomb = apply_move(source, 'right')
        self.assertEqual(bomb.head('A'), (5, 6))
        self.assertEqual(bomb.scores['A'], 0)
        shot = self.shoot(bomb, 'left')
        self.assertEqual(shot.body('A'), ((5, 6),))
        self.assertEqual(shot.scores['B'], 1000)
        self.assertIn((5, 5), shot.shot_food)
        eaten = apply_move(shot, 'left')
        self.assertEqual(eaten.body('A'), ((5, 5), (5, 6)))
        self.assertEqual(eaten.scores['A'], 100)
        self.assertNotIn((5, 5), eaten.shot_food)


if __name__ == '__main__':
    unittest.main()
