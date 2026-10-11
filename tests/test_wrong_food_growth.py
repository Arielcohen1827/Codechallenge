"""Sanitized regression for observed growth on a wrong numbered food."""

from copy import deepcopy
import unittest

from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import GameState, apply_move


OBSERVED_BODY = (
    (8, 12), (7, 12), (7, 13), (8, 13), (9, 13), (9, 14),
    (9, 15), (10, 15), (10, 14), (10, 13), (11, 13), (11, 12),
)
TARGET = (8, 11)
SECOND_COPY = (3, 3)
NEXT_FOOD = (2, 2)


def observed_state(side, score=30199, remaining=70):
    enemy = 'B' if side == 'A' else 'A'
    return GameState(
        rows=16, cols=16, board=(), side=side, enemy=enemy,
        snakes={side: OBSERVED_BODY, enemy: ((0, 1), (0, 0))},
        food=frozenset({NEXT_FOOD}),
        food_values={TARGET: 4, SECOND_COPY: 4, NEXT_FOOD: 1},
        next_food_digit=1, food_copy_targets={4: 2, 1: 1},
        scores={side: score, enemy: 100},
        multipliers={side: 8, enemy: 2}, remaining_moves=remaining,
        pickups=frozenset({(4, 4)}), reliable_tails=frozenset({'A', 'B'}),
        tail_food_a=frozenset(), tail_food_b=frozenset(),
        crashed_a=False, crashed_b=False,
    )


class TestWrongFoodGrowth(unittest.TestCase):
    def assert_transition(self, source, representation):
        source_snapshot = deepcopy(source)
        engine, compact = IterativeSearchEngine.from_game_state(source)
        compact_snapshot = deepcopy(compact)
        side = source.side
        enemy = source.enemy
        if representation == 'game':
            after = apply_move(source, 'left')
            body = after.body(side)
            self.assertEqual(after.body(enemy), source.body(enemy))
            self.assertEqual(after.scores, {side: source.scores[side] - 500, enemy: 100})
            self.assertEqual(after.food_copy_targets, source.food_copy_targets)
            self.assertEqual(after.food_values, {SECOND_COPY: 4, NEXT_FOOD: 1})
            self.assertEqual(after.food, source.food)
            self.assertEqual(after.next_food_digit, 1)
            self.assertEqual(after.pickups, source.pickups)
            self.assertEqual(after.multipliers, source.multipliers)
            self.assertEqual(after.tail_food_a, source.tail_food_a)
            self.assertEqual(after.tail_food_b, source.tail_food_b)
            self.assertEqual(after.reliable_tails, source.reliable_tails)
            self.assertEqual((after.side, after.enemy), (enemy, side))
        else:
            after = engine.apply_move(compact, 'left')
            index = 0 if side == 'A' else 1
            body = tuple(engine.cell(cell) for cell in after.bodies[index])
            self.assertEqual(after.bodies[1 - index], compact.bodies[1 - index])
            scores = list(compact.scores)
            scores[index] -= 500
            self.assertEqual(after.scores, tuple(scores))
            self.assertEqual({engine.cell(cell): digit for cell, digit in after.food_values},
                             {SECOND_COPY: 4, NEXT_FOOD: 1})
            self.assertEqual(after.food, compact.food)
            self.assertEqual(after.next_food_digit, 1)
            self.assertEqual(after.pickups, compact.pickups)
            self.assertEqual(after.multipliers, compact.multipliers)
            self.assertEqual(after.tail_food_a, compact.tail_food_a)
            self.assertEqual(after.tail_food_b, compact.tail_food_b)
            self.assertEqual(after.reliable_tails, compact.reliable_tails)
            self.assertEqual(after.side, 1 - index)
        self.assertEqual(after.walls, source.walls if representation == 'game' else compact.walls)
        self.assertEqual((after.crashed_a, after.crashed_b), (False, False))
        self.assertEqual(after.remaining_moves, source.remaining_moves - 1)
        self.assertEqual(source, source_snapshot)
        self.assertEqual(compact, compact_snapshot)
        self.assertEqual(body, (TARGET,) + OBSERVED_BODY)
        self.assertEqual(len(body), 13)
        self.assertEqual(body[-1], (11, 12))
        return after

    def test_observed_wrong_four_grows_and_preserves_bookkeeping(self):
        for side in ('A', 'B'):
            for representation in ('game', 'compact'):
                with self.subTest(side=side, representation=representation):
                    self.assert_transition(observed_state(side), representation)

    def test_wrong_growth_preserves_strict_elimination_and_last_turn(self):
        for side in ('A', 'B'):
            for score, remaining, terminal, winner in (
                (-2000, 2, False, None),
                (-2001, 2, True, 'B' if side == 'A' else 'A'),
                (30199, 1, True, side),
            ):
                for representation in ('game', 'compact'):
                    with self.subTest(side=side, score=score, remaining=remaining,
                                      representation=representation):
                        after = self.assert_transition(
                            observed_state(side, score, remaining), representation)
                        self.assertEqual(after.is_terminal(), terminal)
                        self.assertEqual(after.terminal_winner(), winner)
