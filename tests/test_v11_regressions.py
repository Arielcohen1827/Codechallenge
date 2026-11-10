"""Cumulative mechanics must retain coordinates and economics on every cube axis."""

from dataclasses import replace
import unittest

from snake_bot.food_planner import choose_food_plan
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import apply_action, apply_move, parse_state
from tests.test_v11_runtime import cube_payload
from tests.test_v11_search import cube


class CubeCumulativeTests(unittest.TestCase):
    def test_each_single_plane_objective_has_a_real_three_dimensional_route(self):
        for target in ((9, 7, 5), (9, 5, 7), (10, 7, 7)):
            with self.subTest(target=target):
                state = parse_state(cube_payload(cells={target: '1'}))
                plan = choose_food_plan(state, 'A')
                self.assertIsNotNone(plan)
                self.assertEqual(plan.food, target)
                self.assertEqual(plan.path[-1], target)
                self.assertEqual(plan.our_distance, 3 if target[0] == 9 else 4)

    def test_all_six_faces_preserve_crash_and_protection_rules(self):
        cases = (
            ((0, 5, 5), 'ascend'),
            ((19, 5, 5), 'descend'),
            ((10, 0, 5), 'up'),
            ((10, 19, 5), 'down'),
            ((10, 5, 0), 'left'),
            ((10, 5, 19), 'right'),
        )
        for head, direction in cases:
            with self.subTest(direction=direction):
                source = cube(snakes={'A': (head,), 'B': ()}, scores={'A': 700, 'B': 0})
                normal = apply_move(source, direction)
                self.assertEqual(normal.head('A'), head)
                self.assertEqual(normal.scores['A'], 0)
                protected = apply_move(replace(source, invincible_a=2), direction)
                self.assertEqual(protected.scores['A'], 700)
                self.assertEqual(protected.invincible_a, 1)

    def test_target_digit_removes_every_copy_across_levels(self):
        target, other_copy = (9, 5, 5), (2, 3, 4)
        following = (11, 5, 5)
        source = cube(
            food=frozenset({target, other_copy}),
            food_values={target: 1, other_copy: 1, following: 2},
            next_food_digit=1,
            multipliers={'A': 3, 'B': 1},
        )
        after = apply_move(source, 'ascend')
        self.assertEqual(after.scores['A'], source.scores['A'] + 300)
        self.assertEqual(after.food_values, {following: 2})
        self.assertEqual(after.next_food_digit, 2)
        self.assertEqual(len(after.body('A')), len(source.body('A')) + 1)

    def test_wrong_digit_preserves_sequence_and_multiplier_in_cube(self):
        target, correct = (9, 5, 5), (11, 5, 5)
        source = cube(
            food=frozenset({correct}),
            food_values={target: 4, correct: 1},
            next_food_digit=1,
            multipliers={'A': 3, 'B': 1},
        )
        after = apply_move(source, 'ascend')
        self.assertEqual(after.scores['A'], source.scores['A'] - 500)
        self.assertEqual(after.next_food_digit, 1)
        self.assertEqual(after.multipliers['A'], 3)
        self.assertEqual(len(after.body('A')), len(source.body('A')) + 1)
        protected = apply_move(replace(source, invincible_a=2), 'ascend')
        self.assertEqual(protected.scores['A'], source.scores['A'])

    def test_vertical_body_crash_produces_tail_food_with_original_levels(self):
        body = ((10, 5, 5), (9, 5, 5), (9, 5, 4), (8, 5, 4), (8, 5, 3))
        source = cube(snakes={'A': body, 'B': ()}, food=frozenset())
        after = apply_move(source, 'ascend')
        self.assertEqual(after.body('A'), body[:3])
        self.assertEqual(after.tail_food_b, frozenset(body[3:]))
        self.assertEqual(after.scores['A'], 0)

    def test_tail_food_beneficiary_and_originator_cleanup_work_vertically(self):
        target = (9, 5, 5)
        base = cube(food=frozenset(), multipliers={'A': 2, 'B': 1})
        beneficiary = apply_move(replace(base, tail_food_a=frozenset({target})), 'ascend')
        self.assertEqual(beneficiary.scores['A'], base.scores['A'] + 200)
        self.assertEqual(len(beneficiary.body('A')), len(base.body('A')) + 1)
        originator = apply_move(replace(base, tail_food_b=frozenset({target})), 'ascend')
        self.assertEqual(originator.scores['A'], base.scores['A'] + 1)
        self.assertEqual(originator.tail_food_b, frozenset())
        self.assertEqual(len(originator.body('A')), len(base.body('A')))

    def test_vertical_ram_cuts_an_invincible_rival_and_stays_put(self):
        rival = ((9, 5, 6), (9, 5, 5), (8, 5, 5))
        source = cube(
            snakes={'A': ((10, 5, 5),), 'B': rival},
            food=frozenset(),
            invincible_a=3,
            invincible_b=5,
        )
        after = apply_move(source, 'ascend')
        self.assertEqual(after.head('A'), source.head('A'))
        self.assertEqual(after.body('B'), rival[:1])
        self.assertEqual(after.shot_food, frozenset(rival[1:]))
        self.assertEqual(after.scores, source.scores)

    def test_star_growth_and_refresh_preserve_level(self):
        source = cube(
            food=frozenset(),
            stars=frozenset({(9, 5, 5)}),
            invincible_a=12,
            multipliers={'A': 2, 'B': 1},
        )
        after = apply_move(source, 'ascend')
        self.assertEqual(after.invincible_a, 50)
        self.assertEqual(after.scores['A'], source.scores['A'] + 200)
        self.assertEqual(after.head('A'), (9, 5, 5))
        self.assertEqual(len(after.body('A')), len(source.body('A')) + 1)

    def test_game_and_compact_actions_have_identical_cube_effects(self):
        source = cube()
        engine, compact = IterativeSearchEngine.from_game_state(source)
        for action in ('move', 'shoot'):
            for direction in ('up', 'down', 'left', 'right', 'ascend', 'descend'):
                with self.subTest(action=action, direction=direction):
                    expected = apply_action(source, action, direction)
                    actual = engine.to_game_state(engine.apply_action(compact, action, direction))
                    self.assertEqual(actual.snakes, expected.snakes)
                    self.assertEqual(actual.scores, expected.scores)
                    self.assertEqual(actual.shot_food, expected.shot_food)
                    self.assertEqual(actual.remaining_moves, expected.remaining_moves)
