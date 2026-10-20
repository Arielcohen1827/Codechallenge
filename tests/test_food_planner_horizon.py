import unittest
from dataclasses import replace
from unittest.mock import patch

from snake_bot import food_planner as planner
from tests.test_food_planner import open_state, sample_plan


def pickup_state(remaining, distance, side='A', actor=None):
    enemy = 'B' if side == 'A' else 'A'
    actor = actor or side
    return open_state(
        rows=7, cols=24, board=tuple(' ' * 24 for _ in range(7)),
        side=actor, enemy='B' if actor == 'A' else 'A',
        snakes={side: ((3, 1),), enemy: ((6, 23),)},
        food=frozenset(), remaining_moves=remaining,
        pickups=frozenset({(3, 1 + distance)}),
    )


class TestMultiplierHorizon(unittest.TestCase):
    def test_impossible_x_has_no_plan_or_economic_denial_evaluation(self):
        for side in ('A', 'B'):
            for remaining, distance in ((18, 14), (32, 17), (9, 8)):
                with self.subTest(side=side, remaining=remaining):
                    state = pickup_state(remaining, distance, side)
                    with patch.object(planner, 'projected_sequence_income', return_value=0) as income, \
                            patch.object(planner, 'food_denial_score', return_value=0) as denial:
                        plan = planner.build_food_plan(state, side, next(iter(state.pickups)), None, 0)
                    self.assertIsNone(plan)
                    income.assert_not_called()
                    denial.assert_not_called()

    def test_last_own_turn_and_one_step_beyond_for_both_actors_and_parities(self):
        for side in ('A', 'B'):
            for actor in ('A', 'B'):
                for remaining in (8, 9):
                    turns = (remaining + (side == actor)) // 2
                    for distance in (turns, turns + 1):
                        with self.subTest(side=side, actor=actor, remaining=remaining, distance=distance):
                            state = pickup_state(remaining, distance, side, actor)
                            plan = planner.build_food_plan(state, side, next(iter(state.pickups)), None, 0)
                            self.assertEqual(plan is not None, distance == turns)

    def test_impossible_x_never_selected_from_stale_plans(self):
        for side in ('A', 'B'):
            for actor in ('A', 'B'):
                for remaining in (8, 9, 18, 32):
                    turns = (remaining + (side == actor)) // 2
                    state = pickup_state(remaining, turns + 1, side, actor)
                    target = next(iter(state.pickups))
                    stale = sample_plan(
                        food=target, our_distance=turns + 1, value=10**12,
                        objective_kind='multiplier', denial_score=10**9,
                        enemy_distance=2, race_margin=20, endgame_acceptable=True,
                    )
                    reachable = sample_plan(food=(3, 2), our_distance=1)
                    with self.subTest(side=side, actor=actor, remaining=remaining):
                        self.assertIsNone(planner.choose_food_plan(state, side, target, plans=[stale]))
                        self.assertIs(planner.choose_food_plan(
                            state, side, target, plans=[stale, reachable]), reachable)

    def test_two_x_same_first_direction_keep_only_reachable_target(self):
        for remaining, far, near in ((18, 14, 5), (9, 8, 3)):
            state = pickup_state(remaining, far)
            near_target = (3, 1 + near)
            state = replace(state, pickups=state.pickups | {near_target})
            plans = planner.build_food_plans(state, 'A')
            self.assertEqual([plan.food for plan in plans], [near_target])
            self.assertEqual(plans[0].first_move, 'right')
            self.assertEqual(planner.choose_food_plan(state, 'A', plans=plans).food, near_target)

    def test_x_economic_arrival_has_exact_remaining_and_opponent_turn(self):
        for side in ('A', 'B'):
            for actor in ('A', 'B'):
                for remaining, distance in ((34, 4), (9, 4), (8, 4), (18, 5)):
                    with self.subTest(side=side, actor=actor, remaining=remaining):
                        state = pickup_state(remaining, distance, side, actor)
                        state = replace(state, walls=frozenset((0, col) for col in range(24)))
                        target = next(iter(state.pickups))
                        path = planner.temporal_shortest_path(state, state.head(side), target, side)
                        legacy_after = planner.simulate_path_to_food(state, side, path)
                        self.assertEqual(legacy_after.body(side), (target,))
                        self.assertEqual(legacy_after.multipliers[side], 2)
                        self.assertEqual(legacy_after.walls, frozenset(
                            (0, col) for col in range(distance, 24 - distance)))
                        cost = 2 * distance - (side == actor)
                        enemy = 'B' if side == 'A' else 'A'
                        expected = replace(legacy_after, remaining_moves=remaining - cost,
                                           side=enemy, enemy=side)
                        with patch.object(planner, 'projected_sequence_income', wraps=planner.projected_sequence_income) as income:
                            plan = planner.build_food_plan(state, side, target, None, 0)
                        self.assertIsNotNone(plan)
                        self.assertEqual(income.call_args_list[0].args, (expected, side))
                        self.assertEqual(income.call_args_list[1].args, (expected, enemy))
                        self.assertEqual(income.call_args_list[2].args, (state, side, distance))
                        safety, space, exits, tail = planner.classify_safety(legacy_after, side)
                        self.assertEqual((plan.safety, plan.space_after, plan.exits_after,
                                          plan.tail_reachable_after), (safety, space, exits, tail))
                        self.assertEqual(plan.path, tuple(path))
                        self.assertEqual(plan.objective_reward, planner.objective_reward(state, side, target))
                        distances = planner.distance_map(legacy_after, legacy_after.head(side), side)
                        hunger = planner.score_gap_urgency(state, side)
                        self.assertEqual(plan.chain_score, planner.food_chain_score(
                            legacy_after, side, target, hunger, distances))
                        self.assertEqual(plan.next_food_distance, planner.nearest_next_food_distance(
                            legacy_after, side, target, distances))

    def test_visible_sequence_cannot_use_extra_turn_after_x(self):
        for side in ('A', 'B'):
            for actor in ('A', 'B'):
                for remaining in (8, 9, 10):
                    state = pickup_state(remaining, 3, side, actor)
                    target = next(iter(state.pickups))
                    numbered = (3, target[1] + 2)
                    state = replace(state, food=frozenset({numbered}),
                                    food_values={numbered: 1}, next_food_digit=1)
                    with patch.object(planner, 'estimated_multiplier_points', wraps=planner.estimated_multiplier_points) as fallback:
                        plan = planner.build_food_plan(state, side, target, None, 0)
                    after_remaining = remaining - (6 - (side == actor))
                    expected_income = 200 if after_remaining // 2 >= 2 else 0
                    with self.subTest(side=side, actor=actor, remaining=remaining):
                        self.assertEqual(plan.projected_income, expected_income)
                        self.assertEqual(plan.rival_projected_income, 0)
                        self.assertEqual(plan.economic_swing, expected_income)
                        self.assertEqual(plan.multiplier_marginal, expected_income)
                        if expected_income == 0:
                            self.assertEqual(fallback.call_args_list[0].args, (state, side, 3))

    def test_numbered_food_uses_unchanged_projection_and_shared_simulation(self):
        for side in ('A', 'B'):
            state = pickup_state(34, 4, side)
            target = next(iter(state.pickups))
            state = replace(state, pickups=frozenset(), food=frozenset({target}),
                            food_values={target: 1}, next_food_digit=1)
            path = planner.temporal_shortest_path(state, state.head(side), target, side)
            after = planner.simulate_path_to_food(state, side, path)
            self.assertEqual(after.remaining_moves, 30)
            self.assertEqual(after.side, side)
            with patch.object(planner, 'projected_sequence_income', wraps=planner.projected_sequence_income) as income:
                plan = planner.build_food_plan(state, side, target, None, 0)
            self.assertIsNotNone(plan)
            self.assertEqual(income.call_args_list[0].args, (after, side))
            self.assertEqual(plan.objective_reward, 100)


if __name__ == '__main__':
    unittest.main()
