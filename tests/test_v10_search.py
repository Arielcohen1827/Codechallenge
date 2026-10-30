"""Search and planning contracts for v10, independent of server RNG."""
from dataclasses import replace
import unittest

from snake_bot import food_planner as planner
from snake_bot.move_search import TurnSearchCache, state_key, static_after_move_score
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import GameState, apply_action


def snapshot(**changes):
    state = GameState(
        rows=9, cols=13, board=(), side='A', enemy='B',
        snakes={'A': ((4, 3), (4, 2), (4, 1)),
                'B': ((4, 8), (5, 8), (6, 8))},
        food=frozenset(), scores={'A': 7000, 'B': 6000},
        remaining_moves=220, reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False, crashed_b=False)
    return replace(state, **changes)


class CompactV10Tests(unittest.TestCase):
    def test_roundtrip_retains_observed_and_unknown_star_state(self):
        for side in ('A', 'B'):
            state = snapshot(side=side, enemy='B' if side == 'A' else 'A',
                stars=frozenset({(3, 3), (3, 7)}), invincible_a=50,
                invincible_b=1, star_spawn_due_round=None, pending_star_spawns=2,
                timeouts_a=3, timeouts_b=7)
            engine, compact = IterativeSearchEngine.from_game_state(state)
            restored = engine.to_game_state(compact)
            self.assertEqual(restored.stars, state.stars)
            self.assertEqual((restored.invincible_a, restored.invincible_b), (50, 1))
            self.assertEqual(restored.star_spawn_due_round, None)
            self.assertEqual(restored.pending_star_spawns, 2)
            self.assertEqual((restored.timeouts_a, restored.timeouts_b), (3, 7))

    def test_cache_keys_distinguish_counter_geometry_clock_and_timeouts(self):
        state = snapshot()
        engine, compact = IterativeSearchEngine.from_game_state(state)
        for changes in ({'invincible_a': 1}, {'invincible_b': 50},
                        {'stars': frozenset({(3, 3)})},
                        {'star_spawn_due_round': 80}, {'pending_star_spawns': 1},
                        {'timeouts_a': 9}, {'timeouts_b': 9}):
            other = replace(state, **changes)
            with self.subTest(changes=changes):
                self.assertNotEqual(state_key(state), state_key(other))
                self.assertNotEqual(compact, engine.to_compact(other))

    def test_protected_moves_remain_four_directions_without_duplicate_ram(self):
        for side in ('A', 'B'):
            state = snapshot(side=side, enemy='B' if side == 'A' else 'A',
                invincible_a=3, invincible_b=3)
            engine, compact = IterativeSearchEngine.from_game_state(state)
            self.assertEqual(set(engine.available_actions(compact)),
                             {'up', 'down', 'left', 'right'})
            self.assertEqual(set(TurnSearchCache().legal_actions(state, side)),
                             {'up', 'down', 'left', 'right'})

    def test_selective_shoots_omit_invincible_target_and_return_after_expiry(self):
        for side in ('A', 'B'):
            state = snapshot(side=side, enemy='B' if side == 'A' else 'A',
                invincible_a=5 if side == 'B' else 0,
                invincible_b=5 if side == 'A' else 0)
            engine, compact = IterativeSearchEngine.from_game_state(state)
            self.assertEqual(engine.shoot_actions(compact), ())
            self.assertFalse(any(isinstance(a, tuple)
                for a in TurnSearchCache().legal_actions(state, side)))
            normal = replace(state, invincible_a=0, invincible_b=0)
            self.assertTrue(engine.shoot_actions(engine.to_compact(normal)))

    def test_compact_transitions_match_shared_engine_for_star_ram_shoot_and_expiry(self):
        cases = [
            (snapshot(stars=frozenset({(4, 4)})), 'move', 'right'),
            (snapshot(invincible_a=2, snakes={'A': ((4, 7),),
                'B': ((4, 8), (5, 8), (6, 8))}), 'move', 'right'),
            (snapshot(invincible_a=2, invincible_b=2), 'shoot', 'right'),
            (snapshot(invincible_a=1), 'invalid', 'left'),
            (snapshot(invincible_a=1), 'timeout', None),
            (snapshot(remaining_moves=101), 'move', 'up'),
        ]
        for state, action, direction in cases:
            with self.subTest(action=action, direction=direction):
                engine, compact = IterativeSearchEngine.from_game_state(state)
                expected = engine.to_compact(apply_action(state, action, direction))
                self.assertEqual(engine.apply_action(compact, action, direction), expected)

    def test_wrong_digit_is_productive_only_during_protection(self):
        base = snapshot(food_values={(4, 4): 2, (3, 3): 1},
                        food=frozenset({(3, 3)}), next_food_digit=1)
        engine, compact = IterativeSearchEngine.from_game_state(base)
        self.assertNotIn('right', engine.productive_moves_for(compact, 0))
        protected = engine.to_compact(replace(base, invincible_a=1))
        self.assertIn('right', engine.productive_moves_for(protected, 0))

    def test_productive_moves_do_not_count_protected_stationary_wall_and_neck(self):
        state = snapshot(snakes={'A': ((0, 0), (1, 0)), 'B': ((8, 12),)},
                         invincible_a=3)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertEqual(engine.productive_moves_for(compact, 0), ('right',))

    def test_compact_preserves_numeric_copy_targets_for_wrong_digit_respawn(self):
        state = snapshot(food_values={(4, 4): 2, (3, 3): 1},
            food=frozenset({(3, 3)}), next_food_digit=1,
            food_copy_targets={1: 4, 2: 5}, invincible_a=2)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertEqual(engine.to_game_state(compact).food_copy_targets, {1: 4, 2: 5})
        expected = engine.to_compact(apply_action(state, 'move', 'right'))
        self.assertEqual(engine.apply_move(compact, 'right'), expected)

    def test_completed_deep_search_prices_star_and_free_ram_without_shoot_branch(self):
        star = snapshot(remaining_moves=3, stars=frozenset({(3, 3)}),
                        scores={'A': 100, 'B': 150})
        engine, compact = IterativeSearchEngine.from_game_state(star, 1000, 3)
        result = engine.search(compact, 'A', ['up', 'right', 'down'])
        self.assertEqual(result.completed_depth, 3)
        self.assertGreater(result.scores['up'], result.scores['right'])
        self.assertGreater(result.scores['up'], result.scores['down'])
        ram = snapshot(invincible_a=3, invincible_b=3,
            snakes={'A': ((4, 7),), 'B': ((4, 8), (5, 8), (6, 8))})
        engine, compact = IterativeSearchEngine.from_game_state(ram, 1000, 2)
        result = engine.search(compact, 'A', list(('up', 'down', 'left', 'right')))
        self.assertEqual(set(result.scores), {'up', 'down', 'left', 'right'})
        self.assertEqual(len(engine.apply_move(compact, 'right').bodies[1]), 1)

    def test_search_values_protection_duration_and_race_on_existing_point_scale(self):
        state = snapshot(invincible_a=1)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertGreater(engine.evaluate(engine.to_compact(replace(state, invincible_a=20))),
                           engine.evaluate(compact))
        near = replace(state, invincible_a=0, stars=frozenset({(4, 4)}))
        lost = replace(state, invincible_a=0, stars=frozenset({(4, 7)}))
        self.assertGreater(engine.star_position_value(engine.to_compact(near), 0),
                           engine.star_position_value(engine.to_compact(lost), 0))


class StarPlanningTests(unittest.TestCase):
    def test_real_policy_selects_body_ram_when_it_opens_our_food_before_rival_arrives(self):
        from snake_bot.snake_brain import SnakeBrain
        from tools.bot_lab import make_turn_data, observe_simulation_state
        state = snapshot(rows=20, cols=30, invincible_a=4, invincible_b=5,
            remaining_moves=240, scores={'A': 6000, 'B': 7000},
            snakes={'A': ((10, 5), (10, 4), (10, 3)),
                    'B': ((12, 6), (11, 6), (10, 6), (9, 6), (8, 6))})
        brain = SnakeBrain(enable_debug=True)
        brain.previous_snakes = {f'active_{i}': state.snakes for i in range(10)}
        data = make_turn_data(state, 'v10-real-ram', 0)
        observe_simulation_state(brain, state, data)
        action = brain.choose_action(data)
        self.assertEqual(action, {'action_type': 'move', 'direction': 'right'})

    def test_star_is_separate_goal_with_scaled_reward_and_real_path(self):
        for multiplier in (1, 4):
            state = snapshot(stars=frozenset({(3, 3)}), multipliers={'A': multiplier, 'B': 1})
            self.assertEqual(planner.objective_kind(state, (3, 3), 'A'), 'star')
            self.assertEqual(planner.objective_reward(state, 'A', (3, 3)), 100 * multiplier)
            plan = planner.build_food_plan(state, 'A', (3, 3), None, 0)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.objective_kind, 'star')
            self.assertEqual(plan.our_distance, 1)
            self.assertEqual(plan.first_move, 'up')
            self.assertIn((3, 3), {p.food for p in planner.build_food_plans(state, 'A')})

    def test_star_strategy_respects_race_existing_duration_and_horizon(self):
        state = snapshot(stars=frozenset({(3, 3)}))
        fresh = planner.star_strategic_points(state, 'A', 1, 6)
        self.assertGreater(fresh, 0)
        self.assertEqual(planner.star_strategic_points(state, 'A', 6, 1), 0)
        self.assertLess(planner.star_strategic_points(replace(state, invincible_a=50), 'A', 1, 6), fresh)
        self.assertEqual(planner.star_strategic_points(replace(state, remaining_moves=1), 'A', 1, 6), 0)

    def test_star_not_reachable_before_end_does_not_create_plan(self):
        state = snapshot(stars=frozenset({(3, 3)}), remaining_moves=0)
        self.assertIsNone(planner.build_food_plan(state, 'A', (3, 3), None, 0))

    def test_protected_enclosure_is_not_suicide_but_expiry_removes_protection(self):
        state = snapshot(snakes={'A': ((0, 0),), 'B': ((8, 12),)},
                         walls=frozenset({(0, 1), (1, 0)}), invincible_a=2)
        self.assertNotEqual(planner.classify_safety(state, 'A')[0], 'SUICIDAL')
        self.assertEqual(planner.classify_safety(replace(state, invincible_a=0), 'A')[0], 'SUICIDAL')

    def test_protected_bomb_route_has_no_economic_loss(self):
        state = snapshot(invincible_a=5, bombs_small=frozenset({(3, 4)}),
                         stars=frozenset({(4, 6)}))
        path = ((4, 3), (4, 4), (4, 5), (4, 6))
        after, loss, triggered = planner.simulate_path_economy(state, 'A', path)
        self.assertIsNotNone(after)
        self.assertEqual(loss, 0)
        self.assertEqual(triggered, 1)

    def test_tactical_wrong_digit_and_wall_penalties_disappear_while_protected(self):
        from snake_bot.bot_weights import get_weights
        state = snapshot(invincible_a=2, food_values={(4, 4): 2}, next_food_digit=1)
        clear = replace(state, food_values={})
        # Exact transitions grow on the wrong digit, so isolate only the legacy
        # hardcoded penalty with a common positional evaluator.
        from unittest.mock import patch
        with patch('snake_bot.move_search.position_score', return_value=0):
            wrong = static_after_move_score(state, 'A', 'right', None, 0, 0, TurnSearchCache())
            empty = static_after_move_score(clear, 'A', 'right', None, 0, 0, TurnSearchCache())
        self.assertGreater(wrong - empty, -get_weights()['deep_wrong_food_penalty'])


if __name__ == '__main__':
    unittest.main()
