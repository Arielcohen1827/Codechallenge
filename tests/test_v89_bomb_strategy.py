"""Economic bomb routing contracts; explosions remain legal, nonfatal actions."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from snake_bot import food_planner as planner
from snake_bot import snake_state
from snake_bot.snake_state import GameState
from snake_bot.game_rules import within_blast_zone
from snake_bot import game_rules


def snapshot(**changes):
    source = GameState(
        rows=20, cols=20, board=(), side='A', enemy='B',
        snakes={'A': ((5, 4), (5, 3)), 'B': ((15, 15), (15, 14))},
        food=frozenset(), shot_food=frozenset({(5, 12)}),
        scores={'A': 6000, 'B': 100}, remaining_moves=250,
        bombs_large=frozenset({(5, 8)}),
        reliable_tails=frozenset({'A', 'B'}), crashed_a=False, crashed_b=False)
    return replace(source, **changes)


def corridor(source):
    walls = frozenset({(row, col) for row in (4, 6) for col in range(3, 11)}
                      | {(5, 3)})
    return replace(source, walls=walls, snakes={**source.snakes, 'A': ((5, 4),)})


class BombEconomicPathTests(unittest.TestCase):
    def test_costly_bomb_gets_reasonable_detour_in_temporal_and_static_routes(self):
        source = snapshot()
        for find in (planner.temporal_shortest_path, planner.shortest_path):
            with self.subTest(find=find.__name__):
                path = find(source, source.head('A'), (5, 12), 'A')
                self.assertGreater(len(path) - 1, 8)
                self.assertFalse(any(within_blast_zone(cell, source.bombs_large,
                                                     source.bombs_small) for cell in path))
                after = planner.simulate_path_to_food(source, 'A', path)
                self.assertGreater(after.scores['A'], source.scores['A'])

    def test_small_positive_wipe_can_be_better_than_long_detour(self):
        source = snapshot(scores={'A': 1, 'B': 100})
        path = planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        self.assertEqual(len(path) - 1, 8)
        self.assertTrue(any(within_blast_zone(cell, source.bombs_large) for cell in path))
        after = planner.simulate_path_to_food(source, 'A', path)
        self.assertGreater(after.scores['A'], 1)
        self.assertEqual(len(after.body('A')), 3)

    def test_inevitable_bomb_does_not_make_route_or_action_illegal(self):
        source = corridor(snapshot())
        path = planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        self.assertEqual(len(path) - 1, 8)
        plan = planner.build_food_plan(source, 'A', (5, 12), None, 0)
        self.assertIsNotNone(plan)
        self.assertGreater(plan.bomb_loss, 6000)
        self.assertEqual(plan.bomb_trigger_count, 1)
        self.assertIn('right', snake_state.legal_moves(source, 'A'))
        clear = planner.build_food_plan(replace(source, bombs_large=frozenset()), 'A', (5, 12), None, 0)
        self.assertGreater(clear.value, plan.value)
        self.assertNotEqual(plan.safety, 'SUICIDAL')

    def test_path_expiration_uses_global_alternating_turn_clock(self):
        source = snapshot(remaining_moves=102)
        path = planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        self.assertEqual(len(path) - 1, 8)
        after = planner.simulate_path_to_food(source, 'A', path)
        self.assertGreater(after.scores['A'], 6000)
        self.assertFalse(after.pending_bomb_respawns)

    def test_expired_bombs_leave_legacy_path_and_planner_values_unchanged(self):
        source = snapshot(remaining_moves=100)
        clear = replace(source, bombs_large=frozenset())
        self.assertEqual(planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A'),
                         planner.temporal_shortest_path(clear, clear.head('A'), (5, 12), 'A'))
        self.assertEqual(planner.build_food_plan(source, 'A', (5, 12), None, 0).value,
                         planner.build_food_plan(clear, 'A', (5, 12), None, 0).value)

    def test_geometric_safe_candidate_is_optional_and_never_relabels_bombs_as_walls(self):
        source = snapshot()
        safe = snake_state.temporal_shortest_path(source, source.head('A'), (5, 12), 'A', avoid_blast=True)
        self.assertGreater(len(safe) - 1, 8)
        trapped = corridor(source)
        self.assertIsNone(snake_state.temporal_shortest_path(
            trapped, trapped.head('A'), (5, 12), 'A', avoid_blast=True))
        self.assertIsNotNone(snake_state.temporal_shortest_path(
            trapped, trapped.head('A'), (5, 12), 'A'))


class BombCandidateValueTests(unittest.TestCase):
    def test_immediate_explosion_is_economic_risk_not_structural_death(self):
        source = snapshot(snakes={'A': ((5, 5), (5, 4)), 'B': ((15, 15),)},
                          bombs_large=frozenset({(7, 8)}), shot_food=frozenset())
        after = planner.apply_move(source, 'right', 'A')
        self.assertNotEqual(planner.classify_safety(after, 'A')[0], 'SUICIDAL')
        bomb_score = planner.move_quality_score(source, 'A', 'right')[0]
        clear_score = planner.move_quality_score(replace(source, bombs_large=frozenset()), 'A', 'right')[0]
        self.assertLess(bomb_score, clear_score - 6000)
        self.assertIn('right', snake_state.legal_moves(source))

    def test_shot_food_competes_with_number_multiplier_and_only_own_tail_food(self):
        for side, multiplier in (('A', 2), ('B', 3)):
            with self.subTest(side=side):
                enemy = 'B' if side == 'A' else 'A'
                own_tail, other_tail = frozenset({(5, 5)}), frozenset({(7, 7)})
                source = snapshot(
                    side=side, enemy=enemy, bombs_large=frozenset(),
                    snakes={side: ((5, 4), (5, 3)), enemy: ((15, 15), (15, 14))},
                    multipliers={side: multiplier, enemy: 1},
                    shot_food=frozenset({(5, 6)}), food=frozenset({(5, 8)}),
                    food_values={(5, 8): 1}, next_food_digit=1,
                    pickups=frozenset({(6, 5)}),
                    tail_food_a=own_tail if side == 'A' else other_tail,
                    tail_food_b=own_tail if side == 'B' else other_tail)
                plans = planner.build_food_plans(source, side)
                by_kind = {plan.objective_kind: plan for plan in plans}
                self.assertTrue({'food', 'shot_food', 'multiplier', 'tail_food'} <= set(by_kind))
                self.assertEqual(by_kind['tail_food'].objective_reward, 100 * multiplier)
                self.assertEqual(by_kind['shot_food'].objective_reward, 100 * multiplier)
                self.assertNotIn((7, 7), {plan.food for plan in plans})
                self.assertEqual(planner.choose_food_plan(source, side, hunger=12).food, (5, 5))
                self.assertEqual(source.next_food_digit, 1)


class BombPlanningReuseTests(unittest.TestCase):
    def test_action_boundary_owns_cache_and_resets_it_on_errors(self):
        from snake_bot.snake_brain import SnakeBrain
        brain = SnakeBrain()
        scopes = []
        def move(data):
            scopes.append(planner._planning_cache.get())
            if data.get('fail'):
                raise RuntimeError('interrupted action')
            return 'up'
        with patch.object(brain, 'choose_move', side_effect=move):
            self.assertEqual(brain.choose_action({}), {'action_type': 'move', 'direction': 'up'})
            self.assertIsNotNone(scopes[0])
            self.assertIsNone(planner._planning_cache.get())
            with self.assertRaisesRegex(RuntimeError, 'interrupted action'):
                brain.choose_action({'fail': True})
            self.assertIsNotNone(scopes[1])
            self.assertIsNot(scopes[0], scopes[1])
            self.assertIsNone(planner._planning_cache.get())

    def test_temporal_bfs_does_not_recheck_blasts_for_already_visited_cells(self):
        source = snapshot(snakes={'A': ((0, 0),), 'B': ((19, 0),)},
                          bombs_large=frozenset(), bombs_small=frozenset({(10, 10)}))
        with patch.object(snake_state, 'path_blast_blocked',
                          wraps=snake_state.path_blast_blocked) as blocked:
            path = snake_state.temporal_shortest_path(source, (0, 0), (19, 19), 'A', avoid_blast=True)
            self.assertEqual(len(path), 39)
            self.assertLess(blocked.call_count, 2 * source.rows * source.cols)
        with patch.object(snake_state, 'path_blast_blocked',
                          wraps=snake_state.path_blast_blocked) as blocked:
            distances = snake_state.temporal_distance_map(source, (0, 0), 'A', avoid_blast=True)
            self.assertEqual(distances[(19, 19)], len(path) - 1)
            self.assertLess(blocked.call_count, 2 * source.rows * source.cols)

    def test_path_extension_reuses_shared_prefix_with_identical_bomb_economy(self):
        source = snapshot()
        path = planner.geometric_temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        expected = planner.simulate_path_economy(source, 'A', path)
        with planner._planning_scope(), patch.object(
                planner, 'official_apply_move', wraps=planner.official_apply_move) as apply:
            planner.simulate_path_economy(source, 'A', path[:4])
            self.assertEqual(planner.simulate_path_economy(source, 'A', path), expected)
            self.assertEqual(apply.call_count, len(path) - 1)

    def test_repeated_distance_queries_preserve_order_and_return_independent_maps(self):
        source = snapshot()
        positions = ((5, 12), (5, 11))
        with planner._planning_scope(), patch.object(
                planner, 'temporal_distance_map', wraps=planner.temporal_distance_map) as find:
            first = planner.food_distances(source, 'A', positions)
            expected = dict(first)
            self.assertEqual(list(first), list(positions))
            first.pop(positions[0])
            second = planner.food_distances(source, 'A', positions)
            self.assertEqual(second, expected)
            self.assertEqual(find.call_count, 1)
            reversed_map = planner.food_distances(source, 'A', positions[::-1])
            self.assertEqual(list(reversed_map), list(positions[::-1]))

    def test_repeated_routes_reuse_queries_without_aliasing_different_score_states(self):
        source = snapshot()
        cheap = replace(source, scores={'A': 1, 'B': 100})
        with planner._planning_scope(), patch.object(
                planner, 'geometric_temporal_shortest_path',
                wraps=planner.geometric_temporal_shortest_path) as find:
            expensive_path = planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
            calls = find.call_count
            self.assertEqual(planner.temporal_shortest_path(source, source.head('A'), (5, 12), 'A'),
                             expensive_path)
            self.assertEqual(find.call_count, calls)
            cheap_path = planner.temporal_shortest_path(cheap, cheap.head('A'), (5, 12), 'A')
            self.assertLess(len(cheap_path), len(expensive_path))

    def test_future_food_geometry_does_not_simulate_from_the_wrong_head(self):
        source = snapshot()
        start, goal = (5, 13), (5, 4)
        expected = planner.geometric_temporal_shortest_path(source, start, goal, 'A')
        with patch.object(planner, 'simulate_path_economy',
                          side_effect=AssertionError('cannot simulate a different starting head')):
            self.assertEqual(planner.temporal_shortest_path(source, start, goal, 'A'), expected)

    def test_precomputed_blast_cells_match_shared_rule_on_rectangular_boards(self):
        for rows, cols in ((20, 20), (20, 30), (30, 20), (30, 30)):
            large, small = frozenset({(0, 0), (rows - 1, cols - 1)}), frozenset({(10, 10)})
            cells = game_rules.blast_zone_cells(rows, cols, large, small)
            expected = frozenset((r, c) for r in range(rows) for c in range(cols)
                                 if within_blast_zone((r, c), large, small))
            self.assertEqual(cells, expected)
            for remaining in (100, 101, 102, 103):
                source = snapshot(rows=rows, cols=cols, remaining_moves=remaining,
                                  bombs_large=large, bombs_small=small)
                for side in ('A', 'B'):
                    for turn in (1, 2):
                        self.assertEqual(
                            snake_state.path_blast_blocked(source, (0, 1), side, turn, cells),
                            snake_state.path_blast_blocked(source, (0, 1), side, turn))

    def test_repeated_simulation_reuses_shared_transition_only_inside_scope(self):
        source = snapshot()
        path = planner.geometric_temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        with patch.object(planner, 'official_apply_move', wraps=planner.official_apply_move) as apply:
            with planner._planning_scope():
                first = planner.simulate_path_economy(source, 'A', path)
                first_calls = apply.call_count
                second = planner.simulate_path_economy(source, 'A', path)
                self.assertGreater(first_calls, 0)
                self.assertEqual(apply.call_count, first_calls)
                self.assertEqual(second, first)
            self.assertEqual(planner.simulate_path_economy(source, 'A', path), first)
            self.assertEqual(apply.call_count, first_calls * 2)

    def test_nested_and_failed_scopes_do_not_leak_results_between_decisions(self):
        source = snapshot(remaining_moves=102)
        path = planner.geometric_temporal_shortest_path(source, source.head('A'), (5, 12), 'A')
        with patch.object(planner, 'official_apply_move', wraps=planner.official_apply_move) as apply:
            with self.assertRaisesRegex(RuntimeError, 'interrupted decision'):
                with planner._planning_scope():
                    expected = planner.simulate_path_economy(source, 'A', path)
                    calls = apply.call_count
                    with planner._planning_scope():
                        self.assertEqual(planner.simulate_path_economy(source, 'A', path), expected)
                    self.assertEqual(apply.call_count, calls)
                    raise RuntimeError('interrupted decision')
            self.assertEqual(planner.simulate_path_economy(source, 'A', path), expected)
            self.assertEqual(apply.call_count, calls * 2)


if __name__ == '__main__':
    unittest.main()
