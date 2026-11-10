"""Cube search must preserve levels, observation boundaries and six-way motion."""

from dataclasses import replace
import unittest

from snake_bot import food_planner
from snake_bot.move_search import TurnSearchCache, state_key
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import GameState, parse_state
from tests.test_v11_runtime import cube_payload
from snake_bot.observation import CubeObservation, KnowledgeState, RememberedCell
from snake_bot.geometry import neighbors


def cube(**changes):
    state = GameState(
        rows=20,
        cols=20,
        levels=20,
        board=(),
        side='A',
        enemy='B',
        snakes={
            'A': ((10, 5, 5), (10, 5, 4), (10, 5, 3)),
            'B': ((12, 5, 5), (12, 5, 6), (12, 5, 7)),
        },
        food=frozenset({(9, 5, 5)}),
        scores={'A': 7000, 'B': 6000},
        remaining_moves=220,
        reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False,
        crashed_b=False,
    )
    return replace(state, **changes)


class CubeCompactTests(unittest.TestCase):
    def test_neighbor_iterators_remain_independent_for_repeated_cube_cells(self):
        first = neighbors([1, 1, 1], 3, 3, 3)
        second = neighbors((1, 1, 1), 3, 3, 3)
        expected = {
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
            (0, 1, 1),
            (2, 1, 1),
        }
        self.assertEqual(set(first), expected)
        self.assertEqual(set(second), expected)

    def test_neighbor_geometry_does_not_reuse_bounds_from_another_cube(self):
        small = set(neighbors((1, 1, 1), 2, 2, 2))
        large = set(neighbors((1, 1, 1), 3, 3, 3))
        self.assertEqual(small, {(1, 0, 1), (1, 1, 0), (0, 1, 1)})
        self.assertEqual(len(large), 6)
        self.assertTrue(small < large)

    def test_visible_disconnected_rival_segment_is_still_occupied(self):
        state = cube(
            observed_snake_cells={'B': frozenset({(9, 5, 5)})},
            snakes={'A': ((10, 5, 5),), 'B': ()},
        )
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertNotIn('ascend', engine.legal_moves(compact))

    def test_cell_indices_roundtrip_without_aliasing_levels(self):
        engine = IterativeSearchEngine(20, 20, levels=20)
        cells = ((0, 0, 0), (1, 0, 0), (19, 19, 19), (10, 5, 5))
        indices = tuple(engine.index(cell) for cell in cells)
        self.assertEqual(len(set(indices)), len(cells))
        self.assertEqual(tuple(engine.cell(index) for index in indices), cells)

    def test_compact_roundtrip_keeps_levels_and_objects(self):
        state = cube(stars=frozenset({(11, 5, 5)}), invincible_a=50)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        restored = engine.to_game_state(compact)
        self.assertEqual(restored.levels, 20)
        self.assertEqual(restored.snakes, state.snakes)
        self.assertEqual(restored.food, state.food)
        self.assertEqual(restored.stars, state.stars)

    def test_vertical_moves_use_shared_transition_rules(self):
        state = cube()
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertIn('ascend', engine.available_moves(compact))
        self.assertIn('descend', engine.available_moves(compact))
        after = engine.to_game_state(engine.apply_move(compact, 'ascend'))
        self.assertEqual(after.head('A'), (9, 5, 5))
        self.assertEqual(after.scores['A'], 7100)

    def test_expand_never_wraps_between_levels_or_rows(self):
        engine = IterativeSearchEngine(20, 20, levels=20)
        origin = engine.index((1, 0, 0))
        expanded = engine.expand(1 << origin)
        actual = {engine.cell(index) for index in range(engine.size) if expanded & (1 << index)}
        self.assertEqual(actual, {(0, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1)})


class CubePlanningTests(unittest.TestCase):
    def test_partial_rival_reply_statistics_are_explicitly_uncertain(self):
        for cells in ({}, {(10, 8, 8): 'B'}):
            with self.subTest(cells=cells):
                state = parse_state(cube_payload(cells=cells))
                stats = food_planner.rival_response_stats(state, 'A', 'ascend')
                self.assertTrue(stats['uncertain'])
                self.assertFalse(stats['forced_zero'])
                self.assertFalse(stats['forced_delayed_zero'])

    def test_partial_survival_does_not_treat_an_invisible_rival_as_no_reply_loss(self):
        for cells in ({}, {(10, 8, 8): 'B'}):
            with self.subTest(cells=cells):
                state = parse_state(cube_payload(cells=cells))
                analysis = food_planner.survival_move_analysis(state, 'A', 'ascend')
                after = food_planner.apply_move(state, 'ascend', 'A')
                self.assertEqual(
                    analysis['future_score'], food_planner.survival_position_score(after, 'A')
                )
                self.assertIsNone(analysis['enemy_reply'])

    def test_center_and_edges_include_level_axis(self):
        state = cube()
        self.assertEqual(food_planner.edge_distance(state, (0, 10, 10)), 0)
        self.assertEqual(food_planner.edge_distance(state, (10, 10, 10)), 9)
        self.assertEqual(food_planner.center_distance(state, (10, 10, 10)), 1.5)

    def test_distance_map_finds_vertical_objective(self):
        state = cube()
        distances = food_planner.distance_map(state, state.head('A'), 'A')
        self.assertEqual(distances[(9, 5, 5)], 1)
        cache = TurnSearchCache()
        self.assertEqual(cache.distance_map(state, state.head('A'), 'A')[(9, 5, 5)], 1)

    def test_cache_key_distinguishes_cube_dimensions(self):
        state = cube()
        self.assertNotEqual(state_key(state), state_key(replace(state, levels=19)))

    def test_partial_paths_do_not_claim_unknown_cells_are_free(self):
        observation = CubeObservation((10, 5, 5), {(10, 5, 5): 'A', (9, 5, 5): '1'}, {})
        state = cube(partial=True, observation=observation, snakes={'A': ((10, 5, 5),), 'B': ()})
        for distances in (
            food_planner.distance_map(state, state.head('A'), 'A'),
            TurnSearchCache().distance_map(state, state.head('A'), 'A'),
        ):
            self.assertEqual(distances, {(10, 5, 5): 0, (9, 5, 5): 1})

    def test_remembered_rival_does_not_create_a_confirmed_shot(self):
        observation = CubeObservation((10, 5, 5), {(10, 5, 5): 'A'}, {})
        knowledge = KnowledgeState(
            remembered={(9, 5, 5): RememberedCell('B', 2)},
            last_enemy_head=(9, 5, 5),
            last_enemy_turn=2,
        )
        state = cube(
            partial=True,
            observation=observation,
            knowledge=knowledge,
            snakes={'A': ((10, 5, 5),), 'B': ()},
        )
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertEqual(engine.shoot_actions(compact), ())
        self.assertEqual(
            TurnSearchCache().legal_actions(state, 'A'),
            tuple(TurnSearchCache().legal_moves(state, 'A')),
        )

    def test_observation_and_memory_are_part_of_transposition_identity(self):
        observation = CubeObservation((10, 5, 5), {(10, 5, 5): 'A'}, {})
        state = cube(partial=True, observation=observation, knowledge=KnowledgeState())
        remembered = KnowledgeState(remembered={(9, 0, 0): RememberedCell('X', 3)})
        other = replace(state, knowledge=remembered)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertNotEqual(compact, engine.to_compact(other))
        self.assertNotEqual(state_key(state), state_key(other))

    def test_partial_search_returns_own_observed_result_without_enemy_head(self):
        observation = CubeObservation((10, 5, 5), {(10, 5, 5): 'A', (9, 5, 5): '1'}, {})
        state = cube(partial=True, observation=observation, snakes={'A': ((10, 5, 5),), 'B': ()})
        engine, compact = IterativeSearchEngine.from_game_state(state)
        result = engine.search(compact, 'A', ('ascend',), target=(9, 5, 5))
        self.assertEqual(set(result.scores), {'ascend'})
        self.assertEqual(result.completed_depth, 1)
        self.assertEqual(result.nodes, 1)


if __name__ == '__main__':
    unittest.main()
