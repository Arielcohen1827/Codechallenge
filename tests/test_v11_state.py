"""Behavioral contracts for cube geometry and partial server observations."""

import unittest
from dataclasses import replace

from snake_bot import snake_state as state
from snake_bot import game_rules


def cube_data(head=(10, 2, 2), objects=None, **fields):
    """Build coherent server slices without exposing the hidden cube."""
    objects = dict(objects or {})
    objects.setdefault(head, 'A')
    planes = {}
    for name, axis in (('board_x', 2), ('board_y', 1), ('board_z', 0)):
        lines = []
        for first in range(20):
            line = []
            for second in range(20):
                cell = list(head)
                variable = [value for value in range(3) if value != axis]
                cell[variable[0]], cell[variable[1]] = first, second
                line.append(objects.get(tuple(cell), ' '))
            lines.append('|' + ''.join(line) + '|')
        planes[name] = '\n'.join(lines)
    return dict(
        levels=20,
        rows=20,
        cols=20,
        board_size='20x20x20',
        head=dict(zip(('level', 'row', 'col'), head)),
        side='A',
        remaining_moves=400,
        score_1=0,
        score_2=0,
        board='\n\n'.join(planes[name] for name in ('board_x', 'board_y', 'board_z')),
        **planes,
        **fields,
    )


class CubeStateTests(unittest.TestCase):
    """Exercise the normal parser and rule entry points with V11 snapshots."""

    def test_three_planes_are_mapped_and_deduplicated(self):
        objects = {(4, 7, 2): 'X', (4, 2, 8): '#', (10, 7, 8): '⭐'}
        parsed = state.parse_state(cube_data(objects=objects))
        self.assertEqual(parsed.head('A'), (10, 2, 2))
        self.assertEqual(parsed.pickups, {(4, 7, 2)})
        self.assertEqual(parsed.walls, {(4, 2, 8)})
        self.assertEqual(parsed.stars, {(10, 7, 8)})
        self.assertEqual(len(parsed.observation.cells), 1141)

    def test_combined_board_has_same_observation(self):
        data = cube_data(objects={(1, 2, 2): 'X'})
        expected = state.parse_state(data)
        for name in ('board_x', 'board_y', 'board_z'):
            del data[name]
        self.assertEqual(state.parse_state(data).pickups, expected.pickups)

    def test_six_directions_preserve_level(self):
        parsed = state.parse_state(cube_data())
        for direction, target in (
            ('up', (10, 1, 2)),
            ('down', (10, 3, 2)),
            ('left', (10, 2, 1)),
            ('right', (10, 2, 3)),
            ('ascend', (9, 2, 2)),
            ('descend', (11, 2, 2)),
        ):
            with self.subTest(direction=direction):
                self.assertEqual(state.apply_move(parsed, direction).head('A'), target)

    def test_top_and_bottom_crashes(self):
        for level, direction in ((0, 'ascend'), (19, 'descend')):
            parsed = state.parse_state(cube_data(head=(level, 2, 2)))
            after = state.apply_move(parsed, direction)
            self.assertEqual(after.head('A'), parsed.head('A'))
            self.assertEqual(after.scores['A'], -500)
            protected = state.apply_move(replace(parsed, invincible_a=2), direction)
            self.assertEqual(protected.scores['A'], 0)

    def test_memory_is_uncertain_and_visible_empty_invalidates(self):
        first = state.parse_state(cube_data(objects={(10, 5, 5): 'X'}))
        second = state.parse_state(cube_data(head=(9, 2, 2)), knowledge=first.knowledge)
        self.assertNotIn((10, 5, 5), second.pickups)
        self.assertEqual(second.knowledge.remembered[(10, 5, 5)].symbol, 'X')
        third = state.parse_state(cube_data(), knowledge=second.knowledge)
        self.assertNotIn((10, 5, 5), third.knowledge.remembered)

    def test_invisible_enemy_is_not_a_remembered_collision(self):
        first = state.parse_state(cube_data(objects={(10, 5, 5): 'B'}))
        second = state.parse_state(cube_data(head=(9, 2, 2)), knowledge=first.knowledge)
        self.assertEqual(second.body('B'), ())
        self.assertNotIn((10, 5, 5), second.occupied())

    def test_bomb_distance_includes_other_level_and_invincibility(self):
        body = ((10, 2, 2),)
        parsed = replace(
            state.parse_state(cube_data()),
            partial=False,
            observation=None,
            snakes={'A': body, 'B': ()},
            bombs_large=frozenset({(7, 3, 3)}),
            scores={'A': 700, 'B': 0},
        )
        after = state.apply_move(parsed, 'ascend')
        self.assertEqual(after.head('A'), (9, 2, 2))
        self.assertEqual(after.scores['A'], 0)
        protected = state.apply_move(replace(parsed, invincible_a=2), 'ascend')
        self.assertEqual(protected.scores['A'], 700)

    def test_six_shooting_rays_and_invincible_target(self):
        for direction, delta in state.directions_for((10, 10, 10)).items():
            target = tuple(10 + 2 * value for value in delta)
            tail = tuple(10 + 3 * value for value in delta)
            parsed = replace(
                state.parse_state(cube_data(head=(10, 10, 10))),
                partial=False,
                observation=None,
                observed_snake_cells={},
                snakes={'A': ((10, 10, 10),), 'B': (target, tail)},
                scores={'A': 8000, 'B': 100},
            )
            with self.subTest(direction=direction):
                after = state.apply_shoot(parsed, direction)
                self.assertEqual(after.scores['A'], 3000)
                self.assertEqual(after.head('A'), (10, 10, 10))
                self.assertEqual(after.shot_food, {tail})
                self.assertEqual(
                    state.apply_shoot(replace(parsed, walls={target}), direction).shot_food, set()
                )
                protected = state.apply_shoot(replace(parsed, invincible_b=2), direction)
                self.assertEqual(protected.body('B'), (target, tail))

    def test_plate_erosion_all_orientations(self):
        for fixed in range(3):
            cells = set()
            for a in range(11):
                for b in range(11):
                    coordinate = [a, b]
                    coordinate.insert(fixed, 10)
                    cells.add(tuple(coordinate))
            shrunk = state.shrink_wall(cells)
            self.assertEqual(len(shrunk), 81)
            self.assertTrue(shrunk < cells)

    def test_star_candidate_equally_distant_and_visible_by_both(self):
        heads = ((0, 0, 0), (2, 2, 2))
        cells = game_rules.star_spawn_candidates(3, 3, heads, levels=3)
        self.assertTrue(cells)
        for cell in cells:
            self.assertEqual(state.manhattan(cell, heads[0]), state.manhattan(cell, heads[1]))
            self.assertTrue(any(a == b for a, b in zip(cell, heads[0])))
            self.assertTrue(any(a == b for a, b in zip(cell, heads[1])))

    def test_frenzy_requests_thirty_stars_and_bombs_expire(self):
        parsed = replace(
            state.parse_state(cube_data()), remaining_moves=101, bombs_large=frozenset({(0, 0, 0)})
        )
        after = state.apply_move(parsed, 'ascend')
        self.assertEqual(after.pending_star_spawns, 30)
        self.assertEqual(after.bombs_large, set())

    def test_partial_digits_do_not_invent_sequence(self):
        data = cube_data(objects={(10, 3, 3): '1', (10, 4, 4): '9'})
        data['remaining_moves'] = 200
        parsed = state.parse_state(data)
        self.assertIsNone(parsed.next_food_digit)
        self.assertEqual(parsed.food, set())


if __name__ == '__main__':
    unittest.main()
