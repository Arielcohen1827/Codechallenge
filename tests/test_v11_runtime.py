"""Production observations retain uncertainty while applying the cube protocol."""

import unittest

from snake_bot.snake_brain import SnakeBrain
from v7_logging import safe_message, verification_record


def cube_payload(head=(10, 5, 5), cells=None, remaining_moves=400, **changes):
    """Build independent wire planes from a sparse true cube fixture."""
    cells = dict(cells or {})
    cells[head] = 'A'
    level, row, col = head
    grids = {
        'board_x': [[cells.get((i, j, col), ' ') for j in range(20)] for i in range(20)],
        'board_y': [[cells.get((i, row, j), ' ') for j in range(20)] for i in range(20)],
        'board_z': [[cells.get((level, i, j), ' ') for j in range(20)] for i in range(20)],
    }
    planes = {
        key: '\n'.join('|' + ''.join(line) + '|' for line in grid) for key, grid in grids.items()
    }
    return {
        'game_id': 'cube-runtime',
        'turn_token': str(remaining_moves),
        'side': 'A',
        'levels': 20,
        'rows': 20,
        'cols': 20,
        'board_size': '20x20x20',
        'head': dict(zip(('level', 'row', 'col'), head)),
        'board': '\n\n'.join(planes.values()),
        **planes,
        'remaining_moves': remaining_moves,
        'score_1': 6000,
        'score_2': 6000,
        'multiplier_1': 1,
        'multiplier_2': 1,
        'invincible_1': 0,
        'invincible_2': 0,
        **changes,
    }


class CubeRuntimeTests(unittest.TestCase):
    def test_real_decision_selects_each_vertical_exit(self):
        head = (10, 5, 5)
        horizontal_walls = {
            (10, 4, 5): '#',
            (10, 6, 5): '#',
            (10, 5, 4): '#',
            (10, 5, 6): '#',
        }
        for direction, destination, blocked in (
            ('ascend', (9, 5, 5), (11, 5, 5)),
            ('descend', (11, 5, 5), (9, 5, 5)),
        ):
            with self.subTest(direction=direction):
                payload = cube_payload(
                    head, {**horizontal_walls, blocked: '#', destination: '1'}, score_1=0
                )
                action = SnakeBrain(enable_debug=False).choose_action(payload)
                self.assertEqual(action, {'action_type': 'move', 'direction': direction})

    def test_play_data_retains_vertical_direction(self):
        import run

        for direction in ('ascend', 'descend'):
            with self.subTest(direction=direction):
                merged = run.merge_action_fields(
                    cube_payload(), {'action': 'shoot', 'direction': direction}
                )
                self.assertEqual(merged['direction'], direction)

    def test_observation_remembers_hidden_objects_and_reobserved_empty_invalidates(self):
        brain = SnakeBrain(enable_debug=False)
        cell = (10, 8, 8)
        first = brain.observe_state(cube_payload(cells={cell: 'X'}))
        self.assertIn(cell, first.pickups)
        hidden = brain.observe_state(cube_payload(head=(11, 6, 6), remaining_moves=398))
        self.assertNotIn(cell, hidden.pickups)
        self.assertIn(cell, hidden.knowledge.remembered)
        visible = brain.observe_state(cube_payload(remaining_moves=396))
        self.assertNotIn(cell, visible.knowledge.remembered)

    def test_occlusion_does_not_infer_enemy_crash(self):
        brain = SnakeBrain(enable_debug=False)
        brain.observe_state(cube_payload(cells={(10, 8, 8): 'B', (10, 8, 9): 'b'}))
        hidden = brain.observe_state(cube_payload(head=(11, 6, 6), remaining_moves=396))
        self.assertEqual(hidden.body('B'), ())
        self.assertIsNot(hidden.crashed_b, True)

    def test_same_clock_corrected_planes_do_not_reuse_old_snapshot(self):
        brain = SnakeBrain(enable_debug=False)
        first = brain.observe_state(cube_payload())
        corrected = brain.observe_state(cube_payload(cells={(2, 7, 5): 'X'}))
        self.assertIsNot(first, corrected)
        self.assertIn((2, 7, 5), corrected.pickups)

    def test_wire_log_preserves_planes_head_and_redacts_turn_token(self):
        payload = cube_payload()
        logged = safe_message({'event': 'your_turn', 'data': payload})['data']
        self.assertEqual(logged['head'], payload['head'])
        self.assertEqual(logged['board_x'], payload['board_x'])
        self.assertEqual(logged['levels'], 20)
        self.assertNotIn('turn_token', logged)

    def test_verification_records_three_dimensional_head(self):
        payload = cube_payload(cells={(10, 8, 8): 'B'})
        state = SnakeBrain().observe_state(payload)
        record = verification_record(payload, observed=state)
        self.assertEqual(record['after']['heads']['A'], [10, 5, 5])
        self.assertTrue(record['after']['partial_observation'])
