"""Check the laboratory's complete cube and player-specific observations."""

from collections import Counter
from dataclasses import replace
import random
import unittest

from tools import bot_lab
from snake_bot.snake_brain import SnakeBrain


class CubeLabTests(unittest.TestCase):
    def test_cube_initial_resources(self):
        state = bot_lab.initial_state(11, levels=20)
        self.assertEqual(state.head('A')[0], 10)
        self.assertEqual((state.levels, state.rows, state.cols), (20, 20, 20))
        self.assertEqual(len(state.pickups), 6)
        self.assertEqual((len(state.bombs_large), len(state.bombs_small)), (6, 15))
        self.assertEqual(set(state.food_values.values()), set(range(1, 6)))
        self.assertTrue(
            all(10 <= count <= 15 for count in Counter(state.food_values.values()).values())
        )

    def test_player_receives_only_three_planes(self):
        state = bot_lab.initial_state(12, levels=20)
        head = state.head('A')
        visible = (head[0], 0, 0)
        hidden = ((head[0] + 1) % 20, 0, 10)
        state = replace(state, food_values={visible: 1, hidden: 2}, food=frozenset({visible}))
        data = bot_lab.make_turn_data(state, 'cube', 0)
        self.assertEqual(data['board_size'], '20x20x20')
        self.assertEqual(data['head'], dict(zip(('level', 'row', 'col'), head)))
        self.assertEqual(data['board'], '\n\n'.join(data[f'board_{axis}'] for axis in 'xyz'))
        self.assertIn('1', data['board_z'])
        self.assertNotIn('2', data['board'])

    def test_plate_each_orientation_and_shrink(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                plate = bot_lab.plate_cells((10, 10, 10), 11, axis)
                self.assertEqual(len(plate), 121)
                self.assertEqual(len({cell[axis] for cell in plate}), 1)
                self.assertEqual(len(bot_lab.shrink_plate(plate)), 81)

    def test_frenzy_replenishes_thirty_stars(self):
        state = bot_lab.initial_state(13, levels=20, max_turns=100)
        self.assertEqual(len(state.stars), 30)
        state = replace(state, stars=frozenset(tuple(state.stars)[1:]))
        state = bot_lab.respawn_stars(state, random.Random(13))
        self.assertEqual(len(state.stars), 30)

    def test_simulator_does_not_inject_hidden_rival_truth(self):
        state = bot_lab.initial_state(15, levels=20)
        snakes = dict(state.snakes, B=((11, 11, 11), (11, 11, 12), (11, 11, 13)))
        state = replace(state, snakes=snakes)
        brain = SnakeBrain(enable_debug=False)
        data = bot_lab.make_turn_data(state, 'hidden-rival', 0)
        bot_lab.observe_simulation_state(brain, state, data)
        observed = brain.current_states['hidden-rival']
        self.assertIsNone(observed.head('B'))
        self.assertNotIn((11, 11, 11), observed.occupied())


if __name__ == '__main__':
    unittest.main()
