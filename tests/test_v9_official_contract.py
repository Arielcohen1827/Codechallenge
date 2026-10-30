"""Remaining explicit v9 contracts, independent of strategic preferences."""
from dataclasses import replace
import unittest

from snake_bot.move_search import TurnSearchCache, state_key
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import GameState, apply_action


def shooting_state(shooter=((8, 2), (9, 2), (9, 3), (8, 3)),
                   rival=((6, 14), (7, 14), (8, 14), (8, 13),
                          (8, 12), (9, 12), (10, 12)), side='A', **changes):
    enemy = 'B' if side == 'A' else 'A'
    source = GameState(
        rows=20, cols=30, board=(), side=side, enemy=enemy,
        snakes={side: shooter, enemy: rival}, food=frozenset(),
        scores={side: 6000, enemy: 123}, remaining_moves=200,
        reliable_tails=frozenset({'A', 'B'}), crashed_a=None, crashed_b=True)
    return replace(source, **changes)


class OfficialShootingContracts(unittest.TestCase):
    def test_ray_crosses_existing_shot_and_tail_food_without_consuming_or_detonating(self):
        for wall in (False, True):
            with self.subTest(wall=wall):
                source = shooting_state(
                    food=frozenset({(8, 4)}), food_values={(8, 4): 1},
                    next_food_digit=1, pickups=frozenset({(8, 5)}),
                    shot_food=frozenset({(8, 6)}), tail_food_a=frozenset({(8, 7)}),
                    bombs_small=frozenset({(8, 10)}),
                    walls=frozenset({(8, 9)}) if wall else frozenset())
                after = apply_action(source, 'shoot', 'right')
                generated = frozenset() if wall else frozenset({(8, 12), (9, 12), (10, 12)})
                self.assertEqual(after.scores, {'A': 1000, 'B': 123})
                self.assertEqual(after.body('A'), source.body('A'))
                self.assertEqual(after.body('B'), source.body('B') if wall else source.body('B')[:4])
                self.assertEqual(after.shot_food, source.shot_food | generated)
                self.assertEqual(after.last_action['shot_food_generated'], sorted(generated))
                self.assertEqual(after.last_action['hit_coordinate'], None if wall else (8, 12))
                self.assertEqual(after.last_action['hit'], not wall)
                self.assertEqual(after.last_action['cut_length'], 0 if wall else 3)
                self.assertEqual(after.food_values, source.food_values)
                self.assertEqual(after.next_food_digit, 1)
                self.assertEqual(after.pickups, source.pickups)
                self.assertEqual(after.tail_food_a, source.tail_food_a)
                self.assertEqual(after.bombs_small, source.bombs_small)
                self.assertFalse(after.pending_bomb_respawns)
                self.assertFalse(after.last_action['bomb_triggered'])
                self.assertEqual((after.crashed_a, after.crashed_b), (None, True))

    def test_impacted_tail_cell_is_cut_but_a_head_is_retained_even_when_alone(self):
        rivals = (((6, 4), (7, 4), (8, 4)),
                  ((8, 6), (8, 7), (8, 8)),
                  ((8, 6),))
        cache = TurnSearchCache()
        for rival in rivals:
            with self.subTest(rival=rival):
                source = shooting_state(rival=rival)
                after = cache.apply_action(source, 'A', 'shoot', 'right')
                hit_index = 2 if rival[0] == (6, 4) else 0
                keep = max(1, hit_index)
                self.assertTrue(after.last_action['hit'])
                self.assertEqual(after.body('B'), rival[:keep])
                self.assertEqual(after.shot_food, frozenset(rival[keep:]))
                self.assertEqual(after.last_action['cut_length'], len(rival) - keep)
                self.assertEqual(after.scores, {'A': 1000, 'B': 123})
                self.assertEqual(after.body('A'), source.body('A'))
                self.assertEqual((after.crashed_a, after.crashed_b), (None, True))
                if hit_index:
                    self.assertIn(rival[hit_index], after.shot_food)
                else:
                    self.assertNotIn(rival[0], after.shot_food)
        same = shooting_state(rival=rivals[1])
        shorter = replace(same, snakes={**same.snakes, 'B': same.body('B')[:-1]})
        self.assertNotEqual(state_key(same), state_key(shorter))
        self.assertNotEqual(cache.apply_action(same, 'A', 'shoot', 'right').shot_food,
                            cache.apply_action(shorter, 'A', 'shoot', 'right').shot_food)

    def test_cut_food_uses_current_blast_zones_and_matches_compact_for_both_sides(self):
        rival = ((7, 6), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10))
        for side in ('A', 'B'):
            for remaining in (101, 100):
                with self.subTest(side=side, remaining=remaining):
                    source = shooting_state(side=side, rival=rival,
                        remaining_moves=remaining, bombs_small=frozenset({(9, 8)}))
                    after = apply_action(source, 'shoot', 'right')
                    expected_food = frozenset({(8, 6), (8, 10)}) if remaining == 101 else frozenset(rival[1:])
                    self.assertEqual(after.shot_food, expected_food)
                    self.assertEqual(after.body(source.enemy), rival[:1])
                    self.assertEqual(after.remaining_moves, remaining - 1)
                    self.assertEqual(after.scores, {side: 1000, source.enemy: 123})
                    self.assertFalse(after.bombs_small)
                    self.assertEqual((after.crashed_a, after.crashed_b), (None, True))
                    engine, compact = IterativeSearchEngine.from_game_state(source)
                    self.assertEqual(engine.apply_action(compact, 'shoot', 'right'),
                                     engine.to_compact(after))


if __name__ == '__main__':
    unittest.main()
