"""Official star/invincibility contracts against the shared transition engine."""
from dataclasses import replace
import unittest

from snake_bot import snake_state as rules
from snake_bot.snake_state import GameState, apply_action, apply_move, parse_state


def state(**changes):
    source = GameState(
        rows=20, cols=20, board=(), side='A', enemy='B',
        snakes={'A': ((5, 5), (5, 4), (5, 3), (5, 2)),
                'B': ((5, 8), (5, 9), (5, 10), (5, 11))},
        food=frozenset(), scores={'A': 6000, 'B': 7000},
        remaining_moves=250, reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False, crashed_b=False)
    return replace(source, **changes)


class StarParsingTests(unittest.TestCase):
    def test_star_is_independent_unicode_cell_and_counter_is_authoritative(self):
        parsed = parse_state({'board': 'Aa ⭐ X1* bB', 'side': 'B',
                              'invincible_1': 0, 'invincible_2': 14})
        self.assertEqual(parsed.stars, frozenset({(0, 3)}))
        self.assertNotIn((0, 3), parsed.food | parsed.pickups | parsed.shot_food)
        self.assertEqual(parsed.head('A'), (0, 0))
        self.assertEqual(parsed.head('B'), (0, 9))
        self.assertEqual(parsed.invincibility('B'), 14)

    def test_inverted_heads_and_tracking_survive_refresh_and_expiry(self):
        for a, b in ((0, 0), (50, 0), (0, 50), (50, 50)):
            with self.subTest(a=a, b=b):
                line = ('aAA' if a else 'Aaa') + '  ' + ('bBB' if b else 'Bbb')
                previous = {'A': ((0, 0), (0, 1), (0, 2)),
                            'B': ((0, 5), (0, 6), (0, 7))}
                parsed = parse_state({'board': line, 'invincible_1': a,
                                      'invincible_2': b}, previous_snakes=previous)
                self.assertEqual(parsed.snakes, previous)
                self.assertEqual(parsed.reliable_tails, frozenset({'A', 'B'}))

    def test_parsed_frenzy_records_missing_stars_without_inventing_cells(self):
        parsed = parse_state({'board': 'A ⭐ B', 'remaining_moves': 100})
        self.assertTrue(parsed.star_frenzy)
        self.assertEqual(parsed.pending_star_spawns, 9)
        self.assertEqual(parsed.stars, frozenset({(0, 2)}))
        self.assertIsNone(parsed.star_spawn_due_round)

    def test_midgame_zero_counters_do_not_invent_normal_spawn_deadline(self):
        parsed = parse_state({'board': 'A   B', 'remaining_moves': 250})
        self.assertIsNone(parsed.star_spawn_due_round)
        self.assertEqual(parsed.pending_star_spawns, 0)


class StarConsumptionTests(unittest.TestCase):
    def test_star_grows_rewards_refreshes_and_never_advances_sequence(self):
        for counter, multiplier in ((0, 1), (3, 4), (50, 2)):
            with self.subTest(counter=counter, multiplier=multiplier):
                source = state(stars=frozenset({(5, 6), (10, 10)}),
                               invincible_a=counter, next_food_digit=4,
                               multipliers={'A': multiplier, 'B': 1})
                after = apply_move(source, 'right')
                self.assertEqual(after.scores['A'], 6000 + 100 * multiplier)
                self.assertEqual(after.body('A'), ((5, 6),) + source.body('A'))
                self.assertEqual(after.invincible_a, 50)
                self.assertEqual(after.next_food_digit, 4)
                self.assertEqual(after.stars, frozenset({(10, 10)}))
                self.assertTrue(after.last_action['star_consumed'])

    def test_own_turns_count_once_and_rival_turn_does_not(self):
        source = state(invincible_a=50, invincible_b=7)
        after = apply_move(source, 'down')
        self.assertEqual((after.invincible_a, after.invincible_b), (49, 7))
        after = apply_move(after, 'down')
        self.assertEqual((after.invincible_a, after.invincible_b), (49, 6))
        final = apply_move(replace(source, invincible_a=1), 'down')
        self.assertEqual(final.invincible_a, 0)

    def test_shoot_invalid_and_timeout_consume_protection(self):
        for kind, direction, score in (('shoot', 'down', 1000),
                                       ('invalid', None, 6000),
                                       ('move', 'bad', 6000),
                                       ('timeout', None, 6000)):
            with self.subTest(kind=kind, direction=direction):
                after = apply_action(state(invincible_a=2), kind, direction)
                self.assertEqual(after.invincible_a, 1)
                self.assertEqual(after.scores['A'], score)
                self.assertEqual(after.remaining_moves, 249)
                self.assertEqual(after.head('A'), (5, 5))
                self.assertFalse(after.crashed_a)

    def test_normal_invalid_penalty_and_tenth_timeout_defeat_remain(self):
        after = apply_action(state(), 'invalid', None)
        self.assertEqual(after.scores['A'], 5500)
        after = apply_action(state(invincible_a=2, timeouts_a=9), 'timeout', None)
        self.assertEqual(after.timeouts_a, 10)
        self.assertTrue(after.is_terminal())
        self.assertEqual(after.terminal_winner(), 'B')
        self.assertEqual(after.scores['A'], 6000)

    def test_all_fifty_own_turns_are_protected_and_next_collision_is_normal(self):
        source = state(invincible_a=50, walls=frozenset({(5, 6)}))
        for remaining in range(49, -1, -1):
            source = apply_move(source, 'right', side='A')
            self.assertEqual(source.invincible_a, remaining)
            self.assertEqual(source.scores['A'], 6000)
            self.assertFalse(source.crashed_a)
        source = apply_move(source, 'right', side='A')
        self.assertEqual(source.scores['A'], 0)
        self.assertTrue(source.crashed_a)

    def test_unaffordable_invincible_shoot_is_invalid_and_free_of_penalty(self):
        source = state(invincible_a=1, scores={'A': 4999, 'B': 7000})
        after = apply_action(source, 'shoot', 'right')
        self.assertEqual(after.scores['A'], 4999)
        self.assertFalse(after.last_action['valid'])
        self.assertFalse(after.last_action['hit'])
        self.assertEqual(after.invincible_a, 0)


class InvincibleCollisionTests(unittest.TestCase):
    def test_protected_own_tail_collision_is_stationary_instead_of_vacating(self):
        body = ((5, 5), (5, 4), (6, 4), (6, 5))
        source = state(invincible_a=1, snakes={'A': body, 'B': ((15, 15),)})
        after = apply_move(source, 'down')
        self.assertEqual(after.body('A'), body)
        self.assertEqual(after.scores['A'], 6000)
        normal = apply_move(replace(source, invincible_a=0), 'down')
        self.assertEqual(normal.head('A'), (6, 5))
        self.assertEqual(normal.scores['A'], 6001)

    def test_border_wall_and_own_body_are_stationary_without_crash(self):
        cases = (dict(snakes={'A': ((0, 5), (0, 4), (0, 3), (0, 2)),
                             'B': ((5, 8),)}, direction='up'),
                 dict(walls=frozenset({(5, 6)}), direction='right'),
                 dict(direction='left'))
        for case in cases:
            direction = case.pop('direction')
            with self.subTest(direction=direction, case=case):
                source = state(invincible_a=1, **case)
                after = apply_move(source, direction)
                self.assertEqual(after.snakes, source.snakes)
                self.assertEqual(after.scores, source.scores)
                self.assertFalse(after.crashed_a)
                self.assertFalse(after.tail_food_a | after.tail_food_b)
                self.assertEqual(after.invincible_a, 0)
                self.assertIn(direction, rules.available_moves(source))

    def test_bomb_still_detonates_but_preserves_score_and_body(self):
        for side in ('A', 'B'):
            kwargs = {f'invincible_{side.lower()}': 1}
            source = state(side=side, enemy='B' if side == 'A' else 'A',
                           snakes={'A': ((5, 5), (5, 4)), 'B': ((15, 5), (15, 4))},
                           bombs_small=frozenset({(5 if side == 'A' else 15, 7)}), **kwargs)
            after = apply_move(source, 'right')
            self.assertEqual(after.scores, source.scores)
            self.assertEqual(len(after.body(side)), len(source.body(side)))
            self.assertFalse(after.bombs_small)
            self.assertEqual(after.pending_bomb_respawns, ('small',))
            self.assertTrue(after.last_action['bomb_triggered'])
            self.assertFalse(after.crashed_a or after.crashed_b)

    def test_wrong_digit_grows_is_consumed_and_keeps_sequence(self):
        source = state(invincible_a=1, food_values={(5, 6): 4, (15, 15): 2},
                       food=frozenset({(15, 15)}), next_food_digit=2,
                       food_copy_targets={2: 1, 4: 1})
        after = apply_move(source, 'right')
        self.assertEqual(after.scores['A'], source.scores['A'])
        self.assertEqual(after.body('A'), ((5, 6),) + source.body('A'))
        self.assertNotIn((5, 6), after.food_values)
        self.assertEqual(after.next_food_digit, 2)
        self.assertEqual(after.food_copy_targets[4], 1)

    def test_ram_cuts_normal_or_invincible_rival_for_free_and_filters_blasts(self):
        for rival_protection in (0, 12):
            for hit in ('head', 'body'):
                with self.subTest(protected=rival_protection, hit=hit):
                    rival = ((5, 6), (5, 7), (5, 8), (5, 9)) if hit == 'head' else (
                        (6, 6), (5, 6), (5, 7), (5, 8))
                    source = state(invincible_a=1, invincible_b=rival_protection,
                                   snakes={'A': ((5, 5), (5, 4)), 'B': rival},
                                   bombs_small=frozenset({(6, 8)}))
                    after = apply_move(source, 'right')
                    self.assertEqual(after.body('A'), source.body('A'))
                    self.assertEqual(after.body('B'), rival[:1])
                    self.assertEqual(after.scores, source.scores)
                    self.assertEqual(after.shot_food, frozenset({(5, 6)}) if hit == 'body' else frozenset())
                    self.assertTrue(after.last_action['ram_cut'])
                    self.assertEqual(after.last_action['ram_hit_coordinate'], (5, 6))
                    self.assertEqual(after.last_action['cut_length'], 3)
                    self.assertEqual(after.last_action['shoot_cost'], 0)
                    self.assertFalse(after.crashed_a or after.crashed_b)

    def test_normal_attacker_crashes_even_when_rival_is_invincible(self):
        source = state(invincible_b=50,
                       snakes={'A': ((5, 5), (5, 4), (5, 3), (5, 2)),
                               'B': ((5, 6), (5, 7))})
        after = apply_move(source, 'right')
        self.assertEqual(after.scores['A'], 0)
        self.assertTrue(after.crashed_a)
        self.assertEqual(after.body('B'), source.body('B'))

    def test_shoot_cannot_cut_invincible_head_or_body_and_still_costs(self):
        for shooter_protection in (0, 4):
            for rival in (((5, 8), (5, 9)), ((6, 8), (5, 8), (5, 9))):
                source = state(invincible_a=shooter_protection, invincible_b=1,
                               snakes={'A': ((5, 5),), 'B': rival})
                after = apply_action(source, 'shoot', 'right')
                self.assertEqual(after.scores['A'], 1000)
                self.assertEqual(after.body('B'), rival)
                self.assertFalse(after.shot_food)
                self.assertTrue(after.last_action['shot_blocked_by_invincibility'])
                self.assertEqual(after.last_action['cut_length'], 0)
                self.assertEqual(after.invincible_b, 1)


class StarSpawnTests(unittest.TestCase):
    def test_initial_star_becomes_pending_after_both_twentieth_turns(self):
        source = state(remaining_moves=362)
        after_a = apply_move(source, 'down')
        self.assertEqual(after_a.pending_star_spawns, 0)
        after_b = apply_move(after_a, 'down')
        self.assertEqual(after_b.remaining_moves, 360)
        self.assertEqual(after_b.pending_star_spawns, 1)
        self.assertFalse(after_b.stars)

    def test_normal_star_pauses_during_protection_and_waits_twenty_rounds(self):
        source = state(remaining_moves=250, invincible_a=1, star_spawn_due_round=None)
        after = apply_move(source, 'down')
        self.assertEqual(after.star_spawn_due_round, 95.5)
        self.assertEqual(after.pending_star_spawns, 0)
        before_due = replace(after, remaining_moves=212, side='A', enemy='B')
        after = apply_move(apply_move(before_due, 'down'), 'down')
        self.assertEqual(after.remaining_moves, 210)
        self.assertEqual(after.pending_star_spawns, 0)
        self.assertEqual(apply_move(after, 'down').pending_star_spawns, 1)

    def test_respawn_waits_exactly_forty_actions_after_either_player_expires(self):
        for side, remaining in (('A', 250), ('B', 249)):
            kwargs = {f'invincible_{side.lower()}': 1}
            source = state(remaining_moves=remaining, side=side,
                           enemy='B' if side == 'A' else 'A',
                           scores={'A': 100000, 'B': 100000}, **kwargs)
            after = apply_move(source, 'down')
            expired_at = after.remaining_moves
            for elapsed in range(1, 41):
                # Isolate clock behavior from the choice of future moves.
                after = apply_action(after, 'invalid', None)
                self.assertEqual(after.pending_star_spawns, int(elapsed == 40))
                self.assertEqual(after.remaining_moves, expired_at - elapsed)

    def test_spawn_candidates_are_tied_nearest_equidistant_and_not_in_blast(self):
        source = state(snakes={'A': ((5, 3),), 'B': ((5, 7),)},
                       bombs_small=frozenset({(5, 5)}))
        candidates = rules.star_spawn_candidates(source)
        self.assertEqual(set(candidates), {(3, 5), (7, 5)})
        for cell in candidates:
            self.assertEqual(rules.manhattan(cell, source.head('A')),
                             rules.manhattan(cell, source.head('B')))

    def test_no_normal_star_candidates_or_pending_while_invincible(self):
        source = state(invincible_a=3, pending_star_spawns=1)
        after = apply_move(source, 'down')
        self.assertEqual(after.pending_star_spawns, 0)
        self.assertEqual(rules.star_spawn_candidates(after), ())

    def test_frenzy_boundary_and_continuous_unknown_replacement(self):
        source = state(remaining_moves=101, stars=frozenset({(5, 6)}))
        self.assertFalse(source.star_frenzy)
        after = apply_move(source, 'right')
        self.assertTrue(after.star_frenzy)
        self.assertEqual(after.pending_star_spawns, 10)
        self.assertFalse(after.bombs_large | after.bombs_small)
        stars = frozenset({(5, 7)} | {(10, i) for i in range(9)})
        source = replace(after, side='A', enemy='B', stars=stars,
                         invincible_b=5, pending_star_spawns=0)
        after = apply_move(source, 'right')
        self.assertEqual((after.invincible_a, after.invincible_b), (50, 5))
        self.assertEqual(len(after.stars), 9)
        self.assertEqual(after.pending_star_spawns, 1)
        self.assertTrue(rules.star_spawn_candidates(after))

    def test_normal_star_candidates_never_overlap_food_or_exceed_one(self):
        source = state(snakes={'A': ((5, 3),), 'B': ((5, 7),)},
                       food_values={(5, 5): 4}, pickups=frozenset({(4, 5)}))
        self.assertEqual(rules.star_spawn_candidates(source), ((6, 5),))
        self.assertFalse(rules.star_spawn_candidates(replace(source, stars=frozenset({(6, 5)}))))
        odd_parity = replace(source, snakes={'A': ((5, 3),), 'B': ((5, 6),)})
        self.assertEqual(rules.star_spawn_candidates(odd_parity), ())


class InvinciblePathTests(unittest.TestCase):
    def test_wrong_digit_paths_only_cross_during_remaining_protection(self):
        source = state(rows=1, cols=6, snakes={'A': ((0, 0),), 'B': ((0, 5),)},
                       invincible_a=1, food_values={(0, 2): 4}, next_food_digit=2)
        self.assertIsNone(rules.shortest_path(source, (0, 0), (0, 3), 'A'))
        self.assertIsNone(rules.temporal_shortest_path(source, (0, 0), (0, 3), 'A'))
        self.assertNotIn((0, 3), rules.temporal_distance_map(source, (0, 0), 'A'))
        protected = replace(source, invincible_a=2)
        self.assertEqual(len(rules.shortest_path(protected, (0, 0), (0, 3), 'A')), 4)
        self.assertEqual(len(rules.temporal_shortest_path(protected, (0, 0), (0, 3), 'A')), 4)
        self.assertEqual(rules.temporal_distance_map(protected, (0, 0), 'A')[(0, 3)], 3)

    def test_blast_avoidance_obeys_own_turn_protection(self):
        source = state(invincible_a=1, bombs_small=frozenset({(5, 7)}))
        self.assertFalse(rules.path_blast_blocked(source, (5, 6), 'A', 1))
        self.assertTrue(rules.path_blast_blocked(source, (5, 6), 'A', 2))


if __name__ == '__main__':
    unittest.main()
