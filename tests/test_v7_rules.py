"""Official v7 transition contracts, independent of strategic search."""
from dataclasses import replace
import unittest

from snake_bot import game_rules as rules
from snake_bot.snake_state import GameState, apply_move, parse_state, track_snake, legal_moves
from snake_bot.search_engine import IterativeSearchEngine


def fixture(side='A', **changes):
    state = GameState(
        rows=7, cols=7, board=(), side=side, enemy='B' if side == 'A' else 'A',
        snakes={'A': ((2, 2), (2, 1), (2, 0)),
                'B': ((5, 4), (5, 5), (5, 6))},
        food=frozenset(), scores={'A': 10, 'B': 20}, remaining_moves=400,
        reliable_tails=frozenset({'A', 'B'}), crashed_a=False, crashed_b=False)
    return replace(state, **changes)


class V7RulesTests(unittest.TestCase):
    def transition(self, state, direction):
        engine, compact = IterativeSearchEngine.from_game_state(state)
        after = apply_move(state, direction)
        actual = engine.apply_move(compact, direction)
        _, expected = IterativeSearchEngine.from_game_state(after)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.terminal_winner(), after.terminal_winner())
        self.assertEqual(actual.is_terminal(), after.is_terminal())
        return after

    def test_official_constants_and_default_horizon(self):
        self.assertEqual((rules.MAX_MOVES, rules.CRASH_KEEP_LENGTH, rules.CRASH_PENALTY,
                          rules.ELIMINATION_SCORE_THRESHOLD, rules.TAIL_FOOD_BASE_REWARD),
                         (400, 3, 500, -2500, 100))
        self.assertEqual(parse_state({'board': 'A B'}).remaining_moves, 400)
        self.assertEqual(parse_state({'board': 'A B', 'remaining_moves': 0}).remaining_moves, 0)

    def test_parse_both_tail_symbols_and_coexistence(self):
        for side in ('A', 'B'):
            state = parse_state({'board': 'A\u24b6\u24b7X#12B', 'side': side})
            self.assertEqual(state.tail_food_a, frozenset({(0, 1)}))
            self.assertEqual(state.tail_food_b, frozenset({(0, 2)}))
            self.assertEqual(state.food_values, {(0, 5): 1, (0, 6): 2})
            self.assertEqual(state.pickups, frozenset({(0, 3)}))
            self.assertEqual(state.walls, frozenset({(0, 4)}))
            engine, compact = IterativeSearchEngine.from_game_state(state)
            self.assertEqual(compact.tail_food_a, (engine.index((0, 1)),))
            self.assertEqual(compact.tail_food_b, (engine.index((0, 2)),))

    def test_missing_history_is_unknown_not_first_crash(self):
        state = parse_state({'board': 'A B', 'score_1': 100})
        self.assertIsNone(state.crashed_a)
        self.assertIsNone(state.crashed_b)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        with self.assertRaises(ValueError):
            apply_move(state, 'up')
        with self.assertRaises(ValueError):
            engine.apply_move(compact, 'up')

    def test_explicit_history_and_unknown_nonpositive_crash(self):
        for history in ({'A': False, 'B': True}, {'A': True, 'B': None}):
            state = parse_state({'board': 'A B'}, crash_history=history)
            self.assertIs(state.crashed_a, history['A'])
            self.assertIs(state.crashed_b, history['B'])
        for score in (0, -100):
            after = self.transition(fixture(scores={'A': score, 'B': 20}, crashed_a=None), 'left')
            self.assertEqual(after.scores['A'], score - 500)
            self.assertTrue(after.crashed_a)

    def crash_case(self, kind, side='A', length=5, score=100, crashed=False):
        body = ((2, 2), (2, 1), (1, 1), (1, 2), (1, 3))[:length]
        enemy = 'B' if side == 'A' else 'A'
        rival = ((5, 4), (5, 5), (5, 6))
        direction, walls = 'up', frozenset()
        if kind == 'border':
            body = tuple((r - 2, c) for r, c in ((2, 2), (3, 2), (3, 1), (4, 1), (4, 2))[:length])
        elif kind == 'rival':
            body = ((2, 2), (2, 1), (2, 0), (3, 0), (3, 1))[:length]
            rival = ((1, 2), (1, 3), (1, 4))
        elif kind == 'wall':
            body = ((2, 2), (2, 1), (2, 0), (3, 0), (3, 1))[:length]
            walls = frozenset({(1, 2), (0, 6)})
        state = fixture(side, snakes={side: body, enemy: rival}, walls=walls,
                        scores={side: score, enemy: 37},
                        **({'crashed_a': crashed} if side == 'A' else {'crashed_b': crashed}))
        after = self.transition(state, direction)
        self.assertEqual(after.body(side), body[:3])
        tail = after.tail_food_b if side == 'A' else after.tail_food_a
        self.assertEqual(tail, frozenset(body[3:]))
        self.assertEqual(after.scores[enemy], 37)
        self.assertEqual(after.remaining_moves, 399)
        self.assertEqual(after.side, enemy)
        self.assertTrue(after.crashed_a if side == 'A' else after.crashed_b)
        engine, before = IterativeSearchEngine.from_game_state(state)
        compact_after = engine.apply_move(before, direction)
        index = 0 if side == 'A' else 1
        self.assertEqual(compact_after.bodies[index], tuple(engine.index(cell) for cell in body[:3]))
        self.assertEqual(getattr(compact_after, 'tail_food_b' if side == 'A' else 'tail_food_a'),
                         tuple(sorted(engine.index(cell) for cell in body[3:])))
        self.assertEqual(compact_after.scores[1 - index], 37)
        self.assertEqual(compact_after.scores[index], 0 if score > 0 and not crashed else score - 500)
        self.assertTrue(compact_after.crashed_a if side == 'A' else compact_after.crashed_b)
        return after

    def test_border_crash_both_players(self):
        for side in ('A', 'B'):
            self.crash_case('border', side)

    def test_own_body_crash_both_players(self):
        for side in ('A', 'B'):
            self.crash_case('self', side)

    def test_rival_crash_both_players(self):
        for side in ('A', 'B'):
            self.crash_case('rival', side)

    def test_wall_crash_both_players(self):
        for side in ('A', 'B'):
            self.crash_case('wall', side)

    def test_exactly_three_and_shorter_have_no_detached_food(self):
        for length in (1, 2, 3):
            for side in ('A', 'B'):
                self.crash_case('border', side, length)

    def test_detached_tail_preserves_existing_food(self):
        for side in ('A', 'B'):
            state = fixture(side, tail_food_a=frozenset({(0, 0)}), tail_food_b=frozenset({(0, 1)}),
                            snakes={side: ((2, 2), (2, 1), (1, 1), (1, 2), (1, 3)),
                                    ('B' if side == 'A' else 'A'): ((5, 5),)})
            after = self.transition(state, 'up')
            self.assertEqual(after.tail_food_a, state.tail_food_a | ({(1, 2), (1, 3)} if side == 'B' else set()))
            self.assertEqual(after.tail_food_b, state.tail_food_b | ({(1, 2), (1, 3)} if side == 'A' else set()))

    def test_first_positive_crash_resets_to_zero(self):
        for score in (1, 1700):
            self.assertEqual(self.crash_case('wall', score=score).scores['A'], 0)

    def test_first_zero_crash_costs_500(self):
        for side in ('A', 'B'):
            self.assertEqual(self.crash_case('wall', side, score=0).scores[side], -500)

    def test_first_negative_crash_costs_500(self):
        for score in (-1, -100):
            self.assertEqual(self.crash_case('border', score=score).scores['A'], score - 500)

    def test_later_crash_always_costs_500(self):
        for score in (1200, 0, -100):
            self.assertEqual(self.crash_case('border', score=score, crashed=True).scores['A'], score - 500)

    def test_elimination_is_strict_and_attributed_to_loser(self):
        for side in ('A', 'B'):
            for score, terminal in ((-2000, False), (-2001, True)):
                after = self.crash_case('wall', side, score=score, crashed=True)
                self.assertEqual(after.is_terminal(), terminal)
                self.assertEqual(after.terminal_winner(), after.side if terminal else None)

    def test_wrong_number_can_also_eliminate(self):
        for score, terminal in ((-2000, False), (-2001, True)):
            after = self.transition(fixture(scores={'A': score, 'B': 0}, food_values={(2, 3): 2},
                                            next_food_digit=1), 'right')
            self.assertEqual(after.is_terminal(), terminal)

    def test_no_actions_after_terminal_or_move_limit(self):
        for state in (fixture(scores={'A': -2501, 'B': 0}), fixture(remaining_moves=0)):
            engine, compact = IterativeSearchEngine.from_game_state(state)
            with self.assertRaises(ValueError):
                apply_move(state, 'right')
            with self.assertRaises(ValueError):
                engine.apply_move(compact, 'right')

    def test_last_turn_ends_by_score_or_draw(self):
        for score, winner in ((20, 'A'), (19, None), (18, 'B')):
            state = fixture(scores={'A': score, 'B': 20}, remaining_moves=1)
            after = self.transition(state, 'right')
            self.assertEqual(after.remaining_moves, 0)
            self.assertTrue(after.is_terminal())
            self.assertEqual(after.terminal_winner(), winner)

    def test_consumers_preserve_v7_fields_in_copies_and_cache_keys(self):
        from snake_bot.food_planner import simulate_path_to_food
        from snake_bot.move_search import state_key
        from tools.bot_lab import spawn_food
        from random import Random
        state = fixture(tail_food_a=frozenset({(0, 0)}), tail_food_b=frozenset({(0, 1)}),
                        crashed_a=True, crashed_b=None)
        for copied in (simulate_path_to_food(state, 'A', ((2, 2), (2, 3))),
                       spawn_food(state, Random(7), 1)):
            for field in ('tail_food_a', 'tail_food_b', 'crashed_a', 'crashed_b'):
                self.assertEqual(getattr(copied, field), getattr(state, field))
        variants = (state, replace(state, crashed_a=False), replace(state, crashed_b=False),
                    replace(state, tail_food_a=frozenset()),
                    replace(state, tail_food_b=frozenset()),
                    replace(state, scores={'A': 99, 'B': 20}))
        self.assertEqual(len({state_key(item) for item in variants}), len(variants))

    def test_beneficiary_tail_reward_growth_and_single_cell_removal(self):
        for side in ('A', 'B'):
            for multiplier in (1, 3):
                target = (2, 3) if side == 'A' else (5, 3)
                field = 'tail_food_a' if side == 'A' else 'tail_food_b'
                enemy = 'B' if side == 'A' else 'A'
                state = fixture(side, multipliers={side: multiplier, enemy: multiplier + 4},
                                **{field: frozenset({target, (0, 0)})})
                after = self.transition(state, 'right' if side == 'A' else 'left')
                self.assertEqual(after.scores[side], state.scores[side] + 100 * multiplier)
                self.assertEqual(after.body(side), (target,) + state.body(side))
                self.assertEqual(getattr(after, field), frozenset({(0, 0)}))
                engine, compact = IterativeSearchEngine.from_game_state(state)
                actual = engine.apply_move(compact, 'right' if side == 'A' else 'left')
                index = 0 if side == 'A' else 1
                self.assertEqual(actual.scores[index], state.scores[side] + 100 * multiplier)
                self.assertEqual(actual.bodies[index], (engine.index(target),) + compact.bodies[index])
                self.assertEqual(getattr(actual, field), (engine.index((0, 0)),))

    def test_originator_clears_only_touched_cell_with_normal_score(self):
        for side in ('A', 'B'):
            target = (2, 3) if side == 'A' else (5, 3)
            field = 'tail_food_b' if side == 'A' else 'tail_food_a'
            state = fixture(side, **{field: frozenset({target, (0, 0)})})
            after = self.transition(state, 'right' if side == 'A' else 'left')
            self.assertEqual(after.scores[side], state.scores[side] + 1)
            self.assertEqual(len(after.body(side)), len(state.body(side)))
            self.assertEqual(getattr(after, field), frozenset({(0, 0)}))

    def test_tail_food_does_not_change_numeric_sequence_or_groups(self):
        for next_digit in (1, 9):
            state = fixture(tail_food_a=frozenset({(2, 3)}), next_food_digit=next_digit,
                            food=frozenset({(0, 0)}), food_values={(0, 0): next_digit})
            after = self.transition(state, 'right')
            self.assertEqual(after.next_food_digit, next_digit)
            self.assertEqual(after.food_values, state.food_values)
            self.assertEqual(after.food, state.food)

    def test_number_and_x_preserve_both_tail_food_sets(self):
        for pickup in (False, True):
            state = fixture(tail_food_a=frozenset({(0, 0)}), tail_food_b=frozenset({(0, 1)}),
                            food=frozenset({(2, 3)}) if not pickup else frozenset({(0, 2)}),
                            food_values={(2, 3): 1, (0, 2): 2} if not pickup else {(0, 2): 1},
                            next_food_digit=1, pickups=frozenset({(2, 3)}) if pickup else frozenset(),
                            walls=frozenset({(6, 0)}))
            after = self.transition(state, 'right')
            self.assertEqual(after.tail_food_a, state.tail_food_a)
            self.assertEqual(after.tail_food_b, state.tail_food_b)
            self.assertEqual(after.walls, state.walls)
            self.assertEqual(after.next_food_digit, 1 if pickup else 2)

    def test_releasing_own_tail_and_existing_candidate_filter(self):
        state = fixture(snakes={'A': ((2, 2), (2, 1), (1, 1), (1, 2)), 'B': ((5, 5),)})
        after = self.transition(state, 'up')
        self.assertFalse(after.crashed_a)
        self.assertEqual(after.head('A'), (1, 2))
        self.assertNotIn('left', legal_moves(state))

    def test_exact_tail_release_does_not_depend_on_tracking_confidence(self):
        for side in ('A', 'B'):
            for reliable in (False, True):
                with self.subTest(side=side, reliable=reliable):
                    enemy = 'B' if side == 'A' else 'A'
                    body = ((2, 2), (2, 1), (1, 1), (1, 2))
                    state = fixture(side, snakes={side: body, enemy: ((5, 5),)},
                                    reliable_tails=frozenset({side}) if reliable else frozenset())
                    engine, compact = IterativeSearchEngine.from_game_state(state)
                    self.assertEqual('up' in legal_moves(state), reliable)
                    self.assertEqual('up' in engine.legal_moves(compact), reliable)
                    after = self.transition(state, 'up')
                    expected = ((1, 2), (2, 2), (2, 1), (1, 1))
                    self.assertEqual(after.body(side), expected)
                    self.assertEqual(after.scores[side], state.scores[side] + 1)
                    self.assertFalse(after.crashed_a if side == 'A' else after.crashed_b)
                    self.assertEqual(after.tail_food_a, frozenset())
                    self.assertEqual(after.tail_food_b, frozenset())
                    actual = engine.apply_move(compact, 'up')
                    index = 0 if side == 'A' else 1
                    self.assertEqual(actual.bodies[index], tuple(engine.index(cell) for cell in expected))
                    self.assertEqual(actual.scores[index], state.scores[side] + 1)
                    self.assertFalse(actual.crashed_a if side == 'A' else actual.crashed_b)

    def test_compact_hash_and_side_copy_preserve_history_and_tail_identity(self):
        engine, compact = IterativeSearchEngine.from_game_state(fixture(
            tail_food_a=frozenset({(0, 0)}), crashed_a=True, crashed_b=None))
        variants = (compact, replace(compact, crashed_a=False), replace(compact, crashed_a=None),
                    replace(compact, tail_food_a=(), tail_food_b=compact.tail_food_a))
        self.assertEqual(len(set(variants)), 4)
        captured = []
        original = engine.legal_moves
        engine.legal_moves = lambda state: captured.append(state) or original(state)
        engine.legal_moves_for(compact, 1)
        self.assertEqual(captured[-1], replace(compact, side=1))

    def test_tracking_recognizes_truncation_without_reordering_prefix(self):
        previous = ((1, 2), (1, 1), (0, 1), (0, 2), (0, 3))
        for side in ('A', 'B'):
            board = (' ' + side.lower() + '  ', ' ' + side.lower() + side + ' ')
            self.assertEqual(track_snake(previous, board, side), previous[:3])

    def test_pure_rewards_and_explicit_crash_history(self):
        for mult in (1, 2, 4):
            self.assertEqual(rules.tail_food_reward(mult), 100 * mult)
        for score, seen, expected in ((10, False, 0), (10, True, -490), (0, False, -500),
                                      (-10, None, -510)):
            self.assertEqual(rules.score_after_crash(score, seen), expected)
            self.assertEqual(rules.crash_scores(score, 123, seen), (expected, 123))
        with self.assertRaises(ValueError):
            rules.score_after_crash(10, None)


if __name__ == '__main__':
    unittest.main()
