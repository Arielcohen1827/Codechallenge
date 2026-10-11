"""Behavioral contracts for game rules, independent of strategy and search."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from tools import bot_lab
from snake_bot.game_rules import (
    MULTIPLIER_PICKUP_SCORE,
    WRONG_FOOD_PENALTY,
    crash_scores,
    cyclic_digit,
    find_next_food_digit,
    food_points,
)
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import GameState, apply_move, legal_moves, parse_state, step


def make_state(**changes):
    state = GameState(
        rows=7, cols=7, board=(), side='A', enemy='B',
        snakes={'A': ((2, 2), (2, 1), (2, 0)),
                'B': ((5, 4), (5, 5), (5, 6))},
        food=frozenset(), scores={'A': 10, 'B': 20}, remaining_moves=12,
        reliable_tails=frozenset({'A', 'B'}),
    )
    return replace(state, **changes)


def last_group_state():
    return make_state(
        food=frozenset({(2, 3), (0, 4), (4, 1)}),
        food_values={(2, 3): 9, (0, 4): 9, (4, 1): 9},
        next_food_digit=9, food_copy_targets={9: 3},
    )


class TestPureGameRules(unittest.TestCase):
    def test_cyclic_digit_preserves_conversion_and_offsets(self):
        for value, offset, expected in ((9, 1, 1), ('1', -1, 9),
                                        (3, 0, 3), (2, 19, 3), (0, 1, 1)):
            with self.subTest(value=value, offset=offset):
                self.assertEqual(cyclic_digit(value, offset), expected)

    def test_next_digit_preserves_legacy_inference_without_mutation(self):
        for values, expected in (([], None), (['8', '9', '1'], 8),
                                 ([2, 2, 3], 2), ([2, 4, 5], 2),
                                 (list(range(1, 10)), 1), ([0, 2], 0)):
            with self.subTest(values=values):
                snapshot = list(values)
                self.assertEqual(find_next_food_digit(iter(values)), expected)
                self.assertEqual(values, snapshot)

    def test_food_points_matches_state_rewards_and_clamps_multiplier(self):
        for digit, multiplier, expected in ((None, 1, 100), (None, 2, 200),
                                            (9, 1, 900), (9, 2, 1800),
                                            (9, 0, 900), (None, -2, 100)):
            with self.subTest(digit=digit, multiplier=multiplier):
                source = make_state(food_values={} if digit is None else {(2, 3): digit},
                                    multipliers={'A': multiplier, 'B': 1})
                snapshot = deepcopy(source)
                self.assertEqual(food_points(digit, multiplier), expected)
                self.assertEqual(source.food_reward('A', (2, 3)), expected)
                self.assertEqual(source, snapshot)

    def test_wrong_food_and_pickup_constants_match_transition_rewards(self):
        for pickup in (False, True):
            with self.subTest(pickup=pickup):
                source = make_state(
                    food_values={} if pickup else {(2, 3): 2},
                    next_food_digit=None if pickup else 1,
                    pickups=frozenset({(2, 3)}) if pickup else frozenset(),
                    multipliers={'A': 2, 'B': 1})
                snapshot = deepcopy(source)
                after = apply_move(source, 'right')
                delta = after.scores['A'] - source.scores['A']
                self.assertEqual(delta, 50 if pickup else -500)
                self.assertEqual(delta, MULTIPLIER_PICKUP_SCORE if pickup else WRONG_FOOD_PENALTY)
                self.assertEqual(source, snapshot)

    def test_crash_scores_v7_preserves_rival_and_source(self):
        for scores, seen, expected in (((2000, 0), False, (0, 0)),
                                       ((1000, 0), True, (500, 0)),
                                       ((1500, 0), False, (0, 0)),
                                       ((-100, -200), False, (-600, -200)),
                                       ((-2000, 0), True, (-2500, 0)),
                                       ((-2500, 0), True, (-3000, 0))):
            with self.subTest(scores=scores, seen=seen):
                source = {'A': scores[0], 'B': scores[1]}
                snapshot = dict(source)
                self.assertEqual(crash_scores(source['A'], source['B'], seen), expected)
                self.assertEqual(source, snapshot)


class TestLastNumberedGroup(unittest.TestCase):
    def test_compact_consumes_all_last_group_copies(self):
        source = last_group_state()
        engine, compact = IterativeSearchEngine.from_game_state(source)
        snapshot = deepcopy(compact)
        after = engine.apply_move(compact, 'right')
        self.assertEqual(tuple(engine.cell(i) for i in after.bodies[0]),
                         ((2, 3), (2, 2), (2, 1), (2, 0)))
        self.assertEqual(after.scores, (910, 20))
        self.assertEqual(after.next_food_digit, 1)
        self.assertEqual(after.food_values, ())
        self.assertEqual(compact, snapshot)
        self.assertEqual(after.food, (), 'All copies of consumed digit must disappear')

    def test_state_consumes_all_last_group_copies(self):
        source = last_group_state()
        snapshot = deepcopy(source)
        after = apply_move(source, 'right')
        self.assertEqual(after.body('A'), ((2, 3), (2, 2), (2, 1), (2, 0)))
        self.assertEqual(after.scores, {'A': 910, 'B': 20})
        self.assertEqual(after.next_food_digit, 1)
        self.assertEqual(after.food_values, {})
        self.assertEqual(after.food_copy_targets, {})
        self.assertEqual(source, snapshot)
        self.assertEqual(after.food, frozenset(), 'All copies of consumed digit must disappear')


class TestGameRuleContracts(unittest.TestCase):
    def test_parser_recognizes_classic_numbered_pickup_and_wall_symbols(self):
        for numbered in (False, True):
            with self.subTest(numbered=numbered):
                state = parse_state({
                    'board': '|A*X#B|\n| 12  |\n| 3   |' if numbered else
                             '|A*X#B|\n|     |\n|     |',
                    'rows': 3, 'cols': 5, 'side': 'B',
                    'score_1': 10, 'score_2': 20, 'remaining_moves': 12,
                    'multiplier_1': 2, 'multiplier_2': 3,
                })
                self.assertEqual((state.rows, state.cols, state.side, state.enemy), (3, 5, 'B', 'A'))
                self.assertEqual(state.snakes, {'A': ((0, 0),), 'B': ((0, 4),)})
                self.assertEqual(state.pickups, frozenset({(0, 2)}))
                self.assertEqual(state.walls, frozenset({(0, 3)}))
                self.assertEqual(state.scores, {'A': 10, 'B': 20})
                self.assertEqual(state.multipliers, {'A': 2, 'B': 3})
                self.assertEqual(state.remaining_moves, 12)
                self.assertEqual(state.reliable_tails, frozenset())
                self.assertEqual(state.food_values, {(1, 1): 1, (1, 2): 2, (2, 1): 3} if numbered else {})
                self.assertEqual(state.next_food_digit, 1 if numbered else None)
                self.assertEqual(state.food, frozenset({(1, 1)}) if numbered else frozenset({(0, 1)}))
                self.assertEqual(state.food_copy_targets, {1: 1, 2: 1, 3: 1} if numbered else {})
                self.assertEqual(state.wrong_food(), frozenset({(1, 2), (2, 1)}) if numbered else frozenset())
                engine, compact = IterativeSearchEngine.from_game_state(state)
                self.assert_equivalent(state, engine, compact)

    def assert_equivalent(self, state, engine, compact):
        self.assertEqual((compact.rows, compact.cols), (state.rows, state.cols))
        self.assertEqual(compact.side, 0 if state.side == 'A' else 1)
        self.assertEqual(tuple(tuple(engine.cell(i) for i in body) for body in compact.bodies),
                         (state.body('A'), state.body('B')))
        for field in ('food', 'pickups', 'walls'):
            self.assertEqual(frozenset(engine.cell(i) for i in getattr(compact, field)),
                             getattr(state, field), field)
        self.assertEqual({engine.cell(i): digit for i, digit in compact.food_values},
                         state.food_values)
        self.assertEqual(compact.next_food_digit, state.next_food_digit or 0)
        self.assertEqual(compact.scores, (state.scores['A'], state.scores['B']))
        self.assertEqual(compact.multipliers, (state.multipliers['A'], state.multipliers['B']))
        self.assertEqual(compact.remaining_moves, state.remaining_moves)
        self.assertEqual(compact.reliable_tails,
                         ('A' in state.reliable_tails, 'B' in state.reliable_tails))

    def transition(self, source, direction, **expected):
        snapshot = deepcopy(source)
        engine, compact = IterativeSearchEngine.from_game_state(source)
        compact_snapshot = deepcopy(compact)
        self.assert_equivalent(source, engine, compact)
        self.assertEqual(set(legal_moves(source)), set(engine.legal_moves(compact)))
        self.assertIn(direction, legal_moves(source))
        after = apply_move(source, direction)
        compact_after = engine.apply_move(compact, direction)
        self.assertEqual(source, snapshot, 'Transition mutated source GameState')
        self.assertEqual(compact, compact_snapshot, 'Transition mutated source CompactState')
        self.assert_equivalent(after, engine, compact_after)
        self.assertEqual(after.side, source.enemy)
        self.assertEqual(after.enemy, source.side)
        self.assertEqual(after.remaining_moves, max(0, source.remaining_moves - 1))
        self.assertEqual(after.reliable_tails, source.reliable_tails | {source.side})
        for field, value in expected.items():
            self.assertEqual(getattr(after, field), value, field)
        return after

    def test_normal_move_body_tail_score_turn_and_source(self):
        for side, direction, body in (
            ('A', 'right', ((2, 3), (2, 2), (2, 1))),
            ('B', 'left', ((5, 3), (5, 4), (5, 5))),
        ):
            for remaining in (1, 12):
                with self.subTest(side=side, remaining=remaining):
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A',
                                        remaining_moves=remaining, reliable_tails=frozenset())
                    bodies = dict(source.snakes, **{side: body})
                    scores = dict(source.scores)
                    scores[side] += 1
                    self.transition(source, direction, snakes=bodies, scores=scores,
                                    food=frozenset(), food_values={}, pickups=frozenset(),
                                    multipliers={'A': 1, 'B': 1}, walls=frozenset())

    def test_neck_reversal_is_illegal_even_with_reliable_tail(self):
        for side in ('A', 'B'):
            for reliable in (frozenset(), frozenset({'A', 'B'})):
                with self.subTest(side=side, reliable=reliable):
                    # A two-cell snake makes its neck also its tail.
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A',
                                        snakes={'A': ((2, 2), (2, 1)),
                                                'B': ((5, 4), (5, 5))}, reliable_tails=reliable)
                    engine, compact = IterativeSearchEngine.from_game_state(source)
                    forbidden = 'left' if side == 'A' else 'right'
                    self.assertNotIn(forbidden, legal_moves(source))
                    self.assertNotIn(forbidden, engine.legal_moves(compact))

    def test_own_tail_requires_reliability_and_no_growth(self):
        body = ((2, 2), (2, 1), (1, 1), (1, 2))
        for side in ('A', 'B'):
            for reliable, grows, allowed in ((True, False, True), (False, False, False),
                                             (True, True, False)):
                with self.subTest(side=side, reliable=reliable, grows=grows):
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A')
                    source = replace(source, snakes=dict(source.snakes, **{side: body}),
                                     reliable_tails=frozenset({side}) if reliable else frozenset(),
                                     food=frozenset({(1, 2)}) if grows else frozenset())
                    engine, compact = IterativeSearchEngine.from_game_state(source)
                    self.assertEqual('up' in legal_moves(source), allowed)
                    self.assertEqual('up' in engine.legal_moves(compact), allowed)
                    if allowed:
                        after = self.transition(source, 'up')
                        self.assertEqual(after.body(side), ((1, 2), (2, 2), (2, 1), (1, 1)))
                        self.assertEqual(after.scores[side], source.scores[side] + 1)

    def test_boundaries_and_rival_body_are_illegal(self):
        for side in ('A', 'B'):
            with self.subTest(side=side):
                enemy = 'B' if side == 'A' else 'A'
                source = make_state(side=side, enemy=enemy,
                                    snakes={side: ((0, 0),), enemy: ((0, 1), (1, 1), (1, 0))})
                engine, compact = IterativeSearchEngine.from_game_state(source)
                self.assertEqual(legal_moves(source), [])
                self.assertEqual(engine.legal_moves(compact), ())

    def test_classic_food_grows_once_and_has_no_normal_score_bonus(self):
        for side, direction, target in (('A', 'right', (2, 3)), ('B', 'left', (5, 3))):
            for multiplier in (1, 3):
                with self.subTest(side=side, multiplier=multiplier):
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A',
                                        food=frozenset({target, (0, 6)}),
                                        multipliers={'A': multiplier, 'B': multiplier})
                    scores = dict(source.scores)
                    scores[side] += 100 * multiplier
                    self.transition(source, direction, scores=scores,
                                    snakes=dict(source.snakes, **{side: (target,) + source.body(side)}),
                                    food=frozenset({(0, 6)}), next_food_digit=None, food_values={})

    def test_correct_number_consumes_group_and_promotes_next_copies(self):
        for digit in (1, 4):
            for copies in (1, 3):
                with self.subTest(digit=digit, copies=copies):
                    targets = ((2, 3), (0, 4), (4, 1))[:copies]
                    values = {pos: digit for pos in targets}
                    values.update({(3, 3): digit + 1, (3, 5): digit + 1})
                    source = make_state(food=frozenset(targets), food_values=values,
                                        next_food_digit=digit, food_copy_targets={digit: copies, digit + 1: 2})
                    self.transition(source, 'right', scores={'A': 10 + digit * 100, 'B': 20},
                                    snakes=dict(source.snakes, A=((2, 3),) + source.body('A')),
                                    food_values={(3, 3): digit + 1, (3, 5): digit + 1},
                                    food=frozenset({(3, 3), (3, 5)}), next_food_digit=digit + 1,
                                    food_copy_targets={digit + 1: 2})

    def test_wrong_number_consumes_only_touched_copy_and_grows(self):
        for wrong in (2, 9):
            for copies in (1, 3):
                with self.subTest(wrong=wrong, copies=copies):
                    values = {pos: wrong for pos in ((2, 3), (0, 4), (4, 1))[:copies]}
                    values[(3, 3)] = 1
                    source = make_state(food=frozenset({(3, 3)}), food_values=values,
                                        next_food_digit=1, multipliers={'A': 4, 'B': 1},
                                        food_copy_targets={1: 1, wrong: copies})
                    self.transition(source, 'right', scores={'A': -490, 'B': 20},
                                    snakes=dict(source.snakes, A=((2, 3),) + source.body('A')),
                                    food_values={pos: value for pos, value in values.items() if pos != (2, 3)},
                                    food=frozenset({(3, 3)}), next_food_digit=1,
                                    food_copy_targets={1: 1, wrong: copies}, multipliers={'A': 4, 'B': 1})

    def test_multiple_copies_wrap_nine_to_one(self):
        for multiplier in (1, 2):
            with self.subTest(multiplier=multiplier):
                source = replace(last_group_state(),
                                 food_values={**last_group_state().food_values, (3, 3): 1, (3, 5): 1},
                                 food_copy_targets={9: 3, 1: 2}, multipliers={'A': multiplier, 'B': 1})
                self.transition(source, 'right', scores={'A': 10 + 900 * multiplier, 'B': 20},
                                snakes=dict(source.snakes, A=((2, 3),) + source.body('A')),
                                food=frozenset({(3, 3), (3, 5)}), food_values={(3, 3): 1, (3, 5): 1},
                                food_copy_targets={1: 2}, next_food_digit=1)

    def test_pickup_permanently_increases_multiplier_for_later_number(self):
        for initial in (1, 3):
            with self.subTest(initial=initial):
                source = make_state(pickups=frozenset({(2, 3)}),
                                    multipliers={'A': initial, 'B': 1}, next_food_digit=2,
                                    food=frozenset({(2, 4)}), food_values={(2, 4): 2, (0, 6): 3})
                after = self.transition(source, 'right', scores={'A': 60, 'B': 20},
                                        snakes=dict(source.snakes, A=((2, 3), (2, 2), (2, 1))),
                                        pickups=frozenset(), multipliers={'A': initial + 1, 'B': 1},
                                        food=source.food, food_values=source.food_values, next_food_digit=2)
                after = self.transition(after, 'left', scores={'A': 60, 'B': 21},
                                        multipliers={'A': initial + 1, 'B': 1})
                self.transition(after, 'right', scores={'A': 60 + 200 * (initial + 1), 'B': 21},
                                snakes=dict(after.snakes, A=((2, 4),) + after.body('A')),
                                multipliers={'A': initial + 1, 'B': 1}, pickups=frozenset(),
                                food=frozenset({(0, 6)}), food_values={(0, 6): 3}, next_food_digit=3)

    def test_first_wall_hit_resets_positive_score_without_moving_and_continues(self):
        for side, direction, target in (('A', 'right', (2, 3)), ('B', 'left', (5, 3))):
            for remaining in (1, 12):
                with self.subTest(side=side, remaining=remaining):
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A',
                                        walls=frozenset({target}), remaining_moves=remaining,
                                        crashed_a=False, crashed_b=False)
                    scores = dict(source.scores)
                    scores[side] = 0
                    after = self.transition(source, direction, snakes=source.snakes, scores=scores,
                                            food=frozenset(), multipliers={'A': 1, 'B': 1},
                                            walls=source.walls if side == 'A' else frozenset())
                    if remaining > 1:
                        continued = self.transition(after, 'left' if after.side == 'B' else 'right')
                        self.assertEqual(continued.remaining_moves, 10)

    def test_walls_shrink_on_b_turn_only_horizontal_vertical_singleton(self):
        for walls, shrunk in (
            (frozenset({(0, c) for c in range(5)}), frozenset({(0, 1), (0, 2), (0, 3)})),
            (frozenset({(r, 6) for r in range(5)}), frozenset({(1, 6), (2, 6), (3, 6)})),
            (frozenset({(0, 0)}), frozenset()),
            (frozenset({(0, 0), (0, 1)}), frozenset()),
            (frozenset(), frozenset()),
        ):
            for side in ('A', 'B'):
                with self.subTest(walls=walls, side=side):
                    source = make_state(side=side, enemy='B' if side == 'A' else 'A', walls=walls)
                    scores = dict(source.scores)
                    scores[side] += 1
                    self.transition(source, 'right' if side == 'A' else 'left', scores=scores,
                                    walls=walls if side == 'A' else shrunk)


class TestTerminalRuleContracts(unittest.TestCase):
    def test_final_score_values_for_both_roots_without_crash_penalties(self):
        for scores, difference in (({'A': 200, 'B': 50}, 150),
                                   ({'A': 50, 'B': 200}, -150),
                                   ({'A': 200, 'B': 200}, 0)):
            for root in (0, 1):
                with self.subTest(scores=scores, root=root):
                    engine, compact = IterativeSearchEngine.from_game_state(
                        make_state(scores=scores, remaining_moves=0))
                    engine.root_side = root
                    delta = difference if root == 0 else -difference
                    expected = 300_000_000 + delta if delta > 0 else -300_000_000 + delta if delta < 0 else 0
                    self.assertEqual(engine.final_score_value(compact), expected)
                    self.assertEqual(compact.scores, (scores['A'], scores['B']))

    def test_ordinary_crash_continues_without_rival_reward(self):
        for dead in ('A', 'B'):
            for lead, delta in ((1000, -500), (1500, 0), (2000, 500)):
                for root in (0, 1):
                    with self.subTest(dead=dead, lead=lead, root=root):
                        enemy = 'B' if dead == 'A' else 'A'
                        scores = {dead: lead, enemy: 0}
                        source = make_state(side=dead, enemy=enemy, scores=scores,
                                            crashed_a=False, crashed_b=False)
                        source = replace(source, walls=frozenset({step(source.head(dead), 'up')}))
                        engine, compact = IterativeSearchEngine.from_game_state(source)
                        engine.root_side = root
                        after = engine.apply_move(compact, 'up')
                        self.assertFalse(after.is_terminal())
                        self.assertEqual(after.scores, (0, 0))
                        self.assertEqual(engine.terminal_value(after), engine.evaluate(after))
                        self.assertEqual(compact.scores, (scores['A'], scores['B']))

    def test_simulator_no_safe_moves_consumes_crash_turn_without_rival_reward(self):
        for dead in ('A', 'B'):
            for lead, winner in ((1000, 'enemy'), (1500, None), (2000, 'dead')):
                with self.subTest(dead=dead, lead=lead):
                    enemy = 'B' if dead == 'A' else 'A'
                    source = make_state(side=dead, enemy=enemy, scores={dead: lead, enemy: 0},
                                        crashed_a=False, crashed_b=False,
                                        snakes={dead: ((0, 0),), enemy: ((0, 1), (1, 1), (1, 0))})
                    snapshot = deepcopy(source)
                    brain = bot_lab.SnakeBrain(enable_debug=False)
                    brain.choose_move = Mock(return_value='up')
                    brain.commit_move = Mock()
                    brain.decision_debug = Mock(return_value={})
                    with patch.object(bot_lab, 'initial_state', return_value=source), \
                            patch.object(bot_lab, 'SnakeBrain', return_value=brain):
                        result = bot_lab.simulate_game(17, max_turns=1, rows=7, cols=7)
                    self.assertIsNone(result.winner)
                    expected_scores = {dead: 0, enemy: 0}
                    self.assertEqual((result.score_a, result.score_b),
                                     (expected_scores['A'], expected_scores['B']))
                    self.assertIsNone(result.no_legal_side)
                    self.assertEqual(result.turns, 1)
                    brain.choose_move.assert_called_once()
                    self.assertEqual(source, snapshot)

    def test_simulator_turn_limit_chooses_winner_by_score(self):
        for scores, winner in (({'A': 200, 'B': 50}, 'A'),
                               ({'A': 50, 'B': 200}, 'B'),
                               ({'A': 200, 'B': 200}, None)):
            with self.subTest(scores=scores):
                source = make_state(scores=scores, remaining_moves=2)
                brain_a = bot_lab.SnakeBrain(enable_debug=False)
                brain_b = bot_lab.SnakeBrain(enable_debug=False)
                brain_a.choose_move = Mock(return_value='right')
                brain_b.choose_move = Mock(return_value='left')
                brain_a.commit_move = Mock()
                brain_b.commit_move = Mock()
                brain_a.decision_debug = Mock(return_value={})
                brain_b.decision_debug = Mock(return_value={})
                with patch.object(bot_lab, 'initial_state', return_value=source), \
                        patch.object(bot_lab, 'SnakeBrain', side_effect=[brain_a, brain_b]), \
                        patch.object(bot_lab, 'spawn_food', side_effect=lambda state, *_args: state):
                    result = bot_lab.simulate_game(17, max_turns=2, rows=7, cols=7)
                self.assertEqual(result.winner, winner)
                self.assertEqual((result.score_a, result.score_b), (scores['A'] + 1, scores['B'] + 1))
                self.assertEqual(result.turns, 2)
                self.assertIsNone(result.no_legal_side)
                brain_a.commit_move.assert_called_once()
                brain_b.commit_move.assert_called_once()


if __name__ == '__main__':
    unittest.main()
