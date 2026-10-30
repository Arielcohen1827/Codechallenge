import json
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from snake_bot import move_search as production_move_search
from snake_bot.bot_weights import get_weights
from snake_bot.search_engine import IterativeSearchEngine, SearchResult
from snake_bot.snake_state import parse_state
from tests.test_food_planner import open_state
from tools import compare_deep_normalization as compare


FIXTURES = Path(__file__).parent / 'fixtures' / 'deep_clipping_real_v7.json'
move_search = compare.load_planner(
    compare.BASELINE.parent / 'move_search.candidate.py',
    'f9b86822a12dffd3941ba2f66c6d963997aa1a522cf5328af4049e457b6f3b07',
    get_weights())


class DeepNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURES.read_text(encoding='utf-8'))['cases']

    def test_production_preserves_accepted_normalization_logic(self):
        baseline = compare.replay.frozen_source('snake_bot/move_search.py', compare.REFERENCE_SHA)
        before = ast.parse(baseline)
        after = ast.parse(Path(production_move_search.__file__).read_text(encoding='utf-8'))
        # v8/v9 legitimately extend caches and transitions; the accepted
        # ranking/normalization and safety arbitration must remain identical.
        for name in ('rank_deep_moves', 'deep_sort_key', 'prefer_safer_close_move'):
            original = next(node for node in before.body if getattr(node, 'name', None) == name)
            current = next(node for node in after.body if getattr(node, 'name', None) == name)
            if name == 'rank_deep_moves':
                allowed = ast.parse('cache.action_search = search').body[0]
                assignments = [node for node in current.body if ast.dump(node, include_attributes=False)
                               == ast.dump(allowed, include_attributes=False)]
                self.assertEqual(len(assignments), 1, 'Exactly one action-result export is permitted')
                current.body.remove(assignments[0])
            self.assertEqual(ast.dump(current, include_attributes=False),
                             ast.dump(original, include_attributes=False), name)

    def test_production_clips_before_centering_both_saturated_signs(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign), patch.object(
                    production_move_search.IterativeSearchEngine, 'search',
                    return_value=SearchResult(
                        {'up': sign * 240582, 'down': sign * 240512}, 4, 1, 0, 0)):
                ranked = production_move_search.rank_deep_moves(
                    open_state(), 'A', ['up', 'down'], search_time_ms=10)
                self.assertEqual({row['direction']: row['search_adjustment']
                                  for row in ranked}, {'up': 0, 'down': 0})

    def adjustments(self, values, state=None, depth=4, **row_changes):
        rows = [dict(direction=direction, search_score=value, safety='SAFE',
                     **row_changes) for direction, value in values.items()]
        return move_search._search_adjustments(
            open_state() if state is None else state, rows, depth, get_weights())

    def test_real_negative_saturation_preserves_raw_differences(self):
        case = next(case for case in self.cases if
                    case['provenance']['filename'] ==
                    'game_8d25279e-c5d3-11f1-96f5-a2aacf8fcf85.log'
                    and case['provenance']['debug_line'] == 498)
        state = parse_state(case['observed'],
                            crash_history=case['diagnostic']['crash_history'])
        rows = case['diagnostic']['deep_moves']
        raw = {row['direction']: row['search_score'] for row in rows}
        self.assertEqual(raw, {'up': -431010, 'down': -322732, 'left': -429420})
        expected = {'up': -12818, 'down': 25000, 'left': -12262}
        with patch.object(move_search.IterativeSearchEngine, 'search',
                          return_value=SearchResult(raw, 4, 1, 0, 0)):
            ranked = move_search.rank_deep_moves(
                state, state.side, list(raw), search_time_ms=10)
        self.assertEqual({row['direction']: row['search_adjustment']
                          for row in ranked}, expected)
        self.assertEqual(move_search._search_adjustments(
            state, rows, 4, get_weights()), expected)

    def test_positive_and_negative_clipping_with_two_safe_moves(self):
        for raw, expected in (
            ({'up': 240582, 'down': 240512}, {'up': 12, 'down': -13}),
            ({'up': -240582, 'down': -240512}, {'up': -13, 'down': 12}),
            ({'up': 190000, 'down': 220000}, {'up': -5250, 'down': 5250}),
            ({'up': -190000, 'down': -220000}, {'up': 5250, 'down': -5250}),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.adjustments(raw), expected)

    def test_terminal_win_loss_and_mixed_group_preserve_whole_mapping(self):
        for scores, remaining, expected_terminal in (
            ({'A': 100, 'B': -2501}, 40, 500002601),
            ({'A': -2501, 'B': 100}, 40, -500002601),
            ({'A': 100, 'B': 0}, 0, 300000100),
            ({'A': 0, 'B': 100}, 0, -300000100),
        ):
            with self.subTest(scores=scores, remaining=remaining):
                engine, compact = IterativeSearchEngine.from_game_state(
                    open_state(scores=scores, remaining_moves=remaining))
                terminal = engine.terminal_value(compact)
                self.assertEqual(terminal, expected_terminal)
                expected = ({'up': 0, 'down': 0, 'left': 0} if terminal > 0 else
                            {'up': -25000, 'down': 25000, 'left': 25000})
                self.assertEqual(self.adjustments(
                    {'up': terminal, 'down': 240582, 'left': 240512}), expected)

    def test_horizon_draw_and_mixed_values_preserve_legacy(self):
        engine, compact = IterativeSearchEngine.from_game_state(
            open_state(remaining_moves=0))
        self.assertEqual(engine.terminal_value(compact), 0)
        for remaining in (0, 1, 4):
            with self.subTest(remaining=remaining):
                state = open_state(remaining_moves=remaining)
                self.assertEqual(self.adjustments(
                    {'up': 0, 'down': 240582, 'left': 240512}, state),
                    {'up': -25000, 'down': 23333, 'left': 23333})
                self.assertEqual(self.adjustments({'up': 0, 'down': 0}, state),
                                 {'up': 0, 'down': 0})

    def test_root_terminal_guard_preserves_mapping_before_horizon(self):
        for scores in ({'A': -2501, 'B': 0}, {'A': 0, 'B': -2501},
                       {'A': -2501, 'B': -500002501}):
            with self.subTest(scores=scores):
                state = open_state(scores=scores)
                self.assertTrue(state.is_terminal())
                self.assertEqual(self.adjustments(
                    {'up': 240582, 'down': 240512}, state), {'up': 0, 'down': 0})

    def test_extreme_normal_evaluation_uses_conservative_fallback(self):
        state = open_state(scores={'A': 20000000, 'B': 0})
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertFalse(compact.is_terminal())
        raw = engine.evaluate(compact)
        self.assertEqual(raw, 440000000)
        self.assertEqual(self.adjustments(
            {'up': raw, 'down': 240582, 'left': 240512}, state),
            {'up': 0, 'down': 0, 'left': 0})

    def test_extreme_guard_boundary_is_inclusive_for_both_signs(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                self.assertEqual(self.adjustments(
                    {'up': sign * 300000000, 'down': sign * 299999900}),
                    {'up': 0, 'down': 0})
                self.assertEqual(self.adjustments(
                    {'up': sign * 299999999, 'down': sign * 299999899}),
                    {'up': 17, 'down': -18} if sign > 0 else
                    {'up': -18, 'down': 17})

    def test_forced_loss_and_missing_raw_are_excluded_from_guard_and_center(self):
        for excluded in (500000100, -500000100, None):
            with self.subTest(excluded=excluded):
                rows = [
                    {'direction': 'up', 'search_score': excluded, 'forced_loss': True},
                    {'direction': 'down', 'search_score': 240582},
                    {'direction': 'left', 'search_score': 240512},
                    {'direction': 'right', 'search_score': None},
                ]
                self.assertEqual(move_search._search_adjustments(
                    open_state(), rows, 4, get_weights()), {'down': 12, 'left': -13})
        self.assertEqual(self.adjustments({'up': 500000100}, forced_loss=True), {})
        self.assertEqual(self.adjustments({}), {})
        self.assertEqual(self.adjustments({'up': 5000}), {'up': 0})

    def test_final_cap_preserves_relative_order_and_can_still_create_ties(self):
        adjustments = self.adjustments(
            {'up': 300000, 'down': 400000, 'left': 450000, 'right': 500000})
        self.assertEqual(adjustments,
                         {'up': -25000, 'down': -4375, 'left': 13125, 'right': 25000})
        self.assertEqual(sorted(adjustments, key=adjustments.get),
                         ['up', 'down', 'left', 'right'])
        self.assertEqual(self.adjustments({'up': 0, 'down': 1000000, 'left': 1100000}),
                         {'up': -25000, 'down': 25000, 'left': 25000})

    def test_within_bounds_and_integer_translation_keep_mapping(self):
        for values, expected in (
            ((1, 2, 4), {'up': -1, 'down': 0, 'left': 0}),
            ((-199999, -199997, -199995), {'up': -1, 'down': 0, 'left': 0}),
            ((-200000, 0, 200000), {'up': -25000, 'down': 0, 'left': 25000}),
        ):
            for offset in (0, 600000, -600000):
                with self.subTest(values=values, offset=offset):
                    raw = dict(zip(('up', 'down', 'left'),
                                   (value + offset for value in values)))
                    self.assertEqual(self.adjustments(raw), expected)

    def test_real_guarded_and_within_bounds_fixtures_match_recorded_legacy(self):
        checked = 0
        changed_flat = 0
        for case in self.cases:
            diagnostic = case['diagnostic']
            rows = diagnostic.get('deep_moves', [])
            depth = case['analysis']['depth']
            if depth < get_weights()['iterative_search_min_apply_depth']:
                continue
            state = parse_state(case['observed'],
                                crash_history=diagnostic['crash_history'])
            actual = move_search._search_adjustments(state, rows, depth, get_weights())
            raw = [row['search_score'] for row in rows
                   if row['search_score'] is not None and not row.get('forced_loss')]
            guarded = (state.is_terminal() or state.remaining_moves <= depth
                       or any(abs(value) >= 300000000 for value in raw))
            recorded = {row['direction']: row['search_adjustment'] for row in rows
                        if row['search_score'] is not None and not row.get('forced_loss')}
            with self.subTest(provenance=case['provenance']):
                if guarded or all(abs(value) <= 200000 for value in raw):
                    self.assertEqual(actual, recorded)
                    checked += 1
                elif case['analysis']['all_clip_identical']:
                    self.assertNotEqual(actual, recorded)
                    changed_flat += 1
        self.assertGreaterEqual(checked, 8)
        self.assertGreaterEqual(changed_flat, 2)

    def test_rank_preserves_minimum_depth_and_empty_search_gate(self):
        for depth, raw, expected in (
            (3, {'up': 240582, 'down': 240512}, {'up': 0, 'down': 0}),
            (4, {'up': 240582, 'down': 240512}, {'up': 12, 'down': -13}),
            (6, {}, {'up': 0, 'down': 0}),
        ):
            with self.subTest(depth=depth, raw=raw):
                with patch.object(move_search.IterativeSearchEngine, 'search',
                                  return_value=SearchResult(raw, depth, 1, 0, 0)):
                    ranked = move_search.rank_deep_moves(
                        open_state(), 'A', ['up', 'down'], search_time_ms=10)
                self.assertEqual({row['direction']: row['search_adjustment']
                                  for row in ranked}, expected)

    def test_safety_and_forced_loss_keep_lexicographic_priority(self):
        base = dict(food_now=False, wall_hit=False, our_replies=2, exits=2,
                    region=30, voronoi_score=0, safety='SAFE', forced_loss=False)
        rows = [dict(base, direction='up', score=500000, forced_loss=True),
                dict(base, direction='down', score=400000, safety='SUICIDAL'),
                dict(base, direction='left', score=-25000),
                dict(base, direction='right', score=25000)]
        rows.sort(key=move_search.deep_sort_key, reverse=True)
        self.assertEqual([row['direction'] for row in rows],
                         ['right', 'left', 'down', 'up'])
        move_search.prefer_safer_close_move(rows)
        self.assertEqual(rows[0]['direction'], 'right')
        rows = [dict(base, direction='up', score=5, safety='SUICIDAL'),
                dict(base, direction='down', score=0)]
        move_search.prefer_safer_close_move(rows)
        self.assertEqual(rows[0]['direction'], 'down')


if __name__ == '__main__':
    unittest.main()
