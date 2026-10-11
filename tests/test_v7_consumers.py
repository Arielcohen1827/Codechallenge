import unittest
from dataclasses import replace
from time import perf_counter

from snake_bot.food_planner import (
    winning_suicide_move, forced_kill_move, objective_kind, objective_reward,
    build_food_plans, apply_move as simulated_move,
)
from snake_bot.move_search import TurnSearchCache, terminal_trap_value
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import available_moves
from tests.test_food_planner import open_state


class TestV7Consumers(unittest.TestCase):
    def trapped(self, **changes):
        return open_state(rows=2, cols=2,
                          snakes={'A': ((0, 0),), 'B': ((0, 1), (1, 1), (1, 0))},
                          food=frozenset(), crashed_a=False, crashed_b=False, **changes)

    def test_trapped_snake_has_crash_actions_not_terminal(self):
        state = self.trapped()
        self.assertTrue(available_moves(state, 'A'))
        self.assertEqual(terminal_trap_value(state, 'A', 'B'), (0, None))
        engine, compact = IterativeSearchEngine.from_game_state(state)
        engine.deadline = perf_counter() + 2
        engine.alphabeta(compact, 1, -10**12, 10**12)
        self.assertGreater(engine.nodes, 1)

    def test_positive_lead_does_not_enable_suicide_or_forced_kill(self):
        for lead in (1501, 9000):
            state = self.trapped(scores={'A': lead, 'B': 0})
            self.assertIsNone(winning_suicide_move(state, 'A'))
            self.assertIsNone(forced_kill_move(state, 'A', available_moves(state, 'A')))

    def test_unknown_positive_crash_enumerates_both_official_outcomes(self):
        state = replace(self.trapped(scores={'A': 1000, 'B': 0}), crashed_a=None)
        engine, compact = IterativeSearchEngine.from_game_state(state)
        outcomes = engine.move_outcomes(compact, 'up')
        self.assertEqual({s.scores[0] for s in outcomes}, {0, 500})
        self.assertTrue(all(s.crashed_a is True for s in outcomes))
        self.assertIsNone(state.crashed_a)
        engine.deadline = perf_counter() + 2
        engine.alphabeta(compact, 1, -10**12, 10**12)

    def test_terminal_uses_elimination_not_score_difference(self):
        state = self.trapped(scores={'A': -2501, 'B': -3000})
        engine, compact = IterativeSearchEngine.from_game_state(state)
        engine.root_side = 0
        self.assertLess(engine.terminal_value(compact), 0)
        self.assertEqual(terminal_trap_value(state, 'A', 'B')[1], 'loss')

    def test_tail_objective_identity_and_cache_copy_targets(self):
        state = open_state(tail_food_a=frozenset({(0, 1)}), tail_food_b=frozenset({(0, 2)}))
        self.assertEqual(objective_kind(state, (0, 1), 'A'), 'tail_food')
        self.assertEqual(objective_kind(state, (0, 1), 'B'), 'tail_cleanup')
        from snake_bot.move_search import state_key
        self.assertNotEqual(state_key(state), state_key(replace(state, food_copy_targets={1: 2})))

    def test_tail_reward_depends_on_beneficiary_without_new_planner_targets(self):
        for side, field in (('A', 'tail_food_a'), ('B', 'tail_food_b')):
            cell = (0, 1)
            state = open_state(food=frozenset(), multipliers={'A': 3, 'B': 3},
                               **{field: frozenset({cell})})
            enemy = 'B' if side == 'A' else 'A'
            self.assertEqual(objective_reward(state, side, cell), 300)
            self.assertEqual(objective_reward(state, enemy, cell), 0)
            self.assertEqual(build_food_plans(state, side), [])

    def test_unknown_rival_crash_uses_root_perspective_without_mutating_history(self):
        state = self.trapped(scores={'A': 0, 'B': 1000})
        state = replace(state, crashed_b=None)
        after = simulated_move(state, 'up', 'B', perspective='A')
        self.assertEqual(after.scores['B'], 500)
        self.assertTrue(after.crashed_b)
        self.assertIsNone(state.crashed_b)
        cache = TurnSearchCache()
        cache.root_side = 'A'
        self.assertEqual(cache.apply(state, 'B', 'up').scores['B'], 500)

    def test_tactical_search_stops_only_for_official_terminal_result(self):
        from snake_bot.move_search import two_ply_score, fast_two_ply_score, extra_depth_score
        for scorer in (two_ply_score, fast_two_ply_score):
            for score, expected_loss in ((-2000, False), (-2001, True)):
                state = self.trapped(scores={'A': score, 'B': 0})
                _score, info = scorer(state, 'A', 'up', None, 0, 0, TurnSearchCache())
                self.assertEqual(info['forced_loss'], expected_loss)
                if expected_loss:
                    self.assertEqual(info['terminal_outcome'], 'loss')
                else:
                    self.assertIsNotNone(info['enemy_reply'])
        state = self.trapped(scores={'A': -2001, 'B': 0})
        self.assertLess(extra_depth_score(state, 'A', 'up', TurnSearchCache()), 0)

    def test_move_limit_finalizes_without_an_extra_reply(self):
        from snake_bot.move_search import two_ply_score, fast_two_ply_score
        for scores, outcome in (({'A': 200, 'B': 0}, 'win'),
                                ({'A': 0, 'B': 200}, 'loss'),
                                ({'A': -1, 'B': 0}, 'draw')):
            state = open_state(scores=scores, remaining_moves=1)
            for scorer in (two_ply_score, fast_two_ply_score):
                _score, info = scorer(state, 'A', 'up', None, 0, 0, TurnSearchCache())
                self.assertEqual(info['terminal_outcome'], outcome)
                self.assertIsNone(info['enemy_reply'])

    def test_planner_rival_response_simulates_forced_crash(self):
        from snake_bot.food_planner import rival_response_stats
        state = open_state(rows=2, cols=2, food=frozenset(),
                           snakes={'A': ((1, 1), (0, 1), (1, 0)), 'B': ((0, 0),)},
                           crashed_a=False, crashed_b=False)
        stats = rival_response_stats(state, 'A', 'left', depth=0)
        self.assertEqual(stats['enemy_moves'], 4)
        self.assertFalse(stats['forced_zero'])

    def test_planner_path_cannot_continue_past_official_terminal(self):
        from snake_bot.food_planner import simulate_path_to_food
        state = open_state(remaining_moves=1)
        self.assertIsNone(simulate_path_to_food(state, 'A', ((2, 1), (1, 1), (0, 1))))
        after = simulate_path_to_food(state, 'A', ((2, 1), (1, 1)))
        self.assertTrue(after.is_terminal())

    def test_planner_rival_elimination_is_a_win_not_zero_escape_loss(self):
        from snake_bot.food_planner import rival_response_stats, survival_move_analysis
        from snake_bot.bot_weights import get_weights
        for score in (-2000, -2001):
            with self.subTest(score=score):
                state = open_state(rows=3, cols=3, food=frozenset(),
                                   snakes={'A': ((1, 1), (0, 1), (1, 0)), 'B': ((0, 0),)},
                                   scores={'A': 0, 'B': score}, crashed_a=False, crashed_b=False)
                stats = rival_response_stats(state, 'A', 'left')
                self.assertFalse(stats['forced_zero'])
                self.assertEqual(stats['zero_escape_replies'], 0)
                analysis = survival_move_analysis(state, 'A', 'left')
                if score == -2001:
                    self.assertEqual(analysis['future_score'], get_weights()['enemy_trapped_future_bonus'])
                    self.assertEqual(analysis['reply_count'], 0)
                else:
                    self.assertGreater(analysis['reply_count'], 0)

    def test_third_and_fourth_ply_use_official_terminal_values(self):
        from snake_bot.move_search import best_reply_score, extra_depth_score
        from snake_bot.bot_weights import get_weights
        weights = get_weights()
        for remaining in (3, 4):
            for scores, result in (({'A': 200, 'B': 0}, weights['deep_enemy_trapped_bonus']),
                                   ({'A': 0, 'B': 200}, -weights['deep_no_reply_penalty']),
                                   ({'A': -1 if remaining == 3 else 0, 'B': 0}, 0)):
                with self.subTest(scores=scores, remaining=remaining):
                    state = open_state(food=frozenset(), scores=scores, remaining_moves=remaining)
                    self.assertEqual(extra_depth_score(state, 'A', 'up', TurnSearchCache()), result)
        for scores, result in (({'A': 200, 'B': 0}, weights['deep_enemy_trapped_bonus']),
                               ({'A': 0, 'B': 200}, -weights['deep_no_reply_penalty']),
                               ({'A': -1, 'B': 0}, 0)):
            with self.subTest(last_reply=scores):
                state = open_state(food=frozenset(), scores=scores, remaining_moves=1)
                self.assertEqual(best_reply_score(state, 'A', TurnSearchCache()), result)
