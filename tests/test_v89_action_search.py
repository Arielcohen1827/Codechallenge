import unittest
from dataclasses import replace
from unittest.mock import patch
from snake_bot.snake_state import GameState
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.move_search import TurnSearchCache, rank_deep_moves


def snapshot(**changes):
    initial = GameState(
        rows=20,
        cols=30,
        board=(),
        side='A',
        enemy='B',
        snakes={
            'A': ((10, 5), (10, 4), (10, 3)),
            'B': ((10, 13), (11, 13), (12, 13)),
        },
        food=frozenset(),
        scores={'A': 7000, 'B': 7000},
        remaining_moves=200,
        reliable_tails=frozenset({'A', 'B'}),
        crashed_a=False,
        crashed_b=False,
    )
    return replace(initial, **changes)


class SelectiveActionSearchTests(unittest.TestCase):
    def test_candidate_shots_require_affordability_hit_and_nonempty_cut_for_both_sides(
        self,
    ):
        for side in ('A', 'B'):
            state = snapshot(side=side, enemy='B' if side == 'A' else 'A')
            engine, compact = IterativeSearchEngine.from_game_state(state)
            shots = [
                action
                for action in engine.available_actions(compact)
                if isinstance(action, tuple)
            ]
            self.assertEqual(
                shots, [('shoot', 'right' if side == 'A' else 'left')]
            )
            poor = replace(state, scores={'A': 4999, 'B': 4999})
            self.assertFalse(
                any(
                    isinstance(action, tuple)
                    for action in engine.available_actions(
                        engine.from_game_state(poor)[1]
                    )
                )
            )
            no_cut = replace(
                state,
                snakes={key: body[:1] for key, body in state.snakes.items()},
            )
            self.assertFalse(
                any(
                    isinstance(action, tuple)
                    for action in engine.available_actions(
                        engine.from_game_state(no_cut)[1]
                    )
                )
            )

    def test_wall_cover_eliminates_shoot_in_compact_and_tactical_candidates(
        self,
    ):
        state = snapshot(walls=frozenset({(10, 9)}))
        engine, compact = IterativeSearchEngine.from_game_state(state)
        self.assertFalse(
            any(
                isinstance(action, tuple)
                for action in engine.available_actions(compact)
            )
        )
        self.assertFalse(
            any(
                isinstance(action, tuple)
                for action in TurnSearchCache().legal_actions(state, 'A')
            )
        )

    def test_root_search_adds_real_shot_without_requiring_legacy_caller_to_supply_it(
        self,
    ):
        state = snapshot(rows=14, cols=20, remaining_moves=1)
        engine, compact = IterativeSearchEngine.from_game_state(state, 1000, 2)
        result = engine.search(compact, 'A', ['up', 'right', 'down'])
        self.assertIn(('shoot', 'right'), result.scores)
        self.assertNotIn(('shoot', 'up'), result.scores)
        self.assertGreater(
            max(
                result.scores[direction]
                for direction in ('up', 'right', 'down')
            ),
            result.scores[('shoot', 'right')],
        )

    def test_ranking_exports_same_single_search_without_changing_move_score_logic(
        self,
    ):
        state = snapshot()
        cache = TurnSearchCache()
        from snake_bot.search_engine import SearchResult

        result = SearchResult(
            {'up': 100, 'down': 90, ('shoot', 'right'): 20}, 4, 10, 0, 1
        )
        with patch.object(
            IterativeSearchEngine, 'search', return_value=result
        ) as search:
            ranked = rank_deep_moves(
                state, 'A', ['up', 'down'], cache=cache, search_time_ms=18
            )
        search.assert_called_once()
        self.assertIs(cache.action_search, result)
        self.assertEqual(
            {row['direction']: row['search_score'] for row in ranked},
            {'up': 100, 'down': 90},
        )

    def test_tactical_shoot_prices_5000_for_shooter_and_opponent_perspectives(
        self,
    ):
        from snake_bot.move_search import (
            best_reply_score,
            two_ply_score,
            fast_two_ply_score,
        )
        from snake_bot.bot_weights import get_weights

        state = snapshot()
        values = {}
        for action in ('up', ('shoot', 'right')):
            cache = TurnSearchCache()
            with patch.object(cache, 'legal_actions', return_value=(action,)):
                values[str(action)] = best_reply_score(state, 'A', cache)
        half_cost_value = 5000 * get_weights()['deep_score_point_value'] // 2
        self.assertGreater(
            values['up'] - values[str(('shoot', 'right'))], half_cost_value
        )
        for scorer in (two_ply_score, fast_two_ply_score):
            replies = {}
            for enemy_action in ('up', ('shoot', 'left')):
                cache = TurnSearchCache()
                original = cache.legal_actions
                with patch.object(
                    cache,
                    'legal_actions',
                    side_effect=lambda observed, side: (
                        (enemy_action,)
                        if side == 'B'
                        else original(observed, side)
                    ),
                ):
                    replies[str(enemy_action)] = scorer(
                        state, 'A', 'right', None, 0, 0, cache
                    )[0]
            self.assertGreater(
                replies[str(('shoot', 'left'))] - replies['up'],
                half_cost_value,
            )
