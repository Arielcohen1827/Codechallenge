import unittest
from dataclasses import replace
from snake_bot.snake_state import GameState
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.move_search import state_key
from tools import bot_lab


def state(**changes):
    base = GameState(rows=20, cols=30, board=(), side='A', enemy='B',
        snakes={'A': ((5, 5), (5, 4)), 'B': ((5, 10), (6, 10), (7, 10))},
        food=frozenset(), scores={'A': 6000, 'B': 100}, remaining_moves=200,
        reliable_tails=frozenset({'A', 'B'}), crashed_a=False, crashed_b=False)
    return replace(base, **changes)


class CompactV89Tests(unittest.TestCase):
    def test_new_mechanics_in_cache_identity(self):
        original = state()
        for field in ('bombs_large', 'bombs_small', 'shot_food'):
            changed = replace(original, **{field: frozenset({(10, 10)})})
            self.assertNotEqual(state_key(original), state_key(changed))
            self.assertNotEqual(IterativeSearchEngine.from_game_state(original)[1],
                                IterativeSearchEngine.from_game_state(changed)[1])

    def test_compact_actions_respect_shoot_threshold(self):
        for score in (4999, 5000):
            engine, compact = IterativeSearchEngine.from_game_state(state(scores={'A': score, 'B': 0}))
            actions = engine.available_actions(compact)
            self.assertEqual(('shoot', 'right') in actions, score >= 5000)

    def test_compact_and_game_transitions_equivalent(self):
        from snake_bot.snake_state import apply_action
        cases = [(state(bombs_large=frozenset({(7, 7)})), 'move', 'right'),
                 (state(shot_food=frozenset({(5, 6)})), 'move', 'right'),
                 (state(), 'shoot', 'right'), (state(), 'shoot', 'up'),
                 (state(remaining_moves=101, bombs_small=frozenset({(12, 12)})), 'move', 'right')]
        for before, kind, direction in cases:
            with self.subTest(kind=kind, direction=direction):
                engine, compact = IterativeSearchEngine.from_game_state(before)
                expected = IterativeSearchEngine.from_game_state(apply_action(before, kind, direction))[1]
                self.assertEqual(engine.apply_action(compact, kind, direction), expected)

    def test_large_rectangular_board_roundtrips_emoji(self):
        from snake_bot.snake_state import parse_state
        for rows, cols in ((20, 20), (20, 30), (30, 20), (30, 30)):
            before = state(rows=rows, cols=cols, bombs_large=frozenset({(12, 12)}),
                           bombs_small=frozenset({(14, 14)}), shot_food=frozenset({(2, 2)}))
            observed = parse_state(bot_lab.make_turn_data(before, 'smoke', 0))
            self.assertEqual(observed.bombs_large, before.bombs_large)
            self.assertEqual(observed.bombs_small, before.bombs_small)
            self.assertEqual(observed.shot_food, before.shot_food)
            self.assertEqual((observed.rows, observed.cols), (rows, cols))

    def test_lab_shoot_delegates_rules(self):
        from snake_bot.snake_state import apply_action
        before = state()
        after, metrics = bot_lab.lab_transition(before, 'right', action_type='shoot')
        self.assertEqual(after, apply_action(before, 'shoot', 'right'))
        self.assertEqual(metrics['shots_A'], 1)
        self.assertEqual(after.body('B'), ((5, 10),))

    def test_lab_respawns_avoid_blast_zones_and_preserve_food(self):
        import random
        before = state(bombs_large=frozenset({(12, 12)}), shot_food=frozenset({(2, 2)}))
        after = bot_lab.spawn_food(before, random.Random(2), 5)
        self.assertEqual(after.shot_food, before.shot_food)
        for cell in set(after.food_values) | set(after.pickups):
            self.assertGreater(max(abs(cell[0]-12), abs(cell[1]-12)), 2)
    def test_tactical_action_cache_and_shot_food_identity(self):
        from snake_bot.move_search import TurnSearchCache
        from snake_bot.food_planner import objective_kind, objective_reward
        cache = TurnSearchCache()
        before = state(shot_food=frozenset({(3, 3)}), multipliers={'A': 3, 'B': 1})
        self.assertIn(('shoot', 'right'), cache.legal_actions(before, 'A'))
        after = cache.apply(before, 'A', ('shoot', 'right'))
        self.assertEqual(after.scores['A'], 1000)
        self.assertEqual(objective_kind(before, (3, 3)), 'shot_food')
        self.assertEqual(objective_reward(before, 'A', (3, 3)), 300)
        self.assertNotIn(('shoot', 'right'), cache.legal_actions(after, 'A'))

    def test_bounded_smoke_chain_30x30(self):
        from snake_bot.snake_state import apply_action, parse_state
        before = state(rows=30, cols=30, bombs_small=frozenset({(5, 7)}),
                       snakes={'A': ((5, 5), (5, 4)), 'B': ((3, 6), (4, 6), (4, 7))})
        parsed = parse_state(bot_lab.make_turn_data(before, 'smoke', 0))
        self.assertEqual(parsed.bombs_small, frozenset({(5, 7)}))
        bombed = apply_action(before, 'move', 'right')
        self.assertEqual(bombed.scores['A'], 0)
        self.assertEqual(bombed.head('A'), (5, 6))
        # B shoots down into A's head; A keeps its head, detached tail becomes *.
        ready = replace(bombed, scores={'A': 0, 'B': 6000})
        shot = apply_action(ready, 'shoot', 'down')
        self.assertEqual(shot.body('A'), ((5, 6),))
        self.assertIn((5, 5), shot.shot_food)
        eaten = apply_action(shot, 'move', 'left')
        self.assertEqual(eaten.head('A'), (5, 5))
        self.assertEqual(len(eaten.body('A')), 2)
        self.assertEqual(eaten.scores['A'], 100)
        self.assertNotIn((5, 5), eaten.shot_food)
    def test_lab_bomb_respawns_safe_or_remain_pending_and_expire(self):
        import random
        before = state(pending_bomb_respawns=('large', 'small'),
                       shot_food=frozenset({(12, 12)}))
        after = bot_lab.respawn_bombs(before, random.Random(8))
        self.assertEqual((len(after.bombs_large), len(after.bombs_small)), (1, 1))
        self.assertEqual(after.pending_bomb_respawns, ())
        for bombs, radius in ((after.bombs_large, 2), (after.bombs_small, 1)):
            for bomb in bombs:
                for protected in (before.head('A'), before.head('B'), (12, 12)):
                    self.assertGreater(max(abs(bomb[0]-protected[0]), abs(bomb[1]-protected[1])), radius)
        full = state(rows=1, cols=2, snakes={'A': ((0, 0),), 'B': ((0, 1),)},
                     pending_bomb_respawns=('large', 'small'))
        self.assertEqual(bot_lab.respawn_bombs(full, random.Random(1)).pending_bomb_respawns,
                         ('large', 'small'))
        initial = bot_lab.initial_state(2, rows=20, cols=30, max_turns=400)
        self.assertEqual((len(initial.bombs_large), len(initial.bombs_small)), (2, 5))
        for cell in set(initial.food_values) | set(initial.pickups) | set(initial.shot_food):
            self.assertFalse(bot_lab.in_blast_zone(initial, cell))
        expired = bot_lab.respawn_bombs(replace(after, remaining_moves=100), random.Random(1))
        self.assertFalse(expired.bombs_large | expired.bombs_small)
        self.assertFalse(expired.pending_bomb_respawns)

    def test_tactical_search_models_rival_shots_with_existing_evaluator(self):
        from snake_bot.move_search import TurnSearchCache, two_ply_score, fast_two_ply_score
        before = state(scores={'A': 6000, 'B': 6000})
        for scorer in (two_ply_score, fast_two_ply_score):
            cache = TurnSearchCache()
            value, _ = scorer(before, 'A', 'down', None, 0, 0, cache)
            self.assertIsInstance(value, (int, float))
            self.assertTrue(any('shoot' in key for key in cache.after if isinstance(key, tuple)))
    def test_root_search_accepts_official_tuple_and_protocol_dict_actions(self):
        before = state(rows=8, cols=12)
        for legal in ([('move', 'down'), ('shoot', 'right')],
                      [{'action_type': 'move', 'direction': 'down'},
                       {'action_type': 'shoot', 'direction': 'right'}],
                      [{'action': 'move', 'data': {'direction': 'down'}},
                       {'action': 'shoot', 'data': {'direction': 'right'}}]):
            engine, compact = IterativeSearchEngine.from_game_state(before, time_budget_ms=1000, max_depth=2)
            result = engine.search(compact, 'A', legal)
            self.assertIn('down', result.scores)
            self.assertIn(('shoot', 'right'), result.scores)

    def test_tactical_cache_distinguishes_board_dimensions(self):
        self.assertNotEqual(state_key(state(rows=20, cols=30)), state_key(state(rows=30, cols=20)))

    def test_compact_equivalence_for_B_and_v7_number_growth(self):
        from snake_bot.snake_state import apply_action
        for before, kind, direction in (
            (state(side='B', enemy='A', scores={'A': 0, 'B': 6000}), 'shoot', 'left'),
            (state(side='B', enemy='A', shot_food=frozenset({(5, 11)})), 'move', 'right'),
            (state(food_values={(5, 6): 2, (2, 2): 1}, food=frozenset({(2, 2)}), next_food_digit=1), 'move', 'right'),
            (state(remaining_moves=101, side='B', enemy='A', bombs_large=frozenset({(12, 12)})), 'shoot', 'up')):
            engine, compact = IterativeSearchEngine.from_game_state(before)
            after = apply_action(before, kind, direction)
            self.assertEqual(engine.apply_action(compact, kind, direction),
                             IterativeSearchEngine.from_game_state(after)[1])
    def test_shot_food_planning_survives_parser_separation_and_growth(self):
        from snake_bot.food_planner import build_food_plans, choose_food_plan, simulate_path_to_food
        from snake_bot.snake_state import parse_state
        before = state(shot_food=frozenset({(5, 6)}), multipliers={'A': 3, 'B': 1})
        parsed = parse_state(bot_lab.make_turn_data(before, 'shot-plan', 0))
        self.assertFalse(parsed.food)
        plans = build_food_plans(parsed, 'A')
        plan = choose_food_plan(parsed, 'A', plans=plans)
        self.assertEqual(plan.objective_kind, 'shot_food')
        self.assertEqual(plan.objective_reward, 300)
        self.assertEqual(plan.first_move, 'right')
        after = simulate_path_to_food(parsed, 'A', plan.path)
        self.assertEqual(after.scores['A'], before.scores['A'] + 300)
        self.assertEqual(len(after.body('A')), len(parsed.body('A')) + 1)
        self.assertFalse(after.shot_food)

    def test_search_prices_stationary_shoot_at_last_turn_without_move_point(self):
        before = state(rows=8, cols=12, scores={'A': 5000, 'B': 1}, remaining_moves=1)
        engine, compact = IterativeSearchEngine.from_game_state(before, time_budget_ms=1000, max_depth=2)
        result = engine.search(compact, 'A', [('move', 'down'), ('shoot', 'right')])
        self.assertGreater(result.scores['down'], 0)
        self.assertLess(result.scores[('shoot', 'right')], 0)
        shot = engine.apply_action(compact, 'shoot', 'right')
        self.assertEqual(shot.scores, (0, 1))
        self.assertEqual(shot.bodies[0], compact.bodies[0])
        self.assertEqual(shot.bodies[1], (compact.bodies[1][0],))

    def test_cache_move_action_and_shoot_stay_distinct_at_same_direction(self):
        from snake_bot.move_search import TurnSearchCache
        before = state()
        cache = TurnSearchCache()
        moved = cache.apply_action(before, 'A', 'move', 'right')
        shot = cache.apply_action(before, 'A', 'shoot', 'right')
        self.assertEqual(moved.head('A'), (5, 6))
        self.assertEqual(shot.head('A'), (5, 5))
        self.assertEqual(moved.scores['A'], 6001)
        self.assertEqual(shot.scores['A'], 1000)
        self.assertEqual(cache.apply_action(before, 'A', 'shoot', 'right'), shot)

    def test_lab_respawn_does_not_replenish_consumed_shot_food(self):
        import random
        from snake_bot.snake_state import apply_action
        before = state(shot_food=frozenset({(5, 6)}))
        consumed = apply_action(before, 'move', 'right')
        replenished = bot_lab.spawn_food(consumed, random.Random(4), 5)
        self.assertFalse(replenished.shot_food)
        self.assertTrue(replenished.food_values)
        self.assertEqual(replenished.scores['A'], 6100)
    def test_multi_step_path_preserves_bombs_and_unconsumed_shot_food(self):
        from snake_bot.food_planner import simulate_path_to_food
        before = state(bombs_large=frozenset({(12, 12)}), bombs_small=frozenset({(15, 15)}),
                       shot_food=frozenset({(5, 7), (2, 2)}), pending_bomb_respawns=('small',))
        after = simulate_path_to_food(before, 'A', ((5, 5), (5, 6), (5, 7)))
        self.assertEqual(after.scores['A'], 6101)
        self.assertEqual(after.bombs_large, before.bombs_large)
        self.assertEqual(after.bombs_small, before.bombs_small)
        self.assertEqual(after.shot_food, frozenset({(2, 2)}))
        self.assertEqual(after.pending_bomb_respawns, ('small',))

    def test_shot_food_uses_existing_lost_race_risk_with_numbered_food_present(self):
        from snake_bot.food_planner import hot_lost_food_penalty
        before = state(rows=15, cols=15, side='B', enemy='A', scores={'A': 0, 'B': 0},
                       snakes={'A': ((9, 9), (9, 8), (9, 7)), 'B': ((8, 9), (8, 8), (8, 7))},
                       shot_food=frozenset({(9, 10), (10, 10), (14, 6)}))
        self.assertGreater(hot_lost_food_penalty(before, 'B', 'right'), 0)
        mixed = replace(before, food_values={(0, 0): 1}, food=frozenset({(0, 0)}), next_food_digit=1)
        self.assertGreater(hot_lost_food_penalty(mixed, 'B', 'right'), 0)
        self.assertEqual(hot_lost_food_penalty(mixed, 'B', 'up'), 0)
