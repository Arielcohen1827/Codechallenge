import unittest
from dataclasses import replace
from unittest.mock import patch

import food_planner
import run
from food_planner import (
    FoodPlan,
    advantage_control_move,
    best_numbered_copy,
    build_food_plan,
    center_distance,
    center_control_score,
    classify_safety,
    fallback_denial_score,
    fallback_hunger_food_score,
    edge_distance,
    estimated_multiplier_points,
    food_plan_risk_penalty,
    food_set_race,
    current_food_race,
    rank_survival_moves,
    sequence_gate_hold_move,
    survival_move_analysis,
    survival_position_score,
    territory_control_score,
    winning_suicide_move,
)
from snake_brain import SnakeBrain, final_safe_direction
from snake_state import GameState, apply_move, legal_moves, parse_state
from test_run import turn


def state_from(board, **changes):
    return parse_state(turn(board, **changes))


def open_state(**changes):
    values = {
        'rows': 5,
        'cols': 7,
        'board': tuple(' ' * 7 for _ in range(5)),
        'side': 'A',
        'enemy': 'B',
        'snakes': {'A': ((2, 1),), 'B': ((2, 5),)},
        'food': frozenset({(0, 3)}),
        'scores': {'A': 0, 'B': 0},
        'remaining_moves': 40,
        'reliable_tails': frozenset({'A', 'B'}),
    }
    values.update(changes)
    return GameState(**values)


def sample_plan(**changes):
    values = {
        'food': (1, 1),
        'path': ((2, 1), (1, 1)),
        'our_distance': 1,
        'enemy_distance': 4,
        'race_margin': 3,
        'space_after': 20,
        'exits_after': 2,
        'tail_reachable_after': True,
        'next_food_distance': 3,
        'safety': 'SAFE',
        'value': 10_000,
    }
    values.update(changes)
    return FoodPlan(**values)


class TestGlobalCopyStrategy(unittest.TestCase):
    def test_global_race_uses_rivals_best_copy(self):
        state = state_from(
            '|     1B|\n'
            '| 2 3   |\n'
            '|A  1   |\n'
            '| 4   5 |\n'
            '|       |'
        )

        race = food_set_race(state, 'A')
        plan = build_food_plan(state, 'A', (2, 3), None, 0, food_race=race)

        self.assertEqual(race['our_food'], (2, 3))
        self.assertEqual(race['enemy_food'], (0, 5))
        self.assertEqual((race['our_distance'], race['enemy_distance']), (3, 1))
        self.assertLess(race['margin'], 0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.enemy_distance, 1)
        self.assertLess(plan.race_margin, 0)
        self.assertEqual(current_food_race(state, 'A')[1:], (3, 1, -2))

    def test_best_copy_and_unreachable_race_edges(self):
        state = open_state(food=frozenset({(0, 1), (0, 5)}))
        choice = best_numbered_copy(state, 'A', state.food)
        self.assertEqual(choice[0], (0, 1))

        no_head = replace(state, snakes={'A': (), 'B': state.body('B')})
        self.assertIsNone(best_numbered_copy(no_head, 'A', state.food))
        race = food_set_race(no_head, 'A')
        self.assertIsNone(race['our_distance'])
        self.assertEqual(race['margin'], -99)

    def test_advantage_control_scores_all_copy_access(self):
        state = open_state(
            food=frozenset({(1, 1), (0, 5)}),
            scores={'A': 5000, 'B': 0},
        )
        deep = [
            {
                'direction': 'up', 'score': 100, 'safety': 'SAFE',
                'our_replies': 3, 'exits': 3, 'voronoi_ours': 8,
                'voronoi_enemy': 2, 'enemy_food': False,
            },
            {
                'direction': 'right', 'score': 50, 'safety': 'SAFE',
                'our_replies': 3, 'exits': 3, 'voronoi_ours': 5,
                'voronoi_enemy': 4, 'enemy_food': True,
            },
        ]
        access = [
            {(1, 1): 5, (0, 5): 1},
            {(1, 1): 6, (0, 5): 3},
            {(1, 1): 4, (0, 5): 1},
        ]
        with (
            patch.object(food_planner, 'advantage_control_context', return_value={'active': True}),
            patch.object(food_planner, 'food_distances', side_effect=access),
        ):
            result = advantage_control_move(
                state,
                'A',
                ['up', 'right'],
                deep,
                reference_move='right',
            )

        self.assertTrue(result['override'])
        self.assertEqual(result['direction'], 'up')
        self.assertEqual(result['moves'][0]['enemy_food_distance_before'], 1)
        self.assertTrue(result['moves'][0]['secured_food'])
        self.assertEqual(result['moves'][0]['copies_denied'], 2)


class TestTerminalCloseout(unittest.TestCase):
    def test_suicide_requires_more_than_1500_points(self):
        winning = open_state(
            snakes={'A': ((0, 2),), 'B': ((4, 6),)},
            scores={'A': 1501, 'B': 0},
        )
        tied = replace(winning, scores={'A': 1500, 'B': 0})

        self.assertEqual(winning_suicide_move(winning, 'A'), 'up')
        self.assertIsNone(winning_suicide_move(tied, 'A'))

    def test_brain_preserves_a_winning_suicide(self):
        data = turn(
            '| A   |\n|     |\n|    B|',
            score_1=2000,
            score_2=0,
            game_id='closeout',
        )
        brain = SnakeBrain(enable_debug=False)

        direction = brain.choose_move(data)

        self.assertEqual(direction, 'up')
        self.assertEqual(brain.safe_direction(data, direction), 'up')
        self.assertEqual(final_safe_direction(data, direction), 'up')
        self.assertEqual(brain.decision_debug('closeout')['reason'], 'winning_suicide')

    def test_no_body_or_head_has_no_suicide(self):
        state = open_state(snakes={'A': (), 'B': ((2, 5),)}, scores={'A': 3000, 'B': 0})
        self.assertIsNone(winning_suicide_move(state, 'A'))

    def test_suicide_skips_hash_walls_and_a_releasing_tail(self):
        blocked_up = open_state(
            snakes={'A': ((2, 2), (3, 2), (3, 1)), 'B': ((4, 6),)},
            walls=frozenset({(1, 2)}),
            scores={'A': 3000, 'B': 0},
        )
        self.assertEqual(winning_suicide_move(blocked_up, 'A'), 'down')

        all_hash = replace(
            blocked_up,
            snakes={'A': ((2, 2),), 'B': ((4, 6),)},
            walls=frozenset({(1, 2), (3, 2), (2, 1), (2, 3)}),
        )
        self.assertIsNone(winning_suicide_move(all_hash, 'A'))

        length_two = replace(
            blocked_up,
            snakes={'A': ((2, 2), (3, 2)), 'B': ((4, 6),)},
            walls=frozenset({(1, 2)}),
        )
        self.assertEqual(winning_suicide_move(length_two, 'A'), 'down')


class TestSequenceAndRiskBranches(unittest.TestCase):
    def test_sequence_hold_uses_global_next_race(self):
        state = state_from(
            '|     B|\n'
            '| A1  2|\n'
            '|  1 2 |\n'
            '| 3 4 5|',
            score_1=1,
            remaining=20,
        )
        plan = sample_plan(food=(1, 2), path=((1, 1), (1, 2)))
        deep = [
            {'direction': 'up', 'score': 20, 'safety': 'SAFE', 'our_replies': 3},
            {'direction': 'down', 'score': 10, 'safety': 'SAFE', 'our_replies': 2},
            {'direction': 'right', 'score': 5, 'safety': 'SAFE', 'our_replies': 2},
        ]
        next_positions = state.numbered_groups(limit=2)[1][1]
        with (
            patch.object(
                food_planner,
                'food_set_race',
                return_value={'enemy_distance': 5},
            ),
            patch.object(
                food_planner,
                'best_numbered_copy',
                return_value=(next_positions[0], ((1, 2), next_positions[0]), 2, 1, -1),
            ),
        ):
            move = sequence_gate_hold_move(
                state,
                'A',
                ['up', 'down', 'right'],
                plan,
                deep,
            )

        self.assertIn(move, {'up', 'down'})

    def test_risk_penalties_and_safety_classes(self):
        state = open_state()
        self.assertEqual(food_plan_risk_penalty(state, (0, 3), 1, 'SAFE', 100, 3), 0)
        penalty = food_plan_risk_penalty(state, (0, 3), 5, 'ACCEPTABLE_RISK', 20, 1)
        self.assertGreater(penalty, 0)
        self.assertLess(center_control_score(state, (0, 0)), center_control_score(state, (2, 3)))
        self.assertIsInstance(territory_control_score(state, 'A'), int)

        with (
            patch.object(food_planner, 'flood_region', return_value={(2, 1)}),
            patch.object(food_planner, 'count_exits', return_value=0),
            patch.object(food_planner, 'can_reach_tail', return_value=False),
        ):
            self.assertEqual(classify_safety(state, 'A')[0], 'SUICIDAL')

    def test_empty_and_expired_helpers(self):
        state = open_state(food=frozenset())
        no_head = replace(state, snakes={'A': (), 'B': state.body('B')})

        self.assertEqual(classify_safety(no_head, 'A'), ('SUICIDAL', 0, 0, False))
        self.assertEqual(edge_distance(state, None), 0)
        self.assertEqual(center_distance(state, None), state.rows + state.cols)
        self.assertEqual(center_control_score(state, None), -10_000)
        self.assertEqual(territory_control_score(no_head, 'A'), 0)
        self.assertEqual(estimated_multiplier_points(state, 'A'), 0)
        self.assertIsNone(current_food_race(state, 'A'))
        self.assertEqual(survival_position_score(no_head, 'A'), -1_000_000)

    def test_sequence_hold_rejects_invalid_contexts(self):
        state = open_state()
        plan = sample_plan()
        self.assertIsNone(sequence_gate_hold_move(state, 'A', ['up'], None, []))

        numbered = replace(
            state,
            food_values={(0, 3): 1, (1, 3): 2},
            next_food_digit=1,
            remaining_moves=100,
        )
        self.assertIsNone(sequence_gate_hold_move(numbered, 'A', ['up'], plan, []))

        no_lead = replace(numbered, remaining_moves=20, scores={'A': 0, 'B': 0})
        self.assertIsNone(sequence_gate_hold_move(no_lead, 'A', ['up'], plan, []))


class TestFallbackAndSurvivalBranches(unittest.TestCase):
    def test_hunger_and_global_denial_helpers(self):
        state = open_state(
            food=frozenset({(0, 1), (0, 5)}),
            food_values={(0, 1): 1, (0, 5): 1, (1, 3): 2},
            next_food_digit=1,
        )
        after = apply_move(state, 'up', 'A')
        hunger_score = fallback_hunger_food_score(state, 'A', after, after.head('A'), 20)
        denial_score = fallback_denial_score(state, 'A', after, after.head('A'))
        self.assertGreaterEqual(hunger_score, 0)
        self.assertGreaterEqual(denial_score, 0)
        self.assertEqual(fallback_hunger_food_score(state, 'A', after, after.head('A'), 2), 0)

    def test_survival_analysis_invalid_and_normal_paths(self):
        state = open_state()
        invalid = survival_move_analysis(state, 'A', 'bogus')
        self.assertEqual(invalid['score'], -1_000_000)

        analyses = rank_survival_moves(state, 'A', legal_moves(state, 'A'))
        self.assertTrue(analyses)
        self.assertGreaterEqual(analyses[0]['score'], analyses[-1]['score'])
        self.assertIn('enemy_reply', analyses[0])

    def test_survival_enemy_trapped_and_suicidal_branches(self):
        state = open_state()
        with (
            patch.object(food_planner, 'legal_moves', side_effect=[['up'], []]),
            patch.object(food_planner, 'survival_position_score', return_value=123),
        ):
            trapped = survival_move_analysis(state, 'A', 'up')
        self.assertEqual(trapped['reply_count'], 99)

        with (
            patch.object(food_planner, 'legal_moves', side_effect=[['up'], ['left']]),
            patch.object(food_planner, 'classify_safety', return_value=('SUICIDAL', 1, 0, False)),
        ):
            suicidal = survival_move_analysis(state, 'A', 'up')
        self.assertEqual(suicidal['score'], -1_000_000)

    def test_survival_position_score_penalty_paths(self):
        state = open_state(
            snakes={
                'A': ((2, 2), (2, 1), (2, 0), (1, 0)),
                'B': ((4, 6),),
            }
        )
        with (
            patch.object(food_planner, 'flood_region', return_value={(2, 2)}),
            patch.object(food_planner, 'count_exits', return_value=1),
            patch.object(food_planner, 'legal_moves', return_value=['up']),
            patch.object(food_planner, 'can_reach_tail', return_value=True),
            patch.object(food_planner, 'classify_safety', return_value=('ACCEPTABLE_RISK', 1, 1, True)),
        ):
            score = survival_position_score(state, 'A')
        self.assertIsInstance(score, int)


if __name__ == '__main__':
    unittest.main()
