import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    'food_base_value': 10_000,
    'objective_point_value': 22,
    'multiplier_base_value': 1800,
    'multiplier_future_point_value': 7,
    'multiplier_hunger_penalty': 650,
    'multiplier_stack_penalty': 1200,
    'multiplier_gap_value': 4500,
    'numbered_food_hunger_value': 500,
    'sequence_hold_max_remaining': 30,
    'sequence_hold_min_enemy_distance': 3,
    'sequence_hold_score_buffer': 200,
    'food_distance_cost': 550,
    'race_margin_value': 700,
    'hunger_value': 220,
    'space_after_value': 16,
    'exits_after_value': 260,
    'tail_reachable_bonus': 450,
    'center_unit_value': 80,
    'edge_depth_value': 450,
    'center_edge0_penalty': 3200,
    'center_edge1_penalty': 1000,
    'next_food_value': 170,
    'chain_food_value': 260,
    'chain_race_value': 180,
    'chain_close_food_bonus': 900,
    'sequence_reward_point_value': 4,
    'sequence_setup_base_value': 9000,
    'sequence_setup_reward_point_value': 18,
    'sequence_setup_readiness_value': 1200,
    'sequence_setup_rank_penalty': 4000,
    'sequence_setup_late_slack': 3,
    'sequence_setup_hunger_value': 200,
    'hunger_close_food_value': 90,
    'denial_food_bonus': 1200,
    'denial_chain_bonus': 650,
    'rival_food_threat_value': 360,
    'rival_chain_threat_value': 260,
    'rival_chain_ignore_penalty': 1200,
    'current_target_bonus': 500,
    'safe_bonus': 2000,
    'acceptable_risk_bonus': 700,
    'dangerous_penalty': 6500,
    'lost_race_penalty': 18_000,
    'non_immediate_edge_food_penalty': 5000,
    'low_space_food_penalty': 14_000,
    'long_low_space_food_penalty': 10_000,
    'one_exit_food_penalty': 8000,
    'position_edge0_penalty': 6500,
    'position_edge1_penalty': 1800,
    'position_edge2_penalty': 600,
    'position_corner_penalty': 2500,
    'zero_escape_penalty': 9000,
    'delayed_zero_penalty': 18_000,
    'one_escape_penalty': 2500,
    'small_region_penalty': 1400,
    'one_exit_penalty': 3500,
    'enemy_no_reply_threat': 80_000,
    'territory_gap_limit': 35,
    'territory_gap_penalty': 250,
    'territory_gap_penalty_cap': 25_000,
    'territory_control_value': 35,
    'territory_control_cap': 7000,
    'food_plan_switch_margin': 1200,
    'food_plan_big_switch_margin': 5000,
    'plan_tactical_override_margin': 1800,
    'adjacent_plan_tactical_override_margin': 10_000,
    'one_escape_tactical_override_margin': 450,
    'edge_tactical_override_margin': 700,
    'survival_region_value': 80,
    'survival_exit_value': 2200,
    'survival_reply_value': 1400,
    'survival_tail_bonus': 1200,
    'survival_immediate_food_bonus': 6500,
    'enemy_trapped_future_bonus': 220_000,
    'future_no_reply_penalty': 160_000,
    'future_one_reply_penalty': 28_000,
    'future_corridor_penalty': 12_000,
    'future_small_region_penalty': 1800,
    'future_enemy_food_penalty': 4500,
    'endgame_food_risk_window': 2,
    'endgame_food_bonus': 35_000,
    'voronoi_cell_value': 95,
    'voronoi_center_value': 45,
    'voronoi_food_value': 4200,
    'voronoi_frontier_penalty': 80,
    'deep_region_value': 55,
    'deep_exit_value': 1900,
    'deep_reply_value': 1200,
    'deep_tail_bonus': 900,
    'deep_food_now_bonus': 9000,
    'deep_pickup_now_bonus': 8500,
    'deep_score_point_value': 18,
    'deep_wrong_food_penalty': 140_000,
    'deep_enemy_trapped_bonus': 120_000,
    'deep_enemy_food_penalty': 6500,
    'deep_enemy_pickup_penalty': 14_000,
    'deep_no_reply_penalty': 180_000,
    'deep_one_reply_penalty': 30_000,
    'deep_risk_penalty': 7000,
    'deep_danger_penalty': 20_000,
    'deep_small_region_penalty': 1400,
    'deep_plan_override_margin': 2600,
    'deep_hot_lost_multiplier': 4,
    'deep_safe_preference_margin': 30_000,
    'deep_lost_plan_step_penalty': 42_000,
    'deep_extra_depth_margin': 2200,
    'hot_lost_food_base': 2500,
    'hot_lost_food_enemy_bonus': 1200,
    'fallback_progress_value': 1000,
    'fallback_hunger_progress_value': 80,
    'fallback_food_distance_cost': 80,
    'fallback_hunger_food_progress': 120,
    'fallback_hunger_food_distance': 55,
    'fallback_hunger_food_race': 90,
    'fallback_denial_progress': 120,
    'fallback_denial_reach_bonus': 500,
    'fallback_cycle_penalty': 500,
    'fallback_danger_penalty': 10_000,
}

ACTIVE_WEIGHTS = dict(DEFAULT_WEIGHTS)


def get_weights():
    return ACTIVE_WEIGHTS


def reset_active_weights():
    ACTIVE_WEIGHTS.clear()
    ACTIVE_WEIGHTS.update(DEFAULT_WEIGHTS)


def set_active_weights(weights):
    reset_active_weights()
    ACTIVE_WEIGHTS.update({key: int(value) for key, value in weights.items() if key in DEFAULT_WEIGHTS})


def load_weights(path):
    raw = json.loads(Path(path).read_text())
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({key: int(value) for key, value in raw.items() if key in DEFAULT_WEIGHTS})
    return weights


def load_active_weights(path='weights/active_weights.json'):
    weight_path = Path(path)
    if not weight_path.exists():
        return False
    set_active_weights(load_weights(weight_path))
    return True


def save_weights(path, weights):
    weight_path = Path(path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: int(weights[key]) for key in sorted(DEFAULT_WEIGHTS)}
    weight_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
