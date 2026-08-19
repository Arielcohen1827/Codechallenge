import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    'food_base_value': 10_000,
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
    'current_target_bonus': 1800,
    'safe_bonus': 2000,
    'acceptable_risk_bonus': 700,
    'dangerous_penalty': 6500,
    'lost_race_penalty': 18_000,
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
    'hot_lost_food_base': 2500,
    'hot_lost_food_enemy_bonus': 1200,
    'fallback_progress_value': 1000,
    'fallback_hunger_progress_value': 80,
    'fallback_food_distance_cost': 80,
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
