from collections import deque
from dataclasses import dataclass

from bot_weights import get_weights
from snake_state import (
    CRASH_PENALTY,
    FOOD_SCORE,
    RIVAL_CRASH_REWARD,
    GameState,
    apply_move,
    blocked_for_path,
    count_exits,
    direction_between,
    flood_region,
    legal_moves,
    manhattan,
    neighbors,
    shortest_distance,
    shortest_path,
    step,
    temporal_shortest_distance,
    temporal_shortest_path,
)


@dataclass(frozen=True)
class FoodPlan:
    food: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    our_distance: int
    enemy_distance: int | None
    race_margin: int
    space_after: int
    exits_after: int
    tail_reachable_after: bool
    next_food_distance: int | None
    safety: str
    value: int
    chain_score: int = 0
    denial_score: int = 0
    rival_threat: int = 0
    tempo_urgency: int = 0
    temporal_tail_path: bool = False
    endgame_acceptable: bool = False
    objective_kind: str = 'food'
    objective_reward: int = 100
    sequence_rank: int = 0

    @property
    def first_move(self):
        if len(self.path) < 2:
            return None
        return direction_between(self.path[0], self.path[1])


def simulate_path_to_food(state, side, path):
    current = state
    for cell in path[1:]:
        direction = direction_between(current.head(side), cell)
        if direction is None or direction not in legal_moves(current, side):
            return None
        current = apply_move(current, direction, side)
        current = GameState(
            rows=current.rows,
            cols=current.cols,
            board=current.board,
            side=side,
            enemy=current.enemy if current.enemy != side else ('B' if side == 'A' else 'A'),
            snakes=current.snakes,
            food=current.food,
            scores=current.scores,
            remaining_moves=current.remaining_moves,
            reliable_tails=current.reliable_tails,
            food_values=current.food_values,
            next_food_digit=current.next_food_digit,
            pickups=current.pickups,
            multipliers=current.multipliers,
        )
    return current


def can_reach_tail(state, side):
    snake = state.body(side)
    head = state.head(side)
    if not snake or head is None:
        return False
    return temporal_shortest_path(state, head, snake[-1], side) is not None


def distance_map(state, start, side):
    if start is None:
        return {}
    blocked = blocked_for_path(state, side)
    blocked.discard(start)
    q = deque([start])
    distances = {start: 0}
    while q:
        current = q.popleft()
        for nb in neighbors(current, state.rows, state.cols):
            if nb in distances or nb in blocked:
                continue
            distances[nb] = distances[current] + 1
            q.append(nb)
    return distances


def tempo_race_margin(state, side, our_distance, enemy_distance):
    if enemy_distance is None:
        return 6
    margin = enemy_distance - our_distance
    if margin == 0:
        return 1 if side == state.side else -1
    return margin


def wins_food_race(state, side, our_distance, enemy_distance):
    return tempo_race_margin(state, side, our_distance, enemy_distance) >= 0


def remaining_after_our_food(state, our_distance):
    return state.remaining_moves - max(1, our_distance * 2 - 1)


def endgame_accepts_food_risk(state, our_distance, race_margin, safety):
    if race_margin < 0 or safety == 'SAFE':
        return False
    weights = get_weights()
    after_food_remaining = remaining_after_our_food(state, our_distance)
    return after_food_remaining <= weights['endgame_food_risk_window']


def nearest_next_food_distance(state, side, consumed_food, distances=None):
    head = state.head(side)
    if head is None:
        return None
    if state.food_values:
        sequence = state.numbered_sequence(limit=1)
        if not sequence:
            return None
        path = temporal_shortest_path(state, head, sequence[0][1], side)
        return None if path is None else len(path) - 1
    distances = distances if distances is not None else distance_map(state, head, side)
    food_distances = [
        distances.get(food)
        for food in state.food
        if food != consumed_food and distances.get(food) is not None
    ]
    return min(food_distances) if food_distances else None


def food_chain_score(state, side, consumed_food, hunger, distances=None):
    head = state.head(side)
    if head is None:
        return 0

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    weights = get_weights()
    if state.food_values:
        current = state
        total = 0
        for rank in range(3):
            sequence = current.numbered_sequence(limit=1)
            if not sequence:
                break
            digit, food = sequence[0]
            current_head = current.head(side)
            path = temporal_shortest_path(current, current_head, food, side)
            if path is None or len(path) < 2:
                break
            distance = len(path) - 1
            current_enemy_head = current.head(enemy)
            enemy_distance = None
            if current_enemy_head is not None:
                enemy_distance = temporal_shortest_distance(current, current_enemy_head, food, enemy)
            race_margin = tempo_race_margin(current, side, distance, enemy_distance)
            reward = digit * FOOD_SCORE * max(1, current.multipliers.get(side, 1))
            score = reward * weights['sequence_reward_point_value']
            score += max(0, 12 - distance) * weights['chain_food_value']
            score += max(-2, min(6, race_margin)) * weights['chain_race_value']
            if distance <= 4 and race_margin >= -1:
                score += weights['chain_close_food_bonus']
            if hunger >= 14 and distance <= 7 and race_margin >= -1:
                score += (hunger - 13) * max(0, 8 - distance) * weights['hunger_close_food_value']
            if race_margin < 0:
                score //= 4
            total += score // (rank + 1)
            current = simulate_path_to_food(current, side, path)
            if current is None:
                break
        return total

    distances = distances if distances is not None else distance_map(state, head, side)
    best = 0
    for food in state.food:
        if food == consumed_food:
            continue
        distance = distances.get(food)
        if distance is None:
            continue

        enemy_distance = None
        if enemy_head is not None:
            enemy_distance = manhattan(enemy_head, food)
        race_margin = tempo_race_margin(state, side, distance, enemy_distance)
        score = max(0, 12 - distance) * weights['chain_food_value']
        score += max(-2, min(6, race_margin)) * weights['chain_race_value']
        if distance <= 4 and race_margin >= -1:
            score += weights['chain_close_food_bonus']
        if hunger >= 14 and distance <= 7 and race_margin >= -1:
            score += (hunger - 13) * max(0, 8 - distance) * weights['hunger_close_food_value']
        best = max(best, score)
    return best


def score_gap_urgency(state, side):
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    gap = state.scores.get(enemy, 0) - state.scores.get(side, 0)
    if gap <= 0:
        return 0
    food_gap = gap // 100
    return min(14, food_gap * 2 + gap // 400)


def food_cluster_followup_score(state, food):
    weights = get_weights()
    best = 0
    candidates = [pos for _, pos in state.numbered_sequence()] if state.food_values else state.food
    for other in candidates:
        if other == food:
            continue
        distance = manhattan(food, other)
        if distance <= 7:
            best = max(best, (8 - distance) * weights['rival_chain_threat_value'])
    return best


def rival_food_threat_score(state, side, food, our_distance, enemy_distance):
    if enemy_distance is None:
        return 0

    weights = get_weights()
    threat = 0
    if enemy_distance <= our_distance + 2:
        threat += max(0, 8 - enemy_distance) * weights['rival_food_threat_value']
        threat += food_cluster_followup_score(state, food)
    if tempo_race_margin(state, side, our_distance, enemy_distance) < 0:
        threat += (our_distance - enemy_distance) * weights['rival_food_threat_value']
    return threat


def food_denial_score(state, side, food, our_distance, enemy_distance, rival_threat):
    if enemy_distance is None or not wins_food_race(state, side, our_distance, enemy_distance):
        return 0

    weights = get_weights()
    if enemy_distance > 6 and rival_threat < weights['denial_chain_bonus'] * 2:
        return 0

    urgency = max(0, 7 - enemy_distance)
    score = weights['denial_food_bonus'] + urgency * weights['rival_food_threat_value']
    if our_distance == enemy_distance:
        score += weights['denial_chain_bonus']
    score += min(rival_threat, weights['denial_chain_bonus'] * 4)
    return score


def classify_safety(after, side):
    head = after.head(side)
    if head is None:
        return 'SUICIDAL', 0, 0, False
    region = flood_region(after, head, after.occupied())
    body_len = len(after.body(side))
    exits = count_exits(after, side)
    tail_ok = can_reach_tail(after, side)
    enough_space = len(region) >= body_len + 4

    if exits == 0 or len(region) <= max(2, body_len):
        return 'SUICIDAL', len(region), exits, tail_ok
    if not enough_space and not tail_ok:
        return 'DANGEROUS', len(region), exits, tail_ok
    if exits == 1 and len(region) < body_len * 2 + 6 and not tail_ok:
        return 'DANGEROUS', len(region), exits, tail_ok
    if exits == 1 or len(region) < body_len * 2 + 4:
        return 'ACCEPTABLE_RISK', len(region), exits, tail_ok
    return 'SAFE', len(region), exits, tail_ok


def edge_distance(state, pos):
    if pos is None:
        return 0
    return min(pos[0], pos[1], state.rows - 1 - pos[0], state.cols - 1 - pos[1])


def center_distance(state, pos):
    if pos is None:
        return state.rows + state.cols
    center_r = (state.rows - 1) / 2
    center_c = (state.cols - 1) / 2
    return abs(pos[0] - center_r) + abs(pos[1] - center_c)


def center_control_score(state, pos):
    if pos is None:
        return -10_000
    weights = get_weights()

    max_center_distance = (state.rows - 1) / 2 + (state.cols - 1) / 2
    center_score = int((max_center_distance - center_distance(state, pos)) * weights['center_unit_value'])
    edge = edge_distance(state, pos)
    edge_score = min(edge, 4) * weights['edge_depth_value']
    if edge == 0:
        edge_score -= weights['center_edge0_penalty']
    elif edge == 1:
        edge_score -= weights['center_edge1_penalty']
    return center_score + edge_score


def territory_control_score(state, side, head=None):
    head = head if head is not None else state.head(side)
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    if head is None or enemy_head is None:
        return 0

    our_distances = distance_map(state, head, side)
    enemy_distances = distance_map(state, enemy_head, enemy)
    if not our_distances or not enemy_distances:
        return 0

    occupied = state.occupied()
    our_cells = 0
    enemy_cells = 0
    for r in range(state.rows):
        for c in range(state.cols):
            cell = (r, c)
            if cell in occupied and cell not in (head, enemy_head):
                continue
            our_distance = our_distances.get(cell)
            enemy_distance = enemy_distances.get(cell)
            if our_distance is None and enemy_distance is None:
                continue
            if enemy_distance is None or (our_distance is not None and our_distance < enemy_distance):
                our_cells += 1
            elif our_distance is None or enemy_distance < our_distance:
                enemy_cells += 1

    weights = get_weights()
    score = (our_cells - enemy_cells) * weights['territory_control_value']
    cap = weights['territory_control_cap']
    return max(-cap, min(cap, score))


def food_plan_risk_penalty(state, food, our_distance, safety, space_after, exits_after):
    if our_distance <= 1:
        return 0

    weights = get_weights()
    penalty = 0
    target_edge = edge_distance(state, food)

    if our_distance > 2 and target_edge <= 1:
        penalty += (2 - target_edge) * weights['non_immediate_edge_food_penalty']
        if safety != 'SAFE':
            penalty += weights['low_space_food_penalty'] // 2

    if space_after <= 45:
        penalty += weights['low_space_food_penalty']
        if our_distance >= 4:
            penalty += weights['long_low_space_food_penalty']

    if exits_after <= 1 and space_after <= 80:
        penalty += weights['one_exit_food_penalty']
        if safety == 'ACCEPTABLE_RISK':
            penalty += weights['one_exit_food_penalty'] // 2

    return penalty


def objective_kind(state, target):
    return 'multiplier' if target in state.pickups else 'food'


def estimated_multiplier_points(state, side, travel_cost=0):
    if not state.food_values:
        return 0
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    head = state.head(side)
    enemy_head = state.head(enemy)
    own_turns_left = max(0, (state.remaining_moves + (1 if side == state.side else 0)) // 2 - travel_cost)
    if own_turns_left <= 0 or head is None:
        return 0
    expected = 0
    for rank, (digit, food) in enumerate(state.numbered_sequence(limit=5)):
        our_distance = temporal_shortest_distance(state, head, food, side)
        if our_distance is None or our_distance + rank > own_turns_left:
            continue
        enemy_distance = None
        if enemy_head is not None:
            enemy_distance = temporal_shortest_distance(state, enemy_head, food, enemy)
        margin = tempo_race_margin(state, side, our_distance, enemy_distance)
        capture_percent = 100 if margin >= 0 else 30
        expected += digit * FOOD_SCORE * capture_percent // 100
    return expected


def current_food_race(state, side):
    if not state.food:
        return None
    food = next(iter(state.food))
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    our_distance = temporal_shortest_distance(state, state.head(side), food, side)
    enemy_distance = temporal_shortest_distance(state, state.head(enemy), food, enemy)
    if our_distance is None:
        return food, None, enemy_distance, -99
    return food, our_distance, enemy_distance, tempo_race_margin(state, side, our_distance, enemy_distance)


def sequence_gate_hold_move(state, side, legal, plan, deep_moves):
    if (
        plan is None
        or plan.objective_kind != 'food'
        or plan.our_distance != 1
        or not state.food_values
    ):
        return None
    weights = get_weights()
    if state.remaining_moves > weights['sequence_hold_max_remaining'] or state.remaining_moves <= 4:
        return None

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    lead = state.scores.get(side, 0) - state.scores.get(enemy, 0)
    if lead <= 0:
        return None
    enemy_current_distance = temporal_shortest_distance(
        state, state.head(enemy), plan.food, enemy
    )
    if enemy_current_distance is None or enemy_current_distance < weights['sequence_hold_min_enemy_distance']:
        return None

    sequence = state.numbered_sequence(limit=2)
    if len(sequence) < 2:
        return None
    next_digit, next_food = sequence[1]
    after_eat = apply_move(state, plan.first_move, side)
    our_next_distance = temporal_shortest_distance(after_eat, after_eat.head(side), next_food, side)
    enemy_next_distance = temporal_shortest_distance(after_eat, after_eat.head(enemy), next_food, enemy)
    if (
        our_next_distance is None
        or enemy_next_distance is None
        or tempo_race_margin(after_eat, side, our_next_distance, enemy_next_distance) >= 0
    ):
        return None

    projected_lead = lead + plan.objective_reward
    enemy_next_reward = next_digit * FOOD_SCORE * max(1, state.multipliers.get(enemy, 1))
    if projected_lead >= enemy_next_reward + weights['sequence_hold_score_buffer']:
        return None

    deep_by_direction = {item['direction']: item for item in deep_moves}
    candidates = []
    for direction in legal:
        if direction == plan.first_move:
            continue
        after = apply_move(state, direction, side)
        if manhattan(after.head(side), plan.food) > 2:
            continue
        analysis = deep_by_direction.get(direction)
        if analysis is None or analysis.get('safety') == 'SUICIDAL' or analysis.get('our_replies', 0) < 2:
            continue
        candidates.append(analysis)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item['score'])['direction']


def objective_reward(state, side, target):
    if target in state.pickups:
        return 50
    return state.food_reward(side, target)


def is_structural_food_risk(plan):
    if plan is None or plan.our_distance <= 1:
        return False
    if plan.safety == 'ACCEPTABLE_RISK' and plan.space_after <= 45:
        return True
    if plan.our_distance >= 4 and plan.space_after <= 40:
        return True
    if plan.our_distance > 2 and plan.exits_after <= 1 and plan.space_after <= 80:
        return True
    return False


def position_control_penalty(state, side, direction):
    if direction not in legal_moves(state, side):
        return 1_000_000

    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return 1_000_000
    if len(after.body(side)) <= 3:
        return 0

    weights = get_weights()
    edge = edge_distance(after, head)
    penalty = 0
    if edge == 0:
        penalty += weights['position_edge0_penalty']
    elif edge == 1:
        penalty += weights['position_edge1_penalty']
    elif edge == 2:
        penalty += weights['position_edge2_penalty']

    borders_touching = sum(
        (
            head[0] == 0,
            head[1] == 0,
            head[0] == after.rows - 1,
            head[1] == after.cols - 1,
        )
    )
    if borders_touching >= 2:
        penalty += weights['position_corner_penalty']
    return penalty


def build_food_plan(state, side, food, current_target, hunger):
    head = state.head(side)
    enemy_head = state.head(state.enemy)
    if head is None:
        return None
    path = temporal_shortest_path(state, head, food, side)
    if path is None or len(path) < 2:
        return None

    after = simulate_path_to_food(state, side, path)
    if after is None:
        return None

    enemy_distance = None
    if enemy_head is not None:
        enemy_distance = temporal_shortest_distance(state, enemy_head, food, state.enemy)
    our_distance = len(path) - 1
    race_margin = tempo_race_margin(state, side, our_distance, enemy_distance)
    tempo_urgency = score_gap_urgency(state, side)
    effective_hunger = hunger + tempo_urgency
    safety, space_after, exits_after, tail_ok = classify_safety(after, side)
    after_distances = distance_map(after, after.head(side), side)
    next_food = nearest_next_food_distance(after, side, food, after_distances)
    chain_score = food_chain_score(after, side, food, effective_hunger, after_distances)
    rival_threat = rival_food_threat_score(state, side, food, our_distance, enemy_distance)
    denial_score = food_denial_score(state, side, food, our_distance, enemy_distance, rival_threat)
    occupied = state.occupied()
    temporal_tail_path = any(cell in occupied for cell in path[1:])
    endgame_acceptable = endgame_accepts_food_risk(state, our_distance, race_margin, safety)
    kind = objective_kind(state, food)
    reward = objective_reward(state, side, food)

    weights = get_weights()
    value = weights['food_base_value']
    value += max(0, reward - FOOD_SCORE) * weights['objective_point_value']
    if kind == 'multiplier':
        value += estimated_multiplier_points(state, side, our_distance) * weights['multiplier_future_point_value']
        value += weights['multiplier_base_value']
        enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
        multiplier_gap = state.multipliers.get(enemy, 1) - state.multipliers.get(side, 1)
        value += multiplier_gap * weights['multiplier_gap_value']
        current_race = current_food_race(state, side)
        if current_race is not None and current_race[3] >= 0:
            value -= hunger * weights['multiplier_hunger_penalty']
        value -= max(0, state.multipliers.get(side, 1) - 2) * weights['multiplier_stack_penalty']
    elif state.food_values:
        value += min(hunger, 30) * weights['numbered_food_hunger_value']
    value -= our_distance * weights['food_distance_cost']
    value += race_margin * weights['race_margin_value']
    value += min(effective_hunger, 50) * weights['hunger_value']
    value += min(space_after, 80) * weights['space_after_value']
    value += exits_after * weights['exits_after_value']
    value += weights['tail_reachable_bonus'] if tail_ok else 0
    value += center_control_score(after, after.head(side))
    if next_food is not None:
        value += max(0, 12 - next_food) * weights['next_food_value']
    value += chain_score
    value += denial_score
    if effective_hunger >= 12 and safety == 'SAFE' and race_margin >= 0:
        value += (effective_hunger - 11) * max(0, 10 - our_distance) * weights['hunger_close_food_value']
    if food == current_target:
        value += weights['current_target_bonus']
    if safety == 'SAFE':
        value += weights['safe_bonus']
    elif safety == 'ACCEPTABLE_RISK':
        value += weights['acceptable_risk_bonus']
    elif safety == 'DANGEROUS':
        if endgame_acceptable:
            value += weights['endgame_food_bonus']
        else:
            value -= weights['dangerous_penalty']
    else:
        if endgame_acceptable:
            value += weights['endgame_food_bonus'] // 2
        else:
            value -= 100_000
    if enemy_distance is not None and enemy_distance + 3 < our_distance:
        value -= weights['lost_race_penalty']
    if denial_score == 0 and rival_threat and tempo_race_margin(state, side, our_distance, enemy_distance) < 0:
        value -= min(rival_threat, weights['rival_chain_ignore_penalty'])
    if endgame_acceptable:
        value += max(0, weights['endgame_food_bonus'] - max(0, remaining_after_our_food(state, our_distance)) * 1000)
    else:
        value -= food_plan_risk_penalty(state, food, our_distance, safety, space_after, exits_after)

    return FoodPlan(
        food=food,
        path=path,
        our_distance=our_distance,
        enemy_distance=enemy_distance,
        race_margin=race_margin,
        space_after=space_after,
        exits_after=exits_after,
        tail_reachable_after=tail_ok,
        next_food_distance=next_food,
        safety=safety,
        value=value,
        chain_score=chain_score,
        denial_score=denial_score,
        rival_threat=rival_threat,
        tempo_urgency=tempo_urgency,
        temporal_tail_path=temporal_tail_path,
        endgame_acceptable=endgame_acceptable,
        objective_kind=kind,
        objective_reward=reward,
    )


def build_sequence_setup_plans(state, side, current_target, hunger):
    race = current_food_race(state, side)
    sequence = state.numbered_sequence(limit=3)
    if race is None or race[3] >= 0 or len(sequence) < 2:
        return []

    current_food, _, enemy_current_distance, _ = race
    if enemy_current_distance is None:
        return []
    head = state.head(side)
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    weights = get_weights()
    plans = []
    occupied = state.occupied()

    for rank, (digit, future_food) in enumerate(sequence[1:3], start=1):
        candidates = []
        for staging in neighbors(future_food, state.rows, state.cols):
            if staging in occupied or staging in state.food_values or staging in state.pickups:
                continue
            path = temporal_shortest_path(state, head, staging, side)
            if path is None or len(path) < 2 or current_food in path[1:]:
                continue
            candidates.append((len(path) - 1, path, staging))
        if not candidates:
            continue

        _, path, staging = min(candidates, key=lambda item: (item[0], manhattan(item[2], head)))
        our_distance = len(path) - 1
        if our_distance > enemy_current_distance + weights['sequence_setup_late_slack']:
            continue
        after = simulate_path_to_food(state, side, path)
        if after is None:
            continue
        safety, space_after, exits_after, tail_ok = classify_safety(after, side)
        if safety in ('SUICIDAL', 'DANGEROUS'):
            continue

        enemy_followup = max(1, manhattan(current_food, future_food))
        readiness = enemy_current_distance - our_distance
        race_margin = enemy_followup - 1 + readiness
        future_reward = digit * FOOD_SCORE * max(1, state.multipliers.get(side, 1))
        value = weights['sequence_setup_base_value']
        value += future_reward * weights['sequence_setup_reward_point_value']
        value += readiness * weights['sequence_setup_readiness_value']
        value += race_margin * weights['race_margin_value']
        value -= our_distance * weights['food_distance_cost']
        value -= (rank - 1) * weights['sequence_setup_rank_penalty']
        value += min(space_after, 80) * weights['space_after_value']
        value += exits_after * weights['exits_after_value']
        value += weights['tail_reachable_bonus'] if tail_ok else 0
        value += center_control_score(after, after.head(side))
        if staging == current_target:
            value += weights['current_target_bonus']
        value += min(hunger, 30) * weights['sequence_setup_hunger_value']

        plans.append(
            FoodPlan(
                food=staging,
                path=path,
                our_distance=our_distance,
                enemy_distance=enemy_followup,
                race_margin=race_margin,
                space_after=space_after,
                exits_after=exits_after,
                tail_reachable_after=tail_ok,
                next_food_distance=1,
                safety=safety,
                value=value,
                chain_score=future_reward * weights['sequence_reward_point_value'],
                tempo_urgency=score_gap_urgency(state, side),
                temporal_tail_path=any(cell in occupied for cell in path[1:]),
                objective_kind='sequence_setup',
                objective_reward=future_reward,
                sequence_rank=rank,
            )
        )
    return plans


def choose_food_plan(state, side, current_target=None, hunger=0):
    plans = [
        plan
        for food in state.objective_cells()
        if (plan := build_food_plan(state, side, food, current_target, hunger)) is not None
    ]
    plans.extend(build_sequence_setup_plans(state, side, current_target, hunger))
    viable = [p for p in plans if p.safety in ('SAFE', 'ACCEPTABLE_RISK') or p.endgame_acceptable]
    if not viable:
        viable = [p for p in plans if p.safety == 'DANGEROUS' and hunger >= 35 and p.race_margin >= 0]
    if not viable:
        return None

    immediate = [p for p in viable if p.our_distance == 1]
    if immediate:
        return max(immediate, key=food_tiebreak_key)

    if hunger >= 10:
        harvest = [
            p for p in viable
            if p.objective_kind == 'food'
            and p.race_margin >= 0
            and p.safety in ('SAFE', 'ACCEPTABLE_RISK')
        ]
        if harvest:
            return max(harvest, key=hunger_food_key)

    strategic = [p for p in viable if not is_structural_food_risk(p)]
    if strategic:
        viable = strategic

    weights = get_weights()
    urgent_denials = [
        p
        for p in viable
        if p.enemy_distance is not None
        and (p.enemy_distance <= 4 or p.denial_score >= weights['denial_food_bonus'] + weights['denial_chain_bonus'])
        and p.race_margin >= 0
    ]
    if urgent_denials:
        return max(urgent_denials, key=denial_key)

    race_viable = [
        p
        for p in viable
        if p.enemy_distance is None or p.race_margin >= 0
    ]
    if race_viable:
        viable = race_viable

    nearest_distance = min(p.our_distance for p in viable)
    tempo_hunger = hunger + score_gap_urgency(state, side)
    if tempo_hunger >= 14:
        urgent_food = [
            p
            for p in viable
            if p.objective_kind == 'food'
            and p.safety == 'SAFE'
            and p.race_margin >= 0
            and p.our_distance <= nearest_distance + 2
        ]
        if urgent_food:
            return max(urgent_food, key=hunger_food_key)

    best = max(viable, key=food_tiebreak_key)
    current = next((p for p in viable if p.food == current_target), None)
    if current and tempo_hunger < 12 and current.race_margin >= 0 and current.value > 0:
        close_to_best = current.value >= best.value - weights['food_plan_switch_margin']
        no_extra_detour = current.our_distance <= best.our_distance + 1
        if close_to_best and no_extra_detour:
            return current

    close_enough = [p for p in viable if p.our_distance <= nearest_distance + 1]
    best_close = max(close_enough, key=food_tiebreak_key)
    if best.value >= best_close.value + weights['food_plan_big_switch_margin']:
        return best
    if best_close.value < 0 and best.value > best_close.value:
        return best
    return best_close


def food_tiebreak_key(plan):
    safety_rank = {'SAFE': 3, 'ACCEPTABLE_RISK': 2, 'DANGEROUS': 1}.get(plan.safety, 0)
    race_rank = 0 if plan.enemy_distance is None else max(-4, min(4, plan.race_margin))
    next_food = 0 if plan.next_food_distance is None else max(0, 12 - plan.next_food_distance)
    return (
        safety_rank,
        plan.value,
        race_rank,
        plan.denial_score,
        plan.chain_score,
        -plan.our_distance,
        next_food,
        plan.space_after,
    )


def hunger_food_key(plan):
    next_food = 0 if plan.next_food_distance is None else max(0, 10 - plan.next_food_distance)
    return (
        -plan.our_distance,
        plan.denial_score,
        plan.race_margin,
        plan.chain_score,
        next_food,
        plan.value,
    )


def denial_key(plan):
    enemy_urgency = 9 if plan.enemy_distance is None else 9 - plan.enemy_distance
    return (
        enemy_urgency,
        plan.race_margin,
        -plan.our_distance,
        plan.space_after,
        plan.value,
    )


def should_commit_to_food(plan, hunger, repeat_count):
    if plan.endgame_acceptable:
        return True
    if plan.safety == 'SUICIDAL':
        return False
    if plan.our_distance == 1 and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
        return True
    if plan.race_margin < 0:
        return False
    if is_structural_food_risk(plan):
        return False
    if plan.safety == 'DANGEROUS':
        return hunger >= 35
    tempo_hunger = hunger + plan.tempo_urgency
    if plan.denial_score > 0 and plan.safety == 'SAFE' and plan.race_margin >= 0:
        return True
    if repeat_count >= 2 and plan.safety != 'SUICIDAL':
        return True
    if tempo_hunger >= 20 and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
        return True
    return plan.safety in ('SAFE', 'ACCEPTABLE_RISK') and plan.value > 0


def should_use_plan_for_positioning(plan):
    return plan is not None and plan.race_margin >= 0 and (plan.endgame_acceptable or not is_structural_food_risk(plan))


def fallback_hunger_food_score(state, side, after, head, hunger):
    if hunger < 10 or head is None or not after.food:
        return 0

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    current_head = state.head(side)
    weights = get_weights()
    distances = distance_map(after, head, side)
    best = 0
    for food in after.food:
        after_distance = distances.get(food)
        if after_distance is None:
            continue
        before_distance = None if current_head is None else manhattan(current_head, food)
        enemy_distance = None
        if enemy_head is not None:
            enemy_distance = manhattan(enemy_head, food)
        race_margin = tempo_race_margin(after, side, after_distance, enemy_distance)
        if race_margin < -1:
            continue

        progress = 0 if before_distance is None else max(0, before_distance - after_distance)
        hunger_pressure = hunger - 9
        score = progress * hunger_pressure * weights['fallback_hunger_food_progress']
        score += max(0, 8 - after_distance) * hunger_pressure * weights['fallback_hunger_food_distance']
        score += max(0, min(5, race_margin)) * weights['fallback_hunger_food_race']
        best = max(best, score)
    return best


def fallback_denial_score(state, side, after, head):
    if head is None or not state.food:
        return 0

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    current_head = state.head(side)
    if enemy_head is None or current_head is None:
        return 0

    weights = get_weights()
    after_distances = distance_map(after, head, side)
    score = 0
    for food in state.food:
        enemy_distance = temporal_shortest_distance(state, enemy_head, food, enemy)
        if enemy_distance is None or enemy_distance > 5:
            continue
        before_distance = temporal_shortest_distance(state, current_head, food, side)
        after_distance = after_distances.get(food)
        if before_distance is None or after_distance is None:
            continue
        if wins_food_race(after, side, after_distance, enemy_distance) and after_distance < before_distance:
            progress = before_distance - after_distance
            score += weights['fallback_denial_reach_bonus'] + progress * weights['fallback_denial_progress']
    return score


def rival_response_stats(state, side, direction, depth=1):
    if direction not in legal_moves(state, side):
        return {
            'enemy_moves': 0,
            'zero_escape_replies': 0,
            'one_escape_replies': 0,
            'delayed_forced_zero_replies': 0,
            'forced_zero': True,
            'forced_delayed_zero': True,
        }

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    after = apply_move(state, direction, side)
    enemy_moves = legal_moves(after, enemy)
    if not enemy_moves:
        return {
            'enemy_moves': 0,
            'zero_escape_replies': 0,
            'one_escape_replies': 0,
            'delayed_forced_zero_replies': 0,
            'forced_zero': False,
            'forced_delayed_zero': False,
        }

    zero_escape_replies = 0
    one_escape_replies = 0
    delayed_forced_zero_replies = 0
    for enemy_move in enemy_moves:
        after_enemy = apply_move(after, enemy_move, enemy)
        our_replies = legal_moves(after_enemy, side)
        if not our_replies:
            zero_escape_replies += 1
        elif len(our_replies) == 1:
            one_escape_replies += 1
        if depth > 0 and our_replies and all(
            rival_response_stats(after_enemy, side, our_move, depth - 1)['forced_zero']
            for our_move in our_replies
        ):
            delayed_forced_zero_replies += 1

    return {
        'enemy_moves': len(enemy_moves),
        'zero_escape_replies': zero_escape_replies,
        'one_escape_replies': one_escape_replies,
        'delayed_forced_zero_replies': delayed_forced_zero_replies,
        'forced_zero': zero_escape_replies == len(enemy_moves),
        'forced_delayed_zero': delayed_forced_zero_replies == len(enemy_moves),
    }


def rival_response_penalty(state, side, direction):
    stats = rival_response_stats(state, side, direction)
    if stats['forced_zero']:
        return -1_000_000
    if stats['forced_delayed_zero']:
        return -80_000
    weights = get_weights()
    return -(
        stats['zero_escape_replies'] * weights['zero_escape_penalty']
        + stats['delayed_forced_zero_replies'] * weights['delayed_zero_penalty']
        + stats['one_escape_replies'] * weights['one_escape_penalty']
        + rival_space_penalty(state, side, direction)
        + position_control_penalty(state, side, direction)
    )


def rival_space_penalty(state, side, direction):
    if direction not in legal_moves(state, side):
        return 1_000_000

    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return 1_000_000

    body_len = len(after.body(side))
    immediate_region = len(flood_region(after, head, after.occupied()))
    immediate_exits = count_exits(after, side)
    weights = get_weights()
    base = 0
    if immediate_region < body_len * 2 + 6:
        base += (body_len * 2 + 6 - immediate_region) * weights['small_region_penalty']
    if immediate_exits <= 1:
        base += weights['one_exit_penalty']

    enemy_moves = legal_moves(after, enemy)
    if not enemy_moves:
        return base

    worst_reply = 0
    for enemy_move in enemy_moves:
        after_enemy = apply_move(after, enemy_move, enemy)
        our_head = after_enemy.head(side)
        enemy_head = after_enemy.head(enemy)
        if our_head is None:
            worst_reply = max(worst_reply, 1_000_000)
            continue

        replies = legal_moves(after_enemy, side)
        own_region = len(flood_region(after_enemy, our_head, after_enemy.occupied()))
        enemy_region = 0 if enemy_head is None else len(flood_region(after_enemy, enemy_head, after_enemy.occupied()))
        threat = 0
        if not replies:
            threat += weights['enemy_no_reply_threat']
        if own_region < body_len * 2 + 6:
            threat += (body_len * 2 + 6 - own_region) * weights['small_region_penalty']
        if count_exits(after_enemy, side) <= 1:
            threat += weights['one_exit_penalty']
        territory_gap = enemy_region - own_region
        if territory_gap > weights['territory_gap_limit']:
            threat += min(
                weights['territory_gap_penalty_cap'],
                (territory_gap - weights['territory_gap_limit']) * weights['territory_gap_penalty'],
            )
        worst_reply = max(worst_reply, threat)

    return base + worst_reply


def hot_lost_food_penalty(state, side, direction):
    head = state.head(side)
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    if head is None or enemy_head is None or direction not in legal_moves(state, side):
        return 0

    after = apply_move(state, direction, side)
    after_head = after.head(side)
    if after_head is None:
        return 0

    weights = get_weights()
    penalty = 0
    for food in state.food:
        our_distance = temporal_shortest_distance(state, head, food, side)
        enemy_distance = temporal_shortest_distance(state, enemy_head, food, enemy)
        if (
            our_distance is None
            or enemy_distance is None
            or enemy_distance >= our_distance
            or enemy_distance > 2
        ):
            continue
        after_distance = temporal_shortest_distance(after, after_head, food, side)
        if (
            after_distance is not None
            and after_distance < our_distance
            and (our_distance <= 6 or after_distance <= 3)
        ):
            penalty += weights['hot_lost_food_base'] + (3 - enemy_distance) * weights['hot_lost_food_enemy_bonus']
    return penalty


def forced_kill_move(state, side, legal):
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    for direction in legal:
        after = apply_move(state, direction, side)
        enemy_moves = legal_moves(after, enemy)
        own_head = after.head(side)
        if own_head is None:
            continue
        own_region = flood_region(after, own_head, after.occupied())
        projected_our_score = after.scores.get(side, 0) + RIVAL_CRASH_REWARD
        projected_enemy_score = after.scores.get(enemy, 0) + CRASH_PENALTY
        wins_on_score = projected_our_score > projected_enemy_score
        if not enemy_moves and wins_on_score and len(own_region) >= len(after.body(side)) + 3:
            return direction
    return None


def survival_position_score(state, side):
    head = state.head(side)
    if head is None:
        return -1_000_000

    weights = get_weights()
    region = flood_region(state, head, state.occupied())
    exits = count_exits(state, side)
    replies = legal_moves(state, side)
    body_len = len(state.body(side))
    tail_ok = can_reach_tail(state, side)
    safety, _, _, _ = classify_safety(state, side)

    score = min(len(region), 130) * weights['survival_region_value']
    score += min(exits, 4) * weights['survival_exit_value']
    score += min(len(replies), 4) * weights['survival_reply_value']
    score += center_control_score(state, head) if body_len > 3 else 0
    score += territory_control_score(state, side, head) if body_len > 3 else 0
    if tail_ok:
        score += weights['survival_tail_bonus']
    if safety == 'DANGEROUS':
        score -= weights['fallback_danger_penalty']
    elif safety == 'ACCEPTABLE_RISK':
        score -= weights['future_corridor_penalty'] // 2

    if not replies:
        score -= weights['future_no_reply_penalty']
    elif len(replies) == 1:
        score -= weights['future_one_reply_penalty']
    if exits <= 1:
        score -= weights['future_corridor_penalty']

    wanted_region = body_len * 2 + 6
    if len(region) < wanted_region:
        score -= (wanted_region - len(region)) * weights['future_small_region_penalty']
    return score


def survival_move_analysis(state, side, direction, plan=None, hunger=0, repeat_count=0):
    weights = get_weights()
    if direction not in legal_moves(state, side):
        return {
            'direction': direction,
            'score': -1_000_000,
            'base_score': -1_000_000,
            'future_score': -1_000_000,
            'enemy_reply': None,
            'reply_count': 0,
            'region': 0,
            'exits': 0,
            'food_now': False,
            'enemy_food': False,
        }

    after = apply_move(state, direction, side)
    head = after.head(side)
    region = flood_region(after, head, after.occupied()) if head is not None else set()
    exits = count_exits(after, side) if head is not None else 0
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = after.head(enemy)
    enemy_moves = legal_moves(after, enemy)
    if head is not None and not enemy_moves:
        future = weights['enemy_trapped_future_bonus']
        return {
            'direction': direction,
            'score': future + survival_position_score(after, side),
            'base_score': 0,
            'future_score': future,
            'enemy_reply': None,
            'reply_count': 99,
            'region': len(region),
            'exits': exits,
            'food_now': head in state.food,
            'enemy_food': False,
        }
    safety, _, _, _ = classify_safety(after, side)
    if safety == 'SUICIDAL':
        return {
            'direction': direction,
            'score': -1_000_000,
            'base_score': -1_000_000,
            'future_score': 0,
            'enemy_reply': None,
            'reply_count': 0,
            'region': len(region),
            'exits': exits,
            'food_now': head in state.food if head is not None else False,
            'enemy_food': False,
        }
    base = move_quality_score(state, side, direction, plan, hunger, repeat_count)[0]
    if head in state.food:
        base += weights['survival_immediate_food_bonus']

    worst_future = None
    worst_enemy = None
    worst_reply_count = 0
    worst_enemy_food = False
    for enemy_move in enemy_moves:
        enemy_target = None if enemy_head is None else step(enemy_head, enemy_move)
        enemy_food = enemy_target in after.food if enemy_target is not None else False
        after_enemy = apply_move(after, enemy_move, enemy)
        replies = legal_moves(after_enemy, side)
        if not replies:
            reply_score = -weights['future_no_reply_penalty']
        else:
            reply_score = max(
                survival_position_score(apply_move(after_enemy, reply, side), side)
                + (
                    weights['survival_immediate_food_bonus']
                    if step(after_enemy.head(side), reply) in after_enemy.food
                    else 0
                )
                for reply in replies
            )
        if enemy_food:
            reply_score -= weights['future_enemy_food_penalty']
        if worst_future is None or reply_score < worst_future:
            worst_future = reply_score
            worst_enemy = enemy_move
            worst_reply_count = len(replies)
            worst_enemy_food = enemy_food

    future = worst_future if worst_future is not None else -weights['future_no_reply_penalty']
    return {
        'direction': direction,
        'score': base + future,
        'base_score': base,
        'future_score': future,
        'enemy_reply': worst_enemy,
        'reply_count': worst_reply_count,
        'region': len(region),
        'exits': exits,
        'food_now': head in state.food,
        'enemy_food': worst_enemy_food,
    }


def rank_survival_moves(state, side, legal, plan=None, hunger=0, repeat_count=0):
    analyses = [
        survival_move_analysis(state, side, direction, plan, hunger, repeat_count)
        for direction in legal
    ]
    analyses.sort(
        key=lambda item: (
            item['score'],
            1 if item['food_now'] else 0,
            item['reply_count'],
            item['exits'],
            item['region'],
        ),
        reverse=True,
    )
    return analyses


def move_quality_score(state, side, direction, plan=None, hunger=0, repeat_count=0):
    weights = get_weights()
    tempo_hunger = hunger + score_gap_urgency(state, side)
    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return (-999999, 0, 0, 0)
    safety, _, _, _ = classify_safety(after, side)
    if safety == 'SUICIDAL':
        return (-999999, 0, 0, 0)
    danger_penalty = weights['fallback_danger_penalty'] if safety == 'DANGEROUS' and tempo_hunger < 35 else 0
    nearest = None
    if after.food:
        distances = [shortest_distance(after, head, food, side) for food in after.food]
        distances = [d for d in distances if d is not None]
        nearest = min(distances) if distances else None
    progress = 0
    if plan and plan.race_margin >= 0:
        before = manhattan(state.head(side), plan.food)
        progress = before - manhattan(head, plan.food)
    region = flood_region(after, head, after.occupied())
    exits = count_exits(after, side)
    food_distance_score = 0 if nearest is None else -nearest * weights['fallback_food_distance_cost']
    cycle_penalty = repeat_count * weights['fallback_cycle_penalty'] if progress <= 0 and tempo_hunger >= 10 else 0
    rival_penalty = rival_response_penalty(state, side, direction)
    hot_lost_penalty = hot_lost_food_penalty(state, side, direction)
    control_score = center_control_score(after, head) if len(after.body(side)) > 3 else 0
    territory_score = territory_control_score(after, side, head) if len(after.body(side)) > 3 else 0
    hunger_food_score = fallback_hunger_food_score(state, side, after, head, tempo_hunger)
    denial_score = fallback_denial_score(state, side, after, head)
    return (
        progress * (weights['fallback_progress_value'] + tempo_hunger * weights['fallback_hunger_progress_value'])
        + food_distance_score
        + hunger_food_score
        + denial_score
        + control_score
        + territory_score
        - cycle_penalty
        + rival_penalty
        - hot_lost_penalty
        - danger_penalty,
        1 if head in state.food else 0,
        exits,
        min(len(region), 60),
    )


def fallback_food_positioning(state, side, legal, plan, hunger, repeat_count):
    return max(legal, key=lambda direction: move_quality_score(state, side, direction, plan, hunger, repeat_count))
