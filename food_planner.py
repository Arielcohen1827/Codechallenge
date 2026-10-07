from collections import deque
from dataclasses import dataclass

from bot_weights import get_weights
from snake_state import (
    CRASH_PENALTY,
    DIRS,
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
    shrink_wall,
    step,
    strategic_blocked,
    temporal_distance_map,
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
    projected_income: int = 0
    rival_projected_income: int = 0
    economic_swing: int = 0
    multiplier_marginal: int = 0

    @property
    def first_move(self):
        if len(self.path) < 2:
            return None
        return direction_between(self.path[0], self.path[1])


def simulate_path_to_food(state, side, path):
    current = state
    for cell in path[1:]:
        if cell in current.walls:
            return None
        direction = direction_between(current.head(side), cell)
        if direction is None or direction not in legal_moves(current, side):
            return None
        current = apply_move(current, direction, side)
        walls = shrink_wall(current.walls) if side == 'A' else current.walls
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
            walls=walls,
            food_copy_targets=current.food_copy_targets,
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


def food_distances(state, side, positions=None):
    """Return temporal distances to every reachable copy in one objective set."""
    head = state.head(side)
    if head is None:
        return {}
    positions = state.food if positions is None else positions
    temporal_distances = temporal_distance_map(state, head, side, goals=positions)
    return {
        food: temporal_distances[food]
        for food in positions
        if food in temporal_distances
    }


def food_set_race(state, side, positions=None):
    """Compare each player's fastest route to any copy of the current digit."""
    positions = frozenset(state.food if positions is None else positions)
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    ours = food_distances(state, side, positions)
    theirs = food_distances(state, enemy, positions)
    our_best = min(ours.items(), key=lambda item: (item[1], item[0])) if ours else None
    enemy_best = min(theirs.items(), key=lambda item: (item[1], item[0])) if theirs else None
    our_distance = None if our_best is None else our_best[1]
    enemy_distance = None if enemy_best is None else enemy_best[1]
    margin = -99 if our_distance is None else tempo_race_margin(
        state,
        side,
        our_distance,
        enemy_distance,
    )
    return {
        'our_food': None if our_best is None else our_best[0],
        'our_distance': our_distance,
        'enemy_food': None if enemy_best is None else enemy_best[0],
        'enemy_distance': enemy_distance,
        'margin': margin,
        'our_reachable': len(ours),
        'enemy_reachable': len(theirs),
        'our_distances': ours,
        'enemy_distances': theirs,
    }


def best_numbered_copy(state, side, positions):
    """Choose the copy we can contest best, accounting for move order."""
    head = state.head(side)
    if head is None:
        return None
    race = food_set_race(state, side, positions)
    enemy_distance = race['enemy_distance']
    candidates = []
    for food, our_distance in race['our_distances'].items():
        margin = tempo_race_margin(state, side, our_distance, enemy_distance)
        candidates.append((margin, -our_distance, food))
    if not candidates:
        return None
    margin, neg_distance, food = max(candidates)
    path = temporal_shortest_path(state, head, food, side)
    if path is None:
        return None
    return food, path, -neg_distance, enemy_distance, margin


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
        candidate_distances = []
        for food in state.food:
            path = temporal_shortest_path(state, head, food, side)
            if path is not None:
                candidate_distances.append(len(path) - 1)
        return min(candidate_distances) if candidate_distances else None
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
            if current.next_food_digit is None or not current.food:
                break
            choice = best_numbered_copy(current, side, current.food)
            if choice is None:
                break
            food, path, distance, enemy_distance, race_margin = choice
            if len(path) < 2:
                break
            digit = current.next_food_digit
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
    if state.food_values:
        groups = state.numbered_groups(limit=3)
        candidates = [pos for _, positions in groups[1:] for pos in positions]
    else:
        candidates = state.food
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
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    threat = 0
    if enemy_distance <= our_distance + 2:
        threat += max(0, 8 - enemy_distance) * weights['rival_food_threat_value']
        threat += food_cluster_followup_score(state, food)
        threat += state.food_reward(enemy, food) * weights['rival_reward_point_value']
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


def advantage_control_context(state, side):
    """Describe whether a score lead is large enough to defend positionally."""
    weights = get_weights()
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    lead = state.scores.get(side, 0) - state.scores.get(enemy, 0)
    own_length = len(state.body(side))
    enemy_length = len(state.body(enemy))
    own_turns_left = max(
        0,
        (state.remaining_moves + (1 if side == state.side else 0)) // 2,
    )

    if state.food_values:
        recoverable_threat = projected_sequence_income(state, enemy, limit=5)
        if state.pickups:
            recoverable_threat += estimated_multiplier_points(state, enemy) * min(2, len(state.pickups))
    else:
        recoverable_threat = min(2, len(state.food)) * FOOD_SCORE
    recoverable_threat += min(2, len(state.pickups)) * 50
    current_enemy_reward = max(
        (state.food_reward(enemy, food) for food in state.food),
        default=0,
    )
    crash_score_swing = RIVAL_CRASH_REWARD - CRASH_PENALTY
    protected_lead = crash_score_swing + current_enemy_reward

    dynamic_lead = (
        recoverable_threat * weights['advantage_threat_ratio'] // 100
        + weights['advantage_score_buffer']
    )
    activation_lead = max(
        weights['advantage_min_lead'],
        protected_lead,
        min(weights['advantage_max_activation_lead'], dynamic_lead),
    )
    active = (
        weights['advantage_control_enabled'] > 0
        and bool(state.food)
        and lead >= activation_lead
        and own_length >= weights['advantage_min_body']
        and own_length + weights['advantage_body_deficit'] >= enemy_length
        and own_turns_left >= weights['advantage_min_own_turns']
    )
    return {
        'active': active,
        'lead': lead,
        'activation_lead': activation_lead,
        'recoverable_threat': recoverable_threat,
        'current_enemy_reward': current_enemy_reward,
        'protected_lead': protected_lead,
        'own_length': own_length,
        'enemy_length': enemy_length,
        'own_turns_left': own_turns_left,
    }


def advantage_control_move(state, side, legal, deep_moves, reference_move=None, adjacent_food=False):
    """Choose a safe move that delays the rival while preserving our mobility."""
    context = advantage_control_context(state, side)
    context['moves'] = []
    context['direction'] = None
    context['override'] = False
    if not context['active'] or not deep_moves:
        return context

    weights = get_weights()
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    if enemy_head is None:
        return context

    enemy_distances = food_distances(state, enemy)
    reachable_food = [(distance, food) for food, distance in enemy_distances.items()]
    threatened_food = min(reachable_food)[1] if reachable_food else next(iter(state.food), None)
    before_food_distance = min(enemy_distances.values(), default=None)
    before_reachable_copies = len(enemy_distances)
    baseline_enemy_moves = len(legal_moves(state, enemy))
    context['baseline_enemy_moves'] = baseline_enemy_moves
    context['enemy_reachable_copies'] = before_reachable_copies

    analyses = {item['direction']: item for item in deep_moves}
    candidates = []
    for direction in legal:
        analysis = analyses.get(direction)
        if analysis is None:
            continue
        if analysis.get('forced_loss') or analysis.get('safety') != 'SAFE':
            continue
        if analysis.get('our_replies', 0) < 2 or analysis.get('exits', 0) < 2:
            continue

        after = apply_move(state, direction, side)
        enemy_moves_after = len(legal_moves(after, enemy))
        if enemy_moves_after == 0:
            # forced_kill_move handles score-winning traps before this phase.
            continue

        secured_food = step(state.head(side), direction) in state.food
        after_food_distance = None
        delay = 0
        food_unreachable = False
        after_enemy_distances = {}
        if state.food and not secured_food:
            after_enemy_distances = food_distances(after, enemy, after.food)
            after_food_distance = min(after_enemy_distances.values(), default=None)
            if after_food_distance is None:
                food_unreachable = True
            elif before_food_distance is not None:
                delay = max(-3, min(5, after_food_distance - before_food_distance))
        copies_denied = max(0, before_reachable_copies - len(after_enemy_distances))

        score = analysis['score']
        score += analysis.get('voronoi_ours', 0) * weights['advantage_our_cell_value']
        score -= analysis.get('voronoi_enemy', 0) * weights['advantage_enemy_cell_value']
        score += max(0, baseline_enemy_moves - enemy_moves_after) * weights['advantage_enemy_move_reduction']
        score += delay * weights['advantage_food_delay_value']
        score += copies_denied * weights['advantage_copy_denial_value']
        if food_unreachable:
            score += weights['advantage_food_unreachable_bonus']
        if analysis.get('enemy_food'):
            score -= weights['advantage_enemy_food_penalty']

        candidates.append(
            {
                'direction': direction,
                'score': score,
                'enemy_moves': enemy_moves_after,
                'enemy_food_distance_before': before_food_distance,
                'enemy_food_distance_after': after_food_distance,
                'food_delay': delay,
                'food_unreachable': food_unreachable,
                'enemy_reachable_copies_after': len(after_enemy_distances),
                'copies_denied': copies_denied,
                'secured_food': secured_food,
                'meaningful_denial': (
                    secured_food
                    or food_unreachable
                    or delay > 0
                    or copies_denied > 0
                    or enemy_moves_after < baseline_enemy_moves
                ),
                'voronoi_ours': analysis.get('voronoi_ours', 0),
                'voronoi_enemy': analysis.get('voronoi_enemy', 0),
            }
        )

    candidates.sort(key=lambda item: item['score'], reverse=True)
    context['moves'] = candidates
    if not candidates:
        return context

    best = candidates[0]
    if not best['meaningful_denial']:
        return context
    reference = next(
        (item for item in candidates if item['direction'] == reference_move),
        None,
    )
    required_margin = weights['advantage_switch_margin']
    if adjacent_food:
        required_margin += weights['advantage_adjacent_food_switch_margin']
    improvement = None if reference is None else best['score'] - reference['score']
    should_use = reference_move is None or (
        best['direction'] != reference_move
        and (reference is None or improvement >= required_margin)
    )
    context.update(
        {
            'direction': best['direction'],
            'reference_direction': reference_move,
            'improvement': improvement,
            'required_margin': required_margin,
            'override': should_use,
            'threatened_food': threatened_food,
        }
    )
    return context


def classify_safety(after, side):
    head = after.head(side)
    if head is None:
        return 'SUICIDAL', 0, 0, False
    region = flood_region(after, head, strategic_blocked(after))
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


def projected_sequence_income(state, side, travel_cost=0, multiplier=None, limit=5):
    """Estimate score from the visible numbered sequence without path searches.

    This is intentionally cheaper than the tactical planner. It follows the
    nearest copy of each visible digit, discounts contested races, and stops
    when the player's remaining turns cannot cover the route.
    """
    if not state.food_values:
        return 0
    head = state.head(side)
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    enemy_head = state.head(enemy)
    own_turns_left = max(0, (state.remaining_moves + (1 if side == state.side else 0)) // 2 - travel_cost)
    if own_turns_left <= 0 or head is None:
        return 0

    multiplier = max(1, state.multipliers.get(side, 1) if multiplier is None else multiplier)
    cursor = head
    enemy_cursor = enemy_head
    elapsed = 0
    projected = 0
    for digit, positions in state.numbered_groups(limit=limit):
        target = min(positions, key=lambda cell: (manhattan(cursor, cell), cell))
        our_distance = manhattan(cursor, target)
        arrival = elapsed + max(1, our_distance)
        if arrival > own_turns_left:
            break

        enemy_distance = None
        enemy_target = None
        if enemy_cursor is not None:
            enemy_target = min(positions, key=lambda cell: (manhattan(enemy_cursor, cell), cell))
            enemy_distance = manhattan(enemy_cursor, enemy_target)
        margin = tempo_race_margin(state, side, our_distance, enemy_distance)
        if margin >= 2:
            capture_percent = 100
        elif margin >= 0:
            capture_percent = 82
        elif margin == -1:
            capture_percent = 35
        else:
            capture_percent = 10

        time_percent = max(35, 100 - max(0, arrival - 4) * 3)
        reward = digit * FOOD_SCORE * multiplier
        projected += reward * capture_percent * time_percent // 10_000
        elapsed = arrival
        cursor = target
        if enemy_target is not None:
            enemy_cursor = enemy_target
    return projected


def estimated_multiplier_points(state, side, travel_cost=0):
    """Return the marginal future score of adding one multiplier level."""
    current = max(1, state.multipliers.get(side, 1))
    baseline = projected_sequence_income(state, side, travel_cost, current)
    boosted = projected_sequence_income(state, side, travel_cost, current + 1)
    return max(0, boosted - baseline)


def current_food_race(state, side):
    if not state.food:
        return None
    race = food_set_race(state, side)
    food = race['our_food'] or race['enemy_food'] or next(iter(state.food))
    return food, race['our_distance'], race['enemy_distance'], race['margin']


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
    race = food_set_race(state, side)
    enemy_current_distance = race['enemy_distance']
    if enemy_current_distance is None or enemy_current_distance < weights['sequence_hold_min_enemy_distance']:
        return None

    sequence = state.numbered_groups(limit=2)
    if len(sequence) < 2:
        return None
    next_digit, next_positions = sequence[1]
    after_eat = apply_move(state, plan.first_move, side)
    next_choice = best_numbered_copy(after_eat, side, next_positions)
    if next_choice is None:
        return None
    next_food, _path, our_next_distance, enemy_next_distance, next_margin = next_choice
    if (
        our_next_distance is None
        or enemy_next_distance is None
        or next_margin >= 0
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


def build_food_plan(state, side, food, current_target, hunger, food_race=None):
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
    if food in state.food and state.food_values:
        food_race = food_race or food_set_race(state, side)
        enemy_distance = food_race['enemy_distance']
    elif enemy_head is not None:
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
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    projected_income = projected_sequence_income(after, side)
    rival_projected_income = projected_sequence_income(after, enemy)
    economic_swing = projected_income - rival_projected_income
    multiplier_marginal = 0

    weights = get_weights()
    value = weights['food_base_value']
    value += max(0, reward - FOOD_SCORE) * weights['objective_point_value']
    economic_value = economic_swing * weights['economic_projection_value']
    economic_cap = weights['economic_projection_cap']
    value += max(-economic_cap, min(economic_cap, economic_value))
    if kind == 'multiplier':
        baseline_income = projected_sequence_income(state, side, our_distance)
        multiplier_marginal = max(0, projected_income - baseline_income)
        if multiplier_marginal == 0:
            multiplier_marginal = estimated_multiplier_points(state, side, our_distance)
        value += multiplier_marginal * weights['multiplier_future_point_value']
        value += weights['multiplier_base_value']
        multiplier_gap = state.multipliers.get(enemy, 1) - state.multipliers.get(side, 1)
        value += multiplier_gap * weights['multiplier_gap_value']
        if enemy_distance is not None and race_margin >= 0:
            enemy_marginal = estimated_multiplier_points(state, enemy, enemy_distance)
            value += enemy_marginal * weights['multiplier_denial_point_value']
        current_race = current_food_race(state, side)
        if current_race is not None and current_race[3] >= 0:
            value -= hunger * weights['multiplier_hunger_penalty']
        if multiplier_marginal < FOOD_SCORE * 5:
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
        projected_income=projected_income,
        rival_projected_income=rival_projected_income,
        economic_swing=economic_swing,
        multiplier_marginal=multiplier_marginal,
    )


def build_sequence_setup_plans(state, side, current_target, hunger):
    race = current_food_race(state, side)
    sequence = state.numbered_groups(limit=3)
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

    for rank, (digit, future_positions) in enumerate(sequence[1:3], start=1):
        future_choice = best_numbered_copy(state, side, future_positions)
        if future_choice is None:
            continue
        future_food = future_choice[0]
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
        projected_income = projected_sequence_income(after, side)
        rival_projected_income = projected_sequence_income(after, enemy)
        economic_swing = projected_income - rival_projected_income
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
        economic_value = economic_swing * weights['economic_projection_value']
        economic_cap = weights['economic_projection_cap']
        value += max(-economic_cap, min(economic_cap, economic_value))
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
                projected_income=projected_income,
                rival_projected_income=rival_projected_income,
                economic_swing=economic_swing,
            )
        )
    return plans


def build_food_plans(state, side, current_target=None, hunger=0):
    current_food_race = food_set_race(state, side) if state.food_values else None
    plans = [
        plan
        for food in state.objective_cells()
        if (
            plan := build_food_plan(
                state,
                side,
                food,
                current_target,
                hunger,
                food_race=current_food_race,
            )
        ) is not None
    ]
    plans.extend(build_sequence_setup_plans(state, side, current_target, hunger))
    return plans


def choose_food_plan(state, side, current_target=None, hunger=0, plans=None):
    plans = build_food_plans(state, side, current_target, hunger) if plans is None else list(plans)
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
            best_harvest = max(harvest, key=hunger_food_key)
            investments = [
                p for p in viable
                if p.objective_kind == 'multiplier'
                and p.safety == 'SAFE'
                and p.race_margin >= 0
                and p.our_distance <= best_harvest.our_distance + 3
            ]
            best_investment = max(investments, key=food_tiebreak_key) if investments else None
            strong_investment = (
                best_investment is not None
                and best_investment.multiplier_marginal >= max(500, best_harvest.objective_reward * 2)
                and best_investment.value >= best_harvest.value - get_weights()['food_plan_switch_margin']
            )
            if not strong_investment:
                return best_harvest

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

    if not state.food_values:
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

    race = food_set_race(state, side)
    enemy_distance = race['enemy_distance']
    before_distance = race['our_distance']
    if enemy_distance is None or enemy_distance > 5 or before_distance is None:
        return 0
    after_distances = food_distances(after, side, after.food)
    after_distance = min(after_distances.values(), default=None)
    if after_distance is None:
        return 0
    if wins_food_race(after, side, after_distance, enemy_distance) and after_distance < before_distance:
        weights = get_weights()
        progress = before_distance - after_distance
        return weights['fallback_denial_reach_bonus'] + progress * weights['fallback_denial_progress']
    return 0


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
    immediate_region = len(flood_region(after, head, strategic_blocked(after)))
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
        blocked = strategic_blocked(after_enemy)
        own_region = len(flood_region(after_enemy, our_head, blocked))
        enemy_region = 0 if enemy_head is None else len(flood_region(after_enemy, enemy_head, blocked))
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

    if not state.food_values:
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

    race = food_set_race(state, side)
    our_distance = race['our_distance']
    enemy_distance = race['enemy_distance']
    if (
        our_distance is None
        or enemy_distance is None
        or enemy_distance >= our_distance
        or enemy_distance > 2
    ):
        return 0
    after_distances = food_distances(after, side, after.food)
    after_distance = min(after_distances.values(), default=None)
    if (
        after_distance is None
        or after_distance >= our_distance
        or (our_distance > 6 and after_distance > 3)
    ):
        return 0
    weights = get_weights()
    return weights['hot_lost_food_base'] + (3 - enemy_distance) * weights['hot_lost_food_enemy_bonus']


def winning_suicide_move(state, side):
    """Return a guaranteed crash that ends the match with us still ahead."""
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    projected_our_score = state.scores.get(side, 0) + CRASH_PENALTY
    projected_enemy_score = state.scores.get(enemy, 0) + RIVAL_CRASH_REWARD
    if projected_our_score <= projected_enemy_score:
        return None

    snake = state.body(side)
    head = state.head(side)
    if not snake or head is None:
        return None
    occupied_snakes = set()
    for body in state.snakes.values():
        occupied_snakes.update(body)
    own_tail = snake[-1]
    for direction in DIRS:
        target = step(head, direction)
        if not (0 <= target[0] < state.rows and 0 <= target[1] < state.cols):
            return direction
        if target in state.walls:
            continue
        if target in occupied_snakes:
            if len(snake) > 1 and target == snake[1]:
                return direction
            if target == own_tail and side in state.reliable_tails:
                continue
            return direction
    return None


def forced_kill_move(state, side, legal):
    enemy = state.enemy if side == state.side else ('B' if side == 'A' else 'A')
    for direction in legal:
        after = apply_move(state, direction, side)
        enemy_moves = legal_moves(after, enemy)
        own_head = after.head(side)
        if own_head is None:
            continue
        own_region = flood_region(after, own_head, strategic_blocked(after))
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
    region = flood_region(state, head, strategic_blocked(state))
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
    region = flood_region(after, head, strategic_blocked(after)) if head is not None else set()
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
    wall_hit = step(state.head(side), direction) in state.walls
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
    region = flood_region(after, head, strategic_blocked(after))
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
        - danger_penalty
        - (weights['fallback_wall_hit_penalty'] if wall_hit else 0),
        1 if head in state.food else 0,
        exits,
        min(len(region), 60),
    )


def fallback_food_positioning(state, side, legal, plan, hunger, repeat_count):
    return max(legal, key=lambda direction: move_quality_score(state, side, direction, plan, hunger, repeat_count))
