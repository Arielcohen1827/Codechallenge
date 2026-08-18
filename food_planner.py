from dataclasses import dataclass

from snake_state import (
    GameState,
    apply_move,
    count_exits,
    direction_between,
    flood_region,
    legal_moves,
    manhattan,
    shortest_distance,
    shortest_path,
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
        )
    return current


def can_reach_tail(state, side):
    snake = state.body(side)
    head = state.head(side)
    if not snake or head is None:
        return False
    return shortest_path(state, head, snake[-1], side) is not None


def nearest_next_food_distance(state, side, consumed_food):
    head = state.head(side)
    if head is None:
        return None
    distances = [
        shortest_distance(state, head, food, side)
        for food in state.food
        if food != consumed_food
    ]
    distances = [d for d in distances if d is not None]
    return min(distances) if distances else None


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

    max_center_distance = (state.rows - 1) / 2 + (state.cols - 1) / 2
    center_score = int((max_center_distance - center_distance(state, pos)) * 80)
    edge = edge_distance(state, pos)
    edge_score = min(edge, 4) * 450
    if edge == 0:
        edge_score -= 3200
    elif edge == 1:
        edge_score -= 1000
    return center_score + edge_score


def position_control_penalty(state, side, direction):
    if direction not in legal_moves(state, side):
        return 1_000_000

    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return 1_000_000
    if len(after.body(side)) <= 3:
        return 0

    edge = edge_distance(after, head)
    penalty = 0
    if edge == 0:
        penalty += 6500
    elif edge == 1:
        penalty += 1800
    elif edge == 2:
        penalty += 600

    borders_touching = sum(
        (
            head[0] == 0,
            head[1] == 0,
            head[0] == after.rows - 1,
            head[1] == after.cols - 1,
        )
    )
    if borders_touching >= 2:
        penalty += 2500
    return penalty


def build_food_plan(state, side, food, current_target, hunger):
    head = state.head(side)
    enemy_head = state.head(state.enemy)
    if head is None:
        return None
    path = shortest_path(state, head, food, side)
    if path is None or len(path) < 2:
        return None

    after = simulate_path_to_food(state, side, path)
    if after is None:
        return None

    enemy_distance = None
    if enemy_head is not None:
        enemy_distance = shortest_distance(state, enemy_head, food, state.enemy)
    our_distance = len(path) - 1
    race_margin = 6 if enemy_distance is None else enemy_distance - our_distance
    safety, space_after, exits_after, tail_ok = classify_safety(after, side)
    next_food = nearest_next_food_distance(after, side, food)

    value = 10_000
    value -= our_distance * 550
    value += race_margin * 700
    value += min(hunger, 50) * 220
    value += min(space_after, 80) * 16
    value += exits_after * 260
    value += 450 if tail_ok else 0
    value += center_control_score(after, after.head(side))
    if next_food is not None:
        value += max(0, 12 - next_food) * 170
    if food == current_target:
        value += 1800
    if safety == 'SAFE':
        value += 2000
    elif safety == 'ACCEPTABLE_RISK':
        value += 700
    elif safety == 'DANGEROUS':
        value -= 6500
    else:
        value -= 100_000
    if enemy_distance is not None and enemy_distance + 3 < our_distance:
        value -= 18_000

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
    )


def choose_food_plan(state, side, current_target=None, hunger=0):
    plans = [
        plan
        for food in state.food
        if (plan := build_food_plan(state, side, food, current_target, hunger)) is not None
    ]
    viable = [p for p in plans if p.safety in ('SAFE', 'ACCEPTABLE_RISK')]
    if not viable:
        viable = [p for p in plans if p.safety == 'DANGEROUS' and hunger >= 35 and p.race_margin >= 0]
    if not viable:
        return None

    immediate = [p for p in viable if p.our_distance == 1]
    if immediate:
        return max(immediate, key=food_tiebreak_key)

    urgent_denials = [
        p
        for p in viable
        if p.enemy_distance is not None
        and p.enemy_distance <= 4
        and p.our_distance <= p.enemy_distance
    ]
    if urgent_denials:
        return max(urgent_denials, key=denial_key)

    race_viable = [
        p
        for p in viable
        if p.enemy_distance is None or p.our_distance <= p.enemy_distance
    ]
    if race_viable:
        viable = race_viable

    nearest_distance = min(p.our_distance for p in viable)
    current = next((p for p in viable if p.food == current_target), None)
    if current and current.our_distance <= nearest_distance + 2 and current.race_margin >= -1:
        return current

    close_enough = [p for p in viable if p.our_distance <= nearest_distance + 1]
    return max(close_enough, key=food_tiebreak_key)


def food_tiebreak_key(plan):
    safety_rank = {'SAFE': 3, 'ACCEPTABLE_RISK': 2, 'DANGEROUS': 1}.get(plan.safety, 0)
    race_rank = 0 if plan.enemy_distance is None else max(-4, min(4, plan.race_margin))
    next_food = 0 if plan.next_food_distance is None else max(0, 12 - plan.next_food_distance)
    return (
        safety_rank,
        race_rank,
        -plan.our_distance,
        next_food,
        plan.space_after,
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
    if plan.safety == 'SUICIDAL':
        return False
    if plan.our_distance == 1 and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
        return True
    if plan.race_margin < 0:
        return False
    if repeat_count >= 2 and plan.safety != 'SUICIDAL':
        return True
    if hunger >= 20 and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
        return True
    return plan.safety in ('SAFE', 'ACCEPTABLE_RISK') and plan.value > 0


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
    return -(
        stats['zero_escape_replies'] * 9000
        + stats['delayed_forced_zero_replies'] * 18000
        + stats['one_escape_replies'] * 2500
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
    base = 0
    if immediate_region < body_len * 2 + 6:
        base += (body_len * 2 + 6 - immediate_region) * 1400
    if immediate_exits <= 1:
        base += 3500

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
            threat += 80_000
        if own_region < body_len * 2 + 6:
            threat += (body_len * 2 + 6 - own_region) * 1400
        if count_exits(after_enemy, side) <= 1:
            threat += 3500
        territory_gap = enemy_region - own_region
        if territory_gap > 35:
            threat += min(25_000, (territory_gap - 35) * 250)
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

    penalty = 0
    for food in state.food:
        our_distance = shortest_distance(state, head, food, side)
        enemy_distance = shortest_distance(state, enemy_head, food, enemy)
        if (
            our_distance is None
            or enemy_distance is None
            or enemy_distance >= our_distance
            or enemy_distance > 2
        ):
            continue
        after_distance = shortest_distance(after, after_head, food, side)
        if (
            after_distance is not None
            and after_distance < our_distance
            and (our_distance <= 6 or after_distance <= 3)
        ):
            penalty += 2500 + (3 - enemy_distance) * 1200
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
        if not enemy_moves and len(own_region) >= len(after.body(side)) + 3:
            return direction
    return None


def fallback_food_positioning(state, side, legal, plan, hunger, repeat_count):
    def key(direction):
        after = apply_move(state, direction, side)
        head = after.head(side)
        if head is None:
            return (-999999, 0, 0, 0)
        safety, _, _, _ = classify_safety(after, side)
        if safety == 'SUICIDAL':
            return (-999999, 0, 0, 0)
        danger_penalty = 10000 if safety == 'DANGEROUS' and hunger < 35 else 0
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
        food_distance_score = 0 if nearest is None else -nearest * 80
        cycle_penalty = repeat_count * 500 if progress <= 0 and hunger >= 10 else 0
        rival_penalty = rival_response_penalty(state, side, direction)
        hot_lost_penalty = hot_lost_food_penalty(state, side, direction)
        control_score = center_control_score(after, head) if len(after.body(side)) > 3 else 0
        return (
            progress * (1000 + hunger * 80)
            + food_distance_score
            + control_score
            - cycle_penalty
            + rival_penalty
            - hot_lost_penalty
            - danger_penalty,
            1 if head in state.food else 0,
            exits,
            min(len(region), 60),
        )

    return max(legal, key=key)
