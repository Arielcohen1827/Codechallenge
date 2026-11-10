from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from snake_bot.bot_weights import get_weights
from snake_bot.game_rules import (
    MULTIPLIER_PICKUP_SCORE,
    food_points,
    tail_food_reward,
    shot_food_reward,
    bombs_active_on_turn,
    STAR_TURNS,
)
from snake_bot.snake_state import (
    CRASH_PENALTY,
    FOOD_SCORE,
    apply_move as official_apply_move,
    available_moves,
    blocked_for_path,
    count_exits,
    direction_between,
    flood_region,
    legal_moves,
    manhattan,
    neighbors,
    known_neighbors,
    shortest_path as geometric_shortest_path,
    shrink_wall,
    step,
    strategic_blocked,
    temporal_distance_map,
    temporal_shortest_path as geometric_temporal_shortest_path,
    path_blast_blocked,
)

_planning_cache = ContextVar('food_planning_cache', default=None)


@contextmanager
def _planning_scope():
    """Share immutable transition results only within one planning decision."""
    existing = _planning_cache.get()
    if existing is not None:
        yield existing
        return
    token = _planning_cache.set({'states': {}, 'simulations': {}, 'paths': {}, 'distances': {}})
    try:
        yield _planning_cache.get()
    finally:
        _planning_cache.reset(token)


planning_scope = _planning_scope


def move_outcomes(state, direction, side=None):
    """Simulate both histories without treating an unknown snapshot as initial."""
    side = side or state.side
    field = {'A': 'crashed_a', 'B': 'crashed_b'}[side]
    if getattr(state, field) is not None or state.scores.get(side, 0) <= 0:
        return (official_apply_move(state, direction, side),)
    first = official_apply_move(replace(state, **{field: False}), direction, side)
    if getattr(first, field) is True:
        repeated = official_apply_move(replace(state, **{field: True}), direction, side)
        return (first, repeated)
    return (replace(first, **{field: None}),)


def apply_move(state, direction, side=None, perspective=None):
    """Conservative simulated successor, never a persisted history prediction."""
    side = side or state.side
    perspective = perspective or side
    enemy = 'B' if perspective == 'A' else 'A'

    def outcome_key(result):
        winner = result.terminal_winner()
        terminal_rank = -1 if winner == enemy else 1 if winner == perspective else 0
        return (
            terminal_rank if result.is_terminal() else 0,
            result.scores.get(perspective, 0) - result.scores.get(enemy, 0),
        )

    return min(move_outcomes(state, direction, side), key=outcome_key)


@dataclass(frozen=True)
class FoodPlan:
    """A reachable observed objective with route, safety and economic estimates."""

    food: tuple[int, ...]
    path: tuple[tuple[int, ...], ...]
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
    bomb_loss: int = 0
    bomb_trigger_count: int = 0

    @property
    def first_move(self):
        if len(self.path) < 2:
            return None
        return direction_between(self.path[0], self.path[1])


def simulate_path_to_food(state, side, path):
    return simulate_path_economy(state, side, path)[0]


def simulate_path_economy(state, side, path):
    cache = _planning_cache.get()
    if cache is None:
        return _simulate_path_economy(state, side, path)
    # Keep a strong reference so recycled ids cannot alias another state.
    cache['states'][id(state)] = state
    key = (id(state), side, tuple(path))
    if key not in cache['simulations']:
        cache['simulations'][key] = _simulate_path_economy(state, side, path)
    return cache['simulations'][key]


def _simulate_path_economy(state, side, path):
    """Project one route while reusing cached prefixes within the decision."""
    return PathEconomy(state, side).run(path)


class PathEconomy:
    """Track deterministic route effects, bomb losses and cached prefix results."""

    def __init__(self, state, side):
        self.initial, self.current, self.side = state, state, side
        self.bomb_loss = self.trigger_count = 0
        self.cache = _planning_cache.get()

    def result(self):
        """Return the projected state and accumulated explosion economics."""
        return self.current, self.bomb_loss, self.trigger_count

    def run(self, path):
        """Advance only uncached route prefixes, stopping on invalid movement."""
        for number, cell in enumerate(path[1:], 1):
            key = (id(self.initial), self.side, tuple(path[: number + 1]))
            if not self.restore(key):
                self.advance(cell, number)
                self.remember(key)
            if self.current is None:
                break
        return self.result()

    def restore(self, key):
        """Load an existing projected prefix while preserving the original state reference."""
        if self.cache is None or key not in self.cache['simulations']:
            return False
        self.current, self.bomb_loss, self.trigger_count = self.cache['simulations'][key]
        return True

    def remember(self, key):
        """Save successful and blocked prefixes for other objective routes."""
        if self.cache is not None:
            self.cache['simulations'][key] = self.result()

    def advance(self, cell, number):
        """Reject terminal or blocked prefixes before applying the common rules."""
        if self.current.is_terminal() or cell in self.current.walls:
            self.current = None
            return
        self.current = route_bomb_expiry(self.initial, self.current, self.side, number)
        direction = direction_between(self.current.head(self.side), cell)
        if direction is None or direction not in legal_moves(self.current, self.side):
            self.current = None
            return
        self.current = apply_move(self.current, direction, self.side)
        self.record_explosion()
        self.current = route_next_turn(self.current, self.side)

    def record_explosion(self):
        """Keep explosion counts separate from protection-dependent score loss."""
        metadata = self.current.last_action
        if metadata.get('bomb_triggered'):
            self.trigger_count += 1
            if not metadata.get('invincibility_before', 0):
                self.bomb_loss += metadata['score_before'] - metadata['score_after'] + 1


def route_bomb_expiry(initial, current, side, number):
    """Account for intervening rival turns without changing the X income horizon."""
    if bombs_active_on_turn(initial.remaining_moves, side == initial.side, number):
        return current
    return replace(
        current, bombs_large=frozenset(), bombs_small=frozenset(), pending_bomb_respawns=()
    )


def opposite(side):
    """Return the other player identifier."""
    return 'B' if side == 'A' else 'A'


def route_next_turn(current, side):
    """Restore the projected actor and apply the missing opposing wall erosion."""
    walls = shrink_wall(current.walls) if side == 'A' and not current.partial else current.walls
    enemy = current.enemy if current.enemy != side else opposite(side)
    return replace(current, side=side, enemy=enemy, walls=walls)


def economical_path(state, start, goal, side, finder, max_time=None):
    cache = _planning_cache.get()
    if cache is None:
        return _economical_path(state, start, goal, side, finder, max_time)
    cache['states'][id(state)] = state
    key = (id(state), start, goal, side, finder, max_time)
    if key not in cache['paths']:
        cache['paths'][key] = _economical_path(state, start, goal, side, finder, max_time)
    return cache['paths'][key]


def _economical_path(state, start, goal, side, finder, max_time=None):
    """Compare two legal routes using the existing point/distance valuation."""
    options = {} if max_time is None else {'max_time': max_time}
    direct = finder(state, start, goal, side, **options)
    if not detour_is_relevant(state, start, side, direct):
        return direct
    if not path_has_blast_exposure(state, side, direct):
        return direct
    detour = finder(state, start, goal, side, avoid_blast=True, **options)
    if not distinct_detour(direct, detour):
        return direct
    return lowest_economic_route(state, side, direct, detour)


def detour_is_relevant(state, start, side, direct):
    """Only compare blast costs for a known path from the actor before bomb expiry."""
    return (
        direct is not None
        and start == state.head(side)
        and active_bombs_present(state)
        and state.remaining_moves > 100
    )


def active_bombs_present(state):
    """Test both observed bomb collections without materializing their zones."""
    return bool(state.bombs_large or state.bombs_small)


def path_has_blast_exposure(state, side, path):
    """Respect bomb expiry separately for each projected arrival on the route."""
    return any(
        path_blast_blocked(state, cell, side, index) for index, cell in enumerate(path[1:], 1)
    )


def distinct_detour(direct, detour):
    return detour is not None and detour != direct


def lowest_economic_route(state, side, direct, detour):
    """Preserve the distance floor and point-loss comparison of the two routes."""
    weights = get_weights()
    candidates = []
    for path in (direct, detour):
        distance_cost = (len(path) - 1) * weights['food_distance_cost']
        if route_floor_reached(candidates, distance_cost):
            # The blast-avoiding route cannot beat its distance-only floor.
            break
        after, loss, _ = simulate_path_economy(state, side, path)
        if after is not None:
            cost = distance_cost + loss * weights['objective_point_value']
            candidates.append((cost, len(path), path))
    return min(candidates)[2] if candidates else direct


def route_floor_reached(candidates, distance_cost):
    return bool(candidates) and candidates[0][0] <= distance_cost


def temporal_shortest_path(state, start, goal, side, max_time=None):
    return economical_path(state, start, goal, side, geometric_temporal_shortest_path, max_time)


def temporal_shortest_distance(state, start, goal, side, max_time=None):
    path = temporal_shortest_path(state, start, goal, side, max_time)
    return None if path is None else len(path) - 1


def shortest_path(state, start, goal, side):
    return economical_path(state, start, goal, side, geometric_shortest_path)


def shortest_distance(state, start, goal, side):
    path = shortest_path(state, start, goal, side)
    return None if path is None else len(path) - 1


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
        extend_distance_frontier(state, current, blocked, distances, q)
    return distances


def extend_distance_frontier(state, current, blocked, distances, queue):
    """Visit each unblocked known neighbor once in the distance traversal."""
    for cell in known_neighbors(state, current):
        if cell not in distances and cell not in blocked:
            distances[cell] = distances[current] + 1
            queue.append(cell)


def tempo_race_margin(state, side, our_distance, enemy_distance):
    if enemy_distance is None:
        return 6
    margin = enemy_distance - our_distance
    if margin == 0:
        return 1 if side == state.side else -1
    return margin


def food_objectives(state):
    """Edible food targets for shared geometry and existing positional weights."""
    return state.food | state.shot_food | state.stars


def food_distances(state, side, positions=None):
    """Return temporal distances to every reachable copy in one objective set."""
    head = state.head(side)
    if head is None:
        return {}
    positions = tuple(food_objectives(state) if positions is None else positions)
    cache = _planning_cache.get()
    key = (id(state), side, positions)
    existing = cached_food_distances(cache, state, key)
    if existing is not None:
        return existing
    result = reachable_food_distances(state, head, side, positions)
    if cache is not None:
        cache['distances'][key] = result
    return dict(result)


def cached_food_distances(cache, state, key):
    """Return an independent cached mapping, including a cached empty result."""
    if cache is None:
        return None
    cache['states'][id(state)] = state
    if key in cache['distances']:
        return dict(cache['distances'][key])
    return None


def reachable_food_distances(state, head, side, positions):
    """Restrict one temporal distance map to the requested objective copies."""
    distances = temporal_distance_map(state, head, side, goals=positions)
    return {food: distances[food] for food in positions if food in distances}


def food_set_race(state, side, positions=None):
    """Compare each players fastest confirmed route to any current target copy."""
    if positions is None:
        positions = state.food if state.food_values else food_objectives(state)
    positions = frozenset(positions)
    enemy = state.enemy if side == state.side else opposite(side)
    ours = food_distances(state, side, positions)
    theirs = food_distances(state, enemy, positions)
    return food_race_summary(state, side, ours, theirs)


def best_distance_entry(distances):
    """Return a stable nearest target and distance, or two absent values."""
    if not distances:
        return None, None
    return min(distances.items(), key=lambda item: (item[1], item[0]))


def food_race_summary(state, side, ours, theirs):
    """Preserve per-copy maps together with turn-order-adjusted race margins."""
    our_food, our_distance = best_distance_entry(ours)
    enemy_food, enemy_distance = best_distance_entry(theirs)
    margin = (
        -99
        if our_distance is None
        else tempo_race_margin(state, side, our_distance, enemy_distance)
    )
    return dict(
        our_food=our_food,
        our_distance=our_distance,
        enemy_food=enemy_food,
        enemy_distance=enemy_distance,
        margin=margin,
        our_reachable=len(ours),
        enemy_reachable=len(theirs),
        our_distances=ours,
        enemy_distances=theirs,
    )


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
    """Find the closest remaining objective using temporal routes for numbered food."""
    head = state.head(side)
    if head is None:
        return None
    if state.food_values:
        distances = numbered_followup_distances(state, side, head)
    else:
        distances = distances if distances is not None else distance_map(state, head, side)
        distances = remaining_objective_distances(state, consumed_food, distances)
    return min(distances.values(), default=None)


def numbered_followup_distances(state, side, head):
    """Compute reachable distances to current numbered target copies."""
    distances = {}
    for food in state.food:
        path = temporal_shortest_path(state, head, food, side)
        if path is not None:
            distances[food] = len(path) - 1
    return distances


def remaining_objective_distances(state, consumed_food, distances):
    """Remove the consumed objective and unreachable cells from a supplied map."""
    return {
        food: distances[food]
        for food in food_objectives(state)
        if food != consumed_food and food in distances
    }


def food_chain_score(state, side, consumed_food, hunger, distances=None):
    """Estimate the unchanged next-food chain contribution after an arrival."""
    head = state.head(side)
    if head is None:
        return 0
    if state.food_values:
        return numbered_chain_score(state, side, hunger)
    distances = distances if distances is not None else distance_map(state, head, side)
    return unnumbered_chain_score(state, side, consumed_food, hunger, distances)


def next_chain_choice(current, side):
    """Require an active numbered target and a route with a real first step."""
    if current.next_food_digit is None or not current.food:
        return None
    choice = best_numbered_copy(current, side, current.food)
    if choice is None or len(choice[1]) < 2:
        return None
    return choice


def numbered_chain_score(state, side, hunger):
    """Project up to three successive targets with the existing rank discount."""
    current, total = state, 0
    weights = get_weights()
    for rank in range(3):
        choice = next_chain_choice(current, side)
        if choice is None:
            break
        _, path, distance, _, margin = choice
        reward = food_points(current.next_food_digit, current.multipliers.get(side, 1))
        score = reward * weights['sequence_reward_point_value']
        score += chain_route_value(distance, margin, hunger, weights)
        if margin < 0:
            score //= 4
        total += score // (rank + 1)
        current = simulate_path_to_food(current, side, path)
        if current is None:
            break
    return total


def unnumbered_chain_score(state, side, consumed_food, hunger, distances):
    """Keep the best ordinary-food follow-up under geometric rival race estimates."""
    enemy = state.enemy if side == state.side else opposite(side)
    enemy_head = state.head(enemy)
    candidates = remaining_objective_distances(state, consumed_food, distances)
    values = (
        unnumbered_chain_value(state, side, food, distance, enemy_head, hunger)
        for food, distance in candidates.items()
    )
    return max(values, default=0)


def unnumbered_chain_value(state, side, food, distance, enemy_head, hunger):
    """Combine one ordinary-food race with the existing chain coefficients."""
    enemy_distance = None if enemy_head is None else manhattan(enemy_head, food)
    margin = tempo_race_margin(state, side, distance, enemy_distance)
    return chain_route_value(distance, margin, hunger, get_weights())


def chain_route_value(distance, margin, hunger, weights):
    """Value proximity and race margin without changing chain thresholds."""
    score = max(0, 12 - distance) * weights['chain_food_value']
    score += max(-2, min(6, margin)) * weights['chain_race_value']
    if distance <= 4 and margin >= -1:
        score += weights['chain_close_food_bonus']
    return score + hungry_chain_value(distance, margin, hunger, weights)


def hungry_chain_value(distance, margin, hunger, weights):
    """Apply the established hunger bonus only to sufficiently close winnable food."""
    if hunger >= 14 and distance <= 7 and margin >= -1:
        return (hunger - 13) * max(0, 8 - distance) * weights['hunger_close_food_value']
    return 0


def score_gap_urgency(state, side):
    enemy = state.enemy if side == state.side else opposite(side)
    gap = state.scores.get(enemy, 0) - state.scores.get(side, 0)
    if gap <= 0:
        return 0
    food_gap = gap // 100
    return min(14, food_gap * 2 + gap // 400)


def food_cluster_followup_score(state, food):
    weights = get_weights()
    best = 0
    candidates = followup_candidates(state)
    for other in candidates:
        if other == food:
            continue
        distance = manhattan(food, other)
        if distance <= 7:
            best = max(best, (8 - distance) * weights['rival_chain_threat_value'])
    return best


def followup_candidates(state):
    """Return future numbered copies or the current ordinary-food objective set."""
    if state.food_values:
        return [cell for _, positions in state.numbered_groups(limit=3)[1:] for cell in positions]
    return food_objectives(state)


def rival_food_threat_score(state, side, food, our_distance, enemy_distance):
    if enemy_distance is None:
        return 0

    weights = get_weights()
    enemy = state.enemy if side == state.side else opposite(side)
    threat = 0
    if enemy_distance <= our_distance + 2:
        threat += max(0, 8 - enemy_distance) * weights['rival_food_threat_value']
        threat += food_cluster_followup_score(state, food)
        rival_reward = rival_objective_reward(state, enemy, food)
        threat += rival_reward * weights['rival_reward_point_value']
    if tempo_race_margin(state, side, our_distance, enemy_distance) < 0:
        threat += (our_distance - enemy_distance) * weights['rival_food_threat_value']
    return threat


def rival_objective_reward(state, enemy, food):
    """Respect beneficiary-specific tail rewards in rival-food threat estimates."""
    if food in state.tail_food_a or food in state.tail_food_b:
        return objective_reward(state, enemy, food)
    return state.food_reward(enemy, food)


def food_denial_score(state, side, food, our_distance, enemy_distance, rival_threat):
    if not can_deny_rival_food(state, side, our_distance, enemy_distance):
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


def can_deny_rival_food(state, side, ours, theirs):
    """Require a known rival arrival and a turn-order-winning food race."""
    return theirs is not None and wins_food_race(state, side, ours, theirs)


def advantage_control_context(state, side):
    """Describe whether a score lead is large enough to defend positionally."""
    weights = get_weights()
    enemy = state.enemy if side == state.side else opposite(side)
    lead = state.scores.get(side, 0) - state.scores.get(enemy, 0)
    own_length = len(state.body(side))
    enemy_length = len(state.body(enemy))
    own_turns_left = remaining_actor_turns(state, side)
    recoverable_threat = recoverable_food_threat(state, enemy)
    current_enemy_reward = max(
        (state.food_reward(enemy, food) for food in food_objectives(state)),
        default=0,
    )
    crash_score_swing = CRASH_PENALTY
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
    active = advantage_enabled(
        state, lead, activation_lead, own_length, enemy_length, own_turns_left, weights
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


def recoverable_food_threat(state, enemy):
    """Estimate the existing near-term rival food and multiplier opportunity."""
    if state.food_values:
        threat = projected_sequence_income(state, enemy, limit=5)
        if state.pickups:
            threat += estimated_multiplier_points(state, enemy) * min(2, len(state.pickups))
    else:
        threat = min(2, len(food_objectives(state))) * FOOD_SCORE
    return threat + min(2, len(state.pickups)) * MULTIPLIER_PICKUP_SCORE


def advantage_enabled(state, lead, required, length, rival_length, turns, weights):
    """Require enabled control, real food and enough score lead before body checks."""
    return (
        weights['advantage_control_enabled'] > 0
        and bool(food_objectives(state))
        and lead >= required
        and advantage_body_ready(length, rival_length, turns, weights)
    )


def advantage_body_ready(length, rival_length, turns, weights):
    """Keep the original body-deficit and remaining-time activation thresholds."""
    return (
        length >= weights['advantage_min_body']
        and length + weights['advantage_body_deficit'] >= rival_length
        and turns >= weights['advantage_min_own_turns']
    )


def advantage_control_move(
    state, side, legal, deep_moves, reference_move=None, adjacent_food=False
):
    """Choose a safe move that delays the rival while preserving our mobility."""
    return AdvantageControl(state, side, legal, deep_moves).choose(reference_move, adjacent_food)


class AdvantageControl:
    """Compare rival food access only after the existing safety gates have passed."""

    def __init__(self, state, side, legal, deep_moves):
        self.state, self.side, self.legal, self.deep_moves = state, side, legal, deep_moves
        self.context = advantage_control_context(state, side)
        self.context.update(moves=[], direction=None, override=False)
        self.weights = get_weights()
        self.enemy = state.enemy if side == state.side else opposite(side)

    def choose(self, reference_move, adjacent_food):
        """Return control diagnostics and an optional override after all safety gates."""
        if not self.context['active'] or not self.deep_moves:
            return self.context
        if self.state.head(self.enemy) is None:
            return self.context
        self.prepare_food_race()
        candidates = self.candidate_moves()
        candidates.sort(key=lambda item: item['score'], reverse=True)
        self.context['moves'] = candidates
        return self.select_candidate(candidates, reference_move, adjacent_food)

    def prepare_food_race(self):
        distances = food_distances(self.state, self.enemy)
        reachable = [(distance, food) for food, distance in distances.items()]
        self.threatened_food = (
            min(reachable)[1] if reachable else next(iter(food_objectives(self.state)), None)
        )
        self.before_distance = min(distances.values(), default=None)
        self.before_copies = len(distances)
        self.baseline_moves = len(legal_moves(self.state, self.enemy))
        self.context.update(
            baseline_enemy_moves=self.baseline_moves, enemy_reachable_copies=self.before_copies
        )
        self.analyses = {item['direction']: item for item in self.deep_moves}

    def candidate_moves(self):
        candidates = []
        for direction in self.legal:
            candidate = self.candidate(direction)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def candidate(self, direction):
        analysis = self.analyses.get(direction)
        if not advantage_analysis_is_safe(analysis):
            return None
        after = apply_move(self.state, direction, self.side)
        enemy_moves = len(legal_moves(after, self.enemy))
        if enemy_moves == 0:
            return None
        secured = step(self.state.head(self.side), direction) in food_objectives(self.state)
        distance, delay, unreachable, distances = self.food_delay(after, secured)
        denied = max(0, self.before_copies - len(distances))
        score = self.candidate_score(analysis, enemy_moves, delay, denied, unreachable)
        return {
            'direction': direction,
            'score': score,
            'enemy_moves': enemy_moves,
            'enemy_food_distance_before': self.before_distance,
            'enemy_food_distance_after': distance,
            'food_delay': delay,
            'food_unreachable': unreachable,
            'enemy_reachable_copies_after': len(distances),
            'copies_denied': denied,
            'secured_food': secured,
            'meaningful_denial': any(
                (
                    secured,
                    unreachable,
                    delay > 0,
                    denied > 0,
                    enemy_moves < self.baseline_moves,
                )
            ),
            'voronoi_ours': analysis.get('voronoi_ours', 0),
            'voronoi_enemy': analysis.get('voronoi_enemy', 0),
        }

    def food_delay(self, after, secured):
        if not food_objectives(self.state) or secured:
            return None, 0, False, {}
        distances = food_distances(after, self.enemy, food_objectives(after))
        distance = min(distances.values(), default=None)
        if distance is None:
            return distance, 0, True, distances
        delay = (
            0 if self.before_distance is None else max(-3, min(5, distance - self.before_distance))
        )
        return distance, delay, False, distances

    def candidate_score(self, analysis, enemy_moves, delay, denied, unreachable):
        weights = self.weights
        score = analysis['score']
        score += analysis.get('voronoi_ours', 0) * weights['advantage_our_cell_value']
        score -= analysis.get('voronoi_enemy', 0) * weights['advantage_enemy_cell_value']
        score += (
            max(0, self.baseline_moves - enemy_moves) * weights['advantage_enemy_move_reduction']
        )
        score += delay * weights['advantage_food_delay_value']
        score += denied * weights['advantage_copy_denial_value']
        if unreachable:
            score += weights['advantage_food_unreachable_bonus']
        if analysis.get('enemy_food'):
            score -= weights['advantage_enemy_food_penalty']
        return score

    def select_candidate(self, candidates, reference_move, adjacent_food):
        if not candidates or not candidates[0]['meaningful_denial']:
            return self.context
        best = candidates[0]
        reference = next(
            (item for item in candidates if item['direction'] == reference_move), None
        )
        return self.commit_candidate(best, reference, reference_move, adjacent_food)

    def commit_candidate(self, best, reference, reference_move, adjacent_food):
        margin = self.weights['advantage_switch_margin']
        if adjacent_food:
            margin += self.weights['advantage_adjacent_food_switch_margin']
        improvement = None if reference is None else best['score'] - reference['score']
        self.context.update(
            direction=best['direction'],
            reference_direction=reference_move,
            improvement=improvement,
            required_margin=margin,
            override=advantage_should_override(
                best, reference, reference_move, improvement, margin
            ),
            threatened_food=self.threatened_food,
        )
        return self.context


def advantage_analysis_is_safe(analysis):
    if analysis is None:
        return False
    if analysis.get('forced_loss') or analysis.get('safety') != 'SAFE':
        return False
    return analysis.get('our_replies', 0) >= 2 and analysis.get('exits', 0) >= 2


def advantage_should_override(best, reference, reference_move, improvement, margin):
    return reference_move is None or (
        best['direction'] != reference_move and (reference is None or improvement >= margin)
    )


def classify_safety(after, side):
    """Return the safety category, reachable volume, exits and tail reachability."""
    head = after.head(side)
    if head is None:
        return 'SUICIDAL', 0, 0, False
    region = flood_region(after, head, strategic_blocked(after))
    body_len = len(after.body(side))
    exits = count_exits(after, side)
    tail_ok = can_reach_tail(after, side)
    category = safety_category(after, side, len(region), body_len, exits, tail_ok)
    return category, len(region), exits, tail_ok


def safety_category(after, side, space, length, exits, tail_ok):
    """Classify the measured region while respecting protected stationary impacts."""
    if region_traps_actor(space, length, exits):
        return 'ACCEPTABLE_RISK' if after.invincibility(side) else 'SUICIDAL'
    if dangerous_region(space, length, exits, tail_ok):
        return 'DANGEROUS'
    if limited_safe_region(space, length, exits):
        return 'ACCEPTABLE_RISK'
    return 'SAFE'


def limited_safe_region(space, length, exits):
    """Retain the established single-exit and small-region safety thresholds."""
    return exits == 1 or space < length * 2 + 4


def region_traps_actor(space, length, exits):
    """Identify the established no-exit or near-body-size trap threshold."""
    return exits == 0 or space <= max(2, length)


def dangerous_region(space, length, exits, tail_ok):
    """Detect insufficient escape volume unless a tail route remains available."""
    if tail_ok:
        return False
    return space < length + 4 or (exits == 1 and space < length * 2 + 6)


def edge_distance(state, pos):
    """Distance to the nearest face in the active two- or three-axis geometry."""
    if pos is None:
        return 0
    return min(min(axis, size - 1 - axis) for axis, size in zip(pos, state_dimensions(state)))


def state_dimensions(state):
    """Return axis sizes in coordinate order, preserving historical 2D boards."""
    return (state.levels, state.rows, state.cols) if state.levels > 1 else (state.rows, state.cols)


def center_distance(state, pos):
    """Manhattan distance from a cell to the geometric board or cube centre."""
    if pos is None:
        return sum(state_dimensions(state))
    return sum(abs(axis - (size - 1) / 2) for axis, size in zip(pos, state_dimensions(state)))


def center_control_score(state, pos):
    if pos is None:
        return -10_000
    weights = get_weights()

    max_center_distance = sum((size - 1) / 2 for size in state_dimensions(state))
    center_score = int(
        (max_center_distance - center_distance(state, pos)) * weights['center_unit_value']
    )
    edge = edge_distance(state, pos)
    edge_score = min(edge, 4) * weights['edge_depth_value']
    if edge == 0:
        edge_score -= weights['center_edge0_penalty']
    elif edge == 1:
        edge_score -= weights['center_edge1_penalty']
    return center_score + edge_score


def territory_control_score(state, side, head=None):
    """Compare only reached cells, avoiding complete cube scans in production."""
    head = head if head is not None else state.head(side)
    enemy = state.enemy if side == state.side else opposite(side)
    rival = state.head(enemy)
    if head is None or rival is None:
        return 0
    ours = distance_map(state, head, side)
    theirs = distance_map(state, rival, enemy)
    return territory_from_maps(state, ours, theirs, head, rival)


def territory_from_maps(state, ours, theirs, head, rival):
    """Bound the existing territory score after removing occupied non-head cells."""
    if not ours or not theirs:
        return 0
    occupied = state.occupied() - {head, rival}
    difference = sum(
        territory_cell_delta(ours.get(cell), theirs.get(cell))
        for cell in ours.keys() | theirs.keys()
        if cell not in occupied
    )
    weights = get_weights()
    value = difference * weights['territory_control_value']
    cap = weights['territory_control_cap']
    return max(-cap, min(cap, value))


def territory_cell_delta(ours, theirs):
    """Count one reached cell for the earlier actor, with no contribution on ties."""
    if wins_territory(ours, theirs):
        return 1
    return -1 if wins_territory(theirs, ours) else 0


def wins_territory(ours, theirs):
    """A reachable cell belongs to whichever snake can reach it first."""
    return ours is not None and (theirs is None or ours < theirs)


def food_plan_risk_penalty(state, food, our_distance, safety, space_after, exits_after):
    """Combine established non-immediate edge, low-space and single-exit risks."""
    if our_distance <= 1:
        return 0
    weights = get_weights()
    value = edge_food_risk(state, food, our_distance, safety, weights)
    value += low_space_food_risk(our_distance, space_after, weights)
    return value + one_exit_food_risk(safety, space_after, exits_after, weights)


def edge_food_risk(state, food, distance, safety, weights):
    """Price a distant edge objective with its existing safety adjustment."""
    edge = edge_distance(state, food)
    if distance <= 2 or edge > 1:
        return 0
    value = (2 - edge) * weights['non_immediate_edge_food_penalty']
    return value + (weights['low_space_food_penalty'] // 2 if safety != 'SAFE' else 0)


def low_space_food_risk(distance, space, weights):
    """Keep the additional long-route penalty in a small destination region."""
    if space > 45:
        return 0
    value = weights['low_space_food_penalty']
    return value + (weights['long_low_space_food_penalty'] if distance >= 4 else 0)


def one_exit_food_risk(safety, space, exits, weights):
    """Apply the established narrow-exit cost and acceptable-risk surcharge."""
    if exits > 1 or space > 80:
        return 0
    value = weights['one_exit_food_penalty']
    return value + (value // 2 if safety == 'ACCEPTABLE_RISK' else 0)


def objective_kind(state, target, side=None):
    """Identify observed food semantics without promoting remembered objectives."""
    side = side or state.side
    special = (('star', state.stars), ('shot_food', state.shot_food))
    for kind, cells in special:
        if target in cells:
            return kind
    return tail_objective_kind(state, target, side)


def tail_objective_kind(state, target, side):
    """Distinguish beneficiary consumption from originator tail cleanup."""
    for beneficiary, cells in (('A', state.tail_food_a), ('B', state.tail_food_b)):
        if target in cells:
            return 'tail_food' if side == beneficiary else 'tail_cleanup'
    return 'multiplier' if target in state.pickups else 'food'


def projected_sequence_income(state, side, travel_cost=0, multiplier=None, limit=5):
    """Estimate visible ordered income using the established global-turn horizon."""
    if not state.food_values:
        return 0
    return SequenceIncome(state, side, travel_cost, multiplier).project(limit)


class SequenceIncome:
    """Cheap geometric income projection with explicit actor and rival cursors."""

    def __init__(self, state, side, travel_cost, multiplier):
        self.state, self.side = state, side
        enemy = state.enemy if side == state.side else opposite(side)
        self.cursor, self.enemy_cursor = state.head(side), state.head(enemy)
        self.turns = remaining_actor_turns(state, side, travel_cost)
        current = state.multipliers.get(side, 1) if multiplier is None else multiplier
        self.multiplier = max(1, current)
        self.elapsed = self.projected = 0

    def project(self, limit):
        """Stop when the actor cannot cover the route before the movement limit."""
        if self.turns <= 0 or self.cursor is None:
            return 0
        for digit, positions in self.state.numbered_groups(limit=limit):
            target = nearest_geometric_copy(self.cursor, positions)
            distance = manhattan(self.cursor, target)
            arrival = self.elapsed + max(1, distance)
            if arrival > self.turns:
                break
            self.capture(digit, positions, target, distance, arrival)
        return self.projected

    def capture(self, digit, positions, target, distance, arrival):
        """Apply the previous race/time discounts and advance both cursors."""
        enemy_distance = None
        if self.enemy_cursor is not None:
            enemy_target = nearest_geometric_copy(self.enemy_cursor, positions)
            enemy_distance = manhattan(self.enemy_cursor, enemy_target)
            self.enemy_cursor = enemy_target
        margin = tempo_race_margin(self.state, self.side, distance, enemy_distance)
        time_percent = max(35, 100 - max(0, arrival - 4) * 3)
        reward = food_points(digit, self.multiplier)
        self.projected += reward * capture_percentage(margin) * time_percent // 10_000
        self.elapsed, self.cursor = arrival, target


def remaining_actor_turns(state, side, travel_cost=0):
    """Count this players remaining alternating turns after the requested travel cost."""
    return max(0, (state.remaining_moves + (1 if side == state.side else 0)) // 2 - travel_cost)


def nearest_geometric_copy(cursor, positions):
    """Choose the nearest copy using full-axis Manhattan distance and stable ties."""
    return min(positions, key=lambda cell: (manhattan(cursor, cell), cell))


def capture_percentage(margin):
    """Preserve the established four race confidence discounts."""
    if margin >= 2:
        return 100
    if margin >= 0:
        return 82
    return 35 if margin == -1 else 10


def estimated_multiplier_points(state, side, travel_cost=0):
    """Return the marginal future score of adding one multiplier level."""
    current = max(1, state.multipliers.get(side, 1))
    baseline = projected_sequence_income(state, side, travel_cost, current)
    boosted = projected_sequence_income(state, side, travel_cost, current + 1)
    return max(0, boosted - baseline)


def current_food_race(state, side):
    if not food_objectives(state):
        return None
    race = food_set_race(state, side)
    food = race['our_food'] or race['enemy_food'] or next(iter(food_objectives(state)))
    return food, race['our_distance'], race['enemy_distance'], race['margin']


def sequence_gate_hold_move(state, side, legal, plan, deep_moves):
    """Hold an adjacent digit only under the established short-endgame race conditions."""
    weights = get_weights()
    if not sequence_hold_allowed(state, side, plan, weights):
        return None
    if not next_sequence_loses_lead(state, side, plan, weights):
        return None
    candidates = holding_candidates(state, side, legal, plan, deep_moves)
    return max(candidates, key=lambda item: item['score'])['direction'] if candidates else None


def adjacent_numbered_plan(state, plan):
    return (
        plan is not None
        and plan.objective_kind == 'food'
        and plan.our_distance == 1
        and bool(state.food_values)
    )


def sequence_hold_allowed(state, side, plan, weights):
    if not adjacent_numbered_plan(state, plan):
        return False
    enemy = state.enemy if side == state.side else opposite(side)
    lead = state.scores.get(side, 0) - state.scores.get(enemy, 0)
    return (
        4 < state.remaining_moves <= weights['sequence_hold_max_remaining']
        and lead > 0
        and distant_current_enemy(state, side, weights)
    )


def distant_current_enemy(state, side, weights):
    distance = food_set_race(state, side)['enemy_distance']
    return distance is not None and distance >= weights['sequence_hold_min_enemy_distance']


def next_sequence_loses_lead(state, side, plan, weights):
    sequence = state.numbered_groups(limit=2)
    if len(sequence) < 2:
        return False
    next_digit, next_positions = sequence[1]
    after_eat = apply_move(state, plan.first_move, side)
    choice = best_numbered_copy(after_eat, side, next_positions)
    if choice is None:
        return False
    if not opponent_wins_next_copy(choice):
        return False
    enemy = state.enemy if side == state.side else opposite(side)
    lead = state.scores.get(side, 0) - state.scores.get(enemy, 0)
    projected_lead = lead + plan.objective_reward
    reward = food_points(next_digit, state.multipliers.get(enemy, 1))
    return projected_lead < reward + weights['sequence_hold_score_buffer']


def opponent_wins_next_copy(choice):
    _food, _path, ours, enemy, margin = choice
    return ours is not None and enemy is not None and margin < 0


def holding_candidates(state, side, legal, plan, deep_moves):
    analyses = {item['direction']: item for item in deep_moves}
    candidates = []
    for direction in legal:
        analysis = holding_move_analysis(state, side, direction, plan, analyses)
        if analysis is not None:
            candidates.append(analysis)
    return candidates


def holding_move_analysis(state, side, direction, plan, analyses):
    if direction == plan.first_move:
        return None
    after = apply_move(state, direction, side)
    if manhattan(after.head(side), plan.food) > 2:
        return None
    analysis = analyses.get(direction)
    return analysis if valid_holding_analysis(analysis) else None


def valid_holding_analysis(analysis):
    return (
        analysis is not None
        and analysis.get('safety') != 'SUICIDAL'
        and analysis.get('our_replies', 0) >= 2
    )


def objective_reward(state, side, target):
    """Price the destination according to beneficiary and multiplier semantics."""
    kind = objective_kind(state, target, side)
    multiplier = state.multipliers.get(side, 1)
    rewards = {
        'tail_cleanup': 0,
        'shot_food': shot_food_reward(multiplier),
        'star': FOOD_SCORE * multiplier,
        'tail_food': tail_food_reward(multiplier),
        'multiplier': MULTIPLIER_PICKUP_SCORE,
    }
    if kind in rewards:
        return rewards[kind]
    return state.food_reward(side, target)


def star_strategic_points(state, side, our_distance, enemy_distance=None):
    """Bound a visible star's value by reachable protection and real cut food.

    This estimates opportunity, never changes a rule or grants future score.
    The existing objective point weight prices it; there are no star weights.
    """
    enemy = 'B' if side == 'A' else 'A'
    if star_race_is_lost(state, side, our_distance, enemy_distance):
        return 0
    arrival, extra = added_star_protection(state, side, our_distance)
    if not measurable_star_benefit(state, side, enemy, extra):
        return 0
    avoided_loss = avoided_star_loss(state, side)
    distance = min(manhattan(state.head(side), cell) for cell in state.body(enemy))
    cut_value = max(0, len(state.body(enemy)) - 1) * FOOD_SCORE * state.multipliers.get(side, 1)
    benefit = (avoided_loss + cut_value) * extra // (STAR_TURNS * max(1, distance))
    return benefit + exclusive_star_value(state, arrival, extra, cut_value, enemy_distance)


def star_race_is_lost(state, side, our_distance, enemy_distance):
    return not state.star_frenzy and not wins_food_race(state, side, our_distance, enemy_distance)


def added_star_protection(state, side, our_distance):
    """Price only invincibility turns still usable after arrival."""
    arrival = 2 * our_distance - (1 if side == state.side else 0)
    turns = max(0, (state.remaining_moves - arrival) // 2)
    current = max(0, state.invincibility(side) - our_distance)
    extra = max(0, min(STAR_TURNS, turns) - current)
    return arrival, extra


def measurable_star_benefit(state, side, enemy, extra):
    return bool(extra) and bool(state.body(enemy)) and state.head(side) is not None


def avoided_star_loss(state, side):
    history = state.crashed_a if side == 'A' else state.crashed_b
    return max(0, state.scores.get(side, 0)) if history is False else CRASH_PENALTY


def exclusive_star_value(state, arrival, extra, cut_value, enemy_distance):
    """The ordinary star's exclusive race value ends before the frenzy begins."""
    if state.star_frenzy or enemy_distance is None:
        return 0
    exclusive_turns = min(extra, max(0, (state.remaining_moves - arrival - 100) // 2))
    return cut_value * exclusive_turns // (STAR_TURNS * max(1, enemy_distance))


def is_structural_food_risk(plan):
    """Detect persistent space/exit risk beyond an immediately adjacent objective."""
    if plan is None or plan.our_distance <= 1:
        return False
    return small_food_region(plan) or narrow_food_exit(plan)


def small_food_region(plan):
    """Preserve the acceptable-risk and long-route small-region thresholds."""
    return (
        plan.safety == 'ACCEPTABLE_RISK'
        and plan.space_after <= 45
        or plan.our_distance >= 4
        and plan.space_after <= 40
    )


def narrow_food_exit(plan):
    """Flag a non-immediate objective with a narrow and bounded escape region."""
    return plan.our_distance > 2 and plan.exits_after <= 1 and plan.space_after <= 80


def position_control_penalty(state, side, direction):
    """Price edge exposure only after a valid move with a sufficiently long body."""
    if direction not in legal_moves(state, side):
        return 1_000_000
    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return 1_000_000
    return body_position_penalty(after, side, head)


def body_position_penalty(after, side, head):
    """Apply the existing nearest-face and corner penalties in either geometry."""
    if len(after.body(side)) <= 3:
        return 0
    weights = get_weights()
    edges = {
        0: weights['position_edge0_penalty'],
        1: weights['position_edge1_penalty'],
        2: weights['position_edge2_penalty'],
    }
    value = edges.get(edge_distance(after, head), 0)
    if borders_touching(after, head) >= 2:
        value += weights['position_corner_penalty']
    return value


def borders_touching(state, head):
    """Count board/cube faces sharing the current cell."""
    return sum(axis in (0, size - 1) for axis, size in zip(head, state_dimensions(state)))


def build_food_plan(state, side, food, current_target, hunger, food_race=None):
    """Build one reachable objective plan using unchanged safety and economic terms."""
    path = food_plan_route(state, side, food)
    if path is None:
        return None
    after, loss, triggers = simulate_path_economy(state, side, path)
    if after is None:
        return None
    plan = food_plan_facts(state, side, food, path, after, loss, triggers, hunger, food_race)
    value = food_plan_value(state, side, plan, after, current_target, hunger)
    return replace(plan, value=value)


def food_plan_route(state, side, food):
    """Require a real first step and a reachable arrival for expiring investments."""
    head = state.head(side)
    if head is None:
        return None
    path = temporal_shortest_path(state, head, food, side)
    if path is None or len(path) < 2:
        return None
    return path if objective_arrives_in_time(state, side, food, len(path) - 1) else None


def objective_arrival_cost(state, side, distance):
    """Count alternating global actions until this actors arrival."""
    return 2 * distance - (1 if side == state.side else 0)


def objective_arrives_in_time(state, side, food, distance):
    """Keep the existing movement-budget gate for X and star objectives."""
    if objective_kind(state, food, side) in ('multiplier', 'star'):
        return objective_arrival_cost(state, side, distance) <= state.remaining_moves
    return True


def objective_enemy_distance(state, side, food, food_race):
    """Use the fastest-copy rival race for numbered targets and a route otherwise."""
    if food in state.food and state.food_values:
        return (food_race or food_set_race(state, side))['enemy_distance']
    enemy_head = state.head(state.enemy)
    if enemy_head is not None:
        return temporal_shortest_distance(state, enemy_head, food, state.enemy)
    return None


def economic_arrival_state(state, after, side, kind, distance):
    """Preserve the corrected global horizon only for multiplier investment income."""
    if kind != 'multiplier':
        return after
    enemy = state.enemy if side == state.side else opposite(side)
    return replace(
        after,
        remaining_moves=state.remaining_moves - objective_arrival_cost(state, side, distance),
        side=enemy,
        enemy=side,
    )


def food_plan_facts(state, side, food, path, after, loss, triggers, hunger, food_race):
    """Gather route safety, race and economic facts once before applying value weights."""
    distance = len(path) - 1
    enemy_distance = objective_enemy_distance(state, side, food, food_race)
    margin = tempo_race_margin(state, side, distance, enemy_distance)
    urgency = score_gap_urgency(state, side)
    safety, space, exits, tail = classify_safety(after, side)
    distances = distance_map(after, after.head(side), side)
    chain = food_chain_score(after, side, food, hunger + urgency, distances)
    threat = rival_food_threat_score(state, side, food, distance, enemy_distance)
    denial = food_denial_score(state, side, food, distance, enemy_distance, threat)
    kind = objective_kind(state, food, side)
    economic_after = economic_arrival_state(state, after, side, kind, distance)
    enemy = state.enemy if side == state.side else opposite(side)
    income = projected_sequence_income(economic_after, side)
    rival_income = projected_sequence_income(economic_after, enemy)
    marginal = multiplier_plan_marginal(state, side, kind, distance, income)
    return FoodPlan(
        food=food,
        path=path,
        our_distance=distance,
        enemy_distance=enemy_distance,
        race_margin=margin,
        space_after=space,
        exits_after=exits,
        tail_reachable_after=tail,
        next_food_distance=nearest_next_food_distance(after, side, food, distances),
        safety=safety,
        value=0,
        chain_score=chain,
        denial_score=denial,
        rival_threat=threat,
        tempo_urgency=urgency,
        temporal_tail_path=path_crosses_occupancy(state, path),
        endgame_acceptable=endgame_accepts_food_risk(state, distance, margin, safety),
        objective_kind=kind,
        objective_reward=objective_reward(state, side, food),
        projected_income=income,
        rival_projected_income=rival_income,
        economic_swing=income - rival_income,
        multiplier_marginal=marginal,
        bomb_loss=loss,
        bomb_trigger_count=triggers,
    )


def path_crosses_occupancy(state, path):
    """Flag routes that depend on a currently occupied cell releasing in time."""
    occupied = state.occupied()
    return any(cell in occupied for cell in path[1:])


def multiplier_plan_marginal(state, side, kind, distance, income):
    """Estimate X improvement against the existing global-tempo baseline."""
    if kind != 'multiplier':
        return 0
    baseline = projected_sequence_income(state, side, distance)
    marginal = max(0, income - baseline)
    return marginal if marginal else estimated_multiplier_points(state, side, distance)


def capped_economic_value(swing, weights):
    """Bound the unchanged future-income projection contribution."""
    value = swing * weights['economic_projection_value']
    cap = weights['economic_projection_cap']
    return max(-cap, min(cap, value))


def food_plan_value(state, side, plan, after, target, hunger):
    """Combine the established value components in the original scoring order."""
    weights = get_weights()
    value = weights['food_base_value'] - plan.bomb_loss * weights['objective_point_value']
    value += max(0, plan.objective_reward - FOOD_SCORE) * weights['objective_point_value']
    value += star_plan_value(state, side, plan, weights)
    value += capped_economic_value(plan.economic_swing, weights)
    value += objective_investment_value(state, side, plan, hunger, weights)
    value += route_plan_value(plan, after, side, hunger, weights)
    value += close_hungry_food_value(plan, hunger, weights)
    if plan.food == target:
        value += weights['current_target_bonus']
    value += plan_safety_value(plan, weights)
    value += lost_race_plan_value(state, side, plan, weights)
    value += endgame_plan_value(state, plan, weights)
    return value


def star_plan_value(state, side, plan, weights):
    """Apply the existing tactical star benefit only to a star objective."""
    if plan.objective_kind == 'star':
        return (
            star_strategic_points(state, side, plan.our_distance, plan.enemy_distance)
            * weights['objective_point_value']
        )
    return 0


def objective_investment_value(state, side, plan, hunger, weights):
    """Select X investment economics or ordinary numbered-food hunger."""
    if plan.objective_kind == 'multiplier':
        return multiplier_plan_value(state, side, plan, hunger, weights)
    if plan.objective_kind != 'star' and state.food_values:
        return min(hunger, 30) * weights['numbered_food_hunger_value']
    return 0


def multiplier_plan_value(state, side, plan, hunger, weights):
    """Value future X income, permanent stacks and rival denial separately."""
    enemy = state.enemy if side == state.side else opposite(side)
    gap = state.multipliers.get(enemy, 1) - state.multipliers.get(side, 1)
    value = plan.multiplier_marginal * weights['multiplier_future_point_value']
    value += weights['multiplier_base_value'] + gap * weights['multiplier_gap_value']
    value += multiplier_race_denial(state, enemy, plan, weights)
    race = current_food_race(state, side)
    if race is not None and race[3] >= 0:
        value -= hunger * weights['multiplier_hunger_penalty']
    if plan.multiplier_marginal < FOOD_SCORE * 5:
        value -= max(0, state.multipliers.get(side, 1) - 2) * weights['multiplier_stack_penalty']
    return value


def multiplier_race_denial(state, enemy, plan, weights):
    """Count a rival X opportunity only when our arrival wins its race."""
    if plan.enemy_distance is not None and plan.race_margin >= 0:
        marginal = estimated_multiplier_points(state, enemy, plan.enemy_distance)
        return marginal * weights['multiplier_denial_point_value']
    return 0


def route_plan_value(plan, after, side, hunger, weights):
    """Combine travel, hunger, escape space and follow-up food contributions."""
    value = -plan.our_distance * weights['food_distance_cost']
    value += plan.race_margin * weights['race_margin_value']
    value += min(hunger + plan.tempo_urgency, 50) * weights['hunger_value']
    value += min(plan.space_after, 80) * weights['space_after_value']
    value += plan.exits_after * weights['exits_after_value']
    value += weights['tail_reachable_bonus'] if plan.tail_reachable_after else 0
    value += center_control_score(after, after.head(side))
    if plan.next_food_distance is not None:
        value += max(0, 12 - plan.next_food_distance) * weights['next_food_value']
    return value + plan.chain_score + plan.denial_score


def close_hungry_food_value(plan, hunger, weights):
    """Reward close safe food under the established effective-hunger threshold."""
    effective = hunger + plan.tempo_urgency
    if effective >= 12 and plan.safety == 'SAFE' and plan.race_margin >= 0:
        return (
            (effective - 11) * max(0, 10 - plan.our_distance) * weights['hunger_close_food_value']
        )
    return 0


def plan_safety_value(plan, weights):
    """Apply the original safety categories and endgame exceptions."""
    values = {'SAFE': weights['safe_bonus'], 'ACCEPTABLE_RISK': weights['acceptable_risk_bonus']}
    if plan.safety in values:
        return values[plan.safety]
    if plan.safety == 'DANGEROUS':
        return (
            weights['endgame_food_bonus']
            if plan.endgame_acceptable
            else -weights['dangerous_penalty']
        )
    return weights['endgame_food_bonus'] // 2 if plan.endgame_acceptable else -100_000


def lost_race_plan_value(state, side, plan, weights):
    """Penalize late arrival and ignored rival chain opportunities."""
    value = late_race_penalty(plan, weights)
    if plan.denial_score == 0 and plan.rival_threat and plan.race_margin < 0:
        value -= min(plan.rival_threat, weights['rival_chain_ignore_penalty'])
    return value


def late_race_penalty(plan, weights):
    """Apply the established large-gap arrival penalty."""
    if plan.enemy_distance is not None and plan.enemy_distance + 3 < plan.our_distance:
        return -weights['lost_race_penalty']
    return 0


def endgame_plan_value(state, plan, weights):
    """Trade ordinary risk against a final winning-food opportunity."""
    if plan.endgame_acceptable:
        return max(
            0,
            weights['endgame_food_bonus']
            - max(0, remaining_after_our_food(state, plan.our_distance)) * 1000,
        )
    return -food_plan_risk_penalty(
        state, plan.food, plan.our_distance, plan.safety, plan.space_after, plan.exits_after
    )


def build_sequence_setup_plans(state, side, current_target, hunger):
    """Stage near upcoming digits only when the current target's race is already lost."""
    return SequenceSetupPlanner(state, side, current_target, hunger).build()


class SequenceSetupPlanner:
    """Evaluate neighboring staging cells with the existing sequence economics."""

    def __init__(self, state, side, current_target, hunger):
        self.state, self.side, self.current_target, self.hunger = (
            state,
            side,
            current_target,
            hunger,
        )
        self.race = current_food_race(state, side)
        self.sequence = state.numbered_groups(limit=3)
        self.head = state.head(side)
        self.enemy = state.enemy if side == state.side else opposite(side)
        self.weights, self.occupied = get_weights(), state.occupied()

    def build(self):
        """Return viable staging plans for the next two visible sequence groups."""
        if not lost_sequence_setup_race(self.race, self.sequence):
            return []
        self.current_food, _, self.enemy_distance, _ = self.race
        if self.enemy_distance is None:
            return []
        plans = []
        for rank, (digit, positions) in enumerate(self.sequence[1:3], start=1):
            plan = self.future_plan(rank, digit, positions)
            if plan is not None:
                plans.append(plan)
        return plans

    def future_plan(self, rank, digit, positions):
        choice = best_numbered_copy(self.state, self.side, positions)
        if choice is None:
            return None
        food = choice[0]
        candidates = self.staging_candidates(food)
        if not candidates:
            return None
        _, path, staging = min(
            candidates, key=lambda item: (item[0], manhattan(item[2], self.head))
        )
        return self.evaluate_staging(rank, digit, food, path, staging)

    def staging_candidates(self, future_food):
        candidates = []
        for cell in neighbors(future_food, self.state.rows, self.state.cols, self.state.levels):
            if not sequence_staging_available(self.state, self.occupied, cell):
                continue
            path = temporal_shortest_path(self.state, self.head, cell, self.side)
            if valid_sequence_staging_path(path, self.current_food):
                candidates.append((len(path) - 1, path, cell))
        return candidates

    def evaluate_staging(self, rank, digit, food, path, staging):
        if len(path) - 1 > self.enemy_distance + self.weights['sequence_setup_late_slack']:
            return None
        after, loss, triggers = simulate_path_economy(self.state, self.side, path)
        if after is None:
            return None
        assessment = classify_safety(after, self.side)
        if assessment[0] in ('SUICIDAL', 'DANGEROUS'):
            return None
        return self.assemble_plan(
            rank, digit, food, path, staging, after, loss, triggers, assessment
        )

    def assemble_plan(
        self, rank, digit, future_food, path, staging, after, loss, triggers, assessment
    ):
        state, side, weights = self.state, self.side, self.weights
        safety, space, exits, tail_ok = assessment
        distance = len(path) - 1
        enemy_followup = max(1, manhattan(self.current_food, future_food))
        readiness = self.enemy_distance - distance
        margin = enemy_followup - 1 + readiness
        reward = food_points(digit, state.multipliers.get(side, 1))
        projected = projected_sequence_income(after, side)
        rival_projected = projected_sequence_income(after, self.enemy)
        swing = projected - rival_projected
        value = weights['sequence_setup_base_value']
        value -= loss * weights['objective_point_value']
        value += reward * weights['sequence_setup_reward_point_value']
        value += readiness * weights['sequence_setup_readiness_value']
        value += margin * weights['race_margin_value']
        value -= distance * weights['food_distance_cost']
        value -= (rank - 1) * weights['sequence_setup_rank_penalty']
        value += min(space, 80) * weights['space_after_value']
        value += exits * weights['exits_after_value']
        value += weights['tail_reachable_bonus'] if tail_ok else 0
        value += center_control_score(after, after.head(side))
        economic_value = swing * weights['economic_projection_value']
        economic_cap = weights['economic_projection_cap']
        value += max(-economic_cap, min(economic_cap, economic_value))
        if staging == self.current_target:
            value += weights['current_target_bonus']
        value += min(self.hunger, 30) * weights['sequence_setup_hunger_value']
        return FoodPlan(
            food=staging,
            path=path,
            our_distance=distance,
            enemy_distance=enemy_followup,
            race_margin=margin,
            space_after=space,
            exits_after=exits,
            tail_reachable_after=tail_ok,
            next_food_distance=1,
            safety=safety,
            value=value,
            chain_score=reward * weights['sequence_reward_point_value'],
            tempo_urgency=score_gap_urgency(state, side),
            temporal_tail_path=any(cell in self.occupied for cell in path[1:]),
            objective_kind='sequence_setup',
            objective_reward=reward,
            sequence_rank=rank,
            projected_income=projected,
            rival_projected_income=rival_projected,
            economic_swing=swing,
            bomb_loss=loss,
            bomb_trigger_count=triggers,
        )


def lost_sequence_setup_race(race, sequence):
    return race is not None and race[3] < 0 and len(sequence) >= 2


def sequence_staging_available(state, occupied, cell):
    return (
        cell not in occupied
        and cell not in state.food_values
        and cell not in state.pickups
        and cell not in state.stars
    )


def valid_sequence_staging_path(path, current_food):
    return path is not None and len(path) >= 2 and current_food not in path[1:]


def build_food_plans(state, side, current_target=None, hunger=0):
    """Build available objective and sequence-staging plans inside a shared planning cache."""
    with _planning_scope():
        return _build_food_plans(state, side, current_target, hunger)


def _build_food_plans(state, side, current_target=None, hunger=0):
    current_food_race = food_set_race(state, side) if state.food_values else None
    own_tail_food = state.tail_food_a if side == 'A' else state.tail_food_b
    plans = [
        plan
        for food in state.objective_cells() | own_tail_food
        if (
            plan := build_food_plan(
                state,
                side,
                food,
                current_target,
                hunger,
                food_race=current_food_race,
            )
        )
        is not None
    ]
    plans.extend(build_sequence_setup_plans(state, side, current_target, hunger))
    return plans


def choose_food_plan(state, side, current_target=None, hunger=0, plans=None):
    """Select an immediate, harvest or strategic plan after viability filtering."""
    plans = available_food_plans(state, side, current_target, hunger, plans)
    viable = viable_food_plans(plans, hunger)
    if not viable:
        return None
    immediate = best_immediate_plan(viable)
    if immediate is not None:
        return immediate
    harvest = hungry_harvest_plan(viable, hunger)
    if harvest is not None:
        return harvest
    return strategic_food_plan(state, side, viable, current_target, hunger)


def available_food_plans(state, side, target, hunger, plans):
    if plans is None:
        plans = build_food_plans(state, side, target, hunger)
    own_turns_left = (state.remaining_moves + (side == state.side)) // 2
    return [plan for plan in plans if reachable_plan(plan, state.pickups, own_turns_left)]


def reachable_plan(plan, pickups, own_turns_left):
    if plan.objective_kind != 'multiplier' and plan.food not in pickups:
        return True
    return plan.our_distance <= own_turns_left


def viable_food_plans(plans, hunger):
    viable = [plan for plan in plans if safe_food_plan(plan)]
    if viable:
        return viable
    if hunger < 35:
        return []
    return dangerous_food_plans(plans)


def dangerous_food_plans(plans):
    return [plan for plan in plans if plan.safety == 'DANGEROUS' and plan.race_margin >= 0]


def safe_food_plan(plan):
    return plan.safety in ('SAFE', 'ACCEPTABLE_RISK') or plan.endgame_acceptable


def best_immediate_plan(plans):
    return max(
        (plan for plan in plans if plan.our_distance == 1),
        key=food_tiebreak_key,
        default=None,
    )


HARVEST_KINDS = {'food', 'shot_food', 'tail_food'}


def harvestable(plan):
    return plan.objective_kind in HARVEST_KINDS and plan.race_margin >= 0


def eligible_harvests(plans):
    return (
        plan for plan in plans if harvestable(plan) and plan.safety in ('SAFE', 'ACCEPTABLE_RISK')
    )


def hungry_harvest_plan(plans, hunger):
    if hunger < 10:
        return None
    harvest = max(eligible_harvests(plans), key=hunger_food_key, default=None)
    if harvest is None:
        return None
    investment = nearby_investment(plans, harvest.our_distance + 3)
    if not beats_harvest(investment, harvest):
        return harvest
    return None


def nearby_investment(plans, max_distance):
    return max(
        (plan for plan in plans if viable_investment(plan, max_distance)),
        key=food_tiebreak_key,
        default=None,
    )


def viable_investment(plan, max_distance):
    return (
        plan.objective_kind == 'multiplier'
        and plan.safety == 'SAFE'
        and plan.race_margin >= 0
        and plan.our_distance <= max_distance
    )


def beats_harvest(investment, harvest):
    if investment is None:
        return False
    if investment.multiplier_marginal < max(500, harvest.objective_reward * 2):
        return False
    margin = get_weights()['food_plan_switch_margin']
    return investment.value >= harvest.value - margin


def strategic_food_plan(state, side, viable, target, hunger):
    strategic = [plan for plan in viable if not is_structural_food_risk(plan)]
    viable = strategic or viable
    weights = get_weights()
    denial = urgent_denial(viable, weights)
    if denial is not None:
        return denial
    return race_food_plan(state, side, viable, target, hunger, weights)


def urgent_denial(plans, weights):
    threshold = weights['denial_food_bonus'] + weights['denial_chain_bonus']
    return max(
        (plan for plan in plans if can_deny_food(plan, threshold)),
        key=denial_key,
        default=None,
    )


def can_deny_food(plan, threshold):
    return (
        plan.enemy_distance is not None
        and (plan.enemy_distance <= 4 or plan.denial_score >= threshold)
        and plan.race_margin >= 0
    )


def winnable_food_races(plans):
    return [plan for plan in plans if plan.enemy_distance is None or plan.race_margin >= 0]


def race_food_plan(state, side, viable, target, hunger, weights):
    plans = winnable_food_races(viable) or viable
    nearest = min(plan.our_distance for plan in plans)
    tempo_hunger = hunger + score_gap_urgency(state, side)
    urgent = urgent_harvest(plans, nearest, tempo_hunger)
    if urgent is not None:
        return urgent
    return preferred_food_plan(plans, nearest, target, tempo_hunger, weights)


def urgent_harvest(plans, nearest, tempo_hunger):
    if tempo_hunger < 14:
        return None
    return max(
        (plan for plan in plans if nearby_safe_harvest(plan, nearest + 2)),
        key=hunger_food_key,
        default=None,
    )


def nearby_safe_harvest(plan, max_distance):
    return harvestable(plan) and plan.safety == 'SAFE' and plan.our_distance <= max_distance


def preferred_food_plan(plans, nearest, target, hunger, weights):
    best = max(plans, key=food_tiebreak_key)
    current = next((plan for plan in plans if plan.food == target), None)
    if keep_current_plan(current, best, hunger, weights):
        return current
    return prefer_close_plan(plans, nearest, best, weights)


def keep_current_plan(current, best, hunger, weights):
    if current is None or hunger >= 12:
        return False
    if current.race_margin < 0 or current.value <= 0:
        return False
    return close_to_best_plan(current, best, weights)


def close_to_best_plan(current, best, weights):
    return (
        current.value >= best.value - weights['food_plan_switch_margin']
        and current.our_distance <= best.our_distance + 1
    )


def prefer_close_plan(plans, nearest, best, weights):
    nearby = max(
        (plan for plan in plans if plan.our_distance <= nearest + 1),
        key=food_tiebreak_key,
    )
    return best if worthwhile_detour(best, nearby, weights) else nearby


def worthwhile_detour(best, nearby, weights):
    if best.value >= nearby.value + weights['food_plan_big_switch_margin']:
        return True
    return nearby.value < 0 and best.value > nearby.value


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
    """Keep immediate safety decisions before considering longer-term food pressure."""
    immediate = immediate_food_commitment(plan)
    if immediate is not None:
        return immediate
    if plan.race_margin < 0 or is_structural_food_risk(plan):
        return False
    if plan.safety == 'DANGEROUS':
        return hunger >= 35
    return food_commitment_motives(plan, hunger, repeat_count)


def immediate_food_commitment(plan):
    """Endgame and adjacent-food decisions retain precedence over the race margin."""
    if plan.endgame_acceptable:
        return True
    if plan.safety == 'SUICIDAL':
        return False
    if plan.our_distance == 1 and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
        return True
    return None


def food_commitment_motives(plan, hunger, repeat_count):
    """Combine denial, repeated-state and hunger motives after structural safety checks."""
    tempo_hunger = hunger + plan.tempo_urgency
    return any(
        (
            commit_for_denial(plan),
            repeat_count >= 2 and plan.safety != 'SUICIDAL',
            commit_for_hunger_or_value(plan, tempo_hunger),
        )
    )


def commit_for_denial(plan):
    """An uncontested safe denial may justify holding the current route."""
    return plan.denial_score > 0 and plan.safety == 'SAFE' and plan.race_margin >= 0


def commit_for_hunger_or_value(plan, tempo_hunger):
    """Require acceptable safety for either urgent hunger or a positive plan value."""
    return plan.safety in ('SAFE', 'ACCEPTABLE_RISK') and (tempo_hunger >= 20 or plan.value > 0)


def should_use_plan_for_positioning(plan):
    return (
        plan is not None
        and plan.race_margin >= 0
        and (plan.endgame_acceptable or not is_structural_food_risk(plan))
    )


def fallback_hunger_food_score(state, side, after, head, hunger):
    """Reward reachable food progress only after hunger reaches the existing threshold."""
    if not hunger_food_available(after, head, hunger):
        return 0
    enemy = state.enemy if side == state.side else opposite(side)
    distances = distance_map(after, head, side)
    scores = (
        hunger_food_value(
            after, side, food, distances.get(food), state.head(side), state.head(enemy), hunger
        )
        for food in food_objectives(after)
    )
    return max(0, max(scores, default=0))


def hunger_food_available(after, head, hunger):
    return hunger >= 10 and head is not None and bool(food_objectives(after))


def optional_manhattan(head, cell):
    return None if head is None else manhattan(head, cell)


def hunger_food_value(after, side, food, distance, before_head, enemy_head, hunger):
    if distance is None:
        return 0
    before_distance = optional_manhattan(before_head, food)
    enemy_distance = optional_manhattan(enemy_head, food)
    margin = tempo_race_margin(after, side, distance, enemy_distance)
    if margin < -1:
        return 0
    progress = distance_progress(before_distance, distance)
    weights = get_weights()
    pressure = hunger - 9
    return (
        progress * pressure * weights['fallback_hunger_food_progress']
        + max(0, 8 - distance) * pressure * weights['fallback_hunger_food_distance']
        + max(0, min(5, margin)) * weights['fallback_hunger_food_race']
    )


def distance_progress(before, after):
    return 0 if before is None else max(0, before - after)


def fallback_denial_score(state, side, after, head):
    """Reward approaching a reachable target while preserving its rival-race advantage."""
    if head is None or not food_objectives(state):
        return 0
    enemy = state.enemy if side == state.side else opposite(side)
    enemy_head, current_head = state.head(enemy), state.head(side)
    if not denial_heads_present(current_head, enemy_head):
        return 0
    return denial_food_score(state, side, after, head, enemy, enemy_head, current_head)


def denial_heads_present(own_head, enemy_head):
    return own_head is not None and enemy_head is not None


def denial_food_score(state, side, after, head, enemy, enemy_head, current_head):
    if not state.food_values:
        return legacy_denial_score(state, side, after, head, enemy, enemy_head, current_head)
    return numbered_denial_score(state, side, after)


def legacy_denial_score(state, side, after, head, enemy, enemy_head, current_head):
    distances = distance_map(after, head, side)
    return sum(
        legacy_food_denial_value(
            state, side, after, food, distances, enemy, enemy_head, current_head
        )
        for food in food_objectives(state)
    )


def legacy_food_denial_value(state, side, after, food, distances, enemy, enemy_head, current_head):
    enemy_distance = temporal_shortest_distance(state, enemy_head, food, enemy)
    if enemy_distance is None or enemy_distance > 5:
        return 0
    before = temporal_shortest_distance(state, current_head, food, side)
    return denial_race_value(after, side, before, distances.get(food), enemy_distance)


def numbered_denial_score(state, side, after):
    race = food_set_race(state, side)
    enemy, before = race['enemy_distance'], race['our_distance']
    if enemy is None or enemy > 5 or before is None:
        return 0
    distances = food_distances(after, side, food_objectives(after))
    distance = min(distances.values(), default=None)
    return denial_race_value(after, side, before, distance, enemy)


def denial_race_value(after, side, before_distance, after_distance, enemy_distance):
    if before_distance is None or after_distance is None:
        return 0
    if (
        wins_food_race(after, side, after_distance, enemy_distance)
        and after_distance < before_distance
    ):
        weights = get_weights()
        return (
            weights['fallback_denial_reach_bonus']
            + (before_distance - after_distance) * weights['fallback_denial_progress']
        )
    return 0


def rival_response_stats(state, side, direction, depth=1):
    """Count immediate and delayed forced losses under each available rival reply."""
    if direction not in legal_moves(state, side):
        return empty_rival_response(True)
    if state.partial:
        return {**empty_rival_response(False), 'uncertain': True}
    return complete_rival_response_stats(state, side, direction, depth)


def complete_rival_response_stats(state, side, direction, depth):
    """Evaluate adversarial continuations only when the state is fully known."""
    enemy = state.enemy if side == state.side else opposite(side)
    after = apply_move(state, direction, side)
    enemy_moves = available_moves(after, enemy)
    if not enemy_moves:
        lost = after.is_terminal() and after.terminal_winner() == enemy
        return empty_rival_response(lost)
    zero, one, delayed = count_rival_responses(after, side, enemy, enemy_moves, depth)
    return {
        'enemy_moves': len(enemy_moves),
        'zero_escape_replies': zero,
        'one_escape_replies': one,
        'delayed_forced_zero_replies': delayed,
        'forced_zero': zero == len(enemy_moves),
        'forced_delayed_zero': delayed == len(enemy_moves),
    }


def empty_rival_response(lost):
    return {
        'enemy_moves': 0,
        'zero_escape_replies': 0,
        'one_escape_replies': 0,
        'delayed_forced_zero_replies': 0,
        'forced_zero': lost,
        'forced_delayed_zero': lost,
    }


def count_rival_responses(after, side, enemy, enemy_moves, depth):
    zero, one, delayed = 0, 0, 0
    for move in enemy_moves:
        counts = classify_rival_reply(after, side, enemy, move, depth)
        zero += counts[0]
        one += counts[1]
        delayed += counts[2]
    return zero, one, delayed


def classify_rival_reply(after, side, enemy, move, depth):
    after_enemy = apply_move(after, move, enemy, perspective=side)
    if after_enemy.is_terminal():
        return int(after_enemy.terminal_winner() == enemy), 0, 0
    replies = available_moves(after_enemy, side)
    return (
        int(not replies),
        int(len(replies) == 1),
        int(all_replies_lose_later(after_enemy, side, replies, depth)),
    )


def all_replies_lose_later(state, side, replies, depth):
    return (
        depth > 0
        and bool(replies)
        and all(
            rival_response_stats(state, side, move, depth - 1)['forced_zero'] for move in replies
        )
    )


def rival_response_penalty(state, side, direction):
    if state.partial:
        return 0
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
    """Measure the worst space loss among the rival's available responses."""
    if direction not in legal_moves(state, side):
        return 1_000_000
    enemy = state.enemy if side == state.side else opposite(side)
    after = apply_move(state, direction, side)
    head = after.head(side)
    if head is None:
        return 1_000_000
    weights = get_weights()
    length = len(after.body(side))
    base = immediate_space_cost(after, side, head, length, weights)
    threats = known_rival_space_costs(after, side, enemy, length, weights)
    return base + max(0, max(threats, default=0))


def known_rival_space_costs(state, side, enemy, length, weights):
    """A partial observation does not determine every possible opponent continuation."""
    if state.partial:
        return ()
    return (
        rival_reply_space_cost(state, side, enemy, move, length, weights)
        for move in available_moves(state, enemy)
    )


def small_region_cost(region, length, weights):
    if region < length * 2 + 6:
        return (length * 2 + 6 - region) * weights['small_region_penalty']
    return 0


def immediate_space_cost(state, side, head, length, weights):
    region = len(flood_region(state, head, strategic_blocked(state)))
    cost = small_region_cost(region, length, weights)
    if count_exits(state, side) <= 1:
        cost += weights['one_exit_penalty']
    return cost


def rival_reply_space_cost(after, side, enemy, move, length, weights):
    state = apply_move(after, move, enemy, perspective=side)
    head, rival_head = state.head(side), state.head(enemy)
    if head is None:
        return 1_000_000
    replies = available_moves(state, side)
    blocked = strategic_blocked(state)
    own_region = len(flood_region(state, head, blocked))
    enemy_region = 0 if rival_head is None else len(flood_region(state, rival_head, blocked))
    threat = small_region_cost(own_region, length, weights)
    if not replies:
        threat += weights['enemy_no_reply_threat']
    if count_exits(state, side) <= 1:
        threat += weights['one_exit_penalty']
    return threat + territory_gap_cost(enemy_region - own_region, weights)


def territory_gap_cost(gap, weights):
    if gap > weights['territory_gap_limit']:
        return min(
            weights['territory_gap_penalty_cap'],
            (gap - weights['territory_gap_limit']) * weights['territory_gap_penalty'],
        )
    return 0


def hot_lost_food_penalty(state, side, direction):
    """Retain the lost-race penalty only when this move approaches a contested food."""
    head = state.head(side)
    enemy = state.enemy if side == state.side else opposite(side)
    enemy_head = state.head(enemy)
    if not valid_hot_food_actor(state, side, direction, head, enemy_head):
        return 0
    after = apply_move(state, direction, side)
    if after.head(side) is None:
        return 0
    penalty = legacy_hot_food_penalty(state, side, enemy, head, enemy_head, after)
    return penalty + numbered_hot_food_penalty(state, side, after)


def valid_hot_food_actor(state, side, direction, head, enemy_head):
    return head is not None and enemy_head is not None and direction in legal_moves(state, side)


def hot_lost_race(ours, enemy):
    return ours is not None and enemy is not None and enemy < ours and enemy <= 2


def recovers_hot_food(ours, after):
    return after is not None and after < ours and (ours <= 6 or after <= 3)


def hot_food_penalty_value(enemy_distance):
    weights = get_weights()
    return (
        weights['hot_lost_food_base'] + (3 - enemy_distance) * weights['hot_lost_food_enemy_bonus']
    )


def legacy_hot_food_penalty(state, side, enemy, head, enemy_head, after):
    penalty = 0
    foods = state.shot_food | (state.food if not state.food_values else frozenset())
    for food in foods:
        ours = temporal_shortest_distance(state, head, food, side)
        rival = temporal_shortest_distance(state, enemy_head, food, enemy)
        if not hot_lost_race(ours, rival):
            continue
        distance = temporal_shortest_distance(after, after.head(side), food, side)
        if recovers_hot_food(ours, distance):
            penalty += hot_food_penalty_value(rival)
    return penalty


def numbered_hot_food_penalty(state, side, after):
    if not state.food_values:
        return 0
    race = food_set_race(state, side)
    ours, enemy = race['our_distance'], race['enemy_distance']
    if not hot_lost_race(ours, enemy):
        return 0
    distances = food_distances(after, side, after.food)
    distance = min(distances.values(), default=None)
    return hot_food_penalty_value(enemy) if recovers_hot_food(ours, distance) else 0


def winning_suicide_move(state, side):
    """Compatibility: ordinary crashes cannot close a match in v7."""
    return None


def forced_kill_move(state, side, legal):
    """Do not deliberately seek crashes or mistake an enclosure for a kill."""
    return None


def survival_position_score(state, side):
    """Value current escape space and replies with the established survival weights."""
    if state.is_terminal():
        return terminal_survival_score(state, side)
    head = state.head(side)
    if head is None:
        return -1_000_000
    weights = get_weights()
    region = flood_region(state, head, strategic_blocked(state))
    exits = count_exits(state, side)
    replies = legal_moves(state, side)
    length = len(state.body(side))
    tail_ok = can_reach_tail(state, side)
    safety, _, _, _ = classify_safety(state, side)
    return survival_volume_score(
        state, side, head, len(region), exits, len(replies), length, tail_ok, safety, weights
    )


def terminal_survival_score(state, side):
    winner = state.terminal_winner()
    weights = get_weights()
    if winner == side:
        return weights['enemy_trapped_future_bonus']
    return -weights['future_no_reply_penalty'] if winner else 0


def mature_position_control(state, side, head, length):
    if length > 3:
        return center_control_score(state, head) + territory_control_score(state, side, head)
    return 0


def survival_volume_score(
    state, side, head, space, exits, replies, length, tail_ok, safety, weights
):
    score = min(space, 130) * weights['survival_region_value']
    score += min(exits, 4) * weights['survival_exit_value']
    score += min(replies, 4) * weights['survival_reply_value']
    score += mature_position_control(state, side, head, length)
    score += weights['survival_tail_bonus'] if tail_ok else 0
    safety_penalties = {
        'DANGEROUS': weights['fallback_danger_penalty'],
        'ACCEPTABLE_RISK': weights['future_corridor_penalty'] // 2,
    }
    score -= safety_penalties.get(safety, 0)
    score -= {0: weights['future_no_reply_penalty'], 1: weights['future_one_reply_penalty']}.get(
        replies, 0
    )
    score -= weights['future_corridor_penalty'] if exits <= 1 else 0
    score -= max(0, length * 2 + 6 - space) * weights['future_small_region_penalty']
    return score


def survival_move_analysis(state, side, direction, plan=None, hunger=0, repeat_count=0):
    """Compare a move against the worst available rival reply and our best continuation."""
    return SurvivalMoveAnalysis(state, side, direction, plan, hunger, repeat_count).run()


class SurvivalMoveAnalysis:
    """Keep the measured move, its response tree and the diagnostic schema together."""

    def __init__(self, state, side, direction, plan, hunger, repeat_count):
        self.state, self.side, self.direction = state, side, direction
        self.plan, self.hunger, self.repeat_count = plan, hunger, repeat_count
        self.weights = get_weights()
        self.enemy = state.enemy if side == state.side else opposite(side)
        self.head, self.space, self.exits = None, 0, 0

    def run(self):
        """Return the original diagnostic schema for this move and its worst rival reply."""
        if self.direction not in legal_moves(self.state, self.side):
            return self.result(-1_000_000, -1_000_000, score=-1_000_000)
        self.after = apply_move(self.state, self.direction, self.side)
        self.head = self.after.head(self.side)
        self.space, self.exits = survival_move_geometry(self.after, self.side, self.head)
        if self.after.is_terminal():
            future = terminal_survival_score(self.after, self.side)
            return self.result(
                0, future, score=future + survival_position_score(self.after, self.side)
            )
        safety, _, _, _ = classify_safety(self.after, self.side)
        if safety == 'SUICIDAL':
            return self.result(-1_000_000, 0)
        base = move_quality_score(
            self.state, self.side, self.direction, self.plan, self.hunger, self.repeat_count
        )[0]
        if self.head in food_objectives(self.state):
            base += self.weights['survival_immediate_food_bonus']
        future, move, replies, enemy_food = self.worst_response()
        return self.result(base, future, move, replies, enemy_food)

    def worst_response(self):
        if self.state.partial:
            return survival_position_score(self.after, self.side), None, 0, False
        replies = (self.enemy_response(move) for move in available_moves(self.after, self.enemy))
        return min(
            replies,
            key=lambda item: item[0],
            default=(-self.weights['future_no_reply_penalty'], None, 0, False),
        )

    def enemy_response(self, move):
        enemy_head = self.after.head(self.enemy)
        target = None if enemy_head is None else step(enemy_head, move)
        enemy_food = target in food_objectives(self.after) if target is not None else False
        state = apply_move(self.after, move, self.enemy, perspective=self.side)
        replies = available_moves(state, self.side)
        score = best_own_survival_reply(state, self.side, replies, self.weights)
        if enemy_food and not state.is_terminal():
            score -= self.weights['future_enemy_food_penalty']
        return score, move, len(replies), enemy_food

    def result(self, base, future, enemy_reply=None, reply_count=0, enemy_food=False, score=None):
        return {
            'direction': self.direction,
            'score': base + future if score is None else score,
            'base_score': base,
            'future_score': future,
            'enemy_reply': enemy_reply,
            'reply_count': reply_count,
            'region': self.space,
            'exits': self.exits,
            'food_now': self.head in food_objectives(self.state),
            'enemy_food': enemy_food,
        }


def survival_move_geometry(state, side, head):
    if head is None:
        return 0, 0
    return len(flood_region(state, head, strategic_blocked(state))), count_exits(state, side)


def best_own_survival_reply(state, side, replies, weights):
    if state.is_terminal():
        return survival_position_score(state, side)
    if not replies:
        return -weights['future_no_reply_penalty']
    return max(survival_reply_value(state, side, reply, weights) for reply in replies)


def survival_reply_value(state, side, reply, weights):
    score = survival_position_score(apply_move(state, reply, side), side)
    if step(state.head(side), reply) in food_objectives(state):
        score += weights['survival_immediate_food_bonus']
    return score


def rank_survival_moves(state, side, legal, plan=None, hunger=0, repeat_count=0):
    """Order movement diagnostics by worst-reply survival and escape space."""
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
    """Rank a safe move by its unchanged economic, spatial and rival-response terms."""
    return MoveQuality(state, side, direction, plan, hunger, repeat_count).rank()


class MoveQuality:
    """Keep the original score terms explicit while sharing one simulated successor."""

    def __init__(self, state, side, direction, plan, hunger, repeat_count):
        self.state, self.side, self.direction, self.plan = state, side, direction, plan
        self.repeat_count = repeat_count
        self.weights = get_weights()
        self.tempo_hunger = hunger + score_gap_urgency(state, side)
        self.after = apply_move(state, direction, side)
        self.wall_hit = step(
            state.head(side), direction
        ) in state.walls and not state.invincibility(side)
        self.head = self.after.head(side)

    def rank(self):
        """Return the score and tie-break fields used to compare movement choices."""
        if self.head is None:
            return -999999, 0, 0, 0
        self.safety, _, _, _ = classify_safety(self.after, self.side)
        if self.safety == 'SUICIDAL':
            return -999999, 0, 0, 0
        self.progress = move_plan_progress(self.state, self.after, self.side, self.plan)
        self.nearest = nearest_objective_distance(self.after, self.head, self.side)
        region = flood_region(self.after, self.head, strategic_blocked(self.after))
        exits = count_exits(self.after, self.side)
        return (
            self.score(),
            1 if self.head in food_objectives(self.state) else 0,
            exits,
            min(len(region), 60),
        )

    def score(self):
        weights = self.weights
        score = self.progress * (
            weights['fallback_progress_value']
            + self.tempo_hunger * weights['fallback_hunger_progress_value']
        )
        score += objective_distance_score(self.nearest, weights)
        score += fallback_hunger_food_score(
            self.state, self.side, self.after, self.head, self.tempo_hunger
        )
        score += fallback_denial_score(self.state, self.side, self.after, self.head)
        score += mature_position_control(
            self.after, self.side, self.head, len(self.after.body(self.side))
        )
        score -= stalled_cycle_cost(self.progress, self.tempo_hunger, self.repeat_count, weights)
        score += rival_response_penalty(self.state, self.side, self.direction)
        score -= hot_lost_food_penalty(self.state, self.side, self.direction)
        score -= fallback_danger_cost(self.safety, self.tempo_hunger, weights)
        score -= (
            move_bomb_loss(self.state, self.after, self.side) * weights['objective_point_value']
        )
        score -= weights['fallback_wall_hit_penalty'] if self.wall_hit else 0
        return score


def nearest_objective_distance(state, head, side):
    distances = (shortest_distance(state, head, food, side) for food in food_objectives(state))
    return min((distance for distance in distances if distance is not None), default=None)


def move_plan_progress(state, after, side, plan):
    if not plan or plan.race_margin < 0:
        return 0
    return manhattan(state.head(side), plan.food) - manhattan(after.head(side), plan.food)


def objective_distance_score(distance, weights):
    return 0 if distance is None else -distance * weights['fallback_food_distance_cost']


def stalled_cycle_cost(progress, hunger, repeats, weights):
    if progress <= 0 and hunger >= 10:
        return repeats * weights['fallback_cycle_penalty']
    return 0


def fallback_danger_cost(safety, hunger, weights):
    if safety == 'DANGEROUS' and hunger < 35:
        return weights['fallback_danger_penalty']
    return 0


def move_bomb_loss(state, after, side):
    if after.last_action.get('bomb_triggered') and not state.invincibility(side):
        return after.last_action['score_before'] - after.last_action['score_after'] + 1
    return 0


def fallback_food_positioning(state, side, legal, plan, hunger, repeat_count):
    """Choose the highest-ranked legal movement when no committed food route is used."""
    return max(
        legal,
        key=lambda direction: move_quality_score(
            state, side, direction, plan, hunger, repeat_count
        ),
    )
