from snake_bot.game_rules import SHOOT_COST, shoot_directions
from collections import Counter

from snake_bot.bot_weights import get_weights
from snake_bot.food_planner import (
    center_control_score,
    center_distance,
    state_dimensions,
    distance_map as planner_distance_map,
    classify_safety,
    hot_lost_food_penalty,
    food_objectives,
    move_quality_score,
    apply_move,
)
from snake_bot.snake_state import (
    available_moves,
    count_exits,
    flood_region,
    manhattan,
    step,
    strategic_blocked,
)
from snake_bot.search_engine import IterativeSearchEngine, observation_key, observed_body_key


def state_key(state):
    """Include geometry and observation certainty in every turn-local cache key."""
    return (
        state.rows,
        state.cols,
        state.levels,
        state.partial,
        observation_key(state),
        observed_body_key(state.observed_snake_cells),
        state.side,
        state.enemy,
        tuple(state.snakes.get('A', ())),
        tuple(state.snakes.get('B', ())),
        tuple(sorted(state.food)),
        tuple(sorted(state.food_values.items())),
        state.next_food_digit,
        tuple(sorted(state.pickups)),
        tuple(sorted(state.multipliers.items())),
        tuple(sorted(state.walls)),
        state.remaining_moves,
        tuple(sorted(state.reliable_tails)),
        tuple(sorted(state.scores.items())),
        tuple(sorted(state.tail_food_a)),
        tuple(sorted(state.tail_food_b)),
        state.crashed_a,
        state.crashed_b,
        tuple(sorted(state.bombs_large)),
        tuple(sorted(state.bombs_small)),
        tuple(sorted(state.shot_food)),
        state.pending_bomb_respawns,
        tuple(sorted(state.food_copy_targets.items())),
        tuple(sorted(state.stars)),
        state.invincible_a,
        state.invincible_b,
        state.star_spawn_due_round,
        state.pending_star_spawns,
        state.timeouts_a,
        state.timeouts_b,
    )


class TurnSearchCache:
    """Reuse immutable outcomes and geometry calculations within one decision."""

    def __init__(self):
        """Initialize independent caches for one decision and its root perspective."""
        self.root_side = None
        self.action_search = None
        self.after = {}
        self.legal = {}
        self.region = {}
        self.exits = {}
        self.safety = {}
        self.distances = {}
        self.voronoi = {}
        self.position = {}
        self.quick_position = {}
        self.move_quality_scores = {}

    def apply(self, state, side, direction):
        """Cache a conservative rule successor for a direction or action tuple."""
        if isinstance(direction, tuple):
            return self.apply_action(state, side, *direction)
        key = (state_key(state), side, direction, self.root_side)
        if key not in self.after:
            self.after[key] = apply_move(
                state, direction, side, perspective=self.root_side or side
            )
        return self.after[key]

    def apply_action(self, state, side, action_type, direction):
        """Cache explicit actions while preserving the ordinary-move call boundary."""
        if action_type == 'move':
            return self.apply(state, side, direction)
        from snake_bot.snake_state import apply_action

        key = (state_key(state), side, action_type, direction, self.root_side)
        if key not in self.after:
            self.after[key] = apply_action(state, action_type, direction, side)
        return self.after[key]

    def legal_actions(self, state, side):
        """Combine cached movement with currently confirmed selective shooting rays."""
        moves = tuple(self.legal_moves(state, side))
        return moves + selected_shots(state, side)

    def legal_moves(self, state, side):
        """Cache allowed directions for the specified actor and observation."""
        key = (state_key(state), side)
        if key not in self.legal:
            self.legal[key] = tuple(available_moves(state, side))
        return self.legal[key]

    def region_size(self, state, side):
        """Cache confirmed free space reachable from the actor head."""
        head = state.head(side)
        key = (state_key(state), side, head)
        if key not in self.region:
            self.region[key] = len(flood_region(state, head, strategic_blocked(state)))
        return self.region[key]

    def exits_count(self, state, side):
        """Cache immediate escape count under the current occupancy rules."""
        key = (state_key(state), side)
        if key not in self.exits:
            self.exits[key] = count_exits(state, side)
        return self.exits[key]

    def safety_class(self, state, side):
        """Cache survival classification and its space, exits and tail evidence."""
        key = (state_key(state), side)
        if key not in self.safety:
            self.safety[key] = classify_safety(state, side)
        return self.safety[key]

    def distance_map(self, state, start, side):
        """Cache geometry distances while respecting partial observation boundaries."""
        key = (state_key(state), start, side)
        if key not in self.distances:
            self.distances[key] = planner_distance_map(state, start, side)
        return self.distances[key]

    def move_quality(self, state, side, direction, plan, hunger, repeat_count):
        """Cache move quality without conflating plans with different targets."""
        plan_target = None if plan is None else plan.food
        key = (state_key(state), side, direction, plan_target, hunger, repeat_count)
        if key not in self.move_quality_scores:
            self.move_quality_scores[key] = move_quality_score(
                state, side, direction, plan, hunger, repeat_count
            )[0]
        return self.move_quality_scores[key]


def opponent_of(state, side):
    """Resolve the opposing player from the requested perspective."""
    return state.enemy if side == state.side else ('B' if side == 'A' else 'A')


def selected_shots(state, side):
    """Limit tactical branching to currently observed, affordable rival intersections."""
    enemy = 'B' if side == 'A' else 'A'
    if not can_select_shoot(state, side, enemy):
        return ()
    return tuple(
        ('shoot', direction)
        for direction in shoot_directions(
            state.head(side), state.rows, state.cols, state.walls, state.body(enemy)
        )
    )


def can_select_shoot(state, side, enemy):
    """Reject invalid or immune shots before generating their rays."""
    return (
        not state.is_terminal()
        and state.scores.get(side, 0) >= SHOOT_COST
        and not state.invincibility(enemy)
    )


def terminal_trap_value(state, side, enemy):
    """Only official elimination or the move limit is terminal."""
    if not state.is_terminal():
        return 0, None
    weights = get_weights()
    winner = state.terminal_winner()
    if winner == side:
        return weights['deep_enemy_trapped_bonus'], 'win'
    if winner == enemy:
        return -weights['deep_no_reply_penalty'], 'loss'
    return 0, 'draw'


def voronoi_control(state, side, cache):
    """Cache territory inferred from reachable confirmed cells and current heads."""
    key = (state_key(state), side)
    if key not in cache.voronoi:
        cache.voronoi[key] = calculate_voronoi(state, side, cache)
    return cache.voronoi[key]


def calculate_voronoi(state, side, cache):
    """Compare confirmed reachable cells without scanning the entire cube."""
    enemy = opponent_of(state, side)
    heads = (state.head(side), state.head(enemy))
    counts = Counter(ours=0, enemy=0, tie=0, frontier=0, food_delta=0, center=0)
    if None in heads:
        return voronoi_result(counts)
    ours = cache.distance_map(state, heads[0], side)
    theirs = cache.distance_map(state, heads[1], enemy)
    excluded = state.occupied() - set(heads)
    objectives = state.objective_cells()
    for cell in (ours.keys() | theirs.keys()) - excluded:
        count_voronoi_cell(state, cell, ours.get(cell), theirs.get(cell), objectives, counts)
    return voronoi_result(counts)


def count_voronoi_cell(state, cell, ours, theirs, objectives, counts):
    """Accumulate territory, centre, objective and frontier terms for one cell."""
    owner = territory_owner(ours, theirs)
    if owner is None:
        return
    counts[('enemy', 'tie', 'ours')[owner + 1]] += 1
    max_center = sum((size - 1) / 2 for size in state_dimensions(state))
    center_weight = max(0, int(max_center - center_distance(state, cell) + 1))
    counts['center'] += owner * center_weight
    counts['food_delta'] += owner * int(cell in objectives)
    counts['frontier'] += int(is_frontier(ours, theirs))


def territory_owner(ours, theirs):
    """Return +1 for ours, -1 for theirs, zero for a tie and None if unreachable."""
    if ours is None:
        return None if theirs is None else -1
    if theirs is None:
        return 1
    return int(ours < theirs) - int(theirs < ours)


def is_frontier(ours, theirs):
    """Whether both actors reach a cell within one move of each other."""
    return ours is not None and theirs is not None and abs(ours - theirs) <= 1


def voronoi_result(counts):
    """Apply the established territory weights to accumulated cell counts."""
    weights = get_weights()
    score = (counts['ours'] - counts['enemy']) * weights['voronoi_cell_value']
    score += counts['center'] * weights['voronoi_center_value']
    score += counts['food_delta'] * weights['voronoi_food_value']
    score -= counts['frontier'] * weights['voronoi_frontier_penalty']
    return {
        'score': score,
        **{name: counts[name] for name in ('ours', 'enemy', 'tie', 'frontier', 'food_delta')},
    }


def position_score(state, side, cache):
    """Evaluate confirmed position, territory and adequate room for the current body."""
    key = (state_key(state), side)
    if key not in cache.position:
        cache.position[key] = complete_position_score(state, side, cache)
    return cache.position[key]


def complete_position_score(state, side, cache):
    """Add territory and body-space adequacy to the shared survival score."""
    if state.head(side) is None:
        return -1_000_000
    score = quick_position_score(state, side, cache)
    score += voronoi_control(state, side, cache)['score']
    _, region, _, _ = cache.safety_class(state, side)
    wanted = len(state.body(side)) * 2 + 6
    return score - max(0, wanted - region) * get_weights()['deep_small_region_penalty']


def quick_position_score(state, side, cache):
    """Survival-focused evaluation for turns with many concurrent games."""
    key = (state_key(state), side)
    if key not in cache.quick_position:
        cache.quick_position[key] = base_position_score(state, side, cache)
    return cache.quick_position[key]


def base_position_score(state, side, cache):
    """Combine confirmed escape space, exits, replies and centre preference."""
    head = state.head(side)
    if head is None:
        return -1_000_000
    weights = get_weights()
    safety, region, exits, tail_ok = cache.safety_class(state, side)
    replies = len(cache.legal_moves(state, side))
    score = min(region, 130) * weights['deep_region_value']
    score += min(exits, 4) * weights['deep_exit_value']
    score += min(replies, 4) * weights['deep_reply_value']
    score += center_control_score(state, head)
    score += int(tail_ok) * weights['deep_tail_bonus']
    score -= safety_penalty(safety, weights)
    score -= reply_penalty(replies, weights)
    return score


def safety_penalty(safety, weights):
    """Map a safety classification to its established positional penalty."""
    return {
        'DANGEROUS': weights['deep_danger_penalty'],
        'ACCEPTABLE_RISK': weights['deep_risk_penalty'],
    }.get(safety, 0)


def reply_penalty(replies, weights):
    """Map zero or one productive replies to the accepted mobility penalty."""
    return {0: weights['deep_no_reply_penalty'], 1: weights['deep_one_reply_penalty']}.get(
        replies, 0
    )


def static_after_move_score(
    state, side, direction, plan, hunger, repeat_count, cache, quick=False
):
    """Score one legal action with the accepted position and objective terms."""
    if direction not in cache.legal_moves(state, side):
        return -1_000_000
    after = cache.apply(state, side, direction)
    weights = get_weights()
    score = plan_progress_score(state, side, direction, plan, hunger, after.head(side), weights)
    score -= hot_lost_food_penalty(state, side, direction) * weights['deep_hot_lost_multiplier']
    evaluator = quick_position_score if quick else position_score
    score += evaluator(after, side, cache)
    score += immediate_target_score(state, side, action_target(state, side, direction), weights)
    if lost_plan_step(plan, direction):
        score -= weights['deep_lost_plan_step_penalty']
    return score


def plan_progress_score(state, side, direction, plan, hunger, head, weights):
    """Value progress along the current objective route without changing its weights."""
    if plan is None or head is None:
        return 0
    before_distance = manhattan(state.head(side), plan.food)
    after_distance = manhattan(head, plan.food)
    score = (before_distance - after_distance) * (
        weights['fallback_progress_value'] + hunger * weights['fallback_hunger_progress_value']
    )
    score -= after_distance * weights['fallback_food_distance_cost']
    if direction == plan.first_move:
        score += weights['current_target_bonus']
    return score


def lost_plan_step(plan, direction):
    """Whether a move commits to an objective already lost in the race."""
    return plan is not None and plan.race_margin < 0 and direction == plan.first_move


def action_target(state, side, action):
    """Return the moving head destination, or None for stationary actions."""
    if isinstance(action, tuple) or state.head(side) is None:
        return None
    return step(state.head(side), action)


def immediate_target_score(state, side, target, weights):
    """Price actual target contents, preserving invincibility penalty immunity."""
    if target in food_objectives(state):
        return weights['deep_food_now_bonus'] + (
            state.food_reward(side, target) * weights['deep_score_point_value']
        )
    if target in state.pickups:
        return weights['deep_pickup_now_bonus']
    if state.invincibility(side):
        return 0
    return ordinary_target_penalty(state, target, weights)


def ordinary_target_penalty(state, target, weights):
    """Price a wrong digit or wall for an unprotected actor."""
    if target in state.wrong_food():
        return -weights['deep_wrong_food_penalty']
    if target in state.walls:
        return -weights['deep_wall_hit_penalty']
    return 0


def shoot_score_adjustment(before, after, action, perspective):
    """Price stationary shots on the same point scale as tactical food income."""
    if not isinstance(action, tuple) or action[0] != 'shoot' or after.is_terminal():
        return 0
    enemy = 'B' if perspective == 'A' else 'A'
    before_gap = before.scores.get(perspective, 0) - before.scores.get(enemy, 0)
    after_gap = after.scores.get(perspective, 0) - after.scores.get(enemy, 0)
    return (after_gap - before_gap) * get_weights()['deep_score_point_value']


def best_reply_score(state, side, cache):
    """Best known own reply using the unchanged tactical reward scale."""
    if state.is_terminal():
        return terminal_trap_value(state, side, opponent_of(state, side))[0]
    replies = cache.legal_actions(state, side)
    if not replies:
        return -get_weights()['deep_no_reply_penalty']
    return max(
        -1_000_000, max(reply_position_score(state, side, reply, cache) for reply in replies)
    )


def reply_position_score(state, side, reply, cache):
    """Evaluate one own reply including its actual immediate objective outcome."""
    target = action_target(state, side, reply)
    after = cache.apply(state, side, reply)
    if after.is_terminal():
        return terminal_trap_value(after, side, opponent_of(state, side))[0]
    score = position_score(after, side, cache)
    score += shoot_score_adjustment(state, after, reply, side)
    return score + immediate_target_score(state, side, target, get_weights())


def two_ply_score(state, side, direction, plan, hunger, repeat_count, cache):
    """Evaluate exact rival replies only for complete observations."""
    return tactical_score(state, side, direction, plan, hunger, repeat_count, cache, quick=False)


def fast_two_ply_score(state, side, direction, plan, hunger, repeat_count, cache):
    """Keep the same rival-reply contract with the cheaper positional evaluator."""
    return tactical_score(state, side, direction, plan, hunger, repeat_count, cache, quick=True)


def tactical_score(state, side, direction, plan, hunger, repeat_count, cache, quick):
    """Keep complete-state adversarial scoring separate from uncertain partial replies."""
    cache.root_side = side
    after = cache.apply(state, side, direction)
    immediate = static_after_move_score(
        state, side, direction, plan, hunger, repeat_count, cache, quick=quick
    )
    if after.is_terminal():
        value, info = terminal_reply_analysis(after, side)
        return immediate + value, info
    if state.partial:
        return immediate, uncertain_reply_analysis(after, side, cache)
    score, info = worst_opponent_reply(after, side, cache, quick)
    return immediate + score, info


def terminal_reply_analysis(state, side):
    """Package an official terminal result in the established diagnostic format."""
    value, outcome = terminal_trap_value(state, side, opponent_of(state, side))
    return value, {
        'enemy_reply': None,
        'reply_score': value,
        'our_replies': 0,
        'enemy_food': False,
        'forced_loss': outcome == 'loss',
        'terminal_outcome': outcome,
    }


def uncertain_reply_analysis(state, side, cache):
    """Represent hidden future replies explicitly instead of simulating remembered bodies."""
    return {
        'enemy_reply': None,
        'reply_score': 0,
        'our_replies': len(cache.legal_moves(state, side)),
        'enemy_food': False,
        'forced_loss': False,
        'reply_uncertain': True,
    }


def worst_opponent_reply(state, side, cache, quick):
    """Select the lowest-valued known rival reply with terminal-loss priority."""
    enemy = opponent_of(state, side)
    worst_key = None
    worst_score, worst_info = 0, None
    for move in cache.legal_actions(state, enemy):
        score, info = opponent_reply_analysis(state, side, enemy, move, cache, quick)
        key = (int(not info['forced_loss']), score)
        if worst_key is None or key < worst_key:
            worst_key, worst_score, worst_info = key, score, info
    return worst_score, worst_info


def opponent_reply_analysis(state, side, enemy, move, cache, quick):
    """Evaluate a rival action and expose its confirmed tactical consequences."""
    target = action_target(state, enemy, move)
    enemy_food = target in state.objective_cells()
    enemy_wall = target in state.walls and not state.invincibility(enemy)
    after = cache.apply(state, enemy, move)
    score = after_reply_value(after, side, enemy, cache, quick)
    score += shoot_score_adjustment(state, after, move, side)
    score -= enemy_objective_penalty(state, enemy, target, enemy_food)
    score += int(enemy_wall) * get_weights()['deep_enemy_wall_hit_bonus']
    return score, {
        'enemy_reply': move,
        'reply_score': score,
        'our_replies': len(cache.legal_moves(after, side)),
        'enemy_food': enemy_food,
        'enemy_wall_hit': enemy_wall,
        'forced_loss': after.is_terminal() and after.terminal_winner() == enemy,
    }


def after_reply_value(state, side, enemy, cache, quick):
    """Choose the full or quick established evaluator after a rival reply."""
    if state.is_terminal():
        return terminal_trap_value(state, side, enemy)[0]
    if quick:
        return quick_position_score(state, side, cache)
    return best_reply_score(state, side, cache)


def enemy_objective_penalty(state, enemy, target, is_food):
    """Price confirmed rival food and multiplier collection on the existing scale."""
    if not is_food:
        return 0
    weights = get_weights()
    penalty = weights['deep_enemy_food_penalty']
    if target in food_objectives(state):
        penalty += state.food_reward(enemy, target) * weights['deep_score_point_value']
    elif target in state.pickups:
        penalty += weights['deep_enemy_pickup_penalty']
    return penalty


def extra_depth_score(state, side, direction, cache):
    """Extend close full-information positions while leaving hidden replies uncertain."""
    if state.partial:
        return 0
    cache.root_side = side
    after = cache.apply(state, side, direction)
    enemy = opponent_of(state, side)
    if after.is_terminal():
        return terminal_trap_value(after, side, enemy)[0]
    values = [
        extra_enemy_reply_value(after, side, enemy, move, cache)
        for move in cache.legal_actions(after, enemy)
    ]
    return min(values, default=0)


def extra_enemy_reply_value(state, side, enemy, move, cache):
    """Evaluate the next known rival reply and the best available own continuation."""
    after = cache.apply(state, enemy, move)
    if after.is_terminal():
        return terminal_trap_value(after, side, enemy)[0]
    replies = cache.legal_actions(after, side)
    if not replies:
        value = -get_weights()['deep_no_reply_penalty']
    else:
        value = max(
            -1_000_000,
            max(extra_own_reply_value(after, side, enemy, reply, cache) for reply in replies),
        )
    return value + shoot_score_adjustment(state, after, move, side)


def extra_own_reply_value(state, side, enemy, reply, cache):
    """Choose the worst final rival response after an own continuation."""
    after = cache.apply(state, side, reply)
    if after.is_terminal():
        value = terminal_trap_value(after, side, enemy)[0]
    else:
        value = min(
            extra_final_reply_value(after, side, enemy, move, cache)
            for move in cache.legal_actions(after, enemy)
        )
    return value + shoot_score_adjustment(state, after, reply, side)


def extra_final_reply_value(state, side, enemy, move, cache):
    """Evaluate the final exact reply while retaining shot spending adjustments."""
    final = cache.apply(state, enemy, move)
    if final.is_terminal():
        return terminal_trap_value(final, side, enemy)[0]
    return position_score(final, side, cache) + shoot_score_adjustment(state, final, move, side)


def rank_deep_moves(
    state, side, legal, plan=None, hunger=0, repeat_count=0, cache=None, search_time_ms=None
):
    """Rank legal directions using tactical safety and accepted centered search adjustments."""
    cache = cache or TurnSearchCache()
    fast_mode = search_time_ms is not None and search_time_ms <= 18
    analyses = [
        initial_move_analysis(state, side, direction, plan, hunger, repeat_count, cache, fast_mode)
        for direction in legal
    ]
    analyses.sort(key=deep_sort_key, reverse=True)
    weights = get_weights()
    extend_close_moves(state, side, analyses, cache, fast_mode, weights)
    search = run_iterative_search(state, side, legal, plan, search_time_ms, weights)
    cache.action_search = search
    annotate_search(analyses, search)
    apply_search_adjustments(analyses, search, weights)
    prefer_safer_close_move(analyses)
    return analyses


def initial_move_analysis(state, side, direction, plan, hunger, repeat_count, cache, fast_mode):
    """Collect tactical score, safety and spatial diagnostics for a legal move."""
    scorer = fast_two_ply_score if fast_mode else two_ply_score
    score, info = scorer(state, side, direction, plan, hunger, repeat_count, cache)
    after = cache.apply(state, side, direction)
    voronoi = (
        {'score': 0, 'ours': 0, 'enemy': 0} if fast_mode else voronoi_control(after, side, cache)
    )
    safety, region, exits, tail_ok = cache.safety_class(after, side)
    return {
        'direction': direction,
        'score': score,
        'deep_score': score,
        'extra_score': None,
        'region': region,
        'exits': exits,
        'tail_reachable': tail_ok,
        'safety': safety,
        'replies': len(cache.legal_moves(after, side)),
        'food_now': after.head(side) in state.objective_cells(),
        'wall_hit': step(state.head(side), direction) in state.walls,
        'voronoi_score': voronoi['score'],
        'voronoi_ours': voronoi['ours'],
        'voronoi_enemy': voronoi['enemy'],
        **(info or {}),
    }


def extend_close_moves(state, side, analyses, cache, fast_mode, weights):
    """Extend the leading close or risky moves before iterative search adjustments."""
    if fast_mode or not needs_extra_depth(analyses, weights):
        return
    for item in analyses[:2]:
        extra = extra_depth_score(state, side, item['direction'], cache)
        item['extra_score'] = extra
        item['score'] += extra // 2
    analyses.sort(key=deep_sort_key, reverse=True)


def needs_extra_depth(analyses, weights):
    """Whether the current leading scores or survival evidence justify an extension."""
    if not analyses:
        return False
    return close_score_race(analyses, weights) or risky_best_move(analyses[0])


def close_score_race(analyses, weights):
    """Compare the top score gap against the existing extension threshold."""
    return (
        len(analyses) >= 2
        and analyses[0]['score'] - analyses[1]['score'] <= weights['deep_extra_depth_margin']
    )


def risky_best_move(best):
    """Identify forced-loss, mobility and rival-food warnings in the leading row."""
    return (
        best.get('forced_loss')
        or best['our_replies'] <= 1
        or best['exits'] <= 1
        or best.get('enemy_food')
    )


def run_iterative_search(state, side, legal, plan, search_time_ms, weights):
    """Create a bounded engine with matching geometry and the current objective."""
    budget = (
        weights['iterative_search_time_ms']
        if search_time_ms is None
        else max(5, int(search_time_ms))
    )
    engine, compact = IterativeSearchEngine.from_game_state(
        state, time_budget_ms=budget, max_depth=weights['iterative_search_max_depth']
    )
    return engine.search(compact, side, legal, target=None if plan is None else plan.food)


def annotate_search(analyses, search):
    """Attach completed search effort and scores to existing move diagnostics."""
    for item in analyses:
        item.update(
            search_score=search.scores.get(item['direction']),
            search_adjustment=0,
            search_depth=search.completed_depth,
            search_nodes=search.nodes,
            search_transposition_hits=search.transposition_hits,
            search_elapsed_ms=search.elapsed_ms,
        )


def apply_search_adjustments(analyses, search, weights):
    """Keep the accepted clip-before-center arithmetic and its exact integer rounding."""
    if not can_apply_search(search, weights):
        return
    clipped_scores = clipped_search_scores(analyses)
    center = sum(clipped_scores.values()) // len(clipped_scores) if clipped_scores else 0
    for item in analyses:
        update_search_adjustment(item, clipped_scores, center, weights)
    analyses.sort(key=deep_sort_key, reverse=True)


def can_apply_search(search, weights):
    """Require the configured minimum completed depth before adjusting rankings."""
    return search.completed_depth >= weights['iterative_search_min_apply_depth'] and bool(
        search.scores
    )


def clipped_search_scores(analyses):
    """Retain accepted saturation limits before calculating the shared centre."""
    applicable = [
        item
        for item in analyses
        if item['search_score'] is not None and not item.get('forced_loss')
    ]
    return {
        item['direction']: max(-200_000, min(200_000, item['search_score'])) for item in applicable
    }


def update_search_adjustment(item, clipped_scores, center, weights):
    """Apply the exact centred integer adjustment and existing per-move cap."""
    search_score = clipped_scores.get(item['direction'])
    if search_score is None:
        return
    adjustment = (search_score - center) * weights['iterative_search_weight_percent'] // 100
    cap = weights['iterative_search_adjustment_cap']
    adjustment = max(-cap, min(cap, adjustment))
    item['search_adjustment'] = adjustment
    item['score'] += adjustment


def prefer_safer_close_move(analyses):
    """Promote the first viable safer candidate within the accepted score margin."""
    if len(analyses) < 2:
        return
    best = analyses[0]
    for index, candidate in enumerate(analyses[1:], start=1):
        if prefer_candidate(best, candidate, get_weights()):
            analyses.insert(0, analyses.pop(index))
            return


def doomed_move(item):
    """Whether official forced loss or suicidal safety disqualifies a candidate."""
    return item.get('forced_loss') or item.get('safety') == 'SUICIDAL'


def prefer_candidate(best, candidate, weights):
    """Apply the accepted score-margin and safety arbitration in order."""
    if doomed_move(candidate):
        return False
    if doomed_move(best):
        return True
    if best['score'] - candidate['score'] > weights['deep_safe_preference_margin']:
        return False
    return improves_safety(best, candidate)


def improves_safety(best, candidate):
    """Combine independent mobility, reply, exit and rival-food improvements."""
    return (
        better_mobility(best, candidate)
        or better_replies(best, candidate)
        or better_exits(best, candidate)
        or avoids_enemy_food(best, candidate)
    )


def better_mobility(best, candidate):
    """Whether combined replies and exits improve by the accepted two-step margin."""
    return candidate['our_replies'] + candidate['exits'] >= best['our_replies'] + best['exits'] + 2


def better_replies(best, candidate):
    """Require more replies without sacrificing the current exit count."""
    return candidate['our_replies'] > best['our_replies'] and candidate['exits'] >= best['exits']


def better_exits(best, candidate):
    """Require more exits without sacrificing the current reply count."""
    return candidate['exits'] > best['exits'] and candidate['our_replies'] >= best['our_replies']


def avoids_enemy_food(best, candidate):
    """Whether the candidate removes the leading move rival-food warning."""
    return best.get('enemy_food') and not candidate.get('enemy_food')


def deep_sort_key(item):
    """Order terminal safety, score and tactical tie-breakers consistently."""
    return (
        0 if item.get('forced_loss') else 1,
        0 if item.get('safety') == 'SUICIDAL' else 1,
        item['score'],
        1 if item['food_now'] else 0,
        0 if item.get('wall_hit') else 1,
        item['our_replies'],
        item['exits'],
        item['region'],
        item['voronoi_score'],
    )
