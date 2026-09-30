from collections import deque

from bot_weights import get_weights
from food_planner import (
    center_control_score,
    classify_safety,
    hot_lost_food_penalty,
    move_quality_score,
)
from snake_state import (
    CRASH_PENALTY,
    RIVAL_CRASH_REWARD,
    apply_move,
    blocked_for_path,
    count_exits,
    flood_region,
    legal_moves,
    manhattan,
    neighbors,
    step,
)
from search_engine import IterativeSearchEngine


def state_key(state):
    return (
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
    )


class TurnSearchCache:
    def __init__(self):
        self.after = {}
        self.legal = {}
        self.region = {}
        self.exits = {}
        self.safety = {}
        self.distances = {}
        self.voronoi = {}
        self.position = {}
        self.move_quality_scores = {}

    def apply(self, state, side, direction):
        key = (state_key(state), side, direction)
        if key not in self.after:
            self.after[key] = apply_move(state, direction, side)
        return self.after[key]

    def legal_moves(self, state, side):
        key = (state_key(state), side)
        if key not in self.legal:
            self.legal[key] = tuple(legal_moves(state, side))
        return self.legal[key]

    def region_size(self, state, side):
        head = state.head(side)
        key = (state_key(state), side, head)
        if key not in self.region:
            self.region[key] = len(flood_region(state, head, state.occupied()))
        return self.region[key]

    def exits_count(self, state, side):
        key = (state_key(state), side)
        if key not in self.exits:
            self.exits[key] = count_exits(state, side)
        return self.exits[key]

    def safety_class(self, state, side):
        key = (state_key(state), side)
        if key not in self.safety:
            self.safety[key] = classify_safety(state, side)
        return self.safety[key]

    def distance_map(self, state, start, side):
        key = (state_key(state), start, side)
        if key in self.distances:
            return self.distances[key]
        if start is None:
            self.distances[key] = {}
            return self.distances[key]

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
        self.distances[key] = distances
        return distances

    def move_quality(self, state, side, direction, plan, hunger, repeat_count):
        plan_target = None if plan is None else plan.food
        key = (state_key(state), side, direction, plan_target, hunger, repeat_count)
        if key not in self.move_quality_scores:
            self.move_quality_scores[key] = move_quality_score(
                state, side, direction, plan, hunger, repeat_count
            )[0]
        return self.move_quality_scores[key]


def opponent_of(state, side):
    return state.enemy if side == state.side else ('B' if side == 'A' else 'A')


def terminal_trap_value(state, side, enemy):
    """Return a search value for ending the game by trapping the rival."""
    weights = get_weights()
    our_final = state.scores.get(side, 0) + RIVAL_CRASH_REWARD
    enemy_final = state.scores.get(enemy, 0) + CRASH_PENALTY
    if our_final > enemy_final:
        return weights['deep_enemy_trapped_bonus'], 'win'
    if our_final < enemy_final:
        return -weights['deep_no_reply_penalty'], 'loss'
    return 0, 'draw'


def voronoi_control(state, side, cache):
    enemy = opponent_of(state, side)
    head = state.head(side)
    enemy_head = state.head(enemy)
    key = (state_key(state), side)
    if key in cache.voronoi:
        return cache.voronoi[key]
    if head is None or enemy_head is None:
        result = {'score': 0, 'ours': 0, 'enemy': 0, 'tie': 0, 'frontier': 0, 'food_delta': 0}
        cache.voronoi[key] = result
        return result

    ours = cache.distance_map(state, head, side)
    theirs = cache.distance_map(state, enemy_head, enemy)
    occupied = state.occupied()
    our_cells = enemy_cells = ties = frontier = food_delta = 0
    center_score = 0
    center_r = (state.rows - 1) / 2
    center_c = (state.cols - 1) / 2
    max_center = center_r + center_c

    for r in range(state.rows):
        for c in range(state.cols):
            cell = (r, c)
            if cell in occupied and cell not in (head, enemy_head):
                continue
            our_d = ours.get(cell)
            enemy_d = theirs.get(cell)
            if our_d is None and enemy_d is None:
                continue
            center_weight = max(0, int(max_center - (abs(r - center_r) + abs(c - center_c)) + 1))
            if enemy_d is None or (our_d is not None and our_d < enemy_d):
                our_cells += 1
                center_score += center_weight
                if cell in state.objective_cells():
                    food_delta += 1
            elif our_d is None or enemy_d < our_d:
                enemy_cells += 1
                center_score -= center_weight
                if cell in state.objective_cells():
                    food_delta -= 1
            else:
                ties += 1
            if our_d is not None and enemy_d is not None and abs(our_d - enemy_d) <= 1:
                frontier += 1

    weights = get_weights()
    score = (our_cells - enemy_cells) * weights['voronoi_cell_value']
    score += center_score * weights['voronoi_center_value']
    score += food_delta * weights['voronoi_food_value']
    score -= frontier * weights['voronoi_frontier_penalty']
    result = {
        'score': score,
        'ours': our_cells,
        'enemy': enemy_cells,
        'tie': ties,
        'frontier': frontier,
        'food_delta': food_delta,
    }
    cache.voronoi[key] = result
    return result


def position_score(state, side, cache):
    key = (state_key(state), side)
    if key in cache.position:
        return cache.position[key]
    head = state.head(side)
    if head is None:
        cache.position[key] = -1_000_000
        return cache.position[key]

    weights = get_weights()
    safety, region, exits, tail_ok = cache.safety_class(state, side)
    replies = len(cache.legal_moves(state, side))
    body_len = len(state.body(side))
    voronoi = voronoi_control(state, side, cache)

    score = voronoi['score']
    score += min(region, 130) * weights['deep_region_value']
    score += min(exits, 4) * weights['deep_exit_value']
    score += min(replies, 4) * weights['deep_reply_value']
    score += center_control_score(state, head)
    if tail_ok:
        score += weights['deep_tail_bonus']
    if safety == 'DANGEROUS':
        score -= weights['deep_danger_penalty']
    elif safety == 'ACCEPTABLE_RISK':
        score -= weights['deep_risk_penalty']
    if not replies:
        score -= weights['deep_no_reply_penalty']
    elif replies == 1:
        score -= weights['deep_one_reply_penalty']
    wanted = body_len * 2 + 6
    if region < wanted:
        score -= (wanted - region) * weights['deep_small_region_penalty']

    cache.position[key] = score
    return score


def static_after_move_score(state, side, direction, plan, hunger, repeat_count, cache):
    if direction not in cache.legal_moves(state, side):
        return -1_000_000
    after = cache.apply(state, side, direction)
    head = after.head(side)
    weights = get_weights()
    score = 0
    if plan is not None and head is not None:
        before_distance = manhattan(state.head(side), plan.food)
        after_distance = manhattan(head, plan.food)
        progress = before_distance - after_distance
        score += progress * (weights['fallback_progress_value'] + hunger * weights['fallback_hunger_progress_value'])
        score -= after_distance * weights['fallback_food_distance_cost']
        if direction == plan.first_move:
            score += weights['current_target_bonus']
    score -= hot_lost_food_penalty(state, side, direction) * weights['deep_hot_lost_multiplier']
    score += position_score(after, side, cache)
    target = None if state.head(side) is None else step(state.head(side), direction)
    if target in state.food:
        score += weights['deep_food_now_bonus']
        score += state.food_reward(side, target) * weights['deep_score_point_value']
    elif target in state.pickups:
        score += weights['deep_pickup_now_bonus']
    elif target in state.wrong_food():
        score -= weights['deep_wrong_food_penalty']
    elif target in state.walls:
        score -= weights['deep_wall_hit_penalty']
    if plan is not None and plan.race_margin < 0 and direction == plan.first_move:
        score -= weights['deep_lost_plan_step_penalty']
    return score


def best_reply_score(state, side, cache):
    replies = cache.legal_moves(state, side)
    if not replies:
        return -get_weights()['deep_no_reply_penalty']
    best = -1_000_000
    for reply in replies:
        target = step(state.head(side), reply)
        after = cache.apply(state, side, reply)
        score = position_score(after, side, cache)
        if target in state.food:
            score += get_weights()['deep_food_now_bonus']
            score += state.food_reward(side, target) * get_weights()['deep_score_point_value']
        elif target in state.pickups:
            score += get_weights()['deep_pickup_now_bonus']
        elif target in state.wrong_food():
            score -= get_weights()['deep_wrong_food_penalty']
        elif target in state.walls:
            score -= get_weights()['deep_wall_hit_penalty']
        best = max(best, score)
    return best


def two_ply_score(state, side, direction, plan, hunger, repeat_count, cache):
    weights = get_weights()
    after = cache.apply(state, side, direction)
    enemy = opponent_of(state, side)
    enemy_moves = cache.legal_moves(after, enemy)
    immediate = static_after_move_score(state, side, direction, plan, hunger, repeat_count, cache)
    if not enemy_moves:
        terminal_value, terminal_outcome = terminal_trap_value(after, side, enemy)
        return immediate + terminal_value, {
            'enemy_reply': None,
            'reply_score': terminal_value,
            'our_replies': 99,
            'enemy_food': False,
            'forced_loss': terminal_outcome == 'loss',
            'terminal_outcome': terminal_outcome,
        }

    enemy_head = after.head(enemy)
    worst = None
    worst_key = None
    worst_info = None
    for enemy_move in enemy_moves:
        enemy_target = None if enemy_head is None else step(enemy_head, enemy_move)
        enemy_food = enemy_target in after.objective_cells() if enemy_target is not None else False
        enemy_wall_hit = enemy_target in after.walls if enemy_target is not None else False
        after_enemy = cache.apply(after, enemy, enemy_move)
        our_replies = cache.legal_moves(after_enemy, side)
        forced_loss = not our_replies
        reply_score = best_reply_score(after_enemy, side, cache)
        if enemy_food:
            reply_score -= weights['deep_enemy_food_penalty']
            if enemy_target in after.food:
                reply_score -= after.food_reward(enemy, enemy_target) * weights['deep_score_point_value']
            elif enemy_target in after.pickups:
                reply_score -= weights['deep_enemy_pickup_penalty']
        if enemy_wall_hit:
            reply_score += weights['deep_enemy_wall_hit_bonus']
        # Terminal survival is lexicographic. A rival reply that leaves us no
        # legal move must always be treated as worse than any playable state,
        # regardless of the heuristic score accumulated before it.
        reply_key = (0 if forced_loss else 1, reply_score)
        if worst_key is None or reply_key < worst_key:
            worst = reply_score
            worst_key = reply_key
            worst_info = {
                'enemy_reply': enemy_move,
                'reply_score': reply_score,
                'our_replies': len(our_replies),
                'enemy_food': enemy_food,
                'enemy_wall_hit': enemy_wall_hit,
                'forced_loss': forced_loss,
            }
    return immediate + (worst or 0), worst_info


def extra_depth_score(state, side, direction, cache):
    weights = get_weights()
    after = cache.apply(state, side, direction)
    enemy = opponent_of(state, side)
    enemy_moves = cache.legal_moves(after, enemy)
    if not enemy_moves:
        return weights['deep_enemy_trapped_bonus']

    worst = None
    for enemy_move in enemy_moves:
        after_enemy = cache.apply(after, enemy, enemy_move)
        replies = cache.legal_moves(after_enemy, side)
        if not replies:
            reply_value = -weights['deep_no_reply_penalty']
        else:
            best = -1_000_000
            for reply in replies:
                after_reply = cache.apply(after_enemy, side, reply)
                enemy2 = cache.legal_moves(after_reply, enemy)
                if not enemy2:
                    terminal_value, _ = terminal_trap_value(after_reply, side, enemy)
                    score = position_score(after_reply, side, cache) + terminal_value
                else:
                    score = min(position_score(cache.apply(after_reply, enemy, enemy_reply), side, cache) for enemy_reply in enemy2)
                best = max(best, score)
            reply_value = best
        if worst is None or reply_value < worst:
            worst = reply_value
    return worst or 0


def rank_deep_moves(state, side, legal, plan=None, hunger=0, repeat_count=0, cache=None):
    cache = cache or TurnSearchCache()
    analyses = []
    for direction in legal:
        score, info = two_ply_score(state, side, direction, plan, hunger, repeat_count, cache)
        after = cache.apply(state, side, direction)
        voronoi = voronoi_control(after, side, cache)
        safety, region, exits, _ = cache.safety_class(after, side)
        analyses.append(
            {
                'direction': direction,
                'score': score,
                'deep_score': score,
                'extra_score': None,
                'region': region,
                'exits': exits,
                'safety': safety,
                'replies': len(cache.legal_moves(after, side)),
                'food_now': after.head(side) in state.objective_cells(),
                'wall_hit': step(state.head(side), direction) in state.walls,
                'voronoi_score': voronoi['score'],
                'voronoi_ours': voronoi['ours'],
                'voronoi_enemy': voronoi['enemy'],
                **(info or {}),
            }
        )

    analyses.sort(key=deep_sort_key, reverse=True)
    weights = get_weights()
    close_race = len(analyses) >= 2 and analyses[0]['score'] - analyses[1]['score'] <= weights['deep_extra_depth_margin']
    risky_best = analyses and (
        analyses[0].get('forced_loss')
        or analyses[0]['our_replies'] <= 1
        or analyses[0]['exits'] <= 1
        or analyses[0].get('enemy_food')
    )
    if close_race or risky_best:
        for item in analyses[:2]:
            extra = extra_depth_score(state, side, item['direction'], cache)
            item['extra_score'] = extra
            item['score'] += extra // 2
        analyses.sort(key=deep_sort_key, reverse=True)

    engine, compact = IterativeSearchEngine.from_game_state(
        state,
        time_budget_ms=weights['iterative_search_time_ms'],
        max_depth=weights['iterative_search_max_depth'],
    )
    search = engine.search(
        compact,
        side,
        legal,
        target=None if plan is None else plan.food,
    )
    for item in analyses:
        item['search_score'] = search.scores.get(item['direction'])
        item['search_adjustment'] = 0
        item['search_depth'] = search.completed_depth
        item['search_nodes'] = search.nodes
        item['search_transposition_hits'] = search.transposition_hits
        item['search_elapsed_ms'] = search.elapsed_ms
    if search.completed_depth >= weights['iterative_search_min_apply_depth'] and search.scores:
        applicable = [
            item
            for item in analyses
            if item['search_score'] is not None
            and item.get('safety') != 'SUICIDAL'
            and not item.get('forced_loss')
        ]
        clipped_scores = {
            item['direction']: max(-200_000, min(200_000, item['search_score']))
            for item in applicable
        }
        center = (
            sum(clipped_scores.values()) // len(clipped_scores)
            if clipped_scores
            else 0
        )
        for item in analyses:
            search_score = clipped_scores.get(item['direction'])
            if search_score is None:
                continue
            adjustment = (
                (search_score - center)
                * weights['iterative_search_weight_percent']
                // 100
            )
            cap = weights['iterative_search_adjustment_cap']
            adjustment = max(-cap, min(cap, adjustment))
            item['search_adjustment'] = adjustment
            item['score'] += adjustment
        analyses.sort(key=deep_sort_key, reverse=True)
    prefer_safer_close_move(analyses)
    return analyses


def prefer_safer_close_move(analyses):
    if len(analyses) < 2:
        return
    weights = get_weights()
    best = analyses[0]
    for index, candidate in enumerate(analyses[1:], start=1):
        if candidate.get('forced_loss') or candidate.get('safety') == 'SUICIDAL':
            continue
        if best.get('forced_loss') or best.get('safety') == 'SUICIDAL':
            analyses.insert(0, analyses.pop(index))
            return
        if best['score'] - candidate['score'] > weights['deep_safe_preference_margin']:
            continue
        best_mobility = best['our_replies'] + best['exits']
        candidate_mobility = candidate['our_replies'] + candidate['exits']
        safer_mobility = candidate_mobility >= best_mobility + 2
        safer_reply = candidate['our_replies'] > best['our_replies'] and candidate['exits'] >= best['exits']
        safer_exit = candidate['exits'] > best['exits'] and candidate['our_replies'] >= best['our_replies']
        avoids_enemy_food = best.get('enemy_food') and not candidate.get('enemy_food')
        if safer_mobility or safer_reply or safer_exit or avoids_enemy_food:
            analyses.insert(0, analyses.pop(index))
            return


def deep_sort_key(item):
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
