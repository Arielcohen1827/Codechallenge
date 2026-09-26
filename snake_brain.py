from collections import Counter, deque

from bot_weights import get_weights
from food_planner import (
    build_food_plan,
    build_sequence_setup_plans,
    classify_safety,
    choose_food_plan,
    center_control_score,
    fallback_food_positioning,
    forced_kill_move,
    hot_lost_food_penalty,
    move_quality_score,
    position_control_penalty,
    rival_response_penalty,
    rival_response_stats,
    rank_survival_moves,
    should_commit_to_food,
    should_use_plan_for_positioning,
    sequence_gate_hold_move,
)
from move_search import TurnSearchCache, rank_deep_moves
from snake_state import (
    FOOD_SCORE,
    apply_move,
    count_exits,
    flood_region,
    legal_moves,
    manhattan,
    parse_state,
    shortest_distance,
    step,
)


class SnakeBrain:
    def __init__(self, enable_debug=True):
        self.enable_debug = enable_debug
        self.targets = {}
        self.turns_since_food = Counter()
        self.last_scores = {}
        self.recent_states = {}
        self.last_progress = {}
        self.previous_snakes = {}
        self.current_states = {}
        self.last_decisions = {}

    def choose_move(self, data):
        game_id = str(data.get('game_id', 'default'))
        state = parse_state(data, self.previous_snakes.get(game_id))
        self.current_states[game_id] = state
        side = state.side
        self._update_food_clock(game_id, side, state)

        legal = legal_moves(state, side)
        debug = self._base_debug(data, state, legal)
        if not legal:
            self._save_debug(game_id, debug, 'no_legal_moves', 'up')
            return 'up'

        kill = forced_kill_move(state, side, legal)
        if kill:
            self._save_debug(game_id, debug, 'forced_kill', kill)
            self._remember(game_id, state)
            return kill

        non_penalty_legal = [
            direction for direction in legal
            if step(state.head(side), direction) not in state.wrong_food()
        ]
        if non_penalty_legal:
            if len(non_penalty_legal) != len(legal):
                debug['avoided_wrong_digit_moves'] = sorted(set(legal) - set(non_penalty_legal))
            legal = non_penalty_legal
            debug['legal'] = legal

        current_target = self.targets.get((game_id, side))
        hunger = self.turns_since_food[(game_id, side)]
        repeat_count = self._repeat_count(game_id, state)
        search_cache = TurnSearchCache()
        plan = choose_food_plan(state, side, current_target, hunger)
        deep_moves = rank_deep_moves(state, side, legal, plan, hunger, repeat_count, search_cache)
        debug.update(
            {
                'current_target': current_target,
                'hunger': hunger,
                'repeat_count': repeat_count,
                'plan': self._plan_debug(plan),
                'commit_to_plan': should_commit_to_food(plan, hunger, repeat_count) if plan else False,
                'deep_moves': deep_moves[:4],
            }
        )
        if self.enable_debug:
            debug.update(
                {
                    'food_options': self._food_options(state, side, current_target, hunger),
                    'candidate_moves': self._candidate_moves(state, side, legal),
                }
            )

        commit_to_plan = should_commit_to_food(plan, hunger, repeat_count) if plan else False
        debug['commit_to_plan'] = commit_to_plan

        if plan and commit_to_plan:
            self.targets[(game_id, side)] = plan.food
            move = plan.first_move
            reason = 'food_plan'
            if move in legal:
                best_deep = deep_moves[0] if deep_moves else None
                plan_deep = next((item for item in deep_moves if item['direction'] == move), None)
                adjacent_trap = (
                    plan.our_distance == 1
                    and not plan.endgame_acceptable
                    and plan_deep is not None
                    and (
                        plan_deep.get('our_replies', 0) == 0
                        or plan_deep.get('safety') == 'SUICIDAL'
                    )
                    and best_deep is not None
                    and best_deep['direction'] != move
                    and best_deep.get('safety') != 'SUICIDAL'
                    and best_deep.get('our_replies', 0) > 0
                )
                debug['adjacent_trap_detected'] = adjacent_trap
                if adjacent_trap:
                    move = best_deep['direction']
                    reason = 'adjacent_trap_avoidance'
                    self._remember_progress(game_id, side, state, plan)
                    self._save_debug(game_id, debug, reason, move)
                    self._remember(game_id, state)
                    return move

                hold_move = sequence_gate_hold_move(state, side, legal, plan, deep_moves)
                if hold_move is not None:
                    debug['sequence_gate_held'] = True
                    move = hold_move
                    reason = 'sequence_gate_hold'
                    self._remember_progress(game_id, side, state, plan)
                    self._save_debug(game_id, debug, reason, move)
                    self._remember(game_id, state)
                    return move

                stats = rival_response_stats(state, side, move)
                move_penalty = rival_response_penalty(state, side, move)
                plan_position_penalty = position_control_penalty(state, side, move)
                plan_hot_lost_penalty = hot_lost_food_penalty(state, side, move)
                debug['plan_move_stats'] = stats
                debug['plan_move_penalty'] = move_penalty
                debug['plan_position_penalty'] = plan_position_penalty
                debug['plan_hot_lost_food_penalty'] = plan_hot_lost_penalty
                hard_pressure = stats['forced_zero'] or stats['forced_delayed_zero']
                if plan.our_distance > 1:
                    hard_pressure = hard_pressure or bool(stats['zero_escape_replies'])
                soft_pressure = (
                    stats['one_escape_replies'] >= 2
                    or stats['delayed_forced_zero_replies']
                    or move_penalty <= -6000
                )
                if (
                    hard_pressure
                    or (plan.our_distance > 1 and soft_pressure)
                ):
                    fallback = fallback_food_positioning(state, side, legal, plan, hunger, repeat_count)
                    if rival_response_penalty(state, side, fallback) > move_penalty:
                        move = fallback
                        reason = 'fallback_rival_pressure'
                    least_trappable = max(legal, key=lambda candidate: rival_response_penalty(state, side, candidate))
                    if rival_response_penalty(state, side, least_trappable) - rival_response_penalty(state, side, move) >= 5000:
                        move = least_trappable
                        reason = 'least_trappable'
                weights = get_weights()
                should_arbitrate = (
                    plan.our_distance > 1
                    and (
                        hard_pressure
                        or soft_pressure
                        or plan.safety != 'SAFE'
                        or plan.race_margin <= 1
                        or plan.exits_after <= 1
                        or plan.space_after <= 80
                        or repeat_count >= 1
                        or plan_hot_lost_penalty > 0
                        or plan_position_penalty >= weights['position_edge2_penalty']
                    )
                )
                debug['tactical_arbitration'] = should_arbitrate
                plan_deep = next((item for item in deep_moves if item['direction'] == move), None)
                deep_override_allowed = (
                    hard_pressure
                    or soft_pressure
                    or plan.safety != 'SAFE'
                    or plan.exits_after <= 1
                    or plan.space_after <= 80
                    or repeat_count >= 1
                    or plan_hot_lost_penalty > 0
                    or plan_position_penalty >= weights['position_edge1_penalty']
                )
                debug['deep_override_allowed'] = deep_override_allowed
                if deep_override_allowed and plan.our_distance > 1 and best_deep and best_deep['direction'] != move:
                    if plan_deep and best_deep['score'] >= plan_deep['score'] + weights['deep_plan_override_margin']:
                        move = best_deep['direction']
                        reason = 'deep_control'
                if should_arbitrate:
                    best_tactical = max(
                        legal,
                        key=lambda candidate: search_cache.move_quality(state, side, candidate, plan, hunger, repeat_count),
                    )
                    plan_quality = (
                        search_cache.move_quality(state, side, move, plan, hunger, repeat_count),
                        0,
                        0,
                        0,
                    )
                    best_quality = (
                        search_cache.move_quality(state, side, best_tactical, plan, hunger, repeat_count),
                        0,
                        0,
                        0,
                    )
                    best_stats = rival_response_stats(state, side, best_tactical)
                    debug['plan_move_quality'] = plan_quality[0]
                    debug['best_tactical_move'] = best_tactical
                    debug['best_tactical_quality'] = best_quality[0]
                    debug['best_tactical_stats'] = best_stats
                    margin = weights['plan_tactical_override_margin']
                    if stats['one_escape_replies'] or stats['delayed_forced_zero_replies']:
                        margin = weights['one_escape_tactical_override_margin']
                    elif plan_position_penalty >= weights['position_edge1_penalty']:
                        margin = weights['edge_tactical_override_margin']
                    if best_tactical != move and best_quality[0] >= plan_quality[0] + margin:
                        move = best_tactical
                        reason = 'tactical_override'
                selected_deep = next((item for item in deep_moves if item['direction'] == move), None)
                survivable_deep = next(
                    (
                        item for item in deep_moves
                        if not item.get('forced_loss') and item.get('safety') != 'SUICIDAL'
                    ),
                    None,
                )
                if selected_deep and selected_deep.get('forced_loss') and survivable_deep:
                    move = survivable_deep['direction']
                    reason = 'forced_loss_avoidance'
                    debug['forced_loss_avoided'] = selected_deep['direction']
                self._remember_progress(game_id, side, state, plan)
                self._save_debug(game_id, debug, reason, move)
                self._remember(game_id, state)
                return move

        self.targets[(game_id, side)] = plan.food if plan and should_use_plan_for_positioning(plan) else None
        positioning_plan = plan if should_use_plan_for_positioning(plan) else None
        fallback_deep = deep_moves or rank_deep_moves(state, side, legal, positioning_plan, hunger, repeat_count, search_cache)
        if fallback_deep:
            debug['survival_candidates'] = fallback_deep[:4]
            move = fallback_deep[0]['direction']
            reason = 'deep_control'
            if not plan or not should_use_plan_for_positioning(plan):
                reason = 'survival_control'
        else:
            move = fallback_food_positioning(state, side, legal, positioning_plan, hunger, repeat_count)
            reason = 'fallback_positioning'
        self._save_debug(game_id, debug, reason, move)
        self._remember(game_id, state)
        return move

    def safe_direction(self, data, direction):
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = parse_state(data, self.previous_snakes.get(game_id))
            self.current_states[game_id] = state
        legal = legal_moves(state, state.side)
        if direction in legal:
            return direction
        if legal:
            safe = fallback_food_positioning(state, state.side, legal, None, 0, 0)
            self._mark_safe_repair(game_id, direction, safe)
            return safe
        self._mark_safe_repair(game_id, direction, 'up')
        return 'up'

    def commit_move(self, data, direction):
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = parse_state(data, self.previous_snakes.get(game_id))
        if direction in legal_moves(state, state.side):
            state = apply_move(state, direction, state.side)
        self.previous_snakes[game_id] = state.snakes
        self.current_states.pop(game_id, None)

    def forget(self, game_id):
        for key in list(self.targets):
            if key[0] == str(game_id):
                self.targets.pop(key, None)
                self.turns_since_food.pop(key, None)
                self.last_scores.pop(key, None)
                self.last_progress.pop(key, None)
        self.recent_states.pop(str(game_id), None)
        self.previous_snakes.pop(str(game_id), None)
        self.current_states.pop(str(game_id), None)
        self.last_decisions.pop(str(game_id), None)

    def decision_debug(self, game_id):
        return self.last_decisions.get(str(game_id))

    def _update_food_clock(self, game_id, side, state):
        key = (game_id, side)
        score = state.scores.get(side, 0)
        previous = self.last_scores.get(key)
        if previous is None or score - previous >= FOOD_SCORE:
            self.turns_since_food[key] = 0
            self.targets[key] = None
        else:
            self.turns_since_food[key] += 1
        self.last_scores[key] = score

    def _remember(self, game_id, state):
        key = (
            tuple(state.snakes.get(state.side, ())),
            tuple(state.snakes.get(state.enemy, ())),
            tuple(sorted(state.food)),
            tuple(sorted(state.food_values.items())),
            tuple(sorted(state.pickups)),
            tuple(sorted(state.multipliers.items())),
            tuple(sorted(state.walls)),
            self.targets.get((game_id, state.side)),
        )
        self.recent_states.setdefault(game_id, deque(maxlen=80)).append(key)

    def _repeat_count(self, game_id, state):
        key = (
            tuple(state.snakes.get(state.side, ())),
            tuple(state.snakes.get(state.enemy, ())),
            tuple(sorted(state.food)),
            tuple(sorted(state.food_values.items())),
            tuple(sorted(state.pickups)),
            tuple(sorted(state.multipliers.items())),
            tuple(sorted(state.walls)),
            self.targets.get((game_id, state.side)),
        )
        return sum(1 for item in self.recent_states.get(game_id, ()) if item == key)

    def _remember_progress(self, game_id, side, state, plan):
        head = state.head(side)
        if head is not None:
            self.last_progress[(game_id, side)] = (plan.food, manhattan(head, plan.food))

    def _base_debug(self, data, state, legal):
        side = state.side
        return {
            'event': 'decision_debug',
            'game_id': str(data.get('game_id', 'default')),
            'turn_token': data.get('turn_token'),
            'side': side,
            'head': state.head(side),
            'enemy_head': state.head(state.enemy),
            'food': sorted(state.food),
            'numbered_food': sorted((pos, value) for pos, value in state.food_values.items()),
            'next_food_digit': state.next_food_digit,
            'pickups': sorted(state.pickups),
            'walls': sorted(state.walls),
            'multipliers': state.multipliers,
            'legal': legal,
            'scores': state.scores,
            'remaining_moves': state.remaining_moves,
        }

    def _save_debug(self, game_id, debug, reason, direction):
        debug['reason'] = reason
        debug['direction'] = direction
        self.last_decisions[str(game_id)] = debug

    def _mark_safe_repair(self, game_id, original, repaired):
        debug = self.last_decisions.setdefault(str(game_id), {'event': 'decision_debug'})
        debug['safe_repaired_from'] = original
        debug['direction'] = repaired
        debug['reason'] = 'safe_direction_repair'

    def _plan_debug(self, plan):
        if plan is None:
            return None
        return {
            'food': plan.food,
            'our_distance': plan.our_distance,
            'enemy_distance': plan.enemy_distance,
            'race_margin': plan.race_margin,
            'safety': plan.safety,
            'first_move': plan.first_move,
            'space_after': plan.space_after,
            'exits_after': plan.exits_after,
            'tail_reachable_after': plan.tail_reachable_after,
            'next_food_distance': plan.next_food_distance,
            'chain_score': plan.chain_score,
            'denial_score': plan.denial_score,
            'rival_threat': plan.rival_threat,
            'tempo_urgency': plan.tempo_urgency,
            'temporal_tail_path': plan.temporal_tail_path,
            'endgame_acceptable': plan.endgame_acceptable,
            'objective_kind': plan.objective_kind,
            'objective_reward': plan.objective_reward,
            'sequence_rank': plan.sequence_rank,
            'value': plan.value,
        }

    def _food_options(self, state, side, current_target, hunger):
        options = []
        for food in sorted(state.objective_cells()):
            plan = build_food_plan(state, side, food, current_target, hunger)
            if plan is not None:
                options.append(self._plan_debug(plan))
        for plan in build_sequence_setup_plans(state, side, current_target, hunger):
            options.append(self._plan_debug(plan))
        options.sort(key=lambda item: item['value'], reverse=True)
        return options

    def _candidate_moves(self, state, side, legal):
        candidates = []
        for direction in legal:
            after = apply_move(state, direction, side)
            head = after.head(side)
            safety, space, exits_after, tail_ok = classify_safety(after, side)
            nearest_food = None
            if head is not None:
                distances = [shortest_distance(after, head, food, side) for food in after.objective_cells()]
                distances = [distance for distance in distances if distance is not None]
                nearest_food = min(distances) if distances else None
            candidates.append(
                {
                    'direction': direction,
                    'safety': safety,
                    'space': space,
                    'exits': exits_after,
                    'tail_reachable': tail_ok,
                    'region': 0 if head is None else len(flood_region(after, head, after.occupied())),
                    'count_exits': count_exits(after, side),
                    'nearest_food_distance': nearest_food,
                    'center_control_score': center_control_score(after, head),
                    'position_control_penalty': position_control_penalty(state, side, direction),
                    'rival_penalty': rival_response_penalty(state, side, direction),
                    'hot_lost_food_penalty': hot_lost_food_penalty(state, side, direction),
                    'wrong_food_penalty': step(state.head(side), direction) in state.wrong_food(),
                    'wall_hit': step(state.head(side), direction) in state.walls,
                    'rival_stats': rival_response_stats(state, side, direction),
                }
            )
        return candidates


def final_safe_direction(data, direction):
    state = parse_state(data)
    legal = legal_moves(state, state.side)
    if direction in legal:
        return direction
    if legal:
        return fallback_food_positioning(state, state.side, legal, None, 0, 0)
    return 'up'
