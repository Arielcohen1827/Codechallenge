from collections import Counter, deque

from food_planner import (
    build_food_plan,
    classify_safety,
    choose_food_plan,
    center_control_score,
    fallback_food_positioning,
    forced_kill_move,
    hot_lost_food_penalty,
    position_control_penalty,
    rival_response_penalty,
    rival_response_stats,
    should_commit_to_food,
)
from snake_state import (
    FOOD_SCORE,
    apply_move,
    count_exits,
    flood_region,
    legal_moves,
    manhattan,
    parse_state,
    shortest_distance,
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

        current_target = self.targets.get((game_id, side))
        hunger = self.turns_since_food[(game_id, side)]
        repeat_count = self._repeat_count(game_id, state)
        plan = choose_food_plan(state, side, current_target, hunger)
        debug.update(
            {
                'current_target': current_target,
                'hunger': hunger,
                'repeat_count': repeat_count,
                'plan': self._plan_debug(plan),
                'commit_to_plan': should_commit_to_food(plan, hunger, repeat_count) if plan else False,
            }
        )
        if self.enable_debug:
            debug.update(
                {
                    'food_options': self._food_options(state, side, current_target, hunger),
                    'candidate_moves': self._candidate_moves(state, side, legal),
                }
            )

        if plan and should_commit_to_food(plan, hunger, repeat_count):
            self.targets[(game_id, side)] = plan.food
            move = plan.first_move
            reason = 'food_plan'
            if move in legal:
                stats = rival_response_stats(state, side, move)
                move_penalty = rival_response_penalty(state, side, move)
                debug['plan_move_stats'] = stats
                debug['plan_move_penalty'] = move_penalty
                hard_pressure = (
                    stats['forced_zero']
                    or stats['forced_delayed_zero']
                    or stats['zero_escape_replies']
                )
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
                self._remember_progress(game_id, side, state, plan)
                self._save_debug(game_id, debug, reason, move)
                self._remember(game_id, state)
                return move

        self.targets[(game_id, side)] = plan.food if plan else None
        move = fallback_food_positioning(state, side, legal, plan, hunger, repeat_count)
        self._save_debug(game_id, debug, 'fallback_positioning', move)
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
            self.targets.get((game_id, state.side)),
        )
        self.recent_states.setdefault(game_id, deque(maxlen=80)).append(key)

    def _repeat_count(self, game_id, state):
        key = (
            tuple(state.snakes.get(state.side, ())),
            tuple(state.snakes.get(state.enemy, ())),
            tuple(sorted(state.food)),
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
            'value': plan.value,
        }

    def _food_options(self, state, side, current_target, hunger):
        options = []
        for food in sorted(state.food):
            plan = build_food_plan(state, side, food, current_target, hunger)
            if plan is not None:
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
                distances = [shortest_distance(after, head, food, side) for food in after.food]
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
