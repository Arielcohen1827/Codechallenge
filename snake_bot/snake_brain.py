from collections import Counter, deque
from dataclasses import replace
from snake_bot import food_planner as planning_module

from snake_bot.bot_weights import get_weights
from snake_bot.game_rules import CRASH_KEEP_LENGTH, CRASH_PENALTY, MAX_MOVES, SHOOT_COST, shoot_directions
from snake_bot.food_planner import (
    advantage_control_move,
    apply_move as simulated_move,
    build_food_plans,
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
from snake_bot.move_search import TurnSearchCache, rank_deep_moves
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import (
    FOOD_SCORE,
    DIRS,
    apply_action,
    apply_move,
    available_moves,
    direction_between,
    legal_moves,
    manhattan,
    parse_state,
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
        self.crash_history = {}
        self.observed_states = {}
        self.pending_moves = {}
        self.observation_tokens = {}
        self.action_searches = {}

    def observe_state(self, data):
        """Reconcile server snapshots; predictions never become confirmed history."""
        game_id = str(data.get('game_id', 'default'))
        previous = self.observed_states.get(game_id)
        history = dict(self.crash_history.get(game_id, {'A': None, 'B': None}))
        state = parse_state(data, self.previous_snakes.get(game_id), history)
        if (previous is not None and data.get('remaining_moves') is not None
                and state.remaining_moves > previous.remaining_moves):
            return None
        token = (data.get('turn_token'), state.remaining_moves)
        pending = self.pending_moves.get(game_id)
        reference = previous
        if (pending
                and pending[0].remaining_moves - state.remaining_moves == 1
                and data.get('action', data.get('action_type')) == (pending[2] if len(pending) > 2 else 'move')):
            reference = pending[0]
        single = (reference is not None and reference.remaining_moves - state.remaining_moves == 1
                  and state.side != reference.side)
        coherent_action = None
        if single:
            history, coherent_action = self._single_action_history(reference, state, data, history)
        duplicate = previous is not None and previous.remaining_moves == state.remaining_moves
        if duplicate and not single:
            if (state.board == previous.board and state.scores == previous.scores
                    and state.multipliers == previous.multipliers and state.side == previous.side
                    and state.invincible_a == previous.invincible_a
                    and state.invincible_b == previous.invincible_b
                    and state.timeouts_a == previous.timeouts_a
                    and state.timeouts_b == previous.timeouts_b):
                self.current_states[game_id] = previous
                self.observation_tokens[game_id] = token
                return previous
            state = parse_state(data, previous.snakes, history)
            for side in ('A', 'B'):
                if history[side] is False and (
                        state.body(side) != previous.body(side)
                        or state.scores[side] != previous.scores[side]
                        or state.multipliers[side] != previous.multipliers[side]):
                    history[side] = None
        if previous is None:
            # Only an explicit initial horizon proves nobody has taken a turn.
            if data.get('remaining_moves') is not None:
                if state.remaining_moves == MAX_MOVES and state.side == 'A':
                    history = {'A': False, 'B': False}
                elif state.remaining_moves == MAX_MOVES - 1 and state.side == 'B':
                    history['B'] = False
        elif single:
            if (pending and reference is pending[0]
                    and coherent_action == (pending[2] if len(pending) > 2 else 'move', pending[1])):
                self.pending_moves.pop(game_id, None)
        elif not duplicate:
            delta = previous.remaining_moves - state.remaining_moves
            contiguous = delta == 2 and previous.side == state.side
            if not contiguous:
                history = {side: True if value is True else None
                           for side, value in history.items()}
            else:
                for side in ('A', 'B'):
                    before, after = previous.body(side), state.body(side)
                    if not before or not after:
                        if history[side] is False:
                            history[side] = None
                        continue
                    prefix = before[:CRASH_KEEP_LENGTH]
                    fixed_prefix = (before[0] == after[0]
                                    and set(after) == set(prefix)
                                    and len(after) == len(prefix))
                    old_score, score = previous.scores[side], state.scores[side]
                    # This side had exactly one protected own turn. A harmless
                    # stationary move or a ram never becomes first-crash evidence.
                    if (previous.invincibility(side) > 0
                            and state.invincibility(side) in (previous.invincibility(side) - 1, 50)):
                        continue
                    pending = self.pending_moves.get(game_id)
                    own_shot = (pending and len(pending) > 2 and pending[2] == 'shoot'
                                and side == previous.side)
                    shot_geometry = (before[0] == after[0]
                                     and set(after).issubset(before)
                                     and len(after) <= len(before))
                    if own_shot and shot_geometry and score == old_score - 5000:
                        # A stationary shot at exactly 5000 also produces zero
                        # score, but does not provide crash evidence.
                        continue
                    if (side != previous.side and old_score >= 5000
                            and score == old_score - 5000 and shot_geometry):
                        # The rival action was not observed. In particular,
                        # shooting at 5000 and a first crash can both end at 0.
                        if history[side] is False:
                            history[side] = None
                        continue
                    crash_scores = ({0, old_score - CRASH_PENALTY} if old_score > 0
                                    else {old_score - CRASH_PENALTY})
                    if fixed_prefix and score in crash_scores:
                        history[side] = True
                        snakes = dict(state.snakes)
                        snakes[side] = prefix
                        state = replace(state, snakes=snakes,
                                        reliable_tails=state.reliable_tails | {side})
                    elif history[side] is False:
                        normal = (manhattan(before[0], after[0]) == 1
                                  and ((len(after) == len(before)
                                        and set(after) == set((after[0],) + before[:-1]))
                                       or (len(after) == len(before) + 1
                                           and set(after) == set((after[0],) + before))))
                        pending = self.pending_moves.get(game_id)
                        if pending and side == previous.side:
                            if len(pending) > 2 and pending[2] == 'shoot':
                                normal = (after == before and score == old_score - 5000)
                            else:
                                normal = normal and after[0] == step(before[0], pending[1])
                        if not normal:
                            history[side] = None
            if delta > 2:
                for side in ('A', 'B'):
                    # A shot can shorten a body without any crash. Legacy
                    # matches with no affordable shot retain their v7 proof.
                    enemy = 'B' if side == 'A' else 'A'
                    shot_possible = (previous.scores[enemy] >= 5000
                                     or state.scores[enemy] >= 5000
                                     or previous.invincibility(enemy) > 0 or state.invincibility(enemy) > 0
                                     or bool(previous.stars) or bool(state.stars)
                                     or ('invincible_1' in data and state.remaining_moves <= MAX_MOVES - 40))
                    if (not shot_possible and len(state.body(side)) < len(previous.body(side))
                            and state.body(side)):
                        history[side] = True
            if not pending or len(pending) < 3 or pending[2] != 'shoot' or (
                    contiguous and delta == 2
                    and state.head(previous.side) == previous.head(previous.side)
                    and state.scores[previous.side] == previous.scores[previous.side] - SHOOT_COST
                    and len(state.body(previous.side)) <= len(previous.body(previous.side))
                    and set(state.body(previous.side)) == set(previous.body(previous.side)[:len(state.body(previous.side))])):
                self.pending_moves.pop(game_id, None)
        if state.tail_food_b:
            history['A'] = True
        if state.tail_food_a:
            history['B'] = True
        state = self._observed_star_timing(previous, state)
        state = replace(state, crashed_a=history['A'], crashed_b=history['B'])
        self.crash_history[game_id] = history
        self.observed_states[game_id] = state
        self.observation_tokens[game_id] = token
        self.current_states[game_id] = state
        self.previous_snakes[game_id] = state.snakes
        return state

    @staticmethod
    def _observed_star_timing(before, after):
        """Retain only a provable normal spawn deadline; never predict a cell."""
        if after.star_frenzy:
            return replace(after, pending_star_spawns=max(0, 10 - len(after.stars)))
        if after.stars or after.invincible_a or after.invincible_b:
            return replace(after, star_spawn_due_round=None, pending_star_spawns=0)
        if before is None:
            return after
        gap = before.remaining_moves - after.remaining_moves
        due = before.star_spawn_due_round
        if before.invincible_a or before.invincible_b:
            expirations = [(1 if actor == before.side else 2) + 2 * (before.invincibility(actor) - 1)
                           for actor in ('A', 'B') if before.invincibility(actor)]
            due = ((MAX_MOVES - before.remaining_moves + max(expirations)) / 2 + 20
                   if 0 < gap <= 2 and max(expirations) <= gap else None)
        elif before.stars:
            due = None  # The star may have been consumed in an unobserved interval.
        return replace(after, star_spawn_due_round=due,
                       pending_star_spawns=int(due is not None
                           and (MAX_MOVES - after.remaining_moves) / 2 >= due))

    @staticmethod
    def _single_action_history(before, after, data, history):
        """Infer history only from a coherent one-action server snapshot."""
        actor, passive = before.side, before.enemy
        result = {side: True if history[side] is True else None for side in ('A', 'B')}
        body = after.body(passive)
        old = before.body(passive)
        if (body and old and body[0] == old[0] and len(body) <= len(old)
                and set(body) == set(old[:len(body)]) and after.scores[passive] == before.scores[passive]):
            result[passive] = history[passive]
        if (before.invincibility(actor) > 0
                and after.invincibility(actor) in (before.invincibility(actor) - 1, 50)):
            result[actor] = history[actor]
        action = data.get('action', data.get('action_type'))
        direction = data.get('direction')
        if action is None and before.head(actor) != after.head(actor):
            action = 'move'
            direction = direction_between(before.head(actor), after.head(actor))
        if (action not in ('move', 'shoot') or direction not in DIRS
                or data.get('action_type', action) != action):
            return result, None
        try:
            expected = apply_action(before, action, direction, actor)
        except ValueError:
            return result, None
        matches = (expected.scores == after.scores and expected.side == after.side
                   and expected.shot_food == after.shot_food
                   and expected.invincible_a == after.invincible_a
                   and expected.invincible_b == after.invincible_b
                   and all(expected.head(side) == after.head(side)
                           and set(expected.body(side)) == set(after.body(side))
                           and len(expected.body(side)) == len(after.body(side)) for side in ('A', 'B')))
        if not matches:
            return result, None
        for side in ('A', 'B'):
            flag = expected.crashed_a if side == 'A' else expected.crashed_b
            result[side] = True if history[side] is True else flag
        return result, (action, direction)

    def choose_action(self, data):
        with planning_module.planning_scope():
            return self._choose_action(data)

    def _choose_action(self, data):
        """Compare actions using the same completed tree as movement ranking."""
        game_id = str(data.get('game_id', 'default'))
        self.action_searches.pop(game_id, None)
        direction = self.choose_move(data)
        if direction is None:
            return None
        fallback = {'action_type': 'move', 'direction': direction}
        state = self.current_states.get(game_id)
        cached = self.action_searches.pop(game_id, None)
        if (state is not None and cached is None and not legal_moves(state, state.side)
                and not state.is_terminal() and state.scores[state.side] >= SHOOT_COST
                and shoot_directions(state.head(state.side), state.rows, state.cols,
                                     state.walls, state.body(state.enemy))):
            # The legacy trapped-head path returns before ranking. This is its
            # only tree: compare official forced crashes against useful shots.
            weights = get_weights()
            engine, compact = IterativeSearchEngine.from_game_state(
                state, time_budget_ms=self._search_time_budget(len(self.previous_snakes)),
                max_depth=weights['iterative_search_max_depth'])
            cached = (state, engine.search(compact, state.side, available_moves(state, state.side)))
        if state is None or cached is None or cached[0] is not state:
            return fallback
        search = cached[1]
        debug = self.last_decisions.setdefault(game_id, {'event': 'decision_debug'})
        diagnostic = {'completed_depth': search.completed_depth, 'nodes': search.nodes,
                      'elapsed_ms': search.elapsed_ms,
                      'scores': [{'action_type': key[0] if isinstance(key, tuple) else 'move',
                                  'direction': key[1] if isinstance(key, tuple) else key,
                                  'score': value} for key, value in search.scores.items()],
                      'shoot_cost': SHOOT_COST, 'source': 'shared_root_search',
                      'fallback_direction': direction}
        debug['action_search'] = diagnostic
        # A completed iteration already compares movement economics. Spending
        # 5000 requires the existing, more conservative depth gate below.
        if search.completed_depth == 0:
            diagnostic['selection'] = 'insufficient_completed_depth'
            return fallback
        comparable_moves = available_moves(state, state.side)
        moves = {key: value for key, value in search.scores.items()
                 if isinstance(key, str) and key in comparable_moves}
        if not moves:
            diagnostic['selection'] = 'no_comparable_moves'
            return fallback
        best_move = max(moves, key=moves.get)
        shots = {}
        for key, value in search.scores.items():
            if (search.completed_depth >= get_weights()['iterative_search_min_apply_depth']
                    and isinstance(key, tuple) and len(key) == 2 and key[0] == 'shoot'):
                outcome = self._effective_shot(state, key[1])
                if outcome is not None:
                    shots[key[1]] = (value, outcome.last_action)
        if shots:
            shot = max(shots, key=lambda key: shots[key][0])
            if shots[shot][0] > moves[best_move]:
                diagnostic.update(selection='profitable_hit', selected_score=shots[shot][0],
                                  best_move_score=moves[best_move], **shots[shot][1])
                debug.update(action_type='shoot', direction=shot, reason='shared_search_profitable_shoot')
                self.targets[(game_id, state.side)] = None
                return {'action_type': 'shoot', 'direction': shot}
        # Raw search can repair a concrete new tactical exposure without
        # replacing established v1-v7 planning in ordinary positions.
        risk = self._action_exposure(state, direction, moves)
        if (risk and direction in moves and moves[best_move] > moves[direction]):
            diagnostic.update(selection=risk, selected_score=moves[best_move],
                              fallback_score=moves[direction])
            debug.update(action_type='move', direction=best_move, reason='shared_search_' + risk)
            return {'action_type': 'move', 'direction': best_move}
        diagnostic['selection'] = 'conservative_move'
        return fallback

    @staticmethod
    def _effective_shot(state, direction):
        if (direction not in DIRS or state.is_terminal()
                or state.scores[state.side] < SHOOT_COST):
            return None
        outcome = apply_action(state, 'shoot', direction, state.side)
        return outcome if outcome.last_action['hit'] and outcome.last_action['cut_length'] > 0 else None

    @staticmethod
    def _action_exposure(state, direction, moves):
        if direction not in moves:
            return None
        # Geometry-only diagnostics share the planner's conservative handling
        # of unknown v7 crash history; this successor is never persisted.
        successor = simulated_move(state, direction, state.side, perspective=state.side)
        if successor.last_action.get('bomb_triggered'):
            return 'bomb_economics'
        enemy = state.enemy
        if state.scores[enemy] < SHOOT_COST or successor.invincibility(state.side):
            return None
        target = successor.body(state.side)
        if (len(target) >= 2 and successor.head(enemy) is not None
                and shoot_directions(successor.head(enemy), successor.rows, successor.cols,
                                     successor.walls, target)):
            return 'opponent_shoot_reply'
        return None

    def safe_action(self, data, action):
        """Never allow an unaffordable or malformed shot onto the wire."""
        action_type = action.get('action_type', 'move')
        direction = action.get('direction')
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
        if state is None:
            return None
        if action_type == 'shoot' and self._effective_shot(state, direction) is not None:
            return {'action_type': 'shoot', 'direction': direction}
        return {'action_type': 'move', 'direction': self.safe_direction(data, direction)}

    def choose_move(self, data):
        game_id = str(data.get('game_id', 'default'))
        state = self.observe_state(data)
        if state is None:
            return None
        self.current_states[game_id] = state
        side = state.side
        self._update_food_clock(game_id, side, state)

        legal = legal_moves(state, side)
        debug = self._base_debug(data, state, legal)
        if not legal:
            move = next(iter(available_moves(state, side)), 'up')
            self._save_debug(game_id, debug, 'no_safe_moves', move)
            return move

        kill = forced_kill_move(state, side, legal)
        if kill:
            self._save_debug(game_id, debug, 'forced_kill', kill)
            self._remember(game_id, state)
            return kill

        clean_legal = [
            direction for direction in legal
            if (state.invincibility(side) or step(state.head(side), direction) not in state.wrong_food())
            and (state.invincibility(side) or step(state.head(side), direction) not in state.walls)
        ]
        if clean_legal:
            avoided = set(legal) - set(clean_legal)
            wrong_avoided = [
                direction for direction in avoided
                if step(state.head(side), direction) in state.wrong_food()
            ]
            wall_avoided = [
                direction for direction in avoided
                if step(state.head(side), direction) in state.walls
            ]
            if wrong_avoided:
                debug['avoided_wrong_digit_moves'] = sorted(wrong_avoided)
            if wall_avoided:
                debug['avoided_wall_moves'] = sorted(wall_avoided)
            legal = clean_legal
            debug['legal'] = legal

        current_target = self.targets.get((game_id, side))
        hunger = self.turns_since_food[(game_id, side)]
        repeat_count = self._repeat_count(game_id, state)
        search_cache = TurnSearchCache()
        active_games = len(self.previous_snakes) + (game_id not in self.previous_snakes)
        search_time_ms = self._search_time_budget(active_games)
        food_plans = build_food_plans(state, side, current_target, hunger)
        plan = choose_food_plan(state, side, current_target, hunger, plans=food_plans)
        deep_moves = rank_deep_moves(
            state,
            side,
            legal,
            plan,
            hunger,
            repeat_count,
            search_cache,
            search_time_ms=search_time_ms,
        )
        action_search = getattr(search_cache, 'action_search', None)
        if action_search is not None:
            self.action_searches[game_id] = (state, action_search)
        debug.update(
            {
                'current_target': current_target,
                'hunger': hunger,
                'repeat_count': repeat_count,
                'active_games': active_games,
                'search_time_budget_ms': search_time_ms,
                'plan': self._plan_debug(plan),
                'commit_to_plan': should_commit_to_food(plan, hunger, repeat_count) if plan else False,
                'deep_moves': deep_moves[:4],
            }
        )
        if self.enable_debug:
            debug.update(
                {
                    'food_options': self._food_options(food_plans),
                    'candidate_moves': self._candidate_moves(state, side, legal, deep_moves),
                }
            )

        commit_to_plan = should_commit_to_food(plan, hunger, repeat_count) if plan else False
        debug['commit_to_plan'] = commit_to_plan

        reference_move = plan.first_move if plan and commit_to_plan and plan.first_move in legal else None
        advantage = advantage_control_move(
            state,
            side,
            legal,
            deep_moves,
            reference_move=reference_move,
            adjacent_food=bool(plan and plan.our_distance == 1),
        )
        debug['advantage_control'] = advantage
        if advantage['active'] and advantage['override'] and advantage['direction'] in legal:
            move = advantage['direction']
            self.targets[(game_id, side)] = None
            self._save_debug(game_id, debug, 'advantage_control', move)
            self._remember(game_id, state)
            return move

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

    @staticmethod
    def _search_time_budget(active_games):
        full_budget = get_weights()['iterative_search_time_ms']
        if active_games >= 16:
            return min(full_budget, 10)
        if active_games >= 8:
            return min(full_budget, 18)
        if active_games >= 4:
            return min(full_budget, 30)
        return full_budget

    def safe_direction(self, data, direction):
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
            self.current_states[game_id] = state
        legal = legal_moves(state, state.side)
        if direction in legal:
            return direction
        if legal:
            safe = fallback_food_positioning(state, state.side, legal, None, 0, 0)
            self._mark_safe_repair(game_id, direction, safe)
            return safe
        fallback = next(iter(available_moves(state, state.side)), 'up')
        self._mark_safe_repair(game_id, direction, fallback)
        return fallback

    def commit_move(self, data, direction):
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
        if direction in DIRS and not state.is_terminal():
            self.pending_moves[game_id] = (state, direction)
            try:
                predicted = apply_move(state, direction, state.side)
                snakes = predicted.snakes
            except ValueError:
                unknown = state.crashed_a if state.side == 'A' else state.crashed_b
                if unknown is not None or state.scores[state.side] <= 0:
                    raise
                # Unknown positive-score crash: predict geometry, never economics.
                snakes = dict(state.snakes)
                snakes[state.side] = state.body(state.side)[:CRASH_KEEP_LENGTH]
            self.previous_snakes[game_id] = snakes
        self.current_states.pop(game_id, None)

    def commit_action(self, data, action):
        if action.get('action_type', 'move') == 'move':
            return self.commit_move(data, action.get('direction'))
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
        direction = action.get('direction')
        if (state is not None and direction in DIRS and not state.is_terminal()
                and state.scores[state.side] >= 5000):
            predicted = apply_action(state, 'shoot', direction, state.side)
            self.pending_moves[game_id] = (state, direction, 'shoot')
            self.previous_snakes[game_id] = predicted.snakes
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
        self.crash_history.pop(str(game_id), None)
        self.observed_states.pop(str(game_id), None)
        self.pending_moves.pop(str(game_id), None)
        self.observation_tokens.pop(str(game_id), None)
        self.action_searches.pop(str(game_id), None)

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
            tuple(sorted(state.scores.items())),
            state.crashed_a, state.crashed_b,
            tuple(sorted(state.tail_food_a)), tuple(sorted(state.tail_food_b)),
            tuple(sorted(state.bombs_large)), tuple(sorted(state.bombs_small)),
            tuple(sorted(state.shot_food)), state.pending_bomb_respawns,
            tuple(sorted(state.stars)), state.invincible_a, state.invincible_b,
            state.star_frenzy, state.star_spawn_due_round, state.pending_star_spawns,
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
            tuple(sorted(state.scores.items())),
            state.crashed_a, state.crashed_b,
            tuple(sorted(state.tail_food_a)), tuple(sorted(state.tail_food_b)),
            tuple(sorted(state.bombs_large)), tuple(sorted(state.bombs_small)),
            tuple(sorted(state.shot_food)), state.pending_bomb_respawns,
            tuple(sorted(state.stars)), state.invincible_a, state.invincible_b,
            state.star_frenzy, state.star_spawn_due_round, state.pending_star_spawns,
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
            'crash_history': {'A': state.crashed_a, 'B': state.crashed_b},
            'tail_food_a': sorted(state.tail_food_a),
            'tail_food_b': sorted(state.tail_food_b),
            'bombs_large': sorted(state.bombs_large),
            'bombs_small': sorted(state.bombs_small),
            'shot_food': sorted(state.shot_food),
            'stars': sorted(state.stars),
            'invincible_self': state.invincibility(side),
            'invincible_rival': state.invincibility(state.enemy),
            'star_frenzy': state.star_frenzy,
            'legal': legal,
            'scores': state.scores,
            'remaining_moves': state.remaining_moves,
        }

    def _save_debug(self, game_id, debug, reason, direction):
        debug['reason'] = reason
        debug['action_type'] = 'move'
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
            'bomb_loss': getattr(plan, 'bomb_loss', 0),
            'bomb_trigger_count': getattr(plan, 'bomb_trigger_count', 0),
            'sequence_rank': plan.sequence_rank,
            'projected_income': plan.projected_income,
            'rival_projected_income': plan.rival_projected_income,
            'economic_swing': plan.economic_swing,
            'multiplier_marginal': plan.multiplier_marginal,
            'value': plan.value,
        }

    def _food_options(self, plans):
        options = [self._plan_debug(plan) for plan in plans]
        options.sort(key=lambda item: item['value'], reverse=True)
        return options

    def _candidate_moves(self, state, side, legal, deep_moves):
        deep_by_direction = {item['direction']: item for item in deep_moves}
        candidates = []
        for direction in legal:
            after = simulated_move(state, direction, side)
            head = after.head(side)
            analysis = deep_by_direction.get(direction, {})
            nearest_food = (
                min((manhattan(head, food) for food in after.objective_cells()), default=None)
                if head is not None
                else None
            )
            candidates.append(
                {
                    'direction': direction,
                    'safety': analysis.get('safety'),
                    'space': analysis.get('region', 0),
                    'exits': analysis.get('exits', 0),
                    'tail_reachable': analysis.get('tail_reachable', False),
                    'region': analysis.get('region', 0),
                    'count_exits': analysis.get('exits', 0),
                    'nearest_food_distance': nearest_food,
                    'center_control_score': center_control_score(after, head),
                    'position_control_penalty': position_control_penalty(state, side, direction),
                    'rival_penalty': None,
                    'hot_lost_food_penalty': None,
                    'wrong_food_penalty': step(state.head(side), direction) in state.wrong_food(),
                    'wall_hit': step(state.head(side), direction) in state.walls,
                    'rival_stats': {
                        'enemy_reply': analysis.get('enemy_reply'),
                        'our_replies': analysis.get('our_replies'),
                        'forced_loss': analysis.get('forced_loss', False),
                    },
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
    return next(iter(available_moves(state, state.side)), 'up')
