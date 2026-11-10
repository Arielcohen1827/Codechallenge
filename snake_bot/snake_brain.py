"""Coordinate observation, bounded planning and action commitment for each match."""

from collections import Counter, deque
from dataclasses import dataclass, replace
from snake_bot import food_planner as planning_module

from snake_bot.bot_weights import get_weights
from snake_bot.game_rules import (
    CRASH_KEEP_LENGTH,
    CRASH_PENALTY,
    MAX_MOVES,
    SHOOT_COST,
    shoot_directions,
)
from snake_bot.food_planner import (
    advantage_control_move,
    apply_move as simulated_move,
    build_food_plans,
    choose_food_plan,
    center_control_score,
    fallback_food_positioning,
    forced_kill_move,
    hot_lost_food_penalty,
    position_control_penalty,
    rival_response_penalty,
    rival_response_stats,
    should_commit_to_food,
    should_use_plan_for_positioning,
    sequence_gate_hold_move,
)
from snake_bot.move_search import TurnSearchCache, rank_deep_moves
from snake_bot.search_engine import IterativeSearchEngine
from snake_bot.snake_state import (
    FOOD_SCORE,
    apply_action,
    apply_move,
    available_moves,
    direction_between,
    legal_moves,
    manhattan,
    parse_state,
    step,
)
from snake_bot.geometry import directions_for


@dataclass
class MoveOptions:
    game_id: str
    state: object
    legal: list
    debug: dict
    plan: object
    ranked: list
    cache: TurnSearchCache
    hunger: int
    repeats: int
    commit: bool


@dataclass
class FoodMovePressure:
    stats: dict
    reply_penalty: int
    position_penalty: int
    lost_food_penalty: int
    hard: bool
    soft: bool


def directions_in_cells(targets, cells):
    return {direction for direction, cell in targets.items() if cell in cells}


def record_avoided_moves(debug, wrong, walls):
    if wrong:
        debug['avoided_wrong_digit_moves'] = sorted(wrong)
    if walls:
        debug['avoided_wall_moves'] = sorted(walls)


def valid_advantage(advantage, legal):
    if advantage['active'] and advantage['override'] and advantage['direction'] in legal:
        return advantage['direction']
    return None


def ranked_move(ranked, direction):
    return next((entry for entry in ranked if entry['direction'] == direction), None)


def adjacent_trap(plan, current, alternative, move):
    if plan.our_distance != 1 or plan.endgame_acceptable:
        return False
    if not trapped_entry(current):
        return False
    return viable_escape(alternative, move)


def trapped_entry(entry):
    return entry is not None and (
        entry.get('our_replies', 0) == 0 or entry.get('safety') == 'SUICIDAL'
    )


def viable_escape(entry, current):
    return (
        entry is not None
        and entry['direction'] != current
        and entry.get('safety') != 'SUICIDAL'
        and entry.get('our_replies', 0) > 0
    )


def review_food_move(options, move, best):
    pressure = food_move_pressure(options, move)
    weights = get_weights()
    move, reason = reply_pressure_move(options, move, pressure)
    override = deep_override_allowed(options, pressure, weights)
    arbitrate = should_arbitrate_food_move(options, pressure, weights)
    options.debug['tactical_arbitration'] = arbitrate
    options.debug['deep_override_allowed'] = override
    if override and deep_override_improves(options, move, best, weights):
        move, reason = best['direction'], 'deep_control'
    if arbitrate:
        move, reason = tactical_food_move(options, move, reason, pressure, weights)
    return avoid_forced_loss(options, move, reason)


def food_move_pressure(options, move):
    state, side = options.state, options.state.side
    stats = rival_response_stats(state, side, move)
    reply_penalty = rival_response_penalty(state, side, move)
    position_penalty = position_control_penalty(state, side, move)
    lost_food_penalty = hot_lost_food_penalty(state, side, move)
    options.debug.update(
        plan_move_stats=stats,
        plan_move_penalty=reply_penalty,
        plan_position_penalty=position_penalty,
        plan_hot_lost_food_penalty=lost_food_penalty,
    )
    hard = hard_reply_pressure(stats, options.plan.our_distance)
    soft = soft_reply_pressure(stats, reply_penalty)
    return FoodMovePressure(stats, reply_penalty, position_penalty, lost_food_penalty, hard, soft)


def hard_reply_pressure(stats, distance):
    forced = stats['forced_zero'] or stats['forced_delayed_zero']
    if distance > 1:
        return forced or bool(stats['zero_escape_replies'])
    return forced


def soft_reply_pressure(stats, reply_penalty):
    return (
        stats['one_escape_replies'] >= 2
        or stats['delayed_forced_zero_replies']
        or reply_penalty <= -6000
    )


def reply_pressure_move(options, move, pressure):
    if not (pressure.hard or (options.plan.our_distance > 1 and pressure.soft)):
        return move, 'food_plan'
    state, side = options.state, options.state.side
    fallback = fallback_food_positioning(
        state, side, options.legal, options.plan, options.hunger, options.repeats
    )
    reason = 'food_plan'
    if rival_response_penalty(state, side, fallback) > pressure.reply_penalty:
        move, reason = fallback, 'fallback_rival_pressure'
    return least_trappable_move(options, move, reason)


def least_trappable_move(options, move, reason):
    state, side = options.state, options.state.side
    best = max(options.legal, key=lambda direction: rival_response_penalty(state, side, direction))
    improvement = rival_response_penalty(state, side, best) - rival_response_penalty(
        state, side, move
    )
    if improvement >= 5000:
        return best, 'least_trappable'
    return move, reason


def weak_food_position(options):
    plan = options.plan
    return (
        plan.safety != 'SAFE'
        or plan.exits_after <= 1
        or plan.space_after <= 80
        or options.repeats >= 1
    )


def threatened_food_plan(options, pressure):
    return (
        pressure.hard
        or pressure.soft
        or weak_food_position(options)
        or pressure.lost_food_penalty > 0
    )


def deep_override_allowed(options, pressure, weights):
    return (
        threatened_food_plan(options, pressure)
        or pressure.position_penalty >= weights['position_edge1_penalty']
    )


def should_arbitrate_food_move(options, pressure, weights):
    if options.plan.our_distance <= 1:
        return False
    return (
        threatened_food_plan(options, pressure)
        or options.plan.race_margin <= 1
        or pressure.position_penalty >= weights['position_edge2_penalty']
    )


def deep_override_improves(options, move, best, weights):
    if options.plan.our_distance <= 1:
        return False
    if best is None or best['direction'] == move:
        return False
    current = ranked_move(options.ranked, move)
    return (
        current is not None
        and best['score'] >= current['score'] + weights['deep_plan_override_margin']
    )


def tactical_food_move(options, move, reason, pressure, weights):
    state, plan = options.state, options.plan

    def quality(direction):
        return options.cache.move_quality(
            state, state.side, direction, plan, options.hunger, options.repeats
        )

    best = max(options.legal, key=quality)
    plan_quality, best_quality = quality(move), quality(best)
    options.debug.update(
        plan_move_quality=plan_quality,
        best_tactical_move=best,
        best_tactical_quality=best_quality,
        best_tactical_stats=rival_response_stats(state, state.side, best),
    )
    margin = tactical_override_margin(pressure, weights)
    if best != move and best_quality >= plan_quality + margin:
        return best, 'tactical_override'
    return move, reason


def tactical_override_margin(pressure, weights):
    stats = pressure.stats
    if stats['one_escape_replies'] or stats['delayed_forced_zero_replies']:
        return weights['one_escape_tactical_override_margin']
    if pressure.position_penalty >= weights['position_edge1_penalty']:
        return weights['edge_tactical_override_margin']
    return weights['plan_tactical_override_margin']


def avoid_forced_loss(options, move, reason):
    selected = ranked_move(options.ranked, move)
    alternative = first_survivable_entry(options.ranked)
    if selected and selected.get('forced_loss') and alternative:
        options.debug['forced_loss_avoided'] = selected['direction']
        return alternative['direction'], 'forced_loss_avoidance'
    return move, reason


def first_survivable_entry(ranked):
    return next((entry for entry in ranked if survivable_entry(entry)), None)


def survivable_entry(entry):
    return not entry.get('forced_loss') and entry.get('safety') != 'SUICIDAL'


def positioning_reason(plan):
    return 'deep_control' if plan else 'survival_control'


def observed_star_deadline(before, remaining_moves):
    if max(before.invincible_a, before.invincible_b) > 0:
        return deadline_after_protection(before, remaining_moves)
    if before.stars:
        return None
    return before.star_spawn_due_round


def deadline_after_protection(before, remaining_moves):
    gap = before.remaining_moves - remaining_moves
    expiration = max(
        own_expiration_turn(before, side) for side in ('A', 'B') if before.invincibility(side)
    )
    if 0 < gap <= 2 and expiration <= gap:
        return (MAX_MOVES - before.remaining_moves + expiration) / 2 + 20
    return None


def own_expiration_turn(state, side):
    first_turn = 1 if side == state.side else 2
    return first_turn + 2 * (state.invincibility(side) - 1)


def star_is_due(remaining_moves, deadline):
    return deadline is not None and (MAX_MOVES - remaining_moves) / 2 >= deadline


def passive_player_matches(before, after):
    side = before.enemy
    if after.scores[side] != before.scores[side]:
        return False
    return stationary_body_prefix(before.body(side), after.body(side))


def stationary_body_prefix(before, after):
    return bool(before and after) and observed_body_prefix(before, after)


def observed_body_prefix(before, after):
    return (
        before[0] == after[0]
        and len(after) <= len(before)
        and set(after) == set(before[: len(after)])
    )


def observed_action(before, after, data):
    action = data.get('action', data.get('action_type'))
    direction = data.get('direction')
    first, second = before.head(before.side), after.head(before.side)
    if action is None and first != second:
        action = 'move'
        direction = direction_between(first, second)
    if valid_observed_action(action, direction, data):
        return action, direction
    return None


def valid_observed_action(action, direction, data):
    return (
        action in ('move', 'shoot')
        and direction in directions_for(tuple(data.get('head', {}).values()) or (0, 0))
        and data.get('action_type', action) == action
    )


def simulate_observed_action(before, action):
    try:
        return apply_action(before, *action, before.side)
    except ValueError:
        return None


def matching_action_result(expected, after):
    if expected is None:
        return False
    if action_result_fields(expected) != action_result_fields(after):
        return False
    return all(same_observed_body(expected, after, side) for side in ('A', 'B'))


def action_result_fields(state):
    return (state.scores, state.side, state.shot_food, state.invincible_a, state.invincible_b)


def same_observed_body(expected, after, side):
    expected_body, actual_body = expected.body(side), after.body(side)
    return (
        expected.head(side) == after.head(side)
        and len(expected_body) == len(actual_body)
        and set(expected_body) == set(actual_body)
    )


def verified_crash_history(expected, history):
    confirmed = {'A': expected.crashed_a, 'B': expected.crashed_b}
    return {side: True if history[side] is True else flag for side, flag in confirmed.items()}


def trapped_head_has_shot(state):
    if legal_moves(state, state.side) or not can_shoot(state):
        return False
    return bool(
        shoot_directions(
            state.head(state.side),
            state.rows,
            state.cols,
            state.walls,
            state.body(state.enemy),
        )
    )


def can_shoot(state):
    return not state.is_terminal() and state.scores[state.side] >= SHOOT_COST


def is_shot_key(key):
    return isinstance(key, tuple) and len(key) == 2 and key[0] == 'shoot'


def action_search_diagnostic(search, direction):
    return {
        'completed_depth': search.completed_depth,
        'nodes': search.nodes,
        'elapsed_ms': search.elapsed_ms,
        'scores': [action_score_entry(key, value) for key, value in search.scores.items()],
        'shoot_cost': SHOOT_COST,
        'source': 'shared_root_search',
        'fallback_direction': direction,
    }


def action_score_entry(key, value):
    if isinstance(key, tuple):
        return {'action_type': key[0], 'direction': key[1], 'score': value}
    return {'action_type': 'move', 'direction': key, 'score': value}


def comparable_move_scores(state, search):
    moves = available_moves(state, state.side)
    return {
        key: value for key, value in search.scores.items() if isinstance(key, str) and key in moves
    }


def move_destinations(state, legal):
    head = state.head(state.side)
    return {move: step(head, move) for move in legal}


def opponent_has_shot(state, successor):
    enemy = state.enemy
    if state.scores[enemy] < SHOOT_COST or successor.invincibility(state.side):
        return False
    return bool(
        shoot_directions(
            successor.head(enemy),
            successor.rows,
            successor.cols,
            successor.walls,
            successor.body(state.side),
        )
    )


def can_commit_shot(state, direction):
    return state is not None and direction in directions_for(state) and can_shoot(state)


def predicted_move_bodies(state, direction):
    try:
        return apply_move(state, direction, state.side).snakes
    except ValueError:
        prior = state.crashed_a if state.side == 'A' else state.crashed_b
        if prior is not None or state.scores[state.side] <= 0:
            raise
        # Unknown positive-score crash: predict geometry, never economics.
        snakes = dict(state.snakes)
        snakes[state.side] = state.body(state.side)[:CRASH_KEEP_LENGTH]
        return snakes


class SnakeBrain:
    """Reconcile each match and choose actions from current observations and memory."""

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
        self.knowledge_states = {}

    def observe_state(self, data):
        """Reconcile server snapshots without confirming predicted actions."""
        game_id = str(data.get('game_id', 'default'))
        previous = self.observed_states.get(game_id)
        history = dict(self.crash_history.get(game_id, {'A': None, 'B': None}))
        state = parse_state(
            data,
            self.previous_snakes.get(game_id),
            history,
            knowledge=self.knowledge_states.get(game_id),
        )
        if self._stale_snapshot(previous, state, data):
            return None
        token = (data.get('turn_token'), state.remaining_moves)
        state, history = self._reconcile_snapshot(game_id, previous, state, data, history)
        if state is previous:
            self.current_states[game_id] = previous
            self.observation_tokens[game_id] = token
            return previous
        return self._store_observation(game_id, previous, state, history, token)

    @staticmethod
    def _stale_snapshot(previous, state, data):
        return (
            previous is not None
            and data.get('remaining_moves') is not None
            and state.remaining_moves > previous.remaining_moves
        )

    def _reconcile_snapshot(self, game_id, previous, state, data, history):
        if state.partial:
            return self._partial_snapshot(game_id, previous, state, data, history)
        return self._complete_snapshot(game_id, previous, state, data, history)

    def _complete_snapshot(self, game_id, previous, state, data, history):
        pending = self.pending_moves.get(game_id)
        reference = self._action_reference(previous, state, data, pending)
        if self._one_observed_turn(reference, state):
            history, action = self._single_action_history(reference, state, data, history)
            if previous is None:
                return state, self._initial_history(state, data, history)
            self._confirm_pending_action(game_id, pending, reference, action)
            return state, history
        if previous is None:
            return state, self._initial_history(state, data, history)
        if previous.remaining_moves == state.remaining_moves:
            return self._same_turn_snapshot(previous, state, data, history)
        return self._reconcile_gap(game_id, previous, state, data, history)

    def _partial_snapshot(self, game_id, previous, state, data, history):
        """Occlusion cannot establish a cut, a crash, or an opponent action."""
        if previous is None:
            return state, self._initial_history(state, data, history)
        if previous.remaining_moves != state.remaining_moves:
            self.pending_moves.pop(game_id, None)
        return state, history

    @staticmethod
    def _pending_kind(pending):
        return pending[2] if len(pending) > 2 else 'move'

    def _action_reference(self, previous, state, data, pending):
        if not pending:
            return previous
        action = data.get('action', data.get('action_type'))
        if pending[
            0
        ].remaining_moves - state.remaining_moves == 1 and action == self._pending_kind(pending):
            return pending[0]
        return previous

    @staticmethod
    def _one_observed_turn(reference, state):
        return (
            reference is not None
            and reference.remaining_moves - state.remaining_moves == 1
            and state.side != reference.side
        )

    def _confirm_pending_action(self, game_id, pending, reference, action):
        if (
            pending
            and reference is pending[0]
            and action == (self._pending_kind(pending), pending[1])
        ):
            self.pending_moves.pop(game_id, None)

    @staticmethod
    def _initial_history(state, data, history):
        if data.get('remaining_moves') is None:
            return history
        opening = (state.remaining_moves, state.side)
        if opening == (MAX_MOVES, 'A'):
            return {'A': False, 'B': False}
        if opening == (MAX_MOVES - 1, 'B'):
            history['B'] = False
        return history

    @staticmethod
    def _snapshot_identity(state):
        return (
            state.board,
            state.observation,
            state.scores,
            state.multipliers,
            state.side,
            state.invincible_a,
            state.invincible_b,
            state.timeouts_a,
            state.timeouts_b,
        )

    @staticmethod
    def _player_snapshot(state, side):
        return state.body(side), state.scores[side], state.multipliers[side]

    def _same_turn_snapshot(self, previous, state, data, history):
        if self._snapshot_identity(previous) == self._snapshot_identity(state):
            return previous, history
        state = parse_state(data, previous.snakes, history)
        for side in ('A', 'B'):
            if history[side] is False and self._player_snapshot(
                state, side
            ) != self._player_snapshot(previous, side):
                history[side] = None
        return state, history

    def _reconcile_gap(self, game_id, previous, state, data, history):
        delta = previous.remaining_moves - state.remaining_moves
        contiguous = delta == 2 and previous.side == state.side
        if contiguous:
            for side in ('A', 'B'):
                state = self._reconcile_player(game_id, previous, state, side, history)
        else:
            history = self._unconfirmed_history(history)
        if delta > 2:
            self._infer_gap_crashes(previous, state, data, history)
        self._clear_gap_prediction(game_id, previous, state, contiguous)
        return state, history

    @staticmethod
    def _unconfirmed_history(history):
        return {side: True if value is True else None for side, value in history.items()}

    def _reconcile_player(self, game_id, previous, state, side, history):
        before, after = previous.body(side), state.body(side)
        if not before or not after:
            self._forget_unconfirmed_crash(history, side)
            return state
        if self._protected_turn(previous, state, side):
            return state
        return self._reconcile_unprotected_player(
            self._pending_for_player(game_id, previous, side),
            previous,
            state,
            side,
            history,
        )

    def _pending_for_player(self, game_id, previous, side):
        if side == previous.side:
            return self.pending_moves.get(game_id)
        return None

    @staticmethod
    def _forget_unconfirmed_crash(history, side):
        if history[side] is False:
            history[side] = None

    @staticmethod
    def _protected_turn(previous, state, side):
        remaining = previous.invincibility(side)
        return remaining > 0 and state.invincibility(side) in (remaining - 1, 50)

    def _reconcile_unprotected_player(self, own_action, previous, state, side, history):
        before, after = previous.body(side), state.body(side)
        old_score, score = previous.scores[side], state.scores[side]
        if self._could_be_stationary_shot(
            before, after, old_score, score, own_action, side == previous.side
        ):
            if side != previous.side:
                self._forget_unconfirmed_crash(history, side)
            return state
        if self._confirmed_crash(before, after, old_score, score):
            history[side] = True
            return self._restore_crash_prefix(state, side, before)
        if self._unconfirmed_normal_mismatch(
            history[side], before, after, old_score, score, own_action
        ):
            self._forget_unconfirmed_crash(history, side)
        return state

    def _unconfirmed_normal_mismatch(self, crashed, before, after, old_score, score, pending):
        return crashed is False and not self._matches_normal_action(
            before, after, old_score, score, pending
        )

    def _could_be_stationary_shot(self, before, after, old_score, score, own_action, our_turn):
        if not self._stationary_suffix_loss(before, after):
            return False
        if score != old_score - SHOOT_COST:
            return False
        if not our_turn:
            return old_score >= SHOOT_COST
        return bool(own_action and self._pending_kind(own_action) == 'shoot')

    @staticmethod
    def _stationary_suffix_loss(before, after):
        return before[0] == after[0] and set(after).issubset(before) and len(after) <= len(before)

    @staticmethod
    def _confirmed_crash(before, after, old_score, score):
        prefix = before[:CRASH_KEEP_LENGTH]
        crash_scores = {old_score - CRASH_PENALTY}
        if old_score > 0:
            crash_scores.add(0)
        return (
            before[0] == after[0]
            and set(after) == set(prefix)
            and len(after) == len(prefix)
            and score in crash_scores
        )

    @staticmethod
    def _restore_crash_prefix(state, side, before):
        snakes = dict(state.snakes)
        snakes[side] = before[:CRASH_KEEP_LENGTH]
        return replace(state, snakes=snakes, reliable_tails=state.reliable_tails | {side})

    def _matches_normal_action(self, before, after, old_score, score, pending):
        if self._is_pending_shot(pending):
            return after == before and score == old_score - SHOOT_COST
        if not self._normal_body_step(before, after):
            return False
        return not pending or after[0] == step(before[0], pending[1])

    def _is_pending_shot(self, pending):
        return bool(pending and self._pending_kind(pending) == 'shoot')

    @staticmethod
    def _normal_body_step(before, after):
        if manhattan(before[0], after[0]) != 1:
            return False
        expected = {
            len(before): (after[0],) + before[:-1],
            len(before) + 1: (after[0],) + before,
        }
        return set(after) == set(expected.get(len(after), ()))

    def _infer_gap_crashes(self, previous, state, data, history):
        for side in ('A', 'B'):
            rival = 'B' if side == 'A' else 'A'
            length = len(state.body(side))
            if 0 < length < len(previous.body(side)):
                if not self._gap_cut_possible(previous, state, data, rival):
                    history[side] = True

    @staticmethod
    def _gap_cut_possible(previous, state, data, rival):
        if max(previous.scores[rival], state.scores[rival]) >= SHOOT_COST:
            return True
        if max(previous.invincibility(rival), state.invincibility(rival)) > 0:
            return True
        return SnakeBrain._gap_star_possible(previous, state, data)

    @staticmethod
    def _gap_star_possible(previous, state, data):
        if previous.stars or state.stars:
            return True
        return 'invincible_1' in data and state.remaining_moves <= MAX_MOVES - 40

    def _clear_gap_prediction(self, game_id, previous, state, contiguous):
        pending = self.pending_moves.get(game_id)
        if not pending or self._pending_kind(pending) != 'shoot':
            self.pending_moves.pop(game_id, None)
        elif contiguous and self._predicted_shot_observed(previous, state):
            self.pending_moves.pop(game_id, None)

    @staticmethod
    def _predicted_shot_observed(previous, state):
        side = previous.side
        before, after = previous.body(side), state.body(side)
        return (
            state.head(side) == previous.head(side)
            and state.scores[side] == previous.scores[side] - SHOOT_COST
            and len(after) <= len(before)
            and set(after) == set(before[: len(after)])
        )

    def _store_observation(self, game_id, previous, state, history, token):
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
        self.knowledge_states[game_id] = state.knowledge
        return state

    @staticmethod
    def _observed_star_timing(before, after):
        """Keep only a spawn deadline that follows from observed turns."""
        if after.partial:
            return after
        return SnakeBrain._complete_star_timing(before, after)

    @staticmethod
    def _complete_star_timing(before, after):
        if after.star_frenzy:
            return replace(after, pending_star_spawns=max(0, 10 - len(after.stars)))
        if after.stars or max(after.invincible_a, after.invincible_b) > 0:
            return replace(after, star_spawn_due_round=None, pending_star_spawns=0)
        if before is None:
            return after
        due = observed_star_deadline(before, after.remaining_moves)
        return replace(
            after,
            star_spawn_due_round=due,
            pending_star_spawns=int(star_is_due(after.remaining_moves, due)),
        )

    @staticmethod
    def _single_action_history(before, after, data, history):
        """Infer history only from a coherent one-action server snapshot."""
        result = SnakeBrain._unconfirmed_history(history)
        if passive_player_matches(before, after):
            result[before.enemy] = history[before.enemy]
        if SnakeBrain._protected_turn(before, after, before.side):
            result[before.side] = history[before.side]
        action = observed_action(before, after, data)
        if action is None:
            return result, None
        expected = simulate_observed_action(before, action)
        if not matching_action_result(expected, after):
            return result, None
        return verified_crash_history(expected, history), action

    def choose_action(self, data):
        """Select a legal move or a verified useful shot under the search budget."""
        with planning_module.planning_scope():
            return self._choose_action(data)

    def _choose_action(self, data):
        """Compare actions in the same completed tree used to rank moves."""
        game_id = str(data.get('game_id', 'default'))
        self.action_searches.pop(game_id, None)
        direction = self.choose_move(data)
        if direction is None:
            return None
        fallback = {'action_type': 'move', 'direction': direction}
        state = self.current_states.get(game_id)
        search = self._action_tree(game_id, state)
        if search is None:
            return fallback
        return self._select_search_action(game_id, state, search, fallback)

    def _action_tree(self, game_id, state):
        cached = self.action_searches.pop(game_id, None)
        if state is None:
            return None
        if cached is not None:
            return cached[1] if cached[0] is state else None
        if trapped_head_has_shot(state):
            return self._search_trapped_head(state)
        return None

    def _search_trapped_head(self, state):
        """Compare the forced moves and shots when movement ranking returned early."""
        weights = get_weights()
        engine, compact = IterativeSearchEngine.from_game_state(
            state,
            time_budget_ms=self._search_time_budget(len(self.previous_snakes)),
            max_depth=weights['iterative_search_max_depth'],
        )
        return engine.search(compact, state.side, available_moves(state, state.side))

    def _select_search_action(self, game_id, state, search, fallback):
        direction = fallback['direction']
        debug = self.last_decisions.setdefault(game_id, {'event': 'decision_debug'})
        diagnostic = action_search_diagnostic(search, direction)
        debug['action_search'] = diagnostic
        if search.completed_depth == 0:
            diagnostic['selection'] = 'insufficient_completed_depth'
            return fallback
        moves = comparable_move_scores(state, search)
        if not moves:
            diagnostic['selection'] = 'no_comparable_moves'
            return fallback
        best_move = max(moves, key=moves.get)
        shot = self._profitable_shot(state, search, moves[best_move])
        if shot is not None:
            return self._record_search_shot(
                game_id, state, debug, diagnostic, shot, moves[best_move]
            )
        repair = self._repair_search_exposure(
            state, direction, moves, best_move, debug, diagnostic
        )
        return repair or fallback

    def _profitable_shot(self, state, search, best_move_score):
        if search.completed_depth < get_weights()['iterative_search_min_apply_depth']:
            return None
        shots = self._effective_shot_scores(state, search)
        if not shots:
            return None
        direction = max(shots, key=lambda key: shots[key][0])
        value, metadata = shots[direction]
        if value > best_move_score:
            return direction, value, metadata
        return None

    def _effective_shot_scores(self, state, search):
        shots = {}
        for key, value in search.scores.items():
            if is_shot_key(key):
                outcome = self._effective_shot(state, key[1])
                if outcome is not None:
                    shots[key[1]] = value, outcome.last_action
        return shots

    def _record_search_shot(self, game_id, state, debug, diagnostic, shot, best_move_score):
        direction, value, metadata = shot
        diagnostic.update(
            selection='profitable_hit',
            selected_score=value,
            best_move_score=best_move_score,
            **metadata,
        )
        debug.update(
            action_type='shoot',
            direction=direction,
            reason='shared_search_profitable_shoot',
        )
        self.targets[(game_id, state.side)] = None
        return {'action_type': 'shoot', 'direction': direction}

    def _repair_search_exposure(self, state, direction, moves, best, debug, diagnostic):
        risk = self._action_exposure(state, direction, moves)
        if risk and direction in moves and moves[best] > moves[direction]:
            diagnostic.update(
                selection=risk,
                selected_score=moves[best],
                fallback_score=moves[direction],
            )
            debug.update(action_type='move', direction=best, reason='shared_search_' + risk)
            return {'action_type': 'move', 'direction': best}
        diagnostic['selection'] = 'conservative_move'
        return None

    @staticmethod
    def _effective_shot(state, direction):
        if direction not in directions_for(state) or not can_shoot(state):
            return None
        outcome = apply_action(state, 'shoot', direction, state.side)
        return (
            outcome
            if outcome.last_action['hit'] and outcome.last_action['cut_length'] > 0
            else None
        )

    @staticmethod
    def _action_exposure(state, direction, moves):
        if direction not in moves:
            return None
        # Geometry-only diagnostics share the planner's conservative handling
        # of unknown v7 crash history; this successor is never persisted.
        successor = simulated_move(state, direction, state.side, perspective=state.side)
        if successor.last_action.get('bomb_triggered'):
            return 'bomb_economics'
        if opponent_has_shot(state, successor):
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
        """Reconcile one turn and rank its safe movement options within the planning scope."""
        game_id = str(data.get('game_id', 'default'))
        state = self.observe_state(data)
        if state is None:
            return None
        self.current_states[game_id] = state
        self._update_food_clock(game_id, state.side, state)
        legal = legal_moves(state, state.side)
        debug = self._base_debug(data, state, legal)
        if not legal:
            move = next(iter(available_moves(state, state.side)), 'up')
            self._save_debug(game_id, debug, 'no_safe_moves', move)
            return move
        kill = forced_kill_move(state, state.side, legal)
        if kill:
            return self._save_selected_move(game_id, state, debug, kill, 'forced_kill')
        legal = self._clean_legal_moves(state, legal, debug)
        choice = self._rank_move_options(game_id, state, legal, debug)
        return self._select_ranked_move(choice)

    def _save_selected_move(self, game_id, state, debug, move, reason):
        self._save_debug(game_id, debug, reason, move)
        self._remember(game_id, state)
        return move

    @staticmethod
    def _clean_legal_moves(state, legal, debug):
        if state.invincibility(state.side):
            return legal
        targets = move_destinations(state, legal)
        wrong = directions_in_cells(targets, state.wrong_food())
        walls = directions_in_cells(targets, state.walls)
        clean = [move for move in legal if move not in (wrong | walls)]
        if not clean:
            return legal
        record_avoided_moves(debug, wrong, walls)
        debug['legal'] = clean
        return clean

    def _rank_move_options(self, game_id, state, legal, debug):
        side = state.side
        current_target = self.targets.get((game_id, side))
        hunger = self.turns_since_food[(game_id, side)]
        repeats = self._repeat_count(game_id, state)
        cache = TurnSearchCache()
        active_games = len(self.previous_snakes) + (game_id not in self.previous_snakes)
        time_ms = self._search_time_budget(active_games)
        plans = build_food_plans(state, side, current_target, hunger)
        plan = choose_food_plan(state, side, current_target, hunger, plans=plans)
        ranked = rank_deep_moves(
            state,
            side,
            legal,
            plan,
            hunger,
            repeats,
            cache,
            search_time_ms=time_ms,
        )
        action_search = getattr(cache, 'action_search', None)
        if action_search is not None:
            self.action_searches[game_id] = (state, action_search)
        commit = bool(plan and should_commit_to_food(plan, hunger, repeats))
        debug.update(
            {
                'current_target': current_target,
                'hunger': hunger,
                'repeat_count': repeats,
                'active_games': active_games,
                'search_time_budget_ms': time_ms,
                'plan': self._plan_debug(plan),
                'commit_to_plan': commit,
                'deep_moves': ranked[:4],
            }
        )
        if self.enable_debug:
            debug['food_options'] = self._food_options(plans)
            debug['candidate_moves'] = self._candidate_moves(state, side, legal, ranked)
        return MoveOptions(
            game_id, state, legal, debug, plan, ranked, cache, hunger, repeats, commit
        )

    def _select_ranked_move(self, options):
        plan = options.plan
        advantage = self._advantage_move(options)
        if advantage is not None:
            self.targets[(options.game_id, options.state.side)] = None
            return self._finish_ranked_move(options, advantage, 'advantage_control')
        if options.commit:
            self.targets[(options.game_id, options.state.side)] = plan.food
            if plan.first_move in options.legal:
                move, reason = self._committed_food_move(options)
                self._remember_progress(options.game_id, options.state.side, options.state, plan)
                return self._finish_ranked_move(options, move, reason)
        move, reason = self._positioning_move(options)
        return self._finish_ranked_move(options, move, reason)

    def _finish_ranked_move(self, options, move, reason):
        return self._save_selected_move(
            options.game_id, options.state, options.debug, move, reason
        )

    @staticmethod
    def _advantage_move(options):
        state, plan, legal = options.state, options.plan, options.legal
        reference = plan.first_move if options.commit else None
        advantage = advantage_control_move(
            state,
            state.side,
            legal,
            options.ranked,
            reference_move=reference if reference in legal else None,
            adjacent_food=bool(plan and plan.our_distance == 1),
        )
        options.debug['advantage_control'] = advantage
        return valid_advantage(advantage, legal)

    @staticmethod
    def _committed_food_move(options):
        state, plan, legal = options.state, options.plan, options.legal
        best = next(iter(options.ranked), None)
        move = plan.first_move
        trapped = adjacent_trap(plan, ranked_move(options.ranked, move), best, move)
        options.debug['adjacent_trap_detected'] = trapped
        if trapped:
            return best['direction'], 'adjacent_trap_avoidance'
        hold = sequence_gate_hold_move(state, state.side, legal, plan, options.ranked)
        if hold is not None:
            options.debug['sequence_gate_held'] = True
            return hold, 'sequence_gate_hold'
        return review_food_move(options, move, best)

    def _positioning_move(self, options):
        state, plan, legal = options.state, options.plan, options.legal
        positioning = plan if should_use_plan_for_positioning(plan) else None
        self.targets[(options.game_id, state.side)] = positioning.food if positioning else None
        ranked = options.ranked or rank_deep_moves(
            state, state.side, legal, positioning, options.hunger, options.repeats, options.cache
        )
        if ranked:
            options.debug['survival_candidates'] = ranked[:4]
            return ranked[0]['direction'], positioning_reason(positioning)
        move = fallback_food_positioning(
            state, state.side, legal, positioning, options.hunger, options.repeats
        )
        return move, 'fallback_positioning'

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
        """Repair an unsafe proposal using currently observed legal moves."""
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
        """Keep an unconfirmed own-body prediction until the next server observation."""
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
        if direction in directions_for(state) and not state.is_terminal():
            self.pending_moves[game_id] = (state, direction)
            self.previous_snakes[game_id] = predicted_move_bodies(state, direction)
        self.current_states.pop(game_id, None)

    def commit_action(self, data, action):
        """Record the sent action without treating its predicted effects as server facts."""
        if action.get('action_type', 'move') == 'move':
            return self.commit_move(data, action.get('direction'))
        game_id = str(data.get('game_id', 'default'))
        state = self.current_states.get(game_id)
        if state is None:
            state = self.observe_state(data)
        direction = action.get('direction')
        if can_commit_shot(state, direction):
            predicted = apply_action(state, 'shoot', direction, state.side)
            self.pending_moves[game_id] = (state, direction, 'shoot')
            self.previous_snakes[game_id] = predicted.snakes
        self.current_states.pop(game_id, None)

    def forget(self, game_id):
        """Release predictions, memory and strategy data belonging to a finished match."""
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
        self.knowledge_states.pop(str(game_id), None)

    def decision_debug(self, game_id):
        """Return the bounded decision diagnostics for one match when debugging is enabled."""
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
            state.crashed_a,
            state.crashed_b,
            tuple(sorted(state.tail_food_a)),
            tuple(sorted(state.tail_food_b)),
            tuple(sorted(state.bombs_large)),
            tuple(sorted(state.bombs_small)),
            tuple(sorted(state.shot_food)),
            state.pending_bomb_respawns,
            tuple(sorted(state.stars)),
            state.invincible_a,
            state.invincible_b,
            state.star_frenzy,
            state.star_spawn_due_round,
            state.pending_star_spawns,
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
            state.crashed_a,
            state.crashed_b,
            tuple(sorted(state.tail_food_a)),
            tuple(sorted(state.tail_food_b)),
            tuple(sorted(state.bombs_large)),
            tuple(sorted(state.bombs_small)),
            tuple(sorted(state.shot_food)),
            state.pending_bomb_respawns,
            tuple(sorted(state.stars)),
            state.invincible_a,
            state.invincible_b,
            state.star_frenzy,
            state.star_spawn_due_round,
            state.pending_star_spawns,
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
            'knowledge': decision_knowledge(state),
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
    """Repair a proposed direction using the currently observed legal neighbors."""
    state = parse_state(data)
    legal = legal_moves(state, state.side)
    if direction in legal:
        return direction
    if legal:
        return fallback_food_positioning(state, state.side, legal, None, 0, 0)
    return next(iter(available_moves(state, state.side)), 'up')


def decision_knowledge(state):
    """Summarize live uncertainty without copying the entire remembered map."""
    if not state.partial:
        return {}
    return {
        'levels': state.levels,
        'partial_observation': True,
        'observed_cell_count': len(state.observation.cells),
        'remembered_object_count': len(state.knowledge.remembered),
        'memory_confidence': 'possibly_outdated',
        'last_enemy_head': state.knowledge.last_enemy_head,
        'last_enemy_turn': state.knowledge.last_enemy_turn,
    }
