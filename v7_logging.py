"""Observational copies only; never used by move selection or reconciliation."""

from dataclasses import replace
from snake_bot.snake_state import (
    apply_action,
    in_bounds,
    parse_board,
    step,
)
from snake_bot.geometry import directions_for
from snake_bot.game_rules import ELIMINATION_SCORE_THRESHOLD

DATA_FIELDS = frozenset(
    (
        'game_id side board board_x board_y board_z head levels rows cols board_size '
        'score_1 score_2 score_A '
        'score_B multiplier_1 multiplier_2 multiplier_A multiplier_B '
        'remaining_moves direction challenge_id opponent game row col '
        'orientation tournament tournament_id championship championship_id '
        'source mode type kind challenge_type winner result'
    ).split()
)
DATA_FIELDS = DATA_FIELDS | {
    'reason',
    'cause',
    'action',
    'action_type',
    'invincible_1',
    'invincible_2',
    'invincible_A',
    'invincible_B',
    'timeouts_1',
    'timeouts_2',
    'timeouts_A',
    'timeouts_B',
}
DEBUG_FIELDS = frozenset(
    (
        'event game_id side head enemy_head food numbered_food '
        'next_food_digit pickups walls multipliers crash_history tail_food_a '
        'tail_food_b legal scores remaining_moves current_target hunger '
        'repeat_count active_games search_time_budget_ms plan commit_to_plan '
        'deep_moves food_options candidate_moves advantage_control '
        'adjacent_trap_detected plan_move_stats plan_move_penalty '
        'plan_position_penalty plan_hot_lost_food_penalty '
        'tactical_arbitration deep_override_allowed plan_move_quality '
        'best_tactical_move best_tactical_quality best_tactical_stats '
        'forced_loss_avoided survival_candidates reason direction '
        'safe_repaired_from avoided_wrong_digit_moves avoided_wall_moves '
        'sequence_gate_held verification'
    ).split()
)
DEBUG_FIELDS = DEBUG_FIELDS | {
    'knowledge',
    'action_type',
    'bombs_large',
    'bombs_small',
    'shot_food',
    'action_search',
}
DEBUG_FIELDS = DEBUG_FIELDS | {
    'invincible_self',
    'invincible_rival',
    'stars',
    'star_frenzy',
    'star_consumed',
    'invincibility_before',
    'invincibility_after',
    'ram_cut',
    'ram_hit_coordinate',
    'cut_length',
    'shot_blocked_by_invincibility',
}


def redact_reason(value):
    if isinstance(value, str) and any(
        marker in value.lower()
        for marker in (
            'token',
            'auth',
            'secret',
            'password',
            'credential',
            'bearer',
            'api_key',
            'apikey',
        )
    ):
        return '[redacted]'
    return value


def safe_debug(value):
    if isinstance(value, dict):
        return safe_debug_mapping(value)
    if isinstance(value, (list, tuple)):
        return [safe_debug(item) for item in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


def safe_debug_mapping(value):
    return {
        key: safe_debug(redact_field(key, item))
        for key, item in value.items()
        if not sensitive_debug_key(key)
    }


def redact_field(key, value):
    return redact_reason(value) if key in ('reason', 'cause') else value


def sensitive_debug_key(key):
    lowered = str(key).lower()
    return any(
        word in lowered
        for word in (
            'token',
            'secret',
            'password',
            'credential',
            'auth',
            'api_key',
            'apikey',
        )
    )


def safe_data(data):
    return {
        key: redact_reason(value) if key in ('reason', 'cause') else value
        for key, value in data.items()
        if key in DATA_FIELDS and safe_wire_value(key, value)
    }


def safe_wire_value(key, value):
    """Permit the structured cube head without exposing arbitrary nested fields."""
    if key == 'head' and isinstance(value, dict):
        return set(value) == {'level', 'row', 'col'} and all(
            type(coordinate) is int for coordinate in value.values()
        )
    return isinstance(value, (str, int, float, bool, type(None)))


def safe_message(message):
    if 'data' in message:
        return {**message_envelope(message), 'data': safe_data(message['data'])}
    return safe_debug({key: value for key, value in message.items() if key in DEBUG_FIELDS})


def message_envelope(message):
    return {
        key: message[key]
        for key in ('event', 'action', 'action_type', 'direction')
        if key in message
    }


def number(data, *keys):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def sent_prediction(data, state, direction, action_type='move'):
    if not can_simulate_action(state, direction):
        return None
    side = state.side
    head = state.head(side)
    target = step(head, direction) if action_type == 'move' else head
    cause = predicted_collision(state, target, action_type)
    predicted, known_economics = simulate_prediction(state, direction, action_type, cause)
    if predicted is None:
        return None
    known_economics = observable_prediction_score(
        known_economics, data, state, target, action_type
    )
    return {
        **prediction_metadata(predicted, known_economics),
        **prediction_body_fields(state, predicted, known_economics),
        'source': 'prediction_not_server_confirmation',
        'side': side,
        'direction': direction,
        'target': target,
        'collision_cause': cause,
        'tail_interaction': predicted_tail_interaction(state, target, cause),
        'expected_detached_tail': predicted_detachment(state, predicted),
        'uncertainty': prediction_uncertainty(predicted),
    }


def observable_prediction_score(known, data, state, target, action_type):
    """Hidden bomb effects prevent an exact predicted score in partial observations."""
    return (
        known and not state.partial and prediction_scores_known(data, state, target, action_type)
    )


def can_simulate_action(state, direction):
    return (
        state is not None
        and direction in directions_for(state)
        and bool(state.body(state.side))
        and not state.is_terminal()
    )


def predicted_collision(state, target, action_type):
    if action_type == 'shoot' or state.invincibility(state.side):
        return None
    if not in_bounds(state, target):
        return 'border'
    if target in state.walls:
        return 'wall'
    return predicted_body_collision(state, target)


def own_tail_food(state):
    return state.tail_food_a if state.side == 'A' else state.tail_food_b


def rewarding_food(state):
    return state.food | own_tail_food(state) | state.shot_food | state.stars


def predicted_body_collision(state, target):
    body = state.body(state.side)
    occupied = state.occupied()
    if target not in rewarding_food(state):
        occupied.discard(body[-1])
    if target in body[1:2]:
        return 'self'
    if target in occupied:
        return 'self' if target in body else 'rival'
    return None


def simulate_prediction(state, direction, action_type, cause):
    try:
        return apply_action(state, action_type, direction, state.side), True
    except ValueError:
        if cause is None:
            return None, False
        # An assumed first crash provides geometry, not economic evidence.
        field = 'crashed_a' if state.side == 'A' else 'crashed_b'
        assumed = replace(state, **{field: False})
        return apply_action(assumed, action_type, direction, state.side), False


def prediction_scores_known(data, state, target, action_type):
    side = state.side
    idx = 1 if side == 'A' else 2
    if number(data, f'score_{idx}', f'score_{side}') is None:
        return False
    if action_type != 'move' or target not in rewarding_food(state):
        return True
    return number(data, f'multiplier_{idx}', f'multiplier_{side}') == state.multipliers[side]


def predicted_tail_interaction(state, target, cause):
    for side, symbol, cells in tail_food_sets(state):
        if target in cells:
            return {
                'symbol': symbol,
                'beneficiary': side,
                'originator': 'B' if side == 'A' else 'A',
                'event': tail_interaction_kind(state, side, cause),
            }
    return None


def tail_interaction_kind(state, beneficiary, cause):
    if state.side == beneficiary and cause is None:
        return 'consume'
    return 'cleanup_or_blocked'


def tail_food_sets(state):
    return (('A', 'Ⓐ', state.tail_food_a), ('B', 'Ⓑ', state.tail_food_b))


def predicted_detachment(state, predicted):
    detached = None
    for before, after in zip(tail_food_sets(state), tail_food_sets(predicted)):
        side, symbol, old = before
        new = after[2] - old
        if new:
            detached = {
                'cells': list(map(list, sorted(new))),
                'symbol': symbol,
                'beneficiary': side,
                'originator': state.side,
            }
    return detached


def prediction_metadata(predicted, known_economics):
    fields = (
        'action_type',
        'direction',
        'shoot_cost',
        'hit',
        'hit_coordinate',
        'cut_length',
        'shot_food_generated',
        'bomb_triggered',
        'bomb_type',
        'bomb_types',
        'bomb_positions',
        'score_before',
        'score_after',
        'star_consumed',
        'invincibility_before',
        'invincibility_after',
        'ram_cut',
        'ram_hit_coordinate',
        'shot_blocked_by_invincibility',
    )
    metadata = {key: predicted.last_action.get(key) for key in fields}
    if not known_economics:
        metadata.update(score_before=None, score_after=None)
    return metadata


def prediction_body_fields(state, predicted, known_economics):
    return {
        'expected_body': predicted.body(state.side),
        'expected_opponent_body': predicted.body(state.enemy),
        'expected_shot_food': sorted(predicted.shot_food),
        'invincible_self': state.invincibility(state.side),
        'invincible_rival': state.invincibility(state.enemy),
        'stars': sorted(state.stars),
        'star_frenzy': state.star_frenzy,
        'expected_invincibility': {side: predicted.invincibility(side) for side in ('A', 'B')},
        'expected_stars': sorted(predicted.stars),
        'pending_bomb_respawns': predicted.pending_bomb_respawns,
        'pending_star_spawns': predicted.pending_star_spawns,
        'expected_score': predicted.scores[state.side] if known_economics else None,
    }


def prediction_uncertainty(predicted):
    uncertainty = ['rival_action_not_predicted', 'body_order_from_brain_tracking']
    if predicted.partial:
        uncertainty.extend(('off_plane_objects_unknown', 'hidden_bomb_effects_unknown'))
    if predicted.pending_bomb_respawns:
        uncertainty.append('bomb_respawn_position_unknown')
    if predicted.pending_star_spawns:
        uncertainty.append('star_spawn_position_unknown')
    return uncertainty


def verification_record(
    data,
    observed=None,
    previous=None,
    direction=None,
    finished=False,
    action_type='move',
):
    after = server_snapshot(data, observed)
    before = previous.get('after') if previous else None
    gap = horizon_delta(before, after)
    prediction = snapshot_prediction(data, observed, direction, finished, action_type, after)
    return {
        'before': before,
        'after': after,
        'horizon_delta': gap,
        'server_reported_cause': reported_cause(data) if finished else {},
        'sent_action_prediction': prediction,
        'prediction_reconciliation': reconcile_prediction(previous, after, gap),
        'transition': classify_interval(gap),
        'score_deltas': snapshot_score_deltas(before, after, gap),
        'crash_cause': {'A': 'unknown', 'B': 'unknown'},
        'tail_beneficiary_by_symbol': {'Ⓐ': 'A', 'Ⓑ': 'B'},
        'tail_origin_by_symbol': {'Ⓐ': 'B', 'Ⓑ': 'A'},
        'tail_net_changes': snapshot_tail_changes(before, after, gap),
        'tail_cleanup_or_consumption': 'unknown',
        'game_over_cause': game_over_cause(after) if finished else 'unknown',
        'uncertainty': [
            'rival_actions_unobserved',
            'net_deltas_not_action_counts',
            'last_sent_move_not_confirmed',
            'missing_fields_not_defaulted',
        ],
        'last_observed_snapshot': previous_board(before, after, finished),
    }


def previous_board(before, after, finished):
    if finished and after['board'] is None:
        return before
    return None


def reported_cause(data):
    return {key: value for key, value in safe_data(data).items() if key in ('reason', 'cause')}


PLAYER_FIELDS = (('A', 1), ('B', 2))
SPECIAL_SYMBOLS = {
    'bombs_large': '💣',
    'bombs_small': '🧨',
    'shot_food': '*',
    'stars': '⭐',
}


def snapshot_values(data, field):
    return {
        side: number(data, f'{field}_{index}', f'{field}_{side}') for side, index in PLAYER_FIELDS
    }


def server_snapshot(data, observed):
    board = data.get('board')
    if not isinstance(board, str):
        board = None
    side = data.get('side')
    moves = number(data, 'remaining_moves')
    snapshot = {
        'game_id': data.get('game_id'),
        'side': side,
        'remaining_moves': moves,
        'board': board,
        'scores': snapshot_values(data, 'score'),
        'multipliers': snapshot_values(data, 'multiplier'),
        'invincibility': snapshot_values(data, 'invincible'),
        **relative_invincibility(data, side),
        'star_frenzy': moves <= 100 if moves is not None else None,
        'sent_move': None,
        'server_reported_action': reported_action(data),
        'source': 'server_snapshot',
    }
    snapshot.update(board_observation(data, snapshot, observed))
    return snapshot


def relative_invincibility(data, side):
    own, rival = ('invincible_1', 'invincible_2')
    if side != 'A':
        own, rival = rival, own
    return {
        'invincible_self': number(data, own),
        'invincible_rival': number(data, rival),
    }


def reported_action(data):
    action = data.get('action', data.get('action_type'))
    if action in ('move', 'shoot'):
        return {'action_type': action, 'direction': data.get('direction')}
    return None


def board_observation(data, snapshot, observed):
    fields = {
        'body_cells': None,
        'heads': None,
        'tracked_body': None,
        'tail_symbols': None,
        'v8_v9_cells': None,
        'stars': None,
        'crashed': {'A': None, 'B': None},
        'crashed_scope': 'cumulative_brain_history',
    }
    if data.get('levels', 1) > 1:
        return cube_board_observation(data, observed, fields)
    if snapshot['board'] is None:
        return fields
    parsed, _, _ = parse_board(snapshot['board'], data.get('rows'), data.get('cols'))
    fields.update(board_symbols(parsed, snapshot['invincibility']))
    if matches_observation(parsed, snapshot, observed):
        fields['tracked_body'] = tracked_snakes(observed)
        fields['crashed'] = {'A': observed.crashed_a, 'B': observed.crashed_b}
    return fields


def cube_board_observation(data, observed, fields):
    """Log only currently visible cube facts, keeping memory explicitly uncertain."""
    from snake_bot.snake_state import parse_state

    state = observed if observed is not None else parse_state(data)
    cells = state.observation.cells
    special = cube_special_positions(cells)
    return {
        **fields,
        'partial_observation': True,
        'levels': state.levels,
        'heads': {side: logged_cube_head(state, side) for side in ('A', 'B')},
        'body_cells': {
            side: cube_symbol_positions(cells, (side, side.lower())) for side in ('A', 'B')
        },
        'tracked_body': tracked_snakes(state),
        'v8_v9_cells': special,
        'tail_symbols': cube_tail_positions(cells),
        'stars': special['stars'],
        'knowledge': cube_knowledge_summary(state),
    }


def cube_special_positions(cells):
    return {key: cube_symbol_positions(cells, symbols) for key, symbols in SPECIAL_SYMBOLS.items()}


def cube_tail_positions(cells):
    return {symbol: cube_symbol_positions(cells, symbol) for symbol in ('Ⓐ', 'Ⓑ')}


def cube_symbol_positions(cells, symbols):
    """Serialize visible coordinates without merging cells on different levels."""
    return [list(cell) for cell, symbol in cells.items() if symbol in symbols]


def logged_cube_head(state, side):
    """A missing rival head is unknown rather than a remembered confirmed position."""
    head = state.head(side)
    return list(head) if head is not None else None


def cube_knowledge_summary(state):
    """Bound log size to counts and uncertainty rather than dumping cube memory."""
    return {
        'observed_cell_count': len(state.observation.cells),
        'remembered_object_count': len(state.knowledge.remembered),
        'memory_confidence': 'possibly_outdated',
        'enemy_visible': state.head(state.enemy) is not None,
        'last_enemy_head': state.knowledge.last_enemy_head,
        'last_enemy_turn': state.knowledge.last_enemy_turn,
    }


def board_symbols(parsed, invincibility):
    special = {key: symbol_positions(parsed, symbol) for key, symbol in SPECIAL_SYMBOLS.items()}
    return {
        'body_cells': {
            side: symbol_positions(parsed, (side, side.lower())) for side in ('A', 'B')
        },
        'heads': {side: board_head(parsed, side, invincibility[side]) for side in ('A', 'B')},
        'tail_symbols': {symbol: symbol_positions(parsed, symbol) for symbol in ('Ⓐ', 'Ⓑ')},
        'v8_v9_cells': special,
        'stars': special['stars'],
    }


def symbol_positions(parsed, symbols):
    return [
        [row, col]
        for row, line in enumerate(parsed)
        for col, char in enumerate(line)
        if char in symbols
    ]


def board_head(parsed, side, invincibility):
    symbol = side.lower() if (invincibility or 0) > 0 else side
    positions = symbol_positions(parsed, symbol)
    return positions[0] if len(positions) == 1 else None


def matches_observation(parsed, snapshot, observed):
    return (
        observed is not None
        and observed.board == parsed
        and snapshot['remaining_moves'] == observed.remaining_moves
        and matches_player_observation(snapshot, observed)
    )


def matches_player_observation(snapshot, observed):
    return snapshot['scores'] == observed.scores and all(
        value in (None, observed.invincibility(side))
        for side, value in snapshot['invincibility'].items()
    )


def tracked_snakes(observed):
    return {
        side: {
            'body': observed.body(side),
            'order_reliable': side in observed.reliable_tails,
        }
        for side in ('A', 'B')
    }


def snapshot_prediction(data, observed, direction, finished, action, after):
    after['sent_move'] = direction if action == 'move' else None
    after['sent_action'] = {'action_type': action, 'direction': direction} if direction else None
    if can_predict_snapshot(after, observed, finished):
        return sent_prediction(data, observed, direction, action)
    return None


def can_predict_snapshot(after, observed, finished):
    return after['tracked_body'] is not None and not finished and after['side'] == observed.side


def horizon_delta(before, after):
    if not before:
        return None
    start, end = before['remaining_moves'], after['remaining_moves']
    if start is None or end is None:
        return None
    return start - end


def classify_interval(gap):
    if gap is None:
        return 'unavailable'
    if gap == 0:
        return 'duplicate_or_correction'
    if gap < 0:
        return 'stale'
    return 'interval_with_unobserved_actions'


def snapshot_score_deltas(before, after, gap):
    if before and gap is not None and gap > 0:
        return {
            side: score_delta(before['scores'][side], after['scores'][side]) for side in ('A', 'B')
        }
    return {'A': None, 'B': None}


def score_delta(before, after):
    if before is None or after is None:
        return None
    return after - before


def reconcile_prediction(previous, after, gap):
    prior = previous.get('sent_action_prediction') if previous else None
    if not prior or gap != 1 or after['body_cells'] is None:
        return 'unavailable'
    return reconcile_action(prior, after)


def reconcile_action(prior, after):
    if after.get('partial_observation'):
        return 'partial_observation_hidden_effects_uncertain'
    return reconcile_complete_action(prior, after)


def reconcile_complete_action(prior, after):
    side = prior['side']
    score = after['scores'][side]
    expected = prior['expected_score']
    if expected is None or score is None:
        return 'unavailable'
    if different_invincibility(prior, after):
        return 'differs_invincibility'
    if different_head_or_side(prior, after):
        return 'differs_head_or_side'
    return reconcile_effects(prior, after, score == expected)


def different_invincibility(prior, after):
    counts = prior.get('expected_invincibility')
    if not counts:
        return False
    return any(
        value is not None and value != counts[side]
        for side, value in after['invincibility'].items()
    )


def different_head_or_side(prior, after):
    side = prior['side']
    rival = 'B' if side == 'A' else 'A'
    expected_head = list(prior['expected_body'][0])
    return after['heads'][side] != expected_head or after['side'] not in (None, rival)


def reconcile_effects(prior, after, correct_score):
    own_body = after['body_cells'][prior['side']]
    consistent = correct_score and same_cells(own_body, prior['expected_body'])
    if prior.get('action_type') == 'shoot':
        return reconcile_shot(prior, after, consistent)
    return 'consistent_body_and_score_only' if consistent else 'differs_body_or_score'


def reconcile_shot(prior, after, consistent):
    rival = 'B' if prior['side'] == 'A' else 'A'
    if (
        consistent
        and same_cells(after['body_cells'][rival], prior['expected_opponent_body'])
        and same_cells(after['v8_v9_cells']['shot_food'], prior['expected_shot_food'])
    ):
        return 'consistent_predicted_shoot_effects'
    return 'differs_predicted_shoot_effects'


def same_cells(actual, expected):
    return set(map(tuple, actual)) == set(map(tuple, expected))


def snapshot_tail_changes(before, after, gap):
    if not before or gap is None or gap <= 0:
        return None
    return tail_changes(before['tail_symbols'], after['tail_symbols'])


def tail_changes(old, new):
    if old is None or new is None:
        return None
    return {symbol: cell_changes(old[symbol], cells) for symbol, cells in new.items()}


def cell_changes(before, after):
    old, new = set(map(tuple, before)), set(map(tuple, after))
    return {
        'added': list(map(list, sorted(new - old))),
        'removed': list(map(list, sorted(old - new))),
    }


def game_over_cause(after):
    moves = after['remaining_moves']
    horizon_end = moves is not None and moves <= 0
    eliminated = any(
        value is not None and value < ELIMINATION_SCORE_THRESHOLD
        for value in after['scores'].values()
    )
    return GAME_OVER_CAUSES[horizon_end, eliminated]


GAME_OVER_CAUSES = {
    (True, True): 'ambiguous_horizon_or_score_threshold',
    (True, False): 'possible_horizon_exhaustion',
    (False, True): 'possible_score_threshold',
    (False, False): 'unknown',
}
