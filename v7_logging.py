"""Observational copies only; never used by move selection or reconciliation."""
from dataclasses import replace
from snake_bot.snake_state import DIRS, apply_action, in_bounds, parse_board, step
from snake_bot.game_rules import ELIMINATION_SCORE_THRESHOLD


DATA_FIELDS = frozenset('game_id side board rows cols board_size score_1 score_2 score_A score_B multiplier_1 multiplier_2 multiplier_A multiplier_B remaining_moves direction challenge_id opponent game row col orientation tournament tournament_id championship championship_id source mode type kind challenge_type winner result'.split())
DATA_FIELDS = DATA_FIELDS | {'reason', 'cause', 'action', 'action_type', 'invincible_1', 'invincible_2',
                             'invincible_A', 'invincible_B', 'timeouts_1', 'timeouts_2',
                             'timeouts_A', 'timeouts_B'}
DEBUG_FIELDS = frozenset('event game_id side head enemy_head food numbered_food next_food_digit pickups walls multipliers crash_history tail_food_a tail_food_b legal scores remaining_moves current_target hunger repeat_count active_games search_time_budget_ms plan commit_to_plan deep_moves food_options candidate_moves advantage_control adjacent_trap_detected plan_move_stats plan_move_penalty plan_position_penalty plan_hot_lost_food_penalty tactical_arbitration deep_override_allowed plan_move_quality best_tactical_move best_tactical_quality best_tactical_stats forced_loss_avoided survival_candidates reason direction safe_repaired_from avoided_wrong_digit_moves avoided_wall_moves sequence_gate_held verification'.split())
DEBUG_FIELDS = DEBUG_FIELDS | {'action_type', 'bombs_large', 'bombs_small', 'shot_food', 'action_search'}
DEBUG_FIELDS = DEBUG_FIELDS | {'invincible_self', 'invincible_rival', 'stars', 'star_frenzy',
    'star_consumed', 'invincibility_before', 'invincibility_after', 'ram_cut',
    'ram_hit_coordinate', 'cut_length', 'shot_blocked_by_invincibility'}


def redact_reason(value):
    if isinstance(value, str) and any(marker in value.lower() for marker in
                                      ('token', 'auth', 'secret', 'password', 'credential',
                                       'bearer', 'api_key', 'apikey')):
        return '[redacted]'
    return value


def safe_debug(value):
    if isinstance(value, dict):
        return {key: safe_debug(redact_reason(item) if key in ('reason', 'cause') else item)
                for key, item in value.items()
                if not any(word in str(key).lower() for word in
                           ('token', 'secret', 'password', 'credential', 'auth', 'api_key', 'apikey'))}
    if isinstance(value, (list, tuple)):
        return [safe_debug(item) for item in value]
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


def safe_data(data):
    return {key: redact_reason(value) if key in ('reason', 'cause') else value
            for key, value in data.items() if key in DATA_FIELDS
            and isinstance(value, (str, int, float, bool, type(None)))}


def safe_message(message):
    if 'data' in message:
        return {**{key: message[key] for key in ('event', 'action', 'action_type', 'direction') if key in message},
                'data': safe_data(message['data'])}
    return safe_debug({key: value for key, value in message.items() if key in DEBUG_FIELDS})


def number(data, *keys):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def sent_prediction(data, state, direction, action_type='move'):
    if state is None or direction not in DIRS or not state.body(state.side) or state.is_terminal():
        return None
    side = state.side
    body = state.body(side)
    target = step(body[0], direction) if action_type == 'move' else body[0]
    own_tail_food = state.tail_food_a if side == 'A' else state.tail_food_b
    grows = (target in state.food or target in own_tail_food or target in state.shot_food
             or target in state.stars or state.invincibility(side) and target in state.wrong_food())
    blocked = state.occupied()
    if not grows:
        blocked.discard(body[-1])
    neck = len(body) > 1 and target == body[1]
    cause = None if action_type == 'shoot' else ('border' if not in_bounds(state, target) else
             'wall' if target in state.walls else
             'self' if neck or target in blocked and target in body else
             'rival' if target in blocked else None)
    if state.invincibility(side):
        cause = None
    known_economics = True
    try:
        predicted = apply_action(state, action_type, direction, side)
    except ValueError:
        # The official rule refuses an unknown positive-score crash. Geometry
        # is shared by both histories; the first-crash score is not evidence.
        if cause is None:
            return None
        known_economics = False
        predicted = apply_action(replace(state, **{'crashed_a' if side == 'A' else 'crashed_b': False}),
                                 action_type, direction, side)
    idx = 1 if side == 'A' else 2
    if number(data, f'score_{idx}', f'score_{side}') is None:
        known_economics = False
    if action_type == 'move' and (target in state.food or target in own_tail_food
                                  or target in state.shot_food or target in state.stars):
        known_economics = (known_economics
                           and number(data, f'multiplier_{idx}', f'multiplier_{side}') == state.multipliers[side])
    symbol = '\u24b6' if target in state.tail_food_a else '\u24b7' if target in state.tail_food_b else None
    tail = None
    if symbol:
        beneficiary = 'A' if symbol == '\u24b6' else 'B'
        tail = {'symbol': symbol, 'beneficiary': beneficiary,
                'originator': 'B' if beneficiary == 'A' else 'A',
                'event': 'consume' if target in own_tail_food and cause is None else 'cleanup_or_blocked'}
    detached = None
    for beneficiary, symbol, old, new in (
            ('A', '\u24b6', state.tail_food_a, predicted.tail_food_a),
            ('B', '\u24b7', state.tail_food_b, predicted.tail_food_b)):
        if new - old:
            detached = {'cells': [list(cell) for cell in sorted(new - old)],
                        'symbol': symbol, 'beneficiary': beneficiary, 'originator': side}
    metadata = {key: predicted.last_action.get(key) for key in (
        'action_type', 'direction', 'shoot_cost', 'hit', 'hit_coordinate',
        'cut_length', 'shot_food_generated', 'bomb_triggered', 'bomb_type', 'bomb_types',
        'bomb_positions',
        'score_before', 'score_after', 'star_consumed', 'invincibility_before',
        'invincibility_after', 'ram_cut', 'ram_hit_coordinate', 'shot_blocked_by_invincibility')}
    if not known_economics:
        metadata['score_before'] = metadata['score_after'] = None
    return {**metadata, 'source': 'prediction_not_server_confirmation', 'side': side,
            'direction': direction, 'target': target, 'collision_cause': cause,
            'tail_interaction': tail, 'expected_body': predicted.body(side),
            'expected_detached_tail': detached,
            'expected_opponent_body': predicted.body(state.enemy),
            'expected_shot_food': sorted(predicted.shot_food),
            'invincible_self': state.invincibility(side),
            'invincible_rival': state.invincibility(state.enemy),
            'stars': sorted(state.stars), 'star_frenzy': state.star_frenzy,
            'expected_invincibility': {actor: predicted.invincibility(actor) for actor in ('A', 'B')},
            'expected_stars': sorted(predicted.stars),
            'pending_bomb_respawns': predicted.pending_bomb_respawns,
            'pending_star_spawns': predicted.pending_star_spawns,
            'expected_score': predicted.scores[side] if known_economics else None,
            'uncertainty': ['rival_action_not_predicted', 'body_order_from_brain_tracking']
                + (['bomb_respawn_position_unknown'] if predicted.pending_bomb_respawns else [])
                + (['star_spawn_position_unknown'] if predicted.pending_star_spawns else [])}


def verification_record(data, observed=None, previous=None, direction=None, finished=False, action_type='move'):
    board = data.get('board') if isinstance(data.get('board'), str) else None
    cells = None
    heads = None
    tails = None
    tracked = None
    prediction = None
    special_cells = None
    crashed = {'A': None, 'B': None}
    if board is not None:
        parsed, _, _ = parse_board(board, data.get('rows'), data.get('cols'))
        special_cells = {key: [[r, c] for r, row in enumerate(parsed)
                               for c, char in enumerate(row) if char == symbol]
                         for key, symbol in (('bombs_large', '\U0001f4a3'),
                                             ('bombs_small', '\U0001f9e8'), ('shot_food', '*'),
                                             ('stars', '\u2b50'))}
        cells = {side: [[r, c] for r, row in enumerate(parsed)
                        for c, char in enumerate(row) if char in (side, side.lower())]
                 for side in ('A', 'B')}
        markers = {side: [[r, c] for r, row in enumerate(parsed)
                          for c, char in enumerate(row)
                          if char == (side.lower() if (number(data, f'invincible_{idx}',
                                                              f'invincible_{side}') or 0) > 0 else side)]
                   for side, idx in (('A', 1), ('B', 2))}
        heads = {side: positions[0] if len(positions) == 1 else None
                 for side, positions in markers.items()}
        tails = {symbol: [[r, c] for r, row in enumerate(parsed)
                          for c, char in enumerate(row) if char == symbol]
                 for symbol in ('\u24b6', '\u24b7')}
        matches_observed = (observed is not None and observed.board == parsed
                            and number(data, 'remaining_moves') == observed.remaining_moves
                            and all(number(data, f'score_{idx}', f'score_{side}') == observed.scores[side]
                                    for side, idx in (('A', 1), ('B', 2))))
        matches_observed = matches_observed and all(
            number(data, f'invincible_{idx}', f'invincible_{side}') in (None, observed.invincibility(side))
            for side, idx in (('A', 1), ('B', 2)))
        if matches_observed:
            tracked = {side: {'body': observed.body(side),
                              'order_reliable': side in observed.reliable_tails}
                       for side in ('A', 'B')}
            crashed = {'A': observed.crashed_a, 'B': observed.crashed_b}
            if (not finished and data.get('side') == observed.side
                    and number(data, 'remaining_moves') == observed.remaining_moves):
                prediction = sent_prediction(data, observed, direction, action_type)
    after = {'game_id': data.get('game_id'), 'side': data.get('side'),
             'remaining_moves': number(data, 'remaining_moves'), 'board': board,
             'scores': {side: number(data, f'score_{idx}', f'score_{side}')
                        for side, idx in (('A', 1), ('B', 2))},
             'multipliers': {side: number(data, f'multiplier_{idx}', f'multiplier_{side}')
                             for side, idx in (('A', 1), ('B', 2))},
             'body_cells': cells, 'heads': heads, 'tracked_body': tracked, 'tail_symbols': tails,
             'v8_v9_cells': special_cells,
             'invincibility': {side: number(data, f'invincible_{idx}', f'invincible_{side}')
                               for side, idx in (('A', 1), ('B', 2))},
             'invincible_self': number(data, 'invincible_1' if data.get('side') == 'A' else 'invincible_2'),
             'invincible_rival': number(data, 'invincible_2' if data.get('side') == 'A' else 'invincible_1'),
             'stars': special_cells['stars'] if special_cells is not None else None,
             'star_frenzy': (number(data, 'remaining_moves') <= 100
                             if number(data, 'remaining_moves') is not None else None),
             'crashed': crashed, 'crashed_scope': 'cumulative_brain_history',
             'sent_move': direction if action_type == 'move' else None,
             'sent_action': {'action_type': action_type, 'direction': direction} if direction else None,
             'server_reported_action': (
                 {'action_type': data.get('action', data.get('action_type')),
                  'direction': data.get('direction')}
                 if data.get('action', data.get('action_type')) in ('move', 'shoot') else None),
             'source': 'server_snapshot'}
    before = previous.get('after') if previous else None
    gap = None
    transition = 'unavailable'
    if before and before['remaining_moves'] is not None and after['remaining_moves'] is not None:
        gap = before['remaining_moves'] - after['remaining_moves']
        transition = ('duplicate_or_correction' if gap == 0 else 'stale' if gap < 0
                      else 'interval_with_unobserved_actions')
    deltas = {side: (after['scores'][side] - before['scores'][side]
                    if before and gap is not None and gap > 0
                    and after['scores'][side] is not None and before['scores'][side] is not None
                    else None) for side in ('A', 'B')}
    cause = 'unknown'
    reconciliation = 'unavailable'
    prior_prediction = previous.get('sent_action_prediction') if previous else None
    if prior_prediction and gap == 1 and cells is not None:
        side = prior_prediction['side']
        expected_score = prior_prediction['expected_score']
        if expected_score is not None and after['scores'][side] is not None:
            consistent = (set(map(tuple, cells[side])) == set(map(tuple, prior_prediction['expected_body']))
                          and after['scores'][side] == expected_score)
            reconciliation = ('consistent_body_and_score_only' if consistent
                              else 'differs_body_or_score')
            expected_head = list(prior_prediction['expected_body'][0])
            next_side = 'B' if side == 'A' else 'A'
            if heads[side] != expected_head or (after['side'] is not None and after['side'] != next_side):
                reconciliation = 'differs_head_or_side'
            elif prior_prediction.get('action_type') == 'shoot':
                enemy = 'B' if side == 'A' else 'A'
                effects_match = (consistent
                    and set(map(tuple, cells[enemy])) == set(map(tuple, prior_prediction['expected_opponent_body']))
                    and set(map(tuple, special_cells['shot_food'])) == set(map(tuple, prior_prediction['expected_shot_food'])))
                reconciliation = ('consistent_predicted_shoot_effects' if effects_match
                                  else 'differs_predicted_shoot_effects')
            counts = prior_prediction.get('expected_invincibility')
            if counts and any(value is not None and value != counts[actor]
                              for actor, value in after['invincibility'].items()):
                reconciliation = 'differs_invincibility'
    tail_changes = None
    if (before and gap is not None and gap > 0 and tails is not None
            and before['tail_symbols'] is not None):
        tail_changes = {}
        for symbol in tails:
            old = {tuple(cell) for cell in before['tail_symbols'][symbol]}
            new = {tuple(cell) for cell in tails[symbol]}
            tail_changes[symbol] = {'added': [list(cell) for cell in sorted(new - old)],
                                    'removed': [list(cell) for cell in sorted(old - new)]}
    if finished:
        horizon_end = after['remaining_moves'] is not None and after['remaining_moves'] <= 0
        eliminated = any(value is not None and value < ELIMINATION_SCORE_THRESHOLD
                         for value in after['scores'].values())
        if horizon_end or eliminated:
            cause = ('ambiguous_horizon_or_score_threshold' if horizon_end and eliminated
                     else 'possible_horizon_exhaustion' if horizon_end else 'possible_score_threshold')
    return {'before': before, 'after': after, 'horizon_delta': gap,
            'server_reported_cause': {key: value for key, value in safe_data(data).items()
                                      if finished and key in ('reason', 'cause')},
            'sent_action_prediction': prediction, 'prediction_reconciliation': reconciliation,
            'transition': transition, 'score_deltas': deltas,
            'crash_cause': {'A': 'unknown', 'B': 'unknown'},
            'tail_beneficiary_by_symbol': {'\u24b6': 'A', '\u24b7': 'B'},
            'tail_origin_by_symbol': {'\u24b6': 'B', '\u24b7': 'A'},
            'tail_net_changes': tail_changes,
            'tail_cleanup_or_consumption': 'unknown', 'game_over_cause': cause,
            'uncertainty': ['rival_actions_unobserved', 'net_deltas_not_action_counts',
                            'last_sent_move_not_confirmed', 'missing_fields_not_defaulted'],
            'last_observed_snapshot': before if finished and board is None else None}
